from __future__ import annotations

from collections import Counter
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request
import pandas as pd


DATA_FILE = Path("electronics_filtered.csv")
META_FILE = Path("electronics_metadata.csv")

BAD_REVIEW_THRESHOLD = 2.0
GOOD_REVIEW_THRESHOLD = 4.0
MIN_REVIEWS_FOR_INSIGHT = 50
SAMPLE_ROWS = int(os.getenv("APP_SAMPLE_ROWS", "120000"))

app = Flask(__name__)


def build_product_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}")
    if not META_FILE.exists():
        raise FileNotFoundError(f"Missing {META_FILE}")

    reviews = pd.read_csv(
        DATA_FILE,
      nrows=SAMPLE_ROWS,
        dtype={"asin": "string"},
        usecols=["asin", "overall", "reviewText", "summary", "verified"],
    )
    meta = pd.read_csv(
        META_FILE,
        dtype={"asin": "string"},
        usecols=["asin", "title", "brand", "avg_rating", "review_count"],
    )

    reviews["reviewText"] = reviews["reviewText"].fillna("").astype(str)
    reviews["summary"] = reviews["summary"].fillna("").astype(str)
    reviews["verified"] = reviews["verified"].fillna(False).astype(bool)
    reviews["is_bad_review"] = reviews["overall"] <= BAD_REVIEW_THRESHOLD
    reviews["is_good_review"] = reviews["overall"] >= GOOD_REVIEW_THRESHOLD

    stats = (
        reviews.groupby("asin", as_index=False)
        .agg(
            reviews=("overall", "size"),
            avg_rating=("overall", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            good_review_rate=("is_good_review", "mean"),
            verified_rate=("verified", "mean"),
        )
    )

    stats = stats.merge(meta[["asin", "title", "brand"]], on="asin", how="left")

    stats["quality_band"] = "mid_product"
    stats.loc[stats["avg_rating"] <= BAD_REVIEW_THRESHOLD, "quality_band"] = "bad_product"
    stats.loc[stats["avg_rating"] >= GOOD_REVIEW_THRESHOLD, "quality_band"] = "good_product"

    stats["avg_rating"] = stats["avg_rating"].round(3)
    stats["bad_review_rate"] = stats["bad_review_rate"].round(3)
    stats["good_review_rate"] = stats["good_review_rate"].round(3)
    stats["verified_rate"] = stats["verified_rate"].round(3)

    return reviews, stats


@lru_cache(maxsize=1)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Cache avoids repeated CSV reads across requests while keeping startup quick.
    return build_product_stats()


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 3:
            tokens.append(token)
    return tokens


def top_bad_terms_for_product(reviews_df: pd.DataFrame, asin: str, top_n: int = 12) -> list[str]:
    subset = reviews_df[(reviews_df["asin"] == asin) & (reviews_df["overall"] <= BAD_REVIEW_THRESHOLD)]
    if subset.empty:
        return []

    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "you",
        "are",
        "was",
        "have",
        "had",
        "not",
        "but",
        "its",
        "too",
        "very",
        "just",
        "use",
        "used",
        "using",
        "it",
    }

    counter: Counter[str] = Counter()
    for text in subset["reviewText"]:
        for token in tokenize(text):
            if token not in stop:
                counter[token] += 1

    return [word for word, _ in counter.most_common(top_n)]


TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>BT5153 Product Quality Starter</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --good: #0e9f6e;
      --mid: #2563eb;
      --bad: #dc2626;
    }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: Segoe UI, Helvetica, Arial, sans-serif; }
    .wrap { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    h1, h2 { margin: 0 0 10px; }
    .muted { color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
    .kpi { border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .kpi .v { font-size: 1.2rem; font-weight: 700; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }
    input, select, button { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); }
    button { background: #111827; color: #fff; border: 0; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; }
    th { background: #f9fafb; position: sticky; top: 0; }
    .band-good { color: var(--good); font-weight: 700; }
    .band-mid { color: var(--mid); font-weight: 700; }
    .band-bad { color: var(--bad); font-weight: 700; }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>Product Quality Starter Dashboard</h1>
      <div class=\"muted\">Rating-driven pipeline: bad_product if avg_rating <= 2, good_product if avg_rating >= 4.</div>
    </div>

    <div class=\"card\">
      <h2>Quick Metrics</h2>
      <div class=\"grid\">
        <div class=\"kpi\"><div class=\"muted\">Products</div><div class=\"v\">{{ kpis.total_products }}</div></div>
        <div class=\"kpi\"><div class=\"muted\">Bad Products</div><div class=\"v\">{{ kpis.bad_products }}</div></div>
        <div class=\"kpi\"><div class=\"muted\">Good Products</div><div class=\"v\">{{ kpis.good_products }}</div></div>
        <div class=\"kpi\"><div class=\"muted\">Bad Review Rate</div><div class=\"v\">{{ kpis.bad_review_rate }}</div></div>
      </div>
    </div>

    <div class=\"card\">
      <h2>Product Table</h2>
      <form method=\"get\" class=\"row\">
        <label>Quality
          <select name=\"quality\">
            {% for q in qualities %}
              <option value=\"{{ q }}\" {% if q == selected.quality %}selected{% endif %}>{{ q }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Brand
          <input name=\"brand\" value=\"{{ selected.brand }}\" placeholder=\"contains...\" />
        </label>
        <label>Sort
          <select name=\"sort\">
            {% for s in sorts %}
              <option value=\"{{ s }}\" {% if s == selected.sort %}selected{% endif %}>{{ s }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Top N
          <input type=\"number\" min=\"5\" max=\"200\" name=\"topn\" value=\"{{ selected.topn }}\" />
        </label>
        <button type=\"submit\">Apply</button>
      </form>
      <div style=\"max-height: 520px; overflow: auto; margin-top: 10px;\">
        <table>
          <thead>
            <tr>
              <th>ASIN</th><th>Title</th><th>Brand</th><th>Reviews</th><th>Avg Rating</th><th>Quality</th><th>Bad Rate</th>
            </tr>
          </thead>
          <tbody>
            {% for r in rows %}
              <tr>
                <td><a href=\"/?asin={{ r.asin }}\">{{ r.asin }}</a></td>
                <td>{{ r.title }}</td>
                <td>{{ r.brand }}</td>
                <td>{{ r.reviews }}</td>
                <td>{{ r.avg_rating }}</td>
                <td class=\"{{ 'band-bad' if r.quality_band == 'bad_product' else ('band-good' if r.quality_band == 'good_product' else 'band-mid') }}\">{{ r.quality_band }}</td>
                <td>{{ r.bad_review_rate }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <div class=\"card\">
      <h2>Product Drilldown</h2>
      <form method=\"get\" class=\"row\">
        <label>ASIN
          <input name=\"asin\" value=\"{{ selected.asin }}\" placeholder=\"e.g. B000...\" />
        </label>
        <button type=\"submit\">Inspect</button>
      </form>
      {% if drilldown %}
        <p><strong>{{ drilldown.title }}</strong> ({{ drilldown.asin }})</p>
        <p>Brand: {{ drilldown.brand }} | Avg rating: {{ drilldown.avg_rating }} | Quality: {{ drilldown.quality_band }} | Reviews: {{ drilldown.reviews }}</p>
        <p><strong>Top complaint terms:</strong> {{ drilldown.top_terms|join(', ') if drilldown.top_terms else 'No bad-review terms available' }}</p>
      {% else %}
        <p class=\"muted\">Enter an ASIN or click one from the table to inspect.</p>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""


@app.route("/")
def home() -> str:
    reviews_df, product_stats = get_data()

    quality = request.args.get("quality", "ALL")
    brand = request.args.get("brand", "").strip()
    sort = request.args.get("sort", "avg_rating_asc")
    asin = request.args.get("asin", "").strip()
    topn = int(request.args.get("topn", "30"))

    df = product_stats.copy()

    if quality != "ALL":
        df = df[df["quality_band"] == quality]
    if brand:
        df = df[df["brand"].fillna("").str.contains(brand, case=False, na=False)]

    if sort == "avg_rating_asc":
        df = df.sort_values(["avg_rating", "reviews"], ascending=[True, False])
    elif sort == "avg_rating_desc":
        df = df.sort_values(["avg_rating", "reviews"], ascending=[False, False])
    elif sort == "bad_review_rate_desc":
        df = df.sort_values(["bad_review_rate", "reviews"], ascending=[False, False])
    else:
        df = df.sort_values(["reviews", "avg_rating"], ascending=[False, False])

    topn = max(5, min(200, topn))
    rows = df.head(topn).to_dict(orient="records")

    drilldown: dict[str, Any] | None = None
    if asin:
        one = product_stats[product_stats["asin"] == asin]
        if not one.empty:
            row = one.iloc[0].to_dict()
            row["top_terms"] = top_bad_terms_for_product(reviews_df, asin)
            drilldown = row

    kpis = {
        "total_products": int(product_stats["asin"].nunique()),
        "bad_products": int((product_stats["quality_band"] == "bad_product").sum()),
        "good_products": int((product_stats["quality_band"] == "good_product").sum()),
        "bad_review_rate": round(float(reviews_df["is_bad_review"].mean()), 4),
    }

    return render_template_string(
        TEMPLATE,
        kpis=kpis,
        rows=rows,
        drilldown=drilldown,
        qualities=["ALL", "bad_product", "mid_product", "good_product"],
        sorts=["avg_rating_asc", "avg_rating_desc", "bad_review_rate_desc", "reviews_desc"],
        selected={"quality": quality, "brand": brand, "sort": sort, "asin": asin, "topn": topn},
    )


@app.route("/api/product/<asin>")
def api_product(asin: str):
    reviews_df, product_stats = get_data()

    one = product_stats[product_stats["asin"] == asin]
    if one.empty:
        return jsonify({"error": "ASIN not found"}), 404

    row = one.iloc[0].to_dict()
    row["top_terms"] = top_bad_terms_for_product(reviews_df, asin)
    return jsonify(row)


if __name__ == "__main__":
  debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
  app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=False)
