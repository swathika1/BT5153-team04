from __future__ import annotations

from collections import Counter
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


DATA_FILE = Path("electronics_filtered.csv")
META_FILE = Path("electronics_metadata.csv")

BAD_REVIEW_THRESHOLD = 2.0
GOOD_REVIEW_THRESHOLD = 4.0
SAMPLE_ROWS = int(os.getenv("APP_SAMPLE_ROWS", "120000"))

app = Flask(__name__)

_EXTRA_STOP: frozenset[str] = frozenset({
    "product", "buy", "amazon", "bought", "item", "purchase", "purchased",
    "ordered", "order", "got", "get", "one", "just", "really", "also",
    "would", "could", "like", "don", "ve", "ll", "re", "isn", "wasn",
    "great", "good", "work", "works", "didn", "does", "doesn", "bit",
    "lot", "still", "even", "need", "want", "put", "way", "time",
})
ALL_STOP: frozenset[str] = frozenset(ENGLISH_STOP_WORDS) | _EXTRA_STOP

# Ordered so the first matching category wins.
_TOPIC_KEYWORDS: list[tuple[str, frozenset[str]]] = [
    ("Battery life or charging issues", frozenset({
        "battery", "charge", "charging", "charged", "dies", "dead", "drain",
        "draining", "mah", "recharge", "rechargeable", "overheat", "overheating",
    })),
    ("Connectivity or wireless issues", frozenset({
        "wifi", "wireless", "bluetooth", "connect", "connection", "disconnect",
        "disconnects", "signal", "pairing", "pair", "network", "sync", "syncing",
        "latency",
    })),
    ("Build quality or durability issues", frozenset({
        "broke", "broken", "crack", "cracked", "flimsy", "fragile", "peel",
        "peeling", "scratched", "shattered", "hinge", "durability",
    })),
    ("Software or firmware issues", frozenset({
        "firmware", "software", "driver", "drivers", "install", "installation",
        "bug", "bugs", "crash", "crashes", "freeze", "freezes", "reboot",
        "restart", "glitch", "update", "updates", "error", "errors",
    })),
    ("Customer service or support issues", frozenset({
        "support", "warranty", "refund", "replacement", "exchange", "seller",
        "return", "returns", "complaint", "response", "unresponsive",
    })),
    ("Packaging or shipping damage", frozenset({
        "packaging", "package", "shipping", "shipped", "damaged", "dented",
        "box", "arrived", "missing", "incomplete",
    })),
    ("Value for money concerns", frozenset({
        "price", "expensive", "overpriced", "waste", "worth", "value",
        "cost", "money", "dollars",
    })),
    ("Compatibility issues", frozenset({
        "compatible", "compatibility", "incompatible", "fit", "fitting",
        "dimensions", "version",
    })),
]


def interpret_topic(words: list[str]) -> str | None:
    # Expand bigrams so "battery life" also matches on "battery" and "life".
    tokens: set[str] = set()
    for w in words:
        tokens.add(w)
        tokens.update(w.split())
    for label, keywords in _TOPIC_KEYWORDS:
        if tokens & keywords:
            return f"Likely complaint: {label}"
    return None


@lru_cache(maxsize=1)
def load_meta() -> pd.DataFrame:
    """Load pre-computed product metadata — fast: 11K rows, no aggregation needed."""
    if not META_FILE.exists():
        raise FileNotFoundError(f"Missing {META_FILE}")
    meta = pd.read_csv(META_FILE, dtype={"asin": "string"})
    meta["brand"] = meta["brand"].fillna("Unknown")
    meta["title"] = meta["title"].fillna(meta["asin"])
    meta["avg_rating"] = pd.to_numeric(meta["avg_rating"], errors="coerce").fillna(0.0).round(3)
    meta["quality_band"] = "mid_product"
    meta.loc[meta["avg_rating"] <= BAD_REVIEW_THRESHOLD, "quality_band"] = "bad_product"
    meta.loc[meta["avg_rating"] >= GOOD_REVIEW_THRESHOLD, "quality_band"] = "good_product"
    meta = meta.rename(columns={"review_count": "reviews"})
    return meta


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    """Load raw review sample — only used for per-product drilldown analysis."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}")
    reviews = pd.read_csv(
        DATA_FILE,
        nrows=SAMPLE_ROWS,
        dtype={"asin": "string"},
        usecols=["asin", "overall", "reviewText"],
    )
    reviews["reviewText"] = reviews["reviewText"].fillna("").astype(str)
    reviews["overall"] = pd.to_numeric(reviews["overall"], errors="coerce").fillna(3.0)
    return reviews


@lru_cache(maxsize=1)
def get_band_avg_bad_rates() -> dict[str, float]:
    """Mean bad_review_rate per quality_band across all ASINs present in the reviews sample."""
    reviews = load_reviews().copy()
    meta = load_meta()
    reviews["is_bad"] = reviews["overall"] <= BAD_REVIEW_THRESHOLD
    asin_rates = (
        reviews.groupby("asin")["is_bad"]
        .mean()
        .reset_index()
        .rename(columns={"is_bad": "bad_rate"})
    )
    asin_rates = asin_rates.merge(meta[["asin", "quality_band"]], on="asin", how="inner")
    band_avgs = asin_rates.groupby("quality_band")["bad_rate"].mean()
    return {
        band: float(band_avgs.get(band, 0.0))
        for band in ("bad_product", "mid_product", "good_product")
    }


def get_product_bad_rate(asin: str) -> float | None:
    """Bad review rate for a single ASIN from the reviews sample, or None if absent."""
    subset = load_reviews()
    subset = subset[subset["asin"] == asin]
    if subset.empty:
        return None
    return float((subset["overall"] <= BAD_REVIEW_THRESHOLD).mean())


def get_brand_stats(meta: pd.DataFrame) -> list[dict]:
    brands = (
        meta[meta["brand"] != "Unknown"]
        .groupby("brand", as_index=False)
        .agg(
            products=("asin", "count"),
            bad_products=("quality_band", lambda x: (x == "bad_product").sum()),
            avg_rating=("avg_rating", "mean"),
        )
    )
    brands = brands[brands["products"] >= 3].copy()
    brands["bad_pct"] = (brands["bad_products"] / brands["products"] * 100).round(1)
    brands["avg_rating"] = brands["avg_rating"].round(2)
    return brands.sort_values("bad_pct", ascending=False).head(15).to_dict(orient="records")


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 3:
            tokens.append(token)
    return tokens


def get_topics_for_product(asin: str, n_topics: int = 4, top_words: int = 8) -> list[list[str]] | None:
    """Run NMF topic model on a product's bad reviews. Returns None when too few reviews."""
    try:
        reviews_df = load_reviews()
        bad = reviews_df[
            (reviews_df["asin"] == asin) & (reviews_df["overall"] <= BAD_REVIEW_THRESHOLD)
        ]
        if len(bad) < 10:
            return None
        vec = TfidfVectorizer(
            max_features=500, stop_words="english", min_df=2, ngram_range=(1, 2),
        )
        X = vec.fit_transform(bad["reviewText"])
        k = min(n_topics, max(1, len(bad) // 5))
        model = NMF(n_components=k, random_state=42, max_iter=200)
        model.fit(X)
        names = vec.get_feature_names_out()
        return [
            [names[i] for i in topic.argsort()[-top_words:][::-1]]
            for topic in model.components_
        ]
    except Exception:
        return None


def get_rating_distribution(asin: str) -> dict[int, int]:
    reviews_df = load_reviews()
    dist = (
        reviews_df[reviews_df["asin"] == asin]["overall"]
        .dropna()
        .astype(int)
        .value_counts()
    )
    return {i: int(dist.get(i, 0)) for i in range(1, 6)}


def top_bad_terms(asin: str, top_n: int = 12) -> list[str]:
    reviews_df = load_reviews()
    bad = reviews_df[
        (reviews_df["asin"] == asin) & (reviews_df["overall"] <= BAD_REVIEW_THRESHOLD)
    ]
    if bad.empty:
        return []
    counter: Counter[str] = Counter()
    for text in bad["reviewText"]:
        for token in tokenize(text):
            if token not in ALL_STOP:
                counter[token] += 1
    return [word for word, _ in counter.most_common(top_n)]


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BT5153 Product Quality Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --bad: #ef4444;
      --mid: #3b82f6;
      --good: #22c55e;
      --btn: #2563eb;
      --link: #60a5fa;
      --th-bg: #0f172a;
      --row-hover: #334155;
      --row-alt: #253347;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
      font-size: 14px;
    }
    a { color: var(--link); text-decoration: none; }
    a:hover { text-decoration: underline; }

    .site-header {
      width: 100%;
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .site-header h1 {
      margin: 0;
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text);
    }
    .site-header .subtitle { color: var(--muted); font-size: 13px; }

    .wrap {
      max-width: 1400px;
      margin: 24px auto;
      padding: 0 16px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-left: 3px solid var(--btn);
      border-radius: 12px;
      padding: 20px;
    }

    h2 {
      margin: 0 0 16px;
      font-size: 1.1rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text);
    }

    .muted { color: var(--muted); }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }
    .kpi {
      background: var(--card);
      border: 1px solid var(--border);
      border-top: 3px solid var(--border);
      border-radius: 10px;
      padding: 20px;
    }
    .kpi-total { border-top-color: #3b82f6; }
    .kpi-bad   { border-top-color: #ef4444; }
    .kpi-good  { border-top-color: #22c55e; }
    .kpi-rate  { border-top-color: #f59e0b; }
    .kpi .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    .kpi .v { font-size: 1.8rem; font-weight: 700; }

    .metrics-layout { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }
    .metrics-left { flex: 1; min-width: 320px; }
    .metrics-right { width: 280px; }

    .brand-chart-wrap { position: relative; height: 320px; }

    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-end; }
    label {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    input, select {
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 13px;
    }
    button {
      padding: 8px 16px;
      border-radius: 8px;
      background: var(--btn);
      color: #fff;
      border: 0;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      align-self: flex-end;
    }
    button:hover { background: #1d4ed8; }

    .table-container {
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
      margin-top: 16px;
      max-height: 520px;
      overflow-y: auto;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 12px; text-align: left; font-size: 13px; }
    th {
      background: var(--th-bg);
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 1;
      border-bottom: 1px solid var(--border);
    }
    tbody tr:nth-child(odd)  { background: var(--card); }
    tbody tr:nth-child(even) { background: var(--row-alt); }
    tbody tr:hover { background: var(--row-hover); }
    td { border-bottom: 1px solid #253347; }

    .badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
    }
    .badge-bad  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }
    .badge-mid  { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.4); }
    .badge-good { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.4); }

    .topic-card {
      margin: 6px 0;
      padding: 10px 14px;
      background: rgba(239,68,68,0.08);
      border-radius: 6px;
      border-left: 3px solid var(--bad);
      font-size: 0.9rem;
    }
    .topic-label { font-weight: 600; color: var(--bad); margin-right: 6px; }
    .topic-interp { margin-top: 4px; font-size: 0.82rem; color: var(--muted); font-style: italic; }
    details summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
  </style>
</head>
<body>
  <header class="site-header">
    <h1>Product Quality Dashboard</h1>
    <span class="subtitle">BT5153 Group 4 &nbsp;|&nbsp; Amazon Electronics Analysis</span>
  </header>

  <div class="wrap">
    <!-- Quick Metrics -->
    <div class="card">
      <h2>Quick Metrics</h2>
      <div class="metrics-layout">
        <div class="metrics-left">
          <div class="kpi-grid">
            <div class="kpi kpi-total">
              <div class="label">Total Products</div>
              <div class="v">{{ kpis.total_products }}</div>
            </div>
            <div class="kpi kpi-bad">
              <div class="label">Bad Products</div>
              <div class="v" style="color:var(--bad)">{{ kpis.bad_products }}</div>
            </div>
            <div class="kpi kpi-good">
              <div class="label">Good Products</div>
              <div class="v" style="color:var(--good)">{{ kpis.good_products }}</div>
            </div>
            <div class="kpi kpi-rate">
              <div class="label">Bad Product Rate</div>
              <div class="v" style="color:#f59e0b">{{ kpis.bad_product_rate }}</div>
            </div>
          </div>
          <p class="muted" style="margin:12px 0 0; font-size:12px;">
            Rating-driven pipeline &mdash; bad_product: avg_rating &le; 2.0 &nbsp;&bull;&nbsp; good_product: avg_rating &ge; 4.0
          </p>
        </div>
        <div class="metrics-right">
          <canvas id="qualityChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Brand Chart -->
    <div class="card">
      <h2>Top 15 Brands by Bad Product Rate</h2>
      <div class="brand-chart-wrap">
        <canvas id="brandChart"></canvas>
      </div>
    </div>

    <!-- Product Table -->
    <div class="card">
      <h2>Product Table</h2>
      <form method="get" class="row">
        <label>Quality
          <select name="quality">
            {% for q in qualities %}
              <option value="{{ q }}" {% if q == selected.quality %}selected{% endif %}>{{ q }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Brand
          <input name="brand" value="{{ selected.brand }}" placeholder="contains..." />
        </label>
        <label>Title
          <input name="title" value="{{ selected.title }}" placeholder="contains..." />
        </label>
        <label>Sort
          <select name="sort">
            {% for s in sorts %}
              <option value="{{ s }}" {% if s == selected.sort %}selected{% endif %}>{{ s }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Top N
          <input type="number" min="5" max="200" name="topn" value="{{ selected.topn }}" />
        </label>
        <button type="submit">Apply</button>
      </form>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>ASIN</th><th>Title</th><th>Brand</th><th>Reviews</th><th>Avg Rating</th><th>Quality</th>
            </tr>
          </thead>
          <tbody>
            {% for r in rows %}
              <tr>
                <td><a href="/?asin={{ r.asin }}">{{ r.asin }}</a></td>
                <td>{{ r.title }}</td>
                <td>{{ r.brand }}</td>
                <td>{{ r.reviews }}</td>
                <td>{{ r.avg_rating }}</td>
                <td>
                  {% if r.quality_band == 'bad_product' %}
                    <span class="badge badge-bad">{{ r.quality_band }}</span>
                  {% elif r.quality_band == 'good_product' %}
                    <span class="badge badge-good">{{ r.quality_band }}</span>
                  {% else %}
                    <span class="badge badge-mid">{{ r.quality_band }}</span>
                  {% endif %}
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Product Drilldown -->
    <div class="card">
      <h2>Product Drilldown</h2>
      <form method="get" class="row">
        <label>ASIN
          <input name="asin" value="{{ selected.asin }}" placeholder="e.g. B000..." />
        </label>
        <button type="submit">Inspect</button>
      </form>
      {% if drilldown %}
        <div style="margin-top:16px;">
          <p style="margin:0 0 4px; font-weight:600;">
            {{ drilldown.title }} <span class="muted">({{ drilldown.asin }})</span>
          </p>
          <p style="margin:0 0 12px;" class="muted">
            Brand: {{ drilldown.brand }} &nbsp;|&nbsp;
            Avg Rating: {{ drilldown.avg_rating }} &nbsp;|&nbsp;
            Quality:&nbsp;
            {% if drilldown.quality_band == 'bad_product' %}
              <span class="badge badge-bad">{{ drilldown.quality_band }}</span>
            {% elif drilldown.quality_band == 'good_product' %}
              <span class="badge badge-good">{{ drilldown.quality_band }}</span>
            {% else %}
              <span class="badge badge-mid">{{ drilldown.quality_band }}</span>
            {% endif %}
            &nbsp;|&nbsp; Reviews: {{ drilldown.reviews }}
          </p>
          {% if drilldown.bad_rate_pct is not none %}
          <p style="margin:0 0 12px;">
            <span class="muted">Bad Review Rate:</span>
            <span style="font-weight:700; color:{{ 'var(--good)' if drilldown.vs_avg == 'better' else 'var(--bad)' }}">{{ drilldown.bad_rate_pct }}%</span>
            <span class="muted">vs avg</span>
            <span style="font-weight:600;">{{ drilldown.band_avg_pct }}%</span>
            <span style="font-size:0.82rem; color:{{ 'var(--good)' if drilldown.vs_avg == 'better' else 'var(--bad)' }}">
              ({{ '▲ above' if drilldown.vs_avg == 'worse' else '▼ below' }} band avg)
            </span>
          </p>
          {% endif %}
          <div style="display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start;">
            <div style="flex:1; min-width:280px;">
              {% if drilldown.topics %}
                <strong>Complaint Themes (NMF Topic Model):</strong>
                {% for topic in drilldown.topics %}
                  <div class="topic-card">
                    <span class="topic-label">Topic {{ loop.index }}:</span>{{ topic | join(' &middot; ') }}
                    {% set lbl = drilldown.topic_labels[loop.index0] %}
                    {% if lbl %}<div class="topic-interp">{{ lbl }}</div>{% endif %}
                  </div>
                {% endfor %}
              {% elif drilldown.top_terms %}
                <strong>Top Complaint Terms:</strong>
                <p>{{ drilldown.top_terms | join(', ') }}</p>
              {% else %}
                <p class="muted">Not enough bad reviews for term analysis.</p>
              {% endif %}
            </div>
            <div style="width:320px;">
              <canvas id="ratingChart" height="180"></canvas>
            </div>
          </div>
        </div>
      {% else %}
        <p class="muted">Enter an ASIN or click one from the table to inspect.</p>
      {% endif %}
    </div>
  </div>

  <script>
    Chart.register(ChartDataLabels);

    const GRID_COLOR  = '#334155';
    const TICK_COLOR  = '#94a3b8';
    const TITLE_COLOR = '#f1f5f9';

    new Chart(document.getElementById('qualityChart'), {
      type: 'bar',
      data: {
        labels: ['Bad', 'Mid', 'Good'],
        datasets: [{
          data: [{{ kpis.bad_products }}, {{ kpis.mid_products }}, {{ kpis.good_products }}],
          backgroundColor: ['#ef4444', '#3b82f6', '#22c55e'],
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Quality Band Breakdown', color: TITLE_COLOR },
          datalabels: { display: false },
        },
        scales: {
          x: { beginAtZero: true, ticks: { color: TICK_COLOR }, grid: { color: GRID_COLOR } },
          y: { ticks: { color: TICK_COLOR }, grid: { color: GRID_COLOR } },
        },
      }
    });

    const allBrands = {{ brand_stats | tojson }};
    const brands = allBrands.filter(b => b.bad_pct > 0);
    const brandXMax = brands.length
      ? Math.ceil((Math.max(...brands.map(b => b.bad_pct)) * 1.2) / 5) * 5
      : 100;

    new Chart(document.getElementById('brandChart'), {
      type: 'bar',
      data: {
        labels: brands.map(b => b.brand),
        datasets: [{
          data: brands.map(b => b.bad_pct),
          backgroundColor: '#ef4444',
          borderRadius: 4,
          barThickness: 20,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Bad Product Rate (%)', color: TITLE_COLOR },
          datalabels: {
            anchor: 'end',
            align: 'end',
            formatter: v => v + '%',
            color: TITLE_COLOR,
            font: { size: 11, weight: '600' },
          },
        },
        scales: {
          x: { beginAtZero: true, max: brandXMax, ticks: { color: TICK_COLOR }, grid: { color: GRID_COLOR } },
          y: { ticks: { color: TICK_COLOR }, grid: { color: GRID_COLOR } },
        },
        layout: { padding: { right: 44 } },
      }
    });

    {% if drilldown %}
    (function () {
      const rd = {{ drilldown.rating_dist | tojson }};
      const rdValues = [1, 2, 3, 4, 5].map(k => rd[k] ?? 0);
    new Chart(document.getElementById('ratingChart'), {
      type: 'bar',
      data: {
        labels: ['1★', '2★', '3★', '4★', '5★'],
        datasets: [{
          label: 'Reviews',
          data: rdValues,
          backgroundColor: ['#ef4444', '#f87171', '#fbbf24', '#86efac', '#22c55e'],
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Rating Distribution', color: TITLE_COLOR },
          datalabels: { display: false },
        },
        scales: {
          y: { beginAtZero: true, ticks: { color: TICK_COLOR }, grid: { color: GRID_COLOR } },
          x: { ticks: { color: TICK_COLOR }, grid: { color: GRID_COLOR } },
        },
      }
    });
    }());
    {% endif %}
  </script>
</body>
</html>
"""


@app.route("/")
def home() -> str:
    meta = load_meta()

    quality = request.args.get("quality", "ALL")
    brand = request.args.get("brand", "").strip()
    title = request.args.get("title", "").strip()
    sort = request.args.get("sort", "avg_rating_asc")
    asin = request.args.get("asin", "").strip()
    topn = max(5, min(200, int(request.args.get("topn", "30"))))

    df = meta.copy()
    if quality != "ALL":
        df = df[df["quality_band"] == quality]
    if brand:
        df = df[df["brand"].str.contains(brand, case=False, na=False)]
    if title:
        df = df[df["title"].str.contains(title, case=False, na=False)]

    if sort == "avg_rating_asc":
        df = df.sort_values(["avg_rating", "reviews"], ascending=[True, False])
    elif sort == "avg_rating_desc":
        df = df.sort_values(["avg_rating", "reviews"], ascending=[False, False])
    elif sort == "bad_review_rate_desc":
        df = df.sort_values(["avg_rating", "reviews"], ascending=[True, False])
    else:
        df = df.sort_values(["reviews", "avg_rating"], ascending=[False, False])

    rows = df.head(topn).to_dict(orient="records")

    drilldown: dict[str, Any] | None = None
    if asin:
        one = meta[meta["asin"] == asin]
        if not one.empty:
            row = one.iloc[0].to_dict()
            topics = get_topics_for_product(asin)
            row["topics"] = topics
            row["topic_labels"] = [interpret_topic(t) for t in topics] if topics else []
            row["top_terms"] = [] if topics else top_bad_terms(asin)
            row["rating_dist"] = get_rating_distribution(asin)
            product_bad_rate = get_product_bad_rate(asin)
            band_avg = get_band_avg_bad_rates().get(row["quality_band"], 0.0)
            row["bad_rate_pct"] = round(product_bad_rate * 100, 1) if product_bad_rate is not None else None
            row["band_avg_pct"] = round(band_avg * 100, 1)
            row["vs_avg"] = "better" if (product_bad_rate is not None and product_bad_rate < band_avg) else "worse"
            drilldown = row

    kpis = {
        "total_products": int(meta["asin"].nunique()),
        "bad_products": int((meta["quality_band"] == "bad_product").sum()),
        "mid_products": int((meta["quality_band"] == "mid_product").sum()),
        "good_products": int((meta["quality_band"] == "good_product").sum()),
        "bad_product_rate": f"{float((meta['quality_band'] == 'bad_product').mean()):.1%}",
    }

    brand_stats = get_brand_stats(meta)

    return render_template_string(
        TEMPLATE,
        kpis=kpis,
        rows=rows,
        drilldown=drilldown,
        brand_stats=brand_stats,
        qualities=["ALL", "bad_product", "mid_product", "good_product"],
        sorts=["avg_rating_asc", "avg_rating_desc", "bad_review_rate_desc", "reviews_desc"],
        selected={"quality": quality, "brand": brand, "title": title, "sort": sort, "asin": asin, "topn": topn},
    )


@app.route("/api/product/<asin>")
def api_product(asin: str):
    meta = load_meta()
    one = meta[meta["asin"] == asin]
    if one.empty:
        return jsonify({"error": "ASIN not found"}), 404
    row = one.iloc[0].to_dict()
    row["top_terms"] = top_bad_terms(asin)
    row["topics"] = get_topics_for_product(asin)
    row["topic_labels"] = [interpret_topic(t) for t in row["topics"]] if row["topics"] else []
    row["rating_dist"] = get_rating_distribution(asin)
    product_bad_rate = get_product_bad_rate(asin)
    band_avg = get_band_avg_bad_rates().get(row["quality_band"], 0.0)
    row["bad_rate_pct"] = round(product_bad_rate * 100, 1) if product_bad_rate is not None else None
    row["band_avg_pct"] = round(band_avg * 100, 1)
    row["vs_avg"] = "better" if (product_bad_rate is not None and product_bad_rate < band_avg) else "worse"
    return jsonify(row)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=False)
