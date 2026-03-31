# BT5153 Group Project

## Dataset
- Amazon Electronics reviews: 20,994,353 reviews, 786,868 products
- Source: `Electronics.json.gz` (JSON Lines format, gzip compressed)
- Metadata: `meta_Electronics.json.gz` — used for product `title` and `brand`

## Goals
- Analyze top complaints per product
- Analyze sentiment distribution
- Extract “Top 3 reasons for 1-star reviews” per product
- Compare good vs bad products

## Data Cleaning Pipeline

### Step 1 — Stream and filter during load (`data_cleaning.ipynb`)
Read `Electronics.json.gz` in chunks of 50,000 rows, writing directly to `electronics_cleaned.csv` to avoid loading 3GB+ into memory. Progress is saved to `progress.txt` so the job can resume if interrupted.

Fields kept: `asin`, `overall`, `reviewText`, `summary`, `verified`
Fields dropped: `reviewTime`, `unixReviewTime`, `reviewerID`, `reviewerName`, `style`, `vote`

Rows dropped during load:
- Missing `reviewText`
- `reviewText` with fewer than 20 words

Result: ~12.3M reviews retained (down from 22M)

### Step 2 — Filter by review count (`data_cleaning.ipynb`)
Distribution of reviews per product (721,250 products total):
- Median: 3 reviews
- 75th percentile: 10 reviews
- Mean: 26, std: 163 — heavily right-skewed

**Threshold: keep only products with >= 200 reviews** (~11,000 products, top ~2% of all products). Products below this threshold lack sufficient signal for complaint extraction and sentiment analysis.

### Step 3 — Cap popular products (`data_cleaning.ipynb`)
3,442 products have more than 500 reviews. To prevent popular products from dominating the dataset, randomly sample max 500 reviews per product (seed=42 for reproducibility).

**Output: `electronics_filtered.csv`** (~11K products, ~3.94M reviews total)

### Step 4 — Metadata extraction (`data_cleaning.ipynb`)
Read `meta_Electronics.json.gz` and extract `asin`, `title`, `brand`. Inner join with `product_scores` (average `overall` rating and review count per product computed from `electronics_filtered.csv`). Only products with ≥200 reviews are kept (inner join drops the rest).

**Output: `electronics_metadata.csv`** — `asin`, `title`, `brand`, `avg_rating`, `review_count`, `product_label`

**Product labeling thresholds:**
- Good: avg_rating >= 4 → 5,735 products (51.2%)
- Bad: avg_rating < 3 → 499 products (4.4%)
- Mediocre: 3 <= avg_rating < 4 → 4,971 products (44.4%)

We compared 100 vs 200 minimum review thresholds. While 100 gives more bad products (1,259 vs 499), it is insufficient for reliable topic modeling — with only ~10-15 negative reviews per product at that threshold. The 200 threshold ensures more consistent signal, and the good/bad distribution is similar (5.7% vs 4.4% bad), confirming 200 is the better choice.

### Step 5 — Sentiment labeling
Rating mapped to sentiment label for downstream analysis:
- 1–2 stars → negative
- 3 stars → neutral
- 4–5 stars → positive

Result: 72% positive, 19% negative, 9% neutral (across 3.94M reviews)

### Step 6 — Data sharing
File is too large for GitHub (100MB limit). Share via Google Drive. Teammates can mount in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Loading the filtered data
```python
import polars as pl
df = pl.scan_csv('electronics_filtered.csv', schema_overrides={'asin': pl.Utf8})
# filter as needed, e.g.:
negative = df.filter(pl.col('overall') <= 2).collect()
```