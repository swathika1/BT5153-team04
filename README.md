# BT5153 Team 04 - Product Quality Analysis

This project analyzes Amazon Electronics reviews using a rating-driven pipeline.
Instead of training a separate sentiment classifier, we directly use review ratings to define review quality and product quality.

## Project Scope

1. Prepare filtered review and metadata datasets.
2. Build rating-driven review labels:
bad review: overall <= 2
good review: overall >= 4
3. Aggregate to product-level quality metrics:
average rating, bad-review rate, good-review rate, verified rate
4. Identify product quality bands:
bad_product, mid_product, good_product
5. Run topic modeling on bad reviews to extract complaint themes.
6. Provide starter visualizations and an extendable dashboard app.

## Repository Structure

- `data_cleaning.ipynb`: raw data filtering/sampling and metadata construction.
- `modeling_pipeline_v2.ipynb`: rating-driven analysis, product quality outputs, topic modeling, starter insights/visualizations.
- `app.py`: Flask starter dashboard for product quality exploration.
- `requirements.txt`: Python dependencies.

## Environment Setup (WSL)

Run from project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Notebook

1. Open `modeling_pipeline_v2.ipynb` in VS Code.
2. Select your `.venv` Python kernel.
3. Run all cells top to bottom.

Outputs include:
- product-level quality tables
- bad-review topic outputs (LDA/NMF)
- starter insights summary
- starter visualizations

## Running the Dashboard

From project root in the activated environment:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Current dashboard features:
1. Product table with quality-band filtering.
2. Sorting by rating, review volume, and bad-review rate.
3. ASIN drilldown view with top complaint terms from bad reviews.
4. JSON endpoint: `/api/product/<asin>`.

## Team Handoff Notes

- Topic Modeling (Swa): interpret and label topic clusters.
- Insights + Feature Analysis (Megs): expand narrative insights from starter tables.
- Visualization + Dashboard (Ana): improve chart styling, add richer UI and interactions in Flask.
