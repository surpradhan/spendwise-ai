# SpendWise AI

> **Your finances, analysed locally.**
> Drop a bank export in. Get categorised spending, ML-powered insights, and an interactive dashboard — in under a minute. No cloud. No API keys. No data leaves your machine.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](#) [![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](#) [![Local only](https://img.shields.io/badge/data-local--only-f59e0b)](#) [![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](#)

---

## What it does

| Feature | Detail |
|---------|--------|
| **Multi-bank adapter support** | Auto-detects HDFC Bank exports; generic fallback handles any standard CSV with fuzzy column mapping |
| **Multi-currency tracking** | `Currency` column preserved per transaction (INR, GBP, USD, etc.); per-currency breakdown in summary and dashboard |
| **Auto-categorises transactions** | Keyword rules with an ML fallback (TF-IDF + logistic regression) that learns from your history |
| **Detects recurring charges** | Subscriptions and regular payments flagged without any configuration |
| **Budget tracking & alerts** | Set monthly limits per category; get warned at 80 % and 100 % |
| **Anomaly detection** | Flags unusual transactions via modified z-score (median + MAD); per-category with global fallback for singletons |
| **Natural language queries** | Ask questions in plain English: `show groceries`, `top 5 last 3 months`, `sum food & drink` |
| **Interactive HTML dashboard** | 8 charts — donut, trend, top merchants, income vs expenses, anomaly scatter, and more — fully offline |
| **PDF report** | Multi-page export for archiving or sharing |
| **Pipe-friendly JSON mode** | `--json --no-feedback` for scripting and automation |

---

## Pipeline

```mermaid
flowchart LR
    A[/"Bank Export\nCSV / XLSX"/] --> B["Bank Adapter\nHDFC / Generic"]
    B --> C["Ingest and Normalise\nencoding, dates, PII masking\nCurrency column"]
    C --> D{Classify}
    D -->|keyword pass| E["Keyword Rules"]
    D -->|remaining rows| F["ML Model\nTF-IDF + LR"]
    E --> G["Categorised DataFrame"]
    F --> G
    G --> H["HTML Dashboard\n(8 charts incl. anomaly)"]
    G --> I["PDF Report"]
    G --> J["Recurring Detector"]
    G --> K["Budget Alerts"]
    G --> L["JSON Output"]
    G --> M["Anomaly Detection\n--anomalies"]
    G --> N["NL Query Engine\n--query '...'"]
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Drop your bank export into data/raw/, then run
python main.py --file data/raw/export.csv --dashboard
```

Your dashboard is saved to `exports/dashboard_YYYY-MM-DD_to_YYYY-MM-DD.html` — open it in any browser, no server needed.

**Supported formats:** `.csv`, `.xlsx`, `.xls`
**Required columns:** `Date`, `Description`, `Amount` (or you'll be prompted to map them)

---

## Common Commands

```bash
# Interactive review + PDF report
python main.py --file data/raw/export.csv --dashboard --pdf

# Automation / pipe mode — JSON to stdout, no prompts
python main.py --file data/raw/export.csv --json --no-feedback

# HDFC Bank statement (auto-detected, or force with --bank hdfc)
python main.py --file data/raw/hdfc_statement.csv --dashboard

# Non-USD import — set the currency code for the generic adapter
python main.py --file data/raw/barclays.csv --currency GBP --dashboard

# Set monthly budget limits, then run with dashboard
python main.py --file data/raw/export.csv --set-budget "Groceries:400" "Transport:100" --dashboard

# Retrain the ML classifier from your entire labelled history
python main.py --file data/raw/export.csv --retrain-ml

# Detect unusual transactions (modified z-score per category)
python main.py --file data/raw/export.csv --anomalies

# Ask a natural language question about your spending
python main.py --file data/raw/export.csv --query "show groceries"
python main.py --file data/raw/export.csv --query "top 5 last 3 months"
python main.py --file data/raw/export.csv --query "categories"
```

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `--file PATH` | **(required)** Path to raw bank export |
| `--bank HINT` | Force a bank adapter, e.g. `--bank hdfc`. Overrides auto-detection |
| `--currency CODE` | Default currency for generic imports, e.g. `--currency GBP` |
| `--dashboard` | Generate interactive HTML dashboard |
| `--pdf` | Generate multi-page PDF report |
| `--json` | Write JSON summary to stdout |
| `--output-json PATH` | Write JSON summary to a file |
| `--no-feedback` | Skip interactive review (for scripting) |
| `--retrain-ml` | Retrain ML classifier after this run |
| `--anomalies` | Print anomaly report (unusual transactions flagged by modified z-score) |
| `--query QUERY` | Run a natural language query and print the result (run `categories` first to see available categories) |
| `--set-budget CAT:AMT` | Set one or more monthly budget limits |
| `--keywords PATH` | Custom `keywords.json` path |
| `--budgets PATH` | Custom `budgets.json` path |
| `--exports-dir DIR` | Output directory for dashboards and PDFs |

---

## Customising Categories

Edit `config/keywords.json` to add merchants or new categories:

```json
{
  "Pet Care":    ["petco", "petsmart", "chewy"],
  "Food & Drink": ["starbucks", "chipotle", "your local cafe"]
}
```

Keywords are **case-insensitive substring matches** — `"starbucks"` matches `"STARBUCKS #1234"`.
After adding new categories, run `--retrain-ml` so the ML model picks them up.

---

## Privacy

- **100 % local** — no network calls after install.
- Card numbers (12–16 digits) are auto-masked to `****1234` in every output — terminal, CSV, and dashboard.
- No analytics, telemetry, or logging to external services.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas ≥ 2.0` | Data processing |
| `plotly ≥ 5.18` | Interactive charts |
| `scikit-learn ≥ 1.2` | ML classifier |
| `openpyxl ≥ 3.1` | Excel file support |
| `chardet ≥ 5.2` | Encoding detection |
| `kaleido ≥ 0.2.1` | Static PNG rendering (PDF charts) |
| `reportlab ≥ 4.0` | PDF assembly |

---

## Project Structure

```
spendwise-ai/
├── main.py                    # CLI entry point
├── data/raw/                  # Drop raw bank exports here
├── data/processed/            # Cleaned, categorised CSVs (auto-generated)
├── exports/                   # Dashboards and PDF reports (auto-generated)
├── scripts/
│   ├── adapters/              # Bank-format adapters
│   │   ├── __init__.py        #   detect_adapter() registry
│   │   ├── base.py            #   BankAdapter abstract base class
│   │   ├── generic.py         #   GenericAdapter (fallback; infers currency)
│   │   └── hdfc.py            #   HDFCAdapter (HDFC Bank statement format)
│   ├── ingest.py              # Ingestion & normalisation
│   ├── classifier.py          # Keyword categoriser
│   ├── ml_classifier.py       # ML classifier (TF-IDF + logistic regression)
│   ├── recurring.py           # Recurring transaction detector
│   ├── budget.py              # Budget targets & alerts
│   ├── anomaly.py             # Anomaly detection (modified z-score)
│   ├── nl_query.py            # Natural language query engine
│   ├── dashboard.py           # Plotly HTML + PDF dashboard
│   └── terminal_output.py     # Terminal & JSON summary
├── config/
│   ├── keywords.json          # Category → keyword mapping (edit this)
│   ├── budgets.json           # Category → monthly limit (edit this)
│   └── ml_config.json         # ML settings (confidence threshold)
└── models/                    # Trained ML model (git-ignored, auto-generated)
```
