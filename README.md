# SpendWise AI

> Local, open-source personal finance analyser — no cloud, no API keys, no data leaves your machine.

SpendWise AI takes messy bank transaction exports (CSV / XLSX) and produces:
- A clean, **categorized spending summary** in the terminal
- **ML-assisted categorisation** — learns from your labelled history to classify new transactions automatically
- **Recurring charge detection** — subscriptions and regular payments flagged automatically
- **Budget targets & alerts** — set monthly limits per category and get notified when you're approaching or over
- A fully self-contained, **interactive HTML dashboard** (opens offline)
- An optional **multi-page PDF report** for archiving or sharing

---

## Quick Start (< 10 minutes)

### 1. Prerequisites
- Python 3.10+

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Drop your bank export into `data/raw/`
Supported formats: `.csv`, `.xlsx`, `.xls`

Required columns (exact names, or you'll be prompted to map them):
- `Date` — transaction date
- `Description` — merchant / narrative
- `Amount` — numeric amount (expenses negative or positive, both work)

### 4. Run
```bash
# Human-review mode: interactive correction + HTML dashboard
python main.py --file data/raw/your_export.csv --dashboard

# Human-review + PDF report
python main.py --file data/raw/your_export.csv --dashboard --pdf

# Agent / pipe mode: JSON output, no prompts
python main.py --file data/raw/your_export.csv --json --no-feedback

# Train (or retrain) the ML classifier from your labelled history
python main.py --file data/raw/your_export.csv --retrain-ml

# Set monthly budget targets, then run with dashboard
python main.py --file data/raw/your_export.csv --set-budget "Groceries:400" "Transport:100" --dashboard
```

Your dashboard opens from `exports/dashboard_YYYY-MM-DD_to_YYYY-MM-DD.html`.
Your PDF report is saved to `exports/dashboard_YYYY-MM-DD_to_YYYY-MM-DD.pdf`.

---

## CLI Reference

```
python main.py --file PATH [options]

Required:
  --file, -f PATH        Path to the raw bank export (.csv or .xlsx)

Options:
  --dashboard            Generate interactive HTML dashboard
  --pdf                  Generate multi-page PDF report (requires kaleido + reportlab)
  --json                 Write JSON summary to stdout (includes recurring transactions)
  --output-json PATH     Write JSON summary to a file
  --no-feedback          Skip interactive uncategorized-transaction review
  --keywords PATH        Custom path to keywords.json
  --exports-dir DIR      Output directory for dashboards and PDFs
  --retrain-ml           Retrain ML classifier from all processed CSVs after this run
  --budgets PATH         Path to budgets.json (default: config/budgets.json)
  --set-budget CAT:AMT   Set one or more monthly budget targets, e.g. "Groceries:400"
  --help                 Show this message and exit
```

---

## Project Structure

```
spendwise-ai/
├── main.py                    # CLI entry point
├── README.md                  # This file
├── LICENSE                    # MIT licence
├── requirements.txt           # Dependencies
├── data/
│   ├── raw/                   # Drop raw bank exports here
│   └── processed/             # Cleaned, categorized CSVs (auto-generated)
├── exports/                   # HTML dashboards (auto-generated)
├── scripts/
│   ├── ingest.py              # Module 1 — ingestion & normalisation
│   ├── classifier.py          # Module 2 — keyword categoriser
│   ├── terminal_output.py     # Module 3 — terminal / JSON summary
│   ├── dashboard.py           # Module 4 — Plotly HTML + PDF dashboard
│   ├── recurring.py           # Module 5 — recurring transaction detector
│   ├── ml_classifier.py       # Module 6 — ML classifier (TF-IDF + logistic regression)
│   └── budget.py              # Module 7 — budget targets & alerts
├── config/
│   ├── keywords.json          # Category → keyword mapping
│   ├── ml_config.json         # ML settings (confidence threshold, min samples)
│   └── budgets.json           # Category → monthly limit mapping
├── models/                    # Trained ML model (git-ignored, auto-generated)
├── docs/
│   └── workflow.html          # Interactive pipeline flowchart
└── tests/                     # pytest test suite
```

---

## Pipeline Workflow

An interactive flowchart of the full ingestion-to-dashboard pipeline is available at:

```
docs/workflow.html
```

Open it in any browser — no server required. It covers every step from raw file
loading through PII masking, classification, recurring detection, and optional
dashboard / PDF export.

---

## Customising Categories

Edit `config/keywords.json` to add merchants or create new categories:

```json
{
  "Pet Care": ["petco", "petsmart", "chewy", "banfield"],
  "Food & Drink": ["starbucks", "chipotle", "your local cafe"]
}
```

Keywords are **case-insensitive substring matches** — `"starbucks"` will match
`"STARBUCKS #1234"` and `"Starbucks Coffee"`.

---

## Privacy

- All processing is **100% local** — no internet connection required after install.
- Card numbers (12–16 digits) are automatically masked to `****1234` in all outputs.
- No analytics, telemetry, or logging to external services.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas ≥ 2.0 | Data processing |
| plotly ≥ 5.18 | Interactive charts |
| openpyxl ≥ 3.1 | Excel file support |
| chardet ≥ 5.2 | Encoding detection |
| kaleido ≥ 0.2.1 | Static PNG rendering for PDF charts |
| reportlab ≥ 4.0 | PDF assembly |
| scikit-learn ≥ 1.2 | ML classifier training & inference |
