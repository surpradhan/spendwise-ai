# SpendWise AI

> Local, open-source personal finance analyser — no cloud, no API keys, no data leaves your machine.

SpendWise AI takes messy bank transaction exports (CSV / XLSX) and produces:
- A clean, **categorized spending summary** in the terminal
- A fully self-contained, **interactive HTML dashboard** (opens offline)

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

# Agent / pipe mode: JSON output, no prompts
python main.py --file data/raw/your_export.csv --json --no-feedback
```

Your dashboard opens from `exports/dashboard_YYYY-MM-DD_to_YYYY-MM-DD.html`.

---

## CLI Reference

```
python main.py --file PATH [options]

Required:
  --file, -f PATH        Path to the raw bank export (.csv or .xlsx)

Options:
  --dashboard            Generate interactive HTML dashboard
  --json                 Write JSON summary to stdout
  --output-json PATH     Write JSON summary to a file
  --no-feedback          Skip interactive uncategorized-transaction review
  --keywords PATH        Custom path to keywords.json
  --exports-dir DIR      Output directory for HTML dashboards
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
│   └── dashboard.py           # Module 4 — Plotly HTML dashboard
├── config/
│   └── keywords.json          # Category → keyword mapping
└── tests/                     # pytest test suite
```

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
