# SpendWise AI — Roadmap

Current release: **v3.2** (MoM comparison + expanded NL queries)

---

## v3.2 — Richer Analysis ✓ DONE

**Goal:** More insight from data you already have, with no new dependencies.

| Feature | Module | Notes |
|---------|--------|-------|
| Month-over-month comparison | `terminal_output.py`, `dashboard.py` | Per-category spend change with ↑/↓ arrows in terminal; grouped bar chart in dashboard (9th card). |
| Expanded NL query patterns | `nl_query.py` | Added `average <category>`, `compare <YYYY-MM> vs <YYYY-MM>`, `biggest <category>`, `between <date> and <date>`. |
| `--summary-only` flag | `main.py` | Skips recurring, budget, anomaly, NL query, and dashboard/PDF. Prints summary and exits. |

---

## v3.3 — More Bank Adapters

**Goal:** Reduce friction for users whose bank isn't HDFC.

Each adapter is a self-contained file following the documented pattern in `scripts/adapters/`.

| Adapter | Key columns to handle |
|---------|-----------------------|
| Chase (US) | `Transaction Date`, `Post Date`, `Description`, `Category`, `Type`, `Amount` |
| Wells Fargo (US) | `Date`, `Amount`, `*`, `*`, `Description` (positional CSV, no header) |
| Barclays (UK) | `Date`, `Memo`, `Amount` with GBP currency |
| SBI (India) | `Txn Date`, `Description`, `Ref No./Cheque No.`, `Debit`, `Credit`, `Balance` |

---

## v4.0 — Forecasting & Goals

**Goal:** Forward-looking features — predict future spend and track savings targets.

| Feature | New module | Notes |
|---------|-----------|-------|
| Spending forecast | `scripts/forecast.py` | Linear regression + exponential smoothing on monthly category totals. CLI flag: `--forecast 3` (project 3 months forward). Dashboard card with projected vs budget. |
| Savings goal tracking | `scripts/goals.py` | `config/goals.json` maps goal name → target amount + deadline. Dashboard card shows % progress. CLI: `--set-goal "Emergency Fund:5000:2026-12-31"`. |
| Forecast chart in dashboard | `dashboard.py` | Line chart: actuals + projected trend per category, shaded confidence band. |

---

## v5.0 — Web UI (in progress)

**Goal:** Make the tool usable without a terminal.

| Feature | Status | Notes |
|---------|--------|-------|
| Minimal web front-end | ✓ **Shipped** | `app.py` — FastAPI + uvicorn. File upload, summary, NL query, and dashboard all accessible from `http://localhost:8000`. |
| Interactive category review | Pending | Replace CLI prompts with an in-browser table for mapping uncategorised transactions. |
| Persistent session | Pending | Store processed CSVs and model artefacts between uploads; no database — flat files only. |

---

## Ongoing

| Area | Detail |
|------|--------|
| Test coverage | `anomaly.py`, `nl_query.py`, and all adapters have minimal tests. Target: 80 % line coverage across all modules. |
| Python 3.12 compatibility | Verify and pin CI against 3.10, 3.11, and 3.12. |
| `keywords.json` starter pack | Ship a richer default keyword set covering common US, UK, and India merchants out of the box. |
