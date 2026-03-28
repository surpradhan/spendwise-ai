"""
Module 4 — HTML Dashboard Generator
=====================================
Generates a fully self-contained, single-file HTML export with six
interactive Plotly views.  The file opens offline in any modern browser.

Views
-----
1. Donut chart         — Spending by Category
2. Monthly trend       — Expenses bar + Income line
3. Top Merchants       — Horizontal bar chart (top 10)
4. Income vs Expenses  — Grouped bar by month
5. Uncategorized table — Filterable review table
6. Recurring table     — Recurring transactions detected
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Colour palette — consistent across all charts
_PALETTE = [
    "#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0",
    "#4895EF", "#560BAD", "#B5179E", "#F3722C", "#90BE6D",
]


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Spending by Category as a donut (Pie with hole=0.5).

    Parameters
    ----------
    df : pd.DataFrame  (must have Amount, Category columns)

    Returns
    -------
    go.Figure
    """
    expense_df = df[df["Amount"] < 0].copy()
    expense_df["AbsAmount"] = expense_df["Amount"].abs()
    cat_totals = (
        expense_df.groupby("Category")["AbsAmount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = go.Figure(
        go.Pie(
            labels=cat_totals["Category"],
            values=cat_totals["AbsAmount"],
            hole=0.5,
            marker=dict(colors=_PALETTE),
            hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
            textinfo="label+percent",
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="Spending by Category", font=dict(size=16)),
        showlegend=False,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def build_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Total expenses per month as bars; income as an overlay line.

    Parameters
    ----------
    df : pd.DataFrame  (must have Date, Amount columns)

    Returns
    -------
    go.Figure
    """
    tmp = df.copy()
    tmp["Month"] = pd.to_datetime(tmp["Date"]).dt.to_period("M").astype(str)

    monthly_expense = (
        tmp[tmp["Amount"] < 0]
        .groupby("Month")["Amount"]
        .sum()
        .abs()
        .reset_index(name="Expenses")
    )
    monthly_income = (
        tmp[tmp["Amount"] > 0]
        .groupby("Month")["Amount"]
        .sum()
        .reset_index(name="Income")
    )
    monthly = monthly_expense.merge(monthly_income, on="Month", how="outer").fillna(0)
    monthly = monthly.sort_values("Month")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["Month"],
            y=monthly["Expenses"],
            name="Expenses",
            marker_color="#F72585",
            hovertemplate="<b>%{x}</b><br>Expenses: $%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["Month"],
            y=monthly["Income"],
            name="Income",
            mode="lines+markers",
            line=dict(color="#4CC9F0", width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Income: $%{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=dict(text="Monthly Spending Trend", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=60, r=60),
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    fig.update_yaxes(title_text="Expenses ($)", secondary_y=False)
    fig.update_yaxes(title_text="Income ($)", secondary_y=True)
    return fig


def build_top_merchants(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart — top 10 merchants by cumulative spend.

    Parameters
    ----------
    df : pd.DataFrame  (must have Description, Amount columns)

    Returns
    -------
    go.Figure
    """
    expense_df = df[df["Amount"] < 0].copy()
    expense_df["AbsAmount"] = expense_df["Amount"].abs()
    top10 = (
        expense_df.groupby("Description")["AbsAmount"]
        .sum()
        .sort_values(ascending=True)
        .tail(10)
        .reset_index()
    )
    top10["ShortName"] = top10["Description"].str[:40]

    fig = go.Figure(
        go.Bar(
            x=top10["AbsAmount"],
            y=top10["ShortName"],
            orientation="h",
            marker=dict(
                color=top10["AbsAmount"],
                colorscale=[[0, "#4361EE"], [1, "#F72585"]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>",
            text=top10["AbsAmount"].map(lambda v: f"${v:,.2f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="Top 10 Merchants", font=dict(size=16)),
        xaxis=dict(title="Total Spent ($)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=False),
        margin=dict(t=60, b=40, l=200, r=80),
        plot_bgcolor="white",
    )
    return fig


def build_income_vs_expenses(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart — income and expenses side by side per month.

    Parameters
    ----------
    df : pd.DataFrame  (must have Date, Amount columns)

    Returns
    -------
    go.Figure
    """
    tmp = df.copy()
    tmp["Month"] = pd.to_datetime(tmp["Date"]).dt.to_period("M").astype(str)

    monthly_expense = (
        tmp[tmp["Amount"] < 0]
        .groupby("Month")["Amount"]
        .sum()
        .abs()
        .reset_index(name="Expenses")
    )
    monthly_income = (
        tmp[tmp["Amount"] > 0]
        .groupby("Month")["Amount"]
        .sum()
        .reset_index(name="Income")
    )
    monthly = monthly_expense.merge(monthly_income, on="Month", how="outer").fillna(0)
    monthly = monthly.sort_values("Month")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Income",
            x=monthly["Month"],
            y=monthly["Income"],
            marker_color="#4CC9F0",
            hovertemplate="<b>%{x}</b><br>Income: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Expenses",
            x=monthly["Month"],
            y=monthly["Expenses"],
            marker_color="#F72585",
            hovertemplate="<b>%{x}</b><br>Expenses: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Income vs Expenses by Month", font=dict(size=16)),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=60, r=20),
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Amount ($)", gridcolor="#f0f0f0"),
    )
    return fig


def build_uncategorized_table(df: pd.DataFrame) -> go.Figure:
    """Plotly Table of all uncategorized transactions.

    Parameters
    ----------
    df : pd.DataFrame  (must have Date, Description, Amount, Category)

    Returns
    -------
    go.Figure
    """
    uncat = df[df["Category"] == "Uncategorized"][
        ["Date", "Description", "Amount"]
    ].copy()
    uncat["Amount"] = uncat["Amount"].map(lambda v: f"${v:,.2f}")

    if uncat.empty:
        # Return a simple info table
        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["Status"],
                    fill_color="#4361EE",
                    font=dict(color="white", size=13),
                    align="center",
                ),
                cells=dict(
                    values=[["✓ All transactions are categorized!"]],
                    fill_color="white",
                    align="center",
                ),
            )
        )
    else:
        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["Date", "Description", "Amount"],
                    fill_color="#4361EE",
                    font=dict(color="white", size=13),
                    align=["center", "left", "right"],
                    height=36,
                ),
                cells=dict(
                    values=[
                        uncat["Date"].tolist(),
                        uncat["Description"].tolist(),
                        uncat["Amount"].tolist(),
                    ],
                    fill_color=[["#f9f9fb" if i % 2 == 0 else "white" for i in range(len(uncat))]],
                    align=["center", "left", "right"],
                    height=30,
                    font=dict(size=12),
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Uncategorized Transactions (corrections persist on next run)",
            font=dict(size=16),
        ),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Recurring transactions table
# ---------------------------------------------------------------------------

def build_recurring_table(df: pd.DataFrame) -> go.Figure:
    """Plotly Table of recurring transactions detected in *df*.

    Calls :func:`scripts.recurring.detect_recurring` internally.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Date, Description, Amount, Category.

    Returns
    -------
    go.Figure
    """
    from scripts.recurring import detect_recurring

    rec = detect_recurring(df)

    if rec.empty:
        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["Status"],
                    fill_color="#4361EE",
                    font=dict(color="white", size=13),
                    align="center",
                ),
                cells=dict(
                    values=[["✓ No recurring transactions detected."]],
                    fill_color="white",
                    align="center",
                ),
            )
        )
    else:
        amounts = rec["Avg_Amount"].map(lambda v: f"${abs(v):,.2f}/cycle")
        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["Description", "Category", "Avg Amount",
                            "Frequency", "Occurrences", "Last Date"],
                    fill_color="#4361EE",
                    font=dict(color="white", size=13),
                    align=["left", "left", "right", "center", "center", "center"],
                    height=36,
                ),
                cells=dict(
                    values=[
                        rec["Description"].str[:40].tolist(),
                        rec["Category"].tolist(),
                        amounts.tolist(),
                        rec["Frequency"].tolist(),
                        rec["Occurrences"].tolist(),
                        rec["Last_Date"].tolist(),
                    ],
                    fill_color=[
                        ["#f9f9fb" if i % 2 == 0 else "white"
                         for i in range(len(rec))]
                    ],
                    align=["left", "left", "right", "center", "center", "center"],
                    height=30,
                    font=dict(size=12),
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Recurring Transactions",
            font=dict(size=16),
        ),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Budget chart
# ---------------------------------------------------------------------------

def build_budget_chart(df: pd.DataFrame, budgets: dict[str, float]) -> go.Figure:
    """Horizontal grouped bar chart: actual monthly average vs budget target.

    Bars are coloured by status:
    * Red  (#F72585) — actual exceeds budget
    * Amber (#F3722C) — actual is approaching budget (≥ 80 %)
    * Green (#90BE6D) — on track

    Parameters
    ----------
    df : pd.DataFrame
        Must have Date, Amount, Category columns.
    budgets : dict[str, float]
        Mapping of category → monthly budget.  Only categories present in
        this dict are included.  When empty, returns a figure with a
        "No budgets configured" annotation.

    Returns
    -------
    go.Figure
    """
    if not budgets:
        fig = go.Figure()
        fig.add_annotation(
            text="No budgets configured",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888"),
        )
        fig.update_layout(
            title=dict(text="Budget Targets vs Actual Spending", font=dict(size=16)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=60, b=20, l=20, r=20),
        )
        return fig

    # Compute monthly averages per category from the DataFrame
    tmp = df.copy()
    tmp["Month"] = pd.to_datetime(tmp["Date"]).dt.to_period("M").astype(str)
    num_months = tmp["Month"].nunique()
    num_months = max(num_months, 1)

    expense_df = tmp[tmp["Amount"] < 0].copy()
    expense_df["AbsAmount"] = expense_df["Amount"].abs()
    cat_totals = (
        expense_df.groupby("Category")["AbsAmount"]
        .sum()
        .to_dict()
    )

    categories = sorted(budgets.keys())
    monthly_avgs = [cat_totals.get(cat, 0.0) / num_months for cat in categories]
    budget_vals  = [budgets[cat] for cat in categories]

    # Determine per-category status colours
    colours = []
    for avg, bud in zip(monthly_avgs, budget_vals):
        pct = avg / bud
        if pct >= 1.0:
            colours.append("#F72585")   # exceeded — red
        elif pct >= 0.80:
            colours.append("#F3722C")   # approaching — amber
        else:
            colours.append("#90BE6D")   # on track — green

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Actual (monthly avg)",
            x=monthly_avgs,
            y=categories,
            orientation="h",
            marker_color=colours,
            hovertemplate="<b>%{y}</b><br>Actual: $%{x:,.2f}<extra></extra>",
            text=[f"${v:,.2f}" for v in monthly_avgs],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Budget target",
            x=budget_vals,
            y=categories,
            orientation="h",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color="#4361EE", width=2)),
            hovertemplate="<b>%{y}</b><br>Budget: $%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Budget Targets vs Actual Spending", font=dict(size=16)),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Amount ($)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=False),
        margin=dict(t=80, b=40, l=160, r=80),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Dashboard assembly & export
# ---------------------------------------------------------------------------

def build_dashboard(df: pd.DataFrame, budgets: dict | None = None) -> str:
    """Compose all chart views into a single self-contained HTML string.

    When *budgets* is provided and non-empty a 7th card is added showing the
    budget targets vs actual spending chart.  When absent or empty the layout
    is unchanged (backward compatible).

    Parameters
    ----------
    df : pd.DataFrame
        Must be a fully ingested + classified DataFrame.
    budgets : dict | None, optional
        Mapping of category → monthly budget from
        :func:`scripts.budget.load_budgets`.  Pass ``None`` or ``{}`` to
        omit the budget card.

    Returns
    -------
    str
        Complete HTML document with embedded Plotly JS.
    """
    donut     = build_donut_chart(df)
    trend     = build_monthly_trend(df)
    merch     = build_top_merchants(df)
    grouped   = build_income_vs_expenses(df)
    table     = build_uncategorized_table(df)
    recurring = build_recurring_table(df)

    dates = pd.to_datetime(df["Date"])
    start = dates.min().strftime("%Y-%m-%d")
    end   = dates.max().strftime("%Y-%m-%d")

    import plotly.io as pio

    # Convert each figure to a div (no CDN/JS here — we'll inject once below)
    def _div(fig: go.Figure) -> str:
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    donut_div     = _div(donut)
    trend_div     = _div(trend)
    merch_div     = _div(merch)
    grouped_div   = _div(grouped)
    table_div     = _div(table)
    recurring_div = _div(recurring)

    # Optional budget card (7th card) — only when budgets provided and non-empty
    budget_card_html = ""
    if budgets:
        budget_div = _div(build_budget_chart(df, budgets))
        budget_card_html = f'    <div class="card full-width">{budget_div}</div>'

    # Inline Plotly.js for full offline support
    plotly_js = pio.to_html(
        go.Figure(), full_html=False, include_plotlyjs=True
    )
    # Extract just the <script> tag containing the bundle
    import re
    js_match = re.search(
        r'(<script type="text/javascript">[\s\S]*?window\.Plotly[\s\S]*?</script>)',
        plotly_js,
    )
    plotly_script = js_match.group(1) if js_match else (
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SpendWise AI — {start} to {end}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f4f6fb;
      color: #1a1a2e;
    }}
    header {{
      background: linear-gradient(135deg, #4361EE 0%, #3A0CA3 100%);
      color: white;
      padding: 24px 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }}
    header p  {{ font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }}
    .badge {{
      background: rgba(255,255,255,0.2);
      border-radius: 20px;
      padding: 6px 16px;
      font-size: 0.85rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      padding: 24px 32px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
      padding: 8px;
      overflow: hidden;
    }}
    .card.full-width {{ grid-column: 1 / -1; }}
    footer {{
      text-align: center;
      padding: 20px;
      font-size: 0.8rem;
      color: #888;
    }}
    @media (max-width: 768px) {{
      .grid {{ grid-template-columns: 1fr; padding: 12px; }}
    }}
  </style>
  {plotly_script}
</head>
<body>
  <header>
    <div>
      <h1>SpendWise AI</h1>
      <p>Spending Summary</p>
    </div>
    <div class="badge">📅 {start} → {end}</div>
  </header>

  <div class="grid">
    <div class="card">{donut_div}</div>
    <div class="card">{trend_div}</div>
    <div class="card">{merch_div}</div>
    <div class="card">{grouped_div}</div>
    <div class="card full-width">{table_div}</div>
    <div class="card full-width">{recurring_div}</div>
{budget_card_html}
  </div>

  <footer>Generated by SpendWise AI · All data processed locally · No data leaves your machine</footer>
</body>
</html>"""
    return html


def export_pdf(
    df: pd.DataFrame,
    output_dir: str | Path,
    budgets: dict | None = None,
) -> Path:
    """Build and write a multi-page PDF dashboard to *output_dir*.

    Each Plotly chart is rendered to PNG via kaleido and assembled into a
    single PDF with reportlab.  Page 1 is a summary cover with key stats
    and a category breakdown table; pages 2–5 are the four main charts;
    optional final pages cover uncategorized and recurring transactions.
    When *budgets* is provided and non-empty, a budget targets chart page
    is appended.

    File is named: dashboard_{start}_to_{end}.pdf

    Parameters
    ----------
    df : pd.DataFrame
        Must be a fully ingested + classified DataFrame (Date, Description,
        Amount, Category columns).
    output_dir : str | Path
    budgets : dict | None, optional
        Mapping of category → monthly budget.  Adds a budget chart page
        when provided and non-empty.

    Returns
    -------
    Path
        Path of the written PDF file.

    Raises
    ------
    RuntimeError
        If reportlab or kaleido is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Image, Paragraph, Spacer,
            Table, TableStyle, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER
    except ImportError as exc:
        raise RuntimeError(
            "reportlab is required for PDF export. "
            "Install with: pip install reportlab"
        ) from exc

    import io as _io
    import plotly.io as pio

    def _to_png(fig: go.Figure, width: int = 1200, height: int = 630) -> bytes:
        """Render a Plotly figure to PNG bytes via kaleido."""
        try:
            return pio.to_image(fig, format="png", width=width, height=height, scale=2)
        except ValueError as exc:
            if "kaleido" in str(exc).lower():
                raise RuntimeError(
                    "kaleido is required to render charts for PDF export. "
                    "Install with: pip install kaleido"
                ) from exc
            raise

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(df["Date"])
    start = dates.min().strftime("%Y-%m-%d")
    end   = dates.max().strftime("%Y-%m-%d")
    filename = f"dashboard_{start}_to_{end}.pdf"
    out_path = output_dir / filename

    print("\n[PDF] Building PDF dashboard…")

    # ── Styles ────────────────────────────────────────────────────────────
    page_w, page_h = landscape(A4)
    margin = 1.5 * cm
    avail_w = page_w - 2 * margin

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=landscape(A4),
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
    )

    styles = getSampleStyleSheet()
    PRIMARY   = HexColor("#4361EE")
    SECONDARY = HexColor("#3A0CA3")
    LIGHT_BG  = HexColor("#f4f6fb")
    BORDER    = HexColor("#e0e0e0")
    GREY_TEXT = HexColor("#888888")
    WHITE     = HexColor("#ffffff")
    ROW_ALT   = HexColor("#f9f9fb")

    title_style = ParagraphStyle(
        "PDFTitle", parent=styles["Title"],
        textColor=PRIMARY, fontSize=28, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "PDFSubtitle", parent=styles["Normal"],
        textColor=SECONDARY, fontSize=13, spaceAfter=20,
    )
    section_style = ParagraphStyle(
        "PDFSection", parent=styles["Heading2"],
        textColor=PRIMARY, fontSize=13, spaceBefore=10, spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "PDFLabel", parent=styles["Normal"],
        fontSize=9, textColor=GREY_TEXT, alignment=TA_CENTER,
    )
    value_style = ParagraphStyle(
        "PDFValue", parent=styles["Normal"],
        fontSize=15, fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    footer_style = ParagraphStyle(
        "PDFFooter", parent=styles["Normal"],
        fontSize=8, textColor=GREY_TEXT, alignment=TA_CENTER,
    )
    _FOOTER_TEXT = (
        "Generated by SpendWise AI · All data processed locally · "
        "No data leaves your machine"
    )

    # ── Summary stats ─────────────────────────────────────────────────────
    total_expenses = df[df["Amount"] < 0]["Amount"].sum()
    total_income   = df[df["Amount"] > 0]["Amount"].sum()
    net            = total_income + total_expenses
    n_transactions = len(df)
    n_uncat        = int((df["Category"] == "Uncategorized").sum())
    sign           = "+" if net > 0 else ("-" if net < 0 else "")

    # ── Category breakdown ────────────────────────────────────────────────
    expense_df = df[df["Amount"] < 0].copy()
    expense_df["AbsAmount"] = expense_df["Amount"].abs()
    cat_totals = (
        expense_df.groupby("Category")["AbsAmount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    grand_total = cat_totals["AbsAmount"].sum()

    # ── Story ─────────────────────────────────────────────────────────────
    story: list = []

    # Page 1 — Cover / summary
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("SpendWise AI", title_style))
    story.append(Paragraph(f"Spending Summary · {start} → {end}", subtitle_style))
    story.append(Spacer(1, 0.4 * cm))

    col_w = avail_w / 5
    stats_data = [
        [Paragraph(h, label_style) for h in (
            "Total Income", "Total Expenses", "Net",
            "Transactions", "Uncategorized",
        )],
        [
            Paragraph(f"${total_income:,.2f}", value_style),
            Paragraph(f"${abs(total_expenses):,.2f}", value_style),
            Paragraph(f"{sign}${abs(net):,.2f}", value_style),
            Paragraph(str(n_transactions), value_style),
            Paragraph(str(n_uncat), value_style),
        ],
    ]
    stats_tbl = Table(stats_data, colWidths=[col_w] * 5, rowHeights=[18, 28])
    stats_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), LIGHT_BG),
        ("BACKGROUND",    (0, 1), (-1, 1), WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(stats_tbl)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Spending by Category", section_style))
    cat_col_w = [avail_w * r for r in (0.50, 0.25, 0.25)]
    cat_data = [["Category", "Amount", "% of Total"]] + [
        [
            row["Category"],
            f"${row['AbsAmount']:,.2f}",
            f"{row['AbsAmount'] / grand_total * 100:.1f}%" if grand_total else "—",
        ]
        for _, row in cat_totals.iterrows()
    ]
    cat_tbl = Table(cat_data, colWidths=cat_col_w)
    cat_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_ALT, WHITE]),
        ("ALIGN",          (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("LINEBELOW",      (0, 0), (-1, 0), 0.5, BORDER),
    ]))
    story.append(cat_tbl)
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(_FOOTER_TEXT, footer_style))

    # Pages 2–5 — one chart per page
    IMG_W_PX, IMG_H_PX = 1200, 630
    img_aspect = IMG_H_PX / IMG_W_PX
    img_pdf_w  = avail_w
    img_pdf_h  = avail_w * img_aspect

    chart_pages = [
        ("Spending by Category",       build_donut_chart(df)),
        ("Monthly Spending Trend",     build_monthly_trend(df)),
        ("Top 10 Merchants",           build_top_merchants(df)),
        ("Income vs Expenses",         build_income_vs_expenses(df)),
    ]
    for title, fig in chart_pages:
        story.append(PageBreak())
        story.append(Paragraph(title, section_style))
        story.append(Spacer(1, 0.2 * cm))
        png_bytes = _to_png(fig, width=IMG_W_PX, height=IMG_H_PX)
        img = Image(_io.BytesIO(png_bytes), width=img_pdf_w, height=img_pdf_h)
        story.append(img)
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(_FOOTER_TEXT, footer_style))

    # Optional page — uncategorized transactions
    uncat_df = df[df["Category"] == "Uncategorized"][
        ["Date", "Description", "Amount"]
    ].copy()
    if not uncat_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("Uncategorized Transactions", section_style))
        story.append(Spacer(1, 0.3 * cm))
        uncat_col_w = [avail_w * r for r in (0.14, 0.65, 0.21)]
        uncat_data = [["Date", "Description", "Amount"]] + [
            [row["Date"], row["Description"], f"${row['Amount']:,.2f}"]
            for _, row in uncat_df.iterrows()
        ]
        uncat_tbl = Table(uncat_data, colWidths=uncat_col_w)
        uncat_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_ALT, WHITE]),
            ("ALIGN",          (2, 0), (2, -1), "RIGHT"),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(uncat_tbl)
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(_FOOTER_TEXT, footer_style))

    # Optional page — recurring transactions
    from scripts.recurring import detect_recurring
    recurring_df = detect_recurring(df)

    if not recurring_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("Recurring Transactions Detected", section_style))
        story.append(Spacer(1, 0.3 * cm))
        rec_col_w = [avail_w * r for r in (0.30, 0.14, 0.14, 0.14, 0.14, 0.14)]
        rec_headers = ["Description", "Category", "Avg Amount", "Frequency",
                       "Occurrences", "Last Date"]
        rec_data = [rec_headers] + [
            [
                row["Description"][:40],
                row["Category"],
                f"${abs(row['Avg_Amount']):,.2f}",
                row["Frequency"],
                str(row["Occurrences"]),
                row["Last_Date"],
            ]
            for _, row in recurring_df.iterrows()
        ]
        rec_tbl = Table(rec_data, colWidths=rec_col_w)
        rec_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_ALT, WHITE]),
            ("ALIGN",          (2, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_tbl)
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(_FOOTER_TEXT, footer_style))

    # Optional page — budget targets chart
    if budgets:
        story.append(PageBreak())
        story.append(Paragraph("Budget Targets vs Actual Spending", section_style))
        story.append(Spacer(1, 0.2 * cm))
        budget_fig = build_budget_chart(df, budgets)
        png_bytes = _to_png(budget_fig, width=IMG_W_PX, height=IMG_H_PX)
        img = Image(_io.BytesIO(png_bytes), width=img_pdf_w, height=img_pdf_h)
        story.append(img)
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(_FOOTER_TEXT, footer_style))

    doc.build(story)

    size_kb = out_path.stat().st_size // 1024
    print(f"  ✓ PDF dashboard saved → '{out_path}'  ({size_kb:,} KB)")
    return out_path


def export_dashboard(
    df: pd.DataFrame,
    output_dir: str | Path,
    budgets: dict | None = None,
) -> Path:
    """Build and write the HTML dashboard to *output_dir*.

    File is named: dashboard_{start}_to_{end}.html

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str | Path
    budgets : dict | None, optional
        Mapping of category → monthly budget.  When provided and non-empty,
        a 7th budget-targets card is added to the dashboard.

    Returns
    -------
    Path
        Path of the written HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(df["Date"])
    start = dates.min().strftime("%Y-%m-%d")
    end   = dates.max().strftime("%Y-%m-%d")
    filename = f"dashboard_{start}_to_{end}.html"
    out_path = output_dir / filename

    print("\n[Dashboard] Building interactive HTML dashboard…")
    html = build_dashboard(df, budgets=budgets)

    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"  ✓ Dashboard saved → '{out_path}'  ({size_kb:,} KB)")
    return out_path
