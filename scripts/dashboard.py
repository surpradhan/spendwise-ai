"""
Module 4 — HTML Dashboard Generator
=====================================
Generates a fully self-contained, single-file HTML export with five
interactive Plotly views.  The file opens offline in any modern browser.

Views
-----
1. Donut chart  — Spending by Category
2. Monthly trend  — Expenses bar + Income line
3. Top Merchants  — Horizontal bar chart (top 10)
4. Income vs Expenses  — Grouped bar by month
5. Uncategorized table  — Filterable review table
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
# Dashboard assembly & export
# ---------------------------------------------------------------------------

def build_dashboard(df: pd.DataFrame) -> str:
    """Compose all 5 views into a single self-contained HTML string.

    Returns
    -------
    str
        Complete HTML document with embedded Plotly JS.
    """
    donut   = build_donut_chart(df)
    trend   = build_monthly_trend(df)
    merch   = build_top_merchants(df)
    grouped = build_income_vs_expenses(df)
    table   = build_uncategorized_table(df)

    dates = pd.to_datetime(df["Date"])
    start = dates.min().strftime("%Y-%m-%d")
    end   = dates.max().strftime("%Y-%m-%d")

    import plotly.io as pio

    # Convert each figure to a div (no CDN/JS here — we'll inject once below)
    def _div(fig: go.Figure) -> str:
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    donut_div   = _div(donut)
    trend_div   = _div(trend)
    merch_div   = _div(merch)
    grouped_div = _div(grouped)
    table_div   = _div(table)

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
  </div>

  <footer>Generated by SpendWise AI · All data processed locally · No data leaves your machine</footer>
</body>
</html>"""
    return html


def export_dashboard(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Build and write the HTML dashboard to *output_dir*.

    File is named: dashboard_{start}_to_{end}.html

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str | Path

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
    html = build_dashboard(df)

    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"  ✓ Dashboard saved → '{out_path}'  ({size_kb:,} KB)")
    return out_path
