"""
Module 10a — Agent-based Natural Language Query Engine
======================================================
Wraps the existing nl_query pipeline functions as Ollama tool calls, enabling
open-ended natural language queries via a fully local LLM.

Falls back to the regex-based ``execute_query()`` from ``nl_query.py`` when:
  - the ``ollama`` package is not installed, or
  - the Ollama daemon is not running, or
  - the requested model is not pulled locally, or
  - the agent loop raises an unexpected error.

No data ever leaves the machine — the LLM runs locally via Ollama.

Usage
-----
    from scripts.nl_query_agent import execute_query_agent
    result = execute_query_agent("which category cost me the most last 3 months?", df)

Supported models (tool-calling capable)
---------------------------------------
    llama3.1:8b   (default, ~4.7 GB)
    qwen2.5:7b    (~4.4 GB)
    mistral:7b    (~4.1 GB)

Install Ollama: https://ollama.com
Pull a model:  ollama pull llama3.1:8b
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from scripts.nl_query import (
    _filter_between_dates,
    _filter_by_category,
    _filter_last_n_months,
    _fmt_average,
    _fmt_categories,
    _fmt_compare,
    _fmt_monthly,
    _fmt_search,
    _fmt_sum,
    _fmt_transactions,
    _top_n_expenses,
    execute_query,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1:8b"
_MAX_ROUNDS = 6  # max tool-call iterations before giving up

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": (
                "List all spending categories with their total spend. "
                "Call this first whenever you are unsure of exact category names."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_category",
            "description": "Show all transactions for a specific category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Exact category name (case-insensitive). Call list_categories first if unsure.",
                    },
                    "last_n_months": {
                        "type": "integer",
                        "description": "If provided, restrict to the most recent N months of data.",
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_expenses",
            "description": "Return the N largest expenses, optionally filtered to a category and/or recent months.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of top expenses to return."},
                    "category": {"type": "string", "description": "Optional — filter to this category first."},
                    "last_n_months": {"type": "integer", "description": "Optional — restrict to the most recent N months."},
                },
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sum_category",
            "description": "Return the total spend for a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "last_n_months": {"type": "integer"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "average_category",
            "description": "Return the average monthly spend for a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "last_n_months": {"type": "integer"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "monthly_breakdown",
            "description": "Return a month-by-month spend breakdown for a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "last_n_months": {"type": "integer"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "biggest_expense",
            "description": "Return the single largest expense in a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "last_n_months": {"type": "integer"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_months",
            "description": "Side-by-side category spend comparison for two months.",
            "parameters": {
                "type": "object",
                "properties": {
                    "month_a": {"type": "string", "description": "First month in YYYY-MM format."},
                    "month_b": {"type": "string", "description": "Second month in YYYY-MM format."},
                },
                "required": ["month_a", "month_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transactions",
            "description": "Search for transactions containing a keyword in their description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "last_n_months": {"type": "integer"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_date_range",
            "description": "Return all transactions between two dates (inclusive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date in YYYY-MM-DD format."},
                    "end": {"type": "string", "description": "End date in YYYY-MM-DD format."},
                },
                "required": ["start", "end"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch(name: str, args: dict[str, Any], df: pd.DataFrame, currency_sym: str) -> str:
    """Execute a named tool and return a formatted string result."""
    months = args.get("last_n_months")

    match name:
        case "list_categories":
            return _fmt_categories(df, currency_sym)

        case "show_category":
            scope = _filter_last_n_months(df, months) if months else df
            subset = _filter_by_category(scope, args["category"])
            title = f"Transactions — {args['category'].title()}"
            if months:
                title += f" (last {months} months)"
            return _fmt_transactions(subset, title, currency_sym)

        case "top_expenses":
            scope = _filter_last_n_months(df, months) if months else df
            cat = args.get("category")
            if cat:
                scope = _filter_by_category(scope, cat)
            title = f"Top {args['n']} expenses"
            if cat:
                title += f" — {cat.title()}"
            if months:
                title += f" (last {months} months)"
            return _fmt_transactions(_top_n_expenses(scope, args["n"]), title, currency_sym)

        case "sum_category":
            scope = _filter_last_n_months(df, months) if months else df
            return _fmt_sum(scope, args["category"], currency_sym)

        case "average_category":
            scope = _filter_last_n_months(df, months) if months else df
            return _fmt_average(scope, args["category"], currency_sym)

        case "monthly_breakdown":
            scope = _filter_last_n_months(df, months) if months else df
            return _fmt_monthly(scope, args["category"], currency_sym)

        case "biggest_expense":
            scope = _filter_last_n_months(df, months) if months else df
            subset = _filter_by_category(scope, args["category"])
            return _fmt_transactions(
                _top_n_expenses(subset, 1),
                f"Biggest expense — {args['category'].title()}",
                currency_sym,
            )

        case "compare_months":
            return _fmt_compare(df, args["month_a"], args["month_b"], currency_sym)

        case "search_transactions":
            scope = _filter_last_n_months(df, months) if months else df
            return _fmt_search(scope, args["keyword"], currency_sym)

        case "filter_date_range":
            subset = _filter_between_dates(df, args["start"], args["end"])
            return _fmt_transactions(
                subset,
                f"Transactions {args['start']} to {args['end']}",
                currency_sym,
            )

        case _:
            return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Ollama availability check
# ---------------------------------------------------------------------------

def _ollama_available(model: str) -> bool:
    """Return True if Ollama is reachable and *model* is pulled locally."""
    try:
        import ollama  # noqa: F401
        client = ollama.Client()
        pulled = [m.model for m in client.list().models]
        model_base = model.split(":")[0]
        return any(m.startswith(model_base) for m in pulled)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

def _run_agent(query: str, df: pd.DataFrame, currency_sym: str, model: str) -> str:
    """Send the query to Ollama, execute tool calls, and return the final answer."""
    import ollama

    categories = sorted(df["Category"].unique().tolist())
    date_range = f"{df['Date'].min()} to {df['Date'].max()}"

    system = (
        "You are a personal finance assistant. Answer questions about the user's "
        "bank transactions using the available tools.\n\n"
        f"Available categories (use exact names): {categories}\n"
        f"Data covers: {date_range}\n\n"
        "Rules:\n"
        "- Always use a tool to retrieve data; never invent numbers.\n"
        "- When unsure of a category name, call list_categories first.\n"
        "- Return tool output verbatim — do not paraphrase transaction tables.\n"
        "- If a query is ambiguous, make a reasonable assumption and state it."
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    client = ollama.Client()

    for _ in range(_MAX_ROUNDS):
        response = client.chat(model=model, messages=messages, tools=_TOOLS)
        msg = response.message

        if not msg.tool_calls:
            return msg.content or "(no response)"

        # Append assistant turn (Ollama requires content field even if empty)
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            result = _dispatch(tc.function.name, tc.function.arguments, df, currency_sym)
            messages.append({"role": "tool", "content": result})

    return "(agent reached max iterations without producing a final answer)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query_agent(
    query: str,
    df: pd.DataFrame,
    currency_sym: str = "$",
    model: str = _DEFAULT_MODEL,
) -> str:
    """Execute a natural-language query using a local Ollama LLM as the reasoner.

    The LLM decides which pipeline functions to call based on the query, making
    this engine capable of handling questions that fall outside the fixed regex
    patterns in ``nl_query.execute_query()``.

    Falls back to the regex engine automatically when Ollama is unavailable.

    Parameters
    ----------
    query : str
        A plain-English question about the transaction data.
    df : pd.DataFrame
        Must have columns: Date, Description, Amount, Category.
    currency_sym : str
        Currency symbol used when formatting amounts (e.g. ``"$"``, ``"₹"``).
    model : str
        Ollama model name. Must support tool/function calling.
        Defaults to ``"llama3.1:8b"``.

    Returns
    -------
    str
        Formatted, human-readable result.

    Raises
    ------
    ValueError
        If required columns are missing from *df*.
    """
    required = {"Date", "Description", "Amount", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"execute_query_agent: DataFrame is missing columns: {sorted(missing)}"
        )

    if not _ollama_available(model):
        logger.info(
            "Ollama not available (model=%r) — falling back to regex query engine.", model
        )
        return execute_query(query, df, currency_sym)

    try:
        return _run_agent(query, df, currency_sym, model)
    except Exception as exc:
        logger.warning("Agent error (%s) — falling back to regex query engine.", exc)
        return execute_query(query, df, currency_sym)
