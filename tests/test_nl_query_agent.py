"""Tests for scripts/nl_query_agent.py — _dispatch() and execute_query_agent()."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nl_query_agent import _dispatch, _ollama_available, execute_query_agent


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": [
            "2024-01-10", "2024-01-15", "2024-02-05",
            "2024-02-12", "2024-03-01", "2024-03-20",
        ],
        "Description": [
            "Whole Foods", "Netflix", "Amazon Purchase",
            "Whole Foods", "Uber Eats", "Salary",
        ],
        "Amount": [-85.0, -15.99, -42.50, -90.0, -25.0, 3000.0],
        "Category": [
            "Groceries", "Subscriptions", "Shopping",
            "Groceries", "Food & Drink", "Income",
        ],
    })


# ---------------------------------------------------------------------------
# _dispatch — all tool cases
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_list_categories(self):
        result = _dispatch("list_categories", {}, _make_df(), "$")
        assert "Groceries" in result
        assert "Subscriptions" in result
        # income row should not appear (positive amount)
        assert "Income" not in result

    def test_show_category(self):
        result = _dispatch("show_category", {"category": "Groceries"}, _make_df(), "$")
        assert "Whole Foods" in result
        assert "Netflix" not in result

    def test_show_category_last_n_months(self):
        result = _dispatch(
            "show_category", {"category": "Groceries", "last_n_months": 1}, _make_df(), "$"
        )
        # Only March data within last 1 month of 2024-03-20 — Groceries has no March row
        assert "no transactions found" in result.lower() or "Whole Foods" not in result

    def test_top_expenses_global(self):
        result = _dispatch("top_expenses", {"n": 2}, _make_df(), "$")
        # Top 2 expenses: -90.0 (Whole Foods Feb) and -85.0 (Whole Foods Jan)
        assert "90.00" in result
        assert "85.00" in result
        assert "15.99" not in result  # Netflix is 3rd

    def test_top_expenses_with_category(self):
        result = _dispatch("top_expenses", {"n": 1, "category": "Groceries"}, _make_df(), "$")
        assert "90.00" in result
        assert "Netflix" not in result

    def test_top_expenses_with_last_n_months(self):
        result = _dispatch("top_expenses", {"n": 3, "last_n_months": 2}, _make_df(), "$")
        # last 2 months from 2024-03-20 = Feb + Mar; Jan excluded
        assert "2024-01" not in result

    def test_sum_category(self):
        result = _dispatch("sum_category", {"category": "Groceries"}, _make_df(), "$")
        assert "175.00" in result  # 85 + 90

    def test_sum_unknown_category(self):
        result = _dispatch("sum_category", {"category": "zzz-unknown"}, _make_df(), "$")
        assert "no expenses found" in result.lower()

    def test_average_category(self):
        result = _dispatch("average_category", {"category": "Groceries"}, _make_df(), "$")
        assert "Monthly average" in result
        assert "$" in result

    def test_monthly_breakdown(self):
        result = _dispatch("monthly_breakdown", {"category": "Groceries"}, _make_df(), "$")
        assert "2024-01" in result
        assert "2024-02" in result

    def test_biggest_expense(self):
        result = _dispatch("biggest_expense", {"category": "Groceries"}, _make_df(), "$")
        assert "90.00" in result
        assert "85.00" not in result  # only the single biggest

    def test_compare_months(self):
        result = _dispatch(
            "compare_months", {"month_a": "2024-01", "month_b": "2024-02"}, _make_df(), "$"
        )
        assert "2024-01" in result
        assert "2024-02" in result

    def test_search_transactions(self):
        result = _dispatch("search_transactions", {"keyword": "whole foods"}, _make_df(), "$")
        assert "Whole Foods" in result
        assert "Netflix" not in result

    def test_filter_date_range(self):
        result = _dispatch(
            "filter_date_range",
            {"start": "2024-02-01", "end": "2024-02-28"},
            _make_df(), "$",
        )
        assert "2024-02" in result
        assert "2024-01" not in result
        assert "2024-03" not in result

    def test_unknown_tool_name(self):
        result = _dispatch("nonexistent_tool", {}, _make_df(), "$")
        assert "Unknown tool" in result

    def test_currency_sym_applied(self):
        result = _dispatch("sum_category", {"category": "Groceries"}, _make_df(), "₹")
        assert "₹" in result


# ---------------------------------------------------------------------------
# _ollama_available
# ---------------------------------------------------------------------------

class TestOllamaAvailable:
    def test_returns_false_when_ollama_not_installed(self):
        with patch.dict("sys.modules", {"ollama": None}):
            assert _ollama_available("llama3.1:8b") is False

    def test_returns_false_when_daemon_unreachable(self):
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value.list.side_effect = ConnectionError("daemon not running")
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert _ollama_available("llama3.1:8b") is False

    def test_returns_false_when_model_not_pulled(self):
        mock_model = MagicMock()
        mock_model.model = "mistral:7b"
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value.list.return_value.models = [mock_model]
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert _ollama_available("llama3.1:8b") is False

    def test_returns_true_when_model_available(self):
        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value.list.return_value.models = [mock_model]
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert _ollama_available("llama3.1:8b") is True

    def test_matches_model_by_base_name(self):
        """'llama3.1:8b' should match a pulled model listed as 'llama3.1:8b-instruct'."""
        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b-instruct"
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value.list.return_value.models = [mock_model]
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert _ollama_available("llama3.1:8b") is True


# ---------------------------------------------------------------------------
# execute_query_agent — fallback behaviour
# ---------------------------------------------------------------------------

class TestExecuteQueryAgentFallback:
    """When Ollama is unavailable, execute_query_agent must produce the same
    output as the regex-based execute_query()."""

    def test_falls_back_when_ollama_unavailable(self):
        from scripts.nl_query import execute_query

        df = _make_df()
        with patch("scripts.nl_query_agent._ollama_available", return_value=False):
            agent_result = execute_query_agent("categories", df)
        regex_result = execute_query("categories", df)
        assert agent_result == regex_result

    def test_fallback_for_top_query(self):
        from scripts.nl_query import execute_query

        df = _make_df()
        with patch("scripts.nl_query_agent._ollama_available", return_value=False):
            agent_result = execute_query_agent("top 2", df)
        regex_result = execute_query("top 2", df)
        assert agent_result == regex_result

    def test_fallback_for_sum_query(self):
        from scripts.nl_query import execute_query

        df = _make_df()
        with patch("scripts.nl_query_agent._ollama_available", return_value=False):
            agent_result = execute_query_agent("sum groceries", df)
        regex_result = execute_query("sum groceries", df)
        assert agent_result == regex_result

    def test_fallback_uses_currency_sym(self):
        df = _make_df()
        with patch("scripts.nl_query_agent._ollama_available", return_value=False):
            result = execute_query_agent("sum groceries", df, currency_sym="₹")
        assert "₹" in result

    def test_agent_falls_back_on_runtime_error(self):
        """If the agent loop raises, it must fall back rather than propagate."""
        from scripts.nl_query import execute_query

        df = _make_df()
        with patch("scripts.nl_query_agent._ollama_available", return_value=True), \
             patch("scripts.nl_query_agent._run_agent", side_effect=RuntimeError("boom")):
            agent_result = execute_query_agent("categories", df)
        regex_result = execute_query("categories", df)
        assert agent_result == regex_result


# ---------------------------------------------------------------------------
# execute_query_agent — missing columns
# ---------------------------------------------------------------------------

class TestExecuteQueryAgentValidation:
    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"Date": ["2024-01-01"], "Amount": [-10.0]})
        with pytest.raises(ValueError, match="missing columns"):
            execute_query_agent("categories", bad_df)

    def test_raises_before_checking_ollama(self):
        """Column validation must fire even if Ollama would be available."""
        bad_df = pd.DataFrame({"Date": ["2024-01-01"]})
        with patch("scripts.nl_query_agent._ollama_available", return_value=True):
            with pytest.raises(ValueError, match="missing columns"):
                execute_query_agent("categories", bad_df)


# ---------------------------------------------------------------------------
# execute_query_agent — with mocked Ollama (agent path)
# ---------------------------------------------------------------------------

class TestExecuteQueryAgentWithMockedOllama:
    """Smoke-test the full agent loop using a mocked Ollama client."""

    def _make_mock_ollama(self, tool_name: str, tool_args: dict, final_content: str):
        """Return a mock ollama module that performs one tool call then answers."""
        # First response: tool call
        tool_call = MagicMock()
        tool_call.function.name = tool_name
        tool_call.function.arguments = tool_args

        first_msg = MagicMock()
        first_msg.tool_calls = [tool_call]
        first_msg.content = ""

        # Second response: final answer
        second_msg = MagicMock()
        second_msg.tool_calls = None
        second_msg.content = final_content

        client = MagicMock()
        client.chat.side_effect = [
            MagicMock(message=first_msg),
            MagicMock(message=second_msg),
        ]

        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = client
        return mock_ollama

    def test_agent_calls_list_categories_tool(self):
        df = _make_df()
        mock_ollama = self._make_mock_ollama(
            "list_categories", {}, "Here are your categories."
        )
        with patch("scripts.nl_query_agent._ollama_available", return_value=True), \
             patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = execute_query_agent("what categories do I have?", df)
        assert result == "Here are your categories."

    def test_agent_calls_sum_category_tool(self):
        df = _make_df()
        mock_ollama = self._make_mock_ollama(
            "sum_category", {"category": "Groceries"}, "Your total is $175."
        )
        with patch("scripts.nl_query_agent._ollama_available", return_value=True), \
             patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = execute_query_agent("how much did I spend on groceries?", df)
        assert result == "Your total is $175."

    def test_agent_returns_no_response_on_empty_content(self):
        tool_call = MagicMock()
        tool_call.function.name = "list_categories"
        tool_call.function.arguments = {}

        first_msg = MagicMock()
        first_msg.tool_calls = [tool_call]
        first_msg.content = ""

        # Final message with empty content and no tool calls
        final_msg = MagicMock()
        final_msg.tool_calls = None
        final_msg.content = ""

        client = MagicMock()
        client.chat.side_effect = [
            MagicMock(message=first_msg),
            MagicMock(message=final_msg),
        ]
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = client

        df = _make_df()
        with patch("scripts.nl_query_agent._ollama_available", return_value=True), \
             patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = execute_query_agent("categories", df)
        assert result == "(no response)"
