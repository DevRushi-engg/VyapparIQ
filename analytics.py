"""Local, auditable business calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def calculate_total_revenue(data: pd.DataFrame) -> float:
    return float(data["Amount_INR"].sum())


def calculate_transaction_count(data: pd.DataFrame) -> int:
    return int(len(data))


def calculate_average_transaction(data: pd.DataFrame) -> float:
    return calculate_total_revenue(data) / len(data) if len(data) else 0.0


def calculate_category_revenue(data: pd.DataFrame) -> pd.Series:
    return data.groupby("Category")["Amount_INR"].sum().sort_values(ascending=False)


def calculate_category_share(data: pd.DataFrame) -> pd.Series:
    revenue = calculate_category_revenue(data)
    total = revenue.sum()
    return revenue / total * 100 if total else revenue.astype(float)


def calculate_daily_revenue(data: pd.DataFrame) -> pd.Series:
    return data.groupby(data["Date"].dt.normalize())["Amount_INR"].sum().sort_index()


def calculate_weekday_revenue(data: pd.DataFrame) -> pd.Series:
    values = data.assign(Weekday=data["Date"].dt.day_name()).groupby("Weekday")["Amount_INR"].sum()
    return values.reindex([day for day in WEEKDAY_ORDER if day in values.index], fill_value=0)


def calculate_average_weekday_revenue(data: pd.DataFrame) -> pd.Series:
    working = data.assign(Weekday=data["Date"].dt.day_name())
    values = working.groupby("Weekday")["Amount_INR"].mean()
    return values.reindex([day for day in WEEKDAY_ORDER if day in values.index])


def calculate_best_sales_day(data: pd.DataFrame) -> str:
    values = calculate_average_weekday_revenue(data)
    return str(values.idxmax()) if not values.empty else "Insufficient data"


def calculate_weakest_sales_day(data: pd.DataFrame) -> str:
    values = calculate_average_weekday_revenue(data)
    return str(values.idxmin()) if not values.empty else "Insufficient data"


def calculate_highest_revenue_date(data: pd.DataFrame) -> str:
    values = calculate_daily_revenue(data)
    return values.idxmax().strftime("%d %b %Y") if not values.empty else "Insufficient data"


def calculate_lowest_revenue_date(data: pd.DataFrame) -> str:
    values = calculate_daily_revenue(data)
    return values.idxmin().strftime("%d %b %Y") if not values.empty else "Insufficient data"


def calculate_revenue_trend(data: pd.DataFrame) -> dict[str, Any]:
    daily = calculate_daily_revenue(data)
    if len(daily) < 4:
        return {"label": "Insufficient data", "change_percent": None, "confidence": "Insufficient data", "current": calculate_total_revenue(data), "previous": None}
    midpoint = len(daily) // 2
    previous = float(daily.iloc[:midpoint].sum())
    current = float(daily.iloc[midpoint:].sum())
    change = ((current - previous) / previous * 100) if previous else None
    if change is None or abs(change) < 3:
        label = "Stable"
    elif change > 0:
        label = "Increasing"
    else:
        label = "Decreasing"
    confidence = "High" if len(daily) >= 28 else "Medium" if len(daily) >= 10 else "Low"
    return {"label": label, "change_percent": change, "confidence": confidence, "current": current, "previous": previous}


def calculate_period_comparison(data: pd.DataFrame) -> dict[str, Any]:
    trend = calculate_revenue_trend(data)
    return {"previous_period": trend["previous"], "current_period": trend["current"], "change_percent": trend["change_percent"]}


def get_sales_and_expenses(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Separate sales from expenses when a Type column is present."""
    if "Type" in data.columns:
        type_series = data["Type"].astype(str).str.strip().str.lower()
        is_expense = type_series.isin(["expense", "expenses", "debit", "dr", "payout", "purchase", "cost"])
        is_sale = type_series.isin(["sale", "sales", "income", "credit", "cr", "revenue"])
        if is_expense.any():
            sales_data = data[is_sale] if is_sale.any() else data[~is_expense]
            expense_data = data[is_expense]
            return sales_data if not sales_data.empty else data, expense_data, True
    return data, pd.DataFrame(), False


def build_analysis(data: pd.DataFrame) -> dict[str, Any]:
    sales_data, expense_data, has_expenses = get_sales_and_expenses(data)
    category_revenue = calculate_category_revenue(sales_data)
    category_share = calculate_category_share(sales_data)
    trend = calculate_revenue_trend(sales_data)

    total_revenue = calculate_total_revenue(sales_data)
    total_expenses = float(expense_data["Amount_INR"].sum()) if has_expenses else 0.0
    net_profit = total_revenue - total_expenses if has_expenses else total_revenue
    profit_margin = (net_profit / total_revenue * 100) if (has_expenses and total_revenue > 0) else 100.0

    has_payment_mode = "Payment_Mode" in data.columns
    payment_mode_summary = {}
    top_payment_mode = None
    if has_payment_mode:
        pm_grouped = sales_data.groupby("Payment_Mode")["Amount_INR"].sum().sort_values(ascending=False)
        payment_mode_summary = {str(k): round(float(v), 2) for k, v in pm_grouped.items()}
        top_payment_mode = str(pm_grouped.index[0]) if not pm_grouped.empty else None

    expense_category_breakdown = {}
    top_expense_category = None
    if has_expenses and not expense_data.empty:
        exp_grouped = expense_data.groupby("Category")["Amount_INR"].sum().sort_values(ascending=False)
        expense_category_breakdown = {str(k): round(float(v), 2) for k, v in exp_grouped.items()}
        top_expense_category = str(exp_grouped.index[0]) if not exp_grouped.empty else None

    return {
        "total_revenue": total_revenue,
        "transaction_count": calculate_transaction_count(sales_data),
        "average_transaction": calculate_average_transaction(sales_data),
        "category_revenue": category_revenue,
        "category_share": category_share,
        "category_transaction_count": sales_data.groupby("Category").size().sort_values(ascending=False),
        "top_category": str(category_revenue.index[0]) if not category_revenue.empty else "Insufficient data",
        "top_category_share": float(category_share.iloc[0]) if not category_share.empty else 0.0,
        "weekday_revenue": calculate_weekday_revenue(sales_data),
        "average_weekday_revenue": calculate_average_weekday_revenue(sales_data),
        "best_day": calculate_best_sales_day(sales_data),
        "weakest_day": calculate_weakest_sales_day(sales_data),
        "highest_revenue_date": calculate_highest_revenue_date(sales_data),
        "lowest_revenue_date": calculate_lowest_revenue_date(sales_data),
        "active_selling_days": int(sales_data["Date"].dt.normalize().nunique()) if not sales_data.empty else 0,
        "trend": trend,
        "has_quantity": "Quantity" in sales_data.columns,
        "top_units_category": _top_units_category(sales_data),
        "has_expenses": has_expenses,
        "total_expenses": total_expenses,
        "net_profit": net_profit,
        "profit_margin": profit_margin,
        "top_expense_category": top_expense_category,
        "expense_category_breakdown": expense_category_breakdown,
        "has_payment_mode": has_payment_mode,
        "top_payment_mode": top_payment_mode,
        "payment_mode_summary": payment_mode_summary,
    }


def _top_units_category(data: pd.DataFrame) -> str | None:
    if "Quantity" not in data.columns:
        return None
    quantity = pd.to_numeric(data["Quantity"], errors="coerce")
    if quantity.notna().sum() == 0:
        return None
    return str(data.assign(_quantity=quantity).groupby("Category")["_quantity"].sum().idxmax())


def build_ai_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Return only aggregate signals intended for the language model."""
    trend = analysis["trend"]
    summary = {
        "revenue_inr": round(analysis["total_revenue"], 2),
        "transactions": analysis["transaction_count"],
        "average_transaction_inr": round(analysis["average_transaction"], 2),
        "top_revenue_category": analysis["top_category"],
        "top_category_share_percent": round(analysis["top_category_share"], 2),
        "strongest_weekday": analysis["best_day"],
        "weakest_weekday": analysis["weakest_day"],
        "highest_revenue_date": analysis["highest_revenue_date"],
        "lowest_revenue_date": analysis["lowest_revenue_date"],
        "active_selling_days": analysis["active_selling_days"],
        "revenue_trend": trend["label"],
        "period_change_percent": round(trend["change_percent"], 2) if trend["change_percent"] is not None else None,
        "trend_confidence": trend["confidence"],
    }
    if analysis.get("has_expenses"):
        summary["total_expenses_inr"] = round(analysis["total_expenses"], 2)
        summary["net_profit_inr"] = round(analysis["net_profit"], 2)
        summary["profit_margin_percent"] = round(analysis["profit_margin"], 2)
        if analysis.get("top_expense_category"):
            summary["top_expense_category"] = analysis["top_expense_category"]
    if analysis.get("has_payment_mode") and analysis.get("top_payment_mode"):
        summary["top_payment_mode"] = analysis["top_payment_mode"]
        summary["payment_modes"] = analysis.get("payment_mode_summary", {})
    return summary