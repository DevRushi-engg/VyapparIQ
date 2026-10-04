"""CSV validation, auto-detection, and flexible cleaning for any business CSV."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import re
from typing import Any

import pandas as pd

MAX_FILE_BYTES = 10 * 1024 * 1024

# Alias lists for intelligent column discovery
DATE_ALIASES = [
    "date", "datetime", "transaction_date", "txn_date", "sale_date",
    "order_date", "invoice_date", "entry_date", "timestamp", "time", "day", "dated", "created_at"
]

AMOUNT_ALIASES = [
    "amount", "amount_inr", "amount(inr)", "total", "total_amount", "net_amount",
    "sale_amount", "price", "sales", "revenue", "value", "cost", "subtotal",
    "grand_total", "paid", "bill_amount", "txn_amount"
]

CATEGORY_ALIASES = [
    "category", "product_category", "item_category", "department", "dept",
    "group", "classification", "segment", "head", "expense_category", "cat"
]

ITEM_ALIASES = [
    "particulars", "item_name", "item", "product", "product_name",
    "description", "desc", "details", "narration", "memo", "title", "name", "item_description"
]

TYPE_ALIASES = [
    "type", "transaction_type", "txn_type", "entry_type", "credit_debit", "cr_dr", "sale_type"
]

PAYMENT_MODE_ALIASES = [
    "payment_mode", "payment_method", "payment_type", "mode", "payment",
    "paid_via", "method", "pay_mode"
]


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _find_matching_col(columns: list[str], aliases: list[str]) -> str | None:
    norm_map = {_normalize_col_name(c): c for c in columns}

    # 1. Exact alias match
    for alias in aliases:
        norm_alias = _normalize_col_name(alias)
        if norm_alias in norm_map:
            return norm_map[norm_alias]

    # 2. Substring match
    for alias in aliases:
        norm_alias = _normalize_col_name(alias)
        if len(norm_alias) >= 4:
            for norm_c, orig_c in norm_map.items():
                if norm_alias in norm_c or norm_c in norm_alias:
                    return orig_c

    return None


@dataclass
class ValidationResult:
    """Validation outcome and a cleaned dataframe suitable for analysis."""

    valid: bool
    cleaned_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_uploaded: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    duplicate_rows: int = 0
    date_range: str = "Unavailable"
    categories_detected: int = 0
    detected_columns: dict[str, str] = field(default_factory=dict)


def validate_and_clean_csv(source: Any, max_file_bytes: int = MAX_FILE_BYTES) -> ValidationResult:
    """Read, flexibly detect schema, and clean CSV data from any format."""
    try:
        if hasattr(source, "getvalue"):
            raw = source.getvalue()
        elif isinstance(source, bytes):
            raw = source
        else:
            raw = source.read()
        if len(raw) > max_file_bytes:
            return ValidationResult(False, errors=[f"This file is larger than the current {max_file_bytes // (1024 * 1024)} MB limit."])
        if not raw.strip():
            return ValidationResult(False, errors=["The CSV file is empty."])
        frame = pd.read_csv(BytesIO(raw))
    except pd.errors.ParserError:
        return ValidationResult(False, errors=["Something went wrong while reading the CSV. Please check its structure."])
    except Exception:
        return ValidationResult(False, errors=["Something went wrong while reading the CSV."])

    result = ValidationResult(valid=False, rows_uploaded=len(frame))
    frame.columns = [str(column).strip() for column in frame.columns]

    if frame.empty:
        result.errors.append("The CSV does not contain any rows.")
        return result

    cols = list(frame.columns)

    # 1. Detect Date Column
    date_col = _find_matching_col(cols, DATE_ALIASES)
    if not date_col:
        # Fallback: check if any column contains parsable dates
        for c in cols:
            converted = pd.to_datetime(frame[c], errors="coerce", format="mixed")
            if converted.notna().sum() > len(frame) * 0.4:
                date_col = c
                break

    if not date_col:
        result.errors.append("Could not identify a date column (expected e.g. Date, Timestamp, Invoice_Date).")
        return result

    # 2. Detect Amount Column
    amount_col = _find_matching_col(cols, AMOUNT_ALIASES)
    if not amount_col:
        # Fallback: check numeric columns
        for c in cols:
            if c == date_col:
                continue
            numeric = pd.to_numeric(
                frame[c].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.replace("$", "", regex=False).str.strip(),
                errors="coerce"
            )
            if numeric.notna().sum() > len(frame) * 0.4:
                amount_col = c
                break

    if not amount_col:
        result.errors.append("Could not identify an amount column (expected e.g. Amount, Amount_INR, Total, Price, Sales).")
        return result

    # 3. Detect Optional Columns
    category_col = _find_matching_col(cols, CATEGORY_ALIASES)
    item_col = _find_matching_col(cols, ITEM_ALIASES)
    type_col = _find_matching_col(cols, TYPE_ALIASES)
    mode_col = _find_matching_col(cols, PAYMENT_MODE_ALIASES)

    result.detected_columns = {
        "Date": date_col,
        "Amount": amount_col,
        "Category": category_col or "(Auto: General)",
        "Item": item_col or "(Auto: Category)",
    }
    if type_col:
        result.detected_columns["Type"] = type_col
    if mode_col:
        result.detected_columns["Payment_Mode"] = mode_col

    result.duplicate_rows = int(frame.duplicated().sum())
    if result.duplicate_rows:
        result.warnings.append(f"{result.duplicate_rows} duplicate row(s) were removed.")

    cleaned = frame.copy()

    # Parse Date
    cleaned["Date"] = pd.to_datetime(cleaned[date_col], errors="coerce", format="mixed")

    # Parse Amount
    cleaned["Amount_INR"] = pd.to_numeric(
        cleaned[amount_col].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    # Standardize Category
    if category_col:
        cleaned["Category"] = cleaned[category_col].astype("string").str.strip()
    elif item_col:
        cleaned["Category"] = cleaned[item_col].astype("string").str.strip()
    else:
        cleaned["Category"] = "General"

    # Standardize Item_Name / Particulars
    if item_col:
        cleaned["Item_Name"] = cleaned[item_col].astype("string").str.strip()
    elif category_col:
        cleaned["Item_Name"] = cleaned[category_col].astype("string").str.strip()
    else:
        cleaned["Item_Name"] = "Transaction"

    # Standardize Type if present
    if type_col:
        cleaned["Type"] = cleaned[type_col].astype("string").str.strip()

    # Standardize Payment_Mode if present
    if mode_col:
        cleaned["Payment_Mode"] = cleaned[mode_col].astype("string").str.strip()

    # Preserve Quantity if present
    qty_col = _find_matching_col(cols, ["quantity", "qty", "units", "count"])
    if qty_col and qty_col != amount_col:
        cleaned["Quantity"] = pd.to_numeric(cleaned[qty_col], errors="coerce")

    # Filter out invalid rows (missing date or missing amount)
    invalid_mask = (
        cleaned["Date"].isna()
        | cleaned["Amount_INR"].isna()
        | cleaned["Category"].isna()
        | cleaned["Category"].eq("")
    )

    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        result.warnings.append(f"{invalid_count} unusable row(s) with missing dates/amounts were excluded.")

    cleaned = cleaned.loc[~invalid_mask].drop_duplicates().reset_index(drop=True)
    result.valid_rows = len(cleaned)
    result.invalid_rows = result.rows_uploaded - result.valid_rows
    result.categories_detected = int(cleaned["Category"].nunique()) if not cleaned.empty else 0

    if cleaned.empty:
        result.errors.append("No usable rows remain after validation. Check your date and amount values.")
        return result

    start = cleaned["Date"].min().strftime("%d %b %Y")
    end = cleaned["Date"].max().strftime("%d %b %Y")
    result.date_range = start if start == end else f"{start} - {end}"
    result.cleaned_data = cleaned
    result.valid = True
    return result