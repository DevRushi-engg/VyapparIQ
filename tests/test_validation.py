from io import BytesIO

from validation import validate_and_clean_csv


def test_valid_dataset_is_cleaned():
    raw = b"Date,Item_Name,Category,Amount_INR\n2026-09-01, Rice , Grocery ,1200\n"
    result = validate_and_clean_csv(BytesIO(raw))
    assert result.valid
    assert result.valid_rows == 1
    assert result.cleaned_data.iloc[0]["Category"] == "Grocery"


def test_user_format_with_particulars_and_type():
    raw = (
        b"Date,Particulars,Category,Payment_Mode,Type,Amount\n"
        b'2024-01-01,"Packaged Groceries Retail Sale #1",Packaged Groceries,UPI,Sale,350\n'
        b'2024-01-01,"Shop Commercial Rent Payment",Rent,Bank Transfer,Expense,18000\n'
    )
    result = validate_and_clean_csv(raw)
    assert result.valid
    assert result.valid_rows == 2
    assert "Amount_INR" in result.cleaned_data.columns
    assert result.cleaned_data.iloc[0]["Amount_INR"] == 350
    assert result.cleaned_data.iloc[0]["Type"] == "Sale"
    assert result.cleaned_data.iloc[1]["Type"] == "Expense"
    assert result.cleaned_data.iloc[0]["Item_Name"] == "Packaged Groceries Retail Sale #1"


def test_arbitrary_column_names():
    raw = (
        b"Invoice_Date,Product,Department,Total\n"
        b"2024-05-10,Widget,Hardware,499.50\n"
    )
    result = validate_and_clean_csv(raw)
    assert result.valid
    assert result.valid_rows == 1
    assert result.cleaned_data.iloc[0]["Category"] == "Hardware"
    assert result.cleaned_data.iloc[0]["Item_Name"] == "Widget"
    assert result.cleaned_data.iloc[0]["Amount_INR"] == 499.50


def test_missing_amount_column_is_reported():
    result = validate_and_clean_csv(b"Date,Item_Name,Category\n2026-09-01,Rice,Grocery\n")
    assert not result.valid
    assert "amount" in result.errors[0].lower()


def test_bad_dates_and_blank_amounts_are_excluded():
    raw = (
        b"Date,Particulars,Category,Amount\n"
        b"not-a-date,Rice,Grocery,10\n"
        b"2026-09-01,Sugar,Grocery,\n"
        b"2026-09-03,Tea,Beverages,30\n"
    )
    result = validate_and_clean_csv(raw)
    assert result.valid
    assert result.valid_rows == 1
    assert result.invalid_rows == 2


def test_empty_file_is_safe():
    result = validate_and_clean_csv(b"")
    assert not result.valid
    assert result.errors