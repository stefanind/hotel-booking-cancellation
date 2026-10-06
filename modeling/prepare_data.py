"""Shared cleaning steps for the hotel cancellation notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "hotel_bookings.csv"

EXPECTED_ROWS = 119_390
MAX_LEAD_TIME_DAYS = 180

# These fields were not safely available when a booking was first created.
LEAKAGE_COLUMNS = [
    "is_canceled",
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
]

# Candidate inputs under the project's booking-time scoring assumption. The source
# file is not a true point-in-time snapshot, so availability remains a limitation.
CANDIDATE_MODEL_FEATURES = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "children_was_missing",
]

CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "reservation_status",
]


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy while retaining every source row."""

    if len(raw) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {len(raw):,}.")
    if set(raw["is_canceled"].dropna().unique()) != {0, 1}:
        raise ValueError("is_canceled must contain only 0 and 1, with both present.")

    # No-Show is a cancellation outcome in this file. The notebook also shows
    # the literal comparison with status == 'Canceled' to audit the target.
    canceled_outcome = raw["reservation_status"].isin(["Canceled", "No-Show"])
    if not raw["is_canceled"].eq(canceled_outcome.astype(int)).all():
        raise ValueError("is_canceled does not agree with the final status outcome.")

    clean = raw.copy(deep=True)
    clean.insert(0, "source_row_id", np.arange(len(clean), dtype="int64"))

    # Compare only the original columns. Adding source_row_id first would make
    # every row unique.
    clean["is_exact_duplicate"] = raw.duplicated(keep=False)

    # Normalize whitespace only; do not guess that different labels mean the same thing.
    for column in CATEGORICAL_COLUMNS:
        clean[column] = (
            clean[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    clean["children_was_missing"] = clean["children"].isna()
    clean["children"] = clean["children"].fillna(0).astype("int64")
    clean["country"] = clean["country"].fillna("UNKNOWN_COUNTRY")

    # Agent and company are labels. Nullable integer -> string avoids values like '9.0'.
    clean["agent"] = (
        clean["agent"].astype("Int64").astype("string").fillna("NO_AGENT")
    )
    clean["company"] = (
        clean["company"].astype("Int64").astype("string").fillna("NO_COMPANY")
    )

    # The source documentation defines Undefined and SC as the same meal package:
    # both mean no meal package. Other Undefined values remain untouched.
    clean["meal_was_undefined"] = clean["meal"].eq("Undefined")
    clean["meal"] = clean["meal"].replace({"Undefined": "SC"})

    # This is final-status timing metadata for auditing, never a model input.
    clean["reservation_status_date"] = pd.to_datetime(
        clean["reservation_status_date"], errors="raise"
    )

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    month_number = clean["arrival_date_month"].map(
        {month: number for number, month in enumerate(month_names, start=1)}
    )
    clean["arrival_date"] = pd.to_datetime(
        {
            "year": clean["arrival_date_year"],
            "month": month_number,
            "day": clean["arrival_date_day_of_month"],
        },
        errors="raise",
    )
    clean["booking_date_proxy"] = clean["arrival_date"] - pd.to_timedelta(
        clean["lead_time"], unit="D"
    )
    clean["booking_week_start"] = clean["booking_date_proxy"] - pd.to_timedelta(
        clean["booking_date_proxy"].dt.dayofweek, unit="D"
    )

    clean["total_nights"] = (
        clean["stays_in_weekend_nights"] + clean["stays_in_week_nights"]
    )
    clean["total_guests"] = clean["adults"] + clean["children"] + clean["babies"]
    clean["negative_adr_flag"] = clean["adr"].lt(0)
    clean["zero_night_stay_flag"] = clean["total_nights"].eq(0)
    clean["zero_guest_booking_flag"] = clean["total_guests"].eq(0)

    # Complete booking cohorts under the chosen 180-day scope.
    arrival_start = clean["arrival_date"].min()
    arrival_end = clean["arrival_date"].max()
    complete_cohort_end = arrival_end - pd.Timedelta(days=MAX_LEAD_TIME_DAYS)
    clean["lead_time_within_180_days"] = clean["lead_time"].between(
        0, MAX_LEAD_TIME_DAYS
    )
    clean["complete_booking_cohort"] = clean["booking_date_proxy"].between(
        arrival_start, complete_cohort_end
    )

    # Later evaluation is weekly, so exclude the two partial boundary weeks.
    first_full_week = arrival_start + pd.Timedelta(days=(-arrival_start.dayofweek) % 7)
    last_full_week_end = complete_cohort_end - pd.Timedelta(
        days=(complete_cohort_end.dayofweek + 1) % 7
    )
    last_full_week = last_full_week_end - pd.Timedelta(days=6)
    clean["complete_booking_week"] = clean["booking_week_start"].between(
        first_full_week, last_full_week
    )
    clean["complete_180_population"] = (
        clean["lead_time_within_180_days"]
        & clean["complete_booking_cohort"]
        & clean["complete_booking_week"]
    )

    # Chronological modeling periods for the 86 complete booking weeks.
    full_weeks = pd.date_range(first_full_week, last_full_week, freq="7D")
    validation_start = full_weeks[-26]
    holdout_start = full_weeks[-13]
    eligible = clean["complete_180_population"]

    clean["booking_period_split"] = "out_of_scope"
    clean.loc[
        eligible & clean["booking_week_start"].lt(validation_start),
        "booking_period_split",
    ] = "train"
    clean.loc[
        eligible
        & clean["booking_week_start"].ge(validation_start)
        & clean["booking_week_start"].lt(holdout_start),
        "booking_period_split",
    ] = "validation"
    clean.loc[
        eligible & clean["booking_week_start"].ge(holdout_start),
        "booking_period_split",
    ] = "holdout"

    return clean


def model_features(prepared: pd.DataFrame) -> pd.DataFrame:
    """Return the candidate model inputs, excluding known leakage."""

    return prepared[CANDIDATE_MODEL_FEATURES].copy()


def complete_180_rows(prepared: pd.DataFrame) -> pd.DataFrame:
    """Return the complete weekly population within the 180-day scope."""

    keep = prepared["complete_180_population"]
    return prepared.loc[keep].sort_values(
        ["booking_date_proxy", "source_row_id"], kind="stable"
    )
