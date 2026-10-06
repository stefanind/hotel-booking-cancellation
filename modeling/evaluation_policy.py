"""Shared evaluation rules for the weekly hotel-booking call queue."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


DEFAULT_CALL_CAPACITY = 150

# Ranking-time proximity boost. Uses days to arrival at the call, which is
# known when the list is produced.
DEFAULT_URGENCY_WINDOW_DAYS = 30
MAX_URGENCY_BOOST = 0.50

# Evaluation-time resale assumption. Uses days between the actual cancellation
# and arrival, which is NOT known at call time and never enters a ranking.
DEFAULT_RESALE_WINDOW_DAYS = 30

PRIORITY_COLUMNS = {
    "random": "random_priority",
    "risk_only": "risk_priority",
    "risk_and_value": "risk_value_priority",
    "risk_value_and_urgency": "business_priority",
}


def add_first_call_monday(bookings: pd.DataFrame) -> pd.DataFrame:
    """Add the first Monday after each Monday-Sunday booking week."""

    queued = bookings.copy()
    queued["first_call_monday"] = (
        queued["booking_week_start"] + pd.Timedelta(days=7)
    )
    return queued


def call_dates_for_split(bookings: pd.DataFrame, split_name: str) -> pd.DatetimeIndex:
    """Return one Monday decision for every booking week in a split."""

    booking_weeks = pd.DatetimeIndex(
        bookings.loc[
            bookings["booking_period_split"].eq(split_name),
            "booking_week_start",
        ]
        .drop_duplicates()
        .sort_values()
    )
    return booking_weeks + pd.Timedelta(days=7)


def add_priority_columns(
    queue: pd.DataFrame,
    call_date: pd.Timestamp,
    *,
    probability_column: str = "p_cancel",
    urgency_window_days: int = DEFAULT_URGENCY_WINDOW_DAYS,
) -> pd.DataFrame:
    """Recalculate Monday priority while leaving booking-time risk unchanged."""

    if probability_column not in queue:
        raise KeyError(f"Missing booking-time probability column: {probability_column}")
    if not queue[probability_column].between(0, 1).all():
        raise ValueError(f"{probability_column} must contain probabilities from 0 to 1.")
    if not np.isfinite(urgency_window_days) or urgency_window_days <= 0:
        raise ValueError("urgency_window_days must be positive.")

    scored = queue.copy()
    call_date = pd.Timestamp(call_date).normalize()
    scored["days_to_arrival_at_call"] = (
        scored["arrival_date"] - call_date
    ).dt.days
    scored["booking_value_proxy"] = (
        scored["adr"].clip(lower=0) * scored["total_nights"]
    )
    scored["proximity_at_call"] = (
        1 - scored["days_to_arrival_at_call"] / urgency_window_days
    ).clip(lower=0, upper=1)
    scored["urgency_boost_at_call"] = (
        1 + MAX_URGENCY_BOOST * scored["proximity_at_call"]
    )

    risk = scored[probability_column]
    scored["risk_priority"] = risk
    scored["risk_value_priority"] = risk * scored["booking_value_proxy"]
    scored["business_priority"] = (
        scored["risk_value_priority"] * scored["urgency_boost_at_call"]
    )
    return scored


def add_realized_loss_columns(
    calls: pd.DataFrame,
    *,
    resale_window_days: int = DEFAULT_RESALE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Value each reached cancellation by when it actually canceled.

    The business context is that a cancellation is expensive when it lands
    close to arrival, because the room is then resold at a discount or not at
    all. How much notice the hotel received is therefore a property of the
    cancellation date, not of the date the guest happened to be called.

    This is an evaluation-only calculation. It reads
    ``reservation_status_date``, which is unknown when the list is produced, so
    it must never feed a ranking score. Two bounds are returned:

    * ``realized_loss_proxy`` assumes a linear resale recovery, so a
      cancellation with at least ``resale_window_days`` of notice costs
      nothing and a no-show costs the full booked value.
    * ``full_value_loss_proxy`` assumes no resale at all, so every
      cancellation costs the full booked value.

    Neither is observed profit; both rest on ``adr * nights`` as an exposure
    proxy.
    """

    if not np.isfinite(resale_window_days) or resale_window_days <= 0:
        raise ValueError("resale_window_days must be positive.")

    valued = calls.copy()
    if not len(valued):
        for column in [
            "cancellation_notice_days",
            "unrecovered_share_at_cancellation",
            "realized_loss_proxy",
            "full_value_loss_proxy",
        ]:
            valued[column] = pd.Series(dtype="float64")
        return valued

    canceled = valued["is_canceled"].eq(1)
    notice_days = (
        valued["arrival_date"] - valued["reservation_status_date"]
    ).dt.days.clip(lower=0)
    # A no-show is dated on the arrival date, so it receives zero notice and
    # the full unrecovered share. That is the intended treatment.
    unrecovered_share = (1 - notice_days / resale_window_days).clip(0, 1)

    valued["cancellation_notice_days"] = notice_days.where(canceled)
    valued["unrecovered_share_at_cancellation"] = unrecovered_share.where(
        canceled
    )
    valued["realized_loss_proxy"] = (
        valued["booking_value_proxy"] * unrecovered_share
    ).where(canceled, 0.0)
    valued["full_value_loss_proxy"] = valued["booking_value_proxy"].where(
        canceled, 0.0
    )
    return valued


def pre_call_attrition(
    bookings: pd.DataFrame,
    *,
    same_day_cancellations_are_eligible: bool = True,
) -> pd.Series:
    """Count rows that cannot be called on their first possible Monday."""

    queued = add_first_call_monday(bookings)
    canceled_date = queued["reservation_status_date"]
    canceled_before_snapshot = (
        canceled_date.lt(queued["first_call_monday"])
        if same_day_cancellations_are_eligible
        else canceled_date.le(queued["first_call_monday"])
    )
    canceled_before_first_call = (
        queued["reservation_status"].eq("Canceled")
        & canceled_before_snapshot
    )
    arrived_before_first_call = (
        ~canceled_before_first_call
        & queued["arrival_date"].le(queued["first_call_monday"])
    )
    return pd.Series(
        {
            "canceled before first Monday": int(canceled_before_first_call.sum()),
            "arrived by first Monday": int(arrived_before_first_call.sum()),
            "available at first opportunity": int(
                (~canceled_before_first_call & ~arrived_before_first_call).sum()
            ),
        },
        name="bookings",
    )


def _active_queue(
    bookings: pd.DataFrame,
    call_date: pd.Timestamp,
    called_ids: set,
    *,
    same_day_cancellations_are_eligible: bool,
) -> pd.DataFrame:
    """Return bookings that are actionable at the Monday snapshot."""

    call_date = pd.Timestamp(call_date).normalize()
    canceled_date = bookings["reservation_status_date"]
    canceled_before_snapshot = (
        canceled_date.lt(call_date)
        if same_day_cancellations_are_eligible
        else canceled_date.le(call_date)
    )
    known_canceled = (
        bookings["reservation_status"].eq("Canceled")
        & canceled_before_snapshot
    )
    active = (
        bookings["first_call_monday"].le(call_date)
        & bookings["arrival_date"].gt(call_date)
        & ~known_canceled
        & ~bookings["source_row_id"].isin(called_ids)
    )
    return bookings.loc[active].copy()


def simulate_weekly_policy(
    bookings: pd.DataFrame,
    split_name: str,
    *,
    policy: str = "risk_and_value",
    probability_column: str = "p_cancel",
    capacity: int = DEFAULT_CALL_CAPACITY,
    urgency_window_days: int = DEFAULT_URGENCY_WINDOW_DAYS,
    resale_window_days: int = DEFAULT_RESALE_WINDOW_DAYS,
    random_seed: int = 42,
    same_day_cancellations_are_eligible: bool = True,
):
    """Run one fresh standing-queue simulation for one split and policy."""

    if policy not in PRIORITY_COLUMNS:
        choices = ", ".join(PRIORITY_COLUMNS)
        raise ValueError(f"Unknown policy {policy!r}. Choose from: {choices}.")
    if (
        isinstance(capacity, (bool, np.bool_))
        or not isinstance(capacity, (int, np.integer))
        or capacity <= 0
    ):
        raise ValueError("capacity must be a positive integer.")

    split_bookings = bookings.loc[
        bookings["booking_period_split"].eq(split_name)
    ].copy()
    if split_bookings.empty:
        raise ValueError(f"No rows found for split: {split_name}")
    if (
        split_bookings["source_row_id"].isna().any()
        or split_bookings["source_row_id"].duplicated().any()
    ):
        raise ValueError("source_row_id must be unique and non-null within a split.")

    split_bookings = add_first_call_monday(split_bookings)
    call_dates = call_dates_for_split(split_bookings, split_name)
    score_column = PRIORITY_COLUMNS[policy]
    random_generator = np.random.default_rng(random_seed)
    random_scores = dict(
        zip(
            sorted(split_bookings["source_row_id"].unique()),
            random_generator.random(split_bookings["source_row_id"].nunique()),
        )
    )

    called_ids = set()
    previous_carry_ids = set()
    ever_active_ids = set()
    call_frames = []
    weekly_rows = []

    for call_date in call_dates:
        queue = _active_queue(
            split_bookings,
            call_date,
            called_ids,
            same_day_cancellations_are_eligible=(
                same_day_cancellations_are_eligible
            ),
        )
        current_queue_ids = set(queue["source_row_id"])
        carry_departures = previous_carry_ids - current_queue_ids
        ever_active_ids.update(current_queue_ids)

        queue = add_priority_columns(
            queue,
            call_date,
            probability_column=probability_column,
            urgency_window_days=urgency_window_days,
        )
        if policy == "random":
            queue["random_priority"] = queue["source_row_id"].map(random_scores)

        ranked = queue.loc[queue[score_column].gt(0)].sort_values(
            [score_column, "source_row_id"],
            ascending=[False, True],
            kind="stable",
        )
        selected = ranked.head(capacity).copy()
        # Evaluation-only valuation, applied after the calls are already
        # chosen so cancellation timing cannot influence the ranking.
        selected = add_realized_loss_columns(
            selected, resale_window_days=resale_window_days
        )
        selected["call_date"] = call_date
        selected["queue_size_at_call"] = len(queue)
        selected["policy"] = policy
        selected["call_rank"] = np.arange(1, len(selected) + 1)
        call_frames.append(selected)

        selected_ids = set(selected["source_row_id"])
        called_ids.update(selected_ids)
        previous_carry_ids = current_queue_ids - selected_ids

        canceled_selected = selected["is_canceled"].eq(1)
        precision_among_calls = (
            canceled_selected.mean() if len(selected) else np.nan
        )
        precision_at_capacity = canceled_selected.sum() / capacity
        queue_cancel_rate = queue["is_canceled"].mean() if len(queue) else np.nan
        lift = (
            precision_among_calls / queue_cancel_rate
            if pd.notna(precision_among_calls)
            and pd.notna(queue_cancel_rate)
            and queue_cancel_rate > 0
            else np.nan
        )
        weekly_rows.append(
            {
                "call_date": call_date,
                "eligible_queue": len(queue),
                "carry_departures_before_call": len(carry_departures),
                "positive_priority_pool": len(ranked),
                "calls": len(selected),
                "unused_capacity": capacity - len(selected),
                "capacity_utilization": len(selected) / capacity,
                "cancellations_reached": int(canceled_selected.sum()),
                "precision_at_capacity": precision_at_capacity,
                "precision_among_calls": precision_among_calls,
                "queue_cancellation_rate": queue_cancel_rate,
                "lift_over_queue": lift,
                "canceled_value_proxy_reached": selected.loc[
                    canceled_selected, "booking_value_proxy"
                ].sum(),
                "realized_loss_reached": selected.loc[
                    canceled_selected, "realized_loss_proxy"
                ].sum(),
                "full_value_loss_reached": selected.loc[
                    canceled_selected, "full_value_loss_proxy"
                ].sum(),
                "average_days_to_arrival": selected[
                    "days_to_arrival_at_call"
                ].mean(),
            }
        )

    calls = pd.concat(call_frames, ignore_index=True) if call_frames else pd.DataFrame()
    weekly = pd.DataFrame(weekly_rows)
    called_cancellations = int(calls["is_canceled"].sum()) if len(calls) else 0
    total_calls = len(calls)
    reached_cancellations = (
        calls.loc[calls["is_canceled"].eq(1)] if len(calls) else calls
    )
    realized_loss_reached = (
        reached_cancellations["realized_loss_proxy"].sum()
        if len(reached_cancellations)
        else 0.0
    )
    full_value_loss_reached = (
        reached_cancellations["full_value_loss_proxy"].sum()
        if len(reached_cancellations)
        else 0.0
    )

    callable_cancellations = split_bookings.loc[
        split_bookings["source_row_id"].isin(ever_active_ids)
        & split_bookings["is_canceled"].eq(1),
        "source_row_id",
    ].nunique()
    final_queue_cancellations = split_bookings.loc[
        split_bookings["source_row_id"].isin(previous_carry_ids)
        & split_bookings["is_canceled"].eq(1),
        "source_row_id",
    ].nunique()
    weekly_precision = weekly["precision_at_capacity"]
    aggregate = pd.Series(
        {
            "total calls": total_calls,
            "cancellations reached": called_cancellations,
            "overall precision at capacity": (
                called_cancellations / (capacity * len(weekly))
                if len(weekly)
                else np.nan
            ),
            "overall precision among calls": (
                called_cancellations / total_calls if total_calls else np.nan
            ),
            "mean weekly queue precision at capacity": weekly_precision.mean(),
            "median weekly queue precision at capacity": weekly_precision.median(),
            "weekly precision IQR": (
                weekly_precision.quantile(0.75) - weekly_precision.quantile(0.25)
            ),
            "mean capacity utilization": weekly["capacity_utilization"].mean(),
            "within-horizon callable cancellation capture": (
                called_cancellations / callable_cancellations
                if callable_cancellations
                else np.nan
            ),
            "bookings still waiting after final decision": len(previous_carry_ids),
            "eventual cancellations in final queue": final_queue_cancellations,
            "canceled booking value reached": (
                reached_cancellations["booking_value_proxy"].sum()
                if len(reached_cancellations)
                else 0.0
            ),
            "realized loss reached": realized_loss_reached,
            "full value loss reached": full_value_loss_reached,
        },
        name=f"{split_name}: {policy}",
    )
    final_queue = split_bookings.loc[
        split_bookings["source_row_id"].isin(previous_carry_ids)
    ].copy()
    return calls, weekly, aggregate, final_queue


def repeated_random_baseline(
    bookings: pd.DataFrame,
    split_name: str,
    *,
    repeats: int = 100,
    first_seed: int = 42,
    probability_column: str = "p_cancel",
    capacity: int = DEFAULT_CALL_CAPACITY,
    urgency_window_days: int = DEFAULT_URGENCY_WINDOW_DAYS,
    resale_window_days: int = DEFAULT_RESALE_WINDOW_DAYS,
    same_day_cancellations_are_eligible: bool = True,
) -> pd.DataFrame:
    """Repeat the same capacity-matched queue with random booking priorities."""

    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    results = []
    for repeat in range(repeats):
        seed = first_seed + repeat
        _, _, aggregate, _ = simulate_weekly_policy(
            bookings,
            split_name,
            policy="random",
            probability_column=probability_column,
            capacity=capacity,
            urgency_window_days=urgency_window_days,
            resale_window_days=resale_window_days,
            random_seed=seed,
            same_day_cancellations_are_eligible=(
                same_day_cancellations_are_eligible
            ),
        )
        result = aggregate.to_dict()
        result.update({"repeat": repeat + 1, "random_seed": seed})
        results.append(result)
    return pd.DataFrame(results)


def model_metric_summary(y_true, p_cancel) -> pd.Series:
    """Return supporting probability metrics for later model notebooks."""

    return pd.Series(
        {
            "average_precision": average_precision_score(y_true, p_cancel),
            "roc_auc": roc_auc_score(y_true, p_cancel),
            "brier_score": brier_score_loss(y_true, p_cancel),
        }
    )


def estimate_program_return(
    calls: pd.DataFrame,
    *,
    save_rate: float,
    contribution_margin: float,
    call_cost: float,
    effective_credit_cost_per_call: float,
    loss_column: str = "realized_loss_proxy",
) -> pd.Series:
    """Apply explicit causal and cost assumptions to an evaluated call list.

    ``loss_column`` selects the exposure basis. ``realized_loss_proxy`` assumes
    rooms canceled with notice are resold; ``full_value_loss_proxy`` assumes no
    resale. Reporting both brackets the assumption rather than hiding it.
    """

    if not 0 <= save_rate <= 1:
        raise ValueError("save_rate must be between 0 and 1.")
    if not 0 <= contribution_margin <= 1:
        raise ValueError("contribution_margin must be between 0 and 1.")
    if call_cost < 0 or effective_credit_cost_per_call < 0:
        raise ValueError("Call and credit costs cannot be negative.")

    canceled_loss_proxy = calls.loc[
        calls["is_canceled"].eq(1), loss_column
    ].sum()
    benefit = save_rate * contribution_margin * canceled_loss_proxy
    cost = len(calls) * (call_cost + effective_credit_cost_per_call)
    net_return = benefit - cost
    return pd.Series(
        {
            "scenario_benefit": benefit,
            "scenario_cost": cost,
            "net_return": net_return,
            "roi": net_return / cost if cost else np.nan,
            "break_even_save_rate": (
                cost / (contribution_margin * canceled_loss_proxy)
                if canceled_loss_proxy and contribution_margin
                else np.nan
            ),
        }
    )
