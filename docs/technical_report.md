# Hotel Booking Cancellation & Retention Analytics

Technical report on a retrospective cancellation-risk model, a capacity-limited outreach policy, and an illustrative financial evaluation.

## Objective and operating assumptions

The project evaluates which hotel bookings to contact when outreach capacity is limited. A hypothetical team can attempt 150 calls each Monday and offer a $50 property credit. Scores use candidate booking-time information; each booking is called at most once, and uncalled eligible bookings carry forward.

The analysis separates estimating cancellation risk, choosing a queue ranking, and assessing the economics of an intervention whose effect is unknown. A predictive model alone cannot show that outreach causes guests to keep their bookings.

## Data and preparation

The source is the hotel booking demand data published by Nuno Antonio, Ana de Almeida, and Luis Nunes (2019), [Data in Brief, 22, 41–49](https://doi.org/10.1016/j.dib.2018.11.126). The combined CSV contains 119,390 bookings, 32 fields including a hotel-type indicator, and arrivals from July 2015 through August 2017. The overall cancellation rate is 37.0%.

The target `is_canceled` includes both cancellations and no-shows. A literal comparison with `reservation_status == 'Canceled'` finds 1,207 differences, all no-shows. The target is validated against both terminal outcomes. Outcome fields remain available for auditing and evaluation but never enter a risk score.

The source CSV is unchanged. Preparation operates on a copy, normalizes categorical whitespace, fills missing children with zero while preserving a missingness flag, and uses explicit labels for missing country, agent, and company values. Agent and company codes are categorical identifiers. Meal values `Undefined` and `SC` are merged because the source dictionary defines both as no meal package.

There are 40,165 rows belonging to exact-match groups across the original fields. They are flagged and retained: without a reservation identifier, identical anonymized rows cannot be distinguished from duplicate records. One negative ADR, 715 zero-night stays, and 180 zero-guest bookings are also flagged and retained rather than repaired by an unsupported rule.

`arrival_date` is assembled from the arrival fields. `booking_date_proxy = arrival_date - lead_time` reconstructs an approximate booking date. It has no time of day and cannot resolve exact event ordering. Twenty-seven candidate features exclude the target, final status and date, assigned room type, booking changes, and waiting-list duration. The [source paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) describes values extracted relative to the day before arrival, so removing known leakage fields does not establish that all remaining values were available when the booking was created.

## Population and chronological evaluation

Modeling covers lead times of 0–180 days within complete Monday–Sunday booking cohorts: **76,863 rows across 86 weeks**. This limits incomplete cohorts arising from an arrival-bounded dataset. It is a population-scope choice; all 119,390 source rows remain in the prepared table.

The chronological periods contain 60 training weeks, 13 validation weeks, and 13 holdout weeks. Fitting uses only training-period bookings with arrivals before validation starts, reducing 48,738 nominal training rows to **43,264 resolved outcomes**. Applying the same arrival cutoff to every outcome avoids selectively admitting early cancellations while other future bookings remain unresolved.

Validation selects the model and calling policy. The frozen specification is evaluated on holdout without fitting on validation or holdout labels. This is a retrospective cohort experiment: some validation outcomes resolve after holdout bookings were made. A fully prospective model-selection replay would require additional calendar history or an outcome-delay gap.

## Model comparison

The baseline is L2-regularized logistic regression with one-hot categorical encoding and standardized numeric features. A small comparison of `C = 0.1, 1, 10` selects **C = 1** using validation mean weekly precision@150, with Brier score and stronger regularization as tiebreaks.

CatBoost uses the same rows and features, native categorical inputs, 500 iterations, a 0.05 learning rate, and fixed random seed. Depths 4, 6, and 8 are compared; depth 6 wins the operational metric.

| Selected model | Validation cancellations reached | Calls | Mean weekly precision@150 |
| --- | --- | --- | --- |
| Logistic regression, C = 1 | 1,132 | 1,950 | 58.05% |
| CatBoost, depth 6 | 1,137 | 1,950 | 58.31% |

CatBoost has better average precision, but reaches only five additional cancellation outcomes. Under the evaluation rule favoring simplicity when queue performance is materially similar, logistic regression is selected. Average precision and Brier score are supporting diagnostics; ROC AUC is not the selection objective because it does not represent the weekly capacity constraint.

Probability calibration is imperfect. The score is used for ordering, and risk multiplied by value is a priority index rather than a calibrated expected-loss estimate.

## Queue design and policy selection

Bookings first become eligible on the Monday after their reconstructed booking week. Each decision excludes bookings already called, already arrived, or known to have canceled before the snapshot. Same-day cancellations remain eligible because date-only records cannot establish morning ordering. Validation and holdout each start with an empty queue.

The three validation rankings are risk alone, risk times booked value, and risk times value times a mild arrival-proximity boost. Booked value is `max(adr, 0) * total_nights`; flooring applies only to this exposure calculation.

**Risk times booked value is selected.** Compared with risk alone, it reaches fewer cancellation outcomes (1,055 versus 1,132), but 8.7% more loss under the linear-resale assumption and 12.7% more canceled booking value. The urgency boost adds 2.4% more linear-resale loss and only 83 units of canceled booking value while reducing precision further. The simpler value-weighted ranking is preferred.

Loss valuation uses actual cancellation notice before arrival. That outcome information enters evaluation only. It is kept separate from arrival proximity at the call, which is known when ranking. Changing the urgency window therefore cannot change the definition of the evaluation loss.

## Holdout findings

Across 13 holdout weeks, the selected policy fills **1,950 call slots** and reaches **1,146 target-positive bookings**, comprising 1,118 cancellations and 28 no-shows. Mean weekly precision@150 is **58.77%**, compared with **25.99%** for a random queue averaged over 100 seeded simulations.

The selected outcomes represent **803,697 ADR-equivalent units** of canceled booked room value, versus 693,506 for risk-only ranking: a **15.9% increase**. Value prioritization reaches fewer cancellation outcomes than risk-only ranking (1,146 versus 1,278) but greater revenue exposure.

The weekly schedule cannot reach 3,207 bookings that canceled before their first eligible Monday. The quarter ends with 5,621 active, uncalled bookings, of which 717 eventually cancel. These figures describe the simulated queue, not successful guest contacts or prevented cancellations.

## Financial scenarios

The dataset records no outreach, offers, controls, or room-resale revenue. Intervention effectiveness and recovered room value must remain explicit assumptions.

Two loss bases bracket resale recovery: a linear recovery curve that assigns no loss once cancellation notice reaches 30 days, and a no-resale bound that assigns the full booked value to every cancellation. The former is optimistic for room recovery and pessimistic for intervention value; the latter reverses those directions.

```text
benefit = assumed save rate * contribution margin * loss reached
cost = calls * (call cost + effective credit cost per call)
net return = benefit - cost
```

The scenario grid uses 5%, 10%, and 20% save rates, 70% contribution margin, $5 staff cost per attempted call, and $25 or $50 effective credit cost per attempt. The $25 case can represent partial contact, acceptance, redemption, or marginal-cost effects. ADR currency is undocumented; monetary scenarios assume conversion into matching dollar-equivalent units, with no inflation adjustment.

At a $25 effective credit cost, quarterly cost is **$58,500**. Linear resale recovery yields −$49,375 at a 10% save rate and −$40,251 at 20%, with break-even requiring **64.1%**. No resale yields −$2,241 and $54,018 respectively, with break-even at **10.4%**. At a $50 effective credit cost, no-resale break-even rises to **19.1%**.

These are sensitivity scenarios, not forecasts. The model does not establish any save rate, and equal outreach effectiveness across cancellations and no-shows is a simplifying assumption.

## Limitations and next steps

Results are restricted to two historical hotels and the complete 180-day population. Feature values are not verified booking-time snapshots; dates lack timestamps; exact matching rows lack reservation identifiers. Queue simulations omit incumbent backlog and use fixed scores, while the quarter ends with some bookings still waiting. Neither true contactability nor causal outreach effectiveness can be measured from this file.

An operational extension should first obtain genuine booking-time features, reservation IDs, current pricing and margins, and resale recovery by notice period. A randomized pilot can estimate incremental stay completion, contact rates, credit costs, and staff effort. Alternative contact timing and queue designs require separate evaluation.

## Reproducibility and development

The six notebooks include saved outputs and rebuild their required data and models from the original CSV. Shared preparation, model, and evaluation logic lives in `modeling/`. Fixed periods, model settings, and random seeds make the experiment repeatable; dependency versions can affect numerical results.

The final evaluation was verified with Python 3.13, NumPy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1, and CatBoost 1.2.10. Install the project requirements and run notebooks in numerical order as described in the [README](../README.md).

AI coding assistants supported implementation and review of methodological assumptions. The notebooks retain the reasoning behind the choices, and the shared code exposes the calculations used in the findings.
