# Hotel Booking Retention: Executive Summary

An illustrative study of cancellation-risk targeting for a hotel team with capacity for 150 calls per week and a $50 property-credit offer.

## Findings

The selected approach ranks eligible bookings by cancellation risk multiplied by booked room value. In a 13-week historical holdout, 1,950 call slots reached **1,146 bookings that later canceled or became no-shows**: **58.8%**, compared with **26.0%** under random selection. Value weighting reached 15.9% more canceled booking value than risk-only ranking, while reaching fewer cancellation outcomes.

This demonstrates retrospective targeting performance. It does not demonstrate that calling these guests would change their behavior.

## Business case

The hypothetical program costs **$58,500–$107,250 per quarter**, assuming $5 staff cost and $25–$50 effective credit cost per attempted call. At a $25 credit cost and 70% contribution margin:

| Room-resale assumption | Net return at 10% save rate | Net return at 20% save rate | Break-even save rate |
| --- | --- | --- | --- |
| Linear recovery within 30 days | −$49,375 | −$40,251 | 64.1% |
| No resale | −$2,241 | $54,018 | 10.4% |

Save rates, costs, margin, and resale recovery are assumptions. The source does not document ADR currency; these scenarios assume matching dollar-equivalent units. Unobserved resale recovery determines much of the spread between loss and gain.

## Recommendation

Measure room-resale recovery by cancellation notice period, then consider a randomized pilot. Record stay completion, contact and acceptance rates, credit redemption, handling time, and contribution margin. The comparison group is necessary to distinguish prevented cancellations from guests who would have stayed anyway.

The weekly schedule also leaves operational gaps: **3,207** holdout bookings canceled before their first possible Monday, and **5,621** bookings remained uncalled at the end of the quarter. A different contact schedule or staffing level would need separate evaluation.

## Scope

Results concern two hotels in 2015–2017 and complete booking cohorts with lead times up to 180 days. Booking dates are reconstructed and feature values are not verified booking-time snapshots. Historical targeting performance and scenario returns should be validated on current operational data before deployment.

See the [technical report](technical_report.md) and [final evaluation notebook](../modeling/notebooks/06_final_evaluation.ipynb) for methods and supporting tables. Dataset: Antonio, de Almeida, and Nunes (2019), [Hotel booking demand datasets](https://doi.org/10.1016/j.dib.2018.11.126).
