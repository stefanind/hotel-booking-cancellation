# Hotel Booking Cancellation & Retention Analytics

A data science project that predicts hotel booking cancellations and uses those scores to prioritize a limited weekly outreach queue. It combines chronological model evaluation with booking-value prioritization and financial scenarios for a hypothetical retention program.

The central question is practical: **with capacity for 150 calls a week, which bookings should a hotel contact, and what would have to be true for the outreach to pay for itself?**

## Results

Logistic regression and a **cancellation risk × booked room value** ranking were selected on validation data, then evaluated on a separate 13-week holdout period.

| Holdout measure | Result |
| --- | --- |
| Outreach attempts | 1,950 |
| Selected bookings that later canceled or became no-shows | 1,146 |
| Mean weekly precision@150 | **58.8%**, versus **26.0%** for a capacity-matched random queue |
| Booked room value associated with those outcomes | 803,697 ADR-equivalent units |
| Improvement in canceled booking value reached over risk-only ranking | **15.9%** |

These are retrospective targeting results. The dataset contains no outreach experiments, so it cannot establish how many cancellations a call would prevent. Financial viability depends strongly on unobserved room-resale recovery and the effectiveness and cost of the offer.

## Read the project

- [Executive summary](docs/executive_summary.md) ([PDF](docs/executive_summary.pdf)): findings, business implications, and proposed next steps.
- [Technical report](docs/technical_report.md) ([PDF](docs/technical_report.pdf)): data preparation, model and policy selection, evaluation, and limitations.
- [Final evaluation notebook](modeling/notebooks/06_final_evaluation.ipynb): holdout tables, an example call list, return scenarios, and a generated findings summary in section 6.

Notebooks include saved outputs and can be reviewed without running them. Read them in numerical order to follow the full analysis.

| Notebook | Purpose |
| --- | --- |
| [01_prepare_data.ipynb](modeling/notebooks/01_prepare_data.ipynb) | Data quality checks, leakage exclusions, population scope, and chronological splits |
| [02_define_evaluation.ipynb](modeling/notebooks/02_define_evaluation.ipynb) | Evaluation rules, weekly queue mechanics, selection criteria, and financial assumptions defined before fitting |
| [03_logistic_regression.ipynb](modeling/notebooks/03_logistic_regression.ipynb) | Interpretable baseline and a small regularization comparison |
| [04_catboost.ipynb](modeling/notebooks/04_catboost.ipynb) | Gradient-boosted comparison using the same features, populations, and queue |
| [05_select_calling_policy.ipynb](modeling/notebooks/05_select_calling_policy.ipynb) | Validation comparison of risk, value, and arrival-proximity rankings |
| [06_final_evaluation.ipynb](modeling/notebooks/06_final_evaluation.ipynb) | Frozen model and policy evaluated on holdout, with economic scenarios |

Shared implementation lives in [prepare_data.py](modeling/prepare_data.py), [models.py](modeling/models.py), and [evaluation_policy.py](modeling/evaluation_policy.py).

## Illustrative operating scenario

The project models a hypothetical hotel team with up to **150 outreach attempts each Monday**, one call per booking, and a **$50 property-credit offer**. Uncalled eligible bookings carry forward; canceled or arrived bookings leave the queue. A booking receives one fixed risk score when it enters the analysis.

Financial scenarios vary assumed save rates (5%, 10%, and 20%) and effective credit costs ($25 or $50 per attempted call), with a $5 call cost and 70% contribution margin. Cancellation exposure is valued under two assumptions: linear resale recovery within 30 days, and no resale. These are scenario inputs, not facts inferred from the dataset. The source does not document ADR currency; dollar scenarios assume ADR has already been converted into matching units.

Capacity and financial inputs are arguments to the shared evaluation functions. The saved results use the settings above; changing them requires rerunning the relevant analysis and interpreting precision against the new capacity.

## Dataset and attribution

`hotel_bookings.csv` contains **119,390 bookings and 32 columns**, combining resort and city hotel records with a hotel-type indicator. Arrival dates span July 2015 through August 2017.

Original source: Nuno Antonio, Ana de Almeida, and Luis Nunes (2019), *Hotel booking demand datasets*, **Data in Brief**, 22, 41–49. [Publication and data dictionary](https://doi.org/10.1016/j.dib.2018.11.126) · [Open-access article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/).

The repository includes the combined CSV used by the notebooks. The original data publication describes 31 variables per hotel; the combined file adds the `hotel` indicator. Cleaning happens on an in-memory copy, and the source CSV is preserved.

## Run locally

Use Python 3.13 and create a virtual environment from the repository root:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
jupyter lab modeling/notebooks
```

Run the notebooks in numerical order. Each reconstructs the data and model it needs; there are no hidden serialized models or intermediate data files. The notebooks find `hotel_bookings.csv` by walking up from their working directory, so the checkout folder can have any name.

The saved analysis uses fixed model settings, seeds, and chronological periods. The random reference is repeated 100 times with reproducible seeds. Dependency ranges in `requirements.txt` allow compatible updates; see the technical report for the versions used to verify the final evaluation. Runtime is a few minutes, with the CatBoost comparison taking the longest.

## Limitations and next steps

The source is historical data from two hotels, with reconstructed booking dates and no true booking-time feature snapshots. Analysis is restricted to complete booking cohorts with lead times of 0–180 days. Risk scores support ordering but are not calibrated estimates of expected financial loss.

Before using this approach operationally, collect genuine booking-time snapshots, reservation identifiers, contact records, and room-resale economics. A randomized pilot is needed to measure whether outreach prevents cancellations and whether incremental benefits exceed staff and credit costs.
