# Medicaid Enrollment & Expenditure Forecast (2025-2034)

Forecasts national and state-by-state Medicaid enrollment and expenditures
for the next 10 years, using historical annual data from 2014-2024 sourced
from [medicaid.gov](https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data/medicaid-enrollment-data-collected-through-mbes).

## Why not just draw a straight trend line?

COVID-era rules barred states from disenrolling anyone, so enrollment
spiked to ~97M by 2022. Once those rules ended, states re-checked
everyone's eligibility ("unwinding") and enrollment dropped back down
through 2023-2024. A simple trend line ignores this spike-and-drop and
predicts an unrealistic climb back to 110M+. This project instead assumes
enrollment declines a bit more, then levels off — based on real 2024/2025
data and expected effects of the 2025 reconciliation bill. Expenditure had
no such spike, so it's forecast separately with a simpler growth trend.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 forecast_medicaid.py
```

This always produces the **national** forecast, since historical national
totals are built into the script.

For **state-by-state** forecasts, place these two files in `data/` first:

```
data/enrollment_annual_clean.csv     columns: State, medicaid_enrollees, year, month
data/expenditures_annual_clean.csv   columns: State, year, medicaid_expenditure, chip_expenditure, total_expenditure
```

If they're missing, the script just skips state-level output and tells you why.

## Output

Everything is written to `output/`:

| File | Description |
|---|---|
| `national_enrollment_forecast.csv` / `.png` | National enrollment, 2025-2034 |
| `national_expenditure_forecast.csv` / `.png` | National expenditure, 2025-2034 |
| `state_enrollment_forecast.csv` | Per-state enrollment forecast + YoY % change *(needs state data)* |
| `state_enrollment_forecast.png` | Top 10 states by 2024 enrollment *(needs state data)* |
| `state_expenditure_forecast.csv` | Per-state expenditure forecast + YoY % change *(needs state data)* |
| `state_expenditure_forecast.png` | Top 10 states by 2024 expenditure *(needs state data)* |

## Methodology (short version)

- **National enrollment**: keeps dropping through 2026, then flattens into
  slow growth — slower than pre-pandemic, since the 2025 reconciliation
  bill is expected to keep pushing enrollment down.
- **National expenditure**: fit a steady growth-rate trend, since
  expenditure never had the COVID spike/unwind enrollment did.
- **State enrollment**: each state follows the national trend, nudged by
  its own recent history so states that diverged from the national pattern
  keep some of that difference.
- **State expenditure**: each state gets its own independent growth trend,
  since expenditure trends are steady enough per state to trust on their own.

## Data cleaning notes

Raw medicaid.gov files have messy state names (extra spaces, footnote
marks, inconsistent abbreviations) and some enrollee counts stored as text
like `'470,049'` instead of numbers. Both need cleaning before the
enrollment and expenditure files can be merged by state and year.
