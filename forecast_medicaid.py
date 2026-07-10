"""
forecast_medicaid.py

Forecasts Medicaid enrollment and expenditures, nationwide and state-by-state,
for 2025-2034, using historical annual data 2014-2024.

INPUTS (expected in ./data/):
    enrollment_annual_clean.csv    columns: State, medicaid_enrollees, year, month
    expenditures_annual_clean.csv  columns: State, year, medicaid_expenditure,
                                             chip_expenditure, total_expenditure

    If these aren't present, the script still runs the NATIONAL forecast using
    the embedded reference totals below, and skips the state-level section
    with a clear message telling you what to drop into ./data/.

OUTPUTS (written to ./output/):
    national_enrollment_forecast.csv
    national_expenditure_forecast.csv
    state_enrollment_forecast.csv        (if state data available)
    state_expenditure_forecast.csv       (if state data available)
    national_enrollment_forecast.png
    national_expenditure_forecast.png
    state_enrollment_forecast.png        (if state data available, top 10 states by enrollment)
    state_expenditure_forecast.png       (if state data available, top 10 states by expenditure)

METHODOLOGY (see project handoff doc for full reasoning)
    National enrollment:
        - 2025-2026: enrollment keeps falling but the annual decline TAPERS
          (roughly -4% in 2025 easing to 0% by 2027), matching the real
          KFF-reported FY2024/FY2025 pace rather than the sharper -6.3%/yr
          unwinding-era rate.
        - 2027 onward: modest growth, but held BELOW the pre-pandemic CAGR
          (+0.265%/yr) to reflect the 2025 reconciliation bill's downward
          pressure on enrollment for the rest of the decade.
        - Uncertainty band: DOWNSIDE-ONLY, widening linearly from -2% in
          2025 to -10% in 2034. Only the downside is shown because a
          further policy-driven shortfall below this path is much more
          plausible than a meaningful overshoot above it. (The
          enrollment_high column is still written to the CSV in case it's
          useful, but the chart only shades the low side.)

    National expenditure:
        - No structural break in the historical data, so this uses a
          straightforward log-linear (constant-growth-rate) regression
          fit on 2014-2024 and extrapolates. A log-linear fit is
          equivalent to assuming a constant CAGR, which matches the
          steady historical pattern much better than a plain linear fit.
        - No uncertainty band shown on this chart or in its CSV -
          expenditure has been steady enough historically that a band
          wasn't adding useful information.

    State-level (hybrid approach):
        - Fully independent per-state trend fits are noisy: 11 points per
          state, and many states have small populations, so year-to-year
          swings look like "trend" when they're mostly noise.
        - A single national shape copy-pasted onto every state ignores the
          fact that some states never dropped below pre-pandemic
          enrollment while others cratered during unwinding.
        - Hybrid: each state is projected forward using the NATIONAL
          scenario's year-over-year growth path as the baseline, PLUS a
          shrinkage-weighted "drift" term equal to
          (state's own 2019-2024 CAGR) - (national 2019-2024 CAGR).
          The drift is shrunk toward 0 by STATE_DRIFT_WEIGHT (default 0.4)
          so idiosyncratic state trends partially carry forward without
          each state's forecast being dominated by 11-point noise.
        - Expenditure has no structural break, so each state's own
          log-linear fit is used directly (no shrinkage needed) -
          expenditure trends are steady enough that per-state regression
          is reliable on its own.
"""

import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "output"
HIST_START, HIST_END = 2014, 2024
FCST_START, FCST_END = 2025, 2034
STATE_DRIFT_WEIGHT = 0.4  # 0 = pure national shape, 1 = pure independent state trend

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Embedded national reference data (Section 7 of the handoff doc)
# Used directly so the national forecast can run with zero file uploads.
# --------------------------------------------------------------------------
NATIONAL_CSV = """year,total_enrollees,total_medicaid_exp,total_chip_exp,total_exp
2014,74025130,467426379977.0,6215662924,473642042901.0
2015,76137348,526710873983.0,7621752400,534332626383.0
2016,76034466,550881322328.0,9194251220,560075573548.0
2017,76766099,572243939611.0,10755496429,582999436040.0
2018,75799027,587506879752.0,10121581735,597628461487.0
2019,75011359,597385302023.0,11024078300,608409380323.0
2020,84029028,652931212149.0,11368112981,664299325130.0
2021,90288880,717143060778.0,12852327711,729995388489.0
2022,96904625,792734393498.0,14467542210,807201935708.0
2023,93799620,859881646027.31,15262342499,875143988526.31
2024,85058864,908839083557.1,18588119293,927427202850.1
"""


def load_national_history() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(NATIONAL_CSV))
    return df


def round_sig(x, sig=2):
    """Round x to the given number of significant figures."""
    if x == 0 or pd.isna(x):
        return 0.0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))


# --------------------------------------------------------------------------
# National enrollment forecast
# --------------------------------------------------------------------------
def build_national_enrollment_forecast(hist: pd.DataFrame) -> pd.DataFrame:
    base_year = HIST_END
    base_value = hist.loc[hist.year == base_year, "total_enrollees"].iloc[0]

    # Tapering decline 2025-2026, easing linearly from -4% to 0% by 2027,
    # then modest growth ramping from 0% up to +0.15%/yr (held below the
    # +0.265%/yr pre-pandemic CAGR) by 2029 and flat thereafter.
    decline_2025 = -0.04
    decline_2026 = -0.02
    growth_final = 0.0015  # +0.15%/yr, below pre-pandemic +0.265%/yr

    years = list(range(FCST_START, FCST_END + 1))
    growth_rates = []
    for y in years:
        if y == 2025:
            g = decline_2025
        elif y == 2026:
            g = decline_2026
        elif y == 2027:
            g = 0.0
        elif y == 2028:
            g = growth_final / 2  # ramp
        else:
            g = growth_final
        growth_rates.append(g)

    values = []
    v = base_value
    for g in growth_rates:
        v = v * (1 + g)
        values.append(v)

    # Uncertainty band: +/-2% (2025) widening linearly to +/-10% (2034)
    band_pct = np.linspace(0.02, 0.10, len(years))

    out = pd.DataFrame({
        "year": years,
        "yoy_growth_rate": growth_rates,
        "enrollment_most_likely": values,
        "enrollment_low": [v * (1 - b) for v, b in zip(values, band_pct)],
        "enrollment_high": [v * (1 + b) for v, b in zip(values, band_pct)],
    })
    return out, growth_rates


# --------------------------------------------------------------------------
# National expenditure forecast (log-linear / constant-CAGR regression)
# --------------------------------------------------------------------------
def loglinear_fit_extrapolate(hist_years, hist_values, fcst_years):
    """Fit ln(value) = a + b*year via OLS, return extrapolated values."""
    x = np.array(hist_years, dtype=float)
    y = np.log(np.array(hist_values, dtype=float))
    b, a = np.polyfit(x, y, 1)  # slope, intercept
    fitted = np.exp(a + b * np.array(fcst_years, dtype=float))
    annual_growth = np.exp(b) - 1
    return fitted, annual_growth


def build_national_expenditure_forecast(hist: pd.DataFrame) -> pd.DataFrame:
    years = list(range(FCST_START, FCST_END + 1))
    fitted, growth = loglinear_fit_extrapolate(
        hist.year, hist.total_exp, years
    )

    band_pct = np.linspace(0.015, 0.06, len(years))
    out = pd.DataFrame({
        "year": years,
        "implied_annual_growth_rate": growth,
        "expenditure_most_likely": fitted,
        "expenditure_low": fitted * (1 - band_pct),
        "expenditure_high": fitted * (1 + band_pct),
    })
    return out


# --------------------------------------------------------------------------
# State-level data loading
# --------------------------------------------------------------------------
def load_state_data():
    enr_path = os.path.join(DATA_DIR, "enrollment_annual_clean.csv")
    exp_path = os.path.join(DATA_DIR, "expenditures_annual_clean.csv")
    if not (os.path.exists(enr_path) and os.path.exists(exp_path)):
        print(
            "\n[state-level] Skipped: didn't find\n"
            f"    {enr_path}\n    {exp_path}\n"
            "Drop your enrollment_annual_clean.csv and "
            "expenditures_annual_clean.csv into ./data/ and re-run to get "
            "state-by-state forecasts.\n"
        )
        return None, None
    enr = pd.read_csv(enr_path)
    exp = pd.read_csv(exp_path)
    return enr, exp


# --------------------------------------------------------------------------
# State-level enrollment forecast (hybrid: national shape + shrunk drift)
# --------------------------------------------------------------------------
def cagr(v_start, v_end, n_years):
    if v_start <= 0 or n_years <= 0:
        return 0.0
    return (v_end / v_start) ** (1 / n_years) - 1


def build_state_enrollment_forecast(enr: pd.DataFrame, national_growth_rates):
    years_fcst = list(range(FCST_START, FCST_END + 1))

    # National 2019-2024 CAGR, for computing each state's "drift"
    nat = load_national_history()
    nat_2019 = nat.loc[nat.year == 2019, "total_enrollees"].iloc[0]
    nat_2024 = nat.loc[nat.year == 2024, "total_enrollees"].iloc[0]
    national_cagr_19_24 = cagr(nat_2019, nat_2024, 5)

    rows = []
    for state, g in enr.groupby("State"):
        g = g.sort_values("year")
        if 2019 not in g.year.values or 2024 not in g.year.values:
            continue  # incomplete history for this state, skip
        v2019 = g.loc[g.year == 2019, "medicaid_enrollees"].iloc[0]
        v2024 = g.loc[g.year == 2024, "medicaid_enrollees"].iloc[0]
        state_cagr_19_24 = cagr(v2019, v2024, 5)

        drift = STATE_DRIFT_WEIGHT * (state_cagr_19_24 - national_cagr_19_24)

        v = v2024
        for y, g_nat in zip(years_fcst, national_growth_rates):
            v = v * (1 + g_nat + drift)
            rows.append({"State": state, "year": y, "enrollment_forecast": v})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# State-level expenditure forecast (independent per-state log-linear fit)
# --------------------------------------------------------------------------
def build_state_expenditure_forecast(exp: pd.DataFrame):
    years_fcst = list(range(FCST_START, FCST_END + 1))
    rows = []
    for state, g in exp.groupby("State"):
        g = g.sort_values("year")
        if len(g) < 3:
            continue  # not enough points for a stable fit
        fitted, _ = loglinear_fit_extrapolate(
            g.year, g.total_expenditure.clip(lower=1), years_fcst
        )
        for y, val in zip(years_fcst, fitted):
            rows.append({"State": state, "year": y, "expenditure_forecast": val})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def millions_formatter(x, pos):
    return f"{x/1e6:.0f}M"


def billions_formatter(x, pos):
    return f"${x/1e9:.0f}B"


def plot_national_enrollment(hist, fcst):
    fig, ax = plt.subplots(figsize=(11, 6))

    # Prepend the last historical point to the forecast series so the two
    # lines connect visually with no gap at the 2024/2025 seam.
    bridge_year = hist.year.iloc[-1]
    bridge_val = hist.total_enrollees.iloc[-1]
    fcst_x = [bridge_year] + fcst.year.tolist()
    fcst_y = [bridge_val] + fcst.enrollment_most_likely.tolist()
    fcst_low = [bridge_val] + fcst.enrollment_low.tolist()

    ax.plot(hist.year, hist.total_enrollees, marker="o", color="#2c5f8a",
             label="Historical (2014-2024)")
    # Line only (no markers) so the red line can pass through the 2024
    # point without drawing a red dot on top of the blue historical one.
    ax.plot(fcst_x, fcst_y, color="#d1495b",
             label="Forecast, most likely (2025-2034)")
    # Markers only for the actual forecast years (2025+), not the 2024 bridge.
    ax.plot(fcst.year, fcst.enrollment_most_likely, marker="o", linestyle="None",
             color="#d1495b")
    # Only the lower half of the band is shaded: there's a meaningfully
    # higher chance enrollment comes in below this path (further policy-
    # driven declines) than meaningfully above it.
    ax.fill_between(fcst_x, fcst_low, fcst_y,
                      color="#d1495b", alpha=0.15, label="Downside uncertainty")
    ax.set_xticks(range(HIST_START, FCST_END + 1, 1))
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(50e6, 110e6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
    ax.set_title("National Medicaid Enrollment: History & Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total enrollees")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "national_enrollment_forecast.png"), dpi=150)
    plt.close(fig)


def plot_national_expenditure(hist, fcst):
    fig, ax = plt.subplots(figsize=(11, 6))

    bridge_year = hist.year.iloc[-1]
    bridge_val = hist.total_exp.iloc[-1]
    fcst_x = [bridge_year] + fcst.year.tolist()
    fcst_y = [bridge_val] + fcst.expenditure_most_likely.tolist()

    ax.plot(hist.year, hist.total_exp, marker="o", color="#2c5f8a",
             label="Historical (2014-2024)")
    ax.plot(fcst_x, fcst_y, color="#d1495b",
             label="Forecast, most likely (2025-2034)")
    ax.plot(fcst.year, fcst.expenditure_most_likely, marker="o", linestyle="None",
             color="#d1495b")
    ax.set_xticks(range(HIST_START, FCST_END + 1, 1))
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(billions_formatter))
    ax.set_title("National Medicaid + CHIP Expenditure: History & Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total expenditure")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "national_expenditure_forecast.png"), dpi=150)
    plt.close(fig)


def plot_top_states_enrollment(enr_hist, state_fcst, n=10):
    latest = enr_hist[enr_hist.year == HIST_END].sort_values(
        "medicaid_enrollees", ascending=False
    )
    top_states = latest.head(n).State.tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    for state in top_states:
        h = enr_hist[enr_hist.State == state].sort_values("year")
        f = state_fcst[state_fcst.State == state].sort_values("year")
        bridge_year = h.year.iloc[-1]
        bridge_val = h.medicaid_enrollees.iloc[-1]
        f_x = [bridge_year] + f.year.tolist()
        f_y = [bridge_val] + f.enrollment_forecast.tolist()

        line, = ax.plot(h.year, h.medicaid_enrollees, marker="o", markersize=3)
        ax.plot(f_x, f_y, linestyle="--", color=line.get_color(), markersize=3)
        ax.annotate(state, (f_x[-1], f_y[-1]),
                    fontsize=8, color=line.get_color(),
                    xytext=(4, 0), textcoords="offset points")
    ax.set_xticks(range(HIST_START, FCST_END + 1, 1))
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
    ax.set_title(f"Top {n} States by 2024 Medicaid Enrollment: History (solid) & Forecast (dashed)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Enrollees")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "state_enrollment_forecast.png"), dpi=150)
    plt.close(fig)


def plot_top_states_expenditure(exp_hist, state_fcst, n=10):
    latest = exp_hist[exp_hist.year == HIST_END].sort_values(
        "total_expenditure", ascending=False
    )
    top_states = latest.head(n).State.tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    for state in top_states:
        h = exp_hist[exp_hist.State == state].sort_values("year")
        f = state_fcst[state_fcst.State == state].sort_values("year")
        bridge_year = h.year.iloc[-1]
        bridge_val = h.total_expenditure.iloc[-1]
        f_x = [bridge_year] + f.year.tolist()
        f_y = [bridge_val] + f.expenditure_forecast.tolist()

        line, = ax.plot(h.year, h.total_expenditure, marker="o", markersize=3)
        ax.plot(f_x, f_y, linestyle="--", color=line.get_color(), markersize=3)
        ax.annotate(state, (f_x[-1], f_y[-1]),
                    fontsize=8, color=line.get_color(),
                    xytext=(4, 0), textcoords="offset points")
    ax.set_xticks(range(HIST_START, FCST_END + 1, 1))
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(billions_formatter))
    ax.set_title(f"Top {n} States by 2024 Medicaid+CHIP Expenditure: History (solid) & Forecast (dashed)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Expenditure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "state_expenditure_forecast.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    hist = load_national_history()

    # National enrollment
    nat_enr_fcst, national_growth_rates = build_national_enrollment_forecast(hist)
    plot_national_enrollment(hist, nat_enr_fcst)
    nat_enr_out = nat_enr_fcst.copy()
    nat_enr_out["yoy_growth_rate"] = nat_enr_out["yoy_growth_rate"].round(4)
    for c in ["enrollment_most_likely", "enrollment_low", "enrollment_high"]:
        nat_enr_out[c] = nat_enr_out[c].round().astype(int)
    nat_enr_out.to_csv(os.path.join(OUTPUT_DIR, "national_enrollment_forecast.csv"), index=False)
    print("Wrote national_enrollment_forecast.csv + .png")

    # National expenditure
    nat_exp_fcst = build_national_expenditure_forecast(hist)
    plot_national_expenditure(hist, nat_exp_fcst)
    nat_exp_out = nat_exp_fcst[["year", "implied_annual_growth_rate", "expenditure_most_likely"]].copy()
    nat_exp_out["implied_annual_growth_rate"] = nat_exp_out["implied_annual_growth_rate"].round(4)
    nat_exp_out["expenditure_most_likely"] = nat_exp_out["expenditure_most_likely"].round().astype(int)
    nat_exp_out.to_csv(os.path.join(OUTPUT_DIR, "national_expenditure_forecast.csv"), index=False)
    print("Wrote national_expenditure_forecast.csv + .png")

    # State-level
    enr, exp = load_state_data()
    if enr is not None:
        state_enr_fcst = build_state_enrollment_forecast(enr, national_growth_rates)
        plot_top_states_enrollment(enr, state_enr_fcst)
        state_enr_out = state_enr_fcst.copy()
        state_enr_out["enrollment_forecast"] = state_enr_out["enrollment_forecast"].round().astype(int)
        state_enr_out = state_enr_out.sort_values(["State", "year"]).reset_index(drop=True)

        base_2024 = enr[enr.year == HIST_END].set_index("State")["medicaid_enrollees"]
        state_enr_out["prev_value"] = state_enr_out.groupby("State")["enrollment_forecast"].shift(1)
        needs_base = state_enr_out["prev_value"].isna()
        state_enr_out.loc[needs_base, "prev_value"] = state_enr_out.loc[needs_base, "State"].map(base_2024)
        state_enr_out["yoy_pct_change"] = (
            (state_enr_out["enrollment_forecast"] - state_enr_out["prev_value"])
            / state_enr_out["prev_value"] * 100
        ).apply(round_sig)
        state_enr_out = state_enr_out.drop(columns=["prev_value"])
        state_enr_out.to_csv(os.path.join(OUTPUT_DIR, "state_enrollment_forecast.csv"), index=False)
        print("Wrote state_enrollment_forecast.csv + .png")

        state_exp_fcst = build_state_expenditure_forecast(exp)
        plot_top_states_expenditure(exp, state_exp_fcst)
        state_exp_out = state_exp_fcst.copy()
        state_exp_out["expenditure_forecast"] = state_exp_out["expenditure_forecast"].round().astype(int)
        state_exp_out = state_exp_out.sort_values(["State", "year"]).reset_index(drop=True)

        base_2024_exp = exp[exp.year == HIST_END].set_index("State")["total_expenditure"]
        state_exp_out["prev_value"] = state_exp_out.groupby("State")["expenditure_forecast"].shift(1)
        needs_base = state_exp_out["prev_value"].isna()
        state_exp_out.loc[needs_base, "prev_value"] = state_exp_out.loc[needs_base, "State"].map(base_2024_exp)
        state_exp_out["yoy_pct_change"] = (
            (state_exp_out["expenditure_forecast"] - state_exp_out["prev_value"])
            / state_exp_out["prev_value"] * 100
        ).apply(round_sig)
        state_exp_out = state_exp_out.drop(columns=["prev_value"])
        state_exp_out.to_csv(os.path.join(OUTPUT_DIR, "state_expenditure_forecast.csv"), index=False)
        print("Wrote state_expenditure_forecast.csv + .png")

    print("\nDone. See ./output/ for all files.")


if __name__ == "__main__":
    main()
