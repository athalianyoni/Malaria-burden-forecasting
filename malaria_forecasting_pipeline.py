"""
Malaria Burden Forecasting for Hospital Preparedness
Peru proof of concept for a future Howard Mission Hospital adaptation.

This script reproduces the main modelling workflow used in the project.
It expects cleaned malaria surveillance records and hourly weather data.

IMPORTANT:
- The fitted Peru model is not intended for direct operational use in Zimbabwe.
- The transferable part is the workflow, which must be retrained and validated
  using local Howard Mission Hospital data before any operational use.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial
import statsmodels.api as sm


# -----------------------------------------------------------------------------
# 1. FILE PATHS AND COLUMN NAMES
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MALARIA_FILE = DATA_DIR / "peru_malaria_cleaned.csv"
WEATHER_FILE = DATA_DIR / "open_meteo_hourly.xlsx"

YEAR_COL = "year"
WEEK_COL = "epidermiological week"  # spelling preserved from the original data
TEMP_COL = "temperature_2m (°C)"
RAIN_COL = "rain (mm)"
TIME_COL = "time"


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def rmse(y_true, y_pred):
    """Return root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_epi_week(row):
    """
    Convert a Sunday week-start date into the project epidemiological week.

    Week 1 is the Sunday-starting week that contains January 4.
    """
    year = int(row["epi_year"])
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    first_sunday = jan4 - pd.Timedelta(days=(jan4.weekday() + 1) % 7)
    return ((row["week_start"] - first_sunday).days // 7) + 1


def fit_nb(train_df, test_df, features, scale_features=None):
    """
    Fit a statsmodels Negative Binomial regression and return predictions.

    Parameters
    ----------
    train_df, test_df : pandas.DataFrame
        Training and future evaluation data.
    features : list[str]
        Predictor column names.
    scale_features : list[str] or None
        Columns to standardize using training data only.
    """
    train_work = train_df.copy()
    test_work = test_df.copy()

    model_features = list(features)

    if scale_features:
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_work[scale_features])
        scaled_test = scaler.transform(test_work[scale_features])

        for idx, col in enumerate(scale_features):
            scaled_name = f"{col}_scaled"
            train_work[scaled_name] = scaled_train[:, idx]
            test_work[scaled_name] = scaled_test[:, idx]
            model_features = [scaled_name if f == col else f for f in model_features]

    X_train = sm.add_constant(train_work[model_features], has_constant="add")
    y_train = train_work["cases"]

    model = NegativeBinomial(y_train, X_train).fit(
        method="lbfgs",
        maxiter=1000,
        disp=False,
    )

    X_test = sm.add_constant(test_work[model_features], has_constant="add")
    predictions = model.predict(X_test)

    return model, predictions, train_work, test_work, model_features


# -----------------------------------------------------------------------------
# 3. LOAD MALARIA DATA AND AGGREGATE TO WEEKLY COUNTS
# -----------------------------------------------------------------------------

if not MALARIA_FILE.exists():
    raise FileNotFoundError(
        f"Missing malaria file: {MALARIA_FILE}\n"
        "Place the cleaned surveillance CSV in data/ or update MALARIA_FILE."
    )

if not WEATHER_FILE.exists():
    raise FileNotFoundError(
        f"Missing weather file: {WEATHER_FILE}\n"
        "Place the hourly weather workbook in data/ or update WEATHER_FILE."
    )

malaria = pd.read_csv(MALARIA_FILE)

weekly_cases = (
    malaria.groupby([YEAR_COL, WEEK_COL])
    .size()
    .reset_index(name="cases")
)

print("\nWeekly malaria data shape:", weekly_cases.shape)
print(weekly_cases.head())


# -----------------------------------------------------------------------------
# 4. LOAD HOURLY WEATHER AND REBUILD EPIDEMIOLOGICAL WEEKS
# -----------------------------------------------------------------------------

weather = pd.read_excel(WEATHER_FILE)
weather[TIME_COL] = pd.to_datetime(weather[TIME_COL])

# Open-Meteo timestamps in the original workflow were treated as UTC/GMT.
# Peru local time was reconstructed as UTC-5.
weather["peru_time"] = weather[TIME_COL] - pd.Timedelta(hours=5)

# Sunday-Saturday weeks: W-SAT periods end on Saturday.
weather["week_end"] = (
    weather["peru_time"]
    .dt.to_period("W-SAT")
    .dt.end_time
    .dt.normalize()
)
weather["week_start"] = weather["week_end"] - pd.Timedelta(days=6)

# Thursday-based year anchor helps assign year-crossing weeks consistently.
weather["epi_year"] = (
    weather["week_start"] + pd.Timedelta(days=3)
).dt.year

weather["epi_week"] = weather.apply(get_epi_week, axis=1)

weekly_weather = (
    weather.groupby(["epi_year", "epi_week"])
    .agg(
        avg_temperature=(TEMP_COL, "mean"),
        total_rainfall=(RAIN_COL, "sum"),
        n_hours=("peru_time", "count"),
    )
    .reset_index()
)

# Retain complete 168-hour weeks for the modelling years.
weekly_weather_clean = weekly_weather[
    (weekly_weather["n_hours"] == 168)
    & (weekly_weather["epi_year"] >= 2018)
    & (weekly_weather["epi_year"] <= 2024)
].copy()

weekly_weather_clean = weekly_weather_clean.rename(
    columns={
        "epi_year": YEAR_COL,
        "epi_week": WEEK_COL,
    }
)


# -----------------------------------------------------------------------------
# 5. MERGE MALARIA AND WEATHER DATA
# -----------------------------------------------------------------------------

model_data = weekly_cases.merge(
    weekly_weather_clean[
        [YEAR_COL, WEEK_COL, "avg_temperature", "total_rainfall"]
    ],
    on=[YEAR_COL, WEEK_COL],
    how="inner",
    validate="one_to_one",
)

model_data = (
    model_data
    .sort_values([YEAR_COL, WEEK_COL])
    .reset_index(drop=True)
)

print("\nMerged modelling dataset shape:", model_data.shape)
print(model_data[["cases", "avg_temperature", "total_rainfall"]].describe())


# -----------------------------------------------------------------------------
# 6. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

# Case-count history.
for lag in range(1, 9):
    model_data[f"cases_lag_{lag}"] = model_data["cases"].shift(lag)

# Weather history.
for lag in range(1, 9):
    model_data[f"rain_lag_{lag}"] = model_data["total_rainfall"].shift(lag)
    model_data[f"temp_lag_{lag}"] = model_data["avg_temperature"].shift(lag)

# Annual cyclic seasonality.
model_data["sin_week"] = np.sin(
    2 * np.pi * model_data[WEEK_COL] / 52
)
model_data["cos_week"] = np.cos(
    2 * np.pi * model_data[WEEK_COL] / 52
)

# Additional exploratory features used later in the project.
model_data["sin_week_2"] = np.sin(
    4 * np.pi * model_data[WEEK_COL] / 52
)
model_data["cos_week_2"] = np.cos(
    4 * np.pi * model_data[WEEK_COL] / 52
)
model_data["cases_roll_4"] = (
    model_data["cases"].shift(1).rolling(4).mean()
)
model_data["cases_change"] = (
    model_data["cases_lag_1"] - model_data["cases_lag_2"]
)


# -----------------------------------------------------------------------------
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------

train = model_data[model_data[YEAR_COL] <= 2022].copy()
test = model_data[model_data[YEAR_COL] >= 2023].copy()

print("\nTraining years:", sorted(train[YEAR_COL].unique()))
print("Test years:", sorted(test[YEAR_COL].unique()))


# -----------------------------------------------------------------------------
# 8. PERSISTENCE BASELINE
# -----------------------------------------------------------------------------

baseline_test = test.dropna(subset=["cases_lag_1", "cases"]).copy()
baseline_test["baseline_prediction"] = baseline_test["cases_lag_1"]

mae_baseline = mean_absolute_error(
    baseline_test["cases"],
    baseline_test["baseline_prediction"],
)
rmse_baseline = rmse(
    baseline_test["cases"],
    baseline_test["baseline_prediction"],
)


# -----------------------------------------------------------------------------
# 9. MODEL 1: NEGATIVE BINOMIAL WITH CASE LAGS 1-4 + SEASONALITY
# -----------------------------------------------------------------------------

model1_features = [
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
    "sin_week",
    "cos_week",
]

train_m1 = train.dropna(subset=model1_features + ["cases"]).copy()
test_m1 = test.dropna(subset=model1_features + ["cases"]).copy()

nb_model1, pred_model1, _, _, _ = fit_nb(
    train_m1,
    test_m1,
    features=model1_features,
    scale_features=[
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",
    ],
)

mae_model1 = mean_absolute_error(test_m1["cases"], pred_model1)
rmse_model1 = rmse(test_m1["cases"], pred_model1)


# -----------------------------------------------------------------------------
# 10. MODEL 2: NEGATIVE BINOMIAL + TRAINING-SELECTED WEATHER LAGS
# -----------------------------------------------------------------------------

# In the original exploratory training-period correlation check, rain_lag_3 and
# temp_lag_8 had the largest absolute correlations among the tested weather lags.
model2_features = model1_features + ["rain_lag_3", "temp_lag_8"]

train_m2 = train.dropna(subset=model2_features + ["cases"]).copy()
test_m2 = test.dropna(subset=model2_features + ["cases"]).copy()

nb_model2, pred_model2, _, _, _ = fit_nb(
    train_m2,
    test_m2,
    features=model2_features,
    scale_features=[
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",
        "rain_lag_3",
        "temp_lag_8",
    ],
)

mae_model2 = mean_absolute_error(test_m2["cases"], pred_model2)
rmse_model2 = rmse(test_m2["cases"], pred_model2)


# -----------------------------------------------------------------------------
# 11. SIMPLIFIED NEGATIVE BINOMIAL
# -----------------------------------------------------------------------------

# For direct comparison with Model 1, use the same training rows that are
# available after requiring case lags 1-4, although the final simplified model
# itself uses only lag 1 plus seasonality.
common_required = [
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
    "sin_week",
    "cos_week",
    "cases",
]

train_simple = train.dropna(subset=common_required).copy()
test_simple = test.dropna(subset=["cases_lag_1", "sin_week", "cos_week", "cases"]).copy()

simple_features = ["cases_lag_1", "sin_week", "cos_week"]

nb_model_simple, simple_prediction, train_simple_fitted, test_simple_fitted, _ = fit_nb(
    train_simple,
    test_simple,
    features=simple_features,
    scale_features=["cases_lag_1"],
)

test_simple["simple_prediction"] = np.asarray(simple_prediction)

mae_simple = mean_absolute_error(test_simple["cases"], test_simple["simple_prediction"])
rmse_simple = rmse(test_simple["cases"], test_simple["simple_prediction"])


# -----------------------------------------------------------------------------
# 12. RANDOM FOREST COMPARATOR
# -----------------------------------------------------------------------------

rf_features = ["cases_lag_1", "sin_week", "cos_week"]

X_train_rf = train_simple[rf_features]
y_train_rf = train_simple["cases"]
X_test_rf = test_simple[rf_features]
y_test_rf = test_simple["cases"]

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42,
)
rf_model.fit(X_train_rf, y_train_rf)
rf_predictions = rf_model.predict(X_test_rf)

mae_rf = mean_absolute_error(y_test_rf, rf_predictions)
rmse_rf = rmse(y_test_rf, rf_predictions)


# -----------------------------------------------------------------------------
# 13. MODEL SCORECARD
# -----------------------------------------------------------------------------

results = pd.DataFrame(
    {
        "Model": [
            "Persistence baseline",
            "Negative Binomial - Model 1",
            "Negative Binomial + weather",
            "Simplified Negative Binomial",
            "Random Forest",
        ],
        "MAE": [
            mae_baseline,
            mae_model1,
            mae_model2,
            mae_simple,
            mae_rf,
        ],
        "RMSE": [
            rmse_baseline,
            rmse_model1,
            rmse_model2,
            rmse_simple,
            rmse_rf,
        ],
    }
)

print("\nModel comparison")
print(results.to_string(index=False))


# -----------------------------------------------------------------------------
# 14. ACTUAL VS PREDICTED PLOT
# -----------------------------------------------------------------------------

final_compare = test_simple[[YEAR_COL, WEEK_COL, "cases"]].copy()
final_compare["NB_prediction"] = test_simple["simple_prediction"].values
final_compare["RF_prediction"] = rf_predictions

plt.figure(figsize=(15, 6))
plt.plot(final_compare["cases"].values, label="Actual cases", linewidth=2)
plt.plot(final_compare["NB_prediction"].values, label="Simplified Negative Binomial")
plt.plot(final_compare["RF_prediction"].values, label="Random Forest")
plt.axvline(x=52, linestyle="--", alpha=0.6)
plt.xlabel("Test week")
plt.ylabel("Malaria cases")
plt.title("Actual vs Predicted Weekly Malaria Cases, 2023-2024")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "actual_vs_pred_reproduced.png", dpi=200)
plt.close()


# -----------------------------------------------------------------------------
# 15. QUARTILE ERROR DIAGNOSTICS
# -----------------------------------------------------------------------------

final_compare["case_quartile"] = pd.qcut(
    final_compare["cases"],
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"],
)

final_compare["NB_error"] = final_compare["NB_prediction"] - final_compare["cases"]
final_compare["RF_error"] = final_compare["RF_prediction"] - final_compare["cases"]
final_compare["NB_absolute_error"] = final_compare["NB_error"].abs()
final_compare["RF_absolute_error"] = final_compare["RF_error"].abs()
final_compare["NB_squared_error"] = final_compare["NB_error"] ** 2
final_compare["RF_squared_error"] = final_compare["RF_error"] ** 2
final_compare["NB_underpredicted"] = final_compare["NB_prediction"] < final_compare["cases"]
final_compare["RF_underpredicted"] = final_compare["RF_prediction"] < final_compare["cases"]

quartile_results = (
    final_compare.groupby("case_quartile", observed=True)
    .agg(
        n_weeks=("cases", "size"),
        actual_min=("cases", "min"),
        actual_max=("cases", "max"),
        actual_mean=("cases", "mean"),
        NB_mean_prediction=("NB_prediction", "mean"),
        RF_mean_prediction=("RF_prediction", "mean"),
        NB_MAE=("NB_absolute_error", "mean"),
        RF_MAE=("RF_absolute_error", "mean"),
        NB_MSE=("NB_squared_error", "mean"),
        RF_MSE=("RF_squared_error", "mean"),
        NB_bias=("NB_error", "mean"),
        RF_bias=("RF_error", "mean"),
        NB_underprediction_rate=("NB_underpredicted", "mean"),
        RF_underprediction_rate=("RF_underpredicted", "mean"),
    )
)
quartile_results["NB_RMSE"] = np.sqrt(quartile_results["NB_MSE"])
quartile_results["RF_RMSE"] = np.sqrt(quartile_results["RF_MSE"])

print("\nQuartile diagnostics")
print(quartile_results)


# -----------------------------------------------------------------------------
# 16. EXPLORATORY HIGH-BURDEN CLASSIFIER
# -----------------------------------------------------------------------------

# The high-burden definition is fixed from 2018-2021 development data.
dev_train = model_data[model_data[YEAR_COL] <= 2021].copy()
high_burden_threshold = dev_train["cases"].quantile(0.75)

model_data["high_burden"] = (
    model_data["cases"] >= high_burden_threshold
).astype(int)

classifier_features = [
    "cases_lag_1",
    "cases_change",
    "cases_roll_4",
    "sin_week",
    "cos_week",
    "sin_week_2",
    "cos_week_2",
]

final_clf_train = (
    model_data[model_data[YEAR_COL] <= 2022]
    .dropna(subset=classifier_features + ["high_burden"])
    .copy()
)
final_clf_test = (
    model_data[model_data[YEAR_COL] >= 2023]
    .dropna(subset=classifier_features + ["high_burden"])
    .copy()
)

warning_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
)

warning_model.fit(
    final_clf_train[classifier_features],
    final_clf_train["high_burden"],
)

final_clf_test["warning_probability"] = warning_model.predict_proba(
    final_clf_test[classifier_features]
)[:, 1]

# 0.50 was selected provisionally during 2022 validation and kept fixed here.
final_clf_test["alert"] = (
    final_clf_test["warning_probability"] >= 0.50
).astype(int)

print("\nExploratory high-burden classifier")
print(confusion_matrix(final_clf_test["high_burden"], final_clf_test["alert"]))
print(
    classification_report(
        final_clf_test["high_burden"],
        final_clf_test["alert"],
        digits=3,
    )
)


# -----------------------------------------------------------------------------
# 17. NEGATIVE BINOMIAL PREDICTIVE UNCERTAINTY
# -----------------------------------------------------------------------------

alpha_simple = nb_model_simple.params["alpha"]
mu = test_simple["simple_prediction"].values

# statsmodels NB2 variance: Var(Y) = mu + alpha * mu^2.
# scipy nbinom uses size/prob parameters, so convert them here.
size = 1 / alpha_simple
prob = size / (size + mu)

test_simple["pred_lower_80"] = nbinom.ppf(0.10, size, prob)
test_simple["pred_upper_80"] = nbinom.ppf(0.90, size, prob)

# Because the threshold may be fractional (e.g. 206.25) but case counts are
# integers, P(Y >= 207) is equivalent to P(Y > 206).
integer_cutoff = int(np.floor(high_burden_threshold))
test_simple["prob_high_burden"] = nbinom.sf(
    integer_cutoff,
    size,
    prob,
)

uncertainty_results = test_simple[
    [
        YEAR_COL,
        WEEK_COL,
        "cases",
        "simple_prediction",
        "pred_lower_80",
        "pred_upper_80",
        "prob_high_burden",
    ]
].copy()

uncertainty_results = uncertainty_results.rename(
    columns={
        "cases": "actual_cases",
        "simple_prediction": "expected_cases",
    }
)

uncertainty_results["inside_80_interval"] = (
    (uncertainty_results["actual_cases"] >= uncertainty_results["pred_lower_80"])
    & (uncertainty_results["actual_cases"] <= uncertainty_results["pred_upper_80"])
)

coverage_80 = uncertainty_results["inside_80_interval"].mean()

uncertainty_results["actual_high_burden"] = (
    uncertainty_results["actual_cases"] >= np.ceil(high_burden_threshold)
).astype(int)

roc_auc = roc_auc_score(
    uncertainty_results["actual_high_burden"],
    uncertainty_results["prob_high_burden"],
)
pr_auc = average_precision_score(
    uncertainty_results["actual_high_burden"],
    uncertainty_results["prob_high_burden"],
)
brier = brier_score_loss(
    uncertainty_results["actual_high_burden"],
    uncertainty_results["prob_high_burden"],
)

print("\nUncertainty diagnostics")
print(f"High-burden threshold: {high_burden_threshold:.2f}")
print(f"80% predictive interval coverage: {coverage_80:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
print(f"PR AUC: {pr_auc:.3f}")
print(f"Brier score: {brier:.3f}")
print(
    uncertainty_results.groupby("actual_high_burden")["prob_high_burden"].mean()
)


# -----------------------------------------------------------------------------
# 18. FINAL UNCERTAINTY PLOT
# -----------------------------------------------------------------------------

x = np.arange(len(uncertainty_results))

plt.figure(figsize=(16, 7))
plt.fill_between(
    x,
    uncertainty_results["pred_lower_80"].to_numpy(),
    uncertainty_results["pred_upper_80"].to_numpy(),
    alpha=0.2,
    label="80% predictive interval",
)
plt.plot(
    x,
    uncertainty_results["actual_cases"].to_numpy(),
    linewidth=2,
    label="Actual cases",
)
plt.plot(
    x,
    uncertainty_results["expected_cases"].to_numpy(),
    linewidth=2,
    label="Expected cases",
)
plt.axhline(
    y=np.ceil(high_burden_threshold),
    linestyle="--",
    label=f"High-burden threshold ({int(np.ceil(high_burden_threshold))} cases)",
)
plt.axvline(x=51.5, linestyle=":", alpha=0.7)
plt.xlabel("Test week (2023-2024)")
plt.ylabel("Reported malaria cases")
plt.title("Weekly Malaria Forecasts with 80% Predictive Intervals")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "uncertainty_plot_reproduced.png", dpi=200)
plt.close()


# -----------------------------------------------------------------------------
# 19. FINAL SUMMARY
# -----------------------------------------------------------------------------

print("\nFinal interpretation")
print(
    "The simplified Negative Binomial provides the most interpretable retained "
    "count forecast, but abrupt high-burden weeks remain difficult to anticipate. "
    "The model should therefore be presented with predictive uncertainty rather "
    "than as an exact early-warning system."
)
