"""
Data Preprocessing & Cleaning — Dataset-SolarTechLab.csv
=========================================================
Steps:
  1. Load with correct separator
  2. Parse & sort datetime
  3. Select relevant features
  4. Drop nighttime rows  (G_tilt == 0  →  no solar generation)
  5. Drop NaN rows        (PV_Power NaN at night, occasional sensor gaps)
  6. Remove physical outliers (negative power, irradiance spikes)
  7. Remove statistical outliers via IQR on PV_Power
  8. Reset index & report summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8")


# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    print(f"Raw shape          : {df.shape}")
    print(f"Columns            : {list(df.columns)}")
    print(f"NaN counts:\n{df.isnull().sum()}\n")
    return df


# ─────────────────────────────────────────────────────────────
# 2. PARSE DATETIME & SORT
# ─────────────────────────────────────────────────────────────

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Use mixed format inference to tolerate truncated/corrupted entries
    # errors='coerce' turns unparseable values into NaT instead of crashing
    df["Time"] = pd.to_datetime(df["Time"], format="mixed",
                                dayfirst=True, errors="coerce")

    # Drop rows where datetime could not be parsed at all
    bad = df["Time"].isna().sum()
    if bad > 0:
        print(f"  Dropped {bad} rows with unparseable timestamps "
              f"(e.g. truncated entries like '28-Mar-2')")
        df = df.dropna(subset=["Time"])

    df = df.sort_values("Time").reset_index(drop=True)
    print(f"Parsed {len(df)}/{before} timestamps successfully")
    print(f"Time range         : {df['Time'].min()}  →  {df['Time'].max()}")
    return df


# ─────────────────────────────────────────────────────────────
# 3. SELECT FEATURES
#    Keep only what the PINN needs: Time, PV_Power, T_air, G_tilt
#    G_h  : horizontal irradiance   — redundant given G_tilt
#    W_s  : wind speed              — not used in physics model
#    W_d  : wind direction          — not used in physics model
# ─────────────────────────────────────────────────────────────

FEATURES = ["Time", "PV_Power", "T_air", "G_tilt"]

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[FEATURES].copy()
    print(f"After feature selection : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. DROP NIGHTTIME ROWS  (G_tilt == 0)
#    PV panels produce nothing at night → these rows are useless
#    for training a power prediction model.
#    NaN in PV_Power at night is expected, not a data error.
# ─────────────────────────────────────────────────────────────

def drop_nighttime(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["G_tilt"] > 0].copy()
    print(f"Dropped nighttime rows  : {before - len(df):>6d}  →  {len(df)} remain")
    return df


# ─────────────────────────────────────────────────────────────
# 5. DROP REMAINING NaNs
#    After nighttime removal, any leftover NaN is a sensor gap
# ─────────────────────────────────────────────────────────────

def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna().copy()
    print(f"Dropped NaN rows        : {before - len(df):>6d}  →  {len(df)} remain")
    return df


# ─────────────────────────────────────────────────────────────
# 6. PHYSICAL BOUNDS CHECK
#    Remove rows that violate physical constraints:
#      - PV_Power < 0       : impossible
#      - G_tilt   < 0       : impossible
#      - G_tilt   > 1500    : above realistic solar irradiance (W/m²)
#      - T_air    < -30     : unrealistic air temperature
#      - T_air    > 70      : unrealistic air temperature
# ─────────────────────────────────────────────────────────────

def physical_bounds_filter(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = (
        (df["PV_Power"] >= 0)     &
        (df["G_tilt"]   >= 0)     &
        (df["G_tilt"]   <= 1500)  &
        (df["T_air"]    >= -30)   &
        (df["T_air"]    <= 70)
    )
    df = df[mask].copy()
    print(f"Dropped physical outliers: {before - len(df):>5d}  →  {len(df)} remain")
    return df


# ─────────────────────────────────────────────────────────────
# 7. IQR OUTLIER REMOVAL on PV_Power
#    Catches sensor spikes / logging errors that pass physical bounds
# ─────────────────────────────────────────────────────────────

def iqr_filter(df: pd.DataFrame, col: str = "PV_Power",
               factor: float = 3.0) -> pd.DataFrame:
    before = len(df)
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)].copy()
    print(f"Dropped IQR outliers    : {before - len(df):>6d}  →  {len(df)} remain  "
          f"(bounds: [{lower:.2f}, {upper:.2f}])")
    return df


# ─────────────────────────────────────────────────────────────
# 8. FINAL RESET & SUMMARY
# ─────────────────────────────────────────────────────────────

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    print(f"\n{'='*45}")
    print(f"Final dataset shape     : {df.shape}")
    print(f"{'='*45}")
    print(df.describe().round(3))
    return df


# ─────────────────────────────────────────────────────────────
# 9. DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────

def plot_diagnostics(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Cleaned Dataset — Diagnostics", fontsize=13)

    # PV Power distribution
    axes[0, 0].hist(df["PV_Power"], bins=60, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("PV Power Distribution")
    axes[0, 0].set_xlabel("PV Power (W)")
    axes[0, 0].set_ylabel("Count")

    # G_tilt distribution
    axes[0, 1].hist(df["G_tilt"], bins=60, color="orange", edgecolor="white")
    axes[0, 1].set_title("Irradiance (G_tilt) Distribution")
    axes[0, 1].set_xlabel("G_tilt (W/m²)")
    axes[0, 1].set_ylabel("Count")

    # PV Power vs G_tilt
    axes[1, 0].scatter(df["G_tilt"], df["PV_Power"], s=3, alpha=0.3, color="steelblue")
    axes[1, 0].set_title("PV Power vs Irradiance")
    axes[1, 0].set_xlabel("G_tilt (W/m²)")
    axes[1, 0].set_ylabel("PV Power (W)")

    # PV Power vs T_air
    axes[1, 1].scatter(df["T_air"], df["PV_Power"], s=3, alpha=0.3, color="seagreen")
    axes[1, 1].set_title("PV Power vs Air Temperature")
    axes[1, 1].set_xlabel("T_air (°C)")
    axes[1, 1].set_ylabel("PV Power (W)")

    plt.tight_layout()
    plt.savefig("diagnostics.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# 10. NORMALISATION
#     Returns scaled arrays + fitted scalers (needed for inverse
#     transform at evaluation time — pass these into the PINN)
# ─────────────────────────────────────────────────────────────

def normalize(df: pd.DataFrame):
    """
    Builds a 6-feature input matrix:
      [G_tilt, T_air, hour_sin, hour_cos, doy_sin, doy_cos]

    Time features are cyclically encoded so that:
      - Hour 23 and hour 0 are treated as neighbours (not opposites)
      - Dec 31 and Jan 1 are treated as neighbours
    This is critical for the model to learn sunrise/sunset patterns.
    """
    # Cyclic time features
    hour = df["Time"].dt.hour + df["Time"].dt.minute / 60.0
    doy  = df["Time"].dt.dayofyear

    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * doy  / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * doy  / 365)

    # 6 input features
    X = df[["G_tilt", "T_air",
            "hour_sin", "hour_cos",
            "doy_sin",  "doy_cos"]].values
    y = df[["PV_Power"]].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    print(f"\nFeatures           : G_tilt, T_air, hour_sin, hour_cos, doy_sin, doy_cos")
    print(f"X shape            : {X_scaled.shape}")
    print(f"X mean             : {scaler_X.mean_.round(4)}")
    print(f"X std              : {scaler_X.scale_.round(4)}")
    print(f"y mean             : {scaler_y.mean_[0]:.4f}")
    print(f"y std              : {scaler_y.scale_[0]:.4f}")

    return X_scaled, y_scaled, scaler_X, scaler_y


# ─────────────────────────────────────────────────────────────
# PIPELINE — call this from your training script
# ─────────────────────────────────────────────────────────────

def preprocess(path: str, plot: bool = True):
    """
    Full preprocessing pipeline.
    Returns:
        df          : cleaned DataFrame (un-scaled, for inspection)
        X_scaled    : normalised inputs  [G_tilt, T_air,
                      hour_sin, hour_cos, doy_sin, doy_cos]  (N, 6)
        y_scaled    : normalised targets [PV_Power]           (N, 1)
        scaler_X    : fitted StandardScaler for X (6 features)
        scaler_y    : fitted StandardScaler for y
    """
    print("\n── Step 1: Load ─────────────────────────────")
    df = load_raw(path)

    print("\n── Step 2: Parse datetime ───────────────────")
    df = parse_datetime(df)

    print("\n── Step 3: Select features ──────────────────")
    df = select_features(df)

    print("\n── Step 4: Drop nighttime ───────────────────")
    df = drop_nighttime(df)

    print("\n── Step 5: Drop NaNs ────────────────────────")
    df = drop_nans(df)

    print("\n── Step 6: Physical bounds ──────────────────")
    df = physical_bounds_filter(df)

    print("\n── Step 7: IQR outlier removal ──────────────")
    df = iqr_filter(df)

    print("\n── Step 8: Finalize ─────────────────────────")
    df = finalize(df)

    if plot:
        plot_diagnostics(df)

    print("\n── Step 9: Normalise ────────────────────────")
    X_scaled, y_scaled, scaler_X, scaler_y = normalize(df)

    return df, X_scaled, y_scaled, scaler_X, scaler_y


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, X_scaled, y_scaled, scaler_X, scaler_y = preprocess(
        "Dataset-SolarTechLab.csv", plot=True
    )

    # Quick sanity check
    print(f"\nX_scaled shape : {X_scaled.shape}")
    print(f"y_scaled shape : {y_scaled.shape}")
    print(f"\nSample X (first 3 rows, normalised):\n{X_scaled[:3]}")
    print(f"Sample y (first 3 rows, normalised):\n{y_scaled[:3]}")
