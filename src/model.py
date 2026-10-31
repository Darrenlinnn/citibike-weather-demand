"""Citi Bike daily demand vs. weather: regression + high-demand classifier.

All scores are out-of-sample (5-fold cross-validation), so they reflect how
well the models predict days they have not seen.

Usage:
    python src/model.py --data data/citibike_weather_daily_2025.csv --images images
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load(path):
    df = pd.read_csv(path, parse_dates=["DATE"])
    df["day_of_week"] = df["Day_of_Week"].map({d: i for i, d in enumerate(DAYS)})
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rain"] = (df["PRCP"] > 0).astype(int)
    return df.rename(columns={"Month": "month", "Total_Daily_Rides": "rides"})


def regression(df):
    """Compare weather-only vs. weather + calendar features with 5-fold CV."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sets = {"Weather only (TMAX, PRCP, AWND)": ["TMAX", "PRCP", "AWND"],
            "Weather + calendar": ["TMAX", "PRCP", "AWND", "is_weekend", "month", "day_of_week"]}
    rows = []
    for label, feats in sets.items():
        for name, model in [("Linear Regression", LinearRegression()),
                            ("Decision Tree (depth 5)", DecisionTreeRegressor(max_depth=5, random_state=42))]:
            r2 = cross_val_score(model, df[feats], df["rides"], cv=kf, scoring="r2")
            rows.append({"features": label, "model": name, "cv_r2_mean": r2.mean(), "cv_r2_std": r2.std()})
    coefs = LinearRegression().fit(df[["TMAX", "PRCP", "AWND"]], df["rides"]).coef_
    return pd.DataFrame(rows), dict(zip(["TMAX", "PRCP", "AWND"], coefs))


def classifier(df):
    """Predict above- vs. below-median demand days, scored out-of-sample."""
    df = df.copy()
    df["high_demand"] = (df["rides"] > df["rides"].median()).astype(int)
    X, y = df[["TMAX", "PRCP", "AWND", "is_weekend", "month"]], df["high_demand"]
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X, y, cv=skf)
    in_sample = accuracy_score(y, clf.fit(X, y).predict(X))
    return accuracy_score(y, pred), in_sample, classification_report(y, pred, digits=2)


def rain_effect(df):
    """Average ride drop on rainy vs. dry days, weekday vs. weekend."""
    out = {}
    for label, part in [("weekday", df[df.is_weekend == 0]), ("weekend", df[df.is_weekend == 1])]:
        dry, wet = part[part.is_rain == 0]["rides"].mean(), part[part.is_rain == 1]["rides"].mean()
        out[label] = (dry - wet, (dry - wet) / dry * 100)
    return out


def charts(df, img):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
    # 1) Ridership vs. max temperature, rainy days highlighted
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    for flag, color, label in [(0, "#2c7fb8", "Dry day"), (1, "#f03b20", "Rainy day")]:
        d = df[df.is_rain == flag]
        ax.scatter(d.TMAX, d.rides / 1000, s=18, alpha=0.7, color=color, label=label)
    ax.set_xlabel("Daily max temperature (°F)"); ax.set_ylabel("Daily rides (thousands)")
    ax.set_title("Citi Bike rides rise with temperature; rain pulls them down (2025)", loc="left")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{img}/rides_vs_temperature.png")
    # 2) Monthly average rides vs. temperature
    m = df.groupby("month").agg(rides=("rides", "mean"), tmax=("TMAX", "mean"))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.bar(m.index, m.rides / 1000, color="#2c7fb8", alpha=0.8, label="Avg daily rides")
    ax.set_ylabel("Avg daily rides (thousands)"); ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2 = ax.twinx(); ax2.plot(m.index, m.tmax, color="#f03b20", marker="o", label="Avg max temp (°F)")
    ax2.set_ylabel("Avg max temperature (°F)"); ax2.spines["top"].set_visible(False)
    ax.set_title("Ridership peaks in September, two months after temperature", loc="left")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88), frameon=False)
    fig.tight_layout(); fig.savefig(f"{img}/monthly_rides_temperature.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/citibike_weather_daily_2025.csv")
    p.add_argument("--images", default="images")
    a = p.parse_args()
    df = load(a.data)
    print(f"{len(df)} days, {df.rides.sum():,} rides\n")
    res, coefs = regression(df)
    print(res.round(3).to_string(index=False))
    print("\nWeather-only linear coefficients (rides per unit):", {k: round(v) for k, v in coefs.items()})
    acc, ins, rep = classifier(df)
    print(f"\nHigh-demand classifier: 5-fold CV accuracy {acc:.1%} (in-sample {ins:.1%})\n{rep}")
    for k, (n, pct) in rain_effect(df).items():
        print(f"Rain effect, {k}: -{n:,.0f} rides/day ({pct:.1f}%)")
    charts(df, a.images)


if __name__ == "__main__":
    main()
