import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

#Load and preprocess data
data = pd.read_csv("Tea Project Data.csv")
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date"])
data = data.set_index("date").sort_index()
data = data[~data.index.duplicated(keep="last")].copy()

for col in ["wtk_close_kes", "ndvi", "ndwi"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["wtk_close_kes"])
data = data[data["wtk_close_kes"] > 0].copy()

#Feature Engineering
data["log_ret"] = np.log(data["wtk_close_kes"] / data["wtk_close_kes"].shift(1))
data["current_dir"] = np.where(data["log_ret"].notna(), (data["log_ret"] > 0).astype(float),np.nan)
data["target_dir"] = data["current_dir"].shift(-1)
data["price_volatility_3m"] = data["log_ret"].rolling(3).std()
data["price_volatility_6m"] = data["log_ret"].rolling(6).std()
data["ndvi_rolling_mean_6m"] = data["ndvi"].rolling(6).mean()
data["ndvi_rolling_vol_6m"] = data["ndvi"].rolling(6).std()
data["ndwi_rolling_mean_6m"] = data["ndwi"].rolling(6).mean()
data["ndwi_rolling_vol_6m"] = data["ndwi"].rolling(6).std()
data["ndvi_lag_2"] = data["ndvi"].shift(2)
data["ndwi_lag_2"] = data["ndwi"].shift(2)
data["log_ret_lag_1"] = data["log_ret"].shift(1)
data["price_momentum_2m"] = data["log_ret"].rolling(2).sum()

ndvi_change = data["ndvi"].diff()
ndwi_change = data["ndwi"].diff()

ndvi_change_std = ndvi_change.rolling(2).std()
ndwi_change_std = ndwi_change.rolling(2).std()

data["ndvi_velocity_2m"] = (data["ndvi"].diff(2) / ndvi_change_std.replace(0, np.nan))
data["ndwi_velocity_2m"] = (data["ndwi"].diff(2) / ndwi_change_std.replace(0, np.nan))

feature_cols = [
    "log_ret_lag_1",
    "price_momentum_2m",
    "ndvi_velocity_2m",
    "ndwi_velocity_2m",
]

#Define feature matrix X and target y
X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
y = data["target_dir"]
valid = y.notna() #Account for missing values
X = X.loc[valid]
y = y.loc[valid].astype(int)

#Reduced min_child_weight to let shallow trees make splits on small time series data folds
#define our hyperparam space
param_grid = list(ParameterGrid({
    "max_depth": [1,2] ,
    "learning_rate": [0.02, 0.04],
    "n_estimators": [40, 80],
    "min_child_weight":[1, 2],
    "reg_lambda": [2.0, 5.0],
}))

#Time series split and generation of our OOS window
outer_cv = TimeSeriesSplit(n_splits=3)
oos_prob = pd.Series(np.nan, index=data.index, dtype=float)
oos_pred = pd.Series(np.nan, index=data.index, dtype=float)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    inner_cv = TimeSeriesSplit(n_splits=2)
    best_score = -np.inf
    best_params = None
    #hyperparam tuning 
    for params in param_grid:
        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            Xi_train = X_train.iloc[inner_train_idx]
            yi_train = y_train.iloc[inner_train_idx]
            Xi_val = X_train.iloc[inner_val_idx]
            yi_val = y_train.iloc[inner_val_idx]

            if yi_train.nunique() < 2 or yi_val.nunique() < 2:
                continue

            imputer = SimpleImputer(strategy="median")
            Xi_train_imp = imputer.fit_transform(Xi_train)
            Xi_val_imp = imputer.transform(Xi_val)

            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                random_state=42,
                n_jobs=1,
            )

            model.fit(Xi_train_imp, yi_train)
            val_prob = model.predict_proba(Xi_val_imp)[:, 1]
            inner_scores.append(roc_auc_score(yi_val, val_prob))

        if inner_scores:
            score = np.mean(inner_scores)
            if score > best_score:
                best_score = score
                best_params = params

    if best_params is None:
        best_params = {
            "max_depth": 1,
            "learning_rate": 0.02,
            "n_estimators": 40,
            "min_child_weight": 2,
            "reg_lambda": 5.0,
        }

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train_imp, y_train)

    prob = model.predict_proba(X_test_imp)[:, 1]
    
    # FIXED: Replaced hard 0.5 cutoff with dynamic percentile thresholding (Top 35% of probabilities)
    trade_threshold = np.percentile(prob, 65) if len(prob) > 0 else 0.5
    pred = (prob >= trade_threshold).astype(int)
    
    oos_prob.loc[X_test.index] = prob
    oos_pred.loc[X_test.index] = pred

    auc = (
        roc_auc_score(y_test, prob)
        if y_test.nunique() == 2
        else np.nan
    )

    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    accuracy = accuracy_score(y_test, pred)
    balanced_acc = (
        balanced_accuracy_score(y_test, pred)
        if y_test.nunique() == 2
        else np.nan
    )

    fold_results.append({
        "fold": fold,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Best inner AUC": best_score,
        "Best parameters": best_params,
    })

    print(f"\nFold {fold}")
    print("-" * 35)
    print(f"ROC-AUC: {auc:.4f}" if not np.isnan(auc)
          else "ROC-AUC: undefined")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}"
          if not np.isnan(balanced_acc)
          else "Balanced accuracy: undefined")
    print(f"Selected params: {best_params}")

#BACKTESTING ENGINE

# BACKTESTING ENGINE
CAPITAL = 100_000
ALLOCATION = 0.10 #10% to invest
FEE_RATE = 0.021  # Working assumption: 2.1% per traded side

results = pd.DataFrame(index=data.index)
results["signal"] = oos_pred
results["next_market_ret"] = data["wtk_close_kes"].shift(-1) / data["wtk_close_kes"] - 1

# Use only dates with an out-of-sample prediction and a next-month return
bt = results.dropna(subset=["signal", "next_market_ret"]).copy()
bt["strategy_ret"] = bt["signal"] * bt["next_market_ret"]
# Start both curves at the same capital
bt["gross_equity"] = CAPITAL * (1 + bt["strategy_ret"]).cumprod()
bt["buy_hold_equity"] = CAPITAL * (1 + bt["next_market_ret"]).cumprod()
# 10% bc WTK is illquid
bt["position"] = ALLOCATION * bt["gross_equity"] * bt["signal"]
# Trade amount; first observed position is an entry from cash
bt["dV"] = bt["position"].diff()
if len(bt):
    bt.iloc[0, bt.columns.get_loc("dV")] = bt["position"].iloc[0]

bt["fees"] = FEE_RATE * bt["dV"].abs()
bt["strategy_equity"] = np.nan
if len(bt):
    bt.iloc[0, bt.columns.get_loc("strategy_equity")] = (
        CAPITAL * (1 + bt["strategy_ret"].iloc[0]) - bt["fees"].iloc[0]
    )

    for i in range(1, len(bt)):
        previous_equity = bt["strategy_equity"].iloc[i - 1]
        bt.iloc[i, bt.columns.get_loc("strategy_equity")] = (
            previous_equity * (1 + bt["strategy_ret"].iloc[i])
            - bt["fees"].iloc[i]
        )

if len(bt):
    final_value = bt["strategy_equity"].iloc[-1]
    print("RESULTS!!!!!!:")
    print('='*50)
    print(f"Number of trades: {len(bt)}")
    print(f"Starting capital: KSh {CAPITAL:,.2f}")
    print(f"Total fees: KSh {bt['fees'].sum():,.2f}")
    print(f"Final equity after fees: KSh {final_value:,.2f}")
    print(f"Net profit: KSh {final_value - CAPITAL:,.2f}")
    print(f"Total return after fees: {(final_value / CAPITAL - 1) * 100:.2f}%")
    print('='*50)

# Plot both curves in KSh, over the same out-of-sample dates
plt.figure(figsize=(12, 6))
plt.plot(bt.index, bt["strategy_equity"],
         label="XGBoost Strategy (after fees)", color="forestgreen", linewidth=2)
plt.plot(bt.index, bt["buy_hold_equity"],
         label="Buy & Hold", color="black", alpha=0.6)
plt.title("WTK Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (KSh)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
