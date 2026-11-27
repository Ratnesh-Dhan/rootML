"""
train_model.py  -  walk-forward ML pipeline for gold candles (M1/M5/M15).

Usage:
    pip install pandas numpy scikit-learn joblib
    python train_model.py your_data.csv

What it does:
 1. Loads an MT5-style CSV (tab or comma separated, any common column names).
 2. Builds features (returns, ATR, RSI, EMA distance, candle shape, time of day).
 3. Labels each bar: did a BUY hit +TP first, a SELL hit +TP first, or neither
    (SL and TP are ATR multiples, within a time limit).
 4. Walk-forward: train on the past, test on the NEXT unseen chunk, roll forward.
 5. Reports accuracy AND profit in R-multiples after spread cost, for several
    confidence thresholds. Only unseen-data numbers count.
 6. Trains a final model on all data and saves model.joblib.
"""
import sys
from tqdm.auto import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# ----------------------------- settings ------------------------------
TP_ATR      = 1.0    # take profit = ATR x
SL_ATR      = 1.0    # stop loss   = ATR x
HORIZON     = 30     # bars allowed to hit TP/SL
ATR_PERIOD  = 14
SPREAD_COST = 0.30    # extra cost per trade in PRICE units (e.g. 0.30 for $0.30 on gold)
TRAIN_DAYS  = 120    # minimum training window (days)
TEST_DAYS   = 30     # each test chunk (days)
THRESHOLDS  = [0.40, 0.45, 0.50, 0.55, 0.60]   # min probability to take a trade
# ----------------------------------------------------------------------


def load_csv(path):
    with open(path, "r", errors="ignore") as f:
        head = f.readline()
    counts = {";": head.count(";"), "\t": head.count("\t"), ",": head.count(",")}
    sep = max(counts, key=counts.get)
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    elif "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"])
    elif "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("No time/date column found. Columns: %s" % list(df.columns))

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError("Missing column: " + c)
    if "spread" not in df.columns:
        df["spread"] = 0.0
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "spread"]]


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def add_features(df):
    c = df["close"]
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    for n in (1, 3, 5, 15, 30):
        df[f"ret{n}"] = c.pct_change(n)
    df["rsi7"] = rsi(c, 7)
    df["rsi14"] = rsi(c, 14)
    for n in (9, 21, 50):
        df[f"ema{n}_dist"] = (c - c.ewm(span=n, adjust=False).mean()) / df["atr"]
    df["body"] = (df["close"] - df["open"]) / df["atr"]
    df["range"] = (df["high"] - df["low"]) / df["atr"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["atr"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["atr"]
    df["atr_rel"] = df["atr"] / c
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(100).mean()
    h = df["time"].dt.hour + df["time"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow"] = df["time"].dt.dayofweek
    return df


FEATURES = ["ret1", "ret3", "ret5", "ret15", "ret30", "rsi7", "rsi14",
            "ema9_dist", "ema21_dist", "ema50_dist", "body", "range",
            "upper_wick", "lower_wick", "atr_rel", "atr_ratio",
            "hour_sin", "hour_cos", "dow"]


# def make_labels(df):
#     """0 = no clear win, 1 = BUY wins (TP before SL), 2 = SELL wins."""
#     o = df["close"].values
#     h, l, atr = df["high"].values, df["low"].values, df["atr"].values
#     n = len(df)
#     y = np.zeros(n, dtype=int)
#     for i in range(n - HORIZON - 1):
#         a = atr[i]
#         if not np.isfinite(a) or a <= 0:
#             continue
#         entry = o[i]
#         buy_tp, buy_sl = entry + TP_ATR * a, entry - SL_ATR * a
#         sell_tp, sell_sl = entry - TP_ATR * a, entry + SL_ATR * a
#         buy_res = sell_res = 0
#         for j in range(i + 1, i + 1 + HORIZON):
#             if buy_res == 0:
#                 if l[j] <= buy_sl:  buy_res = -1      # SL first (conservative)
#                 elif h[j] >= buy_tp: buy_res = 1
#             if sell_res == 0:
#                 if h[j] >= sell_sl: sell_res = -1
#                 elif l[j] <= sell_tp: sell_res = 1
#             if buy_res != 0 and sell_res != 0:
#                 break
#         if buy_res == 1 and sell_res != 1:
#             y[i] = 1
#         elif sell_res == 1 and buy_res != 1:
#             y[i] = 2
#     df["label"] = y
#     return df

def make_labels(df):
    """0 = no clear win, 1 = BUY wins, 2 = SELL wins."""
    o = df["close"].values
    h, l, atr = df["high"].values, df["low"].values, df["atr"].values

    n = len(df)
    y = np.zeros(n, dtype=int)

    for i in tqdm(
        range(n - HORIZON - 1),
        desc="Generating labels",
        unit="candle",
        dynamic_ncols=True,
    ):
        a = atr[i]

        if not np.isfinite(a) or a <= 0:
            continue

        entry = o[i]

        buy_tp = entry + TP_ATR * a
        buy_sl = entry - SL_ATR * a

        sell_tp = entry - TP_ATR * a
        sell_sl = entry + SL_ATR * a

        buy_res = sell_res = 0

        for j in range(i + 1, i + 1 + HORIZON):
            if buy_res == 0:
                if l[j] <= buy_sl:
                    buy_res = -1
                elif h[j] >= buy_tp:
                    buy_res = 1

            if sell_res == 0:
                if h[j] >= sell_sl:
                    sell_res = -1
                elif l[j] <= sell_tp:
                    sell_res = 1

            if buy_res != 0 and sell_res != 0:
                break

        if buy_res == 1 and sell_res != 1:
            y[i] = 1
        elif sell_res == 1 and buy_res != 1:
            y[i] = 2

    df["label"] = y
    return df

def r_result(label_row_pred, label_true, atr, cost):
    """Profit in R for a taken trade. pred: 1 buy / 2 sell. Win = +TP/SL, else -1."""
    risk = SL_ATR * atr
    gain = TP_ATR * atr / risk
    cost_r = cost / risk if risk > 0 else 0
    return (gain if label_true == label_row_pred else -1.0) - cost_r


# def walk_forward(df):
#     df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
#     df = df.iloc[:-HORIZON]                       # last bars have unfinished labels
#     start = df["time"].iloc[0] + pd.Timedelta(days=TRAIN_DAYS)
#     end = df["time"].iloc[-1]

#     rows, acc_list = [], []
#     t0 = start
#     fold = 0
#     while t0 < end:
#         t1 = t0 + pd.Timedelta(days=TEST_DAYS)
#         # purge: drop the last HORIZON bars of training so labels don't peek into test
#         train = df[df["time"] < t0].iloc[:-HORIZON]
#         test = df[(df["time"] >= t0) & (df["time"] < t1)]
#         if len(train) < 2000 or len(test) < 200:
#             t0 = t1
#             continue

#         model = HistGradientBoostingClassifier(
#             max_depth=4, learning_rate=0.05, max_iter=200,
#             l2_regularization=1.0, random_state=42)
#         model.fit(train[FEATURES], train["label"])
#         proba = model.predict_proba(test[FEATURES])
#         classes = list(model.classes_)

#         pred = np.array(classes)[proba.argmax(axis=1)]
#         acc_list.append((pred == test["label"].values).mean())

#         for cls in (1, 2):
#             if cls in classes:
#                 rows.append(pd.DataFrame({
#                     "fold": fold, "time": test["time"].values, "cls": cls,
#                     "p": proba[:, classes.index(cls)],
#                     "true": test["label"].values, "atr": test["atr"].values}))
#         fold += 1
#         t0 = t1

#     res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
#     return res, acc_list

def walk_forward(df):
    df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    df = df.iloc[:-HORIZON]

    if df.empty:
        raise ValueError("No usable rows remain after feature/label cleanup.")

    start = df["time"].iloc[0] + pd.Timedelta(days=TRAIN_DAYS)
    end = df["time"].iloc[-1]

    if start >= end:
        return pd.DataFrame(), []

    total_folds = int(np.ceil((end - start) / pd.Timedelta(days=TEST_DAYS)))

    rows, acc_list = [], []
    fold = 0

    with tqdm(
        total=total_folds,
        desc="Walk-forward",
        unit="fold",
        dynamic_ncols=True,
    ) as pbar:
        t0 = start

        while t0 < end:
            t1 = t0 + pd.Timedelta(days=TEST_DAYS)

            train = df[df["time"] < t0].iloc[:-HORIZON]
            test = df[(df["time"] >= t0) & (df["time"] < t1)]

            if len(train) < 2000 or len(test) < 200:
                pbar.set_postfix_str("Skipping insufficient data")
                t0 = t1
                pbar.update(1)
                continue

            pbar.set_postfix(
                fold=fold + 1,
                train_rows=len(train),
                test_rows=len(test),
                stage="fitting",
            )

            model = HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.05,
                max_iter=200,
                l2_regularization=1.0,
                random_state=42,
                verbose=1,
            )

            model.fit(train[FEATURES], train["label"])

            proba = model.predict_proba(test[FEATURES])
            classes = list(model.classes_)

            pred = np.array(classes)[proba.argmax(axis=1)]
            acc_list.append(
                (pred == test["label"].values).mean()
            )

            for cls in (1, 2):
                if cls in classes:
                    rows.append(pd.DataFrame({
                        "fold": fold,
                        "time": test["time"].values,
                        "cls": cls,
                        "p": proba[:, classes.index(cls)],
                        "true": test["label"].values,
                        "atr": test["atr"].values,
                    }))

            fold += 1
            t0 = t1
            pbar.update(1)

    res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return res, acc_list

def report(res, acc_list):
    print("\n=== WALK-FORWARD RESULTS (unseen data only) ===")
    print("Folds: %d | mean 3-class accuracy: %.3f" % (len(acc_list), np.mean(acc_list)))
    print("(random baseline is roughly the share of 'no trade' bars, so do not trust accuracy alone)\n")
    print("%-10s %-8s %-9s %-10s %-10s" % ("threshold", "trades", "win%", "avg R", "total R"))
    best = None
    for th in THRESHOLDS:
        sel = res[res["p"] >= th]
        if len(sel) == 0:
            continue
        risk = SL_ATR * sel["atr"].values
        gain = TP_ATR / SL_ATR
        win = (sel["cls"].values == sel["true"].values)
        r = np.where(win, gain, -1.0) - SPREAD_COST / np.where(risk > 0, risk, np.nan)
        r = np.nan_to_num(r)
        wins = (sel["cls"].values == sel["true"].values).mean() * 100
        print("%-10.2f %-8d %-9.1f %-10.3f %-10.1f" % (th, len(sel), wins, r.mean(), r.sum()))
        if best is None or r.mean() > best[1]:
            best = (th, r.mean())
    if best:
        print("\nBest threshold by avg R: %.2f (avg R %.3f)" % best)
        if best[1] <= 0:
            print("-> Average R is not positive: NO EDGE found with these settings.")
    return best


# def main(path):
#     df = load_csv(path)
#     print("Loaded %d candles: %s -> %s" % (len(df), df["time"].iloc[0], df["time"].iloc[-1]))
#     df = add_features(df)
#     print("Labelling...")
#     df = make_labels(df)
#     print("Label counts:", df["label"].value_counts().to_dict())

#     res, acc = walk_forward(df)
#     if res.empty:
#         print("Not enough data for the walk-forward windows. Lower TRAIN_DAYS/TEST_DAYS.")
#         return
#     best = report(res, acc)

#     # final model on everything
#     clean = df.dropna(subset=FEATURES + ["label"]).iloc[:-HORIZON]
#     final = HistGradientBoostingClassifier(
#         max_depth=4, learning_rate=0.05, max_iter=200,
#         l2_regularization=1.0, random_state=42)
#     final.fit(clean[FEATURES], clean["label"])
#     joblib.dump({"model": final, "features": FEATURES,
#                  "tp_atr": TP_ATR, "sl_atr": SL_ATR, "horizon": HORIZON,
#                  "threshold": best[0] if best else 0.55}, "model.joblib")
#     res.to_csv("walk_forward_predictions.csv", index=False)
#     print("\nSaved model.joblib and walk_forward_predictions.csv")

def main(path):
    print("\n[1/6] Loading CSV...")
    df = load_csv(path)

    print(
        f"Loaded {len(df):,} candles: "
        f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}"
    )

    print("\n[2/6] Calculating features...")
    df = add_features(df)

    print("\n[3/6] Generating labels...")
    df = make_labels(df)

    print("Label counts:", df["label"].value_counts().to_dict())

    print("\n[4/6] Walk-forward training and evaluation...")
    res, acc = walk_forward(df)

    if res.empty:
        print("Not enough data for walk-forward windows.")
        print("Check your date range and TRAIN_DAYS / TEST_DAYS.")
        return

    best = report(res, acc)

    print("\n[5/6] Training final model...")
    clean = df.dropna(subset=FEATURES + ["label"]).iloc[:-HORIZON]

    final = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=1.0,
        random_state=42,
        verbose=1,
    )

    final.fit(clean[FEATURES], clean["label"])

    print("\n[6/6] Saving model and predictions...")

    joblib.dump({
        "model": final,
        "features": FEATURES,
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "horizon": HORIZON,
        "threshold": best[0] if best else 0.55,
    }, "model.joblib")

    res.to_csv("walk_forward_predictions.csv", index=False)

    print("\nTraining pipeline completed.")
    print("Saved model.joblib and walk_forward_predictions.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py your_data.csv")
        sys.exit(1)
    main(sys.argv[1])
