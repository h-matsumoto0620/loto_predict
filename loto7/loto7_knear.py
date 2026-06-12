import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

CSV_FILE = "loto7.csv"
NUM_COLS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字", "第7数字"]
BONUS_COLS = ["BONUS数字1", "BONUS数字2"]
WINDOW = 5
LOTO_MIN = 1
LOTO_MAX = 37

df = pd.read_csv(CSV_FILE, encoding="shift_jis")
all_numbers = df[NUM_COLS].values       # (n, 7)
bonus_numbers = df[BONUS_COLS].values   # (n, 2)

print(f"読み込み完了: {len(df)} 回分のデータ")

def make_features(all_numbers, bonus_numbers, window=WINDOW):
    n = len(all_numbers)
    rows = n - window - 1
    idx = np.arange(rows)[:, None] + np.arange(window)[None, :]
    past = all_numbers[idx]                    # (rows, window, 7)
    past_bonus = bonus_numbers[idx]            # (rows, window, 2)
    flat = past.reshape(rows, -1)              # (rows, window*7)
    flat_bonus = past_bonus.reshape(rows, -1)  # (rows, window*2)
    X = np.concatenate([
        flat.mean(axis=1).reshape(-1, 1),
        flat.std(axis=1).reshape(-1, 1),
        flat.max(axis=1).reshape(-1, 1),
        flat.min(axis=1).reshape(-1, 1),
        np.median(flat, axis=1).reshape(-1, 1),
        np.diff(past, axis=1).mean(axis=(1, 2)).reshape(-1, 1),
        flat_bonus.mean(axis=1).reshape(-1, 1),
        flat_bonus.std(axis=1).reshape(-1, 1),
        flat_bonus.max(axis=1).reshape(-1, 1),
        flat_bonus.min(axis=1).reshape(-1, 1),
        past.mean(axis=1),   # (rows, 7)
        past.std(axis=1),    # (rows, 7)
    ], axis=1)
    y = all_numbers[window:n - 1]
    return X, y

X, y = make_features(all_numbers, bonus_numbers)
print(f"特徴量形状: X={X.shape}, y={y.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_neighbors": [3, 5, 7, 10, 15],
    "weights": ["uniform", "distance"],
}

raw_predictions = []

print("\n--- 各数字列の学習 ---")
for col_idx in range(7):
    grid = GridSearchCV(
        KNeighborsRegressor(),
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_scaled, y[:, col_idx])
    best = grid.best_estimator_

    past_w = all_numbers[-WINDOW:]
    past_bonus_w = bonus_numbers[-WINDOW:]
    flat_w = past_w.flatten()
    flat_bonus_w = past_bonus_w.flatten()
    next_feat = np.concatenate([
        [flat_w.mean(), flat_w.std(), flat_w.max(), flat_w.min(), np.median(flat_w)],
        [np.diff(past_w, axis=0).mean()],
        [flat_bonus_w.mean(), flat_bonus_w.std(), flat_bonus_w.max(), flat_bonus_w.min()],
        past_w.mean(axis=0),
        past_w.std(axis=0),
    ]).reshape(1, -1)
    next_feat_scaled = scaler.transform(next_feat)

    pred = int(round(best.predict(next_feat_scaled)[0]))
    pred = max(LOTO_MIN, min(pred, LOTO_MAX))
    raw_predictions.append(pred)
    print(f"  第{col_idx+1}数字: 予測={pred:2d}  (n_neighbors={best.n_neighbors}, weights={best.weights})")

def remove_duplicates(predictions, all_numbers, loto_min=LOTO_MIN, loto_max=LOTO_MAX):
    flat_all = all_numbers.flatten()
    freq = {n: 0 for n in range(loto_min, loto_max + 1)}
    for n in flat_all:
        freq[n] = freq.get(n, 0) + 1
    result = list(dict.fromkeys(predictions))
    if len(result) < 7:
        used = set(result)
        unused = sorted(
            [n for n in range(loto_min, loto_max + 1) if n not in used],
            key=lambda n: freq[n]
        )
        result += unused[:7 - len(result)]
    return sorted(result[:7])

final_predictions = remove_duplicates(raw_predictions, all_numbers)

print("\n" + "=" * 40)
print(f"生予測（重複除去前）: {raw_predictions}")
print(f"最終予測番号（7つ）: {final_predictions}")
dup_count = len(raw_predictions) - len(set(raw_predictions))
print(f"重複数: {dup_count}（補完: {'あり' if dup_count > 0 else 'なし'}）")
print("=" * 40)
