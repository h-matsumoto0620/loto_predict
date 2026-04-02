import csv
import codecs
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

# ============================================================
# 設定
# ============================================================
CSV_FILE = "miniloto.csv"
NUM_COLS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字"]
BONUS_COL = "BONUS数字"
WINDOW = 5
LOTO_MIN = 1
LOTO_MAX = 31

# ============================================================
# データ読み込み
# ============================================================
with codecs.open(CSV_FILE, "r", encoding="shift_jis") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

all_numbers = np.array([[int(row[c]) for c in NUM_COLS] for row in rows])
bonus_numbers = np.array([int(row[BONUS_COL]) for row in rows])

print(f"読み込み完了: {len(all_numbers)} 回分のデータ")

# ============================================================
# 特徴量生成
# ============================================================
def make_features(all_numbers, bonus_numbers, window=WINDOW):
    X, y = [], []
    n = len(all_numbers)
    for i in range(window, n - 1):
        past = all_numbers[i - window:i]
        past_bonus = bonus_numbers[i - window:i]
        flat = past.flatten()
        feat = [
            np.mean(flat), np.std(flat), np.max(flat), np.min(flat), np.median(flat),
            np.mean(np.diff(past, axis=0)),
            np.mean(past_bonus), np.std(past_bonus), np.max(past_bonus), np.min(past_bonus),
        ]
        feat += list(np.mean(past, axis=0))
        feat += list(np.std(past, axis=0))
        X.append(feat)
        y.append(all_numbers[i])
    return np.array(X), np.array(y)

X, y = make_features(all_numbers, bonus_numbers)
print(f"特徴量形状: X={X.shape}, y={y.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# ランダムフォレストで学習・予測
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [3],
}

raw_predictions = []

print("\n--- 各数字列の学習 ---")
for col_idx in range(5):
    col_labels = y[:, col_idx]
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, cv=tscv, scoring="neg_mean_squared_error",
    )
    grid.fit(X_scaled, col_labels)
    best_model = grid.best_estimator_

    past = all_numbers[-WINDOW:]
    past_bonus = bonus_numbers[-WINDOW:]
    flat = past.flatten()
    next_feat = [
        np.mean(flat), np.std(flat), np.max(flat), np.min(flat), np.median(flat),
        np.mean(np.diff(past, axis=0)),
        np.mean(past_bonus), np.std(past_bonus), np.max(past_bonus), np.min(past_bonus),
    ]
    next_feat += list(np.mean(past, axis=0))
    next_feat += list(np.std(past, axis=0))

    next_feat_scaled = scaler.transform(np.array(next_feat).reshape(1, -1))
    pred = int(round(best_model.predict(next_feat_scaled)[0]))
    pred = max(LOTO_MIN, min(pred, LOTO_MAX))
    raw_predictions.append(pred)
    print(f"  第{col_idx+1}数字: 予測={pred:2d}  (n_estimators={best_model.n_estimators}, max_depth={best_model.max_depth})")

# ============================================================
# 重複排除
# ============================================================
def remove_duplicates(predictions, all_numbers, loto_min=LOTO_MIN, loto_max=LOTO_MAX):
    flat_all = all_numbers.flatten()
    freq = {n: 0 for n in range(loto_min, loto_max + 1)}
    for n in flat_all:
        freq[n] = freq.get(n, 0) + 1
    result = list(dict.fromkeys(predictions))
    if len(result) < 5:
        used = set(result)
        unused = sorted(
            [n for n in range(loto_min, loto_max + 1) if n not in used],
            key=lambda n: freq[n]
        )
        result += unused[:5 - len(result)]
    return sorted(result[:5])

final_predictions = remove_duplicates(raw_predictions, all_numbers)

print("\n" + "=" * 40)
print(f"生予測（重複除去前）: {raw_predictions}")
print(f"最終予測番号（5つ）: {final_predictions}")
dup_count = len(raw_predictions) - len(set(raw_predictions))
print(f"重複数: {dup_count}（補完: {'あり' if dup_count > 0 else 'なし'}）")
print("=" * 40)
