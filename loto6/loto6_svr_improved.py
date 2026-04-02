import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

# ============================================================
# 設定
# ============================================================
CSV_FILE = "loto6.csv"
NUM_COLS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字"]
BONUS_COL = "BONUS数字"
WINDOW = 5
LOTO_MIN = 1
LOTO_MAX = 43  # 修正①: 42 → 43（ロト6の正しい上限）

# ============================================================
# データ読み込み
# 修正②: encoding="shift_jis" を明示
# 修正③: next(reader)を2回呼んでデータを1行捨てていたのを修正
#         → pd.read_csv で列名指定して安全に読む
# ============================================================
df = pd.read_csv(CSV_FILE, encoding="shift_jis")
all_numbers = df[NUM_COLS].values       # shape: (回数, 6)
bonus_numbers = df[BONUS_COL].values    # shape: (回数,)

print(f"読み込み完了: {len(df)} 回分のデータ")

# ============================================================
# 特徴量生成
# 修正④: インデックスiを特徴量から除外（未来予測に無意味）
# 修正⑤: bonus_std / bonus_max / bonus_min を追加（Tree版との統一）
# 修正⑥: セット全体を1つのベクトルとして扱い列間の文脈を考慮
# ============================================================
def make_features(all_numbers, bonus_numbers, window=WINDOW):
    X, y = [], []
    n = len(all_numbers)

    for i in range(window, n - 1):
        past = all_numbers[i - window:i]        # shape: (window, 6)
        past_bonus = bonus_numbers[i - window:i]  # shape: (window,)

        flat = past.flatten()

        feat = [
            # 過去window回の全数字の統計
            np.mean(flat),
            np.std(flat),
            np.max(flat),
            np.min(flat),
            np.median(flat),
            # 直前回との差分の平均（トレンド）
            np.mean(np.diff(past, axis=0)),
            # ボーナス数字の統計
            np.mean(past_bonus),
            np.std(past_bonus),
            np.max(past_bonus),
            np.min(past_bonus),
        ]
        # 過去window回の各列の平均・標準偏差
        feat += list(np.mean(past, axis=0))   # 6個
        feat += list(np.std(past, axis=0))    # 6個

        X.append(feat)
        y.append(all_numbers[i])  # 次回の6数字

    return np.array(X), np.array(y)

X, y = make_features(all_numbers, bonus_numbers)
print(f"特徴量形状: X={X.shape}, y={y.shape}")

# ============================================================
# スケーリング
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 各数字列ごとにSVR（線形カーネル）で学習・予測
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "C": [0.1, 1, 10],
    "epsilon": [0.1, 0.2, 0.5],
}

raw_predictions = []

print("\n--- 各数字列の学習 ---")
for col_idx in range(6):
    col_labels = y[:, col_idx]

    grid = GridSearchCV(
        SVR(kernel="linear"),
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_scaled, col_labels)
    best_model = grid.best_estimator_

    # 予測用特徴量（最新WINDOW回分）
    past = all_numbers[-WINDOW:]
    past_bonus = bonus_numbers[-WINDOW:]
    flat = past.flatten()

    next_feat = [
        np.mean(flat),
        np.std(flat),
        np.max(flat),
        np.min(flat),
        np.median(flat),
        np.mean(np.diff(past, axis=0)),
        np.mean(past_bonus),
        np.std(past_bonus),
        np.max(past_bonus),
        np.min(past_bonus),
    ]
    next_feat += list(np.mean(past, axis=0))
    next_feat += list(np.std(past, axis=0))

    next_feat = np.array(next_feat).reshape(1, -1)
    next_feat_scaled = scaler.transform(next_feat)

    pred = best_model.predict(next_feat_scaled)[0]
    pred = int(round(pred))
    pred = max(LOTO_MIN, min(pred, LOTO_MAX))
    raw_predictions.append(pred)

    print(f"  第{col_idx+1}数字: 予測={pred:2d}  (C={best_model.C}, epsilon={best_model.epsilon})")

# ============================================================
# 修正⑦: 重複排除の保証
#   重複した場合は出現頻度の低い番号で補完
# ============================================================
def remove_duplicates(predictions, all_numbers, loto_min=LOTO_MIN, loto_max=LOTO_MAX):
    flat_all = all_numbers.flatten()
    freq = {n: 0 for n in range(loto_min, loto_max + 1)}
    for n in flat_all:
        freq[n] = freq.get(n, 0) + 1

    result = list(dict.fromkeys(predictions))  # 順序保持で重複除去

    if len(result) < 6:
        used = set(result)
        unused = sorted(
            [n for n in range(loto_min, loto_max + 1) if n not in used],
            key=lambda n: freq[n]
        )
        result += unused[:6 - len(result)]

    return sorted(result[:6])

final_predictions = remove_duplicates(raw_predictions, all_numbers)

# ============================================================
# 結果出力
# ============================================================
print("\n" + "=" * 40)
print(f"生予測（重複除去前）: {raw_predictions}")
print(f"最終予測番号（6つ）: {final_predictions}")
dup_count = len(raw_predictions) - len(set(raw_predictions))
print(f"重複数: {dup_count}（補完: {'あり' if dup_count > 0 else 'なし'}）")
print("=" * 40)
