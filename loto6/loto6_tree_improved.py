import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

# ============================================================
# 設定
# ============================================================
CSV_FILE = "loto6.csv"
NUM_COLS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字"]
BONUS_COL = "BONUS数字"
WINDOW = 5        # 過去何回分を特徴量にするか
LOTO_MIN = 1
LOTO_MAX = 43

# ============================================================
# データ読み込み（列名指定で安全に読む）
# ============================================================
df = pd.read_csv(CSV_FILE, encoding="shift_jis")
all_numbers = df[NUM_COLS].values          # shape: (回数, 6)
bonus_numbers = df[BONUS_COL].values       # shape: (回数,)

print(f"読み込み完了: {len(df)} 回分のデータ")

# ============================================================
# 特徴量生成（セット全体を1つのベクトルとして扱う）
#
# 改修ポイント①: 列独立 → セット全体で特徴量を構築
#   過去WINDOW回分の6数字すべてを特徴量に含めることで、
#   列間の相関・全体的なパターンを捉えられるようにする
#
# 改修ポイント②: インデックスiを特徴量から除外
#   「何回目のデータか」は未来予測に無意味なため削除
# ============================================================
def make_features(all_numbers, bonus_numbers, window=WINDOW):
    """
    各時点iの特徴量を生成する。
    特徴量: 過去window回の6数字の統計量 + ボーナス数字の統計量
    """
    X, y = [], []
    n = len(all_numbers)

    for i in range(window, n - 1):
        past = all_numbers[i - window:i]       # shape: (window, 6)
        past_bonus = bonus_numbers[i - window:i]  # shape: (window,)

        flat = past.flatten()  # window×6個の数字

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
        # 過去window回の各列の平均（列ごとの傾向）
        feat += list(np.mean(past, axis=0))   # 6個追加
        # 過去window回の各列の標準偏差
        feat += list(np.std(past, axis=0))    # 6個追加

        X.append(feat)
        y.append(all_numbers[i])  # 次回の6数字すべて（shape: (6,)）

    return np.array(X), np.array(y)

X, y = make_features(all_numbers, bonus_numbers)
print(f"特徴量形状: X={X.shape}, y={y.shape}")

# ============================================================
# スケーリング
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 各数字列ごとにDecisionTreeRegressorで学習・予測
#
# 改修ポイント③: 共通の特徴量(X)を使って各列を予測
#   → 全体の文脈を踏まえた上で各列を予測
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
}

raw_predictions = []

print("\n--- 各数字列の学習 ---")
for col_idx in range(6):
    col_labels = y[:, col_idx]

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
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

    print(f"  第{col_idx+1}数字: 予測={pred:2d}  (best_depth={best_model.max_depth})")

# ============================================================
# 改修ポイント④: 重複排除の保証
#   各列独立予測のため同じ番号が出る可能性がある
#   重複している場合は出現頻度の低い番号で補完する
# ============================================================
def remove_duplicates(predictions, all_numbers, loto_min=LOTO_MIN, loto_max=LOTO_MAX):
    """
    重複を除去し、出現頻度の低い番号で補完する。
    """
    # 全番号の出現頻度を計算
    flat_all = all_numbers.flatten()
    freq = {n: 0 for n in range(loto_min, loto_max + 1)}
    for n in flat_all:
        freq[n] = freq.get(n, 0) + 1

    result = list(dict.fromkeys(predictions))  # 順序を保ちながら重複除去

    if len(result) < 6:
        used = set(result)
        # 未使用番号を出現頻度の低い順に並べて補完
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
