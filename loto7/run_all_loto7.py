import subprocess
import sys
import re
import unicodedata

# ============================================================
# STEP0: CSV更新（update_loto7_csv.py を実行）
# ============================================================
print(f"{'='*55}")
print(f"  STEP0: ロト7 CSV更新")
print(f"{'='*55}")
try:
    result = subprocess.run(
        [sys.executable, "update_loto7_csv.py"],
        capture_output=True, text=True, check=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"  エラー: CSV更新中に問題が発生しました。\n{e.stderr}")
    print("  CSV更新をスキップして予測処理を続行します。")
except FileNotFoundError:
    print("  スキップ: update_loto7_csv.py が見つかりません。")
    print("  CSV更新をスキップして予測処理を続行します。")

# ============================================================
# 実行スクリプト一覧
# ============================================================
scripts = [
    ("K-近傍法 (KNN)",          "loto7_knear.py"),
    ("MLP（ニューラルネット）",   "loto7_mlp.py"),
    ("ランダムフォレスト",        "loto7_random.py"),
    ("RBF SVR",                  "loto7_rbfsvr.py"),
    ("線形 SVR",                 "loto7_svr.py"),
    ("決定木回帰",               "loto7_tree.py"),
]

# ============================================================
# 各スクリプトを順に実行し、予測番号を収集
# ============================================================
model_predictions: dict[str, tuple[int, ...]] = {}

for model_name, script in scripts:
    print(f"\n{'='*55}")
    print(f"  実行中: {model_name}  ({script})")
    print(f"{'='*55}")
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        match = re.search(r"最終予測番号（7つ）:\s*(\[[\d,\s]+\])", result.stdout)
        if match:
            numbers = tuple(sorted(map(int, re.findall(r"\d+", match.group(1)))))
            model_predictions[model_name] = numbers
    except subprocess.CalledProcessError as e:
        print(f"  エラー: 実行中に問題が発生しました。\n{e.stderr}")
    except FileNotFoundError:
        print(f"  スキップ: {script} が見つかりません。")

# ============================================================
# 組み合わせ重複チェック（一覧表示前に集計）
# ============================================================
combo_to_models: dict[tuple[int, ...], list[str]] = {}
for model_name, numbers in model_predictions.items():
    combo_to_models.setdefault(numbers, []).append(model_name)

# ============================================================
# 予測番号一覧（重複モデルは1行にまとめる）
# ============================================================
print(f"\n{'='*55}")
print("  【各モデルの予測番号一覧】")
print(f"{'='*55}")

rows: list[tuple[str, tuple[int, ...]]] = []
printed_models: set[str] = set()
for model_name, numbers in model_predictions.items():
    if model_name in printed_models:
        continue
    group_models = combo_to_models[numbers]
    label = "、".join(group_models)
    rows.append((label, numbers))
    printed_models.update(group_models)

def terminal_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

def ljust_terminal(s: str, width: int) -> str:
    return s + " " * max(0, width - terminal_width(s))

max_label_width = max(terminal_width(label) for label, _ in rows) if rows else 25
for label, numbers in rows:
    nums_str = "  ".join(f"{n:2d}" for n in numbers)
    print(f"  {ljust_terminal(label, max_label_width)} ： {nums_str}")
