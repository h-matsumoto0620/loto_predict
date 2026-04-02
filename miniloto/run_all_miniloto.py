import subprocess
import re
import sys

# ============================================================
# STEP0: CSV更新（update_miniloto_csv.py を実行）
# ============================================================
print(f"{'='*55}")
print(f"  STEP0: ミニロトCSV更新")
print(f"{'='*55}")
try:
    result = subprocess.run(
        [sys.executable, "update_miniloto_csv.py"],
        capture_output=True, text=True, check=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"  エラー: CSV更新中に問題が発生しました。\n{e.stderr}")
    print("  CSV更新をスキップして予測処理を続行します。")
except FileNotFoundError:
    print("  スキップ: update_miniloto_csv.py が見つかりません。")
    print("  CSV更新をスキップして予測処理を続行します。")

# ============================================================
# 実行スクリプト一覧
# ============================================================
scripts = [
    ("K-近傍法 (KNN)",        "miniloto_knear.py"),
    ("MLP（ニューラルネット）", "miniloto_mlp.py"),
    ("ランダムフォレスト",      "miniloto_random.py"),
    ("RBF SVR",               "miniloto_rbfsvr.py"),
    ("線形 SVR",              "miniloto_svr.py"),
    ("決定木回帰",             "miniloto_tree.py"),
]

# ============================================================
# 各スクリプトを順に実行し、予測番号を収集
# ============================================================
model_predictions: dict[str, list[int]] = {}

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

        match = re.search(r"最終予測番号（5つ）:\s*(\[[\d,\s]+\])", result.stdout)
        if match:
            numbers = list(map(int, re.findall(r"\d+", match.group(1))))
            model_predictions[model_name] = numbers
        else:
            print(f"  ※ 予測番号を取得できませんでした（出力フォーマット不一致）")

    except subprocess.CalledProcessError as e:
        print(f"  エラー: 実行中に問題が発生しました。\n{e.stderr}")
    except FileNotFoundError:
        print(f"  スキップ: {script} が見つかりません。")

# ============================================================
# 全モデルの予測番号をまとめて表示
# ============================================================
print(f"\n{'='*55}")
print("  【全モデルの予測番号一覧】")
print(f"{'='*55}")

if not model_predictions:
    print("  実行できたモデルがありません。")
    sys.exit(1)

for model_name, numbers in model_predictions.items():
    nums_str = "  ".join(f"{n:2d}" for n in sorted(numbers))
    print(f"  {model_name:<25}: {nums_str}")

# ============================================================
# 重複チェックと除外表示
# ============================================================
print(f"\n{'='*55}")
print("  【重複チェックと最終集計】")
print(f"{'='*55}")

number_to_models: dict[int, list[str]] = {}
for model_name, numbers in model_predictions.items():
    for n in numbers:
        number_to_models.setdefault(n, []).append(model_name)

duplicates_found = False
for number, models in sorted(number_to_models.items()):
    if len(models) > 1:
        if not duplicates_found:
            print("\n  ▼ 複数モデルで重複していた番号:")
            duplicates_found = True
        models_str = "、".join(models)
        print(f"    番号 {number:2d}  →  {len(models)}モデルが予測  ({models_str})")

if not duplicates_found:
    print("\n  重複番号はありませんでした。")

# ============================================================
# モデルごとの採用 / 除外 内訳
# ============================================================
print("\n  ▼ モデルごとの採用 / 除外 内訳:")

adopted: dict[int, str] = {}
excluded_log: list[tuple[str, int, str]] = []

for model_name, numbers in model_predictions.items():
    adopted_nums = []
    excluded_nums = []
    for n in numbers:
        if n not in adopted:
            adopted[n] = model_name
            adopted_nums.append(n)
        else:
            excluded_nums.append(n)
            excluded_log.append((model_name, n, adopted[n]))

    adopted_str  = "  ".join(f"{n:2d}" for n in sorted(adopted_nums))  or "（なし）"
    excluded_str = "  ".join(f"{n:2d}" for n in sorted(excluded_nums)) or "（なし）"
    print(f"\n  {model_name}")
    print(f"    採用番号 : {adopted_str}")
    print(f"    除外番号 : {excluded_str}")
    if excluded_nums:
        for n in excluded_nums:
            print(f"      ※ {n:2d} は「{adopted[n]}」がすでに予測済みのため除外")

# ============================================================
# 最終結果
# ============================================================
final_unique = sorted(adopted.keys())

print(f"\n{'='*55}")
print("  【最終結果：重複除去後の予測番号】")
print(f"{'='*55}")
print(f"  予測番号 ({len(final_unique)}個): {final_unique}")
print()
for n in final_unique:
    print(f"    {n:2d}  ←  {adopted[n]}")
print(f"{'='*55}\n")
