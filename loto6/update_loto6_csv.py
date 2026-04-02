"""
ロト6当選番号CSV更新スクリプト

処理の流れ:
  1. CSVをダウンロードして不要列を削除（上書き保存）
  2. みずほ銀行のページからSeleniumで最新当選番号を取得
  3. CSVに未追記の回があれば追記する

【事前準備】
    pip install --upgrade selenium
    msedgedriver.exe をこのスクリプトと同じフォルダに置く
"""

import urllib.request
import shutil
import os
import sys
import csv
import codecs
import re
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By

# ============================================================
# 設定
# ============================================================
DOWNLOAD_URL = "https://loto6.thekyo.jp/data/loto6.csv"
MIZUHO_URL   = "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html"
CSV_FILENAME = "loto6.csv"
WAIT_SEC     = 15  # JS描画待機秒数

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(SCRIPT_DIR, CSV_FILENAME)
DRIVER_PATH  = os.path.join(SCRIPT_DIR, "../lib/msedgedriver.exe")

# 削除する列名
DROP_COLS = [
    "1等口数", "2等口数", "3等口数", "4等口数", "5等口数",
    "1等賞金", "2等賞金", "3等賞金", "4等賞金", "5等賞金",
    "キャリーオーバー",
]

NUM_COLS  = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字"]
BONUS_COL = "BONUS数字"

# ============================================================
# ① CSVダウンロード＆不要列削除
# ============================================================
def get_csv_latest_round_from_path(file_path: str) -> int:
    """指定パスのCSVの最新回を返す（ファイルがなければ0）"""
    if not os.path.exists(file_path):
        return 0
    try:
        with codecs.open(file_path, "r", encoding="shift_jis") as f:
            reader = csv.DictReader(f)
            rounds = [int(row["開催回"]) for row in reader if row.get("開催回", "").isdigit()]
        return max(rounds) if rounds else 0
    except Exception:
        return 0


def download_csv(url: str, save_path: str) -> bool:
    """
    CSVをダウンロードして上書きする。
    既存CSVの最新回がダウンロードしたCSVの最新回より新しい場合は上書きしない。
    上書きした場合はTrue、スキップした場合はFalseを返す。
    """
    print(f"\n【STEP1】CSVダウンロード")
    print(f"  URL: {url}")

    # 既存CSVの最新回を事前に確認
    existing_latest = get_csv_latest_round_from_path(save_path)
    if existing_latest > 0:
        old_size  = os.path.getsize(save_path)
        old_mtime = datetime.fromtimestamp(os.path.getmtime(save_path))
        print(f"  既存ファイル: {old_size:,} bytes  (更新日時: {old_mtime:%Y-%m-%d %H:%M:%S})")
        print(f"  既存CSV最新回: 第{existing_latest}回")
    else:
        print("  既存ファイル: なし（新規作成）")

    tmp_path = save_path + ".tmp"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTPステータスエラー: {response.status}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(response, f)

        # ダウンロードしたCSVの最新回を確認
        downloaded_latest = get_csv_latest_round_from_path(tmp_path)
        print(f"  ダウンロードCSV最新回: 第{downloaded_latest}回")

        # 既存CSVの最新回がダウンロードより新しい場合は上書きしない
        if existing_latest > downloaded_latest:
            os.remove(tmp_path)
            print(f"  [SKIP] 上書きスキップ: 既存CSV（第{existing_latest}回）の方が"
                  f"ダウンロードCSV（第{downloaded_latest}回）より新しいため。")
            return False

        shutil.move(tmp_path, save_path)
        print(f"  ダウンロード完了・上書き: {os.path.getsize(save_path):,} bytes")
        return True

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"  [エラー] ダウンロード失敗: {e}", file=sys.stderr)
        sys.exit(1)


def drop_columns(file_path: str, drop_cols: list) -> None:
    print(f"\n【STEP2】不要列の削除")
    with codecs.open(file_path, "r", encoding="shift_jis") as f:
        reader = csv.DictReader(f)
        all_cols   = list(reader.fieldnames)
        fieldnames = [col for col in all_cols if col not in drop_cols]
        rows       = [{k: v for k, v in row.items() if k not in drop_cols} for row in reader]

    removed = [col for col in drop_cols if col in all_cols]
    with codecs.open(file_path, "w", encoding="shift_jis", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  削除した列 ({len(removed)}列): {', '.join(removed)}")
    print(f"  残った列: {', '.join(fieldnames)}")


# ============================================================
# CSVの最新回を読み取る
# ============================================================


# ============================================================
# ② Seleniumで全回の当選番号を取得
# ============================================================
def build_driver() -> webdriver.Edge:
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    service = Service(executable_path=DRIVER_PATH)
    return webdriver.Edge(service=service, options=options)


def parse_table(table, target_round: int) -> dict | None:
    """
    テーブルから指定回の当選番号を解析して返す。
    取得できない場合は None を返す。
    """
    if f"第{target_round}回" not in table.text:
        return None

    cells      = table.find_elements(By.TAG_NAME, "td")
    cell_texts = [c.text.strip() for c in cells]

    # 抽選日
    date = None
    for t in cell_texts:
        m = re.match(r"(\d{4}年\d{1,2}月\d{1,2}日)", t)
        if m:
            date = m.group(1)
            break

    # 本数字（ゼロ埋め対応）
    num_cells = []
    for t in cell_texts:
        if re.fullmatch(r"0?([1-9]|[12]\d|3\d|4[0-3])", t):
            num_cells.append(int(t))

    # ボーナス数字（括弧付き）
    bonus = None
    for t in cell_texts:
        m = re.fullmatch(r"\((\d+)\)", t)
        if m:
            bonus = int(m.group(1))
            break

    if len(num_cells) >= 6 and bonus is not None:
        return {
            "round":   target_round,
            "date":    date or "不明",
            "numbers": sorted(num_cells[:6]),
            "bonus":   bonus,
        }
    return None


def get_date_for_round(csv_path: str, target_round: int):
    """
    CSVから指定回の日付を取得して (year, month) を返す。
    見つからない場合は None を返す。
    """
    try:
        with codecs.open(csv_path, "r", encoding="shift_jis") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("開催回", "").strip() == str(target_round):
                    m = re.match(r"(\d{4})/(\d{1,2})/", row.get("日付", ""))
                    if m:
                        return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def fetch_missing_rounds(missing_rounds: list[int], url: str, wait_sec: int) -> list[dict]:
    """
    みずほ銀行のページから不足している回の当選番号をまとめて取得する。
    当月ページで見つからない回はバックナンバーページを参照する。
    URL形式: {base_url}?year=YYYY&month=M
    """
    print(f"\n【STEP3】みずほ銀行から当選番号を取得")
    print(f"  取得対象: {missing_rounds}")

    # ベースURL（クエリパラメータなし）
    base_url = url.split("?")[0]

    driver = build_driver()
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    results    = []
    remaining  = list(missing_rounds)  # まだ取得できていない回

    try:
        # --- まず当月ページを参照 ---
        print(f"  [当月] ページを開いています: {url}")
        driver.get(url)
        print(f"  JS描画待機中（{wait_sec}秒）...")
        time.sleep(wait_sec)

        page_text  = driver.find_element(By.TAG_NAME, "body").text
        all_rounds = [int(n) for n in re.findall(r"第(\d+)回", page_text)]
        print(f"  ページ上の回: {sorted(set(all_rounds))}")

        tables = driver.find_elements(By.TAG_NAME, "table")
        for target in list(remaining):
            for table in tables:
                data = parse_table(table, target)
                if data:
                    results.append(data)
                    print(f"  [OK] 第{target}回 取得成功: "
                          f"本数字={data['numbers']} ボーナス={data['bonus']} ({data['date']})")
                    remaining.remove(target)
                    break

        # --- 当月ページで取れなかった回をバックナンバーから取得 ---
        if remaining:
            print(f"\n  当月ページで未取得の回: {remaining}")
            print(f"  バックナンバーページを参照します...")

            # 不足回をCSVの日付から年月でグループ化
            # CSVにない場合は現在日時から推定
            from collections import defaultdict
            ym_groups = defaultdict(list)

            for target in remaining:
                ym = get_date_for_round(CSV_PATH, target)
                if ym is None:
                    # CSVにない場合: 当月の前月から順に探す
                    now = datetime.now()
                    ym  = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
                ym_groups[ym].append(target)

            for (year, month), targets in sorted(ym_groups.items()):
                back_url = f"{base_url}?year={year}&month={month}"
                print(f"\n  [{year}年{month}月] ページを開いています: {back_url}")
                driver.get(back_url)
                print(f"  JS描画待機中（{wait_sec}秒）...")
                time.sleep(wait_sec)

                page_text  = driver.find_element(By.TAG_NAME, "body").text
                all_rounds = [int(n) for n in re.findall(r"第(\d+)回", page_text)]
                print(f"  ページ上の回: {sorted(set(all_rounds))}")

                tables = driver.find_elements(By.TAG_NAME, "table")
                for target in list(targets):
                    found = False
                    for table in tables:
                        data = parse_table(table, target)
                        if data:
                            results.append(data)
                            print(f"  [OK] 第{target}回 取得成功: "
                                  f"本数字={data['numbers']} ボーナス={data['bonus']} ({data['date']})")
                            remaining.remove(target)
                            found = True
                            break
                    if not found:
                        # 前月のページも試す
                        prev_month = month - 1 if month > 1 else 12
                        prev_year  = year if month > 1 else year - 1
                        prev_url   = f"{base_url}?year={prev_year}&month={prev_month}"
                        print(f"  [NG] 第{target}回 が {year}年{month}月ページに見つからず。"
                              f"前月({prev_year}年{prev_month}月)を試します...")
                        driver.get(prev_url)
                        time.sleep(wait_sec)
                        tables2 = driver.find_elements(By.TAG_NAME, "table")
                        for table in tables2:
                            data = parse_table(table, target)
                            if data:
                                results.append(data)
                                print(f"  [OK] 第{target}回 取得成功（前月ページ）: "
                                      f"本数字={data['numbers']} ボーナス={data['bonus']} ({data['date']})")
                                remaining.remove(target)
                                found = True
                                break
                        if not found:
                            print(f"  [NG] 第{target}回 は取得できませんでした。")

    finally:
        driver.quit()

    return results


# ============================================================
# ③ CSVに追記
# ============================================================
def append_to_csv(file_path: str, new_records: list[dict]) -> None:
    print(f"\n【STEP4】CSVへの追記")

    # 日付を YYYY/M/D 形式に変換
    def fmt_date(date_str: str) -> str:
        m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_str)
        if m:
            return f"{m.group(1)}/{int(m.group(2))}/{int(m.group(3))}"
        return date_str

    # 現在のCSVのフィールド名を取得
    with codecs.open(file_path, "r", encoding="shift_jis") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

    with open(file_path, "a", encoding="shift_jis", errors="replace", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        for rec in sorted(new_records, key=lambda r: r["round"]):
            row = {
                "開催回":  rec["round"],
                "日付":    fmt_date(rec["date"]),
                NUM_COLS[0]: rec["numbers"][0],
                NUM_COLS[1]: rec["numbers"][1],
                NUM_COLS[2]: rec["numbers"][2],
                NUM_COLS[3]: rec["numbers"][3],
                NUM_COLS[4]: rec["numbers"][4],
                NUM_COLS[5]: rec["numbers"][5],
                BONUS_COL:   rec["bonus"],
            }
            writer.writerow(row)
            print(f"  追記: 第{rec['round']}回  {fmt_date(rec['date'])}  "
                  f"本数字={rec['numbers']} ボーナス={rec['bonus']}")

    print(f"  追記完了: {len(new_records)}件")


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    # STEP1: ダウンロード
    overwritten = download_csv(DOWNLOAD_URL, CSV_PATH)

    # STEP2: 不要列削除（上書きした場合のみ実行）
    if overwritten:
        drop_columns(CSV_PATH, DROP_COLS)
    else:
        print("\n【STEP2】不要列の削除: スキップ（上書きなし）")

    # CSVの最新回を確認
    csv_latest = get_csv_latest_round_from_path(CSV_PATH)
    print(f"\n  CSV最新回: 第{csv_latest}回")

    # STEP3: みずほ銀行の当月ページから最新回を確認
    driver = build_driver()
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.get(MIZUHO_URL)
    print(f"\n  みずほ銀行ページ読み込み中（{WAIT_SEC}秒）...")
    time.sleep(WAIT_SEC)
    page_text   = driver.find_element(By.TAG_NAME, "body").text
    web_rounds  = sorted(set(int(n) for n in re.findall(r"第(\d+)回", page_text)))
    web_latest  = max(web_rounds) if web_rounds else 0
    driver.quit()

    print(f"  Web最新回: 第{web_latest}回")

    # 不足している回を特定（CSV最新回+1 から web_latest まで連番で）
    if web_latest <= csv_latest:
        print(f"\n  CSVはすでに最新です（第{csv_latest}回）。追記不要。")
    else:
        # CSV最新回+1 〜 web_latest を全て不足としてリストアップ
        missing = list(range(csv_latest + 1, web_latest + 1))
        print(f"\n  不足している回: {missing}")

        # STEP3: 不足回の当選番号を取得（月をまたぐ場合はバックナンバーも参照）
        new_records = fetch_missing_rounds(missing, MIZUHO_URL, WAIT_SEC)

        if new_records:
            # STEP4: CSVに追記
            append_to_csv(CSV_PATH, new_records)
        else:
            print("  追記するデータがありませんでした。")

    print(f"\n{'='*50}")
    print(f"  完了: ロト6当選番号CSVを更新しました。")
    print(f"  最終CSV最新回: 第{get_csv_latest_round_from_path(CSV_PATH)}回")
    print(f"{'='*50}")
