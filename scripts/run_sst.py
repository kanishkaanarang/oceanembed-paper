"""Run the monthly SST pipeline. Resumable. Examples:
  python scripts/run_sst.py --first 2022-01 --last 2022-01 --workers 1
  nohup caffeinate -i python -u scripts/run_sst.py --workers 3 < /dev/null >> data/interim/sst/run_sst.log 2>&1 &"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from oceanembed.sst_pipeline import INTERIM, month_range, run_task, task_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="2010-02")
    ap.add_argument("--last", default="2025-12")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    months = month_range(args.first, args.last)
    todo = [m for m in months if not os.path.exists(task_path(m))]
    print(f"sst: months {len(months)} | done {len(months) - len(todo)} | to run now {len(todo)} | workers {args.workers}")
    if args.dry_run or not todo:
        print("first months to run:", todo[:3])
        return
    os.makedirs(INTERIM, exist_ok=True)
    log_path = os.path.join(INTERIM, "log.csv")
    new_log = not os.path.exists(log_path)
    t_start, finished, failed = time.time(), 0, 0
    with open(log_path, "a", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.writer(f)
        if new_log:
            w.writerow(["time", "month", "status", "seconds", "message"])
        for fut in as_completed([pool.submit(run_task, m) for m in todo]):
            r = fut.result()
            w.writerow([datetime.now().isoformat(timespec="seconds"), r["month"], r["status"],
                        round(r["seconds"], 1), r["message"]])
            f.flush()
            finished += 1
            failed += r["status"] == "failed"
            eta = (time.time() - t_start) / finished * (len(todo) - finished) / 60
            print(f"[{finished}/{len(todo)}] {r['month']}: {r['status']} in {r['seconds']:.0f} s | failed so far {failed}"
                  f" | ETA about {eta:.0f} min" + (f" | {r['message']}" if r["status"] == "failed" else ""))


if __name__ == "__main__":
    main()
