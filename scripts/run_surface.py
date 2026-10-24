"""Run the monthly surface-field pipeline. Resumable: finished months are skipped.
Examples:
  python scripts/run_surface.py --product ssh --dry-run
  python scripts/run_surface.py --product ssh --first 2022-01 --last 2022-01 --workers 1
  nohup caffeinate -i python -u scripts/run_surface.py --product ssh --workers 2 >> data/interim/surface/run_ssh.log 2>&1 &
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from oceanembed.surface_pipeline import INTERIM, month_range, run_task, task_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="ssh")
    ap.add_argument("--first", default="2010-02")
    ap.add_argument("--last", default="2025-12")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    months = month_range(args.first, args.last)
    todo = [m for m in months if not os.path.exists(task_path(args.product, m))]
    print(f"{args.product}: months {len(months)} | done {len(months) - len(todo)} | to run now {len(todo)} "
          f"| workers {args.workers}")
    if args.dry_run or not todo:
        print("first months to run:", todo[:3])
        return

    os.makedirs(INTERIM, exist_ok=True)
    log_path = os.path.join(INTERIM, "log.csv")
    new_log = not os.path.exists(log_path)
    t_start = time.time()
    finished = failed = 0
    with open(log_path, "a", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.writer(f)
        if new_log:
            w.writerow(["time", "product", "month", "status", "seconds", "message"])
        futures = [pool.submit(run_task, args.product, m) for m in todo]
        for fut in as_completed(futures):
            r = fut.result()
            w.writerow([datetime.now().isoformat(timespec="seconds"), r["product"], r["month"],
                        r["status"], round(r["seconds"], 1), r["message"]])
            f.flush()
            finished += 1
            failed += r["status"] == "failed"
            eta = (time.time() - t_start) / finished * (len(todo) - finished) / 60
            print(f"[{finished}/{len(todo)}] {r['month']}: {r['status']} in {r['seconds']:.0f} s"
                  f" | failed so far {failed} | ETA about {eta:.0f} min"
                  + (f" | {r['message']}" if r["status"] == "failed" else ""))


if __name__ == "__main__":
    main()
