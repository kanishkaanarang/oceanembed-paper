"""Run the GLORYS pipeline for one time block. Resumable: finished tasks are skipped.
Examples:
  python scripts/run_glorys.py --block 5 --dry-run
  python scripts/run_glorys.py --block 5 --only 71,182 --workers 1
  caffeinate -i python scripts/run_glorys.py --block 3 --workers 3
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from oceanembed.glorys_pipeline import INTERIM, block_dates, make_tasks, run_task, task_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="run at most this many remaining tasks (0 = all)")
    ap.add_argument("--only", type=str, default="", help="run one task, e.g. 71,182")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start, end = block_dates(args.block)
    tasks = make_tasks()
    if args.only:
        a, b = (int(x) for x in args.only.split(","))
        if (a, b) not in tasks:
            raise SystemExit(f"{(a, b)} is not a valid task; examples: {tasks[:3]}")
        tasks = [(a, b)]
    remaining = [t for t in tasks if not os.path.exists(task_path(args.block, *t))]
    todo = remaining[: args.limit] if args.limit else remaining
    print(f"block {args.block}: {start.date()} to {end.date()} | tasks {len(tasks)} | "
          f"done {len(tasks) - len(remaining)} | to run now {len(todo)} | workers {args.workers}")
    if args.dry_run or not todo:
        print("first tasks:", todo[:3])
        return

    os.makedirs(INTERIM, exist_ok=True)
    log_path = os.path.join(INTERIM, "log.csv")
    new_log = not os.path.exists(log_path)
    t_start = time.time()
    finished = failed = 0
    with open(log_path, "a", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.writer(f)
        if new_log:
            w.writerow(["time", "block", "a", "b", "status", "seconds", "message"])
        futures = [pool.submit(run_task, args.block, a, b) for a, b in todo]
        for fut in as_completed(futures):
            r = fut.result()
            w.writerow([datetime.now().isoformat(timespec="seconds"), r["block"], r["a"], r["b"],
                        r["status"], round(r["seconds"], 1), r["message"]])
            f.flush()
            finished += 1
            failed += r["status"] == "failed"
            eta = (time.time() - t_start) / finished * (len(todo) - finished) / 60
            print(f"[{finished}/{len(todo)}] task {r['a']},{r['b']}: {r['status']} in {r['seconds']:.0f} s"
                  f" | failed so far {failed} | ETA about {eta:.0f} min"
                  + (f" | {r['message']}" if r["status"] == "failed" else ""))


if __name__ == "__main__":
    main()
