"""Run the wind pipeline. Resumable: finished tasks are skipped.
Examples:
  python scripts/run_wind.py --dry-run
  python scripts/run_wind.py --only 26,61,7 --workers 1
  nohup caffeinate -i python -u scripts/run_wind.py --workers 3 < /dev/null >> data/interim/wind/run_wind.log 2>&1 &
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from oceanembed.wind_pipeline import INTERIM, make_tasks, needed_chunks, run_task, task_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=-1, help="only this time chunk")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="one task, e.g. 26,61,7")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = make_tasks()
    if args.chunk >= 0:
        tasks = [t for t in tasks if t[2] == args.chunk]
    if args.only:
        want = tuple(int(x) for x in args.only.split(","))
        if want not in tasks:
            raise SystemExit(f"{want} is not a valid task; examples: {tasks[:3]}")
        tasks = [want]
    todo = [t for t in tasks if not os.path.exists(task_path(*t))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"time chunks {needed_chunks()} | tasks {len(tasks)} | done {len(tasks) - len([t for t in tasks if not os.path.exists(task_path(*t))])}"
          f" | to run now {len(todo)} | workers {args.workers}")
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
            w.writerow(["time", "a", "b", "chunk", "status", "seconds", "message"])
        futures = [pool.submit(run_task, *t) for t in todo]
        for fut in as_completed(futures):
            r = fut.result()
            w.writerow([datetime.now().isoformat(timespec="seconds"), r["a"], r["b"], r["k"],
                        r["status"], round(r["seconds"], 1), r["message"]])
            f.flush()
            finished += 1
            failed += r["status"] == "failed"
            eta = (time.time() - t_start) / finished * (len(todo) - finished) / 60
            print(f"[{finished}/{len(todo)}] task {r['a']},{r['b']},{r['k']}: {r['status']} in {r['seconds']:.0f} s"
                  f" | failed so far {failed} | ETA about {eta:.0f} min"
                  + (f" | {r['message']}" if r["status"] == "failed" else ""))


if __name__ == "__main__":
    main()
