import argparse
import re
import subprocess
import sys
import time


def system_free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else 100


def tree_rss_mb(pid):
    out = subprocess.run(["ps", "-o", "pid=,ppid=,rss=", "-ax"], capture_output=True, text=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    children = {}
    rss = {}
    for p, pp, r in rows:
        children.setdefault(int(pp), []).append(int(p))
        rss[int(p)] = int(r)
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += children.get(p, [])
    return total / 1024


def main():
    ap = argparse.ArgumentParser(description="run a command, kill it if system memory gets low or its process tree grows too large")
    ap.add_argument("--min-free-pct", type=int, default=12)
    ap.add_argument("--max-rss-gb", type=float, default=20)
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--log", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    child = subprocess.Popen(cmd)
    peak_rss, min_free, t0 = 0.0, 100, time.time()
    log = open(a.log, "a") if a.log else None
    try:
        while child.poll() is None:
            time.sleep(a.interval)
            rss, free = tree_rss_mb(child.pid), system_free_pct()
            peak_rss, min_free = max(peak_rss, rss), min(min_free, free)
            if log and int(time.time() - t0) % 60 < a.interval:
                log.write(f"{time.strftime('%H:%M:%S')} rss={rss / 1024:.1f}GB free={free}%\n"); log.flush()
            if free < a.min_free_pct or rss / 1024 > a.max_rss_gb:
                msg = f"MEMORY GUARD: killing pid {child.pid} (rss {rss / 1024:.1f} GB, system free {free}%)"
                print(msg, file=sys.stderr, flush=True)
                if log:
                    log.write(msg + "\n"); log.flush()
                child.kill()
                child.wait()
                sys.exit(137)
    finally:
        summary = f"guard summary: exit={child.returncode} peak_rss={peak_rss / 1024:.1f}GB min_free={min_free}% wall={time.time() - t0:.0f}s"
        print(summary, file=sys.stderr, flush=True)
        if log:
            log.write(summary + "\n"); log.close()
    sys.exit(child.returncode or 0)


if __name__ == "__main__":
    main()
