#!venv/bin/python3
"""
Single-node test runner for TxnSailsServer + TriStar + RL adapter.

Startup order:
  1. adapter.py  (Python RL agent, listens on :7654)
  2. TxnSailsServer (Java server, StatisticsWorker connects to adapter)
  3. TriStar client (Java benchmark client, connects to server)

Usage:
  python run_tests.py -w ycsb -e postgresql -f online
  python run_tests.py -w ycsb -e postgresql -f hotspot-128
  python run_tests.py -w tpcc -e postgresql -f scalability -n 3
"""
import os
import sys
import time
import signal
import json
import argparse
import subprocess
import atexit
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths — run_tests.py sits inside TxnSailsServer/
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR   = PROJECT_DIR
TRISTAR_DIR  = os.path.join(os.path.dirname(PROJECT_DIR), "TriStar")

PREFIX_CMD_SERVER = f"java -jar {SERVER_DIR}/build/libs/TxnSailsServer-fat-2.0-all.jar"
PREFIX_CMD_CLIENT = (
    f"java -cp {TRISTAR_DIR}/target/tristar/tristar/lib/"
    f" -jar {TRISTAR_DIR}/target/tristar/tristar/tristar.jar"
)

RESULT_DIR = os.path.join(PROJECT_DIR, "results")
META_DIR   = os.path.join(PROJECT_DIR, "metas")

WORKLOADS  = ["ycsb", "tpcc", "smallbank"]
ENGINES    = ["postgresql"]
FUNCTIONS  = [
    "scalability", "hotspot-128", "skew-128", "wc_ratio-256", "wr_ratio-64",
    "skew-64", "bal_ratio-128", "wc_ratio-128", "random-128", "no_ratio-128",
    "pa_ratio-128", "wr_ratio-128", "distributed-128", "distributed-64",
    "dynamic-128", "switch-128", "scalability_p", "wr_ratio-128_p",
    "skew-128_p", "online",
]
STRATEGIES = ["SERIALIZABLE", "SI_TAILOR", "RC_TAILOR"]

# ---------------------------------------------------------------------------
# Global process registry — cleaned up on exit / Ctrl+C
# ---------------------------------------------------------------------------
_bg_procs: list[subprocess.Popen] = []


def _cleanup_all():
    """Kill every tracked background process group."""
    for proc in _bg_procs:
        _kill_proc(proc)
    _bg_procs.clear()


def _sigint_handler(sig, frame):
    print("\n[INTERRUPT] Cleaning up all child processes …", flush=True)
    _cleanup_all()
    sys.exit(130)


signal.signal(signal.SIGINT,  _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)
atexit.register(_cleanup_all)


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
def _kill_proc(proc: subprocess.Popen, timeout: int = 10):
    """Kill a process group gracefully, then forcefully if needed."""
    if proc is None or proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except ProcessLookupError:
        pass                     # already gone
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass
    except Exception:
        pass


def start_background(cmd: str, log_file: str = None, cwd: str = None) -> subprocess.Popen:
    """Start a command in its own session (new process group). Returns Popen."""
    print(f"[BG ] {cmd}", flush=True)
    fh = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, shell=True,
        preexec_fn=os.setsid,          # new session → own pgid
        stdout=fh or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if fh else subprocess.DEVNULL,
        cwd=cwd,
    )
    _bg_procs.append(proc)
    return proc


def run_foreground(cmd: str, timeout: int = 4200, cwd: str = None, log_file: str = None) -> int:
    """Run a command in the foreground (blocking). Returns exit code."""
    print(f"[CMD] {cmd}", flush=True)
    fh = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, shell=True,
        preexec_fn=os.setsid,
        stdout=fh or None,
        stderr=subprocess.STDOUT if fh else None,
        cwd=cwd,
    )
    _bg_procs.append(proc)
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[WARN] Client timed out — killing.", flush=True)
        _kill_proc(proc)
    finally:
        if fh:
            fh.close()
        if proc in _bg_procs:
            _bg_procs.remove(proc)
    return proc.returncode


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------
def get_config_files(workload: str, func: str, engine: str) -> list[str]:
    """Return sorted XML configs.
    
    For 'online', look in the local SERVER_DIR config first; fall back to TriStar.
    """
    candidates = [
        os.path.join(SERVER_DIR, "config", func, engine),
        os.path.join(TRISTAR_DIR, "config", workload, func, engine),
    ]
    for config_dir in candidates:
        if os.path.isdir(config_dir):
            files = sorted(
                os.path.join(config_dir, f)
                for f in os.listdir(config_dir)
                if f.endswith(".xml") and not f.startswith(".")
            )
            if files:
                print(f"[CFG] Using configs from: {config_dir}", flush=True)
                return files
    print(f"[WARN] No config directory found for {func}/{engine}", flush=True)
    return []


def get_partition_files(pconfig_path: str) -> list[str]:
    """Return YAML partition files from a directory or a single file path."""
    if not pconfig_path:
        return []
    if os.path.isfile(pconfig_path):
        return [pconfig_path]
    if os.path.isdir(pconfig_path):
        return sorted(
            os.path.join(pconfig_path, f)
            for f in os.listdir(pconfig_path)
            if f.endswith(".yaml") and not f.startswith(".")
        )
    return []


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------
def run_once(args, func: str, pconfig_path: str = "") -> str:
    """Run one full sweep of configs for the given function."""
    unique_ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    config_path_local = os.path.join(SERVER_DIR, "config", f"{args.wl}.xml")
    schema_path_local = os.path.join(SERVER_DIR, "config", f"{args.wl}.sql")

    partition_file_list = (
        get_partition_files(pconfig_path)
        or [os.path.join(SERVER_DIR, "config", "partition", args.wl, "partition.yaml")]
    )

    config_files = get_config_files(args.wl, func, args.engine)
    if not config_files:
        print(f"[SKIP] No config files found for {func}/{args.engine}", flush=True)
        return unique_ts

    for conf_file in config_files:
        case_name = os.path.splitext(os.path.basename(conf_file))[0]
        cc_name   = case_name.split("_cc_")[-1]

        if cc_name not in ("FS",):          # only run FS for now
            print(f"[SKIP] cc={cc_name}: {case_name}", flush=True)
            continue

        for partition_file in partition_file_list:
            if not partition_file:
                continue

            partition_name = os.path.splitext(os.path.basename(partition_file))[0]
            result_subdir  = os.path.join("results", args.wl, func, unique_ts, partition_name)
            log_base = os.path.join("logs", args.wl, func, unique_ts, case_name)
            if pconfig_path:
                log_base = os.path.join(log_base, partition_name)

            os.makedirs(log_base, exist_ok=True)
            os.makedirs(os.path.join(result_subdir, case_name), exist_ok=True)

            print(f"\n{'='*60}", flush=True)
            print(f"Run: {case_name} | partition: {partition_name}", flush=True)
            print(f"{'='*60}", flush=True)

            adapter_proc = None
            server_proc  = None

            try:
                # 1. Start RL adapter
                adapter_log = os.path.join(log_base, "adapter.log")
                adapter_cmd = f"python3 {SERVER_DIR}/adapter.py -w {args.wl}"
                adapter_proc = start_background(adapter_cmd, adapter_log, cwd=SERVER_DIR)
                time.sleep(3)

                # 2. Start TxnSailsServer
                server_log = os.path.join(log_base, "server.log")
                server_cmd = (
                    f"{PREFIX_CMD_SERVER}"
                    f" -c {config_path_local}"
                    f" -s {schema_path_local}"
                    f" -d {result_subdir}/{case_name}"
                    f" -t {partition_file}"
                    f" -p offline"
                )
                server_proc = start_background(server_cmd, server_log, cwd=SERVER_DIR)
                time.sleep(15)

                # 3. Run TriStar benchmark client (blocking)
                client_log = os.path.join(log_base, "client.log")
                client_cmd = (
                    f"{PREFIX_CMD_CLIENT}"
                    f" -b {args.wl}"
                    f" -c {conf_file}"
                    f" --execute=true"
                    f" -d {result_subdir}/{case_name}"
                )
                run_foreground(client_cmd, timeout=4200, cwd=TRISTAR_DIR, log_file=client_log)

                print(f"[DONE] {case_name}", flush=True)
                time.sleep(5)

            finally:
                # Always clean up server + adapter even if client crashes / times out
                _kill_proc(server_proc)
                _kill_proc(adapter_proc)
                for proc in (server_proc, adapter_proc):
                    if proc and proc in _bg_procs:
                        _bg_procs.remove(proc)

    return unique_ts


# ---------------------------------------------------------------------------
# Offline label generation
# ---------------------------------------------------------------------------
def generate_offline_labels(meta_folder: str):
    """Compute proportional labels from summary JSON files."""
    files = [
        e.path for e in os.scandir(meta_folder)
        if e.is_file() and e.name.endswith(".summary.json")
    ]
    data: dict[str, float] = {}
    for file in files:
        with open(file) as f:
            j = json.load(f)
        iso     = j.get("Isolation")
        goodput = float(j.get("Goodput (requests/second)", 0))
        if iso in STRATEGIES:
            if iso not in data or goodput > data[iso]:
                data[iso] = goodput

    if len(data) != len(STRATEGIES):
        return

    max_g = max(data.values())
    label = [data[iso] / max_g for iso in STRATEGIES]
    label_path = os.path.join(meta_folder, "label")
    with open(label_path, "w") as f:
        f.write(",".join(str(x) for x in label))

    for file in files:
        os.remove(file)


def preprocess_labels(workload: str, func: str, unique_ts: str):
    meta_dir = os.path.join(META_DIR, workload, func, unique_ts)
    if not os.path.isdir(meta_dir):
        return
    for entry in os.scandir(meta_dir):
        if entry.is_dir():
            generate_offline_labels(entry.path)


def run_cnt(args, func: str, cnt: int, pconfig: str = "") -> list[str]:
    timestamps = []
    for i in range(cnt):
        print(f"\n--- Run {i+1}/{cnt} for {func} ---", flush=True)
        ts = run_once(args, func, pconfig)
        timestamps.append(ts)
    return timestamps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-node test runner: adapter → server → client"
    )
    parser.add_argument("-w", "--workload", dest="wl", choices=WORKLOADS,
                        required=True, help="Workload type")
    parser.add_argument("-e", "--engine", dest="engine", choices=ENGINES,
                        required=True, help="Database engine")
    parser.add_argument("-f", "--function", dest="func", nargs="+",
                        choices=FUNCTIONS, help="Test functions (default: all)")
    parser.add_argument("-p", "--partition", dest="pconfig_path",
                        help="Partition config file or directory")
    parser.add_argument("-n", "--cnt", dest="cnt", type=int, default=1,
                        help="Number of runs per function (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    print(f"Workload: {args.wl}  Engine: {args.engine}  Runs: {args.cnt}", flush=True)

    funcs  = args.func or FUNCTIONS
    pconfig = args.pconfig_path or ""

    for func in funcs:
        run_cnt(args, func, args.cnt, pconfig)

    elapsed = datetime.now() - start_time
    print(f"\nStart:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {elapsed}")
