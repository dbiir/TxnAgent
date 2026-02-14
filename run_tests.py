#!venv/bin/python3
"""
Single-node test runner for TxnSailsServer + TriStar + RL adapter.

Startup order:
  1. adapter.py  (Python RL agent, listens on :7654)
  2. TxnSailsServer (Java server, StatisticsWorker connects to adapter)
  3. TriStar client (Java benchmark client, connects to server)

Usage:
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
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRISTAR_DIR = os.path.join(PROJECT_DIR, "target", "tristar", "tristar")

PREFIX_CMD_SERVER = "java -jar build/libs/TxnSailsServer-fat-2.0-all.jar"
PREFIX_CMD_CLIENT = f"java -cp {TRISTAR_DIR}/lib/ -jar {TRISTAR_DIR}/tristar.jar"

RESULT_DIR = os.path.join(PROJECT_DIR, "results")
META_DIR = os.path.join(PROJECT_DIR, "metas")

WORKLOADS = ["ycsb", "tpcc", "smallbank"]
ENGINES = ["postgresql"]
FUNCTIONS = [
    "scalability", "hotspot-128", "skew-128", "wc_ratio-256", "wr_ratio-64",
    "skew-64", "bal_ratio-128", "wc_ratio-128", "random-128", "no_ratio-128",
    "pa_ratio-128", "wr_ratio-128", "distributed-128", "wr_ratio-64",
    "distributed-64", "dynamic-128", "switch-128", "scalability_p",
    "wr_ratio-128_p", "skew-128_p",
]
STRATEGIES = ["SERIALIZABLE", "SI_TAILOR", "RC_TAILOR"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_shell_command(cmd: str, timeout: int = 600):
    """Run a shell command with a timeout. Returns the exit code."""
    print(f"[CMD] {cmd}", flush=True)
    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    try:
        process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate()
        print("[WARN] Command timed out and was killed.", flush=True)
    return process.returncode


def start_background(cmd: str, log_file: str = None):
    """Start a process in the background. Returns the Popen object."""
    print(f"[BG ] {cmd}", flush=True)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = open(log_file, "w")
        proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid,
                                stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    return proc


def kill_process(proc: subprocess.Popen):
    """Gracefully kill a background process group."""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def get_config_files(workload: str, func: str, engine: str) -> list[str]:
    """Find all .xml config files for a given function/engine."""
    config_dir = os.path.join(PROJECT_DIR, "config", workload, func, engine)
    if not os.path.isdir(config_dir):
        print(f"[WARN] Config directory not found: {config_dir}", flush=True)
        return []
    files = sorted([
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith(".xml") and not f.startswith(".")
    ])
    return files


def get_partition_files(pconfig_path: str) -> list[str]:
    """Find partition config files in the given directory."""
    if not os.path.isdir(pconfig_path):
        return [pconfig_path] if os.path.isfile(pconfig_path) else []
    files = sorted([
        os.path.join(pconfig_path, f)
        for f in os.listdir(pconfig_path)
        if f.endswith(".yaml") and not f.startswith(".")
    ])
    return files


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def run_once(args, func: str, pconfig_path: str = "") -> str:
    """Run one round of tests for a given function."""
    unique_ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    config_path_local = os.path.join("config", f"{args.wl}.xml")
    schema_path_local = os.path.join("config", f"{args.wl}.sql")

    # Partition files
    if pconfig_path:
        partition_file_list = get_partition_files(pconfig_path)
    else:
        partition_file_list = [f"config/partition/{args.wl}/partition.yaml"]

    config_files = get_config_files(args.wl, func, args.engine)
    if not config_files:
        print(f"[SKIP] No config files for {func}/{args.engine}", flush=True)
        return unique_ts

    for conf_file in config_files:
        case_name = os.path.splitext(os.path.basename(conf_file))[0]
        cc_name = case_name.split("_cc_")[-1]

        if cc_name not in ("FS", "SER"):
            print(f"[SKIP] Unsupported CC: {cc_name}", flush=True)
            continue
        if cc_name == "SER":
            continue

        for partition_file in partition_file_list:
            if not partition_file:
                continue

            partition_name = os.path.splitext(os.path.basename(partition_file))[0]
            result_subdir = os.path.join(
                "results", args.wl, func, unique_ts, partition_name
            )

            print(f"\n{'='*60}", flush=True)
            print(f"Run: {case_name} | partition: {partition_name}", flush=True)
            print(f"{'='*60}", flush=True)

            # Log directories
            log_base = os.path.join("logs", args.wl, func, unique_ts, case_name)
            if pconfig_path:
                log_base = os.path.join(log_base, partition_name)
            os.makedirs(log_base, exist_ok=True)
            os.makedirs(os.path.join(result_subdir, case_name), exist_ok=True)

            adapter_proc = None
            server_proc = None

            try:
                # -- Step 1: Start adapter (RL agent) first --
                adapter_log = os.path.join(log_base, "adapter.log")
                adapter_cmd = f"python3 adapter.py -w {args.wl}"
                adapter_proc = start_background(adapter_cmd, adapter_log)
                time.sleep(3)  # wait for adapter to start listening

                # -- Step 2: Start TxnSailsServer --
                server_log = os.path.join(log_base, "server.log")
                server_cmd = (
                    f"{PREFIX_CMD_SERVER}"
                    f" -c {config_path_local}"
                    f" -s {schema_path_local}"
                    f" -d {result_subdir}/{case_name}"
                    f" -t {partition_file}"
                    f" -p offline"
                )
                server_proc = start_background(server_cmd, server_log)
                time.sleep(15)  # wait for server to initialize

                # -- Step 3: Run TriStar client (blocking) --
                client_log = os.path.join(log_base, "client.log")
                client_cmd = (
                    f"{PREFIX_CMD_CLIENT}"
                    f" -b {args.wl}"
                    f" -c {conf_file}"
                    f" --execute=true"
                    f" -d {result_subdir}/{case_name}"
                )
                run_shell_command(f"{client_cmd} > {client_log} 2>&1", timeout=600)

                print(f"[DONE] {case_name}", flush=True)
                time.sleep(5)

            finally:
                # Cleanup: kill server and adapter
                kill_process(server_proc)
                kill_process(adapter_proc)

    return unique_ts


def preprocess_labels(workload: str, func: str, unique_ts: str):
    """Generate offline labels from summary JSONs."""
    meta_dir = os.path.join(META_DIR, workload, func, unique_ts)
    if not os.path.isdir(meta_dir):
        return
    for entry in os.scandir(meta_dir):
        if entry.is_dir():
            generate_offline_labels(entry.path)


def generate_offline_labels(meta_folder: str):
    """Compute proportional labels from summary files."""
    files = [
        entry.path for entry in os.scandir(meta_folder)
        if entry.is_file() and entry.name.endswith(".summary.json")
    ]
    data = {}
    for file in files:
        with open(file, "r") as f:
            json_data = json.load(f)
            isolation = json_data["Isolation"]
            goodput = float(json_data["Goodput (requests/second)"])
            if isolation in STRATEGIES:
                if isolation not in data or goodput > data[isolation]:
                    data[isolation] = goodput

    if len(data) != len(STRATEGIES):
        return

    max_goodput = max(data.values())
    label = [data[iso] / max_goodput for iso in STRATEGIES]

    label_path = os.path.join(meta_folder, "label")
    with open(label_path, "w") as f:
        f.write(",".join(str(x) for x in label))

    for file in files:
        os.remove(file)


def run_cnt(args, func: str, cnt: int, pconfig: str = "") -> list[str]:
    """Run the test `cnt` times."""
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
                        type=str, required=True, help="Workload type")
    parser.add_argument("-e", "--engine", dest="engine", choices=ENGINES,
                        type=str, required=True, help="Database engine")
    parser.add_argument("-f", "--function", dest="func", nargs="+",
                        choices=FUNCTIONS, type=str, help="Test functions")
    parser.add_argument("-p", "--partition", dest="pconfig_path", type=str,
                        required=False, help="Partition config path")
    parser.add_argument("-n", "--cnt", dest="cnt", type=int, default=1,
                        help="Number of runs per function")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    print(f"Workload: {args.wl}  Engine: {args.engine}  Runs: {args.cnt}",
          flush=True)

    funcs = args.func if args.func else FUNCTIONS

    for func in funcs:
        pconfig = args.pconfig_path or ""
        run_cnt(args, func, args.cnt, pconfig)

    print(f"\nStart time: {start_time}")
    print(f"End time:   {datetime.now()}")
