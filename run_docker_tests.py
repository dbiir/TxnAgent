#!venv/bin/python3
"""
Docker-based test runner for TxnSailsServer + TriStar + RL adapter.

Runs each component inside Docker containers, optionally on remote machines:
  - adapter:  runs inside txncompass_server container (or locally)
  - server:   runs inside txncompass_server container on the server machine
  - client:   runs inside txncompass_client container on the client machine

Startup order:
  1. adapter.py  (Python RL agent, listens on :7654)
  2. TxnSailsServer (Java server, StatisticsWorker connects to adapter)
  3. TriStar client (Java benchmark client, connects to server)

Usage:
  python run_docker_tests.py -w ycsb -e postgresql -f hotspot-128
  python run_docker_tests.py -w tpcc -e postgresql -f scalability -n 3
  python run_docker_tests.py -w ycsb -e postgresql -f skew-128 \
      --server-host worker-127 --client-host worker-128
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
# Paths (inside containers)
# ---------------------------------------------------------------------------
SERVER_CONTAINER = "txncompass_server"
CLIENT_CONTAINER = "txncompass_client"
SERVER_PROJECT_DIR = "/data/TxnSailsServer"
CLIENT_PROJECT_DIR = "/data/TriStar"

PREFIX_CMD_SERVER = "java -jar build/libs/TxnSailsServer-fat-2.0-all.jar"
PREFIX_CMD_CLIENT = "java -cp lib/ -jar tristar.jar"

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
# Docker / Remote helpers
# ---------------------------------------------------------------------------
def docker_exec(container: str, cmd: str, workdir: str = None) -> str:
    """Build a 'docker exec' command string."""
    escaped = cmd.replace('"', '\\"')
    if workdir:
        return f'docker exec {container} sh -c "source ~/.bashrc && cd {workdir} && {escaped}"'
    return f'docker exec {container} sh -c "source ~/.bashrc && {escaped}"'


def remote_cmd(host: str, cmd: str) -> str:
    """Wrap a command for SSH execution on a remote host."""
    escaped = cmd.replace('"', '\\"')
    return f'ssh {host} "{escaped}"'


def maybe_remote(host: str, cmd: str) -> str:
    """Wrap cmd for remote execution if host is set, otherwise return as-is."""
    if host:
        return remote_cmd(host, cmd)
    return cmd


def run_shell_command(cmd: str, timeout: int = 600) -> int:
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


def start_background(cmd: str) -> subprocess.Popen:
    """Start a process in the background. Returns the Popen object."""
    print(f"[BG ] {cmd}", flush=True)
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


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


# ---------------------------------------------------------------------------
# Config discovery (via Docker)
# ---------------------------------------------------------------------------
def get_config_files(args, func: str) -> list[str]:
    """Find .xml config files inside the client container."""
    find_cmd = f"find {CLIENT_PROJECT_DIR}/config/{args.wl}/{func}/{args.engine} -type f -name '*.xml'"
    full_cmd = docker_exec(CLIENT_CONTAINER, find_cmd, CLIENT_PROJECT_DIR)
    full_cmd = maybe_remote(args.client_host, full_cmd)

    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    files = sorted([f.strip() for f in result.stdout.strip().split("\n") if f.strip()])
    return files


def get_partition_files(args, pconfig_path: str) -> list[str]:
    """Find partition config files inside the server container."""
    find_cmd = f"find {SERVER_PROJECT_DIR}/{pconfig_path} -type f -name '*.yaml'"
    full_cmd = docker_exec(SERVER_CONTAINER, find_cmd, SERVER_PROJECT_DIR)
    full_cmd = maybe_remote(args.server_host, full_cmd)

    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    files = sorted([f.strip() for f in result.stdout.strip().split("\n") if f.strip()])
    return files


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def run_once(args, func: str, pconfig_path: str = "") -> str:
    """Run one round of tests for a given function."""
    unique_ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    config_path_local = f"config/{args.wl}.xml"
    schema_path_local = f"config/{args.wl}.sql"

    # Partition files
    if pconfig_path:
        partition_file_list = get_partition_files(args, pconfig_path)
    else:
        partition_file_list = [f"config/partition/{args.wl}/partition-2.yaml"]

    config_files = get_config_files(args, func)
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
            result_subdir = f"results/{args.wl}/{func}/{unique_ts}/{partition_name}"

            print(f"\n{'='*60}", flush=True)
            print(f"Run: {case_name} | partition: {partition_name}", flush=True)
            print(f"{'='*60}", flush=True)

            # Log directory (inside server container)
            log_base = f"logs/{args.wl}/{func}/{unique_ts}/{case_name}"
            if pconfig_path:
                log_base += f"/{partition_name}"

            adapter_proc = None
            server_proc = None

            try:
                # -- Create directories --
                mkdir_server = docker_exec(
                    SERVER_CONTAINER, f"mkdir -p {log_base}", SERVER_PROJECT_DIR
                )
                run_shell_command(maybe_remote(args.server_host, mkdir_server), 10)

                mkdir_client = docker_exec(
                    CLIENT_CONTAINER, f"mkdir -p {result_subdir}/{case_name}",
                    CLIENT_PROJECT_DIR
                )
                run_shell_command(maybe_remote(args.client_host, mkdir_client), 10)

                # -- Step 1: Start adapter (RL agent) first --
                adapter_cmd = docker_exec(
                    SERVER_CONTAINER,
                    f"python3 adapter.py -w {args.wl} > {log_base}/adapter.log 2>&1",
                    SERVER_PROJECT_DIR,
                )
                adapter_proc = start_background(
                    maybe_remote(args.server_host, adapter_cmd)
                )
                time.sleep(3)  # wait for adapter to start listening

                # -- Step 2: Start TxnSailsServer --
                server_java_cmd = (
                    f"{PREFIX_CMD_SERVER}"
                    f" -c {config_path_local}"
                    f" -s {schema_path_local}"
                    f" -d {result_subdir}/{case_name}"
                    f" -t {partition_file}"
                    f" -p offline"
                    f" > {log_base}/server.log 2>&1"
                )
                server_cmd = docker_exec(
                    SERVER_CONTAINER, server_java_cmd, SERVER_PROJECT_DIR
                )
                server_proc = start_background(
                    maybe_remote(args.server_host, server_cmd)
                )
                time.sleep(15)  # wait for server to initialize

                # -- Step 3: Run TriStar client (blocking) --
                client_java_cmd = (
                    f"{PREFIX_CMD_CLIENT}"
                    f" -b {args.wl}"
                    f" -c {conf_file}"
                    f" --execute=true"
                    f" -d {result_subdir}/{case_name}"
                    f" > {result_subdir}/{case_name}/stdout.log 2>&1"
                )
                client_cmd = docker_exec(
                    CLIENT_CONTAINER, client_java_cmd, CLIENT_PROJECT_DIR
                )
                run_shell_command(
                    maybe_remote(args.client_host, client_cmd), timeout=600
                )

                print(f"[DONE] {case_name}", flush=True)
                time.sleep(5)

            finally:
                # Cleanup: kill background processes
                kill_process(server_proc)
                kill_process(adapter_proc)

    return unique_ts


def preprocess_labels(workload: str, func: str, unique_ts: str):
    """Generate offline labels from summary JSONs."""
    meta_dir = os.path.join("metas", workload, func, unique_ts)
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
        description="Docker test runner: adapter → server → client"
    )
    parser.add_argument("-w", "--workload", dest="wl", choices=WORKLOADS,
                        type=str, required=True, help="Workload type")
    parser.add_argument("-e", "--engine", dest="engine", choices=ENGINES,
                        type=str, required=True, help="Database engine")
    parser.add_argument("-f", "--function", dest="func", nargs="+",
                        choices=FUNCTIONS, type=str, help="Test functions")
    parser.add_argument("-p", "--partition", dest="pconfig_path", type=str,
                        required=False, help="Partition config path (inside server container)")
    parser.add_argument("-n", "--cnt", dest="cnt", type=int, default=1,
                        help="Number of runs per function")
    parser.add_argument("--server-host", dest="server_host", type=str,
                        default="", help="SSH host for server machine (empty = local)")
    parser.add_argument("--client-host", dest="client_host", type=str,
                        default="", help="SSH host for client machine (empty = local)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()

    server_loc = args.server_host or "local"
    client_loc = args.client_host or "local"
    print(f"Workload: {args.wl}  Engine: {args.engine}  Runs: {args.cnt}",
          flush=True)
    print(f"Server: {SERVER_CONTAINER}@{server_loc}  "
          f"Client: {CLIENT_CONTAINER}@{client_loc}", flush=True)

    funcs = args.func if args.func else FUNCTIONS

    for func in funcs:
        pconfig = args.pconfig_path or ""
        run_cnt(args, func, args.cnt, pconfig)

    print(f"\nStart time: {start_time}")
    print(f"End time:   {datetime.now()}")
