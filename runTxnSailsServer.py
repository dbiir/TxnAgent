#!venv/bin/python3
import os
from datetime import datetime
import time
import argparse
import sys
import subprocess
import signal
import json
import numpy as np

prefix_cmd_local = "java -jar target/txnSailsServer-1.0-SNAPSHOT-fat.jar"
prefix_cmd_remote_java = "java -cp target/tristar/tristar/lib/ -jar target/tristar/tristar/tristar.jar "
remote_client_dir = "/data/workspace/tristar/"
# "-b tpcc -c config/postgres/sample_tpcc_config.xml --execute=true"
result_prefix = "results/"
meta_prefix = "metas/"
# TODO: check the path in the remote server
config_prefix = "/data/workspace/txnSailsServer/config/" 
workloads = ["ycsb", "tpcc", "smallbank"]
engines = ["postgresql"]
functions = ["scalability", "hotspot-128","skew-128", "wc_ratio-256",
            "bal_ratio-128", "wc_ratio-128", "random-128", "no_ratio-128", "pa_ratio-128",
             "wr_ratio-128", "dynamic-128", "switch-128"]
strategies = ["SERIALIZABLE", "SI_TAILOR", "RC_TAILOR"]
remote_machine_ip = "21.6.68.184"


def run_shell_command(cmd: str, timeout):
    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    try:
        process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate()
        print("Command timed out and was killed.")
    return process.returncode


def exec_cmd(cmd: str):
    print("command: " + cmd)
    exit_status = os.system(cmd)

    if exit_status == 0:
        print("Command executed successfully")
    else:
        print(f"Command failed with exit status {exit_status}")


def traverse_dir(dir_name: str) -> list:
    xml_files = []

    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.startswith('.'):
                continue
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))

    return xml_files


def create_output_file(filepath: str):
    os.makedirs(filepath, exist_ok=True)

    file_name = "stdout.log"
    file_path = os.path.join(filepath, file_name)
    open(file_path, 'w').close()

    sys.stdout = open(file_path, 'w')

    print("create new file: " + file_path)
    return file_path


def refresh_output_channel():
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", dest='func', nargs='+', choices=functions, type=str,
                        help="specify the function")
    parser.add_argument("-w", "--workload", dest='wl', choices=workloads, type=str, required=True,
                        help="specify the workload")
    parser.add_argument("-e", "--engine", dest="engine", choices=engines, type=str, required=True,
                        help="specify the workload")
    parser.add_argument("-n", "--cnt", dest="cnt", type=int, required=False, default=1,
                        help="count of execution")
    parser.add_argument("-p", "--phase", dest="phase", type=str, required=True, default="offline",
                        help="online predict or offline training")

    return parser.parse_args()


def run_once(f: str, online: bool):
    phase: str = "offline"
    process: subprocess.Popen = None
    
    if online:
        phase = "online"
    # traverse the dir
    config_path = "config/" + args.wl + "/" + f + "/" + args.engine + "/"
    config_path_local = "config/" + args.wl + ".xml"
    schema_path_local = "config/" + args.wl + ".sql"
    print("config_path: " + config_path)
    unique_ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    for conf_file in traverse_dir(config_path):
        if online:
            process = subprocess.Popen("python3 adapter.py -w " + args.wl, shell=True, preexec_fn=os.setsid)
            time.sleep(5)
        meta_dir = meta_prefix + args.wl + "/" + f + "/" + unique_ts + "/"
        result_dir = result_prefix + args.wl + "/" + f + "/" + unique_ts + "/"
        case_name = os.path.splitext(os.path.basename(conf_file))[0]
        output_file_path = result_dir + case_name + "/stdout.log"
        print("Run config - { " + case_name + " }")
        # 1. start txnSails server in this server
        java_cmd = prefix_cmd_local + " -c " + config_path_local + " -s " + schema_path_local + " -d " + result_dir + case_name + " -p " + phase
        process = subprocess.Popen(java_cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(5)
        # 1. create the remote directory
        remote_cmd = "ssh " + remote_machine_ip + " \"mkdir -p " + remote_client_dir + result_dir + case_name + "\" "
        run_shell_command(remote_cmd, 10)
        cmd_remote_java = prefix_cmd_remote_java + " -b " + args.wl + " -c " + config_path + case_name + ".xml" + \
            " --execute=true -d " + result_dir + case_name + " -p " + phase + " > " + output_file_path
        remote_cmd = "ssh " + remote_machine_ip + " \"cd " + remote_client_dir + " ; " + cmd_remote_java + "\""
        run_shell_command(remote_cmd, 240)
        print("Finish config - { " + case_name + " }")
        # time.sleep(5)
        # refresh_output_channel()
        if online:
            if process is not None:
                try:
                    process.communicate(timeout=240)
                    process = None
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate()
            else:
                print("process is None")

    if not online: # generate label
        preprocess_labels(f, unique_ts)

    time.sleep(5)
    return unique_ts


def preprocess_labels(f, unique_ts, wrk=""):
    if len(wrk) != 0:
        meta_dir = "metas/" + wrk + "/" + f + "/" + unique_ts
    else:
        meta_dir = meta_prefix + args.wl + "/" + f + "/" + unique_ts
    for entry in os.scandir(meta_dir):
        if entry.is_dir():
            generate_offline_labels(entry.path)


def generate_offline_labels(meta_folder: str):
    files = []
    for entry in os.scandir(meta_folder):
        if entry.is_file() and entry.name.endswith('.summary.json'):
            files.append(entry.path)
    data = {}
    for file in files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            isolation = json_data['Isolation']
            goodput = float(json_data['Goodput (requests/second)'])
            print(isolation)
            if isolation in strategies:
                if isolation not in data or goodput > data[isolation]:
                    data[isolation] = goodput

    ''' 
        proportion 
    '''
    if len(data) != len(strategies):
        return
    max_goodput_key = max(data, key=data.get)
    max_goodput = data[max_goodput_key]
    # print("max_goodput's type:", type(max_goodput))
    label = [(data[iso] / max_goodput) for iso in strategies]

    '''
        max_index
    '''
    # max_goodput = max(data, key=data.get)
    # max_goodput_index = strategies.index(max_goodput)
    # label = np.zeros(len(strategies), dtype=int)
    # label[max_goodput_index] = 1

    label_file_path = meta_folder + '/label'
    with open(label_file_path, 'w') as label_file:
        label_file.write(','.join(str(x) for x in label))

    for file in files:
        os.remove(file)


def run_cnt(f: str, online: bool, cnt: int):
    timestamps = []
    for i in range(cnt):
        ts = run_once(f, online)
        timestamps.append(ts)
    return timestamps


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    print("workload: " + args.wl + " engine: " + args.engine + " cnt: " + str(args.cnt))
    online_predict = False
    if args.phase == "online":
        online_predict = True
    ff = functions
    if args.func is not None:
        ff = args.func

    for f in ff:
        tss = run_cnt(f, online_predict, args.cnt)

        if not online_predict and f == "random-128":
            offline_service = OfflineService(args.wl)
            print(f, meta_prefix + "/" + args.wl, tss)
            offline_service.service("train", f, meta_prefix + "/" + args.wl, tss)
            print("success")

    print("start time: ", start_time)
    print("end time: ", datetime.now())
