#!venv/bin/python3
import os
from datetime import datetime
import time
import argparse
import sys
import subprocess
import signal
import json
# import numpy as np

prefix_cmd_java_server = "java -jar build/libs/TxnSailsServer-fat-2.0-all.jar"  # local
prefix_cmd_java_client = "java -cp target/tristar/tristar/lib/ -jar target/tristar/tristar/tristar.jar "
remote_client_dir = "/data/TriStar/"
# "-b tpcc -c config/postgres/sample_tpcc_config.xml --execute=true"
result_prefix = "results/"
meta_prefix = "metas/"
# TODO: check the path in the remote server
workloads = ["ycsb", "tpcc", "smallbank"]
engines = ["postgresql"]
functions = ["scalability", "hotspot-128","skew-128", "wc_ratio-256",
            "bal_ratio-128", "wc_ratio-128", "random-128", "no_ratio-128", "pa_ratio-128",
             "wr_ratio-128", "dynamic-128", "switch-128"]
strategies = ["SERIALIZABLE", "SI_TAILOR", "RC_TAILOR"]
remote_machine_ip = "worker-128"


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
    
    return parser.parse_args()


def gen_docker_cmd(docker_name: str, cmd:str, path: str = None):
    escaped_cmd = cmd.replace('"', '\\"')
    
    if path:
        docker_cmd = f'docker exec {docker_name} sh -c "source ~/.bashrc && cd {path} && {escaped_cmd}"'
    else:
        docker_cmd = f'docker exec {docker_name} sh -c "source ~/.bashrc && {escaped_cmd}"'
    
    return docker_cmd

def gen_remote_cmd(remote_ip: str, cmd:str):
    escaped_cmd = cmd.replace('"', '\\"')
    
    return "ssh " + remote_ip + " \"" + escaped_cmd + "\""


def get_config_files(func: str, engine: str) -> list[str]:
    container_name = "txncompass_client"
    config_path = "/data/TriStar"
    remote_machine_ip = "worker-128"
    
    cmd1 = gen_docker_cmd(container_name, "find /data/TriStar/config/ycsb/" + func + "/" + engine + " -type f", config_path)

    import subprocess
    remote_cmd = gen_remote_cmd(remote_machine_ip, cmd1)
    result = subprocess.run(remote_cmd, shell=True, capture_output=True, text=True)
    # print(f"file list:\n{result.stdout}")
    file_list = result.stdout.split("\n")
    file_list.sort()
    return file_list


def run_once(f: str):
    process: subprocess.Popen = None

    # traverse the dir
    config_path_local = "config/" + args.wl + ".xml"
    schema_path_local = "config/" + args.wl + ".sql"
    
    config_path = "config/" + args.wl + "/" + f + "/" + args.engine + "/"
    print("config_path: " + config_path)
    unique_ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    for conf_file in get_config_files(f, args.engine):
        if conf_file.strip() == "":
            continue
        # if online:
        #     process = subprocess.Popen("python3 adapter.py -w " + args.wl, shell=True, preexec_fn=os.setsid)
        #     time.sleep(5)
        # meta_dir = meta_prefix + args.wl + "/" + f + "/" + unique_ts + "/"
        result_dir = result_prefix + args.wl + "/" + f + "/" + unique_ts + "/"
        case_name = os.path.splitext(os.path.basename(conf_file))[0]
        output_file_path = result_dir + case_name + "/stdout.log"
        print("Run config - { " + case_name + " }")
        cc_name = case_name.split("_cc_")[-1]
        # 1. start txnSails server in this server
        # java -jar build/libs/TxnSailsServer-fat-2.0-all.jar -s config/ycsb.sql -c config/ycsb.xml
        if cc_name == "FS":
            server_output_file_path = "logs/" + args.wl + "/" + f + "/" + unique_ts + "/" + case_name
            mkdir_cmd_server = "mkdir -p " + server_output_file_path
            server_output_file = server_output_file_path + "/stdout.log"
            docker_mkdir_cmd_server = gen_docker_cmd("txncompass_server", mkdir_cmd_server, "/data/TxnSailsServer")
            run_shell_command(docker_mkdir_cmd_server, 10)          
            java_cmd = prefix_cmd_java_server + " -c " + config_path_local + " -s " + schema_path_local + " -d " + result_dir + case_name + " -t config/partition/ycsb/partition-2.yaml -p offline > " + server_output_file + " 2>&1"
            server_docker_cmd = gen_docker_cmd("txncompass_server", java_cmd, "/data/TxnSailsServer")
            
            process = subprocess.Popen(server_docker_cmd, shell=True, preexec_fn=os.setsid)
            time.sleep(15)
        elif cc_name == "SER":
            pass
        else:
            print("Unsupported CC: " + cc_name)
            continue
        
        # 1. create the remote directory
        mkdir_cmd = "mkdir -p " + remote_client_dir + result_dir + case_name
        docker_mkdir_cmd = gen_docker_cmd("txncompass_client", mkdir_cmd, "/data/TriStar")
        remote_docker_mkdir_cmd = gen_remote_cmd(remote_machine_ip, docker_mkdir_cmd)
        run_shell_command(remote_docker_mkdir_cmd, 10)
        
        client_cmd = prefix_cmd_java_client + " -b " + args.wl + " -c " + config_path + case_name + ".xml" + \
            " --execute=true -d " + result_dir + case_name + " > " + output_file_path
        docker_client_cmd = gen_docker_cmd("txncompass_client", client_cmd, "/data/TriStar")
        remote_docker_client_cmd = gen_remote_cmd(remote_machine_ip, docker_client_cmd)
        run_shell_command(remote_docker_client_cmd, 240)
        print("Finish config - { " + case_name + " }")
        time.sleep(5)
        # refresh_output_channel()
        if cc_name == "FS":
            if process is not None:
                try:
                    process.communicate(timeout=240)
                    process = None
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate()
            else:
                print("process is None")

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


def run_cnt(f: str, cnt: int):
    timestamps = []
    for i in range(cnt):
        ts = run_once(f)
        timestamps.append(ts)
    return timestamps


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    print("workload: " + args.wl + " engine: " + args.engine + " cnt: " + str(args.cnt))
    ff = functions
    if args.func is not None:
        ff = args.func

    for f in ff:
        tss = run_cnt(f, args.cnt)

    print("start time: ", start_time)
    print("end time: ", datetime.now())
