import csv
import subprocess
import multiprocessing
import statistics

# metric_keys = [
#     "Total Kernel Execution Time (Forward)",
#     "Kernel Time Span (Forward)",
#     "Total Kernel Execution Time (Backward)",
#     "Kernel Time Span (Backward)",
#     "Total Kernel Execution Time (Backward to Backward)",
#     "Kernel Time Span (Backward to Backward)",
# ]

metric_keys = [
    # "Total Kernel Execution Time (Forward)",
    "Kernel Time Span (Forward)",
    # "Total Kernel Execution Time (Backward)",
    "Kernel Time Span (Backward)",
    # "Total Kernel Execution Time (Backward to Backward)",
    "Kernel Time Span (Backward to Backward)",
]

iter_range = range(1, 10)

method_list = ['E3NN', 'OURS']
# method_list = ['E3NN', 'OURS', 'CUEQ']

# base_dir = ""
folder_name = "result0502_prev"
profile_dir = f"/home2/lsy/flashtp_release/{folder_name}"

def calculate_filtered_average(values):
    if len(values) == 0:
        return None
    
    median = statistics.median(values)
    threshold = 2

    filtered_values = [v for v in values if (v <= median * threshold and v * threshold >= median)]
    if len(filtered_values) == 0:
        return None
    
    return sum(filtered_values) / len(filtered_values)

def parse_profile(i, logfile, method, metric_keys):
    cmd = ["python", "kernel_profiler.py", "--logfile", logfile, "--method", method]
                    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    parsed = {}
    for key in metric_keys:
        parsed[key] = None
    
    lines = proc.stdout.splitlines()
    for line in lines:
        line = line.strip()
        for key in metric_keys:
            if line.startswith(key + ":"):
                try:
                    val_str = line.split(":")[1].strip().split()[0] 
                    val = float(val_str)
                    parsed[key] = val
                except:
                    pass
    return parsed

with open(f"kernel_{folder_name}_profile.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    header = ["dtype", "model", "channel", "lmax", "bsize", "layer", "method"] + metric_keys
    csv_writer.writerow(header)

    # ------------ NEQUIP -------------
    h_list = [32] # , 64
    lmax_list = [1, 2, 3, 4] # 1, 2, 3, 4, 5
    bsize_list = [32768] # 16384, 
    layer_list = [3] # 0, 1, 2, 
    dtype_list = ["fp32"] # "fp32","fp64"

    # h_list = [32] # , 64
    # lmax_list = [5] # 1, 2, 3, 4, 5
    # bsize_list = [32768] # 16384, 
    # layer_list = [3] # 0, 1, 2, 
    # dtype_list = ["fp64"] # "fp32", 

    for dtype in dtype_list:
        for h in h_list:
            for lmax in lmax_list:
                for bsize in bsize_list:
                    for layer in layer_list:
                        for method in method_list:
                            results = {key: [] for key in metric_keys}

                            tasks = []
                            for i in iter_range:
                                # logfile = f"{profile_dir}/nequip_h_{h}_lmax_{lmax}_bsize_{bsize}_layer_{layer}/iter_{i}.json"
                                logfile = f"{profile_dir}/{dtype}/nequip_h_{h}_lmax_{lmax}_bsize_{bsize}_layer_{layer}_exp/iter_{i}.json"
                                tasks.append((i, logfile, method, metric_keys))

                            with multiprocessing.Pool() as pool:
                                outputs = pool.starmap(parse_profile, tasks)

                            for out in outputs:
                                for key in metric_keys:
                                    if out[key] is not None:
                                        results[key].append(out[key])

                            print(f'model: nequip, h: {h}, lmax: {lmax}, bsize: {bsize}, layer: {layer}, trace: {method}')
                            
                            if len(results[metric_keys[0]]) != 0:
                                row = [dtype, "nequip",h, lmax, bsize, layer, method]

                                for key in metric_keys:
                                    values = results[key]
                                    # avg_val = sum(values) / len(values)
                                    print(values)
                                    avg_val = calculate_filtered_average(values) # auto filter 
                                    print(f"{key}: {avg_val:.6f} ms")
                                    row.append(f"{avg_val:.6f}")

                                csv_writer.writerow(row)

                                # for key in metric_keys:
                                #     values = results[key]
                                #     avg_val = sum(values) / len(values)
                                #     print(f"{avg_val:.6f}", end=",")
                            else:
                                print("No profile")

                            print()
                            print()
