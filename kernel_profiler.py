'''
Calculates the kernel times for forward, backward, and backward-backward passes by marking specific sections of the forward pass with user annotations

USAGE
1. Set a user annotation for the function to be profiled using torch.profiler.record_function.
2. Perform profiling using PyTorch Profiler. (Use /profile/all_end2end_profile.py or all_exp_end2end_profile.py)
3-1. If you want to measure individual log files,
    python kernel_profiler.py --logfile {pytorch_profiler.json} --method {user annotation}
3-2. If you want to measure every log files,
    python profile.py
'''

import json
import argparse
import bisect

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--logfile', type=str, default=None)
parser.add_argument('--method', type=str, default=None)
args = parser.parse_args()
file_path = args.logfile
method = args.method

with open(file_path, 'r') as f:
    data = json.load(f)

trace_events = data.get("traceEvents", [])

ccr = []
kernel_dict = {}
user_annotations = []
fwdbwd_events = []
for e in trace_events:
    cat = e.get("cat")
    if cat in ("cpu_op", "cuda_runtime"):
        ccr.append((e.get("ts", 0), e))
    elif cat in ("kernel", "gpu_memcpy"):
        ext_id = e.get("args", {}).get("External id")
        if ext_id:
            kernel_dict.setdefault(ext_id, []).append(e)
    elif cat == "user_annotation":
        user_annotations.append(e)
    elif cat == "fwdbwd":
        fwdbwd_events.append(e)

ccr.sort(key=lambda x: x[0])

# Forward ops
user_annotation_profiles = [
    e for e in user_annotations
    if e.get("name") == method
]
fwd_cpu_ops = []
for ann in user_annotation_profiles:
    start = ann.get("ts", 0)
    end = start + ann.get("dur", 0)
    ts_list = [x[0] for x in ccr]
    left_idx = bisect.bisect_left(ts_list, start)
    right_idx = bisect.bisect_right(ts_list, end)
    ops_in_profile = [ccr[i][1] for i in range(left_idx, right_idx)]
    fwd_cpu_ops.extend(ops_in_profile)

# Forward kernels
fwd_cpu_op_to_kernels = []
processed_kernels = set()
for op in fwd_cpu_ops:
    ext_id = op.get("args", {}).get("External id")
    if not ext_id:
        fwd_cpu_op_to_kernels.append((op, []))
        continue
    kernels = kernel_dict.get(ext_id, [])
    unique_kerns = []
    for k in kernels:
        kid = (k.get("name"), k.get("ts"), k.get("dur"))
        if kid not in processed_kernels:
            processed_kernels.add(kid)
            unique_kerns.append(k)
    fwd_cpu_op_to_kernels.append((op, unique_kerns))

# Backward ops
all_bw_cpu_ops = []
seq_set = set(x.get("args", {}).get("Sequence number") for x in fwd_cpu_ops)
for op in fwd_cpu_ops:
    fwd_ts = op.get("ts")
    fwd_idx = trace_events.index(op)
    if fwd_idx + 1 < len(trace_events):
        potential_start = trace_events[fwd_idx + 1]
        if (potential_start.get("ph") == "s"
            and potential_start.get("cat") == "fwdbwd"
            and potential_start.get("ts") == fwd_ts):
            flow_id = potential_start.get("id")
            flow_finish = next(
                (ev for ev in fwdbwd_events
                 if ev.get("ph") == "f"
                 and ev.get("cat") == "fwdbwd"
                 and ev.get("id") == flow_id),
                None
            )
            if flow_finish:
                finish_ts = flow_finish.get("ts")
                fi_idx = trace_events.index(flow_finish)
                if fi_idx > 0:
                    potential_bw_op = trace_events[fi_idx - 1]
                    if (potential_bw_op.get("cat") == "cpu_op"
                        and potential_bw_op.get("ts") == finish_ts
                        and potential_bw_op.get("args", {}).get("Sequence number") in seq_set):
                        all_bw_cpu_ops.append(potential_bw_op)

seen_bw = set()
unique_bw_cpu_ops = []
for op in all_bw_cpu_ops:
    key = (op.get("ts",0), op.get("name",""), op.get("args",{}).get("External id",""), op.get("dur",0))
    if key not in seen_bw:
        seen_bw.add(key)
        unique_bw_cpu_ops.append(op)
all_bw_cpu_ops = unique_bw_cpu_ops

bw_start = None
bw_end = None
if all_bw_cpu_ops:
    bw_start = min(o.get("ts", 0) for o in all_bw_cpu_ops)
    bw_end = max(o.get("ts", 0) + o.get("dur", 0) for o in all_bw_cpu_ops)

bw_cpu_ops_range = []
if bw_start is not None and bw_end is not None:
    ts_list = [x[0] for x in ccr]
    left_idx = bisect.bisect_left(ts_list, bw_start)
    right_idx = bisect.bisect_right(ts_list, bw_end)
    bw_cpu_ops_range = [ccr[i][1] for i in range(left_idx, right_idx)]

# Backward kernels
backward_cpu_op_to_kernels = []
for op in bw_cpu_ops_range:
    ext_id = op.get("args", {}).get("External id")
    if not ext_id:
        backward_cpu_op_to_kernels.append((op, []))
        continue
    kernels = kernel_dict.get(ext_id, [])
    unique_kerns = []
    for k in kernels:
        kid = (k.get("name"), k.get("ts"), k.get("dur"))
        if kid not in processed_kernels:
            processed_kernels.add(kid)
            unique_kerns.append(k)
    backward_cpu_op_to_kernels.append((op, unique_kerns))


# Backward Backward ops
bwdbwd_cpu_ops = []
if bw_cpu_ops_range:
    bw_all_start = min(o.get("ts", 0) for o in bw_cpu_ops_range)
    bw_all_end = max(o.get("ts", 0) + o.get("dur", 0) for o in bw_cpu_ops_range)
    tmp_bwdbwd_candidates = []
    seq_set_bw = set(x.get("args", {}).get("Sequence number") for x in bw_cpu_ops_range)
    for op in bw_cpu_ops_range:
        ts_v = op.get("ts")
        idx_v = trace_events.index(op)
        if idx_v + 1 < len(trace_events):
            p = trace_events[idx_v + 1]
            if (p.get("ph") == "s" and p.get("cat") == "fwdbwd" and p.get("ts") == ts_v):
                i = p.get("id")
                z = next((ev for ev in fwdbwd_events
                          if ev.get("ph") == "f" and ev.get("cat") == "fwdbwd" and ev.get("id") == i),
                         None)
                if z:
                    ft = z.get("ts")
                    fi_idx = trace_events.index(z)
                    if fi_idx > 0:
                        pb = trace_events[fi_idx - 1]
                        if (pb.get("cat") == "cpu_op"
                            and pb.get("ts") == ft
                            and pb.get("args", {}).get("Sequence number") in seq_set_bw):
                            tmp_bwdbwd_candidates.append(pb)

    seen_bwdbwd = set()
    for op in tmp_bwdbwd_candidates:
        key = (op.get("ts",0), op.get("name",""), op.get("args",{}).get("External id",""), op.get("dur",0))
        if key not in seen_bwdbwd:
            seen_bwdbwd.add(key)
            bwdbwd_cpu_ops.append(op)

bwdbwd_start = None
bwdbwd_end = None
if bwdbwd_cpu_ops:
    bwdbwd_start = min(o.get("ts", 0) for o in bwdbwd_cpu_ops)
    bwdbwd_end = max(o.get("ts", 0) + o.get("dur", 0) for o in bwdbwd_cpu_ops)

bwdbwd_cpu_ops_range = []
if bwdbwd_start is not None and bwdbwd_end is not None:
    ts_list = [x[0] for x in ccr]
    left_idx = bisect.bisect_left(ts_list, bwdbwd_start)
    right_idx = bisect.bisect_right(ts_list, bwdbwd_end)
    bwdbwd_cpu_ops_range = [ccr[i][1] for i in range(left_idx, right_idx)]

# Backward Backward kernels
bwdbwd_cpu_op_to_kernels = []
bwdbwd_kernel_start_times = []
bwdbwd_kernel_end_times = []
processed_kernels = set()
for op in bwdbwd_cpu_ops_range:
    ext_id = op.get("args", {}).get("External id")
    if not ext_id:
        bwdbwd_cpu_op_to_kernels.append((op, []))
        continue
    kernels = kernel_dict.get(ext_id, [])
    unique_kerns = []
    for k in kernels:
        kid = (k.get("name"), k.get("ts"), k.get("dur"))
        if kid not in processed_kernels:
            processed_kernels.add(kid)
            unique_kerns.append(k)
    bwdbwd_cpu_op_to_kernels.append((op, unique_kerns))

# Sort
fwd_cpu_ops.sort(key=lambda x: x.get("ts", 0))
bw_cpu_ops_range.sort(key=lambda x: x.get("ts", 0))
bwdbwd_cpu_ops_range.sort(key=lambda x: x.get("ts", 0))
fwd_cpu_op_to_kernels.sort(key=lambda x: x[1][0].get("ts", 0) if x[1] else float('inf'))
backward_cpu_op_to_kernels.sort(key=lambda x: x[1][0].get("ts", 0) if x[1] else float('inf'))
bwdbwd_cpu_op_to_kernels.sort(key=lambda x: x[1][0].get("ts", 0) if x[1] else float('inf'))

# Metrics
kernels_flat = []
for _, kernels in fwd_cpu_op_to_kernels:
    kernels_flat.extend(kernels)
if kernels_flat:
    overall_kernel_time = sum(k.get("dur", 0) for k in kernels_flat) / 1000
    kernel_time_span = (
        max(k.get("ts", 0) + k.get("dur", 0) for k in kernels_flat)
        - min(k.get("ts", 0) for k in kernels_flat)
    ) / 1000
else:
    overall_kernel_time = 0
    kernel_time_span = 0

bwd_kernels_flat = []
for _, kernels in backward_cpu_op_to_kernels:
    bwd_kernels_flat.extend(kernels)
if bwd_kernels_flat:
    backward_overall_kernel_time = sum(k.get("dur", 0) for k in bwd_kernels_flat) / 1000
    backward_kernel_time_span = (
        max(k.get("ts", 0) + k.get("dur", 0) for k in bwd_kernels_flat)
        - min(k.get("ts", 0) for k in bwd_kernels_flat)
    ) / 1000
else:
    backward_overall_kernel_time = 0
    backward_kernel_time_span = 0

bwdbwd_kernels_flat = []
for _, kernels in bwdbwd_cpu_op_to_kernels:
    bwdbwd_kernels_flat.extend(kernels)
if bwdbwd_kernels_flat:
    bwdbwd_overall_kernel_time = sum(k.get("dur", 0) for k in bwdbwd_kernels_flat) / 1000
    bwdbwd_kernel_time_span = (
        max(k.get("ts", 0) + k.get("dur", 0) for k in bwdbwd_kernels_flat)
        - min(k.get("ts", 0) for k in bwdbwd_kernels_flat)
    ) / 1000
else:
    bwdbwd_overall_kernel_time = 0
    bwdbwd_kernel_time_span = 0

# Prints
# print("Linked Kernels (Forward):")
# for _, kernels in fwd_cpu_op_to_kernels:
#     for k in kernels:
#         print(f"  Kernel: {k.get('name', 'Unknown Kernel')[:100]} "
#               f"(External id: {k.get('args', {}).get('External id', 'N/A')})")

# print("\nLinked Backward Kernels:")
# for _, kernels in backward_cpu_op_to_kernels:
#     for k in kernels:
#         print(f"  Kernel: {k.get('name', 'Unknown Kernel')[:100]} "
#               f"(External id: {k.get('args', {}).get('External id', 'N/A')})")

# print("\nLinked Backward to Backward Kernels:")
# for _, kernels in bwdbwd_cpu_op_to_kernels:
#     for k in kernels:
#         print(f"  Kernel: {k.get('name', 'Unknown Kernel')[:100]} "
#               f"(External id: {k.get('args', {}).get('External id', 'N/A')})")

# print("\nFirst and Last CPU Ops for each range:")
# if fwd_cpu_ops:
#     print(f"  Forward Range: "
#           f"First: {fwd_cpu_ops[0].get('name', 'Unknown')} "
#           f"(External id: {fwd_cpu_ops[0].get('args', {}).get('External id', 'N/A')}), "
#           f"Last: {fwd_cpu_ops[-1].get('name', 'Unknown')} "
#           f"(External id: {fwd_cpu_ops[-1].get('args', {}).get('External id', 'N/A')})")

# if bw_cpu_ops_range:
#     print(f"  Backward Range: "
#           f"First: {bw_cpu_ops_range[0].get('name', 'Unknown')} "
#           f"(External id: {bw_cpu_ops_range[0].get('args', {}).get('External id', 'N/A')}), "
#           f"Last: {bw_cpu_ops_range[-1].get('name', 'Unknown')} "
#           f"(External id: {bw_cpu_ops_range[-1].get('args', {}).get('External id', 'N/A')})")

# if bwdbwd_cpu_ops_range:
#     print(f"  Backward to Backward Range: "
#           f"First: {bwdbwd_cpu_ops_range[0].get('name', 'Unknown')} "
#           f"(External id: {bwdbwd_cpu_ops_range[0].get('args', {}).get('External id', 'N/A')}), "
#           f"Last: {bwdbwd_cpu_ops_range[-1].get('name', 'Unknown')} "
#           f"(External id: {bwdbwd_cpu_ops_range[-1].get('args', {}).get('External id', 'N/A')})")

print("\nKernel Execution Time Metrics:")
print(f"  Total Kernel Execution Time (Forward): {overall_kernel_time} ms")
print(f"  Kernel Time Span (Forward): {kernel_time_span} ms")
print(f"  Total Kernel Execution Time (Backward): {backward_overall_kernel_time} ms")
print(f"  Kernel Time Span (Backward): {backward_kernel_time_span} ms")
print(f"  Total Kernel Execution Time (Backward to Backward): {bwdbwd_overall_kernel_time} ms")
print(f"  Kernel Time Span (Backward to Backward): {bwdbwd_kernel_time_span} ms")

