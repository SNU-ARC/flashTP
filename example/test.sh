CUDA_VISIBLE_DEVICES=0 python ./example/all_exp_end2end_torch_profile.py ./ir_config/nequip/4_32_3_p_sc.txt 3 16384 4 fp32 False dd True
# /home2/lsy/.cache/torch_extensions/py311_cu124


CUDA_VISIBLE_DEVICES=1 nsys profile --gpu-metrics-device=1 --gpu-metrics-frequency=100000 -o e3nn_hwinfo_32k -f true  python ./example/all_exp_end2end_torch_base.py ./ir_config/sevennet/sevennet-l3i5.json 1 32768 4 fp32 False dd True 1
# /home2/lsy/.cache/torch_extensions/py311_cu124

/home2/lsy/flashtp_release/ir_config/sevennet/sevennet-l3i5.json


CUDA_VISIBLE_DEVICES=0 python ./example/all_exp_end2end_torch_debug.py ./ir_config/nequip/4_32_3_p_sc.txt 3 16384 4 fp32 False dd True 1
