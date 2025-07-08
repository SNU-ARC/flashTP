# FlashTP [paper](https://openreview.net/pdf?id=wiQe95BPaB)

# Overview
This repository provides the implementation of FlashTP from "FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials" at ICML '25.

FlashTP is a Tensor-Product library that accelerates Tensor-Product Layer of MLIPs which is equipped with a custom CUDA kernels that incoporates optimization such as kernel fusion, sparse computation and input reuse techniques. For more detail on optimizations please check our paper. 

# Prerequisites
- Python 3.11
- CUDA Toolkit 12 or higher
- Pytorch 2.4.1 or higher
- e3nn 
- torch_scatter (For more accruate _e3nn_ performance evaluation )

# Installation and features
1. Clone this repo.
2. Install CUDA toolkit 12 or higher on your machine.
3. Install required Python packages. We recommend using Python 3.11 with either Virtualenv or Conda to manage the dependencies.
```
pip install -r requirements.txt
pip install . --no-build-isolation
```

# How to use FlashTP
From 
```
tp = e3nn.o3.TensorProduct(i_in1,i_in2,i_out,inst_tuple,shared_weights=False, internal_weights=False)
...
in1 = in1_node[edge_src]
out_large_exp = tp(in1,in2,weight)
out_exp = scatter(out_large_exp, t_edge_dst.to(torch.int64), dim=0, dim_size=total_node, reduce="sum")
```
To 
```
import flashTP_e3nn
flashtp = flashTP_e3nn.uvu_TP(i_in1,i_in2,i_out,inst_tuple, device="cuda", dtype=used_dtype)
...
out_ours = flashtp(in1_node_c,in2_c,weight_c, t_edge_src, t_edge_dst)
```
# Evaluation
TODO

# Citation
Please cite our paper if you find our work useful.
```
@inproceedings{leeflashtp,
  title={FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials},
  author={Lee, Seung Yul and Kim, Hojoon and Park, Yutack and Jeong, Dawoon and Han, Seungwu and Park, Yeonhong and Lee, Jae W},
  booktitle={Forty-second International Conference on Machine Learning}
}
```