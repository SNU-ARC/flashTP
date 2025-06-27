import torch
import e3nn
import json
import os,sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import openequivariance as oeq
import flashTP_e3nn
# from sptp.fused_e3nn import fused_uvu_TP
# from sptp_exp.fused_e3nn_exp import fused_uvu_TP_exp
# from sptp_exp_opt.fused_e3nn_exp_opt import fused_uvu_TP_exp_opt
# from sptp_exp_opt_extcg.fused_e3nn_exp_opt_extcg import fused_uvu_TP_exp_opt_extcg

import cuequivariance_torch as cuet
from torch_scatter import scatter
# from e3nn import o3 
import re
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def main():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    
    # h = int(sys.argv[1])
    # l_max = int(sys.argv[2])
    filename = sys.argv[1]
    layer_idx = int(sys.argv[2])
    target_batch_size = int(sys.argv[3]) # 16k, 32k
    block_batch_cnt = int(sys.argv[4]) # 4
    used_dtype_str = sys.argv[5]
    run_cueq = sys.argv[6]
    result_dir = sys.argv[7]
    run_e3nn = sys.argv[8]
    channel_multiplier = int(sys.argv[9])
    
    if run_e3nn == "False":
        run_e3nn = False
    else:
        run_e3nn = True

    if run_cueq == "False":
        run_cueq = False
    else:
        run_cueq = True
    
    outdir = os.path.basename(filename)  # 'sevennet-l3i5.json'
    output_dir = os.path.splitext(outdir)[0]  # 'sevennet-l3i5'
    
    if output_dir[0] == "4":
        match = re.search(r'_(\d+)_(\d+)_', output_dir)
        if match:
            h = match.group(1)
            lmax = match.group(2)
            
            output_dir = f"nequip_h_{h}_lmax_{lmax}"
    elif output_dir[0] != "s" or output_dir[1] == "m":
        output_dir = "mace-" + output_dir[0]
    
    ## dytpe selection
    used_dtype = torch.float32
    if (used_dtype_str == "fp64"):
        used_dtype = torch.float64
        
    max_neighbour = 64
    total_node = target_batch_size // max_neighbour
    edge_src, edge_dst = flashTP_e3nn.utils.fixed_generate_edgepair(total_node, max_neighbour)

    batch_size = len(edge_src)
    print("batch_size", batch_size)

    # filename = f"/work/ir_configs/nequip/4_32_2_p_sc.txt"
    # filename = f"/dataset/ir_configs/4_{h}_{l_max}_p_sc.txt"
    # mul_list = {0:128, 1:64}
    e3nn_config, cueq_config = flashTP_e3nn.utils.load_config_e3nn_cueq(filename,layer_idx, channel_mul= channel_multiplier)
    i_in1, i_in2, i_out, inst_tuple = e3nn_config

    # print(e3nn_config)
    # # uvu
    l_max = max([max([x[1].l for x in i_in1]), max([x[1].l for x in i_in2]), max([x[1].l for x in i_out])])

    torch.set_default_dtype(used_dtype)
    tp = e3nn.o3.TensorProduct(i_in1,i_in2,i_out,inst_tuple,shared_weights=False, internal_weights=False) # path_normalization="none", normalization="none"
    tp = tp.to(device="cuda")  
    fused_tp_exp = flashTP_e3nn.uvu_TP(i_in1,i_in2,i_out,inst_tuple, block_batch_cnt=block_batch_cnt, device="cuda", dtype=used_dtype)

       
    # problem = oeq.TPProblem(i_in1,i_in2,i_out,inst_tuple, shared_weights=False, internal_weights=False)
    # tp_conv = oeq.TensorProductConv(problem, torch_op=True, deterministic=False) 
    # tp_fast = oeq.TensorProduct(problem, torch_op=True)

    torch.set_default_dtype(torch.float32)

    cuet_tp = cuet.ChannelWiseTensorProduct(*cueq_config[:-1], shared_weights=False,internal_weights=False, device="cuda", layout="ir_mul", dtype=used_dtype)
    t_edge_src = torch.tensor(edge_src, device="cuda", dtype=torch.int32)
    t_edge_dst = torch.tensor(edge_dst, device="cuda", dtype=torch.int32)


    torch.cuda.cudart().cudaProfilerStart()
    for i in range(10):
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{result_dir}/{used_dtype_str}/{output_dir}_bsize_{batch_size}_layer_{layer_idx}_exp')
        ) as p:
            if run_e3nn:
                in1_node = torch.rand(total_node, i_in1.dim, device="cuda", requires_grad=True, dtype=used_dtype)
                in2 = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True,dtype=used_dtype)
                weight = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True,dtype=used_dtype)

                in1_node_c = in1_node.detach().clone().requires_grad_()
                in2_c = in2.detach().clone().requires_grad_()
                weight_c = weight.detach().clone().requires_grad_()

                # in1_node_o = in1_node.detach().clone().requires_grad_()
                # in2_o = in2.detach().clone().requires_grad_()
                # weight_o = weight.detach().clone().requires_grad_()

            else:
                in1_node_c = torch.rand(total_node, i_in1.dim, device="cuda", requires_grad=True, dtype=used_dtype)
                in2_c = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True,dtype=used_dtype)
                weight_c = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True,dtype=used_dtype)

            if run_cueq:
                cin1_node_cuda = torch.rand(total_node, i_in1.dim, device="cuda", requires_grad=True, dtype=used_dtype)
                cin2_cuda = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True,dtype=used_dtype)
                weight_cueq = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True,dtype=used_dtype)

            if run_e3nn:
                # fwd path
                torch.cuda.nvtx.range_push(f"e3nn_fwd")
                in1 = in1_node[edge_src]
                with torch.profiler.record_function("E3NN"):
                    out_large_exp = tp(in1,in2,weight)
                    out_exp = scatter(out_large_exp, t_edge_dst.to(torch.int64), dim=0, dim_size=total_node, reduce="sum")
                torch.cuda.nvtx.range_pop()
                y = torch.nn.functional.gelu(out_exp).sum()
                # print(out_exp[0])

                # original e3nn
                torch.cuda.nvtx.range_push(f"e3nn_bwd")
                f_in1, f_in2, f_weight = torch.autograd.grad(y, [in1_node,in2,weight], create_graph=True)
                # f_in1_a, f_in2, f_weight = torch.autograd.grad(y, [in1,in2,weight], create_graph=True)
                # torch.cuda.nvtx.range_push(f"e3nn_bwd_dummy")
                # print(f_in1_a[0], f_in1_a.shape)
                # print(f_in1[0])
                # torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()

                f_in1_gelu = torch.nn.functional.gelu(f_in1)
                f_in2_gelu = torch.nn.functional.gelu(f_in2)
                f_weight_gelu = torch.nn.functional.gelu(f_weight)
                fake_loss = f_in1_gelu.sum() + f_in2_gelu.sum() + f_weight_gelu.sum()

                torch.cuda.nvtx.range_push(f"e3nn_bwd_bwd+bwd")
                fake_loss.backward()
                # torch.cuda.nvtx.range_push(f"e3nn_bwd_bwd+bwd_dummy")
                # print(in1_node.grad[0])
                # torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push(f"open_fwd")
            # in1_o = in1_node_o[edge_src]
            # with torch.profiler.record_function("OPEN"):
            #     out_large_exp_o = tp_fast(in1_o,in2_o,weight_o)
            #     out_exp_o = scatter(out_large_exp_o, t_edge_dst.to(torch.int64), dim=0, dim_size=total_node, reduce="sum")
            # torch.cuda.nvtx.range_pop()
            # y_o = torch.nn.functional.gelu(out_exp_o).sum()
            # # print(y)

            # # original e3nn
            # torch.cuda.nvtx.range_push(f"e3nn_bwd")
            # f_in1_o, f_in2_o, f_weight_o = torch.autograd.grad(y_o, [in1_node_o,in2_o,weight_o], create_graph=True)
            # # torch.cuda.nvtx.range_push(f"e3nn_bwd_dummy")
            # # print(f_in1[0])
            # # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_pop()

            # f_in1_gelu_o = torch.nn.functional.gelu(f_in1_o)
            # f_in2_gelu_o = torch.nn.functional.gelu(f_in2_o)
            # f_weight_gelu_o = torch.nn.functional.gelu(f_weight_o)
            # fake_loss_o = f_in1_gelu_o.sum() + f_in2_gelu_o.sum() + f_weight_gelu_o.sum()

            # torch.cuda.nvtx.range_push(f"e3nn_bwd_bwd+bwd")
            # fake_loss_o.backward()
            # # torch.cuda.nvtx.range_push(f"e3nn_bwd_bwd+bwd_dummy")
            # # print(in1_node.grad[0])
            # # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_pop()


            #### ours 

            torch.cuda.nvtx.range_push(f"sptp_fwd")
            with torch.profiler.record_function("OURS"):
                out_ours = fused_tp_exp(in1_node_c,in2_c,weight_c, t_edge_src, t_edge_dst)
            torch.cuda.nvtx.range_pop()
            y_ours = torch.nn.functional.gelu(out_ours).sum()

            # original e3nn
            torch.cuda.nvtx.range_push(f"sptp_bwd")
            f_in1_c, f_in2_c, f_weight_c = torch.autograd.grad(y_ours, [in1_node_c,in2_c,weight_c], create_graph=True)
            # torch.cuda.nvtx.range_push(f"sptp_bwd_dummy")
            # print(f_in1_c[0])
            # torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()

            f_in1_gelu_c = torch.nn.functional.gelu(f_in1_c)
            f_in2_gelu_c = torch.nn.functional.gelu(f_in2_c)
            f_weight_gelu_c = torch.nn.functional.gelu(f_weight_c)
            fake_loss_c = f_in1_gelu_c.sum() + f_in2_gelu_c.sum() + f_weight_gelu_c.sum()
            
            torch.cuda.nvtx.range_push(f"sptp_bwd_bwd+bwd")
            fake_loss_c.backward()
            # torch.cuda.nvtx.range_push(f"sptp_bwd_bwd+bwd_dummy")
            # print(in1_node_c.grad[0])
            # torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()




            if run_cueq:
                torch.cuda.nvtx.range_push(f"cueq_irmul_fwd")
                cin1_cuda = cin1_node_cuda[edge_src]
                with torch.profiler.record_function("CUEQ"):
                    cuet_out_exp = cuet_tp(cin1_cuda,cin2_cuda,weight_cueq)
                    cuet_out = scatter(cuet_out_exp, t_edge_dst.to(torch.int64), dim=0, dim_size=total_node, reduce="sum")
                torch.cuda.nvtx.range_pop()
                y_cu = torch.nn.functional.gelu(cuet_out).sum()

                torch.cuda.nvtx.range_push(f"cueq_irmul_bwd")
                f_in1_cu, f_in2_cu, f_weight_cu = torch.autograd.grad(y_cu, [cin1_node_cuda,cin2_cuda,weight_cueq], create_graph=True)
                # torch.cuda.nvtx.range_push(f"cueq_irmul_bwd_dummy")
                # print(f_in1_cu[0])
                # torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()

            
                f_in1_gelu_cu = torch.nn.functional.gelu(f_in1_cu)
                f_in2_gelu_cu = torch.nn.functional.gelu(f_in2_cu)
                f_weight_gelu_cu = torch.nn.functional.gelu(f_weight_cu)
                fake_loss_cu = f_in1_gelu_cu.sum() + f_in2_gelu_cu.sum() + f_weight_gelu_cu.sum()

                torch.cuda.nvtx.range_push(f"cueq_irmul_bwd_bwd+bwd")
                fake_loss_cu.backward()
                # torch.cuda.nvtx.range_push(f"cueq_irmul_bwd_bwd+bwd_dummy")
                # print(cin1_node_cuda.grad[0])
                # torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
            
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    main()