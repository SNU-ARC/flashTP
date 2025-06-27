import torch
import e3nn
import itertools

from .sptp_exp_opt.fused_e3nn_exp_opt import fused_uvu_TP_exp_opt
from .sptp_exp_opt_large.fused_e3nn_exp_opt_large import fused_uvu_TP_exp_opt_large
from .sptp_exp_opt_extcg.fused_e3nn_exp_opt_extcg import fused_uvu_TP_exp_opt_extcg


def uvu_TP(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions,
    block_batch_cnt=None,
    device="cuda",
    dtype=torch.float32,
):
    """
    Wrapper function to create and return fused_tp_exp instance

    Args:
        irreps_in1: First input irreps
        irreps_in2: Second input irreps
        irreps_out: Output irreps
        instructions: Instruction tuple
        block_batch_cnt: Number of block batches need to be list of 3 int []
        if not set uses flashtp uses config from heruistic
        device: Device to use (default: "cuda")
        dtype: Data type to use (default: torch.float32)

    Returns:
        fused_tp_exp instance
    """
    # assert if ... =>
    # ir_ out is greater than 512
    # num_path > 512 => can be done during actual initalization

    # need to check the number of unique cg matrix value first

    # if out is larger than int16 64K => use out

    # if number of uint8 is larger than

    # Supported TensorProduct options
    # irrep_normalization = "component"
    # path_normalization  = "element"
    # internal_weights = False
    # shared_weights = False


    # create equivalent TP 
    tp = e3nn.o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights=False,
        internal_weights=False,
    )
    
    unique_cg, unique_cg_mat = extract_cg_info(
        irreps_in1,
        irreps_in2,
        irreps_out, tp.instructions, dtype
    )
    unique_cg_val = list(set(unique_cg))

    if len(unique_cg_val) <= 256:
        if True:
            print("fused_uvu_TP_exp_opt_large")
            return fused_uvu_TP_exp_opt_large(
                i_in1=irreps_in1,
                i_in2=irreps_in2,
                i_out=irreps_out,
                instructions=tp.instructions,
                unique_cg_mat=unique_cg_mat,
                unique_cg_val = unique_cg_val,
                per_block_batch=block_batch_cnt,
                device=device,
                dtype=dtype
            )
        else:
            print("fused_uvu_TP_exp_opt")
            return fused_uvu_TP_exp_opt(
                i_in1=irreps_in1,
                i_in2=irreps_in2,
                i_out=irreps_out,
                inst_tuple=instructions,
                per_block_batch=block_batch_cnt,
                device=device,
                dtype=dtype
            )
    else:
        print("fused_uvu_TP_exp_opt_extcg")
        return fused_uvu_TP_exp_opt_extcg(
            i_in1=irreps_in1,
            i_in2=irreps_in2,
            i_out=irreps_out,
            instructions=tp.instructions,
            unique_cg_mat=unique_cg_mat,
            unique_cg_val = unique_cg_val,
            per_block_batch=block_batch_cnt,
            device=device,
            dtype=dtype
        )


def extract_cg_info(i_in1, i_in2, uvuv_i_out, instructions, dtype):
    unique_cg = []
    unique_cg_mat = {}
    idx = 0
    for inst in instructions:
        i = inst.i_in1
        j = inst.i_in2
        k = inst.i_out

        mul_in1, ir_in1 = i_in1[i]
        mul_in2, ir_in2 = i_in2[j]
        mul_out, ir_out = uvuv_i_out[k]

        cg = e3nn.o3.wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
        idx += 1
        # remove small difference in fp64 precision problem
        # by converting it to fp32 then back to target dtype
        cg_fp32 = cg.to(torch.float32)
        unique_cg += cg_fp32.unique().to(dtype).tolist()

        partial_mat_cg = torch.zeros(
            i_in1[i].dim, i_in2[j].dim, uvuv_i_out[k].dim
        )
        unique_cg_mat[f"{ir_in1.l}_{ir_in2.l}_{ir_out.l}"] = cg

        ## uvuv
        for u, v in itertools.product(range(mul_in1), range(mul_in2)):
            partial_mat_cg[
                u * ir_in1.dim : (u + 1) * ir_in1.dim,
                v * ir_in2.dim : (v + 1) * ir_in2.dim,
                (u * mul_in2 + v) * ir_out.dim : (u * mul_in2 + v + 1) * ir_out.dim,
            ] = cg

    return unique_cg, unique_cg_mat
