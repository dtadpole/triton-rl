# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd(
    x_ptr, weight_ptr, output_ptr,
    stride_x, stride_weight, stride_out,
    N_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Applies RMS normalization to the input tensor using Triton.

    Args:
        x_ptr (tensor): Pointer to input tensor. Shape (M, N)
        weight_ptr (tensor): Pointer to weight tensor. Shape (N,)
        output_ptr (tensor): Pointer to output tensor. Shape (M, N)
        stride_x (tuple): Strides for input tensor
        stride_weight (tuple): Strides for weight tensor
        stride_out (tuple): Strides for output tensor
        N_SIZE (tl.constexpr): Size of feature dimension (N)
        EPS (tl.constexpr): Epsilon for numerical stability
        BLOCK_SIZE (tl.constexpr): Triton processing block size

    Returns:
        None: Output is written directly to output_ptr

    Explanation:
        1. Calculate row position using program ID
        2. Compute memory offsets for current block
        3. Load input data with masking for partial blocks
        4. Compute squared sum of elements in row
        5. Calculate RMS value using sum and normalization
        6. Normalize and scale elements with weight
        7. Store results to output with proper masking
    """
    row = tl.program_id(0)
    x_ptr += row * stride_x[0]
    output_ptr += row * stride_out[0]

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_SIZE

    x = tl.load(x_ptr + cols * stride_x[1], mask=mask, other=0.0)
    weight = tl.load(weight_ptr + cols * stride_weight[0], mask=mask, other=0.0)

    x_squared = x * x
    row_rms = tl.sqrt(tl.sum(x_squared) / N_SIZE + EPS)

    normalized = x / row_rms
    out = normalized * weight

    tl.store(output_ptr + cols * stride_out[1], out, mask=mask)

def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton wrapper function for RMS normalization.

    Args:
        x (torch.Tensor): Input tensor. Shape (B, D)
        weight (torch.Tensor): Scaling weights. Shape (D,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor. Shape (B, D)

    Explanation:
        1. Validate input dimensions
        2. Preallocate output tensor
        3. Determine optimal block size (power of two up to 1024)
        4. Launch Triton kernel with appropriate grid/block dimensions
    """
    B, D = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(D)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    grid = (B,)
    _rms_norm_fwd[grid](
        x, weight, output,
        x.stride(0), weight.stride(0), output.stride(0),
        N_SIZE=D,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Reference RMS normalization implementation using PyTorch.

    Args:
        x (torch.Tensor): Input tensor. Shape (B, D)
        weight (torch.Tensor): Scaling weights. Shape (D,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor. Shape (B, D)

    Explanation:
        1. Compute squared values of input
        2. Calculate mean squared along feature dimension
        3. Compute RMS value with epsilon
        4. Normalize input using RMS
        5. Apply learned weights
    """
    rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    x_normed = x / (rms + eps)
    return x_normed * weight

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(rms_norm_triton, rms_norm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(rms_norm_triton, rms_norm_torch, input_generator, [batch_sizes, dim_sizes])
