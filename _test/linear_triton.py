# filename: linear_triton.py
import torch
import triton
import triton.language as tl

# PyTorch reference implementation
class TorchLinear:
    @staticmethod
    def forward(x, weight):
        """Forward pass for linear layer without bias: y = x @ w.T"""
        return x @ weight.t()  # x: [batch_size, in_features], weight: [out_features, in_features]

    @staticmethod
    def backward(grad_output, x, weight):
        """
        Backward pass for linear layer:
        - grad_x = grad_output @ weight
        - grad_weight = grad_output.T @ x
        """
        grad_x = grad_output @ weight  # grad_output: [batch_size, out_features], grad_x: [batch_size, in_features]
        grad_weight = grad_output.transpose(-2, -1) @ x  # grad_weight: [out_features, in_features]
        return grad_x, grad_weight

# Triton implementation for forward pass (GEMM)
@triton.jit
def _linear_forward_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute matrix multiplication C = A @ B.T
    A is of shape (M, K) and B is of shape (N, K)
    """
    # -----------------------------------------------------------
    # Map program ids to C matrix coordinates (m, n)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Compute tile indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # ----------------------------------------------------------
    # Create block pointers
    a_block_ptr = a_ptr + m_start * stride_am  # Input block pointer
    b_block_ptr = b_ptr + n_start * stride_bn  # Weight block pointer
    c_block_ptr = c_ptr + m_start * stride_cm + n_start * stride_cn  # Output block pointer

    # -----------------------------------------------------------
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K

        # Load the next block of A and B
        a = tl.load(a_block_ptr + k_start * stride_ak,
                   mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < M - m_start and
                        tl.arange(0, BLOCK_SIZE_K)[None, :] < K - k_start)  # [BLOCK_SIZE_M, BLOCK_SIZE_K]

        b = tl.load(b_block_ptr + k_start * stride_bk,
                   mask=tl.arange(0, BLOCK_SIZE_N)[:, None] < N - n_start and
                        tl.arange(0, BLOCK_SIZE_K)[None, :] < K - k_start)  # [BLOCK_SIZE_N, BLOCK_SIZE_K]

        # Compute matrix multiplication for this block
        acc += tl.dot(a, b.T)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # -----------------------------------------------------------
    # Write output
    m_mask = tl.arange(0, BLOCK_SIZE_M)[:, None] < M - m_start
    n_mask = tl.arange(0, BLOCK_SIZE_N)[None, :] < N - n_start
    mask = m_mask & n_mask
    tl.store(c_block_ptr, acc, mask=mask)

# Triton implementation for weight gradient (backward pass)
@triton.jit
def _linear_backward_weight_kernel(
    # Pointers to matrices
    grad_output_ptr, x_ptr, grad_weight_ptr,
    # Matrix dimensions
    B, O, I,  # Batch, Output features, Input features
    # The stride variables represent how much to increase the ptr
    stride_gb, stride_go,
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute weight gradient: grad_weight = grad_output.T @ x
    grad_output is of shape (B, O) and x is of shape (B, I)
    """
    # Map program id to weight gradient matrix coordinates (o, i)
    pid = tl.program_id(0)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_in_group = GROUP_SIZE_O * num_pid_i
    group_id = pid // num_pid_in_group
    first_pid_o = group_id * GROUP_SIZE_O
    group_size_o = min(num_pid_o - first_pid_o, GROUP_SIZE_O)
    pid_o = first_pid_o + (pid % group_size_o)
    pid_i = (pid % num_pid_in_group) // group_size_o

    # Compute tile indices
    o_start = pid_o * BLOCK_SIZE_O
    i_start = pid_i * BLOCK_SIZE_I

    # Create block pointers
    grad_output_block_ptr = grad_output_ptr + o_start * stride_go  # grad_output block pointer
    x_block_ptr = x_ptr + i_start * stride_xi  # x block pointer
    grad_weight_block_ptr = grad_weight_ptr + o_start * stride_wo + i_start * stride_wi  # grad_weight block pointer

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    # Iterate to compute a block of the grad_weight matrix
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        b_start = b * BLOCK_SIZE_B

        # Load the next block of grad_output and x
        grad_output = tl.load(grad_output_block_ptr + b_start * stride_gb,
                             mask=tl.arange(0, BLOCK_SIZE_B)[:, None] < B - b_start and
                                  tl.arange(0, BLOCK_SIZE_O)[None, :] < O - o_start)  # [BLOCK_SIZE_B, BLOCK_SIZE_O]

        x = tl.load(x_block_ptr + b_start * stride_xb,
                   mask=tl.arange(0, BLOCK_SIZE_B)[:, None] < B - b_start and
                        tl.arange(0, BLOCK_SIZE_I)[None, :] < I - i_start)  # [BLOCK_SIZE_B, BLOCK_SIZE_I]

        # Compute matrix multiplication for this block
        acc += tl.dot(grad_output.T, x)  # [BLOCK_SIZE_O, BLOCK_SIZE_I]

    # Write output
    o_mask = tl.arange(0, BLOCK_SIZE_O)[:, None] < O - o_start
    i_mask = tl.arange(0, BLOCK_SIZE_I)[None, :] < I - i_start
    mask = o_mask & i_mask
    tl.store(grad_weight_block_ptr, acc, mask=mask)

# Wrapper functions
def triton_linear_forward(x, weight):
    """Compute linear forward pass using Triton kernel."""
    batch_size, in_features = x.shape  # x: [batch_size, in_features]
    out_features, _ = weight.shape  # weight: [out_features, in_features]

    # Allocate output
    y = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)  # y: [batch_size, out_features]

    # Configure meta-parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Set grid size
    grid = lambda meta: (
        triton.cdiv(batch_size, meta['BLOCK_SIZE_M']) *
        triton.cdiv(out_features, meta['BLOCK_SIZE_N']),
    )

    # Launch kernel
    _linear_forward_kernel[grid](
        x, weight, y,
        batch_size, out_features, in_features,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return y

def triton_linear_backward(grad_output, x, weight):
    """Compute linear backward pass using Triton kernels."""
    batch_size, out_features = grad_output.shape  # grad_output: [batch_size, out_features]
    _, in_features = x.shape  # x: [batch_size, in_features]

    # Allocate output for gradients
    grad_x = torch.empty((batch_size, in_features), device=grad_output.device, dtype=grad_output.dtype)  # grad_x: [batch_size, in_features]
    grad_weight = torch.empty((out_features, in_features), device=grad_output.device, dtype=grad_output.dtype)  # grad_weight: [out_features, in_features]

    # For grad_x computation, we reuse forward kernel (it's mathematically the same operation)
    _linear_forward_kernel[(triton.cdiv(batch_size, 128) * triton.cdiv(in_features, 128),)](
        grad_output, weight.T, grad_x,
        batch_size, in_features, out_features,
        grad_output.stride(0), grad_output.stride(1),
        weight.stride(0), weight.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )

    # For grad_weight computation
    BLOCK_SIZE_B = 32
    BLOCK_SIZE_O = 128
    BLOCK_SIZE_I = 128
    GROUP_SIZE_O = 8

    # Set grid size for grad_weight
    grid = lambda meta: (
        triton.cdiv(out_features, meta['BLOCK_SIZE_O']) *
        triton.cdiv(in_features, meta['BLOCK_SIZE_I']),
    )

    # Launch grad_weight kernel
    _linear_backward_weight_kernel[grid](
        grad_output, x, grad_weight,
        batch_size, out_features, in_features,
        grad_output.stride(0), grad_output.stride(1),
        x.stride(0), x.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
        BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_O=BLOCK_SIZE_O, BLOCK_SIZE_I=BLOCK_SIZE_I,
        GROUP_SIZE_O=GROUP_SIZE_O,
    )

    return grad_x, grad_weight

# Export for benchmarking
generated_triton_forward = triton_linear_forward
generated_torch_forward = TorchLinear.forward
generated_triton_backward = triton_linear_backward
generated_torch_backward = TorchLinear.backward
