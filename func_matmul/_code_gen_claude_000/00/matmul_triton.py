# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # Configurations with different GROUP_SIZE_M for autotuning
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how to access a matrix element at position i, j
    stride_am, stride_ak,  
    stride_bk, stride_bn,  
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication C = A @ B.
    
    Args:
        a_ptr (tl.pointer): Pointer to the matrix A. [M, K]
        b_ptr (tl.pointer): Pointer to the matrix B. [K, N]
        c_ptr (tl.pointer): Pointer to the output matrix C. [M, N]
        M (int): Number of rows in A and C.
        N (int): Number of columns in B and C.
        K (int): Number of columns in A and rows in B.
        stride_am (int): Stride for accessing rows of matrix A.
        stride_ak (int): Stride for accessing columns of matrix A.
        stride_bk (int): Stride for accessing rows of matrix B.
        stride_bn (int): Stride for accessing columns of matrix B.
        stride_cm (int): Stride for accessing rows of matrix C.
        stride_cn (int): Stride for accessing columns of matrix C.
        BLOCK_SIZE_M (tl.constexpr): Size of blocks for matrix A's rows.
        BLOCK_SIZE_N (tl.constexpr): Size of blocks for matrix B's columns.
        BLOCK_SIZE_K (tl.constexpr): Size of blocks for shared dimension K.
        GROUP_SIZE_M (tl.constexpr): Size of groups for super-block optimization.
    
    Returns:
        None: Results are written to c_ptr.
    
    Explanation:
        1. Calculate program ID and map it to the block of C this thread should compute.
        2. Create pointers to the specific blocks of A, B, and C matrices.
        3. Initialize accumulators with zeros for the result block.
        4. Iterate through the K dimension in blocks of BLOCK_SIZE_K.
        5. Load blocks of A and B matrices into shared memory.
        6. Compute the matrix multiplication for the current blocks using tl.dot.
        7. Store the accumulated results back to the C matrix.
    """
    # -----------------------------------------------------------
    # Map program ID to the block of C it should compute using super-blocks
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Create pointers to the first blocks of A, B, and C that this program should process
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks to handle boundary conditions
    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N
    
    # Pointers to A, B with offsets
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # -----------------------------------------------------------
    # Initialize accumulator with zeros (in float32 for better precision)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through k dimension in steps of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B into shared memory with boundary masks
        a_mask_k = a_mask & (k + offs_k[None, :] < K)
        b_mask_k = b_mask & (k + offs_k[:, None] < K)
        
        # Load blocks from A and B, using mask to handle boundaries
        a = tl.load(a_ptrs, mask=a_mask_k, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask_k, other=0.0)
        
        # Compute matrix multiplication for this block using tl.dot
        acc += tl.dot(a, b)
        
        # Increment pointers for the next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # -----------------------------------------------------------
    # Store results back to C matrix
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create mask for writing results to handle boundary conditions
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def generated_triton_func(A, B):
    """
    Perform matrix multiplication C = A @ B using Triton kernel.
    
    Args:
        A (torch.Tensor): Input matrix of shape [M, K].
        B (torch.Tensor): Input matrix of shape [K, N].
    
    Returns:
        torch.Tensor: Output matrix of shape [M, N].
    
    Explanation:
        1. Extract the dimensions from input matrices A and B.
        2. Check compatibility of matrix dimensions.
        3. Initialize the output matrix C with the appropriate shape.
        4. Call the Triton kernel with the appropriate grid configuration.
        5. Return the computed result matrix C.
    """
    # Extract dimensions from input matrices
    M, K = A.shape
    K_b, N = B.shape
    
    # Ensure the inner dimensions match
    assert K == K_b, f"Incompatible dimensions for matrix multiplication: A: {A.shape}, B: {B.shape}"
    
    # Initialize output matrix
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Define grid size based on block sizes and matrix dimensions
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Call the Triton kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C

def generated_torch_func(A, B):
    """
    Perform matrix multiplication C = A @ B using PyTorch.
    
    Args:
        A (torch.Tensor): Input matrix of shape [M, K].
        B (torch.Tensor): Input matrix of shape [K, N].
    
    Returns:
        torch.Tensor: Output matrix of shape [M, N].
    
    Explanation:
        1. Use PyTorch's built-in matrix multiplication operator (@).
        2. Return the computed result matrix.
    """
    return A @ B

if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B

    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
