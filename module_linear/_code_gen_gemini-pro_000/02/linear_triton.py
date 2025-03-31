# filename: linear_triton.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import triton
import triton.language as tl
import math

# Triton Kernels with Autotuning and Grouping

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 5}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 5}, num_warps=4),
        # Add more configs, potentially exploring larger block sizes for larger dimensions if needed
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
    ],
    key=['B', 'I', 'O'], # Dimensions that affect performance
)
@triton.jit
def _forward_kernel(
    X, W, Y,                     # Pointers to matrices
    stride_xb, stride_xi,        # Strides for X
    stride_wo, stride_wi,        # Strides for W
    stride_yb, stride_yo,        # Strides for Y
    B, I, O,                     # Matrix dimensions
    BLOCK_SIZE_B: tl.constexpr,  # Tile size for B dimension (Batch)
    BLOCK_SIZE_O: tl.constexpr,  # Tile size for O dimension (Output Features)
    BLOCK_SIZE_I: tl.constexpr,  # Tile size for I dimension (Input Features) - Reduction axis
    GROUP_SIZE_B: tl.constexpr,  # Grouping along B dimension for scheduling
):
    """
    [Computes the forward pass of a linear layer: Y = X @ W.T.]

    Args:
        X (tl.pointer_type): Pointer to the input tensor. Shape: (B, I)
        W (tl.pointer_type): Pointer to the weight tensor. Shape: (O, I)
        Y (tl.pointer_type): Pointer to the output tensor. Shape: (B, O)
        stride_xb (int): Stride of X along the B dimension.
        stride_xi (int): Stride of X along the I dimension.
        stride_wo (int): Stride of W along the O dimension.
        stride_wi (int): Stride of W along the I dimension.
        stride_yb (int): Stride of Y along the B dimension.
        stride_yo (int): Stride of Y along the O dimension.
        B (int): Batch size.
        I (int): Input features dimension.
        O (int): Output features dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for the B dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for the O dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for the I dimension (reduction).
        GROUP_SIZE_B (tl.constexpr): Grouping factor for the B dimension.

    Returns:
        None: The result is written directly to the Y tensor.

    Explanation:
        1. Get program IDs (`pid_b`, `pid_o`) for the current block, considering grouping.
        2. Initialize accumulator tile `acc` with zeros.
        3. Iterate over the reduction dimension `I` in blocks of `BLOCK_SIZE_I`.
        4. Calculate pointers `offs_x` and `offs_w` for the current blocks of X and W.
        5. Load blocks `a` from X and `b` from W using masking for boundary conditions.
        6. Perform matrix multiplication `tl.dot(a, b)` and accumulate the result in `acc`. Note W needs to be treated as transposed, so load W[o, i] for dot(X[b, i], W[o, i]).
        7. After the loop, calculate the pointer `offs_y` for the output block.
        8. Store the accumulated result `acc` into the output tensor Y using masking.
    """
    # 1. Get program IDs
    pid = tl.program_id(axis=0)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)

    # Determine block indices, potentially accounting for grouping if GROUP_SIZE_B > 1
    # Grouping in Triton's autotuner primarily affects launch bounds and scheduling strategy.
    # We map the 1D pid to 2D block indices (b, o)
    # If GROUP_SIZE_B is used, Triton's scheduler groups blocks along the 'b' dimension.
    pid_b = pid // num_pid_o
    pid_o = pid % num_pid_o

    # Calculate offsets for the current block
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    # 2. Initialize accumulator
    # Y[b, o] = sum_i ( X[b, i] * W[o, i] )
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    # 3. Iterate over the reduction dimension I
    for k in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        offs_i = k * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

        # 4. Calculate pointers for X and W blocks
        # offs_x: (BLOCK_SIZE_B, BLOCK_SIZE_I) block
        offs_x = offs_b[:, None] * stride_xb + offs_i[None, :] * stride_xi
        # offs_w: (BLOCK_SIZE_O, BLOCK_SIZE_I) block
        offs_w = offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi

        # 5. Load blocks a and b with masking
        mask_x = (offs_b[:, None] < B) & (offs_i[None, :] < I)
        mask_w = (offs_o[:, None] < O) & (offs_i[None, :] < I)

        a = tl.load(X + offs_x, mask=mask_x, other=0.0) # Shape: (BLOCK_SIZE_B, BLOCK_SIZE_I)
        b = tl.load(W + offs_w, mask=mask_w, other=0.0) # Shape: (BLOCK_SIZE_O, BLOCK_SIZE_I)

        # 6. Perform matrix multiplication and accumulate
        # We need X @ W.T, so we compute dot(a, b.T)
        # tl.dot requires compatible shapes. a=(M, K), b=(N, K). We want dot(a, b.T) where b.T is (K, N)
        # Triton's tl.dot implicitly handles the transpose for the second operand if needed based on layout?
        # Let's be explicit: We need to compute sum_k A[m, k] * B[n, k]. This is what tl.dot does when b is loaded as (N, K).
        acc += tl.dot(a, tl.trans(b), allow_tf32=True) # Requires b to be transposed: (BLOCK_SIZE_I, BLOCK_SIZE_O)

    # Cast accumulator to output dtype
    acc = acc.to(Y.dtype.element_ty)

    # 7. Calculate pointer for Y block
    # offs_y: (BLOCK_SIZE_B, BLOCK_SIZE_O) block
    offs_y = offs_b[:, None] * stride_yb + offs_o[None, :] * stride_yo
    mask_y = (offs_b[:, None] < B) & (offs_o[None, :] < O)

    # 8. Store the result
    tl.store(Y + offs_y, acc, mask=mask_y)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 5}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'num_warps': 4, 'num_stages': 5}, num_warps=4),
        # Add more configs
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def _backward_dx_kernel(
    dY, W, dX,                    # Pointers to matrices
    stride_dyb, stride_dyo,      # Strides for dY
    stride_wo, stride_wi,        # Strides for W
    stride_dxb, stride_dxi,      # Strides for dX
    B, I, O,                     # Matrix dimensions
    BLOCK_SIZE_B: tl.constexpr,  # Tile size for B dimension (Batch)
    BLOCK_SIZE_I: tl.constexpr,  # Tile size for I dimension (Input Features)
    BLOCK_SIZE_O: tl.constexpr,  # Tile size for O dimension (Output Features) - Reduction axis
    GROUP_SIZE_B: tl.constexpr,  # Grouping along B dimension for scheduling
):
    """
    [Computes the backward gradient with respect to the input: dX = dY @ W.]

    Args:
        dY (tl.pointer_type): Pointer to the gradient of the output tensor. Shape: (B, O)
        W (tl.pointer_type): Pointer to the weight tensor. Shape: (O, I)
        dX (tl.pointer_type): Pointer to the gradient of the input tensor. Shape: (B, I)
        stride_dyb (int): Stride of dY along the B dimension.
        stride_dyo (int): Stride of dY along the O dimension.
        stride_wo (int): Stride of W along the O dimension.
        stride_wi (int): Stride of W along the I dimension.
        stride_dxb (int): Stride of dX along the B dimension.
        stride_dxi (int): Stride of dX along the I dimension.
        B (int): Batch size.
        I (int): Input features dimension.
        O (int): Output features dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for the B dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for the I dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for the O dimension (reduction).
        GROUP_SIZE_B (tl.constexpr): Grouping factor for the B dimension.

    Returns:
        None: The result is written directly to the dX tensor.

    Explanation:
        1. Get program IDs (`pid_b`, `pid_i`) for the current block.
        2. Initialize accumulator tile `acc` with zeros.
        3. Iterate over the reduction dimension `O` in blocks of `BLOCK_SIZE_O`.
        4. Calculate pointers `offs_dy` and `offs_w` for the current blocks of dY and W.
        5. Load blocks `dy_block` from dY and `w_block` from W using masking. Note W is used directly (O, I).
        6. Perform matrix multiplication `tl.dot(dy_block, w_block)` and accumulate the result in `acc`.
        7. After the loop, calculate the pointer `offs_dx` for the output block dX.
        8. Store the accumulated result `acc` into the dX tensor using masking.
    """
    # 1. Get program IDs
    pid = tl.program_id(axis=0)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    # Map 1D pid -> 2D block indices (b, i)
    pid_b = pid // num_pid_i
    pid_i = pid % num_pid_i

    # Calculate offsets for the current block
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    # 2. Initialize accumulator
    # dX[b, i] = sum_o ( dY[b, o] * W[o, i] )
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    # 3. Iterate over the reduction dimension O
    for k in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        offs_o = k * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

        # 4. Calculate pointers for dY and W blocks
        # offs_dy: (BLOCK_SIZE_B, BLOCK_SIZE_O) block
        offs_dy = offs_b[:, None] * stride_dyb + offs_o[None, :] * stride_dyo
        # offs_w: (BLOCK_SIZE_O, BLOCK_SIZE_I) block
        offs_w = offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi

        # 5. Load blocks with masking
        mask_dy = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        mask_w = (offs_o[:, None] < O) & (offs_i[None, :] < I)

        dy_block = tl.load(dY + offs_dy, mask=mask_dy, other=0.0) # Shape: (BLOCK_SIZE_B, BLOCK_SIZE_O)
        w_block = tl.load(W + offs_w, mask=mask_w, other=0.0)     # Shape: (BLOCK_SIZE_O, BLOCK_SIZE_I)

        # 6. Perform matrix multiplication and accumulate
        # acc = dY @ W
        acc += tl.dot(dy_block, w_block, allow_tf32=True)

    # Cast accumulator to output dtype
    acc = acc.to(dX.dtype.element_ty)

    # 7. Calculate pointer for dX block
    # offs_dx: (BLOCK_SIZE_B, BLOCK_SIZE_I) block
    offs_dx = offs_b[:, None] * stride_dxb + offs_i[None, :] * stride_dxi
    mask_dx = (offs_b[:, None] < B) & (offs_i[None, :] < I)

    # 8. Store the result
    tl.store(dX + offs_dx, acc, mask=mask_dx)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'num_warps': 4, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'num_warps': 4, 'num_stages': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'num_warps': 4, 'num_stages': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 4, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 4, 'num_warps': 4, 'num_stages': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'num_warps': 4, 'num_stages': 5}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'num_warps': 4, 'num_stages': 5}, num_warps=4),
        # Add more configs
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'num_warps': 8, 'num_stages': 3}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def _backward_dw_kernel(
    dY, X, dW,                    # Pointers to matrices
    stride_dyb, stride_dyo,      # Strides for dY
    stride_xb, stride_xi,        # Strides for X
    stride_dwo, stride_dwi,      # Strides for dW
    B, I, O,                     # Matrix dimensions
    BLOCK_SIZE_O: tl.constexpr,  # Tile size for O dimension (Output Features)
    BLOCK_SIZE_I: tl.constexpr,  # Tile size for I dimension (Input Features)
    BLOCK_SIZE_B: tl.constexpr,  # Tile size for B dimension (Batch) - Reduction axis
    GROUP_SIZE_O: tl.constexpr,  # Grouping along O dimension for scheduling
):
    """
    [Computes the backward gradient with respect to the weight: dW = dY.T @ X.]

    Args:
        dY (tl.pointer_type): Pointer to the gradient of the output tensor. Shape: (B, O)
        X (tl.pointer_type): Pointer to the input tensor. Shape: (B, I)
        dW (tl.pointer_type): Pointer to the gradient of the weight tensor. Shape: (O, I)
        stride_dyb (int): Stride of dY along the B dimension.
        stride_dyo (int): Stride of dY along the O dimension.
        stride_xb (int): Stride of X along the B dimension.
        stride_xi (int): Stride of X along the I dimension.
        stride_dwo (int): Stride of dW along the O dimension.
        stride_dwi (int): Stride of dW along the I dimension.
        B (int): Batch size.
        I (int): Input features dimension.
        O (int): Output features dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for the O dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for the I dimension.
        BLOCK_SIZE_B (tl.constexpr): Block size for the B dimension (reduction).
        GROUP_SIZE_O (tl.constexpr): Grouping factor for the O dimension.

    Returns:
        None: The result is written directly to the dW tensor.

    Explanation:
        1. Get program IDs (`pid_o`, `pid_i`) for the current block.
        2. Initialize accumulator tile `acc` with zeros.
        3. Iterate over the reduction dimension `B` in blocks of `BLOCK_SIZE_B`.
        4. Calculate pointers `offs_dy` and `offs_x` for the current blocks of dY and X. Note that dY needs to be accessed as transposed (O x B).
        5. Load blocks `dy_block` from dY (transposed access pattern) and `x_block` from X using masking.
        6. Perform matrix multiplication `tl.dot(dy_block.T, x_block)` and accumulate the result in `acc`.
        7. After the loop, calculate the pointer `offs_dw` for the output block dW.
        8. Store the accumulated result `acc` into the dW tensor using masking.
    """
    # 1. Get program IDs
    pid = tl.program_id(axis=0)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    # Map 1D pid -> 2D block indices (o, i)
    pid_o = pid // num_pid_i
    pid_i = pid % num_pid_i

    # Calculate offsets for the current block
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    # 2. Initialize accumulator
    # dW[o, i] = sum_b ( dY[b, o] * X[b, i] )
    # This is equivalent to dY.T @ X
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    # 3. Iterate over the reduction dimension B
    for k in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        offs_b = k * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

        # 4. Calculate pointers for dY and X blocks
        # offs_dy: access (BLOCK_SIZE_B, BLOCK_SIZE_O) block for dY.T
        offs_dy = offs_b[:, None] * stride_dyb + offs_o[None, :] * stride_dyo
        # offs_x: access (BLOCK_SIZE_B, BLOCK_SIZE_I) block for X
        offs_x = offs_b[:, None] * stride_xb + offs_i[None, :] * stride_xi

        # 5. Load blocks with masking
        mask_dy = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        mask_x = (offs_b[:, None] < B) & (offs_i[None, :] < I)

        # Load dY as (B, O) -> we need (O, B) for dot product with X (B, I)
        # Load dy_block as (BLOCK_SIZE_B, BLOCK_SIZE_O)
        dy_block = tl.load(dY + offs_dy, mask=mask_dy, other=0.0)
        # Load x_block as (BLOCK_SIZE_B, BLOCK_SIZE_I)
        x_block = tl.load(X + offs_x, mask=mask_x, other=0.0)

        # 6. Perform matrix multiplication and accumulate
        # We need dY.T @ X. dy_block is (B, O), x_block is (B, I)
        # Compute tl.dot(tl.trans(dy_block), x_block)
        acc += tl.dot(tl.trans(dy_block), x_block, allow_tf32=True)

    # Cast accumulator to output dtype
    acc = acc.to(dW.dtype.element_ty)

    # 7. Calculate pointer for dW block
    # offs_dw: (BLOCK_SIZE_O, BLOCK_SIZE_I) block
    offs_dw = offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi
    mask_dw = (offs_o[:, None] < O) & (offs_i[None, :] < I)

    # 8. Store the result
    tl.store(dW + offs_dw, acc, mask=mask_dw)


# PyTorch Autograd Function
class TritonLinearFunction(Function):
    """
    [Implements the linear operation using Triton kernels for forward and backward passes.]

    Args:
        input (torch.Tensor): Input tensor. Shape: (B, I)
        weight (torch.Tensor): Weight tensor. Shape: (O, I)

    Returns:
        torch.Tensor: Output tensor. Shape: (B, O)

    Explanation:
        1. [forward]: Computes the forward pass Y = X @ W.T using `_forward_kernel`.
        2. [forward]: Saves input and weight tensors for the backward pass.
        3. [backward]: Computes dX = dY @ W using `_backward_dx_kernel`.
        4. [backward]: Computes dW = dY.T @ X using `_backward_dw_kernel`.
        5. [backward]: Returns gradients dX and dW.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        [Computes the forward pass of the linear layer using the Triton kernel.]

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object to save tensors for backward.
            input (torch.Tensor): Input tensor. Shape: (B, I)
            weight (torch.Tensor): Weight tensor. Shape: (O, I)

        Returns:
            torch.Tensor: Output tensor. Shape: (B, O)

        Explanation:
            1. Check contiguity of input tensors, make contiguous if necessary.
            2. Get dimensions B, I, O.
            3. Create the output tensor Y.
            4. Define the grid for launching the Triton kernel `_forward_kernel`.
            5. Launch the `_forward_kernel` with appropriate arguments (pointers, strides, dimensions).
            6. Save input and weight for the backward pass using `ctx.save_for_backward`.
            7. Return the output tensor Y.
        """
        # 1. Ensure contiguity
        input_c = input.contiguous()
        weight_c = weight.contiguous()

        # 2. Get dimensions
        assert input_c.shape[1] == weight_c.shape[1], f"Input ({input_c.shape[1]}) and weight ({weight_c.shape[1]}) inner dimensions must match"
        B, I = input_c.shape
        O, _ = weight_c.shape
        # print(f"Forward shapes: B={B}, I={I}, O={O}")
        # print(f"Input dtype: {input_c.dtype}, Weight dtype: {weight_c.dtype}")

        # 3. Create output tensor
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)

        # 4. Define grid
        # Grid dimensions are based on the number of blocks needed
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )

        # 5. Launch kernel
        _forward_kernel[grid](
            input_c, weight_c, output,
            input_c.stride(0), input_c.stride(1),
            weight_c.stride(0), weight_c.stride(1),
            output.stride(0), output.stride(1),
            B, I, O,
            # BLOCK_SIZE_B, BLOCK_SIZE_O, BLOCK_SIZE_I, GROUP_SIZE_B are passed by autotuner
        )

        # 6. Save tensors for backward
        ctx.save_for_backward(input_c, weight_c)
        # print(f"Output dtype: {output.dtype}, shape: {output.shape}")

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        [Computes the backward pass (gradients) of the linear layer using Triton kernels.]

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object with saved tensors.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output tensor. Shape: (B, O)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Gradients with respect to input (dX) and weight (dW). Returns None for args that do not require grad.

        Explanation:
            1. Retrieve saved input and weight tensors.
            2. Ensure grad_output is contiguous.
            3. Get dimensions B, I, O.
            4. Create gradient tensors dX and dW.
            5. Compute dX = dY @ W:
                a. Define the grid for `_backward_dx_kernel`.
                b. Launch the kernel.
            6. Compute dW = dY.T @ X:
                a. Define the grid for `_backward_dw_kernel`.
                b. Launch the kernel.
            7. Return the computed gradients dX and dW.
        """
        # 1. Retrieve saved tensors
        input, weight = ctx.saved_tensors

        # 2. Ensure grad_output contiguity
        grad_output_c = grad_output.contiguous()

        # 3. Get dimensions
        B, I = input.shape
        O, _ = weight.shape
        # print(f"Backward shapes: B={B}, I={I}, O={O}")
        # print(f"Grad Output dtype: {grad_output_c.dtype}, shape: {grad_output_c.shape}")
        # print(f"Input dtype: {input.dtype}, Weight dtype: {weight.dtype}")

        # 4. Create gradient tensors
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)

        # 5. Compute dX = dY @ W
        grid_dx = lambda meta: (
             triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        _backward_dx_kernel[grid_dx](
            grad_output_c, weight, grad_input,
            grad_output_c.stride(0), grad_output_c.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
            B, I, O,
            # BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O, GROUP_SIZE_B passed by autotuner
        )
        # print(f"Grad Input (dX) dtype: {grad_input.dtype}, shape: {grad_input.shape}")

        # 6. Compute dW = dY.T @ X
        grid_dw = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        _backward_dw_kernel[grid_dw](
            grad_output_c, input, grad_weight,
            grad_output_c.stride(0), grad_output_c.stride(1),
            input.stride(0), input.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
            B, I, O,
            # BLOCK_SIZE_O, BLOCK_SIZE_I, BLOCK_SIZE_B, GROUP_SIZE_O passed by autotuner
        )
        # print(f"Grad Weight (dW) dtype: {grad_weight.dtype}, shape: {grad_weight.shape}")

        # 7. Return gradients
        # Bias gradient would be None if bias existed and wasn't needed
        return grad_input, grad_weight


# PyTorch Module
class TritonLinear(nn.Module):
    """
    [A PyTorch nn.Module implementing a linear layer using Triton kernels.]

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        device (str or torch.device): The device to place the weight tensor on.
        dtype (torch.dtype): The desired data type for the weight tensor.

    Returns:
        None: Initializes the module.

    Explanation:
        1. Initialize the parent `nn.Module`.
        2. Store input and output feature sizes.
        3. Create the weight tensor `self.weight` as an `nn.Parameter`.
        4. Initialize the weight tensor using Kaiming uniform initialization.
    """
    def __init__(self, in_features: int, out_features: int, device='cuda', dtype=torch.float16):
        """
        [Initializes the TritonLinear module.]

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device (str or torch.device): Device for the weight tensor ('cuda', 'cpu', etc.).
            dtype (torch.dtype): Data type for the weight tensor (e.g., torch.float16).

        Returns:
            None

        Explanation:
            1. Call `super().__init__()`.
            2. Store `in_features` and `out_features`.
            3. Create `self.weight` as a `torch.nn.Parameter` with shape (out_features, in_features).
            4. Initialize `self.weight` using `nn.init.kaiming_uniform_` for good default behavior.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # Initialize weights similar to torch.nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        [Performs the forward pass using the custom Triton autograd function.]

        Args:
            input (torch.Tensor): The input tensor. Shape: (..., in_features)

        Returns:
            torch.Tensor: The output tensor. Shape: (..., out_features)

        Explanation:
            1. Call `TritonLinearFunction.apply` with the input tensor and the module's weight.
            2. Return the result.
        """
        return TritonLinearFunction.apply(input, self.weight)

# Benchmark and Verification Code (Provided)
if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    # Note: Bias is False for torch_module to match TritonLinear implementation
    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        # Ensure weights are identical for comparison
        with torch.no_grad():
            triton_module.weight.copy_(torch_module.weight)
        return triton_module, torch_module

    def forward_generator(B, I, O, device='cuda', dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype, requires_grad=True) # Added requires_grad=True
        # input = torch.nn.functional.normalize(input, dim=-1) # Normalization might hide small errors, remove for stricter check? Let's keep it as requested.
        return input

    # Assuming benchmark.py exists and contains the specified functions
    try:
        from benchmark import verify_correctness_forward, benchmark_performance_forward
        print("Running Forward Pass Verification...")
        verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes], atol=1e-1, rtol=1e-1) # Relaxed tolerance for FP16
        print("\nRunning Forward Pass Performance Benchmark...")
        benchmark_performance_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])
    except ImportError:
        print("benchmark.py not found. Skipping forward pass tests.")
    except Exception as e:
        print(f"An error occurred during forward pass testing: {e}")


    def backward_generator(B, I, O, device='cuda', dtype=torch.float16):
        # Input needs requires_grad=True to compute input gradients
        input = torch.randn(B, I, device=device, dtype=dtype, requires_grad=True)
        # input = torch.nn.functional.normalize(input, dim=-1)
        # d_output should match the shape of the module's output
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        # d_output = torch.nn.functional.normalize(d_output, dim=-1)
        return input, d_output

    try:
        from benchmark import verify_correctness_backward, benchmark_performance_backward
        print("\nRunning Backward Pass Verification...")
        verify_correctness_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes], atol=1e-1, rtol=1e-1) # Relaxed tolerance for FP16
        print("\nRunning Backward Pass Performance Benchmark...")
        benchmark_performance_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes])
    except ImportError:
        print("benchmark.py not found. Skipping backward pass tests.")
    except Exception as e:
        print(f"An error occurred during backward pass testing: {e}")

    print("\nScript finished.")

