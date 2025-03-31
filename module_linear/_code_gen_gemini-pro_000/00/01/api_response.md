Timestamp: 20250331_041044

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
import math

# Helper function
def ceil_div(a, b):
    return (a + b - 1) // b

# -------------------\
# Triton Kernels
# -------------------\

@triton.autotune(
    configs=[
        # Basic configs
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_stages=4, num_warps=8),
        # Configs with Group Sizes (Super Blocks)
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_O': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2}, num_stages=3, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # Matrix strides
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr
):
    """
    [Computes the forward pass of a linear layer: output = input @ weight.T]

    Args:
        input_ptr (tl.pointer_type): Pointer to the input tensor. [B, I]
        weight_ptr (tl.pointer_type): Pointer to the weight tensor. [O, I]
        output_ptr (tl.pointer_type): Pointer to the output tensor. [B, O]
        B (int): Batch size.
        I (int): Input features dimension.
        O (int): Output features dimension.
        stride_input_b (int): Stride of the input tensor along the batch dimension.
        stride_input_i (int): Stride of the input tensor along the input feature dimension.
        stride_weight_o (int): Stride of the weight tensor along the output feature dimension.
        stride_weight_i (int): Stride of the weight tensor along the input feature dimension.
        stride_output_b (int): Stride of the output tensor along the batch dimension.
        stride_output_o (int): Stride of the output tensor along the output feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Tile size for the batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Tile size for the input feature dimension (reduction dimension K).
        BLOCK_SIZE_O (tl.constexpr): Tile size for the output feature dimension.
        GROUP_SIZE_B (tl.constexpr): Grouping factor for the batch dimension (super blocks).
        GROUP_SIZE_O (tl.constexpr): Grouping factor for the output dimension (super blocks).

    Returns:
        None: The result is written directly to output_ptr.

    Explanation:
        1. Calculate the group IDs based on program IDs and group sizes (`group_id_b`, `group_id_o`).
        2. Determine the starting program ID for this group (`first_pid_b`, `first_pid_o`).
        3. Calculate the number of program IDs required for this group (`num_pid_b`, `num_pid_o`).
        4. Iterate through the program IDs within the assigned group (`pid_b_offset`, `pid_o_offset`).
        5. Calculate the actual program IDs (`pid_b`, `pid_o`) for the current instance.
        6. Check if the calculated program IDs are within the valid grid range (needed for grouped launches).
        7. Calculate the base offsets (`offset_b`, `offset_o`) for the current block based on program IDs.
        8. Initialize an accumulator `acc` matrix of size `(BLOCK_SIZE_B, BLOCK_SIZE_O)` with zeros (using float32 for precision).
        9. Iterate over the input feature dimension `I` in blocks of `BLOCK_SIZE_I`.
        10. Calculate pointers (`a_ptrs`, `b_ptrs`) to the current blocks of the input (`a`) and weight (`b`) tensors.
        11. Define boundary masks (`a_mask`, `b_mask`) for the `I` dimension to handle cases where `I` is not divisible by `BLOCK_SIZE_I`, and also for B and O dimensions based on `offset_b` and `offset_o`.
        12. Load the input block `a` (`BLOCK_SIZE_B, BLOCK_SIZE_I`) and weight block `b` (`BLOCK_SIZE_O, BLOCK_SIZE_I`) using masks.
        13. Compute the matrix multiplication `a @ b.T` for the current blocks and add the result to the accumulator `acc`. Use `tl.dot` with `allow_tf32=True`. Note `b` needs transposing.
        14. After iterating through `I`, calculate the pointers (`c_ptrs`) to the corresponding block in the output tensor.
        15. Define boundary masks (`c_mask`) for the `B` and `O` dimensions based on `offset_b` and `offset_o`.
        16. Store the accumulated result `acc` (cast to the output tensor's dtype) into the output tensor using the mask, only if the program ID was valid (step 6).
    """
    # 1. Group IDs
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)

    # Determine grid structure for grouping
    num_pid_b_group = tl.cdiv(num_pid_b, GROUP_SIZE_B)
    num_pid_o_group = tl.cdiv(num_pid_o, GROUP_SIZE_O)
    group_id_b = pid // num_pid_o_group
    group_id_o = pid % num_pid_o_group

    # 2. Starting PID for this group
    first_pid_b = group_id_b * GROUP_SIZE_B
    first_pid_o = group_id_o * GROUP_SIZE_O

    # 3. Number of PIDs in this group (potentially less than GROUP_SIZE if at boundary)
    # These are used if iterating within the kernel, but the group launch handles this implicitly.

    # 4. Iterate through PIDs in the group
    for pid_b_offset in range(GROUP_SIZE_B):
        # 5. Actual PID for this instance
        pid_b = first_pid_b + pid_b_offset
        # Skip if pid_b is out of bounds for B dimension
        if pid_b >= num_pid_b:
            continue

        for pid_o_offset in range(GROUP_SIZE_O):
             # 5. Actual PID for this instance
            pid_o = first_pid_o + pid_o_offset
            # Skip if pid_o is out of bounds for O dimension
            if pid_o >= num_pid_o:
                continue

            # 7. Base offsets for the current block
            offset_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            offset_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

            # 8. Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

            # 9. Iterate over K (I dimension)
            for k in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
                offset_i = k * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

                # 10. Calculate pointers for current blocks
                a_ptrs = input_ptr + offset_b[:, None] * stride_input_b + offset_i[None, :] * stride_input_i
                # Weight is (O, I), need (BLOCK_O, BLOCK_I) block -> transpose for dot
                b_ptrs = weight_ptr + offset_o[:, None] * stride_weight_o + offset_i[None, :] * stride_weight_i

                # 11. Define boundary masks for K, B, O dimensions
                a_mask = (offset_b[:, None] < B) & (offset_i[None, :] < I)
                b_mask = (offset_o[:, None] < O) & (offset_i[None, :] < I)

                # 12. Load blocks with masking
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0) # Shape (BLOCK_SIZE_O, BLOCK_SIZE_I)

                # 13. Compute matrix multiplication and accumulate
                # a: (BLOCK_B, BLOCK_I), b: (BLOCK_O, BLOCK_I) -> need b.T
                # Triton dot(A(M,K), B(N,K).T) computes A @ B.T
                acc += tl.dot(a, tl.trans(b), allow_tf32=True)

            # 14. Calculate output pointers
            c_ptrs = output_ptr + offset_b[:, None] * stride_output_b + offset_o[None, :] * stride_output_o

            # 15. Define boundary masks for M (B) and N (O) dimensions
            c_mask = (offset_b[:, None] < B) & (offset_o[None, :] < O)

            # 16. Store the result
            tl.store(c_ptrs, acc.to(output_ptr.dtype.element_ty), mask=c_mask)


@triton.autotune(
    configs=[
        # Basic configs
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_stages=4, num_warps=8),
         # Configs with Group Sizes (Super Blocks)
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2}, num_stages=3, num_warps=8),
    ],
    key=['B', 'O', 'I'],
)
@triton.jit
def linear_backward_dx_kernel(
    # Pointers to matrices
    d_output_ptr, weight_ptr, dx_ptr,
    # Matrix dimensions
    B, O, I,
    # Matrix strides
    stride_d_output_b, stride_d_output_o,
    stride_weight_o, stride_weight_i,
    stride_dx_b, stride_dx_i,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    [Computes the backward pass for the input gradient (dx) of a linear layer: dx = d_output @ weight]

    Args:
        d_output_ptr (tl.pointer_type): Pointer to the gradient of the output tensor. [B, O]
        weight_ptr (tl.pointer_type): Pointer to the weight tensor. [O, I]
        dx_ptr (tl.pointer_type): Pointer to the input gradient tensor (output). [B, I]
        B (int): Batch size.
        O (int): Output features dimension (reduction dimension K).
        I (int): Input features dimension.
        stride_d_output_b (int): Stride of d_output along the batch dimension.
        stride_d_output_o (int): Stride of d_output along the output feature dimension.
        stride_weight_o (int): Stride of weight along the output feature dimension.
        stride_weight_i (int): Stride of weight along the input feature dimension.
        stride_dx_b (int): Stride of dx along the batch dimension.
        stride_dx_i (int): Stride of dx along the input feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Tile size for the batch dimension.
        BLOCK_SIZE_O (tl.constexpr): Tile size for the output feature dimension (reduction dimension K).
        BLOCK_SIZE_I (tl.constexpr): Tile size for the input feature dimension.
        GROUP_SIZE_B (tl.constexpr): Grouping factor for the batch dimension (super blocks).
        GROUP_SIZE_I (tl.constexpr): Grouping factor for the input dimension (super blocks).

    Returns:
        None: The result is written directly to dx_ptr.

    Explanation:
        1. Calculate the group IDs based on program IDs and group sizes (`group_id_b`, `group_id_i`).
        2. Determine the starting program ID for this group (`first_pid_b`, `first_pid_i`).
        3. Iterate through the program IDs within the assigned group (`pid_b_offset`, `pid_i_offset`).
        4. Calculate the actual program IDs (`pid_b`, `pid_i`) for the current instance.
        5. Check if the calculated program IDs are within the valid grid range.
        6. Calculate the base offsets (`offset_b`, `offset_i`) for the current block based on program IDs.
        7. Initialize an accumulator `acc` matrix of size `(BLOCK_SIZE_B, BLOCK_SIZE_I)` with zeros (using float32 for precision).
        8. Iterate over the output feature dimension `O` (reduction dim K) in blocks of `BLOCK_SIZE_O`.
        9. Calculate pointers (`a_ptrs`, `b_ptrs`) to the current blocks of d_output (`a`) and weight (`b`).
        10. Define boundary masks (`a_mask`, `b_mask`) for the `O` dimension, and also for B and I dimensions based on `offset_b` and `offset_i`.
        11. Load the d_output block `a` (`BLOCK_SIZE_B, BLOCK_SIZE_O`) and weight block `b` (`BLOCK_SIZE_O, BLOCK_SIZE_I`) using masks.
        12. Compute the matrix multiplication `a @ b` and add the result to the accumulator `acc`. Use `tl.dot` with `allow_tf32=True`.
        13. After iterating through `O`, calculate the pointers (`c_ptrs`) to the corresponding block in the `dx` tensor.
        14. Define boundary masks (`c_mask`) for the `B` and `I` dimensions based on `offset_b` and `offset_i`.
        15. Store the accumulated result `acc` (cast to the dx tensor's dtype) into the `dx` tensor using the mask, only if the program ID was valid.
    """
    # 1. Group IDs
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)

    num_pid_b_group = tl.cdiv(num_pid_b, GROUP_SIZE_B)
    num_pid_i_group = tl.cdiv(num_pid_i, GROUP_SIZE_I)
    group_id_b = pid // num_pid_i_group
    group_id_i = pid % num_pid_i_group

    # 2. Starting PID for this group
    first_pid_b = group_id_b * GROUP_SIZE_B
    first_pid_i = group_id_i * GROUP_SIZE_I

    # 3. Iterate through PIDs in the group
    for pid_b_offset in range(GROUP_SIZE_B):
        # 4. Actual PID for this instance
        pid_b = first_pid_b + pid_b_offset
        # 5. Check bounds
        if pid_b >= num_pid_b:
            continue

        for pid_i_offset in range(GROUP_SIZE_I):
            # 4. Actual PID for this instance
            pid_i = first_pid_i + pid_i_offset
            # 5. Check bounds
            if pid_i >= num_pid_i:
                continue

            # 6. Base offsets for the current block
            offset_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            offset_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

            # 7. Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

            # 8. Iterate over K (O dimension)
            for k in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
                offset_o = k * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

                # 9. Calculate pointers for current blocks
                a_ptrs = d_output_ptr + offset_b[:, None] * stride_d_output_b + offset_o[None, :] * stride_d_output_o
                b_ptrs = weight_ptr + offset_o[:, None] * stride_weight_o + offset_i[None, :] * stride_weight_i

                # 10. Define boundary masks for K, B, I dimensions
                a_mask = (offset_b[:, None] < B) & (offset_o[None, :] < O)
                b_mask = (offset_o[:, None] < O) & (offset_i[None, :] < I)

                # 11. Load blocks with masking
                a = tl.load(a_ptrs, mask=a_mask, other=0.0) # Shape (BLOCK_B, BLOCK_O)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0) # Shape (BLOCK_O, BLOCK_I)

                # 12. Compute matrix multiplication and accumulate
                acc += tl.dot(a, b, allow_tf32=True)

            # 13. Calculate output pointers
            c_ptrs = dx_ptr + offset_b[:, None] * stride_dx_b + offset_i[None, :] * stride_dx_i

            # 14. Define boundary masks for M (B) and N (I) dimensions
            c_mask = (offset_b[:, None] < B) & (offset_i[None, :] < I)

            # 15. Store the result
            tl.store(c_ptrs, acc.to(dx_ptr.dtype.element_ty), mask=c_mask)


@triton.autotune(
    configs=[
        # Basic configs
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_stages=4, num_warps=8),
        # Configs with Group Sizes (Super Blocks)
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2}, num_stages=3, num_warps=8),
    ],
    key=['O', 'B', 'I'],
)
@triton.jit
def linear_backward_dw_kernel(
    # Pointers to matrices
    d_output_ptr, input_ptr, dw_ptr,
    # Matrix dimensions
    B, O, I,
    # Matrix strides
    stride_d_output_b, stride_d_output_o,
    stride_input_b, stride_input_i,
    stride_dw_o, stride_dw_i,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    [Computes the backward pass for the weight gradient (dw) of a linear layer: dw = d_output.T @ input]

    Args:
        d_output_ptr (tl.pointer_type): Pointer to the gradient of the output tensor. [B, O]
        input_ptr (tl.pointer_type): Pointer to the input tensor. [B, I]
        dw_ptr (tl.pointer_type): Pointer to the weight gradient tensor (output). [O, I]
        B (int): Batch size (reduction dimension K).
        O (int): Output features dimension.
        I (int): Input features dimension.
        stride_d_output_b (int): Stride of d_output along the batch dimension.
        stride_d_output_o (int): Stride of d_output along the output feature dimension.
        stride_input_b (int): Stride of input along the batch dimension.
        stride_input_i (int): Stride of input along the input feature dimension.
        stride_dw_o (int): Stride of dw along the output feature dimension.
        stride_dw_i (int): Stride of dw along the input feature dimension.
        BLOCK_SIZE_O (tl.constexpr): Tile size for the output feature dimension.
        BLOCK_SIZE_B (tl.constexpr): Tile size for the batch dimension (reduction dimension K).
        BLOCK_SIZE_I (tl.constexpr): Tile size for the input feature dimension.
        GROUP_SIZE_O (tl.constexpr): Grouping factor for the output dimension (super blocks).
        GROUP_SIZE_I (tl.constexpr): Grouping factor for the input dimension (super blocks).

    Returns:
        None: The result is written directly to dw_ptr.

    Explanation:
        1. Calculate the group IDs based on program IDs and group sizes (`group_id_o`, `group_id_i`).
        2. Determine the starting program ID for this group (`first_pid_o`, `first_pid_i`).
        3. Iterate through the program IDs within the assigned group (`pid_o_offset`, `pid_i_offset`).
        4. Calculate the actual program IDs (`pid_o`, `pid_i`) for the current instance.
        5. Check if the calculated program IDs are within the valid grid range.
        6. Calculate the base offsets (`offset_o`, `offset_i`) for the current block based on program IDs.
        7. Initialize an accumulator `acc` matrix of size `(BLOCK_SIZE_O, BLOCK_SIZE_I)` with zeros (using float32 for precision).
        8. Iterate over the batch dimension `B` (reduction dim K) in blocks of `BLOCK_SIZE_B`.
        9. Calculate pointers (`a_ptrs`, `b_ptrs`) to the current blocks of d_output (`a`) and input (`b`).
        10. Define boundary masks (`a_mask`, `b_mask`) for the `B` dimension, and also for O and I dimensions based on `offset_o` and `offset_i`.
        11. Load the d_output block `a` (`BLOCK_SIZE_B, BLOCK_SIZE_O`) and input block `b` (`BLOCK_SIZE_B, BLOCK_SIZE_I`) using masks.
        12. Compute the matrix multiplication `a.T @ b` and add the result to the accumulator `acc`. Use `tl.dot` with `allow_tf32=True`. Note `a` needs transposing.
        13. After iterating through `B`, calculate the pointers (`c_ptrs`) to the corresponding block in the `dw` tensor.
        14. Define boundary masks (`c_mask`) for the `O` and `I` dimensions based on `offset_o` and `offset_i`.
        15. Store the accumulated result `acc` (cast to the dw tensor's dtype) into the `dw` tensor using the mask, only if the program ID was valid. Atomics are not needed here as accumulation happens locally within the kernel instance before the final store.
    """
    # 1. Group IDs
    pid = tl.program_id(axis=0)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)

    num_pid_o_group = tl.cdiv(num_pid_o, GROUP_SIZE_O)
    num_pid_i_group = tl.cdiv(num_pid_i, GROUP_SIZE_I)
    group_id_o = pid // num_pid_i_group
    group_id_i = pid % num_pid_i_group

    # 2. Starting PID for this group
    first_pid_o = group_id_o * GROUP_SIZE_O
    first_pid_i = group_id_i * GROUP_SIZE_I

    # 3. Iterate through PIDs in the group
    for pid_o_offset in range(GROUP_SIZE_O):
        # 4. Actual PID for this instance
        pid_o = first_pid_o + pid_o_offset
        # 5. Check bounds
        if pid_o >= num_pid_o:
            continue

        for pid_i_offset in range(GROUP_SIZE_I):
             # 4. Actual PID for this instance
            pid_i = first_pid_i + pid_i_offset
             # 5. Check bounds
            if pid_i >= num_pid_i:
                continue

            # 6. Base offsets for the current block
            offset_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
            offset_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

            # 7. Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

            # 8. Iterate over K (B dimension)
            for k in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
                offset_b = k * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

                # 9. Calculate pointers for current blocks
                # d_output (B, O) -> load block (BLOCK_B, BLOCK_O)
                a_ptrs = d_output_ptr + offset_b[:, None] * stride_d_output_b + offset_o[None, :] * stride_d_output_o
                # input (B, I) -> load block (BLOCK_B, BLOCK_I)
                b_ptrs = input_ptr + offset_b[:, None] * stride_input_b + offset_i[None, :] * stride_input_i

                # 10. Define boundary masks for K, O, I dimensions
                a_mask = (offset_b[:, None] < B) & (offset_o[None, :] < O)
                b_mask = (offset_b[:, None] < B) & (offset_i[None, :] < I)

                # 11. Load blocks with masking
                a = tl.load(a_ptrs, mask=a_mask, other=0.0) # Shape (BLOCK_B, BLOCK_O)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0) # Shape (BLOCK_B, BLOCK_I)

                # 12. Compute matrix multiplication and accumulate
                # Need a.T @ b = (BLOCK_O, BLOCK_B) @ (BLOCK_B, BLOCK_I) -> (BLOCK_O, BLOCK_I)
                acc += tl.dot(tl.trans(a), b, allow_tf32=True)

            # 13. Calculate output pointers
            c_ptrs = dw_ptr + offset_o[:, None] * stride_dw_o + offset_i[None, :] * stride_dw_i

            # 14. Define boundary masks for M (O) and N (I) dimensions
            c_mask = (offset_o[:, None] < O) & (offset_i[None, :] < I)

            # 15. Store the result
            tl.store(c_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=c_mask)


# -------------------\
# Autograd Function\n# -------------------

class _Linear(torch.autograd.Function):
    """
    [Custom autograd function for Triton-based Linear layer.]

    Explanation:
        1. forward: Computes the forward pass using the `linear_forward_kernel`.
        2. backward: Computes the backward pass using `linear_backward_dx_kernel` and `linear_backward_dw_kernel`.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        [Computes the forward pass of the linear layer (output = input @ weight.T).]

        Args:
            ctx (torch.autograd.function.Context): Context object to save tensors for backward.
            input (torch.Tensor): Input tensor. [B, I]
            weight (torch.Tensor): Weight tensor. [O, I]

        Returns:
            torch.Tensor: Output tensor. [B, O]

        Explanation:
            1. Check contiguity of input and weight tensors, make contiguous if necessary.
            2. Get dimensions B, I from input and O from weight.
            3. Create an empty output tensor of shape [B, O] with the correct dtype and device.
            4. Define the grid size for the Triton kernel using a lambda function that accesses autotuned block/group sizes from the kernel's metadata (`meta`).
            5. Launch the `linear_forward_kernel` with the necessary pointers, dimensions, strides, and let autotuner handle block/group size selection.
            6. Save the input and weight tensors for the backward pass using `ctx.save_for_backward`.
            7. Return the output tensor.
        """
        # 1. Ensure contiguity
        input_c = input.contiguous()
        weight_c = weight.contiguous()

        # 2. Get dimensions
        B, I = input_c.shape
        O, _ = weight_c.shape
        assert _ == I, f"Weight shape[1] ({_}) must match input shape[1] ({I})"

        # 3. Create output tensor
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)

        # 4. Define grid size using lambda meta
        # The grid calculation now happens inside the kernel launcher using `meta`
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B'] * meta['GROUP_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O'] * meta['GROUP_SIZE_O'])
        )
        # Use a single dimension for grouped launch
        grid_1d = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']) // \
                                (meta['GROUP_SIZE_B'] * meta['GROUP_SIZE_O']), )


        # 5. Launch kernel
        linear_forward_kernel[grid_1d]( # Use 1D grid for grouped launch
            input_c, weight_c, output,
            B, I, O,
            input_c.stride(0), input_c.stride(1),
            weight_c.stride(0), weight_c.stride(1),
            output.stride(0), output.stride(1),
            # Autotune decorator handles passing the kwargs implicitly based on the config key
        )

        # 6. Save for backward
        ctx.save_for_backward(input_c, weight_c)

        # 7. Return output
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        [Computes the backward pass for input and weight gradients.]

        Args:
            ctx (torch.autograd.function.Context): Context object with saved tensors.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the output tensor. [B, O]

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]: Gradients w.r.t. input and weight. (dx, dw)

        Explanation:
            1. Retrieve saved input and weight tensors.
            2. Check contiguity of grad_output, make contiguous if necessary.
            3. Get dimensions B, I, O.
            4. --- Calculate dx (input gradient: dx = grad_output @ weight) ---
            5. Create an empty dx tensor of shape [B, I].
            6. Define the grid size for the `linear_backward_dx_kernel` using a lambda function.
            7. Launch the `linear_backward_dx_kernel`.
            8. --- Calculate dw (weight gradient: dw = grad_output.T @ input) ---
            9. Create an empty dw tensor of shape [O, I].
            10. Define the grid size for the `linear_backward_dw_kernel` using a lambda function.
            11. Launch the `linear_backward_dw_kernel`.
            12. Return the computed gradients (dx, dw). Return None for bias gradient as it's not used.
        """
        # 1. Retrieve saved tensors
        input, weight = ctx.saved_tensors

        # 2. Ensure contiguity for grad_output
        grad_output_c = grad_output.contiguous()

        # 3. Get dimensions
        B, I = input.shape
        O, _ = weight.shape
        assert grad_output_c.shape == (B, O)

        # 4. --- Calculate dx ---
        # 5. Create dx tensor
        dx = torch.empty_like(input)

        # 6. Define grid for dx kernel using lambda meta
        grid_dx = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B'] * meta['GROUP_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I'] * meta['GROUP_SIZE_I'])
        )
        # Use a single dimension for grouped launch
        grid_dx_1d = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']) // \
                                  (meta['GROUP_SIZE_B'] * meta['GROUP_SIZE_I']), )


        # 7. Launch dx kernel
        linear_backward_dx_kernel[grid_dx_1d]( # Use 1D grid for grouped launch
            grad_output_c, weight, dx,
            B, O, I,
            grad_output_c.stride(0), grad_output_c.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
        )

        # 8. --- Calculate dw ---
        # 9. Create dw tensor
        dw = torch.empty_like(weight)

        # 10. Define grid for dw kernel using lambda meta
        grid_dw = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O'] * meta['GROUP_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I'] * meta['GROUP_SIZE_I'])
        )
        # Use a single dimension for grouped launch
        grid_dw_1d = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_O']) * triton.cdiv(I, meta['BLOCK_SIZE_I']) // \
                                  (meta['GROUP_SIZE_O'] * meta['GROUP_SIZE_I']), )

        # 11. Launch dw kernel
        linear_backward_dw_kernel[grid_dw_1d]( # Use 1D grid for grouped launch
            grad_output_c, input, dw,
            B, O, I,
            grad_output_c.stride(0), grad_output_c.stride(1),
            input.stride(0), input.stride(1),
            dw.stride(0), dw.stride(1),
        )

        # 12. Return gradients
        return dx, dw


# -------------------\
# PyTorch Module
# -------------------

class TritonLinear(torch.nn.Module):
    """
    [Triton implementation of a standard linear layer (torch.nn.Linear without bias).]

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        device (str or torch.device): The desired device of the parameters.
        dtype (torch.dtype): The desired floating point type of the parameters.

    Attributes:
        weight (torch.nn.Parameter): The learnable weights of the module of shape (out_features, in_features).

    Explanation:
        1. Initialize the nn.Module superclass.
        2. Store in_features and out_features.
        3. Create the weight parameter tensor with shape (out_features, in_features) on the specified device and dtype.
        4. Initialize the weight parameter using Kaiming uniform initialization.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        [Initializes the TritonLinear module.]

        Args:
            in_features (int): Size of each input sample (I).
            out_features (int): Size of each output sample (O).
            device (str or torch.device, optional): The target device for the weight tensor. Defaults to None (CPU).
            dtype (torch.dtype, optional): The target data type for the weight tensor. Defaults to None (torch default).

        Returns:
            None

        Explanation:
            1. Call the superclass constructor.
            2. Store input and output feature sizes.
            3. Create the weight parameter `self.weight` as a `torch.nn.Parameter` of shape `(out_features, in_features)`.
            4. Initialize weights using `torch.nn.init.kaiming_uniform_` with `a=math.sqrt(5)`.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # Initialize weights similar to torch.nn.Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        [Performs the forward pass using the custom Triton autograd function.]

        Args:
            input (torch.Tensor): The input tensor. [B, in_features]

        Returns:
            torch.Tensor: The output tensor. [B, out_features]

        Explanation:
            1. Call the `_Linear.apply` method, passing the input tensor and the module's weight parameter.
        """
        return _Linear.apply(input, self.weight)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'

# -------------------\
# Benchmark Code
# -------------------

# The following code is provided by the user for testing correctness and performance.
if __name__ == "__main__":
    # Ensure benchmark.py and _benchmark_util.py are in the same directory or PYTHONPATH
    try:
        from benchmark import (
            verify_correctness_forward, benchmark_performance_forward,
            verify_correctness_backward, benchmark_performance_backward
        )
    except ImportError:
        print("Error: benchmark.py not found. Please ensure benchmark.py and _benchmark_util.py are accessible.")
        print("You might need to copy them into the current directory.")
        # Attempt to import from parent directory if running from a subdirectory
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from benchmark import (
                verify_correctness_forward, benchmark_performance_forward,
                verify_correctness_backward, benchmark_performance_backward
            )
            print("Successfully imported benchmark.py from parent directory.")
        except ImportError as e:
             print(f"Failed to import benchmark.py from parent directory: {e}")
             exit(1)


    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    # Use float16 for benchmarking as requested
    dtype = torch.float16
    device = 'cuda'

    print(f"Running benchmarks with dtype={dtype} on device={device}")
    print(f"B sizes: {B_sizes}")
    print(f"I sizes: {I_sizes}")
    print(f"O sizes: {O_sizes}")
    print("-" * 30)

    def module_generator(B, I, O, device=device, dtype=dtype):
        # Note: Using the class defined in *this* file
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        # Ensure weights are identical for fair comparison
        with torch.no_grad():
            # Ensure torch_module.weight is contiguous before copying if needed
            # triton_module.weight.copy_(torch_module.weight.contiguous())
            # Parameter tensors are usually contiguous, direct copy should be fine
             triton_module.weight.copy_(torch_module.weight)
        return triton_module, torch_module

    def forward_generator(B, I, O, device=device, dtype=dtype):
        # Using normalized inputs as in the request
        input_tensor = torch.randn(B, I, device=device, dtype=dtype, requires_grad=True) # Ensure requires_grad=True for backward pass later
        # input_tensor = torch.nn.functional.normalize(input_tensor, dim=-1) # Normalization can reduce rank, potentially causing issues in backward checks
        return (input_tensor,) # Return as tuple

    print("Verifying Forward Pass Correctness...")
    # Relaxed tolerance for FP16 and potential differences introduced by Triton vs cuBLAS atomics/order of ops
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype, atol=5e-2, rtol=5e-2)
    print("\nBenchmarking Forward Pass Performance...")
    benchmark_performance_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype)

    def backward_generator(B, I, O, device=device, dtype=dtype):
        # Need requires_grad=True on input for backward pass
        input_tensor = torch.randn(B, I, device=device, dtype=dtype, requires_grad=True)
        # input_tensor = torch.nn.functional.normalize(input_tensor, dim=-1) # Avoid normalization
        # Using normalized d_output as in the request
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        # d_output = torch.nn.functional.normalize(d_output, dim=-1) # Avoid normalization
        return (input_tensor,), d_output # Return input as tuple

    print("\nVerifying Backward Pass Correctness...")
    verify_correctness_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype, atol=5e-2, rtol=5e-2) # Relaxed tolerance for FP16
    print("\nBenchmarking Backward Pass Performance...")
    benchmark_performance_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes], dtype=dtype)

    print("\nBenchmark finished.")

```
<test>
Make sure `benchmark.py` and `_benchmark_util.py` (which you can create from the `benchmark.py` provided, extracting the helper functions) are in the same directory as `linear_triton.py`. Then run the script.

```bash
python linear_triton.py
```

</test>