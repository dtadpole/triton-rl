Timestamp: 20250315_191815

<think>
Okay, let's tackle this error. The user provided an error message from Triton complaining about incompatible matrix shapes in the dot product. The error says:

AssertionError: First input shape ([constexpr[64], constexpr[32]]) and second input shape [constexpr[64], constexpr[32]] are not compatible for matmul (second index of first shape (32) must be equal to first index of second shape (64)

So, in the linear_forward_kernel, when they're doing the matrix multiplication with tl.dot(input, weight), the shapes are mismatched. The input is [64, 32] and the weight is [64, 32], but matrix multiplication requires that the number of columns of the first matrix (input's second dimension) matches the number of rows of the second matrix (weight's first dimension). Here, input is [BLOCK_SIZE_B, GROUP_SIZE_I] and weight is [BLOCK_SIZE_O, GROUP_SIZE_I]. So input is (B, I) and weight is (O, I), but for matrix multiplication, we need input (B, I) multiplied by weight (I, O), which would produce (B, O). 

Wait, but the weight matrix in PyTorch's Linear layer is stored as (O, I). So when we do the matrix multiplication, we need to transpose the weight matrix. Normally, in PyTorch, the forward pass is input @ weight.T. 

Looking at the Triton kernel code:

The input is loaded with shape [BLOCK_SIZE_B, GROUP_SIZE_I], and the weight is [BLOCK_SIZE_O, GROUP_SIZE_I]. But when you multiply input (B, I) with weight (O, I), that's not compatible. To make it compatible, the weight should be transposed so that it's (I, O). Therefore, the correct multiplication would be input @ weight.T, which would require the weight to be loaded as (GROUP_SIZE_I, BLOCK_SIZE_O) instead of (BLOCK_SIZE_O, GROUP_SIZE_I). 

So the error is because the weight is not transposed. The solution is to transpose the weight matrix before the dot product. However, in Triton's tl.dot, the dimensions need to be compatible. One way to fix this is to transpose the weight matrix when loading it or adjust the pointers so that the weight is accessed in a transposed manner. 

Looking at the current code in the forward kernel:

weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

This is indexing into the weight matrix as (O, I) because stride_weight_o is the stride for the O dimension. So when loaded, the weight has shape [BLOCK_SIZE_O, GROUP_SIZE_I]. But to perform input @ weight.T, the weight.T would be [GROUP_SIZE_I, BLOCK_SIZE_O]. Therefore, the dot product should be between input (BLOCK_SIZE_B, GROUP_SIZE_I) and weight.T (GROUP_SIZE_I, BLOCK_SIZE_O), which would produce (BLOCK_SIZE_B, BLOCK_SIZE_O) as the accumulator. 

To achieve this, the weight matrix should be transposed. However, transposing in Triton isn't straightforward. Instead, perhaps the weight is loaded in a way that effectively transposes it. Alternatively, we can switch the order of the dot product. 

Wait, the tl.dot function in Triton takes two 2D tensors and performs a matrix multiplication. The function's signature is tl.dot(a, b), which computes a @ b. So, if input is (B, I) and weight is (I, O), then input @ weight gives (B, O). But in the current code, the weight is (O, I), so we need to transpose it. 

Therefore, in the kernel, the weight should be transposed. But how is the weight stored? In the TritonLinear class, the weight is initialized as (out_features, in_features), same as PyTorch's Linear layer. So the weight is stored as (O, I). Therefore, during the forward pass, the correct multiplication is input @ weight.T. 

In the Triton kernel, to compute this, we need to access the weight in a transposed manner. So instead of loading weight as (O, I), we need to load it as (I, O). That would require changing the way the weight pointers are calculated. 

Looking at the code:

input_ptrs: input is of shape (B, I), so offsets are for B and I dimensions. 

weight_ptrs: weights are (O, I), so the code uses offs_o (O dimension) and offs_i (I dimension). 

But when performing input @ weight.T, weight.T is (I, O), so we should be loading the weight matrix in a way that the I and O dimensions are swapped. 

Thus, instead of loading a tile of size (BLOCK_SIZE_O, GROUP_SIZE_I) from the weight matrix (O, I), we need to load a tile of size (GROUP_SIZE_I, BLOCK_SIZE_O) from the weight matrix. 

Therefore, the weight_ptrs should be calculated to load a (GROUP_SIZE_I, BLOCK_SIZE_O) tile. 

Wait, in the current code, the weight_ptrs are calculated as:

weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

This is for each O and I in the tile. So the tile is (BLOCK_SIZE_O, GROUP_SIZE_I). But we need the tile to be (GROUP_SIZE_I, BLOCK_SIZE_O), so that when we transpose it, it becomes (BLOCK_SIZE_O, GROUP_SIZE_I). Wait, no. Let me think again. 

Alternatively, if we swap the dimensions when accessing the weight, perhaps by switching the order of offs_i and offs_o. 

Wait, the original weight is stored as O rows and I columns. So, to get a tile of size (GROUP_SIZE_I, BLOCK_SIZE_O), we need to iterate over I and O in a different way. 

For example, the weight_ptrs should be calculated as:

weight_ptr + offs_i[:, None] * stride_weight_i + offs_o[None, :] * stride_weight_o

But that would require that the stride_weight_i is the stride for the I dimension, which in the case of the weight matrix (O, I), stride_weight_o is the stride for O (i.e., each row in O), and stride_weight_i is the stride within a row. 

So, in the original code, weight_ptrs are calculated as O rows and I columns. But to load a tile of (GROUP_SIZE_I rows and BLOCK_SIZE_O columns), which is effectively a transposed tile, we need to iterate over I (rows) and O (columns). 

This suggests that the weight_ptrs are being set up incorrectly. Instead of using offs_o (O dimension) and offs_i (I dimension), we should use offs_i and offs_o but swap their positions. 

Wait, the weight in memory is stored as O rows, each of I elements. Each row is contiguous. So, to access a block of (BLOCK_SIZE_O, GROUP_SIZE_I), you would move along O (rows) and then within each row, take GROUP_SIZE_I elements. 

But if we want to transpose the weight matrix, we need to access it as I rows and O columns. However, since the weight is stored as O rows, accessing it in a transposed manner would require striding differently. This might not be feasible without actual transposition. 

Alternatively, perhaps the error is due to the tl.dot(input, weight) where input is (BLOCK_SIZE_B, GROUP_SIZE_I) and weight is (BLOCK_SIZE_O, GROUP_SIZE_I), which would require GROUP_SIZE_I (from input's columns) to match BLOCK_SIZE_O (from weight's rows). But in this case, they don't match, leading to the error. 

So the problem is that the input is (B, I) and the weight is (O, I), which can't be multiplied as is. To perform input @ weight.T, the weight needs to be (I, O). Therefore, the kernel needs to access the weight matrix in a transposed manner. 

To fix this, the kernel should load the weight in such a way that each tile is (GROUP_SIZE_I, BLOCK_SIZE_O), so that when you transpose it to (BLOCK_SIZE_O, GROUP_SIZE_I), you can perform input (BLOCK_SIZE_B, GROUP_SIZE_I) @ weight_tile.T (GROUP_SIZE_I, BLOCK_SIZE_O) -> (BLOCK_SIZE_B, BLOCK_SIZE_O). 

Therefore, the weight_ptrs should be calculated to load tiles of (GROUP_SIZE_I, BLOCK_SIZE_O). 

So, changing the weight_ptrs to:

weight_ptrs = weight_ptr + offs_i[None, :] * stride_weight_i + offs_o[:, None] * stride_weight_o

Wait, but original stride_weight_i is the stride for the I dimension (which is 1 in a row-major layout). Wait, no. The weight is stored as (O, I), so for a row-major (C-like) layout, each row is O elements of I. Therefore, stride_weight_o would be I (since moving to the next row in O requires skipping I elements), and stride_weight_i is 1 (since moving to the next element in I within the same row O is contiguous). 

Therefore, the original code's weight_ptrs are calculated as:

offs_o (O dimension) * stride_weight_o (which is I) + offs_i (I dimension) * stride_weight_i (1). 

But to access the weight in a transposed way (I, O), we need to calculate pointers for the I dimension first, then the O dimension. However, since the weight is stored as (O, I), accessing (I, O) would require non-contiguous strides, which might not be efficient, but for correctness, we need to adjust the way we load the weight. 

So perhaps the correct way is to load the weight as (GROUP_SIZE_I, BLOCK_SIZE_O), but how? 

Alternatively, perhaps the weight matrix should be transposed before passing it to the kernel. But the TritonLinear module initializes the weight as (O, I), same as PyTorch. So the kernel must handle this. 

Wait, let's re-express the matrix multiplication. The correct operation is input (B, I) @ weight (I, O). But since the weight is stored as (O, I), we need to transpose it. Therefore, the matrix multiplication is input @ weight.T. 

In terms of the kernel's tl.dot, to compute this, we can transpose the weight tile. 

But in the current code, the weight is loaded as (BLOCK_SIZE_O, GROUP_SIZE_I). To transpose it, we would need (GROUP_SIZE_I, BLOCK_SIZE_O). Then, input (BLOCK_SIZE_B, GROUP_SIZE_I) @ weight_tile.T (GROUP_SIZE_I, BLOCK_SIZE_O) would produce (BLOCK_SIZE_B, BLOCK_SIZE_O), which is correct. 

So how to transpose the weight tile. In Triton, you can transpose a 2D tensor using tl.trans. However, the code is using tl.load to load a tile, and the way the pointers are set up determines the shape. 

Therefore, the problem is that the current code is loading the weight tile with shape (BLOCK_SIZE_O, GROUP_SIZE_I), and then trying to multiply it with input (BLOCK_SIZE_B, GROUP_SIZE_I). The matrix multiplication input (B, I) * weight (O, I) is invalid because I (GROUP_SIZE_I) does not match O (BLOCK_SIZE_O). 

To fix this, the weight should be loaded in a way that the tl.dot(input, weight) computes input @ weight.T. Therefore, the weight tile should be (GROUP_SIZE_I, BLOCK_SIZE_O), and then input (BLOCK_SIZE_B, GROUP_SIZE_I) @ weight (GROUP_SIZE_I, BLOCK_SIZE_O) would produce (BLOCK_SIZE_B, BLOCK_SIZE_O). 

But how to load the weight as (GROUP_SIZE_I, BLOCK_SIZE_O)? 

Looking at the current code for weight_ptrs:

weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

This is adding offsets for O and I dimensions. The shape here would be (BLOCK_SIZE_O, GROUP_SIZE_I), because offs_o is of size BLOCK_SIZE_O (from pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) and offs_i is GROUP_SIZE_I. 

To get a tile of (GROUP_SIZE_I, BLOCK_SIZE_O), we need to swap the order of offs_i and offs_o. 

So changing weight_ptrs to:

weight_ptrs = weight_ptr + offs_i[:, None] * stride_weight_i + offs_o[None, :] * stride_weight_o

Wait, no. Because stride_weight_i is 1 (assuming the weight is stored in row-major order as (O, I)), so each O row has I elements, contiguous. Therefore, stride_weight_o (stride between O rows) is I, and stride_weight_i (stride within a row) is 1. 

So, if we have offs_i (GROUP_SIZE_I) and offs_o (BLOCK_SIZE_O), then:

For each element in the weight tile, the pointer would be:

weight_ptr + (o * stride_weight_o) + (i * stride_weight_i)

But if we want to load a tile of (GROUP_SIZE_I, BLOCK_SIZE_O), then for each row in I (GROUP_SIZE_I) and column in O (BLOCK_SIZE_O), but that's not possible because the weight is stored as O rows and I columns. 

Alternatively, perhaps the weight should be accessed in a way that for each element in the tile, we take GROUP_SIZE_I elements along the O dimension and BLOCK_SIZE_O along the I dimension. Wait, no, that doesn't make sense. 

Alternatively, maybe the kernel is incorrectly accessing the weight matrix. The weight is (O, I), and in the forward pass, we need to multiply input (B, I) with weight (I, O), which is the transpose. 

Wait, perhaps the kernel's weight_ptrs are set up incorrectly. The weight's shape is (O, I), so each row is of size I. To get a tile from the weight matrix that can be transposed to (I, O), we need to load tiles that cover multiple rows (O) and columns (I). 

Wait, perhaps the correct approach is to load a tile of size (GROUP_SIZE_I, BLOCK_SIZE_O) from the weight matrix. But since the weight is stored as O rows of I elements, accessing a tile of (GROUP_SIZE_I, BLOCK_SIZE_O) would require selecting a block of rows (O) and columns (I). 

But this is getting complicated. Let's step back. 

The error occurs because the input tile is (64, 32) and the weight tile is (64, 32). The matrix multiplication input * weight would require that the columns of input (32) match the rows of weight (64), which they don't. Hence, the assertion fails. 

To fix this, the weight tile must be transposed so that its shape is (32, 64), allowing input (64,32) * weight (32,64) to produce (64,64). 

But how to load the weight tile in a transposed manner. 

In the current code, the weight_ptrs are calculated with offs_o (size BLOCK_SIZE_O) and offs_i (size GROUP_SIZE_I). So each weight tile is (BLOCK_SIZE_O, GROUP_SIZE_I). To transpose it, we need to have (GROUP_SIZE_I, BLOCK_SIZE_O). 

Therefore, changing the weight_ptrs to use offs_i for the rows and offs_o for the columns. But since the weight is stored as (O, I), the rows are O and columns are I. 

Wait, this is getting me confused. Let's consider the indices:

In the original code, for the weight matrix (O, I), the weight_ptrs are calculated as:

offs_o (BLOCK_SIZE_O elements) along the O dimension (rows), and offs_i (GROUP_SIZE_I elements) along the I dimension (columns). 

This gives a tile of (BLOCK_SIZE_O, GROUP_SIZE_I). 

But to compute input (BLOCK_SIZE_B, GROUP_SIZE_I) * weight.T (GROUP_SIZE_I, BLOCK_SIZE_O), we need the weight tile to be (GROUP_SIZE_I, BLOCK_SIZE_O). 

So the weight tile should be transposed. To do this, perhaps the weight_ptrs should index into the I dimension first, then the O dimension. 

But how? The weight matrix is (O, I), so to get a tile of (GROUP_SIZE_I, BLOCK_SIZE_O), we need to read GROUP_SIZE_I rows (O) and BLOCK_SIZE_O columns (I). Wait, no, because the rows are O and columns are I. 

Wait, that doesn't make sense. If we take GROUP_SIZE_I rows (O) and BLOCK_SIZE_O columns (I), the tile would be (GROUP_SIZE_I, BLOCK_SIZE_O) in the I and O dimensions. But the O dimension is the rows. 

This is getting complicated. Perhaps the correct way is to load the weight in a way that each tile is (GROUP_SIZE_I, BLOCK_SIZE_O) in the I and O dimensions. 

Wait, perhaps the kernel should iterate over the I dimension in groups and the O dimension in blocks, and for each iteration, load a tile of the weight matrix that's (GROUP_SIZE_I, BLOCK_SIZE_O) by aggregating across O and I. 

Alternatively, maybe the weight_ptrs should be calculated such that the tile is (GROUP_SIZE_I, BLOCK_SIZE_O), which would require that each element in the tile is accessed as weight[o][i], but with o varying over BLOCK_SIZE_O and i varying over GROUP_SIZE_I. 

Wait, but the weight matrix is stored as O rows (each of size I), so if we take a group of I's and a block of O's, then the tile would be (BLOCK_SIZE_O, GROUP_SIZE_I), which is the same as the current code. 

So perhaps the error is that the code is trying to multiply input (BLOCK_SIZE_B, GROUP_SIZE_I) with weight (BLOCK_SIZE_O, GROUP_SIZE_I) using tl.dot(input, weight). This would require that GROUP_SIZE_I (input's columns) matches BLOCK_SIZE_O (weight's rows), which isn't the case unless BLOCK_SIZE_O == GROUP_SIZE_I. 

But the autotune configs have BLOCK_SIZE_O and GROUP_SIZE_I as separate parameters, which may not be equal. For example, one of the configs has BLOCK_SIZE_O=64 and GROUP_SIZE_I=32. 

Therefore, when GROUP_SIZE_I is 32 and BLOCK_SIZE_O is 64, the input tile is (B, 32), and the weight tile is (64, 32), leading to a matrix multiplication of (B,32) * (64,32), which is invalid. Hence the assertion error. 

The correct approach is to transpose the weight tile before multiplying. Therefore, the code should compute tl.dot(input, tl.trans(weight)), but Triton's tl.dot doesn't support transposing via arguments. So we need to transpose the weight tile explicitly. 

Alternatively, we can switch the order of the loops and the way the weight is loaded. 

Another approach: the weight matrix is (O, I), so the correct multiplication is input (B, I) @ weight.T (I, O). Therefore, each weight tile should be (GROUP_SIZE_I, BLOCK_SIZE_O), which would allow input tile (BLOCK_SIZE_B, GROUP_SIZE_I) @ weight tile (GROUP_SIZE_I, BLOCK_SIZE_O) to produce (BLOCK_SIZE_B, BLOCK_SIZE_O). 

To achieve this, we need to load the weight matrix as (GROUP_SIZE_I, BLOCK_SIZE_O). How? 

The weight is stored as (O, I). So to load a tile of (GROUP_SIZE_I, BLOCK_SIZE_O), we need to iterate over O in groups of BLOCK_SIZE_O and I in groups of GROUP_SIZE_I. 

Wait, maybe the weight_ptrs are set up incorrectly. Originally, the code uses:

weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

Which for each element in the weight tile, combines offs_o (O dimension) and offs_i (I dimension), leading to a tile of (BLOCK_SIZE_O, GROUP_SIZE_I). 

To get a tile of (GROUP_SIZE_I, BLOCK_SIZE_O), perhaps we should swap the offs_o and offs_i in the weight_ptrs calculation. 

Wait, but that might not be correct. Let's think of the calculation again. 

Each element in the weight matrix is accessed as weight[o][i], where o is the row (O) and i is the column (I). 

The current code's weight_ptrs for a tile are:

For each o in offs_o (BLOCK_SIZE_O elements), and each i in offs_i (GROUP_SIZE_I elements), the pointer is weight_ptr + o * stride_weight_o + i * stride_weight_i. 

This gives a tile of BLOCK_SIZE_O rows (O) and GROUP_SIZE_I columns (I). 

But we need to get a tile of GROUP_SIZE_I rows (I) and BLOCK_SIZE_O columns (O), which is the transpose. 

But since the weight is stored as (O, I), the rows are O, so it's impossible to directly load a tile of I rows and O columns. Therefore, the only way is to load the weight tile as (BLOCK_SIZE_O, GROUP_SIZE_I) and then transpose it in the kernel. 

But Triton's tl.trans can transpose a matrix. So after loading the weight tile as (BLOCK_SIZE_O, GROUP_SIZE_I), we can transpose it to (GROUP_SIZE_I, BLOCK_SIZE_O), and then perform the dot product with input (BLOCK_SIZE_B, GROUP_SIZE_I) @ weight_tile_transposed (GROUP_SIZE_I, BLOCK_SIZE_O), which would give the correct (BLOCK_SIZE_B, BLOCK_SIZE_O) result. 

Therefore, modifying the code in the kernel's loop:

After loading input and weight:

input = tl.load(...)
weight = tl.load(...)
weight_t = tl.trans(weight)
acc += tl.dot(input, weight_t, allow_tf32=True)

But does Triton's tl.trans work for arbitrary 2D tensors? Or does the tile need to be square? 

Looking up Triton's documentation, the tl.trans function can transpose a 2D tensor. So applying tl.trans to the weight tile (BLOCK_SIZE_O, GROUP_SIZE_I) would give a (GROUP_SIZE_I, BLOCK_SIZE_O) tensor, which can then be multiplied by the input (BLOCK_SIZE_B, GROUP_SIZE_I) to produce (BLOCK_SIZE_B, BLOCK_SIZE_O). 

Therefore, the fix is to transpose the weight tile before performing the dot product. 

In the original code, the line is:

acc += tl.dot(input, weight, allow_tf32=True)

To transpose the weight, it should be:

acc += tl.dot(input, tl.trans(weight), allow_tf32=True)

But wait, the input is (BLOCK_SIZE_B, GROUP_SIZE_I), and weight is (BLOCK_SIZE_O, GROUP_SIZE_I). After transposing, weight becomes (GROUP_SIZE_I, BLOCK_SIZE_O). Then input (B, I) @ weight_t (I, O) gives (B, O), which is correct. 

So changing the line to include tl.trans(weight) would fix the matrix multiplication dimensions. 

Therefore, the change needed is in the linear_forward_kernel's dot product line. 

Original code line 47:

acc += tl.dot(input, weight, allow_tf32=True)

Change to:

acc += tl.dot(input, tl.trans(weight), allow_tf32=True)

This would transpose the weight tile to (GROUP_SIZE_I, BLOCK_SIZE_O), allowing the matrix multiplication to proceed correctly. 

Testing this change, the matrix dimensions would be:

input: (BLOCK_SIZE_B, GROUP_SIZE_I)

weight_t: (GROUP_SIZE_I, BLOCK_SIZE_O)

Dot product result: (BLOCK_SIZE_B, BLOCK_SIZE_O), which matches the accumulator's shape. 

This should resolve the assertion error. 

Now, checking the error message again. The error was in the forward kernel's dot product line. The input shape was [64,32], and the weight shape was [64,32]. After transposing, the weight becomes [32,64], so input (64,32) @ weight (32,64) = (64,64), which is valid. 

Thus, adding tl.trans(weight) in the dot product should fix the error. 

Looking at the code in the original linear_triton.py provided by the user, in the linear_forward_kernel, the line where the dot product is performed is:

acc += tl.dot(input, weight, allow_tf32=True)

Changing this to:

acc += tl.dot(input, tl.trans(weight), allow_tf32=True)

This should transpose the weight matrix tile before multiplying.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl
import math

# Forward kernel with autotune for block sizes and group sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_I': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    Computes forward pass of linear layer using matrix multiplication.
    
    Args:
        input_ptr (tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tensor): Pointer to output tensor of shape (B, O)
        B, I, O (int): Dimensions of input and weight tensors
        stride_*: Stride information for tensor access
        BLOCK_SIZE_B, BLOCK_SIZE_O (tl.constexpr): Tile sizes for B and O dimensions
        GROUP_SIZE_I (tl.constexpr): Group size for splitting I dimension
        
    Returns:
        None (writes results to output_ptr)
        
    Explanation:
        1. Calculate program IDs for batch and output feature blocks
        2. Initialize output accumulator
        3. Loop over input features in groups
        4. Load input and weight tiles using block pointers
        5. Compute partial matrix product with tl.dot
        6. Accumulate results and store final output
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    offs_i = tl.arange(0, GROUP_SIZE_I)
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, GROUP_SIZE_I):
        input = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] + i < I), other=0.0)
        weight = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] + i < I), other=0.0)
        acc += tl.dot(input, tl.trans(weight), allow_tf32=True)
        input_ptrs += GROUP_SIZE_I * stride_input_i
        weight_ptrs += GROUP_SIZE_I * stride_weight_i
        
    acc = acc.to(tl.float16)
    output_ptrs = output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o
    tl.store(output_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

# Backward input kernel with autotune
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 256, 'GROUP_SIZE_O': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_input_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, I, O,
    stride_grad_output_b, stride_grad_output_o,
    stride_weight_o, stride_weight_i,
    stride_grad_input_b, stride_grad_input_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr
):
    """
    Computes input gradient (dx) using matrix multiplication of grad_output and weight.
    
    Args:
        grad_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        weight_ptr (tensor): Pointer to weight tensor (O, I)
        grad_input_ptr (tensor): Pointer to input gradient tensor (B, I)
        B, I, O (int): Dimensions of tensors
        stride_*: Stride information for tensor access
        BLOCK_SIZE_B, BLOCK_SIZE_I (tl.constexpr): Tile sizes for B and I dimensions
        GROUP_SIZE_O (tl.constexpr): Group size for splitting O dimension
        
    Returns:
        None (writes results to grad_input_ptr)
        
    Explanation:
        1. Calculate program IDs for batch and input feature blocks
        2. Initialize accumulator
        3. Loop over output features in groups
        4. Load grad_output and weight tiles
        5. Compute partial products with tl.dot
        6. Accumulate and store results
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    offs_o = tl.arange(0, GROUP_SIZE_O)
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, GROUP_SIZE_O):
        grad_output = tl.load(grad_output_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] + o < O), other=0.0)
        weight = tl.load(weight_ptrs, mask=(offs_o[:, None] + o < O) & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output, weight, allow_tf32=True)
        grad_output_ptrs += GROUP_SIZE_O * stride_grad_output_o
        weight_ptrs += GROUP_SIZE_O * stride_weight_o
        
    acc = acc.to(tl.float16)
    grad_input_ptrs = grad_input_ptr + offs_b[:, None] * stride_grad_input_b + offs_i[None, :] * stride_grad_input_i
    tl.store(grad_input_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

# Backward weight kernel with autotune
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 256, 'GROUP_SIZE_B': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_weight_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_grad_output_b, stride_grad_output_o,
    stride_grad_weight_o, stride_grad_weight_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Computes weight gradient (dw) using outer product of input and grad_output.
    
    Args:
        input_ptr (tensor): Pointer to input tensor (B, I)
        grad_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        grad_weight_ptr (tensor): Pointer to weight gradient tensor (O, I)
        B, I, O (int): Dimensions of tensors
        stride_*: Stride information for tensor access
        BLOCK_SIZE_O, BLOCK_SIZE_I (tl.constexpr): Tile sizes for O and I dimensions
        GROUP_SIZE_B (tl.constexpr): Group size for splitting B dimension
        
    Returns:
        None (writes results to grad_weight_ptr)
        
    Explanation:
        1. Calculate program IDs for output and input feature blocks
        2. Initialize accumulator
        3. Loop over batches in groups
        4. Load input and grad_output tiles
        5. Compute outer product with tl.dot
        6. Accumulate and store results atomically
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    offs_b = tl.arange(0, GROUP_SIZE_B)
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, GROUP_SIZE_B):
        mask = (offs_b[None, :] + b < B) & (offs_i[:, None] < I) & (offs_o[:, None] < O)
        input = tl.load(input_ptrs, mask=mask, other=0.0)
        grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
        acc += tl.dot(grad_output, input, allow_tf32=True)
        input_ptrs += GROUP_SIZE_B * stride_input_b
        grad_output_ptrs += GROUP_SIZE_B * stride_grad_output_b
        
    acc = acc.to(tl.float16)
    grad_weight_ptrs = grad_weight_ptr + offs_o[:, None] * stride_grad_weight_o + offs_i[None, :] * stride_grad_weight_i
    tl.atomic_add(grad_weight_ptrs, acc, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I))

class TritonLinearFwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )
        
        linear_forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1)
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]
        
        # Calculate input gradient
        grad_input = torch.empty_like(input)
        grid_dx = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        linear_backward_input_kernel[grid_dx](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1)
        )
        
        # Calculate weight gradient
        grad_weight = torch.zeros_like(weight)
        grid_dw = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        linear_backward_weight_kernel[grid_dw](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1)
        )
        
        return grad_input, grad_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return TritonLinearFwd.apply(x, self.weight)

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        return triton_module, torch_module

    def forward_generator(B, I, O, device='cuda', dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        return input

    from benchmark import verify_correctness_forward, benchmark_performance_forward
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])
    benchmark_performance_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])

    def backward_generator(B, I, O, device='cuda', dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        d_output = torch.nn.functional.normalize(d_output, dim=-1)
        return input, d_output

    from benchmark import verify_correctness_backward, benchmark_performance_backward
    verify_correctness_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes])
    benchmark_performance_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes])
```