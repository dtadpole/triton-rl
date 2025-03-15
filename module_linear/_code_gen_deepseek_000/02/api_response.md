API: deepseek
Test Case: ./module_linear/
Timestamp: 20250314_093634

<think>
Okay, I need to create a Triton module for a PyTorch nn.Linear layer, implementing both forward and backward passes using Triton. The filename should be triton_linear.py. Let's start by understanding the requirements.

First, the TritonLinear class needs to mimic the functionality of torch.nn.Linear, but using Triton kernels. The user mentioned using autotune for block sizes BLOCK_SIZE_B, BLOCK_SIZE_I, and BLOCK_SIZE_O. So, I'll need to set up autotuned configurations for these block sizes in both the forward and backward kernels.

For the forward pass, the Triton kernel will perform matrix multiplication. Since PyTorch's linear layer is essentially a matmul between the input and the weight matrix (plus bias if present, but the test code doesn't include bias), the forward kernel needs to handle input of shape (B, I) and weight of shape (O, I) to produce output of shape (B, O). Wait, actually, PyTorch's Linear layer uses weight of shape (O, I), and the input is (B, I), so the matmul is input @ weight.T, resulting in (B, O). So the Triton kernel needs to compute this.

In Triton, the matrix multiplication kernel typically involves tiling both the input and weight matrices into blocks. Autotuning the block sizes for B, I, and O dimensions will help optimize performance across different sizes. The autotune configurations should be specified with possible values for these block sizes, and Triton's autotune will select the best ones.

For the backward pass, we need to compute the gradients of the input and the weight. The input gradient is computed as d_output @ weight, and the weight gradient is input.T @ d_output. However, since this is a backward pass, the Triton kernels need to handle these two operations. But wait, in PyTorch's autograd, the backward function would compute the gradients. So the TritonLinear module's backward method should use custom Triton kernels for these computations.

Wait, but when implementing a custom autograd Function in PyTorch, the forward and backward methods must be defined. So the TritonLinear class will subclass nn.Module and contain a weight parameter. The forward method will use a custom autograd Function that uses Triton kernels for both forward and backward passes.

So the steps are:

1. Create a TritonLinear class inheriting from nn.Module.
2. Initialize the weight parameter with the correct shape (O, I), as in PyTorch's Linear layer.
3. Implement a custom autograd Function (e.g., TritonLinearFunction) that defines the forward and backward passes using Triton kernels.
4. In the forward method of TritonLinear, apply the TritonLinearFunction to the input and the weight.
5. Implement the forward and backward Triton kernels with autotuning.

Now, let's think about the kernels.

For the forward kernel (triton_linear_forward):

- Input is (B, I), weight is (O, I). The output is (B, O).
- The kernel needs to compute each element of the output matrix by taking the dot product of rows from input and columns from weight (since weight is (O, I), but the matmul is input @ weight.T).
- So each element in the output (b, o) is the sum over i of input[b, i] * weight[o, i].
- The kernel should tile the computation with blocks for B and O dimensions. The I dimension will be reduced.

For the backward kernels:

- The input gradient (d_input) is computed as d_output @ weight. The shape of d_output is (B, O), weight is (O, I), so d_input is (B, I).
- The weight gradient (d_weight) is input.T @ d_output. The shape of input is (B, I), so input.T is (I, B), and d_output is (B, O), resulting in (I, O) for d_weight. But since the weight is (O, I), the gradient should be accumulated as the transpose of that? Wait, no. The weight gradient for a linear layer is indeed input.T @ d_output, which is (I, B) @ (B, O) = (I, O). But the weight's gradient should have the same shape as the weight, which is (O, I). So maybe we need to transpose the result. Wait, no. Let's think: The weight is (O, I), so the derivative dL/dweight is computed as (input)^T @ d_output, which gives (I, B) @ (B, O) = (I, O). But the weight's gradient should be (O, I). So maybe we need to transpose the result of input.T @ d_output to get (O, I). Hmm, that's a potential point of confusion here.

Wait, PyTorch's Linear layer without bias computes output = input @ weight.T. So the gradient for weight would be the input.T @ d_output, which gives (I, B) @ (B, O) = (I, O). But the weight's shape is (O, I), so the gradient should be (I, O).T = (O, I). So yes, the weight gradient is (d_output.T @ input). Because dL/dweight = d_output.T @ input. That makes sense. So the weight gradient is (O, B) @ (B, I) = (O, I), which matches the weight shape. Therefore, in the backward pass, the weight gradient is computed as d_output.T @ input.

So, for the backward function:

- d_input = d_output @ weight (shape (B, O) @ (O, I) → (B, I))
- d_weight = d_output.T @ input (shape (O, B) @ (B, I) → (O, I))

Therefore, in Triton, the backward pass requires two kernels: one for d_input and one for d_weight.

But when implementing the backward pass in a custom autograd Function, the backward method needs to compute gradients for both input and weight. So the TritonLinearFunction's backward will return the gradients for input and weight (and None for other inputs, like bias if present, but in this case there's no bias).

So the Triton kernels needed are:

1. Forward kernel: input (B, I) @ weight (O, I) → output (B, O)
2. Backward kernel for input gradient: d_output (B, O) @ weight (O, I) → d_input (B, I)
3. Backward kernel for weight gradient: d_output.T (O, B) @ input (B, I) → d_weight (O, I)

Each of these will be implemented as Triton kernels with autotuning.

Now, setting up the autotune configurations. For each kernel, we'll define possible block sizes for the relevant dimensions. For example, in the forward kernel, we might autotune BLOCK_SIZE_B and BLOCK_SIZE_O, with possible values like 16, 32, 64, etc. Similarly for the other kernels.

In Triton, autotuning is done using the @triton.autotune decorator, which takes a list of configs. Each config specifies values for the kernel parameters that Triton will benchmark to find the fastest.

But how to structure the kernels?

For the forward kernel, the block size for B (BLOCK_SIZE_B) and O (BLOCK_SIZE_O) could be tuned. The kernel will process a block of B rows and O columns, and for each element in that block, accumulate the sum over I elements.

Similarly, the backward kernel for d_input would process blocks of B and I, and the kernel for d_weight would process blocks of O and I.

Wait, but the d_input computation is (B, O) @ (O, I) → (B, I). Each element in d_input (B, I) is a sum over O of d_output[b, o] * weight[o, i]. So the kernel for d_input would need to iterate over O dimension. The block sizes for B and I can be tuned.

For the d_weight computation, which is (O, B) @ (B, I) → (O, I), each element in d_weight (O, I) is a sum over B of d_output[o, b] * input[b, i]. So the kernel would iterate over B dimension, and block sizes for O and I can be tuned.

Each of these needs their own Triton kernels with autotuning.

Now, writing the code structure:

The TritonLinear class will have:

- __init__ method: initializes weight as a parameter of shape (O, I), and initializes it similarly to PyTorch's Linear layer (e.g., Kaiming normal).
- forward method: applies the TritonLinearFunction to input and weight.

The TritonLinearFunction (a subclass of torch.autograd.Function) will have:

- forward: uses the forward Triton kernel to compute output.
- backward: uses the two backward Triton kernels to compute gradients for input and weight.

But wait, in the backward pass, the Function's backward method will receive gradients w.r.t. the output (d_output) and needs to compute gradients for input and weight.

Now, implementing the kernels with autotune.

First, the forward kernel:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        # Add more configs as needed
    ],
)
@triton.jit
def triton_linear_forward(...):
    ...

Similarly for the backward kernels.

But for the backward pass, two separate kernels are needed. Each with their own autotuning.

For the d_input kernel (computing input gradient):

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 16}, ...),
        ...
    ],
)
@triton.jit
def triton_linear_backward_input(...):
    ...

For the d_weight kernel (computing weight gradient):

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 16}, ...),
        ...
    ],
)
@triton.jit
def triton_linear_backward_weight(...):
    ...

Now, the details of each kernel.

For the forward kernel:

The input is of shape (B, I), weight is (O, I). The output is (B, O).

Each output element at (b, o) is sum over i of input[b, i] * weight[o, i].

In kernel code:

- The kernel will be launched with a grid that covers all B and O elements in blocks of BLOCK_SIZE_B and BLOCK_SIZE_O.

- For each block, the program ID will correspond to a block of B and O.

- Within each block, each thread (or element) computes the sum for a specific (b, o) pair.

But since I can be large, we need to loop over the I dimension in chunks. The block size for I (BLOCK_SIZE_I) can be another parameter, but perhaps that's part of the autotuning.

Wait, perhaps the kernel uses a reduction over the I dimension. So each thread block handles a block of B and O, and for each element in that block, iterates over the I dimension in chunks, accumulating the sum.

Alternatively, the kernel could be structured to process tiles of input and weight matrices, but given that input is (B, I) and weight is (O, I), the matmul is more of a batched outer product.

Wait, maybe a better way to structure the kernel is to have each thread compute a single element of the output matrix. For each element (b, o), the kernel loops over the I dimension, loading chunks of input and weight into shared memory, and accumulates the sum.

But in Triton, the programming model is a bit different. Let's refer to the Triton matrix multiplication example.

Looking at the Triton matmul example (like in their tutorials), the kernel is structured with two blocks for M and N dimensions, and the K dimension is reduced. So for our case, the input is (B, I) and weight is (O, I), so the matmul is input @ weight.T, which is (B, I) @ (I, O) → (B, O). So the M dimension is B, N is O, and K is I. So the kernel can be similar to the standard matmul example, but with M as B and N as O.

So the kernel will have:

- pid_m and pid_n to iterate over B and O.

Each block handles a block of B (pid_m) and O (pid_n), and each thread in the block computes a portion of the sum over I.

So the kernel code would:

1. Compute the row and column indices for the output matrix.

2. Initialize a sum accumulator.

3. Loop over the I dimension in blocks, loading input and weight tiles into shared memory.

4. Compute the dot product for the current tile and accumulate.

5. After the loop, store the result to the output.

So in code:

For forward kernel:

@triton.jit
def triton_linear_forward(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # Strides
    input_row_stride, input_col_stride,
    weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    # Other parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1)

    # Define the offsets for the block in B and O dimensions
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    # Loop over I dimension in blocks
    for i in range(0, I, BLOCK_SIZE_I):
        offs_i = i + tl.arange(0, BLOCK_SIZE_I)
        # Load input and weight tiles
        # Input is (B, I), so for each row in B, load I elements
        input_tile = tl.load(input_ptr + offs_b[:, None] * input_row_stride + offs_i[None, :] * input_col_stride,
                             mask=(offs_b[:, None] < B) & (offs_i[None, :] < I), other=0.0)
        # Weight is (O, I), so for each row in O, load I elements
        weight_tile = tl.load(weight_ptr + offs_o[:, None] * weight_row_stride + offs_i[None, :] * weight_col_stride,
                              mask=(offs_o[:, None] < O) & (offs_i[None, :] < I), other=0.0)
        # Compute matrix multiplication for the current tile
        acc += tl.dot(input_tile, weight_tile, allow_tf32=True)

    # Store the result to output
    offs_b_out = offs_b[:, None]
    offs_o_out = offs_o[None, :]
    output_ptrs = output_ptr + offs_b_out * output_row_stride + offs_o_out * output_col_stride
    tl.store(output_ptrs, acc, mask=(offs_b_out < B) & (offs_o_out < O))

Wait, but the input_tile would be BLOCK_SIZE_B x BLOCK_SIZE_I, and weight_tile would be BLOCK_SIZE_O x BLOCK_SIZE_I. Then, the dot product of input_tile (BxI) and weight_tile (OxI) transposed would be (BLOCK_SIZE_B x BLOCK_SIZE_O). So maybe the code uses input_tile @ weight_tile.T?

Alternatively, tl.dot can handle this if the shapes are correct. Maybe the code uses the input_tile as (BLOCK_SIZE_B, BLOCK_SIZE_I) and weight_tile as (BLOCK_SIZE_O, BLOCK_SIZE_I), and then the dot product is input_tile @ weight_tile.T, resulting in (BLOCK_SIZE_B, BLOCK_SIZE_O), which is accumulated.

But in Triton, tl.dot(a, b) where a is (M, K) and b is (K, N) results in (M, N). So if input_tile is (BLOCK_SIZE_B, BLOCK_SIZE_I) and weight_tile is (BLOCK_SIZE_O, BLOCK_SIZE_I), then we need to transpose the weight_tile to (BLOCK_SIZE_I, BLOCK_SIZE_O), then input_tile @ weight_tile.T would be input_tile @ weight_tile.T.

Wait, no. If weight_tile is (BLOCK_SIZE_O, BLOCK_SIZE_I), then weight_tile.T is (BLOCK_SIZE_I, BLOCK_SIZE_O). So input_tile (BLOCK_SIZE_B, BLOCK_SIZE_I) @ weight_tile.T (BLOCK_SIZE_I, BLOCK_SIZE_O) gives (BLOCK_SIZE_B, BLOCK_SIZE_O). Which is correct.

But in the code above, the load for weight_tile is rows O and columns I. So weight_tile is (BLOCK_SIZE_O, BLOCK_SIZE_I). So to compute the dot product with input_tile (BLOCK_SIZE_B, BLOCK_SIZE_I), we need to transpose the weight_tile and do input_tile @ weight_tile.T. But tl.dot(input_tile, weight_tile) would compute (BLOCK_SIZE_B, BLOCK_SIZE_I) @ (BLOCK_SIZE_O, BLOCK_SIZE_I) → which is not possible unless we transpose. So perhaps there's a mistake here. The correct way is to transpose the weight_tile.

Wait, no. The weight matrix in the linear layer is (O, I). So each row in weight is a neuron's weights. The input is (B, I), so the output is input @ weight.T, which is (B, I) @ (I, O) → (B, O). Therefore, weight_tile is (BLOCK_SIZE_O, BLOCK_SIZE_I), and we need to multiply input_tile (BLOCK_SIZE_B, BLOCK_SIZE_I) with weight_tile.T (BLOCK_SIZE_I, BLOCK_SIZE_O). So the dot product would be input_tile @ weight_tile.T.

But in Triton's tl.dot, the inputs are (M, K) and (K, N). So to compute input_tile (M=BLOCK_SIZE_B, K=BLOCK_SIZE_I) multiplied by weight_tile.T (K=BLOCK_SIZE_I, N=BLOCK_SIZE_O), you need to have the second matrix as (BLOCK_SIZE_I, BLOCK_SIZE_O). So, yes, you need to transpose the weight_tile.

But in the code above, the weight_tile is (BLOCK_SIZE_O, BLOCK_SIZE_I), so to get (BLOCK_SIZE_I, BLOCK_SIZE_O), we can transpose it. So the code should do:

acc += tl.dot(input_tile, tl.trans(weight_tile))

But tl.trans is a Triton function for transposing. So the correct code would transpose the weight_tile.

So in the kernel code, the line would be:

acc += tl.dot(input_tile, tl.trans(weight_tile))

This way, input_tile is (BLOCK_SIZE_B, BLOCK_SIZE_I) multiplied by weight_tile transposed to (BLOCK_SIZE_I, BLOCK_SIZE_O), resulting in (BLOCK_SIZE_B, BLOCK_SIZE_O).

But in the initial code draft, the line was:

acc += tl.dot(input_tile, weight_tile, allow_tf32=True)

Which would attempt to multiply (BLOCK_SIZE_B, BLOCK_SIZE_I) by (BLOCK_SIZE_O, BLOCK_SIZE_I), which is invalid. So this is a mistake. Hence, the code needs to transpose the weight_tile.

So that's a key correction.

Now, the forward kernel code would include the transpose.

Moving on to the backward pass.

For the input gradient (d_input = d_output @ weight), the shapes are (B, O) @ (O, I) → (B, I). So each element in d_input[b, i] is sum over o of d_output[b, o] * weight[o, i].

The kernel for this would have blocks for B and I dimensions, similar to the forward kernel but with different dimensions.

The kernel code would loop over O in blocks, accumulate the sum over o.

Similarly, for the weight gradient (d_weight = d_output.T @ input), which is (O, B) @ (B, I) → (O, I). Each element d_weight[o, i] is sum over b of d_output[b, o] * input[b, i].

This kernel would have blocks for O and I dimensions and loop over B in blocks.

Each of these kernels would need their own autotuning for block sizes.

Now, putting this all together into code.

The TritonLinearFunction's forward and backward methods will call these kernels.

But when using Triton kernels in PyTorch's autograd Functions, we need to handle the tensors' data pointers, strides, and manage the kernel launches properly.

Also, during the backward pass, the Function will need to save the input and weight for the gradient computations.

In the forward method of TritonLinearFunction:

- Save input and weight for backward.
- Compute output using the forward kernel.

In the backward method:

- Retrieve saved input and weight.
- Compute d_input using the backward_input kernel.
- Compute d_weight using the backward_weight kernel.
- Return d_input and d_weight (and None for bias, but there's no bias here).

Now, potential issues to consider:

- The data types: The test code uses torch.float16. Ensure that the Triton kernels handle fp16 correctly. Triton has support for fp16, but accumulation is done in fp32 for better precision.
- Strides: Need to compute the correct strides for input and weight tensors. For example, if a tensor is not contiguous, the stride may not be as expected. So in the kernel, the stride parameters are passed so that the kernel can correctly access the elements.
- Autotuning configurations: The user specified BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O. The forward kernel uses B and O block sizes, the backward_input uses B and I, backward_weight uses O and I. So the autotune configs for each kernel should focus on those dimensions.

Now, implementing the code.

First, the imports.

We need to import torch, triton, and triton.language as tl.

Then, the TritonLinear class.

Then, the TritonLinearFunction.

The forward and backward functions.

But for the autograd Function, we need to define the forward and backward as static methods.

Now, code structure:

import torch
import triton
import triton.language as tl

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        # save for backward
        ctx.save_for_backward(input, weight)
        # compute output using triton kernel
        B, I = input.shape
        O, I_w = weight.shape
        assert I == I_w, "Input and weight dimensions mismatch"
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        # launch kernel
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )
        triton_linear_forward[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_B=..., BLOCK_SIZE_O=..., BLOCK_SIZE_I=...,
        )
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        # compute gradients
        d_input = torch.empty_like(input)
        d_weight = torch.empty_like(weight)
        B, I = input.shape
        O, _ = weight.shape
        # launch input gradient kernel
        grid_input = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        triton_linear_backward_input[grid_input](
            d_output, weight, d_input,
            B, O, I,
            d_output.stride(0), d_output.stride(1),
            weight.stride(0), weight.stride(1),
            d_input.stride(0), d_input.stride(1),
            BLOCK_SIZE_B=..., BLOCK_SIZE_I=...,
            BLOCK_SIZE_O=...,
        )
        # launch weight gradient kernel
        grid_weight = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        triton_linear_backward_weight[grid_weight](
            d_output, input, d_weight,
            B, O, I,
            d_output.stride(0), d_output.stride(1),
            input.stride(0), input.stride(1),
            d_weight.stride(0), d_weight.stride(1),
            BLOCK_SIZE_O=..., BLOCK_SIZE_I=...,
            BLOCK_SIZE_B=...,
        )
        return d_input, d_weight

But the actual kernel launches need to pass the autotune parameters correctly. However, since we're using autotune, the kernel will automatically select the best parameters based on the configs provided in the decorators.

But in the code above, when launching the kernel, we might need to pass the block sizes as kernel arguments. But Triton's autotune handles this automatically by selecting the best config from the provided list.

Thus, in the code, when launching the kernel, the specific values for BLOCK_SIZE_B, etc., are determined by Triton's autotuner based on the configs provided in the @triton.autotune decorator.

So, for example, the triton_linear_forward kernel would have:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        # other configs...
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def triton_linear_forward(...):
    ...

The key parameters for autotune are the B, I, O sizes.

Now, putting all this together.

But there's a lot to write, and I need to ensure that all the kernel parameters are correctly passed.

Another consideration: the test code module_generator initializes TritonLinear with I and O, and the test uses input with shape (B, I). The TritonLinear class should have in_features I and out_features O.

In the TritonLinear class:

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weight as (out_features, in_features)
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Initialize weight similarly to PyTorch's Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

Now, the __init__ method correctly initializes the weight parameter with shape (O, I).

Testing the code:

The user provided a main block that generates modules and tests correctness and performance. The code should be included as is in the triton_linear.py file.

Potential issues:

- The Triton kernels must be compatible with the data types used (e.g., fp16).
- The autotune configurations must cover a reasonable range of block sizes to ensure performance across different B, I, O sizes.
- Masking in the kernels must properly handle cases where the block size doesn't evenly divide the matrix dimensions.

Another thing to note is that the Triton kernels need to handle the tensor strides correctly. For example, if the input or weight is not contiguous, the stride parameters passed to the kernel will affect how the data is accessed. To handle this, the kernels should use the stride values from the tensors when loading data.

Additionally, the forward and backward kernels must accumulate sums correctly, initializing the accumulator to zero and summing across the appropriate dimensions.

Now, let's draft the code with detailed comments as per the user's request.

For each function (TritonLinearFunction.forward, backward, and the three kernels), we need to add the structured comments.

For example, the triton_linear_forward kernel:

@triton.autotune(...)
@triton.jit
def triton_linear_forward(...):
    """
    Description: Performs forward pass of linear layer using matrix multiplication.

    Args:
        input_ptr (tensor): Input tensor of shape (B, I).
        weight_ptr (tensor): Weight tensor of shape (O, I).
        output_ptr (tensor): Output tensor of shape (B, O).
        B (int): Batch size dimension.
        I (int): Input feature dimension.
        O (int): Output feature dimension.
        input_row_stride (int): Stride for rows of input tensor.
        input_col_stride (int): Stride for columns of input tensor.
        weight_row_stride (int): Stride for rows of weight tensor.
        weight_col_stride (int): Stride for columns of weight tensor.
        output_row_stride (int): Stride for rows of output tensor.
        output_col_stride (int): Stride for columns of output tensor.
        BLOCK_SIZE_B (tl.constexpr): Block size for B dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for O dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for I dimension.

    Returns:
        None: Output is stored in output_ptr.

    Explanation:
        1. Calculate block indices for B and O dimensions.
        2. Initialize accumulator to zero.
        3. Loop over I dimension in blocks of BLOCK_SIZE_I.
        4. Load input and weight tiles from global memory.
        5. Compute dot product and accumulate.
        6. Store accumulated result to output.
    """
    # Kernel code...

Similarly for the other kernels and functions.

Finally, the test section will include the user-provided code for correctness and performance testing.

Now, writing the complete code with all the comments and structure.
</think>

```python
# filename: triton_linear.py
import torch
import triton
import triton.language as tl

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Description: Performs forward pass of linear layer using Triton kernels.

        Args:
            ctx (torch.autograd.function.BackwardCFunction): Context for saving tensors.
            input (torch.Tensor): Input tensor of shape (B, I).
            weight (torch.Tensor): Weight tensor of shape (O, I).

        Returns:
            torch.Tensor: Output tensor of shape (B, O).

        Explanation:
            1. Save input and weight for backward pass.
            2. Initialize output tensor.
            3. Launch Triton kernel with autotuned block sizes.
        """
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )
        
        triton_linear_forward[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Description: Performs backward pass of linear layer using Triton kernels.

        Args:
            ctx (torch.autograd.function.BackwardCFunction): Context with saved tensors.
            grad_output (torch.Tensor): Gradient of output tensor of shape (B, O).

        Returns:
            tuple: (grad_input, grad_weight) tensors of shapes (B, I) and (O, I).

        Explanation:
            1. Retrieve saved tensors from forward pass.
            2. Initialize gradient tensors.
            3. Launch input gradient kernel with autotuned block sizes.
            4. Launch weight gradient kernel with autotuned block sizes.
        """
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]
        
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)

        # Compute input gradient
        grid_input = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        triton_linear_backward_input[grid_input](
            grad_output, weight, grad_input,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )

        # Compute weight gradient
        grid_weight = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        triton_linear_backward_weight[grid_weight](
            grad_output, input, grad_weight,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )

        return grad_input, grad_weight

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def triton_linear_forward(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    input_row_stride, input_col_stride,
    weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    """
    Description: Triton kernel for linear layer forward pass.

    Args:
        input_ptr (tensor): Input tensor of shape (B, I).
        weight_ptr (tensor): Weight tensor of shape (O, I).
        output_ptr (tensor): Output tensor of shape (B, O).
        B (int): Batch size dimension.
        I (int): Input feature dimension.
        O (int): Output feature dimension.
        input_row_stride (int): Input tensor row stride.
        input_col_stride (int): Input tensor column stride.
        weight_row_stride (int): Weight tensor row stride.
        weight_col_stride (int): Weight tensor column stride.
        output_row_stride (int): Output tensor row stride.
        output_col_stride (int): Output tensor column stride.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.

    Returns:
        None: Output is stored in output_ptr.

    Explanation:
        1. Calculate block indices for B and O dimensions.
        2. Initialize accumulator to zero.
        3. Loop over I dimension in BLOCK_SIZE_I increments.
        4. Load input and weight tiles from global memory.
        5. Compute matrix multiplication using tensor cores.
        6. Store accumulated results to output.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * input_row_stride + offs_i[None, :] * input_col_stride
    weight_ptrs = weight_ptr + offs_o[:, None] * weight_row_stride + offs_i[None, :] * weight_col_stride
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        mask_i = i * BLOCK_SIZE_I + offs_i < I
        input = tl.load(input_ptrs, mask=mask_i[None, :] & (offs_b[:, None] < B), other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_i[None, :] & (offs_o[:, None] < O), other=0.0)
        acc += tl.dot(input, tl.trans(weight))
        input_ptrs += BLOCK_SIZE_I * input_col_stride
        weight_ptrs += BLOCK_SIZE_I * weight_col_stride
    
    output_ptrs = output_ptr + offs_b[:, None] * output_row_stride + offs_o[None, :] * output_col_stride
    tl.store(output_ptrs, acc.to(tl.float16), mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def triton_linear_backward_input(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    grad_output_row_stride, grad_output_col_stride,
    weight_row_stride, weight_col_stride,
    grad_input_row_stride, grad_input_col_stride,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    """
    Description: Triton kernel for computing input gradients.

    Args:
        grad_output_ptr (tensor): Output gradient tensor of shape (B, O).
        weight_ptr (tensor): Weight tensor of shape (O, I).
        grad_input_ptr (tensor): Input gradient tensor of shape (B, I).
        B (int): Batch size dimension.
        O (int): Output feature dimension.
        I (int): Input feature dimension.
        grad_output_row_stride (int): Output gradient row stride.
        grad_output_col_stride (int): Output gradient column stride.
        weight_row_stride (int): Weight tensor row stride.
        weight_col_stride (int): Weight tensor column stride.
        grad_input_row_stride (int): Input gradient row stride.
        grad_input_col_stride (int): Input gradient column stride.
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.

    Returns:
        None: Input gradient is stored in grad_input_ptr.

    Explanation:
        1. Calculate block indices for B and I dimensions.
        2. Initialize accumulator to zero.
        3. Loop over O dimension in blocks.
        4. Load gradient output and weight tiles.
        5. Compute matrix multiplication and accumulate.
        6. Store accumulated results to input gradient.
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, 32)
    
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * grad_output_row_stride + offs_o[None, :] * grad_output_col_stride
    weight_ptrs = weight_ptr + offs_o[:, None] * weight_row_stride + offs_i[None, :] * weight_col_stride
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, tl.cdiv(O, 32)):
        mask_o = o * 32 + offs_o < O
        grad_output = tl.load(grad_output_ptrs, mask=mask_o[None, :] & (offs_b[:, None] < B), other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_o[:, None] & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output, weight)
        grad_output_ptrs += 32 * grad_output_col_stride
        weight_ptrs += 32 * weight_row_stride
    
    grad_input_ptrs = grad_input_ptr + offs_b[:, None] * grad_input_row_stride + offs_i[None, :] * grad_input_col_stride
    tl.store(grad_input_ptrs, acc.to(tl.float16), mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 16, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def triton_linear_backward_weight(
    grad_output_ptr, input_ptr, grad_weight_ptr,
    B, O, I,
    grad_output_row_stride, grad_output_col_stride,
    input_row_stride, input_col_stride,
    grad_weight_row_stride, grad_weight_col_stride,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    """
    Description: Triton kernel for computing weight gradients.

    Args:
        grad_output_ptr (tensor): Output gradient tensor of shape (B, O).
        input_ptr (tensor): Input tensor of shape (B, I).
        grad_weight_ptr (tensor): Weight gradient tensor of shape (O, I).
        B (int): Batch size dimension.
        O (int): Output feature dimension.
        I (int): Input feature dimension.
        grad_output_row_stride (int): Output gradient row stride.
        grad_output_col_stride (int): Output gradient column stride.
        input_row_stride (int): Input tensor row stride.
        input_col_stride (int): Input tensor column stride.
        grad_weight_row_stride (int): Weight gradient row stride.
        grad_weight_col_stride (int): Weight gradient column stride.
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension.
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension.

    Returns:
        None: Weight gradient is stored in grad_weight_ptr.

    Explanation:
        1. Calculate block indices for O and I dimensions.
        2. Initialize accumulator to zero.
        3. Loop over B dimension in blocks.
        4. Load gradient output and input tiles.
        5. Compute outer product and accumulate.
        6. Store accumulated results to weight gradient.
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, 32)
    
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * grad_output_row_stride + offs_o[None, :] * grad_output_col_stride
    input_ptrs = input_ptr + offs_b[:, None] * input_row_stride + offs_i[None, :] * input_col_stride
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, tl.cdiv(B, 32)):
        mask_b = b * 32 + offs_b < B
        grad_output = tl.load(grad_output_ptrs, mask=mask_b[:, None] & (offs_o[None, :] < O), other=0.0)
        input = tl.load(input_ptrs, mask=mask_b[:, None] & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(tl.trans(grad_output), input)
        grad_output_ptrs += 32 * grad_output_row_stride
        input_ptrs += 32 * input_row_stride
    
    grad_weight_ptrs = grad_weight_ptr + offs_o[:, None] * grad_weight_row_stride + offs_i[None, :] * grad_weight_col_stride
    tl.atomic_add(grad_weight_ptrs, acc.to(tl.float16), mask=(offs_o[:, None] < O) & (offs_i[None, :] < I))

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
    
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

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

<test>
To test the code, run the following command:

```bash
python triton_linear.py
```

Ensure you have the following dependencies installed:
- PyTorch with CUDA support
- Triton (v2.1.0 or compatible)
- benchmark.py in the same directory (with required verify and benchmark functions)
</test>