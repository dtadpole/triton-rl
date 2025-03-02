CODE_GEN_PROMPT = '''
Respond in the following format, start with a "think" section, followed by one code block in markdown, and followed by a "test" section.
    In the "think" section, give comprehensive, logically structured thinking, use markdown formatting for clear information organization, admit uncertainties for ambiguous situations.
    In the code block, generate one python file. The first line of the code block is always a comment with the filename.
        Assume benchmark.py already exists (do NOT generate benchmark.py), and has with verify_correctness and benchmark_performance functions;
        Generate main code, and call the functions in benchmark.py.
    In the test section, generate instructions and test command to test the code, where the command in a bash code block, do NOT generate pip install command.
<think>
...
</think>
```python
# filename: file_1.py
...
```
```python
# filename: file_2.py
...
```
```python
# filename: file_x.py
...
```
<test>
...
```bash
...
```
</test>
'''

FUNC_FORMAT_PROMPT = '''
For each Python and Triton function, generate a detailed comment section structured as follows:

Description: A summary of the function's purpose.
Args: A list of arguments, each with a type hint and description.  If a type is a tensor or triton pointer, show the shape.
Returns: The return type(s) and description, including None if applicable.
Explanation: A numbered list of sequential steps the function performs.

Follow this template:
```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    [Description of the function's purpose.]

    Args:
        arg1 (type): [Description of arg1.] [Shape if tensor or triton pointer]
        arg2 (type): [Description of arg2.] [Shape if tensor or triton pointer]

    Returns:
        return_type: [Description of returned value(s).] [Shape if tensor or triton pointer]

    Explanation:
        1. [Step 1: What the function does first.]
        2. [Step 2: Next action or logic.]
        3. [Additional steps as needed...]
    """
    # Function code here
```
'''

CODE_FIX_PROMPT = '''Fix the code in the code block, files and error message are provided.
DO NOT change the filename, only fix the code in the code block.
DO NOT add any other files.
DO NOT change the benchmark code.
ONLY respond with the file that contains the fixed code.
'''

BASE_PROMPTS = [
    {
        "role": "user",
        "content": CODE_GEN_PROMPT,
    },
    {
        "role": "user",
        "content": FUNC_FORMAT_PROMPT,
    },
]

FIX_PROMPTS = [
    {
        "role": "user",
        "content": CODE_FIX_PROMPT,
    },
]


TEST_CASES = {

    "softmax": """
Generate triton code for softmax (filename: softmax_triton.py),
    test correctness and performance using following code literally:
```python
    batch_sizes = [256, 512, 1024, 2048];
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
    benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
```
""",

    "rmsnorm": """
Generate triton code for RMSNorm (filename: rmsnorm_triton.py),
    test correctness and performance using following code literally:
```python
    batch_sizes = [256, 512, 1024, 2048];
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
    benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
```
""",

    "rmsnorm_with_linear": """
Generate triton code for RMSNorm with Linear layer (filename: rmsnorm_with_linear_triton.py),
    test correctness and performance using following code literally:
```python
    batch_sizes = [256, 512, 1024, 2048];
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
    benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
```
""",

    "matmul": """
Generate triton code for Matrix Multiplication (filename: matmul_triton.py).
    use autotune for block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), use super blocks and autotune GROUP_SIZE_M,
    test correctness and performance using following code literally:
```python
    M_sizes = [512, 1024, 2048, 4096]
    N_sizes = [512, 1024, 2048, 4096]
    K_sizes = [512, 1024, 2048, 4096]
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=0)
        return A, B
    verify_correctness(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
```
""",
}
