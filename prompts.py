format_prompt = {
    "messages": [
        {
            "role": "system",
            "content": """
            {system_prompt}
            """
        },
        {
            "role": "user",
            "content": """
            {user_prompt}
            """
        },
        {
            "role": "assistant",
            "content": """
            {assistant_response}
            """,
            "weight": """{response_weight}""",
        },
    ]
}

COMMON_PROMPT="""
Respond in the following format, start with a "think" section, followed by one code block in markdown, and followed by a "test" section.
    In the "think" section, give comprehensive, logically structured thinking, use markdown formatting for clear information organization, admit uncertainties for ambiguous situations.
    In the code block, generate one python file. The first line of the code block is always a comment with the filename.
  assume benchmark.py already exists (do NOT generate benchmark.py), and has with verify_correctness and benchmark_performance functions; generate main code, and call the function in benchmark.py.
In the test section, generate instructions and test command to test the code, where the command in a bash code block, do not generate pip install command.
<think>
...
</think>
```python
# filename: file_1.py
from benchmark import verify_correctness, benchmark_performance
...
```
```python
# filename: file_2.py
from benchmark import verify_correctness, benchmark_performance
...
```
```python
# filename: file_x.py
from benchmark import verify_correctness, benchmark_performance
...
```
<test>
...
```bash
...
```
</test>
"""

TEST_CASES = {

    "softmax": """
Generate triton code for softmax, test correctness and performance using following code literally:
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
Generate triton code for RMSNorm, test correctness and performance using following code literally:
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
Generate triton code for Matrix Multiplication,
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
