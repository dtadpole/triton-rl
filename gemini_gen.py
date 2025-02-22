import os
import datetime
import subprocess
from google import genai
from google.genai import types

SYS_PROMPT="""
Respond in the following format, start with a "think" section, followed by one code block in markdown, and followed by a "test" section.
In the "think" section, give comprehensive, logically structured thinking, use markdown formatting for clear information organization, admit uncertainties for ambiguous situations.
In the code block, generate one python file. The first line of the code block is always a comment with the filename.
  assume benchmark.py already exists (do NOT generate benchmark.py), and has with verify_correctness and benchmark_performance functions; generate main code, and call the function in benchmark.py:
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

SOFTMAX_PROMPT="""
    Generate triton code for softmax, test correctness and performance using following code literally:
    ```python
        batch_sizes = [256, 512, 1024, 2048];
        dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
        def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float32):
            x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
            return x
        verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
        benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
    ```
"""

RMSNORM_PROMPT="""
    Generate triton code for RMSNorm, test correctness and performance using following code literally:
    ```python
        batch_sizes = [256, 512, 1024, 2048];
        dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
        def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float32):
            x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
            weight = torch.randn(dim_size, device=device, dtype=dtype)
            eps = 1e-5
            return x, weight, eps
        verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
        benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
    ```
"""

client = genai.Client(api_key="AIzaSyBk5082INxetg6K79MFycnzmiRMFAwTjLQ")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    # model="gemini-2.0-pro-exp-02-05",
    config=types.GenerateContentConfig(
        # response_mime_type="text/markdown",
        system_instruction=SYS_PROMPT,
    ),
    contents=RMSNORM_PROMPT,
)

# create a folder for the code and logs, starts with code_gen_<timestamp>
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"code_gen_{timestamp}"
os.makedirs(folder_name, exist_ok=True)

# write response to log file
with open(f"{folder_name}/gemini_gen.log", "w") as f:
    f.write(response.text)

# copy benchmark.py to the folder
import shutil
shutil.copy("benchmark.py", f"{folder_name}/benchmark.py")

# match response.text with regex and extract the code blocks
import re

code_blocks = re.findall(r'```python\n(.*?)```', response.text, re.DOTALL)

# print(code_blocks)

# write the code blocks to files
for code_block in code_blocks:
    filename = code_block.split("\n")[0].split("# filename: ")[1]
    print(filename)
    with open(f"{folder_name}/{filename}", "w") as f:
        f.write(code_block)

# extract test commands from response.text
test_commands = re.findall(r'```bash\n(.*?)```', response.text, re.DOTALL)
# run test commands and print the output
for test_command in test_commands:
    #cmd = f"bash -c 'cd {folder_name} && /home/centos/miniconda3/envs/torch26/bin/{test_command.strip()}'"
    cmd = f"bash -c 'cd {folder_name} && {test_command.strip()}'"
    print("--------------------------------")
    print(cmd)
    print("--------------------------------")
    # run command and capture the output and error, use subprocess.run
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    print(result.stdout)
    print("--------------------------------")
    print(result.stderr)
    print("--------------------------------")
