import os
import datetime
import subprocess
from google import genai
from google.genai import types
from gen_code_prompts import COMMON_PROMPT, TEST_CASES
from gen_code_api import API_ENDPOINTS

test_case = "rmsnorm"
api_name = "deepseek"

# create a folder for the code and logs, starts with code_gen_<timestamp>
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"code_gen_{timestamp}_{test_case}_{api_name}"
os.makedirs(folder_name, exist_ok=True)

# write benchmark.py to folder
benchmark_code = open("benchmark.py").read()
benchmark_code_file = f"{folder_name}/benchmark.py"
with open(benchmark_code_file, "w") as f:
    f.write(benchmark_code)

messages = [
    {
        "role": "system",
        "content": ""
    },
    {
        "role": "user",
        "content": COMMON_PROMPT
    },
    {
        "role": "user",
        "content": TEST_CASES[test_case]
    },
    {
        "role": "file",
        "name": "benchmark.py",
        "content": benchmark_code
    }
]

# response_text = deepseek_gen(COMMON_PROMPT + TEST_CASES[test_case])
response_text = API_ENDPOINTS[api_name](messages)


# write response to log file
with open(f"{folder_name}/gen_code.log", "w") as f:
    f.write(response_text)

# match response.text with regex and extract the code blocks
import re

code_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)

# print(code_blocks)

# write the code blocks to files
for code_block in code_blocks:
    filename = code_block.split("\n")[0].split("# filename: ")[1]
    print(filename)
    with open(f"{folder_name}/{filename}", "w") as f:
        f.write(code_block)

# extract test commands from response.text
test_commands = re.findall(r'<test>.*?```bash\n(.*?)```.*?</test>', response_text, re.DOTALL)
# run test commands and print the output
for test_command in test_commands:
    #cmd = f"bash -c 'cd {folder_name} && /home/centos/miniconda3/envs/torch26/bin/{test_command.strip()}'"
    cmd = f"bash -c 'cd {folder_name} && {test_command.strip()}'"
    print("--------------------------------")
    print(cmd)
    print("--------------------------------")
    # run command and print output and error separately, use subprocess.run
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
    print(process.stdout.decode('utf-8'))
    print("--------------------------------")
    print(process.stderr.decode('utf-8'))
    print("--------------------------------")
    # get the exit code
    exit_code = process.returncode
    if exit_code != 0:
        print(f"Test failed with exit code {exit_code}")
        break
    else:
        print(f"Test passed with exit code {exit_code}")
    print("--------------------------------")
