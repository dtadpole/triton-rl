import os
import datetime
import subprocess
from google import genai
from google.genai import types
from prompts import COMMON_PROMPT, TEST_CASES
from gemini import gemini_gen
from fireworks import deepseek_gen

test_case = "rmsnorm"

# response_text = deepseek_gen(COMMON_PROMPT + TEST_CASES[test_case])
response_text = gemini_gen(COMMON_PROMPT + TEST_CASES[test_case])

# create a folder for the code and logs, starts with code_gen_<timestamp>
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"code_gen_{timestamp}_{test_case}"
os.makedirs(folder_name, exist_ok=True)

# write response to log file
with open(f"{folder_name}/gen_code.log", "w") as f:
    f.write(response_text)

# copy benchmark.py to the folder
import shutil
shutil.copy("benchmark.py", f"{folder_name}/benchmark.py")

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
test_commands = re.findall(r'```bash\n(.*?)```', response_text, re.DOTALL)
# run test commands and print the output
for test_command in test_commands:
    #cmd = f"bash -c 'cd {folder_name} && /home/centos/miniconda3/envs/torch26/bin/{test_command.strip()}'"
    cmd = f"bash -c 'cd {folder_name} && {test_command.strip()}'"
    print("--------------------------------")
    print(cmd)
    print("--------------------------------")
    # run command and print output as we go, do not wait for the command to finish
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, env=os.environ)
    while True:
        output = process.stdout.readline().decode('utf-8').rstrip()
        if output:
            print(output)  # Print stdout as it becomes available
        if process.poll() is not None:  # Check if process has finished
            break
    process.wait()
