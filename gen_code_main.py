import os
import datetime
import subprocess
import json
from google import genai
from google.genai import types
from prompts import BASE_PROMPTS, FIX_PROMPTS, TEST_CASES
from gen_code_api import API_ENDPOINTS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--api", type=str, default="deepseek")
parser.add_argument("-d", "--dir", type=str, default=".")
parser.add_argument("-s", "--start_iteration", type=int, default=0)
parser.add_argument("-m", "--max_iterations", type=int, default=5)
parser.add_argument("-x", "--fix_mode", type=bool, default=False, action="store_true")
args = parser.parse_args()


if not os.path.exists(args.dir):
    print(f"Directory {args.dir} does not exist")
    exit(1)

if not os.path.exists(os.path.join(args.dir, "prompt.txt")):
    print(f"Prompt file {os.path.join(args.dir, "prompt.txt")} does not exist")
    exit(1)

INPUT_PROMPT = open(os.path.join(args.dir, "prompt.txt")).read()
TEST_COMMANDS = ""

# get the next sequence id
next_sequence_id = 0
# check for next available subfolder name, starts with code_gen_, and ends with a number
while os.path.exists(f"{args.dir}/code_gen_{args.api}_{next_sequence_id:03d}"):
    next_sequence_id += 1
subfolder_name = f"{args.dir}/code_gen_{args.api}_{next_sequence_id:03d}"
# create subfolder_name
os.makedirs(subfolder_name, exist_ok=True)
# for files with name "prompt.txt" or ending with ".py" in the args.folder, read the content and write to subfolder_name
for file in os.listdir(args.dir):
    if file == "prompt.txt" or file.endswith(".py"):
        with open(os.path.join(args.dir, file), "r") as f:
            content = f.read()
        with open(os.path.join(subfolder_name, file), "w") as f:
            f.write(content)

ERR_MESSAGE = ""
if os.path.exists(f"{subfolder_name}/code_stderr.log"):
    with open(f"{subfolder_name}/code_stderr.log", "r") as f:
        ERR_MESSAGE = f.read()

FIX_MODE = args.fix_mode

def codegen_messages(folder_name):
    messages = BASE_PROMPTS + [
        {
            "role": "user",
            "content": INPUT_PROMPT,
        },
    ]
    # list folder with all files ending with .py
    files = [f for f in os.listdir(folder_name) if f.endswith(".py")]
    for file in files:
        with open(f"{folder_name}/{file}", "r") as f:
            messages.append({
                "role": "file",
                "content": f.read()
            })
    # return messages
    return messages

def fix_messages(folder_name, err_message):
    messages = codegen_messages(folder_name) + FIX_PROMPTS + [
        {
            "role": "user",
            "content": f"Error Message: {err_message}",
        },
    ]
    return messages


for iteration in range(args.start_iteration, args.max_iterations):

    if FIX_MODE:
        init_code_folder_name = f"{subfolder_name}/{(iteration-1):02d}"
        request_messages = fix_messages(init_code_folder_name, ERR_MESSAGE)
    else:
        init_code_folder_name = f"{subfolder_name}"
        request_messages = codegen_messages(init_code_folder_name)

    # create a folder for the code and logs, starts with code_gen_<timestamp>
    code_folder_name = f"{subfolder_name}/{iteration:02d}"
    os.makedirs(code_folder_name, exist_ok=True)

    # write request_messages to log file
    with open(f"{code_folder_name}/api_request.log", "w") as f:
        f.write(json.dumps(request_messages, indent=4))

    response_message = API_ENDPOINTS[args.api](request_messages)

    # for all files that named "prompt.txt" or ending with ".py" in the init_code_folder_name, read the content and write to code_folder_name
    for file in os.listdir(init_code_folder_name):
        if file == "prompt.txt" or file.endswith(".py"):
            with open(f"{init_code_folder_name}/{file}", "r") as f:
                content = f.read()
            with open(f"{code_folder_name}/{file}", "w") as f:
                f.write(content)

    # write response to log file
    with open(f"{code_folder_name}/api_response.log", "w") as f:
        f.write(json.dumps(response_message, indent=4))

    with open(f"{code_folder_name}/api_response.md", "w") as f:
        f.write(f"API: {args.api}\n")
        f.write(f"Test Case: {args.dir}\n")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        f.write(f"Timestamp: {timestamp}\n\n")

        if "content" in response_message:
            # if response_message has a "reasoner-content" key, write it to the log file
            if "reasoning_content" in response_message:
                f.write("<think>\n")
                f.write(response_message["reasoning_content"])
                f.write("\n</think>\n")

            f.write(response_message["content"])
        else:
            f.write(json.dumps(response_message, indent=4))

    # match response.text with regex and extract the code blocks
    import re

    code_blocks = re.findall(r'```python\n(.*?)```', response_message['content'], re.DOTALL)

    # print(code_blocks)
    # write the code blocks to files
    for code_block in code_blocks:
        filename = code_block.split("\n")[0].split("# filename: ")[1]
        print(f"writing file: {filename}")
        with open(f"{code_folder_name}/{filename}", "w") as f:
            f.write(code_block)

    # extract test commands from response.text
    test_commands = re.findall(r'.*?```bash\n(.*?)```.*?', response_message['content'], re.DOTALL)
    # get the first test command if available
    if len(test_commands) > 0:
        TEST_COMMANDS = test_commands[0].strip()
    cmd = f"bash -c 'cd {code_folder_name} && {TEST_COMMANDS}'"
    print("--------------------------------")
    print(cmd)
    print("--------------------------------")
    # run command and print output and error separately, use subprocess.run
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
    OUTPUT_MESSAGE = process.stdout.decode('utf-8')
    if OUTPUT_MESSAGE != "":
        with open(f"{code_folder_name}/code_stdout.log", "w") as f:
            f.write(OUTPUT_MESSAGE)
    print(OUTPUT_MESSAGE)
    print("--------------------------------")
    ERR_MESSAGE = process.stderr.decode('utf-8')
    if ERR_MESSAGE != "":
        with open(f"{code_folder_name}/code_stderr.log", "w") as f:
            f.write(ERR_MESSAGE)
    print(ERR_MESSAGE)
    print("--------------------------------")
    # get the exit code
    exit_code = process.returncode
    if exit_code != 0:
        print(f"Test failed with exit code {exit_code}")
        print("--------------------------------")
        print("Fixing code...")
        print("--------------------------------")
        FIX_MODE = True
    else:
        print(f"Test passed with exit code {exit_code}")
        print("--------------------------------")
        break
