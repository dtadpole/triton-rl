import os
import datetime
import subprocess
import json
from google import genai
from google.genai import types
from prompts import BASE_PROMPTS, FIX_PROMPTS
from gen_code_api import API_ENDPOINTS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--api", type=str, default="deepseek")
parser.add_argument("-d", "--dir", type=str, default=".")
parser.add_argument("-b", "--breadth", type=int, default=3)
parser.add_argument("-t", "--depth", type=int, default=3)
args = parser.parse_args()

if not os.path.exists(args.dir):
    print(f"Directory {args.dir} does not exist")
    exit(1)

if not os.path.exists(os.path.join(args.dir, "prompt.txt")):
    print(f"Prompt file {os.path.join(args.dir, "prompt.txt")} does not exist")
    exit(1)

INPUT_PROMPT = open(os.path.join(args.dir, "prompt.txt")).read()
TEST_COMMAND = ""
if os.path.exists(os.path.join(args.dir, "test.command")):
    with open(os.path.join(args.dir, "test.command"), "r") as f:
        TEST_COMMAND = f.read()

def get_folder_name(args):
    # get the next sequence id
    next_sequence_id = 0
    # check for next available subfolder name, starts with code_gen_, and ends with a number
    while os.path.exists(os.path.join(args.dir, f"code_gen_{args.api}_{next_sequence_id:03d}")):
        next_sequence_id += 1
    folder_name = os.path.join(args.dir, f"code_gen_{args.api}_{next_sequence_id:03d}")
    os.makedirs(folder_name, exist_ok=True)
    # for files with name "prompt.txt"  or ending with ".py" or ".stderr" or ".stdout" or ".failed" or ".success" or ".command " in the args.folder, read the content and write to subfolder_name
    for file in os.listdir(args.dir):
        if file == "prompt.txt" or file.endswith(".py") or file.endswith(".stderr") or file.endswith(".stdout") or file.endswith(".success") or file.endswith(".failed") or file.endswith(".command"):
            with open(os.path.join(args.dir, file), "r") as f:
                content = f.read()
            with open(os.path.join(folder_name, file), "w") as f:
                f.write(content)
    return folder_name

folder_name = get_folder_name(args)

# ITER_QUEUE is a list of parent and child folders
def initialize_iter_queue(args, folder_name):
    ITER_QUEUE = []
    # find start_iteration by checking folder names in start_folder_name
    start_iteration = 0
    for _ in range(args.breadth):
        while os.path.exists(os.path.join(folder_name, f"{start_iteration:02d}")):
            start_iteration += 1
        # append start_iteration to ITER_QUEUE
        child_config = {"parent": folder_name, "child": os.path.join(folder_name, f"{start_iteration:02d}"), "depth": 0}
        print(f"Adding {child_config} to ITER_QUEUE")
        ITER_QUEUE.append(child_config)
        start_iteration += 1
    return ITER_QUEUE

ITER_QUEUE = initialize_iter_queue(args, folder_name)

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
        with open(os.path.join(folder_name, file), "r") as f:
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


# WHILE ITER_QUEUE is not empty
while len(ITER_QUEUE) > 0:
    # pop the first item from ITER_QUEUE
    print("--------------------------------")
    iter_config = ITER_QUEUE.pop(0)
    print(f"Processing {iter_config}")
    print("--------------------------------")

    base_folder = iter_config["parent"]
    code_folder = iter_config["child"]

    # ERR_MESSAGE is the error message from the last iteration
    ERR_MESSAGE = ""
    if os.path.exists(os.path.join(base_folder, "code.stderr")):
        with open(os.path.join(base_folder, "code.stderr"), "r") as f:
            ERR_MESSAGE = f.read()

    # if there is a .failed file, set FIX_MODE to True
    if os.path.exists(os.path.join(iter_config["parent"], ".failed")):
        FIX_MODE = True
        request_messages = fix_messages(iter_config["parent"], ERR_MESSAGE)
    else:
        FIX_MODE = False
        request_messages = codegen_messages(iter_config["parent"])

    # create a folder for the code and logs, starts with code_gen_<timestamp>
    code_folder = iter_config["child"]
    os.makedirs(code_folder, exist_ok=True)

    # write request_messages to log file
    with open(os.path.join(code_folder, "api_request.json"), "w") as f:
        f.write(json.dumps(request_messages, indent=4))

    response_message = API_ENDPOINTS[args.api](request_messages)

    # for all files that named "prompt.txt" or ending with ".py" in the init_code_folder_name, read the content and write to code_folder_name
    for file in os.listdir(iter_config["parent"]):
        if file == "prompt.txt" or file.endswith(".py"):
            with open(os.path.join(iter_config["parent"], file), "r") as f:
                content = f.read()
            with open(os.path.join(code_folder, file), "w") as f:
                f.write(content)

    # write response to log file
    with open(os.path.join(code_folder, "api_response.json"), "w") as f:
        f.write(json.dumps(response_message, indent=4))

    with open(os.path.join(code_folder, "api_response.md"), "w") as f:
        f.write(f"API: {args.api}\n")
        f.write(f"Test Case: {args.dir}\n")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        f.write(f"Timestamp: {timestamp}\n\n")

        if "content" in response_message:
            # if response_message has a "reasoner-content" key, write it to the log file
            if "reasoning_content" in response_message and response_message["reasoning_content"]:
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
        print(f"writing file: {os.path.join(code_folder, filename)}")
        with open(os.path.join(code_folder, filename), "w") as f:
            f.write(code_block)

    # extract test commands from response.text
    test_commands = re.findall(r'.*?```bash\n(.*?)```.*?', response_message['content'], re.DOTALL)
    # get the first test command if available
    if len(test_commands) > 0:
        TEST_COMMAND = test_commands[0].strip()
        # write test command to log file
        with open(os.path.join(code_folder, "test.command"), "w") as f:
            f.write(TEST_COMMAND)
    cmd = f"bash -c 'cd {code_folder} && {TEST_COMMAND}'"
    print("--------------------------------")
    print(cmd)
    print("--------------------------------")
    # run command and print output and error separately, use subprocess.run
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
    STDOUT_MESSAGE = process.stdout.decode('utf-8')
    if STDOUT_MESSAGE != "":
        with open(os.path.join(code_folder, "code.stdout"), "w") as f:
            f.write(STDOUT_MESSAGE)
    print(STDOUT_MESSAGE)
    print("--------------------------------")
    STDERR_MESSAGE = process.stderr.decode('utf-8')
    if STDERR_MESSAGE != "":
        with open(os.path.join(code_folder, "code.stderr"), "w") as f:
            f.write(STDERR_MESSAGE)
    print(STDERR_MESSAGE)
    print("--------------------------------")
    # get the exit code
    exit_code = process.returncode
    if exit_code != 0:
        print(f"Test failed with exit code {exit_code} [{code_folder}]")
        # write ".failed" to the code_folder_name
        with open(os.path.join(code_folder, ".failed"), "w") as f:
            f.write(f"{exit_code}")
        print("--------------------------------")
        print(f"Fixing code... [{code_folder}]")
        # add the current folder to ITER_QUEUE
        if iter_config["depth"] < args.depth:
            iter_count = 0
            for _ in range(args.breadth):
                while os.path.exists(os.path.join(code_folder, f"{iter_count:02d}")):
                    iter_count += 1
                # append start_iteration to ITER_QUEUE
                child_config = {"parent": code_folder, "child": os.path.join(code_folder, f"{iter_count:02d}"), "depth": iter_config["depth"] + 1}
                print(f"Adding {child_config} to ITER_QUEUE")
                ITER_QUEUE.append(child_config)
                iter_count += 1
        print("--------------------------------")
    else:
        print(f"Test passed with exit code {exit_code} [{code_folder}]")
        # write ".success" to the code_folder_name
        with open(os.path.join(code_folder, ".success"), "w") as f:
            f.write(f"{exit_code}")
        print("--------------------------------")
        break
