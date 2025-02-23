import subprocess

def run_and_print_live(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    while True:
        output = process.stdout.readline().decode('utf-8').rstrip()
        if output:
            print(output)  # Print stdout as it becomes available

        if process.poll() is not None:  # Check if process has finished
            break

# Example usage
command = "ping 8.8.8.8"
run_and_print_live(command)
