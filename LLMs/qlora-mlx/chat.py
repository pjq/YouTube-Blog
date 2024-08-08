import subprocess
import logging
from mlx_lm import load, generate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command_with_live_output(command: list[str]) -> None:
    """
    Courtesy of ChatGPT:
    Runs a command and prints its output line by line as it executes.

    Args:
        command (List[str]): The command and its arguments to be executed.

    Returns:
        None
    """
    logging.info(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print the output line by line
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output.strip())
        
    # Print the error output, if any
    err_output = process.stderr.read()
    if err_output:
        logging.error(err_output)

def construct_shell_command(command: list[str]) -> str:
    return str(command).replace("'","").replace("[","").replace("]","").replace(",","")

# Define paths and parameters
model_path = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
adapter_path = "adapters.npz"
max_tokens = 140

# Prompt builder function
instructions_string = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
请把现代汉语翻译成古文

### Input:
{comment}

### Response:
"""
def prompt_builder(comment: str) -> str:
    return f'''<s>[INST] {instructions_string.format(comment=comment)} [/INST]\n'''


# Interactive chat loop
while True:
    comment = input("You: ")
    if comment.lower() in ['exit', 'quit']:
        logging.info("Exiting the chat loop.")
        break

    prompt = prompt_builder(comment)
    logging.info(f"Generated prompt: {prompt}")

    # Define command
    command = ['python', 'scripts/lora.py', '--model', model_path, '--adapter-file', adapter_path, '--max-tokens', str(max_tokens), '--prompt', prompt]

    # Run command and print results continuously
    run_command_with_live_output(command)