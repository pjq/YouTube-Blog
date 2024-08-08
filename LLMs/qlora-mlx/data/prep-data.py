import csv
import random
import json
import logging
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define parameters
num_test = 10
num_val = 10

# Load dataset from Hugging Face
logging.info("Loading dataset from Hugging Face")
dataset = load_dataset("AISPIN/shiji-70liezhuan")
logging.info("Dataset loaded successfully")

# Prompt format
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Prepare examples
logging.info("Preparing examples")
example_template = lambda instruction, input_text, output: alpaca_prompt.format(instruction, input_text, output)

example_list = []
for example in dataset['train']:
    formatted_example = {"text": example_template(example['instruction'], example['input'], example['output'])}
    example_list.append(formatted_example)

logging.info(f"Total examples prepared: {len(example_list)}")
logging.info(f"Last example: {example_list[-1]}")

# Create test and validation data
logging.info("Creating test and validation data")
test_val_index_list = random.sample(range(0, len(example_list)-1), num_test+num_val)

test_list = [example_list[index] for index in test_val_index_list[:num_test]]
val_list = [example_list[index] for index in test_val_index_list[num_test:]]

for example in test_list+val_list:
    example_list.remove(example)

logging.info(f"Number of training examples: {len(example_list)}")
logging.info(f"Number of test examples: {len(test_list)}")
logging.info(f"Number of validation examples: {len(val_list)}")

# Write train, test, and validation data to file
logging.info("Writing training data to train.jsonl")
with open('train.jsonl', 'w') as outfile:
    for example in example_list:
        json.dump(example, outfile)
        outfile.write('\n')

logging.info("Writing test data to test.jsonl")
with open('test.jsonl', 'w') as outfile:
    for example in test_list:
        json.dump(example, outfile)
        outfile.write('\n')

logging.info("Writing validation data to valid.jsonl")
with open('valid.jsonl', 'w') as outfile:
    for example in val_list:
        json.dump(example, outfile)
        outfile.write('\n')

logging.info("Data preparation complete")