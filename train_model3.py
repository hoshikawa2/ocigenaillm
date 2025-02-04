# https://drlee.io/step-by-step-guide-fine-tuning-metas-llama-3-2-1b-model-f1262eda36c8

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import PyPDF2

def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_chapter(manual, index, next_index):
    index_time = 0
    start_idx = 0
    end_idx = False
    idx = 0
    sub_text = ""
    manual = manual.replace("\n", "")
    while idx < manual.__len__():
        if manual[idx:idx+index.__len__()] == index:
            if index_time > 0 and start_idx == 0:
                start_idx = idx
            else:
                index_time = index_time + 1
        if manual[idx:idx+next_index.__len__()] == next_index:
            if index_time > 0 and start_idx > 0:
                end_idx = True
            else:
                index_time = index_time + 1
        if start_idx > 0 and end_idx:
            sub_text = sub_text + manual[start_idx:idx]
            if (sub_text.__len__() > index.__len__() + 10 and sub_text.__len__() < 4096):
                break
            else:
                if sub_text.__len__() >= 4096:
                    sub_text = sub_text[0:4096]
                else:
                    sub_text = ""
                break
        idx = idx + 1
    return sub_text

def read_indexes(file_path):
    list_index = []
    with open(file_path, 'rb') as file:
        for line in file.readlines():
            title = treat_line(line)
            list_index.append(title)
    return list_index

def treat_line(line):
    line = remove_index(line)
    line = trim_index(line)
    return line
def remove_index(line):
    is_number = False
    is_space = False
    is_roman = False
    idx = line.__len__() - 1
    while idx >= 0:
        if line[idx:idx+1].decode() == "v" or line[idx:idx+1].decode() == "x" or line[idx:idx+1].decode() == "i":
            is_roman = True
        if line[idx:idx+1].decode().isnumeric():
            is_number = True
        if line[idx:idx+1].decode() == " ":
            is_space = True
        if (is_number or is_roman) and is_space:
            break
        idx = idx - 1
    return line[0:idx].decode()

def trim_index(line):
    line = line.strip()
    return line

def mount_question_answer(index_file, manual_file):
    indexes = read_indexes(index_file)
    manual = extract_text_from_pdf(manual_file)
    prompt = []
    response = []

    idx = 0
    while idx < indexes.__len__() - 1:
        chapter = extract_chapter(manual, indexes[idx], indexes[idx + 1])
        if chapter != "":
            if count_words(chapter) <= 2048:
                prompt.append("How to " + indexes[idx] + "?")
                response.append(chapter)
            else:
                while count_words(chapter) > 2048:
                    first_strs, chapter = extract_words(chapter, 2048)
                    prompt.append("How to " + indexes[idx] + "?")
                    response.append(first_strs)

        idx = idx + 1
    return prompt, response

def count_words(line):
    return len(line.split())

def extract_words(line, words):
    l = 0
    spaces = 0
    sub_str = ""
    rest_str = ""
    while l < line.__len__():
        if line[l:l+1] == " ":
            spaces = spaces + 1
        if spaces >= words:
            sub_str = sub_str + line[0:l]
            rest_str = line[l + 1:line.__len__()]
            return sub_str, rest_str
        l = l + 1
    sub_str = line
    rest_str = ""
    return sub_str, rest_str

# Load the model and tokenizer
model_id = '/Users/cristianohoshikawa/Projects/Python/huggingface/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id)

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# pdf_files = [    './Manuals/using-integrations-oracle-integration-3.pdf',
#                  './Manuals/SOASUITE.pdf',
#                  './Manuals/SOASUITEHL7.pdf'
#                  ]
#
# # Define prompts and responses
# prompts = [
#     "Explain quantum computing in simple terms.",
#     "What are the benefits of renewable energy?",
#     "Describe the process of photosynthesis.",
#     "What is the significance of the Renaissance?",
#     "How does the human immune system work?"
# ]
# responses = [
#     "Quantum computing uses quantum bits, or qubits, to perform calculations. Unlike classical bits that are either 0 or 1, qubits can exist in multiple states simultaneously, allowing quantum computers to solve certain complex problems faster.",
#     "Renewable energy, such as solar and wind, reduces greenhouse gas emissions, decreases air pollution, and conserves natural resources. It also promotes energy independence and sustainability.",
#     "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water. It occurs in the chloroplasts, producing oxygen as a byproduct.",
#     "The Renaissance was a cultural movement from the 14th to the 17th century, characterized by a renewed interest in classical art, science, and philosophy. It led to significant advancements in many fields and a shift towards humanism.",
#     "The human immune system protects the body from infections and diseases. It consists of physical barriers, immune cells, and proteins that identify and destroy pathogens like bacteria and viruses."
# ]

prompt_master, response_master = mount_question_answer("./index/OIC_Index.txt", './Manuals/using-integrations-oracle-integration-3.pdf')

# Set a consistent max length
max_length = 2048

counter = prompt_master.__len__()

while counter >= 0:

    sub_counter = 0
    prompts = []
    responses = []
    while sub_counter < 1:
        prompts.append(prompt_master[sub_counter])
        responses.append(response_master[sub_counter])
        sub_counter = sub_counter + 1
    counter = counter - 1

    # Tokenize prompts and responses
    tokenized_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_labels = tokenizer(responses, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]

    # Ensure labels' padding tokens are ignored in loss computation
    tokenized_labels[tokenized_labels == tokenizer.pad_token_id] = -100

    from torch.utils.data import Dataset

    # Create a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return len(self.inputs["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": self.inputs["input_ids"][idx],
                "attention_mask": self.inputs["attention_mask"][idx],
                "labels": self.labels[idx]
            }

    # Instantiate the dataset
    dataset = CustomDataset(tokenized_inputs, tokenized_labels)

    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

    # Data collator to handle padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Increase the number of training epochs if necessary
    training_args = TrainingArguments(
        output_dir="/Users/cristianohoshikawa/Projects/Python/huggingface/results",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        num_train_epochs=2,  # Increase to 5 or more
        weight_decay=0.01
    )


    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Start training
    try:
        trainer.train()
    except ValueError as e:
        print("\nError during training:")
        print(e)

    # Save the model and tokenizer
    model.save_pretrained("/Users/cristianohoshikawa/Projects/Python/huggingface/trained_model")
    tokenizer.save_pretrained("/Users/cristianohoshikawa/Projects/Python/huggingface/trained_model")

print("Model and tokenizer saved successfully!")

