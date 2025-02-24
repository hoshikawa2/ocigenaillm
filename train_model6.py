# https://drlee.io/step-by-step-guide-fine-tuning-metas-llama-3-2-1b-model-f1262eda36c8
# primeiro voce treina o modelo com informações cruciais para o que deseja
# depois voce utiliza este modelo pre-treinado para utilizar com os PDFs
# usar o modelo nos codigos que consomem os PDFs

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_id = '/Users/cristianohoshikawa/Projects/Python/huggingface/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define prompts and responses
prompts = [
    "Explique o que é o SOA SUITE",
    "Explique o que é o Oracle Integration"
]
responses = [
    "SOA SUITE é uma ferramenta da Oracle utilizada para construir integrações entre sistemas e tecnologias. Através de uma IDE (JDeveloper) consegue-se criar integrações através de design de fluxos, interligando as tecnologias e sistemas através de adaptadores",
    "Oracle Integration é uma ferramenta, evoluída do SOA SUITE utilizada também para construir integrações entre sistemas e tecnologias. Mais no ambito Cloud porém com possibilidades de integrar ambientes on-premises e com outras clouds. Oracle Integration (OIC) também permite utilização de adaptadores e a construção também é feita através de fluxos gráficos, porém em ambiente web, não há IDE para desenvolvimento local."
]

# Set a consistent max length
max_length = 50

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
    per_device_train_batch_size=2,
    num_train_epochs=3,  # Increase to 5 or more
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
