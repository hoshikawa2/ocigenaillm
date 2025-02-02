import torch
from langchain_community.document_loaders import PyPDFLoader
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import PyPDF2

# Extrair texto dos arquivos PDF

def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

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

# Lista de caminhos dos arquivos PDF
pdf_files = [    './Manuals/using-integrations-oracle-integration-3.pdf',
                 './Manuals/SOASUITE.pdf',
                 './Manuals/SOASUITEHL7.pdf'
                 ]

# Concatenar texto dos arquivos PDF
concatenated_text = ''
for file_path in pdf_files:
    text = extract_text_from_pdf(file_path)
    concatenated_text += text

# Tokenize o texto dos arquivos PDF
model_id = '/Users/cristianohoshikawa/Projects/Python/huggingface/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id)

max_length = 2048
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# while concatenated_text.__len__() > 0:
#     sub_concatenated_text = concatenated_text[0:2048000]
#     concatenated_text = concatenated_text[2048000:]
#     print(sub_concatenated_text)
tokenized_pdf_text = tokenizer(concatenated_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

# Criar o dataset com os dados dos arquivos PDF
dataset = CustomDataset(tokenized_pdf_text, tokenized_pdf_text["input_ids"])

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
    per_device_train_batch_size=40,
    num_train_epochs=3,  # Increase to 5 or more
    weight_decay=0.01
)

# Treinar o modelo com os dados dos arquivos PDF
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator)

# Iniciar o treinamento
try:
    result = trainer.train()
except ValueError as e:
    print("\nErro durante o treinamento:")
    print(e)

# Salvar o modelo e o tokenizer
model.save_pretrained("/Users/cristianohoshikawa/Projects/Python/huggingface/trained_model")
tokenizer.save_pretrained("/Users/cristianohoshikawa/Projects/Python/huggingface/trained_model")

print("Modelo e tokenizer salvos com sucesso!")
