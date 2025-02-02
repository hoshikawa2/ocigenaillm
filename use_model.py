import torch
import textwrap
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = LlamaForCausalLM.from_pretrained("/Users/cristianohoshikawa/Projects/Python/huggingface/trained_model")
tokenizer = AutoTokenizer.from_pretrained("/Users/cristianohoshikawa/Projects/Python/huggingface/trained_model")

# Check if GPU is available and move the model to GPU

#NVidia
# device = 0 if torch.cuda.is_available() else -1
#Mac Silicon
device = torch.device("mps")

text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# Test the model
prompt = "Como usar um adapter REST no Oracle Integration passo-a-passo?"

generated_text = text_generation(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9)[0]["generated_text"]

# Format the output for better readability

print("\nGenerated Text:\n")
print(generated_text)