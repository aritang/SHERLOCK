from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="model-cache")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="model-cache")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("using device", device)

input_text = "The meaning of life is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
