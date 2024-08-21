import transformers
import torch

## Here you paste your cloned repos location
model_id = "/Users/gabeschwartz/Documents/GitHub/NewsProject/Meta-Llama-3.1-8B-Instruct" 

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])