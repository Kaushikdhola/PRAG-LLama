# Use a pipeline as a high-level helper
from transformers import pipeline
from huggingface_hub import login


login(token="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")

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")