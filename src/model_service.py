from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
import logging


class TextRequest(BaseModel):
    text: str


class ConfigRequest(BaseModel):
    prompt: str
    do_sample: bool = Field(default=True)
    top_k: int = Field(default=40)
    temperature: float = Field(default=0.6)
    max_new_tokens: int = Field(default=600)


app = FastAPI()

# Load models once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the embedding model
print("Loading QA model...")
logging.getLogger("transformers").setLevel(logging.INFO)

embedding_model_name = "LABSE"
embedder = SentenceTransformer(embedding_model_name).to(device)
print(f'loaded embedding model from {embedding_model_name}')

# Load the main language model
model_name = "lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print(f'loaded model from {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name)


# API Endpoints
@app.get("/health")
def health_check():
    return {"status": "running"}


@app.post("/generate/")
def generate_text(request: ConfigRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    generation_config = GenerationConfig(
        do_sample=request.do_sample,
        top_k=request.top_k,
        temperature=request.temperature,
        max_new_tokens=request.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )
    outputs = model.generate(inputs["input_ids"], generation_config=generation_config)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": answer}


@app.post("/embed/")
def embed_text(request: TextRequest):
    embedding = embedder.encode([request.text]).tolist()
    return {"embedding": embedding}


# Run FastAPI on port 5000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
