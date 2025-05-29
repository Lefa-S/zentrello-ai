from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI()

# Load tokenizer and model once on startup
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    trust_remote_code=True,
    device_map="auto",
)

class InputText(BaseModel):
    prompt: str

@app.post("/generate_reply")
async def generate_reply_endpoint(data: InputText):
    input_text = f"Summarize this email briefly:\n\n{data.prompt}\n\nSummary:"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        use_cache=False
    )

    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary_text.split("Summary:")[-1].strip()

    return {"summary": summary if summary else "No reply generated."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)