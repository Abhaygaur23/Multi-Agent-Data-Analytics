import os
import torch
from src.config import HF_TOKEN, MODEL_NAME, BNB_CONFIG, DEVICE

# Global references for local model and API client
model = None
tokenizer = None
_hf_client = None

def get_hf_client(token=None):
    global _hf_client
    token = token or HF_TOKEN or os.getenv("HF_TOKEN")
    try:
        from huggingface_hub import InferenceClient
        if token:
            _hf_client = InferenceClient(token=token)
        else:
            _hf_client = InferenceClient()
        return _hf_client
    except Exception:
        return None

def load_model_and_tokenizer():
    global model, tokenizer
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"device_map": "auto"}
        if BNB_CONFIG is not None:
            kwargs["quantization_config"] = BNB_CONFIG
        elif DEVICE == "cpu":
            kwargs["torch_dtype"] = torch.float32

        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN if HF_TOKEN else None)
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Could not load local model ({e}). Orchestrator will use HF Inference API or rule-based routing.")
        return None, None

# Function to call the LLM (supports local model or HF Inference API)
def call_llm(prompt: str, max_tokens: int = 500, custom_token: str = None) -> str:
    global model, tokenizer

    # 1. Use local model if already loaded
    if model is not None and tokenizer is not None:
        try:
            full_prompt = (
                f"[INST] {prompt}\n\n"
                f"IMPORTANT: Your response must be a valid JSON object only. Format your response as proper JSON.\n"
                f"DO NOT include any text outside the JSON object.\n"
                f"Example of correct format: {{\"key\": \"value\", \"otherKey\": 123}}\n"
                f"Do not use markdown formatting for JSON. [/INST]"
            )
            inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "[/INST]" in response:
                response = response.split("[/INST]", 1)[1].strip()
            return response
        except Exception as e:
            print(f"Local LLM inference error: {e}")

    # 2. Try Hugging Face Serverless Inference API (ideal for Streamlit Cloud & CPU)
    token = custom_token or HF_TOKEN or os.getenv("HF_TOKEN")
    client = get_hf_client(token)
    if client:
        try:
            messages = [
                {"role": "system", "content": "You are a JSON-only data science orchestration assistant. Always respond with a valid JSON object."},
                {"role": "user", "content": prompt}
            ]
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )
            if resp and resp.choices:
                return resp.choices[0].message.content
        except Exception:
            try:
                return client.text_generation(prompt, model=MODEL_NAME, max_new_tokens=max_tokens)
            except Exception:
                pass

    return ""
