import os
import torch
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve token securely from environment variables or Streamlit secrets
HF_TOKEN = os.getenv("HF_TOKEN", "")
try:
    import streamlit as st
    if not HF_TOKEN and hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
        HF_TOKEN = str(st.secrets["HF_TOKEN"])
except Exception:
    pass

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

# Enable GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit quantization config (only initialized when CUDA is available)
BNB_CONFIG = None
if torch.cuda.is_available():
    try:
        from transformers import BitsAndBytesConfig
        BNB_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    except Exception:
        BNB_CONFIG = None
