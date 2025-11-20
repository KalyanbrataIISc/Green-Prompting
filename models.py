import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from llama_cpp import Llama

device = "cpu"


def load_distilgpt2():
    """
    Load DistilGPT-2 as decoder-only causal LM.
    Returns: (model, tokenizer, model_type)
    """
    name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok, "decoder_only"


def load_flan_t5_small():
    """
    Load Flan-T5-Small as encoder-decoder model.
    Returns: (model, tokenizer, model_type)
    """
    name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
    return model, tok, "encoder_decoder"


def load_llama3_1b():
    """
    Load Llama-3.2-1B-Instruct-Q8_0.gguf using llama-cpp-python.
    Assumes the file is in project root.
    """
    model_path = "Llama-3.2-1B-Instruct-Q8_0.gguf"

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=4,
        logits_all=False,
        n_gpu_layers=0,
    )

    return llm, None, "llama_cpp"