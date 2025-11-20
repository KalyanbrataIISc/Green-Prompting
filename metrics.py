import time
import re
import psutil
import torch
import evaluate

# Estimated CPU power for energy proxy (in watts)
P_EST_WATTS = 25.0

# Process handle for CPU usage
process = psutil.Process()

# ROUGE metric (used for XSum)
rouge = evaluate.load("rouge")


def run_inference_with_metrics(
    model,
    tokenizer,
    prompt_text,
    model_type="decoder_only",
    max_new_tokens=64,
):
    """
    Run a single inference and measure:
    - input_tokens, output_tokens, total_tokens
    - wall_time_s, cpu_time_s, energy_j

    model_type:
        - "decoder_only"     : HF causal LM (e.g. DistilGPT-2)
        - "encoder_decoder"  : HF seq2seq (e.g. Flan-T5)
        - "llama_cpp"        : llama-cpp Llama instance
    """

    # Branch 1 - Llama 3.2 1B via llama-cpp-python
    if model_type == "llama_cpp":
        # Count input tokens
        input_tokens = model.tokenize(prompt_text.encode("utf-8"))
        input_len = len(input_tokens)

        cpu_before = process.cpu_times()
        t0 = time.perf_counter()

        # Deterministic generation
        out = model(
            prompt_text,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        t1 = time.perf_counter()
        cpu_after = process.cpu_times()

        output_text = out["choices"][0]["text"]

        # Usage info may contain counts
        usage = out.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", None)
        completion_tokens = usage.get("completion_tokens", None)

        if prompt_tokens is None:
            prompt_tokens = input_len
        if completion_tokens is None:
            completion_tokens = len(model.tokenize(output_text.encode("utf-8")))

        total_tokens = prompt_tokens + completion_tokens

        cpu_time = (
            (cpu_after.user - cpu_before.user)
            + (cpu_after.system - cpu_before.system)
        )
        energy = cpu_time * P_EST_WATTS

        return output_text, {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "wall_time_s": t1 - t0,
            "cpu_time_s": cpu_time,
            "energy_j": energy,
        }

    # Branch 2 - Hugging Face models (DistilGPT-2, Flan-T5)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    cpu_before = process.cpu_times()
    t0 = time.perf_counter()

    with torch.no_grad():
        if model_type == "decoder_only":
            # Decoder-only LMs: output includes prompt
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_tokens = out_ids[:, input_len:]
        else:
            # Encoder-decoder models: output is just the generated sequence
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_tokens = out_ids

    t1 = time.perf_counter()
    cpu_after = process.cpu_times()

    output_text = tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True,
    )
    out_len = new_tokens.shape[1]

    cpu_time = (
        (cpu_after.user - cpu_before.user)
        + (cpu_after.system - cpu_before.system)
    )
    energy = cpu_time * P_EST_WATTS

    return output_text, {
        "input_tokens": input_len,
        "output_tokens": out_len,
        "total_tokens": input_len + out_len,
        "wall_time_s": t1 - t0,
        "cpu_time_s": cpu_time,
        "energy_j": energy,
    }


def extract_yes_no(text: str):
    """
    Map a free-form model output to True (Yes) / False (No) / None.
    """
    t = text.lower().strip()

    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False

    if not t:
        return None

    first = t.split()[0]
    if first.startswith("y"):
        return True
    if first.startswith("n"):
        return False

    return None


def eval_boolq(ds, model, tok, prompt_fn, model_type, n=50):
    """
    Evaluate BoolQ on first n examples of `ds`.
    """
    correct = 0
    records = []

    for i in range(n):
        ex = ds[i]
        prompt = prompt_fn(ex["question"], ex["passage"], model_type=model_type)

        output_text, met = run_inference_with_metrics(
            model=model,
            tokenizer=tok,
            prompt_text=prompt,
            model_type=model_type,
        )

        pred = extract_yes_no(output_text)
        gold = bool(ex["answer"])
        if pred == gold:
            correct += 1

        records.append(met)

    accuracy = correct / n
    avg_tokens = sum(r["total_tokens"] for r in records) / n
    avg_energy = sum(r["energy_j"] for r in records) / n
    avg_wall_time = sum(r["wall_time_s"] for r in records) / n
    perf_per_100_tokens = accuracy / (avg_tokens / 100.0)

    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "perf_per_100_tokens": perf_per_100_tokens,
        "avg_energy_j": avg_energy,
        "avg_wall_time_s": avg_wall_time,
    }


def extract_final_number(text: str):
    """
    Extract the LAST integer or decimal number from text.
    Used for GSM8K numeric answer evaluation.
    """
    matches = re.findall(r"[-+]?\d+\.?\d*", text)
    if not matches:
        return None
    return matches[-1]


def eval_gsm8k(ds, model, tok, prompt_fn, model_type, n=50):
    """
    Evaluate GSM8K on first n examples.
    Accuracy = exact match of final numeric answer.
    """
    correct = 0
    records = []

    for i in range(n):
        ex = ds[i]
        problem = ex["question"]
        gold_str = ex["answer"]  # e.g. "#### 42"
        gold_num = extract_final_number(gold_str)

        prompt = prompt_fn(problem, model_type=model_type)

        output_text, met = run_inference_with_metrics(
            model=model,
            tokenizer=tok,
            prompt_text=prompt,
            model_type=model_type,
        )

        pred_num = extract_final_number(output_text)

        if pred_num == gold_num:
            correct += 1

        records.append(met)

    accuracy = correct / n
    avg_tokens = sum(r["total_tokens"] for r in records) / n
    avg_energy = sum(r["energy_j"] for r in records) / n
    avg_wall_time = sum(r["avg_wall_time_s"] for r in records) / n if "avg_wall_time_s" in records[0] else sum(r["wall_time_s"] for r in records) / n

    # Correct avg_wall_time key
    avg_wall_time = sum(r["wall_time_s"] for r in records) / n

    perf_per_100_tokens = accuracy / (avg_tokens / 100.0)

    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "perf_per_100_tokens": perf_per_100_tokens,
        "avg_energy_j": avg_energy,
        "avg_wall_time_s": avg_wall_time,
    }


def eval_xsum(ds, model, tok, prompt_fn, model_type, n=50):
    """
    Evaluate XSum summarization on first n examples.
    Metric: ROUGE (1/2/L) and efficiency.
    """
    preds = []
    refs = []
    metrics_list = []

    for i in range(n):
        ex = ds[i]
        doc = ex["document"]
        ref = ex["summary"]

        prompt = prompt_fn(doc, model_type=model_type)

        pred, met = run_inference_with_metrics(
            model=model,
            tokenizer=tok,
            prompt_text=prompt,
            model_type=model_type,
            max_new_tokens=64,
        )

        preds.append(pred)
        refs.append(ref)
        metrics_list.append(met)

    rouge_scores = rouge.compute(predictions=preds, references=refs)

    avg_tokens = sum(m["total_tokens"] for m in metrics_list) / n
    avg_energy = sum(m["energy_j"] for m in metrics_list) / n
    avg_wall_time = sum(m["wall_time_s"] for m in metrics_list) / n

    rougeL = rouge_scores["rougeL"]
    perf_per_100_tokens = rougeL / (avg_tokens / 100.0)

    summary = {
        **rouge_scores,
        "avg_tokens": avg_tokens,
        "perf_per_100_tokens": perf_per_100_tokens,
        "avg_energy_j": avg_energy,
        "avg_wall_time_s": avg_wall_time,
    }

    return summary