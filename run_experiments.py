import pandas as pd

from data_loading import load_boolq, load_gsm8k, load_xsum
from models import load_distilgpt2, load_flan_t5_small, load_llama3_1b

from prompts import (
    boolq_zero_shot,
    boolq_one_shot,
    boolq_cot,
    gsm8k_zero_shot,
    gsm8k_cot,
    xsum_concise,
    xsum_verbose,
)

from metrics import eval_boolq, eval_gsm8k, eval_xsum


def main():
    # Load datasets
    boolq = load_boolq(100)
    gsm8k = load_gsm8k(80)
    xsum = load_xsum(80)

    # Load models
    gpt2_model, gpt2_tok, gpt2_type = load_distilgpt2()
    t5_model, t5_tok, t5_type = load_flan_t5_small()
    llama_model, _, llama_type = load_llama3_1b()

    results = []

    # ----------------- BoolQ -----------------
    for model, tok, mtype, mname in [
        (gpt2_model, gpt2_tok, gpt2_type, "distilgpt2"),
        (t5_model, t5_tok, t5_type, "flan-t5-small"),
        (llama_model, None, llama_type, "llama-3.2-1B-Q8"),
    ]:
        for name, fn in [
            ("zero-shot", boolq_zero_shot),
            ("one-shot", boolq_one_shot),
        ]:
            summary = eval_boolq(boolq, model, tok, fn, mtype, n=40)
            results.append(
                {
                    "model": mname,
                    "prompt_style": name,
                    "task": "BoolQ",
                    **summary,
                }
            )

        if mname == "distilgpt2":
            summary = eval_boolq(boolq, model, tok, boolq_cot, mtype, n=40)
            results.append(
                {
                    "model": mname,
                    "prompt_style": "cot",
                    "task": "BoolQ",
                    **summary,
                }
            )

    # ----------------- GSM8K -----------------
    for model, tok, mtype, mname in [
        (gpt2_model, gpt2_tok, gpt2_type, "distilgpt2"),
        (t5_model, t5_tok, t5_type, "flan-t5-small"),
        (llama_model, None, llama_type, "llama-3.2-1B-Q8"),
    ]:
        for name, fn in [
            ("zero-shot", gsm8k_zero_shot),
            ("cot", gsm8k_cot),
        ]:
            summary = eval_gsm8k(gsm8k, model, tok, fn, mtype, n=30)
            results.append(
                {
                    "model": mname,
                    "prompt_style": name,
                    "task": "GSM8K",
                    **summary,
                }
            )

    # ----------------- XSum -----------------
    for model, tok, mtype, mname in [
        (t5_model, t5_tok, t5_type, "flan-t5-small"),
        (llama_model, None, llama_type, "llama-3.2-1B-Q8"),
    ]:
        for name, fn in [
            ("concise", xsum_concise),
            ("verbose", xsum_verbose),
        ]:
            summary = eval_xsum(xsum, model, tok, fn, mtype, n=30)
            results.append(
                {
                    "model": mname,
                    "prompt_style": name,
                    "task": "XSum",
                    **summary,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv("results/all_results.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()