Green Prompting for Lightweight LLMs
======================================

_Evaluating Energy–Accuracy Trade-offs in CPU-Deployable Models_

## Project Overview

Large language models (LLMs) are powerful but computationally expensive. Recent work (Rubei et al., 2025) shows that prompt engineering significantly affects the energy consumption, latency, and accuracy of LLMs, yet most prior studies target large GPU-backed models. This project extends the investigation to small, CPU-friendly models that fit lightweight systems or edge devices, while measuring both accuracy and energy proxies.

### Models Evaluated

- **DistilGPT-2** – causal, small, and fast on CPU
- **Flan-T5-Small** – encoder-decoder, instruction-tuned
- **Optional:** Quantized LLaMA-2-7B (4-bit)

These models run entirely on CPU within a GitHub Codespace deployment.

## Research Questions

- How do prompt styles (zero-shot, few-shot, chain-of-thought, concise, verbose) affect accuracy, latency, token count, memory usage, and CPU-time-dependent energy? 
- Can we define a Performance-per-Token (or Performance-per-100-Tokens) metric that captures both utility and efficiency? 
- How do small models behave under “green prompting” strategies compared to the large LLMs studied by Rubei et al.?

## Tasks Evaluated

| Task | Dataset | Metric |
| --- | --- | --- |
| **BoolQ** | Yes/No question answering | Accuracy |
| **GSM8K** | Math reasoning | Accuracy |
| **XSum** | Summarization | ROUGE-1/2/L |

Every task is evaluated with multiple prompting strategies.

## Prompt Styles

- Zero-shot
- One-/few-shot
- Chain-of-thought (CoT)
- Concise vs verbose instructions

Each style influences token length, reasoning depth, and compute cost.

## What the Code Does

| Module | Responsibility |
| --- | --- |
| `data_loading.py` | Load and prepare dataset subsets. |
| `models.py` | Load CPU-only models and tokenizers. |
| `prompts.py` | Store prompt templates for every task and style. |
| `metrics.py` | Run inference, count tokens, estimate energy, and compute scores. |
| `run_experiments.py` | Orchestrate experiments and persist results. |

### Workflow

1. Load BoolQ subsets (currently 100 examples).
2. Load DistilGPT-2 and Flan-T5-Small into CPU memory.
3. For each prompt style:
   - Build the prompt for every example. 
   - Run model inference on the CPU.
   - Measure input/output/total tokens, wall time, CPU time, and energy proxy (`energy_j = cpu_time * 25W`).
   - Extract model answers, compute accuracy, and compute Performance-per-100-Tokens.
4. Save results to `results/boolq_results.csv`, which contains the columns `model`, `prompt_style`, `accuracy`, `avg_energy_j`, `avg_tokens`, `perf_per_100_tokens`, `avg_wall_time_s`.

## Energy Measurement

GitHub Codespaces lack real hardware power counters, so energy is approximated as:

```
energy_joules = cpu_time_seconds * 25W
```

This follows the methodology from Rubei et al., with the stated limitation about the proxy.

## Repository Structure

```
/your-project-folder
  data_loading.py
  models.py
  prompts.py
  metrics.py
  run_experiments.py
  results/
      boolq_results.csv   (generated after experiments)
  figs/
      (plots go here)
  README.md
```

## Installation and Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "transformers>=4.44.0" "datasets==2.20.0" accelerate
```

## Running Your First Experiment

From the project directory:

```bash
python run_experiments.py
```

### Expected Behavior

- Scripts download datasets and models on the first run.
- BoolQ evaluation takes ~2–4 minutes on a CPU.
- Results print as a table and appear in `results/boolq_results.csv`.
- Example table structure:

```
             model    prompt_style   task    accuracy    avg_tokens   perf_per_100_tokens   avg_energy_j   avg_wall_time_s
0      distilgpt2      zero-shot     BoolQ      ...
1      distilgpt2      one-shot      BoolQ      ...
2      distilgpt2      cot           BoolQ      ...
3  flan-t5-small      zero-shot      BoolQ      ...
4  flan-t5-small      one-shot       BoolQ      ...
```

## Next Steps (After BoolQ Works)

1. Add GSM8K benchmark and implement numeric answer extraction.
2. Evaluate reasoning performance and compare zero-shot vs CoT.
3. Add XSum with ROUGE scoring and contrast concise vs verbose summarization prompts.
4. Plot accuracy vs energy, accuracy vs wall time, tokens vs accuracy, and perf-per-token vs energy, saving visuals to `figs/`.

## Using These Results in a Paper

- **Methodology:** Describe datasets/subsets, models, prompt styles, inference/metrics, and the energy proxy formula.
- **Results:** Insert tables from CSVs and the plots created above. Compare prompting strategies.
- **Discussion:** Cover accuracy trade-offs, overall efficiency, whether CoT is worth it at small scale, and how your small models differ from Rubei et al.’s large models.
- **Conclusion:** Highlight that concise prompts usually offer the best efficiency, CoT boosts accuracy only sometimes but increases energy, and Performance-per-Token aggregates utility and cost.

## Reproducibility

All randomness is fixed with `seed=42`, so results should be deterministic across machines.

## Citation

If referencing the inspiration paper:

> Rubei et al. (2025). “Prompt Engineering and Its Implications on the Energy Consumption of Large Language Models.”