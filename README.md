# **Green Prompting for Lightweight LLMs**
### *Evaluating Energy–Accuracy Trade-offs in CPU-Deployable Models*

This project investigates how different prompting strategies affect the **accuracy**, **token efficiency**, and **energy consumption** of small, CPU-friendly language models.  
It extends the methodology of:

**Rubei et al. (2025). Prompt Engineering and Its Implications on the Energy Consumption of Large Language Models.**

The goal is to evaluate *green prompting* techniques that improve performance while minimizing computational cost on systems without GPUs (e.g., GitHub Codespaces).

---

# **1. Models Evaluated**

This project compares three lightweight LLM families:

| Model | Type | Parameters | Notes |
|-------|------|------------|-------|
| **DistilGPT-2** | Decoder-only | 82M | Very small, weak, baseline |
| **Flan-T5-Small** | Encoder–decoder | 77M | Best for summarization and energy efficiency |
| **Llama-3.2-1B-Instruct-Q8_0** | Decoder-only | 1B (quantized) | Strongest accuracy, most energy hungry |

All models run **entirely on CPU**.

---

# **2. Tasks Evaluated**

The experiments evaluate models on three standard NLP benchmarks:

### **1. BoolQ (Yes/No Question Answering)**  
- Metric: **Accuracy**

### **2. GSM8K (Math Word Reasoning)**  
- Metric: **Numeric exact match**

### **3. XSum (Abstractive Summarization)**  
- Metrics: **ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum**

---

# **3. Prompt Styles Tested**

For each model–task pair, the following prompt types are evaluated:

| Prompt Style | Description |
|--------------|-------------|
| **Zero-shot** | Direct question or instruction |
| **One-shot** | One worked example is provided |
| **Few-shot** | (optional) multi-example prompts |
| **Chain-of-thought** | Step-by-step reasoning |
| **Concise summary** | Single-sentence summary request |
| **Verbose summary** | More descriptive instructions |

Not all styles apply to all tasks.

---

# **4. Metrics Recorded**

Each inference records:

### **Performance**
- Accuracy (BoolQ, GSM8K)
- ROUGE scores (XSum)
- Performance-per-100-tokens  
  - Meaning: `(accuracy or ROUGE-L) / (avg_tokens / 100)`

### **Efficiency**
- **avg_tokens**: total input + output tokens
- **avg_wall_time_s**: total latency
- **avg_energy_j**: estimated CPU energy (using psutil CPU time x estimated watts)

### **Energy Proxy**
Energy (E) is estimated from:
```
E = CPU_time_seconds × 25 watts
```
Using:
```python
psutil.Process().cpu_times()
```

---

# **5. Project Structure**

```
Green-Prompting/
│
├─ data_loading.py          # Load BoolQ, GSM8K, XSum
├─ prompts.py               # Prompt templates for all tasks
├─ metrics.py               # Inference + energy + performance metrics
├─ models.py                # HF + Llama.cpp model loaders
├─ run_experiments.py       # Main experiment runner
├─ plots.py                 # Generates all figs with one command
│
├─ results/
│   └─ all_results.csv      # Final combined table
│
└─ figs/
    ├─ boolq_accuracy_bar.png
    ├─ boolq_accuracy_vs_energy.png
    ├─ ...
    ├─ xsum_rougeL_vs_energy.png
    └─ (many more)
```

---

# **6. Installation**

### **1. Clone this repo**
```bash
git clone https://github.com/KalyanbrataIISc/Green-Prompting.git
cd Green-Prompting
```

### **2. Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Add the Llama 3.2 model**
Place the quantized model in your root folder:

```
Llama-3.2-1B-Instruct-Q8_0.gguf
```

---

# **7. Running Experiments**

### **Run all experiments**
```bash
python run_experiments.py
```

This will:
- Load datasets  
- Run BoolQ / GSM8K / XSum  
- Evaluate all models in all prompt styles  
- Compute metrics  
- Save everything to:
```
results/all_results.csv
```

---

# **8. Generating Plots**

Just run:
```bash
python plots.py
```

Figures are saved in:
```
figs/
```

---

# **9. Key Findings**

### **BoolQ**
- Llama-3.2-1B: **best accuracy**  
- Flan-T5: **best energy efficiency**

### **GSM8K**
- All small models struggle  
- Llama performs slightly better  

### **XSum**
- Flan-T5 has highest ROUGE scores  
- Llama uses **>25x** more energy

---

# **10. Citation**

```
Rubei et al. (2025). Prompt Engineering and Its Implications on the Energy Consumption of Large Language Models.
```
