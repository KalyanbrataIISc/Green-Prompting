import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "results/all_results.csv"
FIGS_DIR = "figs"


# -------------------------------------------------------------------
# Style helpers: fixed colors per model, markers per prompt style
# -------------------------------------------------------------------

MODEL_COLORS = {
    "distilgpt2": "#1f77b4",        # blue
    "flan-t5-small": "#ff7f0e",     # orange
    "llama-3.2-1B-Q8": "#2ca02c",   # green
}

PROMPT_MARKERS = {
    "zero-shot": "o",
    "one-shot": "s",
    "cot": "D",
    "concise": "^",
    "verbose": "v",
}


def ensure_figs_dir():
    os.makedirs(FIGS_DIR, exist_ok=True)


def load_results():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"{RESULTS_CSV} not found. Run run_experiments.py first.")
    return pd.read_csv(RESULTS_CSV)


# -------------------------------------------------------------------
# Generic plotting utilities
# -------------------------------------------------------------------

def _bar_plot_metric(df, metric, ylabel, title, filename):
    """
    Bar plot: x = model/prompt, grouped and colored by model.
    """
    ensure_figs_dir()

    df_plot = df.dropna(subset=[metric]).copy()
    if df_plot.empty:
        print(f"[WARN] No data for metric '{metric}' for {title}.")
        return

    # Sort by model then prompt_style for consistent grouping
    df_plot = df_plot.sort_values(["model", "prompt_style"])

    labels = []
    heights = []
    colors = []

    for _, row in df_plot.iterrows():
        label = f"{row['model']} | {row['prompt_style']}"
        labels.append(label)
        heights.append(row[metric])
        colors.append(MODEL_COLORS.get(row["model"], "#333333"))

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), heights, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)

    # Build legend from unique models
    used_models = df_plot["model"].unique()
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS.get(m, "#333333"), lw=8)
        for m in used_models
    ]
    plt.legend(handles, used_models, title="Model", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def _scatter_perf_vs_energy(df, perf_col, perf_label, title, filename, log_energy=False):
    """
    Scatter: x = avg_energy_j, y = perf_col, colored by model, marker by prompt_style.
    """
    ensure_figs_dir()

    df_plot = df.dropna(subset=[perf_col, "avg_energy_j"]).copy()
    if df_plot.empty:
        print(f"[WARN] No data for {perf_col} vs energy in {title}.")
        return

    plt.figure(figsize=(10, 6))

    # For legend: collect used (model, prompt_style)
    legend_items = {}

    for _, row in df_plot.iterrows():
        model = row["model"]
        style = row["prompt_style"]
        x = row["avg_energy_j"]
        y = row[perf_col]

        color = MODEL_COLORS.get(model, "#333333")
        marker = PROMPT_MARKERS.get(style, "o")

        plt.scatter(x, y, color=color, marker=marker, s=60, alpha=0.85)

        # Store one example handle per (model, style)
        key = (model, style)
        if key not in legend_items:
            legend_items[key] = (color, marker)

    # Build legend with combined model + prompt
    handles = []
    labels = []
    for (model, style), (color, marker) in legend_items.items():
        handles.append(
            plt.Line2D(
                [0], [0],
                color=color,
                marker=marker,
                linestyle="",
                markersize=8,
            )
        )
        labels.append(f"{model} | {style}")

    plt.xlabel("Average energy (J, proxy)")
    plt.ylabel(perf_label)
    plt.title(title)
    if log_energy:
        plt.xscale("log")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(handles, labels, fontsize=8, title="Model | Prompt", loc="best")
    plt.tight_layout()

    out_path = os.path.join(FIGS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


# -------------------------------------------------------------------
# Task-specific plot groups
# -------------------------------------------------------------------

def make_boolq_plots(df):
    df_boolq = df[df["task"] == "BoolQ"].copy()
    if df_boolq.empty:
        print("[WARN] No BoolQ rows in all_results.csv")
        return

    _bar_plot_metric(
        df_boolq,
        metric="accuracy",
        ylabel="Accuracy",
        title="BoolQ - Accuracy by Model and Prompt Style",
        filename="boolq_accuracy_bar.png",
    )

    _bar_plot_metric(
        df_boolq,
        metric="avg_energy_j",
        ylabel="Average Energy (J)",
        title="BoolQ - Energy by Model and Prompt Style",
        filename="boolq_energy_bar.png",
    )

    _scatter_perf_vs_energy(
        df_boolq,
        perf_col="accuracy",
        perf_label="Accuracy",
        title="BoolQ - Accuracy vs Energy",
        filename="boolq_accuracy_vs_energy.png",
        log_energy=True,
    )

    _scatter_perf_vs_energy(
        df_boolq,
        perf_col="perf_per_100_tokens",
        perf_label="Performance per 100 tokens",
        title="BoolQ - Efficiency vs Energy",
        filename="boolq_efficiency_vs_energy.png",
        log_energy=True,
    )


def make_gsm8k_plots(df):
    df_gsm = df[df["task"] == "GSM8K"].copy()
    if df_gsm.empty:
        print("[WARN] No GSM8K rows in all_results.csv")
        return

    _bar_plot_metric(
        df_gsm,
        metric="accuracy",
        ylabel="Accuracy",
        title="GSM8K - Accuracy by Model and Prompt Style",
        filename="gsm8k_accuracy_bar.png",
    )

    _bar_plot_metric(
        df_gsm,
        metric="avg_energy_j",
        ylabel="Average Energy (J)",
        title="GSM8K - Energy by Model and Prompt Style",
        filename="gsm8k_energy_bar.png",
    )

    _scatter_perf_vs_energy(
        df_gsm,
        perf_col="accuracy",
        perf_label="Accuracy",
        title="GSM8K - Accuracy vs Energy",
        filename="gsm8k_accuracy_vs_energy.png",
        log_energy=True,
    )

    _scatter_perf_vs_energy(
        df_gsm,
        perf_col="perf_per_100_tokens",
        perf_label="Performance per 100 tokens",
        title="GSM8K - Efficiency vs Energy",
        filename="gsm8k_efficiency_vs_energy.png",
        log_energy=True,
    )


def make_xsum_plots(df):
    df_xsum = df[df["task"] == "XSum"].copy()
    if df_xsum.empty:
        print("[WARN] No XSum rows in all_results.csv")
        return

    _bar_plot_metric(
        df_xsum,
        metric="rougeL",
        ylabel="ROUGE-L",
        title="XSum - ROUGE-L by Model and Prompt Style",
        filename="xsum_rougeL_bar.png",
    )

    _bar_plot_metric(
        df_xsum,
        metric="avg_energy_j",
        ylabel="Average Energy (J)",
        title="XSum - Energy by Model and Prompt Style",
        filename="xsum_energy_bar.png",
    )

    _scatter_perf_vs_energy(
        df_xsum,
        perf_col="rougeL",
        perf_label="ROUGE-L",
        title="XSum - ROUGE-L vs Energy",
        filename="xsum_rougeL_vs_energy.png",
        log_energy=True,
    )

    _scatter_perf_vs_energy(
        df_xsum,
        perf_col="perf_per_100_tokens",
        perf_label="ROUGE-L per 100 tokens",
        title="XSum - Efficiency vs Energy",
        filename="xsum_efficiency_vs_energy.png",
        log_energy=True,
    )


def make_all_plots():
    df = load_results()
    make_boolq_plots(df)
    make_gsm8k_plots(df)
    make_xsum_plots(df)


if __name__ == "__main__":
    make_all_plots()