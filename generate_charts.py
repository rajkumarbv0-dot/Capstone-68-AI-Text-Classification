"""
CAPSTONE 68: Visualization Generator
Generates high-resolution, presentation-ready charts comparing
the small trained classifier vs. Google Gemini 1.5 Flash.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure directory exists
os.makedirs("charts", exist_ok=True)

# Load metrics and predictions
with open("data/metrics_summary.json", "r") as f:
    metrics_data = json.load(f)

df_preds = pd.read_csv("data/test_predictions.csv")

# Set global aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

# Palette
COLOR_SMALL_NB = "#3B82F6"      # Classic Blue
COLOR_SMALL_LR = "#1D4ED8"      # Deep Royal Blue
COLOR_GEMINI_NAIVE = "#F59E0B"  # Amber
COLOR_GEMINI_STRUCT = "#10B981" # Emerald Green
COLOR_GEMINI_FEW = "#8B5CF6"    # Vivid Purple

# -------------------------------------------------------------
# CHART 1: METRICS COMPARISON (Accuracy, Precision, Recall, F1)
# -------------------------------------------------------------
def plot_metrics_comparison():
    models = metrics_data["models"]
    names = [
        "Small NB",
        "Small LR",
        "Gemini Zero-Naive",
        "Gemini Zero-Struct",
        "Gemini Few-Shot"
    ]
    
    accuracies = [m["accuracy"] * 100 for m in models]
    precisions = [m["precision_macro"] * 100 for m in models]
    recalls = [m["recall_macro"] * 100 for m in models]
    f1_scores = [m["f1_macro"] * 100 for m in models]
    
    x = np.arange(len(names))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    rects1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#2563EB', alpha=0.9)
    rects2 = ax.bar(x - 0.5*width, precisions, width, label='Precision (Macro)', color='#059669', alpha=0.9)
    rects3 = ax.bar(x + 0.5*width, recalls, width, label='Recall (Macro)', color='#D97706', alpha=0.9)
    rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score (Macro)', color='#7C3AED', alpha=0.9)
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Capstone 68: Performance Metrics Comparison\nSmall Trained Classifiers vs. Google Gemini 1.5 Flash Chatbot', 
                 fontsize=14, fontweight='bold', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax.set_ylim(50, 105)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    
    # Add value labels
    for rects in [rects1, rects2, rects3, rects4]:
        for r in rects:
            height = r.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(r.get_x() + r.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, rotation=0)
            
    plt.tight_layout()
    plt.savefig("charts/01_metrics_comparison.png")
    plt.close()
    print("[OK] Saved charts/01_metrics_comparison.png")

# -------------------------------------------------------------
# CHART 2: CONFUSION MATRICES (Side-by-Side Heatmaps)
# -------------------------------------------------------------
def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=300)
    
    selected_models = [
        ("Small: Naive Bayes", metrics_data["models"][0]["confusion_matrix"], "Blues"),
        ("Small: TF-IDF + LogReg", metrics_data["models"][1]["confusion_matrix"], "Blues"),
        ("Gemini: Zero-Shot Naive", metrics_data["models"][2]["confusion_matrix"], "Oranges"),
        ("Gemini: Few-Shot Optimized", metrics_data["models"][4]["confusion_matrix"], "Purples")
    ]
    
    for ax, (title, cm, cmap) in zip(axes, selected_models):
        cm = np.array(cm)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=60)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Pred Neg', 'Pred Pos'], fontsize=9)
        ax.set_yticklabels(['True Neg', 'True Pos'], fontsize=9)
        
        # Annotate numbers
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                total_in_row = cm[i].sum()
                pct = (val / total_in_row) * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontweight="bold", fontsize=11)
                
        ax.grid(False)
        
    fig.suptitle("Confusion Matrix Breakdown: True Labels vs. Predicted Labels (N=100 Test Samples)", 
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig("charts/02_confusion_matrices.png", bbox_inches='tight')
    plt.close()
    print("[OK] Saved charts/02_confusion_matrices.png")

# -------------------------------------------------------------
# CHART 3: LATENCY VS ACCURACY TRADE-OFF (Pareto Frontier)
# -------------------------------------------------------------
def plot_latency_vs_accuracy():
    models = metrics_data["models"]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = [COLOR_SMALL_NB, COLOR_SMALL_LR, COLOR_GEMINI_NAIVE, COLOR_GEMINI_STRUCT, COLOR_GEMINI_FEW]
    labels = [
        "Small: Naive Bayes (0.15ms)",
        "Small: Logistic Reg (0.20ms)",
        "Gemini Zero-Shot Naive (421ms)",
        "Gemini Zero-Shot Struct (431ms)",
        "Gemini Few-Shot CoT (506ms)"
    ]
    
    latencies = [m["mean_latency_ms"] for m in models]
    accuracies = [m["accuracy"] * 100 for m in models]
    sizes = [160, 200, 220, 240, 280]
    
    for i in range(len(models)):
        ax.scatter(latencies[i], accuracies[i], s=sizes[i], color=colors[i], 
                   alpha=0.9, edgecolors='black', linewidth=1.5, zorder=4, label=labels[i])
        
        # Annotation offset
        offset_x = 1.35 if latencies[i] < 1 else 0.75
        offset_y = -1.8 if "Naive Bayes" in labels[i] else 1.2
        ax.annotate(f"{models[i]['model_name'].split(':')[1].strip() if ':' in models[i]['model_name'] else models[i]['model_name']}\nAcc: {accuracies[i]:.1f}% | Lat: {latencies[i]:.2f}ms",
                    (latencies[i], accuracies[i]),
                    xytext=(latencies[i] * offset_x, accuracies[i] + offset_y),
                    fontsize=8.5, fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FAFAFA", ec="#CCCCCC", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Mean Inference Latency per Sample (Milliseconds, Log Scale)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title("The Fundamental Trade-off: Inference Latency vs. Accuracy\nSmall Local Classifier (<0.3ms) vs. Cloud LLM API (~500ms)", 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Highlight zones
    ax.axvspan(0.05, 2.0, color="#3B82F6", alpha=0.08, label="Ultra-Fast Real-Time Zone (<2ms)")
    ax.axvspan(200, 1000, color="#8B5CF6", alpha=0.08, label="Generative LLM Zone (200-1000ms)")
    
    ax.set_ylim(65, 103)
    ax.set_xlim(0.08, 1000)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig("charts/03_latency_vs_accuracy.png")
    plt.close()
    print("[OK] Saved charts/03_latency_vs_accuracy.png")

# -------------------------------------------------------------
# CHART 4: OPERATIONAL COST & THROUGHPUT (Economic Analysis)
# -------------------------------------------------------------
def plot_cost_and_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    models = metrics_data["models"]
    short_names = ["Small NB", "Small LR", "Gemini Naive", "Gemini Struct", "Gemini Few-Shot"]
    costs = [m["cost_per_100k_usd"] for m in models]
    throughputs = [m["throughput_qps"] for m in models]
    
    # Cost chart (Log scale)
    bar_colors = ["#2563EB", "#1D4ED8", "#F59E0B", "#10B981", "#7C3AED"]
    bars1 = ax1.bar(short_names, costs, color=bar_colors, alpha=0.9, edgecolor="#333333", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Cost per 100,000 Inferences ($ USD, Log Scale)", fontsize=10, fontweight='bold')
    ax1.set_title("Operational Cost per 100,000 Queries\n(12,500x Difference!)", fontsize=12, fontweight='bold', pad=12)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"${yval:.3f}" if yval < 0.1 else f"${yval:.2f}",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax1.set_ylim(0.001, 100)
    ax1.tick_params(axis='x', rotation=15)
    
    # Throughput chart (Queries per second on single thread)
    bars2 = ax2.bar(short_names, throughputs, color=bar_colors, alpha=0.9, edgecolor="#333333", lw=0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("Inference Throughput (Queries / Second, Log Scale)", fontsize=10, fontweight='bold')
    ax2.set_title("Single-Thread Throughput\n(Local CPU vs Cloud API)", fontsize=12, fontweight='bold', pad=12)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"{yval:,.1f} QPS",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax2.set_ylim(0.5, 15000)
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig("charts/04_cost_and_efficiency.png")
    plt.close()
    print("[OK] Saved charts/04_cost_and_efficiency.png")

# -------------------------------------------------------------
# CHART 5: ERROR TAXONOMY BY LINGUISTIC CHALLENGE CATEGORY
# -------------------------------------------------------------
def plot_error_taxonomy():
    cat_data = metrics_data["category_breakdown"]
    
    cats = [c["category"] for c in cat_data]
    acc_small = [c["accuracy_small_lr"] * 100 for c in cat_data]
    acc_gemini_zero = [c["accuracy_gemini_zero"] * 100 for c in cat_data]
    acc_gemini_few = [c["accuracy_gemini_few"] * 100 for c in cat_data]
    
    y = np.arange(len(cats))
    height = 0.25
    
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    r1 = ax.barh(y - height, acc_small, height, label="Small Model (TF-IDF + LR)", color="#2563EB", alpha=0.9)
    r2 = ax.barh(y, acc_gemini_zero, height, label="Gemini 1.5 Flash (Zero-Shot)", color="#F59E0B", alpha=0.9)
    r3 = ax.barh(y + height, acc_gemini_few, height, label="Gemini 1.5 Flash (Few-Shot CoT)", color="#7C3AED", alpha=0.9)
    
    ax.set_xlabel("Category Accuracy (%)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title("Linguistic Vulnerability Analysis: Accuracy by Challenge Type\nWhere Small Models Stumble vs. Where Chatbots Dominate", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=10, fontweight='bold')
    ax.set_xlim(45, 108)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    
    # Add values at the end of each bar
    for bars in [r1, r2, r3]:
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.0, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%",
                    ha="left", va="center", fontsize=8, fontweight="bold")
            
    plt.tight_layout()
    plt.savefig("charts/05_error_taxonomy.png")
    plt.close()
    print("[OK] Saved charts/05_error_taxonomy.png")

if __name__ == "__main__":
    plot_metrics_comparison()
    plot_confusion_matrices()
    plot_latency_vs_accuracy()
    plot_cost_and_efficiency()
    plot_error_taxonomy()
    print("\nAll 5 charts successfully generated in /charts directory!")
