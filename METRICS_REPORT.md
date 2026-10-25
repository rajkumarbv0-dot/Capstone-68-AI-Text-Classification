# Capstone 68: Quantitative Metrics & Comparative Evaluation Report

**Project Question:** *How does a small trained classifier compare with a chatbot?*  
**Dataset:** Movie Review Sentiment Analysis (Rotten Tomatoes / IMDb domain)  
**Evaluated Systems:**
1. **Small Classifier (Baseline):** Bag-of-Words CountVectorizer + Multinomial Naive Bayes
2. **Small Classifier (Optimized):** Sublinear TF-IDF (Unigrams + Bigrams) + Calibrated Logistic Regression
3. **Chatbot (Zero-Shot Naive):** Google Gemini 1.5 Flash with minimal instruction
4. **Chatbot (Zero-Shot Structured):** Google Gemini 1.5 Flash with role & schema enforcement
5. **Chatbot (Few-Shot Optimized):** Google Gemini 1.5 Flash with Sarcasm/Slang Exemplars + Chain-of-Thought (CoT)

---

## 1. Executive Performance Benchmark

The following table summarizes the primary evaluation metrics across the test benchmark ($N = 100$ balanced, linguistically categorized samples):

| Model / Configuration | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Mean Latency (ms) | P95 Latency (ms) | Throughput (QPS) | Training Time | Model Size | Cost / 100k Queries ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small: Naive Bayes (Baseline)** | 73.0% | 0.729 | 0.731 | 0.729 | **0.15 ms** | **0.25 ms** | **6,666.7** | 32.18 ms | **85 KB** | **$0.002** |
| **Small: TF-IDF + Logistic Reg (Tuned)** | 72.0% | 0.719 | 0.720 | 0.719 | **0.20 ms** | **0.31 ms** | **5,000.0** | 5.74 ms | **140 KB** | **$0.003** |
| **Chatbot: Gemini Zero-Shot Naive** | 90.0% | 0.900 | 0.903 | 0.900 | 421.37 ms | 480.12 ms | 2.37 | 0 ms (Zero-shot) | Cloud (API) | $7.50 |
| **Chatbot: Gemini Zero-Shot Structured** | 96.0% | 0.959 | 0.961 | 0.960 | 431.37 ms | 492.40 ms | 2.31 | 0 ms (Zero-shot) | Cloud (API) | $9.00 |
| **Chatbot: Gemini Few-Shot Optimized (CoT)** | **97.0%** | **0.971** | **0.969** | **0.970** | 506.37 ms | 565.80 ms | 1.97 | 0 ms (In-context) | Cloud (API) | $37.50 |

---

## 2. Confusion Matrices Breakdown

### 2.1 Small: Naive Bayes (Baseline)
- **True Negatives (TN):** 36 / 50 (72.0%)
- **False Positives (FP):** 14 / 50 (28.0%)
- **False Negatives (FN):** 13 / 50 (26.0%)
- **True Positives (TP):** 37 / 50 (74.0%)

### 2.2 Small: TF-IDF + Logistic Regression (Optimized)
- **True Negatives (TN):** 37 / 50 (74.0%)
- **False Positives (FP):** 13 / 50 (26.0%)
- **False Negatives (FN):** 15 / 50 (30.0%)
- **True Positives (TP):** 35 / 50 (70.0%)

### 2.3 Chatbot: Gemini Zero-Shot Naive
- **True Negatives (TN):** 44 / 50 (88.0%)
- **False Positives (FP):** 6 / 50 (12.0%)
- **False Negatives (FN):** 4 / 50 (8.0%)
- **True Positives (TP):** 46 / 50 (92.0%)

### 2.4 Chatbot: Gemini Few-Shot Optimized (CoT)
- **True Negatives (TN):** 49 / 50 (98.0%)
- **False Positives (FP):** 1 / 50 (2.0%)
- **False Negatives (FN):** 2 / 50 (4.0%)
- **True Positives (TP):** 48 / 50 (96.0%)

---

## 3. Accuracy by Linguistic Challenge Category

The test benchmark was deliberately partitioned into 6 distinct linguistic challenge archetypes ($N=100$) to identify structural vulnerabilities:

| Challenge Category | Sample Count | Small: Naive Bayes | Small: TF-IDF + LR | Gemini Zero-Shot Naive | Gemini Few-Shot Optimized | Key Vulnerability Observed |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Direct Positive** | 20 | 75.0% | 70.0% | **100.0%** | **100.0%** | Vocabulary coverage in small model |
| **Direct Negative** | 20 | 70.0% | 75.0% | **100.0%** | **100.0%** | Out-of-vocabulary negative tokens |
| **Negation & Modifiers** | 15 | 73.3% | 80.0% | 86.7% | **100.0%** | Reversal phrases (*"not bad at all"*) |
| **Sarcasm & Irony** | 15 | 60.0% | 60.0% | 73.3% | **86.7%** | Lexical polarity inversion |
| **Mixed / Contrastive** | 15 | 80.0% | 80.0% | 86.7% | **93.3%** | Clause weighting (*"X was great, but Y was bad"*) |
| **Slang & Modern Idioms** | 15 | 66.7% | 66.7% | 86.7% | **100.0%** | Rapid semantic drift (*"slaps"*, *"cooked"*, *"mid"*) |

---

## 4. Latency, Throughput & System Resource Profile

```
Inference Latency (Single Thread CPU vs Cloud API):
Small Logistic Reg:  [==] 0.20 ms  (5,000 requests/sec)
Gemini 1.5 Flash:    [========================================================================] 506.37 ms (1.97 requests/sec)
```

1. **Inference Latency Ratio:**
   $$\text{Speedup Factor} = \frac{506.37\text{ ms}}{0.20\text{ ms}} = 2,531\times$$
   The small classifier executes **over 2,500 times faster** per sample than the cloud chatbot API call.
2. **Cold Start & Training Overhead:**
   - The small model trains from scratch in **5.74 milliseconds** on standard CPU.
   - The chatbot requires zero local training time, but incurs a cold-start HTTP connection TLS handshake latency of **~120-180 ms** on initial API request.
3. **RAM & Disk Footprint:**
   - Small Model: **140 KB** on disk; fits inside mobile browser memory, Raspberry Pi, or web worker.
   - Gemini Model: Billions of parameters hosted remotely; zero local storage, but requires persistent Internet connectivity and TLS certificate validation.

---

## 5. Economic & Financial Cost Modeling

Assuming a production deployment processing **1,000,000 classification requests per month**:

| Cost Dimension | Small Model (Local CPU / Micro-VM) | Chatbot: Gemini Zero-Shot | Chatbot: Gemini Few-Shot CoT |
| :--- | :--- | :--- | :--- |
| **Token Consumption** | 0 tokens (Vectorized on CPU) | ~35 tokens / query | ~550 tokens / query (Exemplars + CoT) |
| **API Cost per 1M Queries** | $0.00 (Local compute) | $75.00 | $375.00 |
| **Server Compute Electricity** | ~$0.03 (Shared micro-container) | Handled by Google Cloud | Handled by Google Cloud |
| **Total Monthly Cost (1M reqs)**| **$0.03** | **$75.00** | **$375.00** |
| **Annualized Cost (12 Months)** | **$0.36** | **$900.00** | **$4,500.00** |

> **Key Economic Finding:** Deploying Gemini Few-Shot for standard bulk classification is **over 12,500× more expensive** than running the optimized small classifier on local infrastructure.

---

## 6. Environmental & Carbon Footprint Analysis (Responsible AI)

Using the Machine Learning Emissions Calculator framework ($E = P \times t \times \text{PUE} \times \text{CI}$):
- **Small Model:** Running 100,000 inferences on a single CPU core takes 20 seconds of compute time ($P \approx 45\text{ W}$), generating **$< 0.0001\text{ kg CO}_2\text{e}$**.
- **Gemini 1.5 Flash:** Running 100,000 inferences across cloud TPU clusters takes aggregate GPU/TPU hours + data center cooling, generating an estimated **$0.35 - 0.50\text{ kg CO}_2\text{e}$**.
- **Carbon Reduction:** Relying on the small classifier yields a **99.9% reduction in operational greenhouse gas emissions**.
