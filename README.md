<p align="center">
  <img src="https://img.shields.io/badge/🏷️-Retail_Price_Predictor-blue?style=for-the-badge" alt="Project Title"/>
</p>

<h1 align="center">Retail Price Predictor</h1>

<p align="center">
  <em>From Classical ML to Generative AI to Agentic AI — A Full-Stack Approach to Retail Price Estimation</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Anthropic-191919?style=flat-square&logo=anthropic&logoColor=white" alt="Anthropic"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/Meta_Llama_3.2-0467DF?style=flat-square&logo=meta&logoColor=white" alt="Llama"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/ChromaDB-FF6F61?style=flat-square&logo=googlechrome&logoColor=white" alt="Chroma"/>
  <img src="https://img.shields.io/badge/Gradio-F97316?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"/>
  <img src="https://img.shields.io/badge/Modal-000000?style=flat-square&logo=modal&logoColor=white" alt="Modal"/>
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/emsikes/ml-retail-price-predictor?style=flat-square" alt="Last Commit"/>
  <img src="https://img.shields.io/github/languages/top/emsikes/ml-retail-price-predictor?style=flat-square" alt="Top Language"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## 📌 Overview

**ML Retail Price Predictor** is an end-to-end machine learning project that tackles retail product price estimation through a progressive, multi-phase approach. Starting with classical ML baselines and advancing through NLP-enhanced models, supervised fine-tuning of frontier LLMs, and culminating in a production-grade **Agentic AI serverless application**, the project demonstrates the full spectrum of modern ML/AI engineering.

The pipeline ingests the **Amazon product dataset from Hugging Face**, explores and curates the data for price prediction, benchmarks traditional ML techniques, then pushes into LLM fine-tuning and retrieval-augmented generation (RAG) to build a multi-modal intelligent pricing assistant.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  Hugging Face (Amazon Retail Dataset) → Curation → Preprocessing    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌────────────────┐  ┌───────────────────┐  ┌──────────────────────────┐
│ CLASSICAL ML   │  │  NEURAL NETWORKS  │  │   LLM PIPELINE           │
│                │  │                   │  │                          │
│ • Baselines    │  │  • Vanilla 8L NN  │  │  • Frontier Zero-Shot    │
│ • Linear Reg.  │  │  • Optimized 3L   │  │    (GPT-5.1, Claude,     │
│ • BoW + LR     │  │    w/ BatchNorm   │  │     Gemma, GPT-4.1-nano) │
│ • Random Forest│  │  • 10L Residual   │  │  • Fine-Tuned            │
│ • XGBoost      │  │    w/ Skip Conn.  │  │    (GPT-4.1-nano SFT)   │
└────────┬───────┘  └─────────┬─────────┘  │  • Open-Source           │
         │                    │            │    (Llama-3.2-3B base)   │
         │                    │            └────────────┬─────────────┘
         └────────────────────┼─────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              AGENTIC AI APPLICATION LAYER                           │
│                                                                     │
│  LangChain (Orchestration) + ChromaDB (Vector Store)                │
│  Gradio (UI) + Modal.com (Serverless Compute)                       │
│                                                                     │
│  → Multi-Modal RAG  →  Agentic Pricing Assistant                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Benchmark Results

All models were evaluated on the same held-out test set from the Amazon retail product dataset using MAE, MSE, and R². Confidence intervals are reported at the 95% level. **16 models** were benchmarked across baselines, classical ML, neural networks, frontier LLMs, fine-tuned LLMs, and open-source LLMs.

### Leaderboard (Ranked by MAE)

| Rank | Model | MAE (± 95% CI) | MSE | R² Score |
|:-----|:------|:----------------|:----|:---------|
| 🥇 | **10-Layer Residual NN** | **$40.74 ± $7.33** | **4,460** | **79.7%** |
| 🥈 | GPT-5.1 | $48.24 ± $10.66 | 8,244 | 62.5% |
| 🥉 | Claude Opus 4.6 | $49.14 ± $33.92 | 8,406 | 67.9% |
| 4 | Optimized 3-Layer NN | $51.53 ± $9.07 | 6,934 | 68.4% |
| 5 | Vanilla 8-Layer NN | $58.82 ± $9.38 | 8,039 | 63.4% |
| 6 | XGBoost | $68.23 ± $9.73 | 9,582 | 56.4% |
| 7 | GPT-4.1-nano | $68.29 ± $15.79 | 17,638 | 19.7% |
| 8 | GPT-4.1-nano (Fine-Tuned) | $68.91 ± $13.44 | 14,147 | 35.6% |
| 9 | Random Forest | $73.04 ± $11.93 | 12,747 | 42.0% |
| 10 | NLP Linear Regression (BoW) | $76.81 ± $11.20 | 12,786 | 41.8% |
| 11 | Human | $87.62 ± $24.16 | 22,872 | 6.9% |
| 12 | Linear Regression | $101.56 ± $14.21 | 20,832 | 5.2% |
| 13 | Constant Pricer | $106.18 ± $14.36 | 106,180 | -0.2% |
| 14 | Llama-3.2-3B (Base) | $147.49 ± $53.17 | 168,916 | -668.6% |
| 15 | Gemma 270B | $202.10 ± $46.85 | 155,126 | -605.8% |
| 16 | Random Pricer | $382.08 ± $37.47 | 219,084 | -896.9% |

### Neural Network Architecture Comparison

| Attribute | Vanilla 8-Layer | Optimized 3-Layer | 10-Layer Residual |
|:----------|:----------------|:------------------|:------------------|
| Hidden dimensions | 64 (all layers) | 256→128→64 | 4096 (all layers) |
| Parameters | ~700K | ~1.4M | 289M |
| Normalization | None | BatchNorm | LayerNorm |
| Regularization | None | Dropout (0.3/0.2/0.1) | Dropout (0.2) |
| Skip connections | No | No | Yes (residual blocks) |
| Target transform | None | log1p | log + z-score |
| Loss function | MSE | HuberLoss | L1Loss |
| Optimizer | Adam | Adam + weight decay | AdamW |
| LR schedule | None | CosineAnnealing | CosineAnnealing |
| Gradient clipping | No | No | Yes (max norm 1.0) |
| Epochs | 2 | 30 (early stopping) | 5 |
| Hardware | CPU | CPU | GPU (RTX 4070 Super) |

> **Key Insights:**
> - The **10-layer residual NN** achieved the highest R² (79.7%) and lowest MAE ($40.74) of any model, including frontier LLMs — demonstrating that a well-tuned, task-specific model trained on domain data can outperform general-purpose models orders of magnitude larger.
> - **Architecture optimization delivered compounding gains:** Vanilla 8L (63.4% R²) → Optimized 3L with BatchNorm/Dropout (68.4%) → 10L residual with skip connections (79.7%). Each improvement addressed a specific bottleneck (vanishing gradients, overfitting, model capacity).
> - NLP features (Bag of Words) provided a massive jump from 5.2% → 41.8% R², confirming that textual product descriptions carry substantial pricing signal.
> - **Frontier LLMs performed well zero-shot** but couldn't match a trained specialist. Claude Opus 4.6's wide CI (±$33.92) suggests inconsistent predictions; GPT-5.1 showed the best balance of accuracy and consistency among LLMs.
> - **Fine-tuning GPT-4.1-nano** improved R² (19.7% → 35.6%) but barely moved MAE, suggesting the nano model lacks capacity for this task.
> - **Llama-3.2-3B (base, untuned)** produced a MAE of $147.49 with a very wide CI (±$53.17) and a deeply negative R² (-668.6%), confirming that a small open-source base model without instruction tuning or domain adaptation is not viable for structured price prediction. This establishes the pre-fine-tuning baseline for the QLoRA fine-tuning phase.
> - **Gemma 270B** underperformed despite its scale (MAE $202.10, R² -605.8%), suggesting its instruction-following and numerical reasoning are poorly suited to this pricing format — model size alone does not guarantee task fit.

---

## 🔬 Project Phases

### Phase 1 — Data Curation & Preprocessing
- Source the Amazon retail product dataset from Hugging Face
- Exploratory data analysis (EDA) and statistical profiling
- Handle missing values, outliers, and data type inconsistencies
- Feature engineering for numerical and categorical attributes
- Text preprocessing for product descriptions

### Phase 2 — Classical ML Baselines & Evaluation
- Establish naive baselines (Random Pricer, Constant Pricer, Human)
- Train and evaluate Linear Regression, NLP-enhanced Linear Regression (Bag of Words), Random Forest, and XGBoost
- Compare models using MAE, MSE, R², and confidence intervals
- Error tracking and systematic performance logging

### Phase 3 — Neural Network Development & Optimization
- **Vanilla 8-Layer NN:** Initial deep network (64-wide, MSE loss, 2 epochs) — established 63.4% R² baseline
- **Optimized 3-Layer NN:** Reduced depth, added BatchNorm + Dropout, switched to HuberLoss, log1p target transform, 30 epochs with early stopping — improved to 68.4% R²
- **10-Layer Residual NN:** 4096-wide residual blocks with skip connections, LayerNorm, L1Loss, log + z-score targets, AdamW with gradient clipping, GPU-accelerated on RTX 4070 Super — achieved 79.7% R² in only 5 epochs
- Features: HashingVectorizer (5,000 features, binary BoW, English stop words removed)

### Phase 4 — Frontier LLM Evaluation & Fine-Tuning
- Zero-shot evaluation of frontier LLMs: GPT-5.1, Claude Opus 4.6, GPT-4.1-nano, Gemma 270B
- Supervised fine-tuning of **GPT-4.1-nano** for price prediction
- Prepare JSONL training datasets for SFT workflows
- Evaluate fine-tuned LLMs against classical ML and neural network baselines
- Zero-shot evaluation of **Llama-3.2-3B (base)** to establish open-source pre-fine-tuning baseline (MAE: $147.49, R²: -668.6%)

### Phase 5 — Agentic AI Serverless RAG Application
- Build a **multi-modal RAG pipeline** with ChromaDB as the vector store
- Orchestrate agent workflows with **LangChain**
- Deploy an interactive UI with **Gradio**
- Run serverless inference on **Modal.com** for scalable, cost-efficient compute
- Combine fine-tuned models with retrieval-augmented context for intelligent pricing
- QLoRA fine-tuning of **Llama-3.2-3B** on domain-curated product pricing data

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3.11+ |
| **Notebooks** | Jupyter Notebook |
| **Classical ML** | scikit-learn, XGBoost, NumPy, Pandas |
| **Neural Networks** | PyTorch (CUDA 12.6), BatchNorm, LayerNorm, Residual Blocks |
| **NLP** | Bag of Words (HashingVectorizer), text preprocessing |
| **Dataset** | Hugging Face Datasets (Amazon Retail) |
| **Frontier LLMs** | OpenAI (GPT-5.1, GPT-4.1-nano), Anthropic (Claude Opus 4.6), Google (Gemma 270B) |
| **Open-Source LLMs** | Meta Llama 3.2-3B (base + QLoRA fine-tuning) |
| **LLM Fine-Tuning** | OpenAI API (GPT-4.1-nano SFT), HuggingFace PEFT + QLoRA (Llama-3.2-3B) |
| **Vector DB** | ChromaDB |
| **Orchestration** | LangChain |
| **Frontend** | Gradio |
| **Serverless** | Modal.com |
| **Hardware** | NVIDIA RTX 4070 Super (12GB VRAM) |
| **Data Format** | JSONL (for SFT training data) |

---

## 📁 Repository Structure

```
ml-retail-price-predictor/
├── data_curation.ipynb          # Data sourcing, EDA, and curation
├── data_preprocessing.ipynb     # Feature engineering and preprocessing
├── evaluation_baseline.ipynb    # Classical ML model training and evaluation
├── evaluation_neural_net.ipynb  # Neural network training and optimization
├── evaluation_llm.ipynb         # Frontier LLM and fine-tuned model evaluation
├── error_tracking.txt           # Systematic error and performance logs
├── jsonl/                       # JSONL training data for LLM fine-tuning
├── pricer/                      # Core pricing module and utilities
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.11
pip install jupyter numpy pandas scikit-learn xgboost
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  # GPU support
pip install datasets transformers peft bitsandbytes  # Hugging Face + QLoRA
pip install litellm openai           # LLM providers
pip install langchain chromadb       # RAG pipeline
pip install gradio modal             # App deployment
```

### Run the Notebooks

```bash
# 1. Data Curation & EDA
jupyter notebook data_curation.ipynb

# 2. Preprocessing & Feature Engineering
jupyter notebook data_preprocessing.ipynb

# 3. Classical ML Training & Evaluation
jupyter notebook evaluation_baseline.ipynb

# 4. Neural Network Training (GPU recommended)
jupyter notebook evaluation_neural_net.ipynb

# 5. LLM Evaluation & Fine-Tuning
jupyter notebook evaluation_llm.ipynb
```

---

## 📈 Results Visualization

```
MAE by Model (Worst → Best)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Random Pricer        ████████████████████████████████████████  $382.08
Gemma 270B           █████████████████████                     $202.10
Llama-3.2-3B (Base)  ███████████████▍                          $147.49
Constant Pricer      ███████████                               $106.18
Linear Regression    ██████████▋                               $101.56
Human                █████████▏                                 $87.62
NLP + Linear Reg.    ████████                                   $76.81
Random Forest        ███████▋                                   $73.04
GPT-4.1-nano FT      ███████▏                                   $68.91
GPT-4.1-nano         ███████▏                                   $68.29
XGBoost              ███████                                    $68.23
Vanilla 8L NN        ██████▏                                    $58.82
Optimized 3L NN      █████▍                                     $51.53
Claude Opus 4.6      █████▏                                     $49.14
GPT-5.1              █████                                      $48.24
10L Residual NN      ████▎                                      $40.74  ← 🏆 Best

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              $0       $100       $200       $300       $400
```

---

## 🔮 Roadmap

- [x] Data curation and preprocessing pipeline
- [x] Classical ML baseline evaluation (6 models)
- [x] Neural network development (3 architectures: vanilla, optimized, residual)
- [x] Frontier LLM zero-shot evaluation (GPT-5.1, Claude Opus 4.6, GPT-4.1-nano, Gemma 270B)
- [x] GPT-4.1-nano supervised fine-tuning
- [x] Llama-3.2-3B base model zero-shot evaluation (pre-fine-tuning baseline)
- [ ] QLoRA fine-tuning of Llama-3.2-3B on domain pricing data
- [ ] ChromaDB vector store integration
- [ ] LangChain agentic workflow orchestration
- [ ] Gradio interactive UI
- [ ] Modal.com serverless deployment
- [ ] Multi-modal RAG application (end-to-end)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with ☕ and curiosity — progressing from classical ML to agentic AI</sub>
</p>
