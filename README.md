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
│ • Random Forest│  │  • 10L Residual   │  │  • Closed-Source FT      │
│ • XGBoost      │  │    w/ Skip Conn.  │  │    (GPT-4.1-nano SFT)    │
└────────┬───────┘  └─────────┬─────────┘  │  • Open-Source FT 🏆     │
         │                    │            │    (Llama-3.2-3B QLoRA)  │
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

All models were evaluated on the same held-out test set from the Amazon retail product dataset using MAE, MSE, and R². Confidence intervals are reported at the 95% level. **18 models** benchmarked across baselines, classical ML, neural networks, frontier LLMs, fine-tuned LLMs, and open-source LLMs.

### Leaderboard (Ranked by MAE)

| Rank | Model | Category | MAE (± 95% CI) | MSE | R² |
|:-----|:------|:---------|:----------------|:----|:---|
| 🏆 | **Llama-3.2-3B (Fine-Tuned Full)** | Open-source FT | **$36.58 ± $7.40** | **4,189** | **80.9%** |
| 🥈 | 10-Layer Residual NN | Neural network | $40.74 ± $7.33 | 4,460 | 79.7% |
| 🥉 | GPT-5.1 | Frontier LLM | $48.24 ± $10.66 | 8,244 | 62.5% |
| 4 | Claude Opus 4.6 | Frontier LLM | $49.14 ± $33.92 | 8,406 | 67.9% |
| 5 | Optimized 3-Layer NN | Neural network | $51.53 ± $9.07 | 6,934 | 68.4% |
| 6 | Vanilla 8-Layer NN | Neural network | $58.82 ± $9.38 | 8,039 | 63.4% |
| 7 | Llama-3.2-3B (Fine-Tuned Lite) | Open-source FT | $64.69 ± $12.73 | 12,626 | 42.5% |
| 8 | XGBoost | Classical ML | $68.23 ± $9.73 | 9,582 | 56.4% |
| 9 | GPT-4.1-nano | Frontier LLM | $68.29 ± $15.79 | 17,638 | 19.7% |
| 10 | GPT-4.1-nano (Fine-Tuned) | Fine-tuned LLM | $68.91 ± $13.44 | 14,147 | 35.6% |
| 11 | Random Forest | Classical ML | $73.04 ± $11.93 | 12,747 | 42.0% |
| 12 | NLP Linear Regression (BoW) | Classical ML | $76.81 ± $11.20 | 12,786 | 41.8% |
| 13 | Human | Baseline | $87.62 ± $24.16 | 22,872 | 6.9% |
| 14 | Linear Regression | Classical ML | $101.56 ± $14.21 | 20,832 | 5.2% |
| 15 | Constant Pricer | Baseline | $106.18 ± $14.36 | 106,180 | -0.2% |
| 16 | Llama-3.2-3B (Base, Untuned) | Open-source base | $147.49 ± $53.17 | 168,916 | -668.6% |
| 17 | Gemma 270B | Frontier LLM | $202.10 ± $46.85 | 155,126 | -605.8% |
| 18 | Random Pricer | Baseline | $382.08 ± $37.47 | 219,084 | -896.9% |

### Llama-3.2-3B Fine-Tuning Progression

| | Base (Untuned) | Fine-Tuned (Lite) | Fine-Tuned (Full) |
|:--|:--|:--|:--|
| MAE | $147.49 ± $53.17 | $64.69 ± $12.73 | **$36.58 ± $7.40** |
| MSE | 168,916 | 12,626 | **4,189** |
| R² Score | -668.6% | 42.5% | **80.9%** |
| MAE improvement vs. base | — | −56.1% | **−75.2%** |

> QLoRA fine-tuning on the full dataset cut MAE by **75%** and improved R² by **+749.5 percentage points** — taking a model worse than a constant pricer to the top of the overall leaderboard.

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
> - **Llama-3.2-3B QLoRA fine-tuned on the full dataset** is the overall winner: MAE $36.58, MSE 4,189, R² 80.9%. A 3B parameter open-source model with QLoRA adapters beat every frontier LLM, every custom neural network, and every classical ML approach — outperforming GPT-5.1 zero-shot by $11.66 MAE and the 10L Residual NN by $4.16 MAE.
> - **Data scale was the single most decisive factor:** Lite fine-tune achieved 42.5% R² vs. 80.9% for the full dataset — the largest performance gap in the entire benchmark. More domain-specific training data consistently outweighed raw model size.
> - The **10-layer residual NN** (79.7% R², $40.74 MAE) held the top spot until full fine-tuning landed, confirming that well-optimized task-specific architectures are highly competitive against models orders of magnitude larger.
> - **Architecture optimization delivered compounding gains:** Vanilla 8L (63.4% R²) → Optimized 3L with BatchNorm/Dropout (68.4%) → 10L residual with skip connections (79.7%). Each step addressed a specific bottleneck.
> - NLP features (Bag of Words) delivered a jump from 5.2% → 41.8% R², confirming that textual product descriptions carry substantial pricing signal even with simple representations.
> - **Frontier LLMs performed well zero-shot** but couldn't match a trained specialist. Claude Opus 4.6's wide CI (±$33.92) reflects inconsistent predictions; GPT-5.1 was the most accurate and consistent among frontier models.
> - **Fine-tuning GPT-4.1-nano** improved R² (19.7% → 35.6%) but barely moved MAE, suggesting the nano model lacks sufficient capacity for this task regardless of fine-tuning approach.
> - **Llama-3.2-3B base (untuned)** confirmed that small open-source base models without domain adaptation are not viable out of the box — making domain-specific fine-tuning essential.
> - **Gemma 270B** severely underperformed despite its scale ($202.10 MAE, −605.8% R²), confirming that model size alone does not guarantee task suitability.

---

## 📈 Results Visualization

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MAE by Model — lower is better             ■ = $10 MAE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 🏆 Llama-3.2-3B FT (Full)  ████                              $36.58
    10L Residual NN          ████▎                             $40.74
    GPT-5.1                  █████                             $48.24
    Claude Opus 4.6          █████▏                            $49.14
    Optimized 3L NN          █████▍                            $51.53
    Vanilla 8L NN            ██████▏                           $58.82
    Llama-3.2-3B FT (Lite)   ██████▋                           $64.69
    XGBoost                  ███████                           $68.23
    GPT-4.1-nano             ███████▏                          $68.29
    GPT-4.1-nano (FT)        ███████▏                          $68.91
    Random Forest            ███████▋                          $73.04
    NLP + Linear Reg.        ████████                          $76.81
    Human                    █████████▏                        $87.62
    Linear Regression        ██████████▋                      $101.56
    Constant Pricer          ███████████                      $106.18
    Llama-3.2-3B (Base)      ███████████████▍                 $147.49
    Gemma 270B               █████████████████████            $202.10
    Random Pricer            ████████████████████████████████ $382.08

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
             $0        $100       $200       $300       $400

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 R² Score — higher is better           [negative R² models omitted]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 🏆 Llama-3.2-3B FT (Full)  ████████████████████████████████  80.9%
    10L Residual NN          ███████████████████████████████   79.7%
    Optimized 3L NN          █████████████████████████████     68.4%
    Claude Opus 4.6          ████████████████████████████      67.9%
    Vanilla 8L NN            █████████████████████████         63.4%
    GPT-5.1                  ████████████████████████          62.5%
    XGBoost                  ██████████████████████            56.4%
    Llama-3.2-3B FT (Lite)   █████████████████                 42.5%
    Random Forest            █████████████████                 42.0%
    NLP + Linear Reg.        █████████████████                 41.8%
    GPT-4.1-nano (FT)        ██████████████                    35.6%
    GPT-4.1-nano             ████████                          19.7%
    Human                    ███                                6.9%
    Linear Regression        ██                                 5.2%
    Constant Pricer          ▏                                 -0.2%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         0%        20%       40%       60%       80%      100%
```

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
- **10-Layer Residual NN:** 4096-wide residual blocks with skip connections, LayerNorm, L1Loss, log + z-score targets, AdamW with gradient clipping, GPU-accelerated on RTX 4070 Super — achieved 79.7% R² in 5 epochs
- Features: HashingVectorizer (5,000 features, binary BoW, English stop words removed)

### Phase 4 — Frontier LLM Evaluation & Fine-Tuning
- Zero-shot evaluation: GPT-5.1, Claude Opus 4.6, GPT-4.1-nano, Gemma 270B
- Supervised fine-tuning of **GPT-4.1-nano** for price prediction
- Zero-shot evaluation of **Llama-3.2-3B (base)** — established open-source pre-fine-tuning baseline (MAE: $147.49, R²: −668.6%)
- Prepare JSONL training datasets for SFT workflows

### Phase 5 — Open-Source LLM Fine-Tuning & Agentic RAG Application
- **QLoRA fine-tuning of Llama-3.2-3B** on domain-curated product pricing data
  - Lite dataset: MAE $64.69, R² 42.5%
  - Full dataset: MAE $36.58, R² 80.9% — **overall leaderboard winner** 🏆
- Build a **multi-modal RAG pipeline** with ChromaDB as the vector store
- Orchestrate agent workflows with **LangChain**
- Deploy an interactive UI with **Gradio**
- Run serverless inference on **Modal.com** for scalable, cost-efficient compute

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
| **Quantization** | BitsAndBytes (4-bit NF4), bf16 compute |
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install datasets transformers peft bitsandbytes
pip install litellm openai
pip install langchain chromadb
pip install gradio modal
```

### Run the Notebooks

```bash
jupyter notebook data_curation.ipynb
jupyter notebook data_preprocessing.ipynb
jupyter notebook evaluation_baseline.ipynb
jupyter notebook evaluation_neural_net.ipynb
jupyter notebook evaluation_llm.ipynb
```

---

## 🔮 Roadmap

- [x] Data curation and preprocessing pipeline
- [x] Classical ML baseline evaluation (6 models)
- [x] Neural network development (3 architectures: vanilla, optimized, residual)
- [x] Frontier LLM zero-shot evaluation (GPT-5.1, Claude Opus 4.6, GPT-4.1-nano, Gemma 270B)
- [x] GPT-4.1-nano supervised fine-tuning
- [x] Llama-3.2-3B base model zero-shot evaluation
- [x] Llama-3.2-3B QLoRA fine-tuning — lite dataset (MAE $64.69, R² 42.5%)
- [x] Llama-3.2-3B QLoRA fine-tuning — full dataset (MAE $36.58, R² 80.9%) 🏆
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
