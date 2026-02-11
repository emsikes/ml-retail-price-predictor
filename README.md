<p align="center">
  <img src="https://img.shields.io/badge/🏷️-Retail_Price_Predictor-blue?style=for-the-badge" alt="Project Title"/>
</p>

<h1 align="center">Retail Price Predictor</h1>

<p align="center">
  <em>From Classical ML to Genrative AI to Agentic AI — A Full-Stack Approach to Retail Price Estimation</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost"/>
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
                ┌──────────────┴──────────────┐
                ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│   CLASSICAL ML PIPELINE  │   │       LLM FINE-TUNING PIPELINE      │
│                          │   │                                      │
│  • Random Baseline       │   │  • OpenAI GPT-4o-mini (LoRA / SFT)  │
│  • Constant Baseline     │   │  • Meta Llama 3.2 (Fine-Tune)       │
│  • Linear Regression     │   │                                      │
│  • BoW + Linear Reg.     │   └──────────────┬───────────────────────┘
│  • Random Forest         │                  │
│  • XGBoost               │                  │
└──────────────┬───────────┘                  │
               │                              │
               └──────────────┬───────────────┘
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

All models were evaluated on the Amazon retail product dataset with Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score. Confidence intervals are reported at the 95% level.

| Model | MAE (± 95% CI) | MSE | R² Score |
|:------|:----------------|:----|:---------|
| 🎲 Random Pricer | 382.08 ± 37.47 | 219,084 | -896.9% |
| 📏 Constant Pricer | 106.18 ± 14.36 | 106.18 | -0.2% |
| 📈 Linear Regression | 101.56 ± 14.21 | 20,832 | 5.2% |
| 📝 NLP Linear Regression (BoW) | 76.81 ± 11.20 | 12,786 | 41.8% |
| 🌲 Random Forest | 73.04 ± 11.93 | 12,747 | 42.0% |
| 🚀 **XGBoost** | **68.23 ± 9.73** | **9,582** | **56.4%** |

> **Key Insight:** Incorporating NLP features (Bag of Words on product descriptions) provided a significant jump from 5.2% → 41.8% R², demonstrating that textual product information carries substantial pricing signal. XGBoost achieved the best classical ML performance with a 56.4% R² and the tightest confidence interval.

---

## 🔬 Project Phases

### Phase 1 — Data Curation & Preprocessing
- Source the Amazon retail product dataset from Hugging Face
- Exploratory data analysis (EDA) and statistical profiling
- Handle missing values, outliers, and data type inconsistencies
- Feature engineering for numerical and categorical attributes
- Text preprocessing for product descriptions

### Phase 2 — Classical ML Baselines & Evaluation
- Establish naive baselines (Random Pricer, Constant Pricer)
- Train and evaluate Linear Regression, NLP-enhanced Linear Regression (Bag of Words), Random Forest, and XGBoost
- Compare models using MAE, MSE, R², and confidence intervals
- Error tracking and systematic performance logging

### Phase 3 — Supervised Fine-Tuning (SFT) with Frontier LLMs
- Fine-tune **OpenAI GPT-4o-mini** using LoRA (Low-Rank Adaptation) for price prediction
- Fine-tune **Meta Llama 3.2** for retail domain price estimation
- Prepare JSONL training datasets for SFT workflows
- Evaluate fine-tuned LLMs against classical ML baselines

### Phase 4 — Agentic AI Serverless RAG Application
- Build a **multi-modal RAG pipeline** with ChromaDB as the vector store
- Orchestrate agent workflows with **LangChain**
- Deploy an interactive UI with **Gradio**
- Run serverless inference on **Modal.com** for scalable, cost-efficient compute
- Combine fine-tuned models with retrieval-augmented context for intelligent pricing

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3.11+ |
| **Notebooks** | Jupyter Notebook |
| **Classical ML** | scikit-learn, XGBoost, NumPy, Pandas |
| **NLP** | Bag of Words (CountVectorizer), text preprocessing |
| **Dataset** | Hugging Face Datasets (Amazon Retail) |
| **LLM Fine-Tuning** | OpenAI API (GPT-4o-mini, LoRA), Meta Llama 3.2 |
| **Vector DB** | ChromaDB |
| **Orchestration** | LangChain |
| **Frontend** | Gradio |
| **Serverless** | Modal.com |
| **Data Format** | JSONL (for SFT training data) |

---

## 📁 Repository Structure

```
ml-retail-price-predictor/
├── data_curation.ipynb          # Data sourcing, EDA, and curation
├── data_preprocessing.ipynb     # Feature engineering and preprocessing
├── evaluation_baseline.ipynb    # Classical ML model training and evaluation
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
pip install datasets transformers    # Hugging Face
pip install openai                   # OpenAI fine-tuning
pip install langchain chromadb       # RAG pipeline
pip install gradio modal             # App deployment
```

### Run the Notebooks

```bash
# 1. Data Curation & EDA
jupyter notebook data_curation.ipynb

# 2. Preprocessing & Feature Engineering
jupyter notebook data_preprocessing.ipynb

# 3. Model Training & Evaluation
jupyter notebook evaluation_baseline.ipynb
```

---

## 📈 Results Visualization

```
R² Score Progression Across Models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Random Pricer        ████████████████████████████████████  -896.9%  ← Worse than random
Constant Pricer      ▌                                       -0.2%
Linear Regression    ██▌                                      5.2%
NLP + Linear Reg.    █████████████████████                   41.8%
Random Forest        █████████████████████                   42.0%
XGBoost              ████████████████████████████▌           56.4%  ← Best classical ML

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     0%         25%         50%         75%        100%
```

---

## 🔮 Roadmap

- [x] Data curation and preprocessing pipeline
- [x] Classical ML baseline evaluation (6 models)
- [ ] OpenAI GPT-4o-mini fine-tuning with LoRA
- [ ] Meta Llama 3.2 fine-tuning
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