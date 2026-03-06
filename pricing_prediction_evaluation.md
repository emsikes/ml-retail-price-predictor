# Pricing Prediction Model Evaluation Summary

## Project Overview

This experiment evaluates multiple approaches to predicting product prices from item descriptions, progressing from naive baselines through classical ML, neural networks, frontier LLMs, and fine-tuned models. The goal is to understand which modeling strategies best capture the relationship between product descriptions and their market prices.

## Results

### Baselines

| Model | Mean Error (±95% CI) | MSE | R² |
|---|---|---|---|
| Random Pricer | $382.08 ± $37.47 | 219,084 | -896.9% |
| Constant Pricer | $106.18 ± $14.36 | 106,180 | -0.2% |
| Human (n=100) | $87.62 ± $24.16 | 22,872 | 6.9% |

### Classical ML

| Model | Mean Error (±95% CI) | MSE | R² |
|---|---|---|---|
| Linear Regression | $101.56 ± $14.21 | 20,832 | 5.2% |
| NLP Linear Regression (BoW) | $76.81 ± $11.20 | 12,786 | 41.8% |
| Random Forest | $73.04 ± $11.93 | 12,747 | 42.0% |
| XGBoost | $68.23 ± $9.73 | 9,582 | 56.4% |

### Neural Networks

| Model | Mean Error (±95% CI) | MSE | R² |
|---|---|---|---|
| Vanilla 8-Layer NN | $58.82 ± $9.38 | 8,039 | 63.4% |
| Optimized 3-Layer NN | $51.53 ± $9.07 | 6,934 | 68.4% |
| **10-Layer Residual NN** | **$40.74 ± $7.33** | **4,460** | **79.7%** |

### Frontier LLMs (Zero-Shot)

| Model | Mean Error (±95% CI) | MSE | R² |
|---|---|---|---|
| Gemma 270B | $202.10 ± $46.85 | 155,126 | -605.8% |
| GPT-4.1-nano | $68.29 ± $15.79 | 17,638 | 19.7% |
| Claude Opus 4.6 | $49.14 ± $33.92 | 8,406 | 67.9% |
| GPT-5.1 | $48.24 ± $10.66 | 8,244 | 62.5% |

### Fine-Tuned LLMs

| Model | Mean Error (±95% CI) | MSE | R² |
|---|---|---|---|
| GPT-4.1-nano Fine-Tuned | $68.91 ± $13.44 | 14,147 | 35.6% |

## Leaderboard (Ranked by Mean Error)

| Rank | Model | Mean Error | R² |
|---|---|---|---|
| 🥇 | **10-Layer Residual NN** | **$40.74** | **79.7%** |
| 🥈 | GPT-5.1 | $48.24 | 62.5% |
| 🥉 | Claude Opus 4.6 | $49.14 | 67.9% |
| 4 | Optimized 3-Layer NN | $51.53 | 68.4% |
| 5 | Vanilla 8-Layer NN | $58.82 | 63.4% |
| 6 | XGBoost | $68.23 | 56.4% |
| 7 | GPT-4.1-nano | $68.29 | 19.7% |
| 8 | GPT-4.1-nano Fine-Tuned | $68.91 | 35.6% |
| 9 | Random Forest | $73.04 | 42.0% |
| 10 | NLP Linear Regression (BoW) | $76.81 | 41.8% |
| 11 | Human | $87.62 | 6.9% |
| 12 | Linear Regression | $101.56 | 5.2% |
| 13 | Constant Pricer | $106.18 | -0.2% |
| 14 | Gemma 270B | $202.10 | -605.8% |
| 15 | Random Pricer | $382.08 | -896.9% |

## Key Takeaways

**The 10-layer residual network is the clear winner across every metric.** At $40.74 mean error, 79.7% R², and the tightest confidence interval (±$7.33) of any top model, it dominates the leaderboard — beating the best frontier LLMs by a wide margin despite being trained only on BoW features. Skip connections and z-score normalized log targets proved transformative, allowing 289M parameters across 4096-wide residual blocks to train effectively in just 5 epochs on an RTX 4070 Super.

**The neural network progression tells a clear story.** Vanilla 8-layer (63.4% R²) → Optimized 3-layer with BatchNorm/Dropout (68.4%) → 10-layer residual with LayerNorm/skip connections (79.7%). Each architectural improvement delivered a measurable jump, with the residual architecture unlocking the ability to go deep without vanishing gradients.

**Frontier LLMs showed strong zero-shot performance but couldn't match a trained specialist.** GPT-5.1 and Claude Opus 4.6 both beat all classical ML models and the first two neural networks, but fell well short of the residual NN. Claude's wide confidence interval (±$33.92) suggests inconsistent predictions — very accurate on some items, significantly off on others. GPT-5.1 showed the tightest CI among the LLMs.

**Fine-tuning GPT-4.1-nano delivered mixed results.** R² improved from 19.7% to 35.6%, but mean error barely changed ($68.29 → $68.91) and the CI tightened from ±$15.79 to ±$13.44. The model became more consistent but not meaningfully more accurate — suggesting the nano model may lack the capacity to fully learn price estimation even with task-specific training data.

**Gemma 270B failed catastrophically** with an R² of -605.8%, performing worse than the random baseline on MSE. This suggests the model either misunderstood the task format or has poor calibration for numerical price estimation.

**Classical ML provides a strong cost-effective middle ground.** XGBoost achieved $68.23 mean error with minimal compute cost, making it the best price-to-performance option for production deployment where GPU training or LLM inference costs are a concern.

**Human pricing intuition underperformed all ML models except basic linear regression**, with a wide confidence interval (±$24.16) suggesting high variance in human estimation ability.

## Neural Network Architecture Comparison

| Attribute | Vanilla 8-Layer | Optimized 3-Layer | 10-Layer Residual |
|---|---|---|---|
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
| Device | CPU | CPU | GPU (RTX 4070 Super) |

## Methodology Notes

- All models evaluated on the same held-out test set using the `pricer.evaluator` framework
- Dataset sourced from HuggingFace Hub (`ed-donner/items_full`)
- Neural network features: HashingVectorizer with 5,000 features, binary BoW, English stop words removed
- Human evaluation limited to first 100 test items
- LLM evaluations used zero-shot prompting with no fine-tuning (except GPT-4.1-nano fine-tuned variant)
- 95% confidence intervals reported for mean absolute error
- GPU training performed on NVIDIA RTX 4070 Super (12GB VRAM) on Windows 11
