# Pricing Prediction Model Evaluation Summary

## Project Overview

This experiment evaluates multiple approaches to predicting product prices from item descriptions, progressing from naive baselines through classical ML, neural networks, and frontier LLMs. The goal is to understand which modeling strategies best capture the relationship between product descriptions and their market prices.

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
| **Optimized 3-Layer NN** | **$51.53 ± $9.07** | **6,934** | **68.4%** |

### Frontier LLMs

| Model | Mean Error (±95% CI) | MSE | R² |
|---|---|---|---|
| Gemma 270B | $202.10 ± $46.85 | 155,126 | -605.8% |
| GPT-4.1-nano | $68.29 ± $15.79 | 17,638 | 19.7% |
| Claude Opus 4.6 | $49.14 ± $33.92 | 8,406 | 67.9% |
| GPT-5.1 | $48.24 ± $10.66 | 8,244 | 62.5% |

## Leaderboard (Ranked by Mean Error)

| Rank | Model | Mean Error | R² |
|---|---|---|---|
| 🥇 | GPT-5.1 | $48.24 | 62.5% |
| 🥈 | Claude Opus 4.6 | $49.14 | 67.9% |
| 🥉 | Optimized 3-Layer NN | $51.53 | 68.4% |
| 4 | Vanilla 8-Layer NN | $58.82 | 63.4% |
| 5 | XGBoost | $68.23 | 56.4% |
| 6 | GPT-4.1-nano | $68.29 | 19.7% |
| 7 | Random Forest | $73.04 | 42.0% |
| 8 | NLP Linear Regression (BoW) | $76.81 | 41.8% |
| 9 | Human | $87.62 | 6.9% |
| 10 | Linear Regression | $101.56 | 5.2% |
| 11 | Constant Pricer | $106.18 | -0.2% |
| 12 | Gemma 270B | $202.10 | -605.8% |
| 13 | Random Pricer | $382.08 | -896.9% |

## Key Takeaways

**The optimized neural network achieved the highest R² (68.4%) of any model tested**, including frontier LLMs — demonstrating that a well-tuned, task-specific model trained on domain data can match or beat general-purpose models orders of magnitude larger.

**Architecture optimization delivered a massive gain.** Reducing from 8 identical 64-unit layers to 3 layers (256→128→64) with BatchNorm, Dropout, log-transformed targets, HuberLoss, and 30 epochs with early stopping improved mean error by 12.4% ($58.82 → $51.53) and R² by 5 percentage points (63.4% → 68.4%).

**Frontier LLMs showed strong zero-shot performance but with caveats.** GPT-5.1 and Claude Opus 4.6 achieved the lowest mean errors without any training data, but Claude's wide confidence interval (±$33.92) suggests inconsistent predictions — likely very accurate on some items but significantly off on others. GPT-5.1 showed the tightest CI among the top performers.

**Gemma 270B failed catastrophically** with an R² of -605.8%, performing worse than the random baseline on MSE. This suggests the model either misunderstood the task format or has poor calibration for numerical price estimation.

**Classical ML provides a strong cost-effective middle ground.** XGBoost achieved $68.23 mean error with minimal compute cost, making it the best price-to-performance option for production deployment where LLM inference costs are a concern.

**Human pricing intuition underperformed all ML models except basic linear regression**, with a wide confidence interval (±$24.16) suggesting high variance in human estimation ability.

## Methodology Notes

- All models evaluated on the same held-out test set using the `pricer.evaluator` framework
- Dataset sourced from HuggingFace Hub (`ed-donner/items_full`)
- Neural network features: HashingVectorizer with 5,000 features, binary BoW, English stop words removed
- Human evaluation limited to first 100 test items
- LLM evaluations used zero-shot prompting with no fine-tuning
- 95% confidence intervals reported for mean absolute error
