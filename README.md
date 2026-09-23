# Math Reasoning with GRPO Fine-Tuning

A project exploring the improvement of mathematical reasoning in a small language model using progressive prompting and GRPO (Group Relative Policy Optimization).

## Overview

This project evaluates a **Qwen2.5-0.5B-Instruct** model through three phases:

- **Phase 1:** Baseline evaluation with normal prompting
- **Phase 2:** Chain-of-Thought prompting
- **Phase 3:** GRPO fine-tuning using a synthetic mathematical reasoning dataset

The goal is to study how much performance improvement can be obtained through prompting and reinforcement-learning-based post-training on a very small language model.

---

## Results Summary

| Phase | Approach | Description |
|---|---|---|
| Phase 1 | Normal Prompting | Baseline capability |
| Phase 2 | Chain-of-Thought | Explicit step-by-step reasoning |
| Phase 3 | GRPO Fine-Tuning | Policy optimization using answer-based rewards |

### Overall Evaluation

![All Evaluation Comparison](./eval_graphs/all_evaluation.png)

---

## Project Structure

```text
Ninebar/
├── src/
│   └── grpo_training.ipynb
│
├── eval_graphs/
│   ├── normal_all.png
│   ├── cot_all.png
│   ├── after_finetune_all.png
│   ├── all_evaluation.png
│   ├── performance_gap.png
│   └── mean_std_scatter.png
│
└── README.md
```

---

# Dataset

## Synthetic Math Reasoning Dataset

A synthetic dataset was created to train and evaluate mathematical reasoning.

### Dataset Size

- **Training set:** 1,000 examples
- **Evaluation set:** 100 examples
- **Paraphrased evaluation set:** 100 examples

The training examples cover difficulty levels from 1–7, while the evaluation set contains more difficult examples ranging from levels 5–10.

## Question Templates

The dataset contains 15 mathematical reasoning patterns.

### 1. Arithmetic

Sequential addition, subtraction, and multiplication.

**Example:**

> A number starts at 25. Then 15 is added. Then it is multiplied by 3. What is the final value?

### 2. Money

Shopping and change calculation.

**Example:**

> A customer buys 3 items at $12 each and 2 items at $8 each. The customer pays $65. How many dollars of change do they receive?

### 3. Ratio

Proportional distribution problems.

**Example:**

> A total of 60 objects is divided among A, B, and C in the ratio 2:3:5. How many objects does B receive?

### 4. Percentage

Sequential percentage changes.

**Example:**

> A quantity starts at 500. It is decreased by 20% and then decreased by another 25%. What is the final quantity?

### 5. Average

Mean calculation.

**Example:**

> What is the average of 15, 25, 35, and 45?

### 6. Linear Equation

Basic algebraic equations.

**Example:**

> Solve for x: 3x + 12 = 45

### 7. Two-Step Equation

Parenthesized algebraic equations.

**Example:**

> Solve for x: 4(x + 7) = 52

### 8. Distance

Speed-time-distance problems.

**Example:**

> A vehicle travels at 60 miles per hour for 4 hours and then travels another 25 miles. What total distance does it travel?

### 9. Work

Worker-day calculations.

**Example:**

> A job requires 5 workers working for 8 days. How many total worker-days of work are required?

### 10. Age

Age progression problems.

**Example:**

> A younger person is 18 years old and an older person is 28 years old. After 10 years, how old will the older person be?

### 11. Multistep

Multiple sequential operations.

**Example:**

> A store has 50 units. It receives 30 more units and sells 15 units. Later, it receives another 20 units. How many units does the store have now?

### 12. Area

Rectangle area calculations.

**Example:**

> A rectangle has a length of 15 meters and a width of 8 meters. What is its area in square meters?

### 13. Perimeter

Rectangle perimeter calculations.

**Example:**

> A rectangle has a length of 12 meters and a width of 7 meters. What is its perimeter in meters?

### 14. Units

Box-based inventory calculations.

**Example:**

> A warehouse has 8 boxes with 12 items in each box. It then receives 15 more items. How many items are there in total?

### 15. Difference

Simple subtraction.

**Example:**

> What is the difference between 150 and 45?

---

## Data Format

Each dataset example contains:

```text
id
prompt
completion
_question
_answer
_template
_difficulty
```

Where:

- `id` — unique example identifier
- `prompt` — problem prompt presented to the model
- `completion` — expected numerical answer
- `_question` — original question
- `_answer` — correct numerical answer
- `_template` — question template type
- `_difficulty` — difficulty level

---

# Model

## Base Model

**Qwen2.5-0.5B-Instruct**

The experiments use the 0.5B parameter instruction-tuned Qwen model.

### Configuration

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Quantization:** 4-bit
- **Fine-tuning:** LoRA
- **Framework:** Unsloth
- **RL Algorithm:** GRPO

---

## LoRA Configuration

```text
Rank (r):        8
Alpha:           16
Dropout:         0
Target modules:  q_proj
                 k_proj
                 o_proj
```

Maximum sequence length:

```text
1024 tokens
```

---

# Phase 1 — Baseline Evaluation

## Approach

The base Qwen2.5-0.5B-Instruct model was evaluated using normal prompting without explicit reasoning instructions.

### Methodology

- Base Qwen2.5-0.5B-Instruct model
- 4-bit quantization
- 100-question evaluation set
- 4 generated responses per question
- Mean score calculated across generations

## Results

![Phase 1 - Normal Prompting](./eval_graphs/normal_all.png)

---

# Phase 2 — Chain-of-Thought Prompting

## Approach

The same model was evaluated using prompts explicitly asking the model to reason step-by-step.

### Methodology

- Explicit step-by-step reasoning instruction
- Same 100-question evaluation set
- 4 generations per question
- Same answer-based evaluation procedure

## Results

![Phase 2 - Chain of Thought](./eval_graphs/cot_all.png)

---

# Phase 3 — GRPO Fine-Tuning

## Approach

The model was fine-tuned using **Group Relative Policy Optimization (GRPO)**.

GRPO optimizes the policy using groups of sampled responses and their relative rewards rather than requiring a separate value model.

## Training Configuration

```python
GRPOConfig(
    output_dir="qwen0.5b-math-grpo",
    learning_rate=5e-5,

    use_vllm=True,
    num_generations=8,

    max_prompt_length=256,
    max_completion_length=768,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    optim="adamw_8bit",
    num_train_epochs=1,

    beta=0.01,
)
```

## Reward Function

The reward is based on whether the generated answer matches the target numerical answer.

The reward procedure:

1. Extract numerical values from the generated completion.
2. Check the final few numerical values.
3. Compare them with the expected answer.
4. Assign a binary reward.

```text
Correct answer   -> 1.0
Incorrect answer -> 0.0
```

## Training Data

```text
1,000 synthetic mathematical reasoning examples
```

## Evaluation

The model is evaluated on the same 100-question evaluation set used in the earlier phases.

## Results

![Phase 3 - After Fine-tuning](./eval_graphs/after_finetune_all.png)

---

# Performance Analysis

## Variance Analysis

For Phases 1 and 2, four responses were generated for each question.

The analysis compares the mean and standard deviation across generations.

![Mean and Standard Deviation](./eval_graphs/mean_std_scatter.png)

This analysis is used to examine:

- Prediction consistency
- Response variance
- Changes in reliability across phases

---

## Performance Gap

Comparison of performance across all three experimental phases.

![Performance Gap](./eval_graphs/performance_gap.png)

---

## Combined Evaluation

The following plot combines the results across the three phases.

![All Evaluation Results](./eval_graphs/all_evaluation.png)

---

# Evaluation Metrics

## Scoring

Each generated answer receives a binary score:

```text
Correct   = 1
Incorrect = 0
```

For each question, the mean score across multiple generations is calculated.

### Number of Generations

```text
Phase 1: 4 generations
Phase 2: 4 generations
Phase 3: 8 generations
```

## Metrics Tracked

- Mean accuracy per question
- Standard deviation across generations
- Overall dataset accuracy
- Performance by question template
- Performance by difficulty level

---

# Key Findings

### 1. Prompting Matters

Explicit Chain-of-Thought instructions can change the model's behavior compared with ordinary prompting.

### 2. GRPO Changes the Policy

GRPO fine-tuning optimizes the model toward responses that receive higher rewards under the defined mathematical correctness function.

### 3. Small Models Can Be Post-Trained

The experiment investigates whether reinforcement-learning-based post-training can improve mathematical behavior in a 0.5B parameter model.

### 4. Response Variance Can Be Measured

Generating multiple responses per question allows both mean performance and response variability to be studied.

### 5. Synthetic Data Enables Controlled Experiments

Because the training data is generated from known mathematical templates, the experiment provides direct control over question structure, difficulty, and ground-truth answers.

---

# Setup

## Requirements

- Python 3.8+
- CUDA-capable GPU
- 8 GB+ GPU memory recommended
- CUDA-compatible PyTorch installation

## Installation

```bash
git clone https://github.com/ajheshbasnet/grpo_qwen0.5b_finetuned.git

cd grpo_qwen0.5b_finetuned

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
# venv\Scripts\activate

pip install torch transformers trl datasets
pip install unsloth vllm
pip install matplotlib wandb
```

---

# Running the Experiment

Open:

```text
src/grpo_training.ipynb
```

Then run the notebook sequentially.

The notebook contains the pipeline for:

1. Generating the synthetic dataset
2. Loading the Qwen2.5-0.5B-Instruct model
3. Running baseline evaluation
4. Running Chain-of-Thought evaluation
5. Configuring LoRA
6. Training with GRPO
7. Evaluating the fine-tuned model
8. Generating evaluation plots

---

# Repository Structure

```text
grpo_qwen0.5b_finetuned/
│
├── src/
│   └── grpo_training.ipynb
│
├── eval_graphs/
│   ├── normal_all.png
│   ├── cot_all.png
│   ├── after_finetune_all.png
│   ├── all_evaluation.png
│   ├── performance_gap.png
│   └── mean_std_scatter.png
│
├── README.md
└── LICENSE
```

---

# Citation

```bibtex
@misc{basnet2026mathgrpo,
  title={Math Reasoning with GRPO Fine-Tuning},
  author={Ajhesh Basnet},
  year={2026},
  howpublished={\url{https://github.com/ajheshbasnet/grpo_qwen0.5b_finetuned}}
}
```

---

# Acknowledgments

- **Qwen Team** — Qwen2.5-0.5B-Instruct
- **Unsloth** — efficient model fine-tuning
- **Hugging Face TRL** — GRPO implementation
- **vLLM** — efficient inference

---

# License

This project is licensed under the MIT License.
