# Parameter-Efficient Fine-Tuning (PEFT) with Hugging Face 🤗

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)

A complete implementation of Parameter-Efficient Fine-Tuning (PEFT) using Hugging Face's PEFT library and Transformers. This project demonstrates how to efficiently adapt foundation models with minimal training resources.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🔍 Overview

Parameter-Efficient Fine-Tuning (PEFT) has revolutionized how we adapt large pre-trained models to specific tasks. Instead of fine-tuning all parameters, PEFT methods modify only a small subset, drastically reducing computational requirements while maintaining performance.

This project demonstrates how to implement and use LoRA (Low-Rank Adaptation), one of the most popular PEFT techniques, on a sentiment analysis task.

## ✨ Features

- **Complete PEFT Pipeline**: From loading a pre-trained model to inference with the fine-tuned model
- **LoRA Implementation**: Low-Rank Adaptation for efficient fine-tuning
- **Performance Metrics**: Comprehensive evaluation and comparison of base vs. fine-tuned models
- **Resource Efficiency**: Training with minimal computational resources
- **Practical Examples**: Sample inference on real text examples

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/parameter-efficient-fine-tuning.git
cd parameter-efficient-fine-tuning

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets peft
```

## 📂 Project Structure

The project is organized as a Jupyter notebook with clearly defined sections:

1. **Project Setup**: Importing libraries and setting up the environment
2. **Loading and Evaluating a Foundation Model**: Preparing the base model and dataset
3. **Performing Parameter-Efficient Fine-Tuning**: Applying LoRA and training
4. **Performing Inference with a PEFT Model**: Evaluating and using the fine-tuned model

## 🔧 Implementation Details

### PEFT Technique: LoRA

LoRA (Low-Rank Adaptation) works by:
- Freezing the original model weights
- Adding small trainable low-rank matrices to specific layers
- Training only these additional matrices, typically < 1% of total parameters

```python
# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,                        # Rank of matrices
    lora_alpha=32,              # Scaling factor
    lora_dropout=0.1,           # Regularization
    target_modules=["q_lin", "v_lin"]  # Which layers to modify
)
```

### Model and Dataset

- **Model**: DistilBERT (distilbert-base-uncased)
- **Dataset**: SST-2 (Stanford Sentiment Treebank) from GLUE benchmark
- **Task**: Binary sentiment classification (positive/negative)

### Evaluation Metrics

- Accuracy
- F1 Score
- Parameter efficiency (trainable parameters / total parameters)

## 📊 Results

The project demonstrates significant parameter efficiency with minimal performance loss:

| Metric     | Base Model | Fine-tuned Model | Change    |
|------------|------------|------------------|-----------|
| Parameters | 66M (100%) | ~0.5M (< 1%)     | -99%      |
| Accuracy   | ~0.89      | ~0.92            | +0.03     |
| F1 Score   | ~0.89      | ~0.91            | +0.02     |

## 🚀 Future Improvements

Several ways to extend this project:

1. **Try different PEFT techniques**:
   - QLoRA (Quantized LoRA)
   - Prefix Tuning
   - IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

2. **Experiment with hyperparameters**:
   - Different ranks (r values)
   - Alpha values
   - Target modules

3. **Apply to different tasks and models**:
   - Text generation
   - Named entity recognition
   - Larger models like BERT, RoBERTa, or T5

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

If you find this implementation helpful, please consider leaving a star! For questions or contributions, feel free to open an issue or submit a pull request.