# 🧠 Mini-Transformer Text Generator

A custom-trained Transformer-based language model that generates Shakespearean-style text. Built with modern attention mechanisms and designed for experimentation, creativity, and educational exploration.

---

## 🚀 Project Overview

This repository contains a minimalist Transformer implementation for autoregressive text generation. Leveraging a compact architecture trained on Shakespearean text, it allows you to explore how modern attention mechanisms generate coherent, stylistic text.

### Key Highlights:
- End-to-end Transformer model built from scratch in PyTorch
- Adjustable decoding strategies to balance creativity and coherence
- Interactive web interface powered by Gradio for real-time experimentation
- Modular codebase designed for easy retraining and extension to other corpora

---

## 🔧 Features

- **Transformer Architecture:** Multi-head self-attention, positional embeddings, and feedforward networks
- **Custom Sampling Controls:**
  - 🔥 Temperature: Adjust randomness  
  - 🔢 Top-k sampling: Restrict next-token choices to top-k probabilities  
  - 🧮 Top-p (nucleus) sampling: Probabilistically trim tail tokens  
  - 🎲 Sampling toggle: Switch between stochastic and greedy decoding  
- **Efficient Training Pipeline:** Batch processing, periodic evaluation, and loss monitoring
- **User-Friendly UI:** Seamless text generation with live parameter tuning via Gradio

---

## ⚙️ Model Configuration

| Parameter           | Value |
|--------------------|-------|
| Batch size          | 32    |
| Context length      | 128   |
| Embedding size      | 128   |
| Transformer layers  | 4     |
| Attention heads     | 4     |
| Dropout             | 0.1   |
| Learning rate       | 0.001 |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/mini-transformer-text-generator.git
cd mini-transformer-text-generator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

python app.py


# Technologies
- Python 3.8+
- PyTorch for deep learning
- Gradio for web interface and demo
- Numpy and standard Python libraries

# Contact
Questions, feedback, or collaborations? Feel free to open an issue or reach out via manishdarji.ai@gmail.com.

# Acknowledgements
Inspired by Andrej Karpathy’s minGPT, which provided foundational guidance on building Transformers from scratch.


