# mini - Transformer Text Generator

A custom-trained Transformer-based language model that generates Shakespearean-style text. This project demonstrates state-of-the-art sequence modeling techniques with fine-grained control over text generation through temperature scaling, top-k, and nucleus (top-p) sampling.

---

## 🚀 Project Overview

This repository contains a minimalist Transformer implementation for autoregressive text generation. Leveraging a compact architecture trained on Shakespearean text, it allows you to explore how modern attention mechanisms generate coherent, stylistic text. The project features:

- End-to-end Transformer model built from scratch in PyTorch
- Adjustable decoding strategies to balance creativity and coherence
- Interactive web interface powered by Gradio for real-time experimentation
- Modular codebase designed for easy retraining and extension to other corpora

---

## 🔧 Features

- **Transformer Architecture:** Multi-head self-attention, positional embeddings, and feedforward networks
- **Custom Sampling Controls:**  
  - Temperature: Adjust randomness  
  - Top-k sampling: Restrict next-token choices to top-k probabilities  
  - Top-p (nucleus) sampling: Probabilistically trim tail tokens  
  - Sampling toggle: Switch between stochastic and greedy decoding  
- **Efficient Training Pipeline:** Batch processing, periodic evaluation, and loss monitoring
- **User-Friendly UI:** Seamless text generation with live parameter tuning

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/mini-transformer-text-generator.git
cd mini-transformer-text-generator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
