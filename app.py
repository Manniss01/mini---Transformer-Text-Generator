import torch
import gradio as gr

from model import TransformerLanguageModel
from utils import decode, encode, vocab_size
from config import device  # 'cuda' if available, else 'cpu'

# Load model
model = TransformerLanguageModel(vocab_size)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Generation function
def generate_text(prompt, max_new_tokens, temperature, top_k, top_p, do_sample):
    try:
        prompt = prompt.strip()
        if not prompt:
            return "Please enter a valid prompt."

        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
            do_sample=do_sample
        )[0]
        return decode(out.tolist())
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input Prompt", placeholder="Enter some text...", lines=2),
        gr.Slider(10, 500, value=200, step=10, label="Max Tokens"),
        gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(0, 100, value=50, step=1, label="Top-k (0 = disable)"),
        gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p (1.0 = disable)"),
        gr.Checkbox(value=True, label="Use Sampling")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="mini - Transformer Text Generator",
    description="Generate Shakespearean-style text using a custom-trained Transformer model. Control temperature, top-k, top-p, and sampling.",
    examples=[
        ["To be, or not to be", 200, 1.0, 40, 0.9, True],
        ["Once upon a midnight dreary", 150, 0.8, 30, 0.95, True],
        ["In fair Verona, where we lay our scene", 180, 1.2, 0, 1.0, False]
    ]
)

if __name__ == "__main__":
    iface.launch()
