import torch
from model import TransformerLanguageModel
from utils import get_batch, estimate_loss, decode, vocab_size
from config import device, learning_rate, max_iters, eval_interval

torch.manual_seed(1337)  # for reproducibility

# Initialize model and optimizer
model = TransformerLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Evaluate and print losses periodically
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get a batch of training data
    xb, yb = get_batch('train')

    # Forward pass and loss calculation
    logits, loss = model(xb, yb)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate sample text after training
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start token (usually index 0)
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_indices))

# Save the trained model weights
torch.save(model.state_dict(), "model.pth")
