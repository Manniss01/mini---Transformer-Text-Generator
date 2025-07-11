import torch

batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1