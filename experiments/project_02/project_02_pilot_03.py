"""
Tiny Transformer Model

This code defines a tiny transformer model with a single transformer block, and trains it on a simple sequence prediction task. 
The model is trained to predict the next token in a sequence of token IDs. The code also prints the predicted token IDs before and after training, 
as well as the difference in logits to show how the model's predictions have changed.
"""
import torch

class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            batch_first=True)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)  # Add & Norm

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)  # Add & Norm

        return x

class TinyTransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.block = TransformerBlock(embedding_dim, num_heads)
        self.output_layer = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)      # [batch, seq] -> [batch, seq, embed]
        x = self.block(x)                  # [batch, seq, embed]
        logits = self.output_layer(x)      # [batch, seq, vocab]
        return logits

vocab_size = 6
embedding_dim = 4
num_heads = 1
model = TinyTransformerModel(vocab_size, embedding_dim, num_heads=num_heads)

input_ids = torch.tensor([[0, 1, 2, 3, 4]])
target_ids = torch.tensor([[1, 2, 3, 4, 5]])

logits = model(input_ids)

print("input shape: ", input_ids.shape)
print("target shape:", target_ids.shape)
print("logits shape: ", logits.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(50):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

with torch.no_grad():
    logits = model(input_ids)
    predicted_ids = logits.argmax(dim=-1)

print("input_ids:     ", input_ids)
print("target_ids:    ", target_ids)
print("predicted_ids: ", predicted_ids)