"""
Tiny Transformer Model
This code defines a tiny transformer model with a single transformer block, and trains it on a simple sequence prediction task. i
The model is trained to predict the next token in a sequence of token IDs.
"""
import torch

# Create the initial dataset
initial_data_string = [
    "A A A B B B",
    "C C C D D D",
    "E E E F F F"
]

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

vocab_size = 6  # A, B, C, D, E, F
embedding_dim = 4
num_heads = 1
model = TinyTransformerModel(vocab_size, embedding_dim, num_heads=num_heads)

token_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
logits = model(token_ids)

print(token_ids.shape)   # [1, 6]
print(logits.shape)      # [1, 6, 6]
print(logits)
print("Logits for the first token:", logits[0, 0])  # Logits for the first token (A)

probabilities = torch.softmax(logits[0, 0], dim=0)  # Convert logits to probabilities for the first token (A)
print("Probabilities for the first token:", probabilities)
print(probabilities.sum())
pred_token = probabilities.argmax()
print("Predicted token index:", pred_token.item())
