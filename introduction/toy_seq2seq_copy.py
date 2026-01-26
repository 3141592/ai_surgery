import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

# Toy vocab: digits only, plus special tokens
DIGITS = list("0123456789")
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
itos = [PAD, BOS, EOS] + DIGITS
stoi = {s: i for i, s in enumerate(itos)}
V = len(itos)

PAD_ID = stoi[PAD]
BOS_ID = stoi[BOS]
EOS_ID = stoi[EOS]

def encode_str(s: str):
    # input tokens: digits + EOS
    return [stoi[ch] for ch in s] + [EOS_ID]

def decode_ids(ids):
    # stop at EOS if present; skip special tokens in display
    out = []
    for t in ids:
        if t == EOS_ID:
            break
        if t >= 3:  # digit
            out.append(itos[t])
    return "".join(out)

def make_batch(batch_size=64, min_len=3, max_len=8):
    xs, ys = [], []
    for _ in range(batch_size):
        L = random.randint(min_len, max_len)
        s = "".join(random.choice(DIGITS) for _ in range(L))
        x = encode_str(s)
        y = [BOS_ID] + [stoi[ch] for ch in s] + [EOS_ID]  # teacher-forcing target
        xs.append(x)
        ys.append(y)

    # pad
    x_max = max(len(x) for x in xs)
    y_max = max(len(y) for y in ys)
    x_t = torch.full((batch_size, x_max), PAD_ID, dtype=torch.long)
    y_t = torch.full((batch_size, y_max), PAD_ID, dtype=torch.long)
    x_len = torch.tensor([len(x) for x in xs], dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_t[i, :len(x)] = torch.tensor(x)
        y_t[i, :len(y)] = torch.tensor(y)

    return x_t.to(DEVICE), x_len.to(DEVICE), y_t.to(DEVICE)

# ----------------------------
# Model: GRU encoder-decoder (fixed vector bottleneck)
# ----------------------------
class Seq2SeqGRU(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.enc = nn.GRU(d_model, hidden, batch_first=True)
        self.dec = nn.GRU(d_model, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def encode(self, x, x_len):
        # pack to ignore PAD in encoder
        x_emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, x_len.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.enc(packed)  # h: [1, B, H]
        return h  # the "representation" (fixed vector per sequence)

    def forward(self, x, x_len, y_in):
        """
        y_in is the decoder input sequence (teacher forcing), e.g. [BOS, ..., ...]
        Returns logits for each time step: [B, T, V]
        """
        h = self.encode(x, x_len)          # [1, B, H]
        y_emb = self.emb(y_in)             # [B, T, D]
        dec_out, _ = self.dec(y_emb, h)    # [B, T, H]
        logits = self.out(dec_out)         # [B, T, V]
        return logits, h

    @torch.no_grad()
    def greedy_decode(self, x, x_len, max_steps=32):
        h = self.encode(x, x_len)  # [1,B,H]
        B = x.size(0)
        prev = torch.full((B, 1), BOS_ID, dtype=torch.long, device=x.device)
        outputs = []
        state = h
        for _ in range(max_steps):
            emb = self.emb(prev)            # [B,1,D]
            dec_out, state = self.dec(emb, state)  # [B,1,H]
            logits = self.out(dec_out[:, 0, :])    # [B,V]
            nxt = torch.argmax(logits, dim=-1)     # [B]
            outputs.append(nxt)
            prev = nxt.unsqueeze(1)
        out_ids = torch.stack(outputs, dim=1)  # [B, max_steps]
        return out_ids, h

# ----------------------------
# Train
# ----------------------------
def main():
    model = Seq2SeqGRU(V, d_model=64, hidden=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss_fn(logits, y_true):
        # logits: [B,T,V], y_true: [B,T]
        return F.cross_entropy(
            logits.reshape(-1, V),
            y_true.reshape(-1),
            ignore_index=PAD_ID
        )

    steps = 1200
    model.train()
    for step in range(1, steps + 1):
        x, x_len, y = make_batch(batch_size=96, min_len=3, max_len=8)
        y_in = y[:, :-1]   # input to decoder (starts with BOS)
        y_tgt = y[:, 1:]   # what decoder should predict

        logits, _ = model(x, x_len, y_in)
        L = loss_fn(logits, y_tgt)

        opt.zero_grad()
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            print(f"step {step:4d}  loss {L.item():.4f}")

    # ----------------------------
    # Inspect: Does it copy?
    # ----------------------------
    model.eval()
    x, x_len, y = make_batch(batch_size=8, min_len=5, max_len=10)
    pred_ids, h = model.greedy_decode(x, x_len, max_steps=24)

    print("\nExamples (copy task):")
    for i in range(8):
        # reconstruct the original input string from x
        x_ids = x[i].tolist()
        # stop at EOS
        inp = decode_ids([t for t in x_ids if t != PAD_ID])
        pred = decode_ids(pred_ids[i].tolist())
        print(f"  in:  {inp}")
        print(f"  out: {pred}\n")

    # ----------------------------
    # Peek at the "representation" h
    # ----------------------------
    # h is [1,B,H]. We'll compute cosine similarity between a few.
    h0 = h[0]  # [B,H]
    def cos(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    print("Cosine similarities between encoder states (h):")
    for i in range(3):
        for j in range(3):
            print(f"{cos(h0[i], h0[j]):7.3f}", end=" ")
        print()

if __name__ == "__main__":
    main()

