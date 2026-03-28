# TransformerLens Hook Reference (GPT-style Pre-LN Block)

## Mental Model (One Block)

Attention:
resid_pre → ln1 → q,k,v → attn_scores → pattern → z → attn_out → resid_mid

MLP:
resid_mid → ln2 → mlp_pre → mlp_post → mlp_out → resid_post

---

## Hook-by-Hook Explanation (blocks.11.*)

### Residual Stream

- **hook_resid_pre**  
  Residual stream entering the block.  
  Shape: [batch, pos, d_model]  
  Core communication channel of the model.

- **hook_resid_mid**  
  Residual after attention output is added.  
  Between attention and MLP.

- **hook_resid_post**  
  Residual after MLP output is added.  
  Output of the block → input to next block.

---

### LayerNorm (LN1 – before attention)

- **ln1.hook_scale**  
  Per-token normalization scale (std dev term).

- **ln1.hook_normalized**  
  Normalized residual stream used as input to attention.

---

### Attention Internals

- **attn.hook_q**  
  Queries (what each position is looking for).  
  Shape: [batch, pos, head, d_head]

- **attn.hook_k**  
  Keys (what each position offers).  
  Same shape as q.

- **attn.hook_v**  
  Values (information to be moved).  
  Same shape as q/k.

- **attn.hook_attn_scores**  
  Raw scores before softmax.  
  Computed as q @ k^T / sqrt(d_head)

- **attn.hook_pattern**  
  Softmaxed attention probabilities.  
  Shape: [batch, head, dest_pos, src_pos]  
  Shows *where attention is paid*.

- **attn.hook_z**  
  Per-head output after mixing values using attention.  
  Shape: [batch, pos, head, d_head]  
  Represents what each head *actually contributes*.

- **hook_attn_out**  
  Combined output of all heads projected back to d_model.  
  Added into residual stream.

---

### LayerNorm (LN2 – before MLP)

- **ln2.hook_scale**  
  Normalization scale for MLP input.

- **ln2.hook_normalized**  
  Normalized residual stream fed into MLP.

---

### MLP Internals

- **mlp.hook_pre**  
  Pre-activation (after first linear layer).  
  Shape: [batch, pos, d_mlp]

- **mlp.hook_post**  
  Post-activation (after nonlinearity).  
  Shows which neurons are “on”.

- **hook_mlp_out**  
  Output projected back to d_model.  
  Added into residual stream.

---

### Final LayerNorm

- **ln_final.hook_scale**  
  Final normalization scale before logits.

- **ln_final.hook_normalized**  
  Final hidden representation before unembedding.

---

## Most Useful Hooks (Prioritized)

1. **attn.hook_pattern**  
   First place to look. Shows attention structure.

2. **attn.hook_z**  
   What each head actually outputs (more informative than v).

3. **hook_resid_pre / resid_post**  
   Core signal flow through the model. Critical for patching.

4. **mlp.hook_post**  
   Best place to inspect neuron activations.

5. **attn.hook_attn_scores**  
   Useful when pattern alone is misleading.

---

## Key Clarification

- `"post"` in TransformerLens shorthand refers to:  
  → **mlp.hook_post**

- Residual stream uses explicit names:  
  → resid_pre, resid_mid, resid_post

---

## Summary

Think of the block as:

- Attention reads from normalized residual, routes information (pattern), and writes back (attn_out)
- MLP reads updated residual, applies nonlinear feature transformation, and writes back
- Residual stream accumulates everything

This repeated accumulation is the core of how transformers compute.