from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
prompt = "The capital of France is"
NEW_TOKENS = 12

def gen(m):
    return m.generate(prompt, max_new_tokens=NEW_TOKENS, do_sample=False)

base = gen(model)
print("BASE:", base)

changed = 0
for h in range(model.cfg.n_heads):
    def zero_head(z, hook, h=h):
        z[:, :, h, :] = 0
        return z

    with model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", zero_head)]):
        edited = gen(model)

    if edited != base:
        changed += 1
        print(f"\nHEAD {h} CHANGED:")
        print("EDIT:", edited)

print(f"\nHeads that changed output: {changed}/{model.cfg.n_heads}")

