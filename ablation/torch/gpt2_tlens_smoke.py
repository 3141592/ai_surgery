from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")  # base GPT-2 (124M)
prompt = "The capital of France is"
prompt = "Q: What is the capital of France?\nA:"
out = model(prompt)

# Inspect next-token distribution
logits = out[0, -1]  # last position
topk = logits.softmax(dim=-1).topk(10)

print("Prompt:", prompt)
print("Top next tokens:")
for prob, tok_id in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f"{prob:8.4f}  {repr(model.tokenizer.decode([tok_id]))}")

print("\nSample completion:")
print(model.generate(prompt, max_new_tokens=20, temperature=0.7))

print(model.generate(
    prompt,
    max_new_tokens=20,
    temperature=0.0,   # greedy
    do_sample=False
))