import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

dir_a, dir_b = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained("distilgpt2")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")   # ONE shared base, like the cache
x = tok("Hello, today", return_tensors="pt")

def logits(m):
    m.eval()
    with torch.no_grad():
        return m(**x).logits

ma = PeftModel.from_pretrained(base, dir_a)
a_before = logits(ma)

mb = PeftModel.from_pretrained(base, dir_b)   # same base object
a_after = logits(ma)
b_out = logits(mb)

print("A changed after loading B? ", not torch.allclose(a_before, a_after, atol=1e-6))
print("A vs B differ (adapters distinct)? ", not torch.allclose(a_before, b_out, atol=1e-6))
print("max |A_before - A_after|:", (a_before - a_after).abs().max().item())
