import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from api.adapters import AdapterManager

tok = AutoTokenizer.from_pretrained("distilgpt2")
m = AdapterManager(AutoModelForCausalLM.from_pretrained("distilgpt2"))
a, b = sys.argv[1], sys.argv[2]
g = lambda n, p: m.generate(n, p, tok, "Hello, today", 20)

a1 = g("A", a); b1 = g("B", b); a2 = g("A", a)
print("A stable after B? ", a1 == a2)
print("A differs from B? ", a1 != b1)
