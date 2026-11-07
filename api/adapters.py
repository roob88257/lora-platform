import threading
from collections import OrderedDict
import torch
from peft import PeftModel


class AdapterManager:
    """One PeftModel per base model. Adapters are loaded by name, LRU-evicted."""
    def __init__(self, base_model, max_adapters=3):
        self._base = base_model
        self._model = None                 # becomes a PeftModel on first load
        self._order = OrderedDict()        # adapter names, oldest first
        self._max = max_adapters
        self._lock = threading.Lock()

    def _activate(self, name, path):
        if name in self._order:
            self._order.move_to_end(name)
        else:
            if len(self._order) >= self._max:
                old, _ = self._order.popitem(last=False)
                self._model.delete_adapter(old)
                print(f"Evicting adapter {old} to make room for {name}")
            if self._model is None:
                self._model = PeftModel.from_pretrained(self._base, path, adapter_name=name)
            else:
                self._model.load_adapter(path, adapter_name=name)
            self._order[name] = True
        self._model.set_adapter(name)

    def generate(self, name, path, tok, prompt, max_new_tokens):
        with self._lock:                   # covers activate AND generate
            self._activate(name, path)
            self._model.eval()
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    pad_token_id=tok.eos_token_id)
            return tok.decode(out[0], skip_special_tokens=True)
