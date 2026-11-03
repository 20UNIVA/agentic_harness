"""transformers backend (CPU by default) with client-side rendering and token splicing.

On a machine whose GPUs are busy, hide them with CUDA_VISIBLE_DEVICES="" (see docs/models.md). Triton-based kernels
(fla, causal_conv1d) are blocked so Qwen3.5 linear-attention layers use the pure-torch path.
Logprobs are not captured in this backend (use the vLLM completions backend for RL-grade token data).
"""
from __future__ import annotations

import os
import sys
import time

from .base import Policy, PromptTracker, file_sha256, turn_from_raw


def _block_gpu_kernels():
    for m in ("fla", "causal_conv1d", "triton"):
        sys.modules.setdefault(m, None)


class HFLocalPolicy(Policy):
    name = "hf_local"

    def __init__(self, model_path: str, *, device: str = "cpu", threads: int = 16, temperature: float = 0.7,
                 top_p: float = 0.8, top_k: int = 20, max_new_tokens: int = 1024, seed: int = 0):
        if device == "cpu":
            _block_gpu_kernels()   # Triton kernels (fla, causal_conv1d) cannot run on CPU; use the pure-torch path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.torch = torch
        self.model_path = model_path
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
        self.device = device
        self.sampling = dict(temperature=temperature, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
        self.stop_ids = [self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.pad_id = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.template_sha = file_sha256(os.path.join(model_path, "chat_template.jinja"))

    def info(self):
        return {"name": self.name, "model": self.model_path, "sampling": self.sampling, "device": self.device,
                "chat_template_sha256": self.template_sha}

    def generate_batch(self, episodes):
        torch = self.torch
        prompts, mism = [], []
        for ep in episodes:
            tr = ep.policy_state.setdefault("tracker", PromptTracker(self.tok))
            ids, mm = tr.build(ep)
            prompts.append(ids)
            mism.append(mm)
        L = max(len(p) for p in prompts)
        inp = torch.full((len(prompts), L), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(prompts), L), dtype=torch.long)
        for i, p in enumerate(prompts):
            inp[i, L - len(p):] = torch.tensor(p)
            att[i, L - len(p):] = 1
        s = self.sampling
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                inp.to(self.model.device), attention_mask=att.to(self.model.device), max_new_tokens=s["max_new_tokens"],
                do_sample=s["temperature"] > 0, temperature=s["temperature"] if s["temperature"] > 0 else None,
                top_p=s["top_p"] if s["temperature"] > 0 else None, top_k=s["top_k"] if s["temperature"] > 0 else None,
                eos_token_id=self.stop_ids, pad_token_id=self.pad_id)
        dt = (time.time() - t0) / len(episodes)
        turns = []
        for i, ep in enumerate(episodes):
            gen = out[i, L:].tolist()
            finish = "length"
            for j, t in enumerate(gen):
                if t in self.stop_ids:
                    gen, finish = gen[: j + 1], "stop"
                    break
            text = self.tok.decode(gen, skip_special_tokens=False)
            ep.policy_state["tracker"].record(gen, text)
            n = sum(len(t.tool_calls) for t in ep.turns)
            turns.append(turn_from_raw(text, ep.tools, prompt_opened_think=ep.enable_thinking, finish_reason=finish,
                                       call_offset=n, prompt_token_ids=prompts[i], completion_token_ids=gen,
                                       prefix_mismatch=mism[i], latency_s=round(dt, 3)))
        return turns
