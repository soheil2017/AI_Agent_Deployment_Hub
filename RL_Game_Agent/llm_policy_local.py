"""
llm_policy_local.py
-------------------
Local LLM (Phi-3-mini by default) + LoRA adapter as the RL policy.
Used for Phase 2 (PPO training) where we need direct access to model weights.

The LLM receives the text prompt from TextFrozenLake and outputs
one of: Left, Down, Right, Up.

LoRA adapter is saved/loaded from ./lora_adapter/ so it persists
across training runs.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from game_env import ACTION_MAP
from policy_base import BasePolicy

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "lora_adapter")

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                      # LoRA rank — higher = more capacity, more memory
    lora_alpha=16,            # scaling factor (alpha/r = 2 is a common ratio)
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
    bias="none",
)

GENERATION_CONFIG = {
    "max_new_tokens": 10,     # we only need one word
    "do_sample": True,
    "temperature": 0.3,       # low temp = more deterministic actions
    "top_p": 0.9,
}


# ------------------------------------------------------------------
# Policy class
# ------------------------------------------------------------------

class LocalLLMPolicy(BasePolicy):
    """
    Wraps a causal LLM + LoRA adapter as an RL policy.
    Used in Phase 2 for PPO fine-tuning.

    Usage:
        policy = LocalLLMPolicy()
        action_int, action_text = policy.act(state_dict["text_prompt"])
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        load_in_4bit: bool = True,
        load_existing_adapter: bool = True,
        device: str = None,
    ):
        """
        Args:
            model_id:              HuggingFace model repo ID.
            load_in_4bit:          Use 4-bit quantization (saves ~75% VRAM).
            load_existing_adapter: If True and ./lora_adapter/ exists, load it.
            device:                'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        self.model_id = model_id
        self.device = device or self._detect_device()
        print(f"[LocalLLMPolicy] Using device: {self.device}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(load_in_4bit, load_existing_adapter)

    # ------------------------------------------------------------------
    # Public interface (required by BasePolicy)
    # ------------------------------------------------------------------

    def act(self, text_prompt: str) -> tuple[int, str]:
        """
        Given the text state prompt, return (action_int, action_text).
        Falls back to a random action if the LLM output is unparseable.
        """
        response = self._generate(text_prompt)
        action_int = self.parse_action(response)

        if action_int == -1:
            return self.random_fallback(response, source="LocalLLMPolicy")

        return action_int, ACTION_MAP[action_int]

    def save_adapter(self, path: str = ADAPTER_DIR):
        """Save the LoRA adapter weights to disk."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[LocalLLMPolicy] LoRA adapter saved to: {path}")

    def get_trainable_parameters(self) -> dict:
        """Return counts of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "trainable_pct": round(100 * trainable / total, 4),
        }

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        print(f"[LocalLLMPolicy] Loading tokenizer: {self.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, load_in_4bit: bool, load_existing_adapter: bool):
        print(f"[LocalLLMPolicy] Loading model: {self.model_id}")

        # Quantization config (skip on CPU/MPS — bitsandbytes requires CUDA)
        bnb_config = None
        if load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device != "cuda":
            base_model = base_model.to(self.device)

        # Load existing LoRA adapter or create a fresh one
        adapter_exists = os.path.isdir(ADAPTER_DIR) and os.path.isfile(
            os.path.join(ADAPTER_DIR, "adapter_config.json")
        )

        if load_existing_adapter and adapter_exists:
            print(f"[LocalLLMPolicy] Loading existing LoRA adapter from: {ADAPTER_DIR}")
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        else:
            print("[LocalLLMPolicy] Initializing fresh LoRA adapter.")
            model = get_peft_model(base_model, LORA_CONFIG)

        model.eval()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"[LocalLLMPolicy] Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.3f}%)"
        )
        return model

    # ------------------------------------------------------------------
    # Internal: inference
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Run a forward pass and return the generated text."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **GENERATION_CONFIG,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode the newly generated tokens (not the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ------------------------------------------------------------------
# Quick test — run directly to verify model loads and generates
# ------------------------------------------------------------------
if __name__ == "__main__":
    from game_env import TextFrozenLake

    print("=" * 60)
    print("Local LLM Policy — Smoke Test")
    print("=" * 60)

    env = TextFrozenLake(is_slippery=False)
    state = env.reset()

    print("\n[Loading policy — this may take a minute on first run...]\n")
    policy = LocalLLMPolicy(load_in_4bit=True, load_existing_adapter=False)

    print("\n--- State prompt sent to LLM ---")
    print(state["text_prompt"])

    action_int, action_text = policy.act(state["text_prompt"])
    print(f"\n[LocalLLMPolicy] Chosen action: {action_text} (id={action_int})")

    env.close()
