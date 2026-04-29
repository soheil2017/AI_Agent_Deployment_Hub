"""
llm_policy_openai.py
--------------------
OpenAI-backed policy for the FrozenLake agent.
Used for Phase 1 (play & data collection) before LoRA fine-tuning.

The interface is identical to LocalLLMPolicy — both expose:
    action_int, action_text = policy.act(text_prompt)

Setup:
    1. pip install openai python-dotenv
    2. Copy .env.example → .env and fill in OPENAI_API_KEY
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from policy_base import BasePolicy

load_dotenv()  # reads .env from project root


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"   # fast + cheap; swap to "gpt-4o" for better reasoning

SYSTEM_PROMPT = """You are an expert game-playing agent navigating a FrozenLake grid.
Your job is to choose the single best action to reach the Goal without falling into a Hole.
Think step by step, then reply with exactly one word: Left, Down, Right, or Up."""


class OpenAIPolicy(BasePolicy):
    """
    Uses OpenAI chat completions as the agent policy.

    No weights, no training here — this is the 'play and collect data'
    phase. The collected (state, action, reward) trajectories will later
    be used to fine-tune the local LoRA model.

    Usage:
        policy = OpenAIPolicy()
        action_int, action_text = policy.act(state["text_prompt"])
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        api_key: str = None,
    ):
        """
        Args:
            model:       OpenAI model ID (gpt-4o-mini, gpt-4o, etc.)
            temperature: Low = more deterministic decisions (0.2 recommended).
            api_key:     Override API key (defaults to OPENAI_API_KEY env var).
        """
        self.model = model
        self.temperature = temperature

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "OpenAI API key not found.\n"
                "Set OPENAI_API_KEY in your .env file or as an environment variable.\n"
                "See .env.example for the format."
            )

        self.client = OpenAI(api_key=resolved_key)
        print(f"[OpenAIPolicy] Ready — model: {self.model}, temperature: {self.temperature}")

    # ------------------------------------------------------------------
    # Public interface (required by BasePolicy)
    # ------------------------------------------------------------------

    def act(self, text_prompt: str) -> tuple[int, str]:
        """
        Send the game state to OpenAI and parse the action from the response.

        Returns:
            (action_int, action_text)
        """
        raw_response = self._call_api(text_prompt)
        action_int = self.parse_action(raw_response)

        if action_int == -1:
            return self.random_fallback(raw_response, source="OpenAIPolicy")

        action_text = raw_response.strip().capitalize()
        # Normalize to exact action name from ACTION_MAP
        from game_env import ACTION_MAP
        action_text = ACTION_MAP[action_int]
        return action_int, action_text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, user_prompt: str) -> str:
        """Make a single chat completion call and return the text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=20,       # we only need one word
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from game_env import TextFrozenLake

    print("=" * 60)
    print("OpenAI Policy — Smoke Test")
    print("=" * 60)

    env = TextFrozenLake(is_slippery=False)
    state = env.reset()
    policy = OpenAIPolicy()

    print("\n--- State prompt ---")
    print(state["text_prompt"])

    action_int, action_text = policy.act(state["text_prompt"])
    print(f"\n[OpenAIPolicy] Chosen action: {action_text} (id={action_int})")

    env.close()
