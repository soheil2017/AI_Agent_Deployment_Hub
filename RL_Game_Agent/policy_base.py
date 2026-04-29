"""
policy_base.py
--------------
Abstract base class for all LLM policies.
Both OpenAI and local (Phi-3 + LoRA) implementations share this interface,
making it trivial to swap backends without changing any other file.
"""

import re
from abc import ABC, abstractmethod
from game_env import ACTION_MAP, ACTION_REVERSE


class BasePolicy(ABC):
    """
    All policies must implement act().
    The rest of the system (play.py, train.py) only ever calls act().
    """

    @abstractmethod
    def act(self, text_prompt: str) -> tuple[int, str]:
        """
        Given the text state description, choose an action.

        Returns:
            (action_int, action_text)
            action_int  : 0=Left, 1=Down, 2=Right, 3=Up
            action_text : human-readable string
        """

    # ------------------------------------------------------------------
    # Shared utility — both backends use the same parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_action(response: str) -> int:
        """
        Extract a valid action integer from raw LLM text.
        Handles: 'Left', 'Move left', 'I will go Left', 'go right', etc.
        Returns -1 if no valid action is found.
        """
        for action_name in ACTION_REVERSE:
            if re.search(rf"\b{action_name}\b", response, re.IGNORECASE):
                return ACTION_REVERSE[action_name]
        return -1

    @staticmethod
    def random_fallback(raw_response: str, source: str = "Policy") -> tuple[int, str]:
        """Pick a random action when the LLM output can't be parsed."""
        import random
        action_int = random.randint(0, 3)
        action_text = ACTION_MAP[action_int]
        print(f"[{source}] Could not parse '{raw_response}' → random fallback: {action_text}")
        return action_int, action_text
