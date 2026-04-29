"""
game_env.py
-----------
Wraps Gymnasium's FrozenLake-v1 and converts raw numeric state
into a human-readable text description that the LLM can reason about.

FrozenLake grid (4x4 default):
  S F F F      S = Start, F = Frozen (safe), H = Hole, G = Goal
  F H F H
  F F F H
  H F F G

Actions: 0=Left, 1=Down, 2=Right, 3=Up
"""

import gymnasium as gym
import numpy as np

# Map action integers to readable strings
ACTION_MAP = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
ACTION_REVERSE = {v: k for k, v in ACTION_MAP.items()}

# 4x4 FrozenLake tile types
TILE_MAP = {
    "S": "Start",
    "F": "Frozen (safe)",
    "H": "Hole (danger)",
    "G": "Goal",
}

# Default 4x4 map
FROZEN_LAKE_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]


class TextFrozenLake:
    """
    FrozenLake environment with a text-based state description
    suitable for passing to an LLM as a prompt.
    """

    def __init__(self, map_name: str = "4x4", is_slippery: bool = False, render: bool = False):
        """
        Args:
            map_name:    Gymnasium map size, '4x4' or '8x8'.
            is_slippery: If True, the agent may slip (stochastic transitions).
                         Keep False while learning — deterministic is easier.
            render:      If True, render the grid visually in the terminal.
        """
        render_mode = "human" if render else None
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode=render_mode,
        )
        self.grid = FROZEN_LAKE_MAP  # 4x4 default
        self.grid_size = 4
        self.state: int = 0
        self.done: bool = False
        self.step_count: int = 0
        self.total_reward: float = 0.0

    # ------------------------------------------------------------------
    # Core environment interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the environment and return the initial text state."""
        obs, _ = self.env.reset()
        self.state = int(obs)
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0
        return self._build_state_dict(reward=0.0, action_taken=None, info="Episode started.")

    def step(self, action: int) -> dict:
        """
        Take one step in the environment.

        Args:
            action: integer 0-3 (Left/Down/Right/Up)

        Returns:
            dict with text description, reward, done flag, and metadata.
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")

        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.state = int(obs)
        self.done = terminated or truncated
        self.step_count += 1
        self.total_reward += float(reward)

        # Build a human-readable outcome message
        if terminated and reward == 1.0:
            info = "SUCCESS! The agent reached the Goal."
        elif terminated and reward == 0.0:
            info = "FAILURE. The agent fell into a Hole."
        elif truncated:
            info = "Episode truncated (too many steps)."
        else:
            info = "The agent moved safely."

        return self._build_state_dict(
            reward=float(reward),
            action_taken=ACTION_MAP[action],
            info=info,
        )

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # Text description builder
    # ------------------------------------------------------------------

    def _build_state_dict(self, reward: float, action_taken, info: str) -> dict:
        """Return a dict with all state info plus a text prompt for the LLM."""
        row, col = divmod(self.state, self.grid_size)
        tile = self.grid[row][col]

        text = self._build_text_prompt(row, col, tile, action_taken, info)

        return {
            "state_id": self.state,
            "position": (row, col),
            "tile": tile,
            "tile_meaning": TILE_MAP.get(tile, "Unknown"),
            "action_taken": action_taken,
            "reward": reward,
            "done": self.done,
            "step": self.step_count,
            "total_reward": self.total_reward,
            "info": info,
            "text_prompt": text,          # <-- this goes into the LLM
        }

    def _build_text_prompt(self, row: int, col: int, tile: str, action_taken, info: str) -> str:
        """
        Build a clear natural-language description of the current game state.
        This is what the LLM receives as its 'observation'.
        """
        grid_visual = self._render_grid_text(row, col)

        action_line = (
            f"Last action taken: {action_taken}\n" if action_taken else "This is the start of the episode.\n"
        )

        prompt = f"""You are playing FrozenLake, a grid navigation game.

## Grid Layout (4x4)
S=Start  F=Frozen(safe)  H=Hole(danger)  G=Goal  [A]=Your position

{grid_visual}

## Current Status
{action_line}Outcome: {info}
Step number: {self.step_count}

## Your Position
- Row {row}, Column {col} (0-indexed from top-left)
- Current tile: {tile} — {TILE_MAP.get(tile, "Unknown")}

## Goal
Navigate from Start (top-left) to Goal (bottom-right) without falling into a Hole.

## Available Actions
- Left  (move one column left)
- Down  (move one row down)
- Right (move one column right)
- Up    (move one row up)

Note: Moving into a wall keeps you in place.

What is your next action? Reply with exactly one word: Left, Down, Right, or Up.
"""
        return prompt

    def _render_grid_text(self, agent_row: int, agent_col: int) -> str:
        """Render the 4x4 grid as ASCII with the agent's position marked."""
        lines = []
        for r, row_str in enumerate(self.grid):
            cells = []
            for c, tile in enumerate(row_str):
                if r == agent_row and c == agent_col:
                    cells.append("[A]")
                else:
                    cells.append(f" {tile} ")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_action(response: str) -> int:
        """
        Parse the LLM's text response into an action integer.
        Returns -1 if the response is not a valid action.
        """
        cleaned = response.strip().capitalize()
        # Handle variations like "move left", "go right", etc.
        for action_name, action_id in ACTION_REVERSE.items():
            if action_name in cleaned:
                return action_id
        return -1  # invalid

    def action_names(self) -> list[str]:
        return list(ACTION_MAP.values())


# ------------------------------------------------------------------
# Quick smoke test — run this file directly to verify the env works
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = TextFrozenLake(is_slippery=False)
    state = env.reset()

    print("=" * 60)
    print("FROZEN LAKE — Text Environment Demo")
    print("=" * 60)
    print(state["text_prompt"])

    # Take a few manual steps to demonstrate
    demo_actions = [2, 1, 2, 1, 1, 2]  # rough path toward goal
    for action in demo_actions:
        if state["done"]:
            break
        print(f"\n>>> Taking action: {ACTION_MAP[action]}")
        state = env.step(action)
        print(state["text_prompt"])
        if state["done"]:
            print(f"\nEpisode finished. Total reward: {state['total_reward']}")
            break

    env.close()
