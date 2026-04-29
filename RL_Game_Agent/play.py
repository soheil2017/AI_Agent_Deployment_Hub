"""
play.py
-------
Run the LLM agent through FrozenLake episodes interactively.
Supports both OpenAI (Phase 1) and local Phi-3 + LoRA (Phase 2) backends.

Usage:
    python play.py                          # OpenAI, 1 episode
    python play.py --policy openai          # explicit OpenAI
    python play.py --policy local           # local Phi-3 + LoRA
    python play.py --episodes 5             # run 5 episodes
    python play.py --step-by-step           # pause between steps
    python play.py --max-steps 20           # limit steps per episode
    python play.py --model gpt-4o           # override OpenAI model
"""

import argparse
import time

from game_env import TextFrozenLake
from policy_base import BasePolicy


# ------------------------------------------------------------------
# Policy factory — isolates import cost to whichever backend is used
# ------------------------------------------------------------------

def load_policy(policy_name: str, args: argparse.Namespace) -> BasePolicy:
    if policy_name == "openai":
        from llm_policy_openai import OpenAIPolicy
        return OpenAIPolicy(
            model=args.model or "gpt-4o-mini",
            temperature=0.2,
        )
    elif policy_name == "local":
        from llm_policy_local import LocalLLMPolicy
        return LocalLLMPolicy(
            load_in_4bit=not args.no_4bit,
            load_existing_adapter=True,
        )
    else:
        raise ValueError(f"Unknown policy '{policy_name}'. Choose: openai | local")


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def run_episode(
    env: TextFrozenLake,
    policy: BasePolicy,
    episode_num: int,
    max_steps: int,
    step_by_step: bool,
) -> dict:
    """Run a single episode. Returns a summary dict."""
    state = env.reset()

    print(f"\n{'=' * 60}")
    print(f"  EPISODE {episode_num}")
    print(f"{'=' * 60}")
    print(state["text_prompt"])

    history = []  # each entry: {step, state_prompt, action, action_int, reward, info}

    for step in range(max_steps):
        if state["done"]:
            break

        if step_by_step:
            input("\n[Press Enter to continue...]\n")

        # Policy decides the next action
        action_int, action_text = policy.act(state["text_prompt"])
        print(f"\n>>> Agent chose: {action_text}")

        history.append({
            "step": step,
            "state_prompt": state["text_prompt"],
            "action": action_text,
            "action_int": action_int,
        })

        # Step the environment
        state = env.step(action_int)
        history[-1]["reward"] = state["reward"]
        history[-1]["info"] = state["info"]

        print(state["text_prompt"])

        if state["done"]:
            break

        time.sleep(0.2)

    won = state["total_reward"] > 0
    summary = {
        "episode": episode_num,
        "steps": state["step"],
        "total_reward": state["total_reward"],
        "won": won,
        "history": history,
    }

    print(f"\n{'─' * 60}")
    print(f"  Episode {episode_num} complete")
    print(f"  Steps taken : {state['step']}")
    print(f"  Total reward: {state['total_reward']}")
    print(f"  Result      : {'WIN' if won else 'LOSS'}")
    print(f"{'─' * 60}")

    return summary


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LLM agent on FrozenLake")
    parser.add_argument(
        "--policy", choices=["openai", "local"], default="openai",
        help="Which LLM backend to use (default: openai)",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Override OpenAI model (e.g. gpt-4o). Ignored for local.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (default: 1)")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Max steps per episode (default: 30)")
    parser.add_argument("--step-by-step", action="store_true",
                        help="Pause between each step (hit Enter to advance)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization for local policy (CPU/MPS safe)")
    args = parser.parse_args()

    print(f"\n[play.py] Loading '{args.policy}' policy...\n")
    policy = load_policy(args.policy, args)
    env = TextFrozenLake(is_slippery=False)

    results = []
    for ep in range(1, args.episodes + 1):
        summary = run_episode(
            env=env,
            policy=policy,
            episode_num=ep,
            max_steps=args.max_steps,
            step_by_step=args.step_by_step,
        )
        results.append(summary)

    env.close()

    if args.episodes > 1:
        wins = sum(1 for r in results if r["won"])
        avg_steps = sum(r["steps"] for r in results) / len(results)
        print(f"\n{'=' * 60}")
        print(f"  OVERALL RESULTS ({args.episodes} episodes, policy={args.policy})")
        print(f"  Win rate  : {wins}/{args.episodes} ({100 * wins / args.episodes:.1f}%)")
        print(f"  Avg steps : {avg_steps:.1f}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
