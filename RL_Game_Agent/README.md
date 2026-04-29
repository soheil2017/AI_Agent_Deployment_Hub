# RL Game Agent — Hybrid LLM + Reinforcement Learning

A hands-on project for learning Reinforcement Learning by building a hybrid agent that combines a Large Language Model (LLM) as a decision-making policy with a standard RL training loop. The agent plays **FrozenLake**, a classic grid-navigation game, and improves over time through human feedback and (later) rule-based rewards.

---

## What Is This Project?

Traditional RL agents learn by trial and error using numerical reward signals. This project takes a different approach: instead of training a policy network from scratch, we use an LLM as the policy — it reads a text description of the game state and decides what action to take, just like a human would.

The learning signal comes from **you** (human feedback) and eventually from **game rules** (rule-based rewards), making this a practical introduction to RLHF (Reinforcement Learning from Human Feedback).

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Game Environment               │
│       TextFrozenLake (game_env.py)          │
│  Converts grid state → text description     │
└────────────────────┬────────────────────────┘
                     │  text prompt
                     ▼
┌─────────────────────────────────────────────┐
│              LLM Policy                     │
│  Phase 1: OpenAIPolicy (llm_policy_openai)  │
│  Phase 2: LocalLLMPolicy (llm_policy_local) │
│  Both expose: act(prompt) → (int, str)      │
└────────────────────┬────────────────────────┘
                     │  action
                     ▼
┌─────────────────────────────────────────────┐
│              Reward Module                  │
│  Phase 1: Human feedback (you rate 1–5)     │
│  Phase 2: + Rule-based (win/loss/distance)  │
└────────────────────┬────────────────────────┘
                     │  reward signal
                     ▼
┌─────────────────────────────────────────────┐
│           RL Training Loop (PPO)            │
│      TRL + PEFT (LoRA adapter on Phi-3)     │
└─────────────────────────────────────────────┘
```

---

## The Game: FrozenLake

FrozenLake is a 4×4 grid where the agent must navigate from **Start** (top-left) to the **Goal** (bottom-right) without falling into a **Hole**.

```
 S   F   F   F       S = Start
 F   H   F   H       F = Frozen (safe to walk on)
 F   F   F   H       H = Hole (episode ends, reward = 0)
 H   F   F   G       G = Goal (reward = 1)
```

**Actions:** Left · Down · Right · Up

The agent receives a full text description of the board at each step, including its current position marked as `[A]`, making it easy for an LLM to reason about.

---

## Two-Phase Roadmap

### Phase 1 — Play & Collect (current)
- OpenAI (`gpt-4o-mini`) acts as the policy
- Human provides reward feedback after each episode (rating 1–5)
- Trajectories are saved as a dataset for fine-tuning
- Goal: observe the agent, understand the RL loop, collect training data

### Phase 2 — Train & Improve (upcoming)
- Phi-3-mini (local, 3.8B params) replaces OpenAI as the policy
- LoRA adapters are fine-tuned via PPO using TRL
- Rule-based rewards are added alongside human feedback
- Goal: see measurable improvement in win rate after training

---

## Project Structure

```
RL_Game_Agent/
│
├── game_env.py             # FrozenLake wrapped with text state descriptions
├── policy_base.py          # Abstract base class — shared interface for all policies
├── llm_policy_openai.py    # Phase 1: OpenAI-backed policy (gpt-4o-mini)
├── llm_policy_local.py     # Phase 2: Local Phi-3-mini + LoRA adapter
├── play.py                 # Run the agent interactively (no training)
│
├── reward.py               # [upcoming] Human feedback CLI + reward logging
├── train.py                # [upcoming] PPO training loop via TRL
│
├── lora_adapter/           # [auto-created] Saved LoRA weights after training
├── .env.example            # API key template — copy to .env
└── requirements.txt        # All dependencies
```

---

## Setup

### 1. Install dependencies

```bash
cd RL_Game_Agent
pip install -r requirements.txt
```

### 2. Configure your OpenAI API key

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...
```

### 3. Verify the environment (no API key needed)

```bash
python game_env.py
```

This runs a pre-scripted demo of the text environment so you can see what the LLM receives as input.

---

## Running the Agent

### Basic run (OpenAI, 1 episode)
```bash
python play.py
```

### Multiple episodes with stats
```bash
python play.py --episodes 10
```

### Step-by-step mode (pause between each action)
```bash
python play.py --step-by-step
```

### Use a more powerful OpenAI model
```bash
python play.py --model gpt-4o
```

### Switch to local Phi-3 + LoRA (Phase 2)
```bash
python play.py --policy local
```

### All options
```
--policy      openai | local      Which backend to use (default: openai)
--model       gpt-4o-mini         Override OpenAI model ID
--episodes    1                   Number of episodes to run
--max-steps   30                  Max steps allowed per episode
--step-by-step                    Pause and press Enter between steps
--no-4bit                         Disable 4-bit quantization (for CPU/MPS)
```

---

## Key Concepts Illustrated

### 1. Text as State Representation
Instead of passing raw numbers to a neural network, we convert the game state into natural language that an LLM can reason about. The prompt includes a visual grid, position description, last action, and outcome.

### 2. LLM as Policy
The LLM reads the state and outputs an action word (`Left`, `Down`, `Right`, `Up`). This replaces the traditional neural network policy used in standard RL.

### 3. Policy Abstraction
Both `OpenAIPolicy` and `LocalLLMPolicy` inherit from `BasePolicy` and expose the same `act(prompt) → (action_int, action_text)` interface. The rest of the system (`play.py`, `train.py`) never needs to know which backend is active.

```python
# Both work identically — swap with one argument
policy = OpenAIPolicy()       # Phase 1
policy = LocalLLMPolicy()     # Phase 2

action_int, action_text = policy.act(state["text_prompt"])
```

### 4. LoRA Fine-tuning (Phase 2)
LoRA (Low-Rank Adaptation) freezes the base model weights and adds small trainable matrices to the attention layers. This means:
- Only ~0.1% of parameters are updated during training
- The base model's general knowledge is preserved
- Training fits in a fraction of the memory required for full fine-tuning

```
Total params:     ~3.8B  (Phi-3-mini)
Trainable (LoRA): ~3.7M  (~0.1%)
LoRA targets:     q_proj, k_proj, v_proj, o_proj
LoRA rank (r):    8
```

### 5. RLHF → Rule-Based (Reward Shaping)
Phase 1 uses human ratings (1–5) as the reward signal — you judge whether the agent made good decisions. Phase 2 adds rule-based rewards (e.g., `+0.1` for moving closer to the goal, `-1` for falling in a hole) so training can scale without requiring human input for every episode.

---

## How the State Prompt Looks

This is what the LLM receives at each step:

```
You are playing FrozenLake, a grid navigation game.

## Grid Layout (4x4)
S=Start  F=Frozen(safe)  H=Hole(danger)  G=Goal  [A]=Your position

[A]  F   F   F
 F   H   F   H
 F   F   F   H
 H   F   F   G

## Current Status
This is the start of the episode.
Outcome: Episode started.
Step number: 0

## Your Position
- Row 0, Column 0 (0-indexed from top-left)
- Current tile: S — Start

## Goal
Navigate from Start (top-left) to Goal (bottom-right) without falling into a Hole.

## Available Actions
- Left  (move one column left)
- Down  (move one row down)
- Right (move one column right)
- Up    (move one row up)

What is your next action? Reply with exactly one word: Left, Down, Right, or Up.
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `gymnasium` | FrozenLake game environment |
| `openai` | Phase 1 — OpenAI API policy |
| `python-dotenv` | Load API key from `.env` |
| `torch` | Tensor operations for local model |
| `transformers` | Load Phi-3-mini base model |
| `peft` | LoRA adapter creation and loading |
| `accelerate` | Efficient model loading across devices |
| `bitsandbytes` | 4-bit quantization (CUDA only) |
| `trl` | PPO training loop (Phase 2) |
| `datasets` | Store and load trajectory data (Phase 2) |

---

## Upcoming Files

| File | Description |
|---|---|
| `reward.py` | CLI for collecting human feedback ratings after each episode |
| `train.py` | PPO fine-tuning loop using TRL — trains the local LoRA adapter |
| `evaluate.py` | Compare win rates before and after training across N episodes |

---

## Learning Path

If you are new to RL, work through the project in this order:

1. **Run `game_env.py`** — understand the environment and the text state format
2. **Run `play.py`** — watch the OpenAI agent play and read its reasoning
3. **Read `policy_base.py`** — understand the policy abstraction
4. **Read `llm_policy_openai.py`** — see how OpenAI is used as a policy
5. **Add `reward.py`** — start collecting human feedback on episodes
6. **Add `train.py`** — run your first PPO training loop
7. **Switch to `--policy local`** — compare the trained local model vs OpenAI
