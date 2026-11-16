"""Implement your tabular Q-learning policy and trainer here."""

from lab import Lab


def encode_state(observation):
    """Return the row index of this observation in your Q table."""
    raise NotImplementedError("Choose your observation features and bins")


def decode_action(column):
    """Return a valid (lateral offset in m, target speed in m/s) pair."""
    raise NotImplementedError("Choose your discrete action set")


def reward(signals):
    """Return a scalar reward from racing.reward.RewardSignals."""
    raise NotImplementedError("Design your training reward")


def train(output_path, seed):
    """Use Lab(reward_fn=reward) and save the trained Q table and its metadata."""
    raise NotImplementedError("Implement the Lecture 6 Q-learning update and exploration")


def load_policy(checkpoint):
    """Load your submitted checkpoint and return act(observation) -> (offset, speed).

    Each call receives only the current observation. Evaluation must be deterministic
    with learning and exploration turned off. This factory is called for every lap.
    """
    raise NotImplementedError("Load the Q table and return your greedy policy")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train your tabular racing policy on Shanghai")
    parser.add_argument("--output", type=Path, default=Path("submission.npz"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.output, args.seed)
