"""Monte-Carlo forecast of final tournament rank, as used live during the contest.

Models the remaining arena games of our deployed model as independent Bernoulli trials
with an uncertain win rate and Glicko-like rating steps, while the other teams' ratings
also drift. Outputs local-rank probabilities and a chart.

Usage: python analysis/simulate_final_rank.py [--games 16] [--score 1900]
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local leaderboard (CREIC Cluj-Napoca) at 19:39, excluding our team.
LOCAL_BOARD = [2715, 2235, 2189, 2176, 2142, 2083, 2030, 2005, 1981, 1903,
               1882, 1867, 1858, 1857, 1770, 1757]


def simulate(score, games, n=200_000, win_rate=(0.72, 0.88), win_step=(10, 18),
             loss_step=(14, 24), others_sd=35, seed=5):
    rng = random.Random(seed)
    counts = [0] * (len(LOCAL_BOARD) + 2)
    finals = []
    for _ in range(n):
        p = rng.uniform(*win_rate)
        s = score
        for _ in range(games):
            s += rng.uniform(*win_step) if rng.random() < p else -rng.uniform(*loss_step)
        others = [b + rng.gauss(0, others_sd) for b in LOCAL_BOARD]
        counts[1 + sum(o > s for o in others)] += 1
        finals.append(s)
    finals.sort()
    return [c / n * 100 for c in counts], finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--score", type=float, default=1900)
    ap.add_argument("--out", default="assets/local_rank_forecast.png")
    a = ap.parse_args()

    probs, finals = simulate(a.score, a.games)
    n = len(finals)
    print(f"final score: median {finals[n // 2]:.0f}, 80% band {finals[n // 10]:.0f}-{finals[9 * n // 10]:.0f}")
    for r in range(1, len(probs)):
        if probs[r] >= 0.1:
            print(f"  local #{r:<2d} {probs[r]:5.1f}%")
    print(f"P(local top 10) = {sum(probs[1:11]):.1f}%")

    ranks = list(range(1, len(probs)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f"#{r}" for r in ranks], probs[1:], color=["#2a9d8f" if r <= 10 else "#bbbbbb" for r in ranks])
    ax.set_xlabel("Final local rank")
    ax.set_ylabel("Chance (%)")
    ax.set_title(f"Simulated final local rank: {a.games} games left, start {a.score:.0f}\n"
                 f"Top 10: {sum(probs[1:11]):.0f}%")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
