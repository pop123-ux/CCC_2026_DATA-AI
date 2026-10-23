# Lessons learned

## 1. Distillation beats hand-tuning, quickly
A 559k-parameter CNN trained for ~7 minutes on alpha-beta-labelled positions beat a carefully hand-written evaluation graph **90.5% of the time**. When the referee does the search for you, the network only has to supply good priors and evaluations. Copying a stronger searcher is the fastest way to get both.

## 2. Data quality > model size
| Change | Gain |
|---|---|
| More data (625 → 734 shards), same 64×6 net | +119 Elo |
| Wider net, 64 → 96 channels | +78 Elo |
| Fine-tuning on the same data | ≈ 0 (`c64`: −7, `d96`: −14) |
| Fine-tuning on **new, deeper-labelled** data | +45 Elo (`it1`) |

Once the data was exhausted, more epochs did nothing. New, better-labelled positions did help.

## 3. Gate every candidate
3 of the 6 fine-tuned candidates were no better than their parent. A fixed **200-game match with the real referee** (colours alternated) gives a ±3.5% confidence band on win rate, enough to separate +40 Elo from noise. Without it, weaker models would have been deployed.

## 4. Understand the rating system you are optimising
Glicko-2 tracks a rating **and** an uncertainty (RD). The size of each update grows with RD:

> Δrating ≈ K(RD) · (actual − expected)

A new upload appears to reset RD while keeping the rating:

- Right after an upload, each game moves the rating **by a lot** (we saw +100 and −120 within a few games).
- As RD shrinks, each game moves it **by a little** (~+8 per game for `b64` after 30 games).

**Implication:** higher variance helps when you're behind and need a jump, and hurts when you're defending a rank. **Lock in the strongest model early** and let it settle. Late uploads turn a strong position into a coin flip.

## 5. Internal Elo ≠ arena Elo
Every internal number was measured against our *own* models. At the arena's 1,950–2,000 level, opponents were as strong as our models and the win rate fell from ~83% to ~50%. Next time: download arena replays and opponent games early, and test against them.

## 6. Watch the leaderboard mechanics
The location leaderboard only updated a team's score **at upload time**, while the global board updated live. Reading the wrong board leads to wrong decisions.

## What I'd do with more time
- Full **AlphaZero-style self-play RL** with the referee MCTS, to go beyond imitating alpha-beta.
- Deeper teacher labels across the whole dataset.
- Speed-aware scaling: more blocks at 64 channels instead of wider layers.
- Evaluation against real opponents' models and replays.
