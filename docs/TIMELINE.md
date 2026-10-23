# Contest timeline: 23 October 2026 (local time, UTC+3)

The Level 4 tournament ran until ~20:00. Ratings are Glicko-2 scores from the live leaderboard.

| Time | Event | Rating / rank |
|---|---|---|
| ~17:20 | GPU pod (RunPod, NVIDIA A40) set up. Self-play data generation started on all cores. | Hand-written model deployed: **1,435, global #147** |
| 18:20 | First trained networks: `a64` and `a96` both beat the deployed hand-written model 172/200. | n/a |
| 18:28–18:50 | `b96`, then `b64` trained on the full 734-shard dataset. | n/a |
| 18:53 | `b64` beats `a64` 133/200 (+119 Elo) and passes the validator. | n/a |
| 18:55 | `b64` beats the deployed hand-written model 181/200 (90.5%). | n/a |
| **~18:57** | **`b64` deployed.** | n/a |
| 19:05 | `c96` (96×6) beats `b64` 122/200 (+78 Elo). Kept as a backup, since `b64` was winning in the arena. | `b64`: 8 wins out of 9 |
| 19:13 | `b64` climbing steadily. | 1,733, global #68 |
| 19:16 | Automatic improvement loop started: deeper data plus a 200-game gate on every new model. | n/a |
| 19:22 | n/a | **local #14 → climbing** |
| 19:31 | n/a | **1,907, local #8** (peak local rank) |
| 19:35 | Two losses to *The Posers* (local #7). | 1,886, local #10 |
| 19:38 | 25 wins out of 30 games since the upload (83%). | 1,900, global #43 |
| 19:41 | Teams around #10 jump to 1,984–1,992. **`b64` re-uploaded** to get bigger rating swings. | 1,902, local #11 |
| 19:44 | **96-channel model (`a96`) uploaded.** | **2,009** (+100 in ~3 games) |
| 19:47 | Loop promotes `it1` (beat `c96` 113/200). | 2,000, global #39 |
| 19:50 | **`it1` deployed.** Four losses in a row at the higher rating level. | ~1,880 |
| 19:53 | `a96` deployed again. 2 wins out of 3. | 1,798 → recovering |
| 20:00 | Tournament ends. | **Final: local #14** |

## Takeaway
Steady play by one strong model took the team from global #147 to local #8. In the last 15 minutes, upload-driven rating swings worked once (+100) and then against us (−120). See [LESSONS.md](LESSONS.md).
