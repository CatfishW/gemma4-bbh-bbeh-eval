# Deployment Verification: 2026-07-10

Host: `benwulab-remote`

Public base URL: `https://llm.agaii.org/llm/v1`

## Services

| Unit | State | Enabled | Restarts at verification |
|---|---|---|---:|
| `gemma4-e4b.service` | active/running | yes | 0 |
| `gemma4-e2b.service` | active/running | yes | 0 |
| `gemma4-router.service` | active/running | yes | 0 |
| `gemma4-public-tunnel.service` | active/running | yes | 0 |

User lingering is enabled, so the user service manager starts without an
interactive login.

## Models

| Public model ID | Local model path | Local port | Max model length |
|---|---|---:|---:|
| `SubTokenLLM` | `/data/models/gemma-4-E4B-it` | 8888 | 131,072 |
| `SubTokenLLM-E2B` | `/data/models/gemma-4-E2B-it` | 8889 | 131,072 |

The body-preserving router listens on port 8890. The reverse tunnel publishes it
on Tang port 25570, where Nginx maps `/llm/v1/*` to `/v1/*`.

## Public checks

`GET /llm/v1/models` returned both expected model IDs.

Two chat-completion checks each contained exactly one `user` message and no
system-role message:

- E2B request returned exactly `E2B_SYSTEMD_READY`.
- E4B request returned exactly `E4B_SYSTEMD_READY`.

The unrelated `cap-voice` TTS service remained running throughout the migration.
At the final idle snapshot, GPU temperatures were 41 C and 39 C, power limits
were restored to 300 W, and total GPU memory use was approximately 29.6 GB and
24.9 GB.
