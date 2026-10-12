# Iterative QRFlow

The three stages are separate programs. They exchange a cumulative replay
manifest and actor/critic checkpoints; training never starts environment servers.

| Entry point | Input | Output |
|---|---|---|
| `collect_qrflow.py` | Previous actor and replay, or SFT for round 1 | New episodes, refreshed reference actions, `replay.json` |
| `train_qrflow_critic.py` | Replay and previous critic | Q, target Q, Adam and progress in `checkpoint.pt` |
| `train_qrflow_actor.py` | Replay, completed critic and previous actor | Actor, frozen reference, Adam and progress in `checkpoint.pt` |

```bash
# On a node with four GPUs and the existing OpenPI / LIBERO environments:
bash examples/run_iterative_rl.sh
```

The shell runs collection and reference caching on four GPUs, critic training
on one GPU, and actor training on four GPUs. It starts environment clients only
for collection and stops them before training. Set `NPROC`, `OUT_DIR`,
`CONFIG_NAME`, or `BASE_PORT` as needed. `MODEL_PYTHON` selects the OpenPI
environment interpreter and `ENV_PYTHON` selects the LIBERO interpreter.
Trailing arguments are Hydra overrides.
Output directories must be new, so an interrupted run cannot silently overwrite
its inputs or start the same collection twice.

Each stage can also be launched separately. Start the existing environment
clients with `--modes train` before collection; use the same configuration and
port range as the collector.

```bash
uv run --project model_backends/openpi torchrun --standalone --nproc_per_node=4 \
    examples/collect_qrflow.py hydra.run.dir=outputs/round1/collection

uv run --project model_backends/openpi python examples/train_qrflow_critic.py \
    runner.replay_path=outputs/round1/collection/replay.json \
    hydra.run.dir=outputs/round1/critic

uv run --project model_backends/openpi torchrun --standalone --nproc_per_node=4 \
    examples/train_qrflow_actor.py \
    runner.replay_path=outputs/round1/collection/replay.json \
    runner.critic_checkpoint=outputs/round1/critic/checkpoint.pt \
    runner.save_interval=100 hydra.run.dir=outputs/round1/actor
```

For later rounds, pass `runner.iteration`, `runner.actor_checkpoint` and
`runner.previous_replay` to collection; pass the previous
`runner.critic_checkpoint` to critic training. Actor training uses the new critic
checkpoint and the previous actor checkpoint. Q, target Q and critic Adam
continue across rounds; the actor gets a fresh Adam each round. To resume either
training stage, set `runner.resume_from` to that stage's checkpoint. Actor resume
requires the same world size. `runner.collect_data=False` refreshes reference
actions from `runner.previous_replay` without collecting additional episodes.

Defaults match the iterative experiment: 10 rounds, 500 episodes per round
(50 fixed initial states for each LIBERO-10 task), all cumulative replay, 500,000
critic updates and 1,000 actor updates per round. Collection and actor training
use one randomly chosen flow-SDE node and four ODE nodes. Every round caches
eight five-step ODE actions of the previous actor for **all** replay states.

The critic is a 10-head action-injected ensemble: 10-step TD, primitive discount
0.999, max of eight reference-action lower confidence bounds (coefficient 0.5),
target clipping to [-1000, 0]. The actor uses raw target-Q minus the mean Q of
the eight same-state reference actions, clipped to [-10, 10], beta 0.3, 32 paths,
leave-one-out costs and advantage normalization across the complete global
batch. Gradients accumulate by ordinary summation and are clipped once before
the optimizer step. Actor queries apply the physical motion bounds in normalized
coordinates, matching LIBERO's clipping of the first six motion coordinates.
They do not clamp to the normalization quantile interval or normalize generated
actions a second time. Executed actions are normalized once for replay.

Collection success rates describe the entering policy on fixed training initial
states, and those episodes enter replay. They are not held-out evaluations or
measurements of the actor after that round's update.

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
    model_backends/openpi/.venv/bin/python -m pytest tests/test_qrflow.py \
    tests/test_qrflow_openpi.py -q -p no:cacheprovider
```
