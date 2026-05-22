# Latent Distillation G1 Design

## Goal

Add a first-stage latent distillation task for Unitree G1:

```text
Mjlab-LatentDistillation-Flat-Unitree-G1
```

The task distills the existing whole-body tracking teacher
`Mjlab-Trackingbfm-Flat-Unitree-G1` into a reusable motor decoder:

```text
encoder(teacher_actor_obs) -> latent distribution
decoder(proprio_actor_obs, latent) -> joint action
```

This stage does not implement the second-stage velocity-conditioned RL encoder.

## Scope

In scope:

- Register a new task ID for latent distillation.
- Add a pure proprioception observation group for the decoder.
- Add a latent distillation model with encoder, reparameterized latent, and decoder.
- Add a latent action distillation algorithm with action reconstruction, KL, and latent smoothness losses.
- Extend the existing distillation runner to select either the original MLP student or the new latent student from config.
- Save latent checkpoints with both full policy state and explicit encoder/decoder state dictionaries.
- Preserve the existing `Mjlab-Distillation-Flat-Unitree-G1` behavior.

Out of scope:

- Second-stage velocity latent PPO.
- Frozen decoder policy wrappers.
- Learned prior, mixture prior, flow prior, or VQ latent.
- ONNX export for latent policies.
- Training script changes beyond task registration and config support.

## Architecture

The existing distillation stack remains the owner of first-stage distillation. The new task reuses `DistillationRunner`, but the runner switches implementation based on a config field:

```text
student_model_type = "mlp" | "latent"
```

The original task keeps `student_model_type="mlp"`. The new latent task uses:

```text
student_model_type="latent"
encoder_obs_group="teacher_actor"
decoder_obs_group="proprio_actor"
latent_dim=64
```

During rollout, the latent policy samples `z` from the encoder and decodes actions from proprioception. Teacher/student action mixing is unchanged. During update, batches contain both encoder and decoder observations.

## Observations

The environment config will expose three observation groups:

```text
teacher_actor: original tracking actor observation
student_actor: existing sparse-command student observation
proprio_actor: decoder-only proprioception observation
```

`proprio_actor` contains robot state only:

```text
projected_gravity
base_ang_vel
joint_pos
joint_vel
actions
```

It intentionally excludes motion reference, sparse end-effector command, future command, target body pose, and command-derived base velocity terms. This keeps the decoder interface compatible with the second-stage plan:

```text
decoder(proprio, z) -> action
```

## Model

`LatentDistillationModel` owns:

- `encoder`: normalized MLP over `encoder_obs_group`.
- `mu_head`: latent mean projection.
- `log_std_head`: latent log-standard-deviation projection.
- `decoder`: normalized MLP over `decoder_obs_group + z`.

The model clamps `log_std` to a configured range before sampling. It exposes:

```python
encode(obs) -> tuple[Tensor, Tensor]
sample(mu, log_std) -> Tensor
decode(obs, z) -> Tensor
forward(obs, deterministic=False) -> tuple[Tensor, dict[str, Tensor]]
```

Inference uses deterministic `z=mu` unless stochastic sampling is explicitly requested.

## Losses

The latent algorithm optimizes:

```text
total_loss = action_mse
           + effective_kl_weight * free-bits KL
           + latent_smooth_weight * latent_smooth_loss
```

KL is against a fixed standard normal:

```text
KL(q(z | teacher_actor) || N(0, I))
```

The KL weight warms up linearly over `kl_warmup_iterations`. Free bits are applied per dimension to avoid posterior collapse while keeping the latent distribution regularized.

Latent smoothness is computed over the flattened rollout order by penalizing adjacent latent mean differences within each mini-batch. It is a light regularizer and does not require recurrent state.

## Checkpoint Format

MLP checkpoints remain unchanged.

Latent checkpoints include:

```text
model_type = "latent"
policy_state_dict
encoder_state_dict
decoder_state_dict
optimizer_state_dict
latent_cfg
iter
infos.env_state
```

`policy_state_dict` keeps runner load/play behavior simple. `decoder_state_dict` makes the later second-stage implementation straightforward.

## Testing

Focused tests cover:

- New task registration and config fields.
- `proprio_actor` exists and excludes sparse command terms.
- Latent model forward shape and latent stats.
- Latent algorithm update changes parameters and reports KL metrics.
- Distillation runner latent smoke learning with injected teacher adapter.
- Latent save/load round trip.
- Existing plain distillation tests continue to pass.

Verification commands:

```bash
uv run pytest tests/test_distillation_task.py tests/test_distillation_algorithm.py tests/test_distillation_runner_smoke.py
uv run pytest tests/test_distillation*.py
```
