# Latent Distillation G1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Mjlab-LatentDistillation-Flat-Unitree-G1` as a first-stage CVAE/VIB-style action distillation task.

**Architecture:** Extend the existing distillation stack with a config-selected latent student path while preserving the original MLP student path. The latent path uses `teacher_actor` for the encoder and a new `proprio_actor` group for the decoder, and optimizes action reconstruction plus fixed-standard-normal KL and latent smoothness losses.

**Tech Stack:** Python, PyTorch, TensorDict, RSL-RL `MLPModel`, mjlab task registry, pytest.

---

### Task 1: Environment Config and Task Registration

**Files:**
- Modify: `src/mjlab/tasks/distillation/mdp/observations.py`
- Modify: `src/mjlab/tasks/distillation/distillation_env_cfg.py`
- Modify: `src/mjlab/tasks/distillation/config/g1/rl_cfg.py`
- Modify: `src/mjlab/tasks/distillation/config/g1/__init__.py`
- Test: `tests/test_distillation_task.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
def test_latent_distillation_task_is_registered() -> None:
  import mjlab.tasks.distillation.config.g1  # noqa: F401

  env_cfg, rl_cfg = load_mjlab_cfgs("Mjlab-LatentDistillation-Flat-Unitree-G1")

  assert "teacher_actor" in env_cfg.observations
  assert "proprio_actor" in env_cfg.observations
  assert rl_cfg.student_model_type == "latent"
  assert rl_cfg.encoder_obs_group == "teacher_actor"
  assert rl_cfg.decoder_obs_group == "proprio_actor"


def test_proprio_actor_excludes_command_terms() -> None:
  env_cfg = unitree_g1_flat_distillation_env_cfg()

  terms = set(env_cfg.observations["proprio_actor"].terms.keys())

  assert {"projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"} <= terms
  assert "ee_pose" not in terms
  assert "base_lin_vel_b" not in terms
  assert "anchor_height_w" not in terms
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_distillation_task.py::test_latent_distillation_task_is_registered tests/test_distillation_task.py::test_proprio_actor_excludes_command_terms -q
```

Expected: fail because the new task and `proprio_actor` do not exist yet.

- [ ] **Step 3: Implement config and registration**

Add `build_proprio_actor_terms()` using robot proprio terms only. Add `proprio_actor` to `make_distillation_env_cfg()`. Add latent config fields and `unitree_g1_latent_distillation_runner_cfg()`. Register `Mjlab-LatentDistillation-Flat-Unitree-G1` with the same env cfg and the latent runner cfg.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_distillation_task.py::test_latent_distillation_task_is_registered tests/test_distillation_task.py::test_proprio_actor_excludes_command_terms -q
```

Expected: pass.

### Task 2: Latent Model

**Files:**
- Modify: `src/mjlab/tasks/distillation/rl/models.py`
- Test: `tests/test_distillation_algorithm.py`

- [ ] **Step 1: Write failing model test**

Add a test that builds `LatentDistillationModel` from TensorDict observations and asserts action, latent mean, and latent log-std shapes.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_distillation_algorithm.py::test_latent_distillation_model_outputs_actions_and_latent_stats -q
```

Expected: fail because `build_latent_student_model` is missing.

- [ ] **Step 3: Implement model**

Add `LatentDistillationModel` and `build_latent_student_model()`. Use RSL-RL `MLPModel` for encoder trunk and decoder. The encoder outputs `2 * latent_dim` values split into `mu` and `log_std`; the decoder consumes a TensorDict key containing concatenated decoder observations and sampled latent.

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
uv run pytest tests/test_distillation_algorithm.py::test_latent_distillation_model_outputs_actions_and_latent_stats -q
```

Expected: pass.

### Task 3: Latent Distillation Algorithm

**Files:**
- Modify: `src/mjlab/tasks/distillation/rl/algorithm.py`
- Test: `tests/test_distillation_algorithm.py`

- [ ] **Step 1: Write failing algorithm tests**

Add tests that assert `LatentActionDistillationAlgorithm.update()` changes parameters and returns `action_mse`, `kl_loss`, `kl_weight`, `latent_mu_norm`, `latent_std_mean`, and `total_loss`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_distillation_algorithm.py::test_latent_action_distillation_algorithm_updates_student -q
```

Expected: fail because the algorithm class is missing.

- [ ] **Step 3: Implement algorithm**

Add KL calculation, KL warmup, free-bits per dimension, latent smoothness over mini-batch latent means, optimizer, distributed gradient reduction, save, and load methods.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_distillation_algorithm.py::test_latent_action_distillation_algorithm_updates_student -q
```

Expected: pass.

### Task 4: Runner Wiring and Checkpoints

**Files:**
- Modify: `src/mjlab/tasks/distillation/rl/runner.py`
- Test: `tests/test_distillation_runner_smoke.py`

- [ ] **Step 1: Write failing runner tests**

Add tests for latent runner smoke learn and latent save/load round trip using the existing fake env and injected teacher adapter.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_distillation_runner_smoke.py::test_latent_distillation_runner_learn_smoke tests/test_distillation_runner_smoke.py::test_latent_distillation_runner_save_load_round_trip -q
```

Expected: fail because the runner always builds the MLP student.

- [ ] **Step 3: Implement runner wiring**

Switch on `student_model_type`. For latent mode, build latent student and latent algorithm. During rollout, collect both encoder and decoder observations. During update, pass a TensorDict with both groups to the latent algorithm. Preserve the original MLP path and checkpoint validation.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_distillation_runner_smoke.py::test_latent_distillation_runner_learn_smoke tests/test_distillation_runner_smoke.py::test_latent_distillation_runner_save_load_round_trip -q
```

Expected: pass.

### Task 5: Focused Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused distillation tests**

Run:

```bash
uv run pytest tests/test_distillation_task.py tests/test_distillation_algorithm.py tests/test_distillation_runner_smoke.py -q
```

Expected: pass.

- [ ] **Step 2: Run related distillation tests**

Run:

```bash
uv run pytest tests/test_distillation*.py -q
```

Expected: pass.
