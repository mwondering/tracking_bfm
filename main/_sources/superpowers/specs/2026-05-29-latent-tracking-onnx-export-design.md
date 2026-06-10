# Latent Tracking ONNX Export Design

## Goal

Add a dedicated export path that produces one deployable ONNX file for latent tracking policies. The exported model runs the latent tracking actor, clamps the latent action, runs the frozen latent decoder, and outputs final joint actions.

## Scope

In scope:

- New CLI entry point: `export-latent-tracking-bfm-onnx`.
- New deployment helper module for latent tracking export.
- Single ONNX graph containing actor observation normalization, actor MLP, latent clamp, decoder observation normalization, and decoder MLP.
- Explicit `--checkpoint` for the latent tracking actor checkpoint.
- Explicit `--decoder-checkpoint` for the latent distillation checkpoint used as the frozen decoder.
- Motion source overrides matching the existing tracking exporter: `--motion-path` or `--motion-file`.

Out of scope:

- Changing the existing `export-tracking-bfm-onnx` behavior.
- Exporting separate actor and decoder ONNX files.
- Embedding motion reference data into this deployment ONNX.

## ONNX Contract

Inputs:

- `obs`: flat latent tracking actor observation.
- `proprio`: flat decoder proprio observation.

Output:

- `actions`: final joint action tensor.

The graph computes:

```text
latent = actor(obs)
latent = clamp(latent, -latent_action_clip, latent_action_clip)
actions = decoder(concat(proprio, latent))
```

## Architecture

The exporter rebuilds the latent tracking runner from the task registry. Before runner construction, it injects `latent_decoder_checkpoint_path` into the loaded runner config so the existing `LatentTrackingOnPolicyRunner` can create its latent action environment and actor with the correct latent action dimension.

The exporter then takes:

- `runner.alg.get_policy()` as the latent actor.
- `runner.env.decoder` as the frozen latent distillation model.

Both are wrapped through their existing `as_onnx()` compatible MLP export path. The combined deployment module accepts flat tensors rather than `TensorDict` so Torch ONNX export stays simple and deployment friendly.

## Metadata

The ONNX receives minimal metadata:

- `task_id`
- `checkpoint_family=latent_tracking`
- `decoder_checkpoint`
- `obs_group`
- `proprio_obs_group`
- `robot_name`, when provided

## Testing

Unit tests cover the combined module without constructing a full MuJoCo environment:

- actor latent output is clamped before decoding
- ONNX runtime output matches PyTorch output
- metadata and tensor names are written correctly
- output path defaults to `deploy_<checkpoint_stem>.onnx`

