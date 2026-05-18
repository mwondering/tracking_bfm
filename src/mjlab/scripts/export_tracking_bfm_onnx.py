"""CLI for exporting tracking_bfm checkpoints to deployable ONNX files."""

from __future__ import annotations

import argparse

from mjlab.deployment import export_checkpoint_to_onnx


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export a tracking_bfm checkpoint to a deployable ONNX file."
  )
  parser.add_argument("--checkpoint", required=True, help="Path to the source .pt file.")
  parser.add_argument("--task-id", required=True, help="Registered tracking_bfm task id.")
  parser.add_argument(
    "--checkpoint-family",
    choices=["auto", "tracking", "distillation"],
    default="auto",
    help="Checkpoint family. Defaults to auto-detection.",
  )
  parser.add_argument(
    "--obs-group",
    default=None,
    help="Observation group override. Defaults to the family-specific group.",
  )
  parser.add_argument(
    "--motion-path",
    default=None,
    help="Directory containing .npz motion files for multi-motion tracking tasks.",
  )
  parser.add_argument(
    "--motion-file",
    default=None,
    help="Single .npz motion file for single-motion tracking tasks.",
  )
  parser.add_argument(
    "--student-history-steps",
    type=int,
    default=None,
    help="Distillation student sparse-command history_steps override.",
  )
  parser.add_argument(
    "--student-future-steps",
    type=int,
    default=None,
    help="Distillation student sparse-command future_steps override.",
  )
  parser.add_argument(
    "--student-robot-history-steps",
    type=int,
    default=None,
    help="Distillation student robot-state observation history_length override.",
  )
  parser.add_argument(
    "--output-name",
    default=None,
    help="Optional ONNX filename. Defaults to deploy_<checkpoint_stem>.onnx.",
  )
  parser.add_argument("--robot-name", default=None, help="Optional robot name metadata.")
  parser.add_argument("--device", default="cpu", help="Device used to rebuild the actor.")
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose ONNX export logging.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  onnx_path = export_checkpoint_to_onnx(
    checkpoint_path=args.checkpoint,
    task_id=args.task_id,
    checkpoint_family=args.checkpoint_family,
    obs_group=args.obs_group,
    motion_path=args.motion_path,
    motion_file=args.motion_file,
    student_history_steps=args.student_history_steps,
    student_future_steps=args.student_future_steps,
    student_robot_history_steps=args.student_robot_history_steps,
    output_name=args.output_name,
    robot_name=args.robot_name,
    device=args.device,
    verbose=args.verbose,
  )
  print(onnx_path)


if __name__ == "__main__":
  main()
