"""Replay motion from NPZ file using mjlab.

Usage:
    # Basic replay
    python src/mjlab/scripts/replay_npz.py --motion-path /path/to/motion.npz
    
    # Replay with cropping
    python src/mjlab/scripts/replay_npz.py --motion-path /path/to/motion.npz --crop 100 500 cropped_motion.npz
"""

from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.viewer import NativeMujocoViewer


class MotionData:
  """Container for motion data loaded from NPZ file."""
  
  def __init__(self, motion_file: str, device: str = "cuda:0"):
    self.motion_file = motion_file
    self.device = device
    self._load_motion()
  
  def _load_motion(self):
    """Load motion data from NPZ file."""
    data = np.load(self.motion_file)
    
    self.fps = float(data["fps"][0]) if "fps" in data else 50.0
    self.joint_pos = torch.from_numpy(data["joint_pos"]).to(torch.float32).to(self.device)
    self.joint_vel = torch.from_numpy(data["joint_vel"]).to(torch.float32).to(self.device)
    self.body_pos_w = torch.from_numpy(data["body_pos_w"]).to(torch.float32).to(self.device)
    self.body_quat_w = torch.from_numpy(data["body_quat_w"]).to(torch.float32).to(self.device)
    self.body_lin_vel_w = torch.from_numpy(data["body_lin_vel_w"]).to(torch.float32).to(self.device)
    self.body_ang_vel_w = torch.from_numpy(data["body_ang_vel_w"]).to(torch.float32).to(self.device)
    
    self.num_frames = self.joint_pos.shape[0]
    self.duration = self.num_frames / self.fps
    
    print(f"Motion loaded: {self.motion_file}")
    print(f"  Frames: {self.num_frames}, FPS: {self.fps}, Duration: {self.duration:.2f}s")
  
  def crop(self, start_frame: int, end_frame: int) -> "MotionData":
    """Crop motion data to specified frame range."""
    start_frame = max(0, start_frame)
    end_frame = min(self.num_frames, end_frame)
    
    if start_frame >= end_frame:
      raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}")
    
    cropped = MotionData.__new__(MotionData)
    cropped.motion_file = self.motion_file
    cropped.device = self.device
    cropped.fps = self.fps
    cropped.joint_pos = self.joint_pos[start_frame:end_frame].clone()
    cropped.joint_vel = self.joint_vel[start_frame:end_frame].clone()
    cropped.body_pos_w = self.body_pos_w[start_frame:end_frame].clone()
    cropped.body_quat_w = self.body_quat_w[start_frame:end_frame].clone()
    cropped.body_lin_vel_w = self.body_lin_vel_w[start_frame:end_frame].clone()
    cropped.body_ang_vel_w = self.body_ang_vel_w[start_frame:end_frame].clone()
    cropped.num_frames = end_frame - start_frame
    cropped.duration = cropped.num_frames / cropped.fps
    
    print(f"Motion cropped: frames {start_frame}-{end_frame} ({cropped.num_frames} frames, {cropped.duration:.2f}s)")
    return cropped
  
  def save(self, output_path: str):
    """Save motion data to NPZ file."""
    np.savez(
      output_path,
      fps=np.array([self.fps]),
      joint_pos=self.joint_pos.cpu().numpy(),
      joint_vel=self.joint_vel.cpu().numpy(),
      body_pos_w=self.body_pos_w.cpu().numpy(),
      body_quat_w=self.body_quat_w.cpu().numpy(),
      body_lin_vel_w=self.body_lin_vel_w.cpu().numpy(),
      body_ang_vel_w=self.body_ang_vel_w.cpu().numpy(),
    )
    print(f"Motion saved to: {output_path}")


class ZeroPolicy:
  """Dummy policy that returns zero actions."""
  
  def __init__(self, action_dim: int, device: str):
    self.action_dim = action_dim
    self.device = device
  
  def __call__(self, obs) -> torch.Tensor:
    batch_size = obs.shape[0] if isinstance(obs, torch.Tensor) else 1
    return torch.zeros(batch_size, self.action_dim, device=self.device)


class PolicyWithProgressBar:
  """Policy wrapper that displays progress bar based on motion playback."""
  
  def __init__(self, base_policy, env, motion_path: str):
    self.base_policy = base_policy
    self.env = env
    
    # Get motion info from environment's command manager
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    self.total_frames = motion_cmd.motion.time_step_total
    
    # Load fps from npz file
    data = np.load(motion_path)
    self.fps = float(data["fps"][0]) if "fps" in data else 50.0
    self.duration = self.total_frames / self.fps
    
    # Create progress bar
    self.pbar = tqdm(
      total=self.total_frames,
      desc=f"Replaying motion ({self.duration:.1f}s)",
      unit="frame",
      ncols=100,
      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]",
    )
    self.current_frame = 0
    self.last_time_step = 0
  
  def __call__(self, obs) -> torch.Tensor:
    # Get current motion time step from environment
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    # time_steps has shape (num_envs,), we use the first env
    current_time_step = int(motion_cmd.time_steps[0].item())

    # Update progress bar
    if current_time_step < self.last_time_step:
      # Motion looped back to the beginning, reset and advance to current step
      self.pbar.reset()
      if current_time_step > 0:
        self.pbar.update(current_time_step)
    else:
      # Normal forward progress
      delta = current_time_step - self.last_time_step
      if delta > 0:
        self.pbar.update(delta)

    self.last_time_step = current_time_step

    return self.base_policy(obs)
  
  def __del__(self):
    if hasattr(self, 'pbar'):
      self.pbar.close()


def main(
  motion_path: str,
  device: str = "cuda:0",
  crop_start: int | None = None,
  crop_end: int | None = None,
  crop_output: str | None = None,
  num_envs: int = 1,
):
  """Replay motion from NPZ file.
  
  Args:
    motion_path: Path to the NPZ motion file.
    device: Device to use for simulation.
    crop_start: Optional cropping start frame (inclusive).
    crop_end: Optional cropping end frame (exclusive).
    crop_output: Optional output filename for cropped motion. If not provided,
      a default name based on the input file and frame range is used.
    num_envs: Number of environments (default: 1).
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU.")
    device = "cpu"
  
  # Load motion for cropping if needed
  if crop_start is not None or crop_end is not None:
    if crop_start is None or crop_end is None:
      raise ValueError("Both `crop_start` and `crop_end` must be provided for cropping.")

    motion = MotionData(motion_path, device=device)
    start_frame = crop_start
    end_frame = crop_end
    motion = motion.crop(start_frame, end_frame)

    # Determine output filename
    if crop_output is not None:
      output_filename = crop_output
    else:
      # Default name: original_starte_end.npz
      stem = Path(motion_path).stem
      output_filename = f"{stem}_{start_frame}_{end_frame}.npz"

    # Save cropped motion in same directory as original
    motion_dir = Path(motion_path).parent
    output_path = motion_dir / output_filename
    motion.save(str(output_path))
    print(f"Cropped motion saved. Now replaying cropped motion...")
    motion_path = str(output_path)
  
  # Setup environment with motion file
  env_cfg = unitree_g1_flat_tracking_env_cfg(play=True)
  env_cfg.scene.num_envs = num_envs
  
  # Disable terminations for pure replay
  env_cfg.terminations = {}
  
  # Set motion file
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = motion_path
  
  # Create environment
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=1.0)
  
  # Create dummy policy (returns zero actions)
  action_dim = env.unwrapped.action_manager.total_action_dim
  base_policy = ZeroPolicy(action_dim, device)
  
  # Wrap policy with progress bar
  policy = PolicyWithProgressBar(base_policy, env, motion_path)
  
  # Run viewer
  print(f"\nStarting motion replay...")
  print(f"Press ESC to exit, SPACE to pause")
  viewer = NativeMujocoViewer(env, policy)
  viewer.run()
  
  env.close()
  print("\nReplay finished.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
