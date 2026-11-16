"""Random-number-generator seeding shared by solvers and detectors."""
import random

import numpy as np
import torch


def seed_all_rngs(seed: int) -> None:
  """Seed Python, NumPy and PyTorch (CPU and CUDA) RNGs, and make cuDNN deterministic."""
  seed_value = int(seed)
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

  if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
