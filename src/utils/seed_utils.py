import random
from typing import Optional

import numpy as np
import torch

from config.config import load_config


def set_global_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = load_config().seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
