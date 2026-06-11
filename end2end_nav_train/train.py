import multiprocessing
import os
import sys
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.config import AdvancedNavConfig
    from core.trainer import AdvancedNavTrainer
else:
    from .core.config import AdvancedNavConfig
    from .core.trainer import AdvancedNavTrainer


def main():
    multiprocessing.set_start_method("spawn", force=True)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = AdvancedNavConfig()
    trainer = AdvancedNavTrainer(cfg)
    trainer.run_all()


if __name__ == "__main__":
    main()
