import json
from pathlib import Path
import torch


def save_checkpoint(path, agent, config, step=0):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "networks": agent.networks.state_dict(),
        "target_networks": agent.target_networks.state_dict(),
        "step": step,
    }, path / "agent.pt")
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)


def load_checkpoint(path, agent, map_location=None):
    try:
        ckpt = torch.load(Path(path) / "agent.pt", map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(Path(path) / "agent.pt", map_location=map_location)
    agent.networks.load_state_dict(ckpt["networks"])
    agent.target_networks.load_state_dict(ckpt["target_networks"])
    return int(ckpt.get("step", 0))
