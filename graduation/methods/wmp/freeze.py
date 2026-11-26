"""Build an inference snapshot from a trusted WMP training checkpoint."""

import torch


def freeze_checkpoint(source, output):
    """Export the frozen teacher and return its saved iteration."""
    saved = torch.load(source, map_location="cpu", weights_only=False)
    fields = ("model_state_dict", "world_model_dict", "depth_predictor", "discriminator", "signatures",
              "amp_normalizer", "iter", "train_cfg", "wm_config", "wm_updates_done", "depth_updates_done")
    frozen = {key: saved[key] for key in fields}
    frozen["source"] = str(source.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(frozen, temporary)
    temporary.replace(output)
    return frozen["iter"]
