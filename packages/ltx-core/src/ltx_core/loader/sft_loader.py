import json
import logging
import os
import time

import safetensors
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps

logger = logging.getLogger("ltx_pipelines.progress")


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        total_bytes = 0
        for p in model_paths:
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
        logger.info(
            "safetensors load: %d shard(s), %.2fG total -> device=%s",
            len(model_paths),
            total_bytes / (1024**3),
            device,
        )
        t_total = time.monotonic()
        for shard_path in model_paths:
            t_shard = time.monotonic()
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value
            logger.info(
                "  shard %s loaded in %.2fs",
                os.path.basename(shard_path),
                time.monotonic() - t_shard,
            )
        logger.info(
            "safetensors load done in %.2fs  (%d tensors, %.2fG materialized)",
            time.monotonic() - t_total,
            len(sd),
            size / (1024**3),
        )

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        with safetensors.safe_open(path, framework="pt") as f:
            meta = f.metadata()
            if meta is None or "config" not in meta:
                return {}
            return json.loads(meta["config"])

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)
