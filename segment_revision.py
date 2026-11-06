"""Node-owned complete-result revisions, committed only after successful sampling."""

import copy
import os
from pathlib import Path
import tempfile
import uuid

import torch
from comfy.nested_tensor import NestedTensor

from .segment_timing import frames_for_tokens


def merge_intervals(intervals):
    merged = []
    for start, end in sorted((int(a), int(b)) for a, b in intervals if a < b):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def changed_intervals(metadata, since=None, stream="video"):
    """Return all changes since a consumed revision; unknown lineage requires full refresh."""
    total = metadata["video_tokens" if stream == "video" else "audio_tokens"]
    if not since or since.get("lineage") != metadata["lineage"]:
        return [[0, total]] if total else []
    version = int(since["version"])
    if version == metadata["version"]:
        return []
    if version > metadata["version"] or version < metadata["history_start"]:
        return [[0, total]] if total else []
    return merge_intervals(bounds for change in metadata["changes"]
                           if change["version"] > version for bounds in change[stream])


def _pack(value):
    if isinstance(value, NestedTensor):
        return {"__pj_nested__": [_pack(item) for item in value.unbind()]}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.untyped_storage().nbytes() > tensor.numel() * tensor.element_size():
            tensor = tensor.clone()
        return tensor
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_pack(item) for item in value)
    return value


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {"__pj_nested__"}:
            return NestedTensor(value["__pj_nested__"])
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_unpack(item) for item in value)
    return value


class RevisionStore:
    """A single current disk snapshot plus a temporary candidate; no retained tensors."""

    def __init__(self):
        self.directory = None
        self.metadata = None
        self.request_key = None
        self._completed_requests = []

    def check_request(self, key):
        if key == self.request_key:
            return True
        if key in self._completed_requests:
            raise ValueError("请求已过期，不能重复覆盖更新后的结果；请增加操作编号后重新提交")
        return False

    def load(self):
        if not self.available():
            return None
        path = Path(self.directory.name) / "current.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        return _unpack(saved["outputs"])

    def available(self):
        return self.directory is not None and (Path(self.directory.name) / "current.pt").is_file()

    def commit(self, latent, secondary=None, changes=(), audio_changes=(), fps=24.0,
               request_key=None, reset=False):
        for value in (latent, secondary):
            if value is None:
                continue
            samples = value["samples"]
            parts = samples.unbind() if samples.is_nested else [samples]
            if any(not torch.isfinite(part).all().item() for part in parts):
                raise ValueError("结果包含非有限值，未替换上一次完整结果；请检查模型与采样设置")
        streams = latent["samples"].unbind() if latent["samples"].is_nested else [latent["samples"]]
        total_v = streams[0].shape[2]
        total_a = streams[1].shape[-1] if len(streams) > 1 else 0
        old = None if reset else self.metadata
        version = old["version"] + 1 if old else 1
        history = copy.deepcopy(old["changes"]) if old else []
        history.append({"version": version, "video": merge_intervals(changes),
                        "audio": merge_intervals(audio_changes)})
        history = history[-128:]
        metadata = {"lineage": old["lineage"] if old else uuid.uuid4().hex,
                    "version": version, "video_tokens": total_v, "audio_tokens": total_a,
                    "frames": frames_for_tokens(total_v), "fps": float(fps),
                    "history_start": history[0]["version"] - 1, "changes": history}
        output = dict(latent, pj_revision=metadata)
        if secondary is not None:
            secondary = dict(secondary)
            secondary_samples = secondary["samples"]
            secondary_video = secondary_samples.unbind()[0] if secondary_samples.is_nested else secondary_samples
            if secondary_video.shape[2] == total_v:
                secondary["pj_revision"] = copy.deepcopy(metadata)
            else:
                secondary.pop("pj_revision", None)
        if self.directory is None:
            self.directory = tempfile.TemporaryDirectory(prefix="pj_revisions_")
        path = Path(self.directory.name) / "current.pt"
        pending = path.with_suffix(".pending")
        try:
            torch.save({"outputs": _pack((output, secondary))}, pending)
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
        self.metadata = copy.deepcopy(metadata)
        self.request_key = request_key
        if reset:
            self._completed_requests = []
        if request_key is not None:
            self._completed_requests = (self._completed_requests + [request_key])[-128:]
        return output, secondary
