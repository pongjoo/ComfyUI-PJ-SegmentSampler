from .pj_segment_sampler import PJ_SegmentSampler, PJSegmentSampler

NODE_CLASS_MAPPINGS = {
    "PJ_SegmentSampler": PJ_SegmentSampler,
    "PJSegmentSampler": PJSegmentSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PJ_SegmentSampler": "PJ分段采样器",
    "PJSegmentSampler": "PJ分段采样器 (别名)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
