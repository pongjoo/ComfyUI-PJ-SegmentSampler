try:
    from .pj_segment_sampler import PJ_SegmentSampler, PJSegmentSampler
    from .pj_first_pass_sampler import PJ_FirstPass_SegmentSampler
except ImportError:
    from pj_segment_sampler import PJ_SegmentSampler, PJSegmentSampler
    from pj_first_pass_sampler import PJ_FirstPass_SegmentSampler

NODE_CLASS_MAPPINGS = {
    "PJ_FirstPass_SegmentSampler": PJ_FirstPass_SegmentSampler,
    "PJ_SegmentSampler": PJ_SegmentSampler,
    "PJSegmentSampler": PJSegmentSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PJ_FirstPass_SegmentSampler": "PJ一采分段重抽采样器 (单段重抽/缓存)",
    "PJ_SegmentSampler": "PJ二采分段精修采样器 (高清无损/防爆显存)",
    "PJSegmentSampler": "PJ二采分段精修采样器 (别名)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
