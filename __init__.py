from .pj_segment_sampler import PJ_SegmentSampler, PJSegmentSampler
from .pj_first_pass_sampler import PJ_FirstPass_SegmentSampler
from .pj_segment_planner import PJ_SegmentPlanner
from .pj_h3_edit_prepare import PJ_H3EditPrepare
from .pj_h3_context import PJ_H3SegmentContext

NODE_CLASS_MAPPINGS = {
    "PJ_H3SegmentContext": PJ_H3SegmentContext,
    "PJ_H3EditPrepare": PJ_H3EditPrepare,
    "PJ_SegmentPlanner": PJ_SegmentPlanner,
    "PJ_FirstPass_SegmentSampler": PJ_FirstPass_SegmentSampler,
    "PJ_SegmentSampler": PJ_SegmentSampler,
    "PJSegmentSampler": PJSegmentSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PJ_H3SegmentContext": "PJ H3分段条件准备",
    "PJ_H3EditPrepare": "PJ H3编辑区域准备 (原视频/空间遮罩)",
    "PJ_SegmentPlanner": "PJ分段规划器 (时序总控/可视播放器)",
    "PJ_FirstPass_SegmentSampler": "PJ一采分段重抽采样器 (单段重抽/缓存)",
    "PJ_SegmentSampler": "PJ二采分段精修采样器 (分段精修/释放显存)",
    "PJSegmentSampler": "PJ二采分段精修采样器 (别名)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
