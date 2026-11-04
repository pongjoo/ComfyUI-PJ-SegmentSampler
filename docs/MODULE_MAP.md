# 模块地图

| 模块 | 实际入口 | 职责和数据归属 |
|---|---|---|
| 节点注册 | `__init__.py` | 三个节点及既有二采别名；前端目录声明 |
| 分段时间 | `segment_timing.py` | H3帧/潜步换算、区间规划、按真实输入解析规划规则；不持有张量或运行状态 |
| 规划与预览 | `pj_segment_planner.py` | 规划总线、说明文本、可选FFmpeg预览；调用共享时间模块 |
| 采样支撑 | `segment_sampling.py` | 条件接口适配、噪声、遮罩切片、融合及 `SegmentCache`；只服务本插件 |
| 一采 | `pj_first_pass_sampler.py` | 分段重抽决策和采样编排；每实例拥有一份临时磁盘缓存 |
| 二采 | `pj_segment_sampler.py` | 精修编排、完整结果及去噪结果、音频旁路；每实例拥有一份临时磁盘缓存 |
| 前端 | `js/pj_segment_sampler.js` | 旧工作流控件迁移、控件显隐、播放器与仪表盘尺寸；节点移除时断开尺寸观察器 |
| 回归检查 | `tests/test_segments.py` | 真实ComfyUI条件/噪声接口配合记录式采样替身；CPU小张量检查，不加载模型权重 |
| 浏览器检查 | `tests/serve_dashboard.py`、`tests/dashboard.html` | 仅回环地址提供真实前端扩展的DOM布局检查页 |

数据流：实际LATENT + 规划规则 → 共享时间边界 → 条件与遮罩切片 → ComfyUI guider.sample → CPU结果及节点独占磁盘缓存 → 重叠融合 → 完整LATENT。

公共边界保留节点 `INPUT_TYPES`、`RETURN_TYPES`、`execute`/`plan` 参数，以及 `PJ_SEGMENT_PLAN` 现有字段。`plan_segments` 方法委托共享函数。旧总线的预览切片不作为采样长度权威；两条实际流的长度才是依据。

缓存文件仅包含本节点生成的CPU张量，使用 `weights_only=True` 读取；摘要包含输入和条件内容及模型/采样配置，锁定语义有意排除种子。无模块级持久张量缓存。

复用命令在README维护，不另外维护一份。
