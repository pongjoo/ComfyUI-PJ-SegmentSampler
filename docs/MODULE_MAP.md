# 模块地图

| 模块 | 实际入口 | 职责和数据归属 |
|---|---|---|
| 节点注册 | `__init__.py` | 三个节点及既有二采别名；前端目录声明 |
| 分段时间 | `segment_timing.py` | H3帧/潜步换算、就近匹配目标段长和重叠、按真实输入解析规划规则；不持有张量或运行状态 |
| 规划与预览 | `pj_segment_planner.py` | 规划总线、说明文本、可选FFmpeg预览；调用共享时间模块 |
| 采样支撑 | `segment_sampling.py` | 条件接口适配、噪声、遮罩切片、融合及 `SegmentCache`；只服务本插件 |
| 一采 | `pj_first_pass_sampler.py` | 分段重抽决策和采样编排；每实例拥有一份临时磁盘缓存 |
| 二采 | `pj_segment_sampler.py` | 精修编排、完整结果及去噪结果、音频旁路；每实例拥有一份临时磁盘缓存 |
| 前端 | `js/pj_segment_sampler.js` | 旧工作流控件迁移、操作说明、按模式显隐、高级设置折叠、播放器与仪表盘尺寸；节点移除时断开尺寸观察器 |
| 回归检查 | `tests/test_segments.py` | 真实ComfyUI条件/噪声接口配合记录式采样替身；CPU小张量检查，不加载模型权重 |
| 浏览器检查 | `tests/serve_dashboard.py`、`tests/dashboard.html` | 仅回环地址提供真实前端扩展的DOM布局检查页 |
| 工作流迁移检查 | `tests/test_planner_widgets.mjs` | 规划控件迁移、旧预览失效、采样模式显隐与高级设置值保留；使用Node内置测试器 |

数据流：实际LATENT + 规划规则 → 共享时间边界 → 条件与遮罩切片 → ComfyUI guider.sample → CPU结果及节点独占磁盘缓存 → 重叠融合 → 完整LATENT。

公共边界保留输出类型、`execute`/`plan` 调用参数及 `PJ_SEGMENT_PLAN` 既有字段。规划器输入已移除51帧开关，前端迁移旧控件数组；旧API参数继续接收但统一使用原生网格。总线新增 `timing_policy=nearest_h3`，卡片新增实际重叠秒数与帧数。`plan_segments` 方法委托共享函数，旧预览切片不作为采样长度权威；两条实际流的长度才是依据。

缓存文件仅包含本节点生成的CPU张量，使用 `weights_only=True` 读取；摘要包含输入和条件内容及模型/采样配置，锁定语义有意排除种子。无模块级持久张量缓存。

复用命令在README维护，不另外维护一份。

前端折叠状态由节点属性 `pj_advanced_open` 保存，说明面板不参与采样参数序列化；运行模式原值及后端接口保持不变。
