# 模块地图

| 模块 | 实际入口 | 职责和数据归属 |
|---|---|---|
| 节点注册 | `__init__.py` | 四个节点及既有二采别名；前端目录声明 |
| 分段时间 | `segment_timing.py` | H3帧/潜步换算、就近匹配目标段长和重叠、按真实输入解析规划规则；不持有张量或运行状态 |
| 规划与预览 | `pj_segment_planner.py` | 规划总线、说明文本、可选FFmpeg预览；调用共享时间模块 |
| 采样支撑 | `segment_sampling.py` | 条件接口适配、噪声、遮罩切片、融合及 `SegmentCache`；只服务本插件 |
| 编辑准备 | `pj_h3_edit_prepare.py` / `PJ_H3EditPrepare.prepare` | 检查原视频/目标/遮罩对齐；像素MASK按H3时空跨度合并，填入原视频并输出原生AV LATENT/noise_mask；无缓存 |
| 接缝约束 | `segment_boundaries.py` | 按音视频真实交集复制边界、合并原生遮罩、空间零遮罩优先于缓存边界，恢复固定内容以消除舍入漂移；无持久状态 |
| 顺序接力 | `segment_relay.py:sample_relay` | 当前窗口噪声、前段尾部约束、依赖链缓存校验与后缀重建、CPU去重拼接；使用一采实例拥有的磁盘缓存 |
| 一采 | `pj_first_pass_sampler.py` | 分段重抽决策和采样编排；规划可选，未接时整段采样；每实例拥有一份临时磁盘缓存 |
| 二采 | `pj_segment_sampler.py` | 顺序精修、视频边界承接、后续缓存重建、完整结果及去噪结果、音频旁路；规划可选，未接时整段采样；每实例拥有一份临时磁盘缓存 |
| 前端 | `js/pj_segment_sampler.js` | 旧工作流控件迁移、原生控件按连接/模式/段数显隐、高级设置折叠、播放器与仪表盘尺寸；节点移除时断开尺寸观察器 |
| 回归检查 | `tests/test_segments.py` | 真实ComfyUI条件/噪声接口配合记录式采样替身；CPU小张量检查，不加载模型权重 |
| 浏览器检查 | `tests/serve_dashboard.py`、`tests/dashboard.html` | 仅回环地址提供真实前端扩展的DOM布局检查页 |
| 工作流迁移检查 | `tests/test_planner_widgets.mjs` | 规划控件迁移、旧预览失效、采样模式显隐与高级设置值保留；使用Node内置测试器 |

数据流：实际LATENT + 规划规则 → 共享时间边界 → 条件与遮罩切片 → ComfyUI guider.sample → CPU结果及节点独占磁盘缓存 → 重叠融合 → 完整LATENT。

公共边界保留输出类型、`execute`/`plan` 调用参数及 `PJ_SEGMENT_PLAN` 既有字段。规划器输入已移除51帧开关，前端迁移旧控件数组；旧API参数继续接收但统一使用原生网格。总线新增 `timing_policy=nearest_h3`，卡片新增实际重叠秒数与帧数。`plan_segments` 方法委托共享函数，旧预览切片不作为采样长度权威；两条实际流的长度才是依据。

缓存文件仅包含本节点生成的CPU张量，使用 `weights_only=True` 读取；摘要包含输入和条件内容及模型/采样配置，锁定语义有意排除种子。无模块级持久张量缓存。

复用命令在README维护，不另外维护一份。

前端折叠状态由节点属性 `pj_advanced_open` 保存；采样器无说明面板，使用原生运行模式和高级设置按钮。二采新建节点默认全部精修，载入工作流恢复保存值。未接规划隐藏分段控件，后端执行整段采样。`toggleWidget` 同步控件及所属插口绘制；已连接参数保留布局行，不删除、重排或断开插口。

一采追加可选 `generation_mode`、`relay_action`、`relay_start_segment`，既有控件顺序及输出不变。接力路径由 `plan_relay_segments` 根据真实AV长度与目标段长计算固定7潜步交叠；总线/预览增加 `relay_n_chunks`，普通规划不受影响。接力缓存摘要包含本段输入、种子、条件及前段结果摘要；首次失效处先撤销整个后缀，再逐段采样保存。结果仅追加未写入区间；没有跨执行张量存储。

一采可选 `boundary_constraints`（默认关闭）由采样器先决定保留缓存，再调用 `constrain_boundaries` 构建原生LATENT/遮罩，交给 `guider.sample`；`restore_preserved` 按合成后的零遮罩原地恢复固定内容；最终拼接后再次恢复输入的零遮罩内容，最后按首段拥有重叠区组装。接缝模式缓存摘要包含全规划和完整输入；重建开始时撤销旧缓存索引，完成组装后保存本轮结果，避免中断后读取混合版本。

二采默认调用同一边界模块，仅向视频重叠区传入此前已完成的精修结果；输入空间零遮罩优先。音频遮罩及音频融合保持原有职责。采样和x0结果恢复固定值，视频使用`保留边界`单一归属拼接；内部时间遮罩不传出。缓存版本`refine-boundaries-v1`，从首个需重采/失效/缺失的段起，先用`SegmentCache.retain`删除整个后缀，再顺序保存新结果；前缀复用不改变历史种子锁定语义。前端二采过渡控件仅用于非旁路音频。

空间编辑数据流：目标AV LATENT + 原视频VAE LATENT + 外部MASK → `PJ_H3EditPrepare` → 标准LATENT（原视频内容与noise_mask）→ 一采。准备节点不拥有SAM、VAE或参考编码。接缝约束仅向非零空间遮罩内复制缓存值，黑色区域以输入原视频为准；缓存摘要包含输入及遮罩，普通接缝版本为`boundaries-v2`，接力为`relay-22-v1`。
