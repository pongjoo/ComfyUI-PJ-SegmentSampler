# 模块地图

| 模块 | 实际入口 | 职责和数据归属 |
|---|---|---|
| 节点注册 | __init__.py | 五个节点、既有二采别名及前端目录；不持有结果 |
| 时间规划 | segment_timing.py | 帧/潜步换算、规则解析、zero_overlap、delivery_chunks；plan_relay_segments 默认先分配时长再扣接力重叠，complete_timeline=True 用于回修已有时间线；不持有张量 |
| 规划与预览 | pj_segment_planner.py | PJ_SEGMENT_PLAN、媒体/规则预览、可选 FFmpeg 播放器 |
| 空间编辑准备 | pj_h3_edit_prepare.py / PJ_H3EditPrepare.prepare | 对齐原视频、目标和 MASK，输出原生 AV LATENT/noise_mask；不拥有 SAM 或 VAE 编码 |
| H3 条件准备与目标创建 | pj_h3_context.py / PJ_H3SegmentContext.define_schema、execute、prepare、prepare_segment_guider | V3 图片/视频/配套音轨/独立音频四组动态接口，按编号排序并配对；execute 通过原生节点创建向下对齐的目标 AV 潜空间和空提示词基础条件；上下文拥有原始媒体引用与策略，逐窗独立裁剪各路视频和配套音轨，按实际尺寸建立条件；prepare 保持单上下文返回及旧单路参数契约 |
| 采样基础 | segment_sampling.py | 条件/遮罩切片、噪声、按交付范围融合及 SegmentCache；段缓存归采样器实例所有 |
| 边界保护 | segment_boundaries.py | constrain_boundaries、restore_preserved；空间零遮罩优先，恢复锁定值，无持久状态 |
| 局部回修 | segment_local_repair.py | repair_sigmas、repair_segments；旧结果拥有保留区，新输入可提供精修中部，仅写交付区，返回完整 sampled/x0 及变更范围 |
| 完整版本 | segment_revision.py / RevisionStore | 当前磁盘快照与临时候选；available/load/commit/check_request；原子提交、有限变更历史和过期请求检查，无持久张量 |
| 顺序接力 | segment_relay.py / sample_relay | 固定 22 帧接力、窗口噪声、可选逐段条件、依赖摘要和后缀失效；按最终交付边界分配 CPU 拼合结果并裁齐空间遮罩 |
| 一采编排 | pj_first_pass_sampler.py / execute、execute_with_ui | 原生成、载入基准、局部重绘、完整结果复用；拥有一采段缓存与完整版本，原生入口附带执行摘要 |
| 二采编排 | pj_segment_sampler.py / execute、execute_with_ui | 常规精修、局部替换、一采变更更新、高清复用；拥有高清快照和已消费一采版本，不修改一采存储 |
| 前端 | js/pj_segment_sampler.js | 旧控件迁移、显隐、高级折叠、预览尺寸；按执行摘要更新目标菜单，不决定采样状态 |
| 阶段示例 | workflows/局部回修_执行阶段示例.json | 原生 ComfySwitchNode 选择输出，未选择的放大/采样分支不执行 |
| 输出分析 | scripts/analyze_segment_outputs.py | 本地张量/解码数据检查：形状、有限值、变更范围、保护区、时间覆盖；画面指标只报告线索 |
| 运行监控 | scripts/monitor_comfyui.ps1 | 独立 PowerShell 7 只读本机指定任务、NVIDIA/Windows 资源和日志；记录归输出目录所有，任务结束自动退出，不参与采样 |
| 真实验证 | scripts/validate_h3_repair.py | 只读本机队列检查，独立原生执行器运行局部候选矩阵、完整潜空间放大后二采联动、长视频接力和完整解码；原始数据写 diagnostics/ |
| 音频对照 | scripts/validate_h3_audio.py | 按本地 JSON 配置调用原生接力、可复用固定首窗，导出 AV 张量和解码 WAV；复用验证脚本的本机队列与执行器消息入口，不判断对白语义 |
| 采样与版本检查 | tests/test_segments.py、test_local_repair.py、test_refine_revision.py、test_revision_requests.py、test_revision_review.py | CPU 小张量、原生接口及失败恢复契约 |
| 条件与执行检查 | tests/test_h3_context.py、test_context_sampling.py、test_stage_execution.py、test_execution_cache.py、test_zero_overlap.py | 原生 H3 条件构造、长窗口调度、原生 PromptExecutor 惰性分支及三类缓存、零交付重叠 |
| 分析与前端检查 | tests/test_output_analysis.py、test_validation_media.py、test_planner_widgets.mjs、serve_dashboard.py、dashboard.html | 检查报告、本地素材读取、迁移/显隐和独立 DOM 布局；不代替画质验收 |

## 数据流与接口

    输入 LATENT + 规划 → 合法采样窗口 / 交付窗口
    原始条件或 PJ_H3_CONTEXT → 每窗 guider → 原生采样 → 按时间归属拼合
    完整候选 → RevisionStore 原子提交 → 完整 LATENT + pj_revision
    一采完整结果 → 完整潜空间放大 → 二采窗口采样 → 高清完整版本
    一采完整结果 ──────────────→ 二采 source_result（版本与时间关联）

保留 PJ_SEGMENT_PLAN 旧字段和节点输出类型，用户秒数由后端匹配网格。自动/固定段数重叠为 0 时，delivery_chunks 裁去非末段右侧采样填充，普通采样禁止前段继承和运动引导；22 帧接力采用自身规则。采样后的首段次级输出保留合法采样窗；仅载入外部基准时次级为完整基准，复用时读取已保存结果，不重新请求规划器。完整输出才是最终交付时间线。

prepare_segment_guider(base_guider, context, bounds, video_shape, previous_tail=None, segment_index=0) 接收绝对采样边界与当前窗口形状，调用者不得先重复裁剪 guider。运动尾部须相位对齐且为 2+5k 视频潜步，不构造反向多帧引导。未接上下文时继续原有条件切片，不依赖 Easy 私有模块。

采样编排使用 pj_h3_context.prepare_segment_conditions(context, chunks, video_shape, indices) 上下文管理器，先编码待采样窗口并释放编码模型，再以 prepared.guider(base_guider, index, previous_tail) 读取单段条件、追加实时运动尾部。一次执行独占 pj_conditions_* 临时目录，保持原张量精度，成功或异常退出均清理，不缓存跨执行条件。接力/顺序精修传入失效后缀，独立分段和局部回修传入实际目标；窗口与缓存决策仍归采样编排所有。tests/test_condition_preparation.py 覆盖阶段顺序、选择范围、原生条件等价和失败清理。

repair_segments 的 baseline 拥有保留区，initial_latent 可为新放大结果；delivery_bounds 限制交付，editable_ranges 可精确指定一采变更。空间零遮罩与时间保护共同限制采样，内部时间遮罩不写回输出。局部回修保留原音，sampled 和 x0 分别组装，不用 sampled 冒充已有完整 x0。

## 状态归属

每个采样器实例独占临时目录，张量写磁盘并以 weights_only=True 加载。SegmentCache 管逐段复用，RevisionStore 管完整提交和请求记录，采样器决定提交时机。无全局张量缓存，不承诺跨重启恢复。

ComfyUI 调用 execute_with_ui，以 IS_CHANGED / NOT_IDEMPOTENT 避免原生输出缓存遮住节点当前版本。普通相同请求通过输入指纹与当前版本复用磁盘快照；execute 的既有元组返回契约不变。执行摘要含实际视频潜步数、模式、规则和段数；接力另记原输入潜步数，以区分重新生成的时长预算与已有结果回修（relay_repair）的完整范围。摘要不加入 LATENT；未提交时保留旧摘要，提交后的卸载错误不撤销新摘要。

pj_revision 仅附在完整结果，记录 lineage、version、视频/音频潜步长度及逐版本变更区间。历史有界，较旧消费者需要全量更新。二采通过 source_result 读取完整一采元数据，不依赖放大器透传；二采保留自己的已消费版本，不跨节点写缓存。

局部候选成功后才替换完整快照，失败保留旧版。当前请求可幂等复用，已被后续操作替代的近期请求拒绝覆盖；新抽卡使用新种子或操作编号。连续接力段缓存仍按依赖撤销后缀，两种策略不混用。

前端折叠状态归节点属性 pj_advanced_open。实际长度从 pj_sampling_summary 执行消息读取，仅保留在当前节点运行时；规则改变时重算段数、连接改变时丢弃旧摘要，不写工作流属性。版本消费由后端判断，原生惰性输入与 Switch 决定执行依赖。规划器保留 OUTPUT_NODE 预览行为，因此普通队列仍独立执行其媒体来源；阶段示例使用纯规则规划器，不能把采样器惰性复用等同于禁止其他输出分支执行。完整潜空间放大保持第三方公开接口，不绕过其归一化/分块规则增加局部近似缓存。

复用命令在 [README](../README.md) 维护；具体自动检查、真实权重配置和结果统一见 [验证报告](VALIDATION.md)，阶段决策见 [开发台账](DEV_LOG.md)，未交付项与发布状态见 [任务清单](../tasks/todo.md)。
