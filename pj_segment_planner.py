"""
PJ_SegmentPlanner (PJ分段规划器 / 时序总控与可视分段播放器)
提供统一的前置分段计算，实现一采与二采的绝对时序对齐。
支持输入图像(IMAGE)与音频(AUDIO)，自带可视时间戳仪表盘与分段点击精准试听播放器！
"""

import os
import json
import uuid
import tempfile
import subprocess
import numpy as np
import torch

import folder_paths

from .segment_timing import (FPS, frames_for_tokens, tokens_for_frames_floor, plan_segments, plan_relay_segments,
                             segment_seconds, zero_overlap, delivery_chunks)

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ffmpeg_path = "ffmpeg"


def _generate_preview_mp4(images, audio, fps):
    """
    轻量快速合成低分辨率预览视频 (带音频)，供前端 LiteGraph 画布交互播放。
    """
    if images is None or len(images) == 0:
        return None

    try:
        temp_dir = folder_paths.get_temp_directory()
        rand_id = uuid.uuid4().hex[:8]
        filename = f"pj_planner_preview_{rand_id}.mp4"
        output_path = os.path.join(temp_dir, filename)

        # 1. 图像张量预处理与快速降采样 (限制最大边 512，极速编码且浏览器秒播不卡顿)
        n_frames, h, w, c = images.shape
        max_dim = max(h, w)
        if max_dim > 512:
            scale = 512.0 / float(max_dim)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            new_w = (new_w // 2) * 2
            new_h = (new_h // 2) * 2
            imgs_t = images.permute(0, 3, 1, 2)
            imgs_resized = torch.nn.functional.interpolate(
                imgs_t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            imgs_np = (imgs_resized.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            preview_w, preview_h = new_w, new_h
        else:
            new_w = (w // 2) * 2
            new_h = (h // 2) * 2
            imgs_np = (images[:, :new_h, :new_w, :].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            preview_w, preview_h = new_w, new_h

        # 2. 音频提取至临时 WAV 文件
        audio_temp_path = None
        if audio is not None and isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio and sf:
            try:
                wav_tensor = audio["waveform"]
                if len(wav_tensor.shape) == 3:
                    wav_data = wav_tensor[0].cpu().numpy().transpose()
                else:
                    wav_data = wav_tensor.cpu().numpy().transpose()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    sf.write(temp_wav.name, wav_data, audio["sample_rate"], format="WAV")
                    audio_temp_path = temp_wav.name
            except Exception as e:
                print(f"[PJ分段规划器] 临时音频提取跳过: {e}")

        # 3. 构建极速 FFmpeg 转码命令
        args = [
            ffmpeg_path, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{preview_w}x{preview_h}",
            "-r", str(fps),
            "-i", "-",
        ]
        if audio_temp_path and os.path.exists(audio_temp_path):
            args += ["-i", audio_temp_path]

        args += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "22",
            "-preset", "ultrafast",
        ]
        if audio_temp_path and os.path.exists(audio_temp_path):
            args += ["-c:a", "aac", "-b:a", "128k", "-shortest"]

        args.append(output_path)

        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=imgs_np.tobytes(), timeout=30)

        if audio_temp_path and os.path.exists(audio_temp_path):
            try:
                os.remove(audio_temp_path)
            except Exception:
                pass

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return filename
        return None
    except Exception as e:
        print(f"[PJ分段规划器] 预览视频合成失败: {e}")
        return None


class PJ_SegmentPlanner:
    """
    PJ分段规划器 (时钟计算 / 时序总控与分段播放器)
    作为独立的分段参数前置设定器，输出统一的 PJ_SEGMENT_PLAN 总线，
    直接连接到一采和二采采样器，一处修改全局自动同步！
    支持可选接入潜空间(LATENT)或视频图像(IMAGE)，在画布内实时渲染交互式分段仪表盘与试听播放器。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 1.0,
                        "tooltip": "输入图像序列的播放帧率；H3 分段和潜空间始终使用24fps模型时钟",
                    },
                ),
                "split_mode": (
                    ["按秒自动切分", "固定分段数", "自定义切点"],
                    {"default": "按秒自动切分", "tooltip": "分段规划策略"},
                ),
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 2.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "目标参考时长；后端选择最接近的H3合法帧数，实际时长可能略长或略短，尾段以输入长度为限",
                    },
                ),
                "overlap_seconds": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.25,
                        "tooltip": "0 不继承重叠画面，合法采样所需的右侧填充会裁掉；正值就近匹配H3合法重叠。自定义区间以切点为准",
                    },
                ),
                "num_segments": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "tooltip": "固定分段数模式下的分段数量",
                    },
                ),
                "custom_cuts": (
                    "STRING",
                    {
                        "default": "0-6, 4.5-10.5, 9-15",
                        "multiline": False,
                        "tooltip": "普通分段仅处理所填范围；22帧接力需从0连续覆盖输入预算，如0-15,15-30,30-45，各段时长对齐H3后扣重叠、不补尾。缺口、重叠或倒序会报错。",
                    },
                ),
            },
            "optional": {
                "latent": (
                    "LATENT",
                    {
                        "tooltip": "【可选】读取音视频潜空间实际长度；H3 使用固定24fps模型时钟",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "【可选】接入参考视频图像，按视频实际帧数生成分段与播放器预览",
                    },
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "【可选】接入音频原声，供播放器同步试听与透传",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PJ_SEGMENT_PLAN", "INT", "STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("分段规划(PLAN)", "总分段数", "分段信息明细", "图像透传", "音频透传")
    OUTPUT_NODE = True
    FUNCTION = "plan"
    CATEGORY = "PJ工具箱/分段采样"

    def plan(
        self,
        fps=24.0,
        split_mode="按秒自动切分",
        chunk_seconds=6.0,
        overlap_seconds=1.5,
        num_segments=3,
        custom_cuts="0-6, 4.5-10.5, 9-15",
        align_51_frames=False,
        latent=None,
        images=None,
        audio=None,
        **kwargs,
    ):
        source_fps = fps
        fps = FPS
        has_media = latent is not None or images is not None
        total_a = None
        if latent is not None:
            samples = latent["samples"]
            streams = list(samples.unbind()) if samples.is_nested else [samples]
            video = streams[0]
            total_v = video.shape[2]
            if samples.is_nested and video.ndim == 5 and video.shape[1] == 24:
                fps = FPS
            if len(streams) > 1:
                total_a = streams[1].shape[-1]
            mode_desc = "直连潜空间"
            length_policy = "existing_latent"
            length_note = "已有潜空间保留实际总长度。"
        else:
            source_frames = images.shape[0] * FPS / source_fps if images is not None else round(15 * FPS)
            total_v = tokens_for_frames_floor(source_frames)
            mode_desc = "直连视频图像（H3 帧网格预览）" if images is not None else "规则待命模式"
            length_policy = "floor_h3"
            length_note = ("输入不足5帧，预览使用H3最小5帧窗口。" if source_frames < 5 else
                           "总长度向下取整到H3合法帧数。")
        total_frames = frames_for_tokens(total_v)
        effective_seconds = total_frames / fps
        if total_a is None:
            total_a = round(effective_seconds * 40)
        chunks = plan_segments(total_v, total_a, split_mode, chunk_seconds, overlap_seconds,
                               num_segments, custom_cuts, fps, align_51_frames)
        rules = {"split_mode": split_mode, "overlap_seconds": overlap_seconds}
        delivered = delivery_chunks(chunks, rules)
        no_overlap = zero_overlap(rules)

        n_chunks = len(chunks)
        relay_preview = {"chunks": [], "n_chunks": 0, "error": ""}
        try:
            relay_chunks = plan_relay_segments(total_v, total_a, chunk_seconds, split_mode=split_mode,
                                               num_segments=num_segments, custom_cuts=custom_cuts)
        except ValueError as exc:
            # Ordinary custom ranges may be partial; only relay requires a complete budget.
            relay_preview["error"] = str(exc)
        else:
            for index, (start, end, _, _) in enumerate(relay_chunks):
                first, last = frames_for_tokens(start), frames_for_tokens(end)
                overlap = frames_for_tokens(relay_chunks[index - 1][1]) - first if index else 0
                relay_preview["chunks"].append(dict(segment=index + 1, start_frame=first, end_frame=last,
                    start_sec=round(first / FPS, 2), end_sec=round(last / FPS, 2),
                    duration_sec=round((last - first) / FPS, 2), overlap_frames=overlap,
                    overlap_seconds=round(overlap / FPS, 3)))
            relay_preview.update(n_chunks=len(relay_chunks), total_frames=frames_for_tokens(relay_chunks[-1][1]))
        relay_n_chunks = relay_preview["n_chunks"]

        # 3. 组织人类可读的明细文本与结构化分段信息
        chunks_sec_info = []
        summary_lines = [
            f"📊 【PJ分段时序总控仪表盘】",
        ]
        if not has_media:
            summary_lines.append(f"• 状态提示: ⚙️ 规则待命模式（未直连潜空间，下游采样器将按其输入潜空间实际长度动态切分）")
            summary_lines.append(f"• 参考基准: {effective_seconds:.2f}秒 ({total_frames} 帧, 潜空间 {total_v} 步)")
        else:
            summary_lines.append(f"• 状态提示: 🎬 已直连媒体 ({mode_desc})")
            summary_lines.append(f"• 视频总长: {effective_seconds:.2f}秒 ({total_frames} 帧, 潜空间 {total_v} 步)")

        summary_lines.append(f"• 规划模式: {split_mode} | 总计切分: {n_chunks} 个分段 | FPS: {fps}")
        timing_note = "秒数仅供参考；后端就近匹配H3合法帧数，实际结果以下方明细为准。"
        timing_note += " " + length_note
        if no_overlap:
            timing_note += " 零重叠：合法采样窗口的右侧填充将裁掉，不继承前段画面。"
        summary_lines.append(f"• {timing_note}")
        summary_lines.append("--------------------------------------------------")

        for i, (vs, ve, as_, ae) in enumerate(chunks):
            delivery_vs, delivery_ve, delivery_as, delivery_ae = delivered[i]
            s_sec, e_sec = segment_seconds(delivery_vs, delivery_ve, fps)
            s_frame = int(round(s_sec * fps))
            e_frame = min(total_frames, int(round(e_sec * fps)))
            dur_sec = round((e_frame - s_frame) / fps, 2)
            overlap_frames = max(0, min(e_frame, frames_for_tokens(delivered[i - 1][1])) - s_frame) if i else 0
            overlap_sec = round(overlap_frames / fps, 3)
            padding_frames = frames_for_tokens(ve) - frames_for_tokens(delivery_ve)
            summary_lines.append(
                f"  段 {i + 1:02d}: [{s_sec:5.2f}s ~ {e_sec:5.2f}s] ({dur_sec}s, 帧 {s_frame}~{e_frame}) (交付步 {delivery_vs}:{delivery_ve})"
                + (f" | 与前段重叠 {overlap_sec:.3f}s / {overlap_frames}帧" if i else "")
                + (f" | 采样右填充 {padding_frames}帧（裁除）" if padding_frames else "")
            )
            chunks_sec_info.append({
                "segment": i + 1,
                "start_sec": round(s_sec, 2),
                "end_sec": round(e_sec, 2),
                "duration_sec": dur_sec,
                "overlap_seconds": overlap_sec,
                "overlap_frames": overlap_frames,
                "start_frame": s_frame,
                "end_frame": e_frame,
                "vs": vs,
                "ve": ve,
                "as": as_,
                "ae": ae,
                "delivery_vs": delivery_vs,
                "delivery_ve": delivery_ve,
                "delivery_as": delivery_as,
                "delivery_ae": delivery_ae,
                "sample_start_frame": frames_for_tokens(vs),
                "sample_end_frame": frames_for_tokens(ve),
                "padding_frames": padding_frames,
            })
        summary_text = "\n".join(summary_lines)

        print("\n" + "=" * 65)
        print(summary_text)
        print("=" * 65 + "\n")

        # 4. 合成前端交互播放视频 (如果传入了 images)
        video_filename = _generate_preview_mp4(images, audio, source_fps) if images is not None else None

        plan_dict = {
            "has_media": has_media,
            "chunks": chunks,
            "delivery_chunks": delivered,
            "zero_overlap": no_overlap,
            "chunks_sec_info": chunks_sec_info,
            "n_chunks": n_chunks,
            "relay_n_chunks": relay_n_chunks,
            "relay_preview": relay_preview,
            "effective_seconds": round(effective_seconds, 2),
            "total_frames": total_frames,
            "total_v": total_v,
            "total_a": total_a,
            "split_mode": split_mode,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "num_segments": num_segments,
            "custom_cuts": custom_cuts,
            "align_51_frames": False,
            "timing_policy": "nearest_h3",
            "length_policy": length_policy,
            "fps": fps,
            "summary": summary_text,
            "has_video": video_filename is not None,
            "video_filename": video_filename,
        }

        ui_payload = {
            "preview_data": [{
                "video_filename": video_filename,
                "has_video": video_filename is not None,
                "has_media": has_media,
                "fps": fps,
                "total_frames": total_frames,
                "effective_seconds": round(effective_seconds, 2),
                "n_chunks": n_chunks,
                "relay_n_chunks": relay_n_chunks,
                "relay_preview": relay_preview,
                "chunks": chunks_sec_info,
                "zero_overlap": no_overlap,
                "summary": summary_text,
                "mode_desc": mode_desc,
                "timing_policy": "nearest_h3",
                "length_policy": length_policy,
                "timing_note": timing_note,
            }]
        }

        return {
            "ui": ui_payload,
            "result": (plan_dict, n_chunks, summary_text, images, audio),
        }
