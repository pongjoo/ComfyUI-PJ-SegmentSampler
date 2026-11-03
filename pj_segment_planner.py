"""
PJ_SegmentPlanner (PJ分段规划器 / 时序总控与可视分段播放器)
提供统一的前置分段计算，实现一采与二采的绝对时序对齐。
支持输入图像(IMAGE)与音频(AUDIO)，自带可视时间戳仪表盘与分段点击精准试听播放器！
"""

import os
import sys
import math
import json
import uuid
import tempfile
import subprocess
import numpy as np
import torch

import folder_paths

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
                        "tooltip": "视频帧率，用于换算秒数与潜空间步数，MiniMax H3 官方为 24.0",
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
                        "tooltip": "自动模式下单段最大时长（秒），推荐 5.0 ~ 7.0 秒",
                    },
                ),
                "overlap_seconds": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.25,
                        "tooltip": "相邻分段重叠过渡时长（秒），推荐 1.0 ~ 2.0 秒",
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
                        "tooltip": "自定义切分秒数范围，逗号分隔，如：0-6, 4.5-10.5, 9-15",
                    },
                ),
                "align_51_frames": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "开启51帧吸附",
                        "label_off": "关闭",
                        "tooltip": "开启后将窗口长度吸附到51帧/85音频步网格，确保绝对RoPE相位一致",
                    },
                ),
            },
            "optional": {
                "latent": (
                    "LATENT",
                    {
                        "tooltip": "【可选】接入采样器处理的潜空间。接入后直接读取其实际长度并呈现100%精准分段预览",
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
        # 1. 确定视频总帧数与时长 (秒)
        has_media = False
        mode_desc = ""

        if latent is not None and isinstance(latent, dict) and "samples" in latent:
            # 自动读取传入的实际 Latent 长度
            samples = latent["samples"]
            if hasattr(samples, "is_nested") and samples.is_nested:
                v_stream = list(samples.unbind())[0]
            else:
                v_stream = samples
            total_v = v_stream.shape[2]
            v_fps = fps / 3.4
            effective_seconds = float(total_v) / float(v_fps)
            eff_total_frames = int(round(effective_seconds * fps))
            has_media = True
            mode_desc = f"直连潜空间 ({total_v} 潜步, ~{effective_seconds:.2f}s)"
            print(f"[PJ分段规划器] 🚀 读取接入的实际 Latent 长度: {total_v} 潜步 (~{effective_seconds:.2f}秒, {eff_total_frames} 帧)")
        elif images is not None and hasattr(images, "shape") and len(images.shape) >= 1:
            eff_total_frames = int(images.shape[0])
            effective_seconds = float(eff_total_frames) / float(fps)
            v_fps = fps / 3.4
            total_v = int(round(effective_seconds * v_fps))
            has_media = True
            mode_desc = f"直连视频图像 ({eff_total_frames} 帧, ~{effective_seconds:.2f}s)"
            print(f"[PJ分段规划器] 🎬 采用输入图像序列实际帧数: {eff_total_frames} 帧 (~{effective_seconds:.2f}秒)")
        else:
            # 未连潜空间或视频：规则待命模式
            # 采用 15 秒（或旧工作流传入的 duration_seconds）作为参考时长生成预览规则
            effective_seconds = max(0.1, float(kwargs.get("duration_seconds", 15.0)))
            eff_total_frames = int(round(effective_seconds * fps))
            v_fps = fps / 3.4
            total_v = int(round(effective_seconds * v_fps))
            has_media = False
            mode_desc = "规则待命模式 (未直连潜空间，下游采样器将按其各自输入的潜空间长度动态适配)"
            print(f"[PJ分段规划器] ⚙️ 未直连潜空间输入，按参考基准 {effective_seconds:.2f} 秒 ({eff_total_frames} 帧) 预览分段规则")

        total_frames = eff_total_frames
        a_fps = 40.0
        total_a = int(round(effective_seconds * a_fps))

        chunks = []

        # 2. 精准切片时序规划
        if split_mode in ("按秒自动切分", "auto_by_seconds"):
            if align_51_frames:
                target_chunk_v = max(17, int(round(chunk_seconds * v_fps)))
                m_chunk = max(1, round((target_chunk_v - 2) / 15.0))
                win_v = m_chunk * 15 + 2
                target_ov_v = max(2, int(round(overlap_seconds * v_fps)))
                m_ov = max(1, round((target_ov_v - 2) / 15.0))
                stride_m = max(1, m_chunk - m_ov)
                stride_v = stride_m * 15
                v_cur = 0
                while v_cur < total_v:
                    v_start = v_cur
                    v_end = min(total_v, v_start + win_v)
                    if v_end < total_v:
                        rem = (v_end - 2) % 15
                        if rem != 0:
                            v_end = min(total_v, v_end + (15 - rem))
                    a_start = int(round(v_start * (85.0 / 15.0)))
                    a_end = min(total_a, int(round(v_end * (85.0 / 15.0))))
                    chunks.append((v_start, v_end, a_start, a_end))
                    if v_end >= total_v:
                        break
                    v_cur += stride_v
            else:
                win_v = max(5, int(round(chunk_seconds * v_fps)))
                rem = (win_v - 2) % 5
                if rem != 0:
                    win_v = win_v + (5 - rem)
                stride_v = max(5, int(round((chunk_seconds - overlap_seconds) * v_fps)))
                stride_v = (stride_v // 5) * 5
                if stride_v <= 0:
                    stride_v = 5
                v_cur = 0
                while v_cur < total_v:
                    v_start = (v_cur // 5) * 5
                    v_end = min(total_v, v_start + win_v)
                    if v_end < total_v:
                        rem_end = (v_end - 2) % 5
                        if rem_end != 0:
                            v_end = min(total_v, v_end + (5 - rem_end))
                    sec_start = v_start / v_fps
                    sec_end = v_end / v_fps
                    a_start = int(round(sec_start * a_fps))
                    a_end = min(total_a, int(round(sec_end * a_fps)))
                    chunks.append((v_start, v_end, a_start, a_end))
                    if v_end >= total_v:
                        break
                    v_cur += stride_v

        elif split_mode in ("固定分段数", "fixed_segments"):
            actual_n = max(1, int(num_segments))
            win_v = max(5, int(math.ceil(total_v / actual_n)))
            rem = (win_v - 2) % 5
            if rem != 0:
                win_v += 5 - rem
            ov_v = max(2, int(round(overlap_seconds * v_fps)))
            for i in range(actual_n):
                v_start = max(0, i * (win_v - ov_v))
                v_start = (v_start // 5) * 5
                v_end = min(total_v, v_start + win_v)
                if i == actual_n - 1:
                    v_end = total_v
                sec_s = v_start / v_fps
                sec_e = v_end / v_fps
                a_s = int(round(sec_s * a_fps))
                a_e = min(total_a, int(round(sec_e * a_fps)))
                chunks.append((v_start, v_end, a_s, a_e))

        elif split_mode in ("自定义切点", "custom", "custom_cuts"):
            pairs = [p.strip() for p in custom_cuts.split(",") if "-" in p]
            for p in pairs:
                try:
                    s_str, e_str = p.split("-")
                    s_sec, e_sec = float(s_str.strip()), float(e_str.strip())
                    v_start = max(0, int(round(s_sec * v_fps)))
                    v_start = (v_start // 5) * 5
                    v_end = min(total_v, int(round(e_sec * v_fps)))
                    a_start = max(0, int(round(s_sec * a_fps)))
                    a_end = min(total_a, int(round(e_sec * a_fps)))
                    chunks.append((v_start, v_end, a_start, a_end))
                except Exception as e:
                    print(f"[PJ分段规划器] 解析自定义切点失败: '{p}', 错误: {e}")

        if not chunks:
            chunks = [(0, total_v, 0, total_a)]

        n_chunks = len(chunks)

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
        summary_lines.append("--------------------------------------------------")

        for i, (vs, ve, as_, ae) in enumerate(chunks):
            s_sec = vs / v_fps
            e_sec = ve / v_fps
            s_frame = int(round(s_sec * fps))
            e_frame = min(total_frames, int(round(e_sec * fps)))
            dur_sec = round(e_sec - s_sec, 2)
            summary_lines.append(
                f"  段 {i + 1:02d}: [{s_sec:5.2f}s ~ {e_sec:5.2f}s] ({dur_sec}s, 帧 {s_frame}~{e_frame}) (步 {vs}:{ve})"
            )
            chunks_sec_info.append({
                "segment": i + 1,
                "start_sec": round(s_sec, 2),
                "end_sec": round(e_sec, 2),
                "duration_sec": dur_sec,
                "start_frame": s_frame,
                "end_frame": e_frame,
                "vs": vs,
                "ve": ve,
                "as": as_,
                "ae": ae,
            })
        summary_text = "\n".join(summary_lines)

        print("\n" + "=" * 65)
        print(summary_text)
        print("=" * 65 + "\n")

        # 4. 合成前端交互播放视频 (如果传入了 images)
        video_filename = _generate_preview_mp4(images, audio, fps) if images is not None else None

        plan_dict = {
            "has_media": has_media,
            "chunks": chunks,
            "chunks_sec_info": chunks_sec_info,
            "n_chunks": n_chunks,
            "effective_seconds": round(effective_seconds, 2),
            "total_frames": total_frames,
            "total_v": total_v,
            "total_a": total_a,
            "split_mode": split_mode,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "num_segments": num_segments,
            "custom_cuts": custom_cuts,
            "align_51_frames": align_51_frames,
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
                "chunks": chunks_sec_info,
                "summary": summary_text,
                "mode_desc": mode_desc,
            }]
        }

        return {
            "ui": ui_payload,
            "result": (plan_dict, n_chunks, summary_text, images, audio),
        }
