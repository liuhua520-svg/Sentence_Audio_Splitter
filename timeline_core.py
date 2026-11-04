# -*- coding: utf-8 -*-
"""
timeline_core.py
=================
「逐句音频切分工具」的核心逻辑层：不依赖 tkinter，可独立单元测试。

负责：
  1. 打开/解析 .ttslt / .dtslt（SVS-Lab-Tools 序列时间表调试文件，JSON）
  2. 导入 .srt / .lrc / .lab，转换成同样的内部时间表结构
  3. 内部统一数据结构：TimelineDoc / TimelineRow
  4. 保存回 .ttslt / .dtslt（严格保留原有字段，只更新被编辑过的部分）
  5. 按 index 顺序，把每句从合并音频里切出来，导出 wav + 对应 txt
  6. "剪掉开头"：按 BPM / 拍号计算出 N 拍对应的秒数，从内存中的原始
     音频数组头部裁掉——不触碰磁盘上的原始文件，也不改时间表。

所有秒数字段统一使用「浮点秒」（与 .ttslt/.dtslt 保持一致）。
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TimelineRow:
    """
    一行 = 一句台词（对应 .ttslt 的 sentences[i] 或 .dtslt 的 boxes[i]）。
    "silence" 条目（.ttslt 里两句之间的 {"silence": 0.35}）不作为
    TimelineRow，单独保留在 TimelineDoc.raw_sentences 里按原样透传，
    保证保存时不丢失、不错位。
    """
    index: int
    text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    # 原始字段透传（phonemes / wav_path / title 等），保存时原样写回，
    # 仅 text/start_sec/end_sec/duration_sec 被表格编辑覆盖。
    extra: Dict[str, Any] = field(default_factory=dict)

    def title_field(self) -> str:
        """.dtslt 用 title 字段展示对话框标题；.ttslt 没有则回退用文本。"""
        t = self.extra.get("title")
        return t if t else self.text


@dataclass
class TimelineDoc:
    """
    整份 .ttslt / .dtslt 的内存表示。
    kind: "ttslt" 或 "dtslt"（也可能是 "imported"，表示从 srt/lrc/lab
    临时构建、尚未对应任何磁盘上的 .ttslt/.dtslt 文件）。
    """
    kind: str                      # "ttslt" | "dtslt" | "imported"
    source_path: Optional[Path]    # 打开的原始文件路径；imported 时为 None
    rows: List[TimelineRow]
    raw: Dict[str, Any]            # 原始 JSON 顶层字典（用于保存时原样保留未识别字段）
    row_key: str                   # "sentences"（ttslt）或 "boxes"（dtslt）

    # ── 便捷属性 ────────────────────────────────────────────────────
    @property
    def total_duration_sec(self) -> float:
        if not self.rows:
            return 0.0
        return max(r.end_sec for r in self.rows)


# ══════════════════════════════════════════════════════════════════════
# 打开 .ttslt / .dtslt
# ══════════════════════════════════════════════════════════════════════

def open_timeline_file(path: str | Path) -> TimelineDoc:
    """
    根据扩展名分派到 .ttslt / .dtslt 解析器。其余扩展名交给
    import_subtitle_like() 处理（srt/lrc/lab）。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")  # 兼容可能带 BOM 的文件
    data = json.loads(text)

    if suffix == ".ttslt" or data.get("kind") == "tts_alignment_timeline":
        return _parse_ttslt(path, data)
    elif suffix == ".dtslt" or data.get("kind") == "dialogue_batch_timeline":
        return _parse_dtslt(path, data)
    else:
        raise ValueError(f"无法识别的时间表文件类型: {path.name}（既不是 .ttslt 也不是 .dtslt）")


def _parse_ttslt(path: Path, data: Dict[str, Any]) -> TimelineDoc:
    rows: List[TimelineRow] = []
    for item in data.get("sentences", []):
        if "silence" in item and "index" not in item:
            # 句间静音条目，不是一句台词，跳过——保存时会按位置重新插回
            continue
        rows.append(TimelineRow(
            index=int(item.get("index", len(rows))),
            text=str(item.get("text", "")),
            start_sec=float(item.get("start_sec", 0.0)),
            end_sec=float(item.get("end_sec", 0.0)),
            duration_sec=float(item.get("duration_sec", 0.0)),
            extra={k: v for k, v in item.items()
                   if k not in ("index", "text", "start_sec", "end_sec", "duration_sec")},
        ))
    return TimelineDoc(kind="ttslt", source_path=path, rows=rows, raw=data, row_key="sentences")


def _parse_dtslt(path: Path, data: Dict[str, Any]) -> TimelineDoc:
    rows: List[TimelineRow] = []
    for item in data.get("boxes", []):
        if "silence" in item and "index" not in item:
            # 句间静音条目（与 .ttslt 格式对齐），不是一个对话框，跳过——
            # 保存时会按相邻行的时间间隙自动重新插回，不作为表格行展示/编辑。
            continue
        rows.append(TimelineRow(
            index=int(item.get("index", len(rows))),
            text=str(item.get("text", "")),
            start_sec=float(item.get("start_sec", 0.0)),
            end_sec=float(item.get("end_sec", 0.0)),
            duration_sec=float(item.get("duration_sec", 0.0)),
            extra={k: v for k, v in item.items()
                   if k not in ("index", "text", "start_sec", "end_sec", "duration_sec")},
        ))
    return TimelineDoc(kind="dtslt", source_path=path, rows=rows, raw=data, row_key="boxes")


# ══════════════════════════════════════════════════════════════════════
# 保存 .ttslt / .dtslt
# ══════════════════════════════════════════════════════════════════════

def save_timeline_file(doc: TimelineDoc, path: str | Path) -> None:
    """
    把 TimelineDoc 写回 .ttslt / .dtslt（JSON，UTF-8，缩进 2）。

    - ttslt：按 rows 顺序重建 "sentences" 数组；相邻两行之间若存在真实
      间隙（后一行 start_sec - 前一行 end_sec > 阈值），自动补回
      {"silence": ...} 条目，与 SVS-Lab-Tools 后端写出的格式保持一致；
      若两行编辑后变成无缝衔接，则不再插入静音条目。
    - dtslt：按 rows 顺序重建 "boxes" 数组，不涉及 silence。
    - 每行原有的 extra 字段（phonemes / wav_path / title 等）原样保留，
      只有 text/start_sec/end_sec/duration_sec 使用表格里的最新值。
    - 顶层其它字段（generated_at / total_duration_sec / sentence_count /
      box_count 等）会按当前 rows 重新计算，保证与内容一致；未识别的
      额外顶层字段原样保留。
    """
    path = Path(path)
    out = dict(doc.raw)  # 保留未识别的顶层字段

    if doc.kind == "ttslt":
        sentences: List[Dict[str, Any]] = []
        for i, row in enumerate(doc.rows):
            if i > 0:
                prev_end = doc.rows[i - 1].end_sec
                gap = row.start_sec - prev_end
                if gap > 1e-6:
                    sentences.append({"silence": round(gap, 6)})
            item = dict(row.extra)
            item["index"] = row.index
            item["text"] = row.text
            item["start_sec"] = round(row.start_sec, 6)
            item["end_sec"] = round(row.end_sec, 6)
            item["duration_sec"] = round(row.duration_sec, 6)
            # 保证字段顺序美观：index/text/start/end/duration 在前，其余（如 phonemes）在后
            ordered = {k: item[k] for k in ("index", "text", "start_sec", "end_sec", "duration_sec") if k in item}
            for k, v in item.items():
                if k not in ordered:
                    ordered[k] = v
            sentences.append(ordered)
        out["sentences"] = sentences
        out["sentence_count"] = len(doc.rows)
        out["total_duration_sec"] = round(doc.total_duration_sec, 6)

    elif doc.kind == "dtslt":
        boxes: List[Dict[str, Any]] = []
        for i, row in enumerate(doc.rows):
            if i > 0:
                prev_end = doc.rows[i - 1].end_sec
                gap = row.start_sec - prev_end
                if gap > 1e-6:
                    boxes.append({"silence": round(gap, 6)})
            item = dict(row.extra)
            item["index"] = row.index
            item["text"] = row.text
            item["start_sec"] = round(row.start_sec, 6)
            item["duration_sec"] = round(row.duration_sec, 6)
            item["end_sec"] = round(row.end_sec, 6)
            ordered_keys = ["index", "title", "text", "wav_path", "start_sec", "duration_sec", "end_sec", "phonemes"]
            ordered = {k: item[k] for k in ordered_keys if k in item}
            for k, v in item.items():
                if k not in ordered:
                    ordered[k] = v
            boxes.append(ordered)
        out["boxes"] = boxes
        out["box_count"] = len(doc.rows)
        out["total_duration_sec"] = round(doc.total_duration_sec, 6)

    else:
        raise ValueError(f"未知的时间表类型，无法保存: {doc.kind}")

    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# 导入 .srt / .lrc / .lab → TimelineDoc（kind="imported"，可另存为 .ttslt）
# ══════════════════════════════════════════════════════════════════════

def import_subtitle_like(path: str | Path) -> TimelineDoc:
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig", errors="replace")

    if suffix == ".srt":
        rows = _parse_srt(text)
    elif suffix == ".lrc":
        rows = _parse_lrc(text)
    elif suffix == ".lab":
        rows = _parse_lab(text)
    else:
        raise ValueError(f"不支持的导入格式: {path.suffix}（仅支持 .srt / .lrc / .lab）")

    raw = {
        "kind": "tts_alignment_timeline",
        "stem": path.stem,
        "generated_at": "",
        "total_duration_sec": round(rows[-1].end_sec, 6) if rows else 0.0,
        "sentence_count": len(rows),
        "sentences": [],  # save 时会重建
    }
    return TimelineDoc(kind="imported", source_path=None, rows=rows, raw=raw, row_key="sentences")


_SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _srt_ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    ms_val = int(ms.ljust(3, "0")[:3])
    return int(h) * 3600 + int(m) * 60 + int(s) + ms_val / 1000.0


def _parse_srt(text: str) -> List[TimelineRow]:
    """
    解析标准 SRT 字幕。块结构：
        1
        00:00:00,000 --> 00:00:02,136
        你好，大家好

        2
        00:00:02,486 --> 00:00:04,142
        我的世界
    序号行允许缺失/不连续（有些工具导出不带序号），只依赖时间行 + 后续文本行。
    """
    rows: List[TimelineRow] = []
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    idx = 0
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        time_line_i = None
        for i, ln in enumerate(lines):
            if _SRT_TIME_RE.search(ln):
                time_line_i = i
                break
        if time_line_i is None:
            continue
        m = _SRT_TIME_RE.search(lines[time_line_i])
        start = _srt_ts_to_sec(*m.groups()[0:4])
        end = _srt_ts_to_sec(*m.groups()[4:8])
        content_lines = lines[time_line_i + 1:]
        content = " ".join(content_lines).strip()
        rows.append(TimelineRow(
            index=idx, text=content, start_sec=start, end_sec=end,
            duration_sec=max(0.0, end - start), extra={},
        ))
        idx += 1
    return rows


_LRC_TIME_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")


def _parse_lrc(text: str) -> List[TimelineRow]:
    """
    解析 LRC 歌词。LRC 每行只有起始时间，没有显式结束时间——结束时间
    取「下一条有文本的时间戳」，最后一行没有下一条可参照，直接用
    start + 默认 2 秒兜底（导出音频前用户可在表格里手动核对/调整）。
    元数据行（[ar:] [ti:] [al:] [by:] [offset:] 等）会被忽略。
    """
    raw_entries: List[Tuple[float, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tags = list(_LRC_TIME_RE.finditer(line))
        if not tags:
            continue
        content = _LRC_TIME_RE.sub("", line).strip()
        if not content:
            continue
        for m in tags:
            mm, ss, frac = m.group(1), m.group(2), m.group(3) or "0"
            frac_val = int(frac.ljust(3, "0")[:3]) if len(frac) <= 3 else int(frac[:3])
            sec = int(mm) * 60 + int(ss) + frac_val / 1000.0
            raw_entries.append((sec, content))

    raw_entries.sort(key=lambda t: t[0])
    rows: List[TimelineRow] = []
    for i, (start, content) in enumerate(raw_entries):
        if i + 1 < len(raw_entries):
            end = raw_entries[i + 1][0]
        else:
            end = start + 2.0
        rows.append(TimelineRow(
            index=i, text=content, start_sec=start, end_sec=end,
            duration_sec=max(0.0, end - start), extra={},
        ))
    return rows


def _parse_lab(text: str) -> List[TimelineRow]:
    """
    解析 HTK/MFA 风格的 LAB 音标标注文件：
        <start_100ns> <end_100ns> <label>
    单位固定为 100 纳秒（与 SVS-Lab-Tools 后端 LAB 约定一致）。
    LAB 本身是音素级标注，没有"句子"概念——这里把每一行当作一条
    TimelineRow（text = 该行的音标标签），方便在同一个表格界面里查看
    / 切分逐音素的时间区间；如果整份 LAB 只是想当成一句处理，用户
    可以在表格里手动合并/删除多余行后再导出。
    连续的 sil / SIL / pau 静音标签同样会被解析成行，不做特殊过滤，
    避免静音被静默丢弃导致后续对不上时间轴。
    """
    rows: List[TimelineRow] = []
    idx = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start_100ns = int(parts[0])
            end_100ns = int(parts[1])
        except ValueError:
            continue
        label = " ".join(parts[2:])
        start_sec = start_100ns / 10_000_000.0
        end_sec = end_100ns / 10_000_000.0
        rows.append(TimelineRow(
            index=idx, text=label, start_sec=start_sec, end_sec=end_sec,
            duration_sec=max(0.0, end_sec - start_sec), extra={},
        ))
        idx += 1
    return rows


# ══════════════════════════════════════════════════════════════════════
# 按 index 顺序导出每句音频 + 对应 txt
# ══════════════════════════════════════════════════════════════════════

_SAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def sanitize_filename_fragment(text: str, max_chars: int = 20) -> str:
    """
    从文本里截取前 max_chars 个"字符"（按 Unicode 码位计，中日韩字符
    每个算 1），并去掉 Windows/常见文件系统不允许的字符，用于拼接
    导出文件名。全部被过滤掉后返回 "untitled"，避免生成空文件名。
    """
    text = unicodedata.normalize("NFC", text or "")
    text = _SAFE_FILENAME_RE.sub("", text)
    text = text.strip().strip(".")
    text = text[:max_chars]
    return text if text else "untitled"


def _naming_source_text(row: TimelineRow, name_source: str) -> str:
    """
    根据 name_source（"text" / "title" / "stem"）取出用于导出文件名的
    原始字符串（尚未经过 sanitize_filename_fragment 清洗/截断）：
      - "text"  ：row.text（原有行为）
      - "title" ：row.extra["title"]（.dtslt 常见字段，对话框标题），
                  没有则回退到 row.text
      - "stem"  ：row.extra["wav_path"] 的文件名主干（不含扩展名，
                  兼容正反斜杠路径），没有则回退到 row.text
    最终都可能是空字符串，交给 sanitize_filename_fragment 兜底成
    "untitled"。
    """
    if name_source == "title":
        title = row.extra.get("title")
        return str(title) if title else row.text
    if name_source == "stem":
        wav_path = row.extra.get("wav_path")
        if wav_path:
            # 手动取 stem：wav_path 可能是 Windows 路径（反斜杠），在非
            # Windows 上 pathlib 不会按反斜杠切分，所以先统一替换。
            cleaned = str(wav_path).replace("\\", "/")
            stem = Path(cleaned).stem
            if stem:
                return stem
        return row.text
    return row.text  # "text" 或未识别的值一律回退到原有行为


def export_rows_as_audio_and_txt(
    doc: TimelineDoc,
    output_dir: str | Path,
    filename_text_chars: int = 20,
    source_wav_path: Optional[str | Path] = None,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    name_source: str = "text",
) -> List[Dict[str, str]]:
    """
    按 doc.rows 的 index 顺序，从源音频里依次切出每一行对应的音频区间
    [start_sec, end_sec)，分别导出为：
        {index}_{命名片段}.wav
        {index}_{命名片段}.txt   （内容始终为该行原始文本 row.text，UTF-8，
                                    与命名来源无关）
    例如 index=0 文本"你好，大家好" → "0_你好，大家好.wav" / ".txt"
    index=10 → "10_....wav"（index 直接使用原值，天然支持 1 2 3 10 100
    这种非零填充的顺序展示，不做补零）。

    name_source 控制"命名片段"取自哪个字段（详见 _naming_source_text）：
      - "text"  ：行文本（原有默认行为）
      - "title" ：.dtslt 的 title 字段（对话框标题），常在 text 为空时
                  仍然有意义的可读名字
      - "stem"  ：原始 wav_path 的文件名主干，便于导出文件名与源素材
                  文件一一对应
    三种情况都经过 sanitize_filename_fragment 同样的清洗/截断/兜底逻辑。
    txt 文件内容不受 name_source 影响，始终写入该行的 row.text。

    行按 index 数值升序处理（与原始 JSON 顺序无关，保证即使表格里
    手动调整过顺序，导出文件名里的 index 前缀依然和内容对应）。

    源音频二选一：
      - 传 audio + sr：直接使用内存中已经加载好（可能已被"剪掉开头"
        裁剪过）的音频数组，不重新从磁盘读取——这是 GUI 里"加载一次，
        裁剪+导出共用同一份音频"这一设计的支撑函数。
      - 传 source_wav_path：从磁盘读取该 wav 文件（未做任何裁剪）。
        保留这个参数是为了兼容"只想切分、不需要裁剪"的脚本化调用场景。
    两者都不传 / 都传时抛 ValueError，避免调用方语义不清。

    Returns
    -------
    List[Dict]：每项 {"index":.., "wav": 导出路径, "txt": 导出路径}，
    供 GUI 展示导出结果列表。
    """
    if (audio is None) == (source_wav_path is None):
        raise ValueError("必须且只能提供 audio+sr 或 source_wav_path 二者之一")
    if name_source not in ("text", "title", "stem"):
        raise ValueError(f'name_source 必须是 "text" / "title" / "stem" 之一，收到: {name_source!r}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if audio is None:
        audio, sr = sf.read(str(source_wav_path), always_2d=False)

    total_samples = audio.shape[0]

    results: List[Dict[str, str]] = []
    for row in sorted(doc.rows, key=lambda r: r.index):
        start_sample = max(0, int(round(row.start_sec * sr)))
        end_sample = min(total_samples, int(round(row.end_sec * sr)))
        if end_sample <= start_sample:
            # 时间区间无效（例如手动编辑表格时 end <= start，或裁剪音频
            # 后时间表未同步调整导致区间落在了已被裁掉的部分），跳过并
            # 让调用方在结果列表里看到该行缺失，而不是写出一个空/负长度的 wav
            continue
        clip = audio[start_sample:end_sample]

        raw_name_text = _naming_source_text(row, name_source)
        name_frag = sanitize_filename_fragment(raw_name_text, filename_text_chars)
        base_name = f"{row.index}_{name_frag}"
        wav_path = output_dir / f"{base_name}.wav"
        txt_path = output_dir / f"{base_name}.txt"

        sf.write(str(wav_path), clip, sr)
        txt_path.write_text(row.text, encoding="utf-8")

        results.append({"index": str(row.index), "wav": str(wav_path), "txt": str(txt_path)})

    return results


# ══════════════════════════════════════════════════════════════════════
# 导出 .list 文件（vocal_path|speaker_name|language|text，每行一句）
# ══════════════════════════════════════════════════════════════════════

_LIST_ALLOWED_LANGUAGES = ("ZH", "EN", "JA", "KO", "YUE")


def build_list_file_lines(
    export_results: List[Dict[str, str]],
    doc: TimelineDoc,
    speaker_name: str = "speaker_name",
    language: str = "ZH",
) -> List[str]:
    """
    根据 export_rows_as_audio_and_txt() 的返回结果，生成 .list 文件的
    每一行内容，格式：

        {wav 的绝对路径}|{speaker_name}|{language}|{text}

    - vocal_path：导出的 wav 文件的**绝对路径**（export_results 里的
      "wav" 已经是 export_rows_as_audio_and_txt 写出时用的路径，这里
      统一转成绝对路径，兼容调用方传入相对导出目录的情况）。
    - speaker_name：为空/全空白时回退成 "speaker_name"。
    - language：必须是 ZH/EN/JA/KO/YUE 之一（大小写不敏感，内部统一转
      大写），非法值抛 ValueError，避免静默写出错误标签。
    - text：对应行的原始文本 row.text（与 export_results 按 index 一一
      对应；export_results 只包含实际导出成功的行，跳过的行不会出现
      在 .list 里，行数与成功导出的 wav 数一致）。

    Parameters
    ----------
    export_results : export_rows_as_audio_and_txt() 的返回值
    doc : 对应的 TimelineDoc，用于按 index 取回该行原始文本
    speaker_name : 说话人标签，默认 "speaker_name"
    language : ZH / EN / JA / KO / YUE 之一

    Returns
    -------
    List[str]：每项是一行 .list 内容（不含行尾换行符）
    """
    language_norm = (language or "").strip().upper()
    if language_norm not in _LIST_ALLOWED_LANGUAGES:
        raise ValueError(
            f"language 必须是 {'/'.join(_LIST_ALLOWED_LANGUAGES)} 之一，收到: {language!r}"
        )
    speaker = speaker_name.strip() if speaker_name and speaker_name.strip() else "speaker_name"

    text_by_index: Dict[int, str] = {row.index: row.text for row in doc.rows}

    lines: List[str] = []
    for item in export_results:
        wav_abs_path = str(Path(item["wav"]).resolve())
        row_index = int(item["index"])
        text = text_by_index.get(row_index, "")
        # .list 是竖线分隔的纯文本格式，文本里如果本身带竖线/换行会破坏
        # 列对齐，这里做最小清洗：换行替换成空格，竖线转义成全角，避免
        # 生成的 .list 被下游按 | 分割时错位。
        safe_text = text.replace("\r\n", " ").replace("\n", " ").replace("|", "｜")
        lines.append(f"{wav_abs_path}|{speaker}|{language_norm}|{safe_text}")

    return lines


def write_list_file(
    export_results: List[Dict[str, str]],
    doc: TimelineDoc,
    list_path: str | Path,
    speaker_name: str = "speaker_name",
    language: str = "ZH",
) -> Path:
    """
    调用 build_list_file_lines() 生成内容，写到 list_path（UTF-8，
    每行一句，行尾统一 \\n）。返回实际写入的 Path，方便调用方展示。
    """
    lines = build_list_file_lines(export_results, doc, speaker_name=speaker_name, language=language)
    list_path = Path(list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return list_path


# ══════════════════════════════════════════════════════════════════════
# "剪掉开头"：BPM / 拍号 → 拍长秒数 → 内存裁剪原始音频头部
# ══════════════════════════════════════════════════════════════════════

def beat_duration_sec(bpm: float, beats_per_bar: int = 4, beat_unit: int = 4) -> float:
    """
    计算"一拍"对应的秒数。

    以四分音符为一拍的常见情况（4/4、3/4 等 beat_unit=4）：
        一拍秒数 = 60 / BPM
    例：BPM120，4/4 拍，一拍 = 60/120 = 0.5 秒——注意这与需求描述里
    "BPM120 4/4 一拍为两秒"的说法不同：标准音乐理论下 BPM（每分钟
    拍数）本身就定义了"一拍=60/BPM 秒"，BPM120 意味着一拍 0.5 秒，
    不是 2 秒。beats_per_bar（拍号分子，每小节几拍）不影响单拍时长，
    只影响"一小节等于几拍"；beat_unit（拍号分母，以几分音符为一拍）
    在 GUI 里预留参数位，当前仅支持"以四分音符为一拍"这一常见约定
    （beat_unit=4 时按上式计算；其它分母按比例换算：
        一拍秒数 = 60 / BPM * (4 / beat_unit)
    ，例如 8 分音符为一拍时，一拍时长是四分音符为一拍时的一半）。

    Parameters
    ----------
    bpm : 每分钟拍数（四分音符为基准的速度标记）
    beats_per_bar : 拍号分子，每小节拍数（本函数不使用，仅供 GUI 显示/
                    未来"剪掉整小节"功能使用）
    beat_unit : 拍号分母，以几分音符为一拍，默认 4（四分音符）

    Returns
    -------
    float：一拍对应的秒数
    """
    if bpm <= 0:
        raise ValueError("BPM 必须大于 0")
    if beat_unit <= 0:
        raise ValueError("拍号分母必须大于 0")
    return (60.0 / bpm) * (4.0 / beat_unit)


def trim_audio_head(audio: np.ndarray, sr: int, seconds_to_trim: float) -> np.ndarray:
    """
    从内存中的音频数组头部裁掉 seconds_to_trim 秒，返回新数组（不修改
    原数组本身、不触碰磁盘文件）。裁剪秒数超过总时长时，裁到只剩 0 帧
    （不报错，交由调用方在 GUI 上提示"整段已被裁空"）。
    """
    if seconds_to_trim <= 0:
        return audio.copy()
    n = int(round(seconds_to_trim * sr))
    n = max(0, min(n, audio.shape[0]))
    return audio[n:].copy()
