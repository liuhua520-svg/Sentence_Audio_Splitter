# -*- coding: utf-8 -*-
"""
splitter_gui.py
================
逐句音频切分工具 —— 独立于 SVS-Lab-Tools 的单文件 Tkinter GUI 程序。

功能：
  - 打开 .ttslt / .dtslt（SVS-Lab-Tools 输出的序列时间表调试文件）
  - 导入 .srt / .lrc / .lab，转换成同样的可编辑时间表
  - 表格里编辑文本 / 起始秒 / 结束秒，改完自动重算该行 duration_sec
  - 保存回 .ttslt / .dtslt（或"另存为"）
  - 音频只加载一次：手动选择对应的音频 wav 后，"剪掉开头"与"逐句音频
    导出"共用同一份内存中的音频——裁剪开头后直接导出，用的就是裁剪后
    的音频，不需要重新选择文件
  - 逐句音频导出：按 index 顺序批量导出每句音频（wav）+ 对应文本（txt），
    文件名 "{index}_{文本前N字}.{wav|txt}"
  - "剪掉开头"：按 BPM / 拍号算出 N 拍的秒数，从内存里加载的音频头部
    裁掉一段（仅内存操作，不覆盖磁盘原始文件），可另存为新 wav；
    注意：时间表的 start_sec/end_sec 不会自动跟着偏移，需要对齐时请
    在表格里手动减去已裁掉的秒数
  - 预览播放：选中表格中一行后，点按钮或按空格键，只播放该行
    [start_sec, end_sec) 区间（播完自动停）；再次触发会停止当前播放
    （开关式）。播放用的是"音频"区当前加载的内存音频。
  - 预览进度条：播放过程中实时显示当前位置；拖动进度条可即时跳转
    并从新位置继续播放；「从头重播」按钮从该句起点重新播放。

运行方式：
    python splitter_gui.py
依赖 Python 标准库（tkinter）+ numpy + soundfile + sounddevice：
    pip install numpy soundfile sounddevice
（sounddevice 缺失或所在环境没有可用的音频输出设备时，预览播放按钮/
 空格键会弹出明确的错误提示，但不影响其它功能正常使用。）

────────────────────────────────────────────────────────────────────
关于"剪掉开头"里 BPM 与拍长的换算说明（请务必先看一眼再使用）：

标准音乐理论里，BPM（每分钟拍数）本身就定义了一拍的时长：
    一拍秒数 = 60 ÷ BPM
所以 BPM120 时，一拍 = 60/120 = 0.5 秒，而不是 2 秒。
本程序按这个标准公式计算，界面上会实时显示"当前一拍 = X 秒"，
以及"裁剪 N 拍 = 总共裁掉 X 秒"，方便你在下刀前用眼睛核对一遍，
数字不对可以随时调整 BPM / 拍号分母 / 拍数再裁。
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
    _SOUNDDEVICE_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - 环境缺少 PortAudio 时的兜底
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = str(_e)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import timeline_core as tc


APP_TITLE = "逐句音频切分工具 · Sentence Audio Splitter (独立于 SVS-Lab-Tools)"


class SplitterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x720")
        self.minsize(900, 560)

        # ── 状态 ────────────────────────────────────────────────────
        self.doc: Optional[tc.TimelineDoc] = None
        self.current_doc_path: Optional[Path] = None   # 当前已打开/另存为的 .ttslt/.dtslt 路径

        # 音频只加载一次，裁剪开头与逐句导出共用同一份内存数组——不再
        # 分开选择"合并音频"和"原始音频"。self._audio_source_path 记录
        # 磁盘来源，供"重新加载(撤销裁剪)"使用；loaded_audio 始终代表
        # "当前可用于导出的音频"（可能已被裁剪过，也可能原封不动）。
        self.loaded_audio: Optional[np.ndarray] = None
        self.loaded_audio_sr: Optional[int] = None
        self._audio_source_path: Optional[str] = None
        self.trimmed_seconds_total: float = 0.0          # 累计已裁掉的秒数（仅展示用，纯提示信息）

        # ── 预览播放状态 ────────────────────────────────────────────
        # 选中表格中某一行后，点"预览播放"按钮或按空格键，只播放该行
        # [start_sec, end_sec) 这一段（用当前"音频"区加载的内存音频）。
        # 再次点击/按空格 → 停止（开关式）。进度条实时显示位置，拖动
        # 可即时跳转并继续播放；「从头重播」从该句起点重新播放。
        # 实现：把整句切片保存在 _preview_clip，用 _preview_cursor 记录
        # 当前采样点，sd.play(clip[cursor:]) 从任意位置起播；用
        # time.monotonic() 估算推进量来更新进度条。
        self._is_previewing: bool = False
        self._preview_poll_job: Optional[str] = None
        self._preview_clip: Optional[np.ndarray] = None   # 当前预览句的完整切片
        self._preview_sr: Optional[int] = None
        self._preview_cursor: int = 0                      # 相对 clip 起点的采样位置
        self._preview_play_start_cursor: int = 0           # 本次 sd.play 开始时的 cursor
        self._preview_play_started_at: Optional[float] = None  # monotonic 时间戳
        self._preview_row_pos: Optional[int] = None        # 当前预览对应的表格行
        self._is_dragging_scale: bool = False              # 用户正在拖动进度条
        self._scale_programmatic: bool = False             # 程序更新 Scale 时避免触发 seek
        self._last_seek_at: float = 0.0                    # 拖动时节流 seek 的时间戳
        self._was_playing_before_drag: bool = False        # 按下进度条时是否正在播放

        self._build_menu()
        self._build_layout()
        self._refresh_table()
        self._update_status("就绪。请打开 .ttslt / .dtslt，或导入 .srt / .lrc / .lab。")

    # ══════════════════════════════════════════════════════════════
    # 菜单 & 顶层布局
    # ══════════════════════════════════════════════════════════════

    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开 .ttslt / .dtslt...", command=self.action_open_timeline)
        file_menu.add_separator()
        file_menu.add_command(label="导入 .srt 字幕...", command=lambda: self.action_import(".srt"))
        file_menu.add_command(label="导入 .lrc 歌词...", command=lambda: self.action_import(".lrc"))
        file_menu.add_command(label="导入 .lab 标注...", command=lambda: self.action_import(".lab"))
        file_menu.add_separator()
        file_menu.add_command(label="保存", command=self.action_save, accelerator="Ctrl+S")
        file_menu.add_command(label="另存为...", command=self.action_save_as)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_exit)
        menubar.add_cascade(label="文件", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)

        help_menu.add_command(
            label="关于 / 使用说明",
            command=self.action_show_help
        )

        help_menu.add_command(
            label="许可证",
            command=self.action_show_license
        )

        # 第三方许可证子菜单
        third_party_license_menu = tk.Menu(
            help_menu,
            tearoff=0
        )

        third_party_license_menu.add_command(
            label="NumPy (BSD-3 License)",
            command=self.action_show_numpy_license
        )

        third_party_license_menu.add_command(
            label="SoundFile (BSD-3 License)",
            command=self.action_show_soundfile_license
        )

        third_party_license_menu.add_command(
            label="SoundDevice (MIT License)",
            command=self.action_show_sounddevice_license
        )


        help_menu.add_cascade(
            label="第三方许可证",
            menu=third_party_license_menu
        )

        menubar.add_cascade(
            label="帮助",
            menu=help_menu
        )

        self.config(menu=menubar)
        self.bind_all("<Control-s>", lambda e: self.action_save())
        self.bind_all("<space>", self._on_space_key)
        self.protocol("WM_DELETE_WINDOW", self._on_exit)

    def _build_layout(self):
        # 顶部：当前文档信息 + 打开/导入按钮
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")

        self.doc_info_var = tk.StringVar(value="未打开任何时间表文件")
        ttk.Label(top, textvariable=self.doc_info_var, font=("", 10, "bold")).pack(side="left")

        btns = ttk.Frame(top)
        btns.pack(side="right")
        ttk.Button(btns, text="打开 .ttslt/.dtslt", command=self.action_open_timeline).pack(side="left", padx=3)
        ttk.Button(btns, text="导入字幕...", command=self.action_import_menu_button).pack(side="left", padx=3)
        ttk.Button(btns, text="保存", command=self.action_save).pack(side="left", padx=3)
        ttk.Button(btns, text="另存为", command=self.action_save_as).pack(side="left", padx=3)

        # 中部：可编辑表格
        table_frame = ttk.Frame(self, padding=(10, 0))
        table_frame.pack(fill="both", expand=True)

        columns = ("index", "title", "text", "start_sec", "end_sec", "duration_sec")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        headings = {
            "index": "序号 index", "title": "标题 title", "text": "文本", "start_sec": "起始(秒)",
            "end_sec": "结束(秒)", "duration_sec": "时长(秒)",
        }
        widths = {"index": 70, "title": 200, "text": 320, "start_sec": 110, "end_sec": 110, "duration_sec": 110}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col],
                              anchor="w" if col in ("title", "text") else "center")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        self.tree.bind("<Double-1>", self._on_cell_double_click)

        # 表格右侧：行操作按钮
        row_ops = ttk.Frame(table_frame, padding=(8, 0))
        row_ops.pack(side="left", fill="y")
        ttk.Label(row_ops, text="行操作", font=("", 9, "bold")).pack(pady=(0, 6))
        ttk.Button(row_ops, text="编辑选中行", command=self._edit_selected_row).pack(fill="x", pady=2)
        ttk.Button(row_ops, text="删除选中行", command=self._delete_selected_row).pack(fill="x", pady=2)
        ttk.Button(row_ops, text="上移", command=lambda: self._move_row(-1)).pack(fill="x", pady=2)
        ttk.Button(row_ops, text="下移", command=lambda: self._move_row(1)).pack(fill="x", pady=2)
        ttk.Separator(row_ops, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(row_ops, text="双击单元格\n可直接编辑\n\n选中一行后\n按空格键\n可预览播放",
                  justify="center", foreground="#666").pack()

        # 预览播放条（进度条 + 播放/停止 + 从头重播）
        preview_bar = ttk.LabelFrame(self, text="预览播放（选中行的 [start, end) 区间）", padding=(10, 6))
        preview_bar.pack(fill="x", padx=10, pady=(6, 0))
        self._build_preview_bar(preview_bar)

        # 下部：音频（加载 + 剪掉开头）与逐句导出，左右分栏；两者共用
        # 同一份已加载到内存的音频（self.loaded_audio），不需要分别选择
        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")

        audio_box = ttk.LabelFrame(bottom, text="音频（加载一次，裁剪与导出共用）", padding=10)
        audio_box.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self._build_audio_section(audio_box)

        export_box = ttk.LabelFrame(bottom, text="逐句音频导出", padding=10)
        export_box.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._build_export_section(export_box)

        # 状态栏
        self.status_var = tk.StringVar(value="")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=(6, 2))
        status_bar.pack(fill="x", side="bottom")

    # ── 预览播放条（进度条 + 播放控制）────────────────────────────

    def _build_preview_bar(self, parent):
        # 第一行：播放/停止、从头重播、当前时间 / 总时长
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 4))

        self.preview_btn = ttk.Button(controls, text="▶ 播放", width=10,
                                       command=self.action_toggle_preview)
        self.preview_btn.pack(side="left", padx=(0, 4))

        self.replay_btn = ttk.Button(controls, text="⏮ 从头重播", width=12,
                                      command=self.action_replay_from_start)
        self.replay_btn.pack(side="left", padx=(0, 8))

        self.preview_time_var = tk.StringVar(value="0.000 / 0.000 s")
        ttk.Label(controls, textvariable=self.preview_time_var, width=22).pack(side="left", padx=4)

        self.preview_row_info_var = tk.StringVar(value="（未选中行）")
        ttk.Label(controls, textvariable=self.preview_row_info_var,
                  foreground="#666").pack(side="left", padx=8)

        # 第二行：可拖动进度条
        scale_row = ttk.Frame(parent)
        scale_row.pack(fill="x", pady=(2, 0))

        self.preview_scale_var = tk.DoubleVar(value=0.0)
        self.preview_scale = ttk.Scale(
            scale_row, from_=0.0, to=1.0, orient="horizontal",
            variable=self.preview_scale_var, command=self._on_scale_moved,
        )
        self.preview_scale.pack(side="left", fill="x", expand=True, padx=(0, 4))
        # 按下开始拖动、松开结束拖动（松开时确认 seek 并继续播放）
        self.preview_scale.bind("<ButtonPress-1>", self._on_scale_press)
        self.preview_scale.bind("<ButtonRelease-1>", self._on_scale_release)

        ttk.Label(scale_row, text="拖动可跳转", foreground="#888", font=("", 8)).pack(side="left")

    # ── 音频区（加载 + 剪掉开头，二者共用同一份内存音频）────────────

    def _build_audio_section(self, parent):
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="音频文件(wav)：").pack(side="left")
        self.audio_wav_var = tk.StringVar(value="（未加载）")
        ttk.Label(row1, textvariable=self.audio_wav_var, foreground="#0a6").pack(side="left", padx=4)
        ttk.Button(row1, text="加载...", command=self.action_load_audio).pack(side="right")

        ttk.Label(parent, text="剪掉开头（按拍裁剪，仅内存生效）", font=("", 9, "bold")).pack(
            anchor="w", pady=(8, 2))

        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(2, 2))

        ttk.Label(grid, text="BPM：").grid(row=0, column=0, sticky="e", padx=3, pady=2)
        self.bpm_var = tk.DoubleVar(value=120.0)
        bpm_spin = ttk.Spinbox(grid, from_=1, to=999, increment=1, width=8,
                                textvariable=self.bpm_var, command=self._refresh_beat_preview)
        bpm_spin.grid(row=0, column=1, sticky="w", padx=3, pady=2)
        bpm_spin.bind("<KeyRelease>", lambda e: self._refresh_beat_preview())

        ttk.Label(grid, text="拍号：").grid(row=0, column=2, sticky="e", padx=3, pady=2)
        beats_frame = ttk.Frame(grid)
        beats_frame.grid(row=0, column=3, sticky="w", padx=3, pady=2)
        self.beats_per_bar_var = tk.IntVar(value=4)
        self.beat_unit_var = tk.IntVar(value=4)
        bp_spin = ttk.Spinbox(beats_frame, from_=1, to=32, width=4, textvariable=self.beats_per_bar_var,
                               command=self._refresh_beat_preview)
        bp_spin.pack(side="left")
        ttk.Label(beats_frame, text="/").pack(side="left", padx=2)
        bu_spin = ttk.Spinbox(beats_frame, from_=1, to=32, width=4, textvariable=self.beat_unit_var,
                               command=self._refresh_beat_preview)
        bu_spin.pack(side="left")
        bp_spin.bind("<KeyRelease>", lambda e: self._refresh_beat_preview())
        bu_spin.bind("<KeyRelease>", lambda e: self._refresh_beat_preview())

        ttk.Label(grid, text="裁剪拍数：").grid(row=1, column=0, sticky="e", padx=3, pady=2)
        self.trim_beats_var = tk.DoubleVar(value=4.0)
        beats_spin = ttk.Spinbox(grid, from_=0, to=999, increment=0.25, width=8,
                                  textvariable=self.trim_beats_var, command=self._refresh_beat_preview)
        beats_spin.grid(row=1, column=1, sticky="w", padx=3, pady=2)
        beats_spin.bind("<KeyRelease>", lambda e: self._refresh_beat_preview())

        self.beat_preview_var = tk.StringVar(value="")
        ttk.Label(grid, textvariable=self.beat_preview_var, foreground="#666").grid(
            row=1, column=2, columnspan=2, sticky="w", padx=6)

        row_btn = ttk.Frame(parent)
        row_btn.pack(fill="x", pady=(8, 2))
        ttk.Button(row_btn, text="裁掉开头", command=self.action_trim_head).pack(side="left")
        ttk.Button(row_btn, text="另存为新 wav...", command=self.action_save_trimmed_as).pack(side="left", padx=6)
        ttk.Button(row_btn, text="重新加载(撤销裁剪)", command=self.action_reload_trim_audio).pack(side="left")

        self.trim_status_var = tk.StringVar(
            value="提示：裁剪只在内存中进行，不会覆盖磁盘上的原始文件；下面的逐句导出会使用裁剪后的音频。")
        ttk.Label(parent, textvariable=self.trim_status_var, foreground="#a60", wraplength=460, justify="left").pack(
            fill="x", pady=(6, 0))

        self._refresh_beat_preview()

    # ── 逐句导出区 ────────────────────────────────────────────────

    def _build_export_section(self, parent):
        row2 = ttk.Frame(parent)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="导出目录：").pack(side="left")
        self.export_dir_var = tk.StringVar(value="（未选择）")
        ttk.Label(row2, textvariable=self.export_dir_var, foreground="#0a6").pack(side="left", padx=4)
        ttk.Button(row2, text="选择...", command=self.action_choose_export_dir).pack(side="right")

        row3 = ttk.Frame(parent)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="文件名取文本前几字：").pack(side="left")
        self.filename_chars_var = tk.IntVar(value=20)
        ttk.Spinbox(row3, from_=1, to=100, width=6, textvariable=self.filename_chars_var).pack(side="left", padx=4)
        ttk.Label(row3, text="（文件名格式：index_命名片段.wav / .txt，index 按原值展示，不补零）",
                  foreground="#666").pack(side="left", padx=6)

        row3b = ttk.Frame(parent)
        row3b.pack(fill="x", pady=2)
        ttk.Label(row3b, text="命名片段来源：").pack(side="left")
        # "text"：行文本（原有默认行为，.ttslt 常用）
        # "title"：.dtslt 的 title 字段（对话框标题）——text 常为空时很有用
        # "stem"：原始 wav_path 的文件名主干，方便和源素材文件对应
        self.export_name_source_var = tk.StringVar(value="text")
        ttk.Radiobutton(row3b, text="文本 text", value="text",
                         variable=self.export_name_source_var).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(row3b, text="标题 title", value="title",
                         variable=self.export_name_source_var).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(row3b, text="源文件名 stem", value="stem",
                         variable=self.export_name_source_var).pack(side="left", padx=(0, 8))
        ttk.Label(row3b, text="（title/stem 缺失时自动回退用 text；.txt 内容始终是该行 text，不受影响）",
                  foreground="#666").pack(side="left", padx=6)

        # .list 文件导出（可选）：vocal_path|speaker_name|language|text，
        # 每行一句，vocal_path 用导出 wav 的绝对路径。
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(8, 6))

        row_list_chk = ttk.Frame(parent)
        row_list_chk.pack(fill="x", pady=2)
        self.export_list_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row_list_chk, text="同时输出 .list 文件（vocal_path|speaker_name|language|text，绝对路径）",
            variable=self.export_list_enabled_var,
        ).pack(side="left")

        row_list1 = ttk.Frame(parent)
        row_list1.pack(fill="x", pady=2)
        ttk.Label(row_list1, text="speaker_name：").pack(side="left")
        self.export_speaker_name_var = tk.StringVar(value="speaker_name")
        ttk.Entry(row_list1, textvariable=self.export_speaker_name_var, width=20).pack(side="left", padx=(2, 12))

        ttk.Label(row_list1, text="language：").pack(side="left")
        self.export_language_var = tk.StringVar(value="ZH")
        language_combo = ttk.Combobox(
            row_list1, textvariable=self.export_language_var, width=6, state="readonly",
            values=("ZH", "EN", "JA", "KO", "YUE"),
        )
        language_combo.pack(side="left", padx=(2, 0))

        row_list2 = ttk.Frame(parent)
        row_list2.pack(fill="x", pady=2)
        ttk.Label(
            row_list2,
            text='（示例：D:/Audio/xxx_0000000000_0000149440.wav|speaker_name|JA|やった！ついにできました！）',
            foreground="#666",
        ).pack(side="left")

        row4 = ttk.Frame(parent)
        row4.pack(fill="x", pady=(8, 2))
        ttk.Button(row4, text="按 index 顺序导出全部", command=self.action_export_all).pack(side="left")

        self.export_hint_var = tk.StringVar(
            value="提示：导出使用上方「音频」区当前加载的音频（若已裁剪开头，导出的就是裁剪后的音频）；\n"
                  "时间表未随裁剪自动偏移，如需对齐请先在表格里手动调整每行的起始/结束秒数。")
        ttk.Label(parent, textvariable=self.export_hint_var, foreground="#a60", wraplength=460, justify="left").pack(
            fill="x", pady=(6, 0))

    # ══════════════════════════════════════════════════════════════
    # 表格刷新 / 编辑
    # ══════════════════════════════════════════════════════════════

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        if not self.doc:
            return
        for i, row in enumerate(self.doc.rows):
            self.tree.insert("", "end", iid=str(i), values=(
                row.index, row.extra.get("title", ""), row.text,
                f"{row.start_sec:.6f}", f"{row.end_sec:.6f}", f"{row.duration_sec:.6f}",
            ))

    def _selected_row_position(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def _on_cell_double_click(self, event):
        if not self.doc:
            return
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if not row_id:
            return
        pos = int(row_id)
        col_index = int(col_id.replace("#", "")) - 1
        columns = ("index", "title", "text", "start_sec", "end_sec", "duration_sec")
        col_name = columns[col_index]
        if col_name == "duration_sec":
            messagebox.showinfo("提示", "时长由起始/结束时间自动计算，不能直接编辑。")
            return
        self._inline_edit_cell(pos, col_name)

    def _inline_edit_cell(self, pos: int, col_name: str):
        row = self.doc.rows[pos]
        current_value = {
            "index": str(row.index), "title": row.extra.get("title", ""), "text": row.text,
            "start_sec": f"{row.start_sec:.6f}", "end_sec": f"{row.end_sec:.6f}",
        }[col_name]
        label = {"index": "序号 index", "title": "标题 title", "text": "文本",
                  "start_sec": "起始(秒)", "end_sec": "结束(秒)"}[col_name]
        new_value = simpledialog.askstring("编辑单元格", f"{label}：", initialvalue=current_value, parent=self)
        if new_value is None:
            return
        try:
            if col_name == "index":
                row.index = int(new_value)
            elif col_name == "title":
                if new_value.strip():
                    row.extra["title"] = new_value
                else:
                    row.extra.pop("title", None)  # 清空则删掉该字段，而不是存一个空字符串
            elif col_name == "text":
                row.text = new_value
            elif col_name == "start_sec":
                row.start_sec = float(new_value)
            elif col_name == "end_sec":
                row.end_sec = float(new_value)
            row.duration_sec = max(0.0, row.end_sec - row.start_sec)
        except ValueError:
            messagebox.showerror("输入有误", "请输入合法的数值。")
            return
        self._refresh_table()
        self._update_status(f"已更新第 {pos + 1} 行的「{label}」。")

    def _edit_selected_row(self):
        pos = self._selected_row_position()
        if pos is None:
            messagebox.showinfo("提示", "请先在表格中选中一行。")
            return
        for col_name in ("index", "title", "text", "start_sec", "end_sec"):
            self._inline_edit_cell(pos, col_name)

    def _delete_selected_row(self):
        pos = self._selected_row_position()
        if pos is None or not self.doc:
            return
        if not messagebox.askyesno("确认删除", f"确定删除第 {pos + 1} 行吗？"):
            return
        del self.doc.rows[pos]
        self._refresh_table()
        self._update_status("已删除所选行。")

    def _move_row(self, direction: int):
        pos = self._selected_row_position()
        if pos is None or not self.doc:
            return
        new_pos = pos + direction
        if new_pos < 0 or new_pos >= len(self.doc.rows):
            return
        self.doc.rows[pos], self.doc.rows[new_pos] = self.doc.rows[new_pos], self.doc.rows[pos]
        self._refresh_table()
        self.tree.selection_set(str(new_pos))

    # ══════════════════════════════════════════════════════════════
    # 预览播放：选中行按钮 / 空格键，只播 [start_sec, end_sec) 这一段，
    # 用的是"音频"区当前加载的内存音频（若已裁剪开头，预览的就是裁剪
    # 后的结果）；再次触发 → 停止当前播放（开关式）。
    # 进度条实时更新；拖动进度条即时跳转并从新位置继续播放；
    # 「从头重播」从该句起点重新播放。
    # ══════════════════════════════════════════════════════════════

    def _on_space_key(self, event):
        # 双击编辑单元格时弹出的 simpledialog 是模态对话框，会拿到焦点，
        # 全局 bind_all 的按键不会传导过去，所以这里不需要额外判断"是否
        # 正在编辑"；但若焦点在其它可输入控件（如 BPM/拍号的 Spinbox）
        # 上，空格键应该走控件默认行为（比如空格触发按钮/输入数字），
        # 不应该被这里拦截去播放音频。
        focus_widget = self.focus_get()
        if isinstance(focus_widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Spinbox, tk.Text)):
            return
        self.action_toggle_preview()
        return "break"

    def _prepare_preview_clip(self) -> bool:
        """
        根据当前选中行 + 已加载音频，准备 _preview_clip。
        成功返回 True；失败时已弹提示，返回 False。
        """
        if sd is None:
            messagebox.showerror(
                "无法播放",
                "预览播放功能依赖 sounddevice 库，当前环境未能加载：\n"
                f"{_SOUNDDEVICE_IMPORT_ERROR}\n\n"
                "请先执行：pip install sounddevice\n"
                "（Windows 上通常安装后即可直接用；若仍报错，多半是缺少音频输出设备或系统层 PortAudio 驱动。）",
            )
            return False
        if self.loaded_audio is None:
            messagebox.showinfo("提示", "请先在「音频」区加载对应的音频 (wav)。")
            return False
        pos = self._selected_row_position()
        if pos is None or not self.doc:
            messagebox.showinfo("提示", "请先在表格中选中要预览的一行。")
            return False

        row = self.doc.rows[pos]
        sr = self.loaded_audio_sr
        total_samples = self.loaded_audio.shape[0]
        start_sample = max(0, int(round(row.start_sec * sr)))
        end_sample = min(total_samples, int(round(row.end_sec * sr)))
        if end_sample <= start_sample:
            messagebox.showinfo(
                "无法预览",
                "该行的起止时间在当前音频里是无效区间（可能落在裁剪后已经"
                "不存在的部分，或起始/结束时间本身有误），请检查表格。",
            )
            return False

        self._preview_clip = self.loaded_audio[start_sample:end_sample]
        self._preview_sr = sr
        self._preview_row_pos = pos
        self._preview_cursor = 0
        self.preview_scale.configure(to=max(1, len(self._preview_clip) - 1))
        self._set_scale_value(0)
        self._update_preview_time_labels(0)
        self.preview_row_info_var.set(
            f"第 {pos + 1} 行 · index={row.index} · "
            f"{row.start_sec:.3f}s ~ {row.end_sec:.3f}s"
            f"（时长 {row.duration_sec:.3f}s）"
        )
        return True

    def action_toggle_preview(self):
        if self._is_previewing:
            self._stop_preview(keep_clip=True)
            return
        # 若尚未准备 clip，或选中行已变，重新准备
        pos = self._selected_row_position()
        if (self._preview_clip is None
                or self._preview_row_pos != pos
                or self._preview_sr != self.loaded_audio_sr):
            if not self._prepare_preview_clip():
                return
        self._start_playback_from_cursor()

    def action_replay_from_start(self):
        """从头重播：停掉当前播放，cursor 归零，再从起点播放。"""
        if self._is_previewing:
            self._stop_preview(keep_clip=True)
        pos = self._selected_row_position()
        if (self._preview_clip is None
                or self._preview_row_pos != pos
                or self._preview_sr != self.loaded_audio_sr):
            if not self._prepare_preview_clip():
                return
        self._preview_cursor = 0
        self._set_scale_value(0)
        self._update_preview_time_labels(0)
        self._start_playback_from_cursor()

    def _start_playback_from_cursor(self):
        if self._preview_clip is None or self._preview_sr is None:
            return
        n = len(self._preview_clip)
        cursor = max(0, min(self._preview_cursor, n - 1 if n > 0 else 0))
        if n == 0 or cursor >= n:
            self._finish_preview_naturally()
            return
        remaining = self._preview_clip[cursor:]
        try:
            sd.play(remaining, self._preview_sr)
        except Exception as e:
            self._show_error("播放失败", e)
            return

        self._preview_cursor = cursor
        self._preview_play_start_cursor = cursor
        self._preview_play_started_at = time.monotonic()
        self._is_previewing = True
        self.preview_btn.config(text="■ 停止")
        pos = self._preview_row_pos
        row_hint = f"第 {pos + 1} 行" if pos is not None else "当前句"
        self._update_status(
            f"正在预览播放 {row_hint}（从 {cursor / self._preview_sr:.3f}s 起）"
            f"……再按一次空格/按钮可停止；拖动进度条可跳转。"
        )
        self._poll_preview_progress()

    def _poll_preview_progress(self):
        """
        每 50ms 更新进度条与时间标签；检测播放是否自然结束。
        拖动进度条期间不自动改 Scale 值（避免和用户拖动手势打架）。
        """
        if not self._is_previewing:
            return

        # 估算当前采样位置
        if (self._preview_play_started_at is not None
                and self._preview_sr
                and self._preview_clip is not None):
            elapsed = time.monotonic() - self._preview_play_started_at
            estimated = self._preview_play_start_cursor + int(round(elapsed * self._preview_sr))
            estimated = max(0, min(estimated, len(self._preview_clip)))
            self._preview_cursor = estimated
            if not self._is_dragging_scale:
                self._set_scale_value(estimated)
                self._update_preview_time_labels(estimated)

        # 检查是否还在播
        try:
            still_playing = bool(sd.get_stream().active)
        except Exception:
            still_playing = False

        if still_playing:
            self._preview_poll_job = self.after(50, self._poll_preview_progress)
        else:
            # 自然播完
            self._finish_preview_naturally()

    def _finish_preview_naturally(self):
        """播放自然结束：cursor 归零（回到句首），按钮切回未播放，保留 clip
        方便直接再按一次空格/播放即可从头重播（不强制要求点「从头重播」）。"""
        if self._preview_poll_job is not None:
            try:
                self.after_cancel(self._preview_poll_job)
            except Exception:
                pass
            self._preview_poll_job = None
        self._is_previewing = False
        self._preview_play_started_at = None
        self._preview_cursor = 0
        self._set_scale_value(0)
        self._update_preview_time_labels(0)
        self.preview_btn.config(text="▶ 播放")
        self._update_status("预览播放结束（已回到句首，可直接再次播放）。")

    def _stop_preview(self, keep_clip: bool = False):
        """
        手动停止播放。
        keep_clip=True：保留已切片的 clip / cursor，方便继续拖动或重播；
        keep_clip=False：连 clip 一并清掉（退出程序时用）。
        """
        # 先把 cursor 定格在当前估算位置，再停流
        if (self._is_previewing
                and self._preview_play_started_at is not None
                and self._preview_sr
                and self._preview_clip is not None):
            elapsed = time.monotonic() - self._preview_play_started_at
            estimated = self._preview_play_start_cursor + int(round(elapsed * self._preview_sr))
            estimated = max(0, min(estimated, len(self._preview_clip)))
            self._preview_cursor = estimated

        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        if self._preview_poll_job is not None:
            try:
                self.after_cancel(self._preview_poll_job)
            except Exception:
                pass
            self._preview_poll_job = None
        self._is_previewing = False
        self._preview_play_started_at = None
        self.preview_btn.config(text="▶ 播放")
        if keep_clip:
            if self._preview_clip is not None:
                self._set_scale_value(self._preview_cursor)
                self._update_preview_time_labels(self._preview_cursor)
        else:
            self._preview_clip = None
            self._preview_sr = None
            self._preview_cursor = 0
            self._preview_row_pos = None
            self._set_scale_value(0)
            self._update_preview_time_labels(0)
            self.preview_row_info_var.set("（未选中行）")
        self._update_status("已停止预览播放。")

    # ── 进度条拖动 / 跳转 ─────────────────────────────────────────

    def _set_scale_value(self, sample_pos: int):
        """程序侧更新 Scale，避免触发 command 回调里的 seek。"""
        self._scale_programmatic = True
        try:
            self.preview_scale_var.set(float(sample_pos))
        finally:
            self._scale_programmatic = False

    def _update_preview_time_labels(self, sample_pos: int):
        if self._preview_clip is None or not self._preview_sr:
            self.preview_time_var.set("0.000 / 0.000 s")
            return
        total = len(self._preview_clip)
        cur_sec = sample_pos / self._preview_sr
        total_sec = total / self._preview_sr
        self.preview_time_var.set(f"{cur_sec:.3f} / {total_sec:.3f} s")

    def _on_scale_press(self, event):
        self._is_dragging_scale = True
        self._was_playing_before_drag = self._is_previewing
        # 若还没有 clip（用户先拖进度条），尝试按当前选中行准备
        if self._preview_clip is None:
            self._prepare_preview_clip()

    def _on_scale_release(self, event):
        self._is_dragging_scale = False
        if self._preview_clip is None:
            return
        sample_pos = int(round(self.preview_scale_var.get()))
        sample_pos = max(0, min(sample_pos, len(self._preview_clip) - 1
                                 if len(self._preview_clip) > 0 else 0))
        # 松开时最终确认位置：之前在播 或 拖动过程中已开播 → 继续从新位置出声
        should_resume = self._was_playing_before_drag or self._is_previewing
        self._seek_to(sample_pos, resume=should_resume)
        self._was_playing_before_drag = False

    def _on_scale_moved(self, value_str):
        """
        Scale 的 command 回调：拖动过程中实时更新时间标签，
        并节流后即时跳转播放位置（拖动中持续出声，松开时再精确定位一次）。
        """
        if self._scale_programmatic:
            return
        if self._preview_clip is None or not self._preview_sr:
            return
        try:
            sample_pos = int(round(float(value_str)))
        except (TypeError, ValueError):
            return
        sample_pos = max(0, min(sample_pos, len(self._preview_clip) - 1
                                 if len(self._preview_clip) > 0 else 0))
        self._preview_cursor = sample_pos
        self._update_preview_time_labels(sample_pos)
        # 拖动过程中即时跳转并继续播放（约 120ms 节流，避免 stop/start 过于频繁）
        if self._is_dragging_scale:
            now = time.monotonic()
            if now - self._last_seek_at >= 0.12:
                self._last_seek_at = now
                self._seek_to(sample_pos, resume=True)

    def _seek_to(self, sample_pos: int, resume: bool = True):
        """
        跳转到 clip 内指定采样点。
        resume=True 时从新位置继续出声；否则只更新 cursor / 进度条。
        """
        if self._preview_clip is None or not self._preview_sr:
            return
        n = len(self._preview_clip)
        sample_pos = max(0, min(sample_pos, n - 1 if n > 0 else 0))

        # 先停掉当前播放流
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        if self._preview_poll_job is not None:
            try:
                self.after_cancel(self._preview_poll_job)
            except Exception:
                pass
            self._preview_poll_job = None

        self._preview_cursor = sample_pos
        self._set_scale_value(sample_pos)
        self._update_preview_time_labels(sample_pos)
        self._is_previewing = False
        self._preview_play_started_at = None
        self.preview_btn.config(text="▶ 播放")

        if resume:
            self._start_playback_from_cursor()

    # ══════════════════════════════════════════════════════════════
    # 文件：打开 / 导入 / 保存
    # ══════════════════════════════════════════════════════════════

    def action_open_timeline(self):
        path = filedialog.askopenfilename(
            title="打开 .ttslt / .dtslt 文件",
            filetypes=[("SVS-Lab-Tools 时间表", "*.ttslt *.dtslt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.doc = tc.open_timeline_file(path)
            self.current_doc_path = Path(path)
        except Exception as e:
            self._show_error("打开失败", e)
            return
        self._after_doc_loaded(f"已打开: {Path(path).name}")

    def action_import_menu_button(self):
        path = filedialog.askopenfilename(
            title="导入字幕/标注文件",
            filetypes=[("字幕/标注文件", "*.srt *.lrc *.lab"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self._do_import(path)

    def action_import(self, expected_suffix: str):
        path = filedialog.askopenfilename(
            title=f"导入 {expected_suffix} 文件",
            filetypes=[(expected_suffix, f"*{expected_suffix}"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self._do_import(path)

    def _do_import(self, path: str):
        try:
            self.doc = tc.import_subtitle_like(path)
            self.current_doc_path = None
        except Exception as e:
            self._show_error("导入失败", e)
            return
        self._after_doc_loaded(f"已导入: {Path(path).name}（尚未对应任何 .ttslt/.dtslt 文件，"
                                f"请用「另存为」保存）")

    def _after_doc_loaded(self, status_msg: str):
        self._stop_preview(keep_clip=False)
        self._refresh_table()
        kind_label = {"ttslt": "TTS跟读时间表 (.ttslt)", "dtslt": "对话批量时间表 (.dtslt)",
                      "imported": "导入的字幕/标注（尚未保存）"}.get(self.doc.kind, self.doc.kind)
        self.doc_info_var.set(f"{kind_label} · 共 {len(self.doc.rows)} 行 · "
                               f"总时长 {self.doc.total_duration_sec:.3f}s")
        self._update_status(status_msg)

    def action_save(self):
        if not self.doc:
            messagebox.showinfo("提示", "还没有打开任何时间表文件。")
            return
        if self.current_doc_path is None:
            self.action_save_as()
            return
        try:
            tc.save_timeline_file(self.doc, self.current_doc_path)
        except Exception as e:
            self._show_error("保存失败", e)
            return
        self._update_status(f"已保存: {self.current_doc_path.name}")

    def action_save_as(self):
        if not self.doc:
            messagebox.showinfo("提示", "还没有打开任何时间表文件。")
            return
        default_ext = ".ttslt" if self.doc.kind in ("ttslt", "imported") else ".dtslt"
        default_name = (self.current_doc_path.stem if self.current_doc_path
                         else self.doc.raw.get("stem") or "timeline")
        path = filedialog.asksaveasfilename(
            title="另存为",
            defaultextension=default_ext,
            initialfile=default_name + default_ext,
            filetypes=[("TTS跟读时间表", "*.ttslt"), ("对话批量时间表", "*.dtslt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        path = Path(path)
        # 依据用户选择的扩展名决定写出的 doc.kind（.dtslt 需要 boxes 结构）
        if path.suffix.lower() == ".dtslt" and self.doc.kind != "dtslt":
            self.doc.kind = "dtslt"
            self.doc.row_key = "boxes"
            self.doc.raw.pop("sentences", None)
            self.doc.raw["kind"] = "dialogue_batch_timeline"
        elif path.suffix.lower() == ".ttslt" and self.doc.kind == "dtslt":
            self.doc.kind = "ttslt"
            self.doc.row_key = "sentences"
            self.doc.raw.pop("boxes", None)
            self.doc.raw["kind"] = "tts_alignment_timeline"
        try:
            tc.save_timeline_file(self.doc, path)
        except Exception as e:
            self._show_error("保存失败", e)
            return
        self.current_doc_path = path
        self._update_status(f"已另存为: {path.name}")
        self._after_doc_loaded(f"已另存为: {path.name}")

    # ══════════════════════════════════════════════════════════════
    # 逐句音频导出
    # ══════════════════════════════════════════════════════════════

    def action_choose_export_dir(self):
        path = filedialog.askdirectory(title="选择导出目录")
        if not path:
            return
        self.export_dir_var.set(path)

    def action_export_all(self):
        if not self.doc or not self.doc.rows:
            messagebox.showinfo("提示", "当前没有可导出的时间表数据。")
            return
        if self.loaded_audio is None:
            messagebox.showinfo("提示", "请先在上方「音频」区加载对应的音频 (wav)。")
            return
        export_dir = self.export_dir_var.get()
        if not export_dir or export_dir == "（未选择）":
            messagebox.showinfo("提示", "请先选择导出目录。")
            return
        try:
            results = tc.export_rows_as_audio_and_txt(
                self.doc, export_dir,
                filename_text_chars=int(self.filename_chars_var.get()),
                audio=self.loaded_audio, sr=self.loaded_audio_sr,
                name_source=self.export_name_source_var.get(),
            )
        except Exception as e:
            self._show_error("导出失败", e)
            return

        list_path: Optional[Path] = None
        if self.export_list_enabled_var.get():
            try:
                list_path = tc.write_list_file(
                    results, self.doc, Path(export_dir) / "filelist.list",
                    speaker_name=self.export_speaker_name_var.get(),
                    language=self.export_language_var.get(),
                )
            except Exception as e:
                self._show_error(".list 文件导出失败", e)
                return

        skipped = len(self.doc.rows) - len(results)
        trimmed_note = (f"\n（当前音频已裁剪开头共 {self.trimmed_seconds_total:.4f} 秒，"
                         f"导出使用的是裁剪后的音频）" if self.trimmed_seconds_total > 0 else "")
        msg = f"导出完成：共 {len(results)} 句成功导出到\n{export_dir}{trimmed_note}"
        if skipped > 0:
            msg += f"\n（{skipped} 行因起止时间无效或落在已裁剪掉的区间外被跳过，请检查表格）"
        if list_path is not None:
            msg += f"\n.list 文件已写入：\n{list_path}"
        messagebox.showinfo("导出完成", msg)
        status_msg = f"已导出 {len(results)} 句音频+文本到 {export_dir}"
        if list_path is not None:
            status_msg += f"，并写出 {list_path.name}"
        self._update_status(status_msg)

    # ══════════════════════════════════════════════════════════════
    # 音频加载 & 剪掉开头（同一份内存音频，加载一次两处共用）
    # ══════════════════════════════════════════════════════════════

    def _refresh_beat_preview(self):
        try:
            bpm = float(self.bpm_var.get())
            beat_unit = int(self.beat_unit_var.get())
            beats_per_bar = int(self.beats_per_bar_var.get())
            beat_sec = tc.beat_duration_sec(bpm, beats_per_bar, beat_unit)
            n_beats = float(self.trim_beats_var.get())
            total_trim_sec = beat_sec * n_beats
            self.beat_preview_var.set(f"一拍 = {beat_sec:.4f} 秒 → 裁 {n_beats:g} 拍 = 共 {total_trim_sec:.4f} 秒")
        except Exception:
            self.beat_preview_var.set("（请输入合法的 BPM / 拍号 / 拍数）")

    def action_load_audio(self):
        path = filedialog.askopenfilename(
            title="加载音频 (wav)",
            filetypes=[("WAV 音频", "*.wav"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            audio, sr = sf.read(path, always_2d=False)
        except Exception as e:
            self._show_error("加载音频失败", e)
            return
        self._stop_preview(keep_clip=False)
        self.loaded_audio = audio
        self.loaded_audio_sr = sr
        self.trimmed_seconds_total = 0.0
        self._audio_source_path = path
        self.audio_wav_var.set(Path(path).name)
        dur = len(audio) / sr
        self.trim_status_var.set(f"已加载：{Path(path).name}（{dur:.3f} 秒，采样率 {sr}Hz）。"
                                  f"裁剪只在内存中进行，不会覆盖磁盘文件；下面的逐句导出会使用这份音频。")
        self._update_status(f"已加载音频: {Path(path).name}")

    def action_reload_trim_audio(self):
        if not self._audio_source_path:
            messagebox.showinfo("提示", "还没有加载过音频。")
            return
        try:
            audio, sr = sf.read(self._audio_source_path, always_2d=False)
        except Exception as e:
            self._show_error("重新加载失败", e)
            return
        self._stop_preview(keep_clip=False)
        self.loaded_audio = audio
        self.loaded_audio_sr = sr
        self.trimmed_seconds_total = 0.0
        self.trim_status_var.set("已重新加载原始音频，撤销了之前的裁剪（内存状态已重置）。")
        self._update_status("已重新加载原始音频，裁剪已撤销。")

    def action_trim_head(self):
        if self.loaded_audio is None:
            messagebox.showinfo("提示", "请先加载音频。")
            return
        try:
            bpm = float(self.bpm_var.get())
            beat_unit = int(self.beat_unit_var.get())
            beats_per_bar = int(self.beats_per_bar_var.get())
            n_beats = float(self.trim_beats_var.get())
            beat_sec = tc.beat_duration_sec(bpm, beats_per_bar, beat_unit)
        except Exception as e:
            self._show_error("参数有误", e)
            return
        trim_sec = beat_sec * n_beats
        current_dur = len(self.loaded_audio) / self.loaded_audio_sr
        if trim_sec >= current_dur:
            if not messagebox.askyesno(
                "确认",
                f"要裁掉的 {trim_sec:.3f} 秒 ≥ 当前音频剩余长度 {current_dur:.3f} 秒，"
                f"裁剪后音频会变成空的。确定继续吗？"):
                return
        self._stop_preview(keep_clip=False)
        self.loaded_audio = tc.trim_audio_head(self.loaded_audio, self.loaded_audio_sr, trim_sec)
        self.trimmed_seconds_total += trim_sec
        new_dur = len(self.loaded_audio) / self.loaded_audio_sr
        self.trim_status_var.set(
            f"已裁掉开头 {trim_sec:.4f} 秒（本次），累计裁掉 {self.trimmed_seconds_total:.4f} 秒。"
            f"剩余时长 {new_dur:.3f} 秒。仅内存生效，尚未写入磁盘——「按 index 顺序导出全部」会直接使用"
            f"这份裁剪后的音频；如需单独保存裁剪结果，用「另存为新 wav」。\n"
            f"注意：时间表的 start_sec/end_sec 不会自动跟着偏移，如果需要对齐，请在表格里手动减去"
            f"已裁掉的秒数。"
        )
        self._update_status(f"已在内存中裁掉开头 {trim_sec:.4f} 秒。")

    def action_save_trimmed_as(self):
        if self.loaded_audio is None:
            messagebox.showinfo("提示", "请先加载并裁剪音频。")
            return
        default_name = (Path(self._audio_source_path).stem + "_trimmed.wav"
                         if self._audio_source_path else "trimmed.wav")
        path = filedialog.asksaveasfilename(
            title="另存为新 wav", defaultextension=".wav", initialfile=default_name,
            filetypes=[("WAV 音频", "*.wav"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            sf.write(path, self.loaded_audio, self.loaded_audio_sr)
        except Exception as e:
            self._show_error("保存失败", e)
            return
        messagebox.showinfo("已保存", f"裁剪后的音频已保存到:\n{path}")
        self._update_status(f"已另存为新 wav: {Path(path).name}")

    # ══════════════════════════════════════════════════════════════
    # 通用
    # ══════════════════════════════════════════════════════════════

    def _update_status(self, msg: str):
        self.status_var.set(msg)

    def _on_exit(self):
        self._stop_preview(keep_clip=False)
        self.destroy()

    def _show_error(self, title: str, e: Exception):
        messagebox.showerror(title, str(e))
        traceback.print_exc()

    def action_show_help(self):
        messagebox.showinfo(
            "关于 / 使用说明",
            "逐句音频切分工具（独立于 SVS-Lab-Tools）\n\n"
            "1. 打开 .ttslt / .dtslt，或导入 .srt / .lrc / .lab\n"
            "2. 双击表格单元格可编辑标题/文本/起始/结束时间\n"
            "3. 「音频」区：加载一次对应的 wav——「剪掉开头」和「逐句音频\n"
            "   导出」共用这同一份内存音频，不需要分别选择文件。\n"
            "4. 「剪掉开头」：按 BPM/拍号计算出 N 拍对应的秒数，从内存中\n"
            "   的音频头部裁掉一段——只在内存里生效，不覆盖磁盘上的原始\n"
            "   文件；「时间表」不会自动跟着偏移，需要对齐时请在表格里\n"
            "   手动把每行的起始/结束秒数减去已裁掉的秒数。\n"
            "5. 「逐句音频导出」：选择导出目录后，按 index 顺序把「音频」\n"
            "   区当前加载的音频（若已裁剪，就是裁剪后的）切分导出，\n"
            "   文件名格式：index_命名片段.wav / .txt\n"
            "   命名片段来源可选「文本 text / 标题 title / 源文件名 stem」，\n"
            "   title/stem 缺失时自动回退用 text；.txt 内容始终是该行\n"
            "   原始 text，不受命名来源影响。\n"
            "6. 预览播放：选中表格中一行后，点「▶ 播放」或按空格键，\n"
            "   只播放该行 [起始, 结束) 区间，播完自动停；再按一次空格/\n"
            "   按钮会提前停止。用的是「音频」区当前加载的音频。\n"
            "7. 预览进度条：播放过程中实时显示当前位置；拖动进度条可\n"
            "   即时跳转并从新位置继续播放；「从头重播」从该句起点重播。\n\n"
            "注：BPM 与拍长换算按标准公式 一拍秒数=60/BPM 计算，\n"
            "界面上会实时显示换算结果，请下刀前核对一遍。",
        )

    def action_show_license(self):
        license_text = """
    MIT License

    Copyright (c) 2026 liuhua520-svg (https://github.com/liuhua520-svg/Sentence-Audio-Splitter)


    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software.

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.


    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED.
    """

        messagebox.showinfo(
            "许可证",
            license_text
        )
        
    def action_show_numpy_license(self):
        numpy_license_text = """
        Copyright (c) 2005-2025, NumPy Developers.
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

            * Redistributions of source code must retain the above copyright
               notice, this list of conditions and the following disclaimer.

            * Redistributions in binary form must reproduce the above
               copyright notice, this list of conditions and the following
               disclaimer in the documentation and/or other materials provided
               with the distribution.

            * Neither the name of the NumPy Developers nor the names of any
               contributors may be used to endorse or promote products derived
               from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
        A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """
        
        messagebox.showinfo(
            "NumPy License",
            numpy_license_text
        )


    def action_show_soundfile_license(self):
        soundfile_license_text = """
        Copyright (c) 2013, Bastian Bechtold
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

          * Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.
          * Redistributions in binary form must reproduce the above copyright
            notice, this list of conditions and the following disclaimer in
            the documentation and/or other materials provided with the
            distribution.
          * Neither the name of python-soundfile nor the names
            of its contributors may be used to endorse or promote products
            derived from this software without specific prior written
            permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
        A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """
        
        messagebox.showinfo(
            "SoundFile License",
            soundfile_license_text
        )


    def action_show_sounddevice_license(self):
        sounddevice_license_text = """
        Copyright (c) 2015-2025 Matthias Geier

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.
        """
        
        messagebox.showinfo(
            "SoundDevice License",
            sounddevice_license_text
        )

def main():
    app = SplitterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
