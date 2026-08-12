#!/usr/bin/env python3
# ============================================================================
# av1_pipeline_gui_v0_1.py  V0.23  2026-08-02
#
# Folder boxes start from av1_pipeline_config.txt in this same folder.
# Picking folders here overrides that and is remembered in
# gui_settings.json; delete that file to go back to the config values.
# Front end for av1_pipeline_v0_1.py. The pipeline runs as a child process
# and stays the only encoding engine; this window drives it, streams its
# log, and keeps the common flags one click away. Closing or killing the
# GUI never hurts a run in a way the journal cannot resume.
# Runs the same on the .209 desktop and the G16 laptop.
#
# Hidden flag: --smoke runs a scripted analyze pass for automated checks;
# folders come from AV1GUI_SMOKE_SRC and AV1GUI_SMOKE_DST.
# ============================================================================

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "AV1 Pipeline V0.24"
HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "av1_pipeline_v0_1.py"
SETTINGS = HERE / "gui_settings.json"
CONFIG = HERE / "av1_pipeline_config.txt"
LOG_CAP_LINES = 4000
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SR_MODELS = ["realesr-animevideov3", "realesrgan-x4plus",
             "realesrgan-x4plus-anime"]
LOG_MODES = ["Output folder", "Software folder", "Custom"]
ENCODERS = ["auto", "nvenc", "svtav1"]
AUDIO_MODES = ["auto", "copy", "opus", "aac"]
# Lines of actual picture to target. 1080p first because it is the
# standard and the default; 4K is opt-in and only does anything when the
# source is already close to 4K.
TARGET_RES = ["1080p", "4K"]
TARGET_RES_FLAG = {"1080p": "1080", "4K": "2160"}

# Initial selections for the combo boxes.
#
# Deliberately not the first entry of each list above. The lists are
# ordered for the reader - safest or fastest first, so the tooltips read
# sensibly - while the default lands on what this particular machine
# actually wants. Decoupling the two means neither has to compromise.
#
# Precedence, lowest to highest: these constants, then
# av1_pipeline_config.txt, then gui_settings.json. Note that the last of
# those is deliberately NOT consulted for no_sr and sdr_to_hdr; see
# load_settings for why.
DEFAULT_SR_MODEL = "realesrgan-x4plus"
DEFAULT_ENCODER = "nvenc"
DEFAULT_AUDIO_MODE = "opus"
# 1080p, as instructed. Like no_sr and sdr_to_hdr this is library-wide
# policy, so it comes from the config on every launch and is NOT
# persisted to gui_settings.json.
DEFAULT_TARGET_RES = "1080p"

# log line readers for the status bar
RE_FILE = re.compile(r"\[(\d+)/(\d+)\]\s+(.+?)\s+\(")
RE_PROG = re.compile(r"encode\s+(\S+):\s+([0-9.]+)\s+pct.+speed\s+(\S+)")
RE_SR   = re.compile(r"sr chunk\s+(\d+)/(\d+)\s+([0-9.]+)x")
RE_DONE = re.compile(r"run complete:")


def read_config():
    # same plain "key = value" file the pipeline reads, so the folders
    # only ever get typed in one place; a missing file is not an error
    cfg = {}
    try:
        text = CONFIG.read_text(encoding="utf-8")
    except OSError:
        return cfg
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cfg[key.strip().lower()] = val.strip()
    return cfg


def as_bool(v, default=False):
    # config values are text; anything obviously off reads as off
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "", "no", "off", "false")


def build_args(py, pipeline, s):
    """Translate the window's control values into a pipeline argv.

    Options matching the pipeline's own default are omitted rather than
    passed explicitly, so the config file remains in charge of anything
    the operator has not deliberately changed here. The pipeline
    revalidates everything it receives; this is a convenience layer, not
    a trust boundary.

    Args:
        py: interpreter to run the pipeline with.
        pipeline: path to av1_pipeline_v0_1.py.
        s: control values, as collected in start().

    Returns:
        list[str]: argv for subprocess.Popen.
    """
    a = [str(py), "-u", str(pipeline), "--src", s["src"], "--dst", s["dst"]]
    if s.get("analyze"):
        a.append("--analyze-only")
    if s.get("retry"):
        a.append("--retry-failed")
    if s.get("no_sr"):
        a.append("--no-sr")
    if s.get("sdr_to_hdr"):
        a.append("--sdr-to-hdr")
    tr = TARGET_RES_FLAG.get((s.get("target_res") or "").strip())
    if tr:
        a += ["--target-res", tr]
    if s.get("no_subs"):
        a.append("--no-subs")
    if s.get("force"):
        a.append("--force")
    if int(s.get("limit", 0) or 0) > 0:
        a += ["--limit", str(int(s["limit"]))]
    a += ["--min-free-gb", str(float(s.get("minfree", 20) or 20))]
    ld = (s.get("log_dir") or "").strip()
    if ld:
        a += ["--log-dir", ld]
    # Pass these EXPLICITLY, always.
    #
    # They used to be omitted whenever they matched the first entry of
    # the list above, on the theory that the pipeline's own default
    # would then apply. It does not: with no flag the pipeline falls
    # back to av1_pipeline_config.txt, not to its built-in default. So
    # the one selection that most needed to reach it never did -
    # SR_MODELS[0] is realesr-animevideov3, the config says
    # realesrgan-x4plus, and choosing animevideov3 in this window
    # emitted nothing at all and ran x4plus. The dropdown could not
    # select the value it was showing.
    am = (s.get("audio_mode") or "").strip()
    if am:
        a += ["--audio-mode", am]
    st = str(s.get("sr_strength", "")).strip()
    if st:
        a += ["--sr-strength", str(int(float(st)))]
    enc = (s.get("encoder") or "").strip()
    if enc:
        a += ["--encoder", enc]
    model = s.get("sr_model", "").strip()
    if model:
        a += ["--sr-model", model]
    return a


class Tip:
    """Hover balloon. Plain tk so it needs nothing beyond the stdlib."""

    def __init__(self, widget, text, delay=450, width=460):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.width = width
        self.win = None
        self.after_id = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _=None):
        self.cancel()
        self.after_id = self.widget.after(self.delay, self.show)

    def cancel(self):
        if self.after_id:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def show(self):
        if self.win or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry("+%d+%d" % (x, y))
        tk.Label(self.win, text=self.text, justify="left",
                 background="#ffffe0", foreground="#000000",
                 relief="solid", borderwidth=1, padx=8, pady=6,
                 wraplength=self.width).pack()

    def hide(self, _=None):
        self.cancel()
        if self.win:
            try:
                self.win.destroy()
            except Exception:
                pass
            self.win = None


def tip(widget, text):
    Tip(widget, text)
    return widget


# Hover text. Acronyms get spelled out the first time they appear so the
# window explains itself without a manual open beside it.
TIPS = {
"src": "Folder the files are read from.\n\n"
       "Subfolders are read too, and the same subfolder layout is "
       "recreated in the destination. Files under 5 MB are ignored.\n\n"
       "Source files are only ever read. Nothing here is moved, "
       "changed, or deleted.",

"dst": "Folder the finished files are written to.\n\n"
       "Also holds _journal.json (what is finished), a failed folder "
       "with one diagnostic file per file that did not make it, and "
       "_analysis_report.txt after an Analyze only pass.\n\n"
       "It cannot be inside the source folder, or the other way round.",

"logdir": "Where pipeline.log and av1_trace.log are written.\n\n"
          "Output folder: a logs folder inside the destination. This is "
          "the default and keeps each batch's logs with its results.\n"
          "Software folder: beside the program, so logs build up in one "
          "place across every batch.\n"
          "Custom: anywhere you like.\n\n"
          "pipeline.log is the readable one. av1_trace.log is the deep "
          "debug trace and is the file to send when something breaks.",

"analyze": "Read every file, work out what would be done to it, write "
           "_analysis_report.txt, and encode nothing.\n\n"
           "Costs a few seconds per file and no GPU time. Worth running "
           "before any large batch so you can check the plan first.",

"retry": "Clear the failed marks from the journal so files that failed "
         "on an earlier run are queued again.\n\n"
         "Use after fixing whatever caused the failure. Without this, "
         "failed files stay skipped.",

"force": "Re-encode files even when the journal already says done.\n\n"
         "Normally a finished file is skipped forever. Tick this when "
         "you have changed settings and want the same files done again. "
         "Existing output files are overwritten.",

"limit": "Stop after this many files. 0 means the whole queue.\n\n"
         "Set it to 1 for a first test on a new setting, so you get one "
         "result to look at instead of a full batch.",

"minfree": "GB stands for gigabytes. The run stops before starting a "
           "file if free space on the destination drive is below this.\n\n"
           "Raise it if the drive is getting tight. An encode that runs "
           "out of space part way through wastes the whole file.",

"encoder": "Which encoder does the work.\n\n"
           "auto: test the GPU at startup and fall back to the CPU if it "
           "cannot encode AV1. Right for almost everyone, and it goes "
           "back to the GPU on its own once the GPU works again.\n"
           "nvenc: force the GPU. The run stops if it will not start.\n"
           "svtav1: force the CPU encoder and skip the GPU test.\n\n"
           "NVENC is the encoder chip on the NVIDIA card. AV1 ENCODE "
           "arrived with Ada:\n"
           "  RTX 50 series (Blackwell)  - yes\n"
           "  RTX 40 series (Ada)        - yes\n"
           "  RTX 30 series (Ampere)     - no, decode only\n"
           "  RTX 20 / GTX 16 and older  - no\n\n"
           "An RTX 30 series card is not a fault and needs no fixing: "
           "the run moves to the CPU encoder by itself and every file "
           "still completes. Even on a card that can encode AV1, the "
           "ffmpeg build and the driver have to agree on the NVENC API "
           "version, and ffmpeg is sometimes ahead of the newest driver. "
           "The CPU encoder is slower but the quality is at least as "
           "good, so a batch can keep running either way.",

"srmodel": "SR stands for super resolution, the AI upscaler used on "
           "files smaller than 1080 lines. The tool is Real-ESRGAN.\n\n"
           "realesr-animevideov3: fastest, best on animation and cartoons.\n"
           "realesrgan-x4plus: slower, general purpose, live action.\n"
           "realesrgan-x4plus-anime: slower, tuned for anime line art.\n\n"
           "Only used on sources under 1080 lines. Anything already "
           "1080 or larger ignores this.",

"audio": "What happens to the audio track.\n\n"
         "auto: keep the original stream untouched when it is already "
         "a codec every player handles (AC3, E-AC3, AAC, DTS, TrueHD, "
         "FLAC, MP3, Opus, PCM), and convert anything more awkward to "
         "Opus. Loses nothing on a normal rip.\n"
         "copy: never re-encode. Truest to the source.\n"
         "opus: always convert to Opus. Best quality for the size of "
         "any current lossy codec, and handles surround properly.\n"
         "aac: always convert to AAC.\n\n"
         "Re-encoding audio that is already lossy loses a little every "
         "time, so copying is the honest answer when the source codec "
         "is fine. Surround is kept as surround; turn on audio_downmix "
         "in the config file if you actually want stereo.",

"srstrength": "How much of the AI upscale to keep, 0 to 100.\n\n"
              "The upscaler looks at one frame at a time with no memory "
              "of the frame before it, so it invents slightly different "
              "detail every frame. On a still background that reads as "
              "crawling or shimmering. Tools like Topaz avoid this by "
              "feeding the model several frames at once, which is a "
              "property of the model and cannot be added here.\n\n"
              "This blends the model output back towards a plain lanczos "
              "upscale instead. 100 is the raw model with all its "
              "shimmer, 0 is no AI at all, 50 keeps half the invented "
              "detail and roughly halves the flicker. Drop it further if "
              "backgrounds still crawl.\n\n"
              "A temporal stabiliser runs after the blend as well. Turn "
              "that off in the config file if you ever need to.",

"nosr": "Skip the AI upscaler and use lanczos instead.\n\n"
        "Lanczos is an ordinary maths based resize. It is far faster, "
        "measured in minutes rather than hours, and noticeably softer. "
        "Useful for a quick pass or when the GPU is needed elsewhere.\n\n"
        "This box starts from no_sr in av1_pipeline_config.txt every "
        "time the window opens, and a change here applies to this run "
        "only. To change it for good, edit the config file.",

"nosubs": "Skip making subtitles with whisper.\n\n"
          "Normally, when a wanted language has no subtitle track, the "
          "speech in the file's own audio is transcribed to make one. "
          "Because the timings come from that audio it is always in "
          "sync, but it is slow: roughly a fifth of the running time of "
          "the file, on top of the encode.\n\n"
          "Ticking this skips that step entirely. Subtitle tracks "
          "already in the source are still carried across untouched.",

"sdrhdr": "SDR means standard dynamic range, HDR means high dynamic "
          "range. This expands an SDR source into HDR10 colour.\n\n"
          "It is a mathematical expansion, not real HDR mastering. It "
          "changes the look of every SDR file in the batch, so try it "
          "on one file with Limit set to 1 before running it on many.\n\n"
          "Needs a working Vulkan driver. If the startup check finds it "
          "will not run, the option is switched off for that run and "
          "the log says so.\n\n"
          "This box starts from sdr_to_hdr in av1_pipeline_config.txt "
          "every time the window opens, and a change here applies to "
          "this run only. To change it for good, edit the config file.",

"targetres": "How many lines of actual picture to aim for. Bars are "
             "never counted, so a letterboxed film is measured on the "
             "picture inside them.\n\n"
             "1080p is the standard and the default.\n\n"
             "4K only produces a 4K master when the source is already "
             "close to 4K. A DVD or a 1080p Blu-ray in a 4K batch is "
             "still finished at 1080p - the AI upscaler never aims "
             "above 1080 lines, because inventing 4K detail from a "
             "smaller source bakes in artifacts permanently.\n\n"
             "A 4K source whose picture is within 10 percent of 2160 "
             "lines is brought up to meet it. A wider film, such as a "
             "2.39:1 scope title at 1608 lines, keeps its own pixels "
             "rather than being stretched.\n\n"
             "4K masters aim to stay under 10 GB, but picture quality "
             "wins: if holding the quality floor means going over, the "
             "file goes over and the log says so.\n\n"
             "This box starts from target_res in av1_pipeline_config.txt "
             "every time the window opens, and a change here applies to "
             "this run only. To change it for good, edit the config file.",

"start": "Begin the run.\n\n"
         "Largest file goes first, on purpose: big files take longest "
         "and tell you most about whether the settings are right.",

"stop": "Stop after the current file finishes what it is doing.\n\n"
        "Nothing is lost. The journal remembers what completed, so "
        "pressing Start again picks up where this left off.",

"selftest": "Run the program's internal checks.\n\n"
            "Touches no files and needs no GPU. Takes about a second. "
            "Good first thing to press after installing or updating.",

"opendst": "Open the destination folder in Explorer.",

"opentrace": "Open av1_trace.log, the deep debug log.\n\n"
             "This is the file to send when something goes wrong. It "
             "holds every command that was run and the errors they "
             "produced.",

"log": "Live output from the run.\n\n"
       "CQ is constant quality, the encoder's quality dial, roughly 16 "
       "to 30 where lower means bigger and better. bpp is bits per "
       "pixel, used to judge how good the source is. NVENC is the "
       "encoding chip on the NVIDIA card. VMAF is a quality score used "
       "to check a re-encode did not lose anything.",
}


class App:
    def __init__(self, root, smoke=False):
        self.root = root
        self.smoke = smoke
        self.smoke_ok = False
        self.p = None
        self.q = queue.Queue()
        self.cur = ""
        root.title(APP_TITLE)
        root.geometry("900x620")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        frm = ttk.Frame(root, padding=8)
        frm.pack(fill="both", expand=True)

        self.src = tk.StringVar()
        self.dst = tk.StringVar()
        self.analyze = tk.BooleanVar(value=False)
        self.retry = tk.BooleanVar(value=False)
        self.no_sr = tk.BooleanVar(value=False)
        self.sdr_to_hdr = tk.BooleanVar(value=False)
        self.no_subs = tk.BooleanVar(value=False)
        self.force = tk.BooleanVar(value=False)
        self.limit = tk.StringVar(value="0")
        self.minfree = tk.StringVar(value="20")
        self.sr_model = tk.StringVar(value=DEFAULT_SR_MODEL)
        self.encoder = tk.StringVar(value=DEFAULT_ENCODER)
        self.sr_strength = tk.StringVar(value="50")
        self.audio_mode = tk.StringVar(value=DEFAULT_AUDIO_MODE)
        self.target_res = tk.StringVar(value=DEFAULT_TARGET_RES)
        self.log_mode = tk.StringVar(value=LOG_MODES[0])
        self.log_custom = tk.StringVar()

        # row 0: source picker
        ttk.Label(frm, text="Source").grid(row=0, column=0, sticky="w")
        e_src = ttk.Entry(frm, textvariable=self.src)
        e_src.grid(row=0, column=1, sticky="ew", padx=4)
        tip(e_src, TIPS["src"])
        ttk.Button(frm, text="Browse",
                   command=self.pick_src).grid(row=0, column=2)

        # row 1: destination picker
        ttk.Label(frm, text="Destination").grid(row=1, column=0, sticky="w")
        e_dst = ttk.Entry(frm, textvariable=self.dst)
        e_dst.grid(row=1, column=1, sticky="ew", padx=4)
        tip(e_dst, TIPS["dst"])
        ttk.Button(frm, text="Browse",
                   command=self.pick_dst).grid(row=1, column=2)

        # row 2: log folder selector
        ttk.Label(frm, text="Log folder").grid(row=2, column=0, sticky="w")
        logrow = ttk.Frame(frm)
        logrow.grid(row=2, column=1, columnspan=2, sticky="ew", padx=4)
        c_log = ttk.Combobox(logrow, textvariable=self.log_mode, width=22,
                             values=LOG_MODES, state="readonly")
        c_log.pack(side="left")
        c_log.bind("<<ComboboxSelected>>", self.on_log_mode)
        tip(c_log, TIPS["logdir"])
        self.e_logcustom = ttk.Entry(logrow, textvariable=self.log_custom,
                                     state="disabled")
        self.e_logcustom.pack(side="left", fill="x", expand=True, padx=4)
        tip(self.e_logcustom, TIPS["logdir"])
        self.b_logbrowse = ttk.Button(logrow, text="Browse",
                                      command=self.pick_log,
                                      state="disabled")
        self.b_logbrowse.pack(side="left")

        # row 3: flags
        opts = ttk.Frame(frm)
        opts.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        tip(ttk.Checkbutton(opts, text="Analyze only",
                            variable=self.analyze), TIPS["analyze"]
            ).pack(side="left")
        tip(ttk.Checkbutton(opts, text="Retry failed",
                            variable=self.retry), TIPS["retry"]
            ).pack(side="left", padx=8)
        tip(ttk.Checkbutton(opts, text="Force redo",
                            variable=self.force), TIPS["force"]
            ).pack(side="left", padx=8)
        tip(ttk.Label(opts, text="Limit (0 = all)"),
            TIPS["limit"]).pack(side="left")
        tip(ttk.Spinbox(opts, from_=0, to=99999, width=6,
                        textvariable=self.limit), TIPS["limit"]
            ).pack(side="left", padx=4)
        tip(ttk.Label(opts, text="Min free GB"),
            TIPS["minfree"]).pack(side="left")
        tip(ttk.Spinbox(opts, from_=1, to=5000, width=6,
                        textvariable=self.minfree), TIPS["minfree"]
            ).pack(side="left", padx=4)

        # row 4: upscale controls
        sr_row = ttk.Frame(frm)
        sr_row.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        tip(ttk.Label(sr_row, text="Target"),
            TIPS["targetres"]).pack(side="left")
        tip(ttk.Combobox(sr_row, textvariable=self.target_res, width=7,
                         values=TARGET_RES, state="readonly"),
            TIPS["targetres"]).pack(side="left", padx=(4, 12))
        tip(ttk.Label(sr_row, text="Audio"),
            TIPS["audio"]).pack(side="left")
        tip(ttk.Combobox(sr_row, textvariable=self.audio_mode, width=7,
                         values=AUDIO_MODES, state="readonly"),
            TIPS["audio"]).pack(side="left", padx=(4, 12))
        tip(ttk.Label(sr_row, text="Encoder"),
            TIPS["encoder"]).pack(side="left")
        tip(ttk.Combobox(sr_row, textvariable=self.encoder, width=8,
                         values=ENCODERS, state="readonly"),
            TIPS["encoder"]).pack(side="left", padx=4)
        tip(ttk.Label(sr_row, text="SR model"),
            TIPS["srmodel"]).pack(side="left", padx=(12, 0))
        tip(ttk.Combobox(sr_row, textvariable=self.sr_model, width=26,
                         values=SR_MODELS, state="readonly"),
            TIPS["srmodel"]).pack(side="left", padx=4)
        tip(ttk.Label(sr_row, text="AI strength %"),
            TIPS["srstrength"]).pack(side="left", padx=(12, 0))
        tip(ttk.Spinbox(sr_row, from_=0, to=100, increment=5, width=5,
                        textvariable=self.sr_strength),
            TIPS["srstrength"]).pack(side="left", padx=4)

        # row 5: the switches that turn whole stages off. on their own
        # line because each one changes what the run actually does, not
        # merely how it does it, so they want to be read as a group.
        skips = ttk.Frame(frm)
        skips.grid(row=5, column=0, columnspan=3, sticky="w", pady=2)
        tip(ttk.Checkbutton(skips, text="Skip SR (lanczos fallback)",
                            variable=self.no_sr), TIPS["nosr"]
            ).pack(side="left")
        tip(ttk.Checkbutton(skips, text="SDR to HDR",
                            variable=self.sdr_to_hdr), TIPS["sdrhdr"]
            ).pack(side="left", padx=12)
        tip(ttk.Checkbutton(skips, text="Skip generate subtitles",
                            variable=self.no_subs), TIPS["nosubs"]
            ).pack(side="left", padx=12)

        # row 6: run controls
        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=3, sticky="w", pady=4)
        self.b_start = ttk.Button(btns, text="Start", command=self.start)
        self.b_start.pack(side="left")
        tip(self.b_start, TIPS["start"])
        self.b_stop = ttk.Button(btns, text="Stop", command=self.stop,
                                 state="disabled")
        self.b_stop.pack(side="left", padx=6)
        tip(self.b_stop, TIPS["stop"])
        tip(ttk.Button(btns, text="Selftest",
                       command=lambda: self.start(extra=["--selftest"])),
            TIPS["selftest"]).pack(side="left", padx=6)
        tip(ttk.Button(btns, text="Open destination",
                       command=self.open_dst),
            TIPS["opendst"]).pack(side="left", padx=6)
        tip(ttk.Button(btns, text="Open trace log",
                       command=self.open_trace),
            TIPS["opentrace"]).pack(side="left", padx=6)

        # row 7: one line status
        self.status = tk.StringVar(value="idle")
        ttk.Label(frm, textvariable=self.status,
                  anchor="w").grid(row=7, column=0, columnspan=3, sticky="ew")

        # row 8: log pane
        self.log = tk.Text(frm, wrap="none", state="disabled", height=20)
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew")
        tip(self.log, TIPS["log"])
        sb = ttk.Scrollbar(frm, orient="vertical", command=self.log.yview)
        sb.grid(row=8, column=3, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(8, weight=1)

        self.load_settings()
        root.after(120, self.poll)
        if smoke:
            # scripted analyze pass for automated checks, no dialogs
            self.src.set(os.environ.get("AV1GUI_SMOKE_SRC", ""))
            self.dst.set(os.environ.get("AV1GUI_SMOKE_DST", ""))
            self.analyze.set(True)
            self.minfree.set("1")
            root.after(400, self.start)
            root.after(180000, self.smoke_timeout)

    # ------------------------------------------------------------ helpers
    def running(self):
        return self.p is not None and self.p.poll() is None

    def err(self, msg):
        if self.smoke:
            print("SMOKE-ERR " + msg, file=sys.stderr)
            self.root.destroy()
            return
        messagebox.showerror(APP_TITLE, msg)

    def append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        # cap the pane so day-long runs never eat memory
        n = int(self.log.index("end-1c").split(".")[0])
        if n > LOG_CAP_LINES:
            self.log.delete("1.0", "%d.0" % (n - LOG_CAP_LINES))
        self.log.see("end")
        self.log.configure(state="disabled")

    def parse(self, line):
        m = RE_FILE.search(line)
        if m:
            self.cur = "file %s/%s  %s" % (m.group(1), m.group(2), m.group(3))
            self.status.set(self.cur)
        m = RE_PROG.search(line)
        if m:
            self.status.set("%s  |  %s  %s pct  %s"
                            % (self.cur, m.group(1), m.group(2), m.group(3)))
        m = RE_SR.search(line)
        if m:
            self.status.set("%s  |  sr chunk %s/%s  %sx"
                            % (self.cur, m.group(1), m.group(2), m.group(3)))
        if RE_DONE.search(line):
            self.smoke_ok = True

    # ------------------------------------------------------------ actions
    def log_dir_value(self):
        # blank tells the pipeline to use its own default, a logs folder
        # under the destination, so only the other two modes send a path
        mode = self.log_mode.get()
        if mode == LOG_MODES[1]:
            return str(HERE)
        if mode == LOG_MODES[2]:
            return self.log_custom.get().strip()
        return ""

    def on_log_mode(self, _=None):
        custom = self.log_mode.get() == LOG_MODES[2]
        self.e_logcustom.configure(state="normal" if custom else "disabled")
        self.b_logbrowse.configure(state="normal" if custom else "disabled")

    def pick_log(self):
        d = filedialog.askdirectory(title="Pick the log folder")
        if d:
            self.log_custom.set(d)

    def pick_src(self):
        d = filedialog.askdirectory(title="Pick the SOURCE folder")
        if d:
            self.src.set(d)

    def pick_dst(self):
        d = filedialog.askdirectory(title="Pick the DESTINATION folder")
        if d:
            self.dst.set(d)

    def start(self, extra=None):
        if self.running():
            return
        if not PIPELINE.exists():
            self.err("av1_pipeline_v0_1.py is not next to this GUI file")
            return
        if extra is None:
            s = {"src": self.src.get().strip(), "dst": self.dst.get().strip(),
                 "analyze": self.analyze.get(), "retry": self.retry.get(),
                 "no_sr": self.no_sr.get(), "sdr_to_hdr": self.sdr_to_hdr.get(),
                 "no_subs": self.no_subs.get(),
                 "target_res": self.target_res.get(),
                 "force": self.force.get(), "sr_model": self.sr_model.get(),
                 "limit": self.limit.get(), "minfree": self.minfree.get(),
                 "log_dir": self.log_dir_value(),
                 "encoder": self.encoder.get(),
                 "sr_strength": self.sr_strength.get(),
                 "audio_mode": self.audio_mode.get()}
            if self.log_mode.get() == LOG_MODES[2] and not s["log_dir"]:
                self.err("pick a log folder, or switch back to Output folder")
                return
            if not s["src"] or not Path(s["src"]).is_dir():
                self.err("pick a source folder that exists")
                return
            if not s["dst"]:
                self.err("pick a destination folder")
                return
            try:
                cmd = build_args(sys.executable, PIPELINE, s)
            except (ValueError, TypeError):
                self.err("limit and min free GB need plain numbers")
                return
            self.save_settings()
        else:
            cmd = [str(sys.executable), "-u", str(PIPELINE)] + extra
        flags = NOWIN
        try:
            self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace",
                                      creationflags=flags, cwd=str(HERE))
        except Exception as e:
            self.err("could not start the pipeline: %s" % e)
            return
        self.append(">>> " + " ".join(cmd))
        self.status.set("running")
        self.b_start.configure(state="disabled")
        self.b_stop.configure(state="normal")
        threading.Thread(target=self.reader, daemon=True).start()

    def stop(self):
        if not self.running():
            return
        if not self.smoke and not messagebox.askyesno(
                APP_TITLE, "Stop the run? The journal resumes it later."):
            return
        try:
            if os.name == "nt":
                # tree kill takes ffmpeg down with the pipeline; the
                # journal plus the .tmp sweep make this safe
                subprocess.run(["taskkill", "/PID", str(self.p.pid),
                                "/T", "/F"], capture_output=True,
                               creationflags=NOWIN)
            else:
                self.p.terminate()
        except Exception:
            pass

    def reader(self):
        try:
            for line in self.p.stdout:
                self.q.put(line.rstrip("\n"))
        except Exception:
            pass
        self.p.wait()
        self.q.put(None)

    def poll(self):
        try:
            while True:
                line = self.q.get_nowait()
                if line is None:
                    self.finish()
                else:
                    self.append(line)
                    self.parse(line)
        except queue.Empty:
            pass
        self.root.after(120, self.poll)

    def finish(self):
        rc = self.p.poll() if self.p else None
        self.status.set("finished (exit %s)" % rc)
        self.b_start.configure(state="normal")
        self.b_stop.configure(state="disabled")
        if self.smoke:
            self.root.after(300, self.root.destroy)

    def smoke_timeout(self):
        # hard ceiling so an automated check can never hang
        if self.running():
            self.stop()
        self.root.destroy()

    def open_dst(self):
        d = self.dst.get().strip()
        if d and Path(d).is_dir() and hasattr(os, "startfile"):
            os.startfile(d)

    def open_trace(self):
        # the trace file now follows the log folder selector, so look
        # where the current setting says before falling back
        picked = self.log_dir_value().strip()
        cands = []
        if picked:
            cands.append(Path(picked) / "av1_trace.log")
        else:
            d = self.dst.get().strip()
            if d:
                cands.append(Path(d) / "logs" / "av1_trace.log")
        cands.append(HERE / "av1_trace.log")
        t = next((c for c in cands if c.exists()), None)
        if t and hasattr(os, "startfile"):
            os.startfile(str(t))
        elif t:
            self.err("trace log at: %s" % t)
        else:
            self.err("no trace log yet; run the pipeline once first")

    # ------------------------------------------------------------ settings
    def load_settings(self):
        """Populate the controls from config, then from saved state.

        Two layers, applied in order: av1_pipeline_config.txt provides
        the baseline that the pipeline itself will use, then whatever
        the window was last left showing is layered on top so the
        operator's per-run choices survive a restart.

        The exceptions are documented inline below - two settings are
        read from the config only, because letting them persist caused
        real damage to finished files.

        Deleting gui_settings.json returns everything to config values.
        """
        cfg = read_config()
        self.src.set(cfg.get("src", ""))
        self.dst.set(cfg.get("dst", ""))
        self.minfree.set(cfg.get("min_free_gb", "20"))
        model = cfg.get("sr_model", DEFAULT_SR_MODEL)
        self.sr_model.set(model if model in SR_MODELS else DEFAULT_SR_MODEL)
        enc = (cfg.get("encoder") or DEFAULT_ENCODER).strip().lower()
        self.encoder.set(enc if enc in ENCODERS else DEFAULT_ENCODER)
        self.sr_strength.set(str(cfg.get("sr_strength", "50")))
        am = (cfg.get("audio_mode") or DEFAULT_AUDIO_MODE).strip().lower()
        self.audio_mode.set(am if am in AUDIO_MODES else DEFAULT_AUDIO_MODE)
        # no_sr and sdr_to_hdr are library-wide policy rather than
        # per-run choices, so av1_pipeline_config.txt is their only
        # home. They are read here and deliberately NOT read back from
        # gui_settings.json further down.
        #
        # The reasoning is empirical: that file is rewritten every time
        # the window closes, so a box ticked once for a single file
        # silently became the standing setting. It shadowed the config
        # three separate times, on one occasion expanding an entire
        # batch of SDR films to HDR10 after the config had been set
        # explicitly not to. Settings the operator owns per run stay
        # saved; settings that define the library do not.
        self.no_sr.set(as_bool(cfg.get("no_sr")))
        self.sdr_to_hdr.set(as_bool(cfg.get("sdr_to_hdr")))
        # target_res joins them for the same reason. Which resolution the
        # library is mastered at is not a per-run preference, and a 4K
        # batch left selected would quietly upgrade every later run.
        tr = re.sub(r"[^0-9]", "", str(cfg.get("target_res", "")))
        self.target_res.set("4K" if tr == "2160" else DEFAULT_TARGET_RES)
        # the config key is the positive form, the checkbox the negative
        self.no_subs.set(not as_bool(cfg.get("subs_generate"), True))
        # a log_dir in the config shows up as Custom unless it is blank
        cfg_log = (cfg.get("log_dir") or "").strip()
        if cfg_log:
            self.log_mode.set(LOG_MODES[2])
            self.log_custom.set(cfg_log)
        self.on_log_mode()
        try:
            s = json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return
        try:
            if s.get("src"):
                self.src.set(s["src"])
            if s.get("dst"):
                self.dst.set(s["dst"])
            self.limit.set(str(s.get("limit", "0")))
            self.minfree.set(str(s.get("minfree", self.minfree.get())))
            saved = s.get("sr_model", self.sr_model.get())
            self.sr_model.set(saved if saved in SR_MODELS
                              else DEFAULT_SR_MODEL)
            # no_sr, sdr_to_hdr and target_res are deliberately absent
            # here. they come from the config file and nowhere else, so
            # ticking a box for one run can never quietly become the
            # standing setting for the whole library.
            self.no_subs.set(s.get("no_subs", self.no_subs.get()))
            self.sr_strength.set(str(s.get("sr_strength",
                                          self.sr_strength.get())))
            sa = s.get("audio_mode", self.audio_mode.get())
            self.audio_mode.set(sa if sa in AUDIO_MODES
                                else DEFAULT_AUDIO_MODE)
            se = s.get("encoder", self.encoder.get())
            self.encoder.set(se if se in ENCODERS else DEFAULT_ENCODER)
            mode = s.get("log_mode", self.log_mode.get())
            self.log_mode.set(mode if mode in LOG_MODES else LOG_MODES[0])
            self.log_custom.set(s.get("log_custom", self.log_custom.get()))
            self.on_log_mode()
        except Exception:
            pass

    def save_settings(self):
        """Persist the operator's per-run choices to gui_settings.json.

        Note what is absent: no_sr, sdr_to_hdr and target_res are not
        written. All three are library-wide policy owned by
        av1_pipeline_config.txt, and persisting them here is what
        allowed a box ticked for one file to quietly govern every
        subsequent batch.

        Failures are ignored deliberately. Losing the remembered window
        state is a minor inconvenience; refusing to close, or throwing
        during shutdown, is not.
        """
        try:
            SETTINGS.write_text(json.dumps(
                {"src": self.src.get(), "dst": self.dst.get(),
                 "limit": self.limit.get(), "minfree": self.minfree.get(),
                 "sr_model": self.sr_model.get(),
                 # no_sr, sdr_to_hdr and target_res are NOT saved. they
                 # belong to av1_pipeline_config.txt; writing them here
                 # is what let a one-off tick outlive the run that
                 # wanted it.
                 "no_subs": self.no_subs.get(),
                 "encoder": self.encoder.get(),
                 "sr_strength": self.sr_strength.get(),
                 "audio_mode": self.audio_mode.get(),
                 "log_mode": self.log_mode.get(),
                 "log_custom": self.log_custom.get()},
                indent=1), encoding="utf-8")
        except Exception:
            pass

    def on_close(self):
        if self.running():
            if self.smoke or messagebox.askyesno(
                    APP_TITLE,
                    "A run is active. Stop it and close? "
                    "The journal resumes it later."):
                self.stop()
            else:
                return
        self.save_settings()
        self.root.destroy()


def main():
    smoke = "--smoke" in sys.argv
    root = tk.Tk()
    app = App(root, smoke=smoke)
    root.mainloop()
    if smoke:
        return 0 if app.smoke_ok else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
