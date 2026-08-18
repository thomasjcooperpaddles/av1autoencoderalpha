#!/usr/bin/env python3
# =====================================================================
# av1_metadata_tool.py   V0.246   2026-08-15
#
# Standalone metadata scanner. Walks a folder tree, identifies every
# video in it, and writes a .info sidecar beside each one.
#
# METADATA ONLY. It never touches the video, never re-encodes, never
# remuxes, and by default never renames anything. The only files it
# creates are the sidecars.
#
# It is a front end for av1_metadata.py, which is the same module the
# pipeline uses and which has been in production for weeks. Nothing
# about identification, TMDB lookup or frame verification is
# reimplemented here - a second copy of that logic would drift from the
# first, and the whole value of this tool is that it agrees with the
# pipeline.
#
# The evidence ladder, cheapest first, stopping as soon as one answers:
#
#   1. an id already embedded in the container (IMDB/TMDB tag). An
#      explicit id involves no guessing and beats everything below it.
#   2. the folder structure and filename, read by guessit - which uses
#      the WHOLE relative path, so "Show/Season 02/07.mkv" resolves.
#   3. TMDB search, confirmed against the file's actual runtime.
#   4. SCREENSHOTS. Frames sampled from the file are compared against
#      TMDB's backdrops and episode stills by perceptual hash. This is
#      the tie-breaker for when the text evidence conflicts, and it is
#      the only check that looks at what is actually in the video.
#
# Needs: python 3, tkinter, ffmpeg/ffprobe on PATH, av1_metadata.py
# beside it. guessit is optional but strongly recommended.
# =====================================================================
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

VERSION = "0.246"
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".wmv", ".mpg",
              ".mpeg", ".m2ts", ".ts", ".vob", ".flv", ".webm", ".divx",
              ".mts", ".3gp", ".ogm", ".rm", ".rmvb", ".asf"}
MIN_FILE_BYTES = 5 * 1024 * 1024

CONFIG_NAME = "av1_pipeline_config.txt"


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def which(name):
    from shutil import which as _w
    p = _w(name)
    if p:
        return p
    for d in (r"C:\Program Files\MKVToolNix",):
        c = Path(d) / (name + ".exe")
        if c.exists():
            return str(c)
    return ""


def read_config_key(key):
    """Pull one value out of the pipeline's config, if it is beside us.

    The TMDB key in particular: an operator who already has the pipeline
    configured should not have to type it in again here.
    """
    p = Path(__file__).resolve().parent / CONFIG_NAME
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip().lower() == key:
                    return v.strip()
    except OSError:
        pass
    return ""


def frac(v, default=0.0):
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if "/" in s:
            a, b = s.split("/", 1)
            b = float(b)
            return float(a) / b if b else default
        return float(s)
    except Exception:
        return default


def probe(ffprobe, path):
    """Just enough of the file for identification and the sidecar.

    Deliberately a light probe rather than importing the pipeline's
    probe_file: that would drag the whole encoder in for a tool whose
    entire purpose is to not be the encoder.
    """
    cmd = [ffprobe, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300,
                       creationflags=NOWIN)
    if p.returncode != 0 or not (p.stdout or "").strip():
        raise RuntimeError("ffprobe failed: %s" % (p.stderr or "")[:200])
    j = json.loads(p.stdout)
    fmt = j.get("format") or {}
    streams = j.get("streams") or []
    v = next((s for s in streams
              if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")), None)
    if v is None:
        raise RuntimeError("no video stream")
    dur = frac(fmt.get("duration")) or frac(v.get("duration"))
    fps = frac(v.get("avg_frame_rate")) or frac(v.get("r_frame_rate")) or 24.0
    pix = v.get("pix_fmt") or ""
    try:
        depth = int(v.get("bits_per_raw_sample") or 0)
    except Exception:
        depth = 0
    if depth <= 0:
        depth = 10 if "10" in pix else (12 if "12" in pix else 8)
    trc = (v.get("color_transfer") or "").lower()
    dovi = None
    for sd in (v.get("side_data_list") or []):
        if "DOVI" in (sd.get("side_data_type") or ""):
            dovi = sd.get("dv_profile")
    hdr = ("dv5" if dovi == 5 else "dv78" if dovi is not None
           else "hdr10" if trc == "smpte2084"
           else "hlg" if trc == "arib-std-b67" else "sdr")
    audios, subs = [], []
    for s in streams:
        ct = s.get("codec_type")
        tags = s.get("tags") or {}
        if ct == "audio":
            audios.append({"idx": s.get("index"),
                           "lang": (tags.get("language") or "und").lower(),
                           "ch": int(s.get("channels") or 2),
                           "codec": s.get("codec_name") or "",
                           "title": tags.get("title") or ""})
        elif ct == "subtitle":
            subs.append({"idx": s.get("index"),
                         "lang": (tags.get("language") or "und").lower(),
                         "codec": s.get("codec_name") or "",
                         "title": tags.get("title") or ""})
    return {"path": str(path), "dur": dur, "fps": fps,
            "w": int(v.get("width") or 0), "h": int(v.get("height") or 0),
            "depth": depth, "hdr": hdr,
            "vcodec": (v.get("codec_name") or "").lower(),
            "audios": audios, "subs": subs,
            "tags": fmt.get("tags") or {},
            "size": int(frac(fmt.get("size")) or 0)}


# ---------------------------------------------------------------------
# the actual work, one file at a time
# ---------------------------------------------------------------------
class Scanner(object):
    """Identify one file and write its sidecar.

    Kept separate from the window so the decision logic can be read -
    and tested - without a display.
    """

    def __init__(self, MD, ffmpeg, ffprobe, api_key, opts, emit):
        self.MD = MD
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.api_key = api_key
        self.opts = opts
        self.emit = emit          # emit(level, message)

    class _Log(object):
        """av1_metadata expects a logger; give it one that reports here."""

        def __init__(self, emit):
            self._emit = emit

        def info(self, m, *a):
            self._emit("info", "      " + (m % a if a else m))

        def warning(self, m, *a):
            self._emit("warn", "      " + (m % a if a else m))

        def debug(self, m, *a):
            pass

        def error(self, m, *a):
            self._emit("error", "      " + (m % a if a else m))

    def conflict_reasons(self, meta, parsed, info):
        """Why the text evidence should not be trusted on its own.

        This is what decides whether frames get sampled. Each reason is
        a genuine disagreement between two independent signals, not a
        measure of how much the code likes the answer.
        """
        why = []
        if not meta:
            return ["nothing matched in TMDB"]
        tm_title = (meta.get("title") or meta.get("name") or "").lower()
        gs_title = (parsed.get("title") or "").lower()
        if gs_title and tm_title and gs_title not in tm_title \
                and tm_title not in gs_title:
            why.append("the filename says %r, TMDB says %r"
                       % (parsed.get("title"), meta.get("title")
                          or meta.get("name")))
        date = (meta.get("release_date") or meta.get("first_air_date") or "")
        ty = date[:4]
        py = str(parsed.get("year") or "")
        if ty and py and ty != py:
            why.append("year %s in the name against %s in TMDB" % (py, ty))
        runtime = meta.get("runtime") or 0
        if runtime and info.get("dur"):
            if not self.MD.runtime_agrees(info["dur"], runtime):
                why.append("runtime %.0f min against TMDB's %s min"
                           % (info["dur"] / 60.0, runtime))
        if not py and parsed.get("kind") == "movie":
            why.append("no year in the filename to confirm against")
        return why

    def run_one(self, path, root):
        """Returns a result dict for the table."""
        rel = str(Path(path).relative_to(root)).replace("\\", "/")
        res = {"rel": rel, "title": "", "kind": "", "source": "",
               "verified": "", "info": "", "note": ""}
        log = self._Log(self.emit)
        sidecar = Path(path).with_suffix(".info")
        if sidecar.exists() and not self.opts["overwrite"]:
            res["info"] = "exists"
            res["note"] = "skipped, sidecar already there"
            return res

        info = probe(self.ffprobe, path)
        parsed = self.MD.parse_path(rel)
        franchise = self.MD.find_franchise(rel)
        res["kind"] = parsed.get("kind") or ""

        # 1. an id the file already carries beats every other signal
        ids = self.MD.embedded_ids(info)
        if ids:
            self.emit("info", "      embedded id: %s"
                      % ", ".join("%s=%s" % kv for kv in sorted(ids.items())))

        meta = None
        if self.api_key:
            meta = self.MD.tmdb_lookup(self.api_key, parsed, info.get("dur"),
                                       ids=ids, log=log)
            if meta:
                try:
                    meta = self.MD.enrich(self.api_key, meta, parsed,
                                          log=log) or meta
                except Exception as e:
                    self.emit("warn", "      enrich failed: %s" % e)
        else:
            res["note"] = "no TMDB key: name and container tags only"

        if meta:
            res["title"] = (meta.get("title") or meta.get("name") or "")
            res["source"] = "embedded id" if ids else "TMDB search"
        else:
            res["title"] = parsed.get("title") or Path(path).stem
            res["source"] = "filename"

        # 2. screenshots, when the text evidence disagrees with itself
        verify = None
        why = self.conflict_reasons(meta, parsed, info)
        mode = self.opts["verify"]
        do_verify = (mode == "always" or (mode == "conflict" and why))
        if do_verify and not self.api_key:
            do_verify = False
            self.emit("warn", "      cannot verify by frames without a "
                              "TMDB key to fetch reference images")
        if why:
            for w in why:
                self.emit("warn", "      conflict: %s" % w)
        if do_verify and meta:
            try:
                urls = self.MD.tmdb_images(self.api_key, meta, parsed,
                                           log=log)
                if urls:
                    self.emit("info", "      comparing %d frames against "
                                      "%d reference image(s)"
                              % (self.opts["samples"], len(urls)))
                    verify = self.MD.verify_by_frames(
                        self.ffmpeg, path, urls, info.get("dur"),
                        self.opts["tmp"], log=log,
                        samples=self.opts["samples"])
                else:
                    self.emit("warn", "      TMDB has no reference images "
                                      "for this title")
            except Exception as e:
                self.emit("warn", "      frame check failed: %s" % e)
        if verify:
            # The real contract, read out of av1_metadata rather than
            # assumed: {"verdict", "best_distance", "frames", "images",
            # "matched_image"}. verdict is "confirmed" / "unverified" /
            # "not_checked", and best_distance is a Hamming distance
            # over a 64-bit difference hash, so 0 is identical and 32 is
            # unrelated.
            verdict = verify.get("verdict") or "not_checked"
            dist = verify.get("best_distance")
            label = {"confirmed": "confirmed",
                     "unverified": "inconclusive",
                     "not_checked": "not checked"}.get(verdict, verdict)
            if dist is not None:
                label += " (d=%s of %d frames)" % (dist,
                                                   verify.get("frames") or 0)
            res["verified"] = label
            if verdict == "confirmed":
                self.emit("info", "      pictures confirm the match")
            elif why:
                self.emit("warn", "      the pictures did NOT confirm it; "
                                  "treat this identification as a guess")
        elif why:
            res["verified"] = "conflict, unverified"
        else:
            res["verified"] = "not needed"

        if self.opts["dry"]:
            res["info"] = "would write"
            return res

        plan = {"cq": None, "notes": ["metadata_only_scan"],
                "out_space": "", "vf": ""}
        # write_info's FIRST ARGUMENT IS THE .INFO FILE, not the video.
        # Passing the video path here overwrites the film with text -
        # found by testing this on a throwaway clip rather than on the
        # library, and the reason the sidecar is built with with_suffix
        # and then checked before this returns "written".
        wrote = self.MD.write_info(sidecar, meta, parsed, franchise, info,
                                   plan, verify=verify,
                                   source_name=Path(path).name,
                                   out_name=Path(path).name, log=log,
                                   info_is_source=True)
        if wrote is False or not sidecar.exists():
            res["info"] = "NOT written"
            res["note"] = "write_info reported no file"
        else:
            res["info"] = "written"
        return res


# ---------------------------------------------------------------------
# window
# ---------------------------------------------------------------------
class App(object):
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()
        self.rows = []
        root.title("AV1 Metadata Scanner  %s" % VERSION)
        root.geometry("1080x740")
        self.build()
        self.root.after(120, self.pump)

    def build(self):
        pad = {"padx": 6, "pady": 3}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Folder").grid(row=0, column=0, sticky="w")
        self.folder = tk.StringVar()
        ttk.Entry(top, textvariable=self.folder, width=80).grid(
            row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.pick).grid(row=0,
                                                               column=2)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="TMDB key").grid(row=1, column=0, sticky="w")
        self.key = tk.StringVar(value=read_config_key("tmdb_api_key"))
        ttk.Entry(top, textvariable=self.key, width=44).grid(
            row=1, column=1, sticky="w", padx=4)

        opts = ttk.LabelFrame(self.root, text="What to do")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Verify with screenshots:").grid(
            row=0, column=0, sticky="w", padx=6, pady=4)
        self.verify = tk.StringVar(value="conflict")
        for n, (val, lbl) in enumerate((
                ("conflict", "only when the evidence conflicts"),
                ("always", "always"),
                ("never", "never"))):
            ttk.Radiobutton(opts, text=lbl, value=val,
                            variable=self.verify).grid(row=0, column=1 + n,
                                                       sticky="w", padx=6)

        self.overwrite = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Overwrite sidecars that already exist",
                        variable=self.overwrite).grid(row=1, column=0,
                                                      columnspan=2,
                                                      sticky="w", padx=6)
        self.dry = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Dry run - identify but write nothing",
                        variable=self.dry).grid(row=1, column=2,
                                                columnspan=2, sticky="w")
        ttk.Label(opts,
                  text="This tool never renames, re-encodes or remuxes. "
                       "The only files it creates are .info sidecars.",
                  foreground="#666").grid(row=2, column=0, columnspan=5,
                                          sticky="w", padx=6, pady=(0, 4))

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", **pad)
        self.b_scan = ttk.Button(bar, text="Scan folder",
                                 command=self.do_scan)
        self.b_scan.pack(side="left")
        self.b_run = ttk.Button(bar, text="Start", command=self.do_run,
                                state="disabled")
        self.b_run.pack(side="left", padx=6)
        self.b_stop = ttk.Button(bar, text="Stop", command=self.do_stop,
                                 state="disabled")
        self.b_stop.pack(side="left")
        self.status = tk.StringVar(value="Pick a folder.")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=12)

        self.prog = ttk.Progressbar(self.root, mode="determinate")
        self.prog.pack(fill="x", **pad)

        cols = ("file", "kind", "title", "source", "verified", "info")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings",
                                 height=14)
        widths = (430, 70, 210, 105, 150, 90)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=6)

        self.log = tk.Text(self.root, height=11, bg="#181818",
                           fg="#d7d7d7", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    def pick(self):
        d = filedialog.askdirectory(title="Folder to scan")
        if d:
            self.folder.set(d)

    def say(self, level, msg):
        self.q.put(("log", (level, msg)))

    # ---- scan ------------------------------------------------------
    def do_scan(self):
        root = self.folder.get().strip()
        if not root or not Path(root).is_dir():
            messagebox.showerror("AV1 Metadata Scanner",
                                 "Pick a folder that exists.")
            return
        self.tree.delete(*self.tree.get_children())
        self.log.delete("1.0", "end")
        self.files = []
        for p in Path(root).rglob("*"):
            try:
                if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                    continue
                if p.name.startswith("._"):
                    continue
                if p.stat().st_size < MIN_FILE_BYTES:
                    continue
            except OSError:
                continue
            self.files.append(p)
        self.files.sort()
        for p in self.files:
            rel = str(p.relative_to(root)).replace("\\", "/")
            has = p.with_suffix(".info").exists()
            self.tree.insert("", "end", values=(
                rel, "", "", "", "", "sidecar exists" if has else ""))
        n_has = sum(1 for p in self.files if p.with_suffix(".info").exists())
        self.status.set("%d video file(s); %d already have a sidecar"
                        % (len(self.files), n_has))
        self.say("info", "found %d video file(s) under %s"
                 % (len(self.files), root))
        self.b_run.configure(state="normal" if self.files else "disabled")

    # ---- run -------------------------------------------------------
    def do_run(self):
        if not self.files:
            return
        try:
            import av1_metadata as MD
        except ImportError as e:
            messagebox.showerror(
                "AV1 Metadata Scanner",
                "av1_metadata.py must sit beside this tool.\n\n%s" % e)
            return
        ffmpeg, ffprobe = which("ffmpeg"), which("ffprobe")
        if not ffmpeg or not ffprobe:
            messagebox.showerror("AV1 Metadata Scanner",
                                 "ffmpeg and ffprobe must be on PATH.")
            return
        import tempfile
        opts = {"verify": self.verify.get(),
                "overwrite": bool(self.overwrite.get()),
                "dry": bool(self.dry.get()),
                "samples": getattr(MD, "VERIFY_SAMPLES", 24),
                "tmp": Path(tempfile.gettempdir()) / "av1_meta_tool"}
        opts["tmp"].mkdir(parents=True, exist_ok=True)
        self.stop_flag.clear()
        self.b_run.configure(state="disabled")
        self.b_scan.configure(state="disabled")
        self.b_stop.configure(state="normal")
        self.prog.configure(maximum=len(self.files), value=0)
        self.worker = threading.Thread(
            target=self.work,
            args=(MD, ffmpeg, ffprobe, self.key.get().strip(), opts),
            daemon=True)
        self.worker.start()

    def work(self, MD, ffmpeg, ffprobe, key, opts):
        root = Path(self.folder.get().strip())
        sc = Scanner(MD, ffmpeg, ffprobe, key, opts, self.say)
        t0 = time.time()
        done = written = failed = 0
        for n, p in enumerate(self.files):
            if self.stop_flag.is_set():
                self.say("warn", "stopped by request")
                break
            rel = str(p.relative_to(root)).replace("\\", "/")
            self.say("info", "[%d/%d] %s" % (n + 1, len(self.files), rel))
            try:
                res = sc.run_one(p, root)
                if res["info"] == "written":
                    written += 1
                self.say("info", "      -> %s  [%s]"
                         % (res["title"] or "(unidentified)", res["source"]))
            except Exception as e:
                failed += 1
                res = {"rel": rel, "kind": "", "title": "", "source": "",
                       "verified": "", "info": "FAILED", "note": str(e)}
                self.say("error", "      failed: %s: %s"
                         % (type(e).__name__, e))
                self.say("error", traceback.format_exc().splitlines()[-1])
            self.q.put(("row", (n, res)))
            done += 1
            self.q.put(("prog", done))
        self.q.put(("done", (done, written, failed, time.time() - t0)))

    def do_stop(self):
        self.stop_flag.set()
        self.status.set("stopping...")

    # ---- ui pump ---------------------------------------------------
    def pump(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    level, msg = payload
                    self.log.insert("end", msg + "\n")
                    self.log.see("end")
                elif kind == "row":
                    n, res = payload
                    kids = self.tree.get_children()
                    if n < len(kids):
                        self.tree.item(kids[n], values=(
                            res["rel"], res["kind"], res["title"],
                            res["source"], res["verified"], res["info"]))
                        self.tree.see(kids[n])
                elif kind == "prog":
                    self.prog.configure(value=payload)
                elif kind == "done":
                    d, w, f, secs = payload
                    self.status.set(
                        "%d processed, %d sidecar(s) written, %d failed, "
                        "%.1f min" % (d, w, f, secs / 60.0))
                    self.say("info", "")
                    self.say("info", "done: %d processed, %d written, "
                                     "%d failed in %.1f min"
                             % (d, w, f, secs / 60.0))
                    self.b_run.configure(state="normal")
                    self.b_scan.configure(state="normal")
                    self.b_stop.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(120, self.pump)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
