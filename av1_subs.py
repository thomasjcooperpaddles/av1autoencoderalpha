#!/usr/bin/env python3
# ============================================================================
# av1_subs.py   V0.1   2026-08-02
#
# Subtitle acquisition for av1_pipeline.
#
# The ladder, cheapest and most accurate first. Each rung is only tried
# when the one above it produced nothing:
#
#   English
#     1. an English subtitle track already in the source
#     2. transcribe with Whisper, from a strongly dialogue-boosted mix
#        of the English audio. that boosted mix is also kept as an
#        output track, labelled "English enhanced".
#
#   Japanese
#     1. a Japanese subtitle track already in the source
#     2. transcribe the Japanese audio track directly, if one exists.
#        native speech beats translated text every time.
#     3. translate the English subtitles from above.
#
# Nothing here is allowed to fail an encode. Every entry point returns
# None on trouble and logs why.
# ============================================================================

import json
import logging
import os
import re
import subprocess
import urllib.parse
from pathlib import Path   # module level: three functions imported this
                           # locally and nothing else could see it, so
                           # code added later raised NameError and had it
                           # swallowed by a bare except

# See the note in av1_metadata.py. Without this every whisper pass and
# every audio extract flashes a console window over the desktop.
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
import urllib.request

# Moves with av1_pipeline_v0_1.VERSION, which selftest enforces.
VERSION = "0.256"

NET_TIMEOUT = 60
# Whisper needs a ggml model from whisper.cpp. Bigger is better and
# slower; medium is the usual sweet spot for film dialogue.
MODEL_NAMES = ("ggml-large-v3-turbo.bin", "ggml-medium.bin",
               "ggml-small.bin", "ggml-base.bin", "ggml-tiny.bin")
# Strong dialogue lift for the transcription mix. Far heavier than the
# listening downmix, because clarity matters more than balance here.
ENHANCE_CENTRE = 2.0
ENHANCE_DIALOGUE = 2.5


# ---------------------------------------------------------------------
# SRT handling
# ---------------------------------------------------------------------
def parse_srt(text):
    # returns [(index, timing_line, [text lines]), ...]
    out = []
    for block in re.split(r"\r?\n\r?\n", text.strip()):
        lines = [x for x in block.splitlines() if x.strip() != ""]
        if len(lines) < 2:
            continue
        idx = lines[0].strip()
        timing = lines[1].strip()
        if "-->" not in timing:
            # some files omit the index line
            if "-->" in idx:
                timing, body = idx, lines[1:]
                out.append((str(len(out) + 1), timing, body))
            continue
        out.append((idx, timing, lines[2:]))
    return out


def write_srt(cues, path):
    parts = []
    for n, (_, timing, body) in enumerate(cues, start=1):
        parts.append("%d\n%s\n%s\n" % (n, timing, "\n".join(body)))
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


SRT_TIME = re.compile(
    r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*"
    r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def _t2s(h, m, s, ms):
    """Convert an SRT timestamp's captured groups to seconds.

    The millisecond field is right-padded because SRT permits fewer than
    three digits: ",5" means 500 ms, not 5 ms, and reading it naively
    puts cues a half-second out.

    Args:
        h, m, s: hours, minutes, seconds as captured strings.
        ms: fractional part, one to three digits.

    Returns:
        float: seconds.
    """
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _s2t(v):
    """Render seconds as an SRT timestamp.

    Args:
        v: seconds; negatives are clamped to zero, since a cue before
            the start of the file is not representable.

    Returns:
        str: "HH:MM:SS,mmm" with the comma separator SRT requires.
    """
    if v < 0:
        v = 0.0
    ms = int(round(v * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, (ms // 60000) % 60,
                                    (ms // 1000) % 60, ms % 1000)


def clamp_srt(path, duration, log=None):
    """Constrain a generated subtitle file to the length of the picture.

    Whisper assigns its trailing cues a fixed span of roughly 30 seconds
    regardless of where the audio actually stops. Because Matroska takes
    its container duration from the longest track, a single overhanging
    cue stretches the finished file past the video and fails the
    duration check - on a track that is otherwise perfectly good.

    Cues beginning after the end are dropped outright; cues that merely
    run over are truncated to it. The file is rewritten in place only if
    something actually changed.

    Args:
        path: SRT to repair, modified in place.
        duration: length of the picture in seconds. Non-positive values
            disable the check, since clamping to an unknown length would
            be worse than leaving it alone.
        log: optional logger.

    Returns:
        int: number of cues dropped plus cues trimmed; 0 if the file was
        already within bounds or could not be read.
    """
    if not duration or duration <= 0:
        return 0
    try:
        cues = parse_srt(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return 0
    kept = []
    dropped = trimmed = 0
    for idx, timing, body in cues:
        m = SRT_TIME.search(timing)
        if not m:
            kept.append((idx, timing, body))
            continue
        g = m.groups()
        start = _t2s(*g[0:4])
        end = _t2s(*g[4:8])
        if start >= duration:
            dropped += 1
            continue
        if end > duration:
            end = duration
            timing = "%s --> %s" % (_s2t(start), _s2t(end))
            trimmed += 1
        kept.append((idx, timing, body))
    if dropped or trimmed:
        write_srt(kept, path)
        if log:
            log.info("      subtitles clamped to %.1f s: %d cue(s) past "
                     "the end dropped, %d trimmed back"
                     % (duration, dropped, trimmed))
    return dropped + trimmed


def drop_trailing_repeats(path, log=None, run=3):
    """Strip a run of identical cues from the end of a transcript.

    The companion to clamp_srt, addressing the same root cause from the
    other side. Whisper was trained on subtitled video, so when it is
    handed silence it emits phrases remembered from that training data -
    "thank you for watching" is the canonical example - and repeats them
    until the audio ends. This is the single most documented failure
    mode of the model.

    Only the tail is examined. Repetition mid-transcript can be genuine
    (a chant, a refrain, a stutter), whereas three or more identical
    cues terminating the file are not dialogue.

    Args:
        path: SRT to repair, modified in place.
        log: optional logger.
        run: minimum consecutive identical cues to treat as a
            hallucination rather than legitimate repetition.

    Returns:
        int: number of cues removed; 0 if the tail looked genuine or the
        file could not be read.
    """
    try:
        cues = parse_srt(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return 0
    if len(cues) < run + 1:
        return 0
    def body_of(c):
        return " ".join(c[2]).strip().lower()
    last = body_of(cues[-1])
    if not last:
        return 0
    n = 0
    while n < len(cues) and body_of(cues[-1 - n]) == last:
        n += 1
    if n < run:
        return 0
    write_srt(cues[:len(cues) - n], path)
    if log:
        log.info("      dropped %d repeated trailing cue(s): %r"
                 % (n, last[:40]))
    return n


def srt_is_useful(path, min_cues=20):
    # a transcription that produced almost nothing is a failure wearing
    # a success exit code, so check before trusting it
    try:
        cues = parse_srt(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if len(cues) < min_cues:
        return False
    chars = sum(len(" ".join(b)) for _, _, b in cues)
    return chars > 200


# ---------------------------------------------------------------------
# audio preparation
# ---------------------------------------------------------------------
def enhance_filter(layout, centre=ENHANCE_CENTRE, enhance=ENHANCE_DIALOGUE,
                   pan_expr=None):
    # dialogue lives in the centre channel of a surround mix, so lifting
    # it and dropping everything else is the single biggest accuracy win
    # available to speech recognition
    chain = []
    if pan_expr:
        chain.append(pan_expr)
    chain.append("alimiter=limit=0.97")
    if enhance > 0:
        chain.append("dialoguenhance=original=1:enhance=%.2f"
                     % min(3.0, enhance))
        chain.append("aformat=channel_layouts=stereo")
    chain.append("highpass=f=80")
    chain.append("lowpass=f=8000")
    chain.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    chain.append("aformat=channel_layouts=stereo")
    return ",".join(chain)


def make_asr_wav(ffmpeg, src, stream_idx, filt, out_wav, log=None):
    # whisper wants 16 kHz mono PCM; anything else it resamples itself
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "error",
           "-i", str(src), "-map", "0:%d" % stream_idx]
    if filt:
        cmd += ["-af", filt + ",aresample=16000,aformat=channel_layouts=mono"]
    else:
        cmd += ["-af", "aresample=16000,aformat=channel_layouts=mono"]
    cmd += ["-c:a", "pcm_s16le", "-f", "wav", str(out_wav)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                           creationflags=NOWIN)
        if r.returncode != 0:
            if log:
                log.warning("    asr audio extract failed: %s"
                            % (r.stderr or "")[-300:])
            return None
        return out_wav
    except Exception as e:
        if log:
            log.warning("    asr audio extract failed: %s" % e)
        return None


# ---------------------------------------------------------------------
# whisper
# ---------------------------------------------------------------------
def find_model(model_dir, preferred=""):
    if not model_dir:
        return None
    from pathlib import Path
    d = Path(model_dir)
    if not d.is_dir():
        return None
    if preferred:
        q = d / preferred
        if q.exists():
            return q
    for name in MODEL_NAMES:
        q = d / name
        if q.exists():
            return q
    any_bin = sorted(d.glob("ggml-*.bin"))
    return any_bin[0] if any_bin else None


_WOPTS = {}


def whisper_options(ffmpeg, log=None):
    # option names for this filter vary between builds, and passing one
    # the build does not know kills the whole filterchain. ask it.
    if _WOPTS:
        return _WOPTS["set"]
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-h", "filter=whisper"],
                           capture_output=True, text=True, timeout=60,
                           creationflags=NOWIN)
        blob = (r.stdout or "") + (r.stderr or "")
    except Exception:
        blob = ""
    names = set(re.findall(r"^\s{2,}([a-z0-9_]+)\s+<", blob, re.M))
    _WOPTS["set"] = names
    if log:
        log.debug("whisper filter options: %s" % (sorted(names) or "none found"))
    return names


def esc_filter_path(pth):
    """Escape a path for use inside an ffmpeg filter option value.

    A WINDOWS DRIVE COLON NEEDS TWO BACKSLASHES, NOT ONE.

    ffmpeg escapes in layers - the filter's own option parser, then the
    filtergraph parser - and a colon has to survive both. MEASURED on
    8.1.2 with an argument list (no shell in the way, which is important:
    Git Bash rewrites anything that looks like a POSIX path and turns
    C\\:/x into nonsense that reads as an ffmpeg failure):

        model=C\\:/ProgramData/...      WORKS
        model='C\\:/ProgramData/...'    WORKS
        model=C\\\\:/ProgramData/...    FAILS - one layer short
        model='C\\:/ProgramData/...'    FAILS - quoting alone is not enough

    With one backslash every absolute-path attempt failed with
    "No option name near '/ProgramData/whisper/...'", so two of the three
    fallbacks in whisper_srt could never work and only the bare-name form
    ever transcribed anything. The colon is also why the destination has
    to be escaped, not just the model.

    The old version also replaced backslashes AFTER turning them all into
    forward slashes, so that step could never match anything.
    """
    s = str(pth).replace("\\", "/")
    return s.replace("'", "\\'").replace(":", "\\\\:")


def find_vad_model(model_dir):
    """Locate the Silero voice-activity model, if one is installed.

    Voice activity detection prevents whisper from being handed silence
    in the first place, which is the cleanest fix for the hallucinated
    trailing dialogue that drop_trailing_repeats removes after the fact.
    The ffmpeg filter expects it as a separate ggml file from the
    whisper.cpp project; without that file the vad_* options are inert.

    Note that VAD is off by default despite being the better fix.
    Measured on 60 s of speech it cost 25.4 s against 5.6 s without,
    for a transcript that was otherwise identical, because the detector
    runs single-threaded on the CPU and whisper waits on it.

    Args:
        model_dir: directory to search, normally alongside the whisper
            model itself.

    Returns:
        pathlib.Path | None: the model file, or None if absent. The size
        floor guards against a truncated or failed download being
        mistaken for a usable model.
    """
    from pathlib import Path
    try:
        d = Path(model_dir)
        if not d.is_dir():
            return None
        for p in sorted(d.glob("ggml-silero*.bin")):
            if p.is_file() and p.stat().st_size > 100000:
                return p
    except Exception:
        pass
    return None


def whisper_srt(ffmpeg, wav, model, lang, out_srt, use_gpu=True,
                queue=0, log=None, timeout=21600, vad_model=None):
    """Transcribe wav to out_srt. Returns the path, or None.

    The filter argument parser treats ':' as its option separator, so a
    Windows path like C:/... has to be escaped and that escaping has
    been the single biggest source of trouble here. The way round it is
    to not put a path in the filter at all: run with the working
    directory set to the model's folder and refer to both files by bare
    name. Nothing then needs escaping.
    """
    from pathlib import Path
    model = Path(model).resolve()
    # cwd is changed below, so every path handed to ffmpeg outside the
    # filter string has to be absolute or it will not be found
    wav = Path(wav).resolve()
    out_srt = Path(out_srt).resolve()
    known = whisper_options(ffmpeg, log)

    vad = Path(vad_model).resolve() if vad_model else None
    if vad and not vad.is_file():
        vad = None
    if vad and known and "vad_model" not in known:
        if log:
            log.info("      this ffmpeg has no vad_model option, "
                     "transcribing without voice detection")
        vad = None
    # MEASURED on 60 s of speech, ffmpeg 8.1.2, large-v3-turbo on GPU:
    #   no vad, queue default ...   5.6 s
    #   no vad, queue 20 ........   3.3 s
    #   vad,    queue default ...  25.4 s
    #   vad,    queue 20 ....... 134.1 s
    # so the two must NOT be coupled. the filter docs suggest a longer
    # queue suits voice detection; on this build that combination is
    # pathological. queue is left to the caller and vad is opt-in.
    q = queue

    def build(model_ref, dest_ref, vad_ref=None):
        # only what is actually needed. language defaults to auto,
        # use_gpu defaults to true, and queue is a DURATION whose
        # default of 3 seconds is already right, so none are sent
        # unless they differ from the default.
        parts = ["whisper=model=" + model_ref]
        if lang and lang != "auto" and (not known or "language" in known):
            parts.append("language=" + lang)
        if vad_ref:
            parts.append("vad_model=" + vad_ref)
        if q and (not known or "queue" in known):
            parts.append("queue=%d" % int(q))
        if not use_gpu and (not known or "use_gpu" in known):
            parts.append("use_gpu=false")
        parts.append("destination=" + dest_ref)
        parts.append("format=srt")
        return ":".join(parts)

    # ordered by how likely they are to survive a filter parser
    attempts = []
    if model.parent.is_dir() and os.access(str(model.parent), os.W_OK):
        # the vad model sits beside the whisper model, so in this attempt
        # it is a bare name too and needs no escaping either
        vref = vad.name if (vad and vad.parent == model.parent) else (
            esc_filter_path(vad) if vad else None)
        attempts.append(("bare names, cwd at the model folder",
                         build(model.name, out_srt.name, vref),
                         str(model.parent), model.parent / out_srt.name))
    vabs = esc_filter_path(vad) if vad else None
    # The destination needs the same escaping as the model - it is another
    # option value in the same filter string, and out_srt is an absolute
    # path with a drive colon in it. Passing out_srt.name here relied on
    # cwd instead, which is why this attempt wrote into whichever folder
    # it happened to be run from.
    attempts.append(("escaped absolute model path",
                     build(esc_filter_path(model),
                           esc_filter_path(out_srt), vabs),
                     str(out_srt.parent), out_srt))
    attempts.append(("quoted absolute model path",
                     build("'" + esc_filter_path(model) + "'",
                           out_srt.name, vabs),
                     str(out_srt.parent), out_srt))
    if vad:
        # if every VAD form is rejected, still try plain transcription
        # rather than losing subtitles altogether
        attempts.append(("bare names, no voice detection",
                         build(model.name, out_srt.name),
                         str(model.parent), model.parent / out_srt.name))

    last = ""
    for name, af, workdir, produced in attempts:
        if log:
            log.info("      whisper attempt: %s" % name)
            log.debug("      -af %s   (cwd %s)" % (af, workdir))
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "info",
               "-i", str(wav), "-af", af, "-f", "null", "-"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=workdir,
                               creationflags=NOWIN)
        except Exception as e:
            last = str(e)
            if log:
                log.warning("      failed to launch: %s" % e)
            continue
        if r.returncode != 0:
            err = (r.stderr or "").strip()
            last = err
            if log:
                # the whole error, not the tail. truncating this hid the
                # real cause through three versions.
                log.warning("      rc=%s" % r.returncode)
                for line in err.splitlines()[:12]:
                    log.warning("      ffmpeg: %s" % line[:200])
            continue
        if produced != out_srt and produced.exists():
            try:
                produced.replace(out_srt)
            except Exception:
                import shutil as _sh
                _sh.copy2(str(produced), str(out_srt))
                produced.unlink(missing_ok=True)
        if out_srt.exists() and srt_is_useful(out_srt):
            if log:
                log.info("      whisper succeeded via %s" % name)
            return out_srt
        if log:
            log.warning("      produced nothing usable")
        last = "no usable subtitles produced"
    if log:
        log.warning("    whisper could not transcribe: %s" % last[:200])
    return None


# ---------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------
def _post(url, data, headers, timeout=NET_TIMEOUT):
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ---------------------------------------------------------------------
# Script detection: what language is this text ACTUALLY in?
# ---------------------------------------------------------------------
# A subtitle track's language TAG is a claim, not evidence. Two films in
# the 2026-08-13 batch carried a track tagged "eng" whose text is
# Japanese, and because the tag was believed, the pipeline kept it as
# the English track and skipped generating a real one. The finished
# files have an English-labelled track nobody can read.
#
# This detects SCRIPT, not language. That distinguishes Japanese from
# English with certainty and is exactly the question being asked here;
# it would NOT tell English from French, and nothing below pretends
# otherwise.
_SCRIPTS = (
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("han", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
    ("hangul", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ("cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("greek", ((0x0370, 0x03FF),)),
    ("devanagari", ((0x0900, 0x097F),)),
)


def script_profile(text):
    """Count the letters of each script in a block of text.

    Punctuation, digits and whitespace are ignored: they are shared
    between scripts and would dilute the ratios that matter.
    """
    counts = dict((name, 0) for name, _ in _SCRIPTS)
    counts["latin"] = 0
    for ch in text or "":
        o = ord(ch)
        if o < 0x80:
            if ch.isalpha():
                counts["latin"] += 1
            continue
        hit = None
        for name, ranges in _SCRIPTS:
            for lo, hi in ranges:
                if lo <= o <= hi:
                    hit = name
                    break
            if hit:
                break
        if hit:
            counts[hit] += 1
        elif ch.isalpha() and o < 0x250:
            # Latin-1 supplement and Latin Extended-A: accented Latin.
            counts["latin"] += 1
    return counts


def sniff_language(text, min_chars=40):
    """Best guess at the language of subtitle text, from its script.

    Returns an ISO-639-2 style code where the script settles it, "latin"
    where the script is Latin but the language is not determined, or
    "unknown" when there is too little to judge.

    KANA IS DECISIVE. Hiragana and katakana appear in no other language,
    so any real presence of them means Japanese. The threshold is 4 pct
    rather than zero because an English subtitle for an anime legitimately
    carries a few kana in signs and song titles - measured against a real
    two-hour English track, that is well under one percent, so the two
    cases separate cleanly.
    """
    c = script_profile(text)
    total = sum(c.values())
    if total < min_chars:
        return "unknown"
    frac = dict((k, v / float(total)) for k, v in c.items())
    if frac["kana"] > 0.04:
        return "jpn"
    # Han with no kana at all is Chinese; Japanese prose essentially
    # never goes a whole subtitle file without kana.
    if frac["han"] > 0.20:
        return "jpn" if c["kana"] else "chi"
    if frac["hangul"] > 0.20:
        return "kor"
    if frac["cyrillic"] > 0.30:
        return "rus"
    if frac["arabic"] > 0.30:
        return "ara"
    if frac["hebrew"] > 0.30:
        return "heb"
    if frac["thai"] > 0.30:
        return "tha"
    if frac["greek"] > 0.30:
        return "ell"
    if frac["devanagari"] > 0.30:
        return "hin"
    if frac["latin"] > 0.60:
        return "latin"
    return "unknown"


def srt_language(path, max_bytes=400000):
    """sniff_language() over a subtitle file's TEXT, not its markup.

    Timestamps and cue numbers are stripped first: they are digits and
    punctuation either way, but leaving them in would put a floor under
    the character count on a file with almost no dialogue.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError, ValueError):
        # Only the failures that are genuinely about THIS FILE are
        # swallowed. A blanket `except Exception` here hid a NameError
        # for as long as it took to notice every sniff returning
        # "unknown", which is precisely the silent-no-op shape this
        # project keeps rediscovering.
        return "unknown"
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    body = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.isdigit() or "-->" in s:
            continue
        body.append(s)
    return sniff_language(" ".join(body))


_ARGOS = {}


def _argos_pair(at, src_code, tgt):
    """The installed translation for one DIRECTED pair, or None.

    Not "are both languages installed" - that was the bug. argos lists a
    language as installed if ANY package mentions it, so with only
    en->ja present both "en" and "ja" appear installed, and a ja->en
    lookup sails past the "is it missing" check and then returns None
    from get_translation.

    Measured on the 2026-08-13 machine: installed languages ['en','ja'],
    installed pairs en->ja, en->en, ja->ja. ja->en did not exist, was
    never downloaded because the guard did not notice, and every
    translation call returned an empty list WITHOUT LOGGING ANYTHING.
    """
    langs = at.get_installed_languages()
    src = next((x for x in langs if x.code == src_code), None)
    dst = next((x for x in langs if x.code == tgt), None)
    if not src or not dst:
        return None
    try:
        return src.get_translation(dst) or None
    except Exception:
        return None


def _translate_argos(lines, target, log=None, source="en"):
    """Translate entirely offline with argos-translate.

    No API key, no per-token charge and no network, which is what makes
    it usable here: DeepL was refused on cost, and nothing was installed
    to take its place, so English-only films have been coming out with
    no Japanese subtitles at all.

    argostranslate ships small CTranslate2 NMT models, one per language
    pair, downloaded once. It resolves on Python 3.14 - the ctranslate2
    pin that blocked WhisperX does not apply, as argos takes 4.8.1 which
    has a 3.14 wheel.

    Quality is below DeepL. It is a machine translation of a machine
    transcription, so it is a convenience track, not a substitute for a
    real subtitle. Returns [] if the package or the language pair is
    missing, which leaves the caller to carry on without the track.
    """
    try:
        import argostranslate.package as ap
        import argostranslate.translate as at
    except ImportError:
        if log:
            log.warning("    argos-translate is not installed; no local "
                        "translation available")
        return []
    # argostranslate narrates every internal step through the standard
    # logging module at INFO - the paragraphs, the sentences, the
    # tokens, the raw hypotheses and their scores, for EVERY line. The
    # pipeline's root logger is at INFO, so a single episode's subtitles
    # would bury the run log under thousands of lines of tokeniser
    # output. Its own settings.debug flag does not gate this; only the
    # logger level does.
    for _n in ("argostranslate", "stanza", "sentencepiece"):
        _lg = logging.getLogger(_n)
        _lg.setLevel(logging.WARNING)
        _lg.propagate = False
    tgt = (target or "").lower()[:2]
    src_code = (source or "en").lower()[:2]
    key = src_code + "->" + tgt
    if key not in _ARGOS:
        tr = None
        try:
            # Ask for the PAIR. If it is not installed, install it -
            # whether or not the two languages happen to appear
            # elsewhere in the installed set.
            tr = _argos_pair(at, src_code, tgt)
            if tr is None:
                ap.update_package_index()
                avail = ap.get_available_packages()
                hit = next((p for p in avail
                            if p.from_code == src_code
                            and p.to_code == tgt), None)
                if hit is None:
                    if log:
                        log.warning("    argos publishes no %s->%s model, so "
                                    "this direction cannot be translated"
                                    % (src_code, tgt))
                    _ARGOS[key] = False
                    return []
                if log:
                    log.info("    downloading the argos %s->%s model once "
                             "(it was not installed)" % (src_code, tgt))
                ap.install_from_path(hit.download())
                tr = _argos_pair(at, src_code, tgt)
                if tr is None and log:
                    log.warning("    the argos %s->%s model installed but "
                                "still does not resolve" % (src_code, tgt))
        except Exception as e:
            if log:
                log.warning("    argos setup failed for %s->%s: %s: %s"
                            % (src_code, tgt, type(e).__name__, e))
            tr = False
        _ARGOS[key] = tr or False
        if _ARGOS[key] and log:
            log.info("    argos %s->%s ready" % (src_code, tgt))
    tr = _ARGOS[key]
    if not tr:
        # Said once per pair per run rather than silently. The previous
        # version returned an empty list here with no message at all, so
        # the only symptom was the caller reporting "0 lines" over and
        # over with no cause anywhere in the log.
        if log and not _ARGOS.get(key + "!said"):
            _ARGOS[key + "!said"] = True
            log.warning("    no usable argos %s->%s translator; nothing "
                        "will be translated in this direction"
                        % (src_code, tgt))
        return []
    out = []
    for ln in lines:
        try:
            out.append(tr.translate(ln))
        except Exception:
            # one bad line must not lose the whole batch
            out.append(ln)
    return out


def translate_lines(lines, target, provider, key, endpoint="", model="",
                    log=None, source="en"):
    # translated in batches, keeping one source line per output line so
    # the cue timings never have to be recalculated
    if not lines:
        return []
    if provider == "argos":
        return _translate_argos(lines, target, log, source)
    if provider == "deepl":
        host = ("https://api-free.deepl.com" if key.endswith(":fx")
                else "https://api.deepl.com")
        body = [("target_lang", target.upper())]
        body += [("text", x) for x in lines]
        data = urllib.parse.urlencode(body).encode("utf-8")
        try:
            d = _post(host + "/v2/translate", data,
                      {"Authorization": "DeepL-Auth-Key " + key,
                       "Content-Type": "application/x-www-form-urlencoded"})
            return [t.get("text", "") for t in d.get("translations", [])]
        except Exception as e:
            if log:
                log.warning("    deepl failed: %s" % e)
            return []
    # any OpenAI-compatible chat endpoint, which covers most local and
    # hosted LLM servers
    numbered = "\n".join("%d\t%s" % (i + 1, x) for i, x in enumerate(lines))
    prompt = ("Translate each numbered line into %s. This is film "
              "subtitle dialogue: keep it natural and idiomatic rather "
              "than literal. Reply with exactly the same number of "
              "lines, each as <number><tab><translation>, and nothing "
              "else.\n\n%s" % (target, numbered))
    payload = json.dumps({
        "model": model or "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2}).encode("utf-8")
    try:
        d = _post((endpoint or "https://api.openai.com/v1")
                  + "/chat/completions", payload,
                  {"Authorization": "Bearer " + key,
                   "Content-Type": "application/json"})
        txt = d["choices"][0]["message"]["content"]
    except Exception as e:
        if log:
            log.warning("    translation endpoint failed: %s" % e)
        return []
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)[\t\.\)]\s*(.*)", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return [out.get(i + 1, lines[i]) for i in range(len(lines))]


def translate_srt(in_srt, out_srt, target, provider, key, endpoint="",
                  model="", batch=80, log=None, source="en"):
    try:
        cues = parse_srt(in_srt.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        if log:
            log.warning("    could not read source subtitles: %s" % e)
        return None
    if not cues:
        return None
    # newlines inside a cue are joined so one cue is always one line,
    # then split back out afterwards
    flat = [" ".join(b).replace("\n", " ") for _, _, b in cues]
    done = []
    ok_lines = 0
    bad_batches = 0
    for i in range(0, len(flat), batch):
        chunk = flat[i:i + batch]
        got = translate_lines(chunk, target, provider, key, endpoint,
                              model, log, source)
        if len(got) != len(chunk):
            bad_batches += 1
            if log:
                log.warning("    translation returned %d lines for %d; "
                            "keeping the original for this batch"
                            % (len(got), len(chunk)))
            got = chunk
        else:
            ok_lines += len(chunk)
        done.extend(got)
        if log:
            log.debug("translated %d/%d cues" % (min(i + batch, len(flat)),
                                                 len(flat)))

    # REFUSE TO WRITE AN UNTRANSLATED "TRANSLATION".
    #
    # Every failed batch above falls back to the ORIGINAL text, which is
    # the right thing to do for one batch out of fifty and catastrophic
    # when it is all of them: the caller labels whatever comes back as
    # English and muxes it in. On 2026-08-13 argos was returning nothing
    # for every batch, so two finished films shipped with a full-length
    # English-tagged subtitle track containing nothing but Japanese.
    #
    # Returning None is what the caller already treats as "no translation
    # available", which leaves the file without an English track rather
    # than with a lying one.
    if ok_lines == 0:
        if log:
            log.error("    NOTHING was translated - every batch failed. "
                      "Refusing to write a file that would be tagged %s "
                      "and contain the untranslated original."
                      % (target or "?"))
        return None
    if bad_batches:
        frac = 100.0 * ok_lines / max(1, len(flat))
        if log:
            log.warning("    only %.0f%% of the cues translated; %d batch(es) "
                        "kept their original text" % (frac, bad_batches))
        if frac < 50.0:
            if log:
                log.error("    less than half translated; refusing to write "
                          "it rather than pass it off as %s"
                          % (target or "?"))
            return None

    new = [(c[0], c[1], [done[n]]) for n, c in enumerate(cues)]
    written = write_srt(new, out_srt)
    if not written:
        return None

    # VERIFY THE ARTIFACT. Counting successful batches proves the calls
    # returned something; it does not prove that something is in the
    # target language. A model that echoes its input, or a provider that
    # returns the prompt, would pass every check above. Reading the
    # finished file back and asking what script it is in catches all of
    # those at once, including whatever the next one turns out to be.
    want = (target or "").lower()[:2]
    got_lang = srt_language(written)
    if want == "en" and got_lang not in ("latin", "unknown"):
        if log:
            log.error("    the translated file reads as '%s', not English. "
                      "Discarding it: a track tagged English has to BE "
                      "English." % got_lang)
        try:
            Path(written).unlink()
        except OSError:
            pass
        return None
    if want == "ja" and got_lang == "latin":
        if log:
            log.error("    the Japanese translation came back in Latin "
                      "script; discarding it.")
        try:
            Path(written).unlink()
        except OSError:
            pass
        return None
    return written
