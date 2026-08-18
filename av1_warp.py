# ============================================================================
# av1_warp.py  V0.2
#
# Conform a rendered file's audio to its master's timeline.
#
# THE METHOD, WHICH IS THE MANUAL ONE DONE BY MACHINE
#
# Open both tracks in an editor, overlap the waveforms, and walk the
# film. At each minute mark, adjust the rendered audio's PLAYBACK RATE
# so its peaks and valleys sit on the master's peaks and valleys. Each
# adjustment carries everything after it, so by the end the whole track
# has been conformed to the master.
#
# Two rules follow from that description and both are load-bearing:
#
#   * it adjusts PLAYBACK TIME, never cuts. Every output sample is
#     interpolated from the input, so nothing drops out. A segment that
#     runs 30 ms fast is played 0.05 pct slower, not spliced.
#   * the adjustments are per minute and therefore tiny - parts in ten
#     thousand. Pitch is not preserved, and at that size it does not
#     need to be: 0.05 pct is under a cent, where a semitone is 100.
#
# BEFORE ANY OF THAT: the start points are locked together with a broad
# search across the ENTIRE waveform, so the per-minute walk begins from
# a known-good alignment rather than from a guess. Without it the walk
# can start out by several seconds and spend the film chasing an offset
# it never had a prior for.
#
# WHAT IT CANNOT DO
#
# It cannot restore samples that were never written. A track measured
# shorter than its master is missing content; conforming puts everything
# on the right timecode, and the missing part stays missing.
#
# WHY NOTHING ELSE IS LEFT IN HERE
#
# Earlier attempts corrected sync with mkvmerge --sync - one delay, or a
# delay plus one linear ratio, for a whole track. That is only right
# when the error IS a straight line, and measured on a real output it is
# not: fit rms 253 ms against a linear model, flat near zero for ten
# minutes and then flat near +1.55 s. Steps, not a ramp. Those routines
# have been removed rather than left lying around to be called by
# mistake.
# ============================================================================

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PROBE_SR = 16000            # correlation rate; speech lives well below 8k
SEG_SEC = 90.0              # per-anchor window. long, because film audio
                            # is self-similar and a short window over a
                            # wide search finds convincing wrong peaks
FINE_SEARCH = 0.5           # +/- window once the walk is tracking
WIDE_SEARCH = 6.0           # retry window when the narrow probe fails
MIN_PEAK_RATIO = 8.0        # peak must stand clear of the search field
DEFAULT_STEP = 60.0         # one adjustment a minute, as specified
ENV_HZ = 100.0              # envelope rate for the broad start search
SPIKE_TOL = 0.25            # a lone anchor this far from its neighbours
                            # is a bad read, not a real movement
STEP_NOTE_MS = 400.0        # jumps larger than this are worth reporting
# A track already this close to the master is left alone: rewriting it
# would spend a lossy generation and a full remux to change nothing.
# Lip sync is not detectable much below this, and on a batch of hundreds
# it is the difference between conforming what is wrong and re-encoding
# everything.
SKIP_MS = 40.0
# A jump bigger than this between neighbouring anchors is MISSING AUDIO,
# not drift, and is repaired by inserting silence of exactly that length
# rather than by slowing the surrounding minute down to cover it.
# Stretching a minute by 2 pct to absorb 1.2 s would shift its pitch by
# a third of a semitone and would be plainly audible on music; inserting
# silence where the samples actually went missing is both honest and
# inaudible. The rate form of the test keeps the pitch change from any
# segment under 0.2 pct, which is about 3 cents.
STEP_RATE = 0.002
STEP_FLOOR_SEC = 0.100
SHORT_SEG = 12.0            # window for bisecting the position of a step

# HARD SANITY LIMITS. These existed in the tool this replaced and were
# not carried across, which is how a garbage reading reached a
# destructive rewrite: The Simpsons Movie was measured as 206 SECONDS
# out and had 206 s of silence padded onto its head. No correction this
# large is ever a real one - it is a bad correlation - and nothing may
# act on one again.
SANITY_OFFSET_SEC = 10.0    # a bigger offset than this is a misread
SANITY_RATE = 0.02          # 2 pct; beyond this it is not a conform
SANITY_HEAD_SEC = 10.0      # silence padded at the head
SANITY_TOTAL_GAP_SEC = 30.0  # total silence inserted mid-film

# Moves with av1_pipeline_v0_1.VERSION, which selftest enforces. This
# tool rewrites finished files by hand, so a report from it has to say
# which build produced the rewrite.
VERSION = "0.256"


def _run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, timeout=timeout,
                          creationflags=NOWIN)


def _pcm(ffmpeg, path, stream, start, dur, rate=PROBE_SR):
    """Mono PCM for one window, as a float array."""
    import numpy as np
    p = _run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
              "-ss", "%.3f" % max(0.0, start), "-i", str(path),
              "-t", "%.3f" % dur, "-map", stream, "-vn", "-sn",
              "-af", "aresample=%d,aformat=channel_layouts=mono" % rate,
              "-c:a", "pcm_s16le", "-f", "s16le", "-"], timeout=1800)
    return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32)


def _lag_seconds(a, ref, maxlag_sec, minlag_sec=None, rate=PROBE_SR):
    """Where `a` sits relative to `ref`, by FFT cross-correlation.

    FFT, not the naive form, which is O(n^2) and does not return on
    minutes of audio. Returns (lag_seconds, peak/median ratio).

    minlag_sec bounds the search from below, defaulting to -maxlag_sec.
    Callers that centre a search on an expected offset need a window
    that is genuinely symmetric about it; passing only a maximum gives
    one running from expect-3*search to expect+search.
    """
    import numpy as np
    try:
        from scipy import signal
    except ImportError:
        return None, 0.0
    n = min(len(a), len(ref))
    if n < rate:
        return None, 0.0
    a = a[:n] - a[:n].mean()
    b = ref[:n] - ref[:n].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None, 0.0
    a, b = a / na, b / nb
    k = int(maxlag_sec * rate)
    kmin = int((-maxlag_sec if minlag_sec is None else minlag_sec) * rate)
    c = signal.correlate(b, a, mode="full", method="fft")
    mid = len(c) // 2
    lo, hi = max(0, mid + kmin), min(len(c), mid + k + 1)
    seg = c[lo:hi]
    if not len(seg):
        return None, 0.0
    seg = np.abs(seg)
    i = int(np.argmax(seg))
    med = float(np.median(seg))
    ratio = (float(seg[i]) / med) if med else 0.0
    return (lo + i - mid) / float(rate), ratio


def audio_offset(ffmpeg, out_path, src_path, t, out_stream="a:0",
                 src_stream="a:0", expect=0.0, search=FINE_SEARCH,
                 seg=SEG_SEC):
    """Seconds the OUTPUT audio sits from the SOURCE audio at time t.

    Positive means the output is playing, at t, content the source has
    LATER - the audio is running early and has to be slowed or delayed.

    The search window is centred on `expect` and is `search` wide either
    side. Narrow is the point: film audio is self-similar, and a wide
    search gives the correlator room to find a convincing wrong answer
    several seconds away. Measured on a real film with the full window,
    successive probes on one track returned +0.032, +0.534, +3.009,
    +0.374 and +0.769 s - every one passing the sharpness gate, no two
    agreeing.
    """
    a = _pcm(ffmpeg, out_path, "0:" + out_stream, t, seg)
    b = _pcm(ffmpeg, src_path, "0:" + src_stream, t + expect - search,
             seg + 2 * search)
    lag, conf = _lag_seconds(a, b, search * 2, 0.0)
    if lag is None:
        return None, 0.0
    return lag - search + expect, conf


def robust_line(ts, es, min_keep=3):
    """Fit a line, rejecting outliers on a Theil-Sen estimate first.

    Least squares cannot find its own outliers: a bad point mid-span
    drags the intercept so every honest point looks wrong, and a bad
    point at either end has the leverage to tilt the line toward itself
    and inflate any spread-based tolerance until nothing is rejected.
    Both were measured. The median of pairwise slopes has neither
    failure.
    """
    import numpy as np
    ts = np.asarray(ts, dtype=float)
    es = np.asarray(es, dtype=float)
    keep = np.ones(len(ts), dtype=bool)
    if len(ts) >= 4 and ts.max() > ts.min():
        sl = [(es[j] - es[i]) / (ts[j] - ts[i])
              for i in range(len(ts)) for j in range(i + 1, len(ts))
              if ts[j] != ts[i]]
        if sl:
            rs = float(np.median(sl))
            rb = float(np.median(es - rs * ts))
            dev = np.abs(es - (rs * ts + rb))
            tol = max(0.050, 4.0 * (float(np.median(dev)) or 0.005))
            k = dev <= tol
            if int(k.sum()) >= min_keep:
                keep = k
    if int(keep.sum()) < 2 or ts[keep].max() == ts[keep].min():
        return 0.0, float(np.median(es[keep])), keep, 0.0
    slope, intercept = np.polyfit(ts[keep], es[keep], 1)
    resid = es[keep] - (slope * ts[keep] + intercept)
    return (float(slope), float(intercept), keep,
            float(np.sqrt((resid ** 2).mean())))


# ---------------------------------------------------------------------
# step one: lock the start points together, across the whole waveform
# ---------------------------------------------------------------------
def _envelope(ffmpeg, path, stream, rate=8000):
    """Whole-track amplitude envelope at ENV_HZ, as a float array.

    The envelope rather than the waveform, because this correlation is
    between a raw master and a processed, re-encoded downmix: their fine
    structure differs but their loudness contour does not. It is also
    small enough to correlate whole - a two hour film is 720,000 points.
    """
    import numpy as np
    p = _run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
              "-i", str(path), "-map", "0:" + stream, "-vn", "-sn",
              "-af", "aresample=%d,aformat=channel_layouts=mono" % rate,
              "-c:a", "pcm_s16le", "-f", "s16le", "-"], timeout=7200)
    x = np.frombuffer(p.stdout, dtype="<i2").astype(np.float32)
    if not len(x):
        return None
    per = int(rate / ENV_HZ)
    n = (len(x) // per) * per
    if n < per:
        return None
    return np.abs(x[:n]).reshape(-1, per).mean(axis=1)


_ENV_CACHE = {}


def global_align(ffmpeg, out_path, src_path, out_stream="a:0",
                 src_stream="a:0", log=print):
    """Broad search over the ENTIRE waveform for the start alignment.

    Run before the per-minute walk on every file. It correlates the two
    complete loudness envelopes, so there is no search limit at all -
    the answer can be anywhere in the runtime - and it gives the walk a
    prior that is right to about a hundredth of a second.

    Returns (offset_seconds, confidence). Positive means the output is
    running early with respect to the master.
    """
    import numpy as np
    try:
        from scipy import signal
    except ImportError:
        return None, 0.0
    eo = _envelope(ffmpeg, out_path, out_stream)
    # cached: the master's envelope does not change between the tracks
    # being conformed, and decoding a 50 GB master again for each one
    # costs a minute a track for an identical answer
    key = (str(src_path), src_stream)
    es = _ENV_CACHE.get(key)
    if es is None:
        es = _envelope(ffmpeg, src_path, src_stream)
        if es is not None:
            _ENV_CACHE.clear()
            _ENV_CACHE[key] = es
    if eo is None or es is None:
        return None, 0.0
    a = eo - eo.mean()
    b = es - es.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None, 0.0
    c = signal.correlate(b / nb, a / na, mode="full", method="fft")
    c = np.abs(c)
    i = int(np.argmax(c))
    med = float(np.median(c))
    ratio = (float(c[i]) / med) if med else 0.0
    lag = (i - (len(a) - 1)) / ENV_HZ
    log("  broad start search: %+.2f s over the whole waveform "
        "(sharpness %.1f)" % (lag, ratio))
    if abs(lag) > SANITY_OFFSET_SEC:
        # Nothing this pipeline does moves audio by more than a few
        # seconds. A reading of +201 s at sharpness 14 is the correlator
        # finding a similar-sounding passage three minutes away, and
        # acting on one padded 206 s of silence onto a correct film.
        log("  REJECTED: %+.1f s is not a plausible offset; treating the "
            "start as unknown" % lag)
        return None, ratio
    return lag, ratio


# ---------------------------------------------------------------------
# step two: walk the film, one adjustment a minute
# ---------------------------------------------------------------------
def measure_anchors(ffmpeg, out_path, src_path, duration, out_stream,
                    src_stream="a:0", step=DEFAULT_STEP, start=0.0,
                    log=print):
    """Offset of the rendered waveform against the master, every `step`.

    The narrow window is centred on what the PREVIOUS anchor measured,
    so the walk follows the real shape. When the narrow probe fails -
    which is what happens at a step, where the truth has moved outside
    the window - it retries WIDE rather than giving up. Without that
    retry, anchors on the minority side of a step are all rejected and
    the step is flattened into a uniform shift: measured on a fixture
    with 1.2 s cut at t=200 s, every anchor before the cut was discarded
    and the correction moved the good part instead of fixing the bad.
    """
    anchors = []
    t = max(0.0, duration * 0.01)
    predicted = start
    while t <= duration * 0.99:
        off, conf = audio_offset(ffmpeg, out_path, src_path, t,
                                 out_stream, src_stream,
                                 expect=predicted, search=FINE_SEARCH)
        good = (off is not None and conf >= MIN_PEAK_RATIO
                and abs(off - predicted) < FINE_SEARCH * 0.95
                and abs(off) <= SANITY_OFFSET_SEC)
        how = "fine"
        if not good:
            off2, conf2 = audio_offset(ffmpeg, out_path, src_path, t,
                                       out_stream, src_stream,
                                       expect=predicted,
                                       search=WIDE_SEARCH)
            if (off2 is not None and conf2 >= MIN_PEAK_RATIO
                    and abs(off2) <= SANITY_OFFSET_SEC):
                off, conf, good, how = off2, conf2, True, "wide"
        anchors.append({"t": t, "offset": off, "conf": conf,
                        "good": good, "how": how})
        if good:
            predicted = off
        t += step

    # Isolated spikes go, steps stay. A running median does exactly
    # that; a "moved more than N ms since the last anchor" veto cannot
    # tell a real cut from a bad correlation and rejects both.
    vals = [a["offset"] if a["good"] else None for a in anchors]
    for i, a in enumerate(anchors):
        if not a["good"]:
            continue
        win = [v for v in vals[max(0, i - 2):i + 3] if v is not None]
        if len(win) >= 3:
            med = sorted(win)[len(win) // 2]
            if abs(a["offset"] - med) > SPIKE_TOL:
                a["good"] = False
                a["how"] = "spike"
    used = [a for a in anchors if a["good"]]
    # REFUSE rather than extrapolate over a long blind span.
    #
    # Anchors fail for two very different reasons: a passage the
    # correlator cannot lock onto, or NO AUDIO BEING THERE AT ALL. This
    # code could not tell them apart, so when a real file turned up with
    # its audio truncated at 3396 s of a 6056 s film, every anchor past
    # that point failed, the last good offset was extended flat to the
    # end, and the warp padded 44 minutes of silence onto the tail. A
    # detectable fault was turned into a plausible-looking file and
    # shipped. Never again: if the usable anchors do not cover the film,
    # this reports failure and touches nothing.
    if used:
        first_t, last_t = used[0]["t"], used[-1]["t"]
        blind_tail = duration - last_t
        blind_head = first_t
        gap = 0.0
        for x, y in zip(used, used[1:]):
            gap = max(gap, y["t"] - x["t"])
        # Relative as well as absolute: "a fifth of the film has no
        # usable audio" is a truncation whatever the runtime, and a
        # purely absolute limit let a 600 s fixture missing its last
        # 264 s through. A genuinely quiet credit roll is a couple of
        # minutes and stays under this.
        limit = max(120.0, duration * 0.05)
        if blind_tail > limit:
            return None, ("no usable audio after %.0f s of %.0f s - the "
                          "track looks truncated, not merely hard to "
                          "measure" % (last_t, duration))
        if blind_head > limit:
            return None, ("no usable audio before %.0f s - the track "
                          "looks truncated at the head" % first_t)
        if gap > limit:
            return None, ("a %.0f s stretch has no usable audio; refusing "
                          "to guess across it" % gap)
    jumps = sum(1 for x, y in zip(used, used[1:])
                if abs(y["offset"] - x["offset"]) > STEP_NOTE_MS / 1000.0)
    log("  %d of %d anchors usable%s"
        % (len(used), len(anchors),
           "" if not jumps else ", %d step(s) over %.0f ms"
           % (jumps, STEP_NOTE_MS)))
    if len(used) < 3:
        return None, "only %d usable anchors" % len(used)
    return anchors, None


def localise_step(ffmpeg, out_path, src_path, t_a, t_b, o_a, o_b,
                  out_stream, src_stream, log=print):
    """Bisect to find WHERE between two anchors the audio went missing.

    A minute-wide anchor spacing says only that samples vanished
    somewhere in that minute. Inserting the silence at an arbitrary
    minute boundary would put it in the middle of a line of dialogue;
    bisecting with short windows puts it within a few seconds of where
    the audio actually stops.
    """
    lo, hi = t_a, t_b
    for _ in range(4):
        mid = (lo + hi) / 2.0
        off, conf = audio_offset(ffmpeg, out_path, src_path, mid,
                                 out_stream, src_stream, expect=o_a,
                                 search=WIDE_SEARCH, seg=SHORT_SEG)
        if off is None or conf < MIN_PEAK_RATIO:
            break
        if abs(off - o_a) <= abs(off - o_b):
            lo = mid          # still aligned as before: the step is later
        else:
            hi = mid
    where = (lo + hi) / 2.0
    log("    step localised to %.0f s (+/- %.0f s)"
        % (where, (hi - lo) / 2.0))
    return where


def build_warp(anchors, duration, start=0.0, log=print, localiser=None):
    """Turn the anchors into a time map, separating drift from loss.

    Two different faults need two different repairs and conflating them
    is what makes the result audible:

      DRIFT - the offset creeping by a few tens of ms over a minute.
        The audio is all there, just running at fractionally the wrong
        rate, so the segment's PLAYBACK RATE is adjusted. Parts in ten
        thousand; nothing is cut and nothing is inserted.

      LOSS - the offset jumping. Samples are missing, and no playback
        rate fixes that: slowing a whole minute by 2 pct to cover 1.2 s
        of absent audio shifts its pitch by a third of a semitone and
        smears a minute of dialogue to hide a fault that lasted a
        second. SILENCE of exactly the missing length is inserted at
        the point the audio stops instead, and the surrounding minutes
        keep their own rate.

    Returns (old_t, new_t, gaps) where gaps are intervals in the NEW
    timeline that are to be filled with silence rather than sourced
    from the input.
    """
    import numpy as np
    good = [a for a in anchors if a["good"]]
    # Extrapolate the FIRST measured offset back to t=0, and the last
    # one forward to the end. Do NOT use the broad-search value here:
    # it is a prior for the walk, not a measurement of the map, and if
    # it differs from the first anchor by even 90 ms the short segment
    # between them absorbs the whole difference. Measured: a 4.8 s
    # opening segment came out at 2.5 pct - a third of a semitone of
    # pitch shift over the first five seconds, from a disagreement too
    # small to matter anywhere else.
    pts = [(0.0, good[0]["offset"])]
    pts += [(a["t"], a["offset"]) for a in good]
    pts += [(duration, good[-1]["offset"])]
    # drop coincident anchors
    clean = [pts[0]]
    for t, o in pts[1:]:
        if t > clean[-1][0] + 1e-6:
            clean.append((t, o))

    # The map is simply new = old + offset(old). Where the offset JUMPS,
    # the map is discontinuous, and that discontinuity IS the silence
    # gap - it needs no separate accumulator. Carrying one alongside the
    # measured offsets counted the same correction twice: a 1.5 s loss
    # came out as a 3.0 s longer track with a 2.5 pct rate somewhere in
    # the middle to absorb the surplus.
    old_t, new_t, gaps = [clean[0][0]], [clean[0][0] + clean[0][1]], []
    for (t0, o0), (t1, o1) in zip(clean, clean[1:]):
        span = t1 - t0
        d = o1 - o0
        threshold = max(STEP_FLOOR_SEC, span * STEP_RATE)
        if d > threshold and localiser is not None:
            # missing audio: hold the playback rate either side and put
            # silence of exactly the missing length where it went
            ts = localiser(t0, t1, o0, o1)
            ts = min(max(ts, t0 + 1e-3), t1 - 1e-3)
            old_t.append(ts)
            new_t.append(ts + o0)
            gaps.append((ts + o0, ts + o1))
            old_t.append(ts + 1e-6)
            new_t.append(ts + o1)
            old_t.append(t1)
            new_t.append(t1 + o1)
        else:
            old_t.append(t1)
            new_t.append(t1 + o1)
    old_t = np.array(old_t, dtype=float)
    new_t = np.array(new_t, dtype=float)
    for i in range(1, len(new_t)):
        if new_t[i] <= new_t[i - 1]:
            new_t[i] = new_t[i - 1] + 1e-6
    d_old, d_new = np.diff(old_t), np.diff(new_t)
    rate = np.where(d_old > 1e-5, d_new / np.maximum(d_old, 1e-9), 1.0)
    worst = float(np.max(np.abs(1.0 - rate[d_old > 1e-5])))
    total_gap = sum(b - a for a, b in gaps)
    head = float(new_t[0])
    log("  warp: %d segments, playback rate within %.4f pct of normal, "
        "%d silence insert(s) totalling %.3f s, net %+.3f s"
        % (len(rate), worst * 100, len(gaps), total_gap,
           new_t[-1] - old_t[-1]))
    # LAST LINE OF DEFENCE. Any one of these means the MEASUREMENT is
    # wrong, not the film: conforming is a matter of milliseconds and
    # parts in ten thousand, never of minutes.
    problems = []
    if worst > SANITY_RATE:
        problems.append("a segment would play %.2f pct off speed"
                        % (worst * 100))
    if head > SANITY_HEAD_SEC:
        problems.append("%.1f s of silence would be padded at the head"
                        % head)
    if total_gap > SANITY_TOTAL_GAP_SEC:
        problems.append("%.1f s of silence would be inserted mid-film"
                        % total_gap)
    if abs(float(new_t[-1]) - float(old_t[-1])) > SANITY_OFFSET_SEC:
        problems.append("the track would change length by %.1f s"
                        % (float(new_t[-1]) - float(old_t[-1])))
    if problems:
        raise ValueError("refusing to conform: " + "; ".join(problems)
                         + ". That is a bad measurement, not a broken film.")
    return old_t, new_t, gaps


def warp_pcm(raw_in, raw_out, ch, rate, old_t, new_t, gaps=(),
             log=print):
    """Resample onto the new timeline. Adjusts playback time; never cuts.

    Every output sample is interpolated from the input, so no audio is
    dropped or spliced - a segment running fast is simply played
    slightly slower. Memory-mapped and processed in blocks so a long
    film need not fit in RAM.

    Linear interpolation: the rates here are parts in ten thousand, so
    each output sample lands a small fraction of a sample from an input
    sample and the error is a gentle softening far above anything
    audible.

    `gaps` are intervals in the NEW timeline where the input simply has
    no audio to offer - samples that were never written. They are filled
    with silence rather than by holding a sample or by stretching the
    neighbours across them.
    """
    import numpy as np
    src = np.memmap(raw_in, dtype="<i2", mode="r")
    n_old = len(src) // ch
    src = src.reshape(n_old, ch)
    # Do not manufacture a timeline longer than the audio that exists.
    # Clamping the index and carrying on filled a truncated track's tail
    # with a held sample, which encodes as silence and looks like a
    # complete file. If the map asks for more than the input has, that
    # is a fault to report, not to paper over.
    have = n_old / float(rate)
    # The map's last anchor sits at the PICTURE's duration, but the audio
    # legitimately ends a little earlier - a track with 1.5 s cut out of
    # it holds 478.5 s under a 480 s film. So trim the map to the audio
    # that actually exists rather than asking for samples past its end.
    if float(old_t[-1]) > have:
        short = float(old_t[-1]) - have
        if short > 30.0:
            # far more missing than any real master leaves at the tail:
            # this is a truncated track, and padding it with a held
            # sample is what shipped 44 minutes of silence
            raise RuntimeError(
                "the map needs %.1f s of audio but the track only holds "
                "%.1f s; refusing to invent %.1f s"
                % (float(old_t[-1]), have, short))
        new_end = float(np.interp(have, old_t, new_t))
        keep = np.asarray(old_t) <= have
        old_t = np.append(np.asarray(old_t)[keep], have)
        new_t = np.append(np.asarray(new_t)[keep], new_end)
    n_new = int(float(new_t[-1]) * rate)
    head = max(0.0, float(new_t[0]))       # master has content before the
    head_n = int(head * rate)              # render does: pad, do not hold
    block = 1 << 21
    with open(raw_out, "wb") as fh:
        if head_n > 0:
            log("  padding %.3f s of silence at the head" % head)
            fh.write(np.zeros((head_n, ch), dtype="<i2").tobytes())
        for start_i in range(head_n, n_new, block):
            stop = min(start_i + block, n_new)
            tau = np.arange(start_i, stop, dtype=np.float64) / rate
            t_old = np.interp(tau, new_t, old_t)
            idx = np.clip(t_old * rate, 0, n_old - 1.000001)
            i0 = idx.astype(np.int64)
            frac = (idx - i0).astype(np.float32)
            lo_i, hi_i = int(i0.min()), min(int(i0.max()) + 2, n_old)
            chunk = np.asarray(src[lo_i:hi_i], dtype=np.float32)
            if len(chunk) < 2:
                break
            rel = np.clip(i0 - lo_i, 0, len(chunk) - 2)
            a = chunk[rel]
            b = chunk[rel + 1]
            out = a + (b - a) * frac[:, None]
            np.clip(out, -32768, 32767, out=out)
            for g0, g1 in gaps:
                if g1 > tau[0] and g0 < tau[-1]:
                    out[(tau >= g0) & (tau < g1)] = 0.0
            fh.write(out.astype("<i2").tobytes())
    if gaps:
        log("  %d silence insert(s), %.3f s total, where audio was missing"
            % (len(gaps), sum(b - a for a, b in gaps)))
    log("  wrote %.1f s of conformed audio" % (n_new / float(rate)))
    return raw_out


# ---------------------------------------------------------------------
# rebuilding the file
# ---------------------------------------------------------------------
def audio_streams(ffprobe, path):
    p = _run([ffprobe, "-v", "error", "-select_streams", "a",
              "-show_entries",
              "stream=index,codec_name,channels,sample_rate,bit_rate:"
              "stream_tags=language,title,BPS",
              "-print_format", "json", str(path)], timeout=300)
    try:
        streams = json.loads(p.stdout.decode("utf-8", "replace"))["streams"]
    except Exception:
        return []
    out = []
    for n, s in enumerate(streams):
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        br = s.get("bit_rate") or tags.get("bps") or 0
        try:
            br = int(br)
        except (TypeError, ValueError):
            br = 0
        out.append({"a_index": n, "codec": s.get("codec_name") or "",
                    "ch": int(s.get("channels") or 2),
                    "rate": int(s.get("sample_rate") or 48000),
                    "bit_rate": br,
                    "lang": (tags.get("language") or "und").lower(),
                    "title": tags.get("title") or ""})
    return out


def rebuild_track(ffmpeg, raw, ch, rate, bitrate, dest, log=print):
    """Encode the conformed PCM back to Opus at the track's own rate."""
    # libopus refuses more than 256 kbps a channel, and a lossless or
    # PCM source reports far above that - 1536 kbps for 48 kHz 16-bit
    # stereo - so inheriting the figure hands the encoder an argument it
    # rejects. A rate that high says "lossless", not "spend this much".
    target = int((bitrate or 0) / 1000)
    if not target or target > 256 * ch:
        target = 128 * ch
    kbps = max(96, min(768, 256 * ch, target))
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-y",
           "-f", "s16le", "-ar", str(rate), "-ac", str(ch), "-i", str(raw),
           "-c:a", "libopus", "-b:a", "%dk" % kbps, "-vbr", "on",
           "-application", "audio"]
    if ch > 2:
        cmd += ["-mapping_family", "1"]
    cmd += [str(dest)]
    p = _run(cmd, timeout=7200)
    if p.returncode != 0 or not Path(dest).exists():
        raise RuntimeError("re-encode failed: "
                           + p.stderr.decode("utf-8", "replace")[-300:])
    log("  re-encoded at %d kbps" % kbps)
    return dest


def remux(mkvmerge, original, new_audio, dest, log=print):
    """Original video, subtitles, fonts and chapters; conformed audio.

    Audio tracks that did NOT need conforming are carried through from
    the original untouched, in their own places. This used to pass
    --no-audio and re-add only the rebuilt tracks, which silently
    DELETED any track that was already in sync - and "already in sync"
    is the normal case for at least one track on a file with several.

    Nothing is re-encoded except the audio that was actually conformed.
    Track order is restated so every audio track sits where it did,
    between the video and the subtitles.

    Args:
        new_audio: rebuilt tracks, each carrying the a_index of the
            original track it replaces.
    """
    ident = _run([mkvmerge, "-J", str(original)], timeout=600)
    try:
        d = json.loads(ident.stdout.decode("utf-8-sig", "replace"))
    except Exception:
        return False, "could not identify the original"
    tracks = d.get("tracks", [])
    vids = [t["id"] for t in tracks if t["type"] == "video"]
    subs = [t["id"] for t in tracks if t["type"] == "subtitles"]
    auds = [t["id"] for t in tracks if t["type"] == "audio"]
    slot_of = {a["a_index"]: k for k, a in enumerate(new_audio, start=1)}
    keep_ids = [tid for n, tid in enumerate(auds) if n not in slot_of]

    cmd = [mkvmerge, "-o", str(dest)]
    if keep_ids:
        cmd += ["--audio-tracks", ",".join(str(t) for t in keep_ids)]
    else:
        cmd += ["--no-audio"]
    cmd += [str(original)]
    for a in new_audio:
        cmd += ["--language", "0:%s" % a["lang"]]
        if a.get("title"):
            cmd += ["--track-name", "0:%s" % a["title"]]
        cmd += ["--default-track-flag",
                "0:%d" % (1 if a["a_index"] == 0 else 0), str(a["path"])]
    order = [(0, i) for i in vids]
    for n, tid in enumerate(auds):
        order.append((slot_of[n], 0) if n in slot_of else (0, tid))
    order += [(0, i) for i in subs]
    cmd += ["--track-order", ",".join("%d:%d" % f for f in order)]
    if keep_ids:
        log("  carrying %d untouched audio track(s) through unchanged"
            % len(keep_ids))
    p = _run(cmd, timeout=21600)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode > 1 or not Path(dest).exists():
        return False, out[-400:]
    if p.returncode == 1:
        # mkvmerge's manual: exit 1 means the result "might be ok or
        # not" and both the warning and the file must be checked.
        # Swallowing this is how half a film's audio went missing.
        log("  mkvmerge finished WITH WARNINGS:")
        for line in [x for x in out.splitlines()
                     if "warning" in x.lower()][:8]:
            log("    " + line.strip())
    return True, "ok"


def conform(ffmpeg, ffprobe, mkvmerge, out_path, src_path, workdir,
            step=DEFAULT_STEP, src_stream="a:0", langs=("eng", "en"),
            apply=False, log=print, skip_ms=SKIP_MS):
    """Lock the start, walk the film, conform every wanted track."""
    out_path = Path(out_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    d = _run([ffprobe, "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nk=1:nw=1", str(out_path)], timeout=120)
    try:
        duration = float(d.stdout.decode().strip())
    except Exception:
        return {"ok": False, "why": "could not read the output duration"}
    streams = audio_streams(ffprobe, out_path)
    if not streams:
        return {"ok": False, "why": "no audio in the output"}
    if duration < SEG_SEC * 2:
        # An anchor correlates a 90 s window; a file shorter than a
        # couple of those cannot be walked a minute at a time. Not an
        # error - an extra or a short featurette simply has nothing to
        # conform - so it reports success having done nothing rather
        # than filling the log with failures on every clip in a batch.
        log("  only %.0f s long, too short to conform; left alone"
            % duration)
        return {"ok": True, "report": {}, "applied": False,
                "skipped": True, "why": "too short to measure"}
    # langs empty/None means "conform everything, drop nothing", which
    # is what the pipeline wants: it has already decided which tracks
    # the file gets, and commentary and descriptive tracks never made it
    # this far. Filtering again here would drop the only audio track of
    # a film whose language is neither English nor Japanese.
    if langs:
        wanted = [s for s in streams if s["lang"] in langs]
        if not wanted:
            log("  no track matches %s; conforming all of them instead"
                % ",".join(langs))
            wanted = list(streams)
    else:
        wanted = list(streams)
    dropped = [s for s in streams if s not in wanted]
    log("audio: %d track(s) present, %d to conform"
        % (len(streams), len(wanted)))
    if dropped:
        log("  dropping: %s"
            % ", ".join("a:%d %s %s" % (s["a_index"], s["lang"],
                                        s["title"]) for s in dropped))
    built, report = [], {}
    for s in wanted:
        stream = "a:%d" % s["a_index"]
        log("")
        log("track %s (%s, %dch)" % (stream, s["lang"], s["ch"]))
        start, sconf = global_align(ffmpeg, out_path, src_path, stream,
                                    src_stream, log)
        if start is None:
            start = 0.0
        anchors, why = measure_anchors(ffmpeg, out_path, src_path,
                                       duration, stream, src_stream,
                                       step, start, log)
        if anchors is None:
            return {"ok": False, "why": why}
        good = [a for a in anchors if a["good"]]
        report[stream] = {"start": start, "anchors": anchors,
                          "first_ms": good[0]["offset"] * 1000,
                          "last_ms": good[-1]["offset"] * 1000}
        log("  offset %+.0f ms at %.0f s  ->  %+.0f ms at %.0f s"
            % (good[0]["offset"] * 1000, good[0]["t"],
               good[-1]["offset"] * 1000, good[-1]["t"]))
        worst = max(abs(a["offset"]) for a in good) * 1000.0
        report[stream]["worst_ms"] = worst
        if worst < skip_ms:
            # Already on the master's timeline. Rewriting it would cost
            # a lossy generation and a full remux to change nothing, and
            # this runs on every file in a batch of hundreds.
            log("  worst error %.0f ms, under the %.0f ms threshold; "
                "leaving this track alone" % (worst, skip_ms))
            continue
        if not apply:
            continue
        try:
            old_t, new_t, gaps = build_warp(
                anchors, duration, start, log,
                localiser=lambda ta, tb, oa, ob: localise_step(
                    ffmpeg, out_path, src_path, ta, tb, oa, ob,
                    stream, src_stream, log))
        except ValueError as e:
            log("  %s" % e)
            return {"ok": False, "why": str(e)}
        raw = workdir / ("t%d.raw" % s["a_index"])
        warped = workdir / ("t%d.warp.raw" % s["a_index"])
        log("  decoding")
        p = _run([ffmpeg, "-hide_banner", "-v", "error", "-y",
                  "-i", str(out_path), "-map", "0:" + stream,
                  "-f", "s16le", "-acodec", "pcm_s16le",
                  "-ar", str(s["rate"]), "-ac", str(s["ch"]), str(raw)],
                 timeout=7200)
        if p.returncode != 0:
            return {"ok": False, "why": "decode failed: "
                    + p.stderr.decode("utf-8", "replace")[-200:]}
        log("  conforming")
        warp_pcm(raw, warped, s["ch"], s["rate"], old_t, new_t,
                 gaps, log)
        enc = workdir / ("t%d.opus.mka" % s["a_index"])
        rebuild_track(ffmpeg, warped, s["ch"], s["rate"], s["bit_rate"],
                      enc, log)
        built.append({"path": enc, "lang": s["lang"],
                      "title": s["title"], "a_index": s["a_index"]})
        for f in (raw, warped):
            try:
                f.unlink()
            except OSError:
                pass
    if not apply:
        return {"ok": True, "report": report, "applied": False}

    if not built:
        log("")
        log("every track is already on the master's timeline; nothing rewritten")
        return {"ok": True, "report": report, "applied": False,
                "skipped": True}
    # Same volume as the output, deliberately. Writing it to the local
    # scratch would make the final swap a cross-volume copy of the whole
    # film - 13 GB back across the network for a change to the audio.
    dest = out_path.with_suffix(".conformed.mkv")
    log("")
    log("remuxing")
    ok, detail = remux(mkvmerge, out_path, built, dest, log)
    if not ok:
        # do not leave a part-built film sitting next to the real one
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return {"ok": False, "why": "remux failed: %s" % detail}
    bak = out_path.with_suffix(".prewarp.bak")
    try:
        bak.unlink(missing_ok=True)
        out_path.rename(bak)
        shutil.move(str(dest), str(out_path))
    except OSError as e:
        try:
            if not out_path.exists():
                bak.rename(out_path)
        except OSError:
            pass
        return {"ok": False, "why": "could not swap the file in: %s" % e}
    for b in built:
        try:
            Path(b["path"]).unlink()
        except OSError:
            pass
    return {"ok": True, "report": report, "applied": True,
            "backup": str(bak)}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Conform a rendered file's audio to its master's "
                    "timeline, one adjustment a minute")
    ap.add_argument("output")
    ap.add_argument("source")
    ap.add_argument("--step", type=float, default=DEFAULT_STEP)
    ap.add_argument("--src-astream", default="a:0")
    ap.add_argument("--langs", default="eng,en")
    ap.add_argument("--workdir", default=r"S:\temp")
    ap.add_argument("--apply", action="store_true",
                    help="write the conformed file; default is measure only")
    a = ap.parse_args(argv)
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    mm = (shutil.which("mkvmerge")
          or r"C:\Program Files\MKVToolNix\mkvmerge.exe")
    if not ff or not fp:
        print("ffmpeg/ffprobe not found on PATH")
        return 2
    res = conform(ff, fp, mm, a.output, a.source,
                  Path(a.workdir) / "warp", a.step, a.src_astream,
                  tuple(x.strip().lower()
                        for x in a.langs.split(",") if x.strip()),
                  a.apply, print)
    print()
    if not res.get("ok"):
        print("FAILED: %s" % res.get("why"))
        return 1
    if res.get("skipped"):
        print("nothing to do: %s"
              % res.get("why", "already on the master's timeline"))
    elif not res.get("applied"):
        print("measure only. add --apply to write the conformed file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
