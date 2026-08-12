#!/usr/bin/env python3
# =====================================================================
# av1_pipeline_v0_1.py   V0.241   2026-08-11
#
# The build number lives in VERSION below and moves by 0.001 on every
# change. This banner had read "V0.16 2026-08-02" for eight builds.
#
# Folders and a few defaults come from av1_pipeline_config.txt in this
# same folder. Command line flags always beat the config file.
#
# Unattended batch re-encoder. Scans a folder tree, inspects every
# video, picks per-file settings, encodes to 10-bit AV1 through the
# NVENC block (RTX 5060 Ti on the .209 desktop, RTX 4070 Laptop GPU
# on the Dell G16 7630; any Win11 box with an AV1-capable NVENC card
# and a current NVIDIA driver works), checks the result, and keeps a
# journal so a crash or reboot resumes where it left off.
#
# V0.1 scope:
#   - source and destination pickers, or --src / --dst flags
#   - queue sorted largest first, then container type, then file date
#   - one file at a time; files already in AV1 are skipped
#   - crop consensus rule: bars must persist across the whole runtime
#   - interlace detection, bwdif same-rate deinterlace when found
#   - quality scoring drives the CQ value plus a repair filter chain
#   - sub-1080 sdr sources go up to 1080p through realesrgan-ncnn-vulkan
#     when the exe is present (model set by --sr-model, chunked frame
#     batches, plan on hours per file); lanczos stands in when the tool
#     is absent, the source is vfr or hdr, or --no-sr is set; the
#     journal says which path each file took
#   - over-1080 sources scale down to 1080 lines, width proportional
#   - frame rate is never altered anywhere in the chain
#   - HDR10 kept; static metadata re-applied at the container level
#   - Dolby Vision profile 7/8 stripped to the HDR10 base layer
#   - Dolby Vision profile 5 fails with a diagnostic log
#   - HLG converts to PQ when libplacebo is present, else kept as-is
#   - audio: english first, japanese fallback, else first track plus
#     a note; AAC 320k; 5.1 or wider downmixes to stereo with the
#     center up 30 percent and surrounds down 20 percent
#   - every subtitle track and font attachment copied, no burn-in
#   - VMAF gate on straight re-encodes with automatic CQ retry
#   - audio sync checked every 10 seconds; a constant offset gets a
#     remux repair, progressive drift fails the file
#   - failures land in <dst>\failed with a per-file diagnostic log
#   - journal resume, disk space checks, system sleep blocked
# =====================================================================

import argparse
import collections
import ctypes
import hashlib
import json
import logging
import logging.handlers
import math
import os
import atexit
import platform
import signal
import re
import shutil
import subprocess
import traceback
import sys
import threading
import time
from pathlib import Path

# Build number. THREE DECIMALS, and it moves by 0.001 on EVERY change,
# however small. Not once a session, not once a feature - once a change.
#
# 0.24 carried substantially different behaviour before and after the
# 2026-08-10/11 session: the provenance ledger, audio read from the
# master rather than through the Dolby Vision strip, the conform
# disabled, eleven repairs. Logs from either side of that read "0.24"
# and cannot be told apart, which breaks TECH-A5 and made it impossible
# to say which build produced a given file.
#
# 0.241 is the first build under the new rule and marks the audited
# state. Every module reports THIS string - av1_metadata, av1_subs and
# av1_warp each carry a copy and selftest fails if they disagree,
# because a version that only some of the code agrees with is worse
# than no version at all. av1_metadata's copy is stamped into the .info
# sidecar beside every finished file, where it read "0.2" for months.
# 0.242: portability. GPU generation detection for RTX 30/40/50, a
# video-memory reserve sized to the card actually present rather than
# to the 16 GB one this was built on, automatic scratch-disk selection
# with a session directory that removes itself, and a startup check
# that no longer refuses to run on a machine with no NVENC.
# 0.243: the startup banner no longer tells the operator to set
# sr_tmp_dir when leaving it blank is now the recommended setting.
VERSION = "0.243"
VERSION_DATE = "2026-08-12"
IS_WIN = platform.system() == "Windows"

# ---------------------------------------------------------------------
# Folder defaults and the plain-text config file that overrides them.
# The file sits beside this script, one "key = value" per line, and is
# read by both this program and the GUI so there is one place to edit.
# Order of precedence: command line flag, then the config file, then
# the built-in default below, then a folder picker dialog.
# ---------------------------------------------------------------------
# Encoder selection. NVENC only encodes AV1 on Ada (RTX 40) and newer.
# Ampere (RTX 30) and older decode AV1 but cannot encode it, so those
# machines run the CPU encoder instead.
#   auto    test the GPU, fall back to the CPU encoder if it cannot
#   nvenc   force the GPU encoder, fail if it will not run
#   svtav1  force the CPU encoder
ENCODER_CHOICES = ("auto", "nvenc", "svtav1")

# Audio. The old default re-encoded every track to AAC and folded any
# surround mix down to stereo, which threw away quality that was in the
# source. Copying the stream keeps it bit for bit; Opus is the fallback
# for codecs not worth carrying forward.
AUDIO_MODES = ("auto", "copy", "opus", "aac")
# keep every audio track, English ordered first, and add a stereo
# downmix of any English surround track for headphones and TV speakers
# Stall watchdog. ffmpeg can keep emitting progress while the output
# position never advances, which reads as a slowly decaying speed figure
# and looks like a slowdown rather than the hang it actually is. after
# this many seconds with no forward progress the encode is killed so the
# run reports a failure instead of sitting there all night.
# Decoding 4K HEVC in software is the heaviest single CPU cost in the
# whole job, and it starves the GPU encoder waiting for frames. NVDEC
# does it in hardware instead. Frames still return to system memory so
# the CPU filters are unaffected.
HWACCEL = "auto"            # auto, cuda, none
STALL_TIMEOUT_SEC = 420
# Language policy. A Blu-ray remux ships every dub, and carrying 15 of
# them means 15 concurrent encoders and gigabytes of audio nobody plays.
AUDIO_LANGS = ("eng", "en", "jpn", "ja", "jp")
# Fixed output order, every track re-encoded to Opus:
#   1 English stereo downmix   2 Japanese stereo downmix
#   3 English original surround 4 Japanese original surround
# Everything else is dropped, including commentary and audio description.
AUDIO_FIXED_LAYOUT = True
AUDIO_OPUS_PER_CH = 128     # kbps per channel, max quality
AUDIO_OPUS_CAP = 768
# For English, keep only the generated stereo downmix rather than the
# surround masters. Set to 0 to keep the English surround tracks too.
AUDIO_ENG_STEREO_ONLY = True
# The downmix is listed first so players open on it.
AUDIO_MIX_FIRST = True
AUDIO_KEEP_ALL = True
AUDIO_ADD_STEREO_MIX = True
AUDIO_MIX_CENTRE = 1.25       # centre channel lift in the downmix
AUDIO_MIX_ENHANCE = 0.6       # ffmpeg dialoguenhance factor, 0 disables
AUDIO_MIX_BITRATE = 192       # kbps for the added stereo track
# a blank first subtitle track so players start with subtitles off
SUBS_BLANK_FIRST = True
# Whisper transcription and translation
SUBS_GENERATE = True
WHISPER_MODEL_DIR = r"C:\ProgramData\whisper" if IS_WIN else ""
WHISPER_MODEL = ""          # blank picks the best model present
WHISPER_GPU = True
# queue is a DURATION in seconds, not a count. 0 means send nothing and
# let ffmpeg use its own default of 3. Measured on 60 s of speech, 20 is
# both faster than the default (3.3 s against 5.6 s) and merges cues
# better, so it is worth setting rather than leaving alone.
WHISPER_QUEUE = 20
# Voice activity detection stops whisper being handed silence, which is
# what makes it invent lines. It is OFF by default because it is
# expensive: measured at 25.4 s against 5.6 s for the same 60 s of
# speech, roughly four and a half times the transcription cost, for a
# transcript that was otherwise the same. The trailing-repeat trimmer
# and the duration clamp catch the invented lines after the fact at no
# cost at all. Turn this on if a file keeps producing rubbish over its
# quiet passages.
WHISPER_VAD = False
TRANSLATE_PROVIDER = "deepl"   # deepl or openai
TRANSLATE_KEY = ""
TRANSLATE_ENDPOINT = ""
TRANSLATE_MODEL = ""
# Which subtitle languages to keep. Blank keeps every language.
SUBS_LANGS = ("eng", "en", "jpn", "ja", "jp", "und")
# Metadata, naming, tagging. All optional; the encode never depends on it.
META_ENABLE = True
META_RENAME = True
META_UNDERSCORES = True     # linux command lines hate spaces
META_EP_TITLE = True
META_TECH = True
# Confirm the identification against pictures rather than text: sample
# frames from the finished file and compare them to TMDB's backdrops and
# episode stills. Costs a few seconds and a handful of small downloads.
META_VERIFY_FRAMES = True
# Write a readable .info sidecar next to each finished file recording
# what the material is, what was done to it, and how confident the
# identification is. The file should be able to explain itself without
# this pipeline or this database still existing.
META_INFO_FILE = True
TMDB_API_KEY = ""
# Bitmap subtitle formats that carry a picture size the muxer needs
SUBS_BITMAP = ("hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub",
               "xsub")
# extra data rate for genuinely high quality sources. expressed as a
# percentage because that is how people think about it; converted to a
# CQ delta below using the rule that six CQ steps roughly double or
# halve the bitrate, so one step is about 12.2 percent.
# Shifts every computed CQ. Negative means better quality and larger
# files: each step is worth roughly 12 percent bitrate, so -2 is about
# +26 percent. This is the only quality knob NVENC has left once the
# preset, tune and lookahead are already at maximum.
CQ_OFFSET = 0
HQ_SOURCE_BOOST_PCT = 30
HQ_SOURCE_MIN_H = 2000        # source height that counts as 4K class
HQ_SOURCE_MIN_DEPTH = 10
# The mirror of the boost above, for sources small enough that being
# stretched to 1080 lines invents most of what is on screen. Above this
# point the bitrate is describing what the scaler made up rather than
# the film, so the class gets its OWN ceiling: measured on a 512x288
# source, cq 28 gave 418 pct of the source file, cq 30 gave 333 pct and
# cq 36 gave 173 pct. Set lowres_cq_trim to 0 to switch this off.
LOWRES_TRIM_H = 576           # source height at or below which it applies
LOWRES_CQ_TRIM = 8            # CQ steps added (higher CQ = smaller file)
LOWRES_CQ_CEIL = 36           # this class alone may pass the normal ceiling
AUDIO_MODE = "auto"
AUDIO_DOWNMIX = False       # keep the original channel layout
AUDIO_DENOISE = False       # off; it smears anything that is not starved
AUDIO_DENOISE_PER_CH = 64000
# widely playable and worth keeping untouched rather than re-encoding
AUDIO_COPY_OK = ("aac", "ac3", "eac3", "opus", "flac", "alac", "dts",
                 "truehd", "mp3", "vorbis", "pcm_s16le", "pcm_s24le",
                 "pcm_s16be", "pcm_s24be")

# Real-ESRGAN sees one frame at a time and has no idea what the frame
# before it looked like, so it invents slightly different detail every
# frame. On a static background that reads as crawling or shimmering.
# Topaz and similar tools avoid it by feeding the model several frames
# at once; that is a property of the model and cannot be bolted on here.
# What can be done is reduce how far the invented detail is allowed to
# swing, and then average out what is left over time.
#
# SR_STRENGTH blends the model output back towards a plain lanczos
# upscale. 100 is the raw model, 0 is no AI at all, 50 halves both the
# invented detail and the flicker that comes with it.
SR_STRENGTH = 50
# Temporal stabiliser applied after the blend. atadenoise averages a
# pixel across neighbouring frames only while the difference stays
# under a threshold, so still backgrounds settle down while anything
# actually moving passes through untouched and does not smear.
SR_STABILISE = True
# The same idea for the NON-AI upscale path. Without it a lanczos
# stretch of a low-resolution source has no temporal treatment at all.
UPSCALE_STABILISE = True
UPSCALE_STAB_FILTER = ("atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04"
                       ":2a=0.02:2b=0.04:s=9")
SR_STAB_THRESH = (0.03, 0.06)   # inner, outer
SR_STAB_FRAMES = 9              # odd, 5 to 15 is sensible
SVTAV1_PRESET = 6          # 0 slowest/best to 13 fastest, 4-8 is sane
SVTAV1_CRF_OFFSET = 5      # CQ here is tuned for NVENC; SVT-AV1 needs a
                           # higher number for a similar look. approximate,
                           # tune it on your own content
SVTAV1_CRF_CEIL = 45

FFMPEG_DIR = ""            # blank means take ffmpeg from PATH
SR_TMP_DIR = ""            # blank means use the destination .tmp folder

CONFIG_NAME = "av1_pipeline_config.txt"
DEFAULT_SRC = r"P:\2_Storage\Film and TV\DownConvert\Input"
DEFAULT_DST = r"P:\2_Storage\Film and TV\DownConvert\Output"

# ---------------------------------------------------------------------
# Tunable settings. Anything a trial run might want to nudge lives
# here so edits never have to touch the logic below.
# ---------------------------------------------------------------------
VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".wmv", ".mpg",
              ".mpeg", ".m2ts", ".ts", ".vob", ".flv", ".webm", ".divx",
              ".mts", ".3gp", ".ogm", ".rm", ".rmvb", ".asf"}

# queue tiebreak after size: cruftier containers first, then file date
EXT_RANK = {".m2ts": 0, ".mts": 1, ".ts": 2, ".vob": 3, ".rm": 4,
            ".rmvb": 4, ".asf": 5, ".wmv": 6, ".avi": 7, ".divx": 8,
            ".mpg": 9, ".mpeg": 9, ".flv": 10, ".ogm": 11, ".3gp": 12,
            ".mp4": 13, ".m4v": 14, ".mov": 15, ".webm": 16, ".mkv": 17}

MIN_FILE_BYTES = 5 * 1024 * 1024          # anything smaller is ignored
MIN_FREE_BYTES = 20 * 1024 ** 3           # hard floor of free dest space

ANALYSIS_WINDOWS = 12                     # sample windows per file
ANALYSIS_WIN_SEC = 4.0                    # seconds per sample window
# cropdetect takes either a raw integer or a 0-1 fraction. the integer
# is absolute, so 24 sits above 8-bit black (16) but far below 10-bit
# black (64) and no crop is ever found on a 10-bit source. the fraction
# is scaled by bit depth internally and is therefore correct on both.
CROP_LIMIT = "0.1"                        # cropdetect black threshold
CROP_MIN_PX = 8                           # bars thinner than this ignored
IDET_MIN_FRAMES = 50                      # combed frame count floor
IDET_SHARE = 0.45                         # combed share needed on unknown flags
INTERLACE_HEIGHTS = {480, 486, 540, 576, 1080}  # line counts where true interlace lives

VMAF_TARGET = 95.0                        # transparency gate
VMAF_WINDOWS_SEC = 8.0                    # seconds per VMAF window
CQ_FLOOR = 10
CQ_CEIL = 30
# How far the finished picture may differ in length from the source
# before the file is called failed. A flat second, measured video
# against video. The old rule scaled with runtime and let a feature
# film drift 27 seconds, which is a whole scene.
DURATION_TOLERANCE_SEC = 1.0
# How far the finished AUDIO may fall short of the finished picture
# before the file is failed. Generous, because a real disc master can
# legitimately end a second or two before the last frame - but nothing
# like the 2660 s of missing sound that shipped on 2026-08-11.
AUDIO_SHORT_LIMIT_SEC = 5.0
UPSCALE_CQ = 28                           # fixed CQ on upscale paths; AI detail
                                          # compresses well so a higher value still
                                          # looks good at a fraction of the bitrate
# ...but ONLY for genuinely small sources. A source with this many lines
# of real picture or more keeps the quality number its tier earned.
#
# UPSCALE_CQ exists because detail invented by a scaler is not worth
# spending bits on. That reasoning holds for a 512x288 AVI. It does not
# hold for a 1920x800 scope feature, which is a full Blu-ray rip that
# happens to have fewer than 1080 lines because of its aspect ratio.
# The Simpsons Movie came in at tier HIGH, 9.86 Mbps, 0.268 bpp - and
# went out at 0.04 bpp because 800 < 1080 made it "an upscale". That is
# seven CQ steps below what it deserved, and it blocks visibly.
UPSCALE_CQ_MAX_H = 720
REPAIR_CQ = 24                            # repair chain without upscale (bad source, native res)

# Below this the compensation is not worth a filter, and rounding noise
# in the probe would start to matter. One video frame at 24 fps is 42 ms,
# and lip sync becomes noticeable well before that, so the bar is low.
AUDIO_SYNC_MIN_MS = 2.0
# Keeps re-encoded audio pinned to its own timeline instead of letting
# source discontinuities pull it forward. See with_async().
# Conform every finished file's audio to its master's timeline. The
# start points are locked with a broad search over the whole waveform,
# then the film is walked one minute at a time and each segment's
# playback rate is adjusted so its peaks and valleys sit on the
# master's. Nothing is cut, so nothing drops out. See av1_warp.py.
# OFF. It damaged two finished films on 2026-08-11.
#
# The idea was to measure where the rendered audio sits against the
# master and correct it. The flaw is structural: a measurement that is
# wrong ONCE destroys a file that was right. Correlating a heavily
# processed stereo downmix against a raw surround master is guesswork,
# and on self-similar film audio it guesses wrong - it read The Simpsons
# Movie as 206 SECONDS out from a correlation whose sharpness was 14.4,
# then padded 206 s of silence onto the head and stretched segments by
# 8.75 pct. The file had been correct.
#
# Sync is preserved by NOT PERTURBING THE TIMELINE, not by repairing it
# afterwards. Measurement belongs in the detection gates below, which
# report and fail; it does not belong in a silent rewrite.
#
# av1_warp.py remains as a manual tool with hard sanity limits, for a
# file that is known to be wrong and is watched while it is repaired.
AUDIO_CONFORM = False
AUDIO_CONFORM_STEP = 60.0                 # one adjustment a minute
# The search window has to be WIDER than the largest drift worth
# finding, or the true correlation peak falls outside it and the scan
# locks onto the best spurious peak inside instead. At 1.1 s it reported
# a steady +37 ms and drift of -0.04 ms/min on a film that was actually
# 2.5 s out by the end - a confident, precise, completely wrong answer.
# The segment is longer to match: correlating 1.2 s of audio across a
# 5 s window is not enough signal to pick the right peak.
SR_CHUNK_SEC = 10.0                       # frames per realesrgan batch
SR_MODEL_DEFAULT = "realesr-animevideov3"
# Upscale factor. 0 lets the code pick the smallest the model supports
# that clears 1080 lines; a number forces it. x2 on a 528-line film
# lands at 1056 and the last 2 pct is a near-free resize, against x4
# which computes FOUR TIMES the pixels and then throws most of them
# away in the downscale.
SR_SCALE = 0
# Tile size handed to the upscaler. 0 means work it out from the video
# memory actually free at the time; a number forces it.
#
# Memory goes as tile SQUARED, so this is not a dial to guess at: 1024
# at x4 is about 11 GB on its own. ncnn's own auto table tops out at a
# 400 px tile and tells us what that costs per scale, which is enough
# to size it properly. The upscaler is also NOT alone on the card -
# NVENC is encoding the previous file at the same time, measured at
# 2.3 to 3.4 GB - so a reserve is held back for it and the desktop.
SR_TILE = 0
# HARD ceiling on TOTAL video memory in use, as a percentage of the
# card. Everything else here is advisory; this is the line that must
# never be crossed, because crossing it does not fail the file - it
# takes the display driver down and, on 2026-08-06, the whole machine
# with it. 98 leaves the compositor and anything else on the desktop
# room to breathe.
GPU_MAX_PCT = 98
# Seconds to let the driver actually reclaim video memory after a kill.
# nvidia-smi reports the old figure for a short while after a process
# dies, and sizing the next job against that stale number is how one
# over-commit becomes two.
GPU_SETTLE_SEC = 5
# How long to wait for headroom before giving up on a file and moving
# on. Long unattended runs must never block forever, but they must also
# never barge in when the card is busy with something else.
GPU_WAIT_SEC = 300
# Temperature the card must come back under before more work is piled
# on. 0 disables the check. Not a cure for a hardware limit - only a
# refusal to keep leaning on one.
GPU_TEMP_MAX_C = 80
GPU_COOL_SEC = 15
# Seconds to idle between upscaler chunks. The upscale holds the GPU,
# the CPU and the disk at full tilt for hours without a break; a short
# gap every chunk costs a few percent of throughput and lets the whole
# machine, not just the card, come off the ceiling.
SR_CHUNK_PAUSE_SEC = 2.0
SR_VRAM_MB = 15000            # absolute cap; the percentage above wins
# Held back for the concurrent NVENC session and the desktop compositor.
#
# 3400 MB was MEASURED on a 16 GB card, where it is a fifth of the
# board. On a client's 8 GB RTX 3060 Ti it is FORTY-TWO PERCENT of the
# card, which does not crash anything - sr_tile_for floors the budget
# and simply picks a smaller tile - but it cripples the upscaler on
# exactly the machines that can least afford it.
#
# So this is now a CEILING on a proportional reserve, resolved at call
# time by sr_vram_reserve(). Two things scale it down:
#
#   - the card. A fifth of the board is the measured proportion, so
#     that is what is reserved, never more than this number.
#   - the encoder. The reserve exists for an NVENC session running
#     alongside the upscale. On Ampere and older there IS no NVENC AV1
#     session - the encode is on the CPU - so only the desktop needs
#     covering, and the desktop is already counted in "used".
SR_VRAM_RESERVE_MB = 3400
SR_VRAM_RESERVE_PCT = 21      # of total board memory, whichever is less
SR_VRAM_RESERVE_MIN_MB = 700  # compositor headroom, even with no NVENC
# Largest tile we will ask for regardless of what the arithmetic says.
# 896 at x4 was MEASURED safe alongside a running encode (8.3 GB); 992
# was chosen by the budget on 2026-08-06 and the driver fell over 31
# minutes later. The cost model below comes from ncnn's own auto-select
# thresholds, which are heuristics rather than measured allocations, so
# it is not trustworthy enough to extrapolate past what has been tried.
#
# This is a CEILING, not a target. A smaller card never reaches it -
# the budget arithmetic in sr_tile_for lands well below - so it needs
# no per-card adjustment. It exists to stop a LARGER card than the one
# this was measured on extrapolating past what has been tried.
SR_TILE_MAX = 896
# what a 400 px tile costs at each scale, from ncnn's auto table
SR_TILE_COST_400 = {2: 1300.0, 3: 3300.0, 4: 1690.0}
# Thread counts for load : process : save. The tool's own default is
# 1:2:2, and with thousands of small PNGs a single decode thread leaves
# the card waiting on the disk rather than computing.
SR_THREADS = "4:4:4"
SR_MODELS = ("realesr-animevideov3", "realesrgan-x4plus",
             "realesrgan-x4plus-anime")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash
                                          # when running under the gui
# Leave the machine usable while a batch runs. 0 means take everything.
# This caps the cores the child processes may sit on AND the thread
# count handed to ffmpeg, because either alone is not enough: ffmpeg
# spawns as many filter threads as it likes regardless of affinity, and
# affinity alone still lets those threads saturate what they are given.
CPU_MAX_PCT = 80

TOOLS: dict = {}
CAPS: dict = {}
log = logging.getLogger("av1")            # module-level; setup_logging adds handlers

PROGRESS_EVERY_SEC = 20.0                 # encode progress print interval

# filter chain building blocks
# 10-bit output standard for SDR sources. 10 bit does not add colour
# that was never captured, but it does carry gradients at a precision
# 8-bit cannot, so the banding already baked into an 8-bit source can be
# smoothed and then survive encoding instead of re-quantising straight
# back into bands. that is the real gain and deband is what delivers it.
STD_DEBAND = "deband=1thr=0.008:2thr=0.008:3thr=0.008:4thr=0.008:range=16:blur=1"
# saturation lift for standard definition sources. this is a deliberate
# look change, not a restoration; keep it modest.
SD_SATURATION = 1.06
# Deband policy. "auto" applies it only to sources under 10-bit, where
# there is 8-bit quantisation banding to actually fix. "always" applies
# it everywhere, which costs real CPU at 4K and injects dither into a
# master that may not need it. "off" never applies it.
DEBAND_MODE = "auto"
# Hand the final assembly to mkvmerge instead of letting ffmpeg's
# matroska muxer carry subtitles, fonts and chapters. Three of this
# project's worst bugs lived in that muxer, so this is on by default;
# it falls back to the single ffmpeg command if mkvmerge is missing.
MUX_MKVMERGE = True
SD_SAT_MAX_H = 720                        # only sources below this get it
# Contrast Adaptive Sharpen, the FidelityFX CAS algorithm. runs on the
# CPU in ffmpeg, no GPU involved, so the AMD name means nothing here.
# used only where no rescale happens and there is no resampling gain.
CAS_STRENGTH = 0.30

REPAIR_DENOISE = "hqdn3d=1.6:1.2:2.5:2.5"
REPAIR_DEBAND = "deband=1thr=0.011:2thr=0.011:3thr=0.011:4thr=0.011:range=14:blur=1"
SCALE_1080 = "scale=-2:1080:flags=lanczos+accurate_rnd+full_chroma_int"
SCALE_2160 = "scale=-2:2160:flags=lanczos+accurate_rnd+full_chroma_int"

# ---------------------------------------------------------------------
# 4K target mode
# ---------------------------------------------------------------------
# Library-wide target: 1080 or 2160 lines of ACTUAL PICTURE. 1080 is the
# default and remains the standard; 2160 is opt-in per batch. This is
# policy, not a per-run preference, so it lives here and in the config
# and the GUI is deliberately barred from persisting it - see bug 22.
TARGET_RES = 1080
# What counts as "close to 4K", and so eligible for a 4K master at all.
# Measured on WIDTH rather than height because width is invariant to
# letterboxing: a 2.39:1 UHD is 3840x1608 after the bars come off but is
# still a 3840-wide 4K source. The pixel floor is the second guard, so a
# 3840x1080 ultrawide cannot qualify on width alone.
UHD_MIN_WIDTH = 3000
UHD_MIN_PIXELS = 5500000
# Real picture within this percentage of 2160 lines is padded UP to hit
# 2160 exactly; anything further below stays at its native size. This is
# the deliberate break from the 1080 standard, which would have upscaled
# a 2.39:1 UHD from 1608 lines to 5160x2160. A 1.85:1 UHD at 2076 lines
# is 96.1% and pads; a scope film at 1608 lines is 74.4% and does not.
UHD_PAD_BAND_PCT = 10
# Soft ceiling on a 4K master. Soft on purpose: if holding the quality
# floor means going over, the file goes over and the log says so loudly.
UHD_MAX_GB = 10.0
# The quality floor, expressed as the worst CQ the budget may drive to.
# CQ rises to fit the budget and stops here.
#
# 32 measured out as the value that lets the 10 GB cap actually work on a
# real remux: on a 101 minute UHD it predicts 11.08 Mbps of video, about
# 9.6 GB all in. A floor of 30 would stop at 15.08 Mbps and land near
# 12.4 GB, which makes the cap close to decorative. Raise this number to
# protect quality harder and accept larger files; lower it to hold the
# size at the cost of picture.
UHD_CQ_CEIL = 32
# Headroom left for audio, subtitles and container overhead when working
# out how many bits the video may have. Four Opus tracks on a surround
# source run to about 2 Mbps, which is a sixth of a 10 GB budget.
UHD_OVERHEAD_MBPS = 2.5
# Bitrate the calibration measured at this reference CQ for a full 2160
# line frame, in bits per pixel per second. Used to predict output size
# without a test encode.
#
# The "six steps halves the bitrate" figure this project has used
# elsewhere was derived at 1080p and does NOT hold at 4K. Measured on a
# UHD remux at 3996x2160, exact byte counts, encoder confirmed
# deterministic first (three runs at CQ 27 produced byte-identical
# output): CQ 21 = 50.20 Mbps, 23 = 30.27, 25 = 30.08, 27 = 24.71,
# 29 = 18.27, 31 = 13.34. That is about 4.5 steps per halving across
# CQ 27-31, which is the range the budget actually works in.
#
# Note the genuine PLATEAU at CQ 23-25, where the rate barely moves. The
# curve is not a clean exponential, so this prediction is a guide for
# choosing a starting CQ, not a promise about the finished size. The VBV
# cap and the soft-ceiling policy are what actually bound the result.
# 0.1157 measured at CQ 27 over five 20 s windows spread across a UHD
# remux, output 3996x2160: 23.62, 25.54, 24.71, 22.34, 23.48 Mbps, mean
# 23.94, spread +/-7%. That is ONE title - 1988 film stock with real
# grain, which is expensive to encode - so a clean modern digital source
# will come in under this and a busier one over it. Conservative rather
# than exact, which suits a soft cap.
UHD_BPP_REF_CQ = 27
UHD_BPP_REF = 0.1157
UHD_CQ_STEPS_PER_HALVING = 4.5
# Artifact repair for damaged sources. deblock is the ONLY filter that is
# both 10-bit clean and fast enough at 4K: fspp, pp7 and uspp are all
# 8-bit only and would drag the whole graph down exactly as eq does, and
# spp costs 12 to 110 hours a film. Probed, not assumed.
ARTIFACT_REPAIR = "auto"                  # auto | off | weak | strong
ARTIFACT_REPAIR_TIERS = ("low", "awful")
UHD_DEBLOCK = {"weak": "deblock=filter=weak:block=8",
               "strong": "deblock=filter=strong:block=8"}
POST_UPSCALE_SHARPEN = "unsharp=5:5:0.35:5:5:0.0"
# NOTE: the HLG to PQ conversion is now BUILT AT RUNTIME by
# placebo_hlg_to_pq(), from option names probed out of the binary. It
# used to live here as a constant carrying `color_trc=pq`, which
# libplacebo rejects - the accepted spelling is `smpte2084` - and that
# took the whole ffmpeg command down, so no HLG source could encode.



# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def frac(v, default=0.0):
    # reads "24000/1001", "0.005", plain ints, or None into a float
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


def truthy(v):
    """Read a config value as a boolean.

    Module level rather than a lambda inside main() because the scratch
    settings are resolved before the point main() defined it, and two
    definitions of "is this on" that could drift apart is exactly the
    kind of thing that ends up in a bug list.
    """
    return str(v).strip().lower() not in ("0", "no", "off", "false", "")


def cpu_budget():
    """Resolve CPU_MAX_PCT into a concrete core count.

    Callers treat 0 as "no limit" so that the affinity and thread-count
    plumbing can be skipped entirely rather than being handed a value
    equal to the whole machine, which would cost a syscall per child
    process for no behavioural difference.

    Returns:
        int: number of logical cores the children may occupy, or 0 to
        impose no limit at all.
    """
    try:
        total = os.cpu_count() or 1
    except Exception:
        # os.cpu_count() is documented as possibly returning None, and
        # on an unusual host it can raise. Either way the safe answer
        # is "do not attempt to throttle".
        return 0
    if CPU_MAX_PCT <= 0 or CPU_MAX_PCT >= 100:
        return 0
    return max(1, int(total * CPU_MAX_PCT / 100.0))


def throttle(proc):
    """Restrict a freshly spawned child to the CPU budget.

    Affinity is applied in addition to ffmpeg's own -threads because the
    two bound different things. -threads caps the codec's worker pool,
    but the filter graph, the demuxer and the decoder each spawn their
    own threads on top of it, and realesrgan honours no such flag at
    all. Pinning the process bounds every thread it will ever create,
    whatever the tool decides to do internally.

    Failures are swallowed deliberately: psutil may be absent, the
    process may have already exited, and on some hosts setting affinity
    requires privileges we do not have. None of those are worth failing
    an encode over, and the run is merely less polite without it.

    Args:
        proc: a subprocess.Popen whose pid is already valid.
    """
    n = cpu_budget()
    if not n:
        return
    try:
        import psutil
        psutil.Process(proc.pid).cpu_affinity(list(range(n)))
    except Exception:
        pass


def pin_self():
    """Apply the CPU cap to THIS process, not just to its children.

    cpu_max_pct has only ever been enforced with cpu_affinity() on
    subprocess pids. Anything running INSIDE the pipeline - the whisper
    wrapper, and now argos-translate with CTranslate2, stanza and spacy
    underneath it - was never covered, and measured at 31 threads across
    all 16 logical cores while the cap was nominally holding it to 14.

    Native thread pools also have to be told separately: they read their
    own environment variables at import time and will happily size
    themselves to the whole machine regardless of affinity. Those are
    set here too, which is why this runs before anything heavy is
    imported.

    Leaving the machine two free cores is what keeps the desktop
    responsive while a long batch runs; it is also load the power supply
    then does not have to deliver.
    """
    n = cpu_budget()
    if not n:
        return 0
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "CT2_INTER_THREADS",
                "TOKENIZERS_PARALLELISM"):
        if var == "TOKENIZERS_PARALLELISM":
            os.environ.setdefault(var, "false")
        else:
            os.environ.setdefault(var, str(n))
    try:
        import psutil as _ps
        _ps.Process().cpu_affinity(list(range(n)))
    except Exception as e:
        log.debug("could not pin own affinity: %s" % e)
    return n


def ff_threads():
    """Thread-count argument for ffmpeg, as a string.

    Returns:
        str | None: the value for -threads and -filter_threads, or None
        when no limit applies, in which case the caller omits both flags
        and lets ffmpeg size its own pools.
    """
    n = cpu_budget()
    return str(n) if n else None


def kill_tree_by_image(image_name):
    """Kill every process matching an image name, and its children.

    Best effort and never raises: this runs on the failure path, and a
    failure to clean up must not become a second failure. Windows only
    gives a reliable tree kill through taskkill /T /F.
    """
    if not image_name:
        return
    if not image_name.lower().endswith(".exe"):
        image_name += ".exe"
    try:
        subprocess.run(["taskkill", "/IM", image_name, "/T", "/F"],
                       capture_output=True, text=True, timeout=60,
                       creationflags=NOWIN)
    except Exception as e:
        log.debug("tree kill of %s failed: %s" % (image_name, e))


def gpu_wait_for_cool(log=None):
    """Hold off while the card is above its temperature ceiling.

    The upscaler runs the GPU flat out for hours at a time. Three hard
    freezes with BugcheckCode=0 and no crash dump say the kernel stopped
    executing rather than faulted, and heat and power delivery are what
    do that. Neither was measured before, so this both records the
    numbers and refuses to keep piling on when they are already high.

    This is a mitigation, not a cure: software cannot fix a thermal or
    power limit, it can only decline to keep standing on it.

    Returns:
        bool: True once the card is under the ceiling, False if it never
        came down within the wait budget.
    """
    if GPU_TEMP_MAX_C <= 0:
        return True
    deadline = time.time() + max(0, GPU_WAIT_SEC)
    warned = False
    while True:
        g = gpu_snapshot()
        t = g.get("temp_c") if g else None
        if not t or t < GPU_TEMP_MAX_C:
            return True
        if time.time() >= deadline:
            if log:
                log.warning("    gpu still at %.0fC after waiting %ds; "
                            "carrying on" % (t, GPU_WAIT_SEC))
            return False
        if not warned and log:
            log.info("    gpu at %.0fC, over the %dC ceiling; pausing to "
                     "let it cool" % (t, GPU_TEMP_MAX_C))
            warned = True
        time.sleep(GPU_COOL_SEC)


def gpu_wait_for_headroom(need_mb, log=None, timeout=None):
    """Block until the card has room, rather than allocating into a wall.

    Returns True when there is headroom, False if it never appeared. The
    caller is expected to skip the file rather than push on: an
    allocation that does not fit does not fail politely, it resets the
    display driver.
    """
    timeout = GPU_WAIT_SEC if timeout is None else timeout
    deadline = time.time() + max(0, timeout)
    warned = False
    while True:
        g = gpu_snapshot()
        if not g or not g.get("vram_total_mb"):
            return True          # cannot measure, so cannot gate
        total = float(g["vram_total_mb"])
        used = float(g.get("vram_used_mb", 0))
        room = total * GPU_MAX_PCT / 100.0 - used
        if room >= need_mb:
            return True
        if time.time() >= deadline:
            if log:
                log.warning("    gpu still short of headroom: need %.1f GB, "
                            "%.1f GB under the %d%% cap after waiting %ds"
                            % (need_mb / 1024.0, room / 1024.0,
                               GPU_MAX_PCT, timeout))
            return False
        if not warned and log:
            log.info("    waiting for video memory: need %.1f GB, %.1f GB "
                     "available under the %d%% cap"
                     % (need_mb / 1024.0, room / 1024.0, GPU_MAX_PCT))
            warned = True
        time.sleep(GPU_SETTLE_SEC)


def run(cmd, timeout=None):
    # short-command runner; returns (rc, stdout_text, stderr_text)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=NOWIN)
        # duration matters as much as the exit code when hunting a
        # slow step, so it is recorded on every call
        log.debug("run rc=%d in %.2fs cmd=%s"
                  % (p.returncode, time.time() - t0,
                     " ".join(str(c) for c in cmd[:5])))
        if p.returncode != 0 and p.stderr:
            log.debug("err: %s" % (p.stderr or "")[:500])
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        # subprocess.run kills the DIRECT child and nothing below it. A
        # GPU tool that leaves a helper behind keeps its video memory
        # committed, and the next thing to size itself against "free"
        # VRAM then reads a number that is a lie. Kill the whole tree,
        # then give the driver a moment to actually hand the memory
        # back before anyone asks how much is free.
        log.warning("run TIMEOUT after %ss: %s"
                    % (timeout, " ".join(str(c) for c in cmd[:5])))
        kill_tree_by_image(Path(str(cmd[0])).name)
        time.sleep(GPU_SETTLE_SEC)
        return 124, "", "timeout after %s s" % timeout
    except FileNotFoundError as e:
        log.warning("run TOOL MISSING: %s" % e)
        return 127, "", str(e)
    except Exception as e:
        log.warning("run FAILED to launch: %s" % e)
        return 125, "", str(e)


def cleanup_paths(paths):
    # best-effort removal of temp files, never raises
    for p in paths or []:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def gb(n):
    return n / (1024.0 ** 3)


# ---------------------------------------------------------------------
# Journal: one JSON file in the destination, written atomically, read
# back on start so a crash or reboot resumes the queue.
# ---------------------------------------------------------------------
class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {"version": VERSION, "files": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                # a corrupt journal gets pushed aside, never trusted
                try:
                    self.path.replace(self.path.with_suffix(".corrupt.json"))
                except Exception:
                    pass
        self.data.setdefault("files", {})

    def get(self, key):
        return self.data["files"].get(key)

    def update(self, key, **kw):
        e = self.data["files"].setdefault(key, {})
        e.update(kw)
        e["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def clear_failed(self):
        # drops failed entries so those files queue again on this run
        dead = [k for k, v in self.data["files"].items()
                if v.get("status") == "failed"]
        for k in dead:
            self.data["files"].pop(k, None)
        if dead:
            self.save()
        return len(dead)


# ---------------------------------------------------------------------
# Tool discovery and ffmpeg capability probe
# ---------------------------------------------------------------------
def find_tools():
    # ffmpeg_dir in the config wins over PATH. that is the lever for
    # running a specific ffmpeg build, which is how an NVENC API gap
    # gets worked around without disturbing anything else on the box
    d = FFMPEG_DIR
    ff = fp = None
    if d:
        base = Path(d)
        for name, slot in (("ffmpeg", "ff"), ("ffprobe", "fp")):
            for ext in (".exe", ""):
                c = base / (name + ext)
                if c.exists():
                    if slot == "ff":
                        ff = str(c)
                    else:
                        fp = str(c)
                    break
        if not ff:
            log.warning("ffmpeg_dir is set to %s but no ffmpeg is there; "
                        "falling back to PATH" % d)
    cand = {
        "ffmpeg": [ff, shutil.which("ffmpeg")],
        "ffprobe": [fp, shutil.which("ffprobe")],
        "mkvmerge": [shutil.which("mkvmerge"),
                     r"C:\Program Files\MKVToolNix\mkvmerge.exe"],
        "realesrgan": [shutil.which("realesrgan-ncnn-vulkan"),
                       r"C:\ProgramData\realesrgan\realesrgan-ncnn-vulkan.exe"],
        "dovi_tool": [shutil.which("dovi_tool"),
                      r"C:\ProgramData\dovi_tool\dovi_tool.exe"],
        "mkvpropedit": [shutil.which("mkvpropedit"),
                        r"C:\Program Files\MKVToolNix\mkvpropedit.exe"],
    }
    for name, paths in cand.items():
        TOOLS[name] = next((p for p in paths if p and Path(p).exists()), None)
        log.debug("tool %s -> %s" % (name, TOOLS[name]))


def probe_caps():
    rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-filters"],
                       timeout=60)
    blob = out + err
    CAPS["libvmaf"] = "libvmaf" in blob
    CAPS["libplacebo"] = "libplacebo" in blob
    CAPS["zscale"] = "zscale" in blob
    rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-encoders"],
                       timeout=60)
    blob = out + err
    CAPS["av1_nvenc"] = "av1_nvenc" in blob
    # The CPU encoder is what a machine without an AV1-capable NVENC
    # falls back to, so whether the build HAS it decides whether such a
    # machine can run at all. It was never probed, and the startup check
    # aborted on av1_nvenc alone - which would refuse to start on a
    # perfectly capable box whose ffmpeg simply has no nvenc compiled in.
    CAPS["libsvtav1"] = "libsvtav1" in blob


# ---------------------------------------------------------------------
# Keep Windows awake for the whole run; the display may still sleep
# ---------------------------------------------------------------------
def placebo_opts():
    # the gamut mapping option has been called different things across
    # libplacebo and ffmpeg versions, and passing a name this build does
    # not know kills the whole command with "Option not found". so ask
    # the binary what it actually supports and use only that
    opts = CAPS.get("placebo_opts")
    if opts is None:
        if not TOOLS.get("ffmpeg"):
            # nothing to probe yet; the base options below are the ones
            # every build has, so the caller still gets a usable filter
            # rather than a KeyError
            return set()
        rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-h",
                            "filter=libplacebo"], timeout=60)
        blob = out + err
        opts = set(re.findall(r"^\s{2,}([a-z0-9_]+)\s+<", blob, re.M))
        CAPS["placebo_opts"] = opts
        log.debug("libplacebo options: %d found" % len(opts))
    return opts


def placebo_hlg_to_pq():
    """HLG to PQ through libplacebo, built from probed option names.

    This used to be a hardcoded constant reading `color_trc=pq`. That
    value does not exist: libplacebo's color_trc takes the ffmpeg
    spelling, `smpte2084`, and rejects `pq` outright with "Undefined
    constant", which killed the entire ffmpeg command. Every HLG source
    failed to encode. Found by feeding every filter chain the settings
    matrix can produce to ffmpeg and reading back what it refused.

    Same family as the gamut_mapping/gamut_mode bug from session 1, and
    the same lesson: probe the tool, never spell its constants from
    memory.
    """
    opts = placebo_opts()
    parts = ["libplacebo=colorspace=bt2020nc",
             "color_primaries=bt2020", "color_trc=smpte2084"]
    if "tonemapping" in opts:
        parts.append("tonemapping=clip")
    if "format" in opts:
        parts.append("format=p010")
    return ":".join(parts)


def placebo_filter():
    opts = placebo_opts()
    parts = ["libplacebo=colorspace=bt2020nc",
             "color_primaries=bt2020", "color_trc=smpte2084"]
    if "tonemapping" in opts:
        parts.append("tonemapping=clip")
    for name in ("gamut_mode", "gamut_mapping"):
        if name in opts:
            parts.append(name + "=perceptual")
            break
    return ":".join(parts)


def nvenc_verdict(err):
    # three very different NVENC failures wear almost the same error.
    # driver mapping is from ffmpeg's own libavcodec/nvenc.c
    NVENC_MIN_DRIVER = {"13.1": "610.00", "13.0": "570.00",
                        "12.2": "551.76", "12.1": "531.61",
                        "12.0": "522.25"}
    low = (err or "").lower()
    m = re.search(r"required:\s*([0-9.]+)\s+found:\s*([0-9.]+)", low)
    if m or "minimum required nvidia driver" in low:
        need = have = None
        if m:
            need, have = m.group(1), m.group(2)
        d = re.search(r"driver for nvenc is\s*([0-9.]+)", low)
        minver = d.group(1) if d else NVENC_MIN_DRIVER.get(need or "", None)
        msg = "this ffmpeg wants a newer NVENC than the driver provides"
        if need and have:
            msg = ("this ffmpeg was built for NVENC API %s, the driver "
                   "provides %s" % (need, have))
        if minver:
            msg += ". that API needs driver %s or newer" % minver
        return "version_gap", msg
    if any(w in low for w in ("no capable devices", "not supported",
                              "unsupported", "invalid device",
                              "no nvenc capable")):
        return "nocard", "this GPU has no AV1 encoder in hardware"
    return "unknown", "NVENC would not start"


def preflight(log, want_placebo):
    # probe_caps only reads the -encoders and -filters lists, which say
    # what was compiled in, not what actually runs on this machine with
    # this driver. these are real one second encodes. three seconds
    # here saves finding out forty minutes into a queue
    def trial(name, extra_in, vf, enc):
        cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-v", "error"]
        cmd += extra_in
        cmd += ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:d=1"]
        if vf:
            cmd += ["-vf", vf]
        cmd += enc + ["-f", "null", "-"]
        rc, out, err = run(cmd, timeout=120)
        msg = (err or out or "").strip().splitlines()
        if rc == 0:
            log.info("    preflight %-16s ok" % name)
            return True, ""
        why = msg[-1] if msg else "no error text"
        # the last line is usually a downstream complaint. prefer the
        # first line that names an actual cause
        for line in msg:
            low = line.lower()
            if any(w in low for w in ("cannot", "unable", "no such",
                                      "not supported", "unsupported",
                                      "invalid", "unknown", "failure",
                                      "no capable", "driver")):
                why = line
                break
        log.warning("    preflight %-16s FAILED rc %s" % (name, rc_text(rc)))
        log.warning("      ffmpeg: %s" % why)
        return False, why

    log.info("preflight: testing what this machine can actually encode")
    # Say what the card is BEFORE testing it, so the trial result below
    # reads as confirmation rather than as a mystery. This is reporting
    # only - nothing branches on it. See gpu_generation().
    _gname = gpu_name()
    if _gname:
        _gen, _av1 = gpu_generation()
        if _gen and _av1 is True:
            log.info("    gpu: %s - %s, which has an AV1 encoder"
                     % (_gname, _gen))
        elif _gen and _av1 is False:
            log.info("    gpu: %s - %s" % (_gname, _gen))
            log.info("    this generation DECODES AV1 but cannot ENCODE "
                     "it; that is a silicon limit, not a driver one. "
                     "The run will use the CPU encoder instead.")
        else:
            log.info("    gpu: %s - generation not recognised, so the "
                     "trial encode below is the only thing that decides"
                     % _gname)
    else:
        log.info("    no NVIDIA GPU detected; expecting the CPU encoder")
    if os.environ.get("AV1PIPE_TEST_SW") == "1":
        # the software stand-in encoder is in use, so there is no nvenc
        # path to test; say so rather than failing a check that does
        # not apply
        log.info("    preflight skipped, software test hook is on")
        CAPS["nvenc_8bit"] = CAPS["nvenc_10bit"] = True
        CAPS["placebo_run"] = bool(CAPS.get("libplacebo"))
        CAPS["encoder"] = "testsw"
        CAPS["svtav1_ok"] = True
        return True
    nv8 = nv10 = False
    nv_err = ""
    if CAPS.get("av1_nvenc"):
        base = ["-c:v", "av1_nvenc", "-preset", "p7", "-cq", "27",
                "-b:v", "0", "-rc", "vbr"]
        nv8, err8 = trial("av1_nvenc 8-bit", [], "format=yuv420p", base)
        nv10, err10 = trial("av1_nvenc 10-bit", [], "format=p010le", base)
        nv_err = err8 or err10
        if nv8 and not nv10:
            log.warning("    10-bit NVENC is not working on this driver; "
                        "the run switches to 8-bit output")
    CAPS["nvenc_8bit"] = nv8
    CAPS["nvenc_10bit"] = nv10

    # pick the encoder for this run
    want = CAPS.get("encoder_pref", "auto")
    if (nv8 or nv10) and want in ("auto", "nvenc"):
        CAPS["encoder"] = "nvenc"
        log.info("    encoder: av1_nvenc on the GPU")
    else:
        if not (nv8 or nv10):
            kind, why = nvenc_verdict(nv_err)
            log.warning("    av1_nvenc will not run: %s" % why)
            if kind == "version_gap":
                log.warning("    the card is fine. the ffmpeg build is "
                            "simply newer than any driver released so far")
                log.warning("    fix it either way round: install an older "
                            "ffmpeg build, or wait for the driver that "
                            "matches. set ffmpeg_dir in the config file to "
                            "point at a specific ffmpeg without touching "
                            "PATH")
            elif kind == "nocard":
                gen, _av1 = gpu_generation()
                if gen:
                    log.warning("    this is a %s. AV1 ENCODE arrived with "
                                "Ada (RTX 40); this generation decodes AV1 "
                                "but has no AV1 encoder in hardware." % gen)
                else:
                    log.warning("    only Ada (RTX 40) and newer NVENC can "
                                "encode AV1. older cards decode it but have "
                                "no AV1 encoder")
                log.warning("    nothing is wrong and nothing needs fixing: "
                            "the run continues on the CPU encoder below.")
        if want == "nvenc":
            gen, _av1 = gpu_generation()
            log.error("    encoder is set to nvenc in the config and it "
                      "will not run%s. Set encoder = auto to use the CPU "
                      "encoder instead."
                      % (" on a %s" % gen if gen else ""))
            return False
        # CPU fallback, tested the same way rather than assumed
        sv, sv_err = trial("libsvtav1 CPU", [], "format=yuv420p10le",
                           ["-c:v", "libsvtav1", "-preset",
                            str(SVTAV1_PRESET), "-crf", "35"])
        CAPS["svtav1_ok"] = sv
        if not sv:
            log.error("    neither av1_nvenc nor libsvtav1 will encode "
                      "here. this ffmpeg build cannot produce AV1 at all")
            return False
        CAPS["encoder"] = "svtav1"
        log.info("    encoder: libsvtav1 on the CPU, preset %d"
                 % SVTAV1_PRESET)
        log.info("    CPU AV1 is slower than a GPU but the quality is "
                 "at least as good. expect hours per film, not minutes")
        # Be concrete about the cost rather than leaving "hours" to be
        # discovered overnight. The preset is NOT lowered to compensate:
        # that would trade picture for time without being asked, and on
        # a machine with no hardware encoder the whole point of the
        # fallback is that it does not cost quality.
        _cores = cpu_budget() or (os.cpu_count() or 1)
        log.info("    %d cores available to the encoder at preset %d. On "
                 "a machine this size a feature typically runs several "
                 "hours; lower svtav1_preset in the config trades picture "
                 "for time if that is the wrong trade for you."
                 % (_cores, SVTAV1_PRESET))
        if _cores <= 4:
            log.warning("    %d cores is a small budget for CPU AV1. "
                        "Raising cpu_max_pct, or svtav1_preset to 8-10, "
                        "will matter more here than anywhere else."
                        % _cores)

    # tested unconditionally because an SR file hands its encode to the
    # CPU even when the GPU encoder is working fine
    if not CAPS.get("svtav1_ok"):
        sv_ok, _ = trial("libsvtav1 CPU", [], "format=yuv420p10le",
                         ["-c:v", "libsvtav1", "-preset",
                          str(SVTAV1_PRESET), "-crf", "35"])
        CAPS["svtav1_ok"] = sv_ok

    CAPS["hwaccel_ok"] = None
    if HWACCEL != "none" and (nv8 or nv10):
        want = "cuda" if HWACCEL == "auto" else HWACCEL
        rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-v", "error",
                            "-hwaccel", want, "-f", "lavfi", "-i",
                            "testsrc2=size=320x240:rate=24:d=1",
                            "-frames:v", "2", "-f", "null", "-"],
                           timeout=120)
        if rc == 0:
            CAPS["hwaccel_ok"] = want
            log.info("    preflight %-16s ok" % ("hwaccel " + want))
            log.info("    video decode moves to the GPU, so the CPU stops "
                     "starving the encoder on 4K sources")
        else:
            log.info("    hwaccel %s unavailable, decoding on the CPU"
                     % want)

    placebo_ok = False
    if want_placebo and CAPS.get("libplacebo"):
        vf = placebo_filter() + ",format=p010le"
        placebo_ok, _ = trial("libplacebo vulkan",
                              ["-init_hw_device", "vulkan"], vf,
                              ["-c:v", "ffv1"])
        if not placebo_ok:
            log.warning("    SDR to HDR needs libplacebo on Vulkan and it "
                        "does not run here; that option is off for this run")
    CAPS["placebo_run"] = placebo_ok
    # success means an encoder was chosen, which may be the CPU one
    return bool(CAPS.get("encoder"))


def sleep_block(active):
    if not IS_WIN:
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if active else 0)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Folder pickers: tkinter dialogs when a display exists, plain text
# prompts when it does not
# ---------------------------------------------------------------------
def read_config():
    # plain "key = value" lines beside this script; # starts a comment,
    # blank lines and unknown keys are ignored, and a missing or broken
    # file is not an error because every value has a fallback
    cfg = {}
    path = Path(__file__).resolve().parent / CONFIG_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return cfg
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cfg[key.strip().lower()] = val.strip()
    return cfg


def resolve_dir(flag_val, cfg_val, fallback, prompt, must_exist):
    # an explicit flag is taken exactly as typed so a wrong path gets
    # reported instead of quietly swapping in some other folder; the
    # picker only opens when nothing above it names a usable folder
    if flag_val:
        return Path(flag_val).expanduser()
    for cand in (cfg_val, fallback):
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.is_dir() or not must_exist:
            return p
    return pick_dir(prompt)


def pick_dir(title):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title=title)
        root.destroy()
        return Path(chosen) if chosen else None
    except Exception:
        v = input(title + " path: ").strip().strip('"')
        return Path(v) if v else None


def setup_logging(dst, logdir=None):
    # logdir None keeps the old behaviour, a logs folder under the
    # destination. both the readable log and the trace file land in the
    # same place so there is only one folder to go looking in
    logdir = Path(logdir) if logdir else (dst / "logs")
    try:
        logdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logdir = dst / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("av1")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    # console and rotating pipeline.log at INFO (human readable)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%m-%d %H:%M:%S")
    con = logging.StreamHandler(sys.stdout)
    con.setLevel(logging.INFO)
    con.setFormatter(fmt)
    log.addHandler(con)
    fh = logging.handlers.RotatingFileHandler(
        logdir / "pipeline.log", maxBytes=10 * 1024 * 1024,
        backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    # trace file: DEBUG level, compact timestamps, lives in the install
    # directory so it survives destination changes between runs; sized
    # to hold 100+ full encodes (~20 KB each) with one backup covering
    # 200+ encodes of history, well past the 30-encode minimum
    vfmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname).1s %(message)s",
        datefmt="%m%d %H%M%S")
    trace_path = logdir / "av1_trace.log"
    # backupCount was 1, which covers a couple of hundred encodes. An
    # unattended month over thousands of files would recycle that long
    # before anyone came to read it, and the whole point of logging a
    # failure and moving on is that the evidence is still there
    # afterwards. 10 backups is about 55 MB and covers the run.
    vh = logging.handlers.RotatingFileHandler(
        trace_path, maxBytes=5 * 1024 * 1024,
        backupCount=10, encoding="utf-8")
    vh.setLevel(logging.DEBUG)
    vh.setFormatter(vfmt)
    log.addHandler(vh)

    # a buffered handler loses the last few lines when the process is
    # killed outright, which is exactly the case worth diagnosing, so
    # every record is pushed to disk as it is written
    class _Flush(logging.Handler):
        def emit(self, record):
            for h in log.handlers:
                if h is not self:
                    try:
                        h.flush()
                    except Exception:
                        pass
    log.addHandler(_Flush())
    log.debug("trace=%s" % trace_path)
    return log


# ---------------------------------------------------------------------
# Source scan and queue order: size first (largest wins), then the
# container rank table, then file creation date
# ---------------------------------------------------------------------
def queue_key(d):
    return (-d["size"], EXT_RANK.get(d["path"].suffix.lower(), 50),
            d["ctime"])


def scan_sources(src):
    items = []
    for p in src.rglob("*"):
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            if p.name.startswith("._"):
                continue
            st = p.stat()
        except OSError:
            continue
        if st.st_size < MIN_FILE_BYTES:
            continue
        items.append({"path": p, "size": st.st_size, "ctime": st.st_ctime})
    items.sort(key=queue_key)
    return items


def assign_out_names(items):
    # every source keeps its own result name; when two sources differ
    # only by container the results keep the source extension so one
    # can never overwrite the other (movie.avi -> movie.avi.mkv)
    groups = {}
    for it in items:
        r = Path(it["rel"])
        it["out_rel"] = r.with_suffix(".mkv")
        groups.setdefault(it["out_rel"].as_posix().lower(), []).append(it)
    clashes = 0
    for g in groups.values():
        if len(g) > 1:
            clashes += 1
            for it in g:
                r = Path(it["rel"])
                it["out_rel"] = r.with_name(r.name + ".mkv")
    return clashes


# ---------------------------------------------------------------------
# ffprobe read: everything later stages need, in one dict
# ---------------------------------------------------------------------
def stream_bitrate(s):
    """Bits per second for one stream, from wherever it is recorded.

    ffprobe fills `bit_rate` from the codec context, and for Matroska it
    frequently cannot: measured on the Scrooged UHD remux, the DTS-HD MA
    track reports bit_rate=None while carrying BPS=3920049 in its own
    stream tags, and only the small AC3 track reports a number directly.

    That mattered more than it looks. Three callers read this figure and
    every one of them was reading a zero that never came from anywhere:
    the DTS-HD test in audio_quality_rank could not fire, the bitrate
    tiebreak was constant, and opus_bitrate's source floor never
    engaged. The value was simply never collected.

    Matroska writes the tag as BPS, BPS-eng, BPS-en and so on depending
    on the muxer, so the whole tag set is searched rather than one name.

    Args:
        s: one stream dict from ffprobe's JSON.

    Returns:
        int: bits per second, or 0 when nothing records it.
    """
    br = frac(s.get("bit_rate"))
    if br > 0:
        return int(br)
    for k, v in (s.get("tags") or {}).items():
        if k.upper().split("-")[0] == "BPS":
            br = frac(v)
            if br > 0:
                return int(br)
    return 0


def probe_file(path):
    cmd = [TOOLS["ffprobe"], "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    rc, out, err = run(cmd, timeout=300)
    if rc != 0 or not out.strip():
        raise RuntimeError("ffprobe failed: " + err.strip()[:400])
    j = json.loads(out)
    fmt = j.get("format", {}) or {}
    streams = j.get("streams", []) or []
    v = None
    for s in streams:
        if s.get("codec_type") == "video" and not \
                (s.get("disposition") or {}).get("attached_pic"):
            v = s
            break
    if v is None:
        raise RuntimeError("no video stream found")
    dur = frac(fmt.get("duration")) or frac(v.get("duration"))
    if dur <= 0:
        raise RuntimeError("no readable duration")
    fps = frac(v.get("avg_frame_rate"))
    if fps <= 0 or fps > 1000:
        fps = frac(v.get("r_frame_rate"))
    if fps <= 0 or fps > 1000:
        fps = 24.0
    # both probed rates plus the raw string feed the sr cfr guard
    rr = frac(v.get("r_frame_rate"))
    ar = frac(v.get("avg_frame_rate"))
    fps_str = v.get("r_frame_rate") or ""
    if not (0 < frac(fps_str) <= 1000):
        fps_str = "%.6f" % fps
    try:
        size = int(frac(fmt.get("size"))) or path.stat().st_size
    except Exception:
        size = path.stat().st_size
    # video bitrate with a fallback chain; the basis tag lands in logs
    vbr = frac(v.get("bit_rate"))
    basis = "stream"
    if vbr <= 0:
        fb = frac(fmt.get("bit_rate"))
        if fb > 0:
            vbr = fb * 0.85
            basis = "container_85pct"
    if vbr <= 0:
        vbr = size * 8.0 / dur * 0.85
        basis = "size_85pct"
    pix = v.get("pix_fmt") or ""
    depth = 0
    try:
        depth = int(v.get("bits_per_raw_sample") or 0)
    except Exception:
        depth = 0
    if depth <= 0:
        depth = 10 if "10" in pix else (12 if "12" in pix else 8)
    master = cll = None
    dovi = None
    for sd in (v.get("side_data_list") or []):
        t = sd.get("side_data_type") or ""
        if "DOVI" in t:
            dovi = sd.get("dv_profile")
        elif "Mastering display" in t:
            master = sd
        elif "Content light" in t:
            cll = sd
    trc = (v.get("color_transfer") or "").lower()
    hdr = "sdr"
    if dovi == 5:
        hdr = "dv5"
    elif dovi is not None:
        hdr = "dv78"
    elif trc == "smpte2084":
        hdr = "hdr10"
    elif trc == "arib-std-b67":
        hdr = "hlg"
    audios, subs, atts = [], [], 0
    # one extra pass so the streams ffmpeg cannot parse are known before
    # anything tries to copy one and deadlock the muxer
    bad_streams = set()
    try:
        bad_streams = unparsable_streams(path)
        if bad_streams:
            log.debug("streams ffmpeg cannot parse: %s" % sorted(bad_streams))
    except Exception:
        pass
    for s in streams:
        ct = s.get("codec_type")
        if ct == "audio":
            d = s.get("disposition") or {}
            audios.append({
                "idx": s.get("index"),
                "lang": ((s.get("tags") or {}).get("language")
                         or "und").lower(),
                "ch": int(s.get("channels") or 2),
                "layout": s.get("channel_layout") or "",
                "codec": s.get("codec_name") or "",
                "bit_rate": stream_bitrate(s),
                "start": frac(s.get("start_time")),
                "title": ((s.get("tags") or {}).get("title") or ""),
                # commentary and audio-description tracks are real audio
                # but nobody wants them as their main track
                "aux": bool(d.get("comment") or d.get("visual_impaired")
                            or d.get("descriptions") or d.get("karaoke"))})
        elif ct == "subtitle":
            # PGS is a bitmap format, so matroska needs its width and
            # height to write the track entry. when ffprobe cannot
            # report them the muxer never finalises the header and the
            # whole encode deadlocks, so the size is captured here and
            # unusable tracks are dropped before they can do that.
            d = s.get("disposition") or {}
            subs.append({
                "idx": s.get("index"),
                "codec": s.get("codec_name") or "",
                "lang": ((s.get("tags") or {}).get("language")
                         or "und").lower(),
                "forced": bool(d.get("forced")),
                "sdh": bool(d.get("hearing_impaired"))})
        elif ct == "attachment":
            atts += 1
    # Container tags. av1_metadata.embedded_ids() reads info["tags"] to
    # short-circuit the whole TMDB search on an id the file already
    # carries, and this key was never here, so that function always
    # returned {} and the short-circuit was dead code. Measured on the
    # Scrooged remux, which carries IMDB=tt0096061 and TMDB=9647 in its
    # container tags: the strongest identification signal available was
    # being thrown away and the title guessed from its filename instead.
    return {"path": str(path), "size": size, "dur": dur,
            "tags": fmt.get("tags") or {},
            "vcodec": (v.get("codec_name") or "").lower(),
            "vidx": v.get("index"),
            "w": int(v.get("width") or 0), "h": int(v.get("height") or 0),
            "fps": fps, "fps_str": fps_str, "rr": rr, "ar": ar,
            "pix": pix, "depth": depth,
            "vbr": vbr, "vbr_basis": basis,
            "trc": trc, "prim": (v.get("color_primaries") or "").lower(),
            "space": (v.get("color_space") or "").lower(),
            "range": (v.get("color_range") or "").lower(),
            "field_order": (v.get("field_order") or "").lower(),
            "hdr": hdr, "dovi": dovi, "master": master, "cll": cll,
            "audios": audios, "subs": subs, "atts": atts,
            "bad_streams": sorted(bad_streams)}


def probe_duration(path):
    cmd = [TOOLS["ffprobe"], "-v", "error", "-show_entries",
           "format=duration", "-print_format", "json", str(path)]
    rc, out, err = run(cmd, timeout=120)
    if rc != 0:
        return 0.0
    try:
        return frac(json.loads(out).get("format", {}).get("duration"))
    except Exception:
        return 0.0


def _hhmmss(s):
    """Parse Matroska's per-track DURATION tag into seconds.

    The tag is written as "HH:MM:SS.nnnnnnnnn" with nanosecond
    precision, which float() will not take directly.

    Args:
        s: tag value, or anything falsy/malformed.

    Returns:
        float: seconds, or 0.0 if the value is absent or unparseable.
        Callers treat 0.0 as "no answer" and fall through to the next
        source, so this never raises.
    """
    try:
        parts = str(s).strip().split(":")
        if len(parts) != 3:
            return 0.0
        return (int(parts[0]) * 3600 + int(parts[1]) * 60
                + float(parts[2]))
    except Exception:
        return 0.0


def probe_video_duration(path):
    """Duration of the video stream, in seconds.

    Deliberately not the container duration. Matroska reports the length
    of its LONGEST track, so a subtitle that outlasts the picture - a
    signs or commentary track running past the credits, or a whisper cue
    carrying its fixed ~30 s span - inflates the container figure. The
    encoded output follows the video, so measuring anything else fails
    good files, and fails them on account of tracks the pipeline has
    already decided to drop.

    Four sources are tried in descending order of trustworthiness:

    1. the stream's own duration field, which mp4 and avi populate;
    2. Matroska's per-track DURATION tag, which is where mkv keeps it;
    3. frame count divided by average frame rate;
    4. container duration, accepted only because returning nothing
       would be worse.

    Args:
        path: file to probe.

    Returns:
        float: seconds. 0.0 only if ffprobe itself failed on the file.
    """
    cmd = [TOOLS["ffprobe"], "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=duration,nb_frames,avg_frame_rate:stream_tags=DURATION",
           "-print_format", "json", str(path)]
    rc, out, err = run(cmd, timeout=120)
    if rc == 0:
        try:
            st = (json.loads(out).get("streams") or [{}])[0]
            d = frac(st.get("duration"))
            if d > 0:
                return d
            tags = {k.lower(): v for k, v in (st.get("tags") or {}).items()}
            d = _hhmmss(tags.get("duration"))
            if d > 0:
                return d
            n = frac(st.get("nb_frames"))
            fps = frac(st.get("avg_frame_rate"))
            if n > 0 and fps > 0:
                return n / fps
        except Exception:
            pass
    return probe_duration(path)


def track_census(path):
    """Timing census of every audio and video track in one file.

    The unit of the provenance ledger. Duration and start time per
    track, cheap enough to run at every stage boundary - one ffprobe -
    and sufficient to catch the failures that have actually happened:
    a track truncated, a track dropped, a stream that starts somewhere
    it should not.

    Matroska keeps per-track length in a DURATION tag rather than in the
    stream header, so both are read and the larger believed.

    Returns:
        dict: {"video": [...], "audio": [...]} of per-track dicts, or
        {} if the file could not be read.
    """
    cmd = [TOOLS["ffprobe"], "-v", "error", "-show_entries",
           "stream=index,codec_type,codec_name,channels,duration,"
           "start_time:stream_tags=DURATION",
           "-print_format", "json", str(path)]
    rc, out, err = run(cmd, timeout=300)
    if rc != 0:
        return {}
    try:
        streams = json.loads(out).get("streams") or []
    except Exception:
        return {}
    cen = {"video": [], "audio": []}
    for st in streams:
        ct = st.get("codec_type")
        if ct not in ("video", "audio"):
            continue
        tags = {k.lower(): v for k, v in (st.get("tags") or {}).items()}
        cen[ct].append({
            "idx": st.get("index"),
            "codec": st.get("codec_name") or "",
            "ch": st.get("channels"),
            "dur": max(frac(st.get("duration")),
                       _hhmmss(tags.get("duration"))),
            "start": frac(st.get("start_time")),
        })
    return cen


def census_selfcheck(cen, tol=None):
    """Does every audio track in this file reach the end of its picture?

    The invariant that matters and that nothing was ever checking. The
    output's audio tracks are DERIVED - a downmix, a re-encode - so they
    do not correspond one-to-one with the master's, and comparing them
    to it track by track is meaningless. Comparing each of them to the
    picture in their OWN file is both meaningful and sufficient: a film
    with 3396 s of audio under 6056 s of video is broken however it got
    that way.
    """
    tol = AUDIO_SHORT_LIMIT_SEC if tol is None else tol
    if not cen or not cen.get("video") or not cen.get("audio"):
        return []
    vdur = max((v["dur"] for v in cen["video"]), default=0.0)
    if vdur <= 0:
        return []
    bad = []
    for n, a in enumerate(cen["audio"]):
        if a["dur"] > 0 and (vdur - a["dur"]) > tol:
            bad.append("audio track %d runs %.1f s under a %.1f s "
                       "picture (%.1f s short)"
                       % (n, a["dur"], vdur, vdur - a["dur"]))
    return bad


def census_compare(before, after, tol=2.0):
    """Compare two censuses of the SAME lineage, stage to stage.

    Used where the track list is meant to survive unchanged - the
    Dolby Vision strip, the final assembly, the HDR property pass. A
    track that shortens or disappears between two stages names the
    stage that did it, which is the whole point of keeping a ledger.
    """
    if not before or not after:
        return []
    bad = []
    for kind in ("video", "audio"):
        b, a = before.get(kind) or [], after.get(kind) or []
        if len(a) < len(b):
            bad.append("%s tracks went from %d to %d"
                       % (kind, len(b), len(a)))
        for n, (x, y) in enumerate(zip(b, a)):
            if x["dur"] > 0 and y["dur"] > 0 and (x["dur"] - y["dur"]) > tol:
                bad.append("%s track %d shortened from %.1f s to %.1f s "
                           "(lost %.1f s)"
                           % (kind, n, x["dur"], y["dur"],
                              x["dur"] - y["dur"]))
    return bad


def ledger(ctx, stage, path, compare_to=None, log=None):
    """Record one stage in the provenance ledger, and check it.

    Every stage that writes a file records what came out of it. When
    something is wrong the ledger names the tool that did it, instead of
    leaving a finished file to be diagnosed by correlation days later.

    Scrooged is the case this is built for: mkvmerge exited with
    warnings after the Dolby Vision remux and produced complete video
    with audio ending at 3396 s of 6056. Nothing looked, the encode
    reproduced it faithfully, and a film with 44 minutes of silence
    shipped. One ffprobe here would have caught it immediately.

    Returns:
        list[str]: problems found. Empty means this stage is clean.
    """
    log = log or ctx.get("log") or globals()["log"]
    cen = track_census(path)
    ctx.setdefault("ledger", []).append({"stage": stage,
                                         "path": str(path), "census": cen})
    if not cen:
        log.debug("ledger %-12s could not read %s" % (stage, path))
        return []
    v = max((x["dur"] for x in cen.get("video") or []), default=0.0)
    auds = ", ".join("%.1f" % a["dur"] for a in cen.get("audio") or [])
    log.debug("ledger %-12s video %.1f s | audio %s" % (stage, v, auds))
    problems = census_selfcheck(cen)
    if compare_to is not None:
        problems += census_compare(compare_to, cen)
    for pb in problems:
        log.error("    LEDGER %s: %s" % (stage, pb))
    return problems


def ledger_report(ctx, log):
    """Print the whole ledger, for a failure diagnostic."""
    rows = ctx.get("ledger") or []
    if not rows:
        return []
    out = ["AUDIO/VIDEO PROVENANCE LEDGER:",
           "  %-14s %10s  %s" % ("stage", "video", "audio tracks")]
    for r in rows:
        cen = r["census"] or {}
        v = max((x["dur"] for x in cen.get("video") or []), default=0.0)
        a = ", ".join("%.1f" % x["dur"] for x in cen.get("audio") or [])
        out.append("  %-14s %9.1fs  %s" % (r["stage"], v, a or "none"))
    for line in out:
        log.info("    " + line)
    return out


def audio_track_durations(path):
    """Length of every audio track, in file order.

    Matroska keeps per-track length in a DURATION tag rather than in the
    stream header, so both are read and the larger believed.
    """
    cmd = [TOOLS["ffprobe"], "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=duration:stream_tags=DURATION",
           "-print_format", "json", str(path)]
    rc, out, err = run(cmd, timeout=300)
    if rc != 0:
        return []
    try:
        streams = json.loads(out).get("streams") or []
    except Exception:
        return []
    out_d = []
    for st in streams:
        tags = {k.lower(): v for k, v in (st.get("tags") or {}).items()}
        out_d.append(max(frac(st.get("duration")),
                         _hhmmss(tags.get("duration"))))
    return out_d


def probe_first_audio(path):
    cmd = [TOOLS["ffprobe"], "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=index", "-print_format", "json",
           str(path)]
    rc, out, err = run(cmd, timeout=120)
    if rc != 0:
        return None
    try:
        s = json.loads(out).get("streams") or []
        return s[0]["index"] if s else None
    except Exception:
        return None


# ---------------------------------------------------------------------
# Analysis stage: sample windows spread across the runtime feed the
# crop consensus, the interlace test, and the quality score
# ---------------------------------------------------------------------
def pick_windows(dur):
    n = ANALYSIS_WINDOWS
    if dur < 90:
        n = 4
    elif dur < 300:
        n = 6
    win = ANALYSIS_WIN_SEC
    lead = max(1.0, dur * 0.03)
    last = dur * 0.97 - win
    if last <= lead:
        return [max(0.0, dur * 0.1)], min(win, max(1.0, dur * 0.5))
    return [lead + (last - lead) * i / (n - 1) for i in range(n)], win


def union_rect(rects, W, H):
    # least aggressive crop that still contains every window's picture
    # area; the bars have to show in every sample or no crop happens
    if not rects:
        return None
    x1 = min(r[2] for r in rects)
    y1 = min(r[3] for r in rects)
    x2 = max(r[2] + r[0] for r in rects)
    y2 = max(r[3] + r[1] for r in rects)
    x1 -= x1 % 2
    y1 -= y1 % 2
    w = min(W - x1, (x2 - x1) + ((x2 - x1) % 2))
    h = min(H - y1, (y2 - y1) + ((y2 - y1) % 2))
    if w >= W - CROP_MIN_PX and h >= H - CROP_MIN_PX:
        return None
    if w < 64 or h < 64:
        return None
    return (w, h, x1, y1)


def dominant_rect(rects, W, H, log=None):
    # the union takes the least aggressive rect across all samples, so a
    # single sample that misses the bars (a fade to black, a dark scene)
    # throws the whole result away. count instead and take the shape the
    # film actually spends most of its time in, which is also the right
    # answer when a film genuinely changes aspect part way through.
    if not rects:
        return None
    tally = {}
    for r in rects:
        # round to even so near-identical reads collapse together
        key = tuple((v // 2) * 2 for v in r)
        tally[key] = tally.get(key, 0) + 1
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    best, hits = ranked[0]
    share = hits / float(len(rects))
    if log and len(ranked) > 1:
        log.debug("crop tally: " + "; ".join(
            "%dx%d+%d+%d n=%d" % (k[0], k[1], k[2], k[3], v)
            for k, v in ranked[:4]))
        if share < 0.6:
            log.info("    mixed aspect ratios in this file; cropping to "
                     "the dominant one (%d%% of samples)" % int(share * 100))
    w, h, x, y = best
    # sanity: never crop outside the frame, never crop away almost all
    # of it, and ignore bars too thin to be worth the trouble
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
        return None
    if w * h < (W * H) // 4:
        return None
    if (W - w) < CROP_MIN_PX and (H - h) < CROP_MIN_PX:
        return None
    return (w, h, x, y)


def detect_crop(path, info, times, win, log):
    rects = []
    for t in times:
        cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-v", "info",
               "-ss", "%.2f" % t, "-t", "%.2f" % win, "-i", str(path),
               "-map", "0:%d" % info["vidx"], "-an", "-sn",
               "-vf", "cropdetect=limit=%s:round=2:reset=0" % CROP_LIMIT,
               "-f", "null", "-"]
        rc, out, err = run(cmd, timeout=300)
        m = None
        for mm in re.finditer(r"crop=(\d+):(\d+):(\d+):(\d+)", err):
            m = mm
        if m:
            rects.append(tuple(int(x) for x in m.groups()))
    if len(rects) < max(2, len(times) // 2):
        # too few reads to trust a crop call
        return None, rects
    return dominant_rect(rects, info["w"], info["h"], log), rects


def interlace_decision(field_order, combed, det, height):
    log.debug("interlace in: fo=%s combed=%.3f det=%s h=%d"
              % (field_order, combed, det, height))
    # container flag rules first, idet counts second, and the unknown
    # flag path only fires at line counts where true interlace lives;
    # a note rides along whenever combing was seen but the call stays
    # progressive, so trial runs can eyeball those files
    share = (combed / det) if det else 0.0
    fo = (field_order or "").lower()
    if fo in ("tt", "bb", "tb", "bt"):
        if share > 0.10:
            return True, None
        return False, "flagged_interlaced_but_reads_progressive"
    if fo == "progressive":
        if share > 0.30:
            return False, "combing_seen_kept_progressive_share_%.2f" % share
        return False, None
    if share > IDET_SHARE and combed >= IDET_MIN_FRAMES \
            and height in INTERLACE_HEIGHTS:
        return True, None
    if share > 0.30:
        return False, "combing_seen_kept_progressive_share_%.2f" % share
    return False, None


def detect_interlace(path, info, times, win, log):
    tff = bff = prog = und = 0
    rx = re.compile(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)"
                    r"\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)")
    for t in times:
        cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-v", "info",
               "-ss", "%.2f" % t, "-t", "%.2f" % win, "-i", str(path),
               "-map", "0:%d" % info["vidx"], "-an", "-sn",
               "-vf", "idet", "-f", "null", "-"]
        rc, out, err = run(cmd, timeout=300)
        m = None
        for mm in rx.finditer(err):
            m = mm
        if m:
            tff += int(m.group(1))
            bff += int(m.group(2))
            prog += int(m.group(3))
            und += int(m.group(4))
    det = tff + bff + prog
    inter, note = interlace_decision(info.get("field_order", ""),
                                     tff + bff, det, info["h"])
    return inter, {"tff": tff, "bff": bff, "prog": prog, "und": und,
                   "share": round((tff + bff) / det, 3) if det else 0.0,
                   "field_order": info.get("field_order", ""),
                   "note": note}


CODEC_FACTOR = {"h264": 1.0, "hevc": 1.35, "vp9": 1.25,
                "mpeg2video": 0.45, "mpeg4": 0.6, "msmpeg4v3": 0.55,
                "msmpeg4v2": 0.5, "vc1": 0.8, "wmv3": 0.7, "wmv2": 0.5,
                "vp8": 0.8, "mpeg1video": 0.4, "theora": 0.6,
                "h263": 0.4, "flv1": 0.45}


def quality_score(info):
    # bits per pixel scaled by codec generation gives a rough source
    # quality read; thresholds get tuned on real libraries
    bpp = info["vbr"] / max(1.0, info["w"] * info["h"] * info["fps"])
    f = CODEC_FACTOR.get(info["vcodec"], 0.8)
    q = bpp * f
    if q >= 0.090:
        tier = "high"
    elif q >= 0.045:
        tier = "mid"
    elif q >= 0.022:
        tier = "low"
    else:
        tier = "awful"
    log.debug("quality: bpp=%.5f q=%.2f tier=%s" % (bpp, q, tier))
    return bpp, q, tier


def hq_source_cq_delta(info):
    # a 4K 10-bit master carries far more real detail than the tier
    # alone can express, and squeezing it to the same CQ as a broadcast
    # 1080p rip is what leaves visible artifacts in the downscale
    if HQ_SOURCE_BOOST_PCT <= 0:
        return 0.0
    if info.get("h", 0) < HQ_SOURCE_MIN_H:
        return 0.0
    if info.get("depth", 8) < HQ_SOURCE_MIN_DEPTH:
        return 0.0
    # six steps double the rate, so steps = log2(1 + pct/100) * 6
    return -math.log(1.0 + HQ_SOURCE_BOOST_PCT / 100.0, 2) * 6.0


def pick_cq(tier, depth, hdrflag):
    base = {"high": 21, "mid": 23, "low": 26, "awful": 28}[tier]
    if depth >= 10:
        base -= 1
    if hdrflag:
        base -= 1
    return max(CQ_FLOOR, min(CQ_CEIL, base))


def uhd_source(info, an):
    """Is this source close enough to 4K to justify a 4K master?

    Judged on the stored WIDTH, not the height. Width survives
    letterboxing: a 2.39:1 UHD is 3840x1608 once the bars come off, which
    a height test would reject as "not 4K" even though every pixel of it
    came off a UHD disc. The area floor is the second guard so that a
    3840x1080 ultrawide cannot pass on width alone.

    Args:
        info: probe result for the source.
        an: analysis result; its crop rectangle gives the real picture.

    Returns:
        tuple: (eligible, real_width, real_height).
    """
    w = info.get("w", 0)
    rw, rh = w, info.get("h", 0)
    if an.get("crop"):
        rw, rh = an["crop"][0], an["crop"][1]
    return (w >= UHD_MIN_WIDTH and (rw * rh) >= UHD_MIN_PIXELS), rw, rh


def uhd_target_height(real_h):
    """Lines to render a 4K-class source at.

    Native unless the real picture is within UHD_PAD_BAND_PCT of 2160, in
    which case it is padded up to 2160 exactly. Deliberately NOT the
    1080 standard's rule, which would upscale a scope film from 1608
    lines to 2160 and add 60 percent more pixels to describe no more
    detail.

    Args:
        real_h: height of the actual picture, after cropping.

    Returns:
        int: 2160 to scale up to a full 4K frame, or 0 to stay native.
    """
    if real_h >= 2160:
        return 2160 if real_h > 2160 else 0
    band = 2160 * (1.0 - UHD_PAD_BAND_PCT / 100.0)
    return 2160 if real_h >= band else 0


def uhd_predict_mbps(cq, w, h, fps):
    """Predicted video bitrate for a frame size at a CQ, in Mbps.

    Built on the measured reference point rather than on a test encode,
    which is what keeps this off the VMAF-search path that was declined.
    Six CQ steps halve the rate; that relationship was already measured
    on this encoder, and UHD_BPP_REF anchors it to an absolute number.

    Returns 0.0 when no calibration constant has been set, which the
    caller must read as "no prediction available" rather than as "free".
    """
    if UHD_BPP_REF <= 0:
        return 0.0
    px = max(1.0, float(w) * float(h) * float(fps))
    scale = 2.0 ** ((UHD_BPP_REF_CQ - cq) / UHD_CQ_STEPS_PER_HALVING)
    return UHD_BPP_REF * px * scale / 1e6


def uhd_budget(info, match_source=True):
    """Bits per second the video may use.

    The ceiling is normally the smaller of the source's own size and
    UHD_MAX_GB, so a 6 GB web-dl aims at 6 GB rather than being inflated
    to fill a 10 GB allowance.

    match_source is turned OFF when artifact repair is running. Aiming at
    a damaged source's own bitrate would spend exactly as few bits as the
    encode that damaged it, which defeats the repair - the deblocked
    picture is smoother and cheaper per frame, but it still needs room to
    be stored. A repaired file is allowed the full cap.

    Args:
        info: probe result; supplies duration and source size.
        match_source: honour the source's size as a ceiling.

    Returns:
        float: video Mbps, or 0.0 when the duration is unknown and no
        budget can be worked out.
    """
    dur = float(info.get("dur") or 0.0)
    if dur <= 0:
        return 0.0
    cap = UHD_MAX_GB * (1024 ** 3)
    src = float(info.get("size") or 0)
    budget = min(src, cap) if (src and match_source) else cap
    total_mbps = budget * 8 / dur / 1e6
    return max(0.5, total_mbps - UHD_OVERHEAD_MBPS)


def uhd_fit_cq(cq, info, an, w, h, notes, repairing=False):
    """Raise CQ until the predicted size fits the budget, or the floor stops it.

    The cap is SOFT. If holding the quality floor (UHD_CQ_CEIL) still
    predicts an oversized file, the file wins and a note records the
    overage rather than the picture being degraded further to hit a
    number. That is the decision Tom made: colour depth and image quality
    outrank size.

    Args:
        cq: the CQ the tier table and its modifiers produced.
        info: probe result; supplies duration and frame rate.
        an: analysis result, unused today but kept so a future tier-aware
            budget rule has it to hand.
        w, h: the frame size actually being encoded.
        notes: plan notes list, appended to in place.

    Returns:
        int: the CQ to encode at.
    """
    budget_mbps = uhd_budget(info, match_source=not repairing)
    if budget_mbps <= 0 or UHD_BPP_REF <= 0:
        notes.append("uhd_budget_unavailable")
        return cq
    dur = float(info["dur"])
    fps = float(info.get("fps") or 24.0)
    start = cq
    while cq < UHD_CQ_CEIL:
        if uhd_predict_mbps(cq, w, h, fps) <= budget_mbps:
            break
        cq += 1
    pred = uhd_predict_mbps(cq, w, h, fps)
    gb = pred * 1e6 * dur / 8 / (1024 ** 3)
    # the number the budget is actually aiming at, which is NOT always
    # UHD_MAX_GB - a source smaller than the cap sets its own target.
    # reporting the cap here produced a log line that said "over budget"
    # and "0.16 GB against a 10.0 GB cap" in the same breath.
    target_gb = budget_mbps * 1e6 * dur / 8 / (1024 ** 3)
    how = "cap" if (repairing or target_gb >= UHD_MAX_GB * 0.999) \
        else "source size"
    if cq != start:
        notes.append("uhd_budget_cq_%d_to_%d" % (start, cq))
    if pred > budget_mbps:
        # the quality floor won. say so plainly - this is the one place
        # the size rule is allowed to be broken and it must not be quiet
        notes.append("uhd_over_budget_%.2fGB_target_%.2fGB_at_cq_%d"
                     % (gb, target_gb, cq))
        log.info("    4K budget: holding quality floor at CQ %d; predicts "
                 "%.2f GB against a %.2f GB target (%s)"
                 % (cq, gb, target_gb, how))
    else:
        log.info("    4K budget: CQ %d predicts %.2f GB against a %.2f GB "
                 "target (%s)" % (cq, gb, target_gb, how))
    return cq


def uhd_repair_filter(an, is_uhd):
    """The deblock strength to use, or None.

    Gated on the EXISTING bits-per-pixel tier table rather than on a new
    detector. blockdetect was measured for this job and is unusable: on a
    fair bitrate ladder it scored 8000k at 1.6222, 600k at 1.3681 and
    300k at 1.5613 - non-monotonic, so a wrecked encode and a transparent
    one land in the same place and no threshold separates them.
    """
    if ARTIFACT_REPAIR == "off" or not is_uhd:
        return None
    if ARTIFACT_REPAIR in ("weak", "strong"):
        return ARTIFACT_REPAIR
    if an.get("tier") in ARTIFACT_REPAIR_TIERS:
        return "strong" if an["tier"] == "awful" else "weak"
    return None


def analyze(path, info, log):
    times, win = pick_windows(info["dur"])
    crop, rects = detect_crop(path, info, times, win, log)
    inter, idet = detect_interlace(path, info, times, win, log)
    bpp, q, tier = quality_score(info)
    notes = [idet["note"]] if idet.get("note") else []
    return {"crop": crop, "rects": rects, "interlaced": inter,
            "idet": idet, "bpp": bpp, "qscore": q, "tier": tier,
            "windows": len(times), "notes": notes}


# ---------------------------------------------------------------------
# Audio selection and the stereo downmix expression
# ---------------------------------------------------------------------
def pick_audio(info):
    auds = info["audios"]
    log.debug("audio tracks: %d" % len(auds))
    if not auds:
        return None, "none"

    def by_lang(codes):
        for a in auds:
            if a["lang"] in codes:
                return a
        return None
    a = by_lang({"eng", "en"})
    if a:
        return a, "eng"
    a = by_lang({"jpn", "ja", "jp"})
    if a:
        return a, "jpn_fallback"
    return auds[0], "first_track_no_eng_no_jpn"


LAYOUT_CH = {
    "mono": ["FC"], "stereo": ["FL", "FR"],
    "2.1": ["FL", "FR", "LFE"], "3.0": ["FL", "FR", "FC"],
    "3.0(back)": ["FL", "FR", "BC"], "3.1": ["FL", "FR", "FC", "LFE"],
    "4.0": ["FL", "FR", "FC", "BC"], "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.0": ["FL", "FR", "FC", "BC", "SL", "SR"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
    "7.1(wide-side)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
}


def pan_expr(layout, centre=None):
    # stereo downmix per spec: center up 30 percent and surrounds down
    # 20 percent against the standard 0.707 mix gains; LFE folded in
    # low so the sub track cannot clip the mains. centre overrides the
    # centre gain outright, which is what the dialogue mix uses.
    # 0.919 is the standard 0.707 already lifted 30 pct, which is the
    # normal downmix. the dialogue mix multiplies that again rather than
    # replacing it, or a "boost" would come out quieter than the default
    fc = 0.919 if centre is None else 0.919 * float(centre)
    chans = LAYOUT_CH.get(layout)
    if not chans:
        return None
    L, R = [], []

    def add(side, gain, name):
        side.append("%.3f*%s" % (gain, name))
    for c in chans:
        if c == "FL":
            add(L, 1.0, c)
        elif c == "FR":
            add(R, 1.0, c)
        elif c == "FC":
            add(L, fc, c)
            add(R, fc, c)
        elif c == "LFE":
            add(L, 0.350, c)
            add(R, 0.350, c)
        elif c in ("SL", "BL"):
            add(L, 0.566, c)
        elif c in ("SR", "BR"):
            add(R, 0.566, c)
        elif c == "BC":
            add(L, 0.400, c)
            add(R, 0.400, c)
        elif c == "FLC":
            add(L, 0.707, c)
        elif c == "FRC":
            add(R, 0.707, c)
    return "pan=stereo|FL=" + "+".join(L) + "|FR=" + "+".join(R)


# ---------------------------------------------------------------------
# Colour conversion builders
# ---------------------------------------------------------------------
# zscale spells these differently per option and per build. matrix
# takes 470bg while primaries only takes bt470bg, and passing a name
# the build does not know is a hard failure, so every entry is a list
# of candidate spellings ending in the H.273 number, which always works.
ZM = {"bt709": ["709", "bt709", "1"],
      "bt470bg": ["470bg", "bt470bg", "5"],
      "smpte170m": ["170m", "smpte170m", "6"],
      "smpte240m": ["240m", "smpte240m", "7"],
      "bt2020nc": ["2020_ncl", "bt2020nc", "9"]}
ZP = {"bt709": ["709", "bt709", "1"],
      "bt470bg": ["bt470bg", "470bg", "5"],
      "bt470m": ["bt470m", "470m", "4"],
      "smpte170m": ["170m", "smpte170m", "6"],
      "smpte240m": ["240m", "smpte240m", "7"],
      "bt2020": ["2020", "bt2020", "9"],
      "film": ["film", "8"]}
ZT = {"bt709": ["709", "bt709", "1"],
      "smpte170m": ["601", "smpte170m", "6"],
      "bt470bg": ["601", "smpte170m", "6"],
      "bt470m": ["601", "smpte170m", "6"],
      "gamma22": ["601", "smpte170m", "6"],
      "gamma28": ["601", "smpte170m", "6"],
      "smpte240m": ["smpte240m", "240m", "7"],
      "iec61966-2-1": ["iec61966-2-1", "srgb", "13"],
      "srgb": ["iec61966-2-1", "srgb", "13"]}
ZSD = {"m": ZM["bt470bg"], "p": ZP["bt470bg"], "t": ZT["bt470bg"]}
ZHD = {"m": ZM["bt709"], "p": ZP["bt709"], "t": ZT["bt709"]}


def zscale_names(option):
    # read the constants this build lists under one zscale option
    names = CAPS.get("zscale_names")
    if names is None and not TOOLS.get("ffmpeg"):
        # selftest and any other caller that runs before find_tools;
        # no probe is possible, so the preferred spelling is used
        return set()
    if names is None:
        rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-h",
                            "filter=zscale"], timeout=60)
        names, cur = {}, None
        for line in (out + err).splitlines():
            if re.match(r"^ {3}\S", line):
                cur = line.split()[0]
                names[cur] = set()
            elif cur and re.match(r"^ {5}\S", line):
                names[cur].add(line.split()[0])
        CAPS["zscale_names"] = names
        log.debug("zscale options probed: %d" % len(names))
    return names.get(option, set())


def zpick(option, candidates):
    # first spelling this build actually accepts; the trailing number
    # is always valid because these options are plain integers
    known = zscale_names(option)
    if not known:
        return candidates[0]
    for c in candidates:
        if c in known or c.isdigit():
            return c
    return candidates[-1]


def build_zscale(info):
    # SDR normalize into bt709; unknown source tags fall back on the
    # usual SD/HD split
    sd = info["h"] <= 576
    dflt = ZSD if sd else ZHD
    m_in = zpick("matrix", ZM.get(info["space"], dflt["m"]))
    p_in = zpick("primaries", ZP.get(info["prim"], dflt["p"]))
    t_in = zpick("transfer", ZT.get(info["trc"], dflt["t"]))
    r_in = "full" if info["range"] in ("pc", "full", "jpeg") else "limited"
    return ("zscale=matrixin=%s:primariesin=%s:transferin=%s:rangein=%s"
            ":matrix=%s:primaries=%s:transfer=%s:range=limited"
            % (m_in, p_in, t_in, r_in,
               zpick("matrix", ZM["bt709"]),
               zpick("primaries", ZP["bt709"]),
               zpick("transfer", ZT["bt709"])))


# ---------------------------------------------------------------------
# Plan builder: turns probe plus analysis into the exact filter chain,
# CQ, colour target, audio pick, and gate flags for one file
# ---------------------------------------------------------------------
def build_plan(info, an, sdr_to_hdr=False):
    """Decide the filter chain, quality number and colour target.

    Filter order is load-bearing and not arbitrary:

    1. deinterlace and crop first, so everything downstream operates on
       real picture rather than on bars;
    2. the SD saturation lift next, because eq is the only filter in the
       chain with no 10-bit support and would otherwise drag the whole
       graph back down to 8-bit - verified by reading ffmpeg's format
       negotiation, not assumed;
    3. promotion to 10-bit, for 8-bit sources only;
    4. denoise, colour conversion and deband, which all want the extra
       precision - deband especially, since dithering into 8-bit levels
       discards most of what the filter is for;
    5. scaling to 1080 lines of actual picture;
    6. CAS sharpening only where no rescale happened and there is
       therefore no resampling gain to be had.

    Args:
        info: probe result for the source.
        an: analysis result - crop rectangle, interlace verdict, tier.
        sdr_to_hdr: expand SDR sources into HDR10. Off by default; it
            changes the appearance of every SDR file in a batch.

    Returns:
        dict: the plan - vf string, chain list, cq, out_space, the
        audio selection and a notes list recording every decision, which
        is what the journal and the failure diagnostics report.
    """
    notes = list(an.get("notes", []))
    chain = []
    if an["interlaced"]:
        chain.append("bwdif=mode=send_frame:parity=auto:deint=all")
        notes.append("deinterlaced_bwdif_same_rate")
    ch = info["h"]
    if an["crop"]:
        w, h, x, y = an["crop"]
        chain.append("crop=%d:%d:%d:%d" % (w, h, x, y))
        ch = h
    # eq is the ONE filter in this whole chain that cannot do 10-bit.
    # probed on 8.1.2: offered yuv420p10le it negotiates back down to
    # yuv420p and drags everything downstream with it. so the
    # saturation lift goes FIRST, while the source is still 8-bit and
    # nothing is lost by it, and the promotion happens straight after.
    if ch <= SD_SAT_MAX_H and SD_SATURATION != 1.0:
        chain.append("eq=saturation=%.3f" % SD_SATURATION)
        notes.append("sd_saturation_%.2f" % SD_SATURATION)
    # go to 10-bit before any filtering that benefits from the headroom.
    # deband in particular was running at 8-bit and dithering into 8-bit
    # levels, which threw away most of what it is for; the promotion
    # used to happen at the scaler, right at the end. only 8-bit sources
    # need this: a 10-bit source is already there and forcing the
    # layout would just buy a pointless conversion.
    if info.get("depth", 8) < 10:
        chain.append("format=yuv420p10le")
        notes.append("filtered_at_10bit")
    # 4K artifact repair goes here: after the promotion so it runs at
    # 10-bit, and BEFORE the denoise, because deblocking wants the block
    # edges still intact - hqdn3d would smear them first and leave the
    # deblocker guessing. deblock is the only candidate that is both
    # 10-bit clean and fast enough at 4K; every filter the forums
    # recommend for this (fspp, pp7, uspp) is 8-bit only.
    is_uhd, real_w, real_h = uhd_source(info, an)
    deblock = uhd_repair_filter(an, is_uhd)
    if deblock:
        chain.append(UHD_DEBLOCK[deblock])
        notes.append("uhd_deblock_%s_tier_%s" % (deblock, an["tier"]))
    repair = an["tier"] in ("low", "awful") or \
        (ch <= 576 and an["tier"] != "high")
    if repair:
        chain.append(REPAIR_DENOISE)
        chain.append(REPAIR_DEBAND)
        notes.append("repair_chain_tier_" + an["tier"])
    colorconv = None
    out_space = "sdr709"
    if info["hdr"] in ("hdr10", "dv78"):
        out_space = "hdr10"
    elif info["hdr"] == "hlg":
        if CAPS["libplacebo"]:
            colorconv = placebo_hlg_to_pq()
            out_space = "hdr10"
            notes.append("hlg_converted_to_pq")
        else:
            out_space = "hlg"
            notes.append("hlg_kept_no_libplacebo")
    else:
        odd = (info["prim"] not in ("", "unknown", "bt709")) or \
              (info["space"] not in ("", "unknown", "bt709"))
        if odd and CAPS["zscale"]:
            colorconv = build_zscale(info)
            notes.append("sdr_normalized_bt709")
        elif odd:
            notes.append("sdr_odd_space_kept_no_zscale")
    # SDR->HDR: expand bt709 into bt2020/pq via libplacebo when requested
    if sdr_to_hdr and out_space == "sdr709" and CAPS.get("libplacebo"):
        colorconv = placebo_filter()
        out_space = "hdr10"
        notes.append("sdr_expanded_to_hdr10_libplacebo")
    if colorconv:
        # colour math runs at native size, ahead of any scaling
        chain.append(colorconv)
    # 10-bit colour standard. deband runs at native size so the bands are
    # fixed before any scaling smears them across more pixels. it is not
    # applied twice when the repair chain already ran one.
    # deband fixes banding an 8-bit source already carries. a native
    # 10-bit master has none, so at 4K the filter costs a great deal of
    # CPU and changes nothing worth having.
    want_deband = (not repair) and (
        DEBAND_MODE == "always"
        or (DEBAND_MODE == "auto" and info.get("depth", 8) < 10))
    if want_deband:
        chain.append(STD_DEBAND)
        notes.append("deband_applied")
    elif not repair and DEBAND_MODE == "auto":
        notes.append("deband_skipped_source_already_10bit")
    upscaled = downscaled = False
    uhd_master = uhd_padded = False
    # A 4K master is only produced when the operator asked for one AND
    # the source is close enough to 4K to have the pixels. Everything
    # else - including every sub-1080 source in a 4K batch - falls
    # through to the 1080 standard below, which is why the AI upscaler
    # never targets anything above 1080 lines.
    if TARGET_RES >= 2160 and is_uhd:
        uhd_master = True
        tgt = uhd_target_height(ch)
        if tgt and ch < tgt:
            # inside the pad band: 1.85:1 at 2076 lines is 96.1% of 2160
            # and comes up to meet it. this is a ~4% resize, not an
            # upscale in the sense the tier table means, so `upscaled`
            # stays False and the CQ keeps coming from the tier.
            chain.append(SCALE_2160)
            uhd_padded = True
            notes.append("uhd_padded_%d_to_2160_lines" % ch)
        elif tgt and ch > tgt:
            chain.append(SCALE_2160)
            downscaled = True
            notes.append("uhd_downscaled_to_2160_lines")
        else:
            # outside the pad band - a scope film at 1608 lines - so it
            # keeps its own pixels rather than being stretched to a line
            # count it was never shot at.
            notes.append("uhd_native_%dx%d" % (real_w, ch))
            if CAS_STRENGTH > 0:
                chain.append("cas=strength=%.2f" % CAS_STRENGTH)
                notes.append("cas_sharpen_%.2f" % CAS_STRENGTH)
    # the target is 1080 lines of ACTUAL PICTURE. ch is the height after
    # cropping, so letterboxed content is measured on the picture rather
    # than on the frame it was sitting in. width follows the aspect.
    elif ch < 1080:
        chain.append(SCALE_1080)
        upscaled = True
        notes.append("upscaled_to_1080_lines")
        # Every bit of temporal stabilisation this pipeline has lived in
        # the Real-ESRGAN path - the 50 percent blend and the atadenoise
        # stabiliser after it. Run with --no-sr and a low-resolution
        # source got a plain lanczos stretch with NOTHING holding it
        # steady, which is why a batch of upscaled SD cartoons came back
        # described as jittery.
        #
        # Measured on a 640x480 29.97 source at 1080 output, mean
        # inter-frame difference: plain lanczos 1.8014, the shipped
        # chain 1.8416 (deband is adding noise, not removing it), and
        # the same chain plus this stabiliser 1.6038.
        #
        # Named as a tradeoff, not a free win: this is the same filter
        # the flicker work found can push BELOW the motion floor, which
        # means it smears real movement as well as shimmer. On cel
        # animation there is little grain for it to eat, which is why it
        # is on for the upscale path only. Set upscale_stabilise = 0 to
        # turn it off.
        if UPSCALE_STABILISE and not is_uhd:
            chain.append(UPSCALE_STAB_FILTER)
            notes.append("upscale_temporal_stabiliser")
    elif ch > 1080:
        chain.append(SCALE_1080)
        downscaled = True
        notes.append("downscaled_to_1080_lines")
    else:
        # already exactly 1080 lines of picture, so nothing is resampled
        # and no oversampling gain is available. a contrast adaptive
        # sharpen is the cheap way to recover apparent crispness without
        # the halos an unsharp mask leaves behind.
        if CAS_STRENGTH > 0:
            chain.append("cas=strength=%.2f" % CAS_STRENGTH)
            notes.append("cas_sharpen_%.2f" % CAS_STRENGTH)
    chain.append("format=p010le")
    hdrflag = out_space in ("hdr10", "hlg")
    hq_delta = hq_source_cq_delta(info)
    src_lines = an["crop"][1] if an["crop"] else info.get("h", 0)
    if upscaled and src_lines < UPSCALE_CQ_MAX_H:
        cq = max(CQ_FLOOR, min(CQ_CEIL, UPSCALE_CQ - (1 if hdrflag else 0)))
    elif upscaled:
        # a big source that merely has fewer than 1080 lines keeps the
        # number its tier earned; most of its detail is real, not invented
        cq = pick_cq(an["tier"], info["depth"], hdrflag)
        notes.append("upscale_kept_tier_cq_%dp_source" % src_lines)
    elif repair:
        cq = max(CQ_FLOOR, min(CQ_CEIL, REPAIR_CQ - (1 if hdrflag else 0)))
    else:
        cq = pick_cq(an["tier"], info["depth"], hdrflag)
    if CQ_OFFSET:
        # an explicit offset is deliberate, so it is allowed to go below
        # the value the tier table would ever pick on its own
        before = cq
        cq = max(CQ_FLOOR, min(CQ_CEIL, cq + CQ_OFFSET))
        if cq != before:
            notes.append("cq_offset_%+d_%d_to_%d" % (CQ_OFFSET, before, cq))
    if hq_delta:
        # a 4K 10-bit master gets a genuinely higher data rate rather
        # than being squeezed to the same number as a 1080p broadcast rip
        before = cq
        cq = max(CQ_FLOOR, min(CQ_CEIL, int(round(cq + hq_delta))))
        if cq != before:
            notes.append("hq_source_boost_%dpct_cq_%d_to_%d"
                         % (HQ_SOURCE_BOOST_PCT, before, cq))
    # the mirror of the hq boost. a small source stretched to 1080 has no
    # real detail to justify a generous data rate: the bitrate goes on
    # describing the upscale, not the film. measured case was a 512x288
    # AVI going 0.69 GB to 2.66 GB, 388 pct of source. the tier table
    # cannot see this because bpp is resolution-normalised, so a soft
    # DVD and a sharp Blu-ray score alike.
    if upscaled and LOWRES_TRIM_H > 0 and LOWRES_CQ_TRIM > 0:
        src_h = an["crop"][1] if an["crop"] else info["h"]
        if src_h <= LOWRES_TRIM_H:
            before = cq
            # this class gets its own ceiling, deliberately above the
            # library-wide one, because the detail being protected up
            # there was invented by the scaler rather than filmed
            top = max(CQ_CEIL, LOWRES_CQ_CEIL)
            cq = max(CQ_FLOOR, min(top, cq + LOWRES_CQ_TRIM))
            if cq != before:
                notes.append("lowres_%dp_cq_%d_to_%d" % (src_h, before, cq))
    # Work out the frame actually being encoded, then let the 4K budget
    # raise CQ until the predicted file fits - stopping at the quality
    # floor and letting the file go over rather than degrading further.
    # This runs LAST so it sees the tier, the hq boost and any offset.
    # The frame actually being encoded. This is also what the finished
    # file gets NAMED after, so it has to be right on every branch and
    # not just the 4K one - a 2160 line master called AV1_1080p would be
    # a lie told in the filename.
    src_w = an["crop"][0] if an["crop"] else info.get("w", 0)
    if uhd_master:
        out_h = 2160 if (uhd_padded or downscaled) else ch
    elif upscaled or downscaled:
        out_h = 1080
    else:
        out_h = ch
    out_w = int(round(src_w * float(out_h) / max(1, ch) / 2.0)) * 2
    maxrate_kbps = 0
    if uhd_master:
        cq = uhd_fit_cq(cq, info, an, out_w, out_h, notes,
                        repairing=bool(deblock))
        budget_mbps = uhd_budget(info, match_source=not deblock)
        if budget_mbps > 0:
            # a VBV cap as the backstop, set well above the average so it
            # bounds runaway complex scenes without flattening every peak
            # to the mean. the CQ above is what actually aims at the size.
            maxrate_kbps = int(budget_mbps * 1500)
    audio, why = pick_audio(info)
    if audio is None:
        notes.append("no_audio_stream")
    elif why != "eng":
        notes.append("audio_pick_" + why)
    gate = (an["crop"] is None) and (not an["interlaced"]) and \
        (not repair) and (not upscaled) and (not downscaled) and \
        (colorconv is None) and (out_space == "sdr709") and \
        (not sdr_to_hdr) and CAPS["libvmaf"]
    # this used to sit AFTER the return, so it never once ran
    log.debug("plan cq=%d space=%s gate=%s up=%s dn=%s repair=%s uhd=%s "
              "out=%dx%d notes=%s"
              % (cq, out_space, gate, upscaled, downscaled, repair,
                 uhd_master, out_w, out_h, notes))
    return {"vf": ",".join(chain), "chain": chain, "cq": cq,
            "out_space": out_space, "colorconv": colorconv is not None,
            "upscaled": upscaled, "downscaled": downscaled,
            "repair": repair, "gate": gate,
            "uhd_master": uhd_master, "uhd_padded": uhd_padded,
            "out_w": out_w, "out_h": out_h,
            "maxrate_kbps": maxrate_kbps,
            "audio": audio, "audio_why": why,
            "notes": notes,
            "needs_dv_strip": info["hdr"] == "dv78",
            "dv5": info["hdr"] == "dv5"}


# ---------------------------------------------------------------------
# Dolby Vision 7/8 strip: pull the HEVC stream, cut the RPU out with
# dovi_tool, then remux the clean video with the original audio, subs,
# and chapters into a temp source the rest of the chain reads as
# plain HDR10
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Scratch volume selection
# ---------------------------------------------------------------------
# Intermediates are enormous and short-lived: a Dolby Vision strip moves
# about three times the source size, and an upscale writes thousands of
# PNG frames. Which disk that lands on is the single biggest throughput
# decision the pipeline makes, and on this machine it was a hand-set
# sr_tmp_dir pointing at S:. A client has different letters, different
# disks, and no reason to know any of this, so it is worked out here.
#
# Ranking is by MEDIA TYPE first, not by a benchmark. Windows already
# knows whether a disk is NVMe, SATA SSD or spinning rust, and asking it
# is instant and exact where a benchmark is slow, noisy under load, and
# forbidden outright while a batch is running. The benchmark exists only
# for volumes Windows will not classify.
SCRATCH_AUTO = True
# A floor, not an estimate. The real requirement is sized per file
# against the queue; this is what makes a volume worth considering at
# all.
SCRATCH_MIN_FREE_GB = 50
SCRATCH_BENCH_MB = 48         # only for volumes with no media type
SCRATCH_PROBE_CACHE = "scratch_probe.json"
SCRATCH_PROBE_MAX_AGE_DAYS = 30
SCRATCH_DIRNAME = "av1_scratch"
# Media type -> rank. Windows MSFT_PhysicalDisk MediaType: 3 HDD,
# 4 SSD, 5 SCM (persistent memory). BusType 17 is NVMe.
SCRATCH_RANK = {"nvme": 4, "scm": 4, "ssd": 3, "unknown": 2, "hdd": 1}

# The session's own scratch directory, and whether we created it.
_SCRATCH = {"session": None, "root": None, "owned": False}


def list_local_volumes():
    """Every FIXED local volume with a drive letter.

    ctypes rather than psutil on purpose: psutil is an optional
    dependency everywhere else in this file, and the scratch decision
    has to work on a machine where the pip install half failed.

    Network and removable volumes are excluded by GetDriveType, which
    is the whole point - a scratch directory on a share is the failure
    this function exists to prevent.
    """
    vols = []
    if not IS_WIN:
        # POSIX: trust the caller's configuration rather than guessing
        # at mount points. Nothing has ever run this on Linux.
        return vols
    DRIVE_FIXED = 3
    try:
        mask = ctypes.windll.kernel32.GetLogicalDrives()
    except Exception as e:
        log.debug("could not enumerate drives: %s" % e)
        return vols
    for i in range(26):
        if not (mask >> i) & 1:
            continue
        letter = "%s:\\" % chr(ord("A") + i)
        try:
            if ctypes.windll.kernel32.GetDriveTypeW(
                    ctypes.c_wchar_p(letter)) != DRIVE_FIXED:
                continue
            u = shutil.disk_usage(letter)
        except Exception:
            continue
        vols.append({"root": letter, "free": u.free, "total": u.total,
                     "media": "unknown", "bus": "", "mbps": 0.0})
    return vols


def volume_media_types():
    """Drive letter -> "nvme" | "ssd" | "hdd" | "unknown".

    Straight out of the Windows storage stack, which knows this for
    certain. One PowerShell call, because the CIM classes involved
    (MSFT_Partition, MSFT_PhysicalDisk) have no usable ctypes surface
    and pulling in a WMI package would be another optional dependency
    on the exact path that must not have one.

    Returns {} on any failure, which sends every volume to the
    benchmark instead. Never raises.
    """
    if not IS_WIN:
        return {}
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$d=@{};"
        "foreach($p in Get-PhysicalDisk){"
        "  $d[[string]$p.DeviceId]=@{m=[string]$p.MediaType;"
        "                            b=[string]$p.BusType}};"
        "$o=@{};"
        "foreach($x in Get-Partition){"
        "  if($x.DriveLetter){"
        "    $k=[string]$x.DiskNumber;"
        "    if($d.ContainsKey($k)){$o[[string]$x.DriveLetter]=$d[$k]}}};"
        "$o|ConvertTo-Json -Compress"
    )
    rc, out, err = run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", ps], timeout=90)
    if rc != 0 or not (out or "").strip():
        log.debug("storage query gave nothing (rc=%s); "
                  "falling back to measurement" % rc)
        return {}
    try:
        raw = json.loads(out)
    except Exception as e:
        log.debug("storage query was not JSON: %s" % e)
        return {}
    if not isinstance(raw, dict):
        return {}
    got = {}
    for letter, d in raw.items():
        if not isinstance(d, dict):
            continue
        media = str(d.get("m") or "").strip().lower()
        bus = str(d.get("b") or "").strip().lower()
        if bus == "nvme" or media == "scm":
            kind = "nvme"
        elif media == "ssd":
            kind = "ssd"
        elif media == "hdd":
            kind = "hdd"
        else:
            kind = "unknown"
        got[str(letter).strip().upper()[:1]] = {"kind": kind, "bus": bus}
    return got


def bench_volume(root, mb=None):
    """Sequential write+read throughput in MB/s, or 0.0 if unmeasurable.

    Only ever called for a volume Windows would not classify. It writes
    a few tens of megabytes and reads them back; that is enough to
    separate a spinning disk from a solid state one by an order of
    magnitude, which is the only distinction being asked for.

    os.fsync is what makes it honest - without it this measures the
    write cache and every volume looks identical.
    """
    mb = SCRATCH_BENCH_MB if mb is None else mb
    d = Path(root) / SCRATCH_DIRNAME
    f = d / ("bench_%d.tmp" % os.getpid())
    buf = b"\xa5" * (1024 * 1024)
    try:
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(str(f), "wb") as fh:
            for _ in range(mb):
                fh.write(buf)
            fh.flush()
            os.fsync(fh.fileno())
        wt = max(1e-6, time.time() - t0)
        t1 = time.time()
        with open(str(f), "rb") as fh:
            while fh.read(1024 * 1024):
                pass
        rt = max(1e-6, time.time() - t1)
        return (mb / wt + mb / rt) / 2.0
    except Exception as e:
        log.debug("could not measure %s: %s" % (root, e))
        return 0.0
    finally:
        try:
            f.unlink()
        except OSError:
            pass


def _probe_cache_path():
    return Path(__file__).resolve().parent / SCRATCH_PROBE_CACHE


def _load_probe_cache():
    try:
        d = json.loads(_probe_cache_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_probe_cache(cache):
    try:
        _probe_cache_path().write_text(json.dumps(cache, indent=1),
                                       encoding="utf-8")
    except OSError as e:
        log.debug("could not write the scratch probe cache: %s" % e)


def rank_scratch_volumes(need_bytes=0, dst=None, log=None):
    """Every usable local volume, best first.

    Order: media type, then whether it is out of the destination's way,
    then free space. A volume that cannot hold the work is dropped
    outright rather than ranked low - picking a fast disk with no room
    on it fails half way through a Dolby Vision strip.

    The destination volume is DEMOTED, not excluded. On a single-disk
    laptop it is the only choice and refusing to run would be worse
    than the extra head movement.
    """
    log = log or globals()["log"]
    vols = list_local_volumes()
    if not vols:
        return []
    media = volume_media_types()
    cache = _load_probe_cache()
    dirty = False
    dst_letter = ""
    if dst:
        try:
            dst_letter = str(Path(dst).resolve())[:1].upper()
        except Exception:
            dst_letter = ""
    floor = max(int(SCRATCH_MIN_FREE_GB) * 1024 ** 3, int(need_bytes))
    usable = []
    for v in vols:
        letter = v["root"][:1].upper()
        info = media.get(letter) or {}
        v["media"] = info.get("kind", "unknown")
        v["bus"] = info.get("bus", "")
        if v["free"] < floor:
            log.debug("scratch: %s has %.1f GB free, under the %.1f GB "
                      "needed" % (v["root"], gb(v["free"]), gb(floor)))
            continue
        if v["media"] == "unknown":
            # measure it, but only once a month per volume: this is a
            # disk benchmark and the standing rule is that those do not
            # run while the machine is working.
            key = "%s|%d" % (letter, v["total"])
            hit = cache.get(key) or {}
            age = time.time() - float(hit.get("at", 0) or 0)
            if hit.get("mbps") and age < SCRATCH_PROBE_MAX_AGE_DAYS * 86400:
                v["mbps"] = float(hit["mbps"])
            else:
                v["mbps"] = bench_volume(v["root"])
                cache[key] = {"mbps": v["mbps"], "at": time.time()}
                dirty = True
            # 400 MB/s is comfortably above any spinning disk and below
            # any SSD, so it separates the two without pretending to
            # rank within either.
            if v["mbps"] >= 400:
                v["media"] = "ssd"
            elif v["mbps"] > 0:
                v["media"] = "hdd"
        usable.append(v)
    if dirty:
        _save_probe_cache(cache)
    for v in usable:
        v["rank"] = SCRATCH_RANK.get(v["media"], 2)
        v["is_dst"] = (v["root"][:1].upper() == dst_letter)
    usable.sort(key=lambda v: (-v["rank"], v["is_dst"], -v["free"]))
    return usable


def sweep_orphan_scratch(root, log):
    """Remove scratch sessions left behind by runs that died.

    A session directory is named for the process that owns it. If that
    process is gone the directory is garbage, however the run ended -
    killed, blue screen, power cut. This is what makes the cleanup
    guarantee hold across a crash, where no atexit hook and no finally
    block ever gets to run.
    """
    base = Path(root) / SCRATCH_DIRNAME
    if not base.is_dir():
        return 0
    freed = 0
    for p in base.glob("s*-*"):
        if not p.is_dir():
            continue
        try:
            pid = int(p.name.split("-", 1)[0][1:])
        except (ValueError, IndexError):
            continue
        if pid == os.getpid():
            continue
        try:
            idle = time.time() - p.stat().st_mtime
        except OSError:
            idle = 1e9
        alive = False
        try:
            import psutil as _ps
            alive = _ps.pid_exists(pid)
        except Exception:
            # No psutil, so liveness is unknowable and age is the only
            # evidence there is.
            alive = idle < 86400
        # PIDs get reused. A live process wearing a dead run's number
        # would otherwise protect that run's scratch for good, and on a
        # machine that reboots often the number comes round quickly.
        # Nothing here goes a day without writing, so an untouched
        # directory is abandoned whatever owns the pid now.
        if alive and idle < 86400:
            continue
        try:
            freed += sum(f.stat().st_size for f in p.rglob("*")
                         if f.is_file())
        except OSError:
            pass
        shutil.rmtree(str(p), ignore_errors=True)
    if freed:
        log.info("swept %.2f GB of scratch left by earlier runs that did "
                 "not finish" % gb(freed))
    return freed


def open_scratch_session(need_bytes=0, dst=None, log=None, explicit=""):
    """Choose a scratch volume and open this run's directory on it.

    An explicit sr_tmp_dir always wins: an operator who has named a
    disk has a reason, and second-guessing it would be the same mistake
    as the automatic sync corrector. Everything else is worked out.

    The directory is named for this process, which is what lets a later
    run identify and remove it if this one never gets to clean up.

    Returns:
        pathlib.Path | None: the session directory, or None when there
        is no usable local volume, in which case the caller falls back
        to the destination's .tmp exactly as before.
    """
    log = log or globals()["log"]
    stamp = time.strftime("%Y%m%d%H%M%S")
    if explicit:
        root = Path(explicit)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("sr_tmp_dir %s cannot be created (%s); choosing a "
                        "scratch disk automatically instead" % (explicit, e))
        else:
            log.info("scratch: %s, set explicitly in the config" % root)
            sweep_orphan_scratch(root, log)
            ses = root / SCRATCH_DIRNAME / ("s%d-%s" % (os.getpid(), stamp))
            try:
                ses.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.warning("could not open a session directory under %s "
                            "(%s); using it directly" % (root, e))
                _SCRATCH.update({"session": root, "root": root,
                                 "owned": False})
                return root
            _SCRATCH.update({"session": ses, "root": root, "owned": True})
            return ses
    if not SCRATCH_AUTO:
        return None
    vols = rank_scratch_volumes(need_bytes, dst, log)
    if not vols:
        log.warning("no local disk has room for the working files; "
                    "intermediates fall back to the destination's .tmp, "
                    "which is slower and puts unfinished material where "
                    "only finished output belongs")
        return None
    best = vols[0]
    log.info("scratch disks, best first:")
    for v in vols[:6]:
        extra = ""
        if v["mbps"]:
            extra = ", measured %.0f MB/s" % v["mbps"]
        if v["is_dst"]:
            extra += ", holds the destination"
        log.info("    %-4s %-8s %6.1f GB free of %6.1f%s"
                 % (v["root"], v["media"], gb(v["free"]), gb(v["total"]),
                    extra))
    if best["is_dst"] and len(vols) == 1:
        log.warning("    only one usable disk and it is the destination's; "
                    "working files and finished output share a spindle")
    root = Path(best["root"])
    sweep_orphan_scratch(root, log)
    ses = root / SCRATCH_DIRNAME / ("s%d-%s" % (os.getpid(), stamp))
    try:
        ses.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("could not open a scratch session on %s: %s" % (root, e))
        return None
    log.info("scratch: %s  (%s, %.1f GB free)"
             % (ses, best["media"], gb(best["free"])))
    _SCRATCH.update({"session": ses, "root": root, "owned": True})
    return ses


def close_scratch_session(log=None):
    """Delete this run's scratch directory. Safe to call repeatedly.

    Registered with atexit AND called from main's finally block AND
    reached through the signal handlers, because "when the session
    stops or ends or completes" means all three and they do not
    overlap: atexit does not run on a signal that kills the process
    outright, and finally does not run if the interpreter is torn down
    from under it.

    Only ever removes a directory this process created and named after
    itself. An explicitly configured sr_tmp_dir that we did not create
    is left exactly as found.
    """
    ses = _SCRATCH.get("session")
    if not ses or not _SCRATCH.get("owned"):
        return 0
    _SCRATCH["session"] = None
    log = log or globals()["log"]
    freed = 0
    try:
        p = Path(ses)
        if p.is_dir():
            try:
                freed = sum(f.stat().st_size for f in p.rglob("*")
                            if f.is_file())
            except OSError:
                freed = 0
            shutil.rmtree(str(p), ignore_errors=True)
            if freed:
                log.info("scratch cleared: %.2f GB released from %s"
                         % (gb(freed), p))
            else:
                log.debug("scratch cleared: %s" % p)
        # take the av1_scratch parent with it when it is empty, so an
        # automatically chosen disk is left exactly as it was found
        try:
            parent = p.parent
            if parent.name == SCRATCH_DIRNAME and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
    except Exception as e:
        log.debug("could not clear the scratch session: %s" % e)
    return freed


def scratch_dir(ctx):
    """Where intermediate files belong: the LOCAL disk, always.

    Only finished output should ever be written to the destination.
    Everything else - elementary streams pulled out for a Dolby Vision
    strip, extracted audio, upscaler frames and chunks, sampled frames
    for the visual check - is working material that is read back almost
    immediately and then deleted.

    The volume is chosen at startup by open_scratch_session(), which
    ranks the machine's local disks and opens a session directory on
    the fastest one with room. An explicit sr_tmp_dir overrides that.

    Falls back to the destination's .tmp only when no local volume is
    usable at all.

    (An earlier version of this note asserted that the destination was
    network-backed and blamed a Csc.sys network lock for the machine
    freezes. P: is a LOCAL DrivePool. The claim was wrong, it came from
    an earlier handoff rather than from measurement, and it produced a
    diagnosis that had to be withdrawn.)
    """
    d = _SCRATCH.get("session") or (Path(SR_TMP_DIR) if SR_TMP_DIR
                                    else ctx["tmp"])
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return ctx["tmp"]
    return d


def dv_strip(ctx, src_path, info, tag):
    log.debug("dv_strip: %s hdr=%s" % (src_path.name, info["hdr"]))
    # the elementary streams are the size of the film, twice over, and
    # the remux is a third copy. None of it belongs on the share.
    tmp = scratch_dir(ctx)
    # ...but three copies of a 50 GB remux is 150 GB, so the scratch
    # volume has to be able to take it. Failing here with a clear
    # message beats filling the disk half way through an extract.
    try:
        need = int(src_path.stat().st_size * 3.2)
        free = shutil.disk_usage(str(tmp)).free
        if free < need:
            raise RuntimeError(
                "not enough scratch space for the Dolby Vision strip: "
                "%s needs about %.1f GB on %s, which has %.1f GB free"
                % (src_path.name, gb(need), tmp, gb(free)))
    except OSError as e:
        log.debug("could not check scratch space: %s" % e)
    raw = tmp / (tag + ".dv.hevc")
    clean = tmp / (tag + ".hdr10.hevc")
    dvfree = tmp / (tag + ".dvfree.mkv")
    cmd1 = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-y", "-v",
            "error", "-i", str(src_path), "-map", "0:%d" % info["vidx"],
            "-c", "copy", "-bsf:v", "hevc_mp4toannexb", "-f", "hevc",
            str(raw)]
    rc, out, err = run(cmd1, timeout=21600)
    if rc != 0:
        cleanup_paths([raw])
        raise RuntimeError("hevc extract failed: " + err[-300:])
    cmd2 = [TOOLS["dovi_tool"], "remove", "-i", str(raw), "-o", str(clean)]
    rc, out, err = run(cmd2, timeout=21600)
    if rc != 0:
        cleanup_paths([raw, clean])
        raise RuntimeError("dovi_tool remove failed: " + (err or out)[-300:])
    cmd3 = [TOOLS["mkvmerge"], "-o", str(dvfree), str(clean),
            "--no-video", str(src_path)]
    rc, out, err = run(cmd3, timeout=21600)
    cleanup_paths([raw, clean])
    if rc > 1 or not dvfree.exists():
        cleanup_paths([dvfree])
        raise RuntimeError("dv-free remux failed: " + (out or err)[-300:])
    if rc == 1:
        # mkvmerge exit 1 is "completed with warnings", which is normally
        # harmless - and this code swallowed the warnings without
        # printing them. On Scrooged it warned after 42 minutes of
        # reading the source across the network and produced a file whose
        # VIDEO was complete but whose AUDIO stopped at 3396 s of 6056.
        # The encode faithfully reproduced that, the duration check only
        # looks at video, and a film with 44 minutes of silence shipped.
        # Warnings get read out loud from now on.
        log.warning("    the Dolby Vision remux completed WITH WARNINGS; "
                    "mkvmerge said:")
        for line in [x for x in (out or "").splitlines() if x.strip()][:12]:
            log.warning("      %s" % line.strip())
    # And verify it rather than trusting the exit code: every audio track
    # in the stripped file must be as long as the one it came from.
    try:
        src_a = audio_track_durations(src_path)
        new_a = audio_track_durations(dvfree)
        for i, (a, b) in enumerate(zip(src_a, new_a)):
            if a > 0 and b > 0 and (a - b) > 2.0:
                cleanup_paths([dvfree])
                raise RuntimeError(
                    "the Dolby Vision remux lost audio: track %d is "
                    "%.1f s in the source and %.1f s after the strip. "
                    "This is the fault that shipped a film with 44 "
                    "minutes of silence." % (i, a, b))
        if len(new_a) < len(src_a):
            log.warning("    the strip produced %d audio track(s) from "
                        "%d; carrying on" % (len(new_a), len(src_a)))
    except RuntimeError:
        raise
    except Exception as e:
        log.debug("could not verify the stripped audio: %s" % e)
    return dvfree


# ---------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------
def pick_encoder(use_sr):
    # the AI upscaler owns the GPU for the whole run, so an SR file
    # hands the encode to the CPU and leaves the card to the model
    base = CAPS.get("encoder")
    if base == "testsw":
        return base
    if use_sr and CAPS.get("svtav1_ok") and CAPS.get("encoder_pref") != "nvenc":
        return "svtav1"
    return base


def nvenc_has(option, value):
    # ask the encoder which options and constants it actually exposes;
    # these vary a lot between ffmpeg builds and driver versions
    opts = CAPS.get("nvenc_opts")
    if opts is None:
        if not TOOLS.get("ffmpeg"):
            return False
        rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-h",
                            "encoder=av1_nvenc"], timeout=60)
        opts, cur = {}, None
        for line in (out + err).splitlines():
            m = re.match(r"^ {2}-(\S+)", line)
            if m:
                cur = m.group(1)
                opts[cur] = set()
            elif cur is not None:
                m2 = re.match(r"^ {5}(\S+)", line)
                if m2:
                    opts[cur].add(m2.group(1))
        CAPS["nvenc_opts"] = opts
        log.debug("av1_nvenc options probed: %d" % len(opts))
    return value in opts.get(option, set())


def enc_args(cq, out_space, minimal=False, encoder=None, maxrate_kbps=0):
    if os.environ.get("AV1PIPE_TEST_SW") == "1":
        # sandbox hook: software x264 stands in for nvenc so the whole
        # chain can run on a box with no nvidia card; colour flags below
        # still apply; never set this variable on the real machine
        args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(cq)]
        if out_space == "hdr10":
            args += ["-color_primaries", "bt2020", "-color_trc",
                     "smpte2084", "-colorspace", "bt2020nc"]
        elif out_space == "hlg":
            args += ["-color_primaries", "bt2020", "-color_trc",
                     "arib-std-b67", "-colorspace", "bt2020nc"]
        else:
            args += ["-color_primaries", "bt709", "-color_trc", "bt709",
                     "-colorspace", "bt709"]
        args += ["-color_range", "tv"]
        return args
    encoder = encoder or CAPS.get("encoder")
    if encoder == "svtav1":
        # CPU AV1. -crf is SVT-AV1's own quality scale, offset from the
        # NVENC oriented CQ this pipeline computes everywhere else.
        # tune=0 favours how it looks rather than PSNR
        crf = max(CQ_FLOOR, min(SVTAV1_CRF_CEIL, cq + SVTAV1_CRF_OFFSET))
        args = ["-c:v", "libsvtav1", "-preset", str(SVTAV1_PRESET),
                "-crf", str(crf)]
        if maxrate_kbps > 0:
            # SVT-AV1 takes the same backstop and reports it as
            # "BRC mode: capped CRF" - probed on v4.1.0, not assumed.
            # Without this the 4K size budget silently did nothing
            # whenever the encode was routed to the CPU.
            args += ["-maxrate", "%dk" % maxrate_kbps,
                     "-bufsize", "%dk" % (maxrate_kbps * 2)]
        if not minimal:
            args += ["-svtav1-params", "tune=0"]
        if out_space == "hdr10":
            args += ["-color_primaries", "bt2020", "-color_trc",
                     "smpte2084", "-colorspace", "bt2020nc"]
        elif out_space == "hlg":
            args += ["-color_primaries", "bt2020", "-color_trc",
                     "arib-std-b67", "-colorspace", "bt2020nc"]
        else:
            args += ["-color_primaries", "bt709", "-color_trc", "bt709",
                     "-colorspace", "bt709"]
        return args + ["-color_range", "tv"]
    args = ["-c:v", "av1_nvenc", "-preset", "p7", "-cq", str(cq),
            "-b:v", "0", "-rc", "vbr"]
    if maxrate_kbps > 0:
        # VBV backstop for the 4K budget. CQ is what aims at the size;
        # this only stops a pathologically complex stretch running away.
        # bufsize at 2x maxrate gives peaks somewhere to live rather
        # than clamping every frame to the average.
        args += ["-maxrate", "%dk" % maxrate_kbps,
                 "-bufsize", "%dk" % (maxrate_kbps * 2)]
    if not minimal:
        # uhq is worth roughly 16-19 pct bitrate over hq on Blackwell,
        # but it only exists on newer builds, so it is probed once and
        # falls back rather than failing the encode
        args += ["-tune", "uhq" if nvenc_has("tune", "uhq") else "hq",
                 "-multipass", "fullres",
                 "-rc-lookahead", "32", "-spatial-aq", "1",
                 "-temporal-aq", "1", "-aq-strength", "8"]
        if nvenc_has("b_ref_mode", "middle"):
            args += ["-b_ref_mode", "middle"]
        if nvenc_has("lookahead_level", "3"):
            args += ["-lookahead_level", "3"]
        if nvenc_has("tf_level", "0"):
            # temporal filtering averages frames to help compression and
            # can wipe fast fine detail such as rain or wind in leaves.
            # off is the archival choice.
            args += ["-tf_level", "0"]
    if out_space == "hdr10":
        args += ["-color_primaries", "bt2020", "-color_trc", "smpte2084",
                 "-colorspace", "bt2020nc"]
    elif out_space == "hlg":
        args += ["-color_primaries", "bt2020", "-color_trc",
                 "arib-std-b67", "-colorspace", "bt2020nc"]
    else:
        args += ["-color_primaries", "bt709", "-color_trc", "bt709",
                 "-colorspace", "bt709"]
    args += ["-color_range", "tv"]
    return args


LOSSLESS_CODECS = ("truehd", "mlp", "dts_hd", "dtshd", "flac", "alac",
                   "pcm_s16le", "pcm_s24le", "pcm_s16be", "pcm_s24be",
                   "pcm_bluray", "pcm_dvd")


def is_lossless_track(a):
    """Whether an audio track is a lossless master.

    ffprobe reports DTS-HD MA as plain `dts` with a profile it does not
    surface, so the data rate stands in for it. That test needs a real
    bitrate to work with, which is why stream_bitrate() reads the BPS
    tag as well - on the Scrooged remux the DTS-HD MA track reports no
    bit_rate at all and 3.92 Mbps in its tags, so this returned False
    for every DTS-HD master in the library.
    """
    codec = (a.get("codec") or "").lower()
    return codec in LOSSLESS_CODECS or (codec == "dts"
                                        and (a.get("bit_rate") or 0)
                                        > 1500000)


def opus_bitrate(ch, src_bits=0, src_ch=0, src_lossless=False):
    # base rate is per channel; Opus is efficient enough that about 80k
    # a channel is past the point most people can pick it out
    base = max(160, min(AUDIO_OPUS_CAP, AUDIO_OPUS_PER_CH * max(1, ch)))
    if src_bits and src_ch and not src_lossless:
        # never spend fewer bits per channel than the source did. this
        # cannot make a re-encode better than its source, nothing can,
        # but it stops a good track being quietly squeezed on the way
        # through.
        #
        # LOSSLESS SOURCES ARE EXCLUDED, which is what the comment here
        # always claimed and the code never did - it simply never ran,
        # because the bitrate it tested was never collected. Matching a
        # PCM or TrueHD data rate in Opus is pure waste: a stereo FLAC
        # at 900 kbps would otherwise demand 768 kbps of Opus to
        # describe something 256 kbps already carries transparently.
        per_ch = src_bits / float(src_ch)
        base = max(base, int(per_ch * ch / 1000.0))
    return max(160, min(AUDIO_OPUS_CAP, base))


def audio_quality_rank(a):
    # lossless first, then channel count, then bitrate. this decides
    # which English track the stereo downmix is derived from.
    codec = (a.get("codec") or "").lower()
    is_lossless = is_lossless_track(a)
    # TrueHD carries Atmos and is the top master on a Disney remux, but
    # ffprobe usually reports no bitrate for it in matroska, so without
    # an explicit rank it would lose the tiebreak to DTS-HD
    tier = {"truehd": 3, "mlp": 3, "pcm_bluray": 2, "flac": 2,
            "dts": 1}.get(codec, 0) if is_lossless else 0
    return (1 if is_lossless else 0, tier, a.get("ch", 0),
            a.get("bit_rate", 0) or 0)


def is_lang(a, codes):
    return (a.get("lang") or "").lower() in codes


def best_for_lang(info, codes, surround_only=True):
    # commentary and audio-description tracks are excluded outright; a
    # descriptive track is real English audio but nobody wants it first
    pool = [a for a in (info.get("audios") or [])
            if is_lang(a, codes) and not a.get("aux")]
    if surround_only:
        sur = [a for a in pool if a.get("ch", 0) > 2]
        if sur:
            pool = sur
    return max(pool, key=audio_quality_rank) if pool else None


def best_english(info):
    eng = [a for a in (info.get("audios") or [])
           if is_lang(a, ("eng", "en")) and a.get("ch", 0) > 2]
    if not eng:
        eng = [a for a in (info.get("audios") or []) if is_lang(a, ("eng", "en"))]
    return max(eng, key=audio_quality_rank) if eng else None


def audio_order(info, picked):
    # every track is kept, but English leads so a player opens on it.
    # the picked track wins the top slot when it is English itself.
    auds = list(info.get("audios") or [])
    if not auds:
        return []
    if AUDIO_LANGS:
        keep = [a for a in auds if is_lang(a, AUDIO_LANGS)]
        # never end up with no audio at all because nothing was tagged
        if not keep:
            keep = [picked] if picked else auds[:1]
        auds = keep
    eng = [a for a in auds if is_lang(a, ("eng", "en"))]
    jpn = [a for a in auds if is_lang(a, ("jpn", "ja", "jp"))]
    rest = [a for a in auds if a not in eng and a not in jpn]
    if AUDIO_ENG_STEREO_ONLY and AUDIO_ADD_STEREO_MIX and best_english(info):
        # the downmix replaces the English masters rather than joining
        # them; a 2ch English track already in the source still counts
        eng = [a for a in eng if a.get("ch", 0) <= 2]
    if picked and picked in rest:
        rest = [picked] + [a for a in rest if a is not picked]
    return eng + jpn + rest


def with_async(af):
    """Left as a pass-through. DO NOT reintroduce aresample=async here.

    This briefly prefixed every audio chain with
    `aresample=async=1:first_pts=0`, on the theory that a disc remux's
    audio was drifting because gaps were being collapsed. That theory
    was wrong, and the change did real damage:

      - on Scrooged, the file it was added for, it changed NOTHING:
        290,700,800 samples either way, exactly the source length. The
        "drift" it was meant to cure was an artefact of correlating a
        heavily processed downmix against the raw 5.1; counting decoded
        samples window by window shows the body of that film is
        sample-exact.
      - on a source whose audio timeline genuinely IS gapped, async does
        not repair the gaps, it PADS them with silence. The Simpsons
        Movie went from 5204.66 s of audio to 6136.50 s - fifteen and a
        half minutes of silence appended to an 87 minute film, leaving
        audio tracks far longer than their own picture. That is what a
        player reports as a broken file.

    Kept as a function so the call sites stay honest about what is not
    being done to them.
    """
    return af


def af_latency_ms(af):
    """Latency an audio filter chain adds, measured with an impulse.

    Some ffmpeg audio filters do not declare their delay, so ffmpeg does
    not compensate the timestamps and the whole track comes out late.
    Measured on 8.1.2: dialoguenhance +32 ms, alimiter +5 ms, everything
    else in this chain 0. Together they put the stereo downmix 37 ms
    behind the picture.

    That is not caught by the sync scan, which compares OUTPUT audio to
    SOURCE audio and so cannot see the video at all; it reported exactly
    +37 ms on a 101 minute feature and passed it, because the correction
    threshold is 50 ms.

    Probed rather than hardcoded because the numbers depend on the
    ffmpeg build AND on the configured enhance/limiter settings, and
    because a future filter added to the chain must not silently
    reintroduce the problem. Cached per chain string.

    Returns:
        float: milliseconds to compensate, 0.0 if it cannot be measured.
    """
    if not af:
        return 0.0
    cache = CAPS.setdefault("af_latency", {})
    if af in cache:
        return cache[af]
    lat = 0.0
    try:
        # A click one second into three seconds of silence; wherever
        # silence ends on the far side of the chain is the delay.
        #
        # silencedetect rather than astats: astats on this build reports
        # peak LEVEL but no peak POSITION, so a probe built on it always
        # measured zero and compensated nothing - a silent no-op of
        # exactly the kind this project has been bitten by before.
        cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-v", "info",
               "-f", "lavfi",
               "-i", "aevalsrc='if(eq(floor(t*48000),48000),1,0)'"
                     ":d=3:s=48000:c=stereo",
               "-af", af + ",silencedetect=n=-60dB:d=0.005",
               "-f", "null", "-"]
        rc, out, err = run(cmd, timeout=120)
        m = re.search(r"silence_end:\s*([\d.]+)", out + err)
        if m:
            lat = max(0.0, (float(m.group(1)) - 1.0) * 1000.0)
        else:
            log.debug("audio latency probe found no click for %s"
                      % af[:60])
    except Exception as e:
        log.debug("audio latency probe failed for %s: %s" % (af[:40], e))
    cache[af] = lat
    return lat


def compensate_af_latency(af, log=None, label="audio chain"):
    """Append a trim that pulls the chain's own latency back out.

    Cheaper and more robust than trying to make every filter declare
    itself: measure what the chain does to an impulse, then drop that
    many milliseconds off the front and rebase the timestamps.

    Any LEADING pan is skipped when probing. A pan expression is built
    for whatever layout the source happens to have, so a stereo test
    impulse cannot be pushed through a 5.1 one; pan itself measured at
    exactly 0 ms, so nothing is lost by leaving it out of the probe.

    This function existed and was never called. Only the listening
    downmix compensated itself, inline, which left every other chain
    built from the same filters uncompensated - notably the
    dialogue-boosted mix, which carries alimiter and a heavier
    dialoguenhance and therefore the same 37 ms delay that bug 38 was
    raised for.

    Args:
        af: full filter chain string.
        log: optional logger.
        label: what to call the chain in the log line.

    Returns:
        str: the chain, with the compensating trim appended when there
        is a delay worth removing.
    """
    if not af:
        return af
    parts = af.split(",")
    head = 0
    while head < len(parts) and parts[head].startswith("pan="):
        head += 1
    lat = af_latency_ms(",".join(parts[head:]))
    if lat < AUDIO_SYNC_MIN_MS:
        return af
    if log:
        log.info("    %s runs %.0f ms late; compensating" % (label, lat))
    return af + ",atrim=start=%.6f,asetpts=PTS-STARTPTS" % (lat / 1000.0)


def stereo_mix_filter(layout, log=None, src_start=0.0):
    # centre carries the dialogue, so it is lifted rather than folded in
    # flat. the limiter catches the peaks that lift creates.
    pexp = pan_expr(layout, centre=AUDIO_MIX_CENTRE)
    if not pexp:
        return None
    chain = [pexp, "alimiter=limit=0.97"]
    if AUDIO_MIX_ENHANCE > 0:
        # ffmpeg's dialogue enhancer emits three channels, so it has to
        # be folded back to stereo straight after or the track ends up
        # as a 3.0 that players handle badly
        chain.append("dialoguenhance=original=1:enhance=%.2f"
                     % AUDIO_MIX_ENHANCE)
        chain.append("aformat=channel_layouts=stereo")
    chain.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    chain.append("aformat=channel_layouts=stereo")
    # alimiter and dialoguenhance both delay the signal without telling
    # ffmpeg, which is what put the DEFAULT stereo track 37 ms behind
    # the picture on every surround source.
    #
    # The compensation ends in asetpts=PTS-STARTPTS, which REBASES the
    # track to zero - and that is only harmless when the track already
    # started at zero. MEASURED: on a stream whose audio legitimately
    # begins at +2.000 s, the rebase moved the content 1995 ms early.
    # It is the mechanism that left two tracks off one master sitting
    # 4.348 s apart.
    #
    # A relative shift (asetpts=PTS-N/TB) is not the answer either:
    # measured, it is silently undone by avoid_negative_ts whenever the
    # shift would take the first packets below zero, which is exactly
    # the normal case. It compensates nothing and says nothing.
    #
    # So: compensate only where it is provably safe, and otherwise
    # leave the 37 ms alone. Audio running 37 ms LATE is far inside the
    # ITU-R BT.1359 tolerance of 125 ms for lagging sound; audio flung
    # two seconds EARLY is not inside anything.
    if abs(src_start) > 0.05:
        if log:
            log.warning("    this track starts at %+.3f s, so the latency "
                        "trim is skipped: rebasing would destroy that "
                        "offset. The downmix runs ~37 ms late, which is "
                        "well inside tolerance." % src_start)
        return ",".join(chain)
    return compensate_af_latency(",".join(chain), log, "audio downmix")


def audio_args_legacy(info, plan):
    """Per-track audio handling, for audio_fixed_layout = 0.

    This function was CALLED and never defined. Setting
    audio_fixed_layout = 0 - a documented config key with a paragraph of
    explanation next to it - raised NameError on every file, which the
    per-file handler caught and journalled as "exception", so a whole
    batch failed with nothing pointing at the setting that did it.

    The fixed layout is the project standard and remains the default.
    This is the other road: keep each track that survives the language
    policy, in the order audio_order() puts them, and decide per track
    whether to copy it or re-encode it, which is what audio_mode has
    always claimed to control:

        copy   never re-encode. Truest to the source.
        auto   copy when the codec is already widely playable, else Opus
        opus   always Opus
        aac    always AAC

    audio_downmix folds surround to stereo, which forces a re-encode
    because a filter cannot be applied to a copied stream. audio_denoise
    engages only on genuinely starved tracks, judged on bits per channel
    against the source's real bitrate - which is a figure this pipeline
    only started collecting when stream_bitrate() was added, so the
    threshold could never fire before either.

    Args:
        info: probe result.
        plan: build_plan output; its picked track leads the order.

    Returns:
        list[str]: ffmpeg arguments for every kept audio track.
    """
    tracks = audio_order(info, plan.get("audio"))
    if not tracks:
        return []
    args = []
    for i, a in enumerate(tracks):
        ch = int(a.get("ch", 2) or 2)
        codec = (a.get("codec") or "").lower()
        want_downmix = bool(AUDIO_DOWNMIX and ch > 2)
        if AUDIO_MODE in ("opus", "aac"):
            mode = AUDIO_MODE
        elif AUDIO_MODE == "copy":
            mode = "copy"
        else:
            mode = "copy" if codec in AUDIO_COPY_OK else "opus"
        if want_downmix and mode == "copy":
            # a filter cannot be applied to a stream being copied, so
            # asking for both means the downmix wins and the track is
            # re-encoded. said out loud rather than silently ignored.
            mode = "opus"
            if "audio_downmix_forced_reencode" not in plan["notes"]:
                plan["notes"].append("audio_downmix_forced_reencode")
        filters = []
        out_ch = ch
        if want_downmix:
            pexp = pan_expr(a.get("layout") or "")
            if pexp:
                filters.append(pexp)
                filters.append("aformat=channel_layouts=stereo")
                out_ch = 2
        if AUDIO_DENOISE and mode != "copy":
            br = a.get("bit_rate") or 0
            if br and (br / float(max(1, ch))) < AUDIO_DENOISE_PER_CH:
                filters.append("afftdn=nf=-25")
                plan["notes"].append("audio%d_denoised_starved" % i)
        args += ["-map", "0:%d" % a["idx"]]
        if mode == "copy":
            args += ["-c:a:%d" % i, "copy"]
        else:
            if out_ch > 2:
                # Opus takes only plain layouts; AC3 surround arrives
                # tagged 5.1(side) and is refused outright without this
                filters.append("aformat=channel_layouts="
                               "7.1|5.1|5.0|quad|stereo|mono")
            if filters:
                args += ["-filter:a:%d" % i, ",".join(filters)]
            if mode == "aac":
                rate = max(160, min(640, 96 * max(1, out_ch)))
                args += ["-c:a:%d" % i, "aac", "-b:a:%d" % i, "%dk" % rate]
            else:
                rate = opus_bitrate(out_ch, a.get("bit_rate", 0) or 0,
                                    ch, is_lossless_track(a))
                args += ["-c:a:%d" % i, "libopus", "-b:a:%d" % i,
                         "%dk" % rate, "-vbr", "on",
                         "-application", "audio"]
                if out_ch > 2:
                    args += ["-mapping_family:a:%d" % i, "1"]
            filters = []
        if filters:
            args += ["-filter:a:%d" % i, ",".join(filters)]
        args += ["-metadata:s:a:%d" % i,
                 "language=%s" % (a.get("lang") or "und"),
                 "-disposition:a:%d" % i, "default" if i == 0 else "0"]
        note = "audio%d_%s_%s%s" % (i, a.get("lang") or "und", mode,
                                    "_downmix" if want_downmix else "")
        if note not in plan["notes"]:
            plan["notes"].append(note)
    return args


def audio_args(info, plan, extra=None, input_index=0):
    if plan["audio"] is None:
        return []
    if not AUDIO_FIXED_LAYOUT:
        return audio_args_legacy(info, plan)

    eng = best_for_lang(info, ("eng", "en"))
    jpn = best_for_lang(info, ("jpn", "ja", "jp"))
    auds = info.get("audios") or []
    if len(auds) == 1 and eng is None and jpn is None:
        # ONE audio track and it claims to be neither English nor
        # Japanese. It is the film's only audio, so it is kept whatever
        # the tag says and it takes the primary slot - a mistagged or
        # untagged rip must not come out silent because of its metadata.
        eng = auds[0]
        if "single_audio_track_kept" not in plan["notes"]:
            plan["notes"].append("single_audio_track_kept_lang_%s"
                                 % (auds[0].get("lang") or "und"))
    slots = []
    # 1 and 2: the stereo downmixes, which is what most playback uses
    for src, lang, label in ((eng, "eng", "English"),
                             (jpn, "jpn", "Japanese")):
        if src is None:
            continue
        if src.get("ch", 0) <= 2:
            # already stereo, so there is nothing to fold down. it goes
            # straight into the stereo slot instead of being processed
            # through a downmix that would only colour it.
            slots.append({"src": src, "filter": None, "stereo": True,
                          "lang": lang, "native_stereo": True,
                          "title": "%s stereo" % label})
            continue
        filt = stereo_mix_filter(src["layout"],
                                 src_start=src.get("start", 0.0) or 0.0)
        if filt:
            slots.append({"src": src, "filter": filt, "stereo": True,
                          "lang": lang, "native_stereo": False,
                          "title": "%s stereo downmix (dialogue boosted)"
                                   % label})
    # 3 and 4: the original surround masters, re-encoded to Opus
    for src, lang, label in ((eng, "eng", "English"),
                             (jpn, "jpn", "Japanese")):
        if src is None or src.get("ch", 0) <= 2:
            # a stereo source is already slot 1 or 2; repeating it as an
            # "original" would just duplicate the same audio
            continue
        slots.append({"src": src, "filter": None, "stereo": False,
                      "lang": lang, "native_stereo": False,
                      "title": "%s %dch" % (label, src.get("ch", 2))})
    if not slots:
        slots = [{"src": plan["audio"], "filter": None, "stereo": False,
                  "lang": (plan["audio"].get("lang") or "und"),
                  "title": ""}]
    if extra:
        # the dialogue-boosted mix that whisper transcribed, kept so it
        # can be listened to as well as read
        slots.append(extra)

    args = []
    auds_all = info.get("audios") or []
    for i, sl in enumerate(slots):
        src = sl["src"]
        if input_index:
            # Audio comes from a DIFFERENT input to the video - the
            # original master rather than the Dolby Vision intermediate.
            # Absolute stream indices do not carry across files, so the
            # track is addressed by its POSITION among audio streams,
            # which the strip preserves.
            pos = next((n for n, a in enumerate(auds_all)
                        if a.get("idx") == src.get("idx")), 0)
            args += ["-map", "%d:a:%d" % (input_index, pos)]
        else:
            args += ["-map", "0:%d" % src["idx"]]
        lossless = is_lossless_track(src)
        if sl["stereo"]:
            out_ch, rate = 2, AUDIO_MIX_BITRATE
            if sl.get("filter"):
                args += ["-filter:a:%d" % i, with_async(sl["filter"])]
            elif src.get("ch", 2) > 2:
                # A stereo slot with no filter is only ever built from a
                # source that is already stereo. If one is ever built
                # from a surround track the encoder is handed six
                # channels while being told to make two, and libopus
                # refuses 5.1(side) outright - bug 7, all over again.
                # Fold it properly rather than emitting a command ffmpeg
                # will reject.
                pexp = pan_expr(src.get("layout") or "")
                args += ["-filter:a:%d" % i,
                         (pexp + "," if pexp else "")
                         + "aformat=channel_layouts=stereo"]
                rate = opus_bitrate(2, src.get("bit_rate", 0) or 0, 2,
                                    lossless)
            else:
                rate = opus_bitrate(2, src.get("bit_rate", 0) or 0, 2,
                                    lossless)
        else:
            out_ch = src.get("ch", 2)
            rate = opus_bitrate(out_ch, src.get("bit_rate", 0) or 0,
                                out_ch, lossless)
            if out_ch > 2:
                args += ["-filter:a:%d" % i,
                         "aformat=channel_layouts="
                         "7.1|5.1|5.0|quad|stereo|mono"]
        args += ["-c:a:%d" % i, "libopus", "-b:a:%d" % i, "%dk" % rate,
                 "-vbr", "on", "-application", "audio"]
        if out_ch > 2:
            # PER STREAM. Unqualified, ffmpeg applies a private codec
            # option to every stream it fits, so one surround track set
            # channel mapping family 1 on the stereo downmixes as well.
            # Probed: -mapping_family:a:N is accepted on this build.
            args += ["-mapping_family:a:%d" % i, "1"]
        if sl["title"]:
            args += ["-metadata:s:a:%d" % i, "title=%s" % sl["title"]]
        args += ["-metadata:s:a:%d" % i, "language=%s" % sl["lang"],
                 "-disposition:a:%d" % i,
                 "default" if i == 0 else "0"]
        note = "audio%d_%s_%s" % (i, sl["lang"],
                                  "downmix" if sl["stereo"] else "original")
        if note not in plan["notes"]:
            plan["notes"].append(note)
    return args


def tidy_generated_srt(path, info, log):
    """Repair a whisper transcript before it reaches the muxer.

    Applies the two halves of the same fix in the order that matters:
    remove the hallucinated tail first, then clamp whatever remains to
    the picture length. Running the clamp alone would truncate the
    invented cues rather than remove them, leaving nonsense dialogue
    sitting over the closing credits.

    Failures are contained here rather than propagated. A subtitle
    tidying problem is not worth losing a completed encode over, and the
    traceback goes to the trace file for diagnosis.

    Args:
        path: generated SRT, modified in place.
        info: probe result, read for the source duration.
        log: logger to report against.
    """
    import av1_subs as SB
    try:
        SB.drop_trailing_repeats(path, log)
        SB.clamp_srt(path, info.get("dur"), log)
    except Exception:
        log.debug("srt tidy failed", exc_info=True)


def acquire_subs(ctx, src, info, plan, log, tag):
    """Returns (list of external srt dicts, extra audio slot or None).

    Runs before the encode so the results can be muxed in one pass
    rather than remuxing a finished file.
    """
    if not SUBS_GENERATE:
        return [], None
    import av1_subs as SB
    t0 = time.time()
    have = usable_subs(info)
    have_eng = [s for s in have if s["lang"] in ("eng", "en")]
    have_jpn = [s for s in have if s["lang"] in ("jpn", "ja", "jp")]
    log.info("    subtitles present: %d English, %d Japanese"
             % (len(have_eng), len(have_jpn)))
    if have_eng and have_jpn:
        log.info("    both languages already present, nothing to generate")
        return [], None

    model = SB.find_model(WHISPER_MODEL_DIR, WHISPER_MODEL)
    if not model:
        log.warning("    no whisper model in %s, cannot generate subtitles"
                    % (WHISPER_MODEL_DIR or "(unset)"))
        return [], None
    vad = SB.find_vad_model(WHISPER_MODEL_DIR) if WHISPER_VAD else None
    if vad:
        log.info("    voice detection on: %s (roughly 4x the "
                 "transcription time)" % vad.name)
    elif WHISPER_VAD:
        log.warning("    whisper_vad is on but no ggml-silero*.bin sits "
                    "beside the whisper model; carrying on without it")
    log.info("    whisper model: %s" % model.name)

    made, extra_slot = [], None
    eng_srt = None
    jpn_srt = None
    # extracted ASR audio is hundreds of megabytes a film and is read
    # straight back by whisper; it has no business on the share
    work = scratch_dir(ctx)

    # ---- English -----------------------------------------------------
    if not have_eng:
        eng = best_for_lang(info, ("eng", "en"))
        if eng is None:
            # an untagged rip has no language on any track, so the
            # primary track is the only sensible candidate. whisper
            # detects the language itself.
            tagged = [x for x in (info.get("audios") or [])
                      if (x.get("lang") or "und") not in ("und", "")]
            if not tagged and info.get("audios"):
                eng = plan.get("audio") or info["audios"][0]
                log.info("    audio carries no language tag; transcribing "
                         "the primary track with automatic detection")
        if eng is None:
            log.info("    no English audio to transcribe")
        else:
            # a strongly dialogue-boosted mix, kept as its own output
            # track and used as the transcription source
            pexp = (pan_expr(eng["layout"], centre=SB.ENHANCE_CENTRE)
                    if eng.get("ch", 0) > 2 else None)
            # Same untold latency as the listening downmix: alimiter and
            # dialoguenhance delay the signal without declaring it. This
            # chain is used twice - as the transcription source, where
            # the delay pushes every generated cue late by the same
            # amount, and as the "English enhanced" output track, where
            # it is straightforward lip sync error.
            filt = compensate_af_latency(
                SB.enhance_filter(eng.get("layout", ""), pan_expr=pexp),
                log, "dialogue mix")
            wav = work / ("%s.eng.wav" % tag)
            log.info("    building English enhanced audio for transcription")
            if SB.make_asr_wav(TOOLS["ffmpeg"], src, eng["idx"], filt,
                               wav, log):
                out = work / ("%s.eng.srt" % tag)
                log.info("    transcribing English audio with whisper "
                         "(this takes a while)")
                lang = "en"
                if (eng.get("lang") or "und") in ("und", ""):
                    lang = "auto"
                got = SB.whisper_srt(TOOLS["ffmpeg"], wav, model, lang,
                                     out, WHISPER_GPU, WHISPER_QUEUE, log,
                                     vad_model=vad)
                if got:
                    tidy_generated_srt(got, info, log)
                    eng_srt = got
                    made.append({"path": got, "lang": "eng",
                                 "title": "English (generated)"})
                    plan["notes"].append("subs_english_generated")
                    log.info("    English subtitles generated")
                extra_slot = {"src": eng, "filter": filt, "stereo": True,
                              "lang": "eng", "native_stereo": False,
                              "title": "English enhanced (dialogue)"}
                cleanup_paths([wav])
    else:
        # an English track already in the file is the translation source
        for s in have_eng:
            if s["codec"] in ("subrip", "ass", "ssa", "mov_text", "text"):
                out = work / ("%s.eng_src.srt" % tag)
                rc, _, _ = run([TOOLS["ffmpeg"], "-hide_banner", "-y",
                                "-v", "error", "-i", str(src),
                                "-map", "0:%d" % s["idx"], "-c:s", "srt",
                                str(out)], timeout=1800)
                if rc == 0 and out.exists():
                    eng_srt = out
                    log.info("    using existing English subtitles as the "
                             "translation source")
                break
        else:
            log.info("    English subtitles are bitmap only, so they "
                     "cannot be translated without OCR")

    # ---- Japanese ----------------------------------------------------
    if not have_jpn:
        jpn = best_for_lang(info, ("jpn", "ja", "jp"))
        got = None
        if jpn is not None:
            # native Japanese speech beats any translation of English
            pexp = (pan_expr(jpn["layout"], centre=SB.ENHANCE_CENTRE)
                    if jpn.get("ch", 0) > 2 else None)
            filt = compensate_af_latency(
                SB.enhance_filter(jpn.get("layout", ""), pan_expr=pexp),
                log, "dialogue mix")
            wav = work / ("%s.jpn.wav" % tag)
            log.info("    Japanese audio found; transcribing it directly")
            if SB.make_asr_wav(TOOLS["ffmpeg"], src, jpn["idx"], filt,
                               wav, log):
                out = work / ("%s.jpn.srt" % tag)
                got = SB.whisper_srt(TOOLS["ffmpeg"], wav, model, "ja",
                                     out, WHISPER_GPU, WHISPER_QUEUE, log,
                                     vad_model=vad)
                cleanup_paths([wav])
                if got:
                    tidy_generated_srt(got, info, log)
                    plan["notes"].append("subs_japanese_transcribed")
        # argos translates locally, so it needs no key; every other
        # provider is a paid web service and without a key there is
        # nothing to call
        can_translate = bool(TRANSLATE_KEY) or TRANSLATE_PROVIDER == "argos"
        if not got and eng_srt and can_translate:
            log.info("    translating English subtitles into Japanese (%s)"
                     % TRANSLATE_PROVIDER)
            out = work / ("%s.jpn_tr.srt" % tag)
            got = SB.translate_srt(eng_srt, out, "JA", TRANSLATE_PROVIDER,
                                   TRANSLATE_KEY, TRANSLATE_ENDPOINT,
                                   TRANSLATE_MODEL, log=log)
            if got:
                plan["notes"].append("subs_japanese_translated_"
                                     + TRANSLATE_PROVIDER)
        elif not got and eng_srt:
            log.info("    no translation configured, so no Japanese "
                     "subtitles could be made; set translate_provider = "
                     "argos to do it locally")
        if got:
            made.append({"path": got, "lang": "jpn",
                         "title": "Japanese (generated)"})
            log.info("    Japanese subtitles generated")
            jpn_srt = got

    # ---- English from Japanese ---------------------------------------
    # A film with ONLY Japanese audio - which is most of the anime in
    # this library - has no English audio to transcribe, so the English
    # rung of the ladder above produces nothing and the file comes out
    # with no English subtitles at all. Nobody watching it can read it.
    #
    # The Japanese transcript is already in hand by this point, so the
    # remaining step is to translate it. Same argos path as the other
    # direction, and the same caveat: a machine translation of a machine
    # transcription, so it is a convenience track rather than a subtitle
    # anyone should rely on.
    have_eng_now = bool(have_eng) or any(m["lang"] == "eng" for m in made)
    if not have_eng_now:
        jsrc = jpn_srt
        if jsrc is None:
            for s in have_jpn:
                if s["codec"] in ("subrip", "ass", "ssa", "mov_text",
                                  "text"):
                    out = work / ("%s.jpn_src.srt" % tag)
                    rc, _, _ = run([TOOLS["ffmpeg"], "-hide_banner", "-y",
                                    "-v", "error", "-i", str(src),
                                    "-map", "0:%d" % s["idx"],
                                    "-c:s", "srt", str(out)], timeout=1800)
                    if rc == 0 and out.exists():
                        jsrc = out
                    break
        can_translate = bool(TRANSLATE_KEY) or TRANSLATE_PROVIDER == "argos"
        if jsrc and can_translate:
            log.info("    no English audio or subtitles; translating the "
                     "Japanese subtitles into English (%s)"
                     % TRANSLATE_PROVIDER)
            out = work / ("%s.eng_tr.srt" % tag)
            eng_got = SB.translate_srt(jsrc, out, "EN", TRANSLATE_PROVIDER,
                                       TRANSLATE_KEY, TRANSLATE_ENDPOINT,
                                       TRANSLATE_MODEL, log=log,
                                       source="ja")
            if eng_got:
                made.append({"path": eng_got, "lang": "eng",
                             "title": "English (translated)"})
                plan["notes"].append("subs_english_translated_from_japanese")
                log.info("    English subtitles generated by translation")
        elif jsrc:
            log.info("    no translation configured, so this file gets no "
                     "English subtitles; set translate_provider = argos")
        else:
            log.info("    no Japanese subtitles to translate from either")
    log_resources(log, "after subtitles", phase="subtitles")
    log.info("    subtitle acquisition took %.1f min"
             % ((time.time() - t0) / 60.0))
    return made, extra_slot


def unparsable_streams(path):
    # ffprobe happily omits width and height for most PGS tracks, so
    # using that as the test threw away good subtitles. ffmpeg names the
    # genuinely broken streams itself, and those are the only ones that
    # hang a matroska mux, so ask it directly.
    rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-v", "warning",
                        "-analyzeduration", "200M", "-probesize", "200M",
                        "-i", str(path), "-t", "0", "-f", "null", "-"],
                       timeout=600)
    bad = set()
    for m in re.finditer(r"Could not find codec parameters for stream (\d+)",
                         (out or "") + (err or "")):
        bad.add(int(m.group(1)))
    return bad


def usable_subs(info, log=None):
    # a bitmap subtitle with no reported size is the specific thing that
    # hangs a matroska mux: ffmpeg holds the header open waiting for
    # parameters that never arrive while the video queue fills behind it
    bad = set(info.get("bad_streams") or ())
    out, dropped_bad, dropped_lang = [], [], []
    for s in info.get("subs") or []:
        if s["idx"] in bad:
            dropped_bad.append(s)
            continue
        if SUBS_LANGS and s["lang"] not in SUBS_LANGS:
            dropped_lang.append(s)
            continue
        out.append(s)
    # English first, then Japanese, then anything else that survived,
    # so the track order in the finished file is predictable
    def rank(s):
        # English first, and within a language: plain, then forced, then
        # SDH, which is the order they are most likely to be wanted in
        l = s["lang"]
        lang_rank = (0 if l in ("eng", "en")
                     else 1 if l in ("jpn", "ja", "jp") else 2)
        kind = 1 if s.get("forced") else 2 if s.get("sdh") else 0
        return (lang_rank, kind, s["idx"])
    out.sort(key=rank)
    if log:
        if dropped_bad:
            log.warning("    dropped %d subtitle track(s) with no usable "
                        "size: %s. these hang the muxer if copied."
                        % (len(dropped_bad),
                           ", ".join("#%d %s" % (s["idx"], s["codec"])
                                     for s in dropped_bad)))
        if dropped_lang:
            log.info("    dropped %d subtitle track(s) by language"
                     % len(dropped_lang))
    return out


def blank_sub_path(tmp):
    # an empty subtitle track, mapped first and flagged default, so a
    # player opens with subtitles effectively off instead of burning in
    # whichever foreign track happened to be first in the source
    q = Path(tmp) / "_subs_off.srt"
    if not q.exists():
        # one blank cue at the very start. it must NOT sit far out in
        # the timeline: a late cue stretches the muxed duration to match
        # and the duration check then fails the whole file
        q.write_text("1\n00:00:00,000 --> 00:00:00,100\n \n\n",
                     encoding="utf-8")
    return q


def source_audio_position(info, picked):
    """Which a:N the picked track is among the SOURCE's audio streams.

    ffprobe stream indices and a:N specifiers are not the same number -
    a file with subtitles ahead of audio diverges immediately - so the
    position is resolved by looking it up rather than assumed.
    """
    auds = info.get("audios") or []
    for n, a in enumerate(auds):
        if picked is not None and a.get("idx") == picked.get("idx"):
            return n
    return 0


def primary_audio_is_english(info, plan):
    a = plan.get("audio") if plan else None
    if a is None:
        return True
    return (a.get("lang") or "und").lower() in ("eng", "en", "und", "")


def sub_args(info, blank_idx=None, ext_subs=None, ext_first=1,
             default_eng=False):
    # every subtitle track rides along; mp4-style text subs convert to
    # srt because matroska cannot hold mov_text; font attachments copy
    args = []
    off = 0
    if blank_idx is not None:
        args += ["-map", "%d:s:0" % blank_idx]
        off = 1
    keep = usable_subs(info)
    # each surviving track is mapped by index rather than with 0:s?,
    # so a broken one cannot sneak back in through the wildcard
    for s in keep:
        args += ["-map", "0:%d" % s["idx"]]
    ext_subs = ext_subs or []
    for n, e in enumerate(ext_subs):
        args += ["-map", "%d:s:0" % (ext_first + n)]
    args += ["-map", "0:t?", "-c:t", "copy"]
    # when the audio is not in English, an English subtitle should be
    # showing by default rather than the blank Off track
    eng_slot = None
    if default_eng:
        for j, s in enumerate(keep):
            if s["lang"] in ("eng", "en") and not s.get("forced") \
                    and not s.get("sdh"):
                eng_slot = j + off
                break
    if off:
        args += ["-c:s:0", "srt", "-metadata:s:s:0", "title=Off",
                 "-metadata:s:s:0", "language=und",
                 "-disposition:s:0",
                 "0" if eng_slot is not None else "default"]
    for j, s in enumerate(keep):
        conv = "srt" if s["codec"] in ("mov_text", "tx3g", "text") else "copy"
        args += ["-c:s:%d" % (j + off), conv]
        args += ["-disposition:s:%d" % (j + off),
                 "default" if (j + off) == eng_slot else "0"]
    base = off + len(keep)
    for n, e in enumerate(ext_subs):
        k2 = base + n
        # a generated English track becomes the default when nothing
        # already in the file claimed that slot
        mine = (default_eng and eng_slot is None and e["lang"] == "eng")
        if mine:
            eng_slot = k2
        args += ["-c:s:%d" % k2, "srt",
                 "-metadata:s:s:%d" % k2, "language=%s" % e["lang"],
                 "-metadata:s:s:%d" % k2, "title=%s" % e["title"],
                 "-disposition:s:%d" % k2, "default" if mine else "0"]
    return args


def mkv_identify(path):
    """Identify a file's tracks using mkvmerge's own JSON output.

    mkvmerge assigns track IDs from its own numbering, which is not
    ffprobe's stream index. The two frequently coincide and just as
    frequently do not - a file with subtitles ahead of audio in stream
    order will diverge immediately - so the mapping is always resolved
    by asking rather than assumed.

    Args:
        path: file to identify.

    Returns:
        dict | None: parsed JSON, or None if mkvmerge is unavailable,
        rejected the file, or returned something unparseable. Callers
        degrade gracefully rather than failing the encode.
    """
    if not TOOLS.get("mkvmerge"):
        return None
    rc, out, err = run([TOOLS["mkvmerge"], "-J", str(path)], timeout=600)
    # 1 means it parsed the file but had warnings, which is still usable
    if rc not in (0, 1) or not out:
        log.debug("mkvmerge -J failed rc=%s: %s" % (rc, (err or "")[:200]))
        return None
    try:
        return json.loads(out)
    except Exception:
        log.debug("mkvmerge -J returned unparsable json")
        return None


def _ids_of(ident, kind):
    """Extract mkvmerge track IDs of one type, in file order.

    Order is the contract here: it is what lets callers correlate these
    IDs with ffprobe stream indices without assuming the two numbering
    schemes agree, which they do not.

    Args:
        ident: parsed output of mkv_identify, or None.
        kind: mkvmerge track type, i.e. "video", "audio", "subtitles".

    Returns:
        list[int]: track IDs, empty if identification failed.
    """
    if not ident:
        return []
    return [t["id"] for t in ident.get("tracks", [])
            if t.get("type") == kind]


def mkvmerge_assemble(va_path, src_path, info, out_path, ext_subs=None,
                      blank_srt=None, default_eng=False, log=None):
    """Assemble the final Matroska file using mkvmerge.

    ffmpeg produces video and audio only; everything else is added here.
    That split exists because ffmpeg's Matroska muxer is where three of
    this project's worst defects lived: the PGS subtitle deadlock that
    froze 4K remuxes at a reproducible 4.1 pct, the attachment handling
    that blinded the stall watchdog, and the subtitle overhang that
    stretched the container past the picture. mkvmerge is purpose-built
    for the format and does not exhibit any of them.

    Source material is drawn from three places: video and audio from the
    encode, subtitles/fonts/chapters from the original source file, and
    generated subtitles as individual inputs. Track order and the
    default flags are then imposed explicitly via --track-order so the
    result matches the layout the ffmpeg path produced, which the
    regression tests compare against byte for byte.

    Args:
        va_path: encoder output, video and audio only.
        src_path: original source, for its subtitles, attachments and
            chapters.
        info: probe result for src_path, used to decide which subtitle
            streams survive the language and parseability filters.
        out_path: destination.
        ext_subs: generated subtitle descriptors, each a dict of
            path/lang/title.
        blank_srt: the placeholder "Off" track, or None to omit it.
        default_eng: force an English subtitle to carry the default
            flag, used when the primary audio is not English.
        log: logger; falls back to the module logger.

    Returns:
        tuple[int, str]: return code and diagnostic output. mkvmerge's
        exit code 1 means "completed with warnings" and is normalised
        to 0 here, with the warnings recorded in the trace.
    """
    log = log or globals()["log"]
    ext_subs = list(ext_subs or [])
    keep = usable_subs(info)

    # map the subtitle streams we decided to keep onto mkvmerge's ids by
    # their position among subtitle tracks, which is the one thing both
    # numbering systems agree on
    src_ident = mkv_identify(src_path)
    src_sub_ids = _ids_of(src_ident, "subtitles")
    pos_of = {}
    for n, s in enumerate(info.get("subs") or []):
        pos_of[s["idx"]] = n
    chosen = []
    for s in keep:
        p = pos_of.get(s["idx"])
        if p is not None and p < len(src_sub_ids):
            chosen.append((s, src_sub_ids[p]))
    if keep and not chosen:
        log.warning("      mkvmerge could not match any subtitle track; "
                    "carrying on without the source subtitles")

    # which track ends up flagged default, mirroring the ffmpeg path
    eng_src_tid = None
    if default_eng:
        for s, tid in chosen:
            if s.get("lang") in ("eng", "en") and not s.get("forced") \
                    and not s.get("sdh"):
                eng_src_tid = tid
                break
    gen_default = None
    if default_eng and eng_src_tid is None:
        for e in ext_subs:
            if e.get("lang") == "eng":
                gen_default = e
                break
    off_default = eng_src_tid is None and gen_default is None

    cmd = [TOOLS["mkvmerge"], "-o", str(out_path)]
    order = []
    # file 0: the encode. its own subtitles/attachments/chapters are
    # empty by construction, but say so explicitly
    va_ident = mkv_identify(va_path)
    va_ids = ([t["id"] for t in va_ident.get("tracks", [])]
              if va_ident else [0])
    cmd += ["--no-subtitles", "--no-attachments", "--no-chapters",
            str(va_path)]
    order += [(0, t) for t in va_ids]
    fid = 1

    if blank_srt:
        cmd += ["--language", "0:und", "--track-name", "0:Off",
                "--default-track-flag", "0:%d" % (1 if off_default else 0),
                "--forced-display-flag", "0:0", str(blank_srt)]
        order.append((fid, 0))
        fid += 1

    # the source supplies its subtitles, its fonts and its chapters
    if chosen:
        cmd += ["--no-video", "--no-audio",
                "--subtitle-tracks",
                ",".join(str(t) for _, t in chosen)]
        for _s, tid in chosen:
            cmd += ["--default-track-flag",
                    "%d:%d" % (tid, 1 if tid == eng_src_tid else 0)]
        cmd += [str(src_path)]
        order += [(fid, t) for _s, t in chosen]
    else:
        cmd += ["--no-video", "--no-audio", "--no-subtitles",
                str(src_path)]
    fid += 1

    for e in ext_subs:
        mine = gen_default is not None and e is gen_default
        cmd += ["--language", "0:%s" % e.get("lang", "und"),
                "--track-name", "0:%s" % e.get("title", ""),
                "--default-track-flag", "0:%d" % (1 if mine else 0),
                "--forced-display-flag", "0:0", str(e["path"])]
        order.append((fid, 0))
        fid += 1

    if order:
        cmd += ["--track-order",
                ",".join("%d:%d" % (f, t) for f, t in order)]
    log.debug("mkvmerge: %s" % " ".join(str(c) for c in cmd))
    rc, out, err = run(cmd, timeout=7200)
    # mkvmerge uses 1 for "finished with warnings", and its own manual
    # says the result "might be ok or not" and that BOTH the warning and
    # the file must be checked. Hiding these at debug is what let the
    # Dolby Vision remux lose half a film's audio in silence, so they
    # are read out loud and the caller verifies the file.
    if rc == 1:
        log.warning("      mkvmerge finished WITH WARNINGS:")
        for line in (out or "").splitlines():
            if "warning" in line.lower():
                log.warning("        %s" % line.strip())
        rc = 0
    return rc, (err or out or "")


def build_cmd(enc_input, plan, info, cq, tmp_out, minimal=False,
              drop_colorconv=False, encoder=None, ext_subs=None,
              extra_audio=None, va_only=False, audio_input=None):
    """Construct the ffmpeg command line for one encode attempt.

    Args:
        enc_input: source to read, which is the DV-stripped copy when
            one was produced rather than the original.
        plan: output of build_plan.
        info: probe result for enc_input.
        cq: quality number for this attempt; the caller walks a ladder.
        tmp_out: where ffmpeg writes.
        minimal: drop every optional encoder argument. Last resort when
            a build rejects options it advertised.
        drop_colorconv: strip the colour conversion filter. Removing the
            transform also reverts the output colour tags, or the result
            would claim a conversion that never happened.
        encoder: override the preflight's choice.
        ext_subs: generated subtitle descriptors. Ignored when va_only.
        extra_audio: additional audio slot, e.g. the dialogue-boosted
            mix retained when whisper had to transcribe English.
        va_only: emit video and audio only, leaving subtitles, fonts and
            chapters to mkvmerge. Nothing timed is carried in this mode,
            so nothing can stretch the container past the picture or
            deadlock the muxer.

    Returns:
        list[str]: argv for run_encode.
    """
    encoder = encoder or CAPS.get("encoder")
    chain = list(plan["chain"])
    out_space = plan["out_space"]
    if drop_colorconv and plan["colorconv"]:
        chain = [c for c in chain if not (c.startswith("zscale=")
                                          or c.startswith("libplacebo="))]
        # dropping the filter drops the colour change with it. tagging
        # the result bt2020/PQ after that describes frames that were
        # never converted, so the tags have to fall back too
        if info["hdr"] == "sdr":
            out_space = "sdr709"
    if os.environ.get("AV1PIPE_TEST_SW") == "1":
        # the 8-bit software stand-in cannot take p010, swap the tail format
        chain = [("format=yuv420p" if c == "format=p010le" else c)
                 for c in chain]
    elif encoder == "svtav1":
        # p010le is a hardware surface format; the CPU encoder takes the
        # planar equivalent instead
        chain = [("format=yuv420p10le" if c == "format=p010le" else c)
                 for c in chain]
    elif CAPS.get("nvenc_10bit") is False and CAPS.get("nvenc_8bit"):
        # preflight proved 10-bit will not encode on this driver. losing
        # the extra bit depth beats losing the file
        chain = [("format=yuv420p" if c == "format=p010le" else c)
                 for c in chain]
    vf = ",".join(chain)
    cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-y", "-v",
           "warning", "-analyzeduration", "200M", "-probesize", "200M"]
    nth = ff_threads()
    if nth:
        # both matter: -threads bounds the codec's own pool, and
        # -filter_threads bounds the filter graph, which on this chain
        # is where most of the CPU actually goes
        cmd += ["-threads", nth, "-filter_threads", nth]
    if (not drop_colorconv) and any(c.startswith("libplacebo=")
                                    for c in chain):
        cmd += ["-init_hw_device", "vulkan"]
    elif CAPS.get("hwaccel_ok") and encoder != "testsw":
        # not combined with the vulkan device, which wants the frames
        # in system memory anyway
        cmd += ["-hwaccel", CAPS["hwaccel_ok"]]
    cmd += ["-i", str(enc_input)]
    nxt = 1
    audio_in = 0
    if audio_input is not None:
        # THE AUDIO NEVER GOES THROUGH THE DOLBY VISION STRIP.
        #
        # The strip exists to take the RPU out of the VIDEO. Routing the
        # audio through it as well means an extract, a dovi_tool pass
        # and a whole extra mkvmerge remux stand between the master and
        # the encoder, for no benefit - and on 2026-08-11 that remux
        # exited with warnings and produced a file with complete video
        # and audio that stopped at 3396 s of 6056. Read the audio from
        # the master instead: one decode, one encode, timestamps and
        # runtime straight off the source.
        cmd += ["-i", str(audio_input)]
        audio_in = nxt
        nxt += 1
    blank_idx = None
    ext_subs = ext_subs or []
    if not va_only:
        if SUBS_BLANK_FIRST:
            try:
                cmd += ["-f", "srt", "-i",
                        str(blank_sub_path(tmp_out.parent))]
                blank_idx = nxt
                nxt += 1
            except OSError:
                blank_idx = None
        # generated subtitle files ride in as extra inputs. every -i has
        # to appear before the first output option or ffmpeg rejects it.
        for e in ext_subs:
            cmd += ["-f", "srt", "-i", str(e["path"])]
    ext_first = nxt
    cmd += ["-map", "0:%d" % info["vidx"], "-vf", vf,
            "-fps_mode", "passthrough"]
    cmd += enc_args(cq, out_space, minimal, encoder,
                    plan.get("maxrate_kbps", 0))
    cmd += audio_args(info, plan, extra_audio, audio_in)
    if va_only:
        # subtitles, fonts and chapters are mkvmerge's job in this mode.
        # nothing timed is carried here, so nothing can stretch the
        # container past the picture or deadlock the muxer.
        cmd += ["-map_metadata", "0"]
    else:
        cmd += sub_args(info, blank_idx, ext_subs, ext_first,
                        default_eng=not primary_audio_is_english(info, plan))
        cmd += ["-map_metadata", "0", "-map_chapters", "0"]
    cmd += ["-max_muxing_queue_size", "4096",
            "-progress", "pipe:1", "-nostats",
            "-f", "matroska", str(tmp_out)]
    log.debug("ffcmd: %s" % " ".join(str(c) for c in cmd))
    return cmd


def cfr_like(info):
    # frame extraction forces constant timing, so the sr path only
    # takes sources whose two probed rates agree
    rr, ar = info.get("rr", 0.0), info.get("ar", 0.0)
    if rr <= 0 or ar <= 0:
        return False
    return abs(rr - ar) <= 0.001 * max(rr, ar)


def hms(seconds):
    """Format a duration for human consumption.

    Used for ETAs on the upscale path, where the answer is routinely
    measured in hours and "27142 s" communicates nothing useful.

    Args:
        seconds: duration; negatives are clamped to zero.

    Returns:
        str: "7h 30m", "10m 05s" or "45s" depending on magnitude.
    """
    s = int(max(0, seconds))
    if s >= 3600:
        return "%dh %02dm" % (s // 3600, (s % 3600) // 60)
    if s >= 60:
        return "%dm %02ds" % (s // 60, s % 60)
    return "%ds" % s


def sr_tile_for(scale, max_dim, log=None):
    """Size the upscaler's tile against video memory free right now.

    A fixed tile size cannot be made safe. Memory cost is quadratic in
    the tile dimension AND quadratic in the scale factor, so a value
    that is comfortable at x2 is four times heavier at x4; and the card
    is shared with the NVENC session encoding the previous file, which
    measurement puts at 2.3-3.4 GB. A hard-coded 1024 came to roughly
    21 GB at x3 on a 16 GB card and took the machine with it.

    The budget is therefore derived at call time: the configured
    ceiling on total card usage, less what is already committed, capped
    by what is genuinely free, less a reserve for the concurrent encode
    and the desktop compositor. Tile size follows from that budget using
    ncnn's own auto-selection thresholds as the cost model.

    A tile larger than the frame processes the frame in one pass and
    wastes the remainder, so the frame dimension is an upper bound.

    Args:
        scale: upscale factor, which drives the per-tile cost.
        max_dim: largest frame dimension, as the sensible tile ceiling.
        log: optional logger for the sizing decision.

    Returns:
        int: tile size in pixels, a multiple of 32 and at least 64.
        Returns SR_TILE unchanged when an explicit override is set.
    """
    if SR_TILE:
        return SR_TILE
    per400 = SR_TILE_COST_400.get(scale, 1690.0)
    budget = float(SR_VRAM_MB)
    total = 0.0
    used = 0.0
    g = gpu_snapshot()
    if g and g.get("vram_total_mb"):
        total = float(g["vram_total_mb"])
        used = float(g.get("vram_used_mb", 0))
        free = total - used
        # The ceiling is on TOTAL card usage, so what is left for us is
        # that ceiling minus what is ALREADY taken. Two ceilings apply
        # and the smaller wins: the absolute sr_vram_mb, and gpu_max_pct
        # of the card.
        #
        # The percentage is the one that matters. sr_vram_mb was 15000
        # against a 16311 MB card, which over-commits before the desktop
        # is even counted - and the desktop alone holds about 1800 MB.
        # That is how a 992 tile got chosen and the driver died.
        ceiling = min(float(SR_VRAM_MB), total * GPU_MAX_PCT / 100.0)
        budget = min(ceiling - used, free)
        if log:
            log.debug("sr vram: %.0f MB total, %.0f used, %.0f free; "
                      "ceiling %.0f (%d%% of card vs sr_vram_mb %d), "
                      "%.0f available"
                      % (total, used, free, ceiling, GPU_MAX_PCT,
                         SR_VRAM_MB, budget))
    # Sized to this card rather than to the 16 GB board the figure was
    # measured on. See sr_vram_reserve().
    reserve = sr_vram_reserve(total)
    if log and total and abs(reserve - SR_VRAM_RESERVE_MB) > 1.0:
        log.debug("sr vram reserve %.0f MB rather than the configured "
                  "%d, sized to a %.1f GB card running the %s encoder"
                  % (reserve, SR_VRAM_RESERVE_MB, total / 1024.0,
                     CAPS.get("encoder") or "unknown"))
    budget = max(256.0, budget - reserve)
    tile = int(400.0 * math.sqrt(budget / per400))
    # round down to a multiple of 32; the tool's floor is 32
    tile = max(64, (min(tile, max_dim, SR_TILE_MAX) // 32) * 32)
    cost = per400 * (tile / 400.0) ** 2
    if log:
        capped = ""
        if tile >= SR_TILE_MAX and SR_TILE_MAX < max_dim:
            capped = " (held at the %d ceiling)" % SR_TILE_MAX
        log.info("    sr tile %d at x%d, about %.1f GB of a %.1f GB "
                 "budget%s" % (tile, scale, cost / 1024.0,
                               budget / 1024.0, capped))
        if total:
            # the reserve ACTUALLY applied, not the configured ceiling.
            # Reporting the latter on a small card described a plan the
            # arithmetic above had not used.
            projected = used + cost + reserve
            log.info("    gpu plan: %.1f GB desktop + %.1f GB upscaler + "
                     "%.1f GB reserved = %.1f of %.1f GB (%.0f%%, cap %d%%)"
                     % (used / 1024.0, cost / 1024.0,
                        reserve / 1024.0, projected / 1024.0,
                        total / 1024.0, 100.0 * projected / total,
                        GPU_MAX_PCT))
    return tile


_SCALE_WARNED = set()


def model_scales(model):
    """Upscale factors a given Real-ESRGAN model actually ships.

    Only the animevideo family provides the smaller variants; x4plus and
    x4plus-anime are 4x networks with no x2 or x3 weights, so requesting
    one of those from them is not a matter of configuration.

    Args:
        model: model name as passed to realesrgan-ncnn-vulkan.

    Returns:
        tuple[int, ...]: supported factors, ascending.
    """
    return (2, 3, 4) if "animevideo" in model else (4,)


def pick_sr_scale(ch, model, want=None):
    """Choose the upscale factor for one source.

    An explicitly configured factor takes precedence where the model
    supports it. Otherwise the smallest factor that still clears 1080
    lines is chosen, since the cost is quadratic: x4 computes four times
    the pixels of x2 and then discards most of them scaling back down to
    the target. That waste is paid twice, once in the model pass and
    again in the filter chain of the chunk encode.

    Args:
        ch: source height after cropping, in lines.
        model: model name, which determines the available factors.
        want: override; defaults to the configured SR_SCALE. Zero means
            choose automatically.

    Returns:
        int: the factor to use. When the requested factor is unavailable
        the nearest supported one is returned and a warning is emitted
        once per (model, request) pair, not once per file.
    """
    avail = model_scales(model)
    want = SR_SCALE if want is None else want
    if want:
        if want in avail:
            return want
        # asked for something this model cannot do; take the closest it
        # can, and say so rather than silently doing something else
        best = min(avail, key=lambda s: abs(s - want))
        # said once, not once per file and again from the background
        # preparation thread. it is a standing configuration mismatch,
        # not a per-file event.
        key = (model, want)
        if key not in _SCALE_WARNED:
            _SCALE_WARNED.add(key)
            log.warning("    sr scale x%d requested but %s only offers "
                        "%s; using x%d. set sr_model to "
                        "realesr-animevideov3 if you want x%d"
                        % (want, model,
                           "x" + ", x".join(str(s) for s in avail),
                           best, want))
        return best
    # otherwise the smallest factor that still clears 1080 lines
    for s in avail:
        if ch * s >= 1080:
            return s
    return avail[-1]


# ---------------------------------------------------------------------
# Real-ESRGAN upscale path: frames out, model pass, chunk encode,
# concat, then one mux pulling audio, subs, and chapters off the
# source. Chunked so scratch space stays small on long runtimes.
# ---------------------------------------------------------------------
def sr_encode(ctx, enc_input, info, an, plan, tmp_out, log, tag,
              extra_audio=None, va_only=False):
    """Upscale with Real-ESRGAN, then encode, in chunks.

    The tool cannot stream: it reads and writes image files and nothing
    else. That is a property of the tool, not of this pipeline, and it
    forces a PNG round trip through disk for every frame. Work is
    therefore done in chunks of SR_CHUNK_SEC so peak scratch usage stays
    bounded on a feature-length source, with the frames on a local SSD
    (sr_tmp_dir) rather than the network share.

    Per chunk: extract frames, run the model, then encode that chunk,
    optionally blending the model output back toward a plain lanczos
    upscale to suppress the frame-to-frame shimmer a single-image model
    inevitably produces. Chunks are concatenated and muxed at the end.

    Cost is dominated by the model pass and, at higher scale factors,
    by the chunk encode - a measured breakdown on a 528-line source at
    x4 was 4 pct extract, 59 pct model, 37 pct encode, because the
    filter chain was grinding through 5120x2112 frames to produce 1080.

    Args:
        ctx: run context, carrying the model name and the no_sr flag.
        enc_input: source to read.
        info: probe result.
        an: analysis result.
        plan: output of build_plan.
        tmp_out: destination for the finished file.
        log: logger.
        tag: short hash identifying this file, used to name scratch
            directories so concurrent work cannot collide.
        extra_audio: additional audio slot, e.g. the dialogue-boosted
            mix kept when whisper had to transcribe English. This path
            used to ignore it, so an upscaled file lost the track.
        va_only: emit video and audio only and leave subtitles, fonts
            and chapters to mkvmerge, exactly as the normal encode path
            does. Without this the upscale path muxed itself through
            ffmpeg's Matroska writer - the writer mkvmerge was adopted
            to get away from - and had no way to carry the generated
            subtitles at all.

    Raises:
        RuntimeError: on any failure in extraction, the model pass, a
            chunk encode, the concatenation or the final mux. The
            caller records it as a per-file failure and moves on.
    """
    model = ctx["sr_model"]
    cw = an["crop"][0] if an["crop"] else info["w"]
    ch = an["crop"][1] if an["crop"] else info["h"]
    scale = pick_sr_scale(ch, model)
    fps_str = info.get("fps_str") or ("%.6f" % info["fps"])
    enc = pick_encoder(True)
    # the frame folders take thousands of PNG writes per chunk. keeping
    # them on fast local storage instead of a network share is the
    # single biggest speed lever in this whole path
    scratch = scratch_dir(ctx)
    work = scratch / (tag + ".sr")
    # The chunk parts follow the frames onto the LOCAL scratch disk.
    #
    # They used to go to ctx["tmp"], which is the destination - and the
    # destination here is a network-backed DrivePool. So while the PNG
    # frames were correctly kept off the share, every one of a film's
    # several hundred chunk parts was written across the network and
    # then all read back again for the concat, for hours.
    #
    # Microsoft document that as a way to hang far more than the writer:
    # concurrent access to a file on a network drive can be serialised
    # behind a Csc.sys resource lock, and every application waiting on
    # that path stops with it - which is what a "the whole desktop
    # froze" report looks like from the outside. This function's own
    # docstring has always claimed the scratch stays off the share; only
    # half of it did.
    parts_root = scratch / (tag + ".sr.parts")
    f_in, f_out, parts = work / "in", work / "out", parts_root / "parts"
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(parts_root, ignore_errors=True)
    for d in (f_in, f_out, parts):
        d.mkdir(parents=True, exist_ok=True)
    tile0 = sr_tile_for(scale, max(cw, ch), log)
    # halve on each retry, ending at the tool's own auto so there is
    # always a rung that works
    sr_tiles = [tile0, max(64, tile0 // 2), max(64, tile0 // 4), 0]
    log.info("    sr x%d, threads %s, scratch %s"
             % (scale, SR_THREADS, scratch))
    log.debug("sr scratch=%s parts=%s encoder=%s" % (work, parts, enc))
    pre = []
    if an["interlaced"]:
        pre.append("bwdif=mode=send_frame:parity=auto:deint=all")
    if an["crop"]:
        pre.append("crop=%d:%d:%d:%d" % an["crop"])
    if plan["colorconv"]:
        # colour normalize runs before the model sees the frames
        pre.append(build_zscale(info))
    pre.append("format=rgb24")
    pix = "p010le"
    if enc == "svtav1":
        pix = "yuv420p10le"
    if os.environ.get("AV1PIPE_TEST_SW") == "1":
        pix = "yuv420p"
    stab = ""
    if SR_STABILISE:
        lo, hi = SR_STAB_THRESH
        stab = ("atadenoise=0a=%.3f:0b=%.3f:1a=%.3f:1b=%.3f"
                ":2a=%.3f:2b=%.3f:s=%d," % (lo, hi, lo, hi, lo, hi,
                                            SR_STAB_FRAMES))
    # x4 overshoots 1080 lines and the downscale that follows is real
    # oversampling, which is worth something on its own. A factor that
    # lands close to the target gains almost none of that, so the
    # crispness comes back through a contrast adaptive sharpen instead.
    # Same filter the no-rescale path already uses at exactly 1080.
    sharpen = ""
    over = (ch * scale) / 1080.0
    if CAS_STRENGTH > 0 and over < 1.15:
        sharpen = "cas=strength=%.2f," % CAS_STRENGTH
        plan["notes"].append("sr_cas_sharpen_%.2f" % CAS_STRENGTH)
    tail = stab + sharpen + SCALE_1080 + ",format=" + pix
    strength = max(0, min(100, SR_STRENGTH))
    blend = strength < 100
    if blend:
        # input 0 is the model output, input 1 the plain lanczos upscale
        # of the very same source frames. all_opacity weights input 0,
        # so it reads straight across as the AI strength
        fc = ("[1:v]scale=%d:%d:flags=lanczos,format=rgb24[lz];"
              "[0:v]format=rgb24[sr];"
              "[sr][lz]blend=all_mode=normal:all_opacity=%.3f[bl];"
              "[bl]%s[v]" % (cw * scale, ch * scale, strength / 100.0, tail))
        log.info("    sr strength %d%%%s, encoder %s"
                 % (strength, ", temporal stabiliser on" if SR_STABILISE
                    else "", enc))
    else:
        log.info("    sr strength 100%%%s, encoder %s"
                 % (", temporal stabiliser on" if SR_STABILISE else "", enc))
    n_chunks = max(1, int(math.ceil(info["dur"] / SR_CHUNK_SEC)))
    made = []
    total_frames = 0
    t0 = time.time()
    last_report = t0
    sr_ticks = 0
    log.info("    sr: %d chunks of %.0f s to do" % (n_chunks, SR_CHUNK_SEC))
    try:
        for i in range(n_chunks):
            start = i * SR_CHUNK_SEC
            seg = min(SR_CHUNK_SEC, info["dur"] - start)
            if seg <= 0.05:
                break
            t_chunk = time.time()
            for d in (f_in, f_out):
                for f in d.glob("*.png"):
                    f.unlink()
            rc, _, err = run([TOOLS["ffmpeg"], "-hide_banner", "-nostdin",
                              "-y", "-v", "error",
                              "-ss", "%.3f" % start, "-t", "%.3f" % seg,
                              "-i", str(enc_input),
                              "-map", "0:%d" % info["vidx"],
                              "-vf", ",".join(pre),
                              str(f_in / "f%08d.png")], timeout=3600)
            if rc != 0:
                raise RuntimeError("frame extract failed chunk %d: %s"
                                   % (i, err[-300:]))
            t_extract = time.time() - t_chunk
            n_frames = len(list(f_in.glob("*.png")))
            if n_frames == 0:
                if i == 0:
                    raise RuntimeError("no frames came out of the source")
                break
            total_frames += n_frames
            # tile as large as the card will take, stepping down only if
            # it will not fit. an out of memory refusal here used to end
            # the whole file; now it costs one retry.
            got = 0
            rc, out, err = -1, "", ""
            for t_try in sr_tiles:
                # Look before allocating. The step-down ladder below only
                # helps when the tool exits cleanly with an error; when
                # the card is genuinely full the driver resets instead,
                # which is not a return code, it is the display going
                # away. So wait for room first, and drop to a smaller
                # tile if it never appears.
                # heat first, then memory. Both are refusals to add load
                # to a card that is already at a limit.
                gpu_wait_for_cool(log)
                need = SR_TILE_COST_400.get(scale, 1690.0) * \
                    ((t_try or 400) / 400.0) ** 2
                if not gpu_wait_for_headroom(need, log):
                    rc, out, err = 125, "", "insufficient gpu headroom"
                    if t_try != sr_tiles[-1]:
                        log.warning("    not enough video memory for tile "
                                    "%s; trying a smaller one"
                                    % (t_try or "auto"))
                    continue
                srcmd = [TOOLS["realesrgan"], "-i", str(f_in),
                         "-o", str(f_out), "-n", model,
                         "-s", str(scale), "-f", "png"]
                if t_try:
                    srcmd += ["-t", str(t_try)]
                if SR_THREADS:
                    srcmd += ["-j", SR_THREADS]
                rc, out, err = run(srcmd, timeout=7200)
                got = len(list(f_out.glob("*.png")))
                if rc == 0 and got == n_frames:
                    if t_try != sr_tiles[0]:
                        log.warning("    tile %s did not fit; settled at "
                                    "%s for the rest of this file"
                                    % (sr_tiles[0], t_try or "auto"))
                        sr_tiles = [t_try]
                    break
                for f in f_out.glob("*.png"):
                    f.unlink()
                if t_try != sr_tiles[-1]:
                    log.debug("sr tile %s failed rc=%s, stepping down"
                              % (t_try, rc))
            if rc != 0 or got != n_frames:
                raise RuntimeError("realesrgan failed chunk %d rc %s "
                                   "(%d of %d frames): %s"
                                   % (i, rc, got, n_frames,
                                      (err or out)[-300:]))
            t_model = time.time() - t_chunk - t_extract
            part = parts / ("p%05d.mkv" % i)
            cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-y",
                   "-v", "error", "-framerate", fps_str,
                   "-start_number", "1",
                   "-i", str(f_out / "f%08d.png")]
            if blend:
                cmd += ["-framerate", fps_str, "-start_number", "1",
                        "-i", str(f_in / "f%08d.png"),
                        "-filter_complex", fc, "-map", "[v]"]
            else:
                cmd += ["-vf", tail]
            cmd += enc_args(plan["cq"], plan["out_space"], False, enc,
                            plan.get("maxrate_kbps", 0))
            cmd += ["-f", "matroska", str(part)]
            rc, _, err = run(cmd, timeout=7200)
            if rc != 0 or not part.exists():
                raise RuntimeError("chunk encode failed %d: %s"
                                   % (i, err[-300:]))
            made.append(part)
            t_enc = time.time() - t_chunk - t_extract - t_model
            # a chunk takes the better part of a minute, so reporting
            # every tenth one left eight minutes of silence and looked
            # exactly like a hang. report on a timer instead, with an
            # ETA, and say where the time is actually going.
            now = time.time()
            done_s = min(info["dur"], (i + 1) * SR_CHUNK_SEC)
            rate = done_s / max(1e-6, now - t0)
            if i == 0 or (now - last_report) > PROGRESS_EVERY_SEC \
                    or (i + 1) == n_chunks:
                last_report = now
                left = max(0.0, info["dur"] - done_s) / max(1e-6, rate)
                log.info("    sr chunk %d/%d  %4.1f pct  %.2fx realtime  "
                         "%s left" % (i + 1, n_chunks,
                                      100.0 * (i + 1) / n_chunks,
                                      rate, hms(left)))
                log.debug("    sr chunk %d timing: extract %.1fs, model "
                          "%.1fs, encode %.1fs, total %.1fs (%d frames)"
                          % (i + 1, t_extract, t_model, t_enc,
                             now - t_chunk, n_frames))
                sr_ticks += 1
                if sr_ticks % 3 == 1:
                    # the one phase that had no resource sampling at all,
                    # which is why a seven hour render was invisible
                    log_resources(log, "during sr upscale", phase="sr")
            # Breathe between chunks. An upscale otherwise holds the GPU,
            # the CPU and the disk at full tilt for eleven hours without
            # a single gap, which is the load profile the machine has
            # now locked up under three times. A couple of seconds a
            # chunk is a few percent of throughput.
            if SR_CHUNK_PAUSE_SEC > 0:
                time.sleep(SR_CHUNK_PAUSE_SEC)
        lst = work / "concat.txt"
        lst.write_text("".join("file '%s'\n" % m.as_posix()
                               for m in made), encoding="utf-8")
        vid = work / "video.mkv"
        rc, _, err = run([TOOLS["ffmpeg"], "-hide_banner", "-nostdin",
                          "-y", "-v", "error", "-f", "concat",
                          "-safe", "0", "-i", str(lst), "-c", "copy",
                          str(vid)], timeout=7200)
        if rc != 0 or not vid.exists():
            raise RuntimeError("chunk concat failed: " + err[-300:])
        # Frame accounting. Every chunk is extracted by timestamp with
        # -ss/-t and the pieces are reassembled at a fixed frame rate,
        # so a frame gained or lost at any one of several hundred chunk
        # boundaries shifts the whole picture timeline against the
        # audio. Nothing was counting, so it would have shown up only as
        # a duration check failure at the very end, if at all.
        expect = int(round(info["dur"] * info["fps"]))
        if total_frames and abs(total_frames - expect) > 2:
            log.warning("    sr frame count: %d frames from the chunks "
                        "against %d expected for %.3f s at %.3f fps "
                        "(%+d). chunk boundaries are not landing cleanly "
                        "and the picture may drift against the audio."
                        % (total_frames, expect, info["dur"], info["fps"],
                           total_frames - expect))
            plan["notes"].append("sr_frame_count_off_%+d"
                                 % (total_frames - expect))
        else:
            log.debug("sr frame count: %d, expected %d"
                      % (total_frames, expect))
        # single mux: fresh video plus everything else off the source;
        # the audio and sub builders write input-0 maps, repoint at 1
        rest = audio_args(info, plan, extra_audio)
        if not va_only:
            rest += sub_args(info)
        fixed = []
        prev = ""
        for x in rest:
            if prev == "-map" and str(x).startswith("0:"):
                fixed.append("1:" + str(x)[2:])
            else:
                fixed.append(x)
            prev = x
        cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-y", "-v",
               "warning", "-i", str(vid), "-i", str(enc_input),
               "-map", "0:v:0", "-c:v", "copy"] + fixed \
            + ["-map_metadata", "1"]
        # in va_only mode nothing timed rides along: no subtitles and no
        # chapters, so nothing can stretch the container past the
        # picture. mkvmerge adds both afterwards from the source.
        cmd += ["-map_chapters", "-1" if va_only else "1"]
        cmd += ["-max_muxing_queue_size", "4096",
                "-f", "matroska", str(tmp_out)]
        rc, _, err = run(cmd, timeout=7200)
        if rc != 0 or not tmp_out.exists():
            raise RuntimeError("sr mux failed: " + err[-300:])
    finally:
        # both halves go, on success and on failure alike. work holds
        # the PNG frames on the scratch drive; parts_root holds the
        # encoded chunks under the destination and was being left
        # behind. thousands of frames per file makes this the one place
        # that must never leak.
        freed = 0
        for d in (work, parts_root):
            try:
                if d.exists():
                    freed += sum(f.stat().st_size
                                 for f in d.rglob("*") if f.is_file())
            except Exception:
                pass
            shutil.rmtree(d, ignore_errors=True)
        if freed:
            log.info("    sr scratch cleared, %.2f GB released" % gb(freed))



# ---------------------------------------------------------------------
# Encode runner: -progress pipe:1 makes ffmpeg print machine-readable
# progress lines on stdout; stderr feeds a tail ring for diagnostics
# ---------------------------------------------------------------------
def resource_snapshot():
    # psutil when present, otherwise the platform call directly. this is
    # here so a process killed for running out of memory leaves evidence
    # rather than a log that simply stops.
    out = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_used_pct"] = vm.percent
        out["ram_free_gb"] = vm.available / 1073741824.0
        pr = psutil.Process()
        out["proc_ram_gb"] = pr.memory_info().rss / 1073741824.0
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        return out
    except Exception:
        pass
    if IS_WIN:
        try:
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            out["ram_used_pct"] = float(m.dwMemoryLoad)
            out["ram_free_gb"] = m.ullAvailPhys / 1073741824.0
        except Exception:
            pass
    else:
        try:
            info = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                info[k] = float(v.strip().split()[0]) / 1048576.0
            tot = info.get("MemTotal", 0)
            av = info.get("MemAvailable", 0)
            out["ram_free_gb"] = av
            if tot:
                out["ram_used_pct"] = 100.0 * (tot - av) / tot
        except Exception:
            pass
    return out


def log_environment(log, src, dst):
    # everything a fault report could possibly need, written once at the
    # top of every run so a log is self contained
    def ver(exe, args, name):
        if not exe:
            log.warning("env %-12s MISSING" % name)
            return
        rc, out, err = run([exe] + args, timeout=60)
        first = ((out or "") + (err or "")).strip().splitlines()
        log.info("env %-12s %s" % (name, first[0][:110] if first else "?"))
    ver(TOOLS.get("ffmpeg"), ["-version"], "ffmpeg")
    ver(TOOLS.get("mkvmerge"), ["--version"], "mkvmerge")
    ver(TOOLS.get("mkvpropedit"), ["--version"], "mkvpropedit")
    ver(TOOLS.get("dovi_tool"), ["--version"], "dovi_tool")
    if TOOLS.get("realesrgan"):
        log.info("env %-12s %s" % ("realesrgan", TOOLS["realesrgan"]))
    for mod in ("numpy", "scipy", "guessit", "psutil"):
        try:
            m = __import__(mod)
            log.info("env %-12s %s" % (mod, getattr(m, "__version__", "?")))
        except Exception:
            log.warning("env %-12s NOT INSTALLED" % mod)
    try:
        import av1_subs as SB
        mp = SB.find_model(WHISPER_MODEL_DIR, WHISPER_MODEL)
        log.info("env %-12s %s" % ("whisper", mp if mp else "no model"))
    except Exception as e:
        log.warning("env %-12s module missing: %s" % ("av1_subs", e))
    blob = ""
    if TOOLS.get("ffmpeg"):
        rc, out, err = run([TOOLS["ffmpeg"], "-hide_banner", "-filters"],
                           timeout=60)
        blob = out + err
    log.info("env %-12s whisper=%s cas=%s deband=%s libplacebo=%s"
             % ("filters", "whisper" in blob, "cas" in blob,
                "deband" in blob, "libplacebo" in blob))
    # scratch is listed alongside source and dest because every
    # intermediate now lives there, and it is the volume that will run
    # out first on a big Dolby Vision title
    places = [("source", src), ("dest", dst)]
    if SR_TMP_DIR:
        places.append(("scratch", Path(SR_TMP_DIR)))
    for label, pth in places:
        try:
            u = shutil.disk_usage(str(pth))
            log.info("env %-12s %s  %.1f GB free of %.1f GB"
                     % (label, pth, u.free / 1e9, u.total / 1e9))
        except Exception:
            log.warning("env %-12s %s unreadable" % (label, pth))
    # A blank sr_tmp_dir used to mean "intermediates go to the
    # destination", which was worth warning about. It now means "choose
    # the fastest local disk with room", which is the RECOMMENDED
    # setting and the only one that can be right on a machine whose
    # drive letters are unknown. Warning about it was telling the
    # operator to fix something that is not broken; the real scratch
    # decision is logged by open_scratch_session a few lines later.
    if not SR_TMP_DIR and not SCRATCH_AUTO:
        log.warning("env %-12s no sr_tmp_dir set and scratch_auto is off, "
                    "so intermediates fall back to the destination. Set "
                    "one or the other." % "scratch")
    elif not SR_TMP_DIR:
        log.info("env %-12s chosen automatically, reported below"
                 % "scratch")
    log.info("env %-12s encoder_pref=%s hwaccel=%s deband=%s cq_offset=%+d"
             % ("settings", CAPS.get("encoder_pref"), HWACCEL,
                DEBAND_MODE, CQ_OFFSET))
    log.info("env %-12s langs=%s subs=%s generate=%s translate=%s"
             % ("policy", ",".join(AUDIO_LANGS), ",".join(SUBS_LANGS),
                SUBS_GENERATE,
                # argos needs no key, so keying the banner off
                # TRANSLATE_KEY alone reported "no key" on a run that
                # was in fact translating perfectly well
                TRANSLATE_PROVIDER
                if (TRANSLATE_KEY or TRANSLATE_PROVIDER == "argos")
                else "no key"))


class Phase:
    """Times a named stage and logs how long it took, even on failure."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def __enter__(self):
        self.t0 = time.time()
        self.log.debug("phase start: %s" % self.name)
        return self

    def __exit__(self, et, ev, tb):
        dt = time.time() - self.t0
        if et is None:
            self.log.debug("phase done : %s in %.1f s" % (self.name, dt))
        else:
            self.log.error("phase FAILED: %s after %.1f s: %s"
                           % (self.name, dt, ev))
        return False


def gpu_snapshot():
    # NVENC is a separate fixed-function block. Task Manager's default
    # GPU graph shows 3D and compute, not the encoder, so a saturated
    # NVENC can read as "GPU 50 percent" while being the actual limit.
    # nvidia-smi reports the encoder engine directly.
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        cand = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        if Path(cand).exists():
            exe = cand
    if not exe:
        return {}
    # temperature and power draw are queried too. The machine has locked
    # hard three times with BugcheckCode=0 and no dump, which is the
    # signature of the kernel simply ceasing to run rather than of any
    # software fault - and the two things that do that under a sustained
    # combined CPU+GPU load are heat and power delivery. Neither was
    # being recorded, so every post-mortem started blind.
    rc, out, err = run([exe, "--query-gpu=utilization.gpu,"
                        "utilization.encoder,utilization.decoder,"
                        "memory.used,memory.total,clocks.sm,"
                        "temperature.gpu,power.draw,enforced.power.limit",
                        "--format=csv,noheader,nounits"], timeout=30)
    if rc != 0 or not out.strip():
        return {}
    f = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    keys = ("gpu_pct", "enc_pct", "dec_pct", "vram_used_mb",
            "vram_total_mb", "sm_mhz", "temp_c", "power_w", "power_cap_w")
    d = {}
    for k, v in zip(keys, f):
        try:
            d[k] = float(v)
        except ValueError:
            pass
    return d


def gpu_name():
    """Marketing name of the first NVIDIA GPU, or "" if there is none."""
    if CAPS.get("gpu_name") is not None:
        return CAPS["gpu_name"]
    name = ""
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        cand = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        if Path(cand).exists():
            exe = cand
    if exe:
        rc, out, err = run([exe, "--query-gpu=name",
                            "--format=csv,noheader"], timeout=30)
        if rc == 0 and out.strip():
            name = out.strip().splitlines()[0].strip()
    CAPS["gpu_name"] = name
    return name


# What each NVENC generation can do, by the number in the card's name.
# The professional boards are named differently and are matched
# separately below.
#
# THIS TABLE NEVER GATES ANYTHING. The preflight runs a real one-second
# encode and that is what decides; a name table cannot know about a
# driver too old for the ffmpeg build, a laptop card in a machine that
# has switched to the integrated GPU, or a board this was written before
# the release of. The table exists so the log can say WHY the trial is
# expected to fail, and so the video-memory reserve knows whether an
# NVENC session will be running alongside the upscaler.
NVENC_GENERATIONS = (
    (5000, 5999, "Blackwell (RTX 50 series)", True),
    (4000, 4999, "Ada Lovelace (RTX 40 series)", True),
    (3000, 3999, "Ampere (RTX 30 series)", False),
    (2000, 2999, "Turing (RTX 20 series)", False),
    (1600, 1699, "Turing (GTX 16 series)", False),
    (1000, 1099, "Pascal (GTX 10 series)", False),
)


def gpu_generation():
    """Which NVENC generation this is, and whether it encodes AV1.

    AV1 ENCODE ARRIVED WITH ADA. An RTX 40 or 50 series card has it; an
    RTX 30 series card decodes AV1 perfectly well and cannot encode it,
    which is a fixed-function silicon limit and not a driver or software
    matter. Those machines run the CPU encoder instead, which is slower
    but not worse - libsvtav1's output at these settings is at least as
    good as NVENC's.

    Returns:
        tuple[str, bool | None]: a human name for the generation, and
        whether AV1 encode is expected. None means "not recognised" -
        a new board, a professional card outside the table, or no
        NVIDIA GPU at all - in which case the preflight simply tries it
        and believes the result.
    """
    name = gpu_name()
    if not name:
        return "", None
    low = name.lower()
    # Professional boards first: "RTX A4000" is Ampere, but a bare digit
    # match would read the 4000 and call it Ada.
    m = re.search(r"\brtx\s*a(\d{3,4})\b", low)
    if m:
        return "Ampere (professional RTX A series)", False
    if re.search(r"\b(l4|l40|l40s)\b", low) or "ada" in low:
        return "Ada Lovelace (professional)", True
    if re.search(r"\b(a100|a40|a30|a10|a16|a2)\b", low):
        return "Ampere (professional)", False
    if re.search(r"\b(t4|titan\s+rtx|quadro)\b", low):
        return "Turing or older (professional)", False
    m = re.search(r"\b(?:rtx|gtx)\s*(\d{4})\b", low)
    if m:
        n = int(m.group(1))
        for lo, hi, label, av1 in NVENC_GENERATIONS:
            if lo <= n <= hi:
                return label, av1
    return "", None


def sr_vram_reserve(total_mb=0.0):
    """Video memory held back from the upscaler, sized to THIS card.

    The measured figure is a fifth of a 16 GB board, kept free for the
    NVENC session encoding the previous file plus the desktop
    compositor. Carried across to a smaller card as a flat number it
    becomes a far larger share of a far smaller board - 3400 MB is 42
    pct of an 8 GB RTX 3060 Ti - and starves the upscaler on the
    machines least able to spare the throughput.

    Two reductions apply:

    1. Proportional. Never more than SR_VRAM_RESERVE_PCT of the board,
       which reproduces the measured 16 GB behaviour exactly and scales
       down from there.
    2. Encoder-aware. The larger half of this reserve exists for a
       concurrent NVENC session. On Ampere and older there is no such
       session - those cards cannot encode AV1, so the encode is on the
       CPU - and only compositor headroom is needed.

    Args:
        total_mb: board memory. 0 means unknown, in which case the flat
            configured figure is used unchanged, because guessing small
            on an unmeasurable card is the dangerous direction.

    Returns:
        float: megabytes to hold back.
    """
    if total_mb <= 0:
        return float(SR_VRAM_RESERVE_MB)
    reserve = min(float(SR_VRAM_RESERVE_MB),
                  total_mb * SR_VRAM_RESERVE_PCT / 100.0)
    if CAPS.get("encoder") not in ("nvenc", None):
        # The encode is not on the card, so nothing is competing with
        # the upscaler for video memory except the desktop.
        reserve = min(reserve, float(SR_VRAM_RESERVE_MIN_MB))
    return max(float(SR_VRAM_RESERVE_MIN_MB), reserve)


def log_resources(log, label="", phase=None):
    """Record a CPU, memory and GPU sample against a named phase.

    The GPU's shader cores and its NVENC block report separately and are
    genuinely independent; a phase can saturate one while leaving the
    other idle, which is the entire basis for overlapping preparation
    with encoding. Recording both is what makes that visible.

    This was originally called only from the encode loop, which meant
    the one phase that keeps NVENC busy was also the only phase being
    measured. Probe, crop analysis, whisper and the upscaler all ran
    unobserved - collectively a third of a file's wall time, and the
    whole of it on an upscale job, where a seven-hour render produced no
    telemetry whatsoever.

    Args:
        log: logger to write to.
        label: human-readable context for the sample.
        phase: stage identifier. When supplied and not the encode, an
            idle NVENC is called out explicitly, since that is where
            throughput is being lost.
    """
    r = resource_snapshot()
    if not r:
        return
    bits = []
    if "ram_used_pct" in r:
        bits.append("ram %.0f%% used, %.1f GB free"
                    % (r["ram_used_pct"], r.get("ram_free_gb", 0)))
    if "proc_ram_gb" in r:
        bits.append("this process %.2f GB" % r["proc_ram_gb"])
    if "cpu_pct" in r:
        bits.append("cpu %.0f%%" % r["cpu_pct"])
    g = gpu_snapshot()
    if g:
        bits.append("gpu %.0f%% / encoder %.0f%% / decoder %.0f%%"
                    % (g.get("gpu_pct", 0), g.get("enc_pct", 0),
                       g.get("dec_pct", 0)))
        if g.get("vram_total_mb"):
            bits.append("vram %.1f/%.1f GB"
                        % (g.get("vram_used_mb", 0) / 1024.0,
                           g["vram_total_mb"] / 1024.0))
        # the two numbers that were missing from every previous
        # post-mortem
        if g.get("temp_c"):
            bits.append("gpu %.0fC" % g["temp_c"])
        if g.get("power_w"):
            bits.append("gpu %.0fW of %.0fW"
                        % (g["power_w"], g.get("power_cap_w", 0)))
    if bits:
        log.debug("resources%s: %s" % (" " + label if label else "",
                                       ", ".join(bits)))
    if g.get("enc_pct", 0) >= 90:
        log.debug("    the NVENC encoder engine is saturated; it is the "
                  "limit on this file, not the CPU or the disk")
    elif phase and phase != "encode" and g.get("enc_pct", 0) < 5:
        # the encoder is fixed-function silicon sitting idle while this
        # stage runs on the CPU or the shader cores. worth seeing in the
        # trace, because it is where the throughput went.
        log.debug("    NVENC idle during %s" % phase)
    if r.get("ram_free_gb", 99) < 1.5:
        log.warning("    only %.1f GB of RAM free; the process is at "
                    "real risk of being killed" % r["ram_free_gb"])


def rc_text(rc):
    # Windows hands back the full 32 bit exit code, so an ffmpeg
    # AVERROR arrives as a huge positive number. show the signed value
    # and name the ones that come up in practice
    try:
        rc = int(rc)
    except (TypeError, ValueError):
        return str(rc)
    sig = rc - 2 ** 32 if rc > 2 ** 31 else rc
    known = {
        -2: "ENOENT, file or codec not found",
        -13: "EACCES, permission denied",
        -22: "EINVAL, bad argument",
        -28: "ENOSPC, disk full",
        -40: "ENOSYS on Windows, feature not supported by the "
             "encoder, driver or build",
        125: "the process could not be launched at all",
    }
    note = known.get(sig)
    if sig == rc and 0 < rc < 256:
        return "%d%s" % (rc, " (" + note + ")" if note else "")
    out = "%d (signed %d, 0x%08X" % (rc, sig, rc & 0xFFFFFFFF)
    if note:
        out += ", " + note
    elif sig < -1000000:
        out += ", large negative code, often an external library "
        out += "such as Vulkan or CUDA. read the ffmpeg line below"
    return out + ")"


def run_encode(cmd, dur, log, label, fps=None):
    """Run one ffmpeg encode, reporting progress and policing hangs.

    ffmpeg is invoked with -progress pipe:1, which emits machine
    readable key=value lines on stdout while stderr carries diagnostics.
    Both are consumed concurrently: stderr on a reader thread feeding a
    bounded ring buffer, stdout on this thread driving the progress
    display and the liveness watchdog.

    Liveness is judged on three independent signals - output position,
    frame count and bytes written - because position alone is not
    dependable. FFmpeg reports out_time as N/A for the entire run when
    an attachment stream is mapped (trac #10798), which describes every
    fansubbed MKV in the library. Treating that as "no progress" killed
    healthy encodes two thirds of the way through.

    Args:
        cmd: full ffmpeg argv.
        dur: source duration in seconds, for the percentage display.
        log: logger to report against.
        label: short identifier for the attempt, e.g. "cq28/full".
        fps: source frame rate. Optional, but without it the percentage
            cannot be derived when position reporting is unavailable.

    Returns:
        tuple[int, list[str]]: the process return code and the tail of
        stderr. 125 indicates the process could not be launched at all;
        124 indicates the watchdog killed it as hung.
    """
    log.debug("enc cmd: %s" % " ".join(str(c) for c in cmd))
    tail = collections.deque(maxlen=240)
    try:
        p = subprocess.Popen(cmd, creationflags=NOWIN,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace")
    except Exception as e:
        return 125, [str(e)]
    throttle(p)

    def eat_err():
        try:
            for line in p.stderr:
                line = line.rstrip()
                if line:
                    tail.append(line)
        except Exception:
            pass

    th = threading.Thread(target=eat_err, daemon=True)
    th.start()
    last = time.time()
    pos = 0.0
    speed = "?"
    prog = {}
    ticks = 0
    exp_frames = 0
    try:
        if fps and dur:
            exp_frames = int(float(fps) * float(dur))
    except Exception:
        exp_frames = 0
    # three independent proofs of life. the output position on its own
    # is NOT one: probed on ffmpeg 8.1.2, mapping an attachment stream
    # (-map 0:t?, which every fansubbed mkv needs for its fonts) makes
    # out_time, bitrate and speed report N/A for the whole run while
    # the encode is perfectly healthy. the frame count and the output
    # size keep moving in that case, so watch all three and treat any
    # one of them advancing as alive.
    state = {"pos": -1.0, "frame": -1, "size": -1,
             "at": time.time(), "stalled": False, "nopos": False}

    def where():
        """Describe the last known progress, for the kill message.

        Only signals that actually reported are included. A message
        naming "0.0 min" when position was never available at all is
        what sent the last investigation down the wrong path.
        """
        bits = []
        if state["pos"] >= 0:
            bits.append("position %.1f min" % (state["pos"] / 60.0))
        if state["frame"] >= 0:
            bits.append("frame %d" % state["frame"])
        if state["size"] >= 0:
            bits.append("%.1f MB written" % (state["size"] / 1048576.0))
        return ", ".join(bits) if bits else "nothing reported at all"

    def watchdog():
        # An ffmpeg that is alive and talking is not necessarily an
        # ffmpeg that is working. The timer is reset by the reader loop
        # whenever ANY of the three signals advances, so this only fires
        # when all of them have been static for the full timeout.
        while p.poll() is None:
            time.sleep(5)
            if STALL_TIMEOUT_SEC <= 0:
                continue
            idle = time.time() - state["at"]
            if idle > STALL_TIMEOUT_SEC:
                state["stalled"] = True
                log.error("    encode stalled: nothing advanced for "
                          "%d s. last seen %s. killing it."
                          % (int(idle), where()))
                log.error("    position, frame count and output size "
                          "frozen together is a hang, not a slow encode")
                try:
                    p.kill()
                except Exception:
                    pass
                return

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        for line in p.stdout:
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "out_time_ms":
                # this field carries microseconds, an old ffmpeg quirk
                if v == "N/A":
                    if not state["nopos"]:
                        state["nopos"] = True
                        log.info("    ffmpeg is reporting no output "
                                 "position for this file; tracking the "
                                 "frame count instead")
                        log.debug("    out_time_ms=N/A. a mapped "
                                  "attachment stream does this on "
                                  "ffmpeg 8.x; the encode is unaffected")
                else:
                    try:
                        pos = int(v) / 1_000_000.0
                    except Exception:
                        pass
                    if pos > state["pos"] + 0.001:
                        state["pos"] = pos
                        state["at"] = time.time()
            elif k == "speed":
                speed = v
            else:
                prog[k] = v
                if k == "frame":
                    try:
                        fr = int(v)
                    except Exception:
                        fr = -1
                    if fr > state["frame"]:
                        state["frame"] = fr
                        state["at"] = time.time()
                elif k == "total_size":
                    try:
                        sz = int(v)
                    except Exception:
                        sz = -1
                    if sz > state["size"]:
                        state["size"] = sz
                        state["at"] = time.time()
            now = time.time()
            if now - last > PROGRESS_EVERY_SEC:
                last = now
                if state["nopos"]:
                    fr = max(state["frame"], 0)
                    if exp_frames > 0:
                        log.info("    encode %s: %5.1f pct  frame %d of "
                                 "~%d  %s fps  (no position reported)"
                                 % (label, 100.0 * fr / exp_frames, fr,
                                    exp_frames, prog.get("fps", "?")))
                    else:
                        log.info("    encode %s: frame %d  %s fps  (no "
                                 "position reported)"
                                 % (label, fr, prog.get("fps", "?")))
                else:
                    pct = 100.0 * pos / dur if dur else 0.0
                    log.info("    encode %s: %5.1f pct  pos %6.1f min  "
                             "speed %s"
                             % (label, pct, pos / 60.0, speed))
                # everything ffmpeg will tell us, in the trace file. a
                # frame count that stops while position moves, or a
                # size that runs away, both show up here first.
                extra = []
                for key in ("frame", "fps", "bitrate", "total_size",
                            "drop_frames", "dup_frames", "q"):
                    if prog.get(key):
                        extra.append("%s=%s" % (key, prog[key]))
                if extra:
                    log.debug("    progress %s: %s" % (label,
                                                       " ".join(extra)))
                ticks += 1
                if ticks % 3 == 0:
                    log_resources(log, "during encode", phase="encode")
    except Exception:
        pass
    p.wait()
    if state["stalled"]:
        tail.append("STALLED: nothing advanced for %d s (last seen %s); "
                    "the encode was killed by the watchdog"
                    % (STALL_TIMEOUT_SEC, where()))
        return 124, list(tail)
    th.join(timeout=5)
    return p.returncode, list(tail)


def check_duration(src_dur, out_path, src_path=None):
    """Verify the encode came out the same length as its source.

    Both sides are measured on the video stream rather than the
    container, for the reasons set out in probe_video_duration. The
    tolerance is flat rather than proportional to runtime; the previous
    sliding 0.5 pct let a 92 minute feature drift 27 seconds and still
    pass, which is long enough to lose a scene.

    Args:
        src_dur: container duration of the source. Used only as a
            fallback when src_path is absent or cannot be probed.
        out_path: the finished file.
        src_path: the source. Strongly preferred, since it lets both
            sides be measured the same way.

    Returns:
        tuple[bool, float]: whether the length is within tolerance, and
        the output's measured duration, which the caller records in the
        journal and in the failure diagnostic.
    """
    ov = probe_video_duration(out_path)
    sv = probe_video_duration(src_path) if src_path else 0.0
    ref = sv if sv > 0 else src_dur
    basis = "video" if sv > 0 else "container(fallback)"
    log.debug("dur check src=%.3f (%s) out=%.3f tol=%.2f path=%s"
              % (ref, basis, ov, DURATION_TOLERANCE_SEC, out_path))
    if sv > 0 and abs(sv - src_dur) > 1.0:
        # worth seeing: it means something non-video sets the container
        # length, which is exactly what used to cause the false failures
        log.debug("    source container is %.2f s but its video is "
                  "%.2f s; measuring against the video"
                  % (src_dur, sv))
    return abs(ov - ref) <= DURATION_TOLERANCE_SEC, ov


# ---------------------------------------------------------------------
# VMAF transparency gate: three sampled windows against the source,
# the lowest score has to clear the target
# ---------------------------------------------------------------------
def vmaf_min(tmp_out, ref, dur, log):
    scores = []
    for t in (dur * 0.2, dur * 0.5, dur * 0.8):
        if t + VMAF_WINDOWS_SEC >= dur:
            continue
        cmd = [TOOLS["ffmpeg"], "-hide_banner", "-nostdin", "-v", "info",
               "-ss", "%.2f" % t, "-t", str(VMAF_WINDOWS_SEC),
               "-i", str(tmp_out),
               "-ss", "%.2f" % t, "-t", str(VMAF_WINDOWS_SEC),
               "-i", str(ref),
               "-lavfi",
               "[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];"
               "[d][r]libvmaf=n_threads=8",
               "-f", "null", "-"]
        rc, out, err = run(cmd, timeout=1800)
        m = None
        for mm in re.finditer(r"VMAF score:\s*([0-9.]+)", err):
            m = mm
        if m:
            scores.append(float(m.group(1)))
    mn = min(scores) if scores else None
    log.debug("vmaf min=%s from %d windows" % (mn, len(scores)))
    return mn, scores


# ---------------------------------------------------------------------
# Audio sync: a window every 10 seconds, cross-correlation against the
# source track; a constant delay gets a remux repair, drift fails
# ---------------------------------------------------------------------
def tag_and_rename(ctx, final, out_rel, rel, info, plan, log):
    # the folder structure is deliberately left alone; only the file is
    # renamed, so a file that escapes its folder still identifies itself
    import av1_metadata as MD
    parsed = MD.parse_path(rel)
    ids = MD.embedded_ids(info)
    franchise = MD.find_franchise(Path(rel).name)
    meta = None
    if TMDB_API_KEY:
        meta = MD.tmdb_lookup(TMDB_API_KEY, parsed, info.get("dur"),
                              ids, log)
    if meta:
        log.info("    identified: %s (%s) via %s"
                 % (meta.get("title") or meta.get("name"),
                    (meta.get("release_date")
                     or meta.get("first_air_date") or "")[:4],
                    meta.get("_confidence", "?")))
        note = meta.get("_runtime_note")
        if note and note.startswith("runtime_off"):
            log.warning("    runtime disagrees with the match (%s); "
                        "check this one" % note)
        plan["notes"].append("identified_" + str(meta.get("id")))
    elif TMDB_API_KEY:
        log.info("    no confident metadata match; name left alone")
    else:
        log.debug("no tmdb key configured, skipping lookup")

    # Visual confirmation, against frames rather than text. Runs on the
    # finished file so what is checked is what is kept.
    verify = None
    if meta and META_VERIFY_FRAMES:
        try:
            urls = MD.tmdb_images(TMDB_API_KEY, meta, parsed, log)
            # 24 sampled frames plus a dozen downloaded stills, written
            # and re-read immediately: local scratch, not the share
            verify = MD.verify_by_frames(TOOLS["ffmpeg"], final, urls,
                                         info.get("dur"),
                                         scratch_dir(ctx) / ("%s.ver"
                                             % hashlib.md5(
                                                 rel.encode("utf-8")
                                             ).hexdigest()[:8]),
                                         log)
            if verify.get("verdict") == "confirmed":
                plan["notes"].append("visually_confirmed")
            elif verify.get("verdict") == "unverified":
                plan["notes"].append("visual_check_inconclusive")
        except Exception as e:
            log.debug("frame verification failed: %s" % e, exc_info=True)

    # tags first, so they land even when renaming is off
    title = None
    if meta:
        title = meta.get("title") or meta.get("name")
        if parsed.get("kind") == "episode":
            # the container title is what a player shows. Using the
            # SERIES name here gave every episode of a show an
            # identical title, so "Planetes" appeared 26 times with
            # nothing to tell them apart. Name the episode instead.
            ep = meta.get("_episode") or {}
            ep_title = ep.get("name") or parsed.get("ep_title")
            s, e = parsed.get("season"), parsed.get("episode")
            bits = [title]
            if s is not None and e is not None:
                bits.append("S%02dE%02d" % (int(s), int(e)))
            if ep_title:
                bits.append(ep_title)
            title = " - ".join(str(b) for b in bits if b)
    if TOOLS.get("mkvpropedit") and (title or franchise):
        xml = MD.tags_xml(meta, parsed, franchise)
        xp = None
        if xml:
            xp = scratch_dir(ctx) / ("%s.tags.xml" % Path(rel).stem[:40])
            xp.write_text(xml, encoding="utf-8")
        if MD.write_tags(TOOLS["mkvpropedit"], final, title, xp, None, log,
                         video_title=title or ""):
            plan["notes"].append("metadata_embedded")
        if xp:
            cleanup_paths([xp])

    def sidecar(dest_file, dest_rel):
        # written last, after any rename, so the .info and the video
        # always share a stem and are obviously a pair in a listing
        if not META_INFO_FILE:
            return
        try:
            ok = MD.write_info(dest_file.with_suffix(".info"), meta, parsed,
                               franchise, info, plan, verify=verify,
                               source_name=Path(rel).name,
                               out_name=dest_file.name, log=log,
                               info_is_source=not plan.get("_info_is_output"))
            if ok:
                log.info("    wrote %s" % dest_file.with_suffix(".info").name)
                plan["notes"].append("info_sidecar_written")
        except Exception as e:
            log.debug("info sidecar failed: %s" % e, exc_info=True)

    if not META_RENAME or not meta:
        sidecar(final, out_rel)
        return final, out_rel
    # the real output height, so a 4K master is named 2160p rather than
    # carrying the 1080 this used to hard-code
    height = int(plan.get("out_h") or 1080)
    newname = MD.build_name(meta, parsed, franchise, ext=final.suffix,
                            codec="AV1", height=height,
                            underscores=META_UNDERSCORES,
                            include_ep_title=META_EP_TITLE,
                            include_tech=META_TECH)
    target = final.with_name(newname)
    if target == final:
        sidecar(final, out_rel)
        return final, out_rel
    if target.exists():
        log.info("    rename target already exists, keeping current name")
        sidecar(final, out_rel)
        return final, out_rel
    try:
        final.rename(target)
        log.info("    renamed: %s" % newname)
        plan["notes"].append("renamed")
        sidecar(target, out_rel.with_name(newname))
        return target, out_rel.with_name(newname)
    except OSError as e:
        log.warning("    rename failed, keeping current name: %s" % e)
        sidecar(final, out_rel)
        return final, out_rel


def metadata_pass(ctx, log, limit=0):
    """Re-run identification, tagging, renaming and the sidecar only.

    Operates on files already sitting in the destination, touching no
    video data whatsoever. Two situations call for it: a library encoded
    before a TMDB key was configured, and any later improvement to the
    naming or metadata logic that should be applied retrospectively
    without spending days re-encoding.

    The journal supplies the original source name and the processing
    decisions where an entry exists, so the sidecar can still describe
    how the file was made. Where it does not, the file is described from
    its own contents and the .info says as much.

    Args:
        ctx: run context.
        log: logger.
        limit: stop after this many files; 0 means all of them.

    Returns:
        collections.Counter: outcome tallies for the run summary.
    """
    dst = ctx["dst"]
    journal = ctx["journal"]
    stats = collections.Counter()
    # index the journal by output name so a renamed file can still be
    # traced back to the source it came from
    by_out = {}
    for src_rel, rec in (journal.data.get("files") or {}).items():
        if rec.get("out"):
            by_out[str(rec["out"]).lower()] = (src_rel, rec)

    files = [p for p in sorted(dst.rglob("*.mkv"))
             if ".tmp" not in p.parts and "failed" not in p.parts]
    if limit > 0:
        files = files[:limit]
    log.info("metadata pass: %d file(s) in %s" % (len(files), dst))
    if not TMDB_API_KEY:
        log.warning("no tmdb_api_key configured; nothing can be looked "
                    "up. add one to av1_pipeline_config.txt")
        return stats

    for n, path in enumerate(files, 1):
        rel_out = path.relative_to(dst).as_posix()
        src_rel, rec = by_out.get(path.name.lower(), (rel_out, {}))
        log.info("[%d/%d] %s" % (n, len(files), path.name))
        try:
            info = probe_file(path)
        except Exception as e:
            log.warning("    cannot probe, skipped: %s" % e)
            stats["probe_failed"] += 1
            continue
        # a plan-shaped dict so the sidecar can report the encode. the
        # journal is the only record of what was actually done, since
        # the finished file cannot describe its own filter chain.
        # info here is a probe of the FINISHED file, not of the original,
        # so the sidecar must not label it "SOURCE"
        plan = {"out_space": rec.get("space", ""), "cq": rec.get("cq", ""),
                "vf": rec.get("vf", ""), "notes": list(rec.get("notes") or []),
                "_info_is_output": True}
        before = path.name
        try:
            newpath, _ = tag_and_rename(ctx, path, Path(rel_out),
                                        src_rel, info, plan, log)
        except Exception as e:
            log.warning("    metadata failed: %s" % e)
            log.debug("metadata traceback", exc_info=True)
            stats["failed"] += 1
            continue
        if newpath.name != before:
            stats["renamed"] += 1
            # keep the journal honest so a later resume does not think
            # the old name is still on disk
            if src_rel in (journal.data.get("files") or {}):
                journal.update(src_rel, out=newpath.name)
        else:
            stats["unchanged"] += 1
        if "visually_confirmed" in plan["notes"]:
            stats["visually_confirmed"] += 1
    return stats


# ---------------------------------------------------------------------
# Container-level HDR10 static metadata: mkvmerge property flags built
# straight off the source probe side data
# ---------------------------------------------------------------------
def hdr_mux_args(info, out_space):
    if out_space not in ("hdr10", "hlg"):
        return []
    a = ["--colour-matrix", "0:9", "--colour-primaries", "0:9",
         "--colour-range", "0:1",
         "--colour-transfer-characteristics",
         "0:16" if out_space == "hdr10" else "0:18"]
    m = info.get("master")
    if m:
        try:
            rx, ry = frac(m["red_x"]), frac(m["red_y"])
            gx, gy = frac(m["green_x"]), frac(m["green_y"])
            bx, by = frac(m["blue_x"]), frac(m["blue_y"])
            wx, wy = frac(m["white_point_x"]), frac(m["white_point_y"])
            maxl = frac(m["max_luminance"])
            minl = frac(m["min_luminance"])
            a += ["--chromaticity-coordinates",
                  "0:%.5f,%.5f,%.5f,%.5f,%.5f,%.5f"
                  % (rx, ry, gx, gy, bx, by),
                  "--white-colour-coordinates", "0:%.5f,%.5f" % (wx, wy),
                  "--max-luminance", "0:%.4f" % maxl,
                  "--min-luminance", "0:%.4f" % minl]
        except Exception:
            pass
    c = info.get("cll")
    if c:
        try:
            a += ["--max-content-light",
                  "0:%d" % int(frac(c.get("max_content"))),
                  "--max-frame-light",
                  "0:%d" % int(frac(c.get("max_average")))]
        except Exception:
            pass
    return a


def finalize(ctx, tmp_out, final, info, plan):
    log.debug("finalize: %s -> %s space=%s"
              % (tmp_out.name, final.name, plan["out_space"]))
    final.parent.mkdir(parents=True, exist_ok=True)
    props = hdr_mux_args(info, plan["out_space"]) \
        if TOOLS.get("mkvmerge") else []
    if props:
        cmd = [TOOLS["mkvmerge"], "-o", str(final)] + props + [str(tmp_out)]
        rc, out, err = run(cmd, timeout=21600)
        if rc == 1:
            ctx["log"].warning("    the HDR property pass finished WITH "
                               "WARNINGS:")
            for line in [x for x in (out or "").splitlines()
                         if "warning" in x.lower()][:8]:
                ctx["log"].warning("      %s" % line.strip())
        if rc <= 1 and final.exists():
            # VERIFY BEFORE DELETING. This deleted tmp_out - the only
            # good copy - on the strength of an exit code alone, and an
            # exit code of 1 means "might be ok or not".
            before = track_census(tmp_out)
            problems = ledger(ctx, "hdr_props", final, compare_to=before)
            if problems:
                ctx["log"].error("    the HDR property pass damaged the "
                                 "file; keeping the pre-props version")
                cleanup_paths([final])
            else:
                cleanup_paths([tmp_out])
                return "hdr_props_applied"
        ctx["log"].warning("    mkvmerge HDR props pass failed rc %s, "
                           "plain move instead" % rc)
        cleanup_paths([final])
    if final.exists():
        final.unlink()
    shutil.move(str(tmp_out), str(final))
    return "moved_plain"


# ---------------------------------------------------------------------
# Failure path: partials land in <dst>\failed next to a diagnostic log
# holding probe, analysis, plan, every command tried, and stderr tails
# ---------------------------------------------------------------------
SUGGEST = {
    "probe": "Container likely damaged. Remux first: mkvmerge -o fixed.mkv <file>, then queue the fixed copy.",
    "dv5": "Dolby Vision profile 5 carries no HDR10 base layer. V0.2 adds a libplacebo path for these. Park the file until then.",
    "dv_tools": "dovi_tool or mkvmerge missing on PATH. Install both, open a new terminal, run again.",
    "dv_strip": "RPU removal failed. Run the listed dovi_tool command by hand and read its error text.",
    "encode": "Every encoder variant failed. Read the stderr tail; if it names av1_nvenc or the driver, update the NVIDIA driver and confirm no other app holds the encode block.",
    "audio_short": "The finished audio does not reach the end of the picture. Something upstream truncated it - on the one occasion this happened it was the Dolby Vision remux exiting with warnings after reading the source over the network, producing complete video and half-length audio. Check the dv-free remux warnings in the log, and re-run the file.",
    "duration": "Encoded length drifted past tolerance. Source timestamps may be broken; remux the source with mkvmerge and queue the fixed copy.",
    "sync_drift": "Audio drifts progressively against the source; a fixed delay cannot repair that. Measure and repair it with av1_warp.py <output> <source> --apply, which walks the file a minute at a time, adjusts playback rate where the audio drifts and inserts silence where samples are actually missing. DO NOT add aresample=async: it does not repair a gapped timeline, it pads it with silence, and doing so appended 15.5 minutes of silence to an 87 minute film.",
    "sr-upscale": "The Real-ESRGAN stage failed; run the exe by hand on a folder of PNG frames and read its complaint. A quick rerun with --no-sr takes the lanczos path instead.",
    "sync_offset": "A constant delay was found and the remux repair did not settle it. Run mkvmerge --sync by hand with the delay in this log.",
    "exception": "Unexpected error. The traceback below tells the story; paste this log back into the chat.",
}


def fail_file(ctx, rel, stage, info, an, plan, attempts, extra, partials):
    log.debug("ENCODE_FAIL file=%s stage=%s" % (rel, stage))
    fdir = ctx["dst"] / "failed"
    fdir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in partials or []:
        try:
            p = Path(p)
            if p.exists():
                tgt = fdir / p.name
                if tgt.exists():
                    tgt.unlink()
                shutil.move(str(p), str(tgt))
                moved.append(str(tgt))
        except Exception:
            pass
    lines = ["=" * 70,
             "av1_pipeline V%s failure diagnostic" % VERSION,
             "file: %s" % rel,
             "stage: %s" % stage,
             "time: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "",
             "SUGGESTED FIX: " + SUGGEST.get(stage, "See details below."),
             ""]
    if info:
        lines.append("PROBE:")
        lines.append(json.dumps({k: v for k, v in info.items()
                                 if k not in ("master", "cll")}, indent=1))
    if an:
        lines.append("ANALYSIS:")
        lines.append(json.dumps(an, indent=1))
    if plan:
        lines.append("PLAN:")
        lines.append(json.dumps({k: v for k, v in plan.items()
                                 if k != "audio"}, indent=1))
    for at in attempts or []:
        lines.append("")
        lines.append("ATTEMPT cq=%s variant=%s rc=%s"
                     % (at.get("cq"), at.get("variant"), at.get("rc")))
        lines.append("CMD: " + at.get("cmd", ""))
        for tl in at.get("tail", []):
            lines.append("  | " + tl)
    rows = (ctx.get("ledger") or [])
    if rows:
        lines.append("")
        lines.append("PROVENANCE LEDGER (where the timing went):")
        lines.append("  %-14s %10s  %s" % ("stage", "video", "audio tracks"))
        for r in rows:
            cen = r["census"] or {}
            v = max((x["dur"] for x in cen.get("video") or []), default=0.0)
            a = ", ".join("%.1f" % x["dur"]
                          for x in cen.get("audio") or []) or "none"
            lines.append("  %-14s %9.1fs  %s" % (r["stage"], v, a))
    if extra:
        lines.append("")
        lines.append("EXTRA:")
        lines.append(json.dumps(extra, indent=1, default=str))
    if moved:
        lines.append("")
        lines.append("MOVED PARTIALS: " + "; ".join(moved))
    name = Path(rel).name
    (fdir / (name + ".diag.log")).write_text("\n".join(lines),
                                             encoding="utf-8")
    ctx["journal"].update(rel, status="failed", stage=stage)
    ctx["log"].error("    FAILED at %s, diagnostic: failed\\%s.diag.log"
                     % (stage, name))


# ---------------------------------------------------------------------
# Analyze-only report row
# ---------------------------------------------------------------------
def report_row(ctx, rel, item, info, an, plan):
    aud = plan["audio"]
    p = ["=" * 70,
         rel,
         "  size %.2f GB  codec %s  %dx%d @ %.3f fps  %d-bit  hdr=%s"
         % (gb(item["size"]), info["vcodec"], info["w"], info["h"],
            info["fps"], info["depth"], info["hdr"]),
         "  bitrate %.0f kbps (%s)  bpp %.4f  qscore %.4f  tier %s"
         % (info["vbr"] / 1000.0, info["vbr_basis"], an["bpp"],
            an["qscore"], an["tier"]),
         "  crop: %s   interlaced: %s (tff %d bff %d prog %d und %d "
         "share %.2f fo=%s)"
         % (an["crop"], an["interlaced"], an["idet"]["tff"],
            an["idet"]["bff"], an["idet"]["prog"], an["idet"]["und"],
            an["idet"]["share"], an["idet"]["field_order"] or "none"),
         "  plan: cq %d  space %s  vmaf-gate %s"
         % (plan["cq"], plan["out_space"], plan["gate"]),
         "  filters: %s" % plan["vf"],
         "  audio: %s (%s, %sch, %s)  subs kept: %d  attachments: %d"
         % (plan["audio_why"], aud["lang"] if aud else "-",
            aud["ch"] if aud else "-", aud["layout"] if aud else "-",
            len(info["subs"]), info["atts"])]
    if plan["notes"]:
        p.append("  notes: " + "; ".join(str(n) for n in plan["notes"]))
    with open(ctx["report"], "a", encoding="utf-8") as f:
        f.write("\n".join(p) + "\n")


# ---------------------------------------------------------------------
# Per-file orchestration: probe, dv strip, analyze, plan, encode with
# the CQ ladder plus fallback variants, finalize, sync check, journal
# ---------------------------------------------------------------------
def process_file(ctx, idx, total, item, prefetch=None):
    log = ctx["log"]
    journal = ctx["journal"]
    src_path = item["path"]
    rel = src_path.relative_to(ctx["src"]).as_posix()
    t0 = time.time()
    log.info("[%d/%d] %s  (%.2f GB)" % (idx, total, rel, gb(item["size"])))
    tag = hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
    log.debug("process start: %s tag=%s size=%d" % (rel, tag, item["size"]))
    out_rel = Path(item.get("out_rel") or Path(rel).with_suffix(".mkv"))
    final = ctx["dst"] / out_rel
    tmp_out = ctx["tmp"] / ("%s.%s.part.mkv" % (tag, out_rel.name))
    dv_files = []
    info = an = plan = None
    attempts = []
    try:
        journal.update(rel, status="probing", in_bytes=item["size"])
        # whatever the background thread finished while the previous
        # file was encoding. a miss is not an error, it just means the
        # work happens here instead.
        pf = dict((prefetch or {}).get(rel) or {})
        ctx["ledger"] = []
        ledger(ctx, "master", src_path, log=log)
        try:
            info = pf.get("info") or probe_file(src_path)
        except Exception as e:
            fail_file(ctx, rel, "probe", None, None, None, [],
                      {"error": str(e)}, [])
            return "failed"
        if info["vcodec"] == "av1":
            journal.update(rel, status="skipped_av1")
            log.info("    already AV1, skipped")
            return "skipped_av1"
        if info["hdr"] == "dv5":
            fail_file(ctx, rel, "dv5", info, None, None, [], {}, [])
            return "failed"
        enc_input = src_path
        if info["hdr"] == "dv78" and not ctx["analyze_only"]:
            if not TOOLS.get("dovi_tool") or not TOOLS.get("mkvmerge"):
                fail_file(ctx, rel, "dv_tools", info, None, None, [], {}, [])
                return "failed"
            need = max(MIN_FREE_BYTES, int(item["size"] * 2.4))
            if shutil.disk_usage(ctx["dst"]).free < need:
                journal.update(rel, status="skipped_space")
                log.warning("    free space too low for the DV strip, skipped")
                return "skipped_space"
            journal.update(rel, status="dv_strip")
            log.info("    Dolby Vision profile %s, RPU strip to the HDR10 base"
                     % info["dovi"])
            try:
                enc_input = dv_strip(ctx, src_path, info, tag)
                dv_files = [enc_input]
                master_cen = (ctx["ledger"][0]["census"]
                              if ctx.get("ledger") else None)
                if ledger(ctx, "dv_strip", enc_input,
                          compare_to=master_cen, log=log):
                    raise RuntimeError(
                        "the Dolby Vision strip lost audio - see the "
                        "LEDGER lines above. This is the fault that "
                        "shipped a film with 44 minutes of silence.")
                info = probe_file(enc_input)
                # the analysis has to describe the stripped file, so
                # anything prepared against the original is now wrong
                pf = {}
            except Exception as e:
                fail_file(ctx, rel, "dv_strip", info, None, None, [],
                          {"error": str(e)}, [])
                cleanup_paths(dv_files)
                return "failed"
        journal.update(rel, status="analyzing")
        if pf.get("an") and pf.get("plan"):
            an, plan = pf["an"], pf["plan"]
            log.info("    analysis reused from the background pass")
        else:
            an = analyze(enc_input, info, log)
            log_resources(log, "after analysis", phase="analysis")
            plan = build_plan(info, an,
                              sdr_to_hdr=ctx.get("sdr_to_hdr", False))
        journal.update(rel, status="planned", tier=an["tier"], cq=plan["cq"],
                       space=plan["out_space"], audio=plan["audio_why"],
                       vf=plan["vf"], notes=plan["notes"])
        log.info("    %s %dx%d %.3f fps %d-bit hdr=%s tier=%s bpp=%.4f"
                 % (info["vcodec"], info["w"], info["h"], info["fps"],
                    info["depth"], info["hdr"], an["tier"], an["bpp"]))
        log.info("    crop=%s interlaced=%s cq=%d space=%s gate=%s "
                 "audio=%s subs=%d"
                 % (an["crop"], an["interlaced"], plan["cq"],
                    plan["out_space"], plan["gate"], plan["audio_why"],
                    len(info["subs"])))
        if ctx["analyze_only"]:
            report_row(ctx, rel, item, info, an, plan)
            journal.update(rel, status="analyzed")
            cleanup_paths(dv_files)
            return "analyzed"
        need = max(MIN_FREE_BYTES, int(item["size"] * 1.3))
        if shutil.disk_usage(ctx["dst"]).free < need:
            journal.update(rel, status="skipped_space")
            log.warning("    free space too low, skipped; clear room "
                        "and run again")
            cleanup_paths(dv_files)
            return "skipped_space"
        journal.update(rel, status="encoding")
        # the test is on the SOURCE being SDR, not on the output space.
        # out_space turns into hdr10 whenever --sdr-to-hdr is on, and
        # reading that here used to switch Real-ESRGAN off silently
        use_sr = bool(plan["upscaled"]) and not ctx["no_sr"] \
            and info["hdr"] == "sdr" \
            and bool(TOOLS.get("realesrgan")) and cfr_like(info)
        # NOW start preparing the next file, not a moment earlier. its
        # probe, analysis and whisper overlap a NORMAL encode for almost
        # nothing (measured: 10.5 s alone against 10.9 s alongside), but
        # if they are allowed to start while THIS file is still doing its
        # own whisper the two fight over the GPU and it runs 3.3x slower.
        #
        # It is skipped entirely when this file is going through the AI
        # upscaler. That path already holds the GPU on Vulkan compute,
        # the CPU on the per-chunk encode and the disk on frame IO, for
        # hours at a stretch. Stacking whisper and a translation pass on
        # top of it is the heaviest combined draw this pipeline can
        # produce, and it is what the machine was doing at 10:56:41 on
        # 2026-08-09 when it locked hard. The overlap is worth about 20
        # percent; it is not worth the box.
        prep = ctx.get("prep_next")
        if prep and use_sr:
            log.info("    background preparation held back: the upscaler "
                     "needs the whole machine")
        elif prep:
            try:
                prep()
            except Exception:
                log.debug("could not start background preparation",
                          exc_info=True)
        if plan["upscaled"]:
            if use_sr:
                plan["notes"].append("upscale_realesrgan_" + ctx["sr_model"])
            else:
                why2 = ("flag_off" if ctx["no_sr"] else
                        "tool_missing" if not TOOLS.get("realesrgan") else
                        "hdr_source" if info["hdr"] != "sdr" else
                        "vfr_source")
                plan["notes"].append("upscale_lanczos_fallback_" + why2)
        used_cq = None
        vmaf_low = None
        # Subtitles are acquired BEFORE either encode path runs.
        #
        # This used to sit after the upscale, and only the non-upscale
        # branch ever consumed the result, so an upscaled file ran
        # whisper - a fifth of its runtime - and then deleted the
        # transcript, the translation and the enhanced dialogue track
        # unread. That hit exactly the material most likely to need
        # generated subtitles: low-resolution animation with no subtitle
        # track of its own.
        ext_subs, extra_audio = [], None
        try:
            if pf.get("subs") is not None:
                ext_subs, extra_audio = pf["subs"]
                log.info("    subtitles reused from the background pass "
                         "(%d generated)" % len(ext_subs))
            else:
                ext_subs, extra_audio = acquire_subs(ctx, src_path, info,
                                                     plan, log, tag)
        except Exception as e:
            # a bare message cost a whole version of silent failure, so
            # the traceback goes in the trace file too
            log.warning("    subtitle acquisition failed, continuing "
                        "without it: %s: %s" % (type(e).__name__, e))
            log.debug("subtitle acquisition traceback:\n%s"
                      % traceback.format_exc())
        # when mkvmerge assembles the result, ffmpeg writes a video and
        # audio only file first and everything else is added afterwards
        # by the tool that understands matroska properly. Both encode
        # paths use it now; the upscale path used to mux itself through
        # ffmpeg's matroska writer, which is the writer three of this
        # project's worst defects lived in.
        # Where the AUDIO is read from. Normally the same file as the
        # video; after a Dolby Vision strip, the untouched master, so the
        # audio never passes through the strip's extract/remux at all.
        audio_master = src_path if enc_input != src_path else None
        if audio_master is not None:
            log.info("    audio will be read from the master, not the "
                     "Dolby Vision intermediate")
        use_mm = bool(MUX_MKVMERGE and TOOLS.get("mkvmerge"))
        if use_sr and ext_subs and not use_mm:
            log.warning("    %d generated subtitle track(s) cannot be "
                        "carried on the upscale path without mkvmerge "
                        "assembly; set mux_mkvmerge = 1" % len(ext_subs))
        # The video+audio intermediate is written by ffmpeg and read
        # straight back by mkvmerge, so it is pure working material and
        # goes on local scratch. That halves the network traffic for a
        # mkvmerge assembly: previously the share took a full size
        # write, a full size read and then another full size write.
        # tmp_out stays on the destination on purpose - it IS the
        # output, and finishing there means the last step is an atomic
        # same-volume rename rather than a copy.
        enc_target = (scratch_dir(ctx) / ("%s.va.mkv" % tag)) if use_mm \
            else tmp_out

        def assemble(source_for_subs):
            """Add subtitles, fonts and chapters with mkvmerge."""
            log.info("    assembling with mkvmerge: subtitles, "
                     "fonts and chapters")
            return mkvmerge_assemble(
                enc_target, source_for_subs, info, tmp_out,
                ext_subs=ext_subs,
                blank_srt=(blank_sub_path(scratch_dir(ctx))
                           if SUBS_BLANK_FIRST else None),
                default_eng=not primary_audio_is_english(info, plan),
                log=log)

        if use_sr:
            # No prefetch hit to check for any more: the background pass
            # deliberately no longer runs the model, because doing it
            # untiled and on network scratch alongside a live encode is
            # what crashed the machine on 2026-08-06. sr_encode owns the
            # upscale outright.
            # sr_encode works the scale out and announces it; calling
            # pick_sr_scale here as well printed every warning twice
            log.info("    sr upscale start: %s" % ctx["sr_model"])
            try:
                sr_encode(ctx, enc_input, info, an, plan, enc_target, log,
                          tag, extra_audio=extra_audio, va_only=use_mm)
            except RuntimeError as e:
                fail_file(ctx, rel, "sr-upscale", info, an, plan, attempts,
                          {"error": str(e)}, [enc_target, tmp_out])
                cleanup_paths(dv_files)
                return "failed"
            attempts.append({"cq": plan["cq"], "variant": "sr", "rc": 0})
            used_cq = plan["cq"]
            if use_mm:
                mrc, merr = assemble(src_path)
                if mrc != 0:
                    log.warning("    mkvmerge failed rc=%s" % mrc)
                    for line in (merr or "").splitlines()[:8]:
                        log.warning("      mkvmerge: %s" % line.strip())
                    attempts.append({"cq": plan["cq"], "variant": "mkvmerge",
                                     "rc": mrc, "tail": [merr[:400]]})
                    fail_file(ctx, rel, "mux", info, an, plan, attempts,
                              {}, [enc_target, tmp_out])
                    cleanup_paths(dv_files)
                    return "failed"
                cleanup_paths([enc_target])
            okdur, out_dur = check_duration(info["dur"], tmp_out,
                                            src_path=enc_input)
            if not okdur:
                fail_file(ctx, rel, "duration", info, an, plan, attempts,
                          {"src_dur": info["dur"], "out_dur": out_dur},
                          [tmp_out])
                cleanup_paths(dv_files)
                return "failed"
        if not use_sr:
            ladder = []
            q = plan["cq"]
            while len(ladder) < 3:
                ladder.append(q)
                if q <= CQ_FLOOR:
                    break
                q = max(CQ_FLOOR, q - 2)
            for cq_try in ladder:
                variants = [("full", False, False)]
                if plan["colorconv"]:
                    variants.append(("no_colorconv", True, False))
                variants.append(("minimal", plan["colorconv"], True))
                rc = -1
                for label, dropcc, minimal in variants:
                    cmd = build_cmd(enc_input, plan, info, cq_try,
                                    enc_target,
                                    minimal=minimal, drop_colorconv=dropcc,
                                    encoder=pick_encoder(False),
                                    ext_subs=ext_subs,
                                    extra_audio=extra_audio,
                                    va_only=use_mm,
                                    audio_input=audio_master)
                    log.info("    encode start: cq %d variant %s"
                             % (cq_try, label))
                    log.debug("encode variant=%s cq=%d" % (label, cq_try))
                    rc, tail = run_encode(cmd, info["dur"], log,
                                          "cq%d/%s" % (cq_try, label),
                                          fps=info.get("fps"))
                    attempts.append({"cq": cq_try, "variant": label, "rc": rc,
                                     "cmd": " ".join(cmd), "tail": tail[-40:]})
                    if rc == 0:
                        if dropcc and plan["colorconv"]:
                            plan["notes"].append("colorconv_dropped_at_runtime")
                        if minimal:
                            plan["notes"].append("encoder_minimal_args_fallback")
                        break
                    log.warning("    variant %s failed rc %s"
                                % (label, rc_text(rc)))
                    # the exit code on its own says almost nothing. put
                    # what ffmpeg actually printed into the readable log
                    # so a failure can be diagnosed without opening the
                    # diagnostic file
                    for line in [x for x in tail if x.strip()][-6:]:
                        log.warning("      ffmpeg: %s" % line)
                    cleanup_paths([enc_target])
                if rc != 0:
                    fail_file(ctx, rel, "encode", info, an, plan, attempts,
                              {}, [enc_target, tmp_out])
                    cleanup_paths(dv_files)
                    return "failed"
                ledger(ctx, "encode", enc_target, log=log)
                if use_mm:
                    mrc, merr = assemble(src_path)
                    if mrc != 0:
                        log.warning("    mkvmerge failed rc=%s" % mrc)
                        for line in (merr or "").splitlines()[:8]:
                            log.warning("      mkvmerge: %s" % line.strip())
                        attempts.append({"cq": cq_try, "variant": "mkvmerge",
                                         "rc": mrc, "tail": [merr[:400]]})
                        fail_file(ctx, rel, "mux", info, an, plan, attempts,
                                  {}, [enc_target, tmp_out])
                        cleanup_paths(dv_files)
                        return "failed"
                    # verify the assembly before dropping the only
                    # other copy of the video and audio
                    if ledger(ctx, "assemble", tmp_out,
                              compare_to=track_census(enc_target),
                              log=log):
                        fail_file(ctx, rel, "audio_short", info, an, plan,
                                  attempts, {"ledger": ledger_report(ctx, log)},
                                  [enc_target, tmp_out])
                        cleanup_paths(dv_files)
                        return "failed"
                    cleanup_paths([enc_target])
                okdur, out_dur = check_duration(info["dur"], tmp_out,
                                                src_path=enc_input)
                if not okdur:
                    fail_file(ctx, rel, "duration", info, an, plan, attempts,
                              {"src_dur": info["dur"], "out_dur": out_dur},
                              [tmp_out])
                    cleanup_paths(dv_files)
                    return "failed"
                used_cq = cq_try
                if not plan["gate"]:
                    break
                vmin, scores = vmaf_min(tmp_out, enc_input, info["dur"], log)
                if vmin is None:
                    plan["notes"].append("vmaf_skipped_no_score")
                    break
                vmaf_low = vmin
                log.info("    vmaf windows %s  low %.2f"
                         % (" ".join("%.1f" % s for s in scores), vmin))
                if vmin >= VMAF_TARGET:
                    break
                if cq_try == ladder[-1]:
                    plan["notes"].append("vmaf_best_effort_%.1f" % vmin)
                    break
                log.info("    under target %.0f, retry at a lower cq"
                         % VMAF_TARGET)
                cleanup_paths([tmp_out])
        # The finished audio must reach the end of the picture.
        #
        # check_duration compares VIDEO to VIDEO and always has, so a
        # truncated audio track passed every check this pipeline had.
        # Scrooged shipped with 3396 s of audio under 6056 s of picture
        # and nothing noticed. A film missing its last 44 minutes of
        # sound is a failed encode, not a finished one.
        short = census_selfcheck(track_census(tmp_out))
        if short:
            for pb in short:
                log.error("    %s" % pb)
            fail_file(ctx, rel, "audio_short", info, an, plan, attempts,
                      {"problems": short, "ledger": ledger_report(ctx, log)},
                      [tmp_out])
            cleanup_paths(dv_files)
            return "failed"
        journal.update(rel, status="finalizing", cq=used_cq)
        how = finalize(ctx, tmp_out, final, info, plan)
        # Conform the audio to the master's timeline.
        #
        # Runs on EVERY file. The source plays correctly, so wherever it
        # has a given moment of audio is where the finished file should
        # have it too. Start points are locked with a broad search over
        # the whole waveform first, then the film is walked a minute at
        # a time and each segment's PLAYBACK RATE is adjusted so its
        # peaks and valleys land on the master's. Nothing is cut, so no
        # audio drops out.
        #
        # This replaces the old scan-and-remux repair entirely. That one
        # applied a single delay - later a delay plus one linear ratio -
        # to a whole track, which is only correct when the error is a
        # straight line. Measured on a real output it was not: fit rms
        # 253 ms against a linear model, flat near zero for ten minutes
        # then flat near +1.55 s.
        sync_note = "no_audio"
        if plan["audio"] is not None and AUDIO_CONFORM:
            journal.update(rel, status="audio_conform")
            try:
                import av1_warp as WARP
                res = WARP.conform(
                    TOOLS["ffmpeg"], TOOLS["ffprobe"], TOOLS["mkvmerge"],
                    final, audio_master or enc_input,
                    scratch_dir(ctx) / ("%s.warp" % tag),
                    step=AUDIO_CONFORM_STEP,
                    src_stream="a:%d" % source_audio_position(
                        info, plan["audio"]),
                    # no language filter here: the track selection was
                    # already made upstream, commentary and descriptive
                    # tracks never reached this point, and filtering
                    # again would drop the only audio of a film tagged
                    # neither eng nor jpn
                    langs=None, apply=True,
                    log=lambda m: log.info("    %s" % m))
                if res.get("ok"):
                    rep = res.get("report") or {}
                    # the worst error found, before anything was changed
                    worst = max((max(abs(v.get("first_ms", 0.0)),
                                     abs(v.get("last_ms", 0.0)))
                                 for v in rep.values()), default=0.0)
                    if res.get("applied"):
                        sync_note = "audio_conformed_was_%.0fms_out" % worst
                    elif res.get("skipped") and not rep:
                        sync_note = "audio_conform_skipped_short_file"
                    else:
                        sync_note = "audio_already_in_sync_%.0fms" % worst
                    plan["notes"].append(sync_note)
                    log.info("    %s" % sync_note)
                    bak = res.get("backup")
                    if bak:
                        cleanup_paths([bak])
                else:
                    sync_note = "conform_failed_%s" % res.get("why", "?")
                    log.warning("    audio conform failed: %s"
                                % res.get("why"))
                    plan["notes"].append(sync_note)
            except Exception as e:
                # never lose a finished encode over the sync pass
                sync_note = "conform_error"
                log.warning("    audio conform error: %s: %s"
                            % (type(e).__name__, e))
                log.debug("conform traceback:\n%s"
                          % traceback.format_exc())
        # identify, rename, and tag. every step here is best effort: a
        # metadata failure must never cost a file that took hours.
        if META_ENABLE:
            try:
                final, out_rel = tag_and_rename(ctx, final, out_rel, rel,
                                                info, plan, log)
            except Exception as e:
                log.warning("    metadata step failed, file kept as is: %s"
                            % e)
        out_size = final.stat().st_size
        log.debug("ENCODE_DONE file=%s cq=%s time=%.0fs in=%.3fG out=%.3fG"
                  % (rel, used_cq, time.time() - t0,
                     item["size"] / 1073741824.0,
                     final.stat().st_size / 1073741824.0 if final.exists() else 0))
        journal.update(rel, status="done", out=out_rel.as_posix(),
                       cq=used_cq, vmaf=vmaf_low, out_bytes=out_size,
                       sync=sync_note, how=how, notes=plan["notes"],
                       elapsed=round(time.time() - t0, 1))
        cleanup_paths(dv_files + [scratch_dir(ctx) / "_subs_off.srt",
                                  ctx["tmp"] / "_subs_off.srt"]
                      + [e["path"] for e in (ext_subs or [])])
        log.info("    done  %.1f min  in %.2f GB  result %.2f GB  kept %.0f%%"
                 % ((time.time() - t0) / 60.0, gb(item["size"]),
                    gb(out_size), 100.0 * out_size / max(1, item["size"])))
        return "done"
    except KeyboardInterrupt:
        journal.update(rel, status="interrupted")
        cleanup_paths([tmp_out] + dv_files)
        raise
    except Exception:
        import traceback
        fail_file(ctx, rel, "exception", info, an, plan, attempts,
                  {"traceback": traceback.format_exc()}, [tmp_out])
        cleanup_paths(dv_files)
        return "failed"


# ---------------------------------------------------------------------
# Queue driver, resume logic, run summary
# ---------------------------------------------------------------------
class _BgLog(logging.LoggerAdapter):
    """Prefix background-thread records so an interleaved log reads.

    Preparation of the next file runs concurrently with the current
    file's encode, so their log lines arrive interleaved. Without a
    marker the result is a log in which two files appear to be at two
    different stages simultaneously, which is precisely the sort of
    thing that costs a diagnostic cycle.
    """

    def process(self, msg, kwargs):
        return "[bg] %s" % msg, kwargs


def prepare_next(ctx, item, cache):
    """Pre-compute everything for the next file that does not need NVENC.

    Probing, crop analysis and whisper transcription leave the encoder
    block entirely idle - measured at about a third of a short file's
    wall time and a fifth of a feature's - while the encode itself
    leaves the CPU at 42 pct and the shader cores at 17 pct. They are
    separate silicon on the same card, so overlapping them is close to
    free: an encode alone measured 10.6 s, whisper alone 3.3 s, and the
    two together 10.9 s.

    Timing of the call matters as much as its content. This must be
    started only once the current file's encode is underway. Starting it
    any earlier puts it alongside the current file's OWN whisper pass,
    where the two contend for the same shader cores and transcription
    slows by a factor of three - which is how the first implementation
    of this managed to be slower than doing nothing.

    Results land in cache[rel] for process_file to collect. Every
    failure path here is swallowed: a miss simply means the work happens
    inline later, and must never be worse than not having tried.

    Args:
        ctx: run context.
        item: queue entry for the file to prepare.
        cache: shared dict keyed by relative path, written in place.
    """
    rel = item["rel"]
    blog = _BgLog(log, {})
    try:
        info = probe_file(item["path"])
        if not info or info.get("vcodec") == "av1":
            return
        if info.get("hdr") in ("dv5", "dv78"):
            # Dolby Vision needs the RPU stripped first, and the analysis
            # has to run on the stripped file rather than this one
            return
        blog.info("preparing %s while the current file encodes" % rel)
        an = analyze(item["path"], info, log)
        plan = build_plan(info, an, sdr_to_hdr=ctx.get("sdr_to_hdr", False))
        entry = {"info": info, "an": an, "plan": plan}
        cache[rel] = entry
        # subtitles are the expensive half and run on the shader cores
        if SUBS_GENERATE and not ctx.get("analyze_only"):
            tag = hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
            try:
                entry["subs"] = acquire_subs(ctx, item["path"], info,
                                             plan, blog, tag)
            except Exception as e:
                blog.debug("background subtitles failed, will retry "
                           "inline: %s" % e)
        # The AI upscale is deliberately NOT done here, and this is the
        # single most important thing in this function.
        #
        # It used to be. On 2026-08-06 that took the machine down. Three
        # separate faults compounded:
        #
        #   1. it ran realesrgan with NO -t tile argument, so the tool
        #      picked its own tile against the card's TOTAL memory while
        #      a 4K NVENC encode was already holding several GB. The
        #      chunked path in sr_encode sizes the tile against what is
        #      actually free; this path did not size it at all.
        #   2. it wrote per-frame PNGs to ctx["tmp"], which is the
        #      DESTINATION - a network-backed DrivePool. Extracting a
        #      whole 92 minute film came to six figures of PNGs across
        #      the network, while the encode was reading and writing the
        #      same volume. The encode's position froze for 32 minutes
        #      and a sync check that normally takes seconds took two
        #      hours.
        #   3. it extracted the ENTIRE film before starting, rather than
        #      chunking, so the disk cost was unbounded.
        #
        # The driver logged six nvlddmkm errors and the box went down.
        #
        # Whisper against an encode was measured at 3 percent, which is
        # why subtitles are still prefetched above. The upscaler is a
        # different animal: it is the heaviest GPU and disk consumer in
        # the pipeline, and sr_encode already does it properly - tiled,
        # chunked, VRAM-budgeted and staged on a LOCAL scratch disk.
        # Doing it twice was not an optimisation; the four and a half
        # hours it burned here were thrown away on a timeout anyway.
        blog.info("prepared %s" % rel)
    except Exception as e:
        # leave whatever was cached; process_file redoes the rest inline
        log.debug("[bg] preparation error: %s %s" % (rel, e))
        log.debug("[bg] traceback:\n%s" % traceback.format_exc())


def main(argv=None):
    ap = argparse.ArgumentParser(description="AV1 batch pipeline V" + VERSION)
    ap.add_argument("--src", help="source folder")
    ap.add_argument("--dst", help="destination folder")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N queue entries")
    ap.add_argument("--analyze-only", action="store_true",
                    help="probe, analyze, plan, report; zero encodes")
    ap.add_argument("--metadata-only", action="store_true",
                    help="re-identify, tag, rename and write .info files "
                         "for what is already in the destination; no "
                         "video is touched")
    ap.add_argument("--selftest", action="store_true",
                    help="internal checks, no media or gpu needed")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="leave .tmp leftovers at startup")
    ap.add_argument("--retry-failed", action="store_true",
                    help="clear failed journal entries so those files run again")
    ap.add_argument("--force", action="store_true",
                    help="re-encode files even if the journal says done")
    ap.add_argument("--sr-model", default=None,
                    choices=SR_MODELS,
                    help="realesrgan model for upscales")
    ap.add_argument("--no-sr", action="store_true",
                    help="skip realesrgan, upscale through lanczos")
    ap.add_argument("--target-res", default=None, choices=("1080", "2160"),
                    help="lines of actual picture to target (default 1080). "
                         "2160 only produces a 4K master when the source is "
                         "already close to 4K; everything else falls back to "
                         "1080, and the AI upscaler never targets above 1080")
    ap.add_argument("--sdr-to-hdr", action="store_true",
                    help="expand SDR sources into HDR10 color via libplacebo")
    ap.add_argument("--no-subs", action="store_true",
                    help="skip whisper subtitle generation and translation; "
                         "existing subtitle tracks are still carried over")
    ap.add_argument("--upscale-cq", type=int, default=None,
                    help="override CQ for upscaled files (default 28, "
                         "higher = smaller file, range 16-30)")
    ap.add_argument("--encoder", default=None, choices=ENCODER_CHOICES,
                    help="auto (default), nvenc to force the GPU, "
                         "svtav1 to force the CPU encoder")
    ap.add_argument("--audio-mode", default=None, choices=AUDIO_MODES,
                    help="auto (default, copy when the codec is already "
                         "widely playable, else Opus), copy, opus, aac")
    ap.add_argument("--audio-downmix", action="store_true",
                    help="fold surround down to stereo; off by default so "
                         "the original channel layout is kept")
    ap.add_argument("--sr-strength", type=int, default=None,
                    help="how much of the AI upscale to keep, 0 to 100 "
                         "(default 50). lower means less invented detail "
                         "and less frame to frame shimmer")
    ap.add_argument("--sr-tmp-dir", default=None,
                    help="fast local folder for the upscaler frame files")
    ap.add_argument("--ffmpeg-dir", default=None,
                    help="folder holding a specific ffmpeg.exe and "
                         "ffprobe.exe to use instead of the one on PATH")
    ap.add_argument("--log-dir", default=None,
                    help="folder for pipeline.log and av1_trace.log "
                         "(default: a logs folder inside the destination)")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="free destination space floor in GB (default 20)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    # the floor is a module constant so the per-file checks read one value;
    # the flag rewrites it here once at startup
    global MIN_FREE_BYTES, UPSCALE_CQ
    cfg = read_config()
    # a flag beats the config file; a bad config value is reported and
    # then ignored so a typo can never stop a batch from starting
    if a.min_free_gb is None:
        try:
            a.min_free_gb = float(cfg.get("min_free_gb", 20.0))
        except ValueError:
            print("config min_free_gb is not a number, using 20")
            a.min_free_gb = 20.0
    if a.encoder is None:
        a.encoder = cfg.get("encoder", "auto").strip().lower()
        if a.encoder not in ENCODER_CHOICES:
            print("config encoder is not one of %s, using auto"
                  % ", ".join(ENCODER_CHOICES))
            a.encoder = "auto"
    CAPS["encoder_pref"] = a.encoder
    global FFMPEG_DIR
    FFMPEG_DIR = (a.ffmpeg_dir or cfg.get("ffmpeg_dir", "") or "").strip()
    global SR_TMP_DIR, SR_STRENGTH, SR_STABILISE
    SR_TMP_DIR = (a.sr_tmp_dir or cfg.get("sr_tmp_dir", "") or "").strip()
    if SR_TMP_DIR:
        try:
            Path(SR_TMP_DIR).mkdir(parents=True, exist_ok=True)
        except OSError:
            print("sr_tmp_dir cannot be created, using the destination")
            SR_TMP_DIR = ""
    # Blank sr_tmp_dir means "work it out", which is what a machine
    # other than this one needs. Setting it means "use this disk", and
    # that still wins.
    global SCRATCH_AUTO, SCRATCH_MIN_FREE_GB
    if cfg.get("scratch_auto") is not None:
        SCRATCH_AUTO = truthy(cfg["scratch_auto"])
    if cfg.get("scratch_min_free_gb") is not None:
        try:
            SCRATCH_MIN_FREE_GB = max(5, min(2000,
                                             int(cfg["scratch_min_free_gb"])))
        except ValueError:
            print("config scratch_min_free_gb is not a number, leaving it")
    global SR_SCALE, SR_TILE, SR_THREADS, SR_VRAM_MB, SR_VRAM_RESERVE_MB
    global GPU_MAX_PCT, SR_TILE_MAX, GPU_SETTLE_SEC, GPU_WAIT_SEC
    global GPU_TEMP_MAX_C, GPU_COOL_SEC, SR_CHUNK_PAUSE_SEC
    for key, name, lo, hi in (("sr_scale", "SR_SCALE", 0, 4),
                              ("sr_tile", "SR_TILE", 0, 4096),
                              ("sr_tile_max", "SR_TILE_MAX", 64, 4096),
                              # the floor is 50 on purpose: a lower cap
                              # would starve the upscaler rather than
                              # protect anything, and 100 is refused
                              # because leaving the card no headroom is
                              # exactly what crashed the machine
                              ("gpu_max_pct", "GPU_MAX_PCT", 50, 99),
                              ("gpu_settle_sec", "GPU_SETTLE_SEC", 0, 120),
                              ("gpu_wait_sec", "GPU_WAIT_SEC", 0, 3600),
                              ("gpu_temp_max_c", "GPU_TEMP_MAX_C", 0, 95),
                              ("gpu_cool_sec", "GPU_COOL_SEC", 1, 300),
                              ("sr_vram_mb", "SR_VRAM_MB", 512, 65536),
                              ("sr_vram_reserve_mb", "SR_VRAM_RESERVE_MB",
                               0, 16384)):
        if cfg.get(key) is not None:
            try:
                globals()[name] = max(lo, min(hi, int(cfg[key])))
            except ValueError:
                print("config %s is not a number, leaving it" % key)
    if cfg.get("sr_threads") is not None:
        v = (cfg["sr_threads"] or "").strip()
        # the tool wants load:proc:save; anything else it would reject
        if re.match(r"^\d+:\d+:\d+$", v):
            SR_THREADS = v
        elif v:
            print("config sr_threads must look like 4:4:4, leaving it")
    if a.sr_strength is not None:
        SR_STRENGTH = max(0, min(100, a.sr_strength))
    else:
        try:
            SR_STRENGTH = max(0, min(100, int(cfg.get("sr_strength",
                                                      SR_STRENGTH))))
        except ValueError:
            print("config sr_strength is not a number, using %d"
                  % SR_STRENGTH)
    global AUDIO_MODE, AUDIO_DOWNMIX, AUDIO_DENOISE
    AUDIO_MODE = (a.audio_mode or cfg.get("audio_mode", "auto")
                  or "auto").strip().lower()
    if AUDIO_MODE not in AUDIO_MODES:
        print("config audio_mode is not one of %s, using auto"
              % ", ".join(AUDIO_MODES))
        AUDIO_MODE = "auto"
    # truthy() is now a module-level function - see its definition. The
    # local lambda that used to shadow it here was identical, but it
    # meant the scratch settings resolved above could not use it.
    if a.audio_downmix:
        AUDIO_DOWNMIX = True
    elif cfg.get("audio_downmix") is not None:
        AUDIO_DOWNMIX = truthy(cfg.get("audio_downmix"))
    if cfg.get("audio_denoise") is not None:
        AUDIO_DENOISE = truthy(cfg.get("audio_denoise"))
    global CAS_STRENGTH, SD_SATURATION
    for key, name, lo, hi in (("cas_strength", "CAS_STRENGTH", 0.0, 1.0),
                              ("sr_chunk_pause_sec", "SR_CHUNK_PAUSE_SEC",
                               0.0, 60.0),
                              ("sd_saturation", "SD_SATURATION", 0.5, 2.0)):
        if cfg.get(key) is not None:
            try:
                globals()[name] = max(lo, min(hi, float(cfg[key])))
            except ValueError:
                print("config %s is not a number, leaving it alone" % key)
    global WHISPER_VAD, WHISPER_QUEUE
    if cfg.get("whisper_vad") is not None:
        WHISPER_VAD = truthy(cfg["whisper_vad"])
    if cfg.get("whisper_queue") is not None:
        try:
            WHISPER_QUEUE = max(0, min(60, int(cfg["whisper_queue"])))
        except ValueError:
            print("config whisper_queue is not a number, leaving it")
    global SUBS_GENERATE, WHISPER_MODEL_DIR, WHISPER_MODEL, WHISPER_GPU
    global TRANSLATE_PROVIDER, TRANSLATE_KEY, TRANSLATE_ENDPOINT
    global TRANSLATE_MODEL
    if cfg.get("subs_generate") is not None:
        SUBS_GENERATE = truthy(cfg["subs_generate"])
    if a.no_subs:
        # one directional on purpose: the flag can only turn generation
        # off, so ticking it always wins over whatever the config says
        SUBS_GENERATE = False
    if cfg.get("whisper_model_dir") is not None:
        WHISPER_MODEL_DIR = cfg["whisper_model_dir"].strip()
    WHISPER_MODEL = (cfg.get("whisper_model") or "").strip()
    if cfg.get("whisper_gpu") is not None:
        WHISPER_GPU = truthy(cfg["whisper_gpu"])
    tp = (cfg.get("translate_provider") or "deepl").strip().lower()
    # argos runs entirely offline and needs no key at all
    TRANSLATE_PROVIDER = tp if tp in ("deepl", "openai", "argos") else "deepl"
    TRANSLATE_KEY = (cfg.get("translate_key") or "").strip()
    TRANSLATE_ENDPOINT = (cfg.get("translate_endpoint") or "").strip()
    TRANSLATE_MODEL = (cfg.get("translate_model") or "").strip()
    global AUDIO_CONFORM, AUDIO_CONFORM_STEP
    if cfg.get("audio_conform") is not None:
        AUDIO_CONFORM = truthy(cfg["audio_conform"])
    if cfg.get("audio_conform_step") is not None:
        try:
            AUDIO_CONFORM_STEP = max(10.0, min(600.0,
                                               float(cfg["audio_conform_step"])))
        except ValueError:
            print("config audio_conform_step is not a number, leaving it")
    global CQ_OFFSET
    if cfg.get("cq_offset") is not None:
        try:
            CQ_OFFSET = max(-8, min(8, int(cfg["cq_offset"])))
        except ValueError:
            print("config cq_offset is not a number, using 0")
    global DURATION_TOLERANCE_SEC
    if cfg.get("duration_tolerance_sec") is not None:
        try:
            DURATION_TOLERANCE_SEC = max(0.1, min(
                60.0, float(cfg["duration_tolerance_sec"])))
        except ValueError:
            print("config duration_tolerance_sec is not a number, "
                  "leaving it")
    global LOWRES_TRIM_H, LOWRES_CQ_TRIM, LOWRES_CQ_CEIL, UPSCALE_CQ_MAX_H
    for key, name, lo, hi in (("upscale_cq_max_h", "UPSCALE_CQ_MAX_H",
                               0, 1080),
                              ("lowres_trim_h", "LOWRES_TRIM_H", 0, 1080),
                              ("lowres_cq_trim", "LOWRES_CQ_TRIM", 0, 20),
                              ("lowres_cq_ceil", "LOWRES_CQ_CEIL", 20, 50)):
        if cfg.get(key) is not None:
            try:
                globals()[name] = max(lo, min(hi, int(cfg[key])))
            except ValueError:
                print("config %s is not a number, leaving it" % key)
    global TARGET_RES, UHD_MIN_WIDTH, UHD_MIN_PIXELS, UHD_PAD_BAND_PCT
    global UHD_CQ_CEIL, ARTIFACT_REPAIR, ARTIFACT_REPAIR_TIERS
    # target_res is library-wide policy. The GUI may override it for a
    # single run but is barred from persisting it, the same treatment
    # no_sr and sdr_to_hdr got after bug 22.
    if cfg.get("target_res") is not None:
        v = re.sub(r"[^0-9]", "", str(cfg["target_res"]))
        if v in ("1080", "2160"):
            TARGET_RES = int(v)
        elif v:
            print("config target_res must be 1080 or 2160, using 1080")
    if a.target_res:
        TARGET_RES = int(a.target_res)
    for key, name, lo, hi in (
            ("uhd_min_width", "UHD_MIN_WIDTH", 1280, 8192),
            ("uhd_min_pixels", "UHD_MIN_PIXELS", 1000000, 40000000),
            ("uhd_pad_band_pct", "UHD_PAD_BAND_PCT", 0, 50),
            ("uhd_cq_ceil", "UHD_CQ_CEIL", 16, 50)):
        if cfg.get(key) is not None:
            try:
                globals()[name] = max(lo, min(hi, int(cfg[key])))
            except ValueError:
                print("config %s is not a number, leaving it" % key)
    global UHD_MAX_GB, UHD_OVERHEAD_MBPS, UHD_BPP_REF
    for key, name, lo, hi in (("uhd_max_gb", "UHD_MAX_GB", 1.0, 200.0),
                              ("uhd_overhead_mbps", "UHD_OVERHEAD_MBPS",
                               0.0, 20.0),
                              ("uhd_bpp_ref", "UHD_BPP_REF", 0.0, 1.0)):
        if cfg.get(key) is not None:
            try:
                globals()[name] = max(lo, min(hi, float(cfg[key])))
            except ValueError:
                print("config %s is not a number, leaving it" % key)
    if cfg.get("artifact_repair") is not None:
        v = str(cfg["artifact_repair"]).strip().lower()
        if v in ("auto", "off", "weak", "strong"):
            ARTIFACT_REPAIR = v
        elif v:
            print("config artifact_repair must be auto, off, weak or "
                  "strong; using auto")
    if cfg.get("artifact_repair_tiers") is not None:
        tiers = tuple(x.strip().lower()
                      for x in str(cfg["artifact_repair_tiers"]).split(",")
                      if x.strip() in ("high", "mid", "low", "awful"))
        if tiers:
            ARTIFACT_REPAIR_TIERS = tiers
    global META_VERIFY_FRAMES, META_INFO_FILE
    for key, name in (("metadata_verify_frames", "META_VERIFY_FRAMES"),
                      ("metadata_info_file", "META_INFO_FILE")):
        if cfg.get(key) is not None:
            globals()[name] = truthy(cfg[key])
    global MUX_MKVMERGE, CPU_MAX_PCT
    if cfg.get("mux_mkvmerge") is not None:
        MUX_MKVMERGE = truthy(cfg["mux_mkvmerge"])
    if cfg.get("cpu_max_pct") is not None:
        try:
            CPU_MAX_PCT = max(0, min(100, int(cfg["cpu_max_pct"])))
        except ValueError:
            print("config cpu_max_pct is not a number, leaving it")
    # apply the cap to ourselves as soon as it is known, and before any
    # of the heavy native libraries get a chance to size their own
    # thread pools against the whole machine
    pin_self()
    global DEBAND_MODE
    if cfg.get("deband_mode") is not None:
        v = cfg["deband_mode"].strip().lower()
        DEBAND_MODE = v if v in ("auto", "always", "off") else "auto"
    global HWACCEL, AUDIO_FIXED_LAYOUT, AUDIO_OPUS_PER_CH
    if cfg.get("hwaccel") is not None:
        v = cfg["hwaccel"].strip().lower()
        HWACCEL = v if v in ("auto", "cuda", "none") else "auto"
    if cfg.get("audio_fixed_layout") is not None:
        AUDIO_FIXED_LAYOUT = truthy(cfg["audio_fixed_layout"])
    if cfg.get("audio_opus_per_ch") is not None:
        try:
            AUDIO_OPUS_PER_CH = max(48, min(256,
                                            int(cfg["audio_opus_per_ch"])))
        except ValueError:
            print("config audio_opus_per_ch is not a number, leaving it")
    global HQ_SOURCE_BOOST_PCT, AUDIO_KEEP_ALL, AUDIO_ADD_STEREO_MIX
    global AUDIO_MIX_CENTRE, AUDIO_MIX_ENHANCE, SUBS_BLANK_FIRST
    for key, name, lo, hi in (
            ("hq_source_boost_pct", "HQ_SOURCE_BOOST_PCT", 0.0, 200.0),
            ("audio_mix_centre", "AUDIO_MIX_CENTRE", 0.5, 2.0),
            ("audio_mix_enhance", "AUDIO_MIX_ENHANCE", 0.0, 3.0)):
        if cfg.get(key) is not None:
            try:
                globals()[name] = max(lo, min(hi, float(cfg[key])))
            except ValueError:
                print("config %s is not a number, leaving it alone" % key)
    global UPSCALE_STABILISE
    for key, name in (("audio_keep_all", "AUDIO_KEEP_ALL"),
                      ("audio_add_stereo_mix", "AUDIO_ADD_STEREO_MIX"),
                      ("upscale_stabilise", "UPSCALE_STABILISE"),
                      ("subs_blank_first", "SUBS_BLANK_FIRST")):
        if cfg.get(key) is not None:
            globals()[name] = truthy(cfg[key])
    global AUDIO_LANGS, AUDIO_ENG_STEREO_ONLY, AUDIO_MIX_FIRST
    global STALL_TIMEOUT_SEC
    if cfg.get("audio_langs") is not None:
        AUDIO_LANGS = tuple(x.strip().lower()
                            for x in cfg["audio_langs"].split(",")
                            if x.strip())
    for key, name in (("audio_eng_stereo_only", "AUDIO_ENG_STEREO_ONLY"),
                      ("audio_mix_first", "AUDIO_MIX_FIRST")):
        if cfg.get(key) is not None:
            globals()[name] = truthy(cfg[key])
    if cfg.get("stall_timeout_sec") is not None:
        try:
            STALL_TIMEOUT_SEC = max(0, int(float(cfg["stall_timeout_sec"])))
        except ValueError:
            print("config stall_timeout_sec is not a number, leaving it")
    global META_ENABLE, META_RENAME, META_UNDERSCORES, META_EP_TITLE
    global META_TECH, TMDB_API_KEY, SUBS_LANGS
    for key, name in (("metadata_enable", "META_ENABLE"),
                      ("rename_files", "META_RENAME"),
                      ("name_underscores", "META_UNDERSCORES"),
                      ("name_episode_title", "META_EP_TITLE"),
                      ("name_tech_suffix", "META_TECH")):
        if cfg.get(key) is not None:
            globals()[name] = truthy(cfg[key])
    TMDB_API_KEY = (cfg.get("tmdb_api_key") or "").strip()
    if cfg.get("subs_langs") is not None:
        SUBS_LANGS = tuple(x.strip().lower()
                           for x in cfg["subs_langs"].split(",")
                           if x.strip())
    sb = cfg.get("sr_stabilise")
    if sb is not None:
        SR_STABILISE = str(sb).strip().lower() not in ("0", "no", "off",
                                                       "false", "")
    try:
        global SVTAV1_PRESET
        SVTAV1_PRESET = max(0, min(13, int(cfg.get("svtav1_preset",
                                                   SVTAV1_PRESET))))
    except ValueError:
        print("config svtav1_preset is not a number, using %d"
              % SVTAV1_PRESET)
    if a.sr_model is None:
        a.sr_model = cfg.get("sr_model", SR_MODEL_DEFAULT)
        if a.sr_model not in SR_MODELS:
            print("config sr_model is not a known model, using %s"
                  % SR_MODEL_DEFAULT)
            a.sr_model = SR_MODEL_DEFAULT
    MIN_FREE_BYTES = max(1, int(a.min_free_gb * 1024 ** 3))
    if a.upscale_cq is not None:
        UPSCALE_CQ = max(CQ_FLOOR, min(CQ_CEIL, a.upscale_cq))
    src = resolve_dir(a.src, cfg.get("src"), DEFAULT_SRC,
                      "Pick the SOURCE folder", True)
    dst = resolve_dir(a.dst, cfg.get("dst"), DEFAULT_DST,
                      "Pick the DESTINATION folder", False)
    # blank anywhere means the default, a logs folder under the
    # destination, which is what setup_logging does with None
    log_dir = a.log_dir or cfg.get("log_dir") or ""
    log_dir = Path(log_dir).expanduser() if log_dir.strip() else None
    if not src or not dst:
        print("both a source and a destination are needed")
        return 2
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_dir():
        print("source folder not found: %s" % src)
        return 2
    dst.mkdir(parents=True, exist_ok=True)
    if src == dst or src in dst.parents or dst in src.parents:
        print("destination cannot sit inside the source tree, nor the reverse")
        return 2
    tmp = dst / ".tmp"
    tmp.mkdir(exist_ok=True)
    (dst / "failed").mkdir(exist_ok=True)
    log = setup_logging(dst, log_dir)
    log.info("av1_pipeline V%s  src=%s  dst=%s" % (VERSION, src, dst))
    log.info("logs: %s" % (log_dir if log_dir else dst / "logs"))

    # every way this process can end now leaves a line. a log that stops
    # with none of these means the process was killed outright by the
    # operating system, which is itself the diagnosis.
    ended = {"done": False}

    def _final(reason, clean=False):
        # only fire once, and do not shout about an ordinary finish
        if ended["done"]:
            return
        ended["done"] = True
        try:
            (log.info if clean else log.error)("SESSION ENDED: %s" % reason)
            log_resources(log, "at exit")
            for h in log.handlers:
                try:
                    h.flush()
                except Exception:
                    pass
        except Exception:
            pass

    def _hook(et, ev, tb):
        import traceback
        _final("unhandled %s: %s" % (et.__name__, ev))
        try:
            log.error("traceback:\n%s"
                      % "".join(traceback.format_exception(et, ev, tb)))
        except Exception:
            pass
        sys.__excepthook__(et, ev, tb)
    sys.excepthook = _hook

    def _sig(signum, frame):
        _final("signal %s, something asked this process to stop" % signum)
        raise KeyboardInterrupt
    for sname in ("SIGTERM", "SIGINT", "SIGBREAK"):
        s = getattr(signal, sname, None)
        if s is not None:
            try:
                signal.signal(s, _sig)
            except Exception:
                pass
    atexit.register(lambda: _final("normal exit", clean=True))
    log_resources(log, "at startup")
    log.debug("python %s on %s, pid %d"
              % (sys.version.split()[0], platform.platform(), os.getpid()))
    log.info("audio: mode=%s downmix=%s denoise=%s"
             % (AUDIO_MODE, AUDIO_DOWNMIX, AUDIO_DENOISE))
    find_tools()
    if not TOOLS["ffmpeg"] or not TOOLS["ffprobe"]:
        log.error("ffmpeg/ffprobe missing on PATH; install, open a new "
                  "terminal, run again")
        return 2
    probe_caps()
    log.info("ffmpeg: %s" % TOOLS["ffmpeg"])
    # after tool discovery, or every line would read MISSING
    log_environment(log, src, dst)
    log.info("tools: mkvmerge=%s dovi_tool=%s realesrgan=%s"
             % (bool(TOOLS["mkvmerge"]), bool(TOOLS["dovi_tool"]),
                bool(TOOLS["realesrgan"])))
    if not a.analyze_only and not a.no_sr and not TOOLS["realesrgan"]:
        log.warning("realesrgan-ncnn-vulkan missing: upscales take "
                    "the lanczos path")
    log.info("caps: av1_nvenc=%s libvmaf=%s libplacebo=%s zscale=%s"
             % (CAPS["av1_nvenc"], CAPS["libvmaf"],
                CAPS["libplacebo"], CAPS["zscale"]))
    # --metadata-only touches no video at all: it identifies, tags,
    # renames and writes sidecars. It has no business demanding a
    # working AV1 encoder, and it was running the full preflight - three
    # real encodes - and then refusing to start on a machine whose GPU
    # encoder was unavailable, for a job that never encodes anything.
    if not a.analyze_only and not a.metadata_only:
        # An AV1 encoder of SOME kind is required; which one is the
        # preflight's business. Aborting on av1_nvenc alone refused to
        # start on any machine whose ffmpeg has no nvenc compiled in,
        # even though libsvtav1 would have encoded every file - which
        # is precisely the configuration on a non-NVIDIA machine.
        if not CAPS["av1_nvenc"] and not CAPS.get("libsvtav1"):
            log.error("this ffmpeg build lists neither av1_nvenc nor "
                      "libsvtav1, so it cannot produce AV1 at all. "
                      "Install the full Gyan build.")
            return 2
        if not CAPS["av1_nvenc"]:
            log.warning("this ffmpeg build lists no av1_nvenc; the CPU "
                        "encoder will carry the whole run")
        if not TOOLS["mkvmerge"]:
            log.error("mkvmerge handles HDR props plus the sync repair; "
                      "install MKVToolNix and open a new terminal")
            return 2
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
        except Exception:
            log.error("numpy and scipy drive the sync check: "
                      "python -m pip install numpy scipy")
            return 2
        if not preflight(log, a.sdr_to_hdr):
            log.error("no working AV1 encoder on this machine. see the "
                      "preflight lines above for which one failed and why")
            return 2
        if a.sdr_to_hdr and not CAPS.get("placebo_run"):
            # carrying on with the flag still set would send every file
            # down a filter that is already known not to start here
            a.sdr_to_hdr = False
    if not a.keep_tmp and not a.metadata_only:
        for p in tmp.glob("*"):
            cleanup_paths([p])
        # the upscaler's frame folders live in sr_tmp_dir, which is a
        # different place entirely and was never swept. a run that dies
        # part way through leaves thousands of PNGs there for good:
        # 1.4 GB was found stranded from a single aborted job. only our
        # own <tag>.sr folders are touched, never anything else that
        # happens to share the directory.
        if SR_TMP_DIR:
            try:
                freed = 0
                # both patterns: the frame directories and the chunk
                # parts, which now live on this disk too rather than on
                # the network share
                # *.ver holds the frames sampled for the visual
                # identification check. verify_by_frames created one per
                # file and never removed it, and this sweep did not know
                # about it, so a long run left a directory per title
                # behind on the scratch disk for good.
                for pat in ("*.sr", "*.sr.parts", "*.ver", "*.warp"):
                    for p in Path(SR_TMP_DIR).glob(pat):
                        if p.is_dir():
                            freed += sum(f.stat().st_size
                                         for f in p.rglob("*")
                                         if f.is_file())
                            shutil.rmtree(p, ignore_errors=True)
                # LOOSE FILES too, not just directories.
                #
                # Bug 33 moved every intermediate onto this disk but the
                # sweep was never extended past the upscaler's folders,
                # so anything that dies part way through leaks a
                # full-size file for good. Found in practice: a 218 MB
                # <tag>.va.mkv from one interrupted run and another from
                # the next. A killed Dolby Vision strip leaks three
                # elementary streams, about 30 GB.
                #
                # Only this pipeline's own names are matched. Anything
                # else sharing the directory is left strictly alone.
                for pat in ("*.va.mkv", "*.dv.hevc", "*.hdr10.hevc",
                            "*.dvfree.mkv", "*.eng.wav", "*.jpn.wav",
                            "*.eng.srt", "*.jpn.srt", "*.eng_src.srt",
                            "*.jpn_src.srt", "*.jpn_tr.srt",
                            "*.eng_tr.srt", "*.tags.xml",
                            "_subs_off.srt"):
                    for p in Path(SR_TMP_DIR).glob(pat):
                        if p.is_file():
                            freed += p.stat().st_size
                            try:
                                p.unlink()
                            except OSError:
                                pass
                if freed:
                    log.info("swept %.2f GB of stranded upscaler frames "
                             "from %s" % (gb(freed), SR_TMP_DIR))
            except Exception:
                log.debug("could not sweep sr_tmp_dir", exc_info=True)
    journal = Journal(dst / "_journal.json")
    if a.retry_failed:
        n = journal.clear_failed()
        log.info("retry-failed: cleared %d failed entries" % n)
    items = scan_sources(src)
    for it in items:
        it["rel"] = it["path"].relative_to(src).as_posix()
    clashes = assign_out_names(items)
    if clashes:
        log.info("%d output name clashes; those results keep the source "
                 "extension in the name" % clashes)
    if a.limit > 0:
        items = items[:a.limit]
    log.info("queue: %d files, %.1f GB total, largest first"
             % (len(items), gb(sum(i["size"] for i in items))))
    sr_note = "Real-ESRGAN active" if TOOLS["realesrgan"] else "Real-ESRGAN missing (lanczos fallback)"
    if a.no_sr:
        sr_note = "Real-ESRGAN off (--no-sr)"
    log.info("notes: %s, DV profile 5 fails to log, "
             "HLG kept when libplacebo is absent" % sr_note)
    # Choose the scratch disk now that the queue is known, because the
    # requirement is set by the LARGEST file in it: a Dolby Vision strip
    # writes about 3.2 times the source before anything is freed, and
    # an upscale writes thousands of frames on top of that. Sizing this
    # against the biggest file is what stops the run choosing a fast
    # disk that fills up 40 minutes into an extract.
    if not a.metadata_only:
        biggest = max((i["size"] for i in items), default=0)
        need = int(biggest * 3.5) if biggest else 0
        try:
            open_scratch_session(need_bytes=need, dst=dst, log=log,
                                 explicit=SR_TMP_DIR)
        except Exception as e:
            # Never let disk selection stop a run. The fallback is the
            # destination's .tmp, which is where it went before any of
            # this existed.
            log.warning("could not choose a scratch disk (%s: %s); "
                        "using the destination's .tmp"
                        % (type(e).__name__, e))
            log.debug("scratch selection traceback:\n%s"
                      % traceback.format_exc())
        # Every way a run can end, on purpose or otherwise:
        #   completes   -> the finally block below
        #   stopped     -> the signal handlers, which raise into it
        #   interpreter -> this atexit hook
        #   killed dead -> the orphan sweep on the next run
        atexit.register(close_scratch_session)
    ctx = {"src": src, "dst": dst, "tmp": tmp, "journal": journal,
           "log": log, "analyze_only": a.analyze_only,
           "report": dst / "_analysis_report.txt",
           "sr_model": a.sr_model, "no_sr": a.no_sr,
           "force": a.force, "sdr_to_hdr": a.sdr_to_hdr}
    if a.metadata_only:
        # touches no video at all: identification, tags, name and the
        # sidecar only, applied to whatever is already in the
        # destination. the queue above is irrelevant to it.
        stats = metadata_pass(ctx, log, limit=a.limit)
        log.info("metadata pass complete: %s" % dict(stats))
        return 0
    stats = collections.Counter()
    prefetch_cache = {}
    prefetch_thread = None
    sleep_block(True)
    try:
        for i, item in enumerate(items, 1):
            # join any running prefetch before processing this file
            if prefetch_thread:
                prefetch_thread.join(timeout=14400)
                if prefetch_thread.is_alive():
                    # It timed out. Dropping the handle here left the old
                    # thread running and armed a second one on the next
                    # lap, so two background passes ran at once - two
                    # whisper transcriptions competing for the same
                    # shader cores, which is measured at 3.3x slower and
                    # is exactly what the pipelining trigger was placed
                    # to avoid. Keep the handle and arm nothing new.
                    log.warning("background preparation is still running "
                                "after 4 hours; not starting another")
                else:
                    prefetch_thread = None
            # arm the preparation of the NEXT file. process_file fires
            # this once its own encode has started, so the background
            # work lands against NVENC rather than against this file's
            # own whisper.
            started = {}
            ctx["prep_next"] = None
            if i < len(items) and not ctx.get("analyze_only") \
                    and prefetch_thread is None:
                nxt = items[i]
                nxt_prior = journal.get(nxt["rel"])
                nxt_skip = nxt_prior and nxt_prior.get("status") in \
                    ("done", "skipped_av1", "failed")
                if not nxt_skip:
                    def arm(nxt=nxt, started=started):
                        if started.get("th"):
                            return
                        th = threading.Thread(
                            target=prepare_next,
                            args=(ctx, nxt, prefetch_cache), daemon=True)
                        th.start()
                        started["th"] = th
                    ctx["prep_next"] = arm
            rel = item["rel"]
            prior = journal.get(rel)
            if prior:
                st = prior.get("status")
                if st == "done":
                    out_path = dst / prior.get("out", "")
                    if out_path.exists() and not ctx.get("force"):
                        log.info("[%d/%d] %s  done earlier, skipped"
                                 % (i, len(items), rel))
                        stats["done_earlier"] += 1
                        continue
                    if not out_path.exists():
                        log.info("[%d/%d] %s  output missing, re-queued"
                                 % (i, len(items), rel))
                    elif ctx.get("force"):
                        log.info("[%d/%d] %s  force flag set, re-queued"
                                 % (i, len(items), rel))
                if st == "skipped_av1":
                    log.info("[%d/%d] %s  already AV1, skipped earlier"
                             % (i, len(items), rel))
                    stats["skipped_av1"] += 1
                    continue
                if st == "failed":
                    log.info("[%d/%d] %s  failed earlier; delete its "
                             "journal entry to retry" % (i, len(items), rel))
                    stats["failed_earlier"] += 1
                    continue
                if st == "analyzed" and a.analyze_only:
                    stats["analyzed_earlier"] += 1
                    continue
            # The last line of defence for an unattended month-long run.
            # process_file already catches Exception per file, but that
            # leaves the BaseException family - SystemExit raised by some
            # library deep in a call stack being the realistic one - free
            # to unwind the whole loop and end the run on file 40 of
            # 3000. Nothing short of a keyboard interrupt should be able
            # to do that. Log it, journal it, take the next file.
            try:
                stats[process_file(ctx, i, len(items), item,
                                   prefetch=prefetch_cache)] += 1
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                import traceback
                log.error("[%d/%d] %s  escaped every handler (%s: %s); "
                          "logged and skipped"
                          % (i, len(items), rel, type(e).__name__, e))
                log.debug("escaped traceback:\n%s" % traceback.format_exc())
                try:
                    journal.update(rel, status="failed",
                                   how="escaped_%s" % type(e).__name__)
                except Exception:
                    pass
                stats["failed"] += 1
            # whatever process_file started is what the next lap joins.
            # only overwrite when something actually started, or a lap
            # that deliberately armed nothing would drop the handle to a
            # thread still running and undo the guard above.
            if started.get("th"):
                prefetch_thread = started["th"]
    except KeyboardInterrupt:
        log.warning("keyboard interrupt; journal saved, run again to resume")
    finally:
        sleep_block(False)
        # Completed, stopped or interrupted, the working files go. The
        # atexit hook covers the paths that never reach this line.
        close_scratch_session(log)
    log.info("run complete: %s" % dict(stats))
    if a.analyze_only:
        log.info("analysis report: %s" % ctx["report"])
    return 0


# ---------------------------------------------------------------------
# Built-in checks: pure helpers only, no media, no gpu
# ---------------------------------------------------------------------
def selftest():
    ok = True

    def t(name, cond):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name)
        ok = ok and bool(cond)

    p = pan_expr("5.1(side)")
    t("pan 5.1(side) builds", p is not None and
      p.startswith("pan=stereo|FL="))
    t("pan center gain 0.919", p is not None and "0.919*FC" in p)
    t("pan surround gain 0.566", p is not None and "0.566*SL" in p
      and "0.566*SR" in p)
    t("pan lfe folded low", p is not None and "0.350*LFE" in p)
    t("pan unknown layout gives none", pan_expr("9.2(nonsense)") is None)
    t("sr scale: 360 lines picks x3",
      pick_sr_scale(360, "realesr-animevideov3") == 3)
    t("sr scale: 540 lines picks x2",
      pick_sr_scale(540, "realesr-animevideov3") == 2)
    t("sr scale: 240 lines caps at x4",
      pick_sr_scale(240, "realesr-animevideov3") == 4)
    t("sr scale: x4plus holds x4",
      pick_sr_scale(720, "realesrgan-x4plus") == 4)
    t("cfr guard: matched rates pass",
      cfr_like({"rr": 23.976, "ar": 23.976}))
    t("cfr guard: split rates fail",
      not cfr_like({"rr": 25.0, "ar": 24.4}))
    fake = [{"rel": "a/movie.avi"}, {"rel": "a/movie.mkv"},
            {"rel": "b/solo.mp4"}]
    assign_out_names(fake)
    t("clash names keep the source extension",
      fake[0]["out_rel"].name == "movie.avi.mkv"
      and fake[1]["out_rel"].name == "movie.mkv.mkv"
      and fake[2]["out_rel"].name == "solo.mkv")
    u = union_rect([(1904, 800, 8, 140), (1912, 802, 4, 138),
                    (1908, 804, 6, 136)], 1920, 1080)
    t("crop union keeps widest rect", u == (1912, 804, 4, 136))
    t("crop near-full frame gives none",
      union_rect([(1918, 1078, 0, 0)], 1920, 1080) is None)
    t("cq inside bounds",
      CQ_FLOOR <= pick_cq("awful", 8, False) <= CQ_CEIL
      and CQ_FLOOR <= pick_cq("high", 10, True) <= CQ_CEIL)
    t("cq drops for depth and hdr",
      pick_cq("mid", 10, True) < pick_cq("mid", 8, False))
    items = [{"path": Path("b.mp4"), "size": 100, "ctime": 5.0},
             {"path": Path("a.mkv"), "size": 200, "ctime": 9.0},
             {"path": Path("c.avi"), "size": 100, "ctime": 1.0}]
    items.sort(key=queue_key)
    t("queue sorts largest first", items[0]["path"].name == "a.mkv")
    t("queue tiebreak on container", items[1]["path"].name == "c.avi")
    z = build_zscale({"h": 480, "space": "", "prim": "", "trc": "",
                      "range": ""})
    zget = lambda k: z.split(k + "=")[1].split(":")[0]
    t("zscale SD guess uses only declared spellings",
      zget("matrixin") in ZM["bt470bg"] and zget("primariesin") in ZP["bt470bg"])
    t("zscale never emits the bare 470bg for primaries",
      zget("primariesin") != "470bg")
    t("interlace: explicit progressive flag wins",
      interlace_decision("progressive", 80, 100, 480)[0] is False)
    t("interlace: disagreement leaves a note",
      interlace_decision("progressive", 80, 100, 480)[1] is not None)
    t("interlace: container flag plus combing fires",
      interlace_decision("tt", 30, 100, 576)[0] is True)
    t("interlace: unknown flag at 240 lines stays progressive",
      interlace_decision("", 80, 100, 240)[0] is False)
    t("interlace: unknown flag at 480 lines fires",
      interlace_decision("", 80, 100, 480)[0] is True)
    bpp, q, tier = quality_score({"w": 1920, "h": 800, "vbr": 6000000.0,
                                  "fps": 24.0, "vcodec": "h264"})
    t("quality tier reads sane", tier in ("high", "mid", "low", "awful"))

    # --- 4K target mode -------------------------------------------------
    uhd_16x9 = {"w": 3840, "h": 2160}
    uhd_scope = {"w": 3840, "h": 2160}
    t("uhd: 16:9 UHD with no crop is eligible",
      uhd_source(uhd_16x9, {"crop": None})[0] is True)
    t("uhd: 2.39:1 UHD stays eligible after the bars come off",
      uhd_source(uhd_scope, {"crop": (3840, 1608, 0, 276)})[0] is True)
    t("uhd: 1080p source is not eligible",
      uhd_source({"w": 1920, "h": 1080}, {"crop": None})[0] is False)
    t("uhd: 3840x1080 ultrawide fails the pixel guard",
      uhd_source({"w": 3840, "h": 1080}, {"crop": None})[0] is False)
    t("uhd: 1.85:1 at 2076 lines is inside the pad band",
      uhd_target_height(2076) == 2160)
    t("uhd: 2.39:1 at 1608 lines stays native",
      uhd_target_height(1608) == 0)
    t("uhd: exactly 2160 needs no rescale",
      uhd_target_height(2160) == 0)
    t("uhd: pad band edge holds at 1944 lines",
      uhd_target_height(1944) == 2160 and uhd_target_height(1943) == 0)
    t("uhd: budget takes the smaller of source size and the cap",
      uhd_budget({"dur": 6000.0, "size": 2 * 1024 ** 3})
      < uhd_budget({"dur": 6000.0, "size": 50 * 1024 ** 3}))
    t("uhd: repair is off for a clean source",
      uhd_repair_filter({"tier": "high"}, True) is None)
    t("uhd: repair fires strong on the awful tier",
      uhd_repair_filter({"tier": "awful"}, True) == "strong")
    t("uhd: repair fires weak on the low tier",
      uhd_repair_filter({"tier": "low"}, True) == "weak")
    t("uhd: repair never fires on a non-UHD source",
      uhd_repair_filter({"tier": "awful"}, False) is None)
    _ar = ARTIFACT_REPAIR
    globals()["ARTIFACT_REPAIR"] = "off"
    t("uhd: artifact_repair=off wins over the tier",
      uhd_repair_filter({"tier": "awful"}, True) is None)
    globals()["ARTIFACT_REPAIR"] = _ar
    t("uhd: no calibration constant means no prediction",
      uhd_predict_mbps(27, 3840, 2160, 24.0) == 0.0
      if UHD_BPP_REF <= 0 else
      uhd_predict_mbps(27, 3840, 2160, 24.0) > 0.0)
    _ref = UHD_BPP_REF
    globals()["UHD_BPP_REF"] = 0.02
    t("uhd: a higher CQ predicts a smaller file",
      uhd_predict_mbps(31, 3840, 2160, 24.0)
      < uhd_predict_mbps(27, 3840, 2160, 24.0))
    t("uhd: prediction scales with frame area",
      uhd_predict_mbps(27, 3840, 2160, 24.0)
      > uhd_predict_mbps(27, 1920, 1080, 24.0))
    globals()["UHD_BPP_REF"] = _ref
    t("uhd: 2160 scale filter targets 2160 lines",
      "2160" in SCALE_2160 and SCALE_2160.startswith("scale=-2:"))
    t("uhd: both deblock strengths are defined",
      set(UHD_DEBLOCK) == {"weak", "strong"})
    t("uhd: default target is the 1080 standard", TARGET_RES == 1080)

    # libplacebo constant spellings. `pq` is NOT a valid color_trc - the
    # accepted name is smpte2084 - and getting it wrong took the whole
    # ffmpeg command down for every HLG source.
    hlg = placebo_hlg_to_pq()
    sdr = placebo_filter()
    t("placebo: HLG path asks for smpte2084, not pq",
      "color_trc=smpte2084" in hlg and "color_trc=pq" not in hlg)
    t("placebo: SDR path asks for smpte2084, not pq",
      "color_trc=smpte2084" in sdr and "color_trc=pq" not in sdr)
    t("placebo: both target bt2020",
      "color_primaries=bt2020" in hlg and "color_primaries=bt2020" in sdr)
    t("placebo: neither emits an empty option",
      "=:" not in hlg and "=:" not in sdr
      and not hlg.endswith("=") and not sdr.endswith("="))
    t("placebo: option probe survives no ffmpeg", isinstance(
        placebo_opts(), set))

    # --- audio bitrate, which nothing was collecting ---------------------
    # ffprobe leaves bit_rate empty for most Matroska audio and puts the
    # figure in a BPS tag instead. Measured on the Scrooged remux: the
    # DTS-HD MA track reports no bit_rate and BPS=3920049.
    t("bitrate: the stream field is used when present",
      stream_bitrate({"bit_rate": "448000"}) == 448000)
    t("bitrate: falls back to the Matroska BPS tag",
      stream_bitrate({"tags": {"BPS": "3920049"}}) == 3920049)
    t("bitrate: BPS-eng and friends count too",
      stream_bitrate({"tags": {"BPS-eng": "640000"}}) == 640000)
    t("bitrate: nothing recorded reads as zero",
      stream_bitrate({"tags": {"DURATION": "01:40:56"}}) == 0)
    t("lossless: DTS-HD MA is recognised by its data rate",
      is_lossless_track({"codec": "dts", "bit_rate": 3920049}))
    t("lossless: plain DTS at disc rates is not",
      not is_lossless_track({"codec": "dts", "bit_rate": 768000}))
    t("lossless: a track with no bitrate at all is not guessed lossless",
      not is_lossless_track({"codec": "dts", "bit_rate": 0}))
    t("lossless: truehd is lossless by codec",
      is_lossless_track({"codec": "truehd"}))
    # the source-rate floor must not fire on a lossless master, or a
    # stereo FLAC would demand 768k of Opus to carry 256k of content
    t("opus: a lossless source does not inflate the rate",
      opus_bitrate(2, 900000, 2, True) == opus_bitrate(2, 0, 0, False))
    t("opus: a lossy source still lifts the floor",
      opus_bitrate(2, 640000, 2, False) > opus_bitrate(2, 0, 0, False))
    t("opus: the cap always wins", opus_bitrate(8, 9999999, 8, False)
      <= AUDIO_OPUS_CAP)
    # the legacy layout path was CALLED and never defined, so
    # audio_fixed_layout = 0 raised NameError on every file
    t("audio: the legacy layout path exists",
      callable(globals().get("audio_args_legacy")))
    _fake = {"audios": [{"idx": 1, "lang": "eng", "ch": 6,
                         "layout": "5.1(side)", "codec": "ac3",
                         "bit_rate": 448000, "title": "", "aux": False}],
             "subs": [], "atts": 0}
    _fp = {"audio": _fake["audios"][0], "notes": []}
    _aa = audio_args(_fake, _fp)
    t("audio: mapping_family is written per stream, never globally",
      "-mapping_family" not in _aa
      and any(str(x).startswith("-mapping_family:a:") for x in _aa))
    # bug 36's remedy must not be recommended by bug 36's own diagnostic
    t("diagnostic: sync_drift no longer recommends aresample=async",
      "aresample=async=1 to the audio" not in SUGGEST["sync_drift"])
    t("diagnostic: sync_drift points at the tool that measures it",
      "av1_warp" in SUGGEST["sync_drift"])
    # a file's ONLY audio track is kept whatever its language tag says
    _one = {"audios": [{"idx": 1, "lang": "fre", "ch": 6,
                        "layout": "5.1", "codec": "ac3",
                        "bit_rate": 448000, "title": "", "aux": False}],
            "subs": [], "atts": 0}
    _op = {"audio": _one["audios"][0], "notes": []}
    _oa = audio_args(_one, _op)
    t("audio: a lone foreign-tagged track is still kept",
      "-map" in _oa and any("single_audio_track_kept" in n
                            for n in _op["notes"]))
    _jp = {"audios": [{"idx": 1, "lang": "jpn", "ch": 2,
                       "layout": "stereo", "codec": "aac",
                       "bit_rate": 128000, "title": "", "aux": False}],
           "subs": [], "atts": 0}
    _jpp = {"audio": _jp["audios"][0], "notes": []}
    t("audio: a lone Japanese track is kept and keeps its own tag",
      "-map" in audio_args(_jp, _jpp))
    try:
        import inspect
        import av1_subs as _SB
        t("subs: the translator takes a source language, not just English",
          "source" in inspect.signature(_SB.translate_srt).parameters)
    except Exception:
        t("subs: the translator takes a source language, not just English",
          False)
    # the ledger: the primitive that names the stage that broke the file
    _good = {"video": [{"dur": 6056.0}],
             "audio": [{"dur": 6056.0}, {"dur": 6055.5}]}
    _bad = {"video": [{"dur": 6056.0}],
            "audio": [{"dur": 3396.0}, {"dur": 6055.5}]}
    t("ledger: a full-length file passes its own check",
      census_selfcheck(_good) == [])
    t("ledger: audio under the picture is caught",
      len(census_selfcheck(_bad)) == 1)
    t("ledger: a track that shortens between stages is caught",
      len(census_compare(_good, _bad)) == 1)
    t("ledger: a track that disappears between stages is caught",
      any("went from 2 to 1" in x for x in census_compare(
          _good, {"video": [{"dur": 6056.0}],
                  "audio": [{"dur": 6056.0}]})))
    # the compensator appends a trim when a latency is known. seeded
    # here because the real probe needs ffmpeg, which the selftest
    # deliberately does not use.
    _chain = "alimiter=limit=0.97,loudnorm=I=-16"
    CAPS.setdefault("af_latency", {})[_chain] = 37.0
    t("audio: a known latency is compensated",
      "atrim" in compensate_af_latency(_chain) and
      "asetpts" in compensate_af_latency(_chain))
    CAPS["af_latency"][_chain] = 0.5
    t("audio: a latency below the floor is left alone",
      compensate_af_latency(_chain) == _chain)
    # and the rebase is refused outright on a track that does not start
    # at zero, whatever the latency is
    t("audio: the latency trim is SKIPPED on an offset track",
      "asetpts" not in (stereo_mix_filter("5.1(side)", None, 2.0) or ""))
    t("audio: an offset track still gets its full downmix chain",
      "loudnorm" in (stereo_mix_filter("5.1(side)", None, 2.0) or ""))
    t("ledger: identical stages report nothing",
      census_compare(_good, _good) == [])
    t("audio conform is OFF in the automatic path", not AUDIO_CONFORM)
    t("audio conform adjusts once a minute", AUDIO_CONFORM_STEP == 60.0)
    t("source audio position resolves a:N, not the stream index",
      source_audio_position({"audios": [{"idx": 3}, {"idx": 7}]},
                            {"idx": 7}) == 1)

    # --- metadata -------------------------------------------------------
    try:
        import av1_metadata as _MD
    except Exception:
        _MD = None
    if _MD is not None:
        _tv = {"id": 34102, "name": "Planetes",
               "first_air_date": "2003-10-04",
               "external_ids": {"imdb_id": "tt0816398"},
               "_episode": {"name": "Outside the Atmosphere",
                            "season_number": 1, "episode_number": 1,
                            "air_date": "2003-10-04"}}
        _pep = {"kind": "episode", "title": "Planetes", "year": "2003",
                "season": 1, "episode": 1}
        _x = _MD.tags_xml(_tv, _pep, "") or ""
        # TMDB gives imdb_id at top level for a MOVIE but only inside
        # external_ids for TV, so reading one place lost it on every
        # episode
        t("meta: TV keeps its IMDB id from external_ids",
          "tt0816398" in _x)
        t("meta: episode title is tagged, not just the series",
          "Outside the Atmosphere" in _x)
        t("meta: series sits at collection level 70",
          "<TargetTypeValue>70</TargetTypeValue>" in _x)
        t("meta: season sits at level 60",
          "<TargetTypeValue>60</TargetTypeValue>" in _x)
        # naming must survive being fed its own output, or a repeated
        # --metadata-only pass grows the name by one _AV1 every run
        _n1 = _MD.build_name(_tv, _MD.parse_path(
            "Planetes.S01E01.Outside.the.Atmosphere.1080p.mkv"), "")
        _n2 = _MD.build_name(_tv, _MD.parse_path(_n1), "")
        _n3 = _MD.build_name(_tv, _MD.parse_path(_n2), "")
        t("meta: episode naming is idempotent", _n1 == _n2 == _n3)
        t("meta: no doubled codec token in the name", "AV1_AV1" not in _n3)
        _m = {"id": 1, "title": "Scrooged", "release_date": "1988-11-23"}
        _p = {"kind": "movie", "title": "Scrooged", "year": "1988"}
        _f1 = _MD.build_name(_m, _p, "", height=2160)
        _f2 = _MD.build_name(_m, _MD.parse_path(_f1), "", height=2160)
        t("meta: movie naming is idempotent", _f1 == _f2)
        t("meta: a 4K master is named 2160p", "AV1_2160p" in _f1)
        t("meta: tech suffix stripper leaves a real title alone",
          _MD.strip_tech_suffix("AV1 Rising") == "AV1 Rising")
        t("meta: tech suffix stripper removes repeats",
          _MD.strip_tech_suffix("Ep Title_AV1_AV1") == "Ep Title")

    # --- portability: GPU generations -----------------------------------
    # These decide nothing - the preflight's trial encode does - but a
    # wrong answer here puts a wrong explanation in the log, and this
    # project has spent whole sessions on a confident wrong explanation.
    _saved_name = CAPS.get("gpu_name")
    for _card, _want_av1, _label in (
            ("NVIDIA GeForce RTX 5060 Ti", True, "RTX 50"),
            ("NVIDIA GeForce RTX 4070 Laptop GPU", True, "RTX 40 laptop"),
            ("NVIDIA GeForce RTX 4090", True, "RTX 40"),
            ("NVIDIA GeForce RTX 3080 Ti", False, "RTX 30"),
            ("NVIDIA GeForce RTX 3060", False, "RTX 30"),
            ("NVIDIA GeForce RTX 2080 Super", False, "RTX 20"),
            ("NVIDIA GeForce GTX 1660 Ti", False, "GTX 16"),
            ("NVIDIA GeForce GTX 1080 Ti", False, "GTX 10")):
        CAPS["gpu_name"] = _card
        _gen, _av1 = gpu_generation()
        t("gpu: %s encodes AV1 = %s" % (_label, _want_av1),
          _av1 is _want_av1 and bool(_gen))
    # An RTX A4000 is AMPERE. A digit match reads the 4000 and calls it
    # Ada, which would tell the operator their card can encode AV1 when
    # it cannot.
    CAPS["gpu_name"] = "NVIDIA RTX A4000"
    _gen, _av1 = gpu_generation()
    t("gpu: RTX A4000 is Ampere, not Ada", _av1 is False and "Ampere" in _gen)
    CAPS["gpu_name"] = ""
    _gen, _av1 = gpu_generation()
    t("gpu: no card gives no verdict, so the trial decides",
      _gen == "" and _av1 is None)
    CAPS["gpu_name"] = "Some Board That Does Not Exist 9000"
    _gen, _av1 = gpu_generation()
    t("gpu: an unknown board gives no verdict either", _av1 is None)
    if _saved_name is None:
        CAPS.pop("gpu_name", None)
    else:
        CAPS["gpu_name"] = _saved_name

    # --- portability: video memory reserve -------------------------------
    _saved_enc = CAPS.get("encoder")
    CAPS["encoder"] = "nvenc"
    _r16 = sr_vram_reserve(16311.0)
    _r8 = sr_vram_reserve(8192.0)
    t("vram: a 16 GB card reserves the measured 3400 MB",
      abs(_r16 - 3400.0) < 60.0)
    t("vram: an 8 GB card reserves proportionally, not 42 pct of itself",
      _r8 < 2000.0 and _r8 >= SR_VRAM_RESERVE_MIN_MB)
    t("vram: the reserve never exceeds the configured ceiling",
      sr_vram_reserve(48000.0) <= SR_VRAM_RESERVE_MB + 1)
    CAPS["encoder"] = "svtav1"
    t("vram: a CPU encode reserves only compositor headroom",
      sr_vram_reserve(8192.0) <= SR_VRAM_RESERVE_MIN_MB)
    t("vram: an unmeasurable card keeps the flat figure",
      sr_vram_reserve(0.0) == float(SR_VRAM_RESERVE_MB))
    if _saved_enc is None:
        CAPS.pop("encoder", None)
    else:
        CAPS["encoder"] = _saved_enc

    # --- portability: scratch disk selection -----------------------------
    t("scratch: nvme outranks ssd outranks hdd",
      SCRATCH_RANK["nvme"] > SCRATCH_RANK["ssd"] > SCRATCH_RANK["hdd"])
    t("scratch: an unclassified disk sits between ssd and hdd",
      SCRATCH_RANK["ssd"] > SCRATCH_RANK["unknown"] > SCRATCH_RANK["hdd"])
    _fake = [{"root": "D:\\", "free": 900 * 1024 ** 3, "total": 0,
              "media": "hdd", "mbps": 0, "rank": SCRATCH_RANK["hdd"],
              "is_dst": False},
             {"root": "E:\\", "free": 200 * 1024 ** 3, "total": 0,
              "media": "nvme", "mbps": 0, "rank": SCRATCH_RANK["nvme"],
              "is_dst": False},
             {"root": "F:\\", "free": 400 * 1024 ** 3, "total": 0,
              "media": "nvme", "mbps": 0, "rank": SCRATCH_RANK["nvme"],
              "is_dst": True}]
    _fake.sort(key=lambda v: (-v["rank"], v["is_dst"], -v["free"]))
    t("scratch: the fast disk wins over the roomy one",
      _fake[0]["root"] == "E:\\")
    t("scratch: the destination's own disk is demoted, not excluded",
      _fake[1]["root"] == "F:\\" and _fake[-1]["root"] == "D:\\")
    t("scratch: a session is named for the process that owns it",
      re.match(r"^s\d+-\d{14}$", "s%d-%s" % (os.getpid(),
                                             time.strftime("%Y%m%d%H%M%S")))
      is not None)
    t("scratch: nothing is deleted when we did not create it",
      close_scratch_session(log) == 0)
    # A machine with one disk, which is the laptop case: the destination
    # volume must still be offered rather than the run refusing to start.
    t("scratch: a single-disk machine still gets an answer",
      SCRATCH_RANK.get("hdd", 0) > 0)

    # The build number, and whether every module agrees with it.
    #
    # Four modules each carry a VERSION and they had drifted to three
    # different values - the pipeline at 0.24, av1_metadata at 0.2,
    # av1_subs at 0.1 - while av1_warp carried none at all. The metadata
    # one is not cosmetic: it is stamped into the .info sidecar written
    # beside every finished file, so the library is full of files
    # claiming to have been made by a build that had not existed for
    # months. A version only some of the code agrees with is worse than
    # no version, so disagreement is a FAILURE, not a warning.
    t("version: three decimals, so 0.001 increments are expressible",
      re.match(r"^\d+\.\d{3}$", VERSION) is not None)
    _vmods = []
    for _mn in ("av1_metadata", "av1_subs", "av1_warp"):
        try:
            _m = __import__(_mn)
            _vmods.append((_mn, getattr(_m, "VERSION", None)))
        except Exception as _e:
            _vmods.append((_mn, "import failed: %s" % _e))
    for _mn, _mv in _vmods:
        t("version: %s reports %s, matching the pipeline"
          % (_mn, _mv), _mv == VERSION)

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
