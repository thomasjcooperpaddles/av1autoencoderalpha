# AV1 Pipeline — changelog

Build numbers move by **0.001 on every change**. `VERSION` in
`av1_pipeline_v0_1.py` is the source of truth; `av1_metadata.py`,
`av1_subs.py` and `av1_warp.py` carry copies and selftest fails if they
disagree.

---

## 0.256 — 2026-08-17 — a ceiling on the SDR→HDR saturation, and an audit

New `sdr_to_hdr_saturation`, **hard-capped at 1.30**. That path may never
boost saturation by more than 30 %, whatever the config asks for; the
clamp is in `sdr_hdr_saturation()` rather than only at config load, so no
caller can route around it. The floor is 0.25, and values below 1.0 are
the useful direction. Default 1.0, which omits the option entirely.

### What the measurement actually showed, including my own error

Tom reported the SDR→HDR output looking far more saturated than previous
files. The first measurement appeared to confirm it — source SATAVG 12.90
against output 51.74, **4.0×**.

**That was wrong, and it was a harness error.** `signalstats` reports RAW
code values, and the source is 8-bit while the output is 10-bit: the same
colour is four times the code value. Measured like for like, with the
source forced to 10-bit:

| | SATAVG |
|---|---|
| source at 10-bit | 57.20 |
| shipped HDR output | 56.44 |
| **ratio** | **0.99×** |

The expansion does not boost encoded chroma. Isolating libplacebo
confirms it from the other direction: at default gain it *reduces*
measured chroma to 0.43× of source, which is what a correct
BT.709→BT.2020 primaries conversion should do — a wider gamut needs
smaller chroma excursions for the same colour. A PNG round trip through
the SR path was also tested as a suspect and cleared: 1.88× against
1.84× for the direct path, i.e. no difference.

So what Tom is seeing is real but is a **rendering** difference, not a
data one: the file is tagged bt2020nc/smpte2084, so a display switches to
HDR mode where SDR-derived material looks brighter and more saturated
than the SDR original. This gain is the lever for that.

Measured response of the filter, relative chroma at each gain, so a value
can be chosen rather than guessed:

| gain | 1.00 | 0.60 | 0.50 | 0.40 | 0.35 | 0.30 |
|---|---|---|---|---|---|---|
| relative chroma | 1.00× | 0.57× | 0.47× | 0.36× | 0.31× | 0.26× |

Five selftests cover the ceiling, the floor, pass-through, an unreadable
value falling back to neutral, and the default applying no lift. The
generated filter string was run through ffmpeg at four gains to confirm
it parses and executes.

### Audit

Static pass over all six modules, 14,376 lines: **zero undefined names,
zero duplicate dict keys, zero mutable default arguments, zero
unreachable statements, zero bare excepts, and no config key that nothing
reads.** The only three hits were `__file__`, which the analyser does not
know as a module builtin.

**195 checks, 0 failures.**

---

## 0.255 — 2026-08-17 — realesr-general-x4v3 is the default

Changed at Tom's instruction. It is the deeper SRVGGNetCompact (32 conv
against 16) and is the better choice for live action and mixed material,
which is most of what goes through here.

**The cost is time and it is not small**: measured 10.1 fps against
18.1 at 720×480, so roughly **3.0 hours per hour of 480i instead of
1.7** — about 1.8× longer on every sub-1080 file. Video memory is
unchanged at 205 MB. `realesr-animevideov3` remains offered and is the
right pick for an animation batch.

Set in all four places that decide it, because any one of them left
behind would silently win:

| | |
|---|---|
| `SR_MODEL_DEFAULT` | the pipeline's built-in |
| `sr_model` in av1_pipeline_config.txt | what a CLI run reads |
| `DEFAULT_SR_MODEL` in the GUI | what a fresh window shows |
| **`sr_model` in gui_settings.json** | **outranks the config on any GUI run** |

That last one is the one that mattered on 2026-08-17: the 0.249 default
change reached the config but not the saved settings, and a GUI run
carried on with x4plus for 22 projected hours. The old installer's
config template is updated too, so it cannot reintroduce the previous
default on a fresh install.

Verified by resolving the value through every route rather than by
editing and hoping: pipeline, config, saved GUI settings and GUI constant
all report `realesr-general-x4v3`, it is in the offered list, and
`sr_model_present()` resolves it. **190 checks, 0 failures.**

---

## 0.254 — 2026-08-17 — two models back, and a correction to 0.251

`realesr-general-x4v3` and `realesrgan-x4plus-anime` are installed and
offered alongside the default. **Tom spotted that the second of these is
one of the two models 0.251 removed** — the names differed because the
pipeline uses the ncnn-style lowercase names and the estimate table used
the upstream `.pth` names. `realesrgan-x4plus-anime` IS
`RealESRGAN_x4plus_anime_6B`. That was not made clear and should have
been.

### The reason 0.251 gave for removing them is void

0.251 said the x4plus pair were 4x-only networks with no x2/x3 weights,
computing sixteen times the pixel area for a sub-1080 source. **0.253 is
what made that wrong**: the PyTorch release of animevideov3 is x4-only
too, so on CUDA every model computes x4 and the output is reduced once to
the target. The x2/x3 distinction was an ncnn property throughout.

What surviving reason there is, is speed — and speed is a tradeoff for the
operator, not grounds for removing a choice. The actual fault on
2026-08-17 was settings precedence: a stale `gui_settings.json` pinned
x4plus and outranked the config. That is fixed where it belongs.

### Measured before installing, not estimated

24 real frames at 720×480 (the active picture of NTSC 480i), tile 768,
output 1964×1080, fp16, projected to the 107,892 frames in one hour:

| model | network | fps | 1 h of 480i | peak VRAM |
|---|---|---|---|---|
| **realesr-animevideov3** | SRVGG, 16 conv | 18.1 | **1.7 h** | 204 MB |
| **realesr-general-x4v3** | SRVGG, 32 conv | 10.1 | **3.0 h** | 205 MB |
| **realesrgan-x4plus-anime** | RRDB, 6 blocks | 2.1 | **14.6 h** | 2912 MB |
| realesrgan-x4plus — not offered | RRDB, 23 blocks | 0.7 | 43.1 h | 2682 MB |

All four load **strictly clean** — zero missing and zero unexpected keys —
which is what proves the architecture table rather than merely suggesting
it. **VRAM is not the constraint on a 16 GB card**: the worst of them
peaks at 2.9 GB, 18 % of the board. And tile size is nearly irrelevant to
throughput — 640, 768, 1024 and no tiling span 4 % — so the tile ladder
is about fitting, not speed.

An earlier estimate in conversation put these at 1 h / 2 h / 8 h / 30 h.
**Those were optimistic by 1.4–1.7×**, because they extrapolated from a
model-only rate and ignored the per-frame host transfer, GPU resize and
download. The table above replaces them.

### The architecture dispatch was a substring guess

`load_model` tested `"animevideov3" in name`, then `"anime" in name`, then
fell through to a 23-block RRDB. `realesr-general-x4v3` contains neither
substring, so it would have been built as the wrong network entirely —
and it needs `num_conv=32`, which `build_srvgg` had hardcoded to 16. It
was only safe because strict loading turns a wrong architecture into a
loud failure instead of quiet rubbish.

Now an explicit `MODEL_ARCH` table quoted from `inference_realesrgan.py`,
with the substring logic kept only as a last resort for an unknown name
and a printed note when it fires. Three selftests assert every offered
model has an architecture entry, has a weights URL, and that the table
matches the reference implementation.

**190 checks, 0 failures.**

---

## 0.253 — 2026-08-17 — one reduction, and two things that were quietly broken

### The upscale threw away 38 % of what the model produced

The chain was `640×352 → model x4 → 2560×1408 → reduce to 1280×704 →
PNG → ffmpeg lanczos back UP to 1964×1080`. On a 352-line source x2 is
704 lines, so 704 lines of real model output were discarded and then
interpolated back. MEASURED on 20 real frames, mean absolute laplacian
as a proxy for retained detail:

| | detail |
|---|---|
| plain lanczos upscale, no model | 0.515 |
| **the route that shipped** | **0.740** |
| **one reduction straight to 1080** | **1.021** |
| model native x4 | 1.275 |

Now one reduction, on the GPU, straight to the size the plan asked for
(`--out-w/--out-h`), and `SCALE_1080` comes out of the filter chain
because nothing is left for it to do. The blend leg follows the same
size, since blend refuses mismatched inputs.

**The requested scale factor buys nothing on CUDA.** All three released
`.pth` files are x4 networks — ncnn ships separate x2/x3 weights, the
PyTorch release does not — so the model computes x4 whatever is asked
for. `sr_scale` now only affects how many pixels the encoder sees.

Cost: 1.19× → 1.06× realtime, because the PNGs are 2.35× the pixels.
Worth it for 38 % more detail.

### A VRAM estimate sized from the wrong number

`sr_tile_for` and the headroom gate took the REQUESTED factor, so asking
for x2 budgeted the x2 cost while the model allocated x4 tensors. It
happened to be safe here (3328 MB budgeted, 2125 MB used) but that is
luck, and this is the one place where being wrong resets the display
driver. Both now use `SR_CUDA_NATIVE_SCALE`.

### The whisper filter could never take an absolute path

`esc_filter_path` escaped a Windows drive colon with ONE backslash.
ffmpeg escapes in layers and it needs two. MEASURED on 8.1.2, from
Python with an argument list — through Git Bash the test is worthless,
because it rewrites anything path-shaped and turns `C\:/x` into nonsense
that reads as an ffmpeg failure:

| | |
|---|---|
| `model=C\:/...` (what shipped) | **FAILS** — `No option name near '/ProgramData/...'` |
| `model=C\\:/...` | WORKS |
| `model='C:/...'` | FAILS |
| `model='C\:/...'` | WORKS |

So two of whisper_srt's three fallbacks could never work, and only the
bare-name-with-cwd form ever transcribed anything. The destination is now
escaped too, so that attempt writes where it was told rather than into
whichever directory it was run from. The old code also replaced
backslashes *after* turning them all into forward slashes — a step that
could never match. Three selftests pin it.

**Subtitles themselves were never broken**: a two-file run generated
none because those clips contain almost no dialogue (whisper returned
`*Squad*` and `- -`), and the pipeline correctly declined to build a
track from that.

### Frame verification: sample the film, not 24 guesses

`verify_by_frames` drew 24 frames — 0.018 % of a feature — each through
its own ffmpeg process and seek. The obvious fix, aligning the sample to
the reference's timecode, **is impossible**: TMDB's image records carry
`aspect_ratio, file_path, height, width, iso_3166_1, iso_639_1,
vote_average, vote_count` and nothing else. VERIFIED against the live API.

So the whole film is sampled instead, one decode pass with an `fps`
filter, plus a second **centre-crop hash** per frame because a difference
hash is not crop-invariant and a reference at a different crop changes
effectively every bit.

On Dead Space: Downfall, the same title, same four reference images:

| | verdict | closest |
|---|---|---|
| before | unverified | 17 of 64 bits, 24 frames |
| **after** | **confirmed** | **1 of 64 bits**, 2164 frames, matched at 276.0 s |

One of those backdrops *is* a frame from the film. The old sampler never
looked at it. Cost is 163 s on a 74-minute file, paid only when the text
evidence conflicts, which is already when screenshots run.

`verdict` now distinguishes **`no_reference_images`** from
`unverified` — the database having nothing to compare against says
nothing about the title, and reporting it as a failed match was
misleading. The report names how it matched, where, and at what density.

**Rejected optimisation, recorded in the code so it is not retried:**
`-skip_frame nokey` returned 1849 frames in 0.09 s against 31 in 2.20 s —
25× faster and 60× denser. It is neither. The file has 13 keyframes and
the flag re-emits the last decoded one for every frame it skips: 1849
outputs, **13 distinct hashes**, the first of them `0x0`. Trusting the
count would have shipped a worse sampler that reported a better one.

### Verified

**188 checks, 0 failures.** Two clips end to end at both NTSC rates:
29.97 → 1860 counted in, 1860 out; 23.976 → 1500 in, 1500 out; audio
short by exactly the declared filter latency in both. Note both fixtures
carry an `nb_frames` tag one higher than `count_packets` — the pipeline
uses the count, which is bug 61's fix behaving correctly, and the tag is
wrong in the cut file.

---

## 0.252 — 2026-08-17 — the GPU does the work now

The upscale ran at **0.27× realtime** and the log's own accounting said
why: `3368.3 s of that was model inference (20% of wall)`. The card was
doing one fifth of a cycle it was supposed to own. Now **1.19×**, with
the model at **79% of wall** — measured on the same 640×352 source,
same model, same tile.

| per 300-frame chunk | 0.251 | 0.252 |
|---|---|---|
| model (GPU) | 7.2 s | 6.2 s |
| CPU downscale x4→x2 | **12.2 s** | on the GPU, 0.6 s |
| torch import, per chunk | 3.2 s | once per file |
| ffmpeg extract | 1.0 s | under the model |
| NVENC chunk encode | 4.6 s | under the model |
| deliberate pause | 2.0 s | 0 |
| **wall** | **36.3 s** | **8.4 s** |

### The downscale was on the CPU

`realesr-animevideov3`'s PyTorch release ships **x4 weights only**, so
`sr_scale = 2` runs the network at x4 and reduces afterwards. That
reduction was a PIL LANCZOS resize of a 2560×1408 frame, on the main
thread, after the frame had already crossed PCIe — **40.7 ms a frame
against 1.9 ms on the card**, 12.2 s of a 28.9 s chunk, more than the
model itself.

It now happens in `gpu_resize()` before the frame comes back, which also
makes the transfer a quarter of the size. Quality was measured rather
than assumed, on 20 real frames against the old LANCZOS output:

| | PSNR vs LANCZOS | ms/frame |
|---|---|---|
| **bicubic + antialias** | **55.73 dB** | 1.5 |
| bilinear + antialias | 54.46 dB | 1.5 |
| area (box) | 54.15 dB | 1.6 |
| PIL LANCZOS | — | 41.0 |

Bicubic is both the closest and the fastest, so there is no tradeoff to
declare — and 55.7 dB is well clear of the 52.9 dB accepted for the whole
ncnn-to-CUDA switch.

### One upscaler process per file, not per chunk

`--serve` keeps `av1_upscale_cuda.py` resident, taking one job per line
on stdin. Saves the 3.2 s torch import 447 times over, and stops building
and tearing down a CUDA context 447 times per film — the operation most
likely to upset a driver on a card this new. `SrServer` owns it; the tile
step-down ladder restarts it, which costs one import rather than a file,
and the old one-process-per-chunk path remains as the fallback if the
resident one will not start.

### Three stages at once

Extract for chunk i+1 and the NVENC encode for chunk i-1 now run while
the model works on chunk i, across three rotating slot directories.
NVENC is a **separate hardware block** from the CUDA cores — the trace
recorded it as idle for the whole upscale — so this is not contention.
Measured on chunk 4 of the test: model 6.2 s with a 5.5 s encode running
underneath it and **0.00 s** spent waiting on it.

The model stays strictly one chunk at a time. It is the genuinely
GPU-bound part; overlapping two would make each slower and raise peak
video memory for nothing.

### Two load limits lifted

- **`sr_chunk_pause_sec` 2.0 → 0.0.** Once a chunk is 6 s rather than
  36 s, two seconds is 20% of it — a quarter of an hour per film of
  deliberate idling. It was added when the lockups were thought to be a
  load problem; they were the 2022 Vulkan binary, and the card died cold.
  `gpu_temp_max_c` and `gpu_wait_for_cool` remain and are the real
  protection: they refuse work when the card is actually hot.
- **`sr_prefetch_during` = 1, new.** The next file's probe, crop analysis
  and whisper pass run during the upscale again. Held back since
  2026-08-09 for the same disproven reason.

**If the machine locks up during a CUDA upscale, reverse these two
first** — both are config keys, not code edits, so they can be turned off
from the other side of a hard reboot.

### Verified

62-second clip end to end: **1860 frames in, 1860 out**, 1860 decoded
from the finished file, picture 62.062 s against a 62.062 s source
timeline, and the audio short by exactly the same 1,762 samples (36.7 ms)
as the declared filter latency. **185 checks, 0 failures.**

One bug found and fixed during the work, which is the reason for the
verification above: the first version of the overlapped loop had the
model and encode stages one indentation level out, so they ran **once
after the loop** instead of once per chunk. Every chunk was extracted and
counted, the frame check passed at 1860 in / 1860 out, and the file still
went out 2 seconds long — the duration gate caught it. `made` is now
asserted against the chunk count, and the frame check alone is not
sufficient evidence that the parts exist.

---

## 0.251 — 2026-08-17 — one upscale model

**`realesr-animevideov3` is now the only model offered.** The x4plus pair
are removed from `SR_MODELS` in the pipeline and the GUI, from the
config comments, and from both installers' config templates.

They are 4x-only networks with no x2 or x3 weights, so a sub-1080 source
is computed at **sixteen times** the pixel area and most of it is thrown
away in the downscale to 1080. Measured on 60 identical frames at tile
640: **11.51 fps against 1.53** — a 7.5× penalty, larger than the CUDA
switch itself.

### What prompted it

The first real run after 0.250 reported **0.06× realtime, 22 h
remaining** for one 74-minute film. The log line was unambiguous:

```
sr upscale start: realesrgan-x4plus
WARNING  sr scale x2 requested but realesrgan-x4plus only offers x4
```

The 0.249 default change reached `av1_pipeline_config.txt` but **not**
`gui_settings.json`, which the GUI writes on save and which outranks the
config — and the GUI's own `DEFAULT_SR_MODEL` was still x4plus, so
re-saving would have restored it either way. Both corrected; the saved
setting is rewritten to animevideov3.

`model_scales()` still answers for the 4x-only names, because a stale
config or an old command line can still name one. What must not happen
is one of them being *selectable*, and four selftests now assert that:
the tuple holds only animevideov3, the default is in it, nothing
matching `x4plus` is in it, and everything in it supports x2.

`--sr-model` rejects an unknown name outright (argparse `choices`); a
config or saved-GUI value that names one is reset to the default with a
printed note.

**185 checks, 0 failures.** Both installers rebuilt.

---

## 0.250 — 2026-08-16 — nothing runs at 100 %

The CUDA upscaler shipped in 0.249 **capped nothing**. It was the one
process on the machine with no ceiling on either resource — an omission,
since the ncnn path had the whole GPU budget machinery around it.

- `UTILISATION_CEILING_PCT = 98`. `GPU_MAX_PCT` and `CPU_MAX_PCT` are
  clamped to it at startup whatever the config asks for, with a printed
  note when a value is lowered. Nothing may take 100 % of anything: an
  allocation that does not fit on a GPU resets the display driver rather
  than failing politely, and a machine pinned at 100 % CPU stops
  responding to the person trying to find out why.
- `--vram-pct` on the CUDA backend, enforced by
  `torch.cuda.set_per_process_memory_fraction()`. **Verified directly:**
  with the ceiling at 10 % of a 16 GB board, a 3 GB allocation is
  refused with a catchable `OutOfMemoryError`.
- `--cpu-threads` caps torch's own pool, which otherwise sizes itself to
  every core it can see and ignores `cpu_max_pct` entirely.
- Exit code **3** for out-of-memory, so the tile step-down ladder can
  tell "the tile was too big" from "the tool is broken".

### Two memory bugs in 0.249's upscaler, fixed

The image pipeline submitted **every** load up front and retained
**every** save future to the end. At 300 frames a chunk that is ~200 MB
of decoded input plus up to **3.2 GB** of finished 2560×1408 output held
in RAM waiting on the disk — a footprint that grew with chunk length and
had no ceiling. Both queues are now bounded to a few frames of lead.
Measured cost: 19.56 → 18.09 fps, with `io_wait_s` at 0.03 s, so the GPU
is not being starved.

### Metrics added

`vram_cap_pct`, `vram_cap_mb`, `peak_vram_pct_of_board`, `cpu_threads`,
`io_threads`, `io_wait_s`. On this machine a real chunk peaks at
**13 % of the board**.

### Not proven

The exit-3 OOM handler is **written but unexercised**. Forcing the
ceiling to 2 % of the board during a real run did not trigger it — the
run completed at a reported 1422 MB peak, which is above that ceiling
and should have refused. The ceiling is provably enforced for a single
large allocation, but its behaviour inside the real inference path is
not yet understood and should not be relied on until it is.

---

## 0.249 — 2026-08-16 — Vulkan → CUDA upscaler

**The upscaler moves from ncnn/Vulkan to CUDA.** ncnn remains as the
fallback and is still the only option on machines without CUDA.

### Why

`realesrgan-ncnn-vulkan.exe` is dated **2022-04-24** — roughly three
years older than the Blackwell silicon it runs on. Blackwell requires
**64 KB resource alignment where earlier architectures accepted 4 KB**,
and a mismatch produces a *GPU lockup rather than a clean error*
([vkd3d-proton #2913](https://github.com/HansKristian-Work/vkd3d-proton/issues/2913)).

That matches this machine's freezes exactly:

| observation | fits |
|---|---|
| no `Display` 4101 | a wedged device is not a driver *timeout* |
| no bugcheck, no minidump | dumps are enabled with a 65 GB pagefile — the kernel never got to write one |
| nvlddmkm Event 153 storm | NVIDIA's internal recovery attempts, failing |
| card cold at 19 W of 180 W | it is stuck, not working |
| 10 of 10 resets inside a compute window | on 2026-08-16 |

Corroborated by [ncnn #6843](https://github.com/tencent/ncnn/issues/6843),
an RTX 5060 + ncnn Vulkan crash on the same GPU family.

**Superseded hypotheses, recorded so they are not revisited:** it is not
a TDR (no 4101 anywhere in 14 days, so `TdrDelay` is irrelevant); it is
not power (crashes do not track GPU error volume, and one crash had zero
GPU errors); Event 153 *count does not predict crashes* — 08-04 survived
1667 of them, 08-16 died on 10.

### Measured, 120 real frames, 640×352 → 2560×1408, tile 640

| | end-to-end | model only | peak VRAM |
|---|---|---|---|
| ncnn / Vulkan | 6.07 fps | — | ~4.2 GB budgeted |
| CUDA | **19.56 fps** | 39.75 fps | **2.1 GB** |

Output difference **52.9 dB PSNR, 0.03 % mean absolute** — fp16
rounding, visually identical.

### The model matters more than the backend

| model | params | CUDA | ncnn |
|---|---|---|---|
| `realesr-animevideov3` | 0.62 M | **11.51 fps** | 6.09 fps |
| `realesrgan-x4plus` | 16.70 M | 1.53 fps | 1.30 fps |

The config was on `realesrgan-x4plus`, which is why a real run sat at
0.05× realtime. **Default changed to `realesr-animevideov3`.**
Combined with the backend switch that is roughly **8.8× faster** than
the previous default.

### Changed

- `av1_upscale_cuda.py` — new. Real-ESRGAN on torch, same CLI shape as
  the ncnn binary (`-i -o -n -s -t -g`) so the call site barely moved.
  Networks defined in-file rather than via `realesrgan`/`basicsr`,
  which import `torchvision.transforms.functional_tensor` and break on
  modern torchvision. Weights load **strictly** — a key mismatch fails
  loudly instead of producing quiet rubbish.
- `sr_backend()` — probes torch for kernels matching this card's
  compute capability, not merely "is CUDA available". Cached.
- Threaded image I/O in the upscaler. The first CUDA benchmark ran
  **4.3 s of model inside a 66 s run** because PNG decode/encode was
  single-threaded — the GPU idle for 94 % of a GPU-bound-looking job.
  Now `compress_level=1` on intermediates and a read-ahead/write-behind
  pool.
- `use_sr` now asks the backend, not `TOOLS["realesrgan"]`. A CUDA-only
  machine was being sent down the lanczos path for want of a 2022
  Vulkan exe it no longer needs.
- `sr_model_present()` understands both backends.
- New config `sr_backend = auto | cuda | ncnn | off`.
- Per-chunk telemetry: backend, device, compute capability, torch and
  CUDA versions, precision, tile, model-only fps against wall fps, and
  peak VRAM — into the log and the plan notes, so the next regression
  is visible without reproducing it.

### Requires

`torch` with CUDA (`torch==2.9.1+cu128` for Python 3.14) and `pillow`.
Note the CUDA wheel index lags the CPU one, so this **downgrades torch
from 2.13.0+cpu to 2.9.1+cu128**. Nothing else here depends on torch —
argos uses ctranslate2.

---

## 0.248 — 2026-08-16 — sync-exact upscale timeline

- **Any deviation in length breaks sync**, so the upscale chunker is now
  frame-exact. `-t 10` selected frames by *duration*, and ten seconds is
  not a whole number of frames at any NTSC-fractional rate: 23.976 gives
  239.76 and ffmpeg emitted 240; 29.97 gives 299.7 and it emitted 300.
  Every chunk rounded **up**. Measured gains were exactly
  `1 − fractional part`: +0.242 frames/chunk at 23.976 and +0.302 at
  29.97, which over ~450 chunks became +108, +112 and +135 frames — and
  112 ÷ 23.976 = 4.67 s, precisely the drift that failed the file.
- Chunks now bounded by `-frames:v` with an exact count, starts derived
  from cumulative frame position.
- Source frame count is **counted** (`-count_packets`, ~4 s, no
  decoding), not computed from `duration × fps`.
- A short chunk is padded with a **duplicate frame** rather than
  allowed to shift everything after it.
- The length check now **fails the file**. It previously warned — and it
  did warn, correctly, on all four failures — then let them run 90
  minutes to a duration error. A warning nobody acts on is not a check.
- `Journal.save()` no longer kills a file. `os.replace()` is an atomic
  rename that SMB can refuse (`PermissionError: [WinError 5]` on
  `\\192.168.2.50\shared`), and it fired inside
  `journal.update(status="analyzing")`. Now retries, falls back to a
  direct write, and never raises.
- `probe_video_duration()` reads the **last video packet** before
  falling back to container duration, which is the length of the
  *longest track*. Trigun's Japanese FLAC runs 5463.5 s against 5433.4 s
  of picture; a correct 9 GB encode was failed for being 30 s short of a
  figure that never described the picture.

## 0.247 — 2026-08-16
- Subtitle track naming; `probe_file` now collects source track titles.

## 0.246 — 2026-08-15
- Subtitle language verified by **content**, not by tag.
- argos pair detection: checked the *languages* rather than the directed
  *pair*, so with `en→ja` installed a `ja→en` lookup silently returned
  nothing and never downloaded the model.
- A failed translation no longer writes the untranslated original and
  labels it English.

## 0.245 — 2026-08-13
- Fixed audio/subtitle track layout: a1 English boosted, a2 English
  original, a3 Japanese unboosted; s1 forced/English/blank, s2 English,
  s3 Japanese.

## 0.244 — 2026-08-12
- Upscaler falls back to lanczos when its model is absent, instead of
  failing every sub-1080 file.

## 0.243 — 2026-08-12
- Startup banner no longer tells the operator to set `sr_tmp_dir` when
  blank is the recommended setting.

## 0.242 — 2026-08-12
- Portability: GPU generation detection (RTX 30/40/50), video-memory
  reserve sized to the actual card, automatic scratch-disk selection
  with a self-removing session directory, and startup no longer refuses
  to run on a machine with no NVENC.

## 0.241 — 2026-08-11
- Adopted three-decimal build numbers moving by 0.001 per change, with
  selftest enforcing that all four modules agree. `av1_metadata` had
  been stamping **"0.2"** into every `.info` sidecar.
