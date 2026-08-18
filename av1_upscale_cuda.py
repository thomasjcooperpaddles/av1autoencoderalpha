#!/usr/bin/env python3
# =====================================================================
# av1_upscale_cuda.py   V0.256   2026-08-17
#
# Real-ESRGAN on CUDA, as a drop-in replacement for
# realesrgan-ncnn-vulkan.
#
# WHY THIS EXISTS
#
# The ncnn/Vulkan binary the pipeline used is dated 2022-04-24, roughly
# three years older than the Blackwell silicon it now runs on. Blackwell
# requires 64 KB resource alignment where earlier architectures accepted
# 4 KB, and a mismatch there produces a GPU LOCKUP rather than a clean
# error - which matches this machine's freezes exactly: no TDR (no
# Display 4101), no bugcheck, no minidump, an Event 153 storm, and a
# card sitting cold because it is stuck rather than working.
#
# CUDA is a completely different driver path, and torch 2.9.1+cu128
# carries real sm_120 kernels.
#
# WHY THE NETWORKS ARE DEFINED HERE
#
# Not via the `realesrgan` / `basicsr` pip packages. basicsr imports
# torchvision.transforms.functional_tensor, which modern torchvision
# removed, so that chain breaks on any current install. The two
# architectures Real-ESRGAN actually uses are small and stable, and
# defining them here removes a fragile dependency from the one path
# that has to work. The weights are the official released .pth files
# and are loaded STRICTLY - a key mismatch means the architecture is
# wrong and it fails loudly rather than producing quiet rubbish.
#
# INTERFACE
#
# Deliberately the same shape as realesrgan-ncnn-vulkan so the pipeline
# call site barely changes:
#
#   -i INDIR  -o OUTDIR  -n MODEL  -s SCALE  -t TILE  -g GPUID
#
# plus --json-metrics, which the pipeline reads to log per-chunk
# telemetry.
# =====================================================================
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

VERSION = "0.256"

MODEL_URLS = {
    "realesr-animevideov3":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesr-animevideov3.pth",
    "realesrgan-x4plus":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x4plus-anime":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "realesr-general-x4v3":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesr-general-x4v3.pth",
}

# WHICH NETWORK EACH MODEL IS, BY NAME RATHER THAN BY GUESS.
#
# This was a substring test - "animevideov3" in name, else "anime" in
# name, else 23-block RRDB - and it silently mis-built anything it did
# not recognise. realesr-general-x4v3 is an SRVGGNetCompact with
# num_conv=32, but it contains neither substring, so it fell through to
# the 23-block RRDB branch. Strict weight loading turns that into a loud
# failure rather than quiet rubbish, which is the only reason it was
# safe. An explicit table cannot be wrong by accident.
#
# Values quoted from the reference implementation
# (inference_realesrgan.py):
#   RealESRGAN_x4plus            RRDBNet num_block=23 num_feat=64 x4
#   RealESRNet_x4plus            RRDBNet num_block=23 num_feat=64 x4
#   RealESRGAN_x4plus_anime_6B   RRDBNet num_block=6  num_feat=64 x4
#   RealESRGAN_x2plus            RRDBNet num_block=23 num_feat=64 x2
#   realesr-animevideov3         SRVGGNetCompact num_conv=16 num_feat=64
#   realesr-general-x4v3         SRVGGNetCompact num_conv=32 num_feat=64
MODEL_ARCH = {
    "realesr-animevideov3": ("srvgg", {"num_conv": 16}),
    "realesr-general-x4v3": ("srvgg", {"num_conv": 32}),
    "realesrgan-x4plus": ("rrdb", {"num_block": 23}),
    "realesrgan-x4plus-anime": ("rrdb", {"num_block": 6}),
}

# Every one of these is a x4 network. ncnn ships separate x2/x3 weight
# files for animevideov3; the PyTorch release does not, so a smaller
# output is produced by resizing after inference rather than by a
# cheaper model. That is a real difference from the ncnn backend and it
# is why the scale factor buys less here than it does there.
NATIVE_SCALE = 4


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------------
# architectures
# ---------------------------------------------------------------------
def build_srvgg(num_feat=64, num_conv=16, upscale=4):
    """SRVGGNetCompact - the realesr-animevideov3 network.

    A plain stack of 3x3 convolutions with PReLU between them, a
    pixel-shuffle at the end, and a nearest-neighbour skip. Small and
    fast, which is why it is the animation default.
    """
    torch = _torch()
    import torch.nn as nn

    class SRVGGNetCompact(nn.Module):
        def __init__(self):
            super().__init__()
            self.upscale = upscale
            body = [nn.Conv2d(3, num_feat, 3, 1, 1),
                    nn.PReLU(num_parameters=num_feat)]
            for _ in range(num_conv):
                body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                body.append(nn.PReLU(num_parameters=num_feat))
            body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
            self.body = nn.Sequential(*body)
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = self.upsampler(self.body(x))
            # the skip: nearest-neighbour upsample of the input, added
            # back, so the network only has to learn the residual
            out += nn.functional.interpolate(
                x, scale_factor=self.upscale, mode="nearest")
            return out

    return SRVGGNetCompact()


def build_rrdb(num_feat=64, num_block=23, num_grow_ch=32):
    """RRDBNet - the realesrgan-x4plus network.

    Residual-in-residual dense blocks. Far heavier than SRVGG, which is
    why x4plus is slow and animevideov3 is not.
    """
    torch = _torch()
    import torch.nn as nn

    class ResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch,
                                   3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch,
                                   3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch,
                                   3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat,
                                   3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = ResidualDenseBlock()
            self.rdb2 = ResidualDenseBlock()
            self.rdb3 = ResidualDenseBlock()

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(nn.functional.interpolate(
                feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(nn.functional.interpolate(
                feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def load_model(name, weights, device, half=True, log=print):
    """Build the right architecture and load its weights STRICTLY."""
    torch = _torch()
    kind, kw = MODEL_ARCH.get(name, (None, None))
    if kind is None:
        # Unknown name: fall back to the old heuristic rather than
        # refusing outright, but say so - a wrong guess here fails on the
        # strict load below, which is the point.
        log("model %s is not in MODEL_ARCH; guessing its architecture"
            % name)
        if "animevideov3" in name:
            kind, kw = "srvgg", {"num_conv": 16}
        elif "anime" in name:
            kind, kw = "rrdb", {"num_block": 6}
        else:
            kind, kw = "rrdb", {"num_block": 23}
    net = build_srvgg(**kw) if kind == "srvgg" else build_rrdb(**kw)
    sd = torch.load(str(weights), map_location="cpu", weights_only=True)
    # the released files wrap the weights under one of two keys
    for key in ("params_ema", "params"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "weights do not match the architecture for %s: %d missing, "
            "%d unexpected (first missing: %s, first unexpected: %s)"
            % (name, len(missing), len(unexpected),
               missing[0] if missing else "-",
               unexpected[0] if unexpected else "-"))
    net.eval()
    net = net.to(device)
    if half and device.type == "cuda":
        net = net.half()
    return net


# ---------------------------------------------------------------------
# tiled inference
# ---------------------------------------------------------------------
def gpu_resize(out, size):
    """Resize a 1x3xHxW float tensor to (w, h) ON THE DEVICE.

    This used to be a PIL LANCZOS resize on the main thread, after the
    frame had already been pulled back to the CPU. MEASURED on this card:
    40.7 ms per frame against 1.9 ms here, which over a 300-frame chunk is
    12.2 s of a 28.9 s chunk - more than the model itself - and every
    millisecond of it with the GPU idle. It also meant transferring the
    x4 frame across PCIe (10.8 MB) only to throw three quarters of it
    away; resizing first makes that transfer a quarter of the size.

    Bicubic with antialiasing, which is the closest torch has to LANCZOS.
    MEASURED against the old PIL LANCZOS path on 20 real frames at
    2560x1408 -> 1280x704: bicubic+antialias 55.73 dB PSNR, bilinear
    54.46, area 54.15, all three at ~1.5 ms against LANCZOS at 41.0 ms.
    Bicubic is both the closest and the fastest, so there is no tradeoff
    to declare here - and 55.7 dB is well clear of the 52.9 dB that was
    accepted for the whole ncnn-to-CUDA switch.

    Antialiasing only means anything when reducing; on the way up it is
    ignored, so it is only asked for when the target is smaller.

    Args:
        out: 1x3xHxW tensor, values in 0..1, on the device.
        size: (width, height) wanted.

    Returns:
        The resized tensor, still on the device.
    """
    nnf = _torch().nn.functional
    w, h = size
    sh, sw = int(out.shape[-2]), int(out.shape[-1])
    if (sh, sw) == (h, w):
        return out
    return nnf.interpolate(out.float(), size=(h, w), mode="bicubic",
                           antialias=(h < sh or w < sw),
                           align_corners=False)


def upscale_one(net, img, device, tile, pad, half, scale=NATIVE_SCALE,
                out_size=None):
    """Upscale a HxWx3 uint8 array. Tiled, with overlap to hide seams.

    out_size, when given, is the (width, height) actually wanted. The
    reduction from the network's native x4 happens on the device before
    the frame comes back - see gpu_resize.
    """
    torch = _torch()
    import numpy as np
    h, w = img.shape[:2]
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device).float().div_(255.0)
    if half and device.type == "cuda":
        x = x.half()
    out = torch.zeros((1, 3, h * scale, w * scale), device=device,
                      dtype=x.dtype)
    if tile <= 0:
        with torch.no_grad():
            out = net(x)
    else:
        n_y = int(math.ceil(h / tile))
        n_x = int(math.ceil(w / tile))
        for iy in range(n_y):
            for ix in range(n_x):
                y0, x0 = iy * tile, ix * tile
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                # pad outward so the network sees context either side of
                # the seam, then discard the padding from the result
                py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
                patch = x[:, :, py0:py1, px0:px1]
                with torch.no_grad():
                    o = net(patch)
                # where the wanted region sits inside the padded output
                ty0 = (y0 - py0) * scale
                tx0 = (x0 - px0) * scale
                ty1 = ty0 + (y1 - y0) * scale
                tx1 = tx0 + (x1 - x0) * scale
                out[:, :, y0 * scale:y1 * scale,
                    x0 * scale:x1 * scale] = o[:, :, ty0:ty1, tx0:tx1]
    out = out.clamp_(0, 1)
    if out_size is not None:
        out = gpu_resize(out, out_size).clamp_(0, 1)
    out = out.mul_(255.0).round_()
    return out.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Real-ESRGAN on CUDA. Same arguments as "
                    "realesrgan-ncnn-vulkan so it can stand in for it.")
    ap.add_argument("-i", dest="indir", default="")
    ap.add_argument("-o", dest="outdir", default="")
    ap.add_argument("--serve", action="store_true",
                    help="stay resident and take one job per line on "
                         "stdin as indir<TAB>outdir, writing a JSON "
                         "metrics line per job. Keeps the model and the "
                         "CUDA context alive across chunks.")
    ap.add_argument("-n", dest="model", default="realesr-animevideov3")
    ap.add_argument("-s", dest="scale", type=int, default=4)
    ap.add_argument("-t", dest="tile", type=int, default=0)
    ap.add_argument("--out-w", type=int, default=0,
                    help="exact output width. With --out-h this overrides "
                         "-s: the network runs at its native x4 and the "
                         "result is reduced ONCE, on the GPU, straight to "
                         "the size actually wanted. Reducing to x2 and "
                         "letting the caller scale back up to 1080 threw "
                         "away 38 pct of the detail the model produced.")
    ap.add_argument("--out-h", type=int, default=0,
                    help="exact output height; see --out-w")
    ap.add_argument("-g", dest="gpu", type=int, default=0)
    ap.add_argument("-f", dest="fmt", default="png")
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--fp32", action="store_true",
                    help="full precision; half is the default and is "
                         "about twice as fast with no visible difference")
    ap.add_argument("--json-metrics", default="",
                    help="write per-run telemetry here")
    ap.add_argument("--vram-pct", type=float, default=98.0,
                    help="hard ceiling on this process's share of video "
                         "memory, as a percentage of the board. Never "
                         "100: the compositor and anything else on the "
                         "desktop need room, and an allocation that does "
                         "not fit does not fail politely on a GPU.")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="cap torch's CPU thread pool. 0 leaves it to "
                         "torch, which sizes itself to the whole machine")
    ap.add_argument("--io-threads", type=int, default=0,
                    help="image decode/encode workers. 0 picks half the "
                         "cores")
    a = ap.parse_args(argv)
    if not a.serve and not (a.indir and a.outdir):
        ap.error("-i and -o are required unless --serve is given")

    torch = _torch()
    import numpy as np
    from PIL import Image

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 2
    device = torch.device("cuda:%d" % max(0, a.gpu))
    torch.backends.cudnn.benchmark = True

    # HARD CEILING ON VIDEO MEMORY, enforced by torch rather than hoped
    # for. set_per_process_memory_fraction makes the allocator refuse
    # past the limit with a catchable OutOfMemoryError, instead of the
    # driver being asked for memory that is not there - which on a GPU
    # is not a polite failure. Never 100 pct: the desktop compositor
    # alone holds around 1.8 GB on this machine, and leaving the card
    # no headroom is what took it down on 2026-08-06.
    vram_pct = max(10.0, min(98.0, float(a.vram_pct)))
    try:
        torch.cuda.set_per_process_memory_fraction(vram_pct / 100.0, device)
    except Exception as e:
        print("could not set the vram ceiling: %s" % e, file=sys.stderr)

    # And the CPU. torch sizes its own thread pool to every core it can
    # see, which ignores the pipeline's cpu_max_pct entirely and is felt
    # as the desktop going unresponsive during a long batch.
    if a.cpu_threads > 0:
        try:
            torch.set_num_threads(a.cpu_threads)
        except Exception:
            pass

    mdir = Path(a.models_dir) if a.models_dir else \
        Path(__file__).resolve().parent / "models_cuda"
    mdir.mkdir(parents=True, exist_ok=True)
    weights = mdir / (a.model + ".pth")
    if not weights.exists():
        url = MODEL_URLS.get(a.model)
        if not url:
            print("no weights and no download URL for %s" % a.model,
                  file=sys.stderr)
            return 2
        print("fetching %s" % url, file=sys.stderr)
        import urllib.request
        urllib.request.urlretrieve(url, str(weights))

    t_load = time.time()
    net = load_model(a.model, weights, device, half=not a.fp32)
    t_load = time.time() - t_load

    tile = a.tile if a.tile > 0 else 0
    pad = 16 if tile else 0

    def run_batch(indir, outdir):
        """One folder of frames in, one folder out. Returns metrics.

        Split out of main so a persistent process can serve chunk after
        chunk without paying for torch and the CUDA context each time -
        MEASURED at 3.2 s of a 36.3 s chunk, which over 447 chunks of one
        film is about 24 minutes of importing the same library.
        """
        return _batch(net, device, torch, np, Image, indir, outdir, a,
                      tile, pad, vram_pct, t_load)

    if a.serve:
        # SERVE MODE. One job per line on stdin, "indir<TAB>outdir", one
        # JSON metrics line per job on stdout. EOF or a blank line ends
        # it. The model, the CUDA context and cudnn's autotuning all
        # survive between jobs, which is the whole point.
        print(json.dumps({"ready": True, "version": VERSION,
                          "device": torch.cuda.get_device_name(device),
                          "seconds_load": round(t_load, 3)}), flush=True)
        rc = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            parts = line.split("\t")
            if len(parts) != 2:
                print(json.dumps({"error": "expected indir<TAB>outdir"}),
                      flush=True)
                rc = 1
                continue
            try:
                m = run_batch(Path(parts[0]), Path(parts[1]))
            except Exception as e:
                # A failure on one chunk must not take the server down;
                # the caller decides whether to retry with a smaller tile
                # or give up on the file.
                name = type(e).__name__
                m = {"error": "%s: %s" % (name, e),
                     "oom": "OutOfMemory" in name}
                rc = 3 if m["oom"] else 1
            print(json.dumps(m), flush=True)
        return rc

    metrics = run_batch(Path(a.indir), Path(a.outdir))
    print(json.dumps(metrics))
    if a.json_metrics:
        try:
            Path(a.json_metrics).write_text(json.dumps(metrics, indent=1),
                                            encoding="utf-8")
        except OSError as e:
            print("could not write metrics: %s" % e, file=sys.stderr)
    return 0


def _batch(net, device, torch, np, Image, indir, outdir, a, tile, pad,
           vram_pct=98.0, load_s=0.0):
    """Upscale every frame in indir into outdir. See run_batch."""
    t_start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in indir.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg",
                                            ".webp")])
    if not files:
        raise RuntimeError("no images in %s" % indir)

    t_infer = 0.0
    t_load = load_s
    n_done = 0
    torch.cuda.reset_peak_memory_stats(device)

    # THREAD THE IMAGE I/O, or it dominates everything.
    #
    # Measured on 120 frames at 640x352 -> 2560x1408: the model took
    # 4.3 s and the whole run took 66 s. Sixty-two seconds of that was
    # PNG decode and encode on one thread. The GPU is idle for 94 pct
    # of a run that looks GPU-bound. ncnn does its own I/O in C++ across
    # the load:proc:save threads its -j flag configures, which is
    # exactly why the old backend appeared faster end to end while
    # being four times slower at the actual arithmetic.
    #
    # These PNGs are intermediates - written here and read straight back
    # by ffmpeg, then deleted - so compress_level=1 is free speed.
    # Level 6 is Pillow's default and spends most of its time shaving
    # bytes off a file that lives for seconds.
    from concurrent.futures import ThreadPoolExecutor
    io_threads = (a.io_threads if a.io_threads > 0
                  else max(2, min(8, (os.cpu_count() or 4) // 2)))

    def _load(p):
        return p, np.asarray(Image.open(p).convert("RGB"))

    def _save(job):
        path, arr = job
        Image.fromarray(arr).save(path, compress_level=1)

    # BOUNDED read-ahead and write-behind.
    #
    # The first version submitted every load at once and kept every save
    # future alive to the end. At 300 frames a chunk that is ~200 MB of
    # decoded input plus up to 3.2 GB of finished 2560x1408 output held
    # in RAM waiting on the disk - a memory profile that grows with
    # chunk length and has no ceiling at all. Capping the queues costs
    # nothing in throughput, because a few frames of lead is all the GPU
    # ever needs, and bounds the footprint to a handful of frames.
    lead = max(2, io_threads)
    t_io_wait = 0.0
    with ThreadPoolExecutor(max_workers=io_threads) as ld, \
            ThreadPoolExecutor(max_workers=io_threads) as sv:
        pending = []
        inflight = []
        nxt = 0
        while nxt < min(lead, len(files)):
            inflight.append(ld.submit(_load, files[nxt]))
            nxt += 1
        while inflight:
            fut = inflight.pop(0)
            if nxt < len(files):
                inflight.append(ld.submit(_load, files[nxt]))
                nxt += 1
            t_w = time.time()
            p, img = fut.result()
            t_io_wait += time.time() - t_w
            # keep the write-behind queue short for the same reason
            while len(pending) >= lead:
                pending.pop(0).result()
            # The network is x4. Anything else is a reduction, and it now
            # happens ON THE GPU inside upscale_one, before the frame
            # crosses PCIe - it used to be a PIL LANCZOS resize here, on
            # this thread, at 40.7 ms a frame against 1.9 ms on the card.
            want = None
            if a.out_w > 0 and a.out_h > 0:
                want = (a.out_w, a.out_h)
            elif a.scale != NATIVE_SCALE:
                want = (max(1, int(round(img.shape[1] * a.scale))),
                        max(1, int(round(img.shape[0] * a.scale))))
            t0 = time.time()
            out = upscale_one(net, img, device, tile, pad, not a.fp32,
                              out_size=want)
            torch.cuda.synchronize(device)
            t_infer += time.time() - t0
            pending.append(sv.submit(
                _save, (outdir / (p.stem + "." + a.fmt), out)))
            n_done += 1
        for f in pending:
            f.result()
    peak_vram = torch.cuda.max_memory_allocated(device) / 1048576.0

    total = time.time() - t_start
    metrics = {
        "backend": "cuda",
        "version": VERSION,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": "sm_%d%d" % torch.cuda.get_device_capability(device),
        "model": a.model,
        "scale": a.scale,
        "native_scale": NATIVE_SCALE,
        "out_w": a.out_w,
        "out_h": a.out_h,
        "tile": tile,
        "precision": "fp32" if a.fp32 else "fp16",
        "frames": n_done,
        "seconds_total": round(total, 3),
        "seconds_model": round(t_infer, 3),
        "seconds_load": round(t_load, 3),
        "fps": round(n_done / total, 3) if total > 0 else 0,
        "fps_model_only": round(n_done / t_infer, 3) if t_infer > 0 else 0,
        "peak_vram_mb": round(peak_vram, 1),
        "vram_cap_pct": vram_pct,
        "vram_cap_mb": round(
            torch.cuda.get_device_properties(device).total_memory
            / 1048576.0 * vram_pct / 100.0, 1),
        "peak_vram_pct_of_board": round(
            100.0 * peak_vram /
            (torch.cuda.get_device_properties(device).total_memory
             / 1048576.0), 1),
        "cpu_threads": torch.get_num_threads(),
        "io_threads": io_threads,
        "io_wait_s": round(t_io_wait, 3),
    }
    return metrics


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Out of memory is now a REACHABLE outcome rather than a crash,
        # because --vram-pct makes the allocator refuse past the ceiling
        # instead of asking the driver for memory that is not there.
        # Exit 3 so the caller can tell "the tile was too big" from "the
        # tool is broken" - the pipeline steps the tile down and retries
        # on either, but the log should say which happened.
        name = type(e).__name__
        if "OutOfMemory" in name:
            print("OUT OF MEMORY: the tile does not fit under the "
                  "configured ceiling. %s" % e, file=sys.stderr)
            sys.exit(3)
        import traceback
        traceback.print_exc()
        sys.exit(1)
