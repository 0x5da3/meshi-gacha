#!/usr/bin/env python3
# meshi-gacha BGM generator (source of the embedded BGM_ASSETS in index.html).
# Composes richer per-store loops (saizeriya/ohsho/cocoichi x normal/RUSH) and
# encodes them as compact mono MP3.
#
# Regenerate:
#   pip install numpy lameenc
#   python3 tools/genmusic.py            # -> /tmp/bgm_assets.json
# then replace the `const BGM_ASSETS={...};` literal in index.html with
#   const BGM_ASSETS=<contents of /tmp/bgm_assets.json>;
# The JS side (bgmRenderAsset) trims MP3 decoder delay by amplitude onset and
# folds the reverb tail for a seamless loop, so loops MUST stay gap-free here.
import json, base64, math, io
import numpy as np
import lameenc

SR = 32000
BASE = 261.63  # C4, matches app's BGM_BASE

rng = np.random.default_rng(20260517)

def hz(semi):           # semitone offset from C4
    return BASE * (2.0 ** (semi / 12.0))

def t_arr(dur):
    return np.arange(int(round(dur * SR))) / SR

# ---- oscillators (mild anti-alias: additive for harmonic tones) ----
def osc(wave, f, dur, detune=0.0):
    f = f * (2.0 ** (detune / 1200.0))
    n = int(round(dur * SR))
    ph = 2 * np.pi * f * (np.arange(n) / SR)
    if wave == 'sine':
        return np.sin(ph)
    if wave == 'triangle':
        y = np.zeros(n)
        k = 1
        while True:
            h = 2 * k - 1
            if h * f > SR * 0.45 or h > 25:
                break
            y += ((-1) ** ((h - 1) // 2)) / (h * h) * np.sin(h * ph)
            k += 1
        return (8 / (np.pi ** 2)) * y
    if wave == 'square':
        y = np.zeros(n)
        h = 1
        while h * f < SR * 0.45 and h <= 31:
            y += np.sin(h * ph) / h
            h += 2
        return (4 / np.pi) * y
    if wave == 'saw':
        y = np.zeros(n)
        h = 1
        while h * f < SR * 0.45 and h <= 40:
            y += ((-1) ** (h + 1)) * np.sin(h * ph) / h
            h += 1
        return (2 / np.pi) * y
    return np.sin(ph)

def adsr(n, a, d, s, r, sl=0.7):
    a = max(1, int(a * SR)); d = max(1, int(d * SR))
    r = max(1, int(r * SR))
    env = np.zeros(n)
    i = 0
    ae = min(a, n); env[:ae] = np.linspace(0, 1, ae); i = ae
    if i < n:
        de = min(i + d, n)
        env[i:de] = np.linspace(1, sl, de - i); i = de
    if i < n - r:
        env[i:n - r] = sl; i = n - r
    if i < n:
        env[i:] = np.linspace(env[i - 1] if i > 0 else sl, 0, n - i)
    return env

# FM electric-piano voice (Rhodes-ish): carrier + bell-ish modulator
def epiano(f, dur, amp=1.0):
    n = int(round(dur * SR)); tt = np.arange(n) / SR
    env = np.exp(-tt * 3.2) * (1 - np.exp(-tt * 220))
    mod = np.sin(2 * np.pi * f * 2.0 * tt) * np.exp(-tt * 6.0) * 2.2
    tone = np.sin(2 * np.pi * f * tt + mod)
    tine = np.sin(2 * np.pi * f * 7.0 * tt) * np.exp(-tt * 26) * 0.18
    return (tone + tine) * env * amp

def pluck(f, dur, amp=1.0, wave='saw', bright=0.6):
    n = int(round(dur * SR)); tt = np.arange(n) / SR
    y = osc(wave, f, dur)
    fc = (f * (2 + 8 * bright)) * np.exp(-tt * 7) + f * 1.2
    y = onepole_lp_var(y, fc)
    env = np.exp(-tt * (5.5 - 3 * bright)) * (1 - np.exp(-tt * 300))
    return y * env * amp

def onepole_lp_var(x, fc):
    fc = np.clip(fc, 30, SR * 0.45)
    a = 1 - np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x); acc = 0.0
    av = a if np.isscalar(a) else a
    for i in range(len(x)):
        acc += (av[i] if not np.isscalar(av) else av) * (x[i] - acc)
        y[i] = acc
    return y

def lp_static(x, fc, q=0.7):
    # biquad lowpass
    w0 = 2 * np.pi * fc / SR
    al = math.sin(w0) / (2 * q)
    b0 = (1 - math.cos(w0)) / 2; b1 = 1 - math.cos(w0); b2 = b0
    a0 = 1 + al; a1 = -2 * math.cos(w0); a2 = 1 - al
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    y = np.empty_like(x); x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        xi = x[i]
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, xi; y2, y1 = y1, yi; y[i] = yi
    return y

def hp_static(x, fc):
    w0 = 2 * np.pi * fc / SR
    al = math.sin(w0) / (2 * 0.707)
    b0 = (1 + math.cos(w0)) / 2; b1 = -(1 + math.cos(w0)); b2 = b0
    a0 = 1 + al; a1 = -2 * math.cos(w0); a2 = 1 - al
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    y = np.empty_like(x); x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        xi = x[i]
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, xi; y2, y1 = y1, yi; y[i] = yi
    return y

def noise(dur):
    return rng.standard_normal(int(round(dur * SR)))

# ---- drums ----
def kick(dur=0.34, f0=150, f1=44, amp=1.0, click=0.5):
    n = int(round(dur * SR)); tt = np.arange(n) / SR
    fsw = f1 + (f0 - f1) * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(fsw) / SR
    body = np.sin(ph) * np.exp(-tt * 7.5)
    cl = noise(0.012); cl *= np.exp(-np.arange(len(cl)) / SR * 240) * click
    out = body
    out[:len(cl)] += cl
    return out * amp

def snare(dur=0.2, amp=1.0, tone=190, bright=1.0, soft=False):
    n = int(round(dur * SR)); tt = np.arange(n) / SR
    nz = noise(dur)
    nz = hp_static(nz, 1400 if not soft else 2400)
    nz = lp_static(nz, 8000 if not soft else 6000)
    nz *= np.exp(-tt * (16 if soft else 22)) * (0.6 if soft else 1.0)
    bd = (np.sin(2 * np.pi * tone * tt) + 0.5 * np.sin(2 * np.pi * tone * 1.6 * tt))
    bd *= np.exp(-tt * 30) * (0.0 if soft else 0.5)
    return (nz * bright + bd) * amp

def hat(dur=0.05, amp=1.0, open_=False):
    d = 0.16 if open_ else dur
    n = int(round(d * SR)); tt = np.arange(n) / SR
    nz = hp_static(noise(d), 8200)
    nz *= np.exp(-tt * (14 if open_ else 60))
    return nz * amp

def clap(amp=1.0):
    d = 0.16; out = np.zeros(int(d * SR))
    for off in (0.0, 0.009, 0.018, 0.028):
        s = int(off * SR); seg = hp_static(noise(0.09), 1300)
        tt = np.arange(len(seg)) / SR
        seg *= np.exp(-tt * 34)
        e = min(len(out), s + len(seg)); out[s:e] += seg[:e - s]
    tt = np.arange(len(out)) / SR
    out += hp_static(noise(d), 1500) * np.exp(-tt * 11) * 0.5
    return out * amp * 0.9

def shaker(amp=1.0):
    d = 0.09; tt = np.arange(int(d * SR)) / SR
    nz = hp_static(noise(d), 5000)
    nz *= np.minimum(1, tt * 120) * np.exp(-tt * 26)
    return nz * amp

def tabla(f=180, dur=0.22, amp=1.0):
    n = int(round(dur * SR)); tt = np.arange(n) / SR
    fsw = f * (1 + 1.6 * np.exp(-tt * 45))
    ph = 2 * np.pi * np.cumsum(fsw) / SR
    y = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.exp(-tt * 12)
    y += hp_static(noise(dur), 2500) * np.exp(-tt * 50) * 0.25
    return y * amp

# ---- effects ----
def delay(x, dt, fb, mix):
    d = int(dt * SR); y = x.copy()
    buf = np.zeros(len(x) + d)
    buf[:len(x)] = x
    out = np.zeros(len(x))
    dl = np.zeros(d)
    for i in range(len(x)):
        dv = dl[i % d]
        o = x[i] + dv
        dl[i % d] = x[i] + dv * fb
        out[i] = o
    return x * (1 - mix) + out * mix

def reverb(x, mix=0.16, decay=0.4):
    combs = [(0.0297, 0.78), (0.0371, 0.74), (0.0411, 0.70), (0.0437, 0.68)]
    out = np.zeros(len(x))
    for dt, g in combs:
        d = int(dt * SR); buf = np.zeros(d); acc = np.zeros(len(x))
        gg = g * (0.6 + decay)
        gg = min(0.92, gg)
        for i in range(len(x)):
            j = i % d
            v = x[i] + buf[j] * gg
            buf[j] = v
            acc[i] = v
        out += acc
    out /= len(combs)
    # one allpass for diffusion
    d = int(0.005 * SR); buf = np.zeros(d); ap = np.zeros(len(x)); g = 0.5
    for i in range(len(x)):
        j = i % d
        bv = buf[j]
        v = -g * out[i] + bv
        buf[j] = out[i] + g * v
        ap[i] = v
    return x * (1 - mix) + ap * mix

def soft_limit(x, drive=1.0):
    x = x * drive
    return np.tanh(x * 1.1) / 1.1

def add(buf, sig, at):
    s = int(round(at * SR))
    if s >= len(buf):
        return
    e = min(len(buf), s + len(sig))
    buf[s:e] += sig[:e - s]

def mix_into(dst, src, g=1.0):
    n = min(len(dst), len(src))
    dst[:n] += src[:n] * g

# ============================================================
# composition specs
# scale degrees are semitone offsets from C4 (0). negative = lower 8ve.
# prog = list of chord voicings (one per bar). bass_root per bar.
# ============================================================

def melody(scale, bars, beats, sd, lo, hi, seed, density=0.62, octave=0):
    r = np.random.default_rng(seed)
    notes = []
    pool = []
    o = lo
    while o <= hi:
        for s in scale:
            v = s + o * 12 + octave * 12
            pool.append(v)
        o += 1
    pool = sorted(pool)
    idx = len(pool) // 2
    steps = bars * beats * 4  # 16th grid
    i = 0
    while i < steps:
        beat_pos = i % 4
        on = r.random() < (density if beat_pos in (0, 2) else density * 0.5)
        if on:
            mv = r.choice([-2, -1, -1, 0, 1, 1, 2, 3])
            idx = int(np.clip(idx + mv, 0, len(pool) - 1))
            dur_steps = r.choice([1, 2, 2, 2, 3, 4])
            dur_steps = min(dur_steps, steps - i)
            notes.append((i, dur_steps, pool[idx]))
            i += dur_steps
        else:
            i += 1
    return notes

STORES = {}

# ---------- SAIZERIYA: jazzy cafe / bossa ----------
def build_saizeriya(rush):
    bpm = 132 if rush else 96
    beats = 4
    bars = 4
    spb = 60.0 / bpm
    sd = spb / 4.0
    loop = bars * beats * spb
    tail = 1.6
    buf = np.zeros(int((loop + tail) * SR))
    drum = np.zeros_like(buf)
    # Cmaj9 / Am9 / Dm9 / G13  (semitone voicings rel C4)
    voic = [
        [-12, 0, 4, 7, 11, 14],   # Cmaj9
        [-15, -3, 0, 4, 7, 12],   # Am9
        [-10, -3, 2, 5, 9, 12],   # Dm9
        [-17, -1, 2, 5, 9, 11],   # G13(ish)
    ]
    roots = [-24, -27, -22, -29]
    fifth = [7, 7, 7, 7]
    scale = [0, 2, 4, 5, 7, 9, 11]  # C major
    swing = 0.12 if not rush else 0.06

    for b in range(bars):
        bt = b * beats * spb
        ch = voic[b % 4]
        # bossa e-piano comping: syncopated stabs
        comp_steps = [0, 3, 6, 7, 10, 13] if not rush else [0, 2, 4, 6, 8, 10, 12, 14]
        for st in comp_steps:
            sw = swing * sd if (st % 2 == 1) else 0
            at = bt + st * sd + sw
            for k, semi in enumerate(ch[1:]):
                add(buf, epiano(hz(semi), 0.55, amp=0.085 * (0.9 if k else 1.0)), at)
        # warm pad
        padsig = np.zeros(int(beats * spb * SR))
        for semi in ch[1:5]:
            s = osc('triangle', hz(semi), beats * spb) * 0.5
            s += osc('sine', hz(semi - 12), beats * spb)[:len(s)] * 0.3
            mix_into(padsig, s)
        env = adsr(len(padsig), 0.25, 0.3, 0.85, 0.4, 0.8)
        add(buf, lp_static(padsig * env, 1600) * 0.05, bt)
        # upright bossa bass: root / fifth with passing notes
        bpat = [(0, roots[b % 4]), (3, roots[b % 4]),
                (6, roots[b % 4] + fifth[b % 4]), (8, roots[b % 4]),
                (11, roots[b % 4] + 3), (14, roots[(b + 1) % 4])]
        for st, semi in bpat:
            at = bt + st * sd
            bs = osc('triangle', hz(semi), 0.42) * 0.7 + osc('sine', hz(semi - 12), 0.42) * 0.5
            bs = lp_static(bs, 900)
            bs *= adsr(len(bs), 0.006, 0.12, 0.5, 0.12, 0.5)
            add(buf, bs * 0.42, at)
    # lead melody
    for (st, dn, semi) in melody(scale, bars, beats, sd, 0, 2, 41 + rush,
                                 density=0.5 if not rush else 0.62,
                                 octave=1 if rush else 0):
        at = st * sd + (swing * sd if st % 2 else 0)
        dur = dn * sd * 0.95
        sig = osc('triangle', hz(semi), dur) * 0.6 + osc('sine', hz(semi), dur) * 0.4
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.4 * np.arange(len(sig)) / SR)
        sig = sig * vib
        sig *= adsr(len(sig), 0.02, 0.08, 0.7, 0.12, 0.7)
        add(buf, sig * 0.13, at)
    # drums: soft kick, brush snare backbeat, shaker 8ths, rim
    for b in range(bars):
        bt = b * beats * spb
        for beat in range(beats):
            at = bt + beat * spb
            if beat in (0, 2) or rush:
                add(drum, kick(0.3, 120, 42, amp=0.55, click=0.2), at)
            if beat in (1, 3):
                add(drum, snare(0.18, amp=0.32, soft=True), at)
            for h in range(2):
                add(drum, shaker(0.5), at + h * spb / 2 + (swing * sd if h else 0))
        if rush:
            add(drum, clap(0.3), bt + spb)
            add(drum, clap(0.3), bt + 3 * spb)
    sig = buf + drum
    sig = delay(sig, sd * (2 if rush else 3), 0.22 if rush else 0.28, 0.18)
    sig = reverb(sig, mix=0.17, decay=0.45)
    return sig, loop, tail


# ---------- OHSHO: bright chinese funk ----------
def build_ohsho(rush):
    bpm = 168 if rush else 124
    beats = 4
    bars = 4
    spb = 60.0 / bpm
    sd = spb / 4.0
    loop = bars * beats * spb
    tail = 1.2
    buf = np.zeros(int((loop + tail) * SR))
    drum = np.zeros_like(buf)
    # C  F  G  C  (funk), pentatonic flavor
    chords = [[0, 4, 7, 14], [-7, 5, 9, 12], [-5, 7, 11, 14], [0, 4, 7, 12]]
    roots = [-24, -19, -17, -24]
    penta = [0, 2, 4, 7, 9]  # C major pentatonic

    for b in range(bars):
        bt = b * beats * spb
        ch = chords[b % 4]
        # funky 16th stabs (square synth)
        stab = [0, 2, 5, 6, 8, 11, 13, 14] if not rush else list(range(0, 16, 2))
        for st in stab:
            if rng.random() < 0.85:
                at = bt + st * sd
                s = np.zeros(int(0.16 * SR))
                for semi in ch:
                    s += osc('square', hz(semi), 0.16, detune=6) * 0.25
                    s += osc('square', hz(semi), 0.16, detune=-6) * 0.25
                s = lp_static(s, 3200, 1.0)
                s *= adsr(len(s), 0.004, 0.05, 0.3, 0.05, 0.3)
                add(buf, s * 0.085, at)
        # slap-ish funk bass
        bpat = [0, 2, 3, 6, 8, 10, 11, 14]
        for st in bpat:
            at = bt + st * sd
            semi = roots[b % 4] + (0 if st % 4 == 0 else rng.choice([0, 7, 5, 12]))
            bs = osc('saw', hz(semi), 0.22)
            tt = np.arange(len(bs)) / SR
            fc = hz(semi) * 6 * np.exp(-tt * 22) + hz(semi) * 1.5
            bs = onepole_lp_var(bs, fc)
            bs *= adsr(len(bs), 0.004, 0.08, 0.35, 0.06, 0.35)
            add(buf, bs * 0.42, at)
        # koto-ish pluck arpeggio (pentatonic, bright)
        for k, st in enumerate([0, 4, 8, 12]):
            semi = penta[(b + k) % len(penta)] + 12
            add(buf, pluck(hz(semi), 0.5, amp=0.07, wave='triangle', bright=0.7),
                bt + st * sd)
    # bright pentatonic lead riff
    for (st, dn, semi) in melody(penta, bars, beats, sd, 1, 2, 77 + rush,
                                 density=0.66 if not rush else 0.78,
                                 octave=1 if rush else 0):
        at = st * sd
        dur = dn * sd * 0.9
        sig = osc('square', hz(semi), dur, detune=5) * 0.5
        sig += osc('square', hz(semi), dur, detune=-5) * 0.5
        sig = lp_static(sig, 4200, 1.1)
        sig *= adsr(len(sig), 0.006, 0.06, 0.6, 0.08, 0.6)
        add(buf, sig * 0.1, at)
    # punchy funk drums
    for b in range(bars):
        bt = b * beats * spb
        kp = [0, 6, 10] if not rush else [0, 4, 8, 12]
        for st in kp:
            add(drum, kick(0.26, 165, 48, amp=0.7, click=0.6), bt + st * sd)
        add(drum, snare(0.17, amp=0.5), bt + spb)
        add(drum, snare(0.17, amp=0.5), bt + 3 * spb)
        if not rush:
            add(drum, snare(0.12, amp=0.16), bt + 2.75 * spb)  # ghost
        for st in range(0, 16, 2):
            op = (st % 8 == 6)
            add(drum, hat(amp=0.16 if not op else 0.13, open_=op), bt + st * sd)
        if rush:
            add(drum, clap(0.34), bt + spb)
            add(drum, clap(0.34), bt + 3 * spb)
    sig = buf + drum
    sig = delay(sig, sd * 3, 0.18, 0.12)
    sig = reverb(sig, mix=0.1, decay=0.3)
    return sig, loop, tail


# ---------- COCOICHI: hypnotic ethnic / phrygian ----------
def build_cocoichi(rush):
    bpm = 140 if rush else 100
    beats = 4
    bars = 4
    spb = 60.0 / bpm
    sd = spb / 4.0
    loop = bars * beats * spb
    tail = 1.8
    buf = np.zeros(int((loop + tail) * SR))
    drum = np.zeros_like(buf)
    # C phrygian: C Db Eb F G Ab Bb -> 0 1 3 5 7 8 10
    scale = [0, 1, 3, 5, 7, 8, 10]
    # tonic drone with bII / bVI color
    chord_bars = [[0, 7, 12], [0, 7, 12], [1, 8, 13], [0, 7, 12]]
    root = -24

    # sustained drone bass (root + fifth) across whole loop
    drone = np.zeros(int(loop * SR))
    for semi in (root, root + 7, root + 12):
        sa = osc('saw', hz(semi), loop)
        si = osc('sine', hz(semi), loop)
        s = sa * 0.5 + si[:len(sa)] * 0.5
        mix_into(drone, s)
    tt = np.arange(len(drone)) / SR
    fc = 600 + 500 * (0.5 + 0.5 * np.sin(2 * np.pi * tt / loop))
    drone = onepole_lp_var(drone, fc)
    drone *= 0.16
    add(buf, drone, 0)

    for b in range(bars):
        bt = b * beats * spb
        ch = chord_bars[b % 4]
        # shimmer pad
        ps = np.zeros(int(beats * spb * SR))
        for semi in ch:
            mix_into(ps, osc('triangle', hz(semi + 12), beats * spb), 0.33)
        ps *= adsr(len(ps), 0.4, 0.3, 0.7, 0.5, 0.7)
        add(buf, lp_static(ps, 2200) * 0.035, bt)
        # sitar-ish saw lead motif (phrygian), with bend on accents
        for k, st in enumerate([0, 3, 6, 8, 11, 14]):
            deg = scale[(b * 2 + k) % len(scale)]
            semi = deg + (12 if not rush else 24)
            dur = sd * 3
            sig = osc('saw', hz(semi), dur)
            ttl = np.arange(len(sig)) / SR
            bend = 1 + 0.012 * np.sin(2 * np.pi * 5.0 * ttl) * np.minimum(1, ttl * 6)
            sig = sig * bend
            sig = lp_static(sig, 2600, 1.4)
            sig *= adsr(len(sig), 0.01, 0.1, 0.5, 0.18, 0.55)
            add(buf, sig * 0.085, bt + st * sd)
    # hypnotic hand percussion
    for b in range(bars):
        bt = b * beats * spb
        tpat = [0, 3, 4, 7, 8, 11, 12, 14]
        for st in tpat:
            f = 200 if st % 8 == 0 else (150 if st % 4 == 0 else 240)
            add(drum, tabla(f, 0.22, amp=0.4), bt + st * sd)
        for st in range(0, 16, 2):
            add(drum, shaker(0.45), bt + st * sd)
        # frame-drum-ish low on the 1
        add(drum, kick(0.4, 110, 55, amp=0.5, click=0.1), bt)
        if rush:
            add(drum, kick(0.32, 130, 50, amp=0.5, click=0.2), bt + 2 * spb)
            add(drum, clap(0.26), bt + spb)
            add(drum, clap(0.26), bt + 3 * spb)
    sig = buf + drum
    sig = delay(sig, sd * (3 if rush else 4), 0.3, 0.22)
    sig = reverb(sig, mix=0.22, decay=0.55)
    return sig, loop, tail


BUILDERS = {
    'saizeriya': build_saizeriya,
    'ohsho': build_ohsho,
    'cocoichi': build_cocoichi,
}

def normalize(x, peak=0.89):
    m = np.max(np.abs(x))
    if m > 0:
        x = x * (peak / m)
    return x

def encode_mp3(samples, bitrate):
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(SR)
    enc.set_channels(1)
    enc.set_quality(2)
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767.0).astype('<i2').tobytes()
    data = enc.encode(pcm)
    data += enc.flush()
    return bytes(data)

def main():
    out = {}
    total = 0
    for store, fn in BUILDERS.items():
        for rush in (False, True):
            sig, loop, tail = fn(rush)
            sig = soft_limit(sig, 1.05)
            sig = normalize(sig, 0.9)
            br = 80 if store != 'cocoichi' else 88
            mp3 = encode_mp3(sig, br)
            b64 = base64.b64encode(mp3).decode('ascii')
            key = store + ('R' if rush else 'N')
            out[key] = {'b64': b64, 'loop': round(loop, 4)}
            total += len(b64)
            peak = float(np.max(np.abs(sig)))
            rms = float(np.sqrt(np.mean(sig ** 2)))
            head = float(np.max(np.abs(sig[:int(0.01 * SR)])))
            print(f"{key:14s} dur={len(sig)/SR:5.2f}s loop={loop:5.2f}s "
                  f"mp3={len(mp3)/1024:6.1f}KB b64={len(b64)/1024:6.1f}KB "
                  f"peak={peak:.3f} rms={rms:.3f} head10ms={head:.3f}")
    print(f"--- total base64 ~ {total/1024:.1f} KB ---")
    with open('/tmp/bgm_assets.json', 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':
    main()
