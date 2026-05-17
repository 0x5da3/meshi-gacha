#!/usr/bin/env python3
# meshi-gacha BGM generator (source of the embedded BGM_ASSETS in index.html).
# Composes richer per-store loops (saizeriya/ohsho/cocoichi x normal/RUSH) and
# encodes them as compact mono MP3.
#
# Regenerate (scipy gives a big speedup; multiprocessing renders the 6
# tracks in parallel; both degrade gracefully if unavailable):
#   pip install numpy scipy lameenc
#   python3 tools/genmusic.py            # -> /tmp/bgm_assets.json
# then replace the `const BGM_ASSETS={...};` literal in index.html with
#   const BGM_ASSETS=<contents of /tmp/bgm_assets.json>;
# The JS side (bgmRenderAsset) trims MP3 decoder delay by amplitude onset and
# folds the reverb tail for a seamless loop, so loops MUST stay gap-free here.
#
# Composition aims for a Yoko Kanno flavor: rootless extended/altered jazz
# voicings, ii-V-I turnarounds with tritone subs, walking bass with chromatic
# approach tones, a harmonized countermelody, and a whole-step "chorus" key
# lift on the RUSH variant.
import json, base64, math, io
import numpy as np
import lameenc
try:
    from scipy.signal import lfilter as _lfilter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

SR = 32000
BASE = 261.63  # C4, matches app's BGM_BASE

rng = np.random.default_rng(20260517)

# Yoko Kanno-ish "chorus lift": RUSH variant modulates up a whole step.
# Every pitched voice routes through hz(), so one global transpose is enough
# (drums use raw frequencies and are intentionally unaffected).
_TP = 0

def hz(semi):           # semitone offset from C4 (+ global key lift)
    return BASE * (2.0 ** ((semi + _TP) / 12.0))

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

# --- fast IIR helpers (scipy lfilter when available, else pure-python) ---
def _iir(b, a, x):
    if HAVE_SCIPY:
        return _lfilter(np.asarray(b, float), np.asarray(a, float), x)
    b = list(b) + [0.0] * (3 - len(b))
    a = list(a) + [0.0] * (3 - len(a))
    y = np.empty_like(x); x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        xi = x[i]
        yi = b[0]*xi + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2
        x2, x1 = x1, xi; y2, y1 = y1, yi; y[i] = yi
    return y

def _comb(x, d, g):                      # y[n] = x[n] + g*y[n-d]
    if HAVE_SCIPY:
        a = np.zeros(d + 1); a[0] = 1.0; a[-1] = -g
        return _lfilter(np.array([1.0]), a, x)
    y = np.empty_like(x); buf = np.zeros(d)
    for i in range(len(x)):
        j = i % d; v = x[i] + buf[j] * g; buf[j] = v; y[i] = v
    return y

def onepole_lp_var(x, fc):
    fc = np.clip(fc, 30, SR * 0.45)
    if np.isscalar(fc):
        a = 1 - np.exp(-2 * np.pi * fc / SR)
        return _iir([a], [1.0, -(1 - a)], x)
    if not HAVE_SCIPY:
        av = 1 - np.exp(-2 * np.pi * fc / SR)
        y = np.empty_like(x); acc = 0.0
        for i in range(len(x)):
            acc += av[i] * (x[i] - acc); y[i] = acc
        return y
    # block-wise constant cutoff (avg per block) via stateful lfilter — fast
    n = len(x); blk = 256; y = np.empty(n); zi = np.zeros(1)
    for s in range(0, n, blk):
        e = min(n, s + blk)
        fcm = float(np.mean(fc[s:e])); a = 1 - np.exp(-2 * np.pi * fcm / SR)
        seg, zi = _lfilter([a], [1.0, -(1 - a)], x[s:e], zi=zi)
        y[s:e] = seg
    return y

def lp_static(x, fc, q=0.7):
    w0 = 2 * np.pi * fc / SR
    al = math.sin(w0) / (2 * q)
    b0 = (1 - math.cos(w0)) / 2; b1 = 1 - math.cos(w0); b2 = b0
    a0 = 1 + al; a1 = -2 * math.cos(w0); a2 = 1 - al
    return _iir([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0], x)

def hp_static(x, fc):
    w0 = 2 * np.pi * fc / SR
    al = math.sin(w0) / (2 * 0.707)
    b0 = (1 + math.cos(w0)) / 2; b1 = -(1 + math.cos(w0)); b2 = b0
    a0 = 1 + al; a1 = -2 * math.cos(w0); a2 = 1 - al
    return _iir([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0], x)

def noise(dur):
    return rng.standard_normal(int(round(dur * SR)))

# Big-band brass stab: detuned saw stack -> lowpass, punchy ADSR.
def brass(f, dur, amp=1.0, bright=1.0):
    n = int(round(dur * SR)); tt = np.arange(n) / SR
    y = (osc('saw', f, dur, detune=-7) + osc('saw', f, dur, detune=7)
         + osc('saw', f, dur) * 0.7 + osc('square', f, dur, detune=3) * 0.4)
    fc = f * (3 + 6 * bright) * np.exp(-tt * 9) + f * 1.6
    y = onepole_lp_var(y, fc)
    env = (1 - np.exp(-tt * 260)) * (0.35 + 0.65 * np.exp(-tt * 4.5))
    return y * env * amp * 0.25

# Warm string ensemble: stacked detuned saws + sub, slow bow, soft lowpass.
def strings(f, dur, amp=1.0):
    n = int(round(dur * SR)); tt = np.arange(n) / SR
    y = np.zeros(n)
    for det in (-9, -4, 4, 9):
        y += osc('saw', f, dur, detune=det)
    y += osc('saw', f / 2, dur) * 0.6
    y = lp_static(y, 2400, 0.6)
    bow = 1 + 0.006 * np.sin(2 * np.pi * 5.0 * tt)
    env = np.minimum(1, tt / max(1e-4, dur * 0.28)) * np.minimum(1, (dur - tt) / max(1e-4, dur * 0.3) + 0.0001)
    env = np.clip(env, 0, 1)
    return y * bow * env * amp * 0.16

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
    d = max(1, int(dt * SR))
    a = np.zeros(d + 1); a[0] = 1.0; a[-1] = -fb       # u[n]=x[n]+fb*u[n-d]
    u = _iir([1.0], a, x) if HAVE_SCIPY else _comb(x, d, fb)
    sd = np.concatenate([np.zeros(d), u[:-d]]) if d < len(u) else np.zeros_like(x)
    out = x + sd
    return x * (1 - mix) + out * mix

def reverb(x, mix=0.16, decay=0.4):
    combs = [(0.0297, 0.78), (0.0371, 0.74), (0.0411, 0.70), (0.0437, 0.68)]
    out = np.zeros(len(x))
    for dt, g in combs:
        d = max(1, int(dt * SR)); gg = min(0.92, g * (0.6 + decay))
        out += _comb(x, d, gg)
    out /= len(combs)
    # one Schroeder allpass for diffusion: H = (-g + z^-d)/(1 - g z^-d)
    d = max(1, int(0.005 * SR)); g = 0.5
    if HAVE_SCIPY:
        b = np.zeros(d + 1); b[0] = -g; b[-1] = 1.0
        a = np.zeros(d + 1); a[0] = 1.0; a[-1] = -g
        ap = _lfilter(b, a, out)
    else:
        buf = np.zeros(d); ap = np.zeros(len(out))
        for i in range(len(out)):
            j = i % d; bv = buf[j]; v = -g * out[i] + bv
            buf[j] = out[i] + g * v; ap[i] = v
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

def _pc(x):
    return x % 12

def _scalepool(scale, lo, hi, octave):
    pool = []
    o = lo
    while o <= hi:
        for s in scale:
            pool.append(s + o * 12 + octave * 12)
        o += 1
    return sorted(set(pool))

def _nearest(pool, target_pc, ref):
    cand = [p for p in pool if _pc(p) == target_pc] or pool
    return min(cand, key=lambda p: (abs(p - ref), p))

def _stepfrom(pool, ref, direction):
    i = min(range(len(pool)), key=lambda k: abs(pool[k] - ref))
    return pool[int(np.clip(i + (1 if direction >= 0 else -1), 0, len(pool) - 1))]

# Kanno-informed melodic engine: a 1-bar motif (syncopated rhythm cell + a
# rise-to-apex contour with a leap, then stepwise descent) developed by the
# "Rule of Three" — bar0 statement, bar1 tail varied & sequenced to the next
# chord, bar2 developed with a higher climactic-staccato apex, bar3 cadential
# answer that resolves a 16th early then rests. Strong beats / the apex land
# on chord tones of that bar; weak beats are scale passing/approach tones.
# chord_bars[b] = semitone voicing for bar b. Returns [(start,dur,semi)].
def melody(chord_bars, scale, bars, beats, sd, lo, hi, seed, density=0.62, octave=0):
    r = np.random.default_rng(seed)
    pool = _scalepool(scale, lo, hi, octave)
    bs = beats * 4  # 16th steps per bar
    CELLS = [
        [(0, 2), (2, 2), (4, 3), (7, 1), (8, 2), (10, 2), (12, 4)],
        [(0, 3), (3, 1), (4, 2), (6, 2), (8, 3), (11, 1), (12, 2), (14, 2)],
        [(0, 2), (2, 1), (3, 3), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)],
        [(0, 4), (4, 2), (6, 2), (8, 2), (10, 2), (12, 3), (15, 1)],
    ]
    base = CELLS[int(r.integers(len(CELLS)))]
    ref = pool[len(pool) // 2]
    notes = []
    for b in range(bars):
        chord = chord_bars[b % len(chord_bars)]
        nxt = chord_bars[(b + 1) % len(chord_bars)]
        cts = sorted(set(_pc(c) for c in chord)) or [0]
        ncts = sorted(set(_pc(c) for c in nxt)) or [0]
        stage = b % 4
        cell = [(o, d) for (o, d) in base]
        if stage == 2:  # development: rhythmic displacement (syncopate)
            cell = [((o + 1 if (k % 2 and o + 1 < bs) else o), d)
                    for k, (o, d) in enumerate(cell)]
        # keep within the bar (meter-agnostic: works for 3/4 and 4/4)
        cell = [(o, min(d, bs - o)) for (o, d) in cell if 0 <= o < bs and bs - o > 0]
        apex_k = max(1, int(round(len(cell) * 0.55)))
        for k, (o, d) in enumerate(cell):
            st = b * bs + o
            strong = (o % 4 == 0)
            if stage == 3 and k >= len(cell) - 2:
                p = _nearest(pool, cts[0], ref)            # cadence target
                d = max(1, d - 2)                          # end early -> rest
                notes.append((st, d, p)); ref = p
                break
            if k == apex_k:                                # climactic staccato
                lift = int(r.choice([4, 5, 7]))
                if stage == 2:
                    lift += 3
                p = _nearest([x for x in pool if x > ref] or pool,
                             cts[int(r.integers(len(cts)))], ref + lift)
                notes.append((st, 1, p)); ref = p          # short + trailing gap
                continue
            if strong:
                p = _nearest(pool, cts[int(r.integers(len(cts)))], ref)
            elif r.random() < 0.16:                        # occasional 3rd+ leap
                p = _nearest(pool, cts[int(r.integers(len(cts)))],
                             ref + int(r.choice([-7, -5, 5, 7])))
            else:                                          # step (resolve leaps)
                p = _stepfrom(pool, ref, 1 if ref < notes[-1][2] else -1) \
                    if notes else _stepfrom(pool, ref, int(r.choice([-1, 1])))
            notes.append((st, d, p)); ref = p
        if stage == 1 and notes:                           # sequence into next chord
            s0, d0, _ = notes[-1]
            notes[-1] = (s0, d0, _nearest(pool, ncts[0], ref))
    steps = bars * bs
    return [(s, min(dn, steps - s), p) for (s, dn, p) in notes if 0 <= s < steps and dn > 0]

# Real countermelody: contrary/oblique motion vs the lead, restricted to the
# bar's chord tones (consonant inner voice kept below the lead).
def counter(notes, chord_bars, scale, beats, bars):
    bs = beats * 4
    out = []
    prev_lead = None
    cref = None
    for (st, dn, semi) in notes:
        b = (st // bs) % len(chord_bars)
        cts = sorted(set(_pc(c) for c in chord_bars[b])) or [0]
        if cref is None:
            cref = semi - 12
        if prev_lead is None:
            tgt = semi - 12
        elif semi > prev_lead:                 # lead rises -> counter falls
            tgt = cref - 2
        elif semi < prev_lead:                 # lead falls -> counter rises
            tgt = cref + 2
        else:                                  # lead holds -> oblique (hold)
            tgt = cref
        ceiling = semi - 3                      # stay below the lead
        cands = []
        base_oct = tgt // 12
        for pc in cts:
            for o in (-2, -1, 0, 1):
                p = pc + 12 * (base_oct + o)
                if p <= ceiling:
                    cands.append(p)
        if not cands:
            cands = [semi - 12]
        c = min(cands, key=lambda p: (abs(p - tgt), -p))
        out.append((st, dn, c))
        prev_lead = semi
        cref = c
    return out

# Jazz walking bass: target each bar root, approach the next with a
# diatonic-or-chromatic leading tone on beat 4 (Kanno-flavored motion).
def walk_bass(roots, scale, bars, beats):
    sd = sorted(set(s % 12 for s in scale))
    seq = []
    for b in range(bars):
        r0 = roots[b % len(roots)]
        r1 = roots[(b + 1) % len(roots)]
        for beat in range(beats):
            if beat == 0:
                n = r0
            elif beat == beats - 1:
                n = r1 - 1 if (r1 - r0) > 0 else r1 + 1   # chromatic approach
            else:
                step = sorted(sd)[(beat * 2) % len(sd)]
                n = r0 + step + (0 if beat < beats / 2 else -12 + 12)
            seq.append((b, beat, n))
    return seq

STORES = {}

# ---------- SAIZERIYA: jazzy cafe / bossa ----------
def build_saizeriya(rush):
    # SAIZERIYA: bittersweet cinematic waltz in 3/4 (full reimagining).
    bpm = 178 if rush else 150
    beats = 3                                   # 3/4 waltz
    bars = 4
    spb = 60.0 / bpm
    sd = spb / 4.0
    loop = bars * beats * spb
    tail = 2.3
    buf = np.zeros(int((loop + tail) * SR))
    drum = np.zeros_like(buf)
    global _TP
    _TP = 2 if rush else 0                      # RUSH: whole-step lift
    # Am9 – Fmaj7(9) – Dm7 – E7(b9)  (i – VI – iv – V7b9): bittersweet Kanno
    voic = [
        [-15, 0, 4, 7, 11],     # Am9   (A C E G B)
        [-19, 0, 5, 9, 12],     # Fmaj7 (F C F A E)
        [-22, 2, 5, 9, 12],     # Dm7   (D D F A C)
        [-20, 4, 8, 11, 13],    # E7b9  (E G# D F)
    ]
    scale = [0, 2, 4, 5, 7, 8, 9, 11]           # A-minor + G# (harmonic color)

    for b in range(bars):
        bt = b * beats * spb
        ch = voic[b % 4]
        root = ch[0]
        # lush string-ensemble pad holding the chord
        for semi in ch[1:]:
            add(buf, strings(hz(semi), beats * spb * 0.98, amp=0.5), bt + 0.01)
        # waltz "oom-pah-pah": bass on beat1, Rhodes chord on beats 2 & 3
        bs = osc('triangle', hz(root), spb * 0.9) * 0.8 + osc('sine', hz(root - 12), spb * 0.9) * 0.5
        bs = lp_static(bs, 700)
        bs *= adsr(len(bs), 0.006, 0.14, 0.4, 0.12, 0.45)
        add(buf, bs * 0.42, bt)
        for beat in (1, 2):
            for k, semi in enumerate(ch[1:4]):
                add(buf, epiano(hz(semi), spb * 0.8, amp=0.07 * (0.9 if k else 1.0)),
                    bt + beat * spb)
        # passing bass note leading into the next bar (beat 3)
        nxt = voic[(b + 1) % 4][0]
        pb = osc('triangle', hz(root + (1 if nxt > root else -1) + (0 if b % 2 else -0)), spb * 0.6)
        pb = lp_static(pb, 700) * adsr(len(pb), 0.006, 0.1, 0.3, 0.08, 0.4)
        add(buf, pb * 0.22, bt + 2 * spb)
    # expressive lead — wide leaps, climactic-staccato, early resolution
    mel = melody(voic, scale, bars, beats, sd, 0, 2, 41 + rush,
                 density=0.5 if not rush else 0.6,
                 octave=1 if rush else 0)
    for (st, dn, semi) in mel:
        at = st * sd
        dur = dn * sd * 0.96
        sig = osc('triangle', hz(semi), dur) * 0.55 + osc('sine', hz(semi), dur) * 0.45
        tt = np.arange(len(sig)) / SR
        vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * tt) * np.minimum(1, tt * 4)
        sig = sig * vib
        sig *= adsr(len(sig), 0.03, 0.1, 0.72, 0.22, 0.7)
        add(buf, sig * 0.135, at)
    # cello-ish contrary counter line
    for (st, dn, semi) in counter(mel, voic, scale, beats, bars):
        at = st * sd
        dur = dn * sd * 0.95
        cs = osc('saw', hz(semi - 12), dur) * 0.45 + osc('sine', hz(semi - 12), dur) * 0.55
        cs = lp_static(cs, 1300, 0.8)
        cs *= adsr(len(cs), 0.04, 0.12, 0.6, 0.2, 0.6)
        add(buf, cs * 0.06, at)
    # intimate waltz brushes: soft kick on 1, brush taps on 2 & 3
    for b in range(bars):
        bt = b * beats * spb
        add(drum, kick(0.34, 105, 44, amp=0.4, click=0.08), bt)
        for beat in (1, 2):
            add(drum, snare(0.16, amp=0.16, soft=True), bt + beat * spb)
            add(drum, shaker(0.34), bt + beat * spb)
        if rush:
            add(drum, shaker(0.3), bt + 0.5 * spb)
            add(drum, shaker(0.3), bt + 1.5 * spb)
    sig = buf + drum
    sig = delay(sig, sd * 3, 0.26, 0.16)
    sig = reverb(sig, mix=0.26, decay=0.6)       # cinematic space
    return sig, loop, tail


# ---------- OHSHO: lively town-Chinese theme "餃子日和" (user-provided) ----------
def build_ohsho(rush):
    # Faithful port of the user's Chinese-pentatonic piece: koto/pipa-ish
    # pluck melody (2 phrases x 2 = 8 bars), root bass, woodblock 8ths, gong.
    bpm = 158 if rush else 132
    beats = 4
    bars = 8
    spb = 60.0 / bpm
    sd = spb / 4.0                               # 16th-note seconds
    loop = bars * beats * spb
    tail = 1.6
    buf = np.zeros(int((loop + tail) * SR))
    drum = np.zeros_like(buf)
    global _TP
    _TP = 2 if rush else 0                        # RUSH: whole-step lift

    NOTE = {'C4': 0, 'D4': 2, 'E4': 4, 'G4': 7, 'A4': 9,
            'C5': 12, 'D5': 14, 'E5': 16, 'G5': 19, 'A5': 21, 'C6': 24}
    BASS = {'C3': -12, 'A3': -3, 'F3': -7, 'G3': -5}
    phraseA = [('G5', 2), ('A5', 2), ('G5', 1), ('E5', 1), ('D5', 2),
               ('E5', 2), ('G5', 2), ('A5', 2), ('C6', 2),
               ('A5', 1), ('G5', 1), ('E5', 2), ('D5', 2), ('C5', 2),
               ('D5', 2), ('E5', 2), ('G5', 4)]
    phraseB = [('E5', 2), ('D5', 1), ('C5', 1), ('D5', 2), ('E5', 2),
               ('G5', 2), ('A5', 2), ('G5', 2), ('E5', 2),
               ('D5', 1), ('E5', 1), ('G5', 2), ('A5', 2), ('G5', 2),
               ('E5', 2), ('D5', 2), ('C5', 4)]
    seq = phraseA + phraseB                       # 64 sixteenths = 4 bars
    bass_bars = ['C3', 'C3', 'A3', 'A3', 'F3', 'F3', 'G3', 'G3']  # 8 bars

    def koto(semi, dur, amp):
        n = int(round(dur * SR)); tt = np.arange(n) / SR
        f = hz(semi)
        y = osc('triangle', f, dur) + osc('sine', f * 2.01, dur) * 0.3
        if semi >= 12:                            # octave-down body for highs
            y += osc('sine', f / 2, dur) * 0.36
        env = np.exp(-tt * 3.4) * (1 - np.exp(-tt * 260))
        return y * env * amp

    def woodblock(at, accent):
        f0 = 1400 if accent else 1000
        n = int(round(0.09 * SR)); tt = np.arange(n) / SR
        y = osc('square', f0, 0.09) * np.exp(-tt * 60)
        y += osc('square', f0 * 0.6, 0.09) * np.exp(-tt * 90) * 0.5
        add(drum, y * (0.13 if accent else 0.07), at)

    def gong(at):
        nz = noise(1.2)
        ei = np.arange(len(nz)) / len(nz)
        nz = nz * np.power(np.clip(1 - ei, 0, 1), 1.5)
        nz = hp_static(lp_static(nz, 620), 240)   # ~bandpass around the gong
        add(drum, nz * 0.12, at)

    # melody (two passes of the 4-bar phrase set -> 8-bar loop)
    for rep in (0, 1):
        pos = rep * 64
        for note, ln in seq:
            at = pos * sd
            dur = ln * sd * 0.95
            semi = NOTE[note]
            add(buf, koto(semi, dur, 0.42), at)
            pos += ln
    # root bass on beat 1 & 3 of each bar
    for b in range(bars):
        bt = b * beats * spb
        semi = BASS[bass_bars[b % 8]]
        for off in (0, 2 * spb):                  # beats 1 and 3
            n = int(round(spb * 1.75 * SR)); tt = np.arange(n) / SR
            bs = osc('sine', hz(semi), spb * 1.75) * (1 - np.exp(-tt * 130)) * np.exp(-tt * 2.0)
            bs += osc('triangle', hz(semi), spb * 1.75) * 0.25 * np.exp(-tt * 3.0)
            add(buf, bs * 0.32, bt + off)
    # percussion: woodblock every 8th (accent on the beat), gong at bars 0 & 4
    total = bars * beats * 4
    for s in range(0, total, 2):
        woodblock(s * sd, (s % 8 == 0))
    for b in range(bars):
        if b % 4 == 0:
            gong(b * beats * spb)
    sig = buf + drum
    sig = delay(sig, sd * 3, 0.16, 0.1)
    sig = reverb(sig, mix=0.14, decay=0.34)       # lively diner room
    return sig, loop, tail


# ---------- COCOICHI: hypnotic ethnic / phrygian ----------
def build_cocoichi(rush):
    # hypnotic phrygian "exotic curry house": a pulsing tonic pedal, a santur
    # ostinato over a bII->I "Spanish" cadence, a buzzy saz lead via the Kanno
    # motif engine + a consonant counter voice, and a darbuka groove. Lead
    # gets its own delayed bus; reverb send is high-passed to keep it tight.
    bpm = 140 if rush else 100
    beats = 4
    bars = 4
    spb = 60.0 / bpm
    sd = spb / 4.0
    loop = bars * beats * spb
    tail = 1.8
    buf = np.zeros(int((loop + tail) * SR))
    lead = np.zeros_like(buf)                    # lead routed through a delay
    drum = np.zeros_like(buf)
    global _TP
    _TP = 2 if rush else 0                       # RUSH: whole-step chorus lift
    # C phrygian: C Db Eb F G Ab Bb -> 0 1 3 5 7 8 10
    scale = [0, 1, 3, 5, 7, 8, 10]
    # i  i  bII(Db)  i  — the iconic phrygian cadence at the loop seam
    Cm = [-12, 0, 3, 7, 10]                      # Cm7
    Db = [-11, 1, 5, 8, 13]                      # Dbmaj (bII colour)
    chord_bars = [Cm, Cm, Db, Cm]

    # --- pulsing tonic pedal (lighter than a static saw wall) ---
    pedlen = int(loop * SR)
    tt = np.arange(pedlen) / SR
    ped = (osc('sine', hz(-24), loop)[:pedlen] * 0.6
           + osc('sine', hz(-12), loop)[:pedlen] * 0.25
           + osc('saw', hz(-12), loop)[:pedlen] * 0.18)
    fc = 360 + 300 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * tt / loop))   # 2 cyc
    ped = onepole_lp_var(ped, fc)
    trem = 1.0 - 0.30 * (0.5 + 0.5 * np.sin(2 * np.pi * (bars * beats) * tt / loop))
    add(buf, ped * trem * 0.15, 0)

    # --- santur-like ostinato + soft colour pad per bar ---
    arp = [0, 1, 2, 3, 2, 3, 1, 2]
    for b in range(bars):
        bt = b * beats * spb
        ch = chord_bars[b % 4]
        tones = [s for s in ch if s >= 0] or ch
        for j in range(8):
            semi = tones[arp[j] % len(tones)]
            acc = 0.072 if j % 4 == 0 else 0.05
            add(buf, pluck(hz(semi), 0.46, amp=acc, wave='triangle', bright=0.78),
                bt + j * 2 * sd)
        ps = np.zeros(int(beats * spb * SR))
        for semi in tones:
            mix_into(ps, osc('triangle', hz(semi + 12), beats * spb), 0.3)
        ps *= adsr(len(ps), 0.35, 0.3, 0.75, 0.5, 0.75)
        add(buf, lp_static(ps, 1900) * 0.03, bt)

    # --- buzzy saz lead via the Kanno motif engine (phrygian) ---
    mel = melody(chord_bars, scale, bars, beats, sd, 0, 2, 35 + rush,
                 density=0.52 if not rush else 0.62, octave=1)
    for (st, dn, semi) in mel:
        dur = dn * sd * 0.96
        n = int(round(dur * SR)); ttl = np.arange(n) / SR
        sig = osc('saw', hz(semi), dur) * 0.7 + osc('saw', hz(semi), dur, detune=7) * 0.3
        vib = 1 + 0.011 * np.sin(2 * np.pi * 5.6 * ttl) * np.minimum(1, ttl * 7)
        sig = sig[:n] * vib
        sig = np.tanh(sig * 1.7) / 1.25
        sig = lp_static(sig, 3000, 1.1)
        sig *= adsr(n, 0.012, 0.09, 0.6, max(0.05, dur * 0.22), 0.6)
        add(lead, sig * 0.1, st * sd)
    # consonant oud-ish counter voice (inner line below the lead)
    for (st, dn, semi) in counter(mel, chord_bars, scale, beats, bars):
        dur = dn * sd * 0.92
        cs = osc('saw', hz(semi), dur) * 0.55 + osc('sine', hz(semi), dur) * 0.35
        cs = lp_static(cs, 2000, 1.1)
        cs *= adsr(len(cs), 0.02, 0.12, 0.5, 0.2, 0.5)
        add(buf, cs * 0.045, st * sd)

    # --- darbuka groove (dum / tek / ka) + frame drum + shaker ---
    def tek(amp):
        d = 0.07; n = int(d * SR); te = np.arange(n) / SR
        y = hp_static(noise(d), 4200) * np.exp(-te * 75)
        y += np.sin(2 * np.pi * 920 * te) * np.exp(-te * 70) * 0.3
        return y * amp
    def ka(amp):
        d = 0.045; n = int(d * SR); te = np.arange(n) / SR
        return hp_static(noise(d), 6200) * np.exp(-te * 120) * amp
    for b in range(bars):
        bt = b * beats * spb
        if rush:
            dpat, tpat, kpat = [0, 4, 6, 8, 12, 14], [2, 10, 13], [3, 7, 11, 15]
        else:
            dpat, tpat, kpat = [0, 3, 8, 11], [4, 12, 14], [2, 6, 7, 10, 13]
        for st in dpat:
            add(drum, tabla(96, 0.30, amp=0.5), bt + st * sd)
        for st in tpat:
            add(drum, tek(0.34), bt + st * sd)
        for st in kpat:
            add(drum, ka(0.22), bt + st * sd)
        for st in range(0, 16, 2):
            add(drum, shaker(0.5 if st % 4 == 0 else 0.42), bt + st * sd)
        add(drum, kick(0.42, 104, 52, amp=0.5, click=0.08), bt)   # frame drum
        if rush:
            add(drum, clap(0.3), bt + 4 * sd)
            add(drum, clap(0.3), bt + 12 * sd)
        if b == bars - 1:                                          # turnaround fill
            for st in (12, 13, 14, 15):
                add(drum, tek(0.26) if st % 2 else ka(0.22), bt + st * sd + sd * 0.5)

    lead = delay(lead, sd * (3 if rush else 4), 0.34, 0.30)
    sig = buf + lead + drum
    wet = reverb(hp_static(sig, 220), mix=1.0, decay=0.5)          # tight, hp'd
    sig = sig + wet * (0.15 if not rush else 0.12)
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

def _seed_for(store, rush):
    off = {'saizeriya': 10, 'ohsho': 20, 'cocoichi': 30}[store]
    return 20260517 + off + (1 if rush else 0)

def _render_task(args):
    store, rush = args
    global rng
    rng = np.random.default_rng(_seed_for(store, rush))   # per-track determinism
    sig, loop, tail = BUILDERS[store](rush)
    sig = soft_limit(sig, 1.05)
    sig = normalize(sig, 0.9)
    br = 80 if store != 'cocoichi' else 88
    mp3 = encode_mp3(sig, br)
    b64 = base64.b64encode(mp3).decode('ascii')
    key = store + ('R' if rush else 'N')
    stats = dict(key=key, dur=len(sig) / SR, loop=loop,
                 mp3=len(mp3) / 1024, b64=len(b64) / 1024,
                 peak=float(np.max(np.abs(sig))),
                 rms=float(np.sqrt(np.mean(sig ** 2))),
                 head=float(np.max(np.abs(sig[:int(0.01 * SR)]))))
    return key, {'b64': b64, 'loop': round(loop, 4)}, stats

def main():
    tasks = [(s, r) for s in BUILDERS for r in (False, True)]
    try:
        import multiprocessing as mp
        with mp.Pool(processes=min(len(tasks), mp.cpu_count() or 1)) as pool:
            results = pool.map(_render_task, tasks)
    except Exception as e:
        print(f"(parallel render unavailable: {e}; running sequentially)")
        results = [_render_task(t) for t in tasks]
    out = {}
    total = 0
    for key, rec, st in results:
        out[key] = rec
        total += len(rec['b64'])
        print(f"{st['key']:14s} dur={st['dur']:5.2f}s loop={st['loop']:5.2f}s "
              f"mp3={st['mp3']:6.1f}KB b64={st['b64']:6.1f}KB "
              f"peak={st['peak']:.3f} rms={st['rms']:.3f} head10ms={st['head']:.3f}")
    print(f"--- total base64 ~ {total/1024:.1f} KB ---")
    with open('/tmp/bgm_assets.json', 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':
    main()
