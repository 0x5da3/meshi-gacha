#!/usr/bin/env python3
# OG image generator for 飯ガチャスロ.
# Outputs ogp.png (1200x630) — deep navy x gold theme.
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, math

OUT = os.path.join(os.path.dirname(__file__), '..', 'ogp.png')
W, H = 1200, 630

NAVY_TOP   = (12, 22, 58)
NAVY_BOT   = (4, 8, 28)
GOLD       = (255, 206, 84)
GOLD_DEEP  = (196, 138, 18)
CREAM      = (255, 244, 214)
RED        = (210, 30, 56)
BLUE_NEON  = (102, 178, 255)

JP_FONT = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
JP_FONT_P = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'
EN_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def f(p, sz):
    return ImageFont.truetype(p, sz)

def gradient_bg():
    img = Image.new('RGB', (W, H), NAVY_BOT)
    px = img.load()
    for y in range(H):
        t = y / H
        # radial-ish vertical gradient
        r = int(NAVY_TOP[0]*(1-t) + NAVY_BOT[0]*t)
        g = int(NAVY_TOP[1]*(1-t) + NAVY_BOT[1]*t)
        b = int(NAVY_TOP[2]*(1-t) + NAVY_BOT[2]*t)
        for x in range(W):
            px[x, y] = (r, g, b)
    return img

def add_radial_glow(img, cx, cy, radius, color, alpha=120):
    glow = Image.new('RGBA', (W, H), (0,0,0,0))
    gd = ImageDraw.Draw(glow)
    steps = 28
    for i in range(steps, 0, -1):
        a = int(alpha * (i/steps)**2)
        rr = int(radius * (i/steps))
        gd.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], fill=color+(a,))
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    img.alpha_composite(glow)

def add_confetti(img):
    d = ImageDraw.Draw(img, 'RGBA')
    import random
    random.seed(7)
    palette = [GOLD, CREAM, BLUE_NEON, RED]
    for _ in range(110):
        x = random.randint(0, W); y = random.randint(0, H)
        s = random.randint(2, 6)
        c = random.choice(palette)
        a = random.randint(60, 180)
        d.ellipse([x, y, x+s, y+s], fill=c+(a,))

def draw_text_layered(d, xy, text, font, fill, stroke=None, stroke_w=0, shadow=None, shadow_off=(0,0)):
    x, y = xy
    if shadow:
        d.text((x+shadow_off[0], y+shadow_off[1]), text, font=font, fill=shadow)
    if stroke and stroke_w:
        d.text(xy, text, font=font, fill=fill, stroke_fill=stroke, stroke_width=stroke_w)
    else:
        d.text(xy, text, font=font, fill=fill)

def text_size(d, text, font):
    bb = d.textbbox((0,0), text, font=font)
    return bb[2]-bb[0], bb[3]-bb[1], bb

def draw_gold_text(img, xy, text, font, stroke_color=(60,30,0), stroke_w=6):
    """Render text in gold gradient with deep stroke + drop shadow."""
    d = ImageDraw.Draw(img)
    w, h, bb = text_size(d, text, font)
    pad = stroke_w + 20
    tile = Image.new('RGBA', (w + pad*2, h + pad*2), (0,0,0,0))
    td = ImageDraw.Draw(tile)
    # shadow
    shadow = Image.new('RGBA', tile.size, (0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.text((pad - bb[0], pad - bb[1]), text, font=font, fill=(0,0,0,180),
            stroke_fill=(0,0,0,200), stroke_width=stroke_w)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    tile.alpha_composite(shadow, (6, 8))
    # base text (white) with stroke
    td.text((pad - bb[0], pad - bb[1]), text, font=font, fill=(255,255,255,255),
            stroke_fill=stroke_color+(255,), stroke_width=stroke_w)
    # gold gradient overlay
    grad = Image.new('RGBA', tile.size, (0,0,0,0))
    gpx = grad.load()
    for yy in range(tile.size[1]):
        t = yy / tile.size[1]
        # gold highlight midway
        if t < 0.5:
            k = t/0.5
            r = int(255*(1-k) + 255*k)
            g = int(228*(1-k) + 196*k)
            b = int(140*(1-k) + 64*k)
        else:
            k = (t-0.5)/0.5
            r = int(255*(1-k) + 184*k)
            g = int(196*(1-k) + 120*k)
            b = int(64*(1-k) + 18*k)
        for xx in range(tile.size[0]):
            gpx[xx, yy] = (r, g, b, 255)
    # mask = white text only (not stroke). Re-render alpha for inner text.
    mask = Image.new('L', tile.size, 0)
    md = ImageDraw.Draw(mask)
    md.text((pad - bb[0], pad - bb[1]), text, font=font, fill=255)
    grad.putalpha(mask)
    tile.alpha_composite(grad)
    # highlight stripe
    hi = Image.new('RGBA', tile.size, (0,0,0,0))
    hd = ImageDraw.Draw(hi)
    hd.rectangle([0, int(tile.size[1]*0.18), tile.size[0], int(tile.size[1]*0.36)],
                 fill=(255,255,255,90))
    himask = mask.point(lambda v: 255 if v>0 else 0)
    hi.putalpha(himask)
    hi_combined = Image.new('RGBA', tile.size, (0,0,0,0))
    hi_combined.paste(hi, (0,0))
    # apply highlight only over text
    only_hi = Image.new('RGBA', tile.size, (0,0,0,0))
    only_hi.paste((255,255,255,110), (0, int(tile.size[1]*0.20), tile.size[0], int(tile.size[1]*0.34)))
    only_hi.putalpha(mask.point(lambda v: int(v*0.55)))
    tile.alpha_composite(only_hi)

    img.alpha_composite(tile, (xy[0]-pad, xy[1]-pad))
    return w, h

def draw_capsule(img, cx, cy, r):
    d = ImageDraw.Draw(img, 'RGBA')
    # top red
    d.pieslice([cx-r, cy-r, cx+r, cy+r], 180, 360, fill=(225,60,80,255))
    # bottom cream
    d.pieslice([cx-r, cy-r, cx+r, cy+r],   0, 180, fill=(255,240,210,255))
    # seam
    d.rectangle([cx-r, cy-3, cx+r, cy+3], fill=(40,20,30,200))
    # highlight
    d.ellipse([cx-r*0.55, cy-r*0.85, cx-r*0.15, cy-r*0.35], fill=(255,255,255,180))
    # gold ring
    d.ellipse([cx-r-6, cy-r-6, cx+r+6, cy+r+6], outline=GOLD, width=5)

def draw_reels(img, cx, cy):
    d = ImageDraw.Draw(img, 'RGBA')
    rw, rh, gap = 88, 110, 12
    total = rw*3 + gap*2
    x0 = cx - total//2
    faces = ['🍒','7','★']  # emoji likely won't render; use chars
    chars = ['C', '7', '★']
    for i in range(3):
        x = x0 + i*(rw+gap)
        # outer
        d.rounded_rectangle([x, cy-rh//2, x+rw, cy+rh//2], radius=14,
                            fill=(255,250,235,255), outline=GOLD_DEEP, width=4)
        # inner shade
        d.rounded_rectangle([x+6, cy-rh//2+6, x+rw-6, cy+rh//2-6], radius=10,
                            fill=(255,255,255,255))
        # symbol
        fnt = f(EN_FONT, 64)
        ch = chars[i]
        bb = d.textbbox((0,0), ch, font=fnt)
        tw = bb[2]-bb[0]; th = bb[3]-bb[1]
        color = (200,30,50,255) if ch=='7' else (180,130,20,255)
        d.text((x + (rw-tw)//2 - bb[0], cy - th//2 - bb[1]), ch, font=fnt, fill=color)

def main():
    base = gradient_bg().convert('RGBA')

    # background glows
    add_radial_glow(base, 300, 180, 360, GOLD, alpha=80)
    add_radial_glow(base, 980, 480, 420, BLUE_NEON, alpha=70)

    # diagonal stripes overlay
    stripes = Image.new('RGBA', (W,H), (0,0,0,0))
    sd = ImageDraw.Draw(stripes)
    for i in range(-H, W, 36):
        sd.polygon([(i,0),(i+12,0),(i+12+H,H),(i+H,H)], fill=(255,255,255,8))
    base.alpha_composite(stripes)

    add_confetti(base)

    # gold border
    d = ImageDraw.Draw(base)
    d.rounded_rectangle([18,18,W-18,H-18], radius=28, outline=GOLD, width=4)
    d.rounded_rectangle([28,28,W-28,H-28], radius=22, outline=(60,40,10,255), width=1)

    # tag pill (top)
    pill_font = f(JP_FONT, 22)
    tag = 'パチンコ風 飯選びガチャ'
    tw, th, _ = text_size(d, tag, pill_font)
    px, py = (W - tw)//2, 70
    d.rounded_rectangle([px-22, py-12, px+tw+22, py+th+14], radius=999,
                        fill=(210,30,56,235), outline=GOLD, width=2)
    d.text((px, py-4), tag, font=pill_font, fill=(255,250,230,255))

    # main logo (centered)
    title = '飯ガチャスロ'
    # try a few sizes to fit
    for sz in (170, 160, 150, 140, 130):
        fnt = f(JP_FONT, sz)
        bb = d.textbbox((0,0), title, font=fnt)
        tw = bb[2]-bb[0]
        if tw < W - 180:
            break
    tx = (W - tw)//2
    ty = 175
    draw_gold_text(base, (tx, ty), title, fnt, stroke_color=(50,28,4), stroke_w=8)

    # subtitle (English)
    subtitle = 'MESHI-GACHA  ::  GACHA × SLOT'
    sf = f(EN_FONT, 30)
    bb = d.textbbox((0,0), subtitle, font=sf)
    sw = bb[2]-bb[0]
    d.text(((W-sw)//2 + 2, 380 + 2), subtitle, font=sf, fill=(0,0,0,180))
    d.text(((W-sw)//2,     380),     subtitle, font=sf, fill=GOLD)

    # decorative icons
    draw_capsule(base, 175, 470, 78)
    draw_reels(base, 1010, 470)

    # tagline bottom
    tag2 = '予算もメニューもガチャで決まる、外食シミュレーター。'
    tf = f(JP_FONT, 26)
    bb = d.textbbox((0,0), tag2, font=tf)
    ww = bb[2]-bb[0]
    d.text(((W-ww)//2 + 2, 560+2), tag2, font=tf, fill=(0,0,0,170))
    d.text(((W-ww)//2,     560),   tag2, font=tf, fill=(220,232,255,255))

    base.convert('RGB').save(OUT, 'PNG', optimize=True)
    print('wrote', OUT)

if __name__ == '__main__':
    main()
