"""Plot retained candidates and independently cross-check original endpoints."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'ign-phase-results.json').read_text())
original = json.loads((HERE.parent / 'results.json').read_text())
for method in ('3LHW', 'Animated IGN'):
    rows = {r['frames']: r for r in data['screen'][method]}
    for i, n in enumerate(original['frames']):
        for metric in ('raw', 'blur'):
            assert math.isclose(rows[n][metric], original[metric][method][i], abs_tol=1e-14)

names = ['3LHW', 'Animated IGN', 'IGN phase + Weyl', 'IGN block 16 + Weyl']
colors = ['#c44e52', '#0072b2', '#009e73', '#9b59b6']
im = Image.new('RGB', (1400, 840), 'white')
d = ImageDraw.Draw(im)
d.text((35, 20), 'Short-history improvement attempts: spatially filtered running-average error (lower is better)', fill='black', font_size=24)
for index, (title, screen) in enumerate([('Seed 1, start 0', data['screen']), ('Held out: seed 7, start 113', data['holdout']['seed7-phase113'])]):
    left, top, width, height = 95 + index * 680, 110, 570, 580
    d.text((left, top-40), title, fill='black', font_size=22)
    for exponent in range(-4, 0):
        y = top + height * (-1-exponent)/3
        d.line((left,y,left+width,y), fill='#dddddd')
        d.text((left-60,y-8), f'1e{exponent}', fill='black', font_size=16)
    for n in (1,4,16,64,256,1024):
        x = left + width * math.log2(n)/10
        d.line((x,top,x,top+height), fill='#eeeeee')
        d.text((x-10,top+height+10), str(n), fill='black', font_size=16)
    for name, color in zip(names, colors):
        points = [(left+width*math.log2(r['frames'])/10, top+height*(-1-math.log10(r['blur']))/3) for r in screen[name]]
        d.line(points, fill=color, width=3)
        for x,y in points: d.ellipse((x-3,y-3,x+3,y+3), fill=color)
    d.text((left+200,top+height+40), 'Integrated frames (log scale)', fill='black', font_size=18)
for i, (name,color) in enumerate(zip(names,colors)):
    x = 40 + i*345
    d.line((x,780,x+25,780),fill=color,width=4)
    d.text((x+32,770),name,fill='black',font_size=18)
d.text((35,815), '128x128, wrapped Gaussian sigma=1. CPU references; no claim of universal or rendering-quality dominance.', fill='black', font_size=16)
im.save(HERE / 'improvement-comparison.png')
print('Original endpoints match; wrote improvement-comparison.png')
