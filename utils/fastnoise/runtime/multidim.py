"""Two-dimensional Houston Stratified Noise (HSN): a shared temporal step vs. per-dimension R2/R4 steps.

Bit-exact uint32 port of procedural.js. Scores 14 known 2D integrals (step edges, including the exact
diagonals, plus smooth functions) per pixel over time, after a sigma-1 blur, and under an alpha .1 EMA.
Usage: python utils/fastnoise/runtime/multidim.py
"""
import numpy as np

N = 64  # 64x64 pixels = 64 aligned 8x8 tiles
U32 = np.uint32
y, x = np.mgrid[0:N, 0:N].astype(U32)


def hash_u(v):
	v = v.astype(U32)
	v = (v ^ (v >> U32(16))) * U32(0x21f0aaad)
	v = (v ^ (v >> U32(15))) * U32(0x735a2d97)
	return v ^ (v >> U32(15))


def pixel_hash(px, py, seed):
	return hash_u((px * U32(0x9e3779b1)) ^ (py * U32(0x85ebca77)) ^ U32(seed))


def hsn_spatial(seed):
	rank = np.zeros((N, N), U32)
	for level in range(3):
		scr = pixel_hash(x >> U32(level + 1), y >> U32(level + 1), (seed ^ ((level + 1) * 0x632be5ab)) & 0xffffffff)
		a = (x >> U32(level)) & U32(1)
		b = (y >> U32(level)) & U32(1)
		low = np.where((scr & U32(4)) != 0, a, b)
		rank |= ((((a ^ b) << U32(1)) | low) ^ (scr & U32(3))) << U32(30 - 2 * level)
	return rank + pixel_hash(x >> U32(3), y >> U32(3), seed ^ 0x02e5be93)


def unit(bits):
	return (bits >> U32(8)).astype(np.float64) / 16777216.0


GOLDEN = 2654435769
R2_STEPS = [3242174889, 2447445413]
R4_STEPS = [3679390609, 3152041523, 2700274805, 2313257605]


def hsn_pair(steps, seeds=(1, 2)):
	spatial = [hsn_spatial(s) for s in seeds]
	return lambda i: tuple(unit(spatial[k] + U32((i * steps[k]) & 0xffffffff)) for k in range(2))


def ign(px, py):
	return np.mod(52.9829189 * np.mod(px * 0.06711056 + py * 0.00583715, 1.0), 1.0)


def ign_offset_pair(i):  # second dimension = IGN at a fixed pixel offset, as three.js VXGI did
	px, py = x + 0.5 + i * 5.588238, y + 0.5 + i * 5.588238
	return ign(px, py), ign(px + 5.588238, py + 3.14159)


rng = np.random.default_rng(3)
METHODS = {
	'White': lambda i: (rng.random((N, N)), rng.random((N, N))),
	'Animated IGN, offset second dimension': ign_offset_pair,
	'HSN, shared golden step': hsn_pair([GOLDEN, GOLDEN]),
	'HSN, R2 steps': hsn_pair(R2_STEPS),
}
for a, b in [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]:
	METHODS[f'HSN, R4 steps (dimensions {a}, {b})'] = hsn_pair([R4_STEPS[a], R4_STEPS[b]])

edges = [(1, -1, 0.0), (1, 1, 1.0), (1, -1, 0.3)] + [(np.cos(t), np.sin(t), c) for t, c in zip(rng.uniform(0, np.pi, 9), rng.uniform(-0.3, 1.0, 9))]
TESTS = [(lambda u, v, e=e: (e[0] * u + e[1] * v < e[2]).astype(float)) for e in edges]
TESTS += [lambda u, v: np.exp(-((u - 0.3) ** 2 + (v - 0.6) ** 2) / 0.05), lambda u, v: np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v) * 0.5 + 0.5]
g = (np.arange(1024) + 0.5) / 1024
GU, GV = np.meshgrid(g, g)
TRUTH = [f(GU, GV).mean() for f in TESTS]

kernel = np.exp(-0.5 * np.arange(-3, 4) ** 2)
kernel /= kernel.sum()


def blur(a):
	a = sum(w * np.roll(a, s, 0) for w, s in zip(kernel, range(-3, 4)))
	return sum(w * np.roll(a, s, 1) for w, s in zip(kernel, range(-3, 4)))


def rmse(values):
	return np.mean([np.sqrt(np.mean((v - t) ** 2)) for v, t in zip(values, TRUTH)]) * 1e3


SAMPLES = [4, 16, 64, 256]
print('| Method | ' + ' | '.join(f'Per pixel, {n}' for n in SAMPLES) + ' | Blurred, 1 frame | EMA, blurred | EMA frame delta |')
print('| --- |' + ' ---: |' * (len(SAMPLES) + 3))
for name, generate in METHODS.items():
	acc = [np.zeros((N, N)) for _ in TESTS]
	ema = [None] * len(TESTS)
	per_pixel, ema_err, delta, previous, first = {}, [], [], None, None
	for i in range(256):
		u, v = generate(i)
		current = []
		for t, f in enumerate(TESTS):
			value = f(u, v)
			acc[t] += value
			ema[t] = value if ema[t] is None else ema[t] + 0.1 * (value - ema[t])
			current.append(ema[t].copy())
		if i == 0:
			first = rmse([blur(a) for a in acc])
		if i + 1 in SAMPLES:
			per_pixel[i + 1] = rmse([a / (i + 1) for a in acc])
		if i >= 128:
			ema_err.append(rmse([blur(c) for c in current]))
			delta.append(np.mean([np.sqrt(np.mean((c - p) ** 2)) for c, p in zip(current, previous)]) * 1e3)
		previous = current
	print(f'| {name} | ' + ' | '.join(f'{per_pixel[n]:.2f}' for n in SAMPLES) + f' | {first:.2f} | {np.mean(ema_err):.2f} | {np.mean(delta):.2f} |')
