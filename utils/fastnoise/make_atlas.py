# packs EA FAST 128x128x32 textures into the 256x4096 RGB atlas layout fastNoise() reads
import sys, numpy as np
from PIL import Image
root, kind, out = sys.argv[1:4]
load = lambda name: np.array([np.asarray(Image.open(f'{root}/{name}_{kind}_{t}.png')) for t in range(32)])
v2 = load('noise/vector2/temporal/exp/vector2_uniform_gauss1_0_exp0101')
re = load('noise/real/temporal/exp/real_uniform_gauss1_0_exp0101')[..., 0]
left = np.stack([v2[..., 0], v2[..., 1], re], -1)
w = np.roll(re, (83, 49), axis=(1, 2))
right = np.stack([w, w, w], -1)
atlas = np.concatenate([left, right], 2).reshape(4096, 256, 3).astype(np.uint8)
Image.fromarray(atlas).save(out, optimize=True)
