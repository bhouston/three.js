"""Independent recurrence check for the optimized all-phase EMA evaluation."""
import numpy as np

rng = np.random.default_rng(931)
values = rng.random((159, 3, 5))
truth = rng.random((3, 5))
phases = np.arange(32)
h = np.repeat(values[:32].mean(0)[None], 32, axis=0)
direct = []
for n in range(128):
    h += .1*(values[n+phases]-h)
    if n >= 96:
        direct.append(np.mean((h-truth)**2))
history = np.empty_like(values)
h = np.zeros((3, 5))
for n in range(159):
    h = .9*h+.1*values[n]
    history[n] = h
before = np.concatenate((np.zeros((1, 3, 5)), history[:31]))
correction = values[:32].mean(0)[None]-before
optimized = [np.mean((history[n+phases]+.9**(n+1)*correction-truth)**2) for n in range(96, 128)]
np.testing.assert_allclose(direct, optimized, rtol=0, atol=1e-15)
print('All-phase EMA prefix correction agrees with 32 independent recurrences.')
