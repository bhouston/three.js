# Spatial cloud transmission for the physical sky example

The original point sample changes the direct sun intensity for the entire scene.
The example now offers a sun-aligned, floating-point transmission map. It remains
an ordinary parallel DirectionalLight (or SunLight/CSM), not a RectAreaLight.
Existing WebGPURenderer light.colorNode support multiplies only this light's
contribution, before ordinary building/tree shadows. Ambient IBL is unaffected.
No renderer-core changes or material-by-material edits are needed.

## Controls and representation

Enable **spatial cloud shadows** in Static clouds; disable it to compare the
original scalar result. Map extent is measured in metres perpendicular to the
sun, centered on the world origin. Resolution defaults to 256 square; extent
400 m. Both are configurable. The RGBA16F render target stores transmission in
all channels (512 KiB at 256 square). The visual result uses one bilinear sample
per fragment, with a feather over the outer 20% into the original scalar fallback.
Cloud density itself is smooth, so this prototype needs no extra blur pass.

A stable orthonormal basis projects world positions into the map. For each texel,
CloudLayerNode.getTransmittanceNode(direction, origin) intersects the sun ray with
the same thin cloud shell and noise field used by the visible sky. Receiver height
is handled through the ray origin. The new origin defaults to zero, preserving
existing callers. The local spherical-shell approximation is intended for bounded
scenes, not planet-scale maps or flying above/inside the clouds.

The light color node replaces the built-in color/intensity uniform. The example
therefore explicitly supplies unattenuated, atmosphere-filtered, pre-exposed sun
RGB, then multiplies by the map transmission exactly once. The old point sample
continues to drive the visible sun disc, exposure meter, and distant fallback.
CSM uses the identical color node. Buildings and trees retain their normal shadows.

The map bakes only when sky/cloud/map settings change; camera movement and normal
frames do not regenerate it. Toggling off skips its bake. The shader still has a
map sample because the toggle is a uniform; the unchanged parent is the true
performance baseline.

## What this solves

Cloud boundaries can cross different parts of the yard rather than changing the
entire yard through one global intensity. At the original 2500 m cloud feature
size, a 30 m yard can still have nearly uniform transmission: that is expected.
Try 500 m feature size and vary arrangement to inspect local gradients. Arbitrary
cloud arrangement edits still jump immediately; temporal interpolation is a
separate alternative. This PR does not claim to smooth sky/IBL rebakes or exposure.

## Industry research

Projected cloud transmission/shadow maps are the established approach. Unreal's
[Volumetric Cloud component documentation](https://dev.epicgames.com/documentation/unreal-engine/volumetric-cloud-component-in-unreal-engine)
compares Beer shadow maps with secondary ray marching and recommends shadow maps
where performance matters. Its controls trade map resolution and coverage extent
against quality. This prototype uses the existing thin shell instead of volumetric
marching, while following the same spatial light-modulation principle.

[Sea of Thieves: Simulating Tropical Weather, GDC 2022](https://media.gdcvault.com/GDC%2B2022/Speaker%2BSlides/SimulatingTropicalWeather_Weick_Colin_Zhou_Emily.pdf)
also describes projected cloud shadows. A scalar rate limiter is a useful cheap
fallback, but cannot express spatial shadows. Finite-source integration is another
anti-aliasing option, explored independently in filter-cloud-sun; the physical
solar disc is small, so its effect can be subtle and artistic widening trades
accuracy for stronger filtering.

Hardware validation and performance results are recorded in the PR description
and the issue's downloadable artifacts.

## Validation on MediaPC

Physical AMD Radeon WebGPU (gcn-5, fallback=false), 1280x900, 2048 directional
shadows, 4096 background and 512 IBL. Three sequential warmed runs of at least
five seconds; vsync/frame limits disabled; GPU completion counted and queues
drained between stages. Parent/map median frame throughput: 765.10/751.70 fps
with source modules and 741.54/719.44 fps with the bundled demo renderer. That is
about 2-3% extra time per frame. Runs were noisy: one source-map run reached only
512 fps; bundled runs ranged 697-756 fps versus parent 712-765. These are observed
medians, not worst-case guarantees.

Complete sky rebuilds: parent 26.759/s, map 26.507/s, about 0.36 ms / 1% extra per
rebuild. Ordinary frames do not rebake. Changing map size/resolution changes cost.

Hardware output through the shipped renderer matches the parent exactly under
clear sky across 4,608,000 RGBA image components. Coverage zero and density zero
produce transmission one; all sampled map states remain bounded in [0,1]. Checked
zenith, low sun, twilight, and both light modes. Screenshots use GPU output-target
readback with normal ACES/SRGB output because Chrome compositor captures were
black on the unchanged parent as well. Captures, raw timing runs and replay scripts
are attached to swarm issue 134's artifacts.

Core lint and changed-file lint pass. Core unit tests: 1956 passed, one todo, one
pre-existing subgroupBallot failure (test assumes <=32 lanes, AMD reports 64).
Addon units: 107 passed. The temporary unit runner omitted the stock Linux
software-driver override and verified the physical adapter; no runner changes
are committed. No TypeScript typecheck is configured.
