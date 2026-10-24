# SSGI diffuse source experiment

Base: `8de0e0315c5c`. Toggle **SSGI Settings → Source radiance** between **Beauty** and **Diffuse + emission** (experimental default).

The beauty pass writes a separate linear HDR MRT containing `totalDiffuse + emissive`. SSGI copies and reprojects that history; SSR continues to use full beauty history. This removes the camera-direction specular reflection from the approximate diffuse radiance field while retaining multi-bounce diffuse feedback and emitted light. Both histories resize together and share depth validation/reset state.

This is a diffuse-source approximation. It deliberately loses specular-to-diffuse paths instead of incorrectly reusing the camera-direction specular lobe. The initial prototype targets the standard/metallic materials in Cornell Box and Dual Rooms. Layered physical materials require additional accounting for finish-stage clearcoat/sheen attenuation; unlit materials retain their output. Material emission is assumed to be available through the standard emissive property.

Validate Cornell metallic sphere and mirror wall, then Dual Rooms. Look for false color bleeding from reflections and changes in brightness while orbiting a stationary camera. Compare both source settings with a path-traced reference; less temporal fluctuation alone is not a correctness result. Check diffuse-only colored walls and emissive geometry retain their light contribution. Record GPU completed-frame throughput because another HDR attachment and history copy add bandwidth.

Syntax/lint passed. Hardware rendering is delegated to the central sequential validation run; no GPU result is claimed here.

References: [AMD Brixelizer GI source feedback](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/brixelizer-gi/), [SSILVB](https://cdrinmatane.github.io/cgspotlight-slides/ssilvb_slides.pdf).

The guarded refinement applies diffuse+emission source only to opaque standard/physical materials without active clearcoat, sheen, transmission, corresponding node variants, or custom lighting/output/backdrop nodes. Unsupported materials write beauty into the separate source attachment, preserving legacy source behavior. This is selected per compiled material, not per scene. The metallic regression remains; added clearcoat/sheen/node-layer MRT regressions require source and beauty to match exactly for unsupported materials. Hardware execution of these added guards is pending central validation.
