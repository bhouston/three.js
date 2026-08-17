# MaterialX Neural Parameter Extraction Contract

## Purpose

This document defines a preferred contract for extracting spatial material features from MaterialX graphs for the encoder phase of neural appearance training.

The key distinction is:

- The **teacher** evaluates the complete reference material and produces directional BRDF targets.
- The **parameter extractor** evaluates direction-independent, spatially varying material features that help initialize latent codes.
- The **runtime neural material** uses neither the extractor nor the encoder. Both are training-only.

This contract is not required to explain the visible 8×8 blocks in the torus rendering. That artifact primarily comes from using an 8×8 finest latent level. Parameter extraction becomes important when scaling training to large latent textures, where direct optimization cannot update every texel sufficiently or consistently.

## Why an encoder needs material parameters

The NVIDIA paper does not initialize millions of latent texels independently. It first trains an encoder:

```text
material parameters at x -> encoder -> latent code z(x) -> BRDF decoder
```

The paper's reference renderer has an explicit layered-material representation. For every layer it can export quantities such as albedo, roughness, normal, tangent, and layer weight. Similar source parameters therefore tend to receive similar latent codes. After this phase, the encoder is evaluated over the full texture pyramid, its outputs are baked into latent textures, and those texels are fine-tuned directly.

This provides two benefits:

1. One training sample updates an encoder shared by every spatial location, rather than only four neighboring latent texels.
2. Locations with the same material parameters begin with compatible latent codes instead of unrelated random values.

The current `NeuralAppearanceTeacherEvaluator.encodeInputs()` is not such an encoder input. It returns UV and canonical frame values, and `NeuralAppearanceTrainer` does not feed those values into its model. It should eventually be removed or replaced by the contract below.

## Constraint: arbitrary MaterialX graphs

There is no universal, lossless function:

```text
arbitrary MaterialX graph -> fixed vector of physical parameters
```

An arbitrary graph can contain:

- Standard Surface, OpenPBR, glTF PBR, or custom closure nodes.
- Layering, mixing, conditionals, and implementation-specific closures.
- Procedural textures driven by UV, position, object data, or time.
- Direction-dependent computations embedded before a closure.
- Values with no shared physical interpretation.
- Custom nodes unknown to three.js.

Attempting to infer “albedo” or “roughness” by inspecting arbitrary graph topology would be brittle and often incorrect. The extractor must operate at a compiler boundary where supported surface semantics are already known. Unsupported graphs must be reported as unsupported rather than assigned invented parameters.

## Decision

Use a **schema-driven, graph-specific feature vector** captured at supported MaterialX surface-mapping boundaries.

For the initial implementation:

1. Support opaque `standard_surface`, `open_pbr_surface`, and `gltf_pbr` terminals that successfully map to `MeshPhysicalNodeMaterial`.
2. Capture evaluated surface-input nodes before or while `MaterialXSurfaceMappings.js` maps them to three.js material nodes.
3. Produce a frozen ordered schema for each trained material.
4. Evaluate the captured feature nodes under the same UV, geometry, and footprint overrides as the BRDF teacher.
5. Reject unsupported closures and active transport modes instead of silently substituting defaults.
6. Fall back to direct latent optimization when no parameter schema is available.

The schema may differ between materials. The encoder is training-only and material-specific, so a universal cross-material input dimension is not required.

## Extraction boundary

The preferred extraction point is the resolved `inputs` object received by:

- `applyStandardSurface()`
- `applyOpenPbrSurface()`
- `applyGltfPbrSurface()`

in `examples/jsm/loaders/materialx/MaterialXSurfaceMappings.js`.

At this point:

- Arbitrary upstream texture and procedural graphs have already compiled to evaluable TSL nodes.
- Surface input names have known semantics.
- Defaults can be applied consistently.
- The graph has not yet been irreversibly flattened into a smaller set of `MeshPhysicalNodeMaterial` properties.

The extractor stores references to nodes, not sampled constants. A feature such as `base_color` may still be an arbitrary procedural graph and is evaluated for every requested UV and footprint.

## Contract

### Descriptor

Each loaded material may expose a training-only descriptor:

```ts
interface MaterialXAppearanceFeatureDescriptor {
	version: 1;
	materialName: string;
	surfaceModel: 'standard_surface' | 'open_pbr_surface' | 'gltf_pbr';
	schema: MaterialXAppearanceFeature[];
	capabilities: {
		opaqueReflection: boolean;
		filteredEvaluation: boolean;
	};
	unsupportedReasons: string[];
}

interface MaterialXAppearanceFeature {
	name: string;
	semantic: string;
	components: 1 | 2 | 3 | 4;
	valueNode: Node;
	defaultValue: number[];
	encoding: 'linear' | 'unitVector' | 'angleSinCos' | 'positiveLog';
	filter: 'linearMean' | 'unitVectorMean' | 'leanNormalRoughness' | 'constant';
}
```

The public loader result should associate the descriptor with the corresponding material without making it part of runtime serialization:

```ts
asset.materials[ name ]
asset.appearanceFeatures[ name ]
```

### Evaluator

The trainer creates an evaluator from the descriptor:

```ts
interface MaterialXAppearanceFeatureEvaluator {
	readonly schema: MaterialXAppearanceFeatureSchema;

	evaluateBatch(
		samples: MaterialXAppearanceFeatureSample[]
	): Promise<MaterialXAppearanceFeatureBatch>;

	dispose(): void;
}

interface MaterialXAppearanceFeatureSample {
	uv: [number, number];
	duvDx: [number, number];
	duvDy: [number, number];
	normal: [number, number, number];
	tangent: [number, number, number];
	bitangent: [number, number, number];
	mip: number;
}

interface MaterialXAppearanceFeatureBatch {
	values: Float32Array;
	sampleCount: number;
	stride: number;
	schemaId: string;
}
```

`values` is row-major: one packed schema vector per sample. `schemaId` is a deterministic hash of feature order, component counts, encodings, defaults, and filtering rules. Training must fail if a batch schema differs from the schema used to construct the encoder.

## Initial feature schemas

Only features that affect the supported opaque reflective teacher should be emitted. UV itself, incoming direction, outgoing direction, camera position, and object position are not material parameters and must not be encoder inputs.

### Standard Surface

Recommended ordered features:

```text
base.weight
base.color.rgb
specular.weight
specular.color.rgb
specular.roughness
specular.ior
specular.anisotropy
specular.rotation.sinCos
coat.weight
coat.color.rgb
coat.roughness
coat.ior
coat.normal.xyz
sheen.weight
sheen.color.rgb
sheen.roughness
thinFilm.weight
thinFilm.thickness
thinFilm.ior
geometry.normal.xyz
```

Defaults must match the MaterialX surface definition or the exact defaults used by the three.js mapping. Missing optional inputs still occupy their schema positions.

### OpenPBR Surface

Use corresponding OpenPBR semantics:

```text
base.weight
base.color.rgb
base.metalness
specular.weight
specular.color.rgb
specular.roughness
specular.roughnessAnisotropy
specular.ior
coat.weight
coat.color.rgb
coat.roughness
coat.ior
coat.normal.xyz
fuzz.weight
fuzz.color.rgb
fuzz.roughness
thinFilm.weight
thinFilm.thickness
thinFilm.ior
geometry.normal.xyz
```

### glTF PBR

Use the smaller glTF semantic set:

```text
base.color.rgb
metallic
roughness
specular.weight
specular.color.rgb
ior
clearcoat.weight
clearcoat.roughness
clearcoat.normal.xyz
sheen.color.rgb
sheen.roughness
iridescence.weight
iridescence.ior
iridescence.thickness
anisotropy.strength
anisotropy.rotation.sinCos
geometry.normal.xyz
```

Occlusion is excluded because it is not a local BRDF parameter. Emission is excluded because the current neural material models reflected direct light, not emitted radiance.

## Opaque-reflection scope

The initial encoder contract should reject or disable encoding for materials with active:

- Transmission or refraction.
- Non-unit or cutout opacity.
- Emission when emission is expected to be represented neurally.
- Subsurface or volume closures.
- Displacement that changes the actual shading intersection.
- Custom closures that do not map to a supported surface model.

These restrictions should match the teacher and runtime contract. Extracting transmission parameters would not help if the neural runtime can only evaluate an opaque reflective BRDF.

The descriptor should preserve precise rejection reasons, for example:

```text
Neural appearance parameter encoding is unavailable:
standard_surface.transmission is active, but the current runtime is opaque-reflection-only.
```

Training may still proceed with direct latent optimization if the teacher can evaluate the material correctly, but the UI must identify that the encoder phase is unavailable.

## Layered and mixed graphs

The paper's renderer exposes a list of explicit material layers. General MaterialX closure composition does not necessarily preserve that structure through the current three.js translation.

The initial contract therefore treats each supported terminal surface model as one semantic record containing its supported lobes. It must not pretend to recover arbitrary closure layers after they have been flattened.

A later layered extension may define:

```ts
interface MaterialXAppearanceLayer {
	path: string;
	model: string;
	weightNode: Node;
	features: MaterialXAppearanceFeature[];
}
```

Layer order and `path` would be fixed from the parsed document. The encoder vector would concatenate layer records and masks. This extension is appropriate only after the MaterialX compiler represents closure composition explicitly and the teacher evaluates the same composition.

## Filtering rules

Encoder features must represent the same footprint and mip level as the BRDF target.

Simple hardware mip lookup is not sufficient for every feature:

- Colors and scalar weights: spatial mean over the footprint.
- Positive distances or thicknesses: mean in a documented linear or log domain.
- Angles: average sine and cosine, never the wrapped angle directly.
- Unit normals and tangents: average directional moments and renormalize.
- Normal plus roughness: use LEAN-style moments or an equivalent normal-distribution representation.
- Categorical or graph-structural values: constant per schema, not sampled as floats.

For an arbitrary upstream procedural graph, filtered feature evaluation should use the same Gaussian footprint sampling strategy as the filtered teacher target. Each sub-sample evaluates the compiled feature nodes at an offset UV. This avoids relying on every MaterialX node to implement analytically correct filtering.

The evaluator may use multiple render targets or multiple passes to return vectors wider than four channels. The implementation detail must not change feature order.

## Coordinate and frame rules

Feature evaluation must share the teacher's semantic overrides:

```text
source UV
normal
tangent
bitangent
UV derivatives / footprint
```

Rules:

1. Apply source-space UV overrides before MaterialX UV-space conversion.
2. Evaluate normal features in the canonical local tangent frame used by training.
3. Encode angular anisotropy as sine/cosine.
4. Do not include `wi` or `wo`; those belong to the BRDF decoder.
5. Do not include raw UV unless intentionally training a coordinate-conditioned model. Raw UV undermines the goal that equal material parameters map to equal latent codes.
6. Use the semantic UV resolver proposed in `materialx_semantic_geometry_overrides.md` so explicit texcoord nodes and procedural coordinate fallbacks receive the same override.

## Encoder lifecycle

The preferred training sequence is:

```text
1. Compile teacher and feature descriptor.
2. Freeze the feature schema.
3. Sample UV, footprint, wi, and wo.
4. Evaluate filtered feature vector k(x, footprint).
5. Evaluate filtered teacher BRDF target.
6. Train encoder(k) + frame extractor + BRDF decoder.
7. Evaluate encoder for every texel of every latent mip.
8. Bake encoder outputs into latent textures.
9. Remove encoder from the model.
10. Fine-tune latent texels, frame extractor, and decoder directly.
11. Export only latent textures and runtime decoders.
```

No source MaterialX parameters or encoder weights need to appear in the runtime JSON.

## Unsupported arbitrary graphs: preferred fallback

If no trustworthy semantic schema can be produced:

1. Continue with direct latent optimization.
2. Warn that high-resolution convergence will be worse.
3. Recommend a lower latent resolution or substantially larger training budget.
4. Never fill a generic vector with UV, arbitrary node values, or guessed “roughness.”

An optional future alternative is a **reflectance-probe encoder**: evaluate the teacher at a fixed set of canonical direction pairs and encode that response vector. This can support arbitrary black-box BRDF graphs, but it is not the parameter encoder from the paper. It is more expensive, duplicates part of the decoder's task, and requires careful directional filtering. It should be designed separately.

## API ownership

Suggested responsibilities:

- `MaterialXDocument.js`: preserve the terminal surface model and resolved semantic input nodes.
- `MaterialXSurfaceMappings.js`: declare supported feature schemas at the same point surface inputs are mapped.
- New `MaterialXAppearanceFeatures.js`: normalize schemas, defaults, encodings, capability checks, and schema IDs.
- `NeuralAppearanceTeacherEvaluator.js`: evaluate feature nodes in batches under the same geometry/UV context as teacher targets, or delegate to a sibling evaluator.
- `NeuralAppearanceTrainer.js`: construct and train the encoder, bake latent mips, and fall back when no descriptor exists.

The generic MaterialX loader should expose semantic feature descriptors. Neural-network architecture and optimization remain the trainer's responsibility.

## Validation

### Schema tests

- Stable feature order for each supported surface model.
- Correct MaterialX defaults for omitted inputs.
- Different graph topology with equal evaluated parameters produces equal vectors.
- Unsupported closures and transport features produce explicit reasons.
- Schema IDs change when order, encoding, or filtering changes.

### Evaluation tests

- Constant inputs produce constant vectors across UV.
- Image and procedural inputs respond to UV overrides.
- Explicit texcoord and fallback coordinates use the same semantic override.
- Normal maps evaluate in the canonical training frame.
- Angle wraparound averages correctly through sine/cosine encoding.
- Filtered normal/roughness features match supersampled references.

### Training tests

- Equal parameter vectors initialize equal latent codes.
- A high-resolution repeated material converges faster with encoder bootstrap than direct texel optimization at the same sample budget.
- Baking the encoder reproduces encoder-path predictions before fine-tuning.
- Removing the encoder after baking does not change runtime serialization.
- Unsupported materials still train through the direct-optimization fallback.

## Importance and priority

Parameter extraction is important for **scaling**, not for basic correctness at 8×8:

- It will not by itself remove the current 8×8 pattern. Raising the finest latent resolution is required for that.
- Raising resolution without an encoder makes direct optimization increasingly inefficient and leaves random latent inconsistency.
- Correct footprint-filtered targets are still required; an encoder cannot recover information absent from its targets.

The recommended order is:

1. Demonstrate that torus block size follows latent resolution.
2. Implement footprint-correct teacher targets.
3. Add this supported-surface parameter contract and encoder bootstrap.
4. Scale latent resolution toward source texture resolution.
5. Extend semantic coverage only when teacher and runtime support the same material behavior.

## Summary

The preferred contract does not attempt to understand every arbitrary MaterialX graph. It captures **evaluated, direction-independent inputs of supported surface models at the compiler's semantic boundary**, under the same spatial footprint as the teacher. Arbitrary upstream graphs remain arbitrary; they are simply evaluated to known surface-input semantics.

When that semantic boundary does not exist, the correct behavior is direct latent optimization or a separately designed black-box reflectance encoder—not guessed material parameters.
