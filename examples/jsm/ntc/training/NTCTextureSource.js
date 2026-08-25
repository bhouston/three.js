import * as THREE from 'three';
import { vec4 } from 'three/tsl';

/**
 * Bakes an arbitrary TSL color node (e.g. a MaterialX base-color graph -
 * whether it's a literal image texture or a procedural node graph) to a
 * plain GPU texture by rendering it across a fullscreen quad. This reuses
 * the "fullscreen quad + orthographic camera" pattern from the neural
 * appearance teacher-atlas renderer, simplified to a single full-resolution
 * pass since there's no per-sample atlas tiling to do here.
 */
async function bakeColorNodeToTexture( renderer, colorNode, resolution = 512 ) {

	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
	camera.position.set( 0, 0, 2 );

	const material = new THREE.NodeMaterial();
	material.lights = false;
	material.toneMapped = false;
	// These bake textures store raw packed data (e.g. roughness/metalness may
	// land in the alpha channel), not real alpha-compositing surfaces. Without
	// this, the material's default opaque blending state forces alpha to 1.0
	// on write, silently clobbering any data channel packed into slot 3 of a
	// vec4 (see NeuralMaterialSource.js's packComponentsIntoVec4).
	material.blending = THREE.NoBlending;
	// Deliberately `fragmentNode`, not `colorNode`: assigning `colorNode` runs
	// NodeMaterial's normal (colorNode === null ? built-in : custom) fragment
	// path through `setupDiffuseColor`/`setupLighting`, whose "basicOutput"
	// unconditionally does `vec4(outgoingLightNode, diffuseColor.a).max(0)`
	// ("force unsigned floats - useful for RenderTargets", see
	// NodeMaterial.js's setupFragment) - correct for a material's *actual
	// lit output*, but this bake path (re)uses a colorNode to smuggle
	// arbitrary, potentially *signed* data (e.g. NeuralMaterialFormat.js's
	// tangent-space (dx, dy) normal offset, in [-1, 1]) through a plain
	// unlit NodeMaterial. That `.max(0)` silently clips every negative
	// component of every baked channel to 0 before training ever sees it -
	// found by writing a test that baked a flat known-negative color and got
	// a wrong, non-negative value back (test/vitest/browser/neural-texture/
	// NTCTextureSource.test.js's "preserves negative values" case).
	// `fragmentNode` bypasses setupDiffuseColor/setupLighting/that clamp
	// entirely (see NodeMaterial.js: "this node property can be used if you
	// need complete freedom in implementing the fragment shader"), so the
	// baked value is exactly what `colorNode` computes, sign intact.
	// `vec4( colorNode )` (TSL's ConvertType) auto-pads a vec3 input with
	// alpha = 1 and passes a vec4 input through unchanged, so this works
	// whether the caller's node is already a full RGBA pack (e.g.
	// NeuralMaterialSource.js's buildPackedColorNodes) or a bare vec3 (e.g.
	// a MaterialX albedo colorNode, see webgpu_materials_neural_texture.html).
	material.fragmentNode = vec4( colorNode );

	const geometry = new THREE.PlaneGeometry( 2, 2 );
	// PlaneGeometry's built-in UV has v=1 at the top of the quad and v=0 at
	// the bottom - the opposite of the "v=0 at top" convention three.js's own
	// fullscreen-quad helper (renderers/common/QuadMesh.js's QuadGeometry)
	// uses for render-target passes, which is what makes "render at raw UV
	// (u,v), then later sample the resulting texture back at that same (u,v)"
	// round-trip correctly. Left unflipped, whatever gets baked at (u,v) here
	// reads back later (via texture()/textureLevel() in
	// NeuralTextureGPUComputeTSL's training kernel and NeuralTextureNodeMaterial/
	// NeuralMaterialNodeMaterial's inference-time evaluateNeuralTextureRaw) as
	// (u, 1-v) instead - a spurious vertical flip baked into every trained
	// channel, since none of those consumers know to undo it. Flipping V here
	// once, at the source, keeps the whole pipeline working in plain raw-UV
	// space so the neural mesh's material lines up with the teacher's.
	const uvAttribute = geometry.attributes.uv;
	for ( let i = 0; i < uvAttribute.count; i ++ ) uvAttribute.setY( i, 1 - uvAttribute.getY( i ) );
	uvAttribute.needsUpdate = true;
	geometry.computeTangents();
	const mesh = new THREE.Mesh( geometry, material );
	scene.add( mesh );

	const renderTarget = new THREE.RenderTarget( resolution, resolution, {
		type: THREE.HalfFloatType,
		format: THREE.RGBAFormat,
		colorSpace: THREE.NoColorSpace,
		minFilter: THREE.LinearFilter,
		magFilter: THREE.LinearFilter,
		wrapS: THREE.RepeatWrapping,
		wrapT: THREE.RepeatWrapping,
		depthBuffer: false,
		stencilBuffer: false
	} );

	const previousTarget = renderer.getRenderTarget();
	const previousToneMapping = renderer.toneMapping;
	renderer.toneMapping = THREE.NoToneMapping;

	renderer.setRenderTarget( renderTarget );
	renderer.render( scene, camera );

	renderer.setRenderTarget( previousTarget );
	renderer.toneMapping = previousToneMapping;

	geometry.dispose();
	material.dispose();

	return renderTarget;

}

/**
 * Extracts the base/albedo color TSL node from a MaterialX-loaded
 * MeshPhysicalNodeMaterial. Works whether the base color is a literal image
 * texture or a procedural node graph (checkerboard, noise, etc.) since it
 * doesn't try to walk the graph looking for an `<image>` node - it simply
 * hands the whole color expression to `bakeColorNodeToTexture`.
 */
function extractBaseColorNode( materialXMaterial ) {

	if ( ! materialXMaterial ) return null;
	if ( materialXMaterial.colorNode ) return materialXMaterial.colorNode;

	return null;

}

/**
 * Loads a plain image file (PNG/JPG/etc.) as a GPU texture, decoding sRGB to
 * the renderer's linear working color space like any other albedo map.
 */
function loadImageTexture( url ) {

	return new THREE.TextureLoader().loadAsync( url ).then( ( texture ) => {

		texture.colorSpace = THREE.SRGBColorSpace;
		texture.wrapS = THREE.RepeatWrapping;
		texture.wrapT = THREE.RepeatWrapping;
		texture.needsUpdate = true;

		return texture;

	} );

}

/**
 * Owns a source texture plus the (optional) render target that produced it,
 * so a caller doesn't have to track "is this a baked texture I own a
 * render target for, or a plain loaded texture I own directly?" via a
 * separate nullable variable. `dispose()` releases whichever one applies -
 * disposing the render target's own copy of the texture, never both.
 */
class NTCTextureSource {

	constructor( texture, renderTarget = null ) {

		this.texture = texture;
		this.renderTarget = renderTarget;

	}

	static async fromBakedColorNode( renderer, colorNode, resolution = 512 ) {

		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, resolution );
		return new NTCTextureSource( renderTarget.texture, renderTarget );

	}

	static async fromImage( url ) {

		return new NTCTextureSource( await loadImageTexture( url ) );

	}

	dispose() {

		if ( this.renderTarget ) this.renderTarget.dispose();
		else this.texture.dispose();

	}

}

export { NTCTextureSource, bakeColorNodeToTexture, extractBaseColorNode, loadImageTexture };
