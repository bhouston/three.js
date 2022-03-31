import {
	Color,
	DoubleSide,
	LinearFilter,
	Object3D,
	Matrix4,
	RGBAFormat,
	ShaderMaterial,
	UniformsUtils,
	WebGLRenderTarget
} from 'three';

import { Pass, FullScreenQuad } from './Pass.js';
import { CopyShader } from '../shaders/CopyShader.js';
import { TRAAShader } from '../shaders/TRAAShader.js';
import { VelocityShader } from '../shaders/VelocityShader.js';

class TRAARenderPass extends Pass {

	constructor( scene, camera, resolution ) {

		super();

		this.scene = scene;
		this.camera = camera;

		//this.orthoScene = new Scene();
		//this.orthoCamera = new OrthographicCamera(-1, 1, 1, -1, -0.01, 1000);

		this.fsQuad = new FullScreenQuad( null );

		this.traaMaterial = new ShaderMaterial( {
			uniforms: UniformsUtils.clone( TRAAShader.uniforms ),
			vertexShader: TRAAShader.vertexShader,
			fragmentShader: TRAAShader.fragmentShader,
			side: DoubleSide,
		} );

		this.velocityMaterial = new ShaderMaterial( {
			uniforms: UniformsUtils.clone( VelocityShader.uniforms ),
			vertexShader: VelocityShader.vertexShader,
			fragmentShader: VelocityShader.fragmentShader,
			side: DoubleSide,
		} );
		this.velocityMaterial.extensions.derivatives = true;

		this.oldClearColor = new Color();
		this.oldClearAlpha = 1;
		this.needsSwap = false;

		this.copyMaterial = new ShaderMaterial( {
			uniforms: UniformsUtils.clone( CopyShader.uniforms ),
			vertexShader: CopyShader.vertexShader,
			fragmentShader: CopyShader.fragmentShader,
			transparent: true,
			depthWrite: false,
		} );

		this.accumulatedBeautyRenderTarget = new WebGLRenderTarget(
			256,
			256,
			{
				minFilter: LinearFilter,
				magFilter: LinearFilter,
				format: RGBAFormat,
				stencilBuffer: true,
			}
		);

		this.previousProjectionViewMatrix = new Matrix4();
		this.currentProjectionViewMatrix = new Matrix4();

		this.projectionMatrix = this.camera.projectionMatrix.clone();

		this.numSamplesPerAccumulation = 16;
		this.staticMode = false;

		this.depthTexture = null;

	}

	dispose() {

		if ( this.accumulatedBeautyRenderTarget ) this.accumulatedBeautyRenderTarget.dispose();

	}

	setSize( width, height ) {

		if ( this.accumulatedBeautyRenderTarget ) this.accumulatedBeautyRenderTarget.setSize( width, height );
		if ( this.velocityRenderTarget ) this.velocityRenderTarget.setSize( width, height );

		this.projectionMatrix.copy( this.camera.projectionMatrix );

		this.resetPending = true;

	}

	renderOverride(
		renderer,
		overrideMaterial,
		renderTarget,
		clearColor,
		clearAlpha
	) {

		var originalClearColor = renderer.getClearColor().getHex();
		var originalClearAlpha = renderer.getClearAlpha();
		var originalAutoClear = renderer.autoClear;

		renderer.autoClear = false;

		clearColor = overrideMaterial.clearColor || clearColor;
		clearAlpha = overrideMaterial.clearAlpha || clearAlpha;
		var clearNeeded = clearColor !== undefined && clearColor !== null;

		if ( clearNeeded ) {

			renderer.setClearColor( clearColor );
			renderer.setClearAlpha( clearAlpha || 0.0 );

		}

		this.scene.overrideMaterial = overrideMaterial;
		// if ( this.camera.clearViewOffset ) this.camera.clearViewOffset();

		renderer.render(
			this.scene,
			this.camera,
			renderTarget,
			clearNeeded,
			this.visibilityFunc
		);
		this.scene.overrideMaterial = null;

		// restore original state
		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );
	}

	render( renderer, writeBuffer, readBuffer, delta, maskActive, overrideCamera ) {

		var camera = overrideCamera || this.camera;


		renderer.getClearColor( this.oldClearColor );
		this.oldClearAlpha = renderer.getClearAlpha();
		var oldAutoClear = renderer.autoClear;
		var oldAutoClearDepth = renderer.autoClearDepth;

		var width = writeBuffer.width,
			height = writeBuffer.height;

		if ( ! this.velocityRenderTarget ) {

			var params = {
				minFilter: LinearFilter,
				magFilter: LinearFilter,
				format: RGBAFormat,
			};

			this.velocityRenderTarget = new WebGLRenderTarget( width, height, params );

		}

		this.currentMaterial = this.superSampleTRAAMaterial;
		this.currentProjectionViewMatrix.multiplyMatrices(
			this.projectionMatrix,
			camera.matrixWorldInverse
		);

		renderer.getClearColor( this.oldClearColor );
		this.oldClearAlpha = renderer.getClearAlpha();
		var oldAutoClear = renderer.autoClear;
		var oldAutoClearDepth = renderer.autoClearDepth;
		var oldAutoClearColor = renderer.autoClearColor;

		renderer.autoClear = false;

		renderer.setClearColor( new Color( 0, 0, 0 ) );
		renderer.setClearAlpha( 0 );

		this.velocityMaterial.uniforms.currentProjectionViewMatrix.value.copy(
			this.currentProjectionViewMatrix
		);
		this.velocityMaterial.uniforms.previousProjectionViewMatrix.value.copy(
			this.previousProjectionViewMatrix
		);

		//renderer.autoClearColor = true;
		this.scene.overrideMaterial = this.velocityMaterial;
		renderer.render(
			this.scene,
			camera,
			this.velocityRenderTarget,
			true,
			this.visibilityFunc
		);
		this.scene.overrideMaterial = null;
		this.scene.traverse( function ( obj ) {

			if ( obj instanceof Object3D ) {

				obj.matrixWorldPrevious.copy( obj.matrixWorld );

			}

		} );

		const traaUniforms = this.currentMaterial.uniforms;

		if ( camera.view ) {

			traaUniforms.jitterOffset.value.set(
				camera.view.offsetX,
				camera.view.offsetY
			);

		}

		traaUniforms.currentBeauty.value = readBuffer.texture;
		traaUniforms.previousBeauty.value = this.accumulatedBeautyRenderTarget.texture;
		traaUniforms.DEPTH_PACKING = this.depthTexture.depthPacking;
		traaUniforms.tDepth.value = this.depthTexture;
		traaUniforms.tVelocity.value = this.velocityRenderTarget.texture;

		if ( this.resetPending ) {

			traaUniforms.mode.value = 2;
			this.resetPending = false;

		} else if ( this.staticMode ) {

			traaUniforms.mode.value = 1;

		} else {

			traaUniforms.mode.value = 0;

		}

		traaUniforms.cameraInverseProjectionMatrix.value.getInverse( this.projectionMatrix );
		traaUniforms.cameraProjectionMatrix.value.copy( this.projectionMatrix );
		traaUniforms.cameraInverseViewMatrix.value.copy( camera.matrixWorld );
		traaUniforms.cameraNearFar.value.set( camera.near, camera.far );
		traaUniforms.textureSize.value.set( width, height );
		traaUniforms.minSampleWeight.value = 1.0 / this.numSamplesPerAccumulation;

		//renderer.autoClearColor = true;
		//renderer.autoClearDepth = false;
		renderer.autoClearDepth = false;
		renderer.setRenderTarget( null );
		this.fsQuad.material = this.traaMaterial;
		this.fsQuad.render( renderer );

		this.copyMaterial.uniforms.tDiffuse.value = writeBuffer.texture;
		this.copyMaterial.uniforms.opacity.value = 1;

		this.fsQuad.material = this.copyMaterial;
		renderer.setRenderTarget( this.accumulatedBeautyRenderTarget );
		this.fsQuad.render( renderer );

		renderer.setRenderTarget( readBuffer );
		this.fsQuad.render( renderer );

		renderer.setClearColor( this.oldClearColor );
		renderer.setClearAlpha( this.oldClearAlpha );
		renderer.autoClear = oldAutoClear;
		renderer.autoClearColor = oldAutoClearColor;
		renderer.autoClearDepth = oldAutoClearDepth;
		this.previousProjectionViewMatrix.copy( this.currentProjectionViewMatrix );

	}

}

export { TRAARenderPass };
