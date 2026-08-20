import * as THREE from 'three';
import * as TSL from 'three/tsl';
import { EnvironmentNode, PhysicalLightingModel } from 'three/webgpu';

/**
 * The pure TSL shading-node math the GPU-resident teacher (see
 * `NeuralAppearanceTeacherEvaluator.js`) evaluates a real MeshPhysicalNodeMaterial
 * with, under controlled per-sample overrides, to generate training targets -
 * split into its own file since none of it references the evaluator's
 * instance state (batching, MRT grouping, atlas caching): it's shading-node
 * authoring, not orchestration, and reads more clearly reviewed as such.
 */

class NeuralTeacherLightingModel extends PhysicalLightingModel {

	constructor( material, lightDirectionNode ) {

		if ( THREE.PhysicalLightingModel === undefined ) {

			throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: PhysicalLightingModel is required for directional GPU teacher sampling. Import three/webgpu for training.' );

		}

		super( material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useTransmission, material.useDispersion, material.useRetroreflection );
		this.lightDirectionNode = lightDirectionNode;

	}

	direct( { reflectedLight }, builder ) {

		super.direct( {
			lightDirection: this.lightDirectionNode,
			lightColor: TSL.vec3( 1 ),
			reflectedLight
		}, builder );

	}

	indirect( /*builder*/ ) {}

}

class NeuralTeacherIBLLightingModel extends PhysicalLightingModel {

	// `capture`, when given, is `{ indirectDiffuse, indirectSpecular }` --
	// two vec3 `.toVar()` nodes that get assigned PhysicalLightingModel's
	// final indirect diffuse/specular split once lighting finishes, so a
	// caller can read them out as separate MRT outputs instead of only the
	// combined `output` channel. Used by the merged 'iblIndirect' pass in
	// NeuralAppearanceTeacherEvaluator._createResources(); left null for the
	// standalone isolate-mode debug view (examples/webgpu_materials_neural_appearance.html),
	// which only needs the combined output and must keep working unchanged.
	constructor( material, isolate = 'full', capture = null ) {

		super( material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useTransmission, material.useDispersion, material.useRetroreflection );

		this.isolate = isolate;
		this._capture = capture;

	}

	direct( /*input, builder*/ ) {}

	finish( builder ) {

		super.finish( builder );

		if ( this._capture ) {

			const { reflectedLight } = builder.context;
			this._capture.indirectDiffuse.assign( reflectedLight.indirectDiffuse );
			this._capture.indirectSpecular.assign( reflectedLight.indirectSpecular );

		}

	}

}

class NeuralTeacherIBLEnvironmentNode extends EnvironmentNode {

	static get type() {

		return 'NeuralTeacherIBLEnvironmentNode';

	}

	constructor( envNode = null, isolate = 'full' ) {

		super( envNode );
		this.isolate = isolate;

	}

	customCacheKey() {

		const isolateKey = this.isolate === 'radiance' ? 1 : this.isolate === 'irradiance' ? 2 : 0;
		return super.customCacheKey() + isolateKey;

	}

	setup( builder ) {

		const { material } = builder;
		let envNode = this.envNode;

		if ( envNode.isTextureNode || envNode.isMaterialReferenceNode ) {

			const value = ( envNode.isTextureNode ) ? envNode.value : material[ envNode.property ];
			const cache = this._getPMREMNodeCache( builder.renderer );
			let cacheEnvNode = cache.get( value );

			if ( cacheEnvNode === undefined ) {

				cacheEnvNode = TSL.pmremTexture( value );
				cache.set( value, cacheEnvNode );

			}

			envNode = cacheEnvNode;

		}

		const useAnisotropy = material.useAnisotropy === true || material.anisotropy > 0;
		const radianceNormalView = useAnisotropy ? TSL.bentNormalView : TSL.normalView;
		const radiance = envNode.context( createRadianceContext( TSL.roughness, radianceNormalView ) ).mul( TSL.materialEnvIntensity );
		const irradiance = envNode.context( createIrradianceContext( TSL.normalWorld ) ).mul( Math.PI ).mul( TSL.materialEnvIntensity );

		if ( this.isolate !== 'irradiance' ) {

			builder.context.radiance.addAssign( radiance.isolate() );

		}

		if ( this.isolate !== 'radiance' ) {

			builder.context.iblIrradiance.addAssign( irradiance.isolate() );

		}

		const clearcoatRadiance = builder.context.lightingModel.clearcoatRadiance;

		if ( clearcoatRadiance && this.isolate !== 'irradiance' ) {

			clearcoatRadiance.addAssign(
				envNode.context( createRadianceContext( TSL.clearcoatRoughness, TSL.clearcoatNormalView ) ).mul( TSL.materialEnvIntensity ).isolate()
			);

		}

	}

}

function createRadianceContext( roughnessNode, normalViewNode ) {

	let reflectVec = null;

	return {
		getUV: () => {

			if ( reflectVec === null ) {

				reflectVec = TSL.positionViewDirection.negate().reflect( normalViewNode );
				reflectVec = TSL.pow4( roughnessNode ).mix( reflectVec, normalViewNode ).normalize();
				reflectVec = reflectVec.transformDirection( TSL.cameraWorldMatrix );

			}

			return reflectVec;

		},
		getTextureLevel: () => roughnessNode
	};

}

function createIrradianceContext( normalWorldNode ) {

	return {
		getUV: () => normalWorldNode,
		getTextureLevel: () => TSL.float( 1 )
	};

}

function createTeacherIBLQueryNodes( material ) {

	const shadingNormal = TSL.normalView.normalize();
	const viewDir = TSL.positionViewDirection.normalize();
	const roughnessSource = material.roughnessNode !== undefined && material.roughnessNode !== null ?
		TSL.float( material.roughnessNode ) :
		TSL.materialRoughness;
	const roughness = TSL.getRoughness( { roughness: roughnessSource } );
	const reflectDir = viewDir.negate().reflect( shadingNormal );
	const roughness4 = roughness.mul( roughness ).mul( roughness ).mul( roughness );
	const radianceDir = roughness4.mix( reflectDir, shadingNormal ).normalize();

	return { roughness, radianceDir };

}

export {
	NeuralTeacherLightingModel,
	NeuralTeacherIBLLightingModel,
	NeuralTeacherIBLEnvironmentNode,
	createRadianceContext,
	createIrradianceContext,
	createTeacherIBLQueryNodes
};
