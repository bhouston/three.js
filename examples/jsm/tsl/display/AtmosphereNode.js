import { Node, Vector3 } from 'three/webgpu';
import { Fn, If, Loop, float, int, vec2, vec3, uniform, texture, positionWorldDirection, uv } from 'three/tsl';
import { atmosphere } from '../../math/Atmosphere.js';

const planetRadius = atmosphere.planetRadius;
const atmosphereRadius = atmosphere.atmosphereRadius;
const rayleigh = vec3( ...atmosphere.rayleigh );
const ozone = vec3( ...atmosphere.ozone );
const nightRadiance = vec3( 0.00016, 0.00020, 0.00028 );

const density = Fn( ( [ height ] ) => vec3(
	height.div( - atmosphere.rayleighScaleHeight ).exp(),
	height.div( - atmosphere.mieScaleHeight ).exp(),
	float( 1 ).sub( height.sub( atmosphere.ozoneCenter ).abs().div( atmosphere.ozoneHalfWidth ) ).max( 0 )
) );

const extinction = Fn( ( [ opticalDepth, aerosol ] ) => rayleigh.mul( opticalDepth.x )
	.add( opticalDepth.y.mul( atmosphere.mieExtinction ).mul( aerosol ) )
	.add( ozone.mul( opticalDepth.z ) ) );

const rayLength = Fn( ( [ radius, cosine ] ) => {

	const b = radius.mul( cosine );
	const ground = b.mul( b ).sub( radius.mul( radius ) ).add( planetRadius * planetRadius );
	const length = b.mul( b ).sub( radius.mul( radius ) ).add( atmosphereRadius * atmosphereRadius ).max( 0 ).sqrt().sub( b );
	return b.lessThan( 0 ).and( ground.greaterThan( 0 ) ).select( float( 0 ), length );

} );

/**
 * Atmospheric transmittance for a spherical shell, using 64 midpoint samples.
 * Distances are kilometres; coefficients come from the shared atmosphere model.
 *
 * @tsl
 * @function
 * @param {Node<float>} radius - Distance from the planet centre in kilometres.
 * @param {Node<float>} cosine - Ray zenith cosine.
 * @param {Node<float>} aerosol - Aerosol density multiplier.
 * @return {Node<vec3>} Linear RGB transmittance.
 */
const atmosphereTransmittance = Fn( ( [ radius, cosine, aerosol ] ) => {

	const length = rayLength( radius, cosine );
	const step = length.div( 64 );
	const opticalDepth = vec3( 0 ).toVar();
	Loop( 64, ( { i } ) => {

		const distance = float( i ).add( 0.5 ).mul( step );
		const height = radius.mul( radius ).add( distance.mul( distance ) ).add( radius.mul( cosine ).mul( distance ).mul( 2 ) ).sqrt().sub( planetRadius );
		opticalDepth.addAssign( density( height ).mul( step ) );

	} );
	return extinction( opticalDepth, aerosol ).negate().exp().mul( length.greaterThan( 0 ) );

} );

/**
 * Single-scattering Rayleigh/Mie/ozone sky radiance, excluding the solar disc.
 * Requires a transmittance LUT whose U axis maps sign(mu) * sqrt(abs(mu)) and
 * V axis maps sqrt(height / atmosphere thickness), both to the unit interval.
 * Bake getTransmittanceLUTNode() into a 256×128 half-float render target with
 * linear filtering and clamp-to-edge wrapping. Rebuild it when aerosol changes.
 * Use standard QuadMesh UVs; TextureNode handles backend texture orientation.
 * Multiple scattering and ground interreflection are not modelled.
 *
 * @augments Node
 * @three_import import { AtmosphereNode } from 'three/addons/tsl/display/AtmosphereNode.js';
 */
class AtmosphereNode extends Node {

	static get type() {

		return 'AtmosphereNode';

	}

	/**
	 * @param {Texture} transmittanceTexture - The RGB transmittance LUT.
	 * @param {Node<vec3>} [directionNode=positionWorldDirection] - Viewing direction.
	 */
	constructor( transmittanceTexture, directionNode = positionWorldDirection ) {

		super( 'vec3' );
		this.transmittanceTexture = transmittanceTexture;
		this.directionNode = directionNode;
		/** @type {UniformNode<vec3>} Direction toward the sun in world space. */
		this.sunDirection = uniform( new Vector3( 0, 1, 0 ) );
		/** @type {UniformNode<float>} Observer altitude in kilometres. */
		this.observerAltitude = uniform( atmosphere.observerAltitude );
		/** @type {UniformNode<float>} Aerosol density multiplier. */
		this.aerosol = uniform( atmosphere.aerosol );
		/** @type {UniformNode<float>} Top-of-atmosphere solar illuminance in lux. */
		this.sunIlluminance = uniform( atmosphere.sunIlluminance );
		/** @type {UniformNode<float>} Radiance scale applied before half-float storage. */
		this.preExposure = uniform( 1 );

	}

	/**
	 * Shader used to render the transmittance LUT.
	 *
	 * @param {Node<vec2>} [uvNode=uv()] - Unit-square LUT coordinates.
	 * @return {Node<vec3>} RGB transmittance.
	 */
	getTransmittanceLUTNode( uvNode = uv() ) {

		return Fn( () => {

			const u = uvNode.x.mul( 2 ).sub( 1 );
			const cosine = u.sign().mul( u.mul( u ) );
			const radius = uvNode.y.mul( uvNode.y ).mul( atmosphereRadius - planetRadius ).add( planetRadius );
			return atmosphereTransmittance( radius, cosine, this.aerosol );

		} )();

	}

	setup() {

		return Fn( () => {

			const direction = this.directionNode;
			const radius = this.observerAltitude.add( planetRadius );
			const length = rayLength( radius, direction.y );
			const radiance = nightRadiance.toVar();
			const cosine = direction.dot( this.sunDirection );
			const rayleighPhase = cosine.mul( cosine ).add( 1 ).mul( 3 / ( 16 * Math.PI ) );
			const g = atmosphere.mieG;
			const miePhase = cosine.mul( cosine ).add( 1 ).mul( 3 / ( 8 * Math.PI ) * ( 1 - g * g ) / ( 2 + g * g ) )
				.div( float( 1 + g * g ).sub( cosine.mul( 2 * g ) ).pow( 1.5 ) );
			const step = length.div( 32 );
			const opticalDepth = vec3( 0 ).toVar();
			If( direction.y.greaterThanEqual( 0 ), () => {

				Loop( { start: int( 0 ), end: int( 32 ), type: 'int' }, ( { i } ) => {

					const distance = float( i ).add( 0.5 ).mul( step );
					const point = direction.mul( distance ).add( vec3( 0, radius, 0 ) );
					const pointRadius = point.length();
					const rho = density( pointRadius.sub( planetRadius ) );
					const sunCosine = point.dot( this.sunDirection ).div( pointRadius ).clamp( - 1, 1 );
					const lutUV = vec2( sunCosine.sign().mul( sunCosine.abs().sqrt() ).mul( 0.5 ).add( 0.5 ), pointRadius.sub( planetRadius ).div( atmosphereRadius - planetRadius ).max( 0 ).sqrt() );
					const sunlight = texture( this.transmittanceTexture, lutUV ).level( 0 ).rgb;
					opticalDepth.addAssign( rho.mul( step ).mul( 0.5 ) );
					const scattering = rayleigh.mul( rho.x ).mul( rayleighPhase )
						.add( rho.y.mul( atmosphere.mieScattering ).mul( this.aerosol ).mul( miePhase ) );
					radiance.addAssign( sunlight.mul( extinction( opticalDepth, this.aerosol ).negate().exp() ).mul( scattering ).mul( step ).mul( this.sunIlluminance ) );
					opticalDepth.addAssign( rho.mul( step ).mul( 0.5 ) );

				} );

			} );
			return radiance.mul( this.preExposure );

		} )();

	}

}

export { AtmosphereNode };
