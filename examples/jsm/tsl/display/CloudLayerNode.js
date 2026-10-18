import { Node, Vector2 } from 'three/webgpu';
import { Fn, If, float, vec3, vec4, uniform, mx_fractal_noise_float, smoothstep, mix } from 'three/tsl';

/**
 * Static, thin spherical cloud layer for the physical atmosphere example.
 * RGB is composited radiance; alpha is view transmittance. No solar disc is added.
 * Distances are metres. This is an artistic single-layer approximation, not
 * volumetric transport. Bake it only when the atmosphere or cloud inputs change.
 *
 * @augments Node
 * @three_import import { CloudLayerNode } from 'three/addons/tsl/display/CloudLayerNode.js';
 */
class CloudLayerNode extends Node {

	static get type() {

		return 'CloudLayerNode';

	}

	constructor( skyNode, directionNode, sunDirectionNode, sunIrradianceNode ) {

		super( 'vec4' );
		this.skyNode = skyNode;
		this.directionNode = directionNode;
		this.sunDirectionNode = sunDirectionNode;
		this.sunIrradianceNode = sunIrradianceNode;
		this.coverage = uniform( 0.5 );
		this.altitude = uniform( 2000 );
		this.observerAltitude = uniform( 2 );
		this.density = uniform( 2 );
		this.noiseScale = uniform( 2500 );
		this.noiseOffset = uniform( new Vector2() );

	}

	getTransmittanceNode( direction ) {

		return Fn( () => {

			const transmission = float( 1 ).toVar();
			If( direction.y.greaterThan( 0 ).and( this.coverage.greaterThan( 0 ) ), () => {

				const radius = this.observerAltitude.add( 6371000 );
				const shell = this.altitude.max( this.observerAltitude.add( 1 ) ).add( 6371000 );
				const b = radius.mul( direction.y );
				const distance = b.mul( b ).add( shell.sub( radius ).mul( shell.add( radius ) ) ).sqrt().sub( b );
				const point = direction.mul( distance );
				const noise = mx_fractal_noise_float( vec3( point.xz.div( this.noiseScale ).add( this.noiseOffset ), 0 ), 5 ).mul( 0.5 ).add( 0.5 );
				const threshold = float( 1.1 ).sub( this.coverage.mul( 1.2 ) );
				const amount = this.coverage.greaterThanEqual( 1 ).select( float( 1 ), smoothstep( threshold.sub( 0.18 ), threshold.add( 0.18 ), noise ) );
				const cosine = direction.y.mul( direction.y ).add( float( 1 ).sub( radius.div( shell ).pow( 2 ) ).mul( float( 1 ).sub( direction.y.mul( direction.y ) ) ) ).sqrt().max( 0.04 );
				transmission.assign( amount.mul( this.density ).div( cosine ).negate().exp() );

			} );
			return transmission;

		} )();

	}

	setup() {

		return Fn( () => {

			const transmission = this.getTransmittanceNode( this.directionNode ).toVar();
			const cosine = this.directionNode.dot( this.sunDirectionNode );
			// Broad dual-lobe scattering avoids a second bright, unresolved sun in IBL.
			const phase = float( 0.64 ).div( float( 1.36 ).sub( cosine.mul( 1.2 ) ).pow( 1.5 ) ).mul( 0.8 )
				.add( float( 0.96 ).div( float( 1.04 ).add( cosine.mul( 0.4 ) ).pow( 1.5 ) ).mul( 0.2 ) ).div( 4 * Math.PI );
			const baseLight = this.sunIrradianceNode.mul( transmission.sqrt().mul( 0.12 ).add( 0.035 ) ).div( Math.PI );
			const silver = this.sunIrradianceNode.mul( phase ).mul( transmission.sqrt().mul( 0.7 ).add( 0.3 ) );
			const ambient = this.skyNode.mul( 0.6 ).add( baseLight );
			const cloud = ambient.add( silver );
			return vec4( mix( cloud, this.skyNode, transmission ), transmission );

		} )();

	}

}

export { CloudLayerNode };
