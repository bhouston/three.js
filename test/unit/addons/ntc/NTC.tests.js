import { Matrix4 } from 'three';
import {
	base64FromBytes,
	bytesFromBase64,
	float32ToFloat16,
	float16ToFloat32,
	encodeFloat16Base64,
	decodeFloat16Base64,
	encodeUint8Base64,
	decodeUint8Base64
} from '../../../../examples/jsm/ntc/NTCBinaryCodec.js';
import {
	CHANNELS,
	MAX_TOTAL_CHANNELS,
	buildChannelActivations,
	getChannel,
	layoutChannels
} from '../../../../examples/jsm/ntc/NTCFormat.js';
import { packLayerBiasesVec4, packLayerWeightsMat4 } from '../../../../examples/jsm/ntc/NTCMLPTSL.js';
import { classifyMaterialChannels } from '../../../../examples/jsm/ntc/training/NTCSource.js';
import {
	DEFAULT_MIPS_PER_LEVEL,
	LATENT_INIT_SCALE,
	MAX_GRID_RESOLUTION,
	computeGridLevels,
	createLatentGrid
} from '../../../../examples/jsm/ntc/training/NTCGridModel.js';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';
import { activate, createMLP, forwardMLP, powerLog, sigmoid } from '../../../../examples/jsm/ntc/training/NTCMLP.js';
import { NTC_PROFILES, NTC_PROFILE_NAMES, getNTCProfile } from '../../../../examples/jsm/ntc/training/NTCProfiles.js';
import { DEFAULT_QUANTIZATION_OPTIONS, QUANTIZATION_SCHEMES, resolveQuantizationConfig } from '../../../../examples/jsm/ntc/training/NTCQuantization.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'NTC', () => {

		QUnit.module( 'NTCBinaryCodec', () => {

			QUnit.test( 'round-trips byte arrays through base64', ( assert ) => {

				const bytes = new Uint8Array( 50000 );
				for ( let i = 0; i < bytes.length; i ++ ) bytes[ i ] = i % 256;

				assert.deepEqual( Array.from( bytesFromBase64( base64FromBytes( bytes ) ) ), Array.from( bytes ), 'large byte array round-trips' );
				assert.equal( bytesFromBase64( base64FromBytes( new Uint8Array( 0 ) ) ).length, 0, 'empty byte array round-trips' );

			} );

			QUnit.test( 'round-trips float16 and uint8 payloads', ( assert ) => {

				for ( const value of [ 0, 1, - 1, 0.5, - 0.5, 2, 100, - 100 ] ) {

					assert.true( Math.abs( float16ToFloat32( float32ToFloat16( value ) ) - value ) <= 1e-5, `${ value } round-trips through float16` );

				}

				const floats = new Float32Array( [ 0, 1, - 1, 0.25, 3.5, - 7.125, 0.001 ] );
				const decodedFloats = decodeFloat16Base64( encodeFloat16Base64( floats ), floats.length );

				for ( let i = 0; i < floats.length; i ++ ) {

					assert.true( Math.abs( decodedFloats[ i ] - floats[ i ] ) < Math.max( 1e-3, Math.abs( floats[ i ] ) * 1e-2 ), `float16 payload value ${ i } stays within half precision` );

				}

				const min = - 2;
				const max = 3;
				const values = new Float32Array( [ - 2, - 1, 0, 0.5, 1, 2.9, 3 ] );
				const decodedUint8 = decodeUint8Base64( encodeUint8Base64( values, min, max ), min, max, values.length );
				const step = ( max - min ) / 255;

				for ( let i = 0; i < values.length; i ++ ) {

					assert.true( Math.abs( decodedUint8[ i ] - values[ i ] ) <= step / 2 + 1e-6, `uint8 payload value ${ i } stays within quantization step` );

				}

				assert.deepEqual( Array.from( decodeUint8Base64( encodeUint8Base64( new Float32Array( [ 5, 5 ] ), 5, 5 ), 5, 5, 2 ) ), [ 5, 5 ], 'min === max avoids division by zero' );

			} );

		} );

		QUnit.module( 'NTCFormat', () => {

			QUnit.test( 'lays out channel subsets contiguously', ( assert ) => {

				const subset = [ getChannel( 'normal' ), getChannel( 'albedo' ), getChannel( 'roughness' ) ];
				const layout = layoutChannels( subset );

				assert.equal( layout.channels[ 0 ].key, 'normal', 'keeps caller order' );
				assert.equal( layout.channels[ 0 ].offset, 0, 'first channel starts at zero' );
				assert.equal( layout.channels[ 1 ].offset, 2, 'second channel follows first size' );
				assert.equal( layout.totalChannels, 6, 'sums channel sizes' );
				assert.equal( layout.packCount, 2, 'pack count is ceil(totalChannels / 4)' );
				assert.deepEqual( layoutChannels( [] ), { channels: [], totalChannels: 0, packCount: 0 }, 'empty layout is valid' );
				assert.equal( MAX_TOTAL_CHANNELS, layoutChannels( CHANNELS ).totalChannels, 'MAX_TOTAL_CHANNELS matches the full channel list' );

			} );

			QUnit.test( 'builds output activations from channel metadata', ( assert ) => {

				const { channels } = layoutChannels( [ getChannel( 'albedo' ), getChannel( 'normal' ), getChannel( 'roughness' ) ] );

				assert.deepEqual( buildChannelActivations( channels ), [
					'sigmoid', 'sigmoid', 'sigmoid',
					'tanh', 'tanh',
					'sigmoid'
				], 'activations follow flat channel offsets' );
				assert.throws( () => getChannel( 'nonexistent' ), /unknown channel/, 'unknown channels throw' );

			} );

		} );

		QUnit.module( 'NTCSource', () => {

			QUnit.test( 'classifies active and constant channels', ( assert ) => {

				const material = { colorNode: {}, roughness: 0.4, side: 2, transparent: true };
				const result = classifyMaterialChannels( material );

				assert.deepEqual( result.activeChannels.map( ( channel ) => channel.key ), [ 'albedo' ], 'node-backed channel is active' );
				assert.equal( result.constantValues.roughness, 0.4, 'plain property resolves as a constant' );
				assert.deepEqual( result.renderFlags, { side: 2, transparent: true }, 'render flags are preserved' );

				const empty = classifyMaterialChannels( {} );
				assert.equal( empty.activeChannels.length, 0, 'empty material has no active channels' );
				assert.equal( Object.keys( empty.constantValues ).length, CHANNELS.length, 'empty material resolves every channel constant' );
				assert.deepEqual( empty.constantValues.normal, [ 0, 0, 1 ], 'default normal is preserved' );

			} );

		} );

		QUnit.module( 'NTCGridModel', () => {

			QUnit.test( 'computes grid levels with stock mip spacing', ( assert ) => {

				assert.deepEqual( computeGridLevels( 128, 4, 2 ), [ 128, 32, 8, 2 ], 'mipsPerLevel controls resolution stride' );
				assert.deepEqual( computeGridLevels( 128, 4 ), computeGridLevels( 128, 4, DEFAULT_MIPS_PER_LEVEL ), 'omitted mipsPerLevel uses default' );
				assert.deepEqual( computeGridLevels( 8, 6, 2 ), [ 8, 2, 1, 1, 1, 1 ], 'levels never underflow below 1' );
				assert.deepEqual( computeGridLevels( MAX_GRID_RESOLUTION + 1000, 1, 2 ), [ MAX_GRID_RESOLUTION ], 'base resolution is capped' );

			} );

			QUnit.test( 'creates latent grids with deterministic initialization', ( assert ) => {

				let calls = 0;
				const grid = createLatentGrid( 3, 2, 5, () => {

					calls ++;
					return 0.5;

				} );

				assert.equal( grid.data.length, 3 * 2 * 5, 'grid data length is width * height * channels' );
				assert.true( grid.data instanceof Float32Array, 'grid data is Float32Array' );
				assert.deepEqual( Array.from( createLatentGrid( 1, 1, 4, () => 0.5 ).data ), [ 0, 0, 0, 0 ], 'random midpoint initializes to zero' );
				assert.true( Math.abs( createLatentGrid( 1, 1, 1, () => 0 ).data[ 0 ] + LATENT_INIT_SCALE ) <= 1e-7, 'random zero reaches negative scale' );
				assert.equal( calls, 3 * 2 * 5, 'random is called once per latent value' );

			} );

		} );

		QUnit.module( 'NTCMLP', () => {

			QUnit.test( 'evaluates activations and forward propagation', ( assert ) => {

				assert.equal( activate( 2.5, 'relu' ), 2.5, 'relu keeps positive values' );
				assert.equal( activate( - 1.5, 'relu' ), 0, 'relu clamps negative values' );
				assert.true( Math.abs( sigmoid( 0 ) - 0.5 ) <= 1e-6, 'sigmoid midpoint is 0.5' );
				assert.equal( powerLog( 1, 3 ), 0, 'powerLog(1) is zero' );

				const mlp = createMLP( 2, [ 3 ], 1, () => 0.75, 'relu', 'linear' );
				const run = forwardMLP( mlp, [ 1, 0.5 ] );

				assert.equal( run.activations.length, 3, 'forward pass stores input, hidden and output activations' );
				assert.equal( run.output.length, 1, 'forward pass produces requested output size' );

			} );

			QUnit.test( 'packs weights and biases into fp32 mat4/vec4 blocks', ( assert ) => {

				const inputSize = 6;
				const outputSize = 5;
				const weights = new Array( outputSize * inputSize );

				for ( let o = 0; o < outputSize; o ++ ) {

					for ( let i = 0; i < inputSize; i ++ ) weights[ o * inputSize + i ] = o * inputSize + i + 1;

				}

				const packed = packLayerWeightsMat4( weights, inputSize, outputSize );

				assert.equal( packed.length, 4, 'ceil(output/4) * ceil(input/4) mat4 blocks are emitted' );
				assert.true( packed[ 0 ] instanceof Matrix4, 'weight blocks are Matrix4 values' );
				assert.true( packed[ 0 ].equals( new Matrix4().set(
					1, 2, 3, 4,
					7, 8, 9, 10,
					13, 14, 15, 16,
					19, 20, 21, 22
				) ), 'first block preserves row-major weights' );
				assert.true( packed[ 3 ].equals( new Matrix4().set(
					29, 30, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0
				) ), 'last block is zero padded' );

				const biases = packLayerBiasesVec4( [ 1, 2, 3, 4, 5 ] );
				assert.deepEqual( [ biases[ 0 ].x, biases[ 0 ].y, biases[ 0 ].z, biases[ 0 ].w ], [ 1, 2, 3, 4 ], 'first bias vec4 is full' );
				assert.deepEqual( [ biases[ 1 ].x, biases[ 1 ].y, biases[ 1 ].z, biases[ 1 ].w ], [ 5, 0, 0, 0 ], 'last bias vec4 is padded' );

			} );

		} );

		QUnit.module( 'NTCQuantization', () => {

			QUnit.test( 'quantizes CPU values and resolves config', ( assert ) => {

				const lo = - 2;
				const hi = 3;
				const step = ( hi - lo ) / 255;

				for ( const value of [ - 2, - 1, 0, 0.37, 1, 2.9, 3 ] ) {

					const quantized = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( value, lo, hi );
					assert.true( quantized >= lo - 1e-9 && quantized <= hi + 1e-9, `${ value } stays in range` );
					assert.true( Math.abs( quantized - value ) <= step / 2 + 1e-9, `${ value } stays within half a step` );

				}

				assert.equal( QUANTIZATION_SCHEMES.none.quantizeForwardCPU( 0.123, - 1, 1 ), 0.123, 'none is identity' );
				assert.deepEqual( resolveQuantizationConfig(), DEFAULT_QUANTIZATION_OPTIONS, 'defaults are stable' );
				assert.deepEqual(
					resolveQuantizationConfig( { quantization: { mode: 'uint8', range: [ - 1, 1 ] } } ),
					{ mode: 'uint8', target: 'latents', range: [ - 1, 1 ], perLevel: true },
					'explicit uint8 config resolves defaults'
				);
				assert.throws( () => resolveQuantizationConfig( { quantization: { mode: 'int4' } } ), /quantization\.mode/, 'invalid modes throw' );
				assert.throws( () => resolveQuantizationConfig( { quantization: { range: [ 1, 0 ] } } ), /quantization\.range/, 'invalid ranges throw' );

			} );

		} );

		QUnit.module( 'NTCProfiles', () => {

			QUnit.test( 'exposes valid named profiles', ( assert ) => {

				assert.deepEqual( NTC_PROFILE_NAMES, [ 'mobile-fast', 'mobile-balanced', 'desktop-quality' ], 'profile names are documented order' );
				assert.deepEqual( Object.keys( NTC_PROFILES ), NTC_PROFILE_NAMES, 'profile table matches names' );
				assert.equal( getNTCProfile( 'not-a-real-profile' ), null, 'unknown profile returns null' );

				const options = getNTCProfile( 'desktop-quality' );
				options.hiddenSizes.push( 999 );
				assert.false( NTC_PROFILES[ 'desktop-quality' ].hiddenSizes.includes( 999 ), 'profile hidden sizes are copied' );

				for ( const name of NTC_PROFILE_NAMES ) {

					const profile = NTC_PROFILES[ name ];
					const model = createNTCGridPyramidModel( { channels: 4, outputChannels: 3, ...getNTCProfile( name ) }, () => 0.5 );

					assert.deepEqual( model.hiddenSizes, profile.hiddenSizes, `${ name } hidden sizes apply to model` );
					assert.equal( model.hiddenActivation, profile.hiddenActivation, `${ name } activation applies to model` );
					assert.equal( model.resolutions[ 0 ], profile.baseResolution, `${ name } base resolution applies to model` );

				}

			} );

		} );

	} );

} );
