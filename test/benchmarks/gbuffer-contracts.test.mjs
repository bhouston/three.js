import assert from 'node:assert/strict';
import test from 'node:test';
import { encodeOct, decodeOct, plan, HistoryState, textureBytes, trackTextures } from '../../examples/jsm/gbuffer/contracts.js';

function normalize( vector ) {

	const length = Math.hypot( ...vector );
	return vector.map( value => value / length );

}

test( 'octahedral normals survive RG16F storage across poles, seams and both hemispheres', () => {

	const directions = [
		[ 0, 0, 1 ], [ 0, 0, - 1 ], [ - 0, - 0, - 1 ],
		[ 1, 0, 0 ], [ - 1, 0, 0 ], [ 0, 1, 0 ], [ 0, - 1, 0 ],
		[ 0, 1, - 1 ], [ 0, - 1, - 1 ], [ 1, 0, - 1 ], [ - 1, 0, - 1 ],
		[ 1e-12, - 1e-12, - 1 ], [ - 1e-12, 1e-12, - 1 ],
		[ 1, 1, 1e-12 ], [ 1, 1, - 1e-12 ]
	];
	let seed = 0x31415926;
	const random = () => {

		seed ^= seed << 13;
		seed ^= seed >>> 17;
		seed ^= seed << 5;
		return ( seed >>> 0 ) / 0x100000000;

	};

	for ( let i = 0; i < 4096; i ++ ) {

		const z = random() * 2 - 1;
		const angle = random() * Math.PI * 2;
		const radius = Math.sqrt( 1 - z * z );
		directions.push( [ radius * Math.cos( angle ), radius * Math.sin( angle ), z ] );

	}

	for ( const direction of directions ) {

		const normal = normalize( direction );
		const encoded = encodeOct( normal );
		assert.ok( encoded.every( value => Number.isFinite( value ) && Math.abs( value ) <= 1 ) );
		const decoded = decodeOct( encoded );
		assert.ok( Math.hypot( ...decoded.map( ( value, axis ) => value - normal[ axis ] ) ) < 1e-12 );
		// Exercise actual binary16 storage rather than a float32-only roundtrip.
		const stored = new globalThis.Float16Array( encoded );
		const quantized = decodeOct( stored );
		assert.ok( quantized.every( Number.isFinite ) );
		assert.ok( Math.abs( Math.hypot( ...quantized ) - 1 ) < 1e-12 );
		const dot = quantized.reduce( ( sum, value, axis ) => sum + value * normal[ axis ], 0 );
		const degrees = Math.acos( Math.min( 1, Math.max( - 1, dot ) ) ) * 180 / Math.PI;
		assert.ok( degrees < 0.15, `RG16F angular error ${ degrees } degrees for ${ normal }` );

	}

} );

test( 'mobile and forward presets never retain deferred-only stages regardless of requested switches', () => {

	for ( const mobile of [ false, true ] ) {
		for ( const mode of [ 'forward', 'deferred' ] ) {
			for ( const passMode of [ 'bypass', 'materialized' ] ) {
				for ( let switches = 0; switches < 16; switches ++ ) {

					const options = { mobile, mode, passMode, prepass: !! ( switches & 1 ), ao: !! ( switches & 2 ), traa: !! ( switches & 4 ), shadows: !! ( switches & 8 ) };
					const active = plan( options );
					assert.equal( active.motion, options.traa, 'AA motion inputs must follow AA in every preset' );
					assert.equal( active.shadows, options.shadows, 'preset changes must preserve matched lighting' );

					if ( mobile || mode === 'forward' ) {

						assert.equal( active.deferred, false );
						assert.ok( ! active.prepass && ! active.ao && ! active.materialized, JSON.stringify( options ) );

					} else {

						assert.equal( active.deferred, true );
						assert.equal( active.prepass, options.prepass );
						assert.equal( active.ao, options.ao );
						assert.equal( active.materialized, passMode === 'materialized' );

					}

				}
			}
		}
	}

} );

test( 'published source history owns snapshots and never aliases the current write slot', () => {

	const history = new HistoryState();
	assert.equal( history.valid, false );
	const geometry = new Float32Array( [ 1, 2, 3 ] );
	const source = { frame: 10, camera: { projection: [ 1, 0, 0, 1 ] }, geometry };
	const originalWrite = history.write;
	history.publish( source );
	assert.equal( history.read, originalWrite );
	assert.notEqual( history.read, history.write );
	assert.equal( history.valid, true );
	const first = history.metadata[ history.read ];
	source.frame = 99;
	source.camera.projection[ 0 ] = 7;
	geometry[ 0 ] = 9;
	assert.equal( first.frame, 10 );
	assert.equal( first.camera.projection[ 0 ], 1 );
	assert.equal( first.geometry[ 0 ], 1 );
	assert.notEqual( first.geometry.buffer, geometry.buffer );

	for ( let frame = 11; frame < 30; frame ++ ) {

		const previousRead = history.read;
		const previousMetadata = globalThis.structuredClone( history.metadata[ previousRead ] );
		history.publish( { frame, camera: { projection: [ frame ] }, geometry: new Float32Array( [ frame ] ) } );
		assert.notEqual( history.read, previousRead );
		assert.equal( history.write, previousRead );
		assert.notEqual( history.read, history.write );
		assert.equal( history.metadata[ history.read ].frame, frame );
		assert.deepEqual( history.metadata[ history.write ], previousMetadata, 'publishing may not mutate the previous source' );

	}

	history.reset( 'camera cut / model change / resize' );
	assert.equal( history.valid, false );
	assert.ok( history.metadata.every( metadata => metadata === null ), 'no previous geometry or camera metadata survives reset' );
	assert.notEqual( history.read, history.write );
	history.publish( { frame: 100, camera: {}, geometry: [] } );
	assert.equal( history.metadata[ history.read ].frame, 100 );
	assert.equal( history.metadata[ history.write ], null );

} );

test( 'logical texture bytes account for odd mip extents, array layers, samples and packed HDR', () => {

	// 7*5 + 3*2 + 1*1 texels in each of three independent array layers.
	assert.deepEqual( textureBytes( { format: 'rgba16float', size: { width: 7, height: 5, depthOrArrayLayers: 3 }, mipLevelCount: 3 } ), { bytes: 1008, exact: true } );
	assert.deepEqual( textureBytes( { format: 'rgba16float', size: [ 7, 5 ], sampleCount: 4 } ), { bytes: 1120, exact: true } );
	assert.deepEqual( textureBytes( { format: 'rg11b10ufloat', size: [ 7, 5 ] } ), { bytes: 140, exact: true } );
	assert.deepEqual( textureBytes( { format: 'depth32float', size: [ 7, 5 ] } ), { bytes: 140, exact: true } );

} );

test( 'volume mip depth shrinks while array layer count stays fixed', () => {

	// Three-dimensional mip volumes are 8*4*4, 4*2*2, and 2*1*1 texels.
	assert.deepEqual( textureBytes( { dimension: '3d', format: 'rgba8unorm', size: [ 8, 4, 4 ], mipLevelCount: 3 } ), { bytes: 584, exact: true } );
	assert.deepEqual( textureBytes( { dimension: '2d', format: 'rgba8unorm', size: [ 8, 4, 4 ], mipLevelCount: 3 } ), { bytes: 672, exact: true } );

} );

test( 'compressed textures charge whole blocks for every mip and flag unsupported estimates', () => {

	// 7x5 uses four BC1 blocks, and each smaller mip still uses one block.
	assert.deepEqual( textureBytes( { format: 'bc1-rgba-unorm', size: [ 7, 5, 3 ], mipLevelCount: 3 } ), { bytes: 144, exact: true } );
	assert.deepEqual( textureBytes( { format: 'bc7-rgba-unorm', size: [ 7, 5, 3 ], mipLevelCount: 3 } ), { bytes: 288, exact: true } );
	assert.deepEqual( textureBytes( { format: 'etc2-rgb8unorm', size: [ 1, 1 ] } ), { bytes: 8, exact: true } );
	assert.deepEqual( textureBytes( { format: 'eac-rg11unorm', size: [ 1, 1 ] } ), { bytes: 16, exact: true } );
	assert.equal( textureBytes( { format: 'unknown-format', size: [ 7, 5 ] } ).exact, false );
	assert.equal( textureBytes( { format: 'astc-8x8-unorm', size: [ 7, 5 ] } ).exact, false, 'unsupported ASTC blocks may not be reported as exact' );

} );

test( 'texture tracking includes overlapping allocations and releases disabled targets', () => {

	let destroyed = 0;
	const createResource = () => ( { destroy() { destroyed ++; } } );
	const device = { createTexture: createResource, createBuffer: createResource };
	const tracker = trackTextures( device );
	const descriptor = { label: 'old-model', format: 'rgba16float', size: [ 7, 5 ] };
	const oldModel = device.createTexture( descriptor );
	descriptor.size[ 0 ] = 999;
	assert.deepEqual( tracker.snapshot().textures[ 0 ].size, [ 7, 5 ], 'allocation records own their descriptor snapshot' );
	const newModel = device.createTexture( { label: 'new-model', format: 'rgba8unorm', size: [ 10, 10 ] } );
	assert.equal( tracker.snapshot().liveBytes, 680 );
	assert.equal( tracker.snapshot().peakBytes, 680 );
	oldModel.destroy();
	assert.equal( tracker.snapshot().liveBytes, 400 );
	assert.equal( tracker.snapshot().peakBytes, 680, 'a model switch must retain its overlapping-load peak' );
	tracker.resetPeak();
	assert.equal( tracker.snapshot().peakBytes, 400 );
	assert.equal( tracker.snapshot().sessionPeakBytes, 680, 'benchmark peak resets may not erase model-load overlap evidence' );
	const bufferDescriptor = { label: 'previous geometry', size: 1024, usage: 128 };
	const buffer = device.createBuffer( bufferDescriptor );
	bufferDescriptor.size = 1;
	assert.equal( tracker.snapshot().bufferBytes, 1024 );
	assert.equal( tracker.snapshot().bufferPeakBytes, 1024 );
	assert.equal( tracker.snapshot().liveBytes, 400, 'texture and geometry buffer totals must remain distinguishable' );
	buffer.destroy();
	assert.equal( tracker.snapshot().bufferBytes, 0 );
	assert.deepEqual( tracker.snapshot().buffers, [] );
	newModel.destroy();
	assert.equal( tracker.snapshot().liveBytes, 0 );
	assert.deepEqual( tracker.snapshot().textures, [] );
	assert.equal( destroyed, 3 );

} );
