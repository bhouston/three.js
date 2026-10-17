// Pure contracts shared by the example, accounting and targeted tests.
export function encodeOct( [ x, y, z ] ) {

	const d = Math.abs( x ) + Math.abs( y ) + Math.abs( z );
	x /= d; y /= d; z /= d;
	return z < 0 ? [ ( 1 - Math.abs( y ) ) * ( x < 0 ? - 1 : 1 ), ( 1 - Math.abs( x ) ) * ( y < 0 ? - 1 : 1 ) ] : [ x, y ];

}

export function decodeOct( [ x, y ] ) {

	const z = 1 - Math.abs( x ) - Math.abs( y );
	const t = Math.max( - z, 0 );
	x += x >= 0 ? - t : t;
	y += y >= 0 ? - t : t;
	const d = Math.hypot( x, y, z );
	return [ x / d, y / d, z / d ];

}

export function plan( options ) {

	const deferred = ! options.mobile && options.mode === 'deferred';
	return {
		deferred,
		prepass: deferred && options.prepass,
		ao: deferred && options.ao,
		materialized: deferred && options.passMode === 'materialized',
		motion: options.traa,
		shadows: options.shadows
	};

}

export class HistoryState {

	constructor() {

		this.reset( 'initial' );

	}

	reset( reason ) {

		this.read = 0;
		this.write = 1;
		this.valid = false;
		this.reason = reason;
		this.metadata = [ null, null ];

	}

	publish( metadata ) {

		this.metadata[ this.write ] = structuredClone( metadata );
		[ this.read, this.write ] = [ this.write, this.read ];
		this.valid = true;

	}

}

export function textureBytes( descriptor ) {

	const { format, size, mipLevelCount = 1, sampleCount = 1 } = descriptor;
	const [ width, height, depth = 1 ] = Array.isArray( size ) ? size : [ size.width, size.height, size.depthOrArrayLayers || 1 ];
	const block = /bc1|bc4|etc2-rgb8|eac-r11/.test( format ) ? 8 : 16;
	const compressed = /^(bc|etc|eac|astc)/.test( format );
	const bytes = { r8unorm: 1, rg8unorm: 2, rgba8unorm: 4, 'rgba8unorm-srgb': 4, bgra8unorm: 4, 'bgra8unorm-srgb': 4, r16float: 2, rg16float: 4, rgba16float: 8, r32float: 4, rg32float: 8, rgba32float: 16, rg11b10ufloat: 4, rgb10a2unorm: 4, depth16unorm: 2, depth24plus: 4, depth32float: 4, 'depth24plus-stencil8': 4, 'depth32float-stencil8': 8 }[ format ];
	let total = 0;
	for ( let mip = 0; mip < mipLevelCount; mip ++ ) {

		const w = Math.max( 1, width >> mip ), h = Math.max( 1, height >> mip );
		total += ( compressed ? Math.ceil( w / 4 ) * Math.ceil( h / 4 ) * block : w * h * ( bytes || 0 ) ) * depth * sampleCount;

	}

	return { bytes: total, exact: compressed ? ! format.startsWith( 'astc' ) : bytes !== undefined };

}

// Capture every actual WebGPU texture, including renderer internals and overlapping model loads.
// Logical bytes exclude driver alignment, heap fragmentation and swap-chain-owned textures.
export function trackTextures( device ) {

	const original = device.createTexture.bind( device );
	const live = new Map();
	let sequence = 0, peak = 0;
	device.createTexture = ( descriptor ) => {

		const resource = original( descriptor );
		const id = ++ sequence;
		const record = { id, ...structuredClone( descriptor ), ...textureBytes( descriptor ) };
		live.set( id, record );
		peak = Math.max( peak, [ ...live.values() ].reduce( ( sum, entry ) => sum + entry.bytes, 0 ) );
		const destroy = resource.destroy.bind( resource );
		resource.destroy = () => {

			live.delete( id );
			destroy();

		};

		return resource;

	};

	return { snapshot: () => ( { liveBytes: [ ...live.values() ].reduce( ( sum, entry ) => sum + entry.bytes, 0 ), peakBytes: peak, textures: [ ...live.values() ] } ), resetPeak: () => {

		peak = [ ...live.values() ].reduce( ( sum, entry ) => sum + entry.bytes, 0 );

	} };

}
