import { sampleExponentialMipLevel } from './NeuralAppearanceFilterUtils.js';

const DEFAULT_MINIMUM_TRAINING_COSINE = 0.05;

async function generateTrainingSamples( options, teacher, random, iteration = 0 ) {

	const batchSize = options.batchSize;
	const mipLevelCount = getMipLevelCount( options.resolution, options.resolution );
	const recordCount = options.fixedTrainingMip >= 0 || options.sampleAllMips === true ?
		batchSize :
		batchSize * mipLevelCount;
	const gridSize = Math.max( 1, Math.floor( Math.sqrt( recordCount ) ) );
	const samples = [];
	const augmentationRatio = options.colorAugmentation && teacher.supportsColorAugmentation === true ? 0.5 * ( 1 + Math.cos( Math.PI * Math.min( iteration / Math.max( 1, options.iterations ), 1 ) ) ) : 0;

	for ( let i = 0; i < recordCount; i ++ ) {

		const x = i % gridSize;
		const y = Math.floor( i / gridSize ) % gridSize;
		const uv = [
			( x + random() ) / gridSize,
			( y + random() ) / gridSize
		];
		const directions = sampleTeacherDirections( random );
		const sampledMip = options.fixedTrainingMip < 0 && options.sampleAllMips !== true ?
			sampleExponentialMipLevel( random, mipLevelCount, options.mipSamplingDecay ) :
			0;
		const mipLevels = options.fixedTrainingMip >= 0 ?
			[ options.fixedTrainingMip ] :
			( options.sampleAllMips === true ? createMipLevelIndices( mipLevelCount ) : [ sampledMip ] );

		for ( const mip of mipLevels ) {

			const footprint = Math.pow( 2, mip ) / Math.max( 1, options.resolution );

			samples.push( {
				uv: uv.slice(),
				wi: directions.wi.slice(),
				wo: directions.wo.slice(),
				normal: [ 0, 0, 1 ],
				tangent: [ 1, 0, 0 ],
				bitangent: [ 0, 1, 0 ],
				duvDx: [ footprint, 0 ],
				duvDy: [ 0, footprint ],
				mip,
				encoderInputs: teacher.encodeInputs ? teacher.encodeInputs( uv ) : [ uv[ 0 ], uv[ 1 ] ]
			} );

		}

	}

	await assignTeacherTargets( samples, teacher );
	normalizeDirectLightingTargets( samples, options.minimumTrainingCosine, options.highlightLossScale );
	await assignAuxiliaryTeacherTargets( samples, teacher );

	for ( const sample of samples ) {

		if ( random() < augmentationRatio ) {

			augmentColorChannels( sample, random );

		}

	}

	await assignIBLTeacherTargets( samples, teacher, random );

	return samples;

}

async function generateValidationSamples( options, teacher ) {

	const sampleCount = options.batchSize;
	const gridSize = Math.max( 1, Math.ceil( Math.sqrt( sampleCount ) ) );
	const mipLevelCount = getMipLevelCount( options.resolution, options.resolution );
	const mipLevels = options.fixedTrainingMip >= 0 ?
		[ options.fixedTrainingMip ] :
		createMipLevelIndices( mipLevelCount );
	const cosines = [ 0.025, 0.1, 0.4, 0.8 ];
	const samples = [];

	for ( let i = 0; i < sampleCount; i ++ ) {

		const uv = [
			( i % gridSize + 0.5 ) / gridSize,
			( Math.floor( i / gridSize ) + 0.5 ) / gridSize
		];
		const wiCosine = cosines[ i % cosines.length ];
		const woCosine = cosines[ Math.floor( i / cosines.length ) % cosines.length ];
		const azimuth = 2 * Math.PI * ( i + 0.5 ) / sampleCount;
		const wi = directionFromCosine( wiCosine, azimuth );
		const wo = directionFromCosine( woCosine, azimuth * 1.61803398875 );

		for ( const mip of mipLevels ) {

			const footprint = Math.pow( 2, mip ) / Math.max( 1, options.resolution );
			samples.push( {
				uv: uv.slice(),
				wi: wi.slice(),
				wo: wo.slice(),
				normal: [ 0, 0, 1 ],
				tangent: [ 1, 0, 0 ],
				bitangent: [ 0, 1, 0 ],
				duvDx: [ footprint, 0 ],
				duvDy: [ 0, footprint ],
				mip,
				encoderInputs: teacher.encodeInputs ? teacher.encodeInputs( uv ) : [ uv[ 0 ], uv[ 1 ] ]
			} );

		}

	}

	await assignTeacherTargets( samples, teacher );
	normalizeDirectLightingTargets( samples, options.minimumTrainingCosine );
	await assignAuxiliaryTeacherTargets( samples, teacher );
	await assignIBLTeacherTargets( samples, teacher );

	return samples;

}

async function generateIBLTrainingSamples( options, teacher, random ) {

	const queryCount = Math.max( 1, Math.min( options.iblSampleCount || options.batchSize, options.batchSize ) );
	const gridSize = Math.max( 1, Math.ceil( Math.sqrt( queryCount ) ) );
	const samples = [];

	for ( let i = 0; i < queryCount; i ++ ) {

		const uv = [
			( i % gridSize + random() ) / gridSize,
			( Math.floor( i / gridSize ) + random() ) / gridSize
		];
		const wo = sampleHemisphereCosine( random );
		const footprint = 1 / Math.max( 1, options.resolution );

		samples.push( {
			uv,
			wi: [ 0, 0, 1 ],
			wo,
			normal: [ 0, 0, 1 ],
			tangent: [ 1, 0, 0 ],
			bitangent: [ 0, 1, 0 ],
			duvDx: [ footprint, 0 ],
			duvDy: [ 0, footprint ],
			mip: 0,
			weight: 0,
			target: [ 0, 0, 0 ],
			encoderInputs: teacher.encodeInputs ? teacher.encodeInputs( uv ) : [ uv[ 0 ], uv[ 1 ] ]
		} );

	}

	await assignIBLTeacherTargets( samples, teacher, random );

	return samples;

}

async function assignIBLTeacherTargets( samples, teacher, random = null ) {

	if ( teacher.supportsIBL !== true || typeof teacher.evaluateBatch !== 'function' ) {

		for ( const sample of samples ) {

			sample.iblWeight = 0;
			sample.iblDirection = [ 0, 0, 1 ];
			sample.iblRoughness = 1;
			sample.iblIncoming = [ 0, 0, 0 ];
			sample.iblIrradiance = [ 0, 0, 0 ];
			sample.iblIndirectRadiance = [ 0, 0, 0 ];
			sample.iblIndirectIrradiance = [ 0, 0, 0 ];

		}

		return;

	}

	const teacherSamples = samples.map( ( sample, i ) => createIBLTeacherSample( sample, random || createIndexRandom( i ) ) );
	const queries = await teacher.evaluateBatch( teacherSamples, 'iblQuery' );
	const incoming = await teacher.evaluateBatch( teacherSamples, 'iblIncoming' );
	const irradiance = await teacher.evaluateBatch( teacherSamples, 'iblIrradiance' );
	const indirectRadiance = await teacher.evaluateBatch( teacherSamples, 'iblIndirectRadiance' );
	const indirectIrradiance = await teacher.evaluateBatch( teacherSamples, 'iblIndirectIrradiance' );

	for ( let i = 0; i < samples.length; i ++ ) {

		const query = queries[ i ] || [];
		const direction = toCanonicalDirection( query.slice( 0, 3 ), teacherSamples[ i ] );
		const roughness = Math.min( Math.max( Number( query[ 3 ] ), 0 ), 1 );
		samples[ i ].iblWeight = Number.isFinite( roughness ) && direction.every( Number.isFinite ) ? 1 : 0;
		samples[ i ].iblDirection = direction;
		samples[ i ].iblRoughness = roughness;
		samples[ i ].iblIncoming = ( incoming[ i ] || [ 0, 0, 0 ] ).slice( 0, 3 );
		samples[ i ].iblIrradiance = ( irradiance[ i ] || [ 0, 0, 0 ] ).slice( 0, 3 );
		samples[ i ].iblIndirectRadiance = ( indirectRadiance[ i ] || [ 0, 0, 0 ] ).slice( 0, 3 );
		samples[ i ].iblIndirectIrradiance = ( indirectIrradiance[ i ] || [ 0, 0, 0 ] ).slice( 0, 3 );

	}

}

function createIBLTeacherSample( sample, random ) {

	const normal = randomUnitVector( random );
	const tangent = orthonormalTangent( normal, random );
	const bitangent = normalize( cross( normal, tangent ) );

	return {
		...sample,
		normal,
		tangent,
		bitangent,
		wi: rotateByFrame( sample.wi, tangent, bitangent, normal ),
		wo: rotateByFrame( sample.wo, tangent, bitangent, normal )
	};

}

function toCanonicalDirection( direction, frame ) {

	const view = normalize( direction.length >= 3 ? direction : [ 0, 0, 1 ] );

	return normalize( [
		dot( view, frame.tangent ),
		dot( view, frame.bitangent ),
		dot( view, frame.normal )
	] );

}

function rotateByFrame( vector, tangent, bitangent, normal ) {

	const value = vector || [ 0, 0, 1 ];

	return [
		tangent[ 0 ] * value[ 0 ] + bitangent[ 0 ] * value[ 1 ] + normal[ 0 ] * value[ 2 ],
		tangent[ 1 ] * value[ 0 ] + bitangent[ 1 ] * value[ 1 ] + normal[ 1 ] * value[ 2 ],
		tangent[ 2 ] * value[ 0 ] + bitangent[ 2 ] * value[ 1 ] + normal[ 2 ] * value[ 2 ]
	];

}

function randomUnitVector( random ) {

	const z = 2 * random() - 1;
	const azimuth = 2 * Math.PI * random();
	const radial = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ radial * Math.cos( azimuth ), radial * Math.sin( azimuth ), z ];

}

function orthonormalTangent( normal, random ) {

	const axis = Math.abs( normal[ 2 ] ) < 0.9 ? [ 0, 0, 1 ] : [ 1, 0, 0 ];
	const tangent = normalize( cross( axis, normal ) );
	const bitangent = cross( normal, tangent );
	const angle = 2 * Math.PI * random();
	const cos = Math.cos( angle );
	const sin = Math.sin( angle );

	return normalize( [
		tangent[ 0 ] * cos + bitangent[ 0 ] * sin,
		tangent[ 1 ] * cos + bitangent[ 1 ] * sin,
		tangent[ 2 ] * cos + bitangent[ 2 ] * sin
	] );

}

function createIndexRandom( index ) {

	let state = ( index + 1 ) * 747796405 >>> 0;

	return function random() {

		state = Math.imul( state ^ state >>> 15, 1 | state );
		state ^= state + Math.imul( state ^ state >>> 7, 61 | state );

		return ( ( state ^ state >>> 14 ) >>> 0 ) / 4294967296;

	};

}

function cross( a, b ) {

	return [
		a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ],
		a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ],
		a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ]
	];

}

async function assignTeacherTargets( samples, teacher ) {

	const targets = teacher.evaluateBatch ? await teacher.evaluateBatch( samples ) : await Promise.all( samples.map( ( sample ) => teacher.evaluate( sample ) ) );

	for ( let i = 0; i < samples.length; i ++ ) {

		samples[ i ].target = targets[ i ].slice( 0, 3 );

	}

}

async function assignAuxiliaryTeacherTargets( samples, teacher ) {

	if ( teacher.supportsEmission === true ) {

		const targets = await teacher.evaluateBatch( samples, 'emission' );

		for ( let i = 0; i < samples.length; i ++ ) {

			samples[ i ].emissionTarget = targets[ i ].slice( 0, 3 );

		}

	}

	if ( teacher.supportsOpacity === true ) {

		const targets = await teacher.evaluateBatch( samples, 'opacity' );

		for ( let i = 0; i < samples.length; i ++ ) {

			samples[ i ].opacityTarget = Math.min( Math.max( targets[ i ][ 0 ], 0 ), 1 );

		}

	}

}

function normalizeDirectLightingTargets( samples, minimumCosine = DEFAULT_MINIMUM_TRAINING_COSINE, highlightLossScale = 0 ) {

	for ( const sample of samples ) {

		const nDotL = Math.max( sample.wi[ 2 ], 0 );
		const validTarget = sample.target.length >= 3 && sample.target.every( Number.isFinite );
		sample.directTarget = validTarget ? sample.target.slice( 0, 3 ) : null;

		if ( validTarget && nDotL >= minimumCosine ) {

			sample.target = sample.target.map( ( value ) => value / nDotL );
			const peakRadiance = Math.max( sample.directTarget[ 0 ], sample.directTarget[ 1 ], sample.directTarget[ 2 ], 0 );
			const highlightWeight = 1 + highlightLossScale * peakRadiance / ( 1 + peakRadiance );
			sample.weight = nDotL * highlightWeight;

		} else {

			sample.target = [ 0, 0, 0 ];
			sample.weight = 0;

		}

	}

}

function sampleTeacherDirections( random ) {

	const mode = random();

	if ( mode < 0.35 ) {

		return {
			wi: sampleHemisphereUniform( random ),
			wo: sampleHemisphereUniform( random )
		};

	}

	if ( mode < 0.55 ) {

		return {
			wi: sampleHemisphereCosine( random ),
			wo: sampleHemisphereCosine( random )
		};

	}

	if ( mode < 0.65 ) {

		return {
			wi: sampleSphereUniform( random ),
			wo: sampleHemisphereUniform( random )
		};

	}

	if ( mode < 0.8 ) return sampleMirrorPair( random );

	return sampleRusinkiewicz( random );

}

function sampleMirrorPair( random ) {

	const wo = sampleHemisphereUniform( random );

	return {
		wi: [ - wo[ 0 ], - wo[ 1 ], wo[ 2 ] ],
		wo
	};

}

function sampleRusinkiewicz( random ) {

	for ( let attempts = 0; attempts < 32; attempts ++ ) {

		const exponent = [ 2, 8, 32, 128 ][ Math.floor( random() * 4 ) ];
		const wh = sampleHemispherePower( random, exponent );
		const wo = sampleHemisphereUniform( random );
		const wi = reflect( wo, wh );

		if ( dot( wo, wh ) > 0 && wi[ 2 ] > 1e-5 ) return { wi, wo };

	}

	return { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] };

}

function sampleHemispherePower( random, exponent ) {

	const z = Math.pow( random(), 1 / ( exponent + 1 ) );
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

}

function sampleSphereUniform( random ) {

	const z = random() * 2 - 1;
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

}

function sampleHemisphereUniform( random ) {

	const z = random();
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

}

function sampleHemisphereCosine( random ) {

	const z = Math.sqrt( random() );
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

}

function directionFromCosine( cosine, azimuth ) {

	const radius = Math.sqrt( Math.max( 0, 1 - cosine * cosine ) );
	return [ radius * Math.cos( azimuth ), radius * Math.sin( azimuth ), cosine ];

}

function reflect( vector, normal ) {

	const scale = 2 * dot( vector, normal );

	return normalize( [
		scale * normal[ 0 ] - vector[ 0 ],
		scale * normal[ 1 ] - vector[ 1 ],
		scale * normal[ 2 ] - vector[ 2 ]
	] );

}

function augmentColorChannels( sample, random ) {

	const pattern = [
		Math.floor( random() * 3 ),
		Math.floor( random() * 3 ),
		Math.floor( random() * 3 )
	];
	const albedoOffset = 3;
	const f0Offset = 8;
	const target = sample.target.slice();
	const albedo = sample.encoderInputs.slice( albedoOffset, albedoOffset + 3 );
	const f0 = sample.encoderInputs.slice( f0Offset, f0Offset + 3 );

	for ( let i = 0; i < 3; i ++ ) {

		sample.target[ i ] = target[ pattern[ i ] ];
		sample.encoderInputs[ albedoOffset + i ] = albedo[ pattern[ i ] ];
		sample.encoderInputs[ f0Offset + i ] = f0[ pattern[ i ] ];

	}

}

function getMipLevelCount( width, height ) {

	let levels = 1;

	while ( width > 1 || height > 1 ) {

		width = Math.max( 1, width >> 1 );
		height = Math.max( 1, height >> 1 );
		levels ++;

	}

	return levels;

}

function createMipLevelIndices( levelCount ) {

	const levels = [];

	for ( let level = 0; level < levelCount; level ++ ) levels.push( level );

	return levels;

}

function dot( a, b ) {

	return a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];

}

function normalize( value ) {

	const length = Math.hypot( value[ 0 ], value[ 1 ], value[ 2 ] ) || 1;

	return [ value[ 0 ] / length, value[ 1 ] / length, value[ 2 ] / length ];

}

export {
	generateTrainingSamples,
	generateValidationSamples,
	generateIBLTrainingSamples,
	assignIBLTeacherTargets,
	assignTeacherTargets,
	assignAuxiliaryTeacherTargets,
	normalizeDirectLightingTargets,
	sampleTeacherDirections,
	getMipLevelCount,
	createMipLevelIndices
};
