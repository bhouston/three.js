function computeFootprintArea( duvDx, duvDy, sourceWidth = 1, sourceHeight = sourceWidth ) {

	const dx = duvDx || [ 0, 0 ];
	const dy = duvDy || [ 0, 0 ];
	const determinant = Math.abs( dx[ 0 ] * dy[ 1 ] - dx[ 1 ] * dy[ 0 ] );

	return determinant * sourceWidth * sourceHeight;

}

function getGaussianSampleGridSize( footprintArea, minSamples = 1, maxSamples = 64 ) {

	const minimum = Math.max( 1, Math.ceil( Math.sqrt( minSamples ) ) );
	const maximum = Math.max( minimum, Math.floor( Math.sqrt( maxSamples ) ) );
	const requested = Math.ceil( Math.sqrt( Math.max( footprintArea, minSamples ) ) );

	return Math.min( Math.max( requested, minimum ), maximum );

}

function createGaussianSampleKernel( gridSize, tileSize, sigma = 0.25 ) {

	const samples = [];
	let weightSum = 0;
	const safeSigma = Math.max( sigma, 1e-6 );

	for ( let y = 0; y < gridSize; y ++ ) {

		for ( let x = 0; x < gridSize; x ++ ) {

			const pixelX = Math.min( tileSize - 1, Math.floor( ( x + 0.5 ) * tileSize / gridSize ) );
			const pixelY = Math.min( tileSize - 1, Math.floor( ( y + 0.5 ) * tileSize / gridSize ) );
			const offsetX = ( pixelX + 0.5 ) / tileSize - 0.5;
			const offsetY = ( pixelY + 0.5 ) / tileSize - 0.5;
			const weight = Math.exp( - 0.5 * ( offsetX * offsetX + offsetY * offsetY ) / ( safeSigma * safeSigma ) );

			samples.push( { x: pixelX, y: pixelY, weight } );
			weightSum += weight;

		}

	}

	for ( const sample of samples ) sample.weight /= weightSum;

	return samples;

}

function sampleExponentialMipLevel( random, levelCount, decay = 0.5 ) {

	if ( levelCount <= 1 ) return 0;

	let weightSum = 0;

	for ( let level = 0; level < levelCount; level ++ ) weightSum += Math.pow( decay, level );

	let value = random() * weightSum;

	for ( let level = 0; level < levelCount; level ++ ) {

		value -= Math.pow( decay, level );
		if ( value <= 0 ) return level;

	}

	return levelCount - 1;

}

function prefilterLeanNormalRoughness( samples ) {

	if ( samples.length === 0 ) {

		return {
			normal: [ 0, 0, 1 ],
			roughness: 0,
			meanSlope: [ 0, 0 ],
			secondMoment: [ 0, 0, 0 ]
		};

	}

	let weightSum = 0;
	let meanX = 0;
	let meanY = 0;
	let momentXX = 0;
	let momentYY = 0;
	let momentXY = 0;
	let meanRoughnessSquared = 0;

	for ( const sample of samples ) {

		const normal = sample.normal || [ 0, 0, 1 ];
		const roughness = Math.max( sample.roughness || 0, 0 );
		const weight = sample.weight !== undefined ? Math.max( sample.weight, 0 ) : 1;
		const inverseZ = 1 / Math.max( Math.abs( normal[ 2 ] ), 1e-4 );
		const slopeX = normal[ 0 ] * inverseZ;
		const slopeY = normal[ 1 ] * inverseZ;

		weightSum += weight;
		meanX += slopeX * weight;
		meanY += slopeY * weight;
		momentXX += slopeX * slopeX * weight;
		momentYY += slopeY * slopeY * weight;
		momentXY += slopeX * slopeY * weight;
		meanRoughnessSquared += roughness * roughness * weight;

	}

	const inverseWeight = 1 / Math.max( weightSum, 1e-12 );
	meanX *= inverseWeight;
	meanY *= inverseWeight;
	momentXX *= inverseWeight;
	momentYY *= inverseWeight;
	momentXY *= inverseWeight;
	meanRoughnessSquared *= inverseWeight;

	const variance = Math.max( momentXX - meanX * meanX, 0 ) + Math.max( momentYY - meanY * meanY, 0 );
	const normalLength = Math.hypot( meanX, meanY, 1 );

	return {
		normal: [ meanX / normalLength, meanY / normalLength, 1 / normalLength ],
		roughness: Math.min( Math.sqrt( meanRoughnessSquared + 0.5 * variance ), 1 ),
		meanSlope: [ meanX, meanY ],
		secondMoment: [ momentXX, momentYY, momentXY ]
	};

}

export {
	computeFootprintArea,
	createGaussianSampleKernel,
	getGaussianSampleGridSize,
	prefilterLeanNormalRoughness,
	sampleExponentialMipLevel
};
