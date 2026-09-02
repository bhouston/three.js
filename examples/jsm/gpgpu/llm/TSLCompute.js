function orderedComputeNodes( ...operations ) {

	const nodes = [];

	for ( const operation of operations ) {

		if ( operation === null || operation === undefined ) continue;

		if ( Array.isArray( operation.computeNodes ) ) {

			nodes.push( ...operation.computeNodes );

		} else if ( operation.computeNode ) {

			nodes.push( operation.computeNode );

		} else {

			throw new Error( 'TSLCompute: Operation does not expose compute nodes.' );

		}

	}

	return nodes;

}

export { orderedComputeNodes };
