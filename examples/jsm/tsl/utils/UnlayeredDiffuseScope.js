/**
 * Conservative material scope for the interior SSGI integration experiments.
 * Finish-stage layers/transmission and custom lighting/output paths retain the
 * original beauty-source and whole-indirect AO behavior.
 */
export function supportsUnlayeredDiffuse( material ) {

	if ( material.transparent === true ) return false;
	if ( material.isMeshStandardMaterial !== true && material.isMeshStandardNodeMaterial !== true ) return false;
	for ( const name of [ 'clearcoat', 'sheen', 'transmission' ] ) {

		if ( material[ name ] > 0 || material[ name + 'Node' ] != null ) return false;

	}

	for ( const name of [ 'fragmentNode', 'outputNode', 'backdropNode', 'lightsNode' ] ) {

		if ( material[ name ] != null ) return false;

	}

	return true;

}
