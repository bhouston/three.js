import { mrt, sample } from 'three/tsl';

/**
 * Experimental configuration-time convention, not a resource graph. Each attachment
 * supplies a complete encoded value; fields expose decoded, sampleable views of it.
 * All fields sharing an active attachment are written, even if only one is requested.
 * Call once on a fresh PassNode, before compilation. Reconstruct to change layouts.
 * Attachment zero (`output`) belongs to PassNode and retains its renderer-selected type.
 *
 * @param {PassNode} producer - A fresh, application-owned pass.
 * @param {Object} layout - Attachment descriptors and logical field descriptors.
 * @param {Array<string>} requested - Fields needed by the application's chosen consumers.
 * @return {Object} Frozen views and attachment names. Ownership stays with the caller.
 */
export function surfaceLayout( producer, { attachments, fields }, requested ) {

	if ( producer.getMRT() !== null || producer.renderTarget.textures.length !== 1 ) {

		throw new Error( 'SurfaceLayout: use a fresh PassNode for each configuration.' );

	}

	// Validate the whole declaration before allocating anything. This intentionally
	// does not infer shader reachability, repack storage, or mutate compiled layouts.
	for ( const [ name, field ] of Object.entries( fields ) ) {

		if ( ! Object.hasOwn( attachments, field.attachment ) ) {

			throw new Error( `SurfaceLayout: unresolved attachment for "${ name }".` );

		}

		if ( ! /^(r|g|b|a|rg|rgb|rgba)$/.test( field.channels ) ) {

			throw new Error( `SurfaceLayout: invalid channels for "${ name }".` );

		}

	}

	for ( const name of requested ) {

		if ( ! Object.hasOwn( fields, name ) ) throw new Error( `SurfaceLayout: unknown field "${ name }".` );

	}

	const active = new Set( [ 'output', ...requested.map( name => fields[ name ].attachment ) ] );
	const outputs = {};

	for ( const name of active ) {

		if ( ! attachments[ name ]?.node ) throw new Error( `SurfaceLayout: missing producer for "${ name }".` );
		if ( name === 'output' && attachments[ name ].texture?.type !== undefined ) {

			throw new Error( 'SurfaceLayout: PassNode owns the output attachment type.' );

		}

	}

	const textures = {};

	for ( const name of active ) {

		const descriptor = attachments[ name ];
		outputs[ name ] = descriptor.node;
		textures[ name ] = producer.getTextureNode( name );
		Object.assign( producer.getTexture( name ), descriptor.texture );

	}

	producer.setMRT( mrt( outputs ) );

	const nodes = {};
	const metadata = {};

	for ( const name of new Set( requested ) ) {

		const field = fields[ name ];
		const textureNode = textures[ field.attachment ];
		const channels = field.channels;
		const decode = field.decode || ( value => value );

		// Capture this producer's node. Later layouts cannot retarget earlier consumers.
		nodes[ name ] = sample( uv => decode( textureNode.sample( uv )[ channels ] ) );
		metadata[ name ] = Object.freeze( { ...field.metadata } );

	}

	return Object.freeze( {
		nodes: Object.freeze( nodes ),
		metadata: Object.freeze( metadata ),
		textures: Object.freeze( textures ),
		attachmentNames: Object.freeze( [ ...active ] )
	} );

}
