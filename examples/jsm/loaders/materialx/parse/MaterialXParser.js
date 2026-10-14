function parseMaterialXNodeTree( element, createNode, addNode, nodePath = '' ) {

	const materialXNode = createNode( element, nodePath );
	if ( materialXNode.nodePath ) {

		addNode( materialXNode );

	}

	for ( const childElement of element.children ) {

		const childMXNode = parseMaterialXNodeTree( childElement, createNode, addNode, materialXNode.nodePath );
		materialXNode.add( childMXNode );

	}

	return materialXNode;

}

function invalid( path, message ) {

	throw new Error( `THREE.MaterialXParser: ${ path }: ${ message }` );

}

function fromXML( element ) {

	return {
		element: element.nodeName,
		attributes: Object.fromEntries( Array.from( element.attributes, attribute => [ attribute.name, attribute.value ] ) ),
		children: Array.from( element.children, fromXML )
	};

}

function normalizeJSON( source ) {

	const ancestors = new Set();
	function properties( object, path ) {

		if ( ! object || typeof object !== 'object' || Array.isArray( object ) || ( Object.getPrototypeOf( object ) !== Object.prototype && Object.getPrototypeOf( object ) !== null ) ) invalid( path, 'Expected a plain JSON object.' );
		const descriptors = Object.getOwnPropertyDescriptors( object );
		for ( const key of Reflect.ownKeys( descriptors ) ) {

			if ( typeof key !== 'string' || ! Object.hasOwn( descriptors[ key ], 'value' ) || ! descriptors[ key ].enumerable ) invalid( path, 'Accessors, symbols and runtime metadata are not JSON.' );

		}

		return descriptors;

	}

	function normalize( node, path ) {

		const descriptors = properties( node, path );
		if ( ancestors.has( node ) ) invalid( path, 'Cyclic JSON document.' );
		ancestors.add( node );
		for ( const key of Object.keys( descriptors ) ) {

			if ( ! [ 'element', 'attributes', 'children' ].includes( key ) ) invalid( path, `Unknown JSON element property "${ key }".` );

		}

		const element = descriptors.element && descriptors.element.value;
		if ( typeof element !== 'string' || element === '' ) invalid( path, 'Expected an element category string.' );
		const attributes = Object.create( null );
		for ( const [ key, descriptor ] of Object.entries( properties( descriptors.attributes ? descriptors.attributes.value : {}, `${ path }.attributes` ) ) ) {

			const value = descriptor.value;
			if ( typeof value !== 'string' && typeof value !== 'boolean' && ( typeof value !== 'number' || ! Number.isFinite( value ) ) ) invalid( `${ path }.attributes.${ key }`, 'Attributes must be strings, finite numbers or booleans, as in XML.' );
			attributes[ key ] = String( value );

		}

		const children = descriptors.children ? descriptors.children.value : [];
		if ( ! Array.isArray( children ) ) invalid( `${ path }.children`, 'Expected a child-element array.' );
		const childDescriptors = Object.getOwnPropertyDescriptors( children );
		if ( Reflect.ownKeys( childDescriptors ).length !== children.length + 1 ) invalid( path, 'Child arrays must be dense plain JSON arrays.' );
		const normalizedChildren = [];
		for ( let i = 0; i < children.length; i ++ ) {

			const descriptor = childDescriptors[ i ];
			if ( ! descriptor || ! Object.hasOwn( descriptor, 'value' ) || ! descriptor.enumerable ) invalid( path, 'Child arrays must contain plain JSON elements.' );
			normalizedChildren.push( normalize( descriptor.value, `${ path }.children[${ i }]` ) );

		}

		ancestors.delete( node );
		return { element, attributes, children: normalizedChildren };

	}

	return normalize( source, 'document' );

}

/** Normalizes XML or the addon's neutral JSON element tree for the shared compiler. */
function parseMaterialXData( data ) {

	let source = data;
	if ( typeof data === 'string' ) {

		const text = data.trimStart();
		if ( text.startsWith( '<' ) ) {

			const xml = new DOMParser().parseFromString( data, 'application/xml' );
			if ( ! xml.documentElement || xml.getElementsByTagName( 'parsererror' ).length ) invalid( 'document', 'Invalid MaterialX XML.' );
			source = fromXML( xml.documentElement );

		} else {

			try {

				source = JSON.parse( text );

			} catch ( error ) {

				invalid( 'document', `Invalid MaterialX JSON: ${ error.message }` );

			}

		}

	}

	const normalized = normalizeJSON( source );
	if ( normalized.element !== 'materialx' ) invalid( 'document', 'The root element must be materialx.' );
	return normalized;

}

function parseMaterialXText( text, createNode, addNode ) {

	return parseMaterialXNodeTree( parseMaterialXData( text ), createNode, addNode );

}

export { parseMaterialXNodeTree, parseMaterialXText, parseMaterialXData };
