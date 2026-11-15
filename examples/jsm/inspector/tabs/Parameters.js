import { Tab } from '../ui/Tab.js';
import { List } from '../ui/List.js';
import { Item } from '../ui/Item.js';
import { createValueSpan, info } from '../ui/utils.js';
import { ValueString, ValueNumber, ValueSlider, ValueSelect, ValueCheckbox, ValueColor, ValueButton } from '../ui/Values.js';

class ParametersGroup {

	constructor( parameters, name ) {

		this.parameters = parameters;
		this._urlPath = name;

		this.paramList = new Item( name );
		this.paramList.setCollapsible( true );

		this.objects = [];

	}

	close() {

		this.paramList.close();

		return this;

	}

	name( name ) {

		this.paramList.setValue( 0, name );

		return this;

	}

	show() {

		this.paramList.show();

		return this;

	}

	hide() {

		this.paramList.hide();

		return this;

	}

	add( object, property, ...params ) {

		const value = object[ property ];
		const type = typeof value;

		let item = null;

		if ( typeof params[ 0 ] === 'object' ) {

			item = this.addSelect( object, property, params[ 0 ] );

		} else if ( type === 'number' ) {

			if ( params.length >= 2 ) {

				item = this.addSlider( object, property, ...params );

			} else {

				item = this.addNumber( object, property, ...params );

			}

		} else if ( type === 'boolean' ) {

			item = this.addBoolean( object, property );

		} else if ( type === 'string' ) {

			item = this.addString( object, property );

		} else if ( type === 'function' ) {

			item = this.addButton( object, property, ...params );

		}

		return item;

	}

	_addInfo( editor, itemNode ) {

		editor.info = ( text ) => {

			info( itemNode, text );
			return editor;

		};

	}

	_addParameter( object, property, editor, subItem ) {

		subItem.data[ 0 ].title = property;

		editor.name = ( name ) => {

			subItem.data[ 0 ].title = name;

			if ( subItem.data[ 0 ].childNodes.length > 0 && subItem.data[ 0 ].firstChild.nodeType === 3 /* Node.TEXT_NODE */ ) {

				subItem.data[ 0 ].firstChild.textContent = name;

			} else {

				subItem.data[ 0 ].insertBefore( document.createTextNode( name ), subItem.data[ 0 ].firstChild );

			}

			return editor;

		};

		this._addInfo( editor, subItem.data[ 0 ] );

		editor.listen = () => {

			const update = () => {

				const value = editor.getValue();
				const propertyValue = object[ property ];

				if ( value !== propertyValue ) {

					editor.setValue( propertyValue );

				}

				requestAnimationFrame( update );

			};

			requestAnimationFrame( update );

			return editor;

		};


		this._registerParameter( object, property, editor, subItem );

		this.parameters._urlBindings.push( {
			group: this, object, property, editor,
			defaultValue: this._getURLParameterValue( object, property, editor )
		} );

		if ( this.parameters._urlParameters !== null ) {

			// Wait for chained control configuration.

			queueMicrotask( () => this._applyURLParameter( object, property, editor ) );

		}

	}

	_getURLParameterName( object, property ) {

		return property === 'value' && object.isUniformNode && object.name ? object.name : property;

	}

	_getURLParameterValue( object, property, editor ) {

		const value = object[ property ];
		return editor instanceof ValueColor ? editor._getColorHex( value ).toLowerCase() : String( value );

	}

	_applyURLParameter( object, property, editor ) {

		const parameters = this.parameters._urlParameters;
		const name = this._getURLParameterName( object, property );
		const path = this._urlPath + '.' + name;
		const qualified = parameters.has( path );
		const key = qualified ? path : name;

		if ( ! parameters.has( key ) ) return;

		const matches = this.parameters._urlBindings.filter( ( binding ) => {

			const name = binding.group._getURLParameterName( binding.object, binding.property );
			return ( qualified ? binding.group._urlPath + '.' + name : name ) === key;

		} );

		if ( matches.length !== 1 ) return;

		let value = parameters.get( key );
		parameters.delete( key );

		if ( editor instanceof ValueSelect ) {

			const options = Object.values( editor.options );
			value = options.find( option => String( option ) === value );

		} else if ( editor instanceof ValueColor ) {

			value = parseInt( value.replace( /^(#|0x)/i, '' ), 16 );

		} else if ( editor instanceof ValueCheckbox ) {

			value = value === 'true';

		} else if ( editor instanceof ValueNumber || editor instanceof ValueSlider ) {

			const input = editor instanceof ValueSlider ? editor.slider : editor.input;
			const min = Number( input.min );
			const max = Number( input.max );

			value = Math.max( min, Math.min( max, Number( value ) ) );

			if ( editor instanceof ValueSlider ) {

				// Normalize slider steps.

				input.value = value;
				value = Number( input.value );

			}

		}

		editor.setValue( value );

	}

	_registerParameter( object, property, editor, subItem ) {

		this.objects.push( { object: object, key: property, editor: editor, subItem: subItem } );

		editor.addEventListener( 'show', () => subItem.show() );
		editor.addEventListener( 'hide', () => subItem.hide() );

	}

	addString( object, property ) {

		const value = object[ property ];

		const editor = new ValueString( { value } );
		editor.addEventListener( 'change', ( { value } ) => {

			object[ property ] = value;

		} );

		const description = createValueSpan();
		description.textContent = property;

		const subItem = new Item( description, editor.domElement );
		this.paramList.add( subItem );

		const itemRow = subItem.domElement.firstChild;
		itemRow.classList.add( 'actionable' );

		// extend object property

		this._addParameter( object, property, editor, subItem );

		return editor;

	}

	addFolder( name ) {

		const group = new ParametersGroup( this.parameters, name );
		group._urlPath = this._urlPath + '.' + name;

		this.paramList.add( group.paramList );

		return group;

	}

	addBoolean( object, property ) {

		const value = object[ property ];

		const editor = new ValueCheckbox( { value } );
		editor.addEventListener( 'change', ( { value } ) => {

			object[ property ] = value;

		} );

		const description = createValueSpan();
		description.textContent = property;

		const subItem = new Item( description, editor.domElement );
		this.paramList.add( subItem );

		// extends logic to toggle checkbox when clicking on the row

		const itemRow = subItem.domElement.firstChild;

		itemRow.classList.add( 'actionable' );
		itemRow.addEventListener( 'click', ( e ) => {

			if ( e.target.closest( 'label' ) ) return;

			const checkbox = itemRow.querySelector( 'input[type="checkbox"]' );

			if ( checkbox ) {

				checkbox.checked = ! checkbox.checked;
				checkbox.dispatchEvent( new Event( 'change' ) );

			}

		} );

		// extend object property

		this._addParameter( object, property, editor, subItem );

		return editor;

	}

	addSelect( object, property, options ) {

		const value = object[ property ];

		const editor = new ValueSelect( { options, value } );
		editor.addEventListener( 'change', ( { value } ) => {

			object[ property ] = value;

		} );

		const description = createValueSpan();
		description.textContent = property;

		const subItem = new Item( description, editor.domElement );
		this.paramList.add( subItem );

		const itemRow = subItem.domElement.firstChild;
		itemRow.classList.add( 'actionable' );

		// extend object property

		this._addParameter( object, property, editor, subItem );

		return editor;

	}

	addColor( object, property ) {

		const value = object[ property ];

		const editor = new ValueColor( { value } );
		editor.addEventListener( 'change', ( { value } ) => {

			object[ property ] = value;

		} );

		const description = createValueSpan();
		description.textContent = property;

		const subItem = new Item( description, editor.domElement );
		this.paramList.add( subItem );

		const itemRow = subItem.domElement.firstChild;
		itemRow.classList.add( 'actionable' );

		// extend object property

		this._addParameter( object, property, editor, subItem );

		return editor;

	}

	addSlider( object, property, min = 0, max = 1, step = 0.01 ) {

		const value = object[ property ];

		const editor = new ValueSlider( { value, min, max, step } );
		editor.addEventListener( 'change', ( { value } ) => {

			object[ property ] = value;

		} );

		const description = createValueSpan();
		description.textContent = property;

		const subItem = new Item( description, editor.domElement );
		this.paramList.add( subItem );

		const itemRow = subItem.domElement.firstChild;
		itemRow.classList.add( 'actionable' );

		// extend object property

		this._addParameter( object, property, editor, subItem );

		return editor;

	}

	addNumber( object, property, ...params ) {

		const value = object[ property ];
		const [ min, max ] = params;

		const editor = new ValueNumber( { value, min, max } );
		editor.addEventListener( 'change', ( { value } ) => {

			object[ property ] = value;

		} );

		const description = createValueSpan();
		description.textContent = property;

		const subItem = new Item( description, editor.domElement );
		this.paramList.add( subItem );

		const itemRow = subItem.domElement.firstChild;
		itemRow.classList.add( 'actionable' );

		// extend object property

		this._addParameter( object, property, editor, subItem );

		return editor;

	}

	addButton( object, property ) {

		const value = object[ property ];

		const editor = new ValueButton( { text: property, value } );
		editor.addEventListener( 'change', ( { value } ) => {

			object[ property ] = value;

		} );

		const subItem = new Item( editor.domElement );
		subItem.itemRow.childNodes[ 0 ].style.gridColumn = '1 / -1';
		this.paramList.add( subItem );

		const itemRow = subItem.domElement.firstChild;
		itemRow.classList.add( 'actionable' );

		// extend object property

		editor.name = ( name ) => {

			const buttonNode = editor.domElement.childNodes[ 0 ];

			if ( buttonNode.childNodes.length > 0 && buttonNode.firstChild.nodeType === 3 /* Node.TEXT_NODE */ ) {

				buttonNode.firstChild.textContent = name;

			} else {

				buttonNode.insertBefore( document.createTextNode( name ), buttonNode.firstChild );

			}

			return editor;

		};

		this._addInfo( editor, editor.domElement.childNodes[ 0 ] );

		this._registerParameter( object, property, editor, subItem );

		return editor;

	}

}

class Parameters extends Tab {

	constructor( options = {} ) {

		super( options.name || 'Parameters', options );

		const paramList = new List( 'Property', 'Value' );
		paramList.domElement.classList.add( 'parameters' );
		paramList.setGridStyle( 'minmax(0, .5fr) minmax(0, 1fr)' );
		paramList.domElement.style.minWidth = '300px';

		const scrollWrapper = document.createElement( 'div' );
		scrollWrapper.className = 'list-scroll-wrapper';
		scrollWrapper.appendChild( paramList.domElement );
		this.content.appendChild( scrollWrapper );

		this.paramList = paramList;

		this.groups = [];
		this._urlParameters = options.urlParameters || null;
		this._urlBindings = [];

	}

	_createShareButton() {

		const button = document.createElement( 'button' );
		button.className = 'parameters-share-btn';
		button.title = 'Copy parameter URL';
		button.setAttribute( 'aria-label', button.title );
		button.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 10.5 6.8-4M8.6 13.5l6.8 4"/></svg>';

		button.onclick = async ( event ) => {

			event.stopPropagation();

			const url = this._getShareURL();

			if ( url === null ) {

				window.alert( 'Some changed parameters have the same URL path. Give their groups or uniforms unique names to share these settings.' );
				return;

			}

			try {

				await navigator.clipboard.writeText( url );
				button.title = 'URL copied';
				button.classList.add( 'copied' );
				setTimeout( () => {

					button.title = 'Copy parameter URL';
					button.classList.remove( 'copied' );

				}, 1000 );

			} catch {

				window.prompt( 'Copy parameter URL', url );

			}

		};

		this.builtinButton.classList.add( 'parameters-toggle' );
		this.builtinButton.after( button );

	}

	_getShareURL( url = new URL( window.location.href ) ) {

		for ( const { group, object, property } of this._urlBindings ) {

			const name = group._getURLParameterName( object, property );
			url.searchParams.delete( name );
			url.searchParams.delete( group._urlPath + '.' + name );

		}

		for ( const { group, object, property, editor, defaultValue } of this._urlBindings ) {

			const value = group._getURLParameterValue( object, property, editor );

			if ( value !== defaultValue ) {

				const name = group._getURLParameterName( object, property );
				const path = group._urlPath + '.' + name;
				const matches = this._urlBindings.filter( ( binding ) => {

					return binding.group._urlPath + '.' + binding.group._getURLParameterName( binding.object, binding.property ) === path;

				} );

				if ( matches.length !== 1 ) return null;

				url.searchParams.set( path, value );

			}

		}

		return url.href;

	}

	createGroup( name ) {

		const group = new ParametersGroup( this, name );

		this.paramList.add( group.paramList );
		this.groups.push( group );

		return group;

	}

}

export { Parameters };
