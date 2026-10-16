import { LessEqualDepth, Vector2 } from 'three/webgpu';

// Conservative opaque pre-pass. Bias depth away to tolerate rounding between
// separately compiled vertex programs. Both draws use the same
// camera (including TRAA jitter) and the same depth attachment.
export function enableDepthPrepass( pass ) {

	const update = pass.updateBefore.bind( pass );
	const materials = new Map();
	const size = new Vector2();
	pass.updateBefore = function ( frame ) {

		const { renderer } = frame;
		if ( ! pass.depthPrepass ) return update( frame );
		renderer.getDrawingBufferSize( size );
		pass.setSize( size.x, size.y );
		const target = renderer.getRenderTarget();
		const mrt = renderer.getMRT();
		const autoClear = renderer.autoClear;
		const background = pass.scene.background;
		const autoClearDepth = pass.autoClearDepth;
		const hidden = [];
		const objects = [];
		pass.scene.traverse( object => {

			if ( ! object.isMesh ) return;
			const original = object.material;
			if ( Array.isArray( original ) || original.transparent ) {

				hidden.push( [ object, object.visible ] );
				object.visible = false;
				return;

			}

			if ( ! materials.has( original ) ) {

				const depthMaterial = original.clone();
				depthMaterial.colorWrite = false;
				depthMaterial.depthWrite = true;
				depthMaterial.polygonOffset = true;
				depthMaterial.polygonOffsetFactor = 2;
				depthMaterial.polygonOffsetUnits = 2;
				materials.set( original, depthMaterial );

			}

			objects.push( [ object, original, original.depthFunc, original.depthWrite ] );
			object.material = materials.get( original );

		} );
		try {

			renderer.setRenderTarget( pass.renderTarget );
			renderer.setMRT( null );
			renderer.autoClear = false;
			renderer.clear();
			pass.scene.background = null;
			renderer.render( pass.scene, pass.camera );
			pass.scene.background = background;
			for ( const [ object, visible ] of hidden ) object.visible = visible;
			for ( const [ object, material ] of objects ) {

				object.material = material;
				material.depthFunc = LessEqualDepth;
				material.depthWrite = true;

			}

			renderer.setRenderTarget( target );
			renderer.setMRT( mrt );
			renderer.autoClear = autoClear;
			pass.autoClearDepth = false;
			update( frame );

		} finally {

			for ( const [ object, material, depthFunc, depthWrite ] of objects ) {

				object.material = material;
				material.depthFunc = depthFunc;
				material.depthWrite = depthWrite;

			}

			pass.scene.background = background;
			renderer.setRenderTarget( target );
			renderer.setMRT( mrt );
			renderer.autoClear = autoClear;
			for ( const [ object, visible ] of hidden ) object.visible = visible;
			pass.autoClearDepth = autoClearDepth;

		}

	};

	const dispose = pass.dispose.bind( pass );
	pass.dispose = () => {

		for ( const material of materials.values() ) material.dispose();
		dispose();

	};

	pass.depthPrepass = false;
	return pass;

}
