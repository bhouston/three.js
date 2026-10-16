import { MeshBasicNodeMaterial, EqualDepth, Vector2 } from 'three/webgpu';

// Example-only opaque depth/equal pattern. Both draws use exactly the same
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
		const objects = [];
		pass.scene.traverse( object => {

			if ( ! object.isMesh ) return;
			const original = object.material;
			if ( Array.isArray( original ) || original.transparent ) return;
			if ( ! materials.has( original ) ) materials.set( original, new MeshBasicNodeMaterial( {
				side: original.side, colorWrite: false, depthWrite: true,
				map: original.map, alphaMap: original.alphaMap, alphaTest: original.alphaTest
			} ) );
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
			for ( const [ object, material ] of objects ) {

				object.material = material;
				material.depthFunc = EqualDepth;
				material.depthWrite = false;

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
			pass.autoClearDepth = true;

		}

	};

	const dispose = pass.dispose.bind( pass );
	pass.dispose = () => {

		for ( const material of materials.values() ) material.dispose();
		dispose();

	};

	pass.depthPrepass = true;
	return pass;

}
