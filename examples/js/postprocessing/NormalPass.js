/**
*
* Normal Pass
*
* @author bhouston / http://clara.io/
*
*/

THREE.NormalPass = function( scene, camera ) {

    THREE.Pass.call( this );
  
    this.scene = scene;
    this.camera = camera;
  
 
    this.normalMaterial = new THREE.MeshNormalMaterial();
    this.normalMaterial.blending = THREE.NoBlending;

    this.orthoScene = new THREE.Scene();
    this.orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -0.01, 1000);
    
    var quad = new THREE.PlaneGeometry( 2, 2 );
    var quadMesh = new THREE.Mesh( quad, this.currentMaterial );
    this.orthoScene.add( quadMesh );
  
    this.needsSwap = false;
  
    this.normalRenderTarget = new THREE.WebGLRenderTarget( 256, 256, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat
    } );

    this.side = THREE.FrontSide;

    this.normalTexture = this.normalRenderTarget.texture;
}

THREE.NormalPass.prototype = Object.assign( Object.create( THREE.Pass.prototype ), {

    constructor: THREE.NormalPass,

    dispose: function() {

        if( this.normalRenderTarget ) { 
            this.normalRenderTarget.dispose();
            this.normalRenderTarget = null;
        }

    },

    setSize: function ( width, height ) {

        this.normalRenderTarget.setSize( width, height );
		//console.log( "NormalPass.setSize: ", width, height );

    },

    render: function( renderer, writeBuffer, readBuffer, delta, maskActive ) {
      
        var originalClearColor = renderer.getClearColor();
        var originalClearAlpha = renderer.getClearAlpha();
        var originalAutoClear = renderer.autoClear;

        this.normalMaterial.side = this.side;
        
        renderer.setClearColor( new THREE.Color( 0.5, 0.5, 1.0 ), 1.0 );
        scene.overrideMaterial = this.normalMaterial;
        renderer.render( this.scene, this.camera, this.normalRenderTarget, true );
        scene.overrideMaterial = null;

		// restore original state
		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );
    }

});
