/**
*
* Depth Pass
*
* @author bhouston / http://clara.io/
*
*/

THREE.DepthPass = function( renderer, scene, camera ) {

    THREE.Pass.call( this );
  
    this.scene = scene;
    this.camera = camera;
  
 
    this.depthMaterial = new THREE.MeshDepthMaterial();
    this.depthMaterial.blending = THREE.NoBlending;
  
    this.needsSwap = false;
 
    this.depthRenderTarget = new THREE.WebGLRenderTarget( 256, 256, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat
    } );

    if ( renderer.extensions.get('WEBGL_depth_texture') ) {
        this.depthRenderTarget.depthBuffer = true;
        this.depthRenderTarget.depthTexture = new THREE.DepthTexture();
        this.depthRenderTarget.depthTexture.type = /*isWebGL2 ? THREE.FloatType : */THREE.UnsignedShortType;

        this.depthTexture = this.depthRenderTarget.depthTexture;
        this.depthTexture.depthPacking = THREE.BasicDepthPacking;

    }
    else {

        this.depthTexture = this.depthRenderTarget.texture;
        this.depthTexture.depthPacking = THREE.RGBADepthPacking;

    }

    this.side = THREE.FrontSide;
    this.outputDebug = false;    
}

THREE.DepthPass.prototype = Object.assign( Object.create( THREE.Pass.prototype ), {

    constructor: THREE.DepthPass,

    dispose: function() {

        if( this.depthRenderTarget ) { 
            this.depthRenderTarget.dispose();
            this.depthRenderTarget = null;
        }

    },

    setSize: function ( width, height ) {

        this.depthRenderTarget.setSize( width, height );
        //console.log( "DepthPass.setSize: ", width, height );

    },

    render: function( renderer, writeBuffer, readBuffer, delta, maskActive ) {
      
        var originalClearColor = renderer.getClearColor();
        var originalClearAlpha = renderer.getClearAlpha();
        var originalAutoClear = renderer.autoClear;

        this.depthMaterial.depthPacking = this.depthTexture.depthPacking;
        this.depthMaterial.side = this.side;
    
        renderer.setClearColor( new THREE.Color( 1.0, 1.0, 1.0 ), 1.0 );
        scene.overrideMaterial = this.depthMaterial;
        renderer.render( this.scene, this.camera, this.outputDebug ? null : this.depthRenderTarget, true );
        scene.overrideMaterial = null;
    
		// restore original state
		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );
    }

});
