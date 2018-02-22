/**
*
* Multi Pass
*
* Produce normal and depth passes as efficiently as possible.
* 
* @author bhouston / http://clara.io/
*
*/

THREE.MultiPass = function( renderer, scene, camera ) {

    THREE.Pass.call( this );
  
    this.scene = scene;
    this.camera = camera;
  
    this.normalMaterial = new THREE.MeshNormalMaterial();
    this.normalMaterial.blending = THREE.NoBlending;
  
    this.needsSwap = false;
  
    this.normalRenderTarget = new THREE.WebGLRenderTarget( 256, 256, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat
    } );

    this.side = THREE.FrontSide;

    this.normalTexture = this.normalRenderTarget.texture;

    if ( true || ! renderer.extensions.get('WEBGL_depth_texture') ) {
    
        this.depthMaterial = new THREE.MeshDepthMaterial();
        this.depthMaterial.blending = THREE.NoBlending;

        this.depthRenderTarget = new THREE.WebGLRenderTarget( 256, 256, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            format: THREE.RGBAFormat
        } );

        this.depthTexture = this.depthRenderTarget.texture;
        this.depthTexture.depthPacking = THREE.RGBADepthPacking;
        console.log( "depthMaterial = true" );

    }
    else {

        this.normalRenderTarget.depthBuffer = true;
        this.normalRenderTarget.depthTexture = new THREE.DepthTexture();
        this.normalRenderTarget.depthTexture.type = /*isWebGL2 ? THREE.FloatType : */THREE.UnsignedShortType;

        this.depthTexture = this.normalRenderTarget.depthTexture;
        this.depthTexture.depthPacking = THREE.BasicDepthPacking;

    }

    this.outputDebug = false;

}

THREE.MultiPass.prototype = Object.assign( Object.create( THREE.Pass.prototype ), {

    constructor: THREE.MultiPass,

    dispose: function() {

        if( this.normalRenderTarget ) { 
            this.normalRenderTarget.dispose();
            this.normalRenderTarget = null;
        }

        if( this.depthRenderTarget ) { 
            this.depthRenderTarget.dispose();
            this.depthRenderTarget = null;
        }

    },

    setSize: function ( width, height ) {

        if( this.normalRenderTarget ) this.normalRenderTarget.setSize( width, height );
        if( this.depthRenderTarget ) this.depthRenderTarget.setSize( width, height );
    
		//console.log( "MultiPass.setSize: ", width, height );

    },

    render: function( renderer, writeBuffer, readBuffer, delta, maskActive ) {
      
        var originalClearColor = renderer.getClearColor();
        var originalClearAlpha = renderer.getClearAlpha();
        var originalAutoClear = renderer.autoClear;

        this.normalMaterial.side = this.side;
        
		renderer.setClearColor( new THREE.Color( 0.5, 0.5, 1.0 ), 1.0 );
        scene.overrideMaterial = this.normalMaterial;
        renderer.render( this.scene, this.camera,this.normalRenderTarget, true );
        scene.overrideMaterial = null;
        console.log( "normalMaterial = rendering" );

        if( this.depthRenderTarget ) { // render depth explicitly if the depth texture extension is not available, less efficient but required.
   
            this.depthMaterial.depthPacking = this.depthTexture.depthPacking;
            this.depthMaterial.side = this.side;
        
            renderer.setClearColor( new THREE.Color( 1.0, 1.0, 1.0 ), 1.0 );
            scene.overrideMaterial = this.depthMaterial;
            renderer.render( this.scene, this.camera, this.outputDebug ? writeBuffer : this.depthRenderTarget, true );
            scene.overrideMaterial = null;
            console.log( "depthMaterial = rendering" );

        }

		// restore original state
		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );
    }

});
