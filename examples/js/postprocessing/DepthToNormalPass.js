/**
*
* Depth to Normal Pass
*
* Used to determine precision of depth pass.  Not really that useful otherwise.
*
* @author bhouston / http://clara.io/
*
*/

THREE.DepthToNormalPass = function( renderer, scene, camera ) {

    THREE.Pass.call( this );
  
    this.scene = scene;
    this.camera = camera;
 
	this.depthToNormalMaterial = new THREE.ShaderMaterial( THREE.DepthToNormalShader );
	this.depthToNormalMaterial.uniforms = THREE.UniformsUtils.clone( this.depthToNormalMaterial.uniforms );
	this.depthToNormalMaterial.defines = Object.assign( {}, this.depthToNormalMaterial.defines );
	this.depthToNormalMaterial.blending = THREE.NoBlending;

    this.orthoScene = new THREE.Scene();
    this.orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -0.01, 1000);
    
    var quad = new THREE.PlaneGeometry( 2, 2 );
    var quadMesh = new THREE.Mesh( quad, this.currentMaterial );
    this.orthoScene.add( quadMesh );
  
    this.needsSwap = true;
 
    this.normalRenderTarget = new THREE.WebGLRenderTarget( 256, 256, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat
    } );

    this.normalTexture = this.normalRenderTarget.texture;
    this.depthTexture = null;

    this.outputDebug = false;
}

THREE.DepthToNormalPass.prototype = Object.assign( Object.create( THREE.Pass.prototype ), {

    constructor: THREE.DepthToNormalPass,

    dispose: function() {

        if( this.normalRenderTarget ) { 
            this.normalRenderTarget.dispose();
            this.normalRenderTarget = null;
        }

    },

    setSize: function ( width, height ) {

        this.normalRenderTarget.setSize( width, height );
        //console.log( "DepthPass.setSize: ", width, height );

    },

    render: function( renderer, writeBuffer, readBuffer, delta, maskActive ) {
      
        var originalClearColor = renderer.getClearColor();
        var originalClearAlpha = renderer.getClearAlpha();
        var originalAutoClear = renderer.autoClear;

        var camera = this.camera;

        this.depthToNormalMaterial.uniforms.tDepth.value = this.depthTexture;
        this.depthToNormalMaterial.defines.depthPacking = this.depthTexture.depthPacking;
		this.depthToNormalMaterial.uniforms[ 'cameraNear' ].value = camera.near;
		this.depthToNormalMaterial.uniforms[ 'cameraFar' ].value = camera.far;
		this.depthToNormalMaterial.uniforms[ 'cameraProjectionMatrix' ].value = camera.projectionMatrix;
		this.depthToNormalMaterial.uniforms[ 'cameraInverseProjectionMatrix' ].value.getInverse( camera.projectionMatrix );
    
        renderer.setClearColor( new THREE.Color( 1.0, 1.0, 1.0 ), 1.0 );
        this.orthoScene.overrideMaterial = this.depthToNormalMaterial;
        renderer.render( this.orthoScene, this.orthoCamera, this.outputDebug ? writeBuffer : this.normalRenderTarget, true );
        this.orthoScene.overrideMaterial = null;
    
		// restore original state
		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );
    }

});
