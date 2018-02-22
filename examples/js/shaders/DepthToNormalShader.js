/**
 * @author bhouston / http://clara.io/
 *
 * Depth to normal shader (extracted from SAO)
 * 
 */

THREE.DepthToNormalShader = {

    blending: THREE.NoBlending,

	defines: {
        "DEPTH_PACKING": 3201,
		"DEPTH_MIPS": 0,
		"PERSPECTIVE_CAMERA": 1
    },
    
    extensions: {
		'derivatives': true
    },
    
    uniforms: {

		"tDepth":       { type: "t", value: null },
		"size":         { type: "v2", value: new THREE.Vector2( 512, 512 ) },

		"cameraNear":   { type: "f", value: 1 },
		"cameraFar":    { type: "f", value: 100 },
		"cameraProjectionMatrix": { type: "m4", value: new THREE.Matrix4() },
		"cameraInverseProjectionMatrix": { type: "m4", value: new THREE.Matrix4() },

	},

	vertexShader: [

		"varying vec2 vUv;",

		"void main() {",

			"vUv = uv;",
			"gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",

		"}"

	].join( "\n" ),

	fragmentShader: [

        "#include <common>",

		"varying vec2 vUv;",

		"uniform sampler2D tDepth;",

		"uniform float cameraNear;",
		"uniform float cameraFar;",
		"uniform mat4 cameraProjectionMatrix;",
		"uniform mat4 cameraInverseProjectionMatrix;",

		"uniform vec2 size;",
        
        "#include <packing>",

        "float getDepth( const in vec2 screenPosition ) {",
        
            "#if DEPTH_PACKING == 3201",
                "return unpackRGBAToDepth( texture2D( tDepth, screenPosition ) );",
            "#else",
                "return texture2D( tDepth, screenPosition ).x;",
            "#endif",
        
        "}",
        
        "vec4 setDepth( const in float depth ) {",
        
            "#if DEPTH_PACKING == 3201",
                "return packDepthToRGBA( depth );",
            "#else",
                "return vec4( depth, 0, 0, 0 );",
            "#endif",
        
        "}",
        
        "float getViewZ( const in float depth ) {",
        
            "#if PERSPECTIVE_CAMERA == 1",
                "return perspectiveDepthToViewZ( depth, cameraNear, cameraFar );",
            "#else",
                "return orthographicDepthToViewZ( depth, cameraNear, cameraFar );",
            "#endif",
        
        "}",

        "vec3 getViewPosition( const in vec2 screenPosition, const in float depth, const in float viewZ ) {",

            "float clipW = cameraProjectionMatrix[2][3] * viewZ + cameraProjectionMatrix[3][3];",
            "vec4 clipPosition = vec4( ( vec3( screenPosition, depth ) - 0.5 ) * 2.0, 1.0 );",
            "clipPosition *= clipW;", // unprojection.
            "return ( cameraInverseProjectionMatrix * clipPosition ).xyz;",

        "}",

        "vec3 getViewNormal( const in vec3 viewPosition, const in vec2 screenPosition ) {",

            "return normalize( cross( dFdx( viewPosition ), dFdy( viewPosition ) ) );",
            
        "}",

		"void main() {",
	
            "float centerDepth = getDepth( vUv );",
            "if( centerDepth >= ( 1.0 - EPSILON ) ) {",
                "discard;",
            "}",

            "gl_FragColor.xy = vec2( dFdx( centerDepth ) * 1000.0 + 0.5, dFdy( centerDepth ) * 1000.0 + 0.5 );",
            "gl_FragColor.z = 0.0;",
            "gl_FragColor.a = 1.0;",
            "return;",
            "float centerViewZ = getViewZ( centerDepth );",
            "vec3 viewPosition = getViewPosition( vUv, centerDepth, centerViewZ );",

            "vec3 centerViewNormal = getViewNormal( viewPosition, vUv );",

            "gl_FragColor.rgb = packNormalToRGB( centerViewNormal );",
            "gl_FragColor.a = 1.0;",

		"}"

	].join( "\n" )

};
