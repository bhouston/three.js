/**
 * @author mrdoob / http://mrdoob.com/
 * @author marklundin / http://mark-lundin.com/
 * @author alteredq / http://alteredqualia.com/
 */

THREE.ParallaxBarrierEffect = function ( renderer ) {

	  var _camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  var _scene = new THREE.Scene();

  var _stereo = new THREE.StereoCamera();

  var _params = { minFilter: THREE.LinearFilter, magFilter: THREE.NearestFilter, format: THREE.RGBAFormat };

  var _renderTargetL = new THREE.WebGLRenderTarget(512, 512, _params);
  var _renderTargetR = new THREE.WebGLRenderTarget(512, 512, _params);

  var _material = new THREE.ShaderMaterial({

    uniforms: {

      'mapLeft': { value: _renderTargetL.texture },
      'mapRight': { value: _renderTargetR.texture },
      'resolution': { value: new THREE.Vector2( 512, 512 ) },

    },

    vertexShader: [

      'varying vec2 vUv;',

      'void main() {',

      ' vUv = vec2( uv.x, uv.y );',
      ' gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',

      '}',

    ].join('\n'),

    fragmentShader: [

      'uniform sampler2D mapLeft;',
      'uniform sampler2D mapRight;',
      'uniform vec2 resolution;',
      'varying vec2 vUv;',

      'void main() {',

      ' vec2 uv = vUv;',

      ' if ( ( mod( gl_FragCoord.x + 1.0, 2.0 ) ) > 1.00 ) {',

      '   gl_FragColor = texture2D( mapLeft, uv );',

      ' } else {',

      '   gl_FragColor = texture2D( mapRight, uv - vec2( resolution.x, 0 ) );',

      ' }',

      '}',

    ].join('\n'),

  });

  var mesh = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), _material);
  _scene.add(mesh);

  this.setSize = function (width, height) {

    renderer.setSize(width, height);

    var pixelRatio = renderer.getPixelRatio();

    _material.uniforms['resolution'].value.set( 1.0 / width, 1.0 / height );

    _renderTargetL.setSize(width * pixelRatio / 2 + 1, height * pixelRatio);
    _renderTargetR.setSize(width * pixelRatio / 2 + 1, height * pixelRatio);

  };

  this.render = function (scene, camera) {

    scene.updateMatrixWorld();

    if (camera.parent === null) camera.updateMatrixWorld();

    _stereo.update(camera);

    renderer.render(scene, _stereo.cameraL, _renderTargetL, true);
    renderer.render(scene, _stereo.cameraR, _renderTargetR, true);
    renderer.render(_scene, _camera);

  };

};
