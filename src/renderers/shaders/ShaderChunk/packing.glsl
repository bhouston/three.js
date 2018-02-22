vec3 packNormalToRGB( const in vec3 normal ) {
  return normalize( normal ) * 0.5 + 0.5;
}

vec3 unpackRGBToNormal( const in vec3 rgb ) {
  return 2.0 * rgb.xyz - 1.0;
}
 
#define DEPTH_PACK_BYTES 4

vec4 packDepthToRGBA( in highp float v ) {

  const highp vec4 packFactor = vec4( 1.0, 255.0, 65025.0, 16581375.0 );
  highp vec4 res = fract( v * packFactor );
  res.xyz -= res.yzw * (1.0/255.0);
  
  #if DEPTH_PACK_BYTES == 4
    // do nothing.
  #elif DEPTH_PACK_BYTES == 3
    res.w = 0.0;
  #elif DEPTH_PACK_BYTES == 2
    res.zw = vec2( 0.0 );
  #elif DEPTH_PACK_BYTES == 1
    res.yzw = vec3( 0.0 );
  #endif
  res.xyzw = res.wzyx;

  return res;

}

float unpackRGBAToDepth( const in highp vec4 v ) {

  const highp vec4 unpackFactor = 1.0 / vec4( 1.0, 255.0, 65025.0, 16581375.0 );
 	highp float depth = dot( v.wzyx, unpackFactor );
  return depth;

}

// NOTE: viewZ/eyeZ is < 0 when in front of the camera per OpenGL conventions

float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
  return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float linearClipZ, const in float near, const in float far ) {
  return linearClipZ * ( near - far ) - near;
}

float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
  return (( near + viewZ ) * far ) / (( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float invClipZ, const in float near, const in float far ) {
  return ( near * far ) / ( ( far - near ) * invClipZ - far );
}
