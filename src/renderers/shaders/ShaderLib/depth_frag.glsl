precision highp float;

#if DEPTH_PACKING == 3200

	uniform float opacity;

#endif

#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

varying vec4 clipSpacePosition;

void main() {

	#include <clipping_planes_fragment>

	vec4 diffuseColor = vec4( 1.0 );

	#if DEPTH_PACKING == 3200

		diffuseColor.a = opacity;

	#endif

	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>

	#include <logdepthbuf_fragment>

	// https://stackoverflow.com/a/12904072
	float far = gl_DepthRange.far;
	float near = gl_DepthRange.near;
	highp float ndc_depth = clipSpacePosition.z / clipSpacePosition.w;
	highp float fragCoordZ = (((far-near) * ndc_depth ) + near + far) / 2.0;

	#if DEPTH_PACKING == 3200

		gl_FragColor = vec4( vec3( fragCoordZ ), opacity );

	#elif DEPTH_PACKING == 3201

		gl_FragColor = packDepthToRGBA( fragCoordZ );

	#endif

}
