varying vec3 vWorldPosition;

#include <common>
#include <clipping_planes_pars_vertex>

#if NUM_CLIPPING_PLANES == 0
	#if ! defined( VARYING_VVIEWPOSITION )
   		varying vec3 vViewPosition;
   		#define VARYING_VVIEWPOSITION 1
   #endif
#endif

void main() {

	vWorldPosition = transformDirection( position, modelMatrix );

	#include <begin_vertex>
	#include <project_vertex>
	#include <clipping_planes_vertex>

	vViewPosition = - mvPosition.xyz;

}
