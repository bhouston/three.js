#ifdef USE_AOMAP

#if defined( TEXTURE_SLOTS )
	vec2 aoUv = aoMapUV();
#else
	vec2 aoUv = vUv2;
#endif

	// reads channel R, intended to be used with a combined OcclusionRoughnessMetallic (RGB) texture
	float ambientOcclusion = ( aoMapTexelTransform( texture2D( aoMap, aoUv ) ).r - 1.0 ) * aoMapIntensity + 1.0;

	reflectedLight.indirectDiffuse *= ambientOcclusion;

	#if defined( USE_ENVMAP ) && defined( PHYSICAL )

		float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );

		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.specularRoughness );

	#endif

#endif
