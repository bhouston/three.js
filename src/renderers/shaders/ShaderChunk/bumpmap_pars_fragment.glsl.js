export default /* glsl */`
#ifdef USE_BUMPMAP

	uniform sampler2D bumpMap;
	uniform float bumpScale;

	// Bump Mapping Unparametrized Surfaces on the GPU by Morten S. Mikkelsen
	// https://mmikk.github.io/papers3d/mm_sfgrad_bump.pdf

	// Evaluate the derivative of the height w.r.t. screen-space using forward differencing (listing 2)

	vec2 dHdxy_fwd() {

		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );

		float scale = bumpScale;

		float Hll = scale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = scale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = scale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;

		return vec2( dBx, dBy );

	}

	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {

		// the surface derivatives are not normalized so the bump height keeps its units and the
		// slope stays the same at any view distance
		vec3 vSigmaX = dFdx( surf_pos.xyz );
		vec3 vSigmaY = dFdy( surf_pos.xyz );
		vec3 vN = surf_norm; // normalized

		#ifdef USE_BUMPMAP_OBJECTSPACE

			mat3 viewToObject = inverse( vBumpMatrix );
			vSigmaX = viewToObject * vSigmaX;
			vSigmaY = viewToObject * vSigmaY;
			vN = normalize( transpose( vBumpMatrix ) * surf_norm );

		#endif

		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );

		float fDet = dot( vSigmaX, R1 ) * faceDirection;

		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		vec3 perturbedNormal = abs( fDet ) * vN - vGrad;

		#ifdef USE_BUMPMAP_OBJECTSPACE

			perturbedNormal = transpose( viewToObject ) * perturbedNormal;

		#endif

		return normalize( perturbedNormal );

	}

#endif
`;
