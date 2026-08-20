import { describe, test, expect } from 'vitest';
import * as Constants from '@src/constants.js';

describe( 'Constants', () => {

	test( 'default values', () => {

		expect( Constants.MOUSE ).toEqual( { LEFT: 0, MIDDLE: 1, RIGHT: 2, ROTATE: 0, DOLLY: 1, PAN: 2 } );
		expect( Constants.TOUCH ).toEqual( { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 } );

		expect( Constants.CullFaceNone ).toBe( 0 );
		expect( Constants.CullFaceBack ).toBe( 1 );
		expect( Constants.CullFaceFront ).toBe( 2 );
		expect( Constants.CullFaceFrontBack ).toBe( 3 );

		expect( Constants.BasicShadowMap ).toBe( 0 );
		expect( Constants.PCFShadowMap ).toBe( 1 );
		expect( Constants.PCFSoftShadowMap ).toBe( 2 );
		expect( Constants.VSMShadowMap ).toBe( 3 );

		expect( Constants.FrontSide ).toBe( 0 );
		expect( Constants.BackSide ).toBe( 1 );
		expect( Constants.DoubleSide ).toBe( 2 );

		expect( Constants.NoBlending ).toBe( 0 );
		expect( Constants.NormalBlending ).toBe( 1 );
		expect( Constants.AdditiveBlending ).toBe( 2 );
		expect( Constants.SubtractiveBlending ).toBe( 3 );
		expect( Constants.MultiplyBlending ).toBe( 4 );
		expect( Constants.CustomBlending ).toBe( 5 );

		expect( Constants.AddEquation ).toBe( 100 );
		expect( Constants.SubtractEquation ).toBe( 101 );
		expect( Constants.ReverseSubtractEquation ).toBe( 102 );
		expect( Constants.MinEquation ).toBe( 103 );
		expect( Constants.MaxEquation ).toBe( 104 );

		expect( Constants.ZeroFactor ).toBe( 200 );
		expect( Constants.OneFactor ).toBe( 201 );
		expect( Constants.SrcColorFactor ).toBe( 202 );
		expect( Constants.OneMinusSrcColorFactor ).toBe( 203 );
		expect( Constants.SrcAlphaFactor ).toBe( 204 );
		expect( Constants.OneMinusSrcAlphaFactor ).toBe( 205 );
		expect( Constants.DstAlphaFactor ).toBe( 206 );
		expect( Constants.OneMinusDstAlphaFactor ).toBe( 207 );
		expect( Constants.DstColorFactor ).toBe( 208 );
		expect( Constants.OneMinusDstColorFactor ).toBe( 209 );
		expect( Constants.SrcAlphaSaturateFactor ).toBe( 210 );
		expect( Constants.ConstantColorFactor ).toBe( 211 );
		expect( Constants.OneMinusConstantColorFactor ).toBe( 212 );
		expect( Constants.ConstantAlphaFactor ).toBe( 213 );
		expect( Constants.OneMinusConstantAlphaFactor ).toBe( 214 );

		expect( Constants.NeverDepth ).toBe( 0 );
		expect( Constants.AlwaysDepth ).toBe( 1 );
		expect( Constants.LessDepth ).toBe( 2 );
		expect( Constants.LessEqualDepth ).toBe( 3 );
		expect( Constants.EqualDepth ).toBe( 4 );
		expect( Constants.GreaterEqualDepth ).toBe( 5 );
		expect( Constants.GreaterDepth ).toBe( 6 );
		expect( Constants.NotEqualDepth ).toBe( 7 );

		expect( Constants.MultiplyOperation ).toBe( 0 );
		expect( Constants.MixOperation ).toBe( 1 );
		expect( Constants.AddOperation ).toBe( 2 );

		expect( Constants.NoToneMapping ).toBe( 0 );
		expect( Constants.LinearToneMapping ).toBe( 1 );
		expect( Constants.ReinhardToneMapping ).toBe( 2 );
		expect( Constants.CineonToneMapping ).toBe( 3 );
		expect( Constants.ACESFilmicToneMapping ).toBe( 4 );
		expect( Constants.CustomToneMapping ).toBe( 5 );
		expect( Constants.AgXToneMapping ).toBe( 6 );
		expect( Constants.NeutralToneMapping ).toBe( 7 );

		expect( Constants.AttachedBindMode ).toBe( 'attached' );
		expect( Constants.DetachedBindMode ).toBe( 'detached' );

		expect( Constants.UVMapping ).toBe( 300 );
		expect( Constants.CubeReflectionMapping ).toBe( 301 );
		expect( Constants.CubeRefractionMapping ).toBe( 302 );
		expect( Constants.EquirectangularReflectionMapping ).toBe( 303 );
		expect( Constants.EquirectangularRefractionMapping ).toBe( 304 );
		expect( Constants.CubeUVReflectionMapping ).toBe( 306 );

		expect( Constants.RepeatWrapping ).toBe( 1000 );
		expect( Constants.ClampToEdgeWrapping ).toBe( 1001 );
		expect( Constants.MirroredRepeatWrapping ).toBe( 1002 );

		expect( Constants.NearestFilter ).toBe( 1003 );
		expect( Constants.NearestMipMapNearestFilter ).toBe( 1004 );
		expect( Constants.NearestMipMapLinearFilter ).toBe( 1005 );
		expect( Constants.LinearFilter ).toBe( 1006 );
		expect( Constants.LinearMipMapNearestFilter ).toBe( 1007 );
		expect( Constants.LinearMipMapLinearFilter ).toBe( 1008 );
		expect( Constants.UnsignedByteType ).toBe( 1009 );

		expect( Constants.ByteType ).toBe( 1010 );
		expect( Constants.ShortType ).toBe( 1011 );
		expect( Constants.UnsignedShortType ).toBe( 1012 );
		expect( Constants.IntType ).toBe( 1013 );
		expect( Constants.UnsignedIntType ).toBe( 1014 );
		expect( Constants.FloatType ).toBe( 1015 );
		expect( Constants.HalfFloatType ).toBe( 1016 );
		expect( Constants.UnsignedShort4444Type ).toBe( 1017 );
		expect( Constants.UnsignedShort5551Type ).toBe( 1018 );
		expect( Constants.UnsignedInt248Type ).toBe( 1020 );

		expect( Constants.AlphaFormat ).toBe( 1021 );
		expect( Constants.RGBAFormat ).toBe( 1023 );
		expect( Constants.DepthFormat ).toBe( 1026 );
		expect( Constants.DepthStencilFormat ).toBe( 1027 );
		expect( Constants.RedFormat ).toBe( 1028 );
		expect( Constants.RedIntegerFormat ).toBe( 1029 );
		expect( Constants.RGFormat ).toBe( 1030 );
		expect( Constants.RGIntegerFormat ).toBe( 1031 );
		expect( Constants.RGBAIntegerFormat ).toBe( 1033 );

		expect( Constants.RGB_S3TC_DXT1_Format ).toBe( 33776 );
		expect( Constants.RGBA_S3TC_DXT1_Format ).toBe( 33777 );
		expect( Constants.RGBA_S3TC_DXT3_Format ).toBe( 33778 );
		expect( Constants.RGBA_S3TC_DXT5_Format ).toBe( 33779 );
		expect( Constants.RGB_PVRTC_4BPPV1_Format ).toBe( 35840 );
		expect( Constants.RGB_PVRTC_2BPPV1_Format ).toBe( 35841 );
		expect( Constants.RGBA_PVRTC_4BPPV1_Format ).toBe( 35842 );
		expect( Constants.RGBA_PVRTC_2BPPV1_Format ).toBe( 35843 );
		expect( Constants.RGB_ETC1_Format ).toBe( 36196 );
		expect( Constants.RGB_ETC2_Format ).toBe( 37492 );
		expect( Constants.RGBA_ASTC_4x4_Format ).toBe( 37808 );
		expect( Constants.RGBA_ASTC_5x4_Format ).toBe( 37809 );
		expect( Constants.RGBA_ASTC_5x5_Format ).toBe( 37810 );
		expect( Constants.RGBA_ASTC_6x5_Format ).toBe( 37811 );
		expect( Constants.RGBA_ASTC_6x6_Format ).toBe( 37812 );
		expect( Constants.RGBA_ASTC_8x5_Format ).toBe( 37813 );
		expect( Constants.RGBA_ASTC_8x6_Format ).toBe( 37814 );
		expect( Constants.RGBA_ASTC_8x8_Format ).toBe( 37815 );
		expect( Constants.RGBA_ASTC_10x5_Format ).toBe( 37816 );
		expect( Constants.RGBA_ASTC_10x6_Format ).toBe( 37817 );
		expect( Constants.RGBA_ASTC_10x8_Format ).toBe( 37818 );
		expect( Constants.RGBA_ASTC_10x10_Format ).toBe( 37819 );
		expect( Constants.RGBA_ASTC_12x10_Format ).toBe( 37820 );
		expect( Constants.RGBA_ASTC_12x12_Format ).toBe( 37821 );
		expect( Constants.RGBA_BPTC_Format ).toBe( 36492 );
		expect( Constants.RGB_BPTC_SIGNED_Format ).toBe( 36494 );
		expect( Constants.RGB_BPTC_UNSIGNED_Format ).toBe( 36495 );
		expect( Constants.RED_RGTC1_Format ).toBe( 36283 );
		expect( Constants.SIGNED_RED_RGTC1_Format ).toBe( 36284 );
		expect( Constants.RED_GREEN_RGTC2_Format ).toBe( 36285 );
		expect( Constants.SIGNED_RED_GREEN_RGTC2_Format ).toBe( 36286 );

		expect( Constants.LoopOnce ).toBe( 2200 );
		expect( Constants.LoopRepeat ).toBe( 2201 );
		expect( Constants.LoopPingPong ).toBe( 2202 );

		expect( Constants.InterpolateDiscrete ).toBe( 2300 );
		expect( Constants.InterpolateLinear ).toBe( 2301 );
		expect( Constants.InterpolateSmooth ).toBe( 2302 );

		expect( Constants.ZeroCurvatureEnding ).toBe( 2400 );
		expect( Constants.ZeroSlopeEnding ).toBe( 2401 );
		expect( Constants.WrapAroundEnding ).toBe( 2402 );

		expect( Constants.NormalAnimationBlendMode ).toBe( 2500 );
		expect( Constants.AdditiveAnimationBlendMode ).toBe( 2501 );

		expect( Constants.TrianglesDrawMode ).toBe( 0 );
		expect( Constants.TriangleStripDrawMode ).toBe( 1 );
		expect( Constants.TriangleFanDrawMode ).toBe( 2 );

		expect( Constants.BasicDepthPacking ).toBe( 3200 );
		expect( Constants.RGBADepthPacking ).toBe( 3201 );

		expect( Constants.TangentSpaceNormalMap ).toBe( 0 );
		expect( Constants.ObjectSpaceNormalMap ).toBe( 1 );

		expect( Constants.NoColorSpace ).toBe( '' );
		expect( Constants.SRGBColorSpace ).toBe( 'srgb' );
		expect( Constants.LinearSRGBColorSpace ).toBe( 'srgb-linear' );

		expect( Constants.ZeroStencilOp ).toBe( 0 );
		expect( Constants.KeepStencilOp ).toBe( 7680 );
		expect( Constants.ReplaceStencilOp ).toBe( 7681 );
		expect( Constants.IncrementStencilOp ).toBe( 7682 );
		expect( Constants.DecrementStencilOp ).toBe( 7683 );
		expect( Constants.IncrementWrapStencilOp ).toBe( 34055 );
		expect( Constants.DecrementWrapStencilOp ).toBe( 34056 );
		expect( Constants.InvertStencilOp ).toBe( 5386 );

		expect( Constants.NeverStencilFunc ).toBe( 512 );
		expect( Constants.LessStencilFunc ).toBe( 513 );
		expect( Constants.EqualStencilFunc ).toBe( 514 );
		expect( Constants.LessEqualStencilFunc ).toBe( 515 );
		expect( Constants.GreaterStencilFunc ).toBe( 516 );
		expect( Constants.NotEqualStencilFunc ).toBe( 517 );
		expect( Constants.GreaterEqualStencilFunc ).toBe( 518 );
		expect( Constants.AlwaysStencilFunc ).toBe( 519 );

		expect( Constants.StaticDrawUsage ).toBe( 35044 );
		expect( Constants.DynamicDrawUsage ).toBe( 35048 );
		expect( Constants.StreamDrawUsage ).toBe( 35040 );
		expect( Constants.StaticReadUsage ).toBe( 35045 );
		expect( Constants.DynamicReadUsage ).toBe( 35049 );
		expect( Constants.StreamReadUsage ).toBe( 35041 );
		expect( Constants.StaticCopyUsage ).toBe( 35046 );
		expect( Constants.DynamicCopyUsage ).toBe( 35050 );
		expect( Constants.StreamCopyUsage ).toBe( 35042 );

		expect( Constants.GLSL1 ).toBe( '100' );
		expect( Constants.GLSL3 ).toBe( '300 es' );

	} );

} );
