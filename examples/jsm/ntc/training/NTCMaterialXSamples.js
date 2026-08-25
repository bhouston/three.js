/**
 * Shared catalog of the MaterialX (.mtlx) sample files that ship in
 * examples/materialx/ - used to populate the "Built-in MaterialX" dropdowns
 * in webgpu_materials_neural_texture.html, webgpu_materials_neural_material.html
 * and webgpu_materials_neural_appearance.html so the three examples always
 * offer the same set of sample files instead of three hand-maintained,
 * drifting lists.
 *
 * Entries are listed alphabetically by file name. `key` is the file name
 * without its .mtlx extension and is stable across releases - safe to use as
 * a <select> option value, an object key, or a URL query parameter (see the
 * `?test=` param read by webgpu_materials_neural_appearance.html and by
 * test/e2e/neural-appearance-training.js).
 *
 * Not every sample is a "training-friendly" tiling material - several of the
 * standard_surface_*_test.mtlx / open_pbr_surface_* / gltf_pbr_* files exist
 * as MaterialX conformance/feature tests rather than curated neural-training
 * targets. They are still fully loadable MeshPhysicalNodeMaterial sources
 * (or, for a couple of shading-model tests, will surface the loader's own
 * "unsupported surface" error, same as opening any other unsupported file
 * from disk) so they're included here for completeness rather than filtered
 * out.
 */

export const MATERIALX_SAMPLES = [
	{
		key: 'gltf_pbr_glass_dispersion',
		file: 'gltf_pbr_glass_dispersion.mtlx',
		label: 'glTF PBR dispersion glass',
		description: 'Rough, near-white glass with chromatic dispersion, built from the gltf_pbr shading model.'
	},
	{
		key: 'neural_train_alpha_cutoff',
		file: 'neural_train_alpha_cutoff.mtlx',
		label: 'Alpha cutoff waves',
		description: 'Wavy UV-driven opacity pattern rendered with mask (alpha test) cutoff rather than blending.'
	},
	{
		key: 'neural_train_brick',
		file: 'neural_train_brick.mtlx',
		label: 'Brick wall',
		description: 'Procedural brick-and-mortar tiling pattern generated from UV coordinates.'
	},
	{
		key: 'neural_train_checkerboard',
		file: 'neural_train_checkerboard.mtlx',
		label: 'Checkerboard 4x4',
		description: 'Simple 4x4 black/white checkerboard base color, a minimal high-frequency tiling test.'
	},
	{
		key: 'neural_train_checkerboard_normal',
		file: 'neural_train_checkerboard_normal.mtlx',
		label: 'Checkerboard + normal map',
		description: 'Checkerboard base color combined with a procedural bump/normal pattern.'
	},
	{
		key: 'neural_train_checkerboard_transparency',
		file: 'neural_train_checkerboard_transparency.mtlx',
		label: 'Checkerboard transparency',
		description: 'Checkerboard pattern driving blended (non-cutoff) opacity.'
	},
	{
		key: 'neural_train_emissive_grid',
		file: 'neural_train_emissive_grid.mtlx',
		label: 'Emissive rainbow waves',
		description: 'Checkerboard-driven emissive rainbow gradient over UV space.'
	},
	{
		key: 'neural_train_glossy_constant',
		file: 'neural_train_glossy_constant.mtlx',
		label: 'Glossy constant',
		description: 'Flat, uniform glossy surface with no spatial variation - a minimal specular baseline.'
	},
	{
		key: 'neural_train_glossy_gold',
		file: 'neural_train_glossy_gold.mtlx',
		label: 'Metallic gold',
		description: 'Polished metallic gold, a constant-color high-metalness/low-roughness surface.'
	},
	{
		key: 'neural_train_glossy_red',
		file: 'neural_train_glossy_red.mtlx',
		label: 'Glossy red',
		description: 'Flat glossy red dielectric surface with visible specular highlight.'
	},
	{
		key: 'neural_train_lambert_constant',
		file: 'neural_train_lambert_constant.mtlx',
		label: 'Lambert constant',
		description: 'Flat, uniform matte diffuse surface with no spatial variation - a minimal diffuse baseline.'
	},
	{
		key: 'neural_train_lambert_red',
		file: 'neural_train_lambert_red.mtlx',
		label: 'Lambert red',
		description: 'Flat matte red diffuse surface with no specular response.'
	},
	{
		key: 'neural_train_normal_map',
		file: 'neural_train_normal_map.mtlx',
		label: 'Normal map waves',
		description: 'Procedural wavy normal-map pattern over a flat base color.'
	},
	{
		key: 'neural_train_road_aggregate',
		file: 'neural_train_road_aggregate.mtlx',
		label: 'Road aggregate',
		description: 'Procedural asphalt/aggregate tiling pattern generated from UV coordinates.'
	},
	{
		key: 'neural_train_uv_grid',
		file: 'neural_train_uv_grid.mtlx',
		label: 'UV rainbow waves',
		description: 'Smooth rainbow gradient over UV space, a low-frequency default training target.'
	},
	{
		key: 'neural_train_uv_grid_glossy',
		file: 'neural_train_uv_grid_glossy.mtlx',
		label: 'UV rainbow glossy',
		description: 'UV rainbow gradient combined with a glossy specular response.'
	},
	{
		key: 'neural_train_velvet',
		file: 'neural_train_velvet.mtlx',
		label: 'Velvet',
		description: 'Sheen-driven velvet cloth look with a deep red diffuse base.'
	},
	{
		key: 'open_pbr_surface_honey',
		file: 'open_pbr_surface_honey.mtlx',
		label: 'OpenPBR honey',
		description: 'Amber, translucent honey material built from the open_pbr_surface shading model.'
	},
	{
		key: 'open_pbr_surface_pearl',
		file: 'open_pbr_surface_pearl.mtlx',
		label: 'OpenPBR pearl',
		description: 'Soft, low-roughness pearlescent surface built from the open_pbr_surface shading model.'
	},
	{
		key: 'open_pbr_surface_velvet',
		file: 'open_pbr_surface_velvet.mtlx',
		label: 'OpenPBR velvet',
		description: 'Dark sheen-driven velvet look built from the open_pbr_surface shading model.'
	},
	{
		key: 'standard_surface_color3_vec3_cm_test',
		file: 'standard_surface_color3_vec3_cm_test.mtlx',
		label: 'Color/vector conversion test',
		description: 'MaterialX conformance test exercising color3/vector3 conversions inside a normal-map node graph.'
	},
	{
		key: 'standard_surface_combined_test',
		file: 'standard_surface_combined_test.mtlx',
		label: 'Combined channels test',
		description: 'MaterialX conformance test combining base color, opacity and other standard_surface inputs together.'
	},
	{
		key: 'standard_surface_conditional_if_float',
		file: 'standard_surface_conditional_if_float.mtlx',
		label: 'Conditional (ifgreater) test',
		description: 'MaterialX conformance test driving base color through an ifgreater conditional node graph.'
	},
	{
		key: 'standard_surface_heightnormal',
		file: 'standard_surface_heightnormal.mtlx',
		label: 'Height-to-normal (image) test',
		description: 'MaterialX conformance test converting a height image into a normal map via heighttonormal.'
	},
	{
		key: 'standard_surface_heighttonormal_normal_input',
		file: 'standard_surface_heighttonormal_normal_input.mtlx',
		label: 'Height-to-normal (input) test',
		description: 'MaterialX conformance test feeding a procedural height field into a normal input via heighttonormal.'
	},
	{
		key: 'standard_surface_image_transform',
		file: 'standard_surface_image_transform.mtlx',
		label: 'Image transform (place2d) test',
		description: 'MaterialX conformance test applying a scale/rotate/translate place2d transform to an image texture.'
	},
	{
		key: 'standard_surface_ior_test',
		file: 'standard_surface_ior_test.mtlx',
		label: 'Specular IOR test',
		description: 'MaterialX conformance test of a high, non-metallic specular index of refraction.'
	},
	{
		key: 'standard_surface_opacity_only_test',
		file: 'standard_surface_opacity_only_test.mtlx',
		label: 'Opacity-only test',
		description: 'MaterialX conformance test of blended opacity with no other channel variation.'
	},
	{
		key: 'standard_surface_opacity_test',
		file: 'standard_surface_opacity_test.mtlx',
		label: 'Opacity test',
		description: 'MaterialX conformance test of blended opacity over a colored diffuse surface.'
	},
	{
		key: 'standard_surface_rotate2d_test',
		file: 'standard_surface_rotate2d_test.mtlx',
		label: 'Rotate2D test',
		description: 'MaterialX conformance test rotating a 2D texture coordinate before sampling.'
	},
	{
		key: 'standard_surface_rotate3d_test',
		file: 'standard_surface_rotate3d_test.mtlx',
		label: 'Rotate3D test',
		description: 'MaterialX conformance test of a 3D rotation node graph feeding base color.'
	},
	{
		key: 'standard_surface_roughness_test',
		file: 'standard_surface_roughness_test.mtlx',
		label: 'Roughness map test',
		description: 'MaterialX conformance test driving specular roughness from a procedural map.'
	},
	{
		key: 'standard_surface_sheen_test',
		file: 'standard_surface_sheen_test.mtlx',
		label: 'Sheen test',
		description: 'MaterialX conformance test of the sheen/sheen_color/sheen_roughness inputs over a blue base.'
	},
	{
		key: 'standard_surface_specular_test',
		file: 'standard_surface_specular_test.mtlx',
		label: 'Specular test',
		description: 'MaterialX conformance test of the specular/specular_color inputs over a blue base.'
	},
	{
		key: 'standard_surface_texture_opacity_test',
		file: 'standard_surface_texture_opacity_test.mtlx',
		label: 'Texture opacity test',
		description: 'MaterialX conformance test driving opacity from a procedural texture graph.'
	},
	{
		key: 'standard_surface_thin_film_ior_clamp_test',
		file: 'standard_surface_thin_film_ior_clamp_test.mtlx',
		label: 'Thin film IOR clamp test',
		description: 'MaterialX conformance test of thin-film interference with a clamped, high thin-film IOR.'
	},
	{
		key: 'standard_surface_thin_film_rainbow_test',
		file: 'standard_surface_thin_film_rainbow_test.mtlx',
		label: 'Thin film rainbow test',
		description: 'MaterialX conformance test of thin-film interference producing an iridescent rainbow highlight.'
	},
	{
		key: 'standard_surface_transmission_only_test',
		file: 'standard_surface_transmission_only_test.mtlx',
		label: 'Transmission-only test',
		description: 'MaterialX conformance test of full transmission (clear glass) with no roughness.'
	},
	{
		key: 'standard_surface_transmission_rough',
		file: 'standard_surface_transmission_rough.mtlx',
		label: 'Rough transmission test',
		description: 'MaterialX conformance test of frosted, near-full transmission with visible surface roughness.'
	},
	{
		key: 'standard_surface_transmission_test',
		file: 'standard_surface_transmission_test.mtlx',
		label: 'Transmission test',
		description: 'MaterialX conformance test of high, slightly tinted transmission (clear glass/liquid).'
	},
	{
		key: 'wood',
		file: 'wood.mtlx',
		label: 'Hardwood floor',
		description: 'Tiled hardwood floor built from the webgpu_lights_physical.html diffuse/bump/roughness bitmaps.'
	}
];

/**
 * @param {string} key
 * @returns {{key:string,file:string,label:string,description:string}|undefined}
 */
export function getMaterialXSample( key ) {

	return MATERIALX_SAMPLES.find( ( sample ) => sample.key === key );

}

/**
 * Populates a <select> element with one <option> per sample (in the
 * catalog's alphabetical-by-file order) followed by a trailing, disabled
 * "Custom file" option - matching the pattern all three neural-* examples
 * use for their "Open .mtlx" file picker.
 *
 * @param {HTMLSelectElement} selectElement
 * @param {{customLabel?:string}} [options]
 */
export function populateMaterialXSelect( selectElement, options = {} ) {

	const customLabel = options.customLabel || 'Custom file';

	for ( const sample of MATERIALX_SAMPLES ) {

		const option = document.createElement( 'option' );
		option.value = sample.key;
		option.textContent = sample.label;
		selectElement.appendChild( option );

	}

	const customOption = document.createElement( 'option' );
	customOption.value = '';
	customOption.disabled = true;
	customOption.textContent = customLabel;
	selectElement.appendChild( customOption );

}
