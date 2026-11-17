import { positionWorld, reference, uniform, vec4 } from 'three/tsl';
import { getGridIrradiance } from './LightProbeGridNode.js';

/**
 * An indirect diffuse source for a light probe grid. It replaces the sources
 * behind it inside its bounds, and leaves them unchanged outside its bounds
 * or when hidden. The list order determines priority where grids overlap.
 * Keep the grid out of the scene's light list to avoid adding its light twice.
 *
 * @tsl
 * @function
 * @param {LightProbeGrid} grid - The light probe grid.
 * @returns {function(Object): Node<vec4>} The diffuse source.
 */
export const lightProbeGridDiffuse = ( grid ) => {

	const min = uniform( grid.boundingBox.min ).onRenderUpdate( () => {

		grid.updateBoundingBox();
		return grid.boundingBox.min;

	} );
	const max = uniform( grid.boundingBox.max );
	const resolution = uniform( grid.resolution );
	const intensity = reference( 'intensity', 'float', grid );
	const visible = reference( 'visible', 'bool', grid );

	return ( { normal } ) => {

		// Include surfaces on the bounds despite floating-point rounding.
		const margin = max.sub( min ).div( resolution ).mul( 0.01 );
		const inside = positionWorld.greaterThanEqual( min.sub( margin ) ).all()
			.and( positionWorld.lessThanEqual( max.add( margin ) ).all() );
		const coverage = inside.and( visible ).toFloat();
		const irradiance = getGridIrradiance( grid, min, max, resolution, intensity, positionWorld, normal );

		return vec4( irradiance.mul( coverage ), coverage );

	};

};
