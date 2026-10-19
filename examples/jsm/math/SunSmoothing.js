/**
 * Moves a normalized sunlight transmission toward its current target without
 * exceeding a maximum change per second. Targets can change on every update.
 *
 * @param {number} current - Current transmission in [0, 1].
 * @param {number} target - Target transmission in [0, 1].
 * @param {number} rate - Maximum transmission change per second, nonnegative.
 * @param {number} delta - Elapsed time in seconds, nonnegative.
 * @return {number} Updated transmission, without overshoot.
 */
function moveSunTransmission( current, target, rate, delta ) {

	const step = rate * delta;
	return current + Math.max( - step, Math.min( step, target - current ) );

}

export { moveSunTransmission };
