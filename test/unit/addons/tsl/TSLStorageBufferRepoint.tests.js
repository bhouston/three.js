import * as THREE from 'three/webgpu';
import { Fn, instanceIndex, storage } from 'three/tsl';
import { getSharedRenderer } from './gpu-test-utils.js';

// A minimal, FFT-independent repro of the mechanism suspected in FFT2D.js's ping-pong buffer
// sharing attempt (see that file's constructor comment): a *single* compiled compute kernel,
// built once, whose storage buffer nodes are repointed (`.value = someOtherAttribute`) at a
// *different* buffer immediately before each of many back-to-back `renderer.compute()` calls,
// with no `await` between them. FFT2D's version of this pattern intermittently corrupted data on
// real hardware for large (non-fused, many-dispatch-per-transform) images; this test strips away
// the FFT math entirely to isolate just the repointing mechanism, so a failure here would confirm
// a genuine three.js/TSL WebGPU-backend bug rather than anything FFT-specific.
//
// Two ping-pong `StorageBufferAttribute`s (`bufA`/`bufB`), one `float` each, `count` elements.
// One kernel, built once: `out[i] = in[i] + 1`. `count` iterations run back-to-back, alternating
// which buffer is "in" and which is "out" (mirroring `FFT2D`'s reverted `_dispatchPingPong`), all
// recorded before the first readback. If every dispatch's bind group correctly captured the
// node values *at the time `renderer.compute()` was called*, the final live buffer holds
// `initial + iterations` in every element; any staleness/aliasing in that repointing would show
// up as a wrong value there.

function makeBuffer( count, initialValue ) {

	const data = new Float32Array( count ).fill( initialValue );

	return new THREE.StorageBufferAttribute( data, 1 );

}

async function readBuffer( renderer, attribute, count ) {

	const data = new Float32Array( await renderer.getArrayBufferAsync( attribute ) );

	return data.slice( 0, count );

}

export default QUnit.module( 'TSL', () => {

	QUnit.module( 'Storage buffer node repointing (GPGPU, WebGPU-only)', () => {

		function repointTest( name, run ) {

			QUnit.test( name, async ( assert ) => {

				const renderer = await getSharedRenderer( 'webgpu' );

				if ( renderer === null ) {

					assert.ok( true, 'SKIPPED: "webgpu" backend is not available in this environment.' );
					return;

				}

				await run( assert, renderer );

			} );

		}

		[ 8, 64, 256 ].forEach( ( iterations ) => {

			repointTest( `single shared kernel survives ${ iterations } rapid back-to-back repointed dispatches`, async ( assert, renderer ) => {

				const count = 64;
				const initialValue = 1;

				const attrA = makeBuffer( count, initialValue );
				const attrB = makeBuffer( count, 0 );

				// One shared, repointable read/write node pair -- exactly the pattern reverted in
				// FFT2D.js. `readNode` is read-only, matching FFT2D's usage.
				const readNode = storage( attrA, 'float', count ).toReadOnly();
				const writeNode = storage( attrB, 'float', count );

				// Built once; every iteration below reuses this same compiled kernel, only
				// repointing `readNode`/`writeNode`'s `.value` beforehand -- never rebuilding it.
				const kernel = Fn( () => {

					const v = readNode.element( instanceIndex ).toVar();
					writeNode.element( instanceIndex ).assign( v.add( 1 ) );

				} )().compute( count );

				let current = 'A'; // which attribute currently holds the live data

				for ( let i = 0; i < iterations; i ++ ) {

					readNode.value = current === 'A' ? attrA : attrB;
					writeNode.value = current === 'A' ? attrB : attrA;

					renderer.compute( kernel );

					current = current === 'A' ? 'B' : 'A';

				}

				// After the loop, `current` names the buffer the *next* pass would read from --
				// i.e. the one the last dispatch just wrote to, which holds the live result.
				const liveAttribute = current === 'A' ? attrA : attrB;
				const result = await readBuffer( renderer, liveAttribute, count );

				const expected = initialValue + iterations;
				let firstWrong = -1;

				for ( let i = 0; i < count; i ++ ) {

					if ( result[ i ] !== expected ) {

						firstWrong = i;
						break;

					}

				}

				assert.ok( firstWrong === -1, firstWrong === -1
					? `all ${ count } elements equal ${ expected } after ${ iterations } repointed dispatches`
					: `element ${ firstWrong } was ${ result[ firstWrong ] }, expected ${ expected } after ${ iterations } repointed dispatches (first mismatch)`
				);

				attrA.dispose?.();
				attrB.dispose?.();

			} );

		} );

		repointTest( 'two distinct kernels sharing one repointed node pair stay correct when interleaved', async ( assert, renderer ) => {

			// Closer to FFT2D's real shape: several *different* compiled kernels (not just one)
			// all referencing the same shared read/write node pair, dispatched in an interleaved
			// sequence -- in case cross-kernel bind-group aliasing on the shared nodes, rather
			// than same-kernel reuse, is what matters.

			const count = 64;
			const iterations = 64;

			const attrA = makeBuffer( count, 1 );
			const attrB = makeBuffer( count, 0 );

			const readNode = storage( attrA, 'float', count ).toReadOnly();
			const writeNode = storage( attrB, 'float', count );

			const incrementKernel = Fn( () => {

				const v = readNode.element( instanceIndex ).toVar();
				writeNode.element( instanceIndex ).assign( v.add( 1 ) );

			} )().compute( count );

			const doubleKernel = Fn( () => {

				const v = readNode.element( instanceIndex ).toVar();
				writeNode.element( instanceIndex ).assign( v.mul( 2 ) );

			} )().compute( count );

			let current = 'A';
			let expected = 1;

			for ( let i = 0; i < iterations; i ++ ) {

				readNode.value = current === 'A' ? attrA : attrB;
				writeNode.value = current === 'A' ? attrB : attrA;

				const useDouble = ( i % 3 === 0 );
				renderer.compute( useDouble ? doubleKernel : incrementKernel );
				expected = useDouble ? expected * 2 : expected + 1;

				current = current === 'A' ? 'B' : 'A';

			}

			const liveAttribute = current === 'A' ? attrA : attrB;
			const result = await readBuffer( renderer, liveAttribute, count );

			let firstWrong = -1;

			for ( let i = 0; i < count; i ++ ) {

				if ( result[ i ] !== expected ) {

					firstWrong = i;
					break;

				}

			}

			assert.ok( firstWrong === -1, firstWrong === -1
				? `all ${ count } elements equal ${ expected } after ${ iterations } interleaved dispatches across 2 kernels`
				: `element ${ firstWrong } was ${ result[ firstWrong ] }, expected ${ expected } (first mismatch)`
			);

			attrA.dispose?.();
			attrB.dispose?.();

		} );

		repointTest( 'repeated repoint-and-dispatch runs stay correct across many separate calls (not just one long chain)', async ( assert, renderer ) => {

			// Runs the single-kernel repoint chain from the first test many times over, each with
			// its own fresh buffers/kernel, to catch nondeterministic ("sometimes") corruption that
			// a single run might not hit -- matching what was actually observed (works most of the
			// time, occasionally doesn't).

			const count = 64;
			const iterations = 32;
			const runs = 12;

			for ( let run = 0; run < runs; run ++ ) {

				const initialValue = run + 1;

				const attrA = makeBuffer( count, initialValue );
				const attrB = makeBuffer( count, 0 );

				const readNode = storage( attrA, 'float', count ).toReadOnly();
				const writeNode = storage( attrB, 'float', count );

				const kernel = Fn( () => {

					const v = readNode.element( instanceIndex ).toVar();
					writeNode.element( instanceIndex ).assign( v.add( 1 ) );

				} )().compute( count );

				let current = 'A';

				for ( let i = 0; i < iterations; i ++ ) {

					readNode.value = current === 'A' ? attrA : attrB;
					writeNode.value = current === 'A' ? attrB : attrA;

					renderer.compute( kernel );

					current = current === 'A' ? 'B' : 'A';

				}

				const liveAttribute = current === 'A' ? attrA : attrB;
				const result = await readBuffer( renderer, liveAttribute, count );

				const expected = initialValue + iterations;
				let firstWrong = -1;

				for ( let i = 0; i < count; i ++ ) {

					if ( result[ i ] !== expected ) {

						firstWrong = i;
						break;

					}

				}

				assert.ok( firstWrong === -1, firstWrong === -1
					? `run ${ run }: all elements equal ${ expected }`
					: `run ${ run }: element ${ firstWrong } was ${ result[ firstWrong ] }, expected ${ expected } (first mismatch)`
				);

				attrA.dispose?.();
				attrB.dispose?.();

			}

		} );

	} );

} );
