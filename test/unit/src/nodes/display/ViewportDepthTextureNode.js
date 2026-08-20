import { describe, test, expect } from 'vitest';
import { viewportDepthTexture } from '@src/nodes/display/ViewportDepthTextureNode.js';
import { DepthTexture } from '@src/textures/DepthTexture.js';
import { RenderTarget } from '@src/core/RenderTarget.js';

describe( 'Nodes', () => {

	describe( 'Display', () => {

		describe( 'ViewportDepthTextureNode', () => {

			describe( 'Depth Texture Caching', () => {

				test( 'getTextureForReference returns shared buffer for null reference', () => {

					const node = viewportDepthTexture();
					const texture = node.getTextureForReference( null );

					expect( texture ).toBe( node.defaultFramebuffer );

				} );

				test( 'different references get independent cached depth textures', () => {

					const node = viewportDepthTexture();

					// Create mock canvas targets (simulating different canvases)
					const canvasTarget1 = { isCanvasTarget: true, id: 1 };
					const canvasTarget2 = { isCanvasTarget: true, id: 2 };

					// Get depth textures for each reference
					const depthTex1 = node.getTextureForReference( canvasTarget1 );
					const depthTex2 = node.getTextureForReference( canvasTarget2 );

					// CRITICAL: Different references must get different textures
					expect( depthTex1 ).not.toBe( depthTex2 );

					// Both should be DepthTexture instances
					expect( depthTex1 instanceof DepthTexture ).toBeTruthy();
					expect( depthTex2 instanceof DepthTexture ).toBeTruthy();

					// Neither should be the shared buffer (they should be clones)
					expect( depthTex1 ).not.toBe( node.defaultFramebuffer );
					expect( depthTex2 ).not.toBe( node.defaultFramebuffer );

				} );

				test( 'same reference returns same cached depth texture', () => {

					const node = viewportDepthTexture();

					const canvasTarget = { isCanvasTarget: true };

					// Get texture twice for same reference
					const depthTex1 = node.getTextureForReference( canvasTarget );
					const depthTex2 = node.getTextureForReference( canvasTarget );

					expect( depthTex1 ).toBe( depthTex2 );

				} );

				// Test with RenderTargets
				test( 'RenderTargets get independent cached depth textures', () => {

					const node = viewportDepthTexture();

					const renderTarget1 = new RenderTarget( 512, 512 );
					const renderTarget2 = new RenderTarget( 256, 256 );

					const depthTex1 = node.getTextureForReference( renderTarget1 );
					const depthTex2 = node.getTextureForReference( renderTarget2 );

					expect( depthTex1 ).not.toBe( depthTex2 );

					// Clean up
					renderTarget1.dispose();
					renderTarget2.dispose();

				} );

				// Test mixed CanvasTarget and RenderTarget references
				test( 'CanvasTarget and RenderTarget get independent caches', () => {

					const node = viewportDepthTexture();

					const canvasTarget = { isCanvasTarget: true };
					const renderTarget = new RenderTarget( 512, 512 );

					const canvasDepthTex = node.getTextureForReference( canvasTarget );
					const renderDepthTex = node.getTextureForReference( renderTarget );

					expect( canvasDepthTex ).not.toBe( renderDepthTex );

					// Clean up
					renderTarget.dispose();

				} );

			} );

		} );

	} );

} );
