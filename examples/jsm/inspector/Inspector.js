
import { RendererInspector } from './RendererInspector.js';
import { Profiler } from './ui/Profiler.js';
import { Performance } from './tabs/Performance.js';
import { Memory } from './tabs/Memory.js';
import { Console } from './tabs/Console.js';
import { Parameters } from './tabs/Parameters.js';
import { Settings } from './tabs/Settings.js';
import { Viewer } from './tabs/Viewer.js';
import { Timeline } from './tabs/Timeline.js';
import { setText } from './ui/utils.js';
import { getItem, setItem } from './Storage.js';

import { setConsoleFunction, getConsoleFunction, REVISION } from 'three/webgpu';

class Inspector extends RendererInspector {

	constructor( options = {} ) {

		super();

		const {
			nonce = null
		} = options;

		this.nonce = nonce;

		// init profiler

		const profiler = new Profiler( this, options );
		profiler.addEventListener( 'resize', ( e ) => this.dispatchEvent( e ) );
		profiler.addEventListener( 'orientationchange', ( e ) => this.dispatchEvent( e ) );
		profiler.addEventListener( 'layoutchange', ( e ) => this.dispatchEvent( e ) );

		const parameters = new Parameters( {
			builtin: true,
			icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M14 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M4 6l8 0" /><path d="M16 6l4 0" /><path d="M8 12m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M4 12l2 0" /><path d="M10 12l10 0" /><path d="M17 18m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M4 18l11 0" /><path d="M19 18l1 0" /></svg>'
		} );
		parameters.hide();
		profiler.addTab( parameters );

		const viewer = new Viewer();
		viewer.hide();
		profiler.addTab( viewer );

		const performance = new Performance();
		profiler.addTab( performance );

		const memory = new Memory();
		profiler.addTab( memory );

		const timeline = new Timeline();
		profiler.addTab( timeline );

		const consoleTab = new Console();
		profiler.addTab( consoleTab );

		const settings = new Settings();
		profiler.addTab( settings );

		profiler.loadLayout();

		if ( ! profiler.activeTabId ) {

			profiler.setActiveTab( performance.id );

		}

		this.statsData = new Map();
		this.profiler = profiler;
		this.performance = performance;
		this.memory = memory;
		this.console = consoleTab;
		this.parameters = parameters;
		this.viewer = viewer;
		this.timeline = timeline;
		this.settings = settings;
		this.once = {};
		this.extensionsData = new WeakMap();
		this.previousConsoleFunction = null;
		this._domObserver = null;

		this.displayCycle = {
			text: {
				needsUpdate: false,
				duration: .25,
				time: 0
			},
			graph: {
				needsUpdate: false,
				duration: .02,
				time: 0
			},
			toggleGraph: {
				needsUpdate: false,
				duration: .02,
				time: 0
			}
		};

	}

	get domElement() {

		return this.profiler.domElement;

	}

	isVertical() {

		return this.profiler ? this.profiler.isVertical() : false;

	}

	onExtension( name, callback ) {

		const extensionAdded = ( e ) => {

			if ( e.name === name ) {

				callback( e.tab );

				this.settings.removeEventListener( 'extensionadded', extensionAdded );

			}

		};

		if ( this.settings.extensions[ name ] && this.settings.extensions[ name ].loaded ) {

			callback( this.settings.extensions[ name ] );

		} else {

			this.settings.addEventListener( 'extensionadded', extensionAdded );

		}

		return this;

	}

	hide() {

		this.profiler.hide();

	}

	show() {

		this.profiler.show();

	}

	setVisible( value ) {

		this.domElement.style.display = value ? '' : 'none';

		return this;

	}

	getVisible() {

		return this.domElement.style.display !== 'none';

	}

	getSize() {

		return this.profiler.getSize();

	}

	setActiveTab( tab ) {

		this.profiler.setActiveTab( tab.id );

		return this;

	}

	setHorizontalAlign( value ) {

		this.profiler.setHorizontalAlign( value );

		return this;

	}

	setVerticalAlign( value ) {

		this.profiler.setVerticalAlign( value );

		return this;

	}

	addTab( tab ) {

		this.profiler.addTab( tab );

		return this;

	}

	removeTab( tab ) {

		tab.dispose();

		this.profiler.removeTab( tab );

		return this;

	}

	setActiveExtension( name, value ) {

		this.settings.setActiveExtension( name, value );

		return this;

	}

	resolveConsoleOnce( type, message ) {

		const key = type + message;

		if ( this.once[ key ] !== true ) {

			this.resolveConsole( type, message );
			this.once[ key ] = true;

		}

	}

	resolveConsole( type, message, stackTrace = null ) {

		switch ( type ) {

			case 'log':

				this.console.addMessage( 'info', message );

				console.log( message );

				break;

			case 'warn':

				this.console.addMessage( 'warn', message );

				if ( stackTrace && stackTrace.isStackTrace ) {

					console.warn( stackTrace.getError( message ) );

				} else {

					console.warn( message );

				}

				break;

			case 'error':

				this.console.addMessage( 'error', message );

				if ( stackTrace && stackTrace.isStackTrace ) {

					console.error( stackTrace.getError( message ) );

				} else {

					console.error( message );

				}

				break;

		}

	}

	setRenderer( renderer ) {

		super.setRenderer( renderer );

		if ( renderer !== null ) {

			const previousConsoleFunction = getConsoleFunction();

			this.previousConsoleFunction = previousConsoleFunction;

			setConsoleFunction( ( type, message, ...params ) => {

				if ( previousConsoleFunction ) {

					previousConsoleFunction( type, message, ...params );

				}

				this.resolveConsole( type, message, ...params );

			} );

			if ( this.isAvailable ) {

				const init = async () => {

					if ( renderer.hasInitialized() === false ) await renderer.init();

					if ( renderer.hasFeature( 'timestamp-query' ) !== true ) {

						this.console.addMessage( 'error', 'THREE.Inspector: GPU Timestamp Queries not available.' );

					}

					let sign = `THREE.WebGPURenderer: ${ REVISION } [ "`;

					if ( renderer.backend.isWebGPUBackend ) {

						sign += 'WebGPU';

					} else if ( renderer.backend.isWebGLBackend ) {

						sign += 'WebGL2';

					}

					sign += '" ]';

					this.console.addMessage( 'info', sign );

					//

					const domElement = this.domElement;

					if ( domElement.parentElement === null ) {

						if ( renderer.domElement.parentElement !== null ) {

							renderer.domElement.parentElement.appendChild( domElement );

						} else {

							if ( this._domObserver !== null ) {

								this._domObserver.disconnect();

							}

							this._domObserver = new MutationObserver( () => {

								if ( renderer.domElement && renderer.domElement.parentElement !== null ) {

									renderer.domElement.parentElement.appendChild( domElement );

									if ( this._domObserver !== null ) {

										this._domObserver.disconnect();
										this._domObserver = null;

									}

								}

							} );

							this._domObserver.observe( document.body || document.documentElement, { childList: true, subtree: true } );

						}

					}

				};

				init();

				this.timeline.setRenderer( renderer );

			}

		} else {

			if ( this.previousConsoleFunction ) {

				setConsoleFunction( this.previousConsoleFunction );
				this.previousConsoleFunction = undefined;

			}

			if ( this._domObserver !== null ) {

				this._domObserver.disconnect();
				this._domObserver = null;

			}

			this.profiler.dispose();

			this.statsData.clear();

			super.dispose();

		}

		return this;

	}

	createParameters( name ) {

		if ( this.parameters.isVisible === false ) {

			this.parameters.show();

		}

		return this.parameters.createGroup( name );

	}

	getStatsData( cid ) {

		let data = this.statsData.get( cid );

		if ( data === undefined ) {

			data = {};

			this.statsData.set( cid, data );

		}

		return data;

	}

	_accumulateStats( children, averages, parentId = '' ) {

		const occurrences = new Map();

		for ( const stats of children ) {

			// The backend call number shifts when unrelated passes appear or disappear.
			const key = stats.isGroupStats ? `group:${ stats.name }` : stats.cid.replace( /^([rc]):\d+:/, '$1:' );
			const occurrence = occurrences.get( key ) || 0;
			const cid = `${ parentId }/${ key.length }:${ key }/${ occurrence }`;
			occurrences.set( key, occurrence + 1 );

			let row = averages.get( cid );

			if ( row === undefined ) {

				row = {
					cid, name: stats.name, isGroupStats: stats.isGroupStats, isComputeStats: stats.isComputeStats,
					cpu: 0, gpu: 0, cpuOffset: 0, gpuStart: 0, gpuEnd: 0,
					count: 0, gpuUnavailable: 0, timestampCount: 0, children: new Map()
				};
				averages.set( cid, row );

			}

			row.count ++;
			row.cpu += stats.cpu;
			row.cpuOffset += stats.cpuOffset;

			if ( stats.gpuNotAvailable ) row.gpuUnavailable ++;
			else row.gpu += stats.gpu;

			if ( stats.gpuStart !== null ) {

				row.timestampCount ++;
				row.gpuStart += stats.gpuOffset;
				row.gpuEnd += stats.gpuOffset + ( stats.gpuEnd - stats.gpuStart );

			}

			this._accumulateStats( stats.children, row.children, cid );

		}

	}

	_averageFrame() {

		const windowSize = this.performance.frameCount;
		const frames = [];

		for ( let i = this.frames.length - 1; i >= 0 && frames.length < windowSize; i -- ) {

			const sample = this.frames[ i ];
			if ( sample.gpuSpanTime !== undefined && sample.children.length > 0 ) frames.push( sample );

		}

		const rows = new Map();
		let gpu = 0;
		let period = 0;

		for ( const sample of frames.reverse() ) {

			this._accumulateStats( sample.children, rows );
			gpu += sample.gpuSpanTime;
			period += sample.deltaTime;

		}

		const count = frames.length;
		const cpu = Array.from( rows.values() ).reduce( ( sum, row ) => sum + row.cpu, 0 ) / count;

		return { rows, count, cpu, gpu: gpu / count, period: period / count };

	}

	getNodes() {

		return this.currentNodes;

	}

	updateTabs() {

		// tabs

		const tabs = Object.values( this.profiler.tabs );

		for ( const tab of tabs ) {

			let tabData = this.extensionsData.get( tab );

			if ( tabData === undefined ) {

				tab.init( this );

				tabData = {};

				this.extensionsData.set( tab, tabData );

			}

			tab.update( this );

		}

	}

	resolveFrame( frame ) {

		if ( frame.children.length === 0 ) return;

		this.updateCycle( this.displayCycle.text );
		this.updateCycle( this.displayCycle.graph );
		this.updateCycle( this.displayCycle.toggleGraph );

		const average = this.displayCycle.text.needsUpdate || this.displayCycle.graph.needsUpdate ? this._averageFrame() : null;

		if ( this.displayCycle.text.needsUpdate ) {

			setText( this.profiler.toggleButton.querySelector( '.fps-counter' ), this.fps.toFixed() );

			this.performance.updateText( this, average );
			this.memory.updateText( this );

		}

		if ( this.displayCycle.toggleGraph.needsUpdate ) {

			if ( this.profiler.toggleGraph ) {

				this.profiler.toggleGraph.addPoint( 'fps', this.fps );
				this.profiler.toggleGraph.update();

			}

		}

		if ( this.displayCycle.graph.needsUpdate ) {

			this.performance.updateGraph( this, average );
			this.memory.updateGraph( this );

		}

		this.displayCycle.text.needsUpdate = false;
		this.displayCycle.graph.needsUpdate = false;
		this.displayCycle.toggleGraph.needsUpdate = false;

	}

	updateCycle( cycle ) {

		cycle.time += this.nodeFrame.deltaTime;

		if ( cycle.time >= cycle.duration ) {

			cycle.needsUpdate = true;
			cycle.time = 0;

		}

	}

	dispose() {

		super.dispose();

		this.setRenderer( null );

	}

}

export { Inspector, getItem, setItem };
