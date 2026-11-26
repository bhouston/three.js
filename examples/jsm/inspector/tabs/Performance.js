import { Tab } from '../ui/Tab.js';
import { List } from '../ui/List.js';
import { Graph } from '../ui/Graph.js';
import { Item } from '../ui/Item.js';
import { createValueSpan, setText } from '../ui/utils.js';

// creates the cell of a row in the timeline, with a bar for each type

function createTimelineCell( ...types ) {

	const cell = document.createElement( 'div' );
	cell.className = 'frame-timeline';

	for ( const type of types ) {

		const bar = document.createElement( 'div' );
		bar.className = type;
		cell.appendChild( bar );

	}

	return cell;

}

class Performance extends Tab {

	constructor( options = {} ) {

		super( 'Performance', options );

		const perfList = new List( 'Name', 'CPU', 'GPU', '' );
		perfList.setGridStyle( 'minmax(0, 1fr) 60px 60px var(--frame-timeline-width)' ); // the timeline is as wide in the nested rows
		perfList.domElement.style.containerType = 'inline-size';
		perfList.domElement.style.minWidth = '640px';

		const scrollWrapper = document.createElement( 'div' );
		scrollWrapper.className = 'list-scroll-wrapper';
		scrollWrapper.appendChild( perfList.domElement );
		this.content.appendChild( scrollWrapper );

		//

		const graphContainer = document.createElement( 'div' );
		graphContainer.className = 'graph-container';

		const graph = new Graph();
		graph.addLine( 'fps', 'var( --color-fps )' );
		graph.addLine( 'cpu', 'var( --color-yellow )' );
		graph.addLine( 'gpu', 'var( --color-green )' );
		graphContainer.append( graph.domElement );

		//

		this.graphFpsCounter = createValueSpan();
		this.graphMsCounter = createValueSpan();
		this.graphMsCounter.classList.add( 'secondary' );

		const graphCounters = document.createElement( 'div' );
		graphCounters.append( this.graphFpsCounter, this.graphMsCounter );

		const graphStats = new Item( 'Graph Stats', createValueSpan(), createValueSpan(), graphCounters );
		perfList.add( graphStats );

		const graphItem = new Item( graphContainer );
		graphItem.itemRow.childNodes[ 0 ].style.gridColumn = '1 / -1';
		graphItem.itemRow.classList.add( 'no-hover' ); // keep the background of the graph
		graphStats.add( graphItem );

		//

		const frameStats = new Item( 'Frame Stats', createValueSpan(), createValueSpan(), createTimelineCell() );
		frameStats.domElement.classList.add( 'frame-stats' );
		perfList.domElement.appendChild( perfList.domElement.querySelector( '.list-header' ) ); // move the header below the graph
		perfList.add( frameStats );

		const miscellaneous = new Item( 'Miscellaneous & Idle', createValueSpan(), createValueSpan() );
		miscellaneous.domElement.firstChild.style.backgroundColor = '#00ff0b1a';
		miscellaneous.domElement.firstChild.classList.add( 'no-hover' );
		frameStats.add( miscellaneous );

		//

		const timelineRuler = perfList.domElement.querySelector( '.list-header-cell:last-child' );
		timelineRuler.classList.add( 'frame-timeline-ruler' );

		//

		this.perfList = perfList;
		this.timelineRuler = timelineRuler;
		this.frameStats = frameStats;
		this.graphStats = graphStats;
		this.graph = graph;
		this.miscellaneous = miscellaneous;

		//

		this.frameItems = new Map();

	}

	resolveStats( inspector, rows, count, parent ) {

		let index = 0;

		for ( const row of rows.values() ) {

			const data = inspector.getStatsData( row.cid );
			let item = data.item;

			if ( item === undefined ) {

				item = new Item( createValueSpan(), createValueSpan(), createValueSpan(), createTimelineCell( 'cpu', 'gpu' ) );
				data.item = item;

				if ( row.isGroupStats === true ) {

					item.itemRow.classList.add( 'group' );
					item.close();

				}

			}

			if ( item.parent !== parent || parent.children[ index ] !== item ) parent.add( item, index );
			index ++;

			const name = row.name || `Unnamed ${ row.cid }`;
			const gpuAvailable = row.gpuUnavailable < row.count;

			setText( item.data[ 0 ], row.isComputeStats ? `${ name } [ Compute ]` : name );
			setText( item.data[ 1 ], ( row.cpu / count ).toFixed( 2 ) );
			setText( item.data[ 2 ], gpuAvailable ? ( row.gpu / ( count - row.gpuUnavailable ) ).toFixed( 2 ) : '-' );

			this.frameItems.set( row.cid, { item, row } );
			this.resolveStats( inspector, row.children, count, item );

		}

	}

	updateTimeline( frame ) {

		const calls = [];

		// The clocks have separate origins. Shift the GPU lane for display only.
		let gpuDelay = 0;

		for ( const { row, item } of this.frameItems.values() ) {

			const cpuStart = row.cpuOffset / row.count;
			const cpuEnd = cpuStart + row.cpu / row.count;
			const gpuStart = row.timestampCount > 0 ? row.gpuStart / row.timestampCount : null;
			const gpuEnd = row.timestampCount > 0 ? row.gpuEnd / row.timestampCount : null;

			if ( gpuStart !== null ) gpuDelay = Math.max( gpuDelay, cpuStart - gpuStart );

			calls.push( { row, item, cpuStart, cpuEnd, gpuStart, gpuEnd } );

		}

		const duration = calls.reduce( ( end, call ) => Math.max( end, call.cpuEnd, call.gpuEnd !== null ? call.gpuEnd + gpuDelay : 0 ), Math.max( frame.period, 1e-3 ) );
		const toPercent = ( time ) => ( time / duration ) * 100 + '%';

		const setBar = ( bar, start, end ) => {

			bar.style.left = toPercent( start );
			bar.style.width = toPercent( end - start );

		};

		for ( const { row, item, cpuStart, cpuEnd, gpuStart, gpuEnd } of calls ) {

			const [ cpuBar, gpuBar ] = item.data[ 3 ].children;
			setBar( cpuBar, cpuStart, cpuEnd );
			gpuBar.hidden = gpuStart === null;

			if ( gpuStart !== null ) setBar( gpuBar, gpuStart + gpuDelay, gpuEnd + gpuDelay );

			const samples = `${ row.count }/${ frame.count } frames`;
			const gpuSpan = gpuStart === null ? '' : `\nGPU span when timed: ${ gpuStart.toFixed( 3 ) }–${ gpuEnd.toFixed( 3 ) } ms from GPU frame start (${ row.timestampCount }/${ frame.count } frames)`;
			item.data[ 2 ].title = `Average per frame · Present in ${ samples }${ gpuSpan }`;
			gpuBar.title = item.data[ 2 ].title;

		}

		const getBars = ( { cid } ) => Array.from( this.frameItems.get( cid ).item.data[ 3 ].children );

		this.frameStats.data[ 3 ].replaceChildren( ...Array.from( frame.rows.values() ).flatMap( getBars ).map( ( bar ) => bar.cloneNode() ) );

		// the labels of the ruler are at a multiple of 1, 2 or 5, at least 80 pixels apart

		const count = Math.max( this.timelineRuler.clientWidth / 80, 1 );
		const magnitude = 10 ** Math.floor( Math.log10( duration / count ) );
		const step = magnitude * [ 1, 2, 5, 10 ].find( ( multiple ) => duration / ( magnitude * multiple ) <= count );

		const ticks = [];

		for ( let i = 0; i * step < duration; i ++ ) {

			const tick = document.createElement( 'span' );
			tick.textContent = parseFloat( ( i * step ).toFixed( 3 ) ) + ( i === 0 ? ' ms' : '' );
			tick.style.left = toPercent( i * step );
			ticks.push( tick );

		}

		this.timelineRuler.replaceChildren( ...ticks );
		this.perfList.domElement.style.setProperty( '--frame-timeline-step', toPercent( step ) );

	}

	updateGraph( inspector, frame ) {

		const fps = inspector.fps;

		this.graph.addPoint( 'fps', fps );

		if ( frame ) {

			const cpuValue = Math.min( ( ( frame.cpu || 0 ) * fps ) / 1000, 1.0 ) * fps;
			const gpuValue = Math.min( ( ( frame.gpu || 0 ) * fps ) / 1000, 1.0 ) * fps;

			this.graph.addPoint( 'cpu', cpuValue );
			this.graph.addPoint( 'gpu', gpuValue );

		}

		this.graph.update();

	}

	updateText( inspector, frame ) {

		const oldFrameItems = new Map( this.frameItems );
		this.frameItems.clear();

		this.resolveStats( inspector, frame.rows, frame.count, this.frameStats );

		for ( const [ cid, { item } ] of oldFrameItems ) {

			if ( ! this.frameItems.has( cid ) ) item.parent.remove( item );

		}

		this.updateTimeline( frame );

		const fps = inspector.fps;

		setText( this.graphFpsCounter, fps.toFixed() + ' FPS' );
		setText( this.graphMsCounter, ( fps > 0 ? 1000 / fps : 0 ).toFixed( 2 ) + ' ms' );

		//

		setText( this.frameStats.data[ 1 ], frame.cpu.toFixed( 2 ) );
		setText( this.frameStats.data[ 2 ], inspector.getRenderer().backend.hasTimestamp ? frame.gpu.toFixed( 2 ) : '-' );

		//

		setText( this.miscellaneous.data[ 1 ], Math.max( 0, frame.period - frame.cpu - frame.gpu ).toFixed( 2 ) );
		setText( this.miscellaneous.data[ 2 ], '-' );

	}

}

export { Performance };
