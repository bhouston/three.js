// The same normalized document written independently in XML and addon JSON.
const xml = `<?xml version="1.0"?>
<materialx version="1.39">
	<nodegraph name="test_graph">
		<multiply name="scaled" type="color3">
			<input name="in1" type="color3" value="0.25, 0.5, 0.75" />
			<input name="in2" type="float" value="2" />
		</multiply>
		<output name="out" type="color3" nodename="scaled" />
	</nodegraph>
</materialx>`;

const json = {
	element: 'materialx',
	attributes: { version: '1.39' },
	children: [ {
		element: 'nodegraph',
		attributes: { name: 'test_graph' },
		children: [ {
			element: 'multiply',
			attributes: { name: 'scaled', type: 'color3' },
			children: [ {
				element: 'input',
				attributes: { name: 'in1', type: 'color3', value: '0.25, 0.5, 0.75' }
			}, {
				element: 'input',
				attributes: { name: 'in2', type: 'float', value: '2' }
			} ]
		}, {
			element: 'output',
			attributes: { name: 'out', type: 'color3', nodename: 'scaled' }
		} ]
	} ]
};

export { xml, json };
