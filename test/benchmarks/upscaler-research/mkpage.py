from pathlib import Path
import sys
root=Path(sys.argv[1])
s=(root/'examples/webgpu_upscaling_joint_bilateral.html').read_text()
s=s.replace('\t\t\tinit();', "\t\t\tObject.assign( params, JSON.parse( decodeURIComponent( location.hash.slice( 1 ) ) || '{}' ) );\n\t\t\tinit();")
s=s.replace('renderer.setAnimationLoop( render );', "window.__test = { renderer, render, params, frames: 0, completed: 0, counting: false, stopAt: params.motionTest ? 120 : ( params.rotate ? 0 : 320 ) };\n\t\t\t\trenderer.setAnimationLoop( render );")
s=s.replace('renderPipeline.render();', "renderPipeline.render();\n\t\t\t\tif ( window.__test ) {\n\t\t\t\t\tconst id = ++ window.__test.frames;\n\t\t\t\t\tif ( window.__test.counting && id % 32 === 0 ) window.__device.queue.onSubmittedWorkDone().then( () => { window.__test.completed = Math.max( window.__test.completed, id ); } );\n\t\t\t\t}")
s=s.replace('if ( params.rotate ) model.rotation.y += timer.getDelta() * 0.2;', 'if ( params.motionTest ) model.rotation.y = window.__test.frames * 0.2 / 60; else if ( params.rotate ) model.rotation.y += timer.getDelta() * 0.2;')
s=s.replace('const id = ++ window.__test.frames;', 'const id = ++ window.__test.frames; if ( window.__test.stopAt && id >= window.__test.stopAt ) renderer.setAnimationLoop( null );')
(root/'examples/_cmp.html').write_text(s)
