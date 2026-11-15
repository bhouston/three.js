// usage: node montage.mjs  -> montage-3x.png, 1:1 crops of the beauty screenshots: top bilinear | joint-bilateral | joint-bilateral-traa, bottom fsr1 | taau | native
import { createRequire } from 'module';
import fs from 'fs';
const { PNG } = createRequire( new URL( '../../../package.json', import.meta.url ) )( 'pngjs' );
const names = [ 'joint-bilateral', 'joint-bilateral-mip-fused', 'native', 'taau', 'taau-mip-fast-responsive', 'taau-mip-responsive' ];
const w = 420, h = 320, x0 = 900, y0 = 400;
const out = new PNG( { width: w * 3, height: h * 2 } );
names.forEach( ( name, i ) => {

	const png = PNG.sync.read( fs.readFileSync( name === 'native' ? 'native-beauty.png' : `${name}-beauty-3x.png` ) );
	PNG.bitblt( png, out, x0, y0, w, h, ( i % 3 ) * w, Math.floor( i / 3 ) * h );

} );
fs.writeFileSync( 'montage-3x.png', PNG.sync.write( out ) );
