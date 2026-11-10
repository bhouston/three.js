// Regenerate preview figures directly from measured linear floats; enlarge pixels without smoothing.
import fs from 'node:fs';
const {default:sharp}=await import(process.env.SHARP_MODULE??'sharp');
const ROOT=new URL('./',import.meta.url),RAW=new URL('./raw/',ROOT);
const W=640,H=480,SCALE=3,CROP_W=112,CROP_H=80,PW=CROP_W*SCALE,PH=CROP_H*SCALE,GAP=12,LABEL=32;
const WIDTH=PW*2+GAP;
function label(text){return Buffer.from(`<svg width="${PW}" height="${LABEL}"><rect width="100%" height="100%" fill="#20252a"/><text x="10" y="22" font-family="sans-serif" font-size="17" fill="white">${text}</text></svg>`);}
async function image(name,rect){
 const buf=fs.readFileSync(new URL(name+'.f32',RAW)),a=new Float32Array(buf.buffer,buf.byteOffset,buf.byteLength/4),rgb=Buffer.alloc(W*H*3);
 for(let i=0;i<rgb.length;i++){const v=Math.max(0,a[i]);rgb[i]=Math.round(255*Math.pow(v/(1+v),1/2.2));}
 return sharp(rgb,{raw:{width:W,height:H,channels:3}}).extract({...rect,width:CROP_W,height:CROP_H}).resize(PW,PH,{kernel:'nearest'}).png().toBuffer();
}
const definitions={
 ssr:{reset:{left:280,top:300},motion:{left:330,top:215}},
 vxgi:{reset:{left:345,top:200},motion:{left:280,top:265}}
};
for(const effect of ['ssr','vxgi'])for(const kind of ['reset','motion']){
 const frames=kind==='reset'?[1,2,4,8,16]:[32,64,96,128],images=[],labels=[];
 for(let row=0;row<frames.length;row++)for(let col=0;col<2;col++){
  const variant=col?'new':'old',f=frames[row],left=col*(PW+GAP),top=row*(PH+LABEL+GAP);
  const name=kind==='reset'?`${effect}_${variant}_0_f${f}`:`${effect}_${variant}_motion_f${f}`;
  images.push({input:await image(name,definitions[effect][kind]),left,top:top+LABEL});
  labels.push({input:await sharp(label(`${col?'Ramp-up':'Old'} - frame ${f}`)).png().toBuffer(),left,top});
 }
 // Draw all labels last, keeping them above image panels.
 await sharp({create:{width:WIDTH,height:frames.length*(PH+LABEL+GAP)-GAP,channels:3,background:'#20252a'}}).composite([...images,...labels]).png().toFile(new URL(`${effect}-${kind}-zoom.png`,ROOT).pathname);
}
fs.writeFileSync(new URL('crop-regions.json',ROOT),JSON.stringify({source:[W,H],crop:[CROP_W,CROP_H],scale:SCALE,kernel:'nearest',displayWidth:WIDTH,definitions},null,2));
