import fs from 'node:fs';
const { default: sharp } = await import( process.env.SHARP_MODULE ?? 'sharp' );
const ROOT=new URL('./',import.meta.url),RAW=new URL('./raw/',ROOT);
const W=640,H=480;
async function image(name,rect){
 const buf=fs.readFileSync(new URL(name+'.f32',RAW));const a=new Float32Array(buf.buffer,buf.byteOffset,buf.byteLength/4),rgb=Buffer.alloc(W*H*3);
 for(let i=0;i<rgb.length;i++){const v=Math.max(0,a[i]);rgb[i]=Math.round(255*Math.pow(v/(1+v),1/2.2));}
 let img=sharp(rgb,{raw:{width:W,height:H,channels:3}});
 if(rect)img=img.extract(rect);
 return img.png().toBuffer();
}
function label(text,w){return Buffer.from(`<svg width="${w}" height="32"><rect width="100%" height="100%" fill="#20252a"/><text x="10" y="22" font-family="sans-serif" font-size="17" fill="white">${text}</text></svg>`);}
for(const effect of ['ssr','vxgi']){
 if(!fs.existsSync(new URL(`${effect}_old_motion_f64.f32`,RAW)))continue;
 await sharp({create:{width:W*2,height:H+32,channels:3,background:'#20252a'}}).composite([
  {input:label('Old weighting — slow orbit, frame 64',W),left:0,top:0},
  {input:label('History ramp-up — same orbit',W),left:W,top:0},
  {input:await image(`${effect}_old_motion_f64`),left:0,top:32},
  {input:await image(`${effect}_new_motion_f64`),left:W,top:32}
 ]).png().toFile(new URL(effect+'-motion-full.png',ROOT).pathname);
 const rect=effect==='ssr'?{left:250,top:180,width:240,height:200}:{left:240,top:200,width:240,height:200};
 const inputs=[];
 for(let col=0;col<4;col++)for(let row=0;row<2;row++){
  const variant=row?'new':'old',f=[32,64,96,128][col];
  inputs.push({input:label(`${variant} · frame ${f}`,240),left:col*240,top:row*232});
  inputs.push({input:await image(`${effect}_${variant}_motion_f${f}`,rect),left:col*240,top:row*232+32});
 }
 await sharp({create:{width:960,height:464,channels:3,background:'#20252a'}}).composite(inputs).png().toFile(new URL(effect+'-motion-crops.png',ROOT).pathname);
 const conv=[];
 for(let col=0;col<4;col++)for(let row=0;row<2;row++){
  const variant=row?'new':'old',f=[1,4,8,16][col];
  conv.push({input:label(`${variant} · frame ${f}`,320),left:col*320,top:row*272});
  conv.push({input:await sharp(await image(`${effect}_${variant}_0_f${f}`)).resize(320,240).png().toBuffer(),left:col*320,top:row*272+32});
 }
 await sharp({create:{width:1280,height:544,channels:3,background:'#20252a'}}).composite(conv).png().toFile(new URL(effect+'-reset.png',ROOT).pathname);
}
