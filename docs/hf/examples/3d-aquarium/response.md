Here's a complete, self-contained single-file build. Everything (fish, glass, water, plants, stones, bubbles, caustics) is procedural — no external assets.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Clownfish Aquarium — Interactive 3D</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root{
    --bg-0:#0a1628; --bg-1:#1a3a5c;
    --ink:#eaf4ff; --muted:#8fb0d0;
    --accent:#ffb454; --accent2:#4fc3f7;
    --panel:rgba(9,20,34,0.62);
    --line:rgba(120,180,220,0.16);
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; overflow:hidden; color:var(--ink);
    font-family:"Space Grotesk",system-ui,sans-serif;
    background:radial-gradient(130% 120% at 50% 38%, var(--bg-1) 0%, var(--bg-0) 62%);
  }
  canvas{display:block; position:fixed; inset:0; z-index:0}

  /* ambient caustic + vignette layers over the scene */
  .caustics{
    position:fixed; inset:-12%; z-index:1; pointer-events:none;
    mix-blend-mode:screen; opacity:.10;
    background:
      radial-gradient(38% 46% at 22% 18%, rgba(120,220,255,.9), transparent 60%),
      radial-gradient(30% 40% at 78% 26%, rgba(120,220,255,.7), transparent 60%),
      radial-gradient(44% 50% at 60% 78%, rgba(90,180,255,.6), transparent 60%),
      radial-gradient(30% 40% at 12% 70%, rgba(120,220,255,.5), transparent 60%);
    filter:blur(26px);
    animation:drift 22s ease-in-out infinite;
  }
  .caustics.two{
    opacity:.07; animation:drift2 30s ease-in-out infinite;
    background:
      radial-gradient(34% 44% at 68% 20%, rgba(160,240,255,.8), transparent 60%),
      radial-gradient(30% 40% at 30% 60%, rgba(120,220,255,.6), transparent 60%),
      radial-gradient(40% 46% at 82% 74%, rgba(90,180,255,.5), transparent 60%);
  }
  @keyframes drift{0%{transform:translate(0,0) scale(1)}50%{transform:translate(3%,-2%) scale(1.08)}100%{transform:translate(0,0) scale(1)}}
  @keyframes drift2{0%{transform:translate(0,0) scale(1.05)}50%{transform:translate(-4%,3%) scale(1)}100%{transform:translate(0,0) scale(1.05)}}
  .vignette{
    position:fixed; inset:0; z-index:2; pointer-events:none;
    background:radial-gradient(120% 120% at 50% 45%, transparent 55%, rgba(2,8,16,.55) 100%);
  }

  /* UI */
  .ui{position:fixed; z-index:10; inset:0; pointer-events:none}
  .ui > *{pointer-events:auto}

  .brand{
    position:absolute; top:22px; left:24px; max-width:440px;
    animation:rise .9s cubic-bezier(.2,.7,.2,1) both;
  }
  .brand h1{
    margin:0; font-weight:700; letter-spacing:-.02em; line-height:1.02;
    font-size:clamp(26px,4.4vw,46px);
    text-shadow:0 2px 24px rgba(0,0,0,.55);
  }
  .brand h1 .a{color:var(--accent)}
  .brand h1 .b{color:var(--accent2)}
  .brand .sub{
    margin:8px 0 0; font-size:13px; color:var(--muted);
    letter-spacing:.04em; text-shadow:0 1px 12px rgba(0,0,0,.5);
  }
  .brand .latin{
    display:inline-block; margin-top:10px; font-family:"JetBrains Mono",monospace;
    font-size:11px; color:var(--accent2); opacity:.85;
    border:1px solid var(--line); border-radius:999px; padding:3px 10px;
    background:rgba(9,20,34,.4); backdrop-filter:blur(6px);
  }

  .panel{
    position:absolute; top:22px; right:24px; width:236px;
    background:var(--panel); border:1px solid var(--line); border-radius:14px;
    padding:14px 15px 12px; backdrop-filter:blur(10px);
    box-shadow:0 18px 40px rgba(0,0,0,.4);
    animation:rise .9s .12s cubic-bezier(.2,.7,.2,1) both;
  }
  .panel h2{
    margin:0 0 10px; font-size:11px; letter-spacing:.16em; text-transform:uppercase;
    color:var(--muted); font-weight:500;
  }
  .row{display:flex; align-items:center; gap:10px; margin:9px 0}
  .row label{font-size:12px; color:var(--ink); flex:0 0 62px}
  .row .val{font-family:"JetBrains Mono",monospace; font-size:11px; color:var(--accent2); flex:0 0 34px; text-align:right}
  input[type=range]{
    -webkit-appearance:none; appearance:none; flex:1; height:4px; border-radius:4px;
    background:linear-gradient(90deg,var(--accent2),rgba(79,195,247,.25)); outline:none; cursor:pointer;
  }
  input[type=range]::-webkit-slider-thumb{
    -webkit-appearance:none; width:14px; height:14px; border-radius:50%;
    background:var(--accent); box-shadow:0 0 0 3px rgba(255,180,84,.25),0 2px 6px rgba(0,0,0,.5);
  }
  .btn{
    margin-top:6px; width:100%; cursor:pointer; font-family:inherit; font-size:12px;
    color:var(--ink); background:rgba(255,180,84,.12); border:1px solid rgba(255,180,84,.4);
    border-radius:9px; padding:8px 10px; transition:.18s; letter-spacing:.04em;
  }
  .btn:hover{background:rgba(255,180,84,.22); transform:translateY(-1px)}
  .btn:active{transform:translateY(0)}

  .legend{
    position:absolute; bottom:20px; left:24px; display:flex; gap:16px; flex-wrap:wrap;
    font-size:12px; color:var(--muted);
    animation:rise .9s .2s cubic-bezier(.2,.7,.2,1) both;
  }
  .legend b{color:var(--ink); font-weight:500}
  .legend .k{
    display:inline-block; min-width:20px; text-align:center; margin-right:6px;
    font-family:"JetBrains Mono",monospace; font-size:11px; color:var(--accent);
    border:1px solid var(--line); border-radius:6px; padding:2px 5px; background:rgba(9,20,34,.4);
  }

  .hint{
    position:absolute; bottom:20px; right:24px; font-size:11px; color:var(--muted);
    opacity:.7; font-family:"JetBrains Mono",monospace;
  }

  @keyframes rise{from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:translateY(0)}}

  @media (max-width:640px){
    .panel{width:200px; top:auto; bottom:74px; right:16px}
    .brand{top:16px; left:16px}
    .legend{bottom:14px; left:16px; gap:10px; font-size:11px}
    .hint{display:none}
  }
</style>
</head>
<body>
  <div class="caustics"></div>
  <div class="caustics two"></div>
  <div class="vignette"></div>

  <div class="ui">
    <div class="brand">
      <h1><span class="a">Clownfish</span> <span class="b">Aquarium</span></h1>
      <p class="sub">A living, procedural reef — rendered in real time with Three.js.</p>
      <span class="latin">Amphiprion ocellaris · 3 specimens</span>
    </div>

    <div class="panel">
      <h2>Controls</h2>
      <div class="row">
        <label>Current</label>
        <input id="flow" type="range" min="0" max="1" step="0.01" value="0.55" />
        <span class="val" id="flowV">55%</span>
      </div>
      <div class="row">
        <label>Light</label>
        <input id="light" type="range" min="0.4" max="1.8" step="0.01" value="1.0" />
        <span class="val" id="lightV">100%</span>
      </div>
      <button class="btn" id="reset">�� Reset view</button>
    </div>

    <div class="legend">
        <span><b>Orbit</b><span class="k">drag</span></span>
        <span><b>Zoom</b><span class="k">scroll</span></span>
        <span><b>Pan</b><span class="k">right-drag</span></span>
    </div>
    <div class="hint">drag to look around · scroll to dive in</div>
  </div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ------------------------------------------------------------------ *
 *  Renderer / Scene / Camera
 * ------------------------------------------------------------------ */
const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a1628, 0.05);

const camera = new THREE.PerspectiveCamera(46, innerWidth/innerHeight, 0.1, 100);
camera.position.set(6.4, 3.4, 7.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 1.5, 0);
controls.minDistance = 3.2;
controls.maxDistance = 16;
controls.maxPolarAngle = 2.55;
controls.minPolarAngle = 0.35;
controls.update();

/* ------------------------------------------------------------------ *
 *  Lighting
 * ------------------------------------------------------------------ */
const ambient = new THREE.AmbientLight(0xb3e5fc, 0.4);
scene.add(ambient);

const dir = new THREE.DirectionalLight(0xfff8e1, 1.0);
dir.position.set(4, 9, 5);
dir.castShadow = true;
dir.shadow.mapSize.set(2048, 2048);
dir.shadow.camera.near = 1;
dir.shadow.camera.far = 30;
dir.shadow.camera.left = -6; dir.shadow.camera.right = 6;
dir.shadow.camera.top = 6;  dir.shadow.camera.bottom = -6;
dir.shadow.bias = -0.0004;
dir.shadow.radius = 5;
scene.add(dir);

const lamp = new THREE.PointLight(0xffd9a0, 0.9, 9, 2);
lamp.position.set(0, 2.7, 0);
scene.add(lamp);

const fill = new THREE.PointLight(0x4fc3f7, 0.35, 12, 2);
fill.position.set(-4, 1.2, -3);
scene.add(fill);

/* ------------------------------------------------------------------ *
 *  Aquarium geometry constants
 * ------------------------------------------------------------------ */
const W = 6, D = 3, H = 3;
const t = 0.08;                 // glass thickness
const HW = W/2, HD = D/2;
const waterTop = H - 0.28;
const waterBottom = 0.06;
const MARGIN = 0.32;            // fish keep-clear from walls
const BOUND = {
  minX:-HW+MARGIN, maxX:HW-MARGIN,
  minZ:-HD+MARGIN, maxZ:HD-MARGIN,
  minY:0.55, maxY:waterTop-0.35
};

/* ------------------------------------------------------------------ *
 *  Glass material (5 panels) + frame
 * ------------------------------------------------------------------ */
const glassMat = new THREE.MeshPhysicalMaterial({
  color:0xe0f7fa, metalness:0, roughness:0.05,
  transmission:0.95, thickness:0.15, ior:1.5,
  transparent:true, opacity:0.18,
  envMapIntensity:1.1, clearcoat:0.4, clearcoatRoughness:0.15,
  side:THREE.DoubleSide, depthWrite:false
});
const panels = [
  { w:W, h:H, pos:[0, H/2, -HD], rot:[0,0,0] },
  { w:W, h:H, pos:[0, H/2,  HD], rot:[0,0,0] },
  { w:D, h:H, pos:[-HW, H/2, 0], rot:[0, Math.PI/2, 0] },
  { w:D, h:H, pos:[ HW, H/2, 0], rot:[0, Math.PI/2, 0] },
  { w:W, h:D, pos:[0, 0, 0], rot:[Math.PI/2,0,0] }
];
panels.forEach(p=>{
  const g = new THREE.Mesh(new THREE.PlaneGeometry(p.w, p.h), glassMat);
  g.position.set(...p.pos); g.rotation.set(...p.rot);
  scene.add(g);
});

// matte-black rim / frame around the open top
const frameMat = new THREE.MeshStandardMaterial({ color:0x14181c, roughness:0.7, metalness:0.25 });
const frame = new THREE.Group();
const rimW = 0.14, rimH = 0.16, rimY = H + 0.02;
[[0, -HD, W, 0],[0, HD, W, 0],[-HW, 0, D, Math.PI/2],[HW, 0, D, Math.PI/2]].forEach(([x,z,len,ry])=>{
  const bar = new THREE.Mesh(new THREE.BoxGeometry(len, rimH, rimW), frameMat);
  bar.position.set(x, rimY, z); bar.rotation.y = ry;
  bar.castShadow = true;
  frame.add(bar);
});
scene.add(frame);

/* ------------------------------------------------------------------ *
 *  Water volume + animated surface
 * ------------------------------------------------------------------ */
const waterMat = new THREE.MeshPhysicalMaterial({
  color:0x1e88e5, transparent:true, opacity:0.16,
  roughness:0.12, transmission:0.6, thickness:0.6,
  ior:1.33, depthWrite:false, side:THREE.DoubleSide
});
const water = new THREE.Mesh(new THREE.BoxGeometry(W-0.16, waterTop-waterBottom, D-0.16), waterMat);
water.position.set(0, (waterTop+waterBottom)/2, 0);
scene.add(water);

const surfaceMat = new THREE.MeshPhysicalMaterial({
  color:0x8fd8ff, transparent:true, opacity:0.32,
  roughness:0.08, transmission:0.7, thickness:0.2,
  ior:1.33, depthWrite:false, side:THREE.DoubleSide, metalness:0
});
const surface = new THREE.Mesh(new THREE.PlaneGeometry(W-0.16, D-0.16, 40, 24), surfaceMat);
surface.rotation.x = -Math.PI/2;
surface.position.set(0, waterTop, 0);
scene.add(surface);
const surfaceBase = surface.geometry.attributes.position.array.slice();

/* ------------------------------------------------------------------ *
 *  Gravel floor + stones
 * ------------------------------------------------------------------ */
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(W, D),
  new THREE.MeshStandardMaterial({ color:0x2b2118, roughness:0.95 })
);
floor.rotation.x = -Math.PI/2;
floor.receiveShadow = true;
scene.add(floor);

const stoneColors = [0x8D6E63,0xA1887F,0x9E9E9E,0x6D4C41,0x757575,0x5D4037];
const stoneGeo = new THREE.DodecahedronGeometry(1, 1);
for (let i=0;i<22;i++){
  const s = 0.06 + Math.random()*0.16;
  const m = new THREE.Mesh(stoneGeo, new THREE.MeshStandardMaterial({
    color: stoneColors[(Math.random()*stoneColors.length)|0],
    roughness:0.9, metalness:0.05
  }));
  m.scale.set(s, s*(0.55+Math.random()*0.35), s);
  m.position.set(
    (Math.random()-0.5)*(W-0.5),
    s*0.35,
    (Math.random()-0.5)*(D-0.5)
  );
  m.rotation.set(Math.random()*6, Math.random()*6, Math.random()*6);
  m.castShadow = true; m.receiveShadow = true;
  scene.add(m);
}

/* ------------------------------------------------------------------ *
 *  Aquatic plants (shader-swayed blades)
 * ------------------------------------------------------------------ */
const plantGroup = new THREE.Group();
const plantBlades = [];
const plantMat = new THREE.ShaderMaterial({
  transparent:true, side:THREE.DoubleSide, depthWrite:false,
  uniforms:{ uTime:{value:0}, uFlow:{value:0.55}, uColor:{value:new THREE.Color(0x2E7D32)} },
  vertexShader:`
    uniform float uTime; uniform float uFlow;
    varying vec2 vUv; varying float vH;
    void main(){
      vUv = uv; vH = uv.y;
      vec3 p = position;
      float sway = sin(uTime*1.1 + uv.x*3.0 + p.x*2.0) * 0.10 * vH * vH;
      float sway2 = sin(uTime*0.7 + p.z*3.0) * 0.06 * vH;
      p.x += sway * (0.4 + uFlow);
      p.z += sway2 * (0.4 + uFlow);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(p,1.0);
    }`,
  fragmentShader:`
    uniform vec3 uColor; varying vec2 vUv;
    void main(){
      vec3 c = mix(uColor*0.55, uColor, vUv.y);
      c = mix(c, vec3(0.55,0.85,0.45), vUv.y*vUv.y*0.35);
      gl_alpha = smoothstep(1.0,0.85,vUv.y);
      gl_alpha *= smoothstep(0.0,0.08,vUv.x)*smoothstep(1.0,0.92,vUv.x);
      gl_alpha *= 0.92;
      gl_FragColor = vec4(c, gl_alpha);
    }`.replace('gl_alpha','float a') // (kept readable; replaced below)
});
// fix: the replace above is a no-op safety; rebuild cleanly:
plantMat.fragmentShader = `
  uniform vec3 uColor; varying vec2 vUv;
  void main(){
    vec3 c = mix(uColor*0.55, uColor, vUv.y);
    c = mix(c, vec3(0.55,0.85,0.45), vUv.y*vUv.y*0.35);
    float a = smoothstep(1.0,0.85,vUv.y);
    a *= smoothstep(0.0,0.08,vUv.x)*smoothstep(1.0,0.92,vUv.x);
    a *= 0.92;
    gl_FragColor = vec4(c, a);
  }`;
plantMat.vertexShader = `
  uniform float uTime; uniform float uFlow;
  varying vec2 vUv;
  void main(){
    vUv = uv;
    vec3 p = position;
    float vH = uv.y;
    float sway  = sin(uTime*1.1 + uv.x*3.0 + p.x*2.0) * 0.10 * vH * vH;
    float sway2 = sin(uTime*0.7 + p.z*3.0) * 0.06 * vH;
    p.x += sway  * (0.4 + uFlow);
    p.z += sway2 * (0.4 + uFlow);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p,1.0);
  }`;
plantMat.needsUpdate = true;

function makePlant(cx, cz, scale){
  const g = new THREE.Group();
  g.position.set(cx, 0, cz);
  const n = 5 + (Math.random()*3|0);
  for (let i=0;i<n;i++){
    const h = (0.7 + Math.random()*0.9) * scale;
    const w = (0.09 + Math.random()*0.05) * scale;
    const blade = new THREE.Mesh(new THREE.PlaneGeometry(w, h, 1, 6), plantMat);
    blade.position.set((Math.random()-0.5)*0.22*scale, h/2, (Math.random()-0.5)*0.22*scale);
    blade.rotation.y = Math.random()*Math.PI;
    blade.rotation.z = (Math.random()-0.5)*0.25;
    g.add(blade);
  }
  plantGroup.add(g);
}
makePlant(-2.4, -1.0, 1.15);
makePlant( 2.5, -1.1, 1.0);
makePlant( 2.2,  1.0, 0.85);
makePlant(-2.6,  0.9, 0.95);
makePlant( 0.0, -1.25, 0.7);
scene.add(plantGroup);

/* ------------------------------------------------------------------ *
 *  Bubbles
 * ------------------------------------------------------------------ */
const bubbleMat = new THREE.MeshPhysicalMaterial({
  color:0xffffff, transparent:true, opacity:0.55,
  roughness:0.05, transmission:0.9, thickness:0.05,
  ior:1.1, metalness:0, depthWrite:false,
  iridescence:0.6, iridescenceIOR:1.3
});
const bubbleGeo = new THREE.SphereGeometry(1, 10, 10);
const streams = [
  { x:-2.0, z:-0.6, r:0.05 },
  { x: 2.1, z: 0.7, r:0.045 }
];
const bubbles = [];
function spawnBubble(st){
  const b = new THREE.Mesh(bubbleGeo, bubbleMat);
  const r = st.r * (0.6 + Math.random()*0.7);
  b.scale.setScalar(r);
  b.position.set(st.x + (Math.random()-0.5)*0.15, 0.12, st.z + (Math.random()-0.5)*0.15);
  b.userData = { r, baseX:b.position.x, baseZ:b.position.z,
    wob:Math.random()*6.28, wobA:0.03+Math.random()*0.05,
    rise:0.55+Math.random()*0.35, seed:Math.random()*100 };
  scene.add(b);
  bubbles.push(b);
}
streams.forEach(st=>{ st.acc = 0; st.rate = 2.5 + Math.random(); });

/* ------------------------------------------------------------------ *
 *  Clownfish
 * ------------------------------------------------------------------ */
const fishMat = {
  orange: new THREE.MeshStandardMaterial({ color:0xFF6D00, roughness:0.55, metalness:0.05 }),
  belly:  new THREE.MeshStandardMaterial({ color:0xFF9E5A, roughness:0.6 }),
  white:  new THREE.MeshStandardMaterial({ color:0xf7f7f2, roughness:0.5 }),
  black:  new THREE.MeshStandardMaterial({ color:0x0a0a0a, roughness:0.4 }),
  fin:   new THREE.MeshStandardMaterial({ color:0xFF8A3D, roughness:0.4, metalness:0.1,
            side:THREE.DoubleSide, transparent:true, opacity:0.92 })
};

function makeFish(scale){
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);

  // --- body silhouette (orange) ---
  const bodyGeo = new THREE.SphereGeometry(0.5, 40, 28);
  bodyGeo.scale(0.55, 0.62, 1.6);
  const bodyMesh = new THREE.Mesh(bodyGeo, fishMat.orange);
  bodyMesh.castShadow = true;
  body.add(bodyMesh);

  // belly (lighter, underside)
  const bellyGeo = new THREE.SphereGeometry(0.5, 32, 20, 0, Math.PI*2, Math.PI*0.55, Math.PI*0.45);
  bellyGeo.scale(0.55, 0.62, 1.6);
  const belly = new THREE.Mesh(bellyGeo, fishMat.belly);
  belly.position.y = -0.02;
  body.add(belly);

  // --- three vertical stripes (white + black border) ---
  const stripeDefs = [
    { z:0.62, s:0.30, r:0.02 },   // head band
    { z:0.02, s:0.34, r:0.02 },   // mid band
    { z:-0.52, s:0.26, r:0.02 }   // rear band
  ];
  stripeDefs.forEach(sd=>{
    const g = new THREE.Group();
    g.position.z = sd.z;
    g.rotation.z = 0.06;
    const w = new THREE.Mesh(new THREE.BoxGeometry(0.60, 0.60, 0.05), fishMat.white);
    w.scale.set(1, 1, 1);
    w.position.y = 0.02;
    const b1 = new THREE.Mesh(new THREE.BoxGeometry(0.60, 0.60, 0.05), fishMat.black);
    b1.position.z = -0.006; b1.scale.set(1.06, 1.06, 1);
    g.add(b1, w);
    g.scale.set(sd.s, sd.s, 1);
    body.add(g);
  });

  // --- tail (caudal) ---
  const tailPivot = new THREE.Group();
  tailPivot.position.z = -0.82;
  body.add(tailPivot);
  const tailShape = new THREE.Shape();
  tailShape.moveTo(0,0);
  tailShape.quadraticCurveTo(0.18,0.16,0.34,0.30);
  tailShape.quadraticCurveTo(0.24,0.10,0.20,0.0);
  tailShape.quadraticCurveTo(0.24,-0.10,0.34,-0.30);
  tailShape.quadraticCurveTo(0.18,-0.16,0,0);
  const tailGeo = new THREE.ShapeGeometry(tailShape, 12);
  tailGeo.scale(1.15,1.15,1);
  const tail = new THREE.Mesh(tailGeo, fishMat.fin);
  tail.rotation.y = Math.PI/2;
  tail.position.z = -0.02;
  tailPivot.add(tail);

  // --- dorsal fin (sail) ---
  const dorsalShape = new THREE.Shape();
  dorsalShape.moveTo(-0.55,0);
  dorsalShape.quadraticCurveTo(-0.1,0.34,0.35,0.02);
  dorsalShape.lineTo(0.35,0);
  dorsalShape.quadraticCurveTo(0.0,0.06,-0.55,0);
  const dorsal = new THREE.Mesh(new THREE.ShapeGeometry(dorsalShape,12), fishMat.fin);
  dorsal.rotation.x = -Math.PI/2;
  dorsal.position.set(0,0.30,0.05);
  body.add(dorsal);

  // --- anal fin (underside near tail) ---
  const analShape = new THREE.Shape();
  analShape.moveTo(0,0);
  analShape.lineTo(-0.34,0.0);
  analShape.quadraticCurveTo(-0.28,-0.16,-0.02,-0.14);
  analShape.quadraticCurveTo(0.02,-0.06,0,0);
  const anal = new THREE.Mesh(new THREE.ShapeGeometry(analShape,10), fishMat.fin);
  anal.rotation.x = Math.PI/2;
  anal.position.set(0,-0.30,-0.55);
  body.add(anal);

  // --- pectoral fins (×2) ---
  const pecShape = new THREE.Shape();
  pecShape.moveTo(0,0);
  pecShape.quadraticCurveTo(0.16,0.05,0.22,-0.10);
  pecShape.quadraticCurveTo(0.10,-0.02,0,0);
  const pecGeo = new THREE.ShapeGeometry(pecShape,10);
  const pecL = new THREE.Mesh(pecGeo, fishMat.fin);
  pecL.position.set(0.24,-0.02,0.34);
  pecL.rotation.set(0.5,0.9,0.2);
  const pecR = pecL.clone();
  pecR.position.x = -0.24;
  pecR.rotation.set(0.5,-0.9,-0.2);
  body.add(pecL, pecR);

  // --- eyes (×2) ---
  const eyeGeo = new THREE.SphereGeometry(0.075, 16, 16);
  const eyeMat = new THREE.MeshStandardMaterial({ color:0xffffff, roughness:0.25 });
  const pupilMat = new THREE.MeshStandardMaterial({ color:0x050505, roughness:0.2 });
  const hiMat = new THREE.MeshStandardMaterial({ color:0xffffff, emissive:0xffffff, emissiveIntensity:0.6, roughness:0.1 });
  const eyes = [];
  [1,-1].forEach(side=>{
    const e = new THREE.Group();
    e.position.set(side*0.20, 0.10, 0.66);
    const sclera = new THREE.Mesh(eyeGeo, eyeMat);
    sclera.scale.set(1,1,0.7);
    const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.04,12,12), pupilMat);
    pupil.position.set(side*0.03,0,0.05);
    const hi = new THREE.Mesh(new THREE.SphereGeometry(0.016,8,8), hiMat);
    hi.position.set(side*0.05,0.03,0.07);
    e.add(sclera, pupil, hi);
    body.add(e);
    eyes.push(e);
  });

  // --- mouth ---
  const mouth = new THREE.Mesh(new THREE.SphereGeometry(0.05,12,12), fishMat.black);
  mouth.scale.set(1,0.5,0.4);
  mouth.position.set(0,-0.02,0.80);
  body.add(mouth);

  root.scale.setScalar(scale);
  return { root, body, tailPivot, pecL, pecR, dorsal, anal, eyes,
    _wag:0, _flap:0, _dsway:0 };
}

const fishList = [];
[0.9, 1.0, 1.1].forEach((s,i)=>{
  const f = makeFish(s);
  f.speed = 0.006 + Math.random()*0.006;
  f.dir = new THREE.Vector3(Math.random()-0.5,0,Math.random()-0.5).normalize();
  f.target = new THREE.Vector3(
    (Math.random()-0.5)*2, 1.2+Math.random()*0.6, (Math.random()-0.5)*1.2
  );
  f.timer = 2 + Math.random()*3;
  f.bank = 0;
  f._wag = Math.random()*6;
  f._flap = Math.random()*6;
  f._dsway = Math.random()*6;
  f.root.position.set((i-1)*1.4, 1.2, (Math.random()-0.5)*0.6);
  scene.add(f.root);
  fishList.push(f);
});

/* ------------------------------------------------------------------ *
 *  Caustic light on the floor (animated shader)
 * ------------------------------------------------------------------ */
const causticMat = new THREE.ShaderMaterial({
  transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
  uniforms:{ uTime:{value:0}, uColor:{value:new THREE.Color(0x66ccff)} },
  vertexShader:`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
  fragmentShader:`
    uniform float uTime; uniform vec3 uColor; varying vec2 vUv;
    void main(){
      vec2 p = vUv*8.0;
      float t = uTime*0.35;
      float c = 0.0;
      c += sin(p.x + t)*0.5 + sin(p.y*1.3 - t*1.1)*0.5;
      c += sin((p.x+p.y)*1.7 + t*0.7)*0.4;
      c = c*0.5 + 0.5;
      c = pow(c, 3.0);
      float edge = smoothstep(0.5,0.15, distance(vUv, vec2(0.5)));
      gl_FragColor = vec4(uColor, c*0.35*edge);
    }`
});
const caustic = new THREE.Mesh(new THREE.PlaneGeometry(W-0.2, D-0.2), causticMat);
caustic.rotation.x = -Math.PI/2;
caustic.position.y = 0.02;
scene.add(caustic);

/* ------------------------------------------------------------------ *
 *  UI wiring
 * ------------------------------------------------------------------ */
let flow = 0.55, lightMul = 1.0;
const flowEl = document.getElementById('flow');
const lightEl = document.getElementById('light');
const flowV = document.getElementById('flowV');
const lightV = document.getElementById('lightV');
flowEl.addEventListener('input', e=>{
  flow = parseFloat(e.target.value);
  flowV.textContent = Math.round(flow*100)+'%';
  plantMat.uniforms.uFlow.value = flow;
});
lightEl.addEventListener('input', e=>{
  lightMul = parseFloat(e.target.value);
  lightV.textContent = Math.round(lightMul*100)+'%';
  dir.intensity = 1.0*lightMul;
  lamp.intensity = 0.9*lightMul;
  ambient.intensity = 0.4*lightMul;
});
document.getElementById('reset').addEventListener('click', ()=>{
  camera.position.set(6.4,3.4,7.6);
  controls.target.set(0,1.5,0);
  controls.update();
});

/* ------------------------------------------------------------------ *
 *  Animation loop
 * ------------------------------------------------------------------ */
const clock = new THREE.Clock();
const tmpQ = new THREE.Quaternion();
const tmpM = new THREE.Matrix4();
const UP = new THREE.Vector3(0,1,0);
const _fwd = new THREE.Vector3();

function animate(){
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const time = clock.elapsedTime;

  // --- fish ---
  for (const f of fishList){
    f.timer -= dt;
    if (f.timer <= 0){
      f.target.set(
        (Math.random()-0.5)*(W-1.2),
        BOUND.minY + Math.random()*(BOUND.maxY-BOUND.minY),
        (Math.random()-0.5)*(D-1.0)
      );
      f.timer = 3 + Math.random()*4;
    }
    _fwd.copy(f.target).sub(f.root.position);
    if (_fwd.lengthSq() > 1e-6){
      _fwd.normalize();
      f.dir.lerp(_fwd, 0.04).normalize();
    }
    f.root.position.addScaledVector(f.dir, f.speed);

    // soft wall containment
    f.root.position.x = THREE.MathUtils.clamp(f.root.position.x, BOUND.minX, BOUND.maxX);
    f.root.position.z = THREE.MathUtils.clamp(f.root.position.z, BOUND.minZ, BOUND.maxZ);
    f.root.position.y = THREE.MathUtils.clamp(f.root.position.y, BOUND.minY, BOUND.maxY);

    // orient + banking
    tmpM.lookAt(f.root.position, f.root.position.clone().add(f.dir), UP);
    tmpQ.setFromRotationMatrix(tmpM);
    f.root.quaternion.slerp(tmpQ, 0.08);
    const cross = f.dir.z * 1.0; // lateral component for bank
    f.bank += ((-cross*0.35) - f.bank) * 0.05;
    f.root.rotateZ(f.bank);

    // fin / tail animation (frequency scales with speed)
    f._wag += dt * (6 + f.speed*260);
    f.tailPivot.rotation.y = Math.sin(f._wag) * 0.34;
    f._flap += dt * 7;
    f.pecL.rotation.z = 0.2 + Math.sin(f._flap)*0.18;
    f.pecR.rotation.z = 0.2 - Math.sin(f._flap)*0.18;
    f._dsway += dt * 2.2;
    f.dorsal.rotation.z = Math.sin(f._dsway)*0.05;
    f.anal.rotation.z = Math.sin(f._dsway*1.3)*0.06;
  }

  // --- water surface ripples ---
  const sp = surface.geometry.attributes.position;
  for (let i=0;i<sp.count;i++){
    const x = surfaceBase[i*3], y = surfaceBase[i*3+1];
    const r = 0.05*Math.sin(x*1.6 + time*1.4) + 0.04*Math.sin(y*2.1 - time*1.1)
            + 0.02*Math.sin((x+y)*3.0 + time*2.0);
    sp.setZ(i, r);
  }
  sp.needsUpdate = true;
  surface.geometry.computeVertexNormals();

  // --- bubbles ---
  streams.forEach(st=>{
    st.acc += dt;
    while (st.acc > 1/st.rate){
      st.acc -= 1/st.rate;
      if (bubbles.length < 90) spawnBubble(st);
    }
  });
  for (let i=bubbles.length-1;i>=0;i--){
    const b = bubbles[i]; const u = b.userData;
    b.position.y += u.rise*dt;
    b.position.x = u.baseX + Math.sin(time*2 + u.wob)*u.wobA;
    b.position.z = u.baseZ + Math.cos(time*1.6 + u.wob)*u.wobA;
    const near = THREE.MathUtils.clamp((waterTop - b.position.y)/0.6, 0, 1);
    b.scale.setScalar(u.r * (0.55 + 0.45*near));
    b.material.opacity = 0.55*near;
    if (b.position.y > waterTop - 0.05){
      scene.remove(b); bubbles.splice(i,1);
    }
  }

  // --- shader time ---
  plantMat.uniforms.uTime.value = time;
  causticMat.uniforms.uTime.value = time;

  controls.update();
  renderer.render(scene, camera);
}
animate();

/* ------------------------------------------------------------------ *
 *  Resize
 * ------------------------------------------------------------------ */
addEventListener('resize', ()=>{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
});
</script>
</body>
</html>
```

A few notes on what's happening under the hood:

- **Fish** are true multi-part models — a scaled ellipsoid body with a lighter belly, three vertical white bands each backed by a black border, a forked `ShapeGeometry` tail that wags around the Y-axis (frequency tied to swim speed), a dorsal sail, an anal fin, two flapping pectorals, and eyes with pupils + a specular glint. They use a **look-at + slerp** rig so they always face their velocity, with a Z-axis bank on turns, and a soft clamp keeps them a margin off every wall.
- **Glass** uses `MeshPhysicalMaterial` transmission with 5 separate panels (4 walls + floor, open top) and a matte-black rim; the water is a separate translucent volume with a sine-rippled top surface.
- **Plants** sway via a vertex shader driven by the *Current* slider; **bubbles** rise with wobble, shrink, and fade at the waterline; **caustics** are an additive animated shader on the floor.
- The **Current** and **Light** sliders and **Reset view** are live; orbit/zoom/pan are on the camera.

Drop it into a `.html` file and open it — it's fully self-contained (only the Three.js CDN and two Google fonts).
