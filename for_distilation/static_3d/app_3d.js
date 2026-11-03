import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const el = document.getElementById("scene3d");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(el.clientWidth, el.clientHeight); el.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x02040f);
const cam = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 0.05, 100);
cam.position.set(3, -4, 2); cam.up.set(0, 0, 1);                 // MuJoCo is z-up
const orbit = new OrbitControls(cam, renderer.domElement);
scene.add(new THREE.AmbientLight(0x8899ff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(3, -2, 6); scene.add(sun);
const grid = new THREE.GridHelper(60, 60, 0x2a3aa0, 0x16205a); grid.rotation.x = Math.PI / 2; scene.add(grid);

let meshes = [], geoms = [], frames = [], playing = null;
const $ = id => document.getElementById(id);

function buildBody(gs) {
  meshes.forEach(m => scene.remove(m)); meshes = []; geoms = gs.filter(g => g.type !== "plane");
  for (const g of geoms) {
    const s = g.size; let geo;
    if (g.type === "sphere") geo = new THREE.SphereGeometry(s[0], 16, 12);
    else if (g.type === "capsule") { geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 12); geo.rotateX(Math.PI / 2); }   // local z axis
    else if (g.type === "cylinder") { geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 16); geo.rotateX(Math.PI / 2); }
    else if (g.type === "box") geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    else { geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(s[0], s[1], s[2]); }
    const col = g.body === 0 ? 0x16205a : 0x3d63ff;
    const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: col, metalness: 0.4, roughness: 0.5 }));
    m.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 40), new THREE.LineBasicMaterial({ color: 0x7fe0ff })));
    scene.add(m); meshes.push(m);
  }
}

function showFrame(i) {
  const f = frames[i], n = Math.min(meshes.length, f.p.length);
  for (let k = 0; k < n; k++) {
    meshes[k].position.set(...f.p[k]);
    meshes[k].quaternion.set(f.q[k][1], f.q[k][2], f.q[k][3], f.q[k][0]);
  }
  orbit.target.set(f.x, 0, 0.9); cam.position.x += (f.x - (cam.position.x - 3)) * 0.0;   // keep orbit centred on the robot
  orbit.update();
  $("h-step").textContent = `${i + 1}/${frames.length}`;
  $("h-x").textContent = `${f.x.toFixed(2)} m`; $("h-z").textContent = `${f.z.toFixed(2)} m`;
}

async function init() {
  const b = await (await fetch("/api3d/bodies")).json();
  window.BODIES = b;
  $("body").innerHTML = Object.entries(b).map(([k, v]) => `<option value="${k}">${k} (${v.env})</option>`).join("");
  await loadBody();
  $("body").onchange = loadBody;
}
async function loadBody() {
  const tag = $("body").value, info = window.BODIES[tag];
  $("h-env").textContent = info.env;
  const r = info.report || {};
  const f = o => o ? `reward ${o.reward.toFixed(0)}  steps ${o.steps.toFixed(0)}  x ${o.x.toFixed(2)} m` : "n/a";
  $("report").textContent = r.mw_manifold
    ? `CEM teacher   ${f(r.teacher)}\nplain CVAE    ${f(r.plain_cvae_baseline)}\nMW manifold   ${f(r.mw_manifold)}\n${r.n_rows} (s,a) rows, MW median dist ${r.mw_median.toFixed(2)}`
    : "no report";
  buildBody((await (await fetch(`/api3d/model?body=${tag}`)).json()).geoms);
}

$("run").onclick = async () => {
  clearInterval(playing); $("run").disabled = true; $("summary").textContent = "simulating…";
  const res = await (await fetch("/api3d/rollout", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ body: $("body").value, mode: $("mode").value, seed: +$("seed").value,
                           steps: +$("steps").value, noise: +$("noise").value, push: +$("push").value }) })).json();
  $("run").disabled = false;
  if (res.error) { $("summary").textContent = res.error; return; }
  frames = res.frames; const s = res.summary;
  $("summary").textContent = `steps ${s.steps}${s.fell_at ? "  (FELL at " + s.fell_at + ")" : "  (survived)"}\nreward ${s.reward}\nforward ${s.forward_m} m`;
  let i = 0; showFrame(0);
  playing = setInterval(() => { if (++i >= frames.length) { clearInterval(playing); return; } showFrame(i); }, res.dt * 1000);
};

(function loop() { requestAnimationFrame(loop); renderer.render(scene, cam); })();
window.onresize = () => { renderer.setSize(el.clientWidth, el.clientHeight); cam.aspect = el.clientWidth / el.clientHeight; cam.updateProjectionMatrix(); };
init();
