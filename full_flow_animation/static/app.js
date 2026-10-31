/* app.js — MW Robotics Terminator-HUD demo frontend.
 *
 * A stylised humanoid rig, side view: legs walk (mobile_robot, x only - y and
 * true 2D floor navigation are not visualised here, a deliberate simplification
 * for a readable side-on figure, not a new physics claim), one arm reaches
 * (planar_arm), the head scans (joint_actuator), the whole figure crouches and
 * jumps (legged_contact), the other arm lifts a payload (flexible_robot), and
 * the figure's outline glow intensifies with heat (actuator_thermal).
 *
 * Every motion is decoded frames from a trained MW manifold via the backend
 * (see robot_motion.py) - this file only draws what it is given.
 */

const svg = d3.select("#scene");
const W = 640, H = 420;
const GROUND_Y = 360;
const CENTER_X = 320;
const SCALE_X = 55;          // px per world x unit (mobile_robot)
const ARM_SCALE = 80;        // px per world unit (planar_arm, L1=1.0 L2=0.8 -> reach ~144px)
const HOP_SCALE = 90;        // px per world unit (legged_contact z)
const LEG_LEN = 70, TORSO_LEN = 68, HEAD_R = 17;

let state = { base_x: 0, theta: 0, theta1: -0.2, theta2: 0.3, joint_theta: 0.8,
             payload_w: 0, walkPhase: 0, heat: 0.15 };

// ── static scene furniture ──────────────────────────────────────────────
svg.append("line").attr("class", "ground-line").attr("x1", 0).attr("x2", W).attr("y1", GROUND_Y).attr("y2", GROUND_Y);
for (let i = -6; i <= 6; i++) {
  svg.append("line").attr("x1", CENTER_X + i * SCALE_X).attr("x2", CENTER_X + i * SCALE_X)
    .attr("y1", GROUND_Y).attr("y2", GROUND_Y + 6).attr("stroke", "#14803a").attr("stroke-width", 1);
}

const figure = svg.append("g").attr("id", "figure");
const reachZone = figure.append("circle").attr("class", "reach-zone").attr("r", (1.0 + 0.8) * ARM_SCALE);
const legL = figure.append("line").attr("class", "figure-body");
const legR = figure.append("line").attr("class", "figure-body");
const torso = figure.append("line").attr("class", "figure-body");
const idleArmUpper = figure.append("line").attr("class", "figure-arm");
const idleArmLower = figure.append("line").attr("class", "figure-arm");
const payload = figure.append("circle").attr("class", "figure-payload").attr("r", 10).attr("opacity", 0);
const reachUpper = figure.append("line").attr("class", "figure-arm");
const reachLower = figure.append("line").attr("class", "figure-arm");
const headG = figure.append("g");
const headCircle = headG.append("circle").attr("class", "figure-head").attr("r", HEAD_R);
const headScan = headG.append("line").attr("stroke", "#35ff6e").attr("stroke-width", 2);
const eyeL = headG.append("circle").attr("class", "figure-eye").attr("r", 2.4);
const eyeR = headG.append("circle").attr("class", "figure-eye").attr("r", 2.4);
const reticle = figure.append("circle").attr("class", "target-reticle").attr("r", 14);

function draw() {
  const fx = CENTER_X + state.base_x * SCALE_X;
  const hopOffset = -(state.z || 0) * HOP_SCALE;
  const hipY = GROUND_Y + hopOffset - LEG_LEN;
  const neckY = hipY - TORSO_LEN;
  const facingRight = Math.cos(state.theta) >= 0;
  const dir = facingRight ? 1 : -1;

  // legs: simple alternating walk-cycle swing, squashed toward the ground on hop
  const swing = Math.sin(state.walkPhase) * 16;
  const crouch = Math.max(0, -hopOffset) * 0.3;    // slight knee bend on the way up/down
  legL.attr("x1", fx).attr("y1", hipY).attr("x2", fx - 10 + swing).attr("y2", GROUND_Y - crouch);
  legR.attr("x1", fx).attr("y1", hipY).attr("x2", fx + 10 - swing).attr("y2", GROUND_Y - crouch);
  torso.attr("x1", fx).attr("y1", hipY).attr("x2", fx).attr("y2", neckY);

  // head: scans (rotates) with joint_actuator's angle
  headG.attr("transform", `translate(${fx}, ${neckY - HEAD_R - 2})`);
  const scanAng = state.joint_theta;
  headScan.attr("x1", 0).attr("y1", 0).attr("x2", (HEAD_R + 6) * Math.cos(scanAng)).attr("y2", (HEAD_R + 6) * Math.sin(scanAng));
  eyeL.attr("cx", -6 * dir).attr("cy", -2);
  eyeR.attr("cx", 2 * dir).attr("cy", -2);

  // idle / lifting arm: holds the payload, forearm position driven by beam deflection
  const shoulderX = fx - 6 * dir, shoulderY = neckY + 8;
  const elbowX = shoulderX - 14 * dir, elbowY = shoulderY + 24;
  const handY = elbowY + 22 - (state.payload_w || 0) * 40;
  idleArmUpper.attr("x1", shoulderX).attr("y1", shoulderY).attr("x2", elbowX).attr("y2", elbowY);
  idleArmLower.attr("x1", elbowX).attr("y1", elbowY).attr("x2", elbowX).attr("y2", handY);
  payload.attr("cx", elbowX).attr("cy", handY + 10).attr("opacity", Math.abs(state.payload_w || 0) > 0.02 ? 0.9 : 0);

  // reaching arm: driven directly by planar_arm's (theta1, theta2)
  const rShoulderX = fx + 6 * dir, rShoulderY = neckY + 6;
  const ang1 = -state.theta1 - Math.PI / 2;
  const angTotal = ang1 - state.theta2;
  const elbow2X = rShoulderX + ARM_SCALE * 1.0 * Math.cos(ang1);
  const elbow2Y = rShoulderY + ARM_SCALE * 1.0 * Math.sin(ang1);
  const handX = elbow2X + ARM_SCALE * 0.8 * Math.cos(angTotal);
  const handY2 = elbow2Y + ARM_SCALE * 0.8 * Math.sin(angTotal);
  reachUpper.attr("x1", rShoulderX).attr("y1", rShoulderY).attr("x2", elbow2X).attr("y2", elbow2Y);
  reachLower.attr("x1", elbow2X).attr("y1", elbow2Y).attr("x2", handX).attr("y2", handY2);
  reticle.attr("cx", handX).attr("cy", handY2);
  reachZone.attr("cx", rShoulderX).attr("cy", rShoulderY);

  // heat glow on the whole figure - brand blue intensifying toward bright cyan with activity
  const glow = Math.min(1, state.heat);
  const r = Math.round(0 + glow * 127), g = Math.round(0 + glow * 224), b = 255;
  figure.style("filter", `drop-shadow(0 0 ${2 + glow * 12}px rgba(${r},${g},${b},${0.35 + glow * 0.5}))`);
}
draw();

// ── animation playback ───────────────────────────────────────────────────
function playFrames(frames, onFrame, msPerFrame = 35, walking = false) {
  let i = 0;
  const timer = setInterval(() => {
    if (walking) state.walkPhase += 0.9;
    onFrame(frames[i]);
    draw();
    updateTelemetry(frames[i]);
    i++;
    if (i >= frames.length) { clearInterval(timer); state.walkPhase = 0; draw(); }
  }, msPerFrame);
}

function updateTelemetry(f) {
  const lines = Object.entries(f).map(([k, v]) => `${k.padEnd(10)} ${typeof v === "number" ? v.toFixed(3) : v}`);
  d3.select("#telemetry-box").text(lines.join("\n"));
}

function logLine(text) {
  const el = d3.select("#hud-log").append("div").attr("class", "entry").text(text);
  const node = document.getElementById("hud-log");
  node.scrollTop = node.scrollHeight;
}

function showTarget(domain, equations) {
  d3.select("#hud-domain").text(domain.toUpperCase());
  d3.select("#hud-eq").text(equations);
  logLine(`EXECUTING ${domain}`);
}

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                              body: JSON.stringify(body || {}) });
  return r.json();
}

// ── controls ──────────────────────────────────────────────────────────
d3.select("#btn-walk-l").on("click", async () => {
  const res = await post("/api/base/move", { x: state.base_x - 2.0, y: 0 });
  showTarget(res.domain, res.equations);
  playFrames(res.frames, (f) => { state.base_x = f.x; state.theta = f.theta; }, 35, true);
});
d3.select("#btn-walk-r").on("click", async () => {
  const res = await post("/api/base/move", { x: state.base_x + 2.0, y: 0 });
  showTarget(res.domain, res.equations);
  playFrames(res.frames, (f) => { state.base_x = f.x; state.theta = f.theta; }, 35, true);
});
d3.select("#btn-hop").on("click", async () => {
  const res = await post("/api/hop", { height: 0.5 + Math.random() * 0.5 });
  showTarget(res.domain, res.equations);
  playFrames(res.frames, (f) => { state.z = f.z; }, 30);
});
d3.select("#btn-lift").on("click", async () => {
  const res = await post("/api/lift", { deflection: 0.5 + Math.random() * 0.4 });
  showTarget(res.domain, res.equations);
  playFrames(res.frames, (f) => { state.payload_w = f.w; }, 40);
});
d3.select("#joint-slider").on("input change", async function () {
  const theta = parseFloat(this.value);
  const res = await post("/api/joint/rotate", { theta });
  showTarget(res.domain, res.equations);
  playFrames(res.frames, (f) => { state.joint_theta = f.theta; }, 25);
});

// ── click near the reaching hand -> planar_arm IK, inverse of the draw() mapping ──
svg.on("click", async (ev) => {
  const [mx, my] = d3.pointer(ev);
  const facingRight = Math.cos(state.theta) >= 0;
  const dir = facingRight ? 1 : -1;
  const fx = CENTER_X + state.base_x * SCALE_X;
  const neckY = GROUND_Y - LEG_LEN - TORSO_LEN;
  const shoulderX = fx + 6 * dir, shoulderY = neckY + 6;

  const sx = mx - shoulderX, sy = my - shoulderY;
  const r = Math.hypot(sx, sy) / ARM_SCALE;
  if (r > 1.9 || r < 0.05) return;    // outside the arm's reach - ignore
  const screenAngle = Math.atan2(sy, sx);
  const physAngle = -screenAngle - Math.PI / 2;
  const px = r * Math.cos(physAngle), py = r * Math.sin(physAngle);

  reticle.attr("opacity", 1);
  const res = await post("/api/arm/reach", { x: px, y: py });
  showTarget(res.domain, res.equations);
  playFrames(res.frames, (f) => { state.theta1 = f.theta1; state.theta2 = f.theta2; }, 30);
  setTimeout(() => reticle.attr("opacity", 0), 700);
});

// ── thermal polling ──────────────────────────────────────────────────
async function refreshThermal() {
  const r = await fetch("/api/thermal/status");
  const d = await r.json();
  state.heat = d.excess;
  draw();
  d3.select("#hud-heat").text(`${(d.excess * 100).toFixed(0)}%`);
}
refreshThermal();
setInterval(refreshThermal, 4000);

logLine("SYSTEM BOOT: six manifolds loaded");
logLine("awaiting operator input");
