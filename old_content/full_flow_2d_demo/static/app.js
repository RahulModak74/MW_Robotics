/* app.js — MW Robotics Terminator-HUD demo frontend, two robots.
 *
 * Each humanoid is built by makeFigure(robotId, centerX): legs walk
 * (mobile_robot, x only - a deliberate side-view simplification, not a new
 * physics claim), one arm reaches (planar_arm), the head scans
 * (joint_actuator), the figure crouches and jumps (legged_contact), the
 * other arm lifts a payload (flexible_robot), and the outline glow
 * intensifies with heat (actuator_thermal).
 *
 * Every motion is decoded frames from a trained MW manifold via the
 * backend (see robot_motion.py) - this file only draws what it is given.
 * /api/choreograph hands back a list of {robot, t, frames, ...} resolved
 * actions from mw_agentic (../mw_agentic/) - this file just schedules and
 * plays them.
 */

const svg = d3.select("#scene");
const W = 900, H = 420;
const GROUND_Y = 360;
const SCALE_X = 48;
const ARM_SCALE = 70;
const HOP_SCALE = 90;
const LEG_LEN = 68, TORSO_LEN = 62, HEAD_R = 15;

svg.attr("viewBox", `0 0 ${W} ${H}`);
svg.append("line").attr("class", "ground-line").attr("x1", 0).attr("x2", W).attr("y1", GROUND_Y).attr("y2", GROUND_Y);
for (let i = -8; i <= 8; i++) {
  svg.append("line").attr("x1", W / 2 + i * SCALE_X).attr("x2", W / 2 + i * SCALE_X)
    .attr("y1", GROUND_Y).attr("y2", GROUND_Y + 6).attr("stroke", "#23307a").attr("stroke-width", 1);
}

function makeFigure(robotId, centerX) {
  const state = { base_x: 0, theta: 0, theta1: -0.2, theta2: 0.3, joint_theta: 0.8,
                 payload_w: 0, walkPhase: 0, heat: 0.15, z: 0 };

  const figure = svg.append("g").attr("id", `figure-${robotId}`);
  const reachZone = figure.append("circle").attr("class", "reach-zone").attr("r", (1.0 + 0.8) * ARM_SCALE);
  const legL = figure.append("line").attr("class", "figure-body");
  const legR = figure.append("line").attr("class", "figure-body");
  const torso = figure.append("line").attr("class", "figure-body");
  const idleArmUpper = figure.append("line").attr("class", "figure-arm");
  const idleArmLower = figure.append("line").attr("class", "figure-arm");
  const payload = figure.append("circle").attr("class", "figure-payload").attr("r", 9).attr("opacity", 0);
  const reachUpper = figure.append("line").attr("class", "figure-arm");
  const reachLower = figure.append("line").attr("class", "figure-arm");
  const headG = figure.append("g");
  headG.append("circle").attr("class", "figure-head").attr("r", HEAD_R);
  const headScan = headG.append("line").attr("stroke", "#5577ff").attr("stroke-width", 2);
  const eyeL = headG.append("circle").attr("class", "figure-eye").attr("r", 2.2);
  const eyeR = headG.append("circle").attr("class", "figure-eye").attr("r", 2.2);
  const reticle = figure.append("circle").attr("class", "target-reticle").attr("r", 12);
  const label = figure.append("text").attr("fill", "#23307a").attr("font-size", 11)
    .attr("text-anchor", "middle").text(`UNIT ${robotId.toUpperCase()}`);

  function draw() {
    const fx = centerX + state.base_x * SCALE_X;
    const hopOffset = -(state.z || 0) * HOP_SCALE;
    const hipY = GROUND_Y + hopOffset - LEG_LEN;
    const neckY = hipY - TORSO_LEN;
    const facingRight = Math.cos(state.theta) >= 0;
    const dir = facingRight ? 1 : -1;

    const swing = Math.sin(state.walkPhase) * 14;
    const crouch = Math.max(0, -hopOffset) * 0.3;
    legL.attr("x1", fx).attr("y1", hipY).attr("x2", fx - 9 + swing).attr("y2", GROUND_Y - crouch);
    legR.attr("x1", fx).attr("y1", hipY).attr("x2", fx + 9 - swing).attr("y2", GROUND_Y - crouch);
    torso.attr("x1", fx).attr("y1", hipY).attr("x2", fx).attr("y2", neckY);
    label.attr("x", fx).attr("y", GROUND_Y + 22);

    headG.attr("transform", `translate(${fx}, ${neckY - HEAD_R - 2})`);
    const scanAng = state.joint_theta;
    headScan.attr("x1", 0).attr("y1", 0).attr("x2", (HEAD_R + 5) * Math.cos(scanAng)).attr("y2", (HEAD_R + 5) * Math.sin(scanAng));
    eyeL.attr("cx", -5 * dir).attr("cy", -2);
    eyeR.attr("cx", 2 * dir).attr("cy", -2);

    const shoulderX = fx - 6 * dir, shoulderY = neckY + 8;
    const elbowX = shoulderX - 12 * dir, elbowY = shoulderY + 20;
    const handY = elbowY + 18 - (state.payload_w || 0) * 36;
    idleArmUpper.attr("x1", shoulderX).attr("y1", shoulderY).attr("x2", elbowX).attr("y2", elbowY);
    idleArmLower.attr("x1", elbowX).attr("y1", elbowY).attr("x2", elbowX).attr("y2", handY);
    payload.attr("cx", elbowX).attr("cy", handY + 9).attr("opacity", Math.abs(state.payload_w || 0) > 0.02 ? 0.9 : 0);

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

    const glow = Math.min(1, state.heat);
    const r = Math.round(glow * 127), g = Math.round(glow * 224), b = 255;
    figure.style("filter", `drop-shadow(0 0 ${2 + glow * 12}px rgba(${r},${g},${b},${0.35 + glow * 0.5}))`);
  }

  function shoulderScreenPos() {
    const fx = centerX + state.base_x * SCALE_X;
    const neckY = GROUND_Y - LEG_LEN - TORSO_LEN;
    const dir = Math.cos(state.theta) >= 0 ? 1 : -1;
    return { x: fx + 6 * dir, y: neckY + 6 };
  }

  draw();
  return { id: robotId, state, draw, reticle, shoulderScreenPos };
}

const figures = { a: makeFigure("a", W * 0.32), b: makeFigure("b", W * 0.68) };

// ── animation playback ───────────────────────────────────────────────────
function playFrames(fig, frames, onFrame, msPerFrame = 35, walking = false) {
  let i = 0;
  const timer = setInterval(() => {
    if (walking) fig.state.walkPhase += 0.9;
    onFrame(frames[i]);
    fig.draw();
    if (fig.id === "a") updateTelemetry(frames[i]);
    i++;
    if (i >= frames.length) { clearInterval(timer); fig.state.walkPhase = 0; fig.draw(); }
  }, msPerFrame);
}

function updateTelemetry(f) {
  const lines = Object.entries(f).map(([k, v]) => `${k.padEnd(10)} ${typeof v === "number" ? v.toFixed(3) : v}`);
  d3.select("#telemetry-box").text(lines.join("\n"));
}

function logLine(text) {
  d3.select("#hud-log").append("div").attr("class", "entry").text(text);
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

function applyActionToState(fig, tool, f) {
  if (tool === "walk") { fig.state.base_x = f.x; fig.state.theta = f.theta; }
  else if (tool === "reach") { fig.state.theta1 = f.theta1; fig.state.theta2 = f.theta2; }
  else if (tool === "hop") { fig.state.z = f.z; }
  else if (tool === "lift") { fig.state.payload_w = f.w; }
  else if (tool === "joint") { fig.state.joint_theta = f.theta; }
}

// ── single-robot controls (robot "a") ───────────────────────────────────
d3.select("#btn-walk-l").on("click", async () => {
  const fig = figures.a;
  const res = await post("/api/base/move", { robot: "a", x: fig.state.base_x - 2.0, y: 0 });
  showTarget(res.domain, res.equations);
  playFrames(fig, res.frames, (f) => applyActionToState(fig, "walk", f), 35, true);
});
d3.select("#btn-walk-r").on("click", async () => {
  const fig = figures.a;
  const res = await post("/api/base/move", { robot: "a", x: fig.state.base_x + 2.0, y: 0 });
  showTarget(res.domain, res.equations);
  playFrames(fig, res.frames, (f) => applyActionToState(fig, "walk", f), 35, true);
});
d3.select("#btn-hop").on("click", async () => {
  const fig = figures.a;
  const res = await post("/api/hop", { robot: "a", height: 0.5 + Math.random() * 0.5 });
  showTarget(res.domain, res.equations);
  playFrames(fig, res.frames, (f) => applyActionToState(fig, "hop", f), 30);
});
d3.select("#btn-lift").on("click", async () => {
  const fig = figures.a;
  const res = await post("/api/lift", { robot: "a", deflection: 0.5 + Math.random() * 0.4 });
  showTarget(res.domain, res.equations);
  playFrames(fig, res.frames, (f) => applyActionToState(fig, "lift", f), 40);
});
d3.select("#joint-slider").on("input change", async function () {
  const fig = figures.a;
  const theta = parseFloat(this.value);
  const res = await post("/api/joint/rotate", { robot: "a", theta });
  showTarget(res.domain, res.equations);
  playFrames(fig, res.frames, (f) => applyActionToState(fig, "joint", f), 25);
});

// click near a figure's reaching hand -> planar_arm IK for that robot
svg.on("click", async (ev) => {
  const [mx, my] = d3.pointer(ev);
  for (const fig of Object.values(figures)) {
    const { x: shoulderX, y: shoulderY } = fig.shoulderScreenPos();
    const sx = mx - shoulderX, sy = my - shoulderY;
    const r = Math.hypot(sx, sy) / ARM_SCALE;
    if (r > 1.9 || r < 0.05) continue;
    const screenAngle = Math.atan2(sy, sx);
    const physAngle = -screenAngle - Math.PI / 2;
    const px = r * Math.cos(physAngle), py = r * Math.sin(physAngle);
    fig.reticle.attr("opacity", 1);
    const res = await post("/api/arm/reach", { robot: fig.id, x: px, y: py });
    showTarget(res.domain, res.equations);
    playFrames(fig, res.frames, (f) => applyActionToState(fig, "reach", f), 30);
    setTimeout(() => fig.reticle.attr("opacity", 0), 700);
    return;
  }
});

// ── choreography: ask mw_agentic, then schedule both robots ─────────────
d3.select("#btn-choreograph").on("click", async () => {
  const task = document.getElementById("choreo-input").value.trim();
  if (!task) return;
  logLine(`TASK: "${task}"`);
  const res = await post("/api/choreograph", { task });
  if (!res.ok) {
    logLine(`CHOREOGRAPHY FAILED: ${(res.errors || []).join("; ")}`);
    return;
  }
  logLine(`PLAN: ${res.spec.title} (${res.path}, ${res.llm_calls} model call${res.llm_calls === 1 ? "" : "s"}` +
         (res.recipe ? `, recipe=${res.recipe}` : "") + ")");
  d3.select("#hud-domain").text("CHOREOGRAPHY");
  d3.select("#hud-eq").text(res.spec.title);
  for (const action of res.resolved) {
    const fig = figures[action.robot];
    setTimeout(() => {
      logLine(`[${action.robot}] ${action.tool} @ t=${action.t.toFixed(1)}s`);
      playFrames(fig, action.frames, (f) => applyActionToState(fig, action.tool, f),
                action.tool === "walk" ? 35 : action.tool === "hop" ? 30 : action.tool === "lift" ? 40 : 30,
                action.tool === "walk");
    }, action.t * 1000);
  }
});

// ── thermal polling, robot "a" drives the HUD readout ────────────────────
async function refreshThermal() {
  for (const fig of Object.values(figures)) {
    const r = await fetch(`/api/thermal/status?robot=${fig.id}`);
    const d = await r.json();
    fig.state.heat = d.excess;
    fig.draw();
    if (fig.id === "a") d3.select("#hud-heat").text(`${(d.excess * 100).toFixed(0)}%`);
  }
}
refreshThermal();
setInterval(refreshThermal, 4000);

logLine("SYSTEM BOOT: six manifolds loaded, two units online");
logLine("awaiting operator input");
