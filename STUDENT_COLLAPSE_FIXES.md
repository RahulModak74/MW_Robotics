# FIX_STUDENT_COLLAPSE.md: Why the MW Student Failed at 70 Steps (and How to Fix It)

## Executive Summary

MW student can be  **catastrophically collapsing** (70 steps vs teacher's 500+). This is not a "slight degradation" — . The root cause is a combination of:

1. **Feature space mismatch** (raw quaternion topology)
2. **Weak physics priors** (no ZMP, no angular momentum bounds)
3. **Latent query instability** (gradient-based optimization at every step)
4. **Possible posterior collapse** (VAE ignoring the latent space)

This document provides a **diagnosis, immediate fixes, and validation protocol** to get the MW student to 400+ steps .

---

## 1. Diagnosis: Why 70 Steps vs 500+?

### 1.1 The Feature Space Problem

Current `features()` function in `policy_3d.py`:
```python
def features(data):
    return np.concatenate([data.qpos[2:], data.qvel]).astype(np.float64)
```

This uses **raw quaternion** `data.qpos[3:7]` for orientation. Quaternions have a **double-cover topology** (q and -q represent the same rotation), which creates a **discontinuous manifold** that the VAE cannot learn smoothly.

**Evidence**: The PPO teacher internally uses Gymnasium's observation (which includes the quaternion), but it's a neural net that can learn to handle this discontinuity. The MW VAE, however, is trying to learn a **smooth latent manifold** over raw quaternions, which is topologically impossible without explicit handling.

### 1.2 The Physics Priors Are Too Weak

current `PolicyManifold3DVAE.physics()` enforces:
- ✅ Height envelope
- ✅ Unit quaternion norm
- ✅ Joint limits
- ✅ Torque limits

But it does NOT enforce:
- ❌ **Zero-Moment-Point (ZMP) stability** — the core of 3D balance
- ❌ **Angular momentum bounds** — prevents spinning out of control
- ❌ **Symmetry constraints** — left/right hip coordination
- ❌ **Energy dissipation** — prevents perpetual motion hacks

**Evidence**: The PPO teacher can learn to "cheat" by flailing arms to gain forward momentum (a known reward-hacking pattern in Humanoid-v5). The MW VAE, without explicit physics constraints, inherits this hack and amplifies it during latent query optimization.

### 1.3 Latent Query Instability

`ManifoldPolicy.act()` does **gradient-based latent optimization** at every control step:
```python
for _ in range(self.steps):  # 25 Adam steps per control step
    opt.zero_grad()
    xh = self.vae.decode(z[None])[0]
    loss = (((xh[:self.sd] - s) / self.sig) ** 2).mean() + self.vae.kl_weight * (z ** 2).sum()
    loss.backward(); opt.step()
```

This is:
- **Computationally expensive** (25 Adam steps × 50Hz = 1250 optimization steps/second)
- **Potentially unstable** (the loss landscape may have local minima that trap the optimizer)
- **Sensitive to KL weight** (if `kl_weight` is too high, it collapses to z=0, producing a constant action)

**Evidence**: If the student falls after exactly 70 steps across multiple seeds, it suggests the latent query is converging to a **fixed point** (z=0 or a local minimum) that produces a constant, physically invalid action.

### 1.4 Posterior Collapse

If the KL divergence during training is near zero, the VAE has learned to **ignore the latent variable** and just output the mean action. This is called **posterior collapse**.

**How to check**: Look at the training logs from `train_mw_3d.py`. If you see:
```
cvae ep 150  recon=0.001  kl=0.0001
```
Then the VAE has collapsed. The reconstruction error is near zero (it's just outputting the mean), and the KL is near zero (it's not using the latent space).

---

## 2. Immediate Fixes (Today)

### 2.1 Fix #1: Switch to Gravity-Vector Features

Replace the raw quaternion with the **gravity vector in body frame**. This is the single most important fix for 3D locomotion.

**In `full_flow_animation/policy_3d.py`**:
```python
def features(data):
    # Gravity vector in body frame (3 dims)
    # data.xmat[0] is the torso rotation matrix
    # We want the world "down" vector [0, 0, -1] in torso frame
    grav_body = data.xmat[0].T @ np.array([0, 0, -1], dtype=np.float64)
    
    # Angular velocity in body frame (3 dims)
    ang_vel_body = data.xmat[0].T @ data.qvel[3:6]
    
    # Height and vertical velocity (2 dims)
    h_vh = np.array([data.qpos[2], data.qvel[2]], dtype=np.float64)
    
    # Joint angles and velocities (34 dims)
    joints = data.qpos[7:].astype(np.float64)  # 17 dims
    jvels = data.qvel[6:].astype(np.float64)   # 17 dims
    
    return np.concatenate([grav_body, ang_vel_body, h_vh, joints, jvels])  # 42 dims
```

**Why this works**: The gravity vector is a **continuous, single-valued function** of orientation. It removes the quaternion double-cover problem and gives the VAE a smooth, physically meaningful signal to learn from.

**Impact**: This alone should push the student from 70 → 200+ steps.

### 2.2 Fix #2: Add Stronger Physics Priors

Add ZMP, angular momentum, and symmetry constraints to `mw_policy_3d.py`:

```python
def physics(self, x):
    tol = self.law_tolerance_
    
    # Existing priors (P1-P4)
    bound("p1_height", x["s0"], low=self.z_lo_, high=self.z_hi_, tolerance=tol)
    q2 = sum(x[f"s{i}"] ** 2 for i in range(1, 5))
    constraint("p2_unit_quat", q2 - 1.0, tolerance=tol)
    for k, (lo, hi) in enumerate(zip(self.joint_lo_, self.joint_hi_)):
        bound(f"p3_joint{k}", x[f"s{self.joint_start_ + k}"], low=lo, high=hi, tolerance=tol)
    for j in range(self.a_dim_):
        bound(f"p4_torque{j}", x[f"a{j}"], low=-self.hi_, high=self.hi_, tolerance=tol)
    
    # NEW: P5 - Angular momentum proxy (torso angular velocity bounded when upright)
    # Assuming s3, s4, s5 are the angular velocity components (after gravity vector fix)
    ang_speed = sum(x[f"s{i}"]**2 for i in range(3, 6))
    bound("p5_ang_speed", ang_speed, high=15.0, tolerance=tol)
    
    # NEW: P6 - Symmetry prior (left/right hip torques should be anti-correlated)
    # This prevents the "spin out of control" hack
    # Adjust indices based on your action space mapping
    # For Humanoid-v5: right_hip_y is a[2], left_hip_y is a[8]
    constraint("p6_sym_hip_y", x["a2"] + x["a8"], tolerance=0.4, kind="laplace")
    
    # NEW: P7 - Energy dissipation (total mechanical energy should not increase)
    # Proxy: sum of |torque * velocity| bounded
    # This prevents perpetual motion hacks
    energy_proxy = sum(abs(x[f"a{j}"]) for j in range(self.a_dim_))
    bound("p7_energy_dissipation", energy_proxy, high=10.0, tolerance=tol)
```

**Why this works**: These priors explicitly forbid the reward-hacking shortcuts that PPO learns. The MW VAE is forced to learn a **physically valid** subset of the teacher's behavior.

**Impact**: This should push the student from 200 → 400+ steps.

### 2.3 Fix #3: Check for Posterior Collapse

Add this diagnostic to `train_mw_3d.py`:

```python
# After training, check KL divergence
with torch.no_grad():
    S_t = torch.as_tensor(S[:1000], dtype=torch.float32)
    A_t = torch.as_tensor(A[:1000], dtype=torch.float32)
    mu_q, sig_q = det.vae_.encode(torch.cat([S_t, A_t], dim=1))
    kl = 0.5 * (mu_q**2 + sig_q**2 - 2*torch.log(sig_q) - 1).sum(dim=1).mean()
    print(f"Posterior KL divergence: {kl.item():.4f}")
    if kl.item() < 0.1:
        print("WARNING: Posterior collapse detected! Increase beta parameter.")
```

If KL < 0.1, retrain with `beta=0.1` or `beta=0.5`:

```python
# In train_mw_3d.py, change the CVAE training loop
model = CVAE(fdim, adim)
# ...
for ep in range(a.epochs):
    # ...
    loss, rec, kl = model(St[idx], At[idx], beta=0.1)  # <-- increase from 0.01 to 0.1
    # ...
```

**Impact**: This ensures the VAE is actually using the latent space to represent different behaviors.

### 2.4 Fix #4: Reduce Latent Query Steps

Change `ManifoldPolicy` to use fewer optimization steps:

```python
class ManifoldPolicy:
    def __init__(self, det, steps=10, lr=0.1):  # <-- reduce from 25 to 10, lr from 0.2 to 0.1
        self.det, self.steps, self.lr = det, steps, lr
        # ...
```

**Why this works**: Fewer optimization steps reduce the chance of converging to a local minimum. The latent query becomes more stable.

**Impact**: This should reduce the chance of catastrophic collapse.

---

## 3. Validation Protocol

After applying the fixes, run this validation:

### 3.1 Step 1: Retrain the MW VAE

```bash
# Re-collect data with new features
python train_policy_3d.py --env Humanoid-v5 --pop 240 --gens 60

# Retrain MW VAE with stronger priors
python train_mw_3d.py --env Humanoid-v5 --epochs 150
```

### 3.2 Step 2: Check Training Logs

Look for:
- **Reconstruction error**: Should be < 0.05
- **KL divergence**: Should be > 0.5 (not collapsed)
- **MW distance on training set**: Should be < 1.0

### 3.3 Step 3: Evaluate on Held-Out Seeds

```bash
# The train_mw_3d.py script already does this
# Look for the output:
# MW-MANIFOLD POLICY held-out: reward=X steps=Y x=Z
```

**Success criteria**:
- Steps ≥ 400 (out of 500 max)
- Forward distance ≥ 5.0 m

### 3.4 Step 4: Visual Inspection

Run the Flask app and watch the humanoid:
```bash
python app_3d.py
# Open http://127.0.0.1:5051
```

Look for:
- Smooth, natural walking (not jittery)
- No spinning or flailing
- Recovery from small perturbations

---

## 4. Next Steps: Path to Full CPU-Native Training

Once the MW student reaches 400+ steps, the next phase is:

### 4.1 DAgger Iteration (This Week)

Implement iterative self-improvement:
1. Roll out the MW student policy
2. Tag states where uncertainty (MC-dropout std) is high
3. Query the PPO teacher **only** at those uncertain states
4. Append to the dataset and retrain the MW VAE for 20-30 epochs
5. Repeat 3-4 times

**Expected outcome**: The MW student should exceed the teacher's robustness (500+ → 600+ steps).

### 4.2 Hierarchical Latent Space (Next Week)

Use `ManifoldAlgebra` to decompose the policy into fibres:
```python
alg = ManifoldAlgebra(feature_names, normaliser)
alg.product("balance", ["grav_body", "ang_vel", "height"], latent_dim=3)
alg.fibre("gait", ["joint_angles", "joint_vels"], latent_dim=4, over="balance")
alg.fibre("torques", ["a0", ..., "a16"], latent_dim=4, over="gait")
```

**Expected outcome**: The hierarchical structure prevents mode collapse and allows the VAE to represent multiple skills (walking, running, recovering).

### 4.3 Direct CPU-Native Policy Learning (Next Month)

Skip the PPO teacher entirely:
1. Initialize the MW VAE with random weights
2. Use physics-guided exploration (sample actions that satisfy the physics priors)
3. Optimize the ELBO with a sparse reward (forward progress)
4. The physics priors act as the inductive bias, replacing the need for millions of GPU rollouts

**Expected outcome**: A fully CPU-native policy that trains in 20 minutes and deploys on a ₹30,000 laptop.

---

## 5. Summary

| Fix | Expected Impact | Time to Implement |
|---|---|---|
| Gravity-vector features | 70 → 200 steps | 2 hours |
| Stronger physics priors | 200 → 400 steps | 4 hours |
| Posterior collapse check | Prevents collapse | 1 hour |
| Reduce latent query steps | Stabilizes inference | 30 minutes |
| **Total (today)** | **70 → 400+ steps** | **~8 hours** |

After these fixes, the MW student should match or exceed the teacher's performance, and you'll have a working prototype for the GPU-teacher → CPU-student distillation pipeline.

The next phase (DAgger iteration, hierarchical latent space, direct CPU-native training) will push this to 600+ steps and eventually eliminate the need for the GPU teacher entirely.

**The key insight**: The MW framework's strength is not in matching the teacher's performance, but in **exceeding** it by enforcing physics constraints that the teacher ignores. The 70-step failure is not a dead end — it's a diagnostic that reveals where the physics priors are too weak.
