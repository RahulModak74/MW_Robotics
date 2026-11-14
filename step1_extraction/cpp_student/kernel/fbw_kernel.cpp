// fbw_kernel.cpp - the MW fly-by-wire look-ahead (FlyByWire._score) as one native loop.
//
// Per control step k of the horizon, for every rollout b (in parallel, one MjData per thread):
//   target  = clip(pilot_k + correction_k + noise_k, lo, hi)      (pilot_0 = the pilot's command now)
//   physics = decim x mj_step from the stored full-physics state (warm start zeroed, as mujoco.rollout does)
//   one kinematics pass (mj_kinematics, mj_comPos, mj_comVel, mj_subtreeVel) -> proprioception + envelope features + costs
//   next observation (priors.next_obs)
// then the pilot (the student's action path, float32 MLP in Eigen) on all live rollouts at once.
// Rollouts that have fallen (FBW_FALL=1) are stopped: their cost is final, so they are no longer simulated.
//
// Build: see build.sh (g++ -O3 -mavx2 -mfma -fopenmp, Eigen, pybind11, MuJoCo C API from the mujoco wheel).
#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <omp.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;
using MatF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VecF = Eigen::VectorXf;
using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arrf = py::array_t<float, py::array::c_style | py::array::forcecast>;

static constexpr int NOBS = 349, NJ = 29;

struct Linear { MatF W; VecF b; };          // y = x W^T + b  (torch layout W [out, in])

static inline void silu_(MatF& h) {
  float* p = h.data(); const Eigen::Index n = h.size();
  for (Eigen::Index i = 0; i < n; ++i) { float v = p[i]; p[i] = v / (1.0f + std::exp(-v)); }
}
static inline MatF lin(const MatF& x, const Linear& L) {
  MatF y = x * L.W.transpose(); y.rowwise() += L.b.transpose(); return y;
}

// The student's action path only: norm -> [encoder -> z_mu] -> body([x, z]) -> mean -> denorm.
// (uq() additionally runs the MW-distance decoder, logvar and, for LastLayer, the encoder/body a second time and a 513x513
// product; none of it affects the action.)
struct Pilot {
  VecF xm, xs, ym, ys;
  std::vector<Linear> enc, body; Linear zmu, mean; bool has_enc = false;
  Eigen::VectorXd off, sc;
  // X [B, 349] (double) -> joint targets [B, 29] (double)
  void run(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& X, double* out) const {
    const Eigen::Index B = X.rows();
    MatF xn = X.cast<float>();
    xn = ((xn.rowwise() - xm.transpose()).array().rowwise() / xs.transpose().array()).cwiseMax(-8.f).cwiseMin(8.f).matrix();
    MatF inp;
    if (has_enc) {
      MatF h = xn;
      for (auto& L : enc) { h = lin(h, L); silu_(h); }
      MatF z = lin(h, zmu);
      inp.resize(B, xn.cols() + z.cols()); inp << xn, z;
    } else inp = xn;
    MatF h = inp;
    for (auto& L : body) { h = lin(h, L); silu_(h); }
    MatF mu = lin(h, mean);
    for (Eigen::Index b = 0; b < B; ++b)
      for (int j = 0; j < NJ; ++j) {
        float a = mu(b, j) * ys[j] + ym[j];
        out[b * NJ + j] = off[j] + sc[j] * (double)a;
      }
  }
};

static Linear to_lin(py::dict d) {
  arrf W = d["W"].cast<arrf>(), b = d["b"].cast<arrf>();
  Linear L; L.W = Eigen::Map<const MatF>(W.data(), W.shape(0), W.shape(1));
  L.b = Eigen::Map<const VecF>(b.data(), b.shape(0)); return L;
}
static VecF to_vec(py::handle h) { arrf a = h.cast<arrf>(); return Eigen::Map<const VecF>(a.data(), a.shape(0)); }

class Kernel {
 public:
  Kernel(std::uintptr_t model_addr, std::vector<int> act, std::vector<int> qad, std::vector<int> dof, int torso, int root,
         std::vector<int> feet, int decim, int threads, py::dict pilot, arr lo, arr hi, double timestep)
      : act_(act), qad_(qad), dof_(dof), torso_(torso), root_(root), feet_(feet), decim_(decim), nthr_(threads) {
    const mjModel* src = reinterpret_cast<const mjModel*>(model_addr);
    m_ = mj_copyModel(nullptr, src);                       // the kernel's own prior (never the world's data)
    if (timestep > 0) m_->opt.timestep = timestep;         // fast mode: coarser look-ahead physics
    for (int t = 0; t < nthr_; ++t) d_.push_back(mj_makeData(m_));
    spec_ = mjSTATE_FULLPHYSICS; ns_ = mj_stateSize(m_, spec_);
    lo_.assign(lo.data(), lo.data() + NJ); hi_.assign(hi.data(), hi.data() + NJ);
    P_.xm = to_vec(pilot["xm"]); P_.xs = to_vec(pilot["xs"]); P_.ym = to_vec(pilot["ym"]); P_.ys = to_vec(pilot["ys"]);
    for (auto L : pilot["body"].cast<py::list>()) P_.body.push_back(to_lin(L.cast<py::dict>()));
    P_.mean = to_lin(pilot["mean"].cast<py::dict>());
    if (pilot.contains("enc")) {
      P_.has_enc = true;
      for (auto L : pilot["enc"].cast<py::list>()) P_.enc.push_back(to_lin(L.cast<py::dict>()));
      P_.zmu = to_lin(pilot["zmu"].cast<py::dict>());
    }
    arr off = pilot["off"].cast<arr>(), sc = pilot["sc"].cast<arr>();
    P_.off = Eigen::Map<const Eigen::VectorXd>(off.data(), NJ); P_.sc = Eigen::Map<const Eigen::VectorXd>(sc.data(), NJ);
  }
  ~Kernel() { for (auto d : d_) mj_deleteData(d); mj_deleteModel(m_); }

  int nstate() const { return ns_; }

  // pilot(X) for testing against torch
  arr pilot(arr X) {
    const Eigen::Index B = X.shape(0);
    arr out({(py::ssize_t)B, (py::ssize_t)NJ});
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Xm(X.data(), B, NOBS);
    P_.run(Xm, out.mutable_data());
    return out;
  }

  // dT [n,H,J] corrections, E [S,H,J] noise, s0 [ns], x [349], T_pilot [J], cfg dict.
  // returns parts [B,5] (per rollout b = candidate b/S, draw b%S) and fallen [B].
  py::tuple score(arr dT, arr E, arr s0, arr x, arr Tp0, py::dict cfg, arr warm0) {
    const int n = dT.shape(0), H = dT.shape(1), S = E.shape(0), B = n * S;
    const double w_fall = cfg["w_fall"].cast<double>(), w_cp = cfg["w_cp"].cast<double>(), w_up = cfg["w_up"].cast<double>(),
                 w_z = cfg["w_z"].cast<double>(), w_spin = cfg["w_spin"].cast<double>(), cp_tol = cfg["cp_tol"].cast<double>(),
                 up_min = cfg["up_min"].cast<double>(), z_min = cfg["z_min"].cast<double>(), spin_max = cfg["spin_max"].cast<double>();
    const bool fall = cfg["fall"].cast<int>() != 0;
    // keep_warm = 1: carry MuJoCo's solver warm start along each rollout, seeded from the world's (as the world itself steps);
    // keep_warm = 0: zero it at every control step, exactly as mujoco.rollout() does in fbw.py.
    const bool keep_warm = cfg.contains("keep_warm") && cfg["keep_warm"].cast<int>() != 0;
    const int nv = m_->nv;
    std::vector<double> warm(keep_warm ? B * (size_t)nv : 0);
    if (keep_warm) for (int b = 0; b < B; ++b) std::copy(warm0.data(), warm0.data() + nv, warm.begin() + b * (size_t)nv);
    const double* dTp = dT.data(); const double* Ep = E.data();
    std::vector<double> st(B * (size_t)ns_), X(B * (size_t)NOBS), Tp(B * (size_t)NJ), Ts(B * (size_t)NJ);
    for (int b = 0; b < B; ++b) {
      std::copy(s0.data(), s0.data() + ns_, st.begin() + b * (size_t)ns_);
      std::copy(x.data(), x.data() + NOBS, X.begin() + b * (size_t)NOBS);
    }
    arr parts({(py::ssize_t)B, (py::ssize_t)5}); std::fill(parts.mutable_data(), parts.mutable_data() + B * 5, 0.0);
    py::array_t<bool> fallen(B); std::fill(fallen.mutable_data(), fallen.mutable_data() + B, false);
    double* pp = parts.mutable_data(); bool* fp = fallen.mutable_data();
    std::vector<char> stop(B, 0);
    std::vector<int> live; std::vector<double> Xl;
    const int nq = m_->nq;
    {
      py::gil_scoped_release nogil;
      for (int k = 0; k < H; ++k) {
        // ---- pilot on every live rollout (k = 0: the pilot's command now)
        live.clear();
        for (int b = 0; b < B; ++b) if (!stop[b]) live.push_back(b);
        if (live.empty()) break;
        if (k == 0) {
          for (int b : live) std::copy(Tp0.data(), Tp0.data() + NJ, Tp.begin() + b * (size_t)NJ);
        } else {
          Xl.resize(live.size() * (size_t)NOBS);
          for (size_t i = 0; i < live.size(); ++i)
            std::copy(X.begin() + live[i] * (size_t)NOBS, X.begin() + (live[i] + 1) * (size_t)NOBS, Xl.begin() + i * NOBS);
          std::vector<double> out(live.size() * (size_t)NJ);
          Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Xm(Xl.data(), live.size(), NOBS);
          P_.run(Xm, out.data());
          for (size_t i = 0; i < live.size(); ++i)
            std::copy(out.begin() + i * NJ, out.begin() + (i + 1) * NJ, Tp.begin() + live[i] * (size_t)NJ);
        }
        // ---- physics + features + cost + next observation, in parallel
        const int L = (int)live.size();
#pragma omp parallel for num_threads(nthr_) schedule(dynamic, 1)
        for (int i = 0; i < L; ++i) {
          const int b = live[i], c = b / S, s = b % S;
          mjData* d = d_[omp_get_thread_num()];
          double* T = &Ts[b * (size_t)NJ];
          for (int j = 0; j < NJ; ++j) {
            double v = Tp[b * NJ + j] + dTp[((size_t)c * H + k) * NJ + j] + Ep[((size_t)s * H + k) * NJ + j];
            T[j] = std::min(std::max(v, lo_[j]), hi_[j]);
          }
          mj_setState(m_, d, &st[b * (size_t)ns_], spec_);
          if (keep_warm) mju_copy(d->qacc_warmstart, &warm[b * (size_t)nv], nv); else mju_zero(d->qacc_warmstart, nv);
          mju_zero(d->qfrc_applied, m_->nv); mju_zero(d->xfrc_applied, 6 * m_->nbody);
          mju_zero(d->ctrl, m_->nu);
          for (int j = 0; j < NJ; ++j) d->ctrl[act_[j]] = T[j];
          for (int t = 0; t < decim_; ++t) mj_step(m_, d);
          mj_getState(m_, d, &st[b * (size_t)ns_], spec_);
          if (keep_warm) mju_copy(&warm[b * (size_t)nv], d->qacc_warmstart, nv);
          // one kinematics pass on the new state
          mj_kinematics(m_, d); mj_comPos(m_, d); mj_comVel(m_, d); mj_subtreeVel(m_, d);
          const double* q = d->qpos; const double* v = d->qvel;
          // proprioception [64]
          double p[64];
          for (int j = 0; j < NJ; ++j) { p[j] = q[qad_[j]]; p[NJ + j] = v[dof_[j]]; }
          p[58] = v[3]; p[59] = v[4]; p[60] = v[5];
          const double* R = d->xmat + 9 * torso_;
          p[61] = -R[6]; p[62] = -R[7]; p[63] = -R[8];
          // envelope features
          const double* com = d->subtree_com + 3 * root_; const double* lv = d->subtree_linvel + 3 * root_;
          double P0[3], P1[3]; mju_copy3(P0, d->xpos + 3 * feet_[0]); mju_copy3(P1, d->xpos + 3 * feet_[1]);
          double fzmin = std::min(P0[2], P1[2]);
          bool g0 = P0[2] < fzmin + 0.03, g1 = P1[2] < fzmin + 0.03;
          double h = std::max(com[2] - fzmin, 0.3);
          double om = std::sqrt(9.81 / h);
          double cp[2] = {com[0] + lv[0] / om, com[1] + lv[1] / om};
          double dist;
          if (g0 != g1) {
            const double* G = g0 ? P0 : P1;
            dist = std::hypot(cp[0] - G[0], cp[1] - G[1]);
          } else {
            double ab[2] = {P1[0] - P0[0], P1[1] - P0[1]};
            double t = ((cp[0] - P0[0]) * ab[0] + (cp[1] - P0[1]) * ab[1]) / std::max(ab[0] * ab[0] + ab[1] * ab[1], 1e-9);
            t = std::min(std::max(t, 0.0), 1.0);
            dist = std::hypot(cp[0] - (P0[0] + t * ab[0]), cp[1] - (P0[1] + t * ab[1]));
          }
          double up = R[8], z = q[2], spin = std::sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
          double* pb = pp + 5 * (size_t)b;
          bool cont = true;
          if ((z < 0.45 || up < 0.5) && !fp[b]) {
            fp[b] = true;
            if (fall) { pb[0] += w_fall * (H - k); stop[b] = 1; cont = false; }
          }
          if (cont) {
            double a1 = std::max(dist - cp_tol, 0.0), a2 = std::max(up_min - up, 0.0), a3 = std::max(z_min - z, 0.0),
                   a4 = std::max(spin - spin_max, 0.0);
            pb[1] += w_cp * a1 * a1; pb[2] += w_up * a2 * a2; pb[3] += w_z * a3 * a3; pb[4] += w_spin * a4 * a4;
          }
          // next observation (priors.next_obs)
          double* Xb = &X[b * (size_t)NOBS];
          double Bk[4][64];
          for (int kk = 0; kk < 4; ++kk) std::copy(Xb + 64 + 64 * kk, Xb + 128 + 64 * kk, Bk[kk]);
          std::copy(p, p + 64, Xb);
          for (int j = 0; j < NJ; ++j) { double a = (T[j] - P_.off[j]) / P_.sc[j]; Xb[320 + j] = P_.off[j] + P_.sc[j] * a; }
          for (int e = 0; e < 64; ++e) {
            Xb[64 + e] = Bk[1][e]; Xb[128 + e] = 0.5 * (Bk[1][e] + Bk[2][e]);
            Xb[192 + e] = 0.75 * Bk[2][e] + 0.25 * Bk[3][e]; Xb[256 + e] = Bk[3][e];
          }
        }
      }
    }
    return py::make_tuple(parts, fallen);
  }

 private:
  mjModel* m_; std::vector<mjData*> d_; int spec_, ns_;
  std::vector<int> act_, qad_, dof_; int torso_, root_; std::vector<int> feet_; int decim_, nthr_;
  std::vector<double> lo_, hi_; Pilot P_;
};

PYBIND11_MODULE(fbw_kernel, mod) {
  py::class_<Kernel>(mod, "Kernel")
      .def(py::init<std::uintptr_t, std::vector<int>, std::vector<int>, std::vector<int>, int, int, std::vector<int>, int, int,
                    py::dict, arr, arr, double>())
      .def("score", &Kernel::score).def("pilot", &Kernel::pilot).def_property_readonly("nstate", &Kernel::nstate);
}
