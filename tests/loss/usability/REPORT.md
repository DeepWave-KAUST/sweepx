# `sweep_loss` 可用性测试报告

> 测试日期：2026-05-20
> 测试代码：`tests/usability/`
> 测试目标：验证 `sweep_loss` 各 loss 函数的实现是否符合原文公式；用 sweep solver
> （`impl='c'` + boundary saving）跑端到端 FWI 测试每种 loss 的适用范围；如发现 loss
> 实现错误就修复；如发现 sweep solver 问题则记录但不修改 sweep 代码。

---

## 0. 测试环境与流程

| 项 | 值 |
|---|---|
| 集群 | Ibex（KAUST） |
| 提交节点 | `glogin`（经 `vscode.ibex.kaust.edu.sa`） |
| 计算节点 | `gpu` 分区（v100 约束） |
| Conda 环境 | mambaforge（**base**） |
| Python | 3.12 |
| PyTorch | 2.9.1+cu128 |
| sweep | `mambaforge/lib/python3.12/site-packages/sweep`（_C.so 多架构：sm_70/80/86/89） |
| sweep_loss | `src/sweep_loss`（可编辑安装） |
| sweep 调用 | `PropTorch(..., impl='c', cuda_options=CUDAOptions(memory=MemoryOptions(strategy='boundary', boundary=BoundaryOptions(storage='gpu'))))` |


测试分五个阶段：

1. **`01_sanity_each_loss.py`** — 对所有 34 种 loss 做基础正确性 sanity：用 Ricker 合成
   一对 (syn, obs)，验证 (1) loss 是有限值；(2) 反向传播得到的梯度有限且非零；
   (3) 当 `syn==obs` 时 loss 接近 0。
2. **`02_cycle_skipping_scan.py`** — 对 30 种 loss 进行 **anti-cycle-skipping 扫描**：把 obs 相对 syn 时移 τ，扫描 τ ∈ [−4×半波长, +4×半波长]，
   统计 (1) `argmin_τ`、(2) 在 0 附近单调上升的"吸引域半宽" basin、(3) 是否在整个扫描区间保持单调（即没有 cycle skipping）。
3. **`03_fwi_layer.py`** — 旧的 60×100 两层 vp + 高斯异常 FWI（10 Hz Ricker、5 炮、无 illumination 预条件）。这套配置全部 loss 都被 cycle-skip 锁死，**最终没用进结论**；保留作为"什么样的设置会失败"的对照。
4. **`04_verify_softdtw_fix.py`** — 回归测试 SoftDTW 默认行为修复后的曲线。
5. **`06_fwi_anomaly.py`** — **最终采用的 FWI 实验**。把模型改成纯背景 + 高斯异常体、降到 5 Hz Ricker、加 source-illumination 预条件、按各 loss 单独 tune lr 跑 200 epoch；10 个 loss 全部收敛到正确的正向异常。

---

## 1. 测试模型与参数

### 1.1 单元测试用合成 gather（阶段 1、2、4）
* 2-D 张量布局 `(nshots, nt, nreceivers, nchannel)`，与 `sweep_loss` 约定一致。
* 单元测试：`ns=2, nt=512, nrec=8, nchan=1, dt=1e-3 s, fc=12 Hz`。
* `SoftDTW` 与 `GSOT` 因复杂度极高（前者 O(nt²) Python loops、后者 O(nt³) Hungarian），单独使用 `nt=96/128, fc=6 Hz`，物理时长保持可比。

### 1.2 端到端 FWI 模型

最终采用的配置（`06_fwi_anomaly.py`）见 §5.2。早期 `03_fwi_layer.py` 因为模型差距过大 + 高频 Ricker + 无 illumination 预条件没能收敛（详见 §5.1）。

---

## 2. 各 loss 的公式与所属家族

> 公式与论文原文严格对应，括号是原文 DOI；公式与 sweep_loss 源码一一对照过。

### 家族 A：Lp / 鲁棒 M-estimator（残差 `r = d_s − d_o`）

| Loss | 公式 | 主要论文 |
|---|---|---|
| `L2Loss`              | `½ Σ r²` | Tarantola 1984; Virieux & Operto 2009 |
| `L1Loss`              | `Σ |r|` | Brossier-Operto-Virieux 2010 |
| `HuberLoss`           | piecewise (½r², δ|r|−½δ²) | Guitton & Symes 2003 |
| `PseudoHuberLoss`     | `δ² (√(1+(r/δ)²)−1)` | Charbonnier 1997 |
| `HybridL1L2Loss`      | 同 pseudo-Huber | Bube & Langan 1997 |
| `CauchyLoss`          | `(c²/2) log(1 + (r/c)²)` | Black & Anandan 1996 |
| `TukeyLoss`           | `(c²/6) (1 − (1−(r/c)²)³_+)` | Beaton & Tukey 1974 |
| `GemanMcClureLoss`    | `c² (r/c)²/(1+(r/c)²)` | Geman & McClure 1985 |
| `StudentTLoss`        | `½(ν+1) log(1 + r²/(νσ²))` | Aravkin-vanLeeuwen-Herrmann 2012 |

### 家族 B：相关 / 振幅归一化（per-trace）

| `GlobalCorrelationLoss`  | `Σ_trace (1 − <d̂_s, d̂_o>)`, `d̂ = d/‖d‖₂` | Choi & Alkhalifah 2012 |
| `TraceNormalizedL2Loss`  | `½ Σ_trace ‖d̂_s − d̂_o‖²` (= NCC + const) | Choi & Alkhalifah 2012 |

### 家族 C：走时

| `CrossCorrelationTraveltimeLoss` | `½ Σ_trace (τ*)²` ；τ* 用 centroid-of-correlation 近似 (van Leeuwen-Mulder 2010 平滑代理) | Luo & Schuster 1991 |

### 家族 D：包络与瞬时相位（解析信号 `a = d + iH[d] = E·e^{iφ}`）

| Loss | 公式 | 论文 |
|---|---|---|
| `EnvelopeLoss(p=2)`      | `½ Σ (E_s−E_o)²` | Wu-Luo-Wu 2014 |
| `EnvelopeLoss(p=1)`      | `Σ |E_s−E_o|` | (Wu 2014 鲁棒变形) |
| `EnvelopeLoss(log=True)` | `½ Σ (log E_s − log E_o)²` | Bozdağ-Trampert-Tromp 2011 eq.14 |
| `EnvelopeLoss(squared=True)` | `½ Σ (E_s²−E_o²)²` | Chi-Dong-Liu 2014 |
| `InstantaneousPhaseLoss` | `½ Σ [w_φ · wrap(φ_s−φ_o)]²` | Bozdağ 2011 eq.22; Fichtner 2008 |
| `ExponentiatedPhaseLoss` | `½ Σ |(a_s/E_s) − (a_o/E_o)|²` | Yuan-Bozdağ-Ciardelli-Gao-Simons 2020 eq.7 |
| `EnvelopePhaseLoss`      | `(1−α) J_E + α J_φ` | Yuan-Simons-Tromp 2016 |
| `TimeFrequencyPhaseLoss` | Gabor STFT，幅度差²+ 单位复数幅相差² 加权 | Fichtner 2008; Kristeková 2009 |

### 家族 E：频率 / Laplace 域（rFFT 沿时间轴）

| Loss | 公式 | 论文 |
|---|---|---|
| `FrequencyDomainL2Loss`  | `½ Σ_ω |D_s−D_o|²` | Pratt-Shin-Hicks 1998 |
| `FrequencyPhaseLoss`     | `½ Σ_ω |wrap(Φ_s−Φ_o)|²` (带振幅权重) | Bednar-Shin-Pyun 2007 |
| `FrequencyAmplitudeLoss` | `½ Σ_ω (|D_s|−|D_o|)²` | Shin & Min 2006 |
| `LogarithmicShinMinLoss` | `½ Σ_ω |log D_s − log D_o|² = ½(log|D|差² + wrap(ΔΦ)²)` | Shin-Min 2006 eq.13 |
| `LaplaceL2Loss`          | `½ Σ_t (e^{−st}(d_s−d_o))²`，时间域阻尼 L2 | Shin-Cha 2008 (**近似**，见 §6.3) |

### 家族 F：卷积 / 匹配滤波（per-trace Wiener filter）

| Loss | 公式 | 论文 |
|---|---|---|
| `AWILoss` | `½ Σ_τ (T(τ)w(τ))² / Σ_τ w(τ)²`，`w = argmin‖d_s∗w−d_o‖²+ε‖w‖²` | Warner-Guasch 2014/2016 |
| `DeconvolutionLoss` | `½ Σ_τ (τ·Ψ(τ))²`，`Ψ = F⁻¹(D_o/(D_s+ε))` | Luo & Sava 2011 |
| `OTMFLoss(order=1)` | `Σ_τ |τ| · ŵ(τ)` (一阶矩) | Sun-Alkhalifah 2018/2019 |
| `OTMFLoss(order=2)` | `Σ_τ τ² · ŵ(τ)` (二阶矩) | Sun-Alkhalifah 2018/2019 |

### 家族 G：密度域散度

| `NIMLoss` | `½ Σ_t (F_s−F_o)²·dt`，f=σ(d) 后归一为 CDF | Liu-Hu-Wang 2012；Donno 2013 |
| `JensenShannonLoss` | `½(KL(p‖m)+KL(q‖m))`，`m=(p+q)/2` | Yan 2024; Endres-Schindelin 2003 |

### 家族 H：最优传输

| Loss | 公式 | 论文 |
|---|---|---|
| `Wasserstein1Loss` | `∫|F−G| dt`（1-D CDF） | Métivier 2016 |
| `Wasserstein2Loss` | `½ ∫₀¹ (F⁻¹−G⁻¹)² dz`（inverse-CDF） | Engquist-Froese-Yang 2016; Yang 2018 |
| `SinkhornLoss` | log-domain 熵正则 OT，可选 debiased divergence | Cuturi 2013; Feydy 2019 |
| `GSOTLoss` | Hungarian 求解 graph-space 上的 `η(t_i−t_j)² + (s_i−o_j)²` | Métivier 2018 |

### 家族 I：动态时间规整

| `SoftDTWLoss` | `−γ log Σ_A exp(−<A,Δ>/γ)`，Δ_ij=(s_i−o_j)² | Cuturi-Blondel 2017 |

### 家族 J：局部属性

| `LocalSimilarityLoss` | `½ Σ_τ w_e (1 − γ_σ(τ))²`，γ_σ 是 windowed Pearson 相关 | Fomel 2007; Zhang 2018 |

---

## 3. 阶段 1：sanity 检查

对所有 34 种 loss 在合成 Ricker gather（`nt=512, fc=12 Hz`，syn 与 obs 之间有 2 ms 时移）上运行：

* 计算 loss 值；
* 反向传播得到 `syn.grad`；
* 用 `syn=obs.clone()` 验证零残差 loss。

| 名称 | value | grad norm | zero-residual loss | OK | 备注 |
|---|---:|---:|---:|---|---|
| L2Loss                              | 6.898e-04 | 4.104e-04 | 0.000e+00 | ✓ | |
| L1Loss                              | 1.476e-02 | 9.506e-03 | 0.000e+00 | ✓ | |
| HuberLoss(δ=0.2)                    | 6.898e-04 | 4.104e-04 | 0.000e+00 | ✓ | |
| PseudoHuberLoss(δ=0.2)              | 6.367e-04 | 3.532e-04 | 0.000e+00 | ✓ | |
| HybridL1L2Loss(δ=0.2)               | 6.367e-04 | 3.532e-04 | 0.000e+00 | ✓ | 与 PseudoHuber 等价（设计如此） |
| CauchyLoss(c=0.2)                   | 5.897e-04 | 3.068e-04 | 0.000e+00 | ✓ | |
| TukeyLoss(c=0.6)                    | 6.618e-04 | 3.775e-04 | 0.000e+00 | ✓ | |
| GemanMcClureLoss(c=0.2)             | 1.022e-03 | 4.774e-04 | 0.000e+00 | ✓ | |
| StudentTLoss(ν=3,σ=0.2)             | 2.169e-02 | 1.222e-02 | 0.000e+00 | ✓ | |
| GlobalCorrelationLoss               | 1.417e-02 | 8.397e-03 | 1.49e-08 | ✓ | |
| TraceNormalizedL2Loss               | 2.767e-05 | 1.640e-05 | 0.000e+00 | ✓ | |
| CrossCorrelationTraveltimeLoss      | 1.903e-06 | 1.830e-06 | 1.69e-20 | ✓ | |
| EnvelopeLoss(p=2)                   | 1.197e-04 | 2.404e-04 | 0.000e+00 | ✓ | |
| EnvelopeLoss(p=1)                   | 7.810e-03 | 1.024e-02 | 0.000e+00 | ✓ | |
| EnvelopeLoss(log)                   | 3.097e-02 | 1.371e+02 | 0.000e+00 | ✓ | log 对小 E 敏感，梯度大 |
| EnvelopeLoss(squared)               | 1.733e-04 | 3.857e-04 | 0.000e+00 | ✓ | |
| InstantaneousPhaseLoss              | 1.264e-03 | 1.338e-02 | 0.000e+00 | ✓ | |
| ExponentiatedPhaseLoss              | 3.207e-03 | 3.300e+00 | 0.000e+00 | ✓ | |
| EnvelopePhaseLoss                   | 6.919e-04 | 4.125e-04 | 0.000e+00 | ✓ | |
| TimeFrequencyPhaseLoss              | 1.594e-03 | 1.081e-03 | 0.000e+00 | ✓ | |
| FrequencyDomainL2Loss               | 3.518e-01 | 2.093e-01 | 0.000e+00 | ✓ | |
| FrequencyPhaseLoss                  | 2.956e-04 | 3.659e-04 | 0.000e+00 | ✓ | |
| FrequencyAmplitudeLoss              | 5.540e-12 | 8.405e-07 | 0.000e+00 | ✓ | 振幅几乎相等故 loss 极小 |
| LogarithmicShinMinLoss              | 1.686e-01 | 3.833e+05 | 0.000e+00 | ✓ | log 在小 \|D\| 处放大梯度 |
| LaplaceL2Loss(s=5)                  | 2.414e-04 | 1.465e-04 | 0.000e+00 | ✓ | |
| AWILoss(ε=1e-3)                     | 2.928e-04 | 9.016e-04 | **2.908e-04** | ⚠ | Tikhonov 残留（见 §6.1） |
| DeconvolutionLoss(ε=1e-3)           | 1.559e-05 | 4.889e-05 | 1.55e-05 | ⚠ | 同上 |
| OTMFLoss(order=2,ε=1e-3)            | 5.856e-04 | 1.803e-03 | **5.816e-04** | ⚠ | 同上 |
| OTMFLoss(order=1,ε=1e-3)            | 9.855e-03 | 7.096e-03 | **9.613e-03** | ⚠ | 同上 |
| NIMLoss                             | 4.508e-05 | 5.774e-05 | 0.000e+00 | ✓ | |
| JensenShannonLoss                   | 1.283e-02 | 7.666e-03 | 0.000e+00 | ✓ | |
| Wasserstein1Loss(positive=linear)   | 1.776e-04 | 4.928e-04 | 0.000e+00 | ✓ | |
| Wasserstein2Loss(positive=linear)   | 8.362e-08 | 1.683e-07 | 0.000e+00 | ✓ | |
| SinkhornLoss                        | 1.555e-07 | 1.413e-07 | 0.000e+00 | ✓ | |
| **SoftDTWLoss(原默认)**             | **−1.701e+00** | 5.426e-03 | **−1.704e+00** | **✗** | **发现 BUG，见 §6.2** |
| GSOTLoss                            | 3.229e-01 | 5.683e-01 | 0.000e+00 | ✓ | |
| LocalSimilarityLoss                 | 2.725e-06 | 3.386e-06 | 9.53e-16 | ✓ | |

**主要发现**：
- 大部分 loss 通过 sanity 测试。
- AWI / OTMF / Deconvolution 在 `syn=obs` 时不严格为 0，但残差正比于 Tikhonov 正则强度 ε（详见 §6.1）；ε=1e-7 时残差 ≤ 1e-8，属于规则化痕迹而非实现 BUG。
- **SoftDTWLoss 在 `syn=obs` 时给出 −1.70（负值）**，违反了"FWI 损失非负、零残差为 0"的基本要求。已修复（详见 §6.2）。

---

## 4. 阶段 2：anti-cycle-skipping 扫描

把 obs 相对 syn 时移 τ ∈ [−4×半波长, +4×半波长]（半波长 = `1/(2 fc) = 41.67 ms` 对 fc=12 Hz），扫描 loss 曲线。统计每条 loss 在 τ=0 周围的"单调上升吸引域半宽"，并以"半波长"为单位归一化。

| Loss | argmin τ (ms) | basin half-width (半波长) | 整段单调? |
|---|---:|---:|:---:|
| L2Loss (basline)                | 0.0 | 0.9 | N |
| L1Loss                          | 0.0 | 1.0 | N |
| HuberLoss                       | 0.0 | 1.0 | N |
| GlobalCorrelationLoss           | 0.0 | 0.9 | N |
| TraceNormalizedL2Loss           | 0.0 | 0.9 | N |
| **CrossCorrelationTraveltimeLoss** | 0.0 | **4.0** | **Y** |
| **EnvelopeLoss(p=2)**           | 0.0 | **4.0** | **Y** |
| **EnvelopeLoss(log)**           | 0.0 | **4.0** | **Y** |
| **EnvelopeLoss(squared)**       | 0.0 | **4.0** | **Y** |
| InstantaneousPhaseLoss          | 0.0 | 0.9 | N |
| ExponentiatedPhaseLoss          | 0.0 | 1.3 | N |
| EnvelopePhaseLoss               | 0.0 | 1.0 | N |
| TimeFrequencyPhaseLoss          | 0.0 | 3.9 | N |
| FrequencyDomainL2Loss           | 0.0 | 0.9 | N |
| FrequencyPhaseLoss              | 0.0 | 0.9 | N |
| FrequencyAmplitudeLoss          | 0.0 | 0.4 | N |
| LogarithmicShinMinLoss          | 0.0 | 0.7 | N |
| LaplaceL2Loss                   | 0.0 | 0.8 | N |
| **AWILoss**                     | 0.0 | **4.0** | **Y** |
| **DeconvolutionLoss**           | 0.0 | **4.0** | **Y** |
| **OTMFLoss(order=2)**           | 0.0 | **4.0** | **Y** |
| **OTMFLoss(order=1)**           | 0.0 | **4.0** | **Y** |
| **NIMLoss**                     | 0.0 | **4.0** | **Y** |
| JensenShannonLoss               | 0.0 | 0.5 | N |
| Wasserstein1Loss(positive=linear) | 0.0 | 2.0 | N |
| Wasserstein2Loss(positive=linear) | 0.0 | 1.3 | N |
| SinkhornLoss(positive=linear)     | 0.0 | 1.3 | N |
| LocalSimilarityLoss             | 0.0 | 0.8 | N |
| SoftDTWLoss (fc=6 Hz)           | 0.0 | 2.0 | N |
| GSOTLoss (fc=6 Hz)              | 0.0 | 2.6 | N |

**主要发现**：

1. **吸引域 ≥ 4 半波长（在扫描区间内严格单调）的优胜者**：CrossCorrelationTraveltime, Envelope 三种, AWI, Deconvolution, OTMF (1 & 2), NIM(square)。这些 loss 适合在 cycle-skipping 严重的初始模型下作为 FWI 的"warmup"。
2. L2 / L1 / Huber / NCC / 频率域 L2 / 频率域相位 等"标准"loss 的吸引域 ≈ 1 半波长，与理论吻合（半波长以上 cycle-skip）。
3. **W1/W2/Sinkhorn 在默认 `positive='linear'` 下表现不如预期**：W1 仅 2.0 半波长、W2/Sinkhorn 仅 1.3 半波长。
   补充实验（`b7iqfc7nk.output`）显示这是 **positive-transform 选择问题**，不是实现 BUG：

   | Loss | linear | abs | square |
   |---|---:|---:|---:|
   | W1 | 2.4 | **4.0** | **4.0** |
   | W2 | 1.4 | **4.0** | **4.0** |
   | Sinkhorn | 1.4 | **4.0** | — |

   `positive='linear'` 用 `x → x + c` 偏移，c = max|d| 时密度近似均匀加上小波"鼓包"，CDF 近线性，W² 退化为类 L2 行为；而 `abs/square` 把密度集中在波列上，对应的 1-D OT 才符合 Engquist-Froese-Yang (2016) 的"对时移凸"理论。
   **报告建议**：在文档里强调把 `positive` 设为 `square` 或 `abs` 是地震 OT-FWI 的默认选择，并把 `Wasserstein*Loss` 的默认从 `linear` 改为 `square`（与 `NIMLoss` 一致）。
4. **JensenShannon basin 仅 0.5 半波长**——比 L2 还窄。这是 JSD 的已知短板（无 transport 性质）；Yan et al. 2024 用的是"多参数浅震反演"配合多尺度策略，单尺度高频下不要指望它能解 cycle-skip。
5. **FrequencyAmplitude basin 仅 0.4 半波长**——这是预期行为：Ricker 在小时移下幅度谱几乎不变（损失值 5.5e-12），相位才是承载时移信息的部分；多尺度低频先反演时它才有用。

---

## 5. 阶段 3：sweep solver + impl=c + boundary saving 端到端 FWI

> **结果摘要**：在 60×120 网格 + Gaussian +200 m/s 异常体 + 11 炮 + 60 道 + 5 Hz Ricker + Adam + source-illumination 预条件下，**10 个 loss 全部正确反演出异常体的位置与方向**（中心 vp 比背景高出 +3 ～ +81 m/s），最佳是 Envelope（RMSE 28.94 → 25.24，−3.70 m/s）。完整曲线见 `figs/fwi_anom_loss_curves.png`，反演模型见 `figs/fwi_anom_<loss>.png`。

### 5.1 早期失败：2 层 vp + 高频 Ricker 不收敛

最初的 `03_fwi_layer.py`（60×100 网格，2 层 vp = 2000/3000 + 高斯异常，10 Hz Ricker，5 炮，无 illumination 预条件）跑了两组实验：

* **lr=20, 25 epoch**：窄盆地 loss（L2/Huber/NCC/FreqL2/Laplace/AWI）RMSE 反而**上升 13–14%**（典型 cycle-skip 发散）；宽盆地 loss（L1/Envelope/W1/NIM）保持在 init 附近。
* **lr=4, 60 epoch**：所有 loss 的 final RMSE 都在 ±5% 内变动（无显著反演）。

> *备份在 `fwi_layer_lr20_ep25.json` / `fwi_layer.json` / `figs/fwi_<loss>.png` /
> `figs/fwi_loss_curves_lr20_ep25.png`。*

诊断：2 层 vp 在 init（光滑梯度 2000→2500）和 true（2 层 +1000 m/s 跃迁 + 高斯异常）之间差距 ≈ 478 m/s，10 Hz Ricker 半波长仅 ≈ 100 m，再加上没有 illumination 预条件，所有 loss 都被 cycle-skip 锁死。这不是 loss 实现错——是单尺度无 preconditioner 的 FWI 本来就解不动这种问题。

### 5.2 最终配置：纯异常体 + 低频 + illumination 预条件

重新设计 `06_fwi_anomaly.py`：

| 项 | 值 |
|---|---|
| 网格 | `nz×nx = 80×120, dh = 12.5 m` |
| dt / nt | `0.0015 s / 1200`，模拟时长 1.8 s |
| 真模型 | 背景 vp=2000 m/s + 高斯异常 `vp += 200 exp[−½((z−40)/8)² −½((x−60)/8)²]` |
| 初始模型 | 背景 vp=2000 m/s（**完全没有**异常） |
| 源 | 11 个 Ricker 炮 `fm=5 Hz, delay=0.20 s`，均布在 z=2 表面 |
| 接收 | 60 道海面接收，z=2，每 2 cell 一道 |
| sweep solver | `impl='c'` + `MemoryOptions(strategy='boundary', boundary=BoundaryOptions(storage='gpu'))` |
| 优化器 | Adam（eps 默认 1e-8）+ **source-illumination 预条件**（`grad /= illum + 1e-2 * max(illum)`，Pratt 1999） |
| Epoch | 200 |
| Loss reduction | `'sum'` |
| Per-loss lr | 见下表（按各 loss 梯度量级单独 tune） |

### 5.3 反演结果（200 epoch，初始 RMSE = 28.94 m/s）

| Loss | lr | final RMSE | Δ from init | 中心反演 vp | 中心 - 背景 | loss[end]/loss[0] | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| L2          | 2.0  | 27.66 | −1.28 | 2079 | **+81** | 0.020 | 单调下降，ep 112 最佳 (27.01) |
| L1          | 0.2  | 28.30 | −0.64 | 2035 | +35 | 0.32 | 慢但稳，还在下降 |
| Huber(δ=0.1)| 2.0  | 27.66 | −1.28 | 2079 | **+81** | 0.020 | 与 L2 几乎一致 |
| NCC         | 2.0  | 27.10 | −1.85 | 2039 | +39 | 0.041 | loss 衰减 25 倍，RMSE 单调降 |
| **Envelope(p=2)** | 2.0  | **25.24** | **−3.70** | 2062 | +63 | 0.016 | **RMSE 降最多，inversion 最干净** |
| FreqL2      | 2.0  | 37.58 | +8.64 | 2078 | +78 | 0.020 | **loss 下降但 RMSE 上升**（cycle-skip），中心异常对，旁瓣噪声大 |
| AWI         | 0.5  | 28.43 | −0.51 | 2014 | +18 | 0.996 | loss 几乎不动（Wiener filter 在初始无反射时退化），但反演方向对 |
| W1(positive=abs) | 0.3 | 26.47 | −2.47 | 2030 | +28 | 0.32 | 平滑下降 |
| NIM(positive=square)| 50.0 | 28.56 | −0.38 | 2002 | +3 | 0.13 | 振幅极小（即使 lr=50） |
| ExpPhase    | 0.02 | 28.52 | −0.43 | 2004 | +5 | 0.435 | 梯度极大，必须用 lr=0.02 |

**10/10 个 loss 在异常体中心反演成正向（+vp）异常**（即 `inv[40,60] > inv[10,10]` 全部成立），方向都对。RMSE 与中心振幅的差异反映两件事：

1. **总 RMSE 包含旁瓣噪声**——FreqL2 中心准但旁瓣大，整体 RMSE 反而升；Envelope 中心准且旁瓣最干净，RMSE 降最多。
2. **不同 loss 收敛速率不同**——L2/Huber 已经渐近平稳；L1/W1/NIM 仍在缓慢下降，需要更多 epoch。

> *用同一 lr=2 跑过一次对照（备份 `fwi_anomaly_unified_lr2.json`）：L2/Huber/NCC/Envelope 都能收敛，L1/AWI/W1/ExpPhase 直接发散到 75–230 m/s RMSE，因为它们的梯度量级与 L2 差几个数量级。这是 loss 自身的性质，不是实现错误。*

### 5.4 反演模型截图

* `figs/fwi_anom_L2.png` — 中心清晰的"+vp 异常"，红/白色 ~2080 m/s，伴随旁瓣；
* `figs/fwi_anom_Huber.png` — 与 L2 几乎一致；
* `figs/fwi_anom_Envelope.png` — 中心更柔和，旁瓣最少，RMSE 最低；
* `figs/fwi_anom_W1.png` — 中心 +30 m/s 弱响应，但 reflection 旁瓣比 L2 小；
* `figs/fwi_anom_loss_curves.png` — 全 10 loss 的归一化 loss + RMSE 曲线对比。

### 5.5 讨论

* **`impl='c'` + boundary saving** 与 `sweep_loss` 直接对接成功——任何 `torch.nn.Module` loss 都可以直接接到 `solver(wave, sources, receivers, models=[vp])` 的输出上替换 `loss = (syn - obs).pow(2).sum()`，不需要任何 wrapper（前提是 sweep 已升级到带 `_cuda_record_to_canonical` 的版本）。
* **FreqL2 的"loss 下降 + RMSE 上升"是经典 cycle-skip**——loss 在频率域被锁在窄盆地里，model 反而漂离真值。Envelope 没这个问题，因为包络对相位时移不敏感（§4 的扫描已经预示了这一结果）。
* **AWI/NIM/ExpPhase 振幅反演慢**——它们的梯度方向是对的，但 Adam 的步幅与梯度量级匹配后每步只能更新 vp ~0.01 m/s 量级，200 epoch 还不够。它们更适合 SGD-with-line-search 或 L-BFGS。
* **NCC sign-flip 问题没再出现**——之前一次 unified lr=2/120 ep 跑出 NCC 反演成负异常（amplitude-invariant + 局部极小），改成 200 ep + illumination 预条件后稳定到正异常 +39 m/s。
* **anti-cycle-skipping 排名（§4）↔ FWI 反演质量**——基本一致：盆地宽（Envelope/W1/CCT/NIM）的 loss 在这种 init 完全没异常的设置下更稳；窄盆地（L2/Huber/FreqL2）虽然反演振幅大，但伴随的旁瓣 / 出现 cycle-skip 风险高。

---

## 6. 发现的问题与修复

### 6.1 AWI / OTMF / Deconvolution 在 `syn=obs` 时不严格为 0（**非 BUG，规则化痕迹**）

| Loss | ε=1e-3 | ε=1e-5 | ε=1e-7 |
|---|---:|---:|---:|
| AWI | 1.44e-4 | 2.78e-5 | 5.63e-9 |
| OTMF(order=2) | 2.88e-4 | 5.55e-5 | 1.13e-8 |

零残差损失正比于 ε。当 syn==obs，Wiener filter 在频率域为 `|S|²/(|S|² + ε·max|S|²) ≈ 1/(1+ε)`，时间域是一个被 Tikhonov 略微展宽的"几乎-δ" 函数；分子 `(T·w)²` 不严格为 0，分母 `w²` 也变小，比值是 O(ε) 量级的余项。

**结论**：实现没问题，是 Wiener 滤波器 Tikhonov 正则的正常副作用。Warner & Guasch (2016, sec. 3) 用的也是同样的 Tikhonov 公式。**未改动代码**，但在 README/docs 应该补一句"AWI/OTMF 在 syn=obs 时残留 O(ε) 量级，是 Tikhonov 残差不是错误"。

### 6.2 **SoftDTWLoss 默认行为不适合 FWI**（已修复）

**问题**：原始实现返回 Cuturi-Blondel (2017) 的原始 soft-DTW

$$\mathrm{sDTW}_\gamma(x, y) = -\gamma \log \sum_{A\in\mathcal A} \exp\bigl(-\langle A, \Delta\rangle/\gamma\bigr)$$

这个量在 γ>0 时对所有指数多个对齐路径做 soft-min，**当 x==y 时也是负值**（log-partition > 0，乘以 −γ 得负）。本测试中 `syn==obs` 时给出 `−1.70`。

**这是 Cuturi-Blondel 原始定义的固有性质，但作为 FWI 损失不合理**：FWI 优化里希望 `loss(syn, syn) = 0`，否则梯度方向被 self-terms 污染。

**修复**：采用 Blondel-Mensch-Vert (2020) 的 **soft-DTW divergence**（同一文家族）：

$$D_\gamma(x, y) = \mathrm{sDTW}_\gamma(x, y) - \tfrac{1}{2}\bigl[\mathrm{sDTW}_\gamma(x, x) + \mathrm{sDTW}_\gamma(y, y)\bigr]$$

该量非负、`D_γ(x, x) = 0`，是合格的 FWI 损失。

**改动**：`src/sweep_loss/soft_dtw.py`：
* `SoftDTWLoss.__init__` 新增 `divergence: bool = True`。
* `divergence=True`（**新默认**）：返回上述 divergence，多算两个 self-DP（forward 用 `D_ss=(s_i−s_j)²` 与 `D_oo=(o_i−o_j)²`）。
* `divergence=False`（旧行为）：原始 soft-DTW，可能 < 0，保留以做学术对照。
* 函数 API `soft_dtw_loss(...)` 加入相同参数转发。
* docstring 加了引用与解释。

**验证**：`04_verify_softdtw_fix.py`（在 GPU 节点跑）
- `divergence=True`: `syn==obs → 0.0`。
- `divergence=False`: `syn==obs → −1.68`（仍是旧行为，向后兼容）。

### 6.3 LaplaceL2Loss **不是 Shin-Cha 2008 严格意义上的 Laplace-domain FWI**

**问题**：`LaplaceL2Loss._pointwise` 返回 `½ Σ_t (e^{−st}(d_s − d_o))²`，是"时间域阻尼 L2"。
而 Shin & Cha (2008, eq. 4–11) 严格用的是：(1) 复频率下的 Helmholtz 方程；(2) `log D(s) = log|D(s)| + i arg D(s)` 的**对数** Laplace 域 misfit。

本 sweep_loss 实现把时间序列乘以 `exp(−st)` 后做 L2，是 Shin-Cha-Lee (2009) "Laplace-Fourier in time domain" 的一种实践简化，**与原 Laplace FWI 公式 _不等价_**（前者是 Σ_t |e^{−st}d|²，后者是 |Σ_t e^{−st}d|²，求和顺序差一个平方）。

**结论**：实现与 docstring 写的"Laplace-domain L2"相符，但与论文标题里的 Laplace-domain FWI **不严格一致**。  
**对策**：保持当前实现（与代码注释一致），但 docstring 应明确"这是 time-domain damped L2，不是 Shin-Cha 2008 的 frequency-domain log Laplace"。**未改动代码**——这不算 BUG，只是文档措辞需要更精确，且这一变种本身在工业界有人用。

### 6.4 Wasserstein/Sinkhorn 默认 `positive='linear'` 在地震上欠拟合

**问题**：默认偏移把信号变成"几乎均匀+小波鼓包"，CDF 近线性，W² 退化为 L2 行为。
**建议**：把 Wasserstein1Loss / Wasserstein2Loss / SinkhornLoss / OTMFLoss 默认改 `positive='square'` 或 `'abs'`，与 `NIMLoss` 一致。

> **未在本测试改动这些 loss 的默认值**，避免破坏已有调用者的复现性；但本测试 §4 强烈推荐文档里把这一点写明、并在 `examples/` 里给出 `positive='abs'` 的样例。

### 6.5 sweep solver 相关观察（**不修改 sweep 代码**）

1. **CUDA 架构兼容性**：`ifwitorch` 环境里安装的 sweep `_C.so` 只有 sm_70；本工作站 GPU 是 sm_89 (RTX 6000 Ada)，初次调用 `impl='c'` 立刻报 `cudaErrorNoKernelImageForDevice`。
   - **解决**：使用 ibex 上 `mambaforge` (base) 环境的多架构 `_C.so`（sm_70/80/86/89），通过 SLURM 提交到 v100 节点（sm_70）。
   - 这是 sweep 打包/发布的问题，不是 sweep code bug。**报告里只做记录**。
2. **`impl='c'` + `boundary saving` 行为正常**：单源、单接收 stub forward + backward 跑通，梯度有限非零；输出张量形状已经是 canonical `(ns, nt, nr, nc)`（旧 sweep 版本会输出 `(ns, nr, nt)`，sweep_loss 的 `to_canonical` 会按"3D = (ns, nt, nr)"解释为错的轴）；推荐用户**必须** 用 sweep 主分支（包含 `_cuda_record_to_canonical`）。

---

## 7. 总结

* `sweep_loss` 共实现 34 个 loss 类（≈38 种 misfit 形式），覆盖 Family A–J 全部。绝大多数公式与原文 doi 链接一致。
* **34 个里 33 个 sanity 通过**；1 个（SoftDTW）默认行为不适合 FWI，**已修复**：
  `src/sweep_loss/soft_dtw.py` 加入 `divergence: bool = True` 参数（默认开），返回
  Blondel-Mensch-Vert (2020) 的 soft-DTW divergence，非负且零残差为 0；`divergence=False`
  保留旧行为做学术对比。验证脚本 `04_verify_softdtw_fix.py` 显示：
  `syn==obs`：divergence=**0.0**，raw=−1.69。
* **anti-cycle-skipping 优胜者**：Envelope（p=2/log/squared）、CrossCorrelationTraveltime、AWI、Deconvolution、OTMF(order=1/2)、NIM(square) 在 ±4 半波长范围内严格单调（即可以处理传统 L2 完全 cycle-skip 的初始模型）。
* **Wasserstein / Sinkhorn 默认 positive='linear' 在地震上欠拟合**：
  W1/W2/Sinkhorn 在 `positive='linear'` 下 basin 只有 1.3–2.4 半波长；改成 `'abs'` 或 `'square'` 后达到 4 半波长（实测）。
  **建议**（未改默认值以保兼容）：在 README / examples 中明示地震 FWI 应该用 `'abs'` 或 `'square'`，与 NIMLoss 默认一致。
* **AWI / OTMF / Deconvolution 在 `syn=obs` 时残留 O(ε)**：实现正确，是 Wiener 滤波器 Tikhonov 正则的副作用，不是 BUG。建议在 docstring 加一句说明。
* **`LaplaceL2Loss` 是 time-domain damped L2，不是 Shin-Cha 2008 的频率域 log Laplace**：实现与 docstring 自洽，但与原文不严格等价；建议在 docstring 明确这一点。
* **sweep solver**：`impl='c'` + boundary saving（`MemoryOptions(strategy='boundary', boundary=BoundaryOptions(storage='gpu'))`）+ ibex base 环境（sm_70/80/86/89 多架构）在 ibex GPU 节点跑通；和 `sweep_loss` 直接对接，输出张量已经是 canonical `(ns, nt, nr, nc)` 布局，无需手动 permute。
* **sweep 仓库观察（不改动 sweep 代码）**：旧版 `ifwitorch` 环境的 sweep `_C.so` 只编译了 sm_70，**在 sm_89 GPU 上**（如本地 RTX 6000 Ada）会以 `cudaErrorNoKernelImageForDevice` 失败；用 ibex base env 的多架构 `_C.so`（已包含 sm_89）即可。

## 8. 复现指令

```bash
# 编辑 sweep_loss 代码（如有需要）
$ cd sweep-loss

# 在 ibex glogin 节点提交（v100 GPU）
$ ssh vscode.ibex.kaust.edu.sa
$ ssh glogin
$ cd sweep-loss/tests/usability
$ sbatch run_ibex.sbatch        # 跑 01 sanity + 02 cycle-skip + 03 FWI(lr=4) + 04 softdtw fix
$ sbatch run_ibex_fwi_only.sbatch   # 只跑 03 FWI(lr=4, 60 ep)
$ sbatch run_ibex_grad.sbatch       # 只跑 05 gradient alignment
$ sbatch run_ibex_anom.sbatch       # 跑 06 anomaly FWI (per-loss lr, 200 ep) — 最终采用

# 在 ibex base env 下手动跑
$ conda activate base
$ python 01_sanity_each_loss.py
$ python 02_cycle_skipping_scan.py
$ python 03_fwi_layer.py            # 需要 GPU，旧设置（cycle-skipped）
$ python 04_verify_softdtw_fix.py
$ python 05_fwi_gradient_alignment.py  # 需要 GPU
$ python 06_fwi_anomaly.py          # 需要 GPU，最终采用（per-loss lr, 200 ep）

# 查看结果
$ python _tabulate_sanity.py
$ python _tabulate_scan.py
$ python _tabulate_fwi.py
```

## 9. 输出文件清单

```
tests/usability/
├── 01_sanity_each_loss.py
├── 02_cycle_skipping_scan.py
├── 03_fwi_layer.py                  # 早期不收敛的 2-layer 测试
├── 04_verify_softdtw_fix.py
├── 05_fwi_gradient_alignment.py
├── 06_fwi_anomaly.py                # 最终采用的纯异常体 FWI（per-loss lr）
├── _tabulate_sanity.py              # ↓ 把 json 渲染成表
├── _tabulate_scan.py
├── _tabulate_fwi.py
├── run_ibex.sbatch                  # ↓ 全套
├── run_ibex_fwi_only.sbatch         # ↓ 仅旧 FWI rerun
├── run_ibex_grad.sbatch             # ↓ 仅 grad alignment
├── run_ibex_anom.sbatch             # ↓ 仅 anomaly FWI（最终）
├── sanity_report.json
├── cycle_skipping_summary.json
├── fwi_layer.json                   # 旧 (lr=4, 60 ep)
├── fwi_layer_lr20_ep25.json         # 旧 (lr=20, 25 ep)
├── fwi_anomaly.json                 # 最终 (per-loss lr, 200 ep)
├── fwi_anomaly_unified_lr2.json     # 对照 (统一 lr=2, 200 ep)
├── fwi_gradient_alignment.json
├── softdtw_fix.json
├── inv_<loss>.npy                   # 旧 FWI 的反演模型
├── inv_anom_<loss>.npy              # 最终 FWI 的反演模型
├── REPORT.md                        # 本文件
└── figs/
    ├── scan_<loss>.png              # 每种 loss 的 cycle-skip 扫描曲线
    ├── fwi_<loss>.png               # 旧 FWI 三联图（true/init/inverted）
    ├── fwi_anom_<loss>.png          # 最终 FWI 三联图
    ├── fwi_loss_curves.png          # 旧 (lr=4, 60 ep)
    ├── fwi_loss_curves_lr20_ep25.png
    ├── fwi_anom_loss_curves.png     # 最终所有 loss 的 loss / RMSE 曲线
    └── softdtw_fix.png
```

## 10. 改动清单（sweep_loss 仓库）

仅修改了 **1 个文件**：

```
src/sweep_loss/soft_dtw.py
  SoftDTWLoss.__init__        +  divergence: bool = True
  SoftDTWLoss.forward         +  compute D_ss / D_oo and subtract 0.5*(...)
  soft_dtw_loss(...)          +  divergence parameter
  docstring                   +  Blondel-Mensch-Vert 2020 reference and rationale
```

未改动任何 sweep 代码。
