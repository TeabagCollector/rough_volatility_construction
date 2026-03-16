# 粗糙波动率框架研究汇报（阶段版）


## 问题
1. motivation要更直接的指出
2. 过去时序解决了什么问题？已经达到了什么？
3. 我要解决什么？ why this matters？ what has been down？ what hasn't been done?  what's my contribution
4. 在探索一下分钟级别的数据获取（整理数据需求），沪深300也改为分钟级别做探索开始
5. sigma（过去信息） -> NN(过去信息)
6. to be more decent: 可解释性

## 数据需求整理
1. 标普500指数的分钟级别价格变动
2. 标普500股指期权的日价格变动
   - open：开盘价
   - close：收盘价
   - iv：隐含波动率
3. 美国十年期国债利率水平（用作r_f）


## 1. 研究动机与总体框架

我的核心动机是把衍生品定价中的股票/股指定价问题拆成两个相对独立但可衔接的模块：

1. **波动率持续预测（time-series forecasting）**  
   用历史实现波动率或其对数序列，预测未来若干日的波动率水平。
2. **波动率建模与定价（cross-sectional pricing）**  
   用随机波动率模型解释/拟合期权隐含波动率曲面，并完成截面定价。

传统预测线常见方法是 AR / HAR / GARCH。  
我在阅读 Gatheral 等的 *Volatility is Rough* 后，尝试引入 rough volatility 视角，将“波动率驱动”从经典 BM 平滑路径框架转向 **Hurst 小于 0.5 的 fBM/Volterra 驱动**，并在此基础上继续尝试 Bayer 等的 rough 定价框架。

这里的核心区别是：传统 BM 对应 \(H=0.5\)，其增量独立、无相关结构；而 fBM 由 \(H\) 控制增量相关与路径粗糙度，当 \(H<0.5\) 时增量负相关、路径更粗糙，能够更贴近高频数据中观测到的“短尺度剧烈起伏 + 均值回复”特征。

更具体地，\(H\) 可理解为“路径规则性与记忆强度参数”：\(H=0.5\) 表示近似随机游走、无增量相关；\(H<0.5\) 表示反持续（anti-persistent），冲击后更容易反向回撤，局部更锯齿；\(H>0.5\) 表示持续性（persistent），增量正相关、趋势延续更强。

---

## 2. 已完成的技术路线

当前项目已形成如下链路：

1. **高频波动率代理构建**：MUZ（Model with Uncertainty Zones）从高频价格构建 `RV^UZ`
2. **粗糙性估计**：对 `log(RV^UZ)` 做 variogram 回归估计 `H`
3. **RFSV 预测**：用 Gatheral 文章中的 RFSV 思路进行滚动预测
4. **基准对比**：AR(5), AR(10), HAR(3)
5. **定价扩展**：基于 rBergomi（Bayer et al.）做期权截面拟合（沪深300股指期权）

---

## 3. Gatheral (Volatility is Rough) 复现要点

### 3.1 关键数学关系（H 的估计）

对数波动率过程记为 \(X_t\)。  
使用结构函数（q 阶矩）：

\[
m_q(\Delta)=\mathbb{E}|X_{t+\Delta}-X_t|^q \propto \Delta^{qH}
\]

取对数后可做线性回归：

\[
\log m_q(\Delta)= qH\log \Delta + c
\]

故回归斜率 \(s\) 满足：

\[
H = s/q
\]

我当前实现中主要用 \(q=2\)，并在多尺度做鲁棒性检查。

### 3.2 RFSV 预测思路（实现层）

- 先由历史样本估计 `H`
- 由增量矩回归截距估计 `nu_sq`（vol-of-vol 规模）
- 对 `log variance` 做滚动预测（多个 horizon：\(\Delta=1,5,20\)）
- 用论文风格 `P-ratio` 评估：

\[
P = \frac{\sum (\hat y - y)^2}{\sum (y-\bar y)^2}, \quad P<1\ \text{越好}
\]

---

## 4. 实证结果（阶段性）

### 4.1 沪深300（RFSV vs AR/HAR）

来自 `Notebooks/2_RFSV_predictor.ipynb` 的一组结果：

**log-variance 预测 P 值（越小越好）**

| Δ | AR(5) | AR(10) | HAR(3) | RFSV |
|---|---:|---:|---:|---:|
| 1  | 0.506 | 0.505 | 0.502 | 0.604 |
| 5  | 0.801 | 0.794 | 0.783 | 0.880 |
| 20 | 1.005 | 1.022 | 1.017 | 1.354 |

**方差域 P 值（越小越好）**

| Δ | AR(5) | AR(10) | HAR(3) | RFSV |
|---|---:|---:|---:|---:|
| 1  | 0.601 | 0.609 | 0.601 | 0.753 |
| 5  | 0.858 | 0.851 | 0.844 | 1.108 |
| 20 | 0.893 | 0.911 | 0.905 | 1.128 |

结论：在沪深300样本中，当前配置下 RFSV 预测效果整体弱于 AR/HAR。

### 4.2 AEX（RFSV vs AR/HAR）

来自 `Notebooks/2_1_RFSV_predictor_AEX.ipynb`：

| Δ | AR(5) | AR(10) | HAR(3) | RFSV |
|---|---:|---:|---:|---:|
| 1  | 0.348 | 0.356 | 0.338 | 0.424 |
| 5  | 0.524 | 0.515 | 0.502 | 0.527 |
| 20 | 0.790 | 0.798 | 0.786 | 0.714 |

结论：AEX 中出现“**短期不占优、较长 horizon（Δ=20）开始占优**”的现象。  
这与我原先“rough 更偏短期”的直觉不完全一致，值得继续拆解其来源（市场结构、样本周期、参数稳定性、数据噪声、损失函数偏好等）。

---

## 5. Bayer (Pricing under rough volatility) 的承接与当前进展

### 5.1 我对 Bayer 思路的理解（用于当前复刻）

在 rough Bergomi/rBergomi 框架中，核心结构可写作：

\[
\frac{dS_t}{S_t}=\sqrt{V_t}\,dW_t
\]

\[
V_t=\xi_0(t)\exp\!\left(\eta W_t^H-\frac{1}{2}\eta^2 t^{2H}\right)
\]
该指数结构可理解为“在初始前向方差曲线 \(\xi_0(t)\) 上施加一个对数正态随机乘子”：\(\eta W_t^H\) 负责注入粗糙随机波动与波动簇集，\(-\frac12\eta^2 t^{2H}\) 是方差校正项（保证指数项均值被归一化），从而既保持 \(V_t>0\) 又控制方差尺度。

\[
W_t^H=\sqrt{2H}\int_0^t (t-s)^{H-\frac12}dB_s
\]

参数上重点是 \(H,\eta,\rho\)，并通过 Monte Carlo 在非 Markov 情况下完成定价与校准。

与传统 Bergomi 的关键区别可概括为：

1. **波动率驱动核不同（数学上是 Markov 核 vs 分数 Volterra 核）**：传统 Bergomi 常写成有限维 Markov 因子系统（典型是 OU 因子线性组合，状态满足 Itô SDE，给定当前状态即可描述未来分布）；rBergomi 则引入
\[
W_t^H=\sqrt{2H}\int_0^t (t-s)^{H-\frac12}\,dB_s,
\]
其核 \(K(t,s)=(t-s)^{H-1/2}\) 具有幂律奇异性与历史依赖，导致方差过程是 Volterra 型、一般不具有限维 Markov 表示，这也是 rough 路径与短尺度粗糙性的来源。  


传统 Bergomi（可理解为 forward variance model 的 Markov 版本）通常在风险中性测度下写成：
\[
\frac{dS_t}{S_t}=\sqrt{V_t}\,dW_t^S,\quad V_t=\xi_t^t
\]
其中 \(\xi_t^u\) 表示时刻 \(t\) 对未来 \(u\) 的前向方差。  
在一因子 Bergomi 中，常见设定是前向方差曲线满足指数核驱动：
\[
\frac{d\xi_t^u}{\xi_t^u}=\omega e^{-\kappa(u-t)}\,dW_t^\xi,\quad u\ge t,
\]
\[
d\langle W^S,W^\xi\rangle_t=\rho\,dt.
\]
这意味着模型由有限维 Markov 因子刻画（核心参数如 \(\kappa,\omega,\rho\)，多因子时再增加衰减尺度与权重），可通过指数核描述期限结构影响。
rBergomi 的关键改动是：将传统指数核（短记忆、有限尺度）替换为分数幂律核
\[
K(t,s)=(t-s)^{H-\frac12},
\]
从而得到
\[
W_t^H=\sqrt{2H}\int_0^t (t-s)^{H-\frac12}dB_s.
\]
当 \(H<0.5\) 时路径更粗糙，能更好匹配“短尺度高不规则性”的经验事实。  
因此，从传统 Bergomi 到 rBergomi 的本质是：**从有限维 Markov 指数核驱动，转向分数 Volterra 幂律核驱动**。

2. **状态维度与可约化程度不同**：传统 Bergomi 在实践中常需多因子扩展以同时拟合期限结构与 smile 形状；rBergomi 虽然非 Markov，但结构更“简约”。  
3. **参数维度更低（在给定 \(\xi_0(t)\) 情况下）**：rBergomi 通常只需重点估计 \(H,\eta,\rho\) 三个核心参数（\(\xi_0(t)\) 视为市场可观测输入），这也是 Bayer 论文强调其相对 parsimonious 的原因。

对应到我当前项目，参数估计流程为：

1. **历史先验阶段（P 侧）**：用 `log(RV^UZ)` 的 variogram 回归估计 \(H\)，并由增量矩回归截距估计 `nu_sq`，取 \(\eta_0\approx\sqrt{\nu^2}\) 作为初值。  
2. **前向方差输入（Q 侧近似）**：当前先用平坦 \(\xi_0\) 近似，即把期限结构函数 \(\xi_0(t)\) 退化为常数 \(\bar{\xi}\)（可理解为“仅乘一个常数尺度项”），于是
\[
V_t=\bar{\xi}\cdot\exp\!\left(\eta W_t^H-\frac12\eta^2 t^{2H}\right).
\]
这相当于忽略了不同到期上的初始方差曲线斜率/曲率信息，只保留一个整体方差水平；后续将替换为由市场数据构造的 \(\xi_0(t)\) 以恢复期限结构。  
3. **期权截面校准（Q 侧）**：固定或放开 \(H\)，对 \((\eta,\rho)\) 或 \((H,\eta,\rho)\) 做加权 IV 误差最小化（含可选 skew 惩罚）。  
4. **稳定性与口径校验**：并行比较 `api_iv（米筐）`、`market_iv（本地反解）`、`model_iv（rBergomi）`，检查可解率、ATM term 误差与 skew 误差。

### 5.2 当前 notebook 复刻（沪深300股指期权，2025-01-10）

已完成：

- rBergomi 路径模拟 + MC 定价 + 本地 IV 反解
- 参数比较：`fix_H` vs `free_H`
- 三方 IV 对比链路：`api_iv（米筐）` / `market_iv（本地反解）` / `model_iv（rBergomi）`

一次运行的摘要（`Notebooks/3_rBergomi.ipynb`）：

- `best_option_scenario = free_H_eta_from_nu`
- `best_option_iv_rmse = 0.180685`
- `best_option_atm_term_error = 0.179495`
- `best_option_skew_error = 0.887668`

并行给出的时间序列侧基准（同 notebook 摘要）：

- `avg_P_RFSV = 0.860562`
- `avg_P_AR5 = 0.736236`
- `avg_P_AR10 = 0.739719`
- `avg_P_HAR3 = 0.735032`

当前问题：

- `model_iv` 有较高缺失（一次诊断约 `48.5%` 缺失），导致 smile 对比可视化与 IV 误差稳定性受影响
- 已接入米筐 `options.get_greeks` 的 `api_iv`，用于和本地反解口径交叉校验

---

## 6. 阶段性结论

1. 我已把“预测”和“定价”两块在同一 rough 框架下打通，并能在 notebook 中做联合对比。  
2. Gatheral 线复现已完成，且跨市场对比出现了有研究价值的差异性（沪深300 vs AEX）。  
3. Bayer 线已完成工程化第一版（rBergomi + 校准 + 可视化），但当前效果和稳定性尚不理想，主要瓶颈是 IV 反解有效率与口径一致性。  
4. 从研究推进角度，下一阶段重点应从“能跑通”转向“口径统一 + 误差归因 + 稳定校准”。

---

## 7. 下一步计划

1. **IV 口径统一**：将 `api_iv` 作为主对照，系统比较 `api_iv` 与本地反解差异（按 call/put、到期、moneyness 分解）。  
2. **提升 model_iv 可解率**：改进反解区间与异常价格处理，降低 NaN 比例。  
3. **前向方差曲线升级**：从平坦 `xi0` 升级到按到期插值/拟合的 `xi0(t)`。  
4. **参数稳定性检验**：多初值、多随机种子、多交易日重复校准。  
5. **解释 AEX 长 horizon 优势**：围绕样本长度、市场微观结构、损失函数敏感性做因子化诊断。

---

## 8. 备注

本文件用于阶段汇报，强调“研究逻辑与实证现象”。  
其中 Bayer 相关数学与实验解释是基于当前实现和有限阅读形成的工作理解，后续会随着全文精读继续修订。

