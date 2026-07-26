---
title: "反向传播：按卡住的顺序讲一遍"
date: 2026-07-26 11:00:00 +0800
categories: [技术, 机器学习]
tags: [反向传播, 神经网络, 梯度下降, 学习笔记]
description: "教科书按数学的流向讲，人按困惑的顺序卡。这篇按后者的顺序重讲一遍，每节配一个能动手的交互件，外加一组通过梯度校验的真实数字。"
---

<style>
.bp-fx{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;text-align:center;padding:15px 10px;margin:1.4em 0;border:1px solid rgba(128,128,128,.28);border-radius:9px;overflow-x:auto;white-space:nowrap}
.bp-fx .f{display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;margin:0 3px}
.bp-fx .f .u{border-bottom:1px solid currentColor;padding:0 5px 1px}
.bp-fx .f .d{padding:1px 5px 0}
.bp-fx .c1{color:#0d7383}.bp-fx .c2{color:#9a5c00}.bp-fx .c3{color:#b4145a}
.bp-key{border-left:3px solid #b4145a;padding:2px 0 2px 16px;margin:1.5em 0}
.bp-q{border-left:3px solid rgba(128,128,128,.4);padding:2px 0 2px 16px;margin:1.4em 0;font-style:italic;opacity:.85}
.bp-q::before{content:"当时问的";font-style:normal;font-size:11px;letter-spacing:.14em;opacity:.6;display:block;margin-bottom:3px}
.bp-frame{width:100%;border:0;display:block;margin:1.5em 0}
.bp-cap{font-size:13px;opacity:.6;text-align:center;margin:-1em 0 1.8em}
.bp-pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;line-height:1.85;padding:16px;border:1px solid rgba(128,128,128,.28);border-radius:9px;overflow-x:auto;margin:1.4em 0}
@media (prefers-color-scheme:dark){html:not([data-bs-theme=light]) .bp-fx .c1{color:#4ec9dc}html:not([data-bs-theme=light]) .bp-fx .c2{color:#f0a830}html:not([data-bs-theme=light]) .bp-fx .c3{color:#e85d8a}}
html[data-bs-theme=dark] .bp-fx .c1{color:#4ec9dc}html[data-bs-theme=dark] .bp-fx .c2{color:#f0a830}html[data-bs-theme=dark] .bp-fx .c3{color:#e85d8a}
</style>

上个月我花了大半天，跟 Claude 把反向传播从头到尾问了一遍。不是第一次学——之前看过公式、抄过推导、甚至照着写过代码——但一直有种"每一步都能跟上、合起来说不清"的感觉。

这次不一样的地方在于，我没有按教科书的顺序走，而是卡在哪问哪。回头看聊天记录，卡住的十个地方几乎没有一个是教科书重点讲的。教科书按数学的流向组织：损失函数、链式法则、逐层递推。而人卡住的地方是另一套顺序：为什么导数能当调参依据、这个符号到底是向量还是矩阵、δ 到底是对谁求导、转置为什么会突然冒出来。

这篇按后一种顺序重写一遍。每节开头引一句我当时的原话，配一个能动手的交互件。文末有一组完整的数值演算，全部通过梯度校验。

先看一眼整体。点"前向"再点"反向"，数字会依次亮在导线上——这些数字后面每一节都会再出现：

<iframe class="bp-frame" src="{{ '/assets/backprop/network.html' | relative_url }}" height="430" loading="lazy" title="前向与反向的数值流动"></iframe>
<p class="bp-cap">青色 = 前向信号，琥珀色 = 反向梯度。全文配色统一遵守这个规则。</p>

---

## 01　为什么求个导数，就能当调参的依据

<div class="bp-q">我都不知道为啥求导数就能作为参数调整的依据……以及优化步长怎么定呢？</div>

这是最根本的一个问题，而且大部分教程直接跳过了——它们默认你已经接受"沿负梯度走"这件事。

先把神经网络全部忘掉。假设只有**一个**参数 `w`，损失是 `L(w) = (w − 3)²`，是个碗。你不知道碗底在哪（不知道答案是 3），只能站在某个位置往下摸。

导数 `L′(w) = 2(w − 3)` 回答的问题只有一个：**把 w 增大一丁点，L 是涨还是跌。**

- 站在 `w = 5`：`L′(5) = 4`，正数，意味着"w 越大 L 越大"，所以该往小走。
- 站在 `w = 1`：`L′(1) = −4`，负数，意味着"w 越大 L 越小"，所以该往大走。

不管在碗的哪一侧，往导数的反方向走一定是在往下走。这不是巧合——导数的定义本身就在回答"往正方向挪会涨还是跌"，取个反自然就是"跌"。

<div class="bp-fx">w<span class="c1">新</span> = w<span class="c1">旧</span> &nbsp;−&nbsp; <span class="c2">η</span> · <span class="f"><span class="u">∂L</span><span class="d">∂w</span></span></div>

减号就是"反方向"，η 决定走多远。把下面的滑块拖到 1.0 以上，能看到一步迈过头之后它怎么越跳越远：

<iframe class="bp-frame" src="{{ '/assets/backprop/lr.html' | relative_url }}" height="440" loading="lazy" title="学习率与梯度下降"></iframe>
<p class="bp-cap">粉色是切线，琥珀色箭头是这一步要挪的距离。</p>

<div class="bp-key"><strong>η 没有万能公式。</strong>太大会发散，太小则要迭代很久。实践中靠试（0.1 / 0.01 / 0.001 这类起点），或者用 Adam 这类自适应算法让步长自己动态调整。注意小球越靠近碗底、每步挪得越小——因为导数绝对值在变小，梯度下降会自动"减速"。</div>

神经网络里有成千上万个参数，做法完全一样：每个参数各自算一个"往哪边挪能让损失变小"，然后各自挪一小步。**反向传播存在的唯一意义，就是帮你把每个参数的那个数字算出来。**

---

## 02　W 到底是什么，是向量还是矩阵

<div class="bp-q">你没解释这里面的 W、x 是啥，也没解释他们的维度，是向量还是矩阵……</div>

这是我第二个卡住的地方，而且卡得很实在：符号写了一堆，但完全没有物理图像。

最关键的一个认知：**权重矩阵里的每一个数字，就是网络里的一根导线。**

下面左边点任意一根线，右边对应的格子会亮起来：

<iframe class="bp-frame" src="{{ '/assets/backprop/wire-matrix.html' | relative_url }}" height="340" loading="lazy" title="导线与权重矩阵的对应"></iframe>
<p class="bp-cap">上一层 3 个神经元、这一层 2 个，中间 6 根线，正好对应 2×3 矩阵的 6 个格子。</p>

读法固定下来就不会再乱：`W[i][j]` 是**从第 j 个输入神经元、通往第 i 个输出神经元**的那根线。**行号看"去哪"，列号看"从哪来"。**

维度也就跟着定死了。要让 `z = Wa + b` 的形状对得上：

| 量 | 形状 | 为什么 |
|---|---|---|
| a（上一层输出） | (n₀, 1) | n₀ 个神经元，一个一行 |
| W | (n₁, n₀) | n₁ 行 × n₀ 列，每根线一格 |
| z（这一层加权和） | (n₁, 1) | (n₁,n₀)×(n₀,1) 的结果 |
| b | (n₁, 1) | 每个神经元配一个 |

<div class="bp-key"><strong>权重矩阵形状 = (输出维度, 输入维度)。</strong>记住这一条，任何一层的维度都能当场推出来，不用背。</div>

---

## 03　δ 是对 z 求导，还是对 W 求导

<div class="bp-q">δ⁽²⁾ 和 δ⁽¹⁾ 是 L 对输入 z 的偏导吗，还是 L 对 W 的偏导啊，又混了……</div>

这个我在对话里混了两次。答案是前者：

<div class="bp-fx"><span class="c2">δ</span><sup>(l)</sup> &nbsp;≡&nbsp; <span class="f"><span class="u">∂L</span><span class="d">∂z<sup>(l)</sup></span></span></div>

最可靠的辨别方法不是背定义，是**数个数**——两者的形状根本不一样。用文末那个例子的真实数字：

| | 是什么 | 值 | 形状 | 几个数 |
|---|---|---|---|---|
| δ⁽¹⁾ | ∂L/∂z⁽¹⁾ | [−0.0192, 0.0063] | (2,1) 向量 | **2 个** |
| ∂L/∂W⁽¹⁾ | 对权重 | [[−0.0192, −0.0385], [0.0063, 0.0125]] | (2,2) 矩阵 | **4 个** |

这一层有 2 个神经元、4 根线，数量对不上，所以两者不可能是同一个东西：

- **δ 是"每个神经元一个数"**
- **∂L/∂W 是"每根线一个数"**

δ 先算出来，再和输入做外积**生成** ∂L/∂W。所以 δ 是中间量，梯度是最终产物。

### 为什么偏偏定义在 z 上

因为 z 这个位置信息最浓缩、复用价值最高。拿到一个 δ，能同时干三件事：

<div class="bp-pre">
                    δ⁽ˡ⁾ （= ∂L/∂z⁽ˡ⁾）
                  ╱   │   ╲
        ∂L/∂W⁽ˡ⁾   ∂L/∂b⁽ˡ⁾   δ⁽ˡ⁻¹⁾
        = δ·(a⁽ˡ⁻¹⁾)ᵀ   = δ      = (W⁽ˡ⁾)ᵀδ ⊙ σ′
</div>

如果直接把 ∂L/∂W 当中间量往前传就没法复用了——它已经和具体的输入纠缠在一起，信息被"摊开"到每根线上。

<div class="bp-key"><strong>附赠一个记忆锚点：∂L/∂b = δ，一模一样，不用做任何运算。</strong>因为 z = Wa + b，所以 ∂z/∂b = 1，乘 1 等于没乘。哪天又记混了就想想"δ 等于谁的梯度"——答案是偏置 b，而 b 是直接加在 z 上的，自然说明 δ 住在 z 那个位置。</div>

---

## 04　ŷ − y 为什么长这个形式

<div class="bp-q">这个没看懂，为啥对他求导是这个形式呢……</div>

答案全在损失函数的定义里，不是什么普遍规律。我们选的是 `L = ½(ŷ − y)²`。

关键一点：**y 是数据集里给定的固定数字**，训练时永不改变，所以求导时它当常数。

令 `u = ŷ − y`，那么 `L = ½u²`：

1. `dL/du = ½ × 2u = u`
2. `du/dŷ = 1 − 0 = 1`（y 是常数，导数为 0）
3. 链式相乘：`∂L/∂ŷ = u × 1 = ŷ − y`

结果之所以这么干净，是因为求导把平方降成一次方，而那个 `½` 恰好把冒出来的 2 抵消了。**`½` 就是专门为此塞进去的**——给损失乘个常数不改变最低点位置，等于顺手调了下学习率而已。

**ŷ − y 同时携带两个信息**：符号告诉你往哪边错了，绝对值告诉你错得有多离谱。这就是它能直接当梯度用的原因。

<div class="bp-key"><strong>换损失函数，这个形式就变。</strong>交叉熵 <code>L = −[y·logŷ + (1−y)·log(1−ŷ)]</code> 的导数是 <code>(ŷ−y) / [ŷ(1−ŷ)]</code>，长得完全不同。但分母那个 ŷ(1−ŷ) 恰好是 sigmoid 的导数——两者一乘直接约掉，δ 又变回干净的 ŷ−y。<strong>这才是分类任务几乎都用交叉熵的真正原因</strong>：不是什么"更符合概率直觉"的玄学，而是它数学上恰好治好了下一节要说的梯度消失。</div>

---

## 05　δ 里那个激活函数导数是怎么回事

<div class="bp-q">这个地方是怎么算的来着？好像是和激活函数有关？</div>

z 不是直接影响 L 的，中间隔着一道激活函数。所以 δ 是**两环相乘**：

<div class="bp-fx"><span class="c2">δ</span> = <span class="f"><span class="u">∂L</span><span class="d">∂z</span></span> = <span class="f"><span class="u">∂L</span><span class="d">∂ŷ</span></span> × <span class="f"><span class="u">∂ŷ</span><span class="d">∂z</span></span> = (ŷ − y) × <span class="c3">σ′(z)</span></div>

sigmoid 的导数有个漂亮的形式，这也是它当年"好用"的原因——直接复用前向算好的值，不用重算指数：

<div class="bp-fx"><span class="c3">σ′(z)</span> = σ(z)·(1 − σ(z)) = ŷ(1 − ŷ)</div>

拖动下面的滑块，注意 σ′ 的天花板：

<iframe class="bp-frame" src="{{ '/assets/backprop/sigmoid.html' | relative_url }}" height="400" loading="lazy" title="sigmoid 及其导数"></iframe>
<p class="bp-cap">青色是 σ，粉色是 σ′。两条曲线画在同一个纵轴上——σ′ 的矮，是真的矮。</p>

<div class="bp-key"><strong>σ′ 的最大值只有 0.25，而且 z 一往两端跑就趋近于 0。</strong>这意味着每经过一层 sigmoid，梯度至少被砍掉 3/4。层数一多，梯度乘着乘着就没了——这就是<strong>梯度消失</strong>，也是后来大家改用 ReLU 的直接原因（ReLU 在正半轴导数恒为 1，不缩放）。</div>

### 这个公式只适用于输出层

隐藏层的第一环不一样——它不直接连着 L，得从下一层绕回来：

| 层 | 第一环（责任从哪来） | 第二环（打折） |
|---|---|---|
| 输出层 | ŷ − y（直接来自损失函数） | × σ′(z) |
| 隐藏层 | (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾（从后一层传回） | × σ′(z) |

**第二环是一模一样的**，区别只在第一环。这也正是"反向传播"这个名字的由来：第一环的信息一层层从后往前倒着传，第二环则是每层自己本地就能算的。

---

## 06　∂L/∂W = δ·aᵀ 是怎么推出来的

<div class="bp-q">好像懂了，这一步怎么推导出来的啊……彻底看不懂了，重新来一遍</div>

这一步我卡了三次。问题在于"外积"这个运算如果没有图像，就是一堆下标。

单独看一根线：`W₁₂`，从 a₂ 出发、通往 z₁。先看 z₁ 怎么算出来的：

<div class="bp-fx">z₁ = W₁₁a₁ + <span class="c3">W₁₂a₂</span> + W₁₃a₃ + b₁</div>

只有中间那项含 W₁₂，其余全是常数。所以 **W₁₂ 每增加 1，z₁ 就增加 a₂ 倍**；而 z₁ 变化对 L 的影响倍数，正是 δ₁。两环一乘：

<div class="bp-fx"><span class="f"><span class="u">∂L</span><span class="d">∂W<sub>ij</sub></span></span> = <span class="c2">δ<sub>i</sub></span> × <span class="c1">a<sub>j</sub></span> &nbsp;&nbsp;<span style="opacity:.6">（误差 × 参与度）</span></div>

把这条规则对 6 根线各用一遍，填成表格：

<iframe class="bp-frame" src="{{ '/assets/backprop/outer-product.html' | relative_url }}" height="400" loading="lazy" title="外积逐格计算"></iframe>
<p class="bp-cap">左边一列 δ，上边一行 a，中间就是一张乘法表。</p>

看这张表的形状：**2 行 3 列，和 W 完全一样**。每个格子恰好告诉你对应那根线该怎么调。

这个"乘法表"的生成方式——一**列**数字 × 一**行**数字铺成一张表——就叫**外积**。而 a 本来是竖着的列向量，所以要先转置放倒当表头：

<div class="bp-fx"><span class="f"><span class="u">∂L</span><span class="d">∂W</span></span> = <span class="c2">δ</span> · (<span class="c1">a</span>)<sup>ᵀ</sup> &nbsp;&nbsp;<span style="opacity:.6">(n₂,1) × (1,n₁) = (n₂,n₁)</span></div>

<div class="bp-key"><strong>这里的转置纯粹是为了形状对得上</strong>——把 (n,1) 放倒成 (1,n)，好凑成外积。它和下一节那个转置<strong>不是一回事</strong>。</div>

### 顺带：a 每次训练都不一样，这不是问题吗

<div class="bp-q">a₂ 每次训练都不一样吧，他是输入端的东西，每次都不一样</div>

完全正确，而且这是设计如此。把 aⱼ 理解成"这根线这次到底传了多少信号"：

- **a₂ = 0**（没被激活）→ 梯度 = δ₁ × 0 = 0。这次没送任何东西，不该罚它，一动不动。
- **a₂ = 5.0**（强烈激活）→ 梯度 = δ₁ × 5.0。贡献大，错也主要怪它，重罚。

这就是所谓的**责任分配**（credit assignment）：δ 说"错了多少"，a 说"参与度多高"。a 每次不同恰恰是必须的——每次输入不同，每根线的参与度本来就该不同。

至于单样本梯度噪声大的问题，解决办法是拿一批样本各算各的、取平均再更新一次——这就是 SGD 里 "Stochastic" 的来源。矩阵形式下那个"对 m 求和"的动作，矩阵乘法会自动帮你完成：

<div class="bp-fx"><span class="f"><span class="u">∂L</span><span class="d">∂W</span></span> = <span style="opacity:.6">1/m ·</span> <span class="c2">Δ</span> · <span class="c1">A</span><sup>ᵀ</sup> &nbsp;&nbsp;<span style="opacity:.6">(n₂,m) × (m,n₁) = (n₂,n₁)</span></div>

这就是深度学习框架能把 batch 训练做得这么高效的原因：原本要循环 m 次做外积再累加，现在一次矩阵乘法就搞定。

---

## 07　转置为什么会突然冒出来

<div class="bp-q">隐藏层的这个传回来的东西，具体来说为啥长这样啊？</div>

这是整篇里我觉得最精妙的一步。核心在于：**一个隐藏神经元同时影响了下游所有神经元**，所以它的责任要沿所有路径加起来。

<div class="bp-fx"><span class="f"><span class="u">∂L</span><span class="d">∂a₂</span></span> = <span class="f"><span class="u">∂L</span><span class="d">∂z₁</span></span>·<span class="f"><span class="u">∂z₁</span><span class="d">∂a₂</span></span> + <span class="f"><span class="u">∂L</span><span class="d">∂z₂</span></span>·<span class="f"><span class="u">∂z₂</span><span class="d">∂a₂</span></span> = <span class="c3">W₁₂δ₁ + W₂₂δ₂</span></div>

每一项的含义很直白：δᵢ 是"下游那个神经元错了多少"，Wᵢⱼ 是"这根线的传导强度"。线越粗、下游错得越狠，退回来的责任就越大。

现在看关键。切换下面两个按钮，注意**高亮扫过的方向**：

<iframe class="bp-frame" src="{{ '/assets/backprop/read-direction.html' | relative_url }}" height="430" loading="lazy" title="前向横着读、反向竖着读"></iframe>
<p class="bp-cap">同一张权重表，前向沿行扫，反向沿列扫。</p>

| 方向 | 公式 | 怎么读 |
|---|---|---|
| 前向 | zᵢ = Σⱼ Wᵢⱼaⱼ | 固定行号 i，扫过所有列 → **横着读** |
| 反向 | ∂L/∂aⱼ = Σᵢ Wᵢⱼδᵢ | 固定列号 j，扫过所有行 → **竖着读** |

而矩阵乘法这个运算规则**永远是横着读的**。想让它改成竖着读，唯一的办法就是先把矩阵行列对调，再做普通矩阵乘法：

<div class="bp-fx"><span class="f"><span class="u">∂L</span><span class="d">∂a<sup>(1)</sup></span></span> = (W<sup>(2)</sup>)<sup class="c3">ᵀ</sup> <span class="c2">δ</span><sup>(2)</sup> &nbsp;&nbsp;<span style="opacity:.6">(n₁,n₂) × (n₂,1) = (n₁,1)</span></div>

<div class="bp-key"><strong>所以转置不是数学技巧，是"读取方向"的开关。</strong>而且全流程里出现的两个转置含义完全不同：<code>(W)ᵀ</code> 在传 δ 时出现，真的在改变运算语义；<code>(a)ᵀ</code> 在算本层梯度时出现，纯粹为了凑外积形状。</div>

### 还有一个容易踩的坑：传回去时必须用旧的 W

<div class="bp-q">交给前一层的时候，似乎还是用的旧的版本的 W 值？</div>

对，而且这是数学上唯一正确的做法。梯度永远是"**在某一个具体的参数点上**"求出来的——求这一点的偏导时，其他参数必须保持在这一点不动。算到一半就更新 W，得到的东西不对应任何一个点上的梯度。

```python
# ❌ 错误：边算边更新
for l in reversed(range(L)):
    grad = delta @ a[l-1].T
    W[l] -= lr * grad                          # 这里就把 W[l] 改了
    delta = W[l].T @ delta * dsig(z[l-1])      # 用的是被改过的 W！

# ✅ 正确：先全部算完，最后统一更新
for l in reversed(range(L)):
    grads[l] = delta @ a[l-1].T
    delta = W[l].T @ delta * dsig(z[l-1])      # W[l] 还是原值
for l in range(L):
    W[l] -= lr * grads[l]
```

PyTorch 里这个分离是显式的：`loss.backward()` 只往 `.grad` 里填数字，一个参数都不碰；只有 `optimizer.step()` 才真正更新。

<div class="bp-key"><strong>这个 bug 最难缠的地方：它不报错、不 NaN、损失看起来还在下降。</strong>因为被污染的方向和真梯度夹角通常还小于 90°，所以还是往下走，只是更慢更不稳。排查手段是<strong>梯度检查</strong>——用 <code>[L(w+ε) − L(w−ε)] / 2ε</code> 暴力算一遍做对照。如果只有最后一层对得上、越往前偏得越多，基本就是这类问题。</div>

---

## 08　完整数值演算

前面所有片段串起来跑一遍。网络设定：

<div class="bp-fx">x = [1.0, 2.0]ᵀ &nbsp;&nbsp; W<sup>(1)</sup> = [[0.5, −0.3], [0.2, 0.8]] &nbsp;&nbsp; W<sup>(2)</sup> = [0.6, −0.4] &nbsp;&nbsp; y = 1.0</div>

<iframe class="bp-frame" src="{{ '/assets/backprop/trace.html' | relative_url }}" height="400" loading="lazy" title="十步数值演算"></iframe>
<p class="bp-cap">十步，一步不跳。所有数字都通过了数值梯度校验。</p>

<div class="bp-key"><strong>注意第 6、7 步是从同一个 δ 分叉出去的两条线</strong>，不是一条流水线。拿到 δ⁽²⁾ 之后，"算本层梯度"和"传给前一层"互不干扰，顺序也无所谓。</div>

可运行的脚本在这里：[bp.py]({{ '/assets/backprop/bp.py' | relative_url }})。改改 W、x 或者激活函数，里面的数值梯度检查会自动验证结果对不对。

---

## 09　W 是参数还是自变量

<div class="bp-q">大部分时候我们说 W 是参数、输入是 x……但在计算梯度下降的时候，W 变成自变量，x 是固定的，所以这个的出现有时候也反应不过来</div>

这个困惑是**记号偷懒造成的**，不是理解力问题。完整写出来，损失函数其实长这样：

<div class="bp-fx">L( <span class="c2">W</span>, <span class="c2">b</span>, <span class="c1">x</span>, <span class="c1">y</span> )</div>

这四样在数学上**地位完全平等**。"参数"和"输入"的区别是我们人为赋予的语义，数学本身不关心。每次求偏导，只解锁其中一个，其余全部冻结：

| 场景 | 解锁（自变量） | 冻结（常数） | 在问什么 |
|---|---|---|---|
| 推理 / 前向 | x | W, b | 换个输入，输出怎么变 |
| 训练 / 反向 | W, b | x, y | 换组参数，损失怎么变 |

`∂L/∂W` 这个记号**省略了"此刻 x 和 y 被钉死"这个前提**。前提没写出来，但它一直在——这就是让人反应不过来的根源。

拿 `L = (Wx − y)²`、样本 x=2、y=6 举例：代入 x 和 y 得到 `L(W) = (2W−6)²`，碗底在 W=3；代入 W 和 y 得到 `L(x) = (3x−6)²`，碗底在 x=2。同一个 L，两条完全不同的曲线。

### 对 x 求导是真的会做的事

反向传播传到最前面时，顺手就把 `∂L/∂x` 也算出来了，平时训练用不上直接扔掉而已。最后一步和别的层公式完全一样，只有一个差别：

| 层 | 公式 | 差别 |
|---|---|---|
| 隐藏层 | δ⁽¹⁾ = (W⁽²⁾)ᵀδ⁽²⁾ **⊙ σ′(z⁽¹⁾)** | 要打折 |
| 输入层 | ∂L/∂x = (W⁽¹⁾)ᵀδ⁽¹⁾ | **没有 ⊙σ′** |

原因很朴素：**x 是原始数据，它不是任何激活函数的输出**，没有需要穿过的东西。

上面那个例子算出来 `∂L/∂x = [−0.0084, +0.0108]`，意思是"把 x₁ 调小、x₂ 调大，损失会变大"。FGSM 攻击就是沿这个**正方向**走一小步：

| | x | ŷ | L |
|---|---|---|---|
| 原始 | [1.0, 2.0] | 0.4854 | 0.1324 |
| 扰动后 ε=0.1 | [0.9, 2.1] | 0.4817 | 0.1343 ↑ |

<div class="bp-key"><strong>唯一的区别只有一个符号：</strong>训练是 <code>W ← W − η∇W</code>（让损失变小），攻击是 <code>x ← x + ε·sign(∇x)</code>（让损失变大）。同一套反向传播、同一个链式法则，只是目标反过来。链式法则并不知道也不关心谁是"参数"谁是"输入"，它只是机械地按依赖关系往回乘——你在哪儿喊停、把哪个变量捞出来，完全由你决定。</div>

---

## 10　速查表

### 整个算法就这么多

<div class="bp-pre">
【前向】 x → z⁽¹⁾ → a⁽¹⁾ → z⁽²⁾ → ŷ　　沿途把每层的 z 和 a 全部缓存

【起点】 δ⁽ᴸ⁾ = (∂L/∂ŷ) ⊙ σ′(z⁽ᴸ⁾)
　　　　　└ 损失函数的导数　└ 激活函数的导数

【循环】 从后往前，每层只做两件事：
　　① ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ　　存起来
　　② δ⁽ˡ⁻¹⁾ = (W⁽ˡ⁾)ᵀδ⁽ˡ⁾ ⊙ σ′(z⁽ˡ⁻¹⁾)　　传给前一层

【更新】 全部算完后统一走一步：W ← W − η·∂L/∂W
</div>

### 三条判错规则

| 规则 | 用法 |
|---|---|
| 梯度形状 = 参数形状 | 算完对一下维度，对不上就是某处该转置或乘法顺序反了 |
| δ 的个数 = 神经元个数 | 如果个数等于导线根数，那你算的是 ∂L/∂W 不是 δ |
| 数值梯度对照 | 只有最后一层对得上、越往前偏得越多 → 多半是原地更新污染 |

### 几个"为什么"的一句话答案

| 问题 | 答案 |
|---|---|
| 为什么 ∂L/∂b = δ | z = Wa + b，∂z/∂b = 1，乘 1 等于没乘 |
| 为什么 W 要乘输入、b 不用 | W 是跟 a 相乘的，b 是直接加上去的 |
| 为什么有两种转置 | (W)ᵀ 换读取方向；(a)ᵀ 只为凑外积形状 |
| 为什么必须用旧 W | 梯度是"某一个参数点上"的量，其余参数必须冻结 |
| 为什么分类用交叉熵 | 它的导数分母恰好约掉 σ′，治好梯度消失 |
| 为什么训练比推理吃显存 | 反向要用前向的中间结果，每层 z 和 a 都得缓存 |

---

写完回头看，这十个卡点有个共同特征：它们都不是"这一步的数学我不会算"，而是"我不知道这个符号此刻在指什么"。维度、求导对象、转置的语义、变量的冻结状态——全是记号省略掉的那些前提。

教科书省略它们是合理的，因为写全了会淹没主线。但学的时候，被淹没的恰恰是这些。

下一站大概会写 ReLU：把 sigmoid 换掉之后，上面哪个位置的公式变了，以及为什么这个改动能治梯度消失。

<script>
addEventListener('message', function(e){
  var d = e.data;
  if (!d || !d.backpropWidget) return;
  var frames = document.querySelectorAll('.bp-frame');
  for (var i = 0; i < frames.length; i++) {
    if (frames[i].contentWindow === e.source) {
      frames[i].style.height = d.height + 'px';
      return;
    }
  }
});
</script>
