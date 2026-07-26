import numpy as np

def sig(v): return 1/(1+np.exp(-v))

# ---- 网络设定 ----
x  = np.array([[1.0],[2.0]])                 # 输入 (2,1)
W1 = np.array([[0.5,-0.3],[0.2,0.8]])        # (2,2)
W2 = np.array([[0.6,-0.4]])                  # (1,2)
y  = np.array([[1.0]])                       # 真实标签

def forward(x):
    z1 = W1 @ x;  a1 = sig(z1)
    z2 = W2 @ a1; yh = sig(z2)
    L  = 0.5*((yh-y)**2).item()
    return z1,a1,z2,yh,L

z1,a1,z2,yh,L = forward(x)
print("=== 前向 ===")
print("z1 =", z1.ravel().round(4), " a1 =", a1.ravel().round(4))
print("z2 =", z2.ravel().round(4), " yh =", yh.ravel().round(4), " L =", round(L,6))

# ---- 反向 ----
d2 = (yh-y) * yh*(1-yh)                      # 输出层 delta
dL_da1 = W2.T @ d2
d1 = dL_da1 * a1*(1-a1)                      # 隐藏层 delta
dL_dx = W1.T @ d1                            # <<< 最后一步：对输入求导

print("\n=== 反向 ===")
print("delta2   =", d2.ravel().round(6))
print("dL/da1   =", dL_da1.ravel().round(6))
print("delta1   =", d1.ravel().round(6))
print("dL/dx    =", dL_dx.ravel().round(6))

# ---- 数值梯度验证 ----
eps=1e-6; num=[]
for i in range(2):
    xp=x.copy(); xp[i,0]+=eps; Lp=forward(xp)[4]
    xm=x.copy(); xm[i,0]-=eps; Lm=forward(xm)[4]
    num.append((Lp-Lm)/(2*eps))
print("数值梯度 =", np.array(num).round(6), " <- 和上面一致则正确")

# ---- 对抗样本：沿梯度正方向走 ----
x_adv = x + 0.1*np.sign(dL_dx)
print("\n=== 对抗扰动 (eps=0.1) ===")
print("原始 x    =", x.ravel(), " L =", round(L,6),      " yh =", round(yh.item(),4))
print("扰动 x    =", x_adv.ravel(), " L =", round(forward(x_adv)[4],6), " yh =", round(forward(x_adv)[3].item(),4))
