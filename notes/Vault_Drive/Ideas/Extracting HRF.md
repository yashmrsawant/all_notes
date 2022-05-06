# Extracting HRF

Suppose $Y_i(1:T)$ be $i$th trial fMRI activity of T timepoints for all $q$ regions. Let $X_i(1:T)$ be the latent activity governing the dynamics of these $q$ regions.

$$
Y_i |X_i \sim \mathcal{N}(HCX_i+d, R)
$$

 
$$
X_i=\begin{pmatrix}
X_i(1) && 
X_i(2) &&
X_i(3) &&
... &&
X_i(T_e)
\end{pmatrix}^T
$$
where $X_i(j)$ is the $p$ dimensional latent for $j$th timepoint. 

* Let $u$ be the length of the hemodynamic kernel. Break $u$ into parts $U_1 = [1:\lfloor u/2 \rfloor]$ and $U_2 = []
* Each $Y_i(1:T)$ can be modelled as follows:
$$
Y_i(1) = h_uCX_i(1)+...+h_1CX_i(u)
$$

$$
Y_i(2) = h_uCX_i(2) + ... + h_1CX_i(u+1)
$$

...

$$
Y_i(T) = h_uCX_i(T) + ... + h_1CX_i(T+h-1)
$$



or


$$
Y_i=\begin{pmatrix}
Y_i(1) \\
Y_i(2) \\
\vdots \\
Y_i(T) \\
\phi \\
\phi\\
\vdots\\
\phi
\end{pmatrix}_{qT_e\times 1 } \equiv HC
\begin{pmatrix}
X_i(1)\\
X_i(2)\\
X_i(3)\\
\vdots \\
X_i(T_e)
\end{pmatrix}
$$

$$
O=\sum_{i}Q_i
$$
For the $i$th region fMRI time-series.
$$
Q_i = -2 (Y_d)^TR^{-1}(HCX)+(HCX)^TR^{-1}(HCX)
$$
where $H$ is $T\times T$ hemodynamic kernel matrix for $i$th region, $C$ is mixing matrix of dimension $T\times pT$ and $X$ is vectorized latent matrix of size $pT\times 1$, and $R$ is region noise variance of size $T\times T$.

If we differentiate $Q_i$ with respect to $H$.

$$
\partial Q_i/\partial H=-2Y_d(CX)^T + 2H(C\mathbb{E}[XX^T]C^T)
$$

This equates to 

$$
H = \big(Y_d(CX)^T\big)\big(C\mathbb{E}[XX^T]C^T\big)^{-1}
$$



$$
\pmatrix{C_{i,:}^T && d_i} = \left(\sum_{t=1}^T \pmatrix{y^{i}(t)}\pmatrix{\nu^i(t)\\1}^T\right)\left(\sum_{t=1}^T \pmatrix{\nu^{i}(\nu^{i})^T(t) && \nu^{i}(t)\\(\nu^{i}(t))^T && 1} \right)^{-1}
$$

$$
R=\left<\sum_{k=1}^{N}\sum_{t=1}^{T}\text{diag}\left(\mathbb{E}_{x\sim x|Y_i}[(Y_k(t)-\tilde{C}\nu(t))(Y_k-\tilde{C}\nu(t))^T]\right)\right>
$$

where $\nu^{i}(1:T)=H^i\tilde{x}$

#### Derivation:

Model formulation I :

$$
\tilde{y}|\tilde{x} \sim \mathcal{N}(HC\tilde{x} + \tilde{d}, R)
$$
$$
\tilde{x} \sim \mathcal{N}(0, K)
$$

$\tilde{y}$: Flattened spatio-temporal signal $qT\times 1$, where $q$ is the spatial dimension of the signal.
$\tilde{x}$: Underlying latent explaining $\tilde{y}$ variation in spatio-temporal dimension $pT\times 1$, where $p$ is the dimensionality of the spatial dimension in the latent space.

$H$: $qT\times qT$ filter matrix which acts on mixed latent signal.

$$
\tilde{y}\sim \mathcal{N}(\tilde{d}, L)
$$

where $L=R+HCKC^TH^T$.

Lemma 1: For simplicity, let $C_h = HC$ then,

$$
\tilde{x}|\tilde{y} \sim \mathcal{N}(\Sigma C_h^TR^{-1}(\tilde{y}_d), \Sigma)
$$

where $\Sigma = (K^{-1}+C_h^TR^{-1}C_h)^{-1}$

Proof: 

Given,
$$
x\sim \mathcal{N}(0, K)\\
y|x \sim \mathcal{N}(C_hx+d, R)
$$

First we find an expression for the join distribution over $x$ and $y$. To do this, we define
$$
z=\pmatrix{x\\y}
$$
$$
\begin{eqnarray}
\log p(z) &=& \log p(x) + \log p(y|x)
\\ &=& -\frac{1}{2}\pmatrix{x \\ y}^T\Delta\pmatrix{x \\ y}
\end{eqnarray}
$$
where $\Delta^{-1}$ can be simplified as:

$$
\Delta^{-1} = \pmatrix{K && KC_h^T\\ C_hK && R + C_hKC_h}
$$






Complete conditional expected data loglikelihood can be expressed as follows:

$$
LL = \sum_{i=1}^{n}\mathbb{E}_{\tilde{x}|\tilde{y}=y_i}[\log p(\tilde{y}=y_i, \tilde{x})]
$$

$$
\log p(\tilde{y}=y_i, \tilde{x}) = \log p(y_i|\tilde{x}) + \log p(\tilde{x})
$$

$$
\mathbb{E}_{\tilde{x}|\tilde{y}=y_i}\big[\log p(\tilde{x})\big] = \mathbb{E}_{\tilde{x}|\tilde{y}=y_i}\big[-0.5(pT \log (2\pi) + \log |K| + \tilde{x}^TK^{-1}\tilde{x})\big]
$$


`
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from matplotlib import pyplot as plt
from numpy.matlib import repmat
import subprocess

try:
    from scipy import io as sio
except:
    import sys

    print("Not able to find scipy package: installing...")
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    install("scipy")
    from scipy import io as sio
dtype_ = tf.float64
from os import system, chdir
chdir("/home/yash/Tensorflow223_GPU/gpfa_hrf/parametricSPM/november_2021/")
def plotyxline(array):
    min_v, max_v = np.min(array), np.max(array) + 1
    plt.plot(np.arange(min_v, max_v), np.arange(min_v, max_v))
    
    
# In[]:
RT = 720  # In milliseconds

data = sio.loadmat("./model_gen_HRF.mat")

ks = 32  # In secs.
binWidth = data['binWidth'][0][0]
h = np.ceil(1000 * ks / binWidth).astype(np.int32)
xDim = data['xDim'][0][0]
yDim = data['yDim'][0][0]
nT = data['nT'][0][0]
Taus = data['Taus']
eps_val = data['eps'][0][0]
H_t = data['H_t']
params = {'C': data['C'], 'd': data['d'], \
          'R': data['R'], 'gamma': (binWidth / Taus) ** 2, \
          'eps': eps_val * np.ones((xDim, 1)), 'H_t': H_t}

nbatch = 150
mask_lowerTriangular = np.zeros((nT, nT))
for i in range(nT):
    mask_lowerTriangular[i, :i + 1] = 1

"""
    @ binWidth: In milliseconds (fMRI samples)
    @ ks: In seconds (hemodynamic kernel width)
    @ xDim: dimensionality of latent state
    @ yDim: number of ROIs
"""
stepsize = np.array([binWidth]).astype(np.float64)[0]  # In milliseconds
ks = np.array([ks]).astype(np.float64)[0]  # In secs.
# In[]: utility functions
def getHemodynamicTensors():
    """
    Returns
    -------
    H : Tensor (yDim x h)
        h is the length of the hemodynamic kernel.

    Ref: https://github.com/neurodebian/spm12/blob/master/spm_hrf.m#L43

    """
    dt = stepsize / 1000  # hemodynamic kernel stepsize in seconds
    h = np.ceil(ks / dt).astype(np.int32)
    D = tf.repeat(tf.reshape(tf.exp(logH_t[:, 5]), shape=[yDim, 1]), repeats=h, axis=1)
    U = tf.nn.relu(
        tf.reshape(tf.range(0, np.ceil(ks / dt), dtype=dtype_), shape=[1, h]) - D / dt) + tf.constant(
        [1e-100], dtype=dtype_)
    P1byP3 = tf.repeat(tf.reshape( \
        tf.divide(tf.exp(logH_t[:, 1 - 1]) + tf.exp(logH_t[:, 3 - 1]), tf.exp(logH_t[:, 3 - 1])),
        shape=[yDim, 1]), \
        repeats=h, axis=1)
    dtbyP3 = tf.repeat(tf.reshape(dt / tf.exp(logH_t[:, 3 - 1]), shape=[yDim, 1]), repeats=h, axis=1)
    P2byP4 = tf.repeat(tf.reshape( \
        tf.divide(tf.exp(logH_t[:, 2 - 1]) + tf.exp(logH_t[:, 4 - 1]), tf.exp(logH_t[:, 4 - 1])),
        shape=[yDim, 1]), \
        repeats=h, axis=1)
    dtbyP4 = tf.repeat(tf.reshape(dt / tf.exp(logH_t[:, 4 - 1]), shape=[yDim, 1]), repeats=h, axis=1)
    _1byP5 = tf.repeat(tf.reshape(1 / tf.exp(logH_t[:, 5 - 1]), shape=[yDim, 1]), repeats=h, axis=1)

    S1 = tf.exp(tf.multiply((P1byP3 - 1), tf.log(U)) + tf.multiply(P1byP3, tf.log(dtbyP3)) - \
                tf.multiply(dtbyP3, U) - tf.math.lgamma(P1byP3))
    S2 = tf.exp(tf.multiply((P2byP4 - 1), tf.log(U)) + tf.multiply(P2byP4, tf.log(dtbyP4)) - \
                tf.multiply(dtbyP3, U) - tf.math.lgamma(P2byP4))
    fU = S1 - tf.multiply(S2, _1byP5)

    H = tf.divide(fU, tf.reshape(tf.math.reduce_sum(fU, axis=1), shape=[yDim, 1]))
    return H


def getK_big_inv(nT):
    """
    Parameters
    ----------
    nT : INTEGER
        Number of timepoints.

    Returns
    -------
    K_big_toep_logdet : Tensor (xDim x 1)
        logdet of each of the gaussian process covariance kernel.
    K_big : Tensor (xDim * nT x xDim * nT)
        Covariance kernel for generating xDim-dimensional gaussian processes.
    K_big_inv : Tensor (xDim * nT x xDim * nT)
        Inverse of the covariance kernel K_big.

    Note
    -------
    gamma = (binWidth / Tau) ^ 2
    """

    Ts = tf.repeat(tf.reshape(tf.range(0, nT, dtype=dtype_), shape=[1, nT]), repeats=xDim, axis=0)
    TgammaSq = tf.exp(-0.5 * tf.square(Ts) * tf.reshape(tf.exp(gamma_log), shape=[xDim, 1]))
    TgammaSq_scaled = TgammaSq * tf.reshape(1 - tf.exp(eps_log), shape=[xDim, 1]) + \
                      tf.pad(tf.exp(eps_log), tf.constant([[0, 0], [0, nT - 1]]))
    K_big_toep = tf.linalg.LinearOperatorToeplitz(TgammaSq_scaled, TgammaSq_scaled, \
                                                  is_non_singular=True, is_positive_definite=True, is_square=True,
                                                  is_self_adjoint=True)
    K_big_toep_chol = K_big_toep.cholesky()
    K_big_toep_chol_inv = K_big_toep_chol.inverse().to_dense()
    K_big_toep_logdet = 2 * K_big_toep_chol.log_abs_determinant()
    K_big_inv_T = []
    for i in range(xDim):
        K_big_inv_T.append(tf.reshape(tf.matmul(tf.transpose(K_big_toep_chol_inv[i, :, :]), \
                                                K_big_toep_chol_inv[i, :, :]), shape=[nT * nT, 1]))
    K_big_inv_fl = tf.reshape(tf.concat(K_big_inv_T, axis=1), shape=[xDim * nT * nT])
    indices = np.zeros((xDim * nT * nT, 2)).astype(np.int32)
    count = 0
    for i in range(nT):
        for j in range(nT):
            ixs = np.arange(0, xDim).astype(np.int32)
            indices[count: count + xDim, 0] = ixs + i * xDim
            indices[count: count + xDim, 1] = ixs + j * xDim
            count += xDim
    K_big_inv = tf.sparse.to_dense(tf.sparse.reorder(
        tf.sparse.SparseTensor(indices, K_big_inv_fl, dense_shape=[xDim * nT, xDim * nT])))
    K_big = tf.sparse.to_dense(tf.sparse.reorder(
        tf.sparse.SparseTensor(indices, tf.reshape(tf.transpose(K_big_toep.to_dense(), perm=[1, 2, 0]) \
                                                   , shape=[xDim * nT * nT]),
                               dense_shape=[xDim * nT, xDim * nT])))
    return K_big_toep_logdet, K_big, K_big_inv


def blkDiag(list_tensors):
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    for tensor in list_tensors:
        full_matrix_shape = tensor.get_shape().with_rank_at_least(2)
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    row_blocks = []
    current_column = 0
    for tensor in list_tensors:
        row_before_length = current_column
        current_column += tensor.shape[-1]
        row_after_length = blocked_cols - current_column
        row_blocks.append(
            tf.pad(tensor=tensor, paddings=tf.concat([tf.zeros([tf.rank(tensor) - 1, 2], dtype=tf.int32), \
                                                      [(row_before_length, row_after_length)]], axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape([blocked_rows, blocked_cols])
    return blocked

# In[]: Computation graph
```
tf.reset_default_graph()
p1 = tf.placeholder(dtype_, shape=[yDim * xDim])  # C
p2 = tf.placeholder(dtype_, shape=[yDim])  # d
p3 = tf.placeholder(dtype_, shape=[yDim])  # R_log
p4 = tf.Variable(2 * tf.cast(tf.log(np.ones((xDim)) * binWidth/720), dtype = dtype_)) # gamma_log
p5 = tf.placeholder(dtype_, shape=[xDim])  # eps_log
p6 = tf.Variable(tf.random_normal([yDim * 6], dtype = dtype_)) # spm_hrf
PI = tf.constant([np.math.pi], dtype_)

C = tf.reshape(p1, shape=[yDim, xDim])
d = tf.reshape(p2, shape=[yDim, 1])
R_log = tf.reshape(p3, shape=[yDim, 1])
gamma_log = tf.reshape(p4, shape=[xDim, 1])
eps_log = tf.reshape(p5, shape=[xDim, 1])
offset = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 1e-100]).reshape((1, 6))
logH_t = tf.log(tf.reshape(tf.exp(p6), shape=[yDim, 6]) + tf.constant(offset,
                        dtype_))  # UPDATE: feed log values to p6

h = np.ceil(1000 * ks / stepsize).astype(np.int32)

Y = tf.placeholder(dtype_, shape=[None, yDim, nT])

nTe = nT
X = tf.placeholder(dtype_, shape=[None, xDim, nTe])
V = tf.placeholder(dtype_, shape=[xDim * nTe, \
                                            xDim * nTe])

Yd = Y - d
R_invYd = Yd * tf.exp(-1 * R_log)

HKernels = getHemodynamicTensors()
H_ph = tf.placeholder(dtype_, shape = [yDim, h])
H = tf.where(tf.is_nan(H_ph), tf.ones_like(H_ph) * 0, H_ph)

H_f = tf.reverse_v2(H, [-1])
K_big_logdets, K_big, K_big_inv = getK_big_inv(nTe)

logdetK_big = tf.reduce_sum(K_big_logdets, axis=0)
detR_log = nT * tf.reduce_sum(R_log, axis=0)

```
```
u1 = h//2
u2 = h - 1 - u1
paddings = tf.constant([[0, 0], [0, 0], [u2, u1], [0, 0]])
paddings_t = tf.constant([[0, 0], [0, 0], [u1, u2], [0, 0]])

H_reshaped = tf.expand_dims(tf.expand_dims(tf.transpose(H), \
                                                axis=0), axis=3)
H_f_reshaped = tf.expand_dims(tf.expand_dims(tf.transpose(H_f), \
                                                  axis=0), axis=3)
"""
    computing CtHtR_invYd
"""
HtR_invYd = tf.transpose(tf.squeeze(tf.nn.depthwise_conv2d( \
    tf.pad(tf.expand_dims(tf.transpose(R_invYd, perm=[0, 2, 1]), 1), \
           paddings), H_reshaped, strides=[1, 1, 1, 1], \
    padding='VALID'), axis=1), perm=[0, 2, 1])
CtHtR_invYd = tf.matmul(tf.transpose(C), HtR_invYd)

"""
    computing CtHtR_invHC_bar
"""
Ct_nTcopies = []
for i in range(nT):
    Ct_nTcopies.append(tf.transpose(C))
C_bar = tf.expand_dims(tf.reshape(blkDiag(Ct_nTcopies), shape = [xDim * nTe, nTe, yDim]), 1)

HC_bar = tf.squeeze(tf.nn.depthwise_conv2d(tf.pad(C_bar, paddings_t),H_f_reshaped, strides = [1, 1, 1, 1],
                                           padding = 'VALID'), axis = 1)
R_invHC_bar = HC_bar * tf.reshape(tf.exp(-1 * R_log), shape = [1, 1, yDim])
HtHC_bar = tf.transpose(tf.squeeze(tf.nn.depthwise_conv2d( \
    tf.pad(tf.expand_dims(HC_bar, axis=1), \
            paddings), H_reshaped, \
    strides=[1, 1, 1, 1], padding='VALID'), axis=1), \
    perm=[0, 2, 1])
CtHtHC_bar = tf.layers.flatten(tf.transpose( \
    tf.matmul(tf.transpose(C), HtHC_bar), \
    perm=[0, 2, 1]))
HtR_invHC_bar = tf.transpose(tf.squeeze(tf.nn.depthwise_conv2d( \
    tf.pad(tf.expand_dims(R_invHC_bar, axis=1), \
           paddings), H_reshaped, \
    strides=[1, 1, 1, 1], padding='VALID'), axis=1), \
    perm=[0, 2, 1])
CtHtR_invHC_bar = tf.layers.flatten(tf.transpose( \
    tf.matmul(tf.transpose(C), HtR_invHC_bar), \
    perm=[0, 2, 1]))
detV_log = 2 * tf.reduce_sum(tf.log(tf.diag_part(tf.linalg.cholesky(V))))
detL_log = detR_log + logdetK_big - detV_log

# In[]:
def e_step(nT):
    """

    :param obj: object of Model class
    :param nT: length of latent dynamics
    :return: posterior mean and posterior covariance of latents given observations
    """
    Vsm = tf.linalg.inv(K_big_inv + CtHtR_invHC_bar)

    T1 = tf.matmul(K_big, tf.eye(xDim * nT, dtype = dtype_) - tf.matmul(CtHtR_invHC_bar, Vsm))
    Xsm = tf.transpose(tf.matmul(T1, tf.transpose(tf.layers.flatten(tf.transpose(
        CtHtR_invYd, perm = [0, 2, 1])))))

    X_e = tf.transpose(tf.reshape(Xsm, shape = [-1, nT, xDim]), perm = [0, 2, 1])
    V_e = Vsm
    return X_e, V_e

# In[]: FA params
X_reshaped = tf.expand_dims(tf.transpose(tf.pad(X, \
                                                tf.constant([[0, 0], [0, 0], [u1, u2]])), perm=[0, 2, 1]), axis=1)
constant = tf.reshape(tf.constant(nbatch * nT, dtype_), shape=[1, 1])
def funC_d(acc, yDim_i):
    H_i = tf.repeat(H_f_reshaped[:, :, (yDim_i): (yDim_i + 1), :], \
                    repeats=xDim, axis=2)
    Nu_i = tf.layers.flatten(tf.transpose(tf.squeeze(tf.nn.depthwise_conv2d( \
        X_reshaped, H_i, strides=[1, 1, 1, 1], padding='VALID'), axis=1), perm=[2, 0, 1]))

    NuNut_i = tf.matmul(Nu_i, tf.transpose(Nu_i))

    Vchol = tf.expand_dims(tf.transpose(tf.reshape(tf.linalg.cholesky(V), \
                                                   shape=[nTe, xDim, nTe * xDim]), perm=[2, 0, 1]), axis=1)
    HVchol = tf.reshape(tf.transpose(tf.squeeze(tf.nn.depthwise_conv2d( \
        tf.pad(Vchol, paddings_t), H_i, strides=[1, 1, 1, 1], padding='VALID'), \
        axis=1), perm=[1, 2, 0]), shape=[nT * xDim, nT * xDim])
    HVVtHt = tf.reshape(tf.transpose(tf.reshape(tf.matmul(HVchol, tf.transpose(HVchol)), \
                                                shape=[nT, xDim, nT, xDim]), perm=[2, 0, 1, 3]), \
                        shape=[nT * nT, xDim, xDim])
    idxs = np.arange(0, nT * nT, nT + 1)
    t1 = (nbatch) * tf.reduce_sum(tf.gather(HVVtHt, idxs, axis=0), axis=0) + NuNut_i
    t2 = tf.reshape(tf.reduce_sum(Nu_i, axis=1), shape=[xDim, 1])

    r1 = tf.concat([t1, t2], axis=1)
    r2 = tf.concat([tf.transpose(t2), constant], axis=1)

    term2 = tf.concat([r1, r2], axis=0)
    Y_i = tf.layers.flatten(tf.transpose(Y[:, yDim_i: (yDim_i + 1), :], perm=[1, 0, 2]))
    YNut_i = tf.matmul(Y_i, tf.transpose(Nu_i))

    term1 = tf.concat([YNut_i, tf.reshape(tf.reduce_sum(Y_i, axis=1), shape=[1, 1])], axis=1)
    Cd_i = tf.matmul(term1, tf.linalg.inv(term2))
    return Cd_i
Cd_0 = funC_d(0, 0)
Cd_hat1 = tf.scan(funC_d, tf.range(0, yDim // 2), Cd_0)
Cd_hat2 = tf.scan(funC_d, tf.range(yDim // 2, yDim), Cd_0)
Cd_hat = tf.squeeze(tf.concat([Cd_hat1, Cd_hat2], axis=0), axis=1)

def getR(nbatch, nT):
    Cx = tf.expand_dims(tf.matmul(tf.transpose(X, \
                                               perm=[0, 2, 1]), tf.transpose(C)), axis=1)
    ChX = tf.layers.flatten(tf.squeeze(tf.nn.depthwise_conv2d(tf.pad(Cx, paddings_t), \
                                                              H_f_reshaped, strides=[1, 1, 1, 1], \
                                                              padding='VALID'), axis=1))
    Ch = tf.transpose(tf.reshape(HC_bar, shape=[nT * xDim, nT * yDim]))
    V_ = tf.reduce_sum(tf.square(tf.matmul(Ch, tf.linalg.cholesky(V))), axis=1)
    Yd_ = tf.layers.flatten(tf.transpose(Yd, perm=[0, 2, 1]))
    R_hat = tf.reshape(tf.reduce_mean(tf.square(Yd_ - ChX), axis=0) + V_, shape=[nT, yDim])
    R_hat = tf.reduce_mean(R_hat, axis=0)
    return R_hat

C_hat = Cd_hat[:, :xDim]
d_hat = Cd_hat[:, xDim:]
R_hat = getR(nbatch, nT)

# In[]: posterior mean and posterior variance of latents given observation
Xsm, Vsm = e_step(nT)
# In[]: Q(Taus) = XtK_invX + logdet(K), Q(H) = -2 * Yd^tR^{-1}(HCX) + (HCX)^TR^{-1}(HCX)
X_flat = tf.reshape(tf.transpose(X, perm=[0, 2, 1]),
                 shape=[-1, xDim * (nT)])
XtK_inv_X = tf.reduce_sum(X_flat * tf.matmul(X_flat, K_big_inv), axis=1) + \
            tf.linalg.trace(tf.matmul(V, K_big_inv))
QTaus = nbatch * logdetK_big + tf.reduce_sum(XtK_inv_X) # Minimize
# In[]:

sXXt = tf.reshape(tf.transpose(tf.linalg.cholesky(tf.matmul(tf.transpose(X_flat), X_flat) + nbatch * V)), \
                  shape = [xDim * nT, nT, xDim])
def funH(acc, yDim_i):
    C_i = tf.reshape(tf.gather(C, yDim_i), shape = [xDim, 1])
    Yd_i = Yd[:, yDim_i, :]
    CX_i = tf.squeeze(tf.matmul(tf.transpose(X, perm = [0, 2, 1]), C_i), axis = 2)
    term1 = tf.matmul(tf.transpose(Yd_i), CX_i)
    sXXtC_i = tf.squeeze(tf.matmul(sXXt, C_i), axis = 2)
    term2 = tf.linalg.inv(tf.matmul(tf.transpose(sXXtC_i), sXXtC_i))
    
    H_i = tf.matmul(term1, term2)
    return H_i
H_0 = funH(None, 0)
Hs = tf.reshape(tf.scan(funH, tf.range(0, yDim), H_0), shape = [yDim, nT * nT])

def fun_h1(acc, ix):
    idxs = tf.range(ix, nT ** 2, nT + 1)
    return tf.reduce_mean(tf.gather(Hs, idxs, axis = 1), axis = 1)
def fun_h2(acc, ix):
    idxs = tf.range(nT * ix + nT, nT ** 2, nT + 1)
    return tf.reduce_mean(tf.gather(Hs, idxs, axis = 1), axis = 1)
Hs_1 = tf.scan(fun_h1, tf.range(u2, -1, -1), fun_h1(None, 0))
Hs_2 = tf.scan(fun_h2, tf.range(0, u1), fun_h2(None, 0))
H_hat = tf.transpose(tf.concat([Hs_1, Hs_2], axis = 0))

# In[]:
train_step1 = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(QTaus, var_list = [p4])

# In[]:
    

try:
    sess.close()
    sess.close()
except:
    print("Error closing session")
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# In[]:
fa_startParams = sio.loadmat('./fA_startParams.mat')
startParams = {'C' : fa_startParams['C'].astype(np.float64),
               'd' : fa_startParams['d'].astype(np.float64),
               'R' : fa_startParams['R'].astype(np.float64),
               'eps' : 0.001 * np.ones((xDim)).astype(np.float64)}
plt.scatter(params['d'], fa_startParams['d'])
plotyxline(params['d'])

# In[]:
feed_dict_ = {p1 : params['C'].reshape((yDim * xDim)),
              p2 : params['d'].reshape((yDim)),
              p3 : np.log(np.diag(params['R'])).reshape((yDim)),
              p5 : np.log(startParams['eps']).reshape((xDim))}
feed_dict_[Y] = data['Y'][:nbatch, :, :]
H_t_init = repmat(np.array([6, 16, 1, 1, 6, 1e-100]), yDim, 1)

# In[]: getting actual Hemodynamic profs
p6.load(np.log(params['H_t']).reshape((yDim * 6)), sess)
H_actual_v = sess.run(H)

p4.load(np.log(params['gamma']).reshape((xDim)), sess)
p6.load(np.log(H_t_init.reshape((yDim * 6))), sess)
H_v = sess.run(H)
# In[]: E-M algorithm
num_epochs = 20000
temp = 0
for iters in range(num_epochs):
    # Expectation
    Xsm_v, Vsm_v = sess.run([Xsm, Vsm], feed_dict_)

    feed_dict_[X] = Xsm_v
    feed_dict_[V] = Vsm_v

    # Maximization
    # C_hat_v, d_hat_v = sess.run([C_hat, d_hat], feed_dict_)
    # plt.plot(C_hat_v.reshape((yDim * xDim)) - params['C'].reshape((yDim * xDim)))
    # plt.show()
    
    # feed_dict_[p1] = C_hat_v.reshape((yDim * xDim))
    # feed_dict_[p2] = d_hat_v.reshape((yDim))

    # R_hat_v = sess.run(R_hat, feed_dict_)
    # feed_dict_[p3] = np.log(R_hat_v.reshape((yDim)))
    Z = sess.run(grad_LLpY_p6, feed_dict_)
    if iters % 2 == 0:
        #sess.run(train_step1, feed_dict_)
        A = None # NULL Statement
    else:
        sess.run(train_step2, feed_dict_)
    gammaLog_v = sess.run(p4)
    print("%d) Taus = "%(iters), binWidth * np.exp(-0.5 * gammaLog_v))
    if iters % 50 == 50:
        yDim_i = 0
        plt.plot(H_actual_v[yDim_i, :])
        H_v = sess.run(H)
        plt.plot(H_v[yDim_i, :])
        plt.show()

```
        

`


```

clear; clc;
A = [ 0.00000000e+00  5.92555719e-04  1.84088167e-02  1.01932858e-01 2.78575429e-01  
    5.17056917e-01  7.51356256e-01  9.22129192e-01 1.00000000e+00  9.86402345e-01  
    9.02362862e-01  7.75634423e-01 6.31584721e-01  4.88896831e-01  3.58926382e-01  
    2.47034308e-01 1.54538333e-01  8.04832043e-02  2.29079242e-02 -2.04083919e-02 
    -5.15985077e-02 -7.26015076e-02 -8.51702810e-02 -9.09081661e-02 -9.12935319e-02 
    -8.76813246e-02 -8.12871539e-02 -7.31652156e-02 -6.41903772e-02 -5.50508008e-02 
    -4.62531087e-02 -3.81387262e-02 -3.09080968e-02 -2.46488760e-02 -1.93645501e-02 
    -1.50007964e-02 -1.14679164e-02 -8.65860806e-03 -6.46106807e-03 -4.76789026e-03
     -3.48148191e-03 -2.51679764e-03 -1.80215184e-03 -1.27876382e-03 -8.99555682e-04];
   
   
   
A_d = [-0.00059256 -0.01722371 -0.06570778 -0.09311853 -0.06183892
        0.00418215  0.0635264   0.09290213  0.09146846  0.07044183
        0.04268896  0.01732126 -0.00136181 -0.01271744 -0.01807837
       -0.0193961  -0.01844085 -0.01647985 -0.01425896 -0.0121262
       -0.01018712 -0.00843423 -0.00683089 -0.00535252 -0.00399757
       -0.00278196 -0.00172777 -0.0008529  -0.00016474  0.00034188
        0.00068331  0.00088375  0.00097141  0.00097489  0.00092057
        0.00083087  0.00072357  0.00061177  0.00050436  0.00040677
        0.00032172  0.00025004  0.00019126  0.00014418 -0.00052035];

    
    
A_d = reshape(A_d', 45, 1);
A = reshape(A', 45, 1);
%%
ks = [];
for noise_scale = [0.001 : 0.005 : 0.1]
    A_a = A + randn(45, 1) * noise_scale;
z_temp = [];
for i = 2 : 44
    z_temp = [z_temp; A_a(i - 1) * -1 + A_a(i) * 1];
end
ks = [ks; sum(z_temp .^ 2)];
fprintf("Laplacian Sum = %f\n", sum(z_temp .^ 2));
end
plot([0.001 : 0.005 : 0.1], ks);
line([0.001, 0.1], [ks(2, 1),ks(2, 1)]);
%%
k = 0.056;
figure; hold on;
A_a = A + randn(45, 1) * k;
plot(A_a);

``


### 
Model formulation I :

$$
\tilde{y}|\tilde{x} \sim \mathcal{N}(HC\tilde{x} + \tilde{d}, R)
$$
$$
\tilde{x} \sim \mathcal{N}(0, K)
$$

Generative Model: Dynamics of the $q$ regions is modeled using gaussian processes.


$$
\log p\big(\tilde{y}, \tilde{x}\big) \propto -\bigg(\|R^{-0.5}(\tilde{y}-HC\tilde{x})\|^2+\tilde{x}^TK^{-1}\tilde{x} + \log(|K|.|R|)\bigg)
$$

$$
O = \bigg(\mathbb{E}_{\tilde{x}|\tilde{y}}\left[\|\tilde{y}-HC\tilde{x})\|^2\right]+\lambda\sum_{i=1}^{q}\|\nabla H_i\|^2\bigg)
$$

Let $h_1, h_2, ..., h_u$ be the sequence of the hrf kernel of a particular region.
Then 
$$
\nabla[h_1, h_2, ..., h_u] = [2h_1 - h_2, 2h_2 - (h_1 + h_3), ..., 2h_k - (h_{k - 1} + h_{k + 2}), ...]
$$


# -*- coding: utf-8 -*-

def minimize(sess, length, obj, p, grad_obj_p, feed_dict_):

    """
    Minimize a differentiable multivariate function.

    Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )

    where the starting point is given by "X" (D by 1), and the function named in
    the string "f", must return a function value and a vector of partial
    derivatives of f wrt X, the "length" gives the length of the run: if it is
    positive, it gives the maximum number of line searches, if negative its
    absolute gives the maximum allowed number of function evaluations. You can
    (optionally) give "length" a second component, which will indicate the
    reduction in function value to be expected in the first line-search (defaults
    to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.

    The function returns when either its length is up, or if no further progress
    can be made (ie, we are at a (local) minimum, or so close that due to
    numerical problems, we cannot get any closer). NOTE: If the function
    terminates within a few iterations, it could be an indication that the
    function values and derivatives are not consistent (ie, there may be a bug in
    the implementation of your "f" function). The function returns the found
    solution "X", a vector of function values "fX" indicating the progress made
    and "i" the number of iterations (line searches or function evaluations,
    depending on the sign of "length") used.

    The Polack-Ribiere flavour of conjugate gradients is used to compute search
    directions, and a line search using quadratic and cubic polynomial
    approximations and the Wolfe-Powell stopping criteria is used together with
    the slope ratio method for guessing initial step sizes. Additionally a bunch
    of checks are made to make sure that exploration is taking place and that
    extrapolation will not be unboundedly large.

    See also: checkgrad

    Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

    Permission is granted for anyone to copy, use, or modify these
    programs and accompanying documents for purposes of research or
    education, provided this copyright notice is retained, and note is
    made of any changes that have been made.

    These programs and documents are distributed without any warranty,
    express or implied.  As the programs were written for research
    purposes only, they have not been tested to the degree that would be
    advisable in any important application.  All use of these programs is
    entirely at the user's own risk.
    """
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG/2

    """
    INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
    MAX = 20;                         % max 20 function evaluations per line search
    RATIO = 10;                                       % maximum allowed slope ratio
    SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
    Powell conditions. SIG is the maximum allowed absolute ratio between
    previous and new slopes (derivatives in the search direction), thus setting
    SIG to low (positive) values forces higher precision in the line-searches.
    RHO is the minimum allowed fraction of the expected (from the slope at the
    initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    Tuning of SIG (depending on the nature of the function to be optimized) may
    speed up the minimization; it is probably not worth playing much with RHO.

    The code falls naturally into 3 parts, after the initial line search is
    started in the direction of steepest descent. 1) we first enter a while loop
    which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
    have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
    enter the second loop which takes p2, p3 and p4 chooses the subinterval
    containing a (local) minimum, and interpolates it, unil an acceptable point
    is found (Wolfe-Powell conditions). Note, that points are always maintained
    in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
    conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
    was a problem in the previous line-search. Return the best value so far, if
    two consecutive line-searches fail, or whenever we run out of function
    evaluations or line-searches. During extrapolation, the "f" function may fail
    either with an error or returning Nan or Inf, and minimize should handle this
    gracefully.
    """

    realmin = np.finfo(float).tiny
    len_param = feed_dict_[p].shape[0]
    red = 1
    # if length > 0:
    #     S = 'Linesearch'
    # else:
    #     S = 'Function evaluation'
    i = 0
    ls_failed = 0
    X = feed_dict_[p]

    [f0, df0] = sess.run([obj, grad_obj_p], feed_dict = feed_dict_)
    fX = f0

    if length < 0:
        i += 1
    s = -df0
    d0 = -np.matmul(s.reshape((1, len_param)), s.reshape((len_param, 1))).reshape((1))
    x3 = red / (1 - d0)
    x3 = x3.reshape((1))
    while i < np.abs(length):
        if length > 0:
            i += 1
        X0 = X
        F0 = f0
        dF0 = df0
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        while True:
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0
            while (not success) and (M > 0):
                try:
                    M = M - 1
                    if length < 0:
                        i += 1
                    feed_dict_[p] = (X + x3 * s).reshape((len_param))
                    [f3, df3] = sess.run([obj, grad_obj_p], feed_dict = feed_dict_)
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        raise ValueError('')
                    success = 1;
                except ValueError:
                    x3 = (x2 + x3) / 2
            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3
            d3 = np.matmul(df3.reshape((1, len_param)), s.reshape((len_param, 1))).reshape((1))
            if (d3 > SIG * d0) or (f3 > f0 + x3 * RHO * d0) or M == 0:
                break
            x1 = x2
            f1 = f2
            d1 = d2

            x2 = x3
            f2 = f3
            d2 = d3

            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            x3 = np.array([x1 - d1 * (x2 - x1) ** 2/(B + np.sqrt(B * B - A * d1 * (x2 - x1)))]).reshape((1))
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:
                x3 = x2 * EXT
            elif x3 > x2 * EXT:
                x3 = x2 * EXT
            elif x3 < x2 + INT * (x2 - x1):
                x3 = x2 + INT * (x2 - x1)

        while ((np.abs(d3) > -SIG*d0) or (f3 > (f0 + x3 * RHO * d0))) and M > 0:
            if d3 > 0 or f3 > (f0 + x3 * RHO * d0):
                x4 = x3
                f4 = f3
                d4 = d3
            else:
                x2 = x3
                f2 = f3
                d2 = d3
            if f3 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2)**2)/(f4 - f2 - d2 * (x4 - x2))
            else:
                A = 6 * (f2 - f4)/(x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = np.array([x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B)/A]).reshape((1))
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2
            x3 = np.max([np.array(np.min([x3, (x4 - INT * (x4 - x2))])).reshape((1)), \
                         np.array([(x2 + INT * (x4 - x2))]).reshape((1))])
            feed_dict_[p] = X + x3 * s
            [f3, df3] = sess.run([obj, grad_obj_p], feed_dict = feed_dict_)
            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3
            M = M - 1
            if length < 0:
                i += 1
            d3 = np.matmul(df3.reshape((1, len_param)), s.reshape((len_param, 1)))
        if(np.abs(d3) < -SIG * d0) and (f3 < f0 + x3 * RHO * d0):
            X = X + x3 * s
            f0 = f3
            fX = np.append(fX, f0)
            s = (np.matmul(df3.reshape((1, len_param)), df3.reshape((len_param, 1))).reshape((1)) - \
                np.matmul(df0.reshape((1, len_param)), df3.reshape((len_param, 1))).reshape((1))) / \
                (np.matmul(df0.reshape((1, len_param)), df0.reshape((len_param, 1)))).reshape((1)) * s - df3
            df0 = df3
            d3 = d0
            d0 = np.matmul(df0.reshape((1, len_param)), s.reshape((len_param, 1)))
            if d0 > 0:
                s = -df0
                d0 = np.matmul(s.reshape((1, len_param)), s.reshape((len_param, 1)))
            x3 = x3 * np.min([RATIO, (d3/(d0 - realmin)).reshape((1))])
            ls_failed = 0
        else:
            X = X0
            f0 = F0
            df0 = dF0
            if ls_failed or (i > np.abs(length)):
                break
            s = -df0
            d0 = -np.matmul(s.reshape((1, len_param)), s.reshape((len_param, 1)))
            x3 = 1 / (1 - d0)
            ls_failed = 1
    return X, fX, i



import numpy as np


# In[]:    



$$
    \log p(X) \propto |K|+\text{trace}(XX^T K^{-1})\tag{1}
$$
If we differentiate (1) w.r.t to K, the optimal taus would be when

$$
K\approx\mathbb{E}_{X\sim X|Y}[XX^T]
$$

```
```

```


### Date: 4th Jan, 2022

1) Suppose we have infinite length (in time-domain) fMRI time-series, we can assume that latent dynamics is very close to zero at the start and at the end. 
2) Let the length of the hemodynamic kernel be $h$. Then we can model inifite length fMRI time-series using infinite length gaussian process as below:

$$
Y_{-\infty:T+h-1}|X_{-\infty:T} \sim \mathcal{N}(HCX+d,R)
\\
X_{-\infty:T}\sim \mathcal{N}(0, K)
\tag{1}
$$
3) The hemodynamic kernel matrix $H$ corresponding to above relationship can be defined as:

$$
H=\pmatrix{h_1 & 0 & 0 & ... & 0\\
h_2 & h_1 & 0 & ... & 0\\
h_3 & h_2 & h_1 & ... & 0\\
\vdots & \vdots & \vdots & \ddots & 0\\
0 & 0 & 0 & ... & h_1\\
0 & 0 & 0 & ... & h_2\\
\vdots &\vdots & \vdots & \ddots & \vdots\\
0 & 0 & 0 & ... & h_u\\
}_{-\infty:T+h-1, -\infty:T}
$$

$$
H^T=\pmatrix{h_1 & h_2 & h_3 & ... & 0\\
0 & h_1 & h_2 & ... & 0\\
0 & 0 & h_1 & ... & 0\\
\vdots & \vdots & \vdots & \ddots & 0\\
0 & 0 & 0 & ... & h_u\\
}_{-\infty:T, -\infty:T+h-1}
$$

$h_i$ is diagonal matrix containing the kernel value of $i$th timepoint for each of the $q$ region. 


4) Using set of gaussian relationship (1), we can infer latent given fMRI time-series as follows:
$$
X_{-\infty:T}|Y_{-\infty:T+h-1} \sim \mathcal{N}(VC^TH^TR^{-1}(Y-d), V)
$$

where $V=(K^{-1}+C^TH^TR^{-1}HC)^{-1}$, the dimension of the matrix $V$ is $-\infty:T \times -\infty:T$.

5. For the case when $T\rightarrow\infty$ or can be equally formulated for limited $T$.
Gaussian kernels for $p$ latents for the continous case can be defined as which basically convolves with $p$-dimensional infinite length time-series using the kernels below:

$$
K_{1:p}(1, t;\tau_1, ...,\tau_p) = [(1- \sigma_1^2) \exp\left(-\frac{t^2}{2\tau_1^2}\right) + \delta(t) \sigma_1^2; 0; 0;...; 0]_{1:p\times -\infty:T}
$$

$$
K_{1:p}(2, t; \tau_1, ...,\tau_p) = [0; (1 - \sigma_2^2) \exp\left(-\frac{t^2}{2\tau_2^2}\right)+\delta(t)\sigma^2_2 ; 0;0; ...;0]_{1:p\times -\infty:T}
$$

$$\vdots$$

$$
K_{1:p}(p, t; \tau_1, ...,\tau_p) = [0; 0; ...; (1-\sigma_p^2) \exp\left(-\frac{t^2}{2\tau_p^2}\right)+\delta(t)\sigma^2_2]_{1:p\times -\infty:T}
$$

![](https://i.imgur.com/qKJXFvo.png)

6. To compute $V$ or more specifically compute $VC^TH^TR^{-1}(Y-d)$ using this continous extension,
we can just find kernel $u_{1:p}(l, t)$ and convolve in the time-domain with $C^TH^TR^{-1}(Y-d)$ which will be of dimension $1:p \times -\infty:T+h-1$ such that. 

$$
\int_{-\infty}^{T} u_{1:p}(l, t-k)\left(K^{-1}_{1:p}(l,t)+C^TH^TR^{-1}HC_{1:p}(l,t)\right)dt = \delta(k)
$$

$$
u_{1:p}(l, t) \approx \mathcal{F}^{-1}\left(\frac{1}{\mathcal{F}(K^{-1}(l, t) + z(l, t))}\right)(-t)
$$

**Note**: $K^{-1}_{1:p}(l, t)$ profile should be symmetric about t=0 and also $C^TH^TR^{-1}HC_{1:p}(l, t)$ even for random hemodynamic kernel filter, where $l$ is the latent dimension for which we want to compute the kernel.



























## Date: 9th Jan-2022

1) $\text{TR} \approx 1.778 \text{ sec}$

2) Length of the HRF kernel $h = 18$ for 32 sec duration.

3) Length of fMRI time-series to consider to model $nT$ length fMRI time-series. 
a) Length of latent time-series, $nT_e = nT + h - 1$
b) Length of fMRI time-series need to extract $nTe$ latent time-series, $nT_h = nTe + h - 1 + k - 1$, where $k$ is the length of $V$ kernel.
c) $V$ kernel is estimated such that
$$
\lim_{n\rightarrow \infty} \int_{-\infty}^{T} v(t-k)(k^{-1}(t)+z(t))dt=\delta(k)
$$
where $z(t) = C^TH^TR^{-1}HC(t)$.
4) Parameters of the model $P=\{C, d, R, \tau_1, ...,\tau_p, h_1, h_2, ..., h_q\}$.
a) Estimating $C$:
   $$
   \pmatrix{C^i && d} = \left<\mathbb{E}_{x \sim X|y_n}\left[\pmatrix{\sum_{t=1}^{T} y^i(t)\pmatrix{\nu^{i}(t)\\1 }}\pmatrix{\sum_{t=1}^{T}\pmatrix{\nu^{i}(\nu^{i})^T(t) && \nu^{i}(t)\\ (\nu^{i}(t))^T&& 1}}^{-1}\right]\right>
   $$
   
    Computing $\mathbb{E}[\nu^{i}(\nu^{i})(t)]=\mathbb{E}[\nu^{i}(t)]\mathbb{E}[\nu^{i}(t)] + H^iV(H^i)^T$
    
    b) Estimating R:
    $$
    R^i=\left<\mathbb{E}_{x\sim X|y_n}||y^i-H^iC^ix||^2\right>
    $$
    $$
    x^T(C^i)^T(H^i)^TH^iC^ix=\text{trace}(xx^T (C^i)^T(H^i)^TH^iC^i)
    $$
    c) Estimating $\Gamma=\{\tau_1, \tau_2, ..., \tau_p\}$:
   
5) To prove: $V$ is toeplitz matrix
$$
V=(K^{-1}+C^TH^TR^{-1}HC)^{-1}
$$
$$
\begin{eqnarray}
V&=&K(I-C^TH^TR^{-1}HCV)\\
K&=&(I+KC^TH^TR^{-1}HC)V
\end{eqnarray}
$$

Suppose $S$ and $R$ be two matrices such that in the block-wise manner, it is toeplitz that is:
$$
S=\begin{pmatrix}
s(0) & s(1) & ... & s(T)\\
s(-1) & s(0) & ... & s(T-1)\\
\vdots & \vdots & \ddots & \vdots\\
s(-T) & ... & ... & s(0)
\end{pmatrix}
$$
where $s(t)$ is $p\times p$ matrix.

Similarly, 
$$
R = \begin{pmatrix}
r(0) & r(1) & ... & r(T)\\
r(-1) & r(0) & ... & r(T-1)\\
\vdots & \vdots & \ddots & \vdots \\
r(-T) & ... & ... & r(0)
\end{pmatrix}
$$

Therefore the product $SR$ can be estimated as:
$$
SR= \begin{pmatrix}
\sum_{t=0}^{T} s(t)r(-t) & \sum_{t=0}^{T}s(t)r(1-t) & ... & \sum_{t=0}^Ts(t)r(T-t)\\
\sum_{t=0}^{T} s(t-1)r(-t) & \sum_{t=0}^{T}s(t-1)r(1-t) & ... & \sum_{t=0}^{T}s(t-1)r(T-t) \\
\vdots & \vdots & \ddots & \vdots \\
\sum_{t=0}^{T}s(t-T)r(-t) & \sum_{t=0}^{T}s(t-T)r(1-t) & ... & \sum_{t=0}^{T}s(t-T)r(T-t)
\end{pmatrix}
$$



6. Estimating $\tau$ parameters:
$$
\log p(X) = -\frac{1}{2} \sum_{i=1}^{p} \left[ T \log(2\pi) + \log |K_i|+\text{trace}(K_i^{-1}X_iX_i^T)\right]
$$

where $X_i = [x_i(-n), x_i(-n+1), ..., x_i(T)], T_e = n+1+T$

Therefore,
$$
X_iX_i^T = \pmatrix{x_i(-n)x_i(-n) & x_i(-n)x_i(-n+1) & ... & x_i(-n)x_i(T)\\
x_i(-n+1)x_i(-n) & x_i(-n+1)x_i(-n+1) & ... & x_i(-n+1)x_i(T)\\
\vdots & \vdots & \ddots & \vdots\\
x_i(T)x_i(-n) & x_i(T)x_i(-n+1) & ... & x_i(T)x_i(T)}
$$


$$
K_i^{-1}=\pmatrix{a(0) & a(1) & ... & a(k)\\
a(-1) & a(0) & ... & a(k-1)\\
\vdots & \vdots & \ddots & \vdots\\
a(-k) & a(-k+1) & ... & a(0)}
$$


### Date: 6th Feb, 2022
Let the length of fMRI time-series used be $nT_h$.
Let the length of latent time-series be $nT_e$.
Let the length of kernel which act as $K_i^{-1}$ be $nT_a$.

$nT_e=nT_h-nT_a+1-h+1$
and 
$nT_e=nT_a+nT+h-1$
$$
Y_{-\infty:T+h-1}|X_{-\infty:T} \sim \mathcal{N}(HCX+d,R)
\\
X_{-\infty:T}\sim \mathcal{N}(0, K)
\tag{1}$$


$$
\begin{eqnarray}
p(X) &=& \Pi_{i=1}^{p}\frac{1}{(2\pi)^{T_z/2}|K_i|^{1/2}}\exp\left(-X_i^TK_i^{-1}X_i\right)\\
\log p(X_i) &=& -0.5\left(\text{ln det}(K_i)+X_i^TK_i^{-1}X_i\right) + \text{const.}\\
&\geq& -0.5\left(\text{ln det}(K_i)^{Te/Tz} + X_{i,-nT_a/2:nT_a/2}^T\right)
\end{eqnarray}
$$


### Date: 1st March, 2022

![](https://i.imgur.com/pMFayBR.png)

a) Validate whether this sequence of transformations is valid in the sense that $HCX:q\times T_{eeh}$ reconstruct  a part of $Y:q\times T_h$.
b) If $T_{eeh} = 70$ then $T_{ee}$ should be of length $T_{ee} = T_{eeh} + h - 1$.
c) $T_{ee} = T_{e} - T_v + 1 \implies T_{e} = T_{ee} + T_v - 1$.
d) $T_h = T_e + h - 1 = T_{ee} + T_v + h - 2 = T_{eeh} + T_v + 2h -3$.
e) $T_h = 70 + 85 + 2 \times 18 - 3 = 155+36-3=188$.

















