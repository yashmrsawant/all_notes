# GPFA model with hemodynamic kernel convolution
We model fMRI activity as follows:
$$
y_i | X \sim \mathcal{N}(H_i \circledast C_{(i, :)} X_{q \times T} + d, r_i\mathbb{I})
$$
where $H_i$ is $1 \times h$ hemodynamic kernel which usually is modelled as difference of gamma distribution with 6 parameters governing the shape and dip of the hemodynamic kernel.

The latent priors are distributed as 
$$
x_i \sim \mathcal{N}(0, K_i)
$$
for each of $i = 1, ..., q$ and kernel matrix $K_i$ of size $T\times T$ is defined as
$$
K_i(m, n) = (1 - s)\exp\left(-\frac{1}{2}\gamma_i^2(m - n)^2\right) + s\delta_{m, n}
$$
where step-size is the amount by which each timepoint values are separated and $\gamma=\frac{\text{step-size}}{\tau_i}$.

$$
\begin{eqnarray}
p(Y|\theta)=&\frac{1}{(2\pi)^{D/2}|\Sigma_{Y}|^{1/2}}\exp\left(-\frac{1}{2}(Y-d)^T\Sigma_{Y}^{-1}(Y-d)\right)\\
p(Y|X, \theta)=& \frac{1}{(2\pi)^{qT/2}|R|^{1/2}} \exp(-\frac{1}{2}\left((Y_d - C_hX)^TR^{-1}(Y_d - C_hX)\right)\\
\mathbb{E}_{X \sim X|Y}\left[\log p(Y | X, \theta)\right]=& -0.5\left[qT\log(2\pi)+\log |R| + Y_d^TR^{-1}Y_d-Y_d^TR^{-1}C_HX+\frac{1}{2}\left\{\text{tr}(R^{-1}C_H\Sigma C_H^T)+\text{tr}(R^{-1}C_H XX^TC_H^T)\right\}\right]\\
\mathbb{E}_{X \sim X|Y}\left[\log p(X|\theta)\right] =& -0.5\left[pT\log(2\pi)+\log|K|+\frac{1}{2}\{\text{tr}(K^{-1}\Sigma) + \text{tr}(K^{-1}XX^T)\}\right]\\
\log p(Y|\theta)=&-0.5\left[qT\log(2\pi)+\log|\Sigma_Y|+(Y_d)^T\Sigma_{Y}^{-1}(Y_d)\right]
\end{eqnarray}
$$
where $\Sigma_Y=R+C_hKC_h^T$

$$
\Sigma_{Y}^{-1}=R^{-1}-R^{-1}C_h\left(K^{-1}+C_h^TR^{-1}C_h\right)^{-1}C_h^TR^{-1}
$$

$$
\begin{eqnarray}
\frac{1}{N}\sum_{i=1}^{N}\mathbb{E}_{X \sim X|Y}[\log p(X, Y_i|\theta)]=-0.5\left[\log|R|+\frac{1}{N}\sum_{i=1}^{N}Y_d^T(i)R^{-1}(Y_d(i)-C_HX(i))\\+0.5\frac{1}{N}\sum_{i=1}^{N}\text{tr}(R^{-1}C_H Z(i) C_H^T)+\log|K| + 0.5\frac{1}{N}\sum_{i = 1}^{N}\text{tr}(K^{-1}Z(i))\right]
\end{eqnarray}
$$
where $Z(i) = \Sigma + X_iX_i^T$

$$
C_h=H_{qT\times qT}C_{qT \times pT}
$$



$$
H = \begin{pmatrix}
h_1 & 0 & 0 & ... & 0\\
h_2 & h_1 & 0 & ... & 0\\
h_3 & h_2 & h_1 & ... & 0\\
\vdots & \vdots & \vdots & \ddots & 0\\
h_T & h_{T - 1} & h_{T-2} & ... & h_1
\end{pmatrix}
$$

$$
\text{tr}(R^{-1}C_HZ(i)C_H^T)=\|\text{flat}(C_H^{R^{-1/2}}Z^L(i))\|^2
$$


$$
\frac{1}{N}\sum_{i=1}^{N}\mathbb{E}[\log p(Y_i|\theta)]
$$


Structure of the code:
>**class Model**:
**attributes**: $p, q, C_{p \times q}, d_{p \times 1}, R_{p \times p}, \Gamma_{q \times 1}, T_{p \times 6}$
**methods**: 
    gpfa_CPrediction: returns tensor computing the part 
    $$
    \mathbb{E}_{X | Y=y_i}\left[\log p(Y|X)\right] \approx \sum_{i = 1}^{n} \log p(Y=y_i|X_i)
    $$
    
#### 
```
class ModelInference(object):
    def __init__(self, binWidth, ks, xDim, yDim, nT, dtype_ = tf.float64):
        self.C = tf.placeholder(dtype = dtype_, shape = [yDim, xDim])
        self.d = tf.placeholder(dtype = dtype_, shape = [yDim, 1])
        self.R = tf.placeholder(dtype = dtype_, shape = [yDim, yDim])
        self.gamma = tf.placeholder(dtype = dtype_, shape = [xDim, 1])
        self.eps = tf.placeholder(dtype = dtype_, shape = [xDim, 1])
        self.H_t = tf.placeholder(dtype = dtype_, shape = [yDim, 6])
        self.R_log = tf.reshape(tf.log(tf.diag_part(self.R)), shape = [yDim, 1])
        self.eps_log = tf.log(self.eps)
        self.gamma_log = tf.log(self.gamma)
        self.logH_t = tf.log(self.H_t)
        self.Y = tf.placeholder(dtype_, shape = [None, yDim, nT])
        self.xDim = xDim
        self.yDim = yDim
        self.stepsize = np.array([binWidth]).astype(np.float64)[0] # IN MILLISEC
        self.ks = np.array([ks]).astype(np.float64)[0] # IN SECS
        self.dtype = dtype_
        self.K_big, self.T1, self.X, self.V = self.Estep(nT)

    def getHemodynamicTensors(self):
    	dt = self.stepsize / 1000 # hemodynamic kernel stepsize in seconds
    	h = np.ceil(self.ks / dt).astype(np.int32)
    	D = tf.repeat(tf.reshape(tf.exp(self.logH_t[:, 5]), shape = [self.yDim, 1]), repeats = h, axis = 1)
    	U = tf.nn.relu(tf.reshape(tf.range(0, np.ceil(self.ks/dt), dtype = dtype_), shape = [1, h]) - D/dt)
    	P1byP3 = tf.repeat(tf.reshape(\
    		tf.divide(tf.exp(self.logH_t[:, 1-1]) + tf.exp(self.logH_t[:, 3-1]), tf.exp(self.logH_t[:, 3-1])), shape = [self.yDim, 1]), \
    		repeats = h, axis = 1)
    	dtbyP3 = tf.repeat(tf.reshape(dt/tf.exp(self.logH_t[:, 3-1]), shape = [self.yDim, 1]), repeats = h, axis = 1)
    	P2byP4 = tf.repeat(tf.reshape(\
    		tf.divide(tf.exp(self.logH_t[:, 2-1]) + tf.exp(self.logH_t[:, 4-1]), tf.exp(self.logH_t[:, 4-1])), shape = [self.yDim, 1]), \
    		repeats = h, axis = 1)
    	dtbyP4 = tf.repeat(tf.reshape(dt/tf.exp(self.logH_t[:, 4-1]), shape = [self.yDim, 1]), repeats = h, axis = 1)
    	_1byP5 = tf.repeat(tf.reshape(1/tf.exp(self.logH_t[:, 5-1]), shape = [self.yDim, 1]), repeats = h, axis = 1)
    
    	S1 = tf.exp(tf.multiply((P1byP3 - 1), tf.log(U)) + tf.multiply(P1byP3, tf.log(dtbyP3)) - \
    		tf.multiply(dtbyP3, U) - tf.math.lgamma(P1byP3))
    	S2 = tf.exp(tf.multiply((P2byP4 - 1), tf.log(U)) + tf.multiply(P2byP4, tf.log(dtbyP4)) - \
    		tf.multiply(dtbyP3, U) - tf.math.lgamma(P2byP4))
    	fU = S1 - tf.multiply(S2, _1byP5)
    	H = tf.divide(fU, tf.reshape(tf.math.reduce_sum(fU, axis = 1), shape = [self.yDim, 1]))
    	return H
    def blkDiag(self, list_tensors):
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
            row_blocks.append(tf.pad(tensor = tensor, paddings = tf.concat([tf.zeros([tf.rank(tensor) - 1, 2], dtype = tf.int32),\
    																				 [(row_before_length, row_after_length)]], axis = 0)))
        blocked = tf.concat(row_blocks, -2)
        blocked.set_shape([blocked_rows, blocked_cols])
        return blocked
    def getK_big_inv(self, nT):
        Ts = tf.repeat(tf.reshape(tf.range(0, nT, dtype = self.dtype), shape = [1, nT]), repeats = self.xDim, axis = 0)
        TgammaSq = tf.exp(-1 * tf.square(Ts * tf.reshape(tf.exp(self.gamma_log), shape = [self.xDim, 1])))
        TgammaSq_scaled = TgammaSq * tf.reshape(1 - tf.exp(self.eps_log), shape = [self.xDim, 1]) + \
            tf.pad(tf.exp(self.eps_log), tf.constant([[0, 0], [0, nT - 1]]))
        K_big_toep = tf.linalg.LinearOperatorToeplitz(TgammaSq_scaled, TgammaSq_scaled, \
                                                      is_non_singular = True, is_positive_definite = True, is_square = True, is_self_adjoint = True)
        K_big_toep_chol = K_big_toep.cholesky()
        K_big_toep_chol_inv = K_big_toep_chol.inverse().to_dense()
        K_big_inv_T = []
        for i in range(self.xDim):
            K_big_inv_T.append(tf.reshape(tf.matmul(tf.transpose(K_big_toep_chol_inv[i, :, :]), \
                                                    K_big_toep_chol_inv[i, :, :]), shape = [nT * nT, 1]))
        K_big_inv_fl = tf.reshape(tf.concat(K_big_inv_T, axis = 1), shape = [self.xDim * nT * nT])
        indices = np.zeros((self.xDim * nT * nT, 2)).astype(np.int32)
        count = 0
        for i in range(nT):
            for j in range(nT):
                ixs = np.arange(0, self.xDim).astype(np.int32)
                indices[count : count + self.xDim, 0] = ixs + i * self.xDim
                indices[count : count + self.xDim, 1] = ixs + j * self.xDim
                count += self.xDim
        K_big_inv = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(indices, K_big_inv_fl, dense_shape = [xDim * nT, xDim * nT])))
        K_big = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(indices, \
                            tf.reshape(tf.transpose(K_big_toep.to_dense(), perm = [1, 2, 0]), shape = [xDim * nT * nT]), dense_shape = [xDim * nT, xDim * nT])))

        return K_big, K_big_inv
    def Estep(self, nT):
        h = np.ceil(1000 * self.ks / self.stepsize).astype(np.int32)
        R_invYd = (self.Y - tf.repeat(self.d, repeats = nT, axis = 1)) *\
                  tf.repeat(tf.exp(-1 * self.R_log), repeats = nT, axis = 1)
        paddings = tf.constant([[0, 0], [0, 0], [0, h-1]])
        H = self.getHemodynamicTensors()
        HtR_invYd = tf.transpose(tf.nn.conv1d(tf.transpose(tf.pad(R_invYd, paddings), perm = [0, 2, 1]), \
                                              tf.expand_dims(tf.transpose(H), 1), \
                                                  padding = 'VALID'), perm = [0, 2, 1])
        CtHtR_invYd = tf.matmul(tf.transpose(self.C), HtR_invYd)
        
        C_nTcopies = []
        for i in range(nT):
            C_nTcopies.append(self.C)
        
        C_bar = tf.transpose(tf.reshape(tf.transpose(self.blkDiag(C_nTcopies)), \
                                        shape = [self.xDim * nT, nT, self.yDim]), perm = [0, 2, 1])
        R_invC_bar = C_bar * tf.repeat(tf.exp(-1 * self.R_log), repeats = nT, axis = 1)
        paddings_t = tf.constant([[0, 0], [0, 0], [h - 1, 0]])
        H_f = tf.reverse_v2(H, [-1])
        R_invHC_bar = tf.transpose(tf.nn.conv1d(tf.transpose(tf.pad(R_invC_bar, paddings_t), perm = [0, 2, 1]), \
                                                tf.expand_dims(tf.transpose(H_f), 1), \
                                                    padding = 'VALID'), perm = [0, 2, 1])
        HtR_invHC_bar = tf.transpose(tf.nn.conv1d(tf.transpose(tf.pad(R_invHC_bar, paddings), perm = [0, 2, 1]), \
                                                  tf.expand_dims(tf.transpose(H), 1), \
                                                      padding = 'VALID'), perm = [0, 2, 1])
        CtHtR_invHC_bar = tf.matmul(tf.transpose(self.C), HtR_invHC_bar)
        CtHtR_invHC_bar = tf.layers.flatten(tf.transpose(CtHtR_invHC_bar, perm = [0, 2, 1]))
        
        K_big, K_big_inv = self.getK_big_inv(nT)
        Vsm = tf.linalg.inv(K_big_inv + CtHtR_invHC_bar)
        T1 = tf.matmul(K_big, tf.eye(self.xDim * nT, dtype = self.dtype) - tf.matmul(CtHtR_invHC_bar, Vsm))
        Xsm = tf.matmul(T1, tf.transpose(tf.layers.flatten(tf.transpose(CtHtR_invYd, perm = [0, 2, 1]))))
        return K_big, T1, tf.transpose(Xsm), Vsm

```
Computing $Q = \mathbb{E}[\log p(X,Y|\theta)]$ w.r.t to X|Y
$Q = \mathbb{E}[\log p(X)] + \mathbb{E}[\log p(Y|X)]$
$L = \sum_{i=1}^{N} \log p(Y_i|\theta)$


```python=
tf.reset_default_graph()
H_t = tf.placeholder(dtype_, shape = [yDim, 6])
C = tf.placeholder(dtype_, shape = [yDim, xDim])
d = tf.placeholder(dtype_, shape = [yDim, xDim])
R = tf.placeholder(dtype_, shape = [yDim, yDim])
gamma = tf.placeholder(dtype_, shape = [xDim, 1])
eps = tf.placeholder(dtype_, shape = [xDim, 1])

R_log = tf.reshape(tf.log(tf.diag_part(R)), shape = [yDim, 1])
eps_log = tf.reshape(tf.log(eps))
gamma_log = tf.reshape(tf.log(gamma))
logH_t = tf.log(H_t)

X = tf.placeholder(dtype_, shape = [None, xDim, nT])
Y = tf.placeholder(dtype_, shape = [None, yDim, nT])


# Defining \Sigma_{X|Y} temporary as tensor placeholder
Vsm = tf.placeholder(dtype_, shape = [xDim * nT, xDim * nT])
Vsm_chol = tf.placeholder(dtype_, shape = [xDim * nT, xDim * nT])


Yd = Y - d
paddings_t = tf.constant([[0, 0], [0, 0], [h-1, 0]])
H = getHemodynamicTensors()
H_f = tf.reverse_v2(H, [-1])

C_nTcopies = []
for i in range(nT):
    Ct_nTcopies.append(tf.transpose(C))

C_bar = tf.transpose(tf.reshape(blkDiag(C_nTcopies), \
            shape = [xDim * nT, nT, yDim]), perm = [0, 2, 1])

HC_bar = tf.transpose(tf.nn.conv1d(tf.transpose(tf.pad(C_bar, paddings_t), perm = [0, 2, 1]), \
            tf.expand_dims(tf.transpose(H_f), 1), padding = 'VALID'), perm = [0, 2, 1])

# E[log p(X)]
K_big_logdets, K_big, K_big_inv = getK_big_inv(nT)
X_flat = tf.expand_dims(tf.layers.flatten(tf.transpose(X, perm = [0, 2, 1])), axis = 2)
Zs = Vsm + tf.matmul(X_flat, tf.transpose(X_flat, perm = [0, 2, 1]))
Zs_chol = tf.linalg.cholesky(Zs)
XK_inv_X = 0.5 * tf.linalg.trace(tf.matmul(Zs, K_big_inv))
logdetK_big = tf.reduce_sum(K_big_logdets, axis = 0)

# E[log p(Y|X)]
Ch_X = tf.transpose(tf.nn.conv1d(tf.transpose(tf.pad(tf.matmul(C, X), paddings_t), perm = [0, 2, 1]),\
                                    tf.expand_dims(tf.transpose(H_f), 1), \
                                        padding = 'VALID'), perm = [0, 2, 1])
R_invYd = Yd * tf.repeat(tf.exp(-1 * R_log), repeats = nT, axis = 1)
R_invYd_flat = tf.layers.flatten(tf.transpose(R_invYd, perm = [0, 2, 1]))
R_inYd_Yd_minus_ChX = R_invYd_flat * (tf.layers.flatten(tf.transpose(Yd, perm = [0, 2, 1])) - \
                                        tf.layers.flatten(tf.transpose(C_hX, perm = [0, 2, 1])))

R0p5_invHC_bar = HC_bar * tf.repeat(tf.exp(-0.5 * R_log), repeats = nT, axis = 1)
R_inv_Ch_Z_Cht = 0.5 * tf.matmul(tf.transpose(Zs_chol, perm = [0, 2, 1]), \
                            tf.reshape(tf.transpose(R0p5_invHC_bar, perm = [0, 2, 1]), \
                                        shape = [xDim * nT, yDim * nT]))
logdetR_big = nT * tf.reduce_sum(R_log, axis = 0)

# Sum log p(Y_i); L being covariance matrix of Y
R_invC_bar = C_bar * tf.repeat(tf.exp(-1 * R_log), repeats = nT, axis = 1)
R_invHC_bar = tf.transpose(tf.nn.conv1d(tf.transpose(tf.pad(R_invHC_bar, paddings_t), perm = [0, 2, 1]), \
                                            tf.expand_dims(tf.transpose(H_f), 1), \
                                                padding = 'VALID'), perm = [0, 2, 1])
R_invHC_bar_flat = tf.transpose(tf.layers.flatten(tf.transpose(R_invHC_bar, perm = [0, 2, 1])))
term1 = tf.matmul(tf.layers.flatten(tf.transpose(Yd, perm = [0, 2, 1])), R_invHC_bar_flat)
YdL_invYd = R_invYd_flat * tf.layers.flatten(tf.transpose(Yd, perm = [0, 2, 1]))
YdL_invYd = YdL_invYd - tf.reduce_sum(tf.square(tf.matmul(term1, tf.linalg.cholesky(Vsm))), axis = 1)
logdet_L = logdetK_big + logdetR_big - tf.reduce_sum(tf.log(tf.diag_part(Vsm_chol)))
```

# [2]: Checking E and M steps


In probabilistic PCA,

$$
\textbf{z} \sim \mathcal{N}(0, I)
$$
and
$$
\textbf{x}|\textbf{z} \sim \mathcal{N}(W\textbf{z}+\mu, \sigma^2I)
$$

Importantly, we can view the probabilistic PCA model from a generative viewpoint in which a sampled value of the observed variable is obtained as follows:

$$
\textbf{x} = W\textbf{z} + \mu + \epsilon
$$
where $\epsilon \sim \mathcal{N}(0, \sigma^2I)$.

$$
\mathbb{E}[\textbf{x}\textbf{x}^t]=W\mathbb{E}[\textbf{z}\textbf{z}^t]W^t+\sigma^2I=WW^t+\sigma^2I
$$

Therefore predictive distribution $p(\textbf{x})$ can be defined as 
$$
\textbf{x} \sim \mathcal{N}(\mu, WW^t+\sigma^2I)
$$

The posterior distribution of $\textbf{z}$ can be defined as

$$
\textbf{z}|\textbf{x} \sim \mathcal{N}(M^{-1}W^t(\textbf{x}-\mu), \sigma^{-2}M)
$$
where $M=W^tW+\sigma^2I$.

We can then determine the model parameters using maximum likelihood method.

$$
\ln p(\textbf{X}|\mu,W, \sigma^2)=\sum_{n=1}^{N}\ln p(x_n|W, \mu, \sigma^2)
$$
where $X = {x_n}$.

$$
\begin{eqnarray}
\ln p(\textbf{X}|\mu, W, \sigma^2)&=& -\frac{1}{2}\left[Nq\ln (2\pi)+N\ln|C|+\sum_{n=1}^{N}(x_n-\mu)^tC^{-1}(x_n - \mu)\right]
\\
&=&-\frac{1}{2}\left[Nq\ln(2\pi) +N\ln |C| + N\textbf{tr}(C^{-1}S)\right]
\end{eqnarray}
$$


where $C = WW^t+\sigma^2I$ and $S = \sum_{n=1}^{N}(x_n-\mu)(x_n-\mu)^t/N$.

The complete-data likelihood function takes the form

$$
\ln p(\textbf{X}, \textbf{Z}|\mu, W, \sigma^2)=\sum_{n=1}^{N} \left(\ln p(x_n|z_n)+\ln p(z_n)\right)
$$



## GPFA
$$
\textbf{x} \sim \mathcal{N}(0, \bar{K})
$$


$$
\textbf{y}|\textbf{x}\sim \mathcal{N}(\bar{C_h}\textbf{x}+d, \bar{R})
$$


$$
\textbf{x}|\textbf{y}\sim \mathcal{N}(M(\textbf{y} - d), \bar{K}-M\bar{C_h}\bar{K})
$$
where $M = \bar{K}\bar{C_h}^t(\bar{C_h}\bar{K}\bar{C_h}^t+\bar{R})^{-1}$.

$$
\textbf{y}\sim \mathcal{N}(d, \bar{C}\bar{K}\bar{C}^t+\bar{R})
$$

We can now determine the complete-data log likelihood function

$$
\ln p(\textbf{X}, \textbf{Y}|d, C, H_t, \Gamma, r) = \sum_{n=1}^{N}\left[\ln p(y_n|x_n) + \ln p(x_n)\right]
$$

where $\Gamma$ is gamma parameter governing gaussian process dynamics, $\bar{C_h} =\bar{H}\bar{C}$ and $r$ is channel noise variance.

$$
\ln p(y_n|x_n)=-0.5\left[qT\ln(2\pi)+\left((y_n-d)-C_hx_n)^tR^{-1}((y_n-d)-C_hx_n)\right)\right]
$$

Now expectation with respect to the posterior distribution over the latent variables:
As
$$
\mathbb{E}_{x_n \sim X|y_n}[x_nx_n^t]=V+\mathbb{E}_{x_n \sim X|y_n}[x_n]\mathbb{E}_{x_n \sim X|y_n}[x_n^t]
$$
where $V=(C_h^tR^{-1}C_h+K^{-1})^{-1}$.

\begin{eqnarray}
\mathbb{E}_{x_n\sim X|y_n}\left[\ln p(y_n|x_n)\right]&=&-0.5\left[qT\ln(2\pi)+\text{logdet}(R)+(y_n-d)^tR^{-1}(y_n-d)-2\mathbb{E}_{x_n\sim X|y_n}[x_n^t]C_h^tR^{-1}(y_n-d)+\\\text{tr}(\mathbb{E}_{x_n \sim X|y_n}[x_nx_n^t]C_h^tR^{-1}C_h)\right]\\
&=&-0.5\left[qT\ln(2\pi)+\text{logdet}(R)+\left((y_n-d)-C_h\mathbb{E}_{x_n\sim X|y_n}[x_n])^tR^{-1}((y_n-d)-C_h\mathbb{E}_{x_n\sim X|y_n}[x_n])\right)\\+\text{tr}(VC_h^tR^{-1}C_h)\right]\\
&=&-0.5\left[qT\ln(2\pi)+\text{logdet}(\textbf{R})+((y_n-\textbf{d})-\textbf{C}_{\textbf{h}}\mu_{x_n})^t\textbf{R}^{-1}((y_n-\textbf{d})-\textbf{C}_{\textbf{h}}\mu_{x_n})+\\\text{tr}(V\textbf{C}_{\textbf{h}}^t\textbf{R}^{-1}\textbf{C}_{\textbf{h}})\right]
\end{eqnarray}



$$
\begin{eqnarray}
\mathbb{E}_{x_n\sim X|y_n}\left[\ln p(x_n)\right]&=&-0.5[pT\ln(2\pi)+\text{logdet}(K)+\text{tr}(K^{-1}\mathbb{E}_{x_n \sim X|y_n}[x_n]\mathbb{E}_{x_n \sim X|y_n}[x_n^t]) + \text{tr}(K^{-1}V)]\\
&=&-0.5\left[pT\ln(2\pi)+\text{logdet}(\textbf{K})+\text{tr}(\textbf{K}^{-1}\mu_{x_n}\mu_{x_n}^t)+\text{tr}(\textbf{K}^{-1}V)\right]
\end{eqnarray}
$$

$$
\begin{eqnarray}
\sum_{n=1}^{N}\ln p(y_n|\theta)&=&\sum_{n=1}^{N}-0.5[qT\ln(2\pi)+\text{logdet}(L)+(y_n-d)L^{-1}(y_n-d)]\\
L&=&C_hKC_h^t+R\\
L^{-1}&=&R^{-1}-R^{-1}C_h(K^{-1}+C_h^tR^{-1}C_h)^{-1}C_h^tR^{-1}\\
&=&\sum_{n=1}^{N}-0.5[qT\ln(2\pi)+\text{logdet}(L)+(y_n-d)^tR^{-1}(y_n-d)-(y_n-d)^tR^{-1}C_hVC_h^tR^{-1}(y_n-d)]\\
&=&\sum_{n=1}^{N}-0.5\left[qT\ln(2\pi)+\text{logdet}(L)+(y_n-d)^{t}R^{-1}(y_n-d)-\|V_{chol}C_h^tR^{-1}(y_n-d)\|^2\right]
\end{eqnarray}
$$

# 

Data likelihood

$$
\int p(Y|\theta)\log p(Y|\theta)dY\approx\frac{1}{N}\sum_{Y_{n}\sim p(Y|\theta)}\log p(Y_n|\theta)
$$


Also,

$$
\log p(Y|\theta) \geq \int q(X) \log \frac{p(X,Y|\theta)}{q(X)} dX
$$

Putting $q(X)$ as posterior distribution of $X$ given $Y$ i.e. $q(X)= p(X|Y,\theta)$ makes $\log p(Y|\theta)$ equal to right hand side.

Using Jensen's inequality
$$
-\log p(Y_n|\theta)=-\int p(X|Y_n,\theta) \log \frac{p(X,Y_n|\theta)}{p(X|Y_n,\theta)}dX\geq -\log \int p(X,Y_n|\theta) dX\geq-\log c
$$
where $c<1$.

$$
\sum_{Y_n \sim p(Y|\theta)}\log p(Y_n|\theta)\leq -N\log c
$$


#### Fitting C, d and R


$$
y_i^{(1:T)}\approx C^ix_{-i}H_i^t+d_i+\epsilon_i
$$

where $\epsilon_i \sim \mathcal{N}(0, r_i^2 I)$.


1) Compute $\nu_{-i}^{(1 : T)}=x_{-i}H_i^t$ using the expectation step.
2) $\hat{y}_i(t)=C^i\nu_{-i}(t)+d_i$
3) Fit $C^i, d^i$ using above data to minimize $\|y_i-\hat{y}_i\|^2$.
4) Use estimated $C^i, d^i$ to compute

$$
M = \sum_{t=1}^{T} -0.5\left[\log(2\pi)+\log(r_i^2)+\frac{(y_i(t)-d_i-C^i\nu_{-i}(t))^2}{r_i^2}\right]
$$

Differentiating $M$ w.r.t $r_i$, we get

$$
\frac{\partial M}{\partial r_i} = -0.5\sum_{t=1}^T\frac{2}{r_i^2}-2\frac{(y_i(t)-d_i-C^i\nu_{-i}(t))^2}{(r_i^3)}
$$

equating $\partial M/\partial r_i = 0$.

$$
r_i =\frac{\sum_{t=1}^T(y_i(t)-d_i-C^i\nu_{-i}(t))^2}{T}
$$



```python=
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 21:03:19 2021

@author: yashm
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
dtype_ = tf.float64
# In[]:
stepsize = 720 # In milliseconds
ks = 32 # In seconds
xDim = 2
yDim = 6
dt = stepsize / 1000 # hemodynamic kernel stepsize in seconds
h = np.ceil(ks / dt).astype(np.int32)
nT = 50
params = {'C' : np.random.randn(yDim, xDim), 'H_t' : np.random.rand(yDim, 6) * 2}
# In[]:
    
def getHemodynamicTensors(logH_t):
    dt = stepsize / 1000 # hemodynamic kernel stepsize in seconds
    h = np.ceil(ks / dt).astype(np.int32)
    D = tf.repeat(tf.reshape(tf.exp(logH_t[:, 5]), shape = [yDim, 1]), repeats = h, axis = 1)
    U = tf.nn.relu(tf.reshape(tf.range(0, np.ceil(ks/dt), dtype = dtype_), shape = [1, h]) - D/dt) + tf.constant([1e-100], dtype = dtype_)
    P1byP3 = tf.repeat(tf.reshape(\
		tf.divide(tf.exp(logH_t[:, 1-1]) + tf.exp(logH_t[:, 3-1]), tf.exp(logH_t[:, 3-1])), shape = [yDim, 1]), \
		repeats = h, axis = 1)
    dtbyP3 = tf.repeat(tf.reshape(dt/tf.exp(logH_t[:, 3-1]), shape = [yDim, 1]), repeats = h, axis = 1)
    P2byP4 = tf.repeat(tf.reshape(\
		tf.divide(tf.exp(logH_t[:, 2-1]) + tf.exp(logH_t[:, 4-1]), tf.exp(logH_t[:, 4-1])), shape = [yDim, 1]), \
		repeats = h, axis = 1)
    dtbyP4 = tf.repeat(tf.reshape(dt/tf.exp(logH_t[:, 4-1]), shape = [yDim, 1]), repeats = h, axis = 1)
    _1byP5 = tf.repeat(tf.reshape(1/tf.exp(logH_t[:, 5-1]), shape = [yDim, 1]), repeats = h, axis = 1)

    S1 = tf.exp(tf.multiply((P1byP3 - 1), tf.log(U)) + tf.multiply(P1byP3, tf.log(dtbyP3)) - \
		tf.multiply(dtbyP3, U) - tf.math.lgamma(P1byP3))
    S2 = tf.exp(tf.multiply((P2byP4 - 1), tf.log(U)) + tf.multiply(P2byP4, tf.log(dtbyP4)) - \
		tf.multiply(dtbyP3, U) - tf.math.lgamma(P2byP4))
    fU = S1 - tf.multiply(S2, _1byP5)
    
    H = tf.divide(fU, tf.reshape(tf.math.reduce_sum(fU, axis = 1), shape = [yDim, 1]))
    return H

# In[]:
tf.reset_default_graph()
X = tf.placeholder(dtype_, shape = [None, xDim, nT])
C = tf.placeholder(dtype_, shape = [yDim, xDim])
Cx = tf.expand_dims(tf.transpose(tf.matmul(C, \
        tf.transpose(tf.reshape(X, shape = [-1, nT, xDim]), perm = [0, 2, 1])),\
                             perm = [0, 2, 1]), axis = 1)
p6 = tf.placeholder(dtype_, shape = [yDim * 6]) 
H = getHemodynamicTensors(tf.reshape(p6, shape = [yDim, 6]))
p6_a = tf.Variable(tf.random_normal([yDim * 6], dtype = dtype_))
H_a = getHemodynamicTensors(tf.reshape(p6_a, shape = [yDim, 6]))
paddings = tf.constant([[0, 0], [0, 0], [0, h-1], [0, 0]])
paddings_t = tf.constant([[0, 0], [0, 0], [h-1, 0], [0, 0]])
Ch_X = tf.squeeze(tf.nn.depthwise_conv2d(tf.pad(Cx, paddings_t), \
        tf.expand_dims(tf.expand_dims(tf.transpose(tf.reverse_v2(H, [-1])), axis = 0), axis = 3), \
                              strides = [1, 1, 1, 1], padding = 'VALID'), axis = 1)
Ch_X_a = tf.squeeze(tf.nn.depthwise_conv2d(tf.pad(Cx, paddings_t), \
        tf.expand_dims(tf.expand_dims(tf.transpose(tf.reverse_v2(H_a, [-1])), axis = 0), axis = 3), \
                              strides = [1, 1, 1, 1], padding = 'VALID'), axis = 1)
n = tf.norm(Ch_X_a, axis = 1) * tf.norm(Ch_X_a, axis = 1)
f = tf.divide((tf.reduce_sum(Ch_X * Ch_X_a, axis = 1)), n)
E = tf.losses.mean_squared_error(tf.ones_like(f), f)

gradE_p6_a = tf.gradients(E, p6_a)
# In[]:

train_step = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(E)
# In[]:

X_f = np.random.randn(20000, xDim, nT)
X_f_test = np.random.randn(50, xDim, nT)
feed_dict_ = {C : params['C'], p6 : np.log(params['H_t'].reshape((yDim * 6)))}
try:
    sess.close()
except:
    print("Error closing session")
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

feed_dict_[X] = X_f[:1000, :, :]
grad_E_p6_a_v = sess.run(gradE_p6_a, feed_dict = feed_dict_)
print(np.sum(np.abs(grad_E_p6_a_v)))

# In[]:
p6_a.load(np.log(params['H_t'].reshape((yDim * 6))), sess)
feed_dict_[X] = X_f_test
f_v = sess.run(f, feed_dict = feed_dict_)
# In[]:
    
H_v, H_a_v, Ch_X_v, Ch_X_a_v = sess.run([H, H_a, Ch_X, Ch_X_a], feed_dict = feed_dict_)

ix = np.random.randint(50)
yDim_i = np.random.randint(yDim)

plt.plot(H_v[yDim_i, :])
plt.plot(H_a_v[yDim_i, :])

plt.plot(Ch_X_v[ix, :, yDim_i])
plt.plot(Ch_X_a_v[ix, :, yDim_i])
# In[]:

idxs = np.arange(200)
errors = np.zeros((0, 2))
for i in range(10000):
    np.random.shuffle(idxs)
    feed_dict_[X] = X_f[idxs[:1000], :, :]
    # grad_E_p6_a_v = sess.run(gradE_p6_a, feed_dict = feed_dict_)
    sess.run(train_step, feed_dict = feed_dict_)
    
    feed_dict_[X] = X_f_test
    E_v, p6_a_v = sess.run([E, p6_a], feed_dict = feed_dict_)
    H_t_a_v = np.exp(p6_a_v).reshape((yDim, 6))
    print("%d-> E = %f, %f"%(i, E_v, np.sum(np.abs(H_t_a_v - params['H_t']))))
    errors = np.append(errors, np.array([[np.sum(np.abs(H_t_a_v - params['H_t'])), \
                                          E_v]]), axis = 0)
    
    
plt.plot(errors[:, 0])
plt.plot(errors[:, 1])
# for i in range(yDim):
#     plt.plot(H_a_v[i, :])
#     plt.show()
    
```

#### Estimation of HRF for $q$ regions


Assuming that HRF profile vary smoothly over time, the iterative procedure to minimize cost $C$ to estimate $H_{q \times T}$ where $H$ is a matrix containing T samples of HRF function for each of the $q$ region.


$$
C=-\sum_{i=1}^{N}\mathbb{E}_{x \sim X|Y=y_i}[\log p(X=x, Y=y_i|\theta)]+\lambda\sum_{i=1}^{q}\nabla^2H_{i,1:T}
$$

where 

$$
\nabla^2H_{i, 1 : T} \approx \sum_{t=2}^{T-1}(H_{i, t+1}+H_{i, t-1})- 2 * H_{i, t}
$$

The first term in $C$ is negative of expected complete-data likelihood where expectation runs over conditional $X|Y = y_i$. The second term is smoothness cost for $H$ to vary smoothly over time.

### Updating C, d and R


Model:


$$
Y_{qT\times 1}|X_{pT \times 1} \sim \mathcal{N}(\bar{H}\bar{C}X+\bar{d}, \bar{R})\\
X \sim \mathcal{N}(0, \bar{K}_{pT \times pT})
$$

$$
\begin{eqnarray}
\log p(Y|X)&=&-0.5\left[qT\log(2\pi)+\log|R|+(Y_d-C_hX)^TR^{-1}(Y_d-C_hX)\right]\\
&=&-0.5[qT\log(2\pi)-\log|R^{-1}|+Y_d^TR^{-1}Y_d-2X^TC_h^TR^{-1}Y_d+\text{Tr}(C_h^TR^{-1}C_hXX^T)]
\end{eqnarray}
$$

$$
\begin{eqnarray}
\text{loss}(L)&=&-\sum_{n=1}^{N} \mathbb{E}_{x_n \sim X|Y=y_n}[\log p(X=x_n, Y=y_n|\theta)]\\
&=&-\sum_{n=1}^{N}\left(\mathbb{E}_{x_n \sim X|Y=y_n}[\log p(Y=y_n|X=x_n)]+\mathbb{E}_{x_n \sim X|Y=y_n}[\log p(X=x_n)]\right)
\end{eqnarray}
$$

For $N$ trials where each $Y_n$ is $qT\times 1$ vector and $X$ is $pT \times 1$.
$$
\begin{pmatrix}\hat{C_h} & \hat{d}\end{pmatrix} = \left(\sum_{n = 1}^{N}Y_n \begin{pmatrix}\mathbb{E}_{X|Y}[X]^T & 1\\
\end{pmatrix}\right)\left(\sum_{n=1}^{N}\mathbb{E}_{X|Y}\begin{pmatrix}XX^T & X\\
X^T & 1\end{pmatrix}\right)^{-1}\tag{1}
$$

$$
R = \text{var}\left(Y-C_hX\right)\tag{2}
$$

$$
\bar{H}\bar{C} = \begin{pmatrix}
h_1C& 0 & 0 & ... & 0 \\
h_2C & h_1C & 0 & ... & 0\\
h_3C& h_2C & h_1C & ... & 0 \\
... & ... & ... & ... & ... &\\
0 & 0 & ... & ... & h_{1}C
\end{pmatrix}_{qT \times pT}
$$







Figure out this???
![](https://i.imgur.com/EdAZtG3.png)

#### 
Differentiation of $\mathbb{E}_{x \sim X|Y}[\log p(x)]$

Let $p_4 = \log (\Gamma)$
$$
\begin{eqnarray}
\frac{\partial{K_i}}{\partial p_4^i} &=& \frac{\partial K_i}{\partial \tau_i} \frac{\partial \tau_i}{\partial \gamma_i} \frac{\partial \gamma_i}{\partial p_4^i}\\
&=&\sigma^2_{f, i} \frac{(t_1 - t_2)^2}{\tau_i^3}\exp\left(-\frac{(t_1-t_2)^2}{2\tau_i^2}\right)(-0.5\tau_i^3)\exp(p_4^i)\\
&=&-\frac{1}{2}\sigma^2_{i,f}(t_1-t_2)^2\exp\left(-\frac{(t_1-t_2)^2}{2\tau_i^2}\right)\exp(p_4^i)
\end{eqnarray}
$$

$$
\begin{eqnarray}
z = \mathbb{E}_{x_n\sim X|y_n}\left[\ln p(x_n)\right]&=&-0.5[pT\ln(2\pi)+\text{logdet}(K)+\text{tr}(K^{-1}\mathbb{E}_{x_n \sim X|y_n}[x_n]\mathbb{E}_{x_n \sim X|y_n}[x_n^t]) + \text{tr}(K^{-1}V)]\\
&=&-0.5\left[pT\ln(2\pi)+\text{logdet}(\textbf{K})+\text{tr}(\textbf{K}^{-1}\mu_{x_n}\mu_{x_n}^t)+\text{tr}(\textbf{K}^{-1}V)\right]
\end{eqnarray}
$$


$$
\partial{K_i}/\partial \log(\gamma_i)=\frac{\partial{K_i}}{\partial{\tau_i}} \frac{\partial \tau_i}{\partial \log \gamma_i}
$$

$$
\gamma_i=1/\tau_i^2\implies \log (\gamma_i)=-2\log(\tau_i)
$$

$$
\tau_i=\exp(-0.5 \log (\gamma_i))
$$

$$
\partial{\tau_i}/\partial{\log (\gamma_i)}=-0.5\tau_i
$$

$$
\frac{\partial Z}{\partial K_i^{-1}} =-0.5\left[-K_i^{-1}+(\mu_{x_n}\mu_{x_n}^t + V)\right]
$$

Question: Why GPFA learning algortihm does not help to reach global optimum i.e. why it depends on the initialized values of the parameters specially timescale?



$$
O = \sum_{i=1}^{N}\log p(Y^{i})= D_{KL}(q(X|Y^{i})||p(X|Y^{i})) + \mathbb{E}_{q(X|Y)}\left[-\log q(X|Y) + \log p(X, Y)\right]
$$

As $q(X, Y), q(Y|X), q(X|Y)$ can be efficiently computed using equations A5, A6 (Yu's paper). We can use update form a 

### Deep Kalman Filters






