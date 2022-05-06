# Relevance of recurrent II connections in GI layer for MC-stRNN

[[training mc-stRNN with GI layer having recurrent excitatory inputs from stRNN hidden units]]


### Main question:
Consider MC-stRNN under the influence of GI layer supports mnemonic coding representations in on-off situations i.e. off situation arise at timestep when globally there is demand for resetting the MC representations. 

Now, if we connect to mc-stRNN and GI layer via the two update operations as below:

$$
\begin{eqnarray}
r(t) &=& f(r_{t-1} W_h + x_t W_x + g_t W_g)\\
g(t) &=& f(g_{t-1}W_g^g + r^E_{t-1} W_s^g + i_g(t) W_x^g)
\end{eqnarray}
$$

$$$$


or 

$g(t)$ state can be update without recurrent I-I connections
$$
\begin{eqnarray}
r(t) &=& f(r_{t-1} W_h + x_t W_x + g_t W_g)\\
g(t) &=& f(r^E_{t-1} W_s^g + i_g(t) W_x^g)
\end{eqnarray}
$$


*What is difference between two types of mnemonic coding network i.e. with two different types of GI layer*
a) In persistence?
b) For training?



