

1) Update equation for next hidden state (stRNN units):
$$
r_t= f(r_{t-1}W_h + x_tW_x + g_tW_g)
$$

2) Update equation for next hidden state (GI layer units):
$$
g_t=f(i_g(t)W_x^g+r_{t-1}W_s^g)\tag{1}
$$
or
$$
g_t = f(i_g(t)W_x^g+r_{t-1}W_s^g+g_{t-1}W_h^g)\tag{2}
$$
