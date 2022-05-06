# Review of Timescale separation of Information processing between sensory and associative regions - Sorrentino et al.,

### Questions asked in the study:
1. The authors were trying to understand that *how bottom-up processes and top-down processes integrate and thus helps in producing the behaviour of interpreting the external environment by the brain?* They answered this at the level of large-scale brain networks by functionally segregating brain regions that are involved in the first or second or involved in both, and they find regions which makes distinct sub-networks.
2. Specifically, they wanted to understand how information processing is organized in the brain and wanted to test their hypothesis that the information retained in the fluctuations of any two regions has a characteristic temporal scale.


### Methods:
The authors computed a measure that helps them quantify for a given pair of regions, the temporal scale during which some underlying computations (information processing) by these regions or get reflected in their dynamics. The authors first computed edge time-series using the z-scored signals
using the relation:
$$
E_{ij}(t)=\frac{(Y_A(t) - \bar{Y_A})}{\sigma_{Y_A}}\frac{(Y_B(t)-\bar{Y_B})}{\sigma_{Y_B}}
$$
where $Y_A$ is the activity of region $A$ and $Y_B$ of region $B$.

1. AMI - Auto Mutual Information is computed by first estimating the probability density function of $X(t)=E_{ij}(t)$ and its shifted version $X(t-\tau) = E_{ij}(t-\tau)$ and then computing mutual information
$$
I(X(t), X(t-\tau))=\text{Entropy}(X(t))+\text{Entropy}(X(t-\tau))-\\ \text{Entropy}([X(t), X(t-\tau)])
$$
for different values of $\tau$.

### Results and comments:
1. They computed AMI as the function of different delays $\tau$ and estimate the delay for which they get minimum AMI for each pair of region. They hypothesized that this is some unique property of given edge.
2. In Fig 1c, they see that there is trend across trials that consistently for some edges, the temporal unfolding of edge-timeseries decays much faster or slower. But their interpretation of high decay for some trials is not clear "Line 154: *strong tendency of high decay of top edges (sorted based on AMI minima) for some trials indicate that the average estimate conveys trial-specific information*".  
3. They were able to functionally segregates two sub-networks using AMI minima which they refer as SSN and LSN. From past studies, the unfolding of information in their activity in the regions of these sub-network makes sense i.e. SSN network is more involved in acquiring and transfer of information throughout the cortex and LSN network is more involved in making sense of these acquired information.
4. Next they show that AMI minima reflects some time-scale property of individual edges which is not just due to static correlations among the areas.
5. Their way of finding MSC (multi-storage core) based on multiplex network coefficient which is basically for a given node (region) $i$, they compute the degrees with which it connects to long-storage network ($k_i^{LSN}$) and the degrees with which it connects to short-storage network ($k_i^{SSN}$) may be faultly as for some region, where $k_i^{LSN}, k_i^{SSN}$ are both small but multiplex network coefficient would be close to zero.
6. In the discussion section, they mentioned potential role of MSC (mult-storage core) nodes which might helps in processing information presented to brain from SSN (in time-scales of the external environment) by also co-ordinating with regions in LSN (brain intrinsic timescale).
### Other comment:

Authors' hypothesized that "the information retained in the fluctations of any two regions has a characteristic temporal scale". This may be more better studied if we try to model large-scale brain activity using models such as used in Shagun et al., 2019 NIPS.

For example, fMRI activity of two brain regions $A$ and $B$ are modelled as follows:

$$
\begin{eqnarray}
Y_A&=&C^AX + d_A + \epsilon_A\tag{a}\\\tag{b}
Y_B&=&C^BX +d_B + \epsilon_B
\end{eqnarray}
$$
Here $X$ represent dynamics in the low-dimensional latent space where dynamics in each of the dimension reflects some computational aspect for producing the relevant behaviour out by the brain, $C^A, C^B$ will tells us about the weight with which each of latent computations is reflected or performed by these two pair of regions $A$ and $B$.

The co-activation time-series of region $A$ and $B$ defined by the author as follows:
$$
E_{ij}(t)=\frac{(Y_A(t) - \bar{Y_A})}{\sigma_{Y_A}}\frac{(Y_B(t)-\bar{Y_B})}{\sigma_{Y_B}}
$$
$$
E_{ij}(t)||E_{ij}(t-\tau)
\tag{1}
$$

can be approximated using $(a)$ and $(b)$ as 

$$
E_{ij}(t) \approx C^AX(t)C^BX(t)=C^AX(t)X(t)^T(C^B)^T
$$


Now, the time duration during which information evolves/retains in region $A$ and $B$ is the property of underlying latents timescales and the estimate we get ultimately depends upon 
$$
\begin{eqnarray}
C^AX(t)X(t)^T(C^B)^T &||& C^AX(t-\tau)X(t-\tau)^T(C^B)^T
\end{eqnarray}\tag{2}
$$
how $C^A, C^B$ in combination weight the underlying latents and **so the authors' estimate of temporal scale using mutual information of $(1)$ is related to temporal scales of different latents relevant to regions $A$ and $B$**.

Ref: 

Ajmera, Shagun, et al. "Infra-slow brain dynamics as a marker for cognitive function and decline." Advances in neural information processing systems 32 (2019): 6949.