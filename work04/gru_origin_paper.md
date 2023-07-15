

> Gl和Gr

在文中描述的grConv网络中，$G^l$和$G^r$是两个权重矩阵，用于计算左上下文和右上下文的贡献。具体来说，$G^l$用于计算左上下文的贡献，$G^r$用于计算右上下文的贡献。

在计算$h_j^{(t)}$时，左上下文的贡献由$h_{j-1}^{(t-1)}$和$G^lh_{j-1}^{(t-1)}$共同决定，其中$G^lh_{j-1}^{(t-1)}$表示将$h_{j-1}^{(t-1)}$与$G^l$相乘得到的向量。同理，右上下文的贡献由$h_{j+1}^{(t-1)}$和$G^r h_{j+1}^{(t-1)}$共同决定，其中$G^rh_{j+1}^{(t-1)}$表示将$h_{j+1}^{(t-1)}$与$G^r$相乘得到的向量。

因此，$G^l$和$G^r$的维度都是$d \times d$，其中$d$是隐藏单元的维度。需要注意的是，$G^l$和$G^r$是需要训练的参数，可以通过反向传播算法来学习得到最优的权重值。

> Wl、Wr

在文中描述的grConv网络中，$Wl$和$Wr$是两个权重矩阵，用于计算经过变换的输入的贡献。

在计算$h_j^{(t)}$时，经过变换的输入$h\tilde{(t)}_j$由输入向量$x_j$和$Wl$、$Wr$计算而得。具体来说，$h\tilde{(t)}_j$可以表示为：

$$h\tilde{(t)}_j = Wrx_j + Wlh_{j-1}^{(t)} + Wrh_{j+1}^{(t)}$$

其中，$Wl$和$Wr$的维度均为$d \times d$，$x_j$是输入序列中第$j$个元素对应的向量，$h_{j-1}^{(t)}$和$h_{j+1}^{(t)}$分别表示在前一个递归级别$t-1$中，第$j-1$个和第$j+1$个隐藏单元的激活。

需要注意的是，$Wl$和$Wr$是需要训练的参数，可以通过反向传播算法来学习得到最优的权重值。



The new activation $\tilde{h}_j^{(t)}$ is computed as usual:
$$
\tilde{h}_j^{(t)}=\phi\left(\mathbf{W}^l h_{j-1}^{(t)}+\mathbf{W}^r h_j^{(t)}\right)
$$
where $\phi$ is an element-wise nonlinearity.
The gating coefficients $\omega$ 's are computed by
$$
\left[\begin{array}{c}
\omega_c \\
\omega_l \\
\omega_r
\end{array}\right]=\frac{1}{Z} \exp \left(\mathbf{G}^l h_{j-1}^{(t)}+\mathbf{G}^r h_j^{(t)}\right)
$$
where $\mathbf{G}^l, \mathbf{G}^r \in \mathbb{R}^{3 \times d}$ and
$$
Z=\sum_{k=1}^3\left[\exp \left(\mathbf{G}^l h_{j-1}^{(t)}+\mathbf{G}^r h_j^{(t)}\right)\right]_k
$$