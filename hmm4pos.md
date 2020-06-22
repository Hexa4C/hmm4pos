# 信号与信息处理大作业

## 基于隐马尔可夫模型的词性标注

### 模型说明

本实验是使用的隐马尔可夫模型，

> 隐马尔可夫模型（hidden Markov model, HMM）是可用于标注问题的统计学习模型，描述由隐藏的马尔可夫链随机生成观测序列的过程，属于生成模型。
>
> ……
>
> 隐马尔可夫模型在语音识别、自然语言处理、生物信息、模式识别等领域有着广泛的应用。

——摘自《统计学习方法》李航 著

### 问题说明

本实验中使用的数据集是从[国家语委现代汉语语料库现代汉语语料库检索](http://corpus.zhonghuayuwen.org/CnCindex.aspx)中下载得到的标注好的中文语料。语料样例如下：

>天津/ns 队/n 在/p 同/p 解放军队/ni 的/u 比赛/v 中/nd ，/w 开场/v 打/v 得/u 沉闷/a ，/w 快攻/v 不够/a 顺手/a ，/w 阵地战/n 射门/v 不/d 果断/a ，/w 防守/v 上/nd 漏洞/n 较/d 多/a ，/w 被/p 对方/n ８/m 号/n 孟/a 优胜/v 、/w １１/m 号/n 金百炼/nh 的/u 内线/n 偷袭/v 和/c 外围/n 强打/v 连连/d 得手/v 。/w 

该语料都是分好词并在每个词语后面做标注的语料，每个标注的意义如下：

| 标注 | a            | aq           | as           | c              | d            |
| ---- | ------------ | ------------ | ------------ | -------------- | ------------ |
| 意义 | 形容词       | 性质形容词   | 状态形容词   | 连词           | 副词         |
| 标注 | e            | f            | g            | ga             | gn           |
| 意义 | 叹词         | 区别词       | 语素词       | 形容词性语素字 | 名词性语素字 |
| 标注 | gv           | h            | i            | ia             | ic           |
| 意义 | 动词性语素字 | 前接成分     | 习用语       | 形容词性习用语 | 连词性习用语 |
| 标注 | in           | iv           | j            | ja             | jn           |
| 意义 | 名词性习用语 | 动词性习用语 | 缩略语       | 形容词性缩略语 | 名词性缩略语 |
| 标注 | jv           | k            | m            | n              | nd           |
| 意义 | 动词性缩略语 | 后接成分     | 数词         | 名词           | 方位名词     |
| 标注 | ng           | nh           | ni           | nl             | nn           |
| 意义 | 普通名词     | 人名         | 机构名       | 处所名词       | 族名         |
| 标注 | ns           | nt           | nz           | o              | p            |
| 意义 | 地名         | 时间名词     | 其他专有名词 | 拟声词         | 介词         |
| 标注 | q            | r            | u            | v              | vd           |
| 意义 | 量词         | 代词         | 助词         | 动词           | 趋向动词     |
| 标注 | vi           | vl           | vt           | vu             | w            |
| 意义 | 不及物动词   | 联系动词     | 及物动词     | 能愿动词       | 其他         |
| 标注 | wp           | ws           | wu           | x              |              |
| 意义 | 标点符号     | 非汉字字符串 | 其他位置符号 | 非语素词       |              |

### 数据处理

数据处理的脚本见`data.py`，在脚本中，

1. 遍历整个语料，统计每一个词的词频，并得到标注和索引的两个映射的字典，`attr2idx`与`idx2attr`。
2. 将词典按词频排序，裁剪词典，删除所有只出现一次的词语，并加入`UNK`符号，之后用`UNK`符号来代替所有的词典中没有的词。
3. 由词典得到token和索引的两个映射的字典，`token2idx`与`idx2token`。
4. 重新遍历整个语料，并用相应的索引代替所有的token和标注。
5. 将所有的处理过的信息保存在`corpus_all.pkl`中。
   
   * a
   
     a1

在`hmmpos.py`中，读取`corpus_all.pkl`后，将数据集以4：1的比例分割为训练集和测试集。

### 模型实现

本实验的隐马尔可夫模型的实现主要参考《统计学习方法》中的算法。

1. 隐马尔可夫模型$$\lambda = (A, B, \pi)$$，

   * $$A$$是状态转移矩阵：
     $$
     A = [a_{ij}]_{N \times N}
     $$
     其中，
     $$
     a_{ij} = P(i_{i+1} = q_j | i_t = q_i), i = 1, 2, \cdots, N; j = 1, 2, \cdots , N
     $$
     即时刻$$t$$处于状态$$q_i$$的条件下在时刻$$t+1$$转移到状态$$q_j$$的概率。

   * $$B$$是观测概率矩阵：
     $$
     B = [b_j(k)]_{N\times M}
     $$
     其中，
     $$
     b_j(k) = P(o_t = v_k| i_t = q_j), k = 1, 2, \cdots, M; j = 1, 2, \cdots, N
     $$
     即时刻$$t$$处于状态$$q_j$$的条件下在时生成观测$$v_k$$的概率。

   * $$\pi$$是初始状态概率向量：
     $$
     \pi = (\pi_i)
     $$
     其中，
     $$
     \pi_i = P(i_1 = q_i), i = 1, 2, \cdots , N
     $$
     是时刻$$t = 1$$处于状态$$q_i$$的概率。
     
   * 具体代码实现，使用`numpy.array`来实现上述矩阵和向量：
   
     ```python
     self.transition = np.zeros((n_state, n_state))
     self.observe = np.zeros((n_state, n_observe))
     self.pi = np.zeros((n_state))
     self.n_state = n_state
     self.n_observe = n_observe
     ```
   
2. 有监督的模型参数估计

   * 状态转移矩阵估计
     $$
     transition[q_1][q_2] = \frac{训练集中相邻的标注二元组中，q_1q_2同时出现个数}{训练集中相邻的标注二元组中，第一个是q_1的个数}
     $$

   * 观测概率矩阵估计
     $$
     observe[q][t] = \frac{训练集中标注是q，观测到的词是t的个数}{训练集中标注是q的个数}
     $$

   * 初始状态概率向量估计
     $$
     \pi[q] = \frac{训练集中开头是标注q的个数}{训练集总句子数}
     $$

   * 代码实现如下：

     ```python
     def init_probmat(self, series):
         """
             @Args:
     
             series: list of series, each serie is a tuple of series of observations and states
             """
         self.transition = np.zeros((self.n_state, self.n_state))
         self.observe = np.zeros((self.n_state, self.n_observe))
         self.pi = np.zeros((self.n_state))
         n_series = len(series)
         trans_base = np.zeros((self.n_state))
         obs_base = np.zeros((self.n_state))
         for cop in tqdm(series):
             sent = cop[0]
             attr = cop[1]
             self.pi[attr[0]] += 1
             for i in range(len(sent) - 1):
                 self.transition[attr[i]][attr[i + 1]] += 1
                 trans_base[attr[i]] += 1
                 for i in range(len(sent)):
                     self.observe[attr[i]][sent[i]] += 1
                     obs_base[attr[i]] += 1
                     self.transition = self.transition / trans_base[:, None]
                     self.observe = self.observe / obs_base[:, None]
                     self.pi = self.pi / n_series
     ```

3. 概率计算的前向算法：

   * 算法细节：

     输入：隐马尔可夫模型$$\lambda$$，观测序列$$O$$；

     输出：观测序列概率$$P(O|\lambda)$$。

     （1）初值
     $$
     \alpha_1(i) = \pi_i b_i(o_1), i = 1, 2, \cdots , N
     $$
     （2）递推	对$$t = 1,2, \cdots, T - 1$$，
     $$
     \alpha_{t+1}(i) = [\sum_{j = 1}^N \alpha_t(j)a_{ji}]b_i(o_{t+1}), i = 1, 2, \cdots, N
     $$
     （3）终止
     $$
     P(O|\lambda) = \sum_{i = 1}^N \alpha_T(i)
     $$

   * 代码实现如下：

     ```python
     def forward(self, observations):
         alphas = np.zeros((1, self.n_state))
         states = list(range(len(self.n_state)))
         T = len(observations)
         # initialize
         for s in states:
             pi_s = self.pi[s]
             alphas[0][s] = pi_s * self.observe[s][observations[0]]
             # recursive
             for t in range(1, T):
                 prev_alphas = alphas
                 alphas = np.zeros((1, self.n_state))
                 for s in states:
                     alpha_each = np.sum(np.dot(prev_alphas, self.transition)) * self.observe[s][observations[t]]
                     alphas[0][s] = alpha_each
                     # terminate
                     prob = np.sum(alphas)
                     return prob
     ```

4. 使用动态规划求概率最大路径的维比特算法：

   * 算法细节：

     输入：模型$$\lambda = (A, B, \pi)$$和观测$$O = (o_1, o_2, \cdots, o_T)；$$

     输出：最优路径$$I^* = (i_1^*, i_2^*, \cdots, i_T^*)。$$

     （1）初始化
     $$
     \delta_1(i) = \pi_i b_i(o_1), \ \ i = 1, 2, \cdots, N \\
     \Psi_1(i) = 0, \ \ i = 1, 2, \cdots, N
     $$
     （2）递推	对$$t = 2, 3, \cdots, T$$
     $$
     \delta_t(i) = \max_{1\leq j\leq N}[\delta_{t - 1}(j)a_{ji}]b_i(o_t), \ \ i = 1, 2, \cdots, N \\
     \Psi_t(i) = \arg \max_{1\leq j \leq N}[\delta_{t - 1}(j)a_{ji}], \ \ i = 1, 2, \cdots, N \\
     $$
     （3）终止
     $$
     P^* = \max_{1\leq i \leq N} \delta_T(i) \\ 
     i^*_T = \arg \max_{1\leq i \leq N} [\delta_T(i) ]
     $$
     （4）最优路径回溯	对$$t = T - 1, T - 2, \cdots, 1$$
     $$
     i^*_t = \Psi_{t + 1}(i^*_{t+ 1})
     $$
     球的最优路径$$I^* = (i_1^*, i_2^*, \cdots, i_T^*)$$。

   * 代码实现如下：

     ```python
     def viterbi(self, observations):
         T = len(observations)
         states = list(range(self.n_state))
         delta = np.zeros((T, self.n_state))
         phi = np.zeros((T, self.n_state), dtype=int)
         for s in states:
             delta[0][s] = self.pi[s] * self.observe[s][observations[0]]
             phi[0][s] = 0
             for t in range(1, T):
                 for s in states:
                     max_arg = 0
                     max_prob = 0.0
                     for j in states:
                         p = delta[t - 1][j] * self.transition[j][s]
                         if p > max_prob:
                             max_prob = p
                             max_arg = j
                             delta[t][s] = max_prob * self.observe[s][observations[t]]
                             phi[t][s] = max_arg
                             max_final_prob = 0.0
                             max_final_state = 0
                             for s in states:
                                 if delta[T - 1][s] > max_final_prob:
                                     max_final_prob = delta[T - 1][s]
                                     max_final_state = s
                                     best_path = [max_final_state]
                                     for t in range(T - 1, 0, -1):
                                         prev_state = phi[t][best_path[-1]]
                                         best_path.append(prev_state)
                                         best_path = list(reversed(best_path))
                                         return best_path
     ```

### 实验结果

在上述设置下，得到的准确率是70.37%，查全率是70.37%。

  