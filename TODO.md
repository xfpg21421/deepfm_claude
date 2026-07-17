# TODO — DeepFM 论文复现（落地 deepmodel/arena 框架）

> 论文：`paper/deepfm.pdf` — Guo et al. 2017, *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction* (arXiv:1703.04247)
> 复现目标（论文 Table 2, Criteo）：**AUC 0.8007 / Logloss 0.45083**
>
> 本文件是唯一任务入口。执行规则见 `CLAUDE.md`：每次只做**第一个未完成任务**，做完自查 + 与论文比对 + 跑 pytest，然后更新本文件并提交。

---

## 一、论文 → 代码 追溯锚点表

任何实现必须落在此表某一行上；不在表内的方法一律不写（`CLAUDE.md`：不臆造论文没有提到的方法）。

| # | 论文要素 | 锚点 | 公式 / 原文 | 落地对应 |
|---|---|---|---|---|
| A1 | 输入 `x=[x_field1,…,x_fieldm]`，m 个 field，d 维稀疏 | §2 | 类别 field → one-hot；连续 field → **值本身** 或 **离散化后 one-hot** | `feature-map.yml` |
| A2 | 组合预测 | Eq.1 | `ŷ = sigmoid(y_FM + y_DNN)` | `model.py: call()` |
| A3 | FM 一阶 + 二阶 | Eq.2 | `y_FM = ⟨w,x⟩ + Σ_{j1}Σ_{j2>j1} ⟨V_i,V_j⟩·x_j1·x_j2`，`w∈R^d`，`V_i∈R^k` | `arena.LinearLayer` + `arena.FMLayer` |
| A3.1 | 省略常数偏置 | §2.1 脚注 2 | "We omit a constant offset for simplicity." | **Eq.2 无 w₀**；见决策 D4 |
| A4 | embedding 层输出 | Eq.3 | `a^(0) = [e1,e2,…,em]`，各 field embedding 同维 k | `FMLayer(return_emb=True)` 拼接 |
| A5 | DNN 前向 | Eq.4 | `a^(l+1) = σ(W^(l)a^(l) + b^(l))`；`y_DNN = σ(W^(|H|+1)·a^H + b^(|H|+1))` | `arena.DNN` + 末层投影 |
| A6 | **FM 与 DNN 共享同一份 embedding** | §2.1 末段 | "FM component and deep component share the same feature embedding" | 单一 `fm_embed_dict` 同时喂二阶与 DNN |
| A7 | V 即网络权重，无预训练，端到端联合训练 | §2.1 | "we eliminate the need of pre-training by FM and instead jointly train the overall network in an end-to-end manner" | 无 FNN 式预训练阶段 |
| A8 | 每个 field 向量只有一个非零项 | §2.2 脚注 3 | "Only one entry is non-zero for each field vector." | one-hot 不变量（测试断言） |
| A9 | 数据集 Criteo | §3.1 | 45M 记录；13 continuous + 26 categorical；**90% train / 10% test** | 转换器 + `params.yml` |
| A10 | 评估指标 | §3.1 | AUC (Area Under ROC) 与 Logloss (cross entropy) | `arena.metrics.point_auc_all_scores` |
| A11 | 超参：dropout | §3.1 | `dropout: 0.5` | `dnn_dropout: 0.5` |
| A12 | 超参：网络结构 | §3.1 | `network structure: 400-400-400` | `dnn_hidden: [400,400,400]` |
| A13 | 超参：优化器 | §3.1 | `optimizer: Adam` | `optimizer: adam` |
| A14 | 超参：激活函数 | §3.1 | "tanh for IPNN, relu for other deep models"（DeepFM 属 other） | `activation='relu'` |
| A15 | 超参：FM 隐向量维度 | §3.1 | "the latent dimension of FM is 10" | `embed_size: 10`（**歧义，见 D5**） |
| A16 | 复现目标值 | §3.2 Table 2 | Criteo：DeepFM **AUC 0.8007 / Logloss 0.45083** | 验收基线 |
| A17 | 超参研究结论 | §3.3 | relu 优于 tanh；dropout 0.6–0.9 最佳；200/400 神经元；3 隐层；constant 形状最佳 | 仅作复现后对照，非主线 |

---

## 二、现状调研结论（已核实，含 file:line）

以下为**实测事实**，非推测。路径相对 `deepmodel/`。

### 2.1 arena 可直接复用的部分（无需改造）

| 能力 | 位置 | 与论文关系 |
|---|---|---|
| FM 二阶 | `arena/arena/layers/fm.py:12-32` | `0.5*Σ(square_of_sum − sum_of_squares)`，是 Eq.2 二阶项的 O(kn) 等价式 |
| FM 一阶 | `arena/arena/layers/linear.py:20-42` | `make_embedding_dict(..., 1, ...)` → `embed_size=1`，`reduce_sum` 即 `⟨w,x⟩` |
| DNN | `arena/arena/layers/dnn.py` | 逐层 `tensordot + bias_add + activation + dropout` = Eq.4 |
| 连续值参与乘法 | `arena/arena/layers/embedding.py:7-21` | `DenseEmbedding.call` 返回 `weight_var * expand_dims(x)`，即 `V_i·x_i` → **Eq.2 对 dense field 精确成立** |
| Adam | `arena/arena/arena_estimator.py:71-86` | 支持 `adagrad/adadelta/rmsprop/sgd/adam/ftrl/adamw` — A13 可直接用 |
| Logloss | `arena/arena/loss_util.py:347-413` | `pointwise_loss_fn` = `sigmoid_cross_entropy_with_logits`，`agg='mean'` → A10 |
| AUC + Logloss 指标 | `arena/arena/metrics.py:8-46` | `point_auc_all_scores` 同时产出 `AUC` 与 `BinaryCrossentropy` → A10 |
| 生产参考实现 | `arena/DeepFM/model.py` | 结构已是 Eq.1（`logits = first_order + second_order + dnn_out + bias`），但超参是生产配置，非论文 |

### 2.2 已核实的**阻碍**（必须正面处理）

1. **arena 没有任何 Criteo 支持。** 全仓库 grep `criteo` 零命中。合法 `data_format` 仅 `plain / useq / useq_hist / plain_news_full / reqq_news_full`（`arena/arena/data_source/data_source.py:174-246`），非法值在 `:246` 抛异常。
2. **所有 reader 硬编码 GZIP。** `data_source.py:392-399`，`tfrecord` 与 `text` 均 `compression_type='GZIP'`。转换器必须输出 GZIP。
3. **本地 `import arena` 失败。** `arena/arena/fs_registry.py:10` 模块级 `import tensorflow_io as tfio`，本机未装 `tensorflow_io`（`ARENA_SKIP_S3MULTI=1` **不能**绕过，因为是模块级导入）。实测 `pytest tests/` 因此 collection error 中断（2 errors / 55 collected）。
4. **本地 Keras 版本不匹配。** 本机 TF **2.16 / Keras 3**；生产 `env.sh:21` 为 `tf215py3`（TF **2.15 / Keras 2**）。arena 生产 embedding 层用 Keras-2 的 `add_weight(partitioner=...)`，Keras 3 拒绝。
5. **`./train.py` 本地不可用。** `train.py:146` 执行 `source ./env.sh; …`，而 `env.sh` 依赖 `/opt/miniconda3/envs/tf215py3` 与 CDH Hadoop —— macOS 本机不存在。
6. **`deepmodel/` 是嵌套 git 仓库**（branch `master`，GitLab 生产库，已有 `arena/RankMixer/` 等未跟踪产物）。`deepfm_claude` 本身**零 commit**，`deepmodel/` 在其中显示为 `??`。→ `CLAUDE.md` 规定的 `git add .` 会把它作为 gitlink 吞入。**见决策 D3。**
7. **测试无 conftest.py / pytest.ini。** 既有范式：`importlib.util.find_spec("tensorflow")` 跳过门 + 按文件路径加载模块 + `sys.modules` 打桩（`tests/test_rankmixer_model.py:39-58` 是唯一可用配方）。跑法：`cd arena && python -m pytest tests/test_x.py`。
8. **CI 不跑测试。** `.gitlab-ci.yml` 只做 git-sync 部署；`Jenkinsfile` 是 stub。测试只能手工跑。

### 2.3 环境实测

```
Python 3.11.13 ／ TensorFlow 2.16.2 ／ pytest 9.1.1 ／ numpy 1.26.3
tensorflow_io: 未安装      pyspark: 未安装
```

---

## 三、待决策点（BLOCKING — `CLAUDE.md` 规定遇歧义 / 需决策必须停）

> **T0 之前不得开工。** 每项给出推荐值与论文依据。

### D1 — 连续特征（I1–I13）如何表示？

论文 §2 明确给出**两个都合法**的选项："each continuous field is represented as **the value itself**, or **a vector of one-hot encoding after discretization**"。论文未说明 Criteo 用哪个。

| 方案 | 落地 | 优点 | 风险 |
|---|---|---|---|
| (a) 值本身 | `type: float` → `kind=dense` → `DenseEmbedding` 给出 `V·x` | Eq.2 精确成立；零改造 | Criteo 计数无上界，`V·x` 数值爆炸；论文未提任何归一化，做了即臆造 |
| **(b) 离线离散化 → one-hot（推荐）** | 转换器内分桶 → `type: int` 稀疏 field | §2 明确许可；`x∈{0,1}` 使 Eq.2 与 A8 精确成立；无需 BN；不改 arena | 分桶策略论文未指定（需自选并记录） |
| (c) arena `dense_bin=True` | `DenseBNBin`：BN→sigmoid→100 桶 | 复用现成代码 | 引入论文未提的 **BatchNorm** 可学习参数 → 臆造风险 |

**推荐 (b)**，分桶边界用等频；理由：唯一同时满足「论文许可」「不引入论文外结构」「Eq.2/A8 精确」的选项。

### D2 — 本地如何跑起来？（关系到「所有代码必须可以运行」）

| 方案 | 代价 |
|---|---|
| (a) 建 venv/conda 装 TF 2.15 + tensorflow_io | 一次性；最接近生产 `tf215py3`；macOS arm64 上 `tensorflow_io` 可能无轮子 |
| **(b) 沿用 arena 既有打桩范式（推荐）** | 复制 `test_rankmixer_model.py:39-58` 配方；测试可跑，但**端到端训练仍跑不了** |
| (c) 只在集群跑 | 本地无法验证，违背 CLAUDE.md |

**推荐 (b) 保证 pytest 必过 + 并行尝试 (a) 打通 smoke 训练**；(a) 失败则端到端训练标记为需集群，并明确告知，不谎称本地已验证。

### D3 — 代码放哪？git 怎么提交？

`train.py:115-126` 硬性要求模型位于 `<model_code_dir>/model.py` + `<model_code_dir>/<config>/params.yml`，即必须落在 `deepmodel/arena/DeepFMPaper/` —— 而 `deepmodel/` 是**别人的生产 git 仓库**。

| 方案 | 说明 |
|---|---|
| **(a) 符号链接（推荐）** | 真身在 `deepfm_claude/DeepFMPaper/`（可提交），`deepmodel/arena/DeepFMPaper` → symlink。Python 沿 symlink 正常导入 |
| (b) 镜像 + 同步脚本 | 两份副本易漂移 |
| (c) 直接提交进 deepmodel | 污染生产库，且不进我们的 repo |

无论选哪个：`deepfm_claude/.gitignore` 必须加 `deepmodel/`，否则 `git add .` 会生成 gitlink。

### D4 — 是否保留全局 bias？

论文脚注 2 明说 "We omit a constant offset for simplicity" → Eq.2 **无 w₀**。生产 `arena/DeepFM/model.py:64` 有 `self.bias`。
**推荐：保留，但语义归属 Eq.4 的 `b^(|H|+1)`**（A5 末层偏置本就存在），并在测试中断言其来源为 Eq.4 而非 Eq.2 的 w₀。若求绝对忠实可去掉 —— 需拍板。

### D5 — `embed_size` 到底取多少？

§3.1 原文："The optimizers of LR and FM are FTRL and Adam respectively, and **the latent dimension of FM is 10**." 该句处于**基线 LR/FM 模型**的语境；DeepFM 只被说 "uses the same setting"（指 dropout 0.5 / 400-400-400 / Adam / relu）。k=10 是否适用于 DeepFM 的共享 embedding，论文**未直说**。
**推荐 k=10**（社区复现惯例，且全文唯一出现过的 k 值），并在 README 标注此处为推断而非直引。

---

## 四、任务列表

> 状态：`[ ]` 未开始 ／ `[~]` 进行中 ／ `[x]` 完成
> 每个任务附**验收标准**与**论文锚点**。无锚点的任务不写代码。

### Phase 0 — 决策与基建

- [ ] **T0 决策确认**
  取得 D1–D5 的用户决策，逐条记入 `README.md` 的「决策记录」。
  *验收*：五项均有明确结论与依据；本文件对应处更新。
  *锚点*：§2 / §3.1 / 脚注 2。

- [ ] **T1 本地环境与打桩基座**
  按 D2 结论落地。产出 `tests/_arena_stub.py`（复用 `test_rankmixer_model.py:39-58` 配方：打桩 `tensorflow_io` / `pyspark`，`ARENA_SKIP_S3MULTI=1`）。查明 `SparseEmbedding` / `DenseEmbedding` 在 Keras 3 下能否 `build`（生产 `ShardedSparseEmbedding` 用 `partitioner` 必挂，普通版未知 —— 需实测）。
  *验收*：打桩下 `import arena` 成功；实测结论回写本文件 §2.3。
  *锚点*：无（工程基建）。

- [ ] **T2 仓库骨架**
  `deepfm_claude/.gitignore`（含 `deepmodel/`、`__pycache__/`、`*.tfrecord.gz`、`data/`）、`pyproject.toml`、`README.md`（含决策记录与论文锚点表）、`pytest.ini`。按 D3 建立 `DeepFMPaper/` 与 symlink。
  *验收*：`git status` 不再出现 `deepmodel/`；`pytest` 可发现测试。
  *锚点*：无（工程基建）。

### Phase 1 — 论文参考实现（纯 numpy，作为数值基准）

> 定位：**测试预言（oracle）**，不是交付模型；交付模型是 Phase 3 的 TensorFlow 实现（`CLAUDE.md`：基于 TensorFlow 实现）。
> 理由：本地 arena 不可导入（§2.2-3），需要零依赖的 ground truth 才能满足「pytest 必须通过」；且这与 arena 既有 golden-vector 范式一致（`tests/golden/*.npz` 来自 PyTorch 参考实现）。

- [ ] **T3 `ref/config.py`** — `DeepFMConfig` 冻结 §3.1 超参（k=10、[400,400,400]、dropout 0.5、relu、Adam）。每字段 docstring 标注锚点。
  *验收*：pytest 断言各默认值 == §3.1；非法值抛异常。 *锚点*：A11–A15。

- [ ] **T4 `ref/embedding.py`** — Eq.3：per-field 查表 `[k]` → concat `[m*k]`。
  *验收*：shape 断言 `(B, m*k)`；one-hot field 只取一行（A8）。 *锚点*：A4, A8, Fig.4。

- [ ] **T5 `ref/fm.py`** — Eq.2：一阶 `⟨w,x⟩` + 二阶。
  *验收*：**朴素 O(n²) 双重求和 与 O(kn) 化简式数值对拍**（`assert_allclose`, atol 1e-6）—— 这是 Eq.2 忠实性的核心证据；无 w₀（A3.1）。 *锚点*：A3, A3.1。

- [ ] **T6 `ref/dnn.py`** — Eq.4：逐层 `σ(Wa+b)`，relu + dropout。
  *验收*：逐层 shape 断言；`training=False` 时 dropout 为恒等。 *锚点*：A5, A11, A14。

- [ ] **T7 `ref/deepfm.py`** — Eq.1 组合。
  *验收*：断言 **FM 二阶与 DNN 取到同一个 embedding 对象**（A6 —— 共享性是论文核心，必须由测试锁死）；`ŷ∈(0,1)`。 *锚点*：A2, A6, A7。

- [ ] **T8 `ref/metrics.py`** — AUC + Logloss。
  *验收*：对已知小样本手算值对拍；AUC 用秩公式，Logloss 用 `-[y·log p + (1-y)·log(1-p)]` 均值。 *锚点*：A10。

- [ ] **T9 生成 golden 向量** — `tests/golden/{fm,dnn,deepfm}.npz`（固定 seed），供 Phase 3 对拍。
  *验收*：npz 含 `input/weights/output`；numpy 侧可复现。 *锚点*：A2–A5。

### Phase 2 — 数据管线

- [ ] **T10 Criteo 数据获取**
  论文 §3.1 脚注 5 给的 `labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/` **已失效**，需确认可用镜像。
  *验收*：`dac.tar.gz` 落地 + 校验（记录 size / md5）；行数与 45M 量级核对（A9）。
  *风险*：**可能因网络 / 链接失效阻塞 → 届时须停下询问用户**（CLAUDE.md：信息不足）。

- [ ] **T11 合成小样本 fixture**
  先于 T12：生成符合 Criteo schema 的 tiny 数据（`label` + `I1–I13` + `C1–C26`），供全部本地测试使用，**不依赖 T10**。
  *验收*：可被 `parse_plain_example` 解析。 *锚点*：A9。

- [ ] **T12 Criteo → TFRecord 转换器**
  关键决策：**复用 `data_format: plain`，不新增 arena data_format** —— 避免改动生产共享代码；`plain` 走 `.batch()` + `parse_plain_example`，与 Criteo 扁平行结构精确匹配。
  含：按 D1 结论对 I1–I13 分桶；C1–C26 词表 / hash；**GZIP 压缩**（§2.2-2 硬约束）；**90/10 划分**（A9）。
  *验收*：pytest 断言 schema、GZIP 可读、划分比例 0.9/0.1（±容差）、字段数 = 1+13+26；空值处理有明确规则。 *锚点*：A1, A9。

### Phase 3 — arena 落地

- [ ] **T13 `DeepFMPaper/criteo/feature-map.yml`**
  **注意既有坑**：`features.py:68-94` 中 `kind` 由 `type` 推导 —— `float/double`→dense，其余→**sparse**；`group:` 是自由文本，**不能**决定 dense（只有 `group: label` 会被 `arena_estimator.py:519-520` 剔除）。`dim` 是哈希桶基数，支持 `k/m` 后缀（`features.py:109-114`）。
  *验收*：pytest 解析 yml 后断言每个 field 的 `kind` 与 D1 结论一致（**此处是此前实现搞错过的地方，必须测**）；13+26+1 条目齐全。 *锚点*：A1, A9。

- [ ] **T14 `DeepFMPaper/criteo/params.yml`**
  §3.1 超参逐项锚点注释。**必须含 `grad_clip` 键**（`arena_estimator.py:95` 直接索引 `params['grad_clip']`，缺键即 KeyError）。`label_fun: from_field` + `label_field: label`。`max_steps` / `epochs` 二选一（`arena_estimator.py:636-672`：`epochs` 需 `steps_per_epoch` 或 `auto_count_data`）。
  *验收*：pytest 校验键齐全 + 值 == §3.1（k=10 / [400,400,400] / 0.5 / adam / pointwise）。 *锚点*：A10–A15。

- [ ] **T15 `DeepFMPaper/model.py`**
  Eq.1–4 落地：`logits = first_order + second_order + dnn_out`（+bias 依 D4）；`activation='relu'`；`use_bn=False`（论文只提 dropout，不臆造 BN）；`loss_fn` 只走 `pointwise`（A10），其余分支明确抛错。
  单一 `fm_embed_dict` 同时喂二阶与 DNN（A6）；一阶用独立 `size=1` 字典（对应 `w∈R^d` 与 `V∈R^k` 分离，A3）。
  *验收*：见 T16。 *锚点*：A2–A7, A11–A15。

- [ ] **T16 arena 侧测试**
  基于 T1 打桩基座；用 `find_spec` 跳过门（既有范式）。
  *验收*：① 与 T9 golden 向量对拍（atol 1e-5）；② **断言 FM 二阶与 DNN 共享同一 embedding 变量**（A6）；③ 各 Tensor shape 断言（`CLAUDE.md`：必须解释 Tensor Shape）；④ `loss_fn` 非 pointwise 时抛错。 *锚点*：A2–A6。

- [ ] **T17 端到端 smoke 训练**
  用 T11 合成数据跑 1–2 step，产出 checkpoint。若 D2 走 (b) 而本地跑不通，**停下报告**，不得伪造通过。
  *验收*：loss 有限且下降；checkpoint 落盘；AUC / Logloss 指标可读出（A10）。

### Phase 4 — 复现与验证

- [ ] **T18 全量 Criteo 训练**（依赖 T10）
  90/10 划分（A9），§3.1 超参。
  *验收*：训练完成，记录 AUC / Logloss 曲线。

- [ ] **T19 与论文 Table 2 比对**
  *验收*：AUC 对 **0.8007**、Logloss 对 **0.45083**（A16）；差异写入 `README.md` 并逐条归因（数据版本 / 分桶策略 / k 的歧义 D5 / lr 未给定 等）。**不得为凑数改动与论文冲突的设置。**

- [ ] **T20 全文逐项 Review**
  以 §一 锚点表为清单逐行核对实现；输出 crosswalk 表（论文要素 → 文件:行 → 测试）。
  *验收*：每行锚点都有对应代码与测试；无锚点的代码要么删除，要么在 README 明确标注为「论文外工程必需」。

---

## 五、风险登记

| # | 风险 | 触发点 | 处置 |
|---|---|---|---|
| R1 | Criteo 原始链接失效，拿不到数据 | T10 | 停下问用户（CLAUDE.md：信息不足）；Phase 1–3 与 T17 不依赖它 |
| R2 | 本地 TF 2.16/Keras 3 与 arena（Keras 2）不兼容 | T1 | 打桩范式兜底；端到端训练可能只能上集群 |
| R3 | `V·x` 数值爆炸（若 D1 选 (a)） | T15 | D1 推荐 (b) 离散化规避 |
| R4 | `git add .` 把 `deepmodel/` 吞成 gitlink | 每次提交 | T2 的 `.gitignore` 必须先落地 |
| R5 | 论文未给 learning rate / batch size / 训练轮数 | T14, T18 | 明确标注为「论文外必需参数」，取常用值并记录，不谎称来自论文 |
| R6 | 复现指标够不到 Table 2 | T19 | 如实报告 + 归因；禁止为对齐数字而偏离论文 |

---

## 六、备注

- `arena/DeepFMPaper/`（含 criteo 配置）曾存在，并在本次调研期间被删除；本计划按**从零新建**推进。
- 兄弟目录 `../deepfm_cglm/` 是同一任务的另一次尝试（含其自己的 `TODO.md` / `src/deepfm_ref/`）。本计划基于本次独立调研撰写，未复制其结论；如需对齐或复用，请明示。
