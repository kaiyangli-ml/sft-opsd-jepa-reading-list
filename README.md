# SFT、OPSD 与 JEPA：25 篇研究阅读清单

检索与发表状态核验截至 **2026-10-06**。本清单服务于三个相互关联、又需要分别研究的问题：SFT 的泛化机制与监督目标；OPSD 的教师信息、学习机制与适用边界；JEPA 式表示预测对语言训练与自蒸馏的潜在价值。

收录标准为 **NeurIPS、ICML、ICLR 主会论文，或近期 arXiv 工作**，并核验作者的主要机构。会议论文优先链接正式论文集或会议论文，预印本明确标注 arXiv；机构按论文署名列出，并非对作者当前任职的判断。阅读问题与研究价值是选题建议，不能替代论文结论或实验复现。

## 目录

- [优先级与第一轮阅读顺序](#reading-order)
- [SFT：8 篇](#sft)
  - [泛化机制](#sft-generalization)
  - [监督目标](#sft-objectives)
  - [自训练](#sft-self-training)
  - [基础认识](#sft-foundations)
- [OPSD：10 篇](#opsd)
  - [核心框架](#opsd-frameworks)
  - [教师信息与边界](#opsd-context)
  - [潜在表示](#opsd-latent)
  - [理论基础](#opsd-theory)
- [JEPA：7 篇](#jepa)
  - [语言直接证据](#jepa-language)
  - [表示机制](#jepa-mechanisms)
  - [教师设计](#jepa-teachers)
  - [跨模态](#jepa-multimodal)
- [证据边界与比较时的关键问题](#evidence-boundaries)

<a id="reading-order"></a>
## 优先级与第一轮阅读顺序

| 优先级 | 含义 | 篇数 |
| --- | --- | --- |
| **P0** | 与当前选题直接相关，第一轮精读 | 8 |
| **P1** | 关键机制、对照或基础框架，第二轮深入 | 10 |
| **P2** | 补充背景、扩展情境或长期参考 | 7 |

优先级表示与本次研究问题的相关程度，不是论文质量排名，也不意味着新预印本的结论比经典论文更可靠。

第一轮按以下顺序阅读：

1. [Rethinking Generalization in Reasoning SFT](#paper-01)：先厘清 SFT 何时泛化，以及比较中的条件差异。
2. [DFT](#paper-04)：理解 token 梯度重加权能够改变什么。
3. [SoFT](#paper-05)：从硬标签转向兼顾学习与保留的软目标。
4. [原始 OPSD](#paper-09)：理解学生轨迹、上下文差异与分布监督。
5. [SDFT](#paper-10)：连接示范学习、on-policy 自蒸馏与持续学习。
6. [What Should a Self-Teacher See?](#paper-13)：研究教师应该得到什么信息、以什么抽象层次得到。
7. [From Self-Distillation to Self-Practice](#paper-14)：检验特权信息在多轮任务中的迁移边界。
8. [LLM-JEPA](#paper-19)：再判断表示预测能补足哪些具体缺陷。

<a id="sft"></a>
## SFT：8 篇

<a id="sft-generalization"></a>
### 泛化机制

<a id="paper-01"></a>
#### 01. Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability

**P0 · arXiv，2026-04-08** · 主要机构：上海人工智能实验室、上海交通大学、中国科学技术大学。 [论文](https://arxiv.org/abs/2604.06628)

- **阅读问题：** SFT 的泛化失败来自目标函数本身，还是训练不足、监督质量、推理轨迹结构与底座能力？
- **价值：** 长 CoT SFT 的跨域表现可能先降后升；同样的轨迹也可能被不同能力的模型学成可迁移程序或表面冗长。这篇适合确定公平的 SFT 基线和训练时长。
- **状态说明：** 另有 [ICML 2026 workshop 版本](https://openreview.net/pdf/86e69ff26f20cfab5a133eef2cf12ebabb287e65.pdf)，此处按近期 arXiv 收录，不将 workshop 记为 ICML 主会。

<a id="paper-02"></a>
#### 02. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

**P1 · ICML 2025 主会** · 主要机构：香港大学、UC Berkeley、NYU、Google DeepMind、University of Alberta。 [正式论文集](https://proceedings.mlr.press/v267/chu25c.html)

- **阅读问题：** 在未见规则和视觉分布变化下，SFT 到底失败在哪里？怎样区分记住解题轨迹和掌握可迁移规则？
- **价值：** 提供 GeneralPoints 与 V-IRL 的受控比较，是研究 SFT 时需要认真回应的失败案例。应与第 01 篇对读，不能把标题当作所有 SFT 设置的普适定律。

<a id="paper-03"></a>
#### 03. Two-stage LLM Fine-tuning with Less Specialization and More Generalization

**P1 · ICLR 2024 主会 · ProMoT** · 主要机构：Google、UCLA。 [正式论文集](https://proceedings.iclr.cc/paper_files/paper/2024/hash/597254dc45be8c166d3ccf0ba2d56325-Abstract-Conference.html)

- **阅读问题：** 微调早期学到的格式偏好，会不会妨碍语义能力与跨任务迁移？这两类学习能否分离？
- **价值：** 先用 soft prompt 吸收任务格式，再在该 prompt 下微调模型。它对 JEPA 结合的启发是：先明确哪些表面变化不应被强制模仿，再定义需要保留的语义表示。

<a id="sft-objectives"></a>
### 监督目标

<a id="paper-04"></a>
#### 04. On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

**P0 · ICLR 2026 主会 · DFT** · 主要机构：东南大学、UCLA、上海交通大学、NTU、UC Berkeley、武汉大学、UC Merced。 [正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2e53c02ea028cbf603f4b6b47fef3d97-Abstract-Conference.html)

- **阅读问题：** 在不引入在线 rollout 或额外反馈的情况下，改变 token 的梯度权重能改善多少泛化？
- **价值：** 从 policy-gradient 视角重写 SFT 梯度，利用模型对 token 的概率动态重加权监督。这是评估新 SFT 目标时的重要基线；其数学重写和机制解释应结合推导假设阅读。

<a id="paper-05"></a>
#### 05. SoFT: Soft Targets for Generalizable LLM Fine-Tuning

**P0 · arXiv，2026-09-26** · 主要机构：香港科技大学、香港科技大学（广州）。 [论文](https://arxiv.org/abs/2609.32493)

- **阅读问题：** 能否提高示范 token 的概率，同时避免压低学生本来就支持的其他合理续写？
- **价值：** 为示范 token 设置概率下限，并构造相对 base 分布 KL 改动最小的软目标，将学习示范与保留原能力连接起来。无需 teacher logits 或在线 rollout，适合与 DFT、标准 SFT 及后续表示监督作直接比较。

<a id="sft-self-training"></a>
### 自训练

<a id="paper-06"></a>
#### 06. STaR: Bootstrapping Reasoning With Reasoning

**P1 · NeurIPS 2022 主会** · 主要机构：Stanford、Google Research。 [正式论文集](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html)

- **阅读问题：** 模型如何把自身生成的推理转化为进一步学习的监督？正确答案提示带来的事后解释，有何帮助和局限？
- **价值：** 生成推理、筛选答案正确的轨迹，必要时利用正确答案重新生成 rationale，再迭代 SFT。它是连接 SFT 与自我改进的经典起点，但不等同于逐 token 分布匹配的 OPSD。

<a id="sft-foundations"></a>
### 基础认识

<a id="paper-07"></a>
#### 07. Finetuned Language Models Are Zero-Shot Learners

**P2 · ICLR 2022 主会 · FLAN** · 主要机构：Google Research。 [官方论文页](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/)

- **阅读问题：** 多任务指令微调为什么能够改善未见任务，而不只是训练任务本身？
- **价值：** 建立 instruction tuning 的基本经验事实，展示任务多样性、自然语言指令与模型规模的重要性。它提醒我们：SFT 能否泛化，需要连同任务分布和监督设计一起讨论。

<a id="paper-08"></a>
#### 08. LIMA: Less Is More for Alignment

**P2 · NeurIPS 2023 主会** · 主要机构：Meta AI、Carnegie Mellon University、USC、Tel Aviv University。 [正式论文集](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html)

- **阅读问题：** 少量高质量 SFT 示范是在教授新能力，还是引导模型呈现预训练中已有的能力？
- **价值：** 用 1,000 个精心策划示例进行微调，提出 superficial alignment hypothesis。适合建立“能力激活”和“能力获得”的区别，但不能据此认定所有推理 SFT 都只学格式。

**SFT 阅读路径：** Rethinking → DFT → SoFT → SFT Memorizes → ProMoT；研究数据生成或自训练时接上 STaR。

<a id="opsd"></a>
## OPSD：10 篇

<a id="opsd-frameworks"></a>
### 核心框架

<a id="paper-09"></a>
#### 09. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

**P0 · ICML 2026 主会 · 原始 OPSD** · 主要机构：UCLA、香港大学、Meta Superintelligence Labs。 [正式论文集](https://proceedings.mlr.press/v306/zhao26be.html)

- **阅读问题：** 同一个模型在看到参考答案或推理后，能否在学生自己的轨迹上提供有效、可迁移的监督？
- **价值：** 利用有／无特权上下文的同模型教师和学生，在学生轨迹上匹配 token 分布。重点检查学生能力门槛、教师上下文、教师更新，以及完整词表目标与采样 token 目标的差异。

<a id="paper-10"></a>
#### 10. Self-Distillation Enables Continual Learning

**P0 · ICML 2026 主会 · SDFT** · 主要机构：MIT、ETH Zurich。 [正式论文集](https://proceedings.mlr.press/v306/shenfeld26a.html)

- **阅读问题：** 看过 demonstration 的自己，怎样教会没有看过 demonstration 的自己，同时减少原有能力遗忘？
- **价值：** 将示范转成学生轨迹上的自蒸馏信号，直接连接 SFT、on-policy 学习与持续学习。适合将新任务收益和原能力保持放在同一套评价中，而不是只看新任务分数。

<a id="paper-11"></a>
#### 11. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

**P1 · ICLR 2024 主会 · GKD** · 主要机构：Google DeepMind、Mila、University of Toronto。 [正式论文集](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html)

- **阅读问题：** 学习收益来自新的监督目标，还是来自学生自己生成训练前缀，从而改变了训练分布？
- **价值：** 把轨迹来源与匹配散度分开研究，教师在学生自生成前缀上指导学生。它是理解 SFT 与 on-policy 蒸馏差异的基础，但教师不必是学生自身。

<a id="paper-12"></a>
#### 12. Reinforcement Learning via Self-Distillation

**P1 · ICML 2026 主会 · SDPO** · 主要机构：ETH Zurich、MIT、Stanford、Max Planck Institute for Intelligent Systems。 [正式论文集](https://proceedings.mlr.press/v306/hubotter26a.html)

- **阅读问题：** 执行错误、评测反馈或成功尝试，能否转化为比单个结果奖励更细密的学习信号？
- **价值：** 让教师看到丰富反馈，以分布监督指导学生。值得关注反馈到 credit assignment 的转换、教师更新方式，以及反馈是否提供学生可利用的信息。

<a id="opsd-context"></a>
### 教师信息与边界

<a id="paper-13"></a>
#### 13. What Should a Self-Teacher See? Privileged Context Design for On-Policy Self-Distillation

**P0 · arXiv，2026-09-22** · 主要机构：上海人工智能实验室、复旦大学、南京大学、上海交通大学、北京大学、UCLA、同济大学。 [论文](https://arxiv.org/abs/2609.25623)

- **阅读问题：** 教师应该看到完整解答、答案、策略提示，还是更抽象的问题描述？最佳抽象层次是否随学生与任务变化？
- **价值：** 比较不同特权上下文，说明信息更多或初始师生 KL 更大并不保证教学更好。它直接支持研究教师上下文设计；但语义、长度和桥接措辞等因素尚不能完全隔离，不能据此断言抽象监督一定优越。

<a id="paper-14"></a>
#### 14. From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents

**P0 · arXiv，2026-09-24 · PSP** · 主要机构：AWS AI／Amazon、Texas A&M University。 [论文](https://arxiv.org/abs/2609.29051)

- **阅读问题：** 如果教师的行动依赖学生推理时不可能拥有的信息，蒸馏是否会只转移确定的行动风格，而不能转移行动依据？
- **价值：** 提供多轮 agent 设置中 OPSD 的边界证据，并探索将特权信息用于采样、自我练习的方案。它是重要的反方阅读；其结果与理论条件不能被概括为“特权信息原则上不可蒸馏”。

<a id="paper-15"></a>
#### 15. On-Policy Self-Distillation without Any Supervision

**P2 · arXiv，2026-08-06 · U-OPSD** · 主要机构：UC San Diego、Georgia Institute of Technology、University of Maryland, College Park、ByteDance。 [论文](https://arxiv.org/abs/2608.06296)

- **阅读问题：** 多数投票与自一致性何时能产生可信监督，何时只会强化共同错误？
- **价值：** 利用模型自身输出构造伪解与教师上下文，探索无需外部标签的自蒸馏。在无监督训练中，对轨迹正误的判断来自伪标签，不能等同于由真实答案验证。

<a id="opsd-latent"></a>
### 潜在表示

<a id="paper-16"></a>
#### 16. Latent On-Policy Self-Distillation

**P1 · arXiv，2026-08-13 · LOPD** · 主要机构：National University of Singapore、上海交通大学、北京邮电大学。 [论文](https://arxiv.org/abs/2608.13040)

- **阅读问题：** 能否把“教师看到什么”变成可学习的连续上下文，而不是固定的人类文本提示？
- **价值：** 将检索经验转为可学习的 latent tokens，作为教师特权上下文，并在学生轨迹上做 token 分布蒸馏。**这里 latent 的是教师上下文；它不是 JEPA，也不是预测 latent target 的训练目标。**

<a id="opsd-theory"></a>
### 理论基础

<a id="paper-17"></a>
#### 17. Unifying Distillation and Privileged Information

**P1 · ICLR 2016 主会** · 主要机构：Facebook AI Research、Columbia University、Max Planck。 [论文](https://leon.bottou.org/publications/pdf/iclr-2016.pdf)

- **阅读问题：** 教师在训练时拥有的信息优势，何时可以转移给推理时没有这些信息的学生？
- **价值：** 将知识蒸馏与 privileged information 放在统一框架下讨论，为 OPSD 的上下文不对称提供理论背景。它帮助区分“教师更容易解题”和“教师能提供可学习信号”。

<a id="paper-18"></a>
#### 18. Self-Distillation Amplifies Regularization in Hilbert Space

**P2 · NeurIPS 2020 主会** · 主要机构：Google Research、DeepMind、UC Berkeley。 [正式论文集](https://papers.neurips.cc/paper_files/paper/2020/hash/2288f691b58edecadcc9a8691762b4fd-Abstract.html)

- **阅读问题：** 在没有新增任务信息时，自蒸馏为何仍可能改善泛化？反复自蒸馏是否总是有益？
- **价值：** 在 Hilbert 空间模型中分析自蒸馏如何加强正则化，并揭示过多轮迭代可能导致欠拟合。可用于构造机制假设，不能直接当作深度 Transformer 或 OPSD 的完整理论。

**OPSD 阅读路径：** 原始 OPSD → SDFT → What Should a Self-Teacher See? → From Self-Distillation to Self-Practice；研究环境反馈时读 SDPO，研究 latent 上下文时读 LOPD。

<a id="jepa"></a>
## JEPA：7 篇

<a id="jepa-language"></a>
### 语言直接证据

<a id="paper-19"></a>
#### 19. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

**P0 · ICLR 2026 主会** · 主要机构：Atlassian、NYU、Brown University。 [正式会议论文](https://openreview.net/pdf?id=GbXKPo9QfH) · [arXiv](https://arxiv.org/abs/2509.14252)

- **阅读问题：** 在保留 token 交叉熵的同时，跨视图 latent 预测是否能提供更可迁移的语义约束？
- **价值：** 是 JEPA 与语言 SFT／预训练结合的直接证据，适合作为相关研究的出发点。它结合 CE 与表示预测，但本身不是 OPSD，不能把其结果直接当成 on-policy 自蒸馏有效性的证据。

<a id="paper-20"></a>
#### 20. data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language

**P1 · ICML 2022 主会** · 主要机构：Meta AI、SambaNova（相关工作在 Meta 完成）。 [正式论文集](https://proceedings.mlr.press/v162/baevski22a.html)

- **阅读问题：** 让信息较少的 student 预测 full-context teacher 的上下文表示，与预测离散标签有何本质区别？
- **价值：** 用统一的表示预测思路覆盖语言、语音与视觉，语言实验提供了直接的表征学习证据。其 NLP 证据主要是 encoder 理解任务，不是自回归长推理或 OPSD。

<a id="paper-21"></a>
#### 21. Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA

**P1 · arXiv，2026-02-26 · STP** · 主要机构：Atlassian、NYU、Brown University。 [论文](https://arxiv.org/abs/2602.22617)

- **阅读问题：** 不依赖显式多视图数据时，能否利用 hidden-state 轨迹上的表示约束改善语言训练的数据效率？
- **价值：** 提供从离散 token 拟合转向连续轨迹约束的具体语言方法。其显著数据效率结果需要按任务解读，文中 16 倍结果对应 NL-RX-SYNTH，不能写成普遍的规模定律结论。

<a id="jepa-mechanisms"></a>
### 表示机制

<a id="paper-22"></a>
#### 22. How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks

**P1 · NeurIPS 2024 主会** · 主要机构：Apple。 [正式论文集](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a600f0f740605205133553cb74a1c107-Abstract-Conference.html)

- **阅读问题：** 为什么表示预测可能偏向有用特征，而不是拟合所有噪声细节？这种偏置依赖哪些模型与数据假设？
- **价值：** 在深线性自蒸馏网络中分析 JEPA 的隐式偏置，为“去除无关细节”提供机制依据。结论属于受限理论模型，不能直接推广到语言 Transformer 的推理表示。

<a id="paper-23"></a>
#### 23. Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning

**P2 · NeurIPS 2020 主会 · BYOL** · 主要机构：DeepMind、Imperial College London。 [正式论文集](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)

- **阅读问题：** 没有负样本时，教师更新、predictor 与训练不对称如何影响表示学习和坍缩？
- **价值：** EMA target 与无负样本自蒸馏的经典工作，适合建立教师设计的基本认识。证据来自视觉自监督，迁移到语言 OPSD 需要重新实验。

<a id="jepa-teachers"></a>
### 教师设计

<a id="paper-24"></a>
#### 24. Rethinking JEPA: Compute-Efficient Video Self-Supervised Learning with Frozen Teachers

**P2 · ICLR 2026 主会 · SALT** · 主要机构：Apple。 [正式会议论文](https://openreview.net/pdf?id=3cB9243E9i) · [Apple 官方页](https://machinelearning.apple.com/research/rethinking-jepa)

- **阅读问题：** JEPA 的收益是否必须依赖随训练更新的教师？固定教师在效率和稳定性上有什么取舍？
- **价值：** 在视频自监督中研究 frozen teacher 的表示预测，为固定／EMA／同步更新教师的比较提供启发。其视频结果不能用来证明语言 OPSD 中固定教师更优。

<a id="jepa-multimodal"></a>
### 跨模态

<a id="paper-25"></a>
#### 25. VL-JEPA: Joint Embedding Predictive Architecture for Vision-language

**P2 · ICLR 2026 主会** · 主要机构：Meta FAIR、香港科技大学、Sorbonne Université、NYU；正式版亦列 University of Amsterdam、USC。 [正式会议论文](https://openreview.net/pdf?id=tjimrqc2BU) · [arXiv](https://arxiv.org/abs/2512.10942)

- **阅读问题：** 视觉条件下预测文本 embedding，与逐 token 生成相比，会保留或丢失哪些监督信息？
- **价值：** 提供文本表示预测与视觉语言 SFT 的直接结合案例。它支持研究语义空间中的监督，但不等同于验证纯语言长推理或 OPSD 的效果。

**JEPA 阅读路径：** LLM-JEPA → data2vec → STP → How JEPA Avoids Noisy Features；同时对照 LOPD，区分教师的 latent 上下文与表示预测目标。

<a id="evidence-boundaries"></a>
## 证据边界与比较时的关键问题

**SFT 与 OPSD 本身都值得独立研究，不需要以结合 JEPA 作为前提。** 这份清单中的 SFT 论文分别触及训练条件、格式专门化、梯度分配、软目标和遗忘；OPSD 论文则进一步涉及学生实际访问的状态、教师特权信息，以及这些信息能否被学生恢复。

**JEPA 的相关证据强度并不相同。** LLM-JEPA 与 STP 提供直接的生成式语言实验；data2vec 提供语言 encoder 表征实验；VL-JEPA 提供视觉语言实验。BYOL、SALT 的视觉结果以及深线性理论能启发机制假设，但没有证明 JEPA 一定能改善 OPSD。

尤其需要避免三个概念混淆：

- **latent 上下文不等于 latent 监督目标。** LOPD 改变教师获得的连续上下文，其学生仍接受 token 分布蒸馏。
- **自蒸馏不等于新增知识。** 收益可能来自教师额外信息，也可能来自正则化、训练分布或概率质量重分配，需要分别设置对照。
- **教师更强不等于更可教。** 教师依靠特权信息做出的正确动作，未必是学生在自身信息条件下能学会的动作。

比较新方法时，最值得同时追问的是：收益来自更合适的监督表示，还是更多信息、更多计算、不同轨迹、输出长度变化或更强正则化？应在尽量匹配数据、监督信息与计算预算的基础上，结合跨域泛化、原能力保持、输出长度及多次采样表现进行判断。
