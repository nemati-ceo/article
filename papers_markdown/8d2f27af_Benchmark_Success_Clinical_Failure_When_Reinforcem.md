# Benchmark Success, Clinical Failure: When Reinforcement Learning Optimizes for Benchmarks, Not Patients

**UUID:** `8d2f27af-d657-4d40-93c7-59b114a219a5`

---

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | Armin Berger, Manuela Bergau, Helen Schneider, Saad Ahmad, Tom Anglim Lagones, Gianluca Brugnara, Martha Foltyn-Dumitru, Kai Schlamp, Philipp Vollmuth, Rafet Sifa |
| **Year** | 2025 |
| **Source** | arXiv |
| **arXiv ID** | 2512.23090v2 |
| **DOI** | N/A |
| **Categories** | cs.AI, cs.LG |

**🔗 URL:** [http://arxiv.org/abs/2512.23090v2](http://arxiv.org/abs/2512.23090v2)

---

## 📖 Citation

```
Armin Berger et al. (2025). Benchmark Success, Clinical Failure: When Reinforcement Learning Optimizes for Benchmarks, Not Patients. arXiv preprint arXiv:2512.23090v2.
```

---

## 📝 Abstract

Recent Reinforcement Learning (RL) advances for Large Language Models (LLMs) have improved reasoning tasks, yet their resource-constrained application to medical imaging remains underexplored. We introduce ChexReason, a vision-language model trained via R1-style methodology (SFT followed by GRPO) using only 2,000 SFT samples, 1,000 RL samples, and a single A100 GPU. Evaluations on CheXpert and NIH benchmarks reveal a fundamental tension: GRPO recovers in-distribution performance (23% improvement on CheXpert, macro-F1 = 0.346) but degrades cross-dataset transferability (19% drop on NIH). This mirrors highresource models like NV-Reason-CXR-3B, suggesting the issue stems from the RL paradigm rather than scale. We identify a generalization paradox where the SFT checkpoint uniquely improves on NIH before optimization, indicating teacher-guided reasoning captures more institutionagnostic features. Furthermore, cross-model comparisons show structured reasoning scaffolds benefit general-purpose VLMs but offer minimal gain for medically pre-trained models. Consequently, curated supervised finetuning may outperform aggressive RL for clinical deployment requiring robustness across diverse populations.

<sup>∗</sup>These authors contributed equally and share first authorship.

#

---

## 🔍 Introduction

Recent work demonstrates that reinforcement learning (RL) can substantially improve large language model performance, particularly in settings with a clear reward signal and automatically verifiable outcomes (e.g., mathematics and code generation; see, e.g., DeepSeek-R1). However, it remains less clear how reliably these gains transfer to problems with weaker or more subjective supervision, such as free-form natural language generation and multimodal inputs. In this work, we investigate whether R1-style training, which combines supervised finetuning (SFT) with Group Relative Policy Optimization (GRPO), can enhance multilabel chest X-ray classification in small vision-language models under severe resource constraints. We focus on chest X-ray diagnosis because it represents a clinically critical task where radiologists value both hard diagnostic labels for rapid assessment and accompanying reasoning traces to establish trust in model outputs. Moreover, chest X-rays benefit from large publicly available datasets with multilabel annotations that provide natural reward signals for reinforcement learning. While recent work has explored R1-style reasoning for

medical visual question answering, multilabel chest X-ray classification remains less studied. A notable exception is NVIDIA's NV-Reason-CXR-3B, which utilizes extensive synthetic data and compute. Our work contrasts with this highresource approach by examining R1-style training under extreme constraints: 50 times less training data and 4 times less compute. This setting is particularly relevant for practitioners who lack large-scale annotation pipelines or extensive infrastructure but still seek to leverage reasoning-guided training for improved diagnostic performance. Our work makes three primary contributions.

- Low-Resource R1-Style Training: We present ChexReason, trained with only 2,000 SFT and 1,000 RL samples on a single A100 GPU, demonstrating that R1-style training is feasible without extensive resources.
- Instruction Format Sensitivity: Cross-model analysis reveals that optimal instruction format depends on medical pre-training: structured medically informed reasoning scaffolds benefit general-purpose VLMs while providing minimal gain for domain-specialized models.
- Benchmark-Transferability Trade-off: GRPO improves CheXpert performance (+23%) but degrades NIH transferability (−19%), mirroring NV-Reason-CXR-3B failures and suggesting a paradigm-level issue.
- Generalization Paradox: The SFT checkpoint uniquely improves on outof-distribution data, indicating teacher-guided traces capture more generalizable features than reward-optimized outputs.

#

---

## 🎯 Conclusion

: ...}, from which we extract predicted labels using a rule-based parser.

#### 3.2.1 Supervised Fine-Tuning

In the SFT stage, we train the model to generate teacher-annotated reasoning traces from the 2,000-sample training dataset. The objective minimizes the negative log-likelihood of expert traces t <sup>∗</sup> given input X-ray images x:

$$\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,t^*) \sim \mathcal{D}_{SFT}} \left[ \sum_{j=1}^{|t^*|} \log \pi_{\theta}(t_j^* \mid x, t_{< j}^*) \right], \tag{1}$$

where t ∗ <sup>j</sup> denotes the j-th token in the expert trace, and each trace was generated by Gemini 2.5 provided with ground-truth labels but instructed to produce rationales as if reasoning independently, resulting in clinician-validated reasoning patterns.

We performed SFT on two vision-language models: MedGemma-4B [\[10\]](#page-17-6) and Qwen2.5-VL-3B-Instruct [\[34\]](#page-20-6), using the Accelerate library [\[11\]](#page-17-7). To enable parameter-efficient training, we employed Low-Rank Adaptation (LoRA) [\[12\]](#page-17-8) with rank 16, scaling factor 32, and dropout rate 0.1, applied exclusively to the language model's attention and feed-forward projection layers, while keeping the vision encoder frozen throughout training. Both models were trained for up to 6 epochs with early stopping, patience of 2 epochs based on validation loss, using the AdamW optimizer with a learning rate of 5×10<sup>−</sup><sup>5</sup> , cosine annealing schedule with 5% warmup, weight decay of 0.01, and gradient clipping at norm 5.0. We used an effective batch size of 8 achieved through gradient accumulation with a per-device batch size of 1, and reserved 10% of the data for validation. Training was conducted in BFloat16 precision with gradient checkpointing enabled for memory efficiency. Critically, we applied assistant-only loss masking, where loss computation was restricted to the assistant's response tokens by masking the user prompt, image placeholder tokens, and padding tokens. For visual token processing, we controlled image resolution by constraining visual tokens to the range of 256–512 tokens per image via the processor's minimum and maximum pixels parameters.

#### 3.2.2 Group Relative Policy Optimization

Following SFT, we fine-tune the model using Group Relative Policy Optimization (GRPO) [\[30\]](#page-19-9) on the 1,000-sample RL dataset. GRPO obviates the need for an additional value function approximation and instead uses the average reward of multiple sampled outputs as the baseline. Specifically, for each input X-ray x, GRPO samples a group of text completions {t1, t2, · · · , tG} from the old policy π<sup>θ</sup>old and optimizes the policy model by maximizing the following objective:

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{RL}, \{t_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(t|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|t_i|} \sum_{j=1}^{|t_i|} \min \left[ r_{i,j}(\theta) \hat{A}_i, \right] \right] \\
\text{clip}(r_{i,j}(\theta), 1 - \varepsilon, 1 + \varepsilon) \hat{A}_i - \beta \mathbb{D}_{KL}(\pi_{\theta} \| \pi_{ref}) \right], \qquad (2)$$

where  $r_{i,j}(\theta) = \frac{\pi_{\theta}(t_{i,j}|x,t_{i,<j})}{\pi_{\theta_{\text{old}}}(t_{i,j}|x,t_{i,<j})}$  is the token-level importance ratio,  $\hat{A}_i$  is the sequence-level advantage calculated based on relative rewards within each group, and  $\varepsilon$  and  $\beta$  are hyper-parameters controlling the clipping bounds and KL divergence penalty, respectively. This group-relative approach aligns well with the comparative nature of reward models, which are typically trained on comparison datasets. Furthermore, instead of adding a KL penalty to the reward, GRPO regularizes by directly adding the KL divergence term between the trained policy and the reference policy  $\pi_{ref}$ , the SFT checkpoint, to the objective [30].

Stabilizing Training Against Mode Collapse. In early training runs, we observed training instability characterized by sharp drops in policy entropy, repetitive outputs, and rapid performance degradation on validation metrics, symptoms consistent with mode collapse in RL fine-tuning of LLMs [40]. To address this, we incorporated several stabilization mechanisms informed by recent work on stable RL training for language models. Specifically, our formulation includes: (1) token-level importance sampling correction via the ratio  $\pi_{\theta}/\pi_{\theta_{\text{old}}}$  to account for policy updates between rollout and training steps, (2) clipped policy updates to constrain policy staleness and prevent aggressive parameter changes, and (3) explicit KL divergence regularization against the reference policy to maintain proximity to the SFT checkpoint [40]. These modifications eliminated the collapse behavior observed in preliminary experiments across all training runs.

For each training sample, we generated 4 completions at temperature 0.8 with nucleus sampling (top-p=0.95). The optimization balances reward maximization against a KL divergence penalty ( $\beta=0.15$ ) to prevent excessive deviation from the SFT checkpoint. We employed the Dr. GRPO loss normalization [21] to eliminate length bias, with asymmetric PPO clipping bounds [0.15, 0.22] for stable policy updates.

**Reward Functions.** We developed two alternative reward functions with contrasting design philosophies and evaluated their performance in separate training runs. The *hard reward* enforces strict format compliance combined with label prediction accuracy:

$$r_{\rm hard}(t,Y) = \begin{cases} J(Y,\hat{Y}) - \lambda_{\rm len} & \text{if } t \text{ is valid and } |t| < 250\\ J(Y,\hat{Y}) & \text{if } t \text{ is valid and } |t| \ge 250\\ 0 & \text{otherwise} \end{cases} \tag{3}$$

where valid outputs must contain exactly one <think>...</think> reasoning

block followed by a <solution>...</solution> block, J(Y, Yˆ ) = |Y ∩Yˆ |/|Y ∪Yˆ | is the Jaccard similarity score between predicted and ground truth labels, and λlen is a penalty coefficient for short responses.

The nuanced reward implements a multi-component scoring mechanism designed to provide a more granular reward signal during training:

$$r_{\text{nuanced}}(t, Y) = r_{\text{match}} + r_{\text{partial}} - r_{\text{FP}} - r_{\text{collapse}} - r_{\text{format}},$$
 (4)

where exact matches receive +100 points, partial credit is scaled by recall (30 points per correct label relative to ground truth size) and precision (20 points per correct label relative to prediction size), and various penalties discourage undesirable behaviors. Critically, we incorporated frequency-weighted penalties for false positives: commonly occurring labels such as "No Finding" and "Support Devices" incur higher penalties when incorrectly predicted than rare pathologies, discouraging shotgun predictions. To prevent mode collapse, we monitor a sliding window of the most recent 100 predictions and apply cumulative penalties (−30 per excess repetition, −50 for mode collapse) when any single label exceeds 70% dominance. Additional penalties discourage invalid CheXpert labels (−100 each), duplicate predictions (−25 each), and extraneous text outside the designated blocks. All predicted labels are filtered to the 14 valid CheXpert labels before evaluation.

# 4 Results and Evaluation

### 4.1 Effect of Prompting

To identify the optimal instruction format for multilabel CheXpert classification, we evaluated nine prompt variants on MedGemma-4B, selected as the primary ablation target due to its superior base performance among available models. Each variant implemented a distinct reasoning strategy, ranging from structured, stepwise protocols to free-form narratives and explicit verification mechanisms (Table [1\)](#page-8-0). We initially prioritized a structured approach, denoted as Reasoning A, which aimed to mimic radiological best practices developed by experts from the University Clinic Bonn and Queensland Health. This prompt instructed the LLM to perform a mandatory 12-step chest X-ray analysis within <think> tags, followed by a strict list of derived CheXpert labels within <solution> tags. Contrary to expectations, the less constrained Reasoning Narrative prompt achieved the highest overall performance (micro-F1 = 0.524, macro-F1 = 0.270), followed by Reasoning Self-Check (micro-F1 = 0.514, macro-F1 = 0.253). Both outperformed the structured Reasoning A baseline (micro-F1 = 0.498, macro-F1 = 0.245). The performance differential was driven primarily by recall rather than precision. Reasoning Narrative and Reasoning Self-Check attained substantially higher overall recall (0.599 and 0.558, respectively) compared to Reasoning A (0.504), suggesting that these prompt framings enhanced the model's sensitivity to positive findings. Conversely, certain complex prompts introduced significant decoding failures, most notably Reasoning C (failure rate 0.482) and Reasoning F (0.180), which severely hindered endto-end performance by producing unparseable outputs. Prompts that imposed overly rigid formatting constraints (Reasoning C, Reasoning F) or required explicit differential comparisons for every finding (Reasoning F) suffered from both decoding failures and reduced classification quality. In contrast, free-form narrative reasoning (Reasoning Narrative) and structured self-checking (Reasoning Self-Check) balanced interpretability with output reliability, yielding micro-F1 gains of +0.026 and +0.016 over the baseline. Notably, these results reflect MedGemma-4B's medical pre-training; as we show in the SFT analysis, the structured Reasoning A format, which underperformed here, becomes the optimal choice for general-purpose models lacking domain-specific knowledge.

<span id="page-8-0"></span>Table 1: Prompt ablation on base MedGemma-4B (CheXpert validation).

| Prompt Configuration   | CheXpert Validation Metrics |          |           |  |  |  |
|------------------------|-----------------------------|----------|-----------|--|--|--|
|                        | Micro-F1                    | Macro-F1 | Fail Rate |  |  |  |
| Reasoning Narrative    | 0.524                       | 0.270    | 0.002     |  |  |  |
| Reasoning Self-Check   | 0.514                       | 0.253    | 0.010     |  |  |  |
| Reasoning Hypothesis   | 0.357                       | 0.188    | 0.022     |  |  |  |
| Reasoning A (baseline) | 0.498                       | 0.245    | 0.000     |  |  |  |
| Reasoning B            | 0.427                       | 0.170    | 0.000     |  |  |  |
| Reasoning C            | 0.260                       | 0.132    | 0.482     |  |  |  |
| Reasoning D            | 0.329                       | 0.170    | 0.000     |  |  |  |
| Reasoning E            | 0.404                       | 0.218    | 0.000     |  |  |  |
| Reasoning F            | 0.269                       | 0.108    | 0.180     |  |  |  |

Key design cues:

Reasoning Narrative: Free-form expert narrative with evidence aggregation

Reasoning Self-Check: Region-wise scan then explicit verification pass

Reasoning Hypothesis: Hypothesis-driven comparison of candidate findings

Reasoning A: Structured 12-step checklist with strict format (Baseline)

Reasoning B: Zonal lung review with pathology differentiation

Reasoning C: Evidence-heavy report with label-confidence matrix

Reasoning D: Definition-guided labeling with strict criteria

Reasoning E: Second-look pass over anatomic blind spots

Reasoning F: Differential diagnosis required for each abnormality

#### 4.2 Effect of Supervised Fine-Tuning

To investigate whether instruction format effectiveness depends on medical pretraining, we fine-tuned two vision-language models, MedGemma-4B (medically pre-trained) and Qwen2.5-VL-3B-Instruct (general-purpose, no medical training), on four distinct instruction formats: Reasoning Narrative (free-form radiologic narrative), Reasoning A (structured 12-step reasoning), Free Reasoning (concise analysis followed by labels), and Only Label (direct label output with no rationale).

MedGemma-4B Results. For the medically pre-trained model, a striking trade-off emerged between micro-F1 and macro-F1 performance (Table [2\)](#page-10-0). The Only Label variant achieved the highest micro-F1 (0.461), while Free Reasoning obtained the best macro-F1 (0.253). This divergence reflects label frequency effects: direct label prediction excels at high-support conditions like "No Finding" and "Support Devices" that dominate micro-averaged metrics, but struggles with rare pathologies that carry equal weight in macro-averages. Notably, Reasoning A underperformed significantly across both metrics (micro-F1 = 0.293, macro-F1 = 0.139), suggesting that explicit structured reasoning provides little benefit when the model already possesses domain-specific feature representations.

Qwen2.5-VL-3B-Instruct Results. The general-purpose model exhibited a strikingly different pattern (Table [3\)](#page-10-1). Reasoning A, the structured 12 step reasoning format, achieved the best performance by a substantial margin (micro-F1 = 0.371, macro-F1 = 0.208), outperforming all other variants. In contrast, Only Label, which succeeded on MedGemma, dropped to mediocre performance (micro-F1 = 0.249, macro-F1 = 0.080). Free Reasoning also struggled (micro-F1 = 0.200, macro-F1 = 0.131), while Reasoning Narrative showed moderate scores but high failure rates (17%).

Cross-Model Interpretation. This complete ranking reversal reveals a fundamental insight: medical pre-training encodes domain-specific feature representations that enable direct label mapping, whereas general-purpose VLMs require explicit structured reasoning scaffolds to compensate for missing domain knowledge. The 12-step clinical analysis in Reasoning A provides necessary interpretive guidance for Qwen, directing attention through systematic examination of anatomical structures and pathological indicators. However, this same structure appears redundant for MedGemma, which has already internalized these reasoning patterns through medical pre-training. The failure of free-form reasoning on Qwen (but moderate success on MedGemma) further supports this hypothesis: without either pre-trained medical representations or explicit structural guidance, the model lacks the necessary inductive biases for clinical interpretation.

Decoding Reliability and Training Dynamics. Across both models, decoding reliability correlated inversely with rationale length, with Only Label producing zero format violations while longer formats increased malformation risk. The training dynamics in Figure [1](#page-11-0) show that syntax-constrained variants (Only Label, Reasoning A) converge rapidly to high token accuracy, while Free Reasoning converges gradually and saturates lower. We hypothesize this reflects output entropy rather than learning quality: templated formats have predictable token distributions that inflate accuracy metrics but encourage pattern memorization, whereas free-form traces force the model to learn semantic relationships rather than surface shortcuts.

Given its superior macro-F1 performance and balanced handling of both common and rare pathologies on the medically pre-trained model, we selected the MedGemma-4B Free Reasoning variant as the initialization checkpoint for subsequent GRPO training.

<span id="page-10-0"></span>Table 2: Supervised fine-tuning variants on MedGemma-4B (CheXpert validation).

|                     |       | Micro |       |       | Macro |       |       |  |
|---------------------|-------|-------|-------|-------|-------|-------|-------|--|
| Variant             | P     | R     | F1    | P     | R     | F1    | Fail  |  |
| Only Label          | 0.596 | 0.375 | 0.461 | 0.332 | 0.214 | 0.241 | 0.000 |  |
| Free Reasoning      | 0.480 | 0.365 | 0.415 | 0.358 | 0.227 | 0.253 | 0.032 |  |
| Reasoning Narrative | 0.565 | 0.281 | 0.376 | 0.345 | 0.141 | 0.180 | 0.210 |  |
| Reasoning A         | 0.405 | 0.230 | 0.293 | 0.381 | 0.143 | 0.139 | 0.004 |  |

<span id="page-10-1"></span>Table 3: Supervised fine-tuning variants on Qwen/Qwen2.5-VL-3B-Instruct (CheXpert validation).

|                     |       | Micro |       |       | Macro |       |       |  |
|---------------------|-------|-------|-------|-------|-------|-------|-------|--|
| Variant             | P     | R     | F1    | P     | R     | F1    | Fail  |  |
| Only Label          | 0.333 | 0.199 | 0.249 | 0.153 | 0.122 | 0.080 | 0.000 |  |
| Free Reasoning      | 0.200 | 0.200 | 0.200 | 0.150 | 0.142 | 0.131 | 0.056 |  |
| Reasoning Narrative | 0.396 | 0.198 | 0.264 | 0.122 | 0.076 | 0.075 | 0.170 |  |
| Reasoning A         | 0.322 | 0.436 | 0.371 | 0.194 | 0.265 | 0.208 | 0.002 |  |

![](_page_11_Figure_0.jpeg)

<span id="page-11-0"></span>Figure 1: Comparative training dynamics across all four supervised fine-tuning variants. The right panel shows training loss convergence, while the left panel displays mean token-level prediction accuracy. Notably, syntax-constrained variants (Only Label, Reasoning A, Reasoning Narrative) converge rapidly to higher token accuracy saturation, whereas Free Reasoning exhibits slower convergence and lower saturation levels. This pattern reflects fundamental differences in output entropy rather than learning quality, with Free Reasoning's diverse, unstructured traces requiring more nuanced semantic learning compared to the predictable template patterns of structured formats.

### 4.3 Effect of Group Relative Policy Optimization

We evaluated GRPO training using both reward functions on the Free Reasoning SFT checkpoint (Table [4\)](#page-14-0). The simpler hard reward function achieved marginally better validation performance (micro-F1 = 0.391, macro-F1 = 0.258) compared to the complex nuanced reward (micro-F1 = 0.387, macro-F1 = 0.257), suggesting that explicit format enforcement combined with Jaccard similarity provided sufficient signal for policy optimization. Notably, both reward functions improved macro-F1 relative to the SFT checkpoint (0.253 → 0.258), indicating enhanced performance on low-prevalence, diagnostically challenging conditions, though micro-F1 declined slightly (0.415 → 0.391), reflecting modest performance degradation on high-support labels that dominate the microaveraged metric.

![](_page_12_Figure_0.jpeg)

Figure 2: Evolution of training metrics during the GRPO training process using nuanced rewards. The panels display (left) the total training reward, (center) the mean reward function component, and (right) the mean completion length in tokens. Faint lines represent raw data recorded at each global step, while bold lines indicate an exponential moving average (EMA) with a smoothing factor of 0.95 to highlight the underlying training trends.

![](_page_12_Figure_2.jpeg)

Figure 3: Evolution of training metrics during the GRPO training process using hard rewards. The panels display (left) the total training reward, (center) the mean reward function component, and (right) the mean completion length in tokens. Faint lines represent raw data recorded at each global step, while bold lines indicate an exponential moving average (EMA) with a smoothing factor of 0.95 to highlight the underlying training trends.

On the in-distribution CheXpert test set (Table 5), GRPO training successfully recovered the performance lost during supervised fine-tuning. The ChexReason model (SFT + GRPO) achieved a macro-F1 of 0.346, representing a 23% improvement over the SFT checkpoint (0.282) and nearly matching the MedGemma baseline (0.362). This recovery was particularly pronounced for categories such as Cardiomegaly (0.442  $\rightarrow$  0.664), Lung Opacity (0.161  $\rightarrow$  0.743), and Support Devices (0.728  $\rightarrow$  0.818), where the reward signal effectively guided the model toward more accurate label predictions.

The generalization characteristics of GRPO training reveal a more nuanced picture when evaluated across two out-of-distribution test sets. On the CheXpert test set (Table [5\)](#page-15-0), which originates from Stanford Hospital but shares the same 14-label CheXpert schema as the MIMIC-CXR-JPG training data, the ChexReason model achieves strong performance (macro-F1 = 0.346), substantially improving over the SFT checkpoint (0.282 → 0.346, +23%). However, on the NIH Chest X-ray test set (Table [6\)](#page-15-1), which employs a distinct labeling methodology reduced to nine CheXpert-compatible categories, the ChexReason model regresses to baseline levels (macro-F1 = 0.243), representing a 19% degradation relative to the SFT checkpoint (0.299 → 0.243) and matching the pretrained MedGemma baseline performance.

This divergent behavior across out-of-distribution datasets suggests that GRPO optimization aligns the model to the semantic structure of the CheXpert labeling convention rather than to generalizable radiologic features. The shared label schema between MIMIC-CXR-JPG (training) and CheXpert (test), encompassing identical pathology definitions, labeling granularity, and uncertainty encoding, enables the reward signal to exploit label-specific patterns that transfer within this ecosystem but fail to generalize to alternative labeling methodologies. Paradoxically, the SFT checkpoint achieves its best performance on the NIH dataset (macro-F1 = 0.299) despite exhibiting the weakest scores on the CheXpert test set (0.282), suggesting that teacher-generated reasoning traces capture visual-semantic relationships that transcend specific label taxonomies, even as they degrade performance on the validation distribution used to guide training.

These findings reveal a fundamental tension in reinforcement learning for small vision-language models in medical imaging: reward-based optimization excels at recovering performance within a specific labeling framework but appears to fail eliciting the kind of generalizable diagnostic reasoning that transfers across institutional conventions and annotation methodologies. The fact that the simpler hard reward performs comparably to the carefully engineered nuanced reward further supports this interpretation. Both reward functions provide sufficient supervision to align the model to CheXpert label semantics, but neither addresses the underlying challenge of learning institution-agnostic radiologic representations. This suggests that improving cross-dataset generalization may require architectural modifications, multi-dataset training curricula, or reward formulations that explicitly penalize overfitting to labeling conventions rather than post-hoc RL fine-tuning on single-institution data.

<span id="page-14-0"></span>Table 4: Reinforcement learning using GRPO in various setups (CheXpert validation). All variants use the MedGemma-4B Free-Reasoning SFT model checkpoint.

|                    |                | Micro |       |       | Macro |       |       |       |
|--------------------|----------------|-------|-------|-------|-------|-------|-------|-------|
| Reward<br>Function | Training Steps | P     | R     | F1    | P     | R     | F1    | Fail  |
| Nuanced            |                |       |       |       |       |       |       |       |
|                    | Best F1-Score  | 0.342 | 0.445 | 0.387 | 0.248 | 0.311 | 0.257 | 0.020 |
|                    | Maximum        | 0.335 | 0.424 | 0.374 | 0.228 | 0.271 | 0.232 | 0.030 |
| Hard               |                |       |       |       |       |       |       |       |
|                    | Best F1-Score  | 0.343 | 0.455 | 0.391 | 0.250 | 0.328 | 0.258 | 0.028 |
|                    | Maximum        | 0.340 | 0.418 | 0.375 | 0.243 | 0.279 | 0.242 | 0.018 |

#### 4.4 Generalization Analysis

The evolution of SFT training metrics in Figure [1](#page-11-0) indicates that the Free Reasoning variant successfully masters the structured output format with stable convergence. However, this optimization masks complex generalization challenges revealed during out-of-distribution testing.

A striking pattern emerges in cross-dataset transfer. High-performing models like NV-Reason (macro-F1 = 0.755 on CheXpert) and ChexReason (0.346) exhibit dramatic degradation on the NIH dataset, dropping 61% and 30% respectively (Tables [5](#page-15-0) and [6\)](#page-15-1). This parallel failure suggests aggressive optimization for CheXpert-specific patterns trades generalization for benchmark performance. Empirical evidence supports this: Compton et al. [\[6\]](#page-17-9) demonstrated that models often rely on deep-seated dataset artifacts like hospital source discrimination. In contrast, the SFT checkpoint uniquely improves on NIH (0.282 → 0.299) despite weaker CheXpert scores, whereas the MedGemma baseline drops 33%.

We hypothesize this stems from the susceptibility of small vision-language models (3-4B parameters) to spurious patterns. Murali et al. [\[23\]](#page-19-10) demonstrated that spurious features are learned early and preferentially when they contain high "usable information," a process entrenched by subsequent optimization on benchmark datasets. However, our teacher-guided SFT process appears to disrupt this overfitting without re-introducing benchmark-specific pressures.

Aligned with knowledge distillation literature, Boland et al. [\[5\]](#page-17-10) showed that distillation from unbiased teachers significantly reduces spurious feature learning. By forcing alignment with Gemini-generated reasoning traces that emphasize diagnostic principles rather than shortcuts, our SFT approach acts as implicit knowledge distillation. These relearned, institution-agnostic principles prove less effective on the artifact-heavy CheXpert distribution but transfer more reliably to the NIH dataset.

The observation that NV-Reason and ChexReason lose this generalization

advantage when optimized for higher CheXpert scores supports the notion that current benchmark-driven development practices may inadvertently disfavor models with broader real-world viability. This interpretation aligns with Vogt-Lowell et al. [35], who report that end-to-end fine-tuning can compromise out-of-distribution robustness. Collectively, these findings imply that the deliberate constraint of supervised fine-tuning with reasoning-aligned guidance may help preserve institution-agnostic generalization capabilities by disrupting the learned shortcuts often associated with competitive benchmark performance.

<span id="page-15-0"></span>Table 5: Comparative F1-Score performance across five models on the outof-sample original Chexpert dataset, including overall aggregated performance. The best performing model for each category is highlighted in bold.

|                            | F1-Score  |          |         |          |              |  |  |
|----------------------------|-----------|----------|---------|----------|--------------|--|--|
| Category                   | NV-Reason | MedGemma | Qwen2.5 | MedGemma |              |  |  |
|                            |           |          |         | (SFT)    | (ChexReason) |  |  |
| Atelectasis                | 0.758     | 0.288    | 0.354   | 0.229    | 0.240        |  |  |
| Cardiomegaly               | 0.853     | 0.627    | 0.385   | 0.442    | 0.664        |  |  |
| Consolidation              | 0.188     | 0.000    | 0.097   | 0.205    | 0.000        |  |  |
| Edema                      | 0.807     | 0.174    | 0.128   | 0.385    | 0.132        |  |  |
| Enlarged Cardiomediastinum | 0.879     | 0.016    | 0.407   | 0.075    | 0.000        |  |  |
| Fracture                   | 0.750     | 0.143    | 0.009   | 0.150    | 0.200        |  |  |
| Lung Lesion                | 0.875     | 0.300    | 0.042   | 0.211    | 0.133        |  |  |
| Lung Opacity               | 0.961     | 0.748    | 0.464   | 0.161    | 0.743        |  |  |
| No Finding                 | 0.775     | 0.625    | 0.318   | 0.579    | 0.607        |  |  |
| Pleural Effusion           | 0.822     | 0.634    | 0.264   | 0.464    | 0.625        |  |  |
| Pleural Other              | 0.667     | 0.000    | 0.026   | 0.000    | 0.000        |  |  |
| Pneumonia                  | 0.429     | 0.400    | 0.031   | 0.143    | 0.400        |  |  |
| Pneumothorax               | 0.842     | 0.308    | 0.033   | 0.179    | 0.286        |  |  |
| Support Devices            | 0.970     | 0.808    | 0.637   | 0.728    | 0.818        |  |  |
| Overall Average (Macro F1) | 0.755     | 0.362    | 0.228   | 0.282    | 0.346        |  |  |

<span id="page-15-1"></span>Table 6: Comparative F1-Score performance across five models on the out-ofdistribution dataset, including overall aggregated performance. The best performing model for each category is highlighted in bold.

|                            | F1-Score  |          |         |          |              |  |  |
|----------------------------|-----------|----------|---------|----------|--------------|--|--|
| Category                   | NV-Reason | MedGemma | Qwen2.5 | MedGemma |              |  |  |
|                            |           |          |         | (SFT)    | (ChexReason) |  |  |
| Atelectasis                | 0.422     | 0.300    | 0.000   | 0.264    | 0.260        |  |  |
| Cardiomegaly               | 0.543     | 0.482    | 0.000   | 0.440    | 0.312        |  |  |
| Consolidation              | 0.056     | 0.029    | 0.109   | 0.200    | 0.178        |  |  |
| Edema                      | 0.273     | 0.522    | 0.000   | 0.526    | 0.429        |  |  |
| Lung Lesion                | 0.397     | 0.195    | 0.115   | 0.113    | 0.179        |  |  |
| No Finding                 | 0.283     | 0.275    | 0.215   | 0.343    | 0.250        |  |  |
| Pleural Other              | 0.152     | 0.180    | 0.000   | 0.248    | 0.082        |  |  |
| Pneumonia                  | 0.000     | 0.202    | 0.117   | 0.138    | 0.182        |  |  |
| Pneumothorax               | 0.552     | 0.000    | 0.000   | 0.416    | 0.317        |  |  |
| Overall Average (Macro F1) | 0.297     | 0.243    | 0.062   | 0.299    | 0.243        |  |  |

# 5 Conclusion

This study examined whether R1-style training, combining supervised finetuning with Group Relative Policy Optimization, enhances multilabel chest X-ray classification in small vision-language models under low-resource conditions. We trained MedGemma-4B and Qwen2.5-VL-3B-Instruct using teachergenerated reasoning traces from Gemini 2.5, employing only 2,000 SFT and 1,000 RL samples on a single A100 GPU. Our ChexReason model achieved a 23% improvement over the SFT checkpoint on the standard CheXpert benchmark (macro-F1 = 0.346), despite training on out-of-distribution MIMIC-CXR-JPG data.

However, this success severely compromised cross-dataset generalization. On the NIH Chest X-ray dataset, which uses different labeling methodology, ChexReason performance degraded by 19%, reverting to baseline levels (macro-F1 = 0.243). This mirrors the high-resource NV-Reason-CXR-3B model, which also saw massive drops on NIH data despite state-of-the-art CheXpert scores. Paradoxically, our SFT checkpoint uniquely improved on NIH (0.282 → 0.299 macro-F1) while showing weaker CheXpert performance, suggesting teacherguided reasoning traces capture more generalizable visual-semantic relationships than reward-optimized outputs. Our cross-model comparison further revealed that instruction format effectiveness depends on medical pre-training. Qwen2.5- VL-3B, lacking domain-specific representations, performed best with explicit 12-step structured reasoning (macro-F1 = 0.208), while MedGemma-4B performed best with direct label prediction (macro-F1 = 0.253). This suggests structured reasoning scaffolds can compensate for missing domain knowledge, but become redundant, or even detrimental, when medical pre-training has already internalized clinical reasoning patterns. These findings reveal a tension in reinforcement learning for small medical VLMs: reward signals recover benchmark performance by exploiting dataset-specific semantics, but potentially degrade transferability across institutions. Since both the high-resource NVIDIA model and our low-resource ChexReason model failed similarly, the issue likely stems from the RL fine-tuning paradigm itself when applied to small models on standardized benchmarks. Consequently, R1-style training on hard labels such as CheXpert may be counterproductive for clinical deployment requiring robustness across diverse populations. Practitioners under resource constraints may be better served by curated, supervised fine-tuning rather than aggressive benchmark optimization.

#

---

## 📎 File Information

- **PDF Location:** `papers_pdf\2512.23090v2_Benchmark_Success_Clinical_Failure_When_Reinforcem.pdf`
- **Extracted:** 2026-01-05 18:46:08
- **Extraction Method:** Marker AI

---

*This markdown contains only key sections (Abstract, Introduction, Conclusion) for quick reference. See the full PDF for complete content including Methods, Results, Discussion, and References.*
