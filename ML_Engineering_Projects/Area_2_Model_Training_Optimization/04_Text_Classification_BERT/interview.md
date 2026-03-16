# Text Classification with BERT — Interview Preparation Guide

> **Stack**: transformers (DistilBERT), torch, accelerate, evaluate, scikit-learn, wandb
> **Datasets**: AG News, IMDB, GLUE/SST-2 | **Model**: distilbert-base-uncased

---

## Quick Reference Card

| Component | Key Detail |
|---|---|
| Model | DistilBERT (66M params, 40% smaller than BERT-base) |
| Architecture | 6 layers, 768 hidden, 12 attention heads |
| Tokenizer | WordPiece, vocab 30,522 tokens |
| Fine-tune LR | 2e-5 to 5e-5 (critical range) |
| Optimizer | AdamW (eps=1e-8, betas=(0.9,0.999)) |
| Scheduler | Linear warmup + linear decay |
| Gradient Clip | max_norm=1.0 |
| Best checkpoint | Saved by validation F1 (not val loss) |
| Training trick | Gradient accumulation for small VRAM |
| Evaluation | Macro F1, Confusion Matrix |

---

## 1. Core Concepts & Theory

### 1.1 Transformer Architecture & BERT Family

**Q1. ⭐ Describe the Transformer encoder architecture and how BERT uses it.**

BERT (Bidirectional Encoder Representations from Transformers) is a stack of Transformer encoder blocks — it uses only the encoder portion and discards the decoder entirely. Each encoder block contains a multi-head self-attention sublayer followed by a position-wise feed-forward network (FFN), each wrapped with residual connections and layer normalization. BERT-base has 12 such blocks, 768 hidden dimensions, 12 attention heads, and roughly 110M parameters; BERT-large scales to 24 blocks, 1024 hidden, 16 heads, and ~340M params. The key innovation is bidirectionality: BERT can attend to both left and right context simultaneously, unlike GPT which is autoregressive left-to-right. This bidirectionality is enabled by the Masked Language Model (MLM) pretraining objective where 15% of tokens are masked and the model must predict them using surrounding context from both sides.

**Follow-up 1:** Why can't a standard language model (like GPT) be bidirectional?
Standard LMs predict the next token given all previous tokens — allowing right-to-left attention would cause information leakage (the model could "see" the answer before predicting it). GPT uses causal masking (upper triangular mask set to -inf) in attention to enforce this constraint. BERT sidesteps this by using a cloze-style objective instead of next-token prediction.

**Follow-up 2:** What is the [CLS] token and how is it used for classification?
The [CLS] (classification) token is a special token prepended to every input sequence. After passing through all encoder layers, the final hidden state of [CLS] serves as the aggregate sequence representation. For fine-tuning on classification tasks, a linear head is placed on top of the [CLS] embedding: `logits = W * h_[CLS] + b`. The intuition is that the [CLS] token attends to all other tokens in every layer, accumulating a holistic representation of the entire sequence.

---

**Q2. ⭐ Explain multi-head self-attention. How does it differ from single-head attention?**

Self-attention computes query (Q), key (K), and value (V) matrices from the input via learned projections, then computes attention weights as `softmax(QK^T / sqrt(d_k)) * V`. Single-head attention does this once. Multi-head attention runs h separate attention operations in parallel, each with its own Q/K/V projections of dimension d_model/h, then concatenates and projects the results: `MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_O`. The benefit is that each head can learn to attend to different aspects of the input — syntactic relationships, semantic similarity, coreference, etc. — simultaneously. The scaling factor `sqrt(d_k)` prevents the dot products from growing large and pushing softmax into regions with near-zero gradients.

```
Multi-Head Attention (h=12 for BERT-base, d_k=64)
───────────────────────────────────────────────────
Input X (seq_len x 768)
        |
   ┌────┴────┐
   |  Split  |  → 12 heads, each d_k=64
   └────┬────┘
        |
  Head 1      Head 2   ...   Head 12
  Q1,K1,V1   Q2,K2,V2       Q12,K12,V12
  Attn_1     Attn_2    ...   Attn_12
        |
   ┌────┴────┐
   | Concat  |  → (seq_len x 768)
   └────┬────┘
        |
    Linear W_O
        |
    Output (seq_len x 768)
```

---

**Q3. ⭐ What is positional encoding and why does BERT need it?**

The Transformer attention mechanism is permutation-invariant — it treats the input as a set, not a sequence. Without positional information, "dog bites man" and "man bites dog" would produce identical representations. BERT adds learned positional embeddings (not the sinusoidal encodings from the original Transformer paper) to the token embeddings before the first encoder layer. BERT's positional embeddings are a lookup table with one 768-dim vector per position, up to max_position_embeddings=512. They are learned during pretraining just like all other parameters. The final input embedding is the sum of: token embedding + positional embedding + segment embedding (sentence A vs B).

---

**Q4. ⭐ What are BERT's two pretraining objectives?**

BERT is pretrained with two objectives simultaneously. First, **Masked Language Model (MLM)**: 15% of input tokens are randomly selected; of those, 80% are replaced with [MASK], 10% with a random token, and 10% are left unchanged. The model must predict the original token for all masked positions. This forces the model to develop bidirectional contextual representations. Second, **Next Sentence Prediction (NSP)**: given two sentences A and B, the model must predict whether B is the actual next sentence in the corpus (50% of the time it is, 50% it is a random sentence). This trains the [CLS] representation to capture inter-sentence relationships, useful for tasks like QA and NLI. Note: later research (RoBERTa, DistilBERT) showed NSP may not be necessary and can actually hurt performance on some tasks.

---

**Q5. ⭐⭐ How does the [CLS] token representation change across fine-tuning, and what are its limitations for representing long documents?**

During pretraining, [CLS] is only optimized for the NSP objective, making it a somewhat arbitrary aggregate representation. During fine-tuning for classification, backpropagation through the cross-entropy loss specifically trains [CLS] to encode task-relevant information, and its representation shifts dramatically from the pretrained state. The limitation is that [CLS] is a single 768-dim vector trying to compress an entire document — for long documents (>512 tokens, which DistilBERT also caps at), critical information gets truncated or diluted. Empirically, for longer documents, mean pooling over all token representations often outperforms [CLS] because it spreads the representational burden.

**Follow-up A:** What strategies exist for classifying documents longer than 512 tokens?
Three main strategies: (1) **Truncation**: simply take the first 512 tokens — works well when the most relevant content is at the beginning (news articles, reviews). (2) **Sliding window with aggregation**: process overlapping windows of 512 tokens, then aggregate (mean/max pool) the [CLS] representations or all token representations. (3) **Hierarchical models**: split document into sentences, encode each sentence independently with BERT, then feed sentence-level embeddings into a simpler model (LSTM, Transformer) for document-level classification.

**Follow-up B:** For a news article classification task where articles average 800 tokens, which strategy would you choose and why?
For news classification (AG News is the dataset here), truncation to first 512 tokens is the pragmatic first choice because news articles front-load the most important information in the lede paragraph. A sliding window approach adds latency proportional to the number of windows. I would first benchmark truncation vs. sliding window with mean pooling and check if the accuracy difference justifies the 2-3x inference cost increase.

**Follow-up C:** How does Longformer or BigBird solve this problem architecturally?
Longformer replaces global full attention (O(n^2)) with local sliding window attention (O(n*w) where w is window size) plus global attention for special tokens like [CLS]. BigBird additionally includes random attention. Both allow processing sequences of 4096+ tokens at roughly linear complexity. For the [CLS] token specifically, Longformer gives it global attention (it attends to and is attended by all tokens), preserving the fine-tuning paradigm.

---

**Q6. ⭐ What tokens does BERT/DistilBERT use as special tokens and what are their roles?**

BERT uses three special tokens: **[CLS]** (index 101) is prepended to every sequence and its final hidden state is used as the sequence representation for classification tasks; **[SEP]** (index 102) marks the boundary between sentence A and sentence B in two-sentence tasks, and also terminates each sequence; **[PAD]** (index 0) is used to pad shorter sequences to the same length within a batch, with an accompanying attention mask (0 for pad positions, 1 for real tokens) so the model ignores padding. DistilBERT uses the same tokenizer and special tokens but removes the segment embeddings (since it was trained without NSP).

---

### 1.2 DistilBERT: Knowledge Distillation

**Q7. ⭐ What is knowledge distillation and how was DistilBERT created?**

Knowledge distillation is a model compression technique where a smaller **student** model is trained to mimic a larger **teacher** model's outputs rather than training from scratch on ground truth labels alone. DistilBERT uses BERT-base as the teacher. During distillation pretraining, the student is trained with a combined loss: soft label cross-entropy (matching the teacher's softmax probability distribution over vocabulary), hard label cross-entropy (standard MLM loss against true token), and cosine embedding loss (aligning the student's hidden states with the teacher's). The key insight is that the soft probability distribution from the teacher carries more information than one-hot labels — even the near-zero probabilities for wrong tokens encode similarity structure.

```
Knowledge Distillation Process
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input Text
     │
     ├──────────────────────┐
     ▼                      ▼
 [TEACHER]              [STUDENT]
 BERT-base              DistilBERT
 12 layers              6 layers
 110M params            66M params
     │                      │
  Soft logits             Soft logits
  (vocab dist)           (vocab dist)
     │                      │
     └──────────┬───────────┘
                ▼
         Combined Loss:
    ┌─────────────────────────┐
    │ L = α * L_CE_soft       │  ← soft labels (T=temperature)
    │   + β * L_CE_hard       │  ← ground truth MLM
    │   + γ * L_cosine        │  ← hidden state alignment
    └─────────────────────────┘
```

---

**Q8. ⭐ What are the exact size/speed numbers for DistilBERT vs BERT-base?**

DistilBERT-base has 66M parameters versus BERT-base's 110M — that is 40% fewer parameters. On inference, DistilBERT is approximately 60% faster (roughly 2.5x throughput on CPU) and uses 40% less memory. On GLUE benchmarks, DistilBERT retains approximately 97% of BERT's performance — for example, on SST-2 DistilBERT achieves ~91.3% accuracy vs BERT's ~93.5%. The distillation was done on the same pretraining corpus (English Wikipedia + BookCorpus). The 6-layer architecture was chosen by initializing student layers from every other teacher layer (layers 0,2,4,6,8,10 of BERT-base), which provided a much better starting point than random initialization.

---

**Q9. ⭐⭐ Explain the temperature parameter in knowledge distillation and why it matters.**

In knowledge distillation, temperature T controls the softness of the teacher's probability distribution: `p_i = softmax(z_i / T)`. At T=1 (normal softmax), the distribution is sharply peaked around the top prediction. As T increases (e.g., T=4), the distribution becomes softer, spreading probability mass to non-top tokens. This softer distribution contains more information about inter-class similarities — for instance, the teacher may assign 0.6 probability to "terrible" being masked, 0.3 to "awful," and 0.1 to "horrible," which tells the student these three words are semantically similar in context. If T=1 was used, the distribution would be 0.99/0.005/0.005 and the similarity information would be lost. Hinton et al. found T=4-10 to be optimal in practice.

**Follow-up A:** Why do you use the same temperature T in both teacher and student during distillation, and then T=1 for inference?
Using the same T in both ensures the student is learning to match the teacher's *relative probability ratios*, not absolute values. If teacher uses T=4 but student uses T=1, the student's loss gradients would be mismatched. At inference time, you reset to T=1 to get the sharpest, most confident predictions. The temperature is effectively a training trick, not a deployment parameter.

**Follow-up B:** How does the cosine embedding loss component work in DistilBERT's distillation?
The cosine loss minimizes `1 - cos(h_student, h_teacher)` where h_student and h_teacher are hidden state vectors from corresponding layers. Since student and teacher have the same hidden dimension (768), direct comparison is possible. This loss forces the student's internal representations to align geometrically with the teacher's, not just to match output distributions. Without this, the student might arrive at completely different internal representations that happen to produce similar output distributions.

**Follow-up C:** When would you prefer DistilBERT over BERT-base for a production classification system?
Choose DistilBERT when: inference latency is a hard constraint (e.g., <50ms SLA), batch sizes are small or compute is CPU-only, cost optimization is required (fewer GPU hours), and the task is relatively simple (binary or 4-class classification on clean text). Choose BERT-base when: accuracy improvements of even 1-2% translate to significant business impact (e.g., medical or legal classification), multi-task learning is planned, or the dataset is small (<1000 examples) where extra capacity helps generalization.

---

**Q10. ⭐ How is DistilBERT different from simply pruning or quantizing BERT?**

These are three different compression techniques. Pruning removes individual weights (unstructured pruning) or entire attention heads/neurons (structured pruning) from a trained BERT model — it starts with a full model and removes components. Quantization reduces the numerical precision of weights (FP32 → INT8 or FP16) without changing architecture. Knowledge distillation trains a completely new, architecturally different smaller model from scratch using the teacher's outputs as supervision — it does not start from the teacher's weights except for weight initialization. DistilBERT combines distillation with a specific architectural choice (half the layers). You can further apply quantization to DistilBERT for additional compression — this is called "cascaded compression."

---

### 1.3 Fine-tuning Strategies

**Q11. ⭐ What is the difference between fine-tuning and feature extraction for BERT?**

**Feature extraction** (also called "frozen encoder"): the pretrained BERT weights are kept frozen, and only a task-specific head (linear layer, MLP) is trained. BERT acts as a fixed feature extractor, producing embeddings that are fed to the head. This is fast, requires little data, and avoids catastrophic forgetting — but the representations are not adapted to the target task.

**Fine-tuning**: all BERT weights are updated during training on the target task, along with the head. This typically yields significantly better performance because the representations adapt. However, it requires more data, is more expensive, and risks catastrophic forgetting on small datasets. In this project, `freeze_encoder_layers()` implements a middle-ground: lower layers (capturing universal linguistic features) can be frozen while upper layers (capturing task-specific features) are fine-tuned.

---

**Q12. ⭐⭐ Explain the layer-wise learning rate decay strategy for fine-tuning transformers.**

BERT's lower layers capture universal linguistic features (syntax, morphology) that transfer across tasks, while upper layers capture more task-specific, semantic features. Applying a single uniform learning rate to all layers is suboptimal — the lower layers need to change very little while upper layers need to adapt more. Layer-wise learning rate decay (LLRD) assigns a different learning rate to each layer: `lr_layer_i = lr_base * decay_factor^(num_layers - i)`. For example, with lr_base=5e-5 and decay=0.9: the top layer gets 5e-5, the next 4.5e-5, then 4.05e-5, down to about 1.74e-5 for the first layer. This significantly improves fine-tuning stability, especially on small datasets.

**Follow-up A:** In the project's `freeze_encoder_layers()`, what is the recommended freezing strategy?
A common strategy is to freeze the embedding layer and the first N/2 transformer blocks (e.g., layers 0-2 for DistilBERT's 6 layers), and fine-tune layers 3-5 plus the classification head. This provides regularization on small datasets and reduces training time. The embedding layer almost never needs to be updated for downstream tasks since WordPiece embeddings are already high quality.

**Follow-up B:** How do you decide how many layers to freeze?
This is a hyperparameter to tune. Practical heuristic: if your training set has <5000 examples, freeze more layers (bottom 2/3); if you have 10k-100k examples, fine-tune all layers with LLRD; if >100k, full fine-tuning with uniform LR often works. Also monitor if training loss decreases faster than validation loss (overfitting) — if yes, freeze more layers.

**Follow-up C:** What is "probing" in the context of BERT, and how does it inform fine-tuning strategy?
Probing is a diagnostic technique where you train a simple linear classifier on top of each BERT layer's representations to predict a linguistic property (POS tags, syntactic trees, NER). Probing studies have shown: lower layers (1-3) capture surface features like morphology and POS; middle layers (4-8) capture syntax; upper layers (9-12) capture semantic and task-specific features. This empirically justifies freezing lower layers for downstream tasks.

---

**Q13. ⭐ What is catastrophic forgetting in fine-tuning and how do you mitigate it?**

Catastrophic forgetting occurs when fine-tuning on a target task causes the model to lose the general linguistic knowledge captured during pretraining. It manifests as strong performance on the fine-tuning task but degraded performance on general NLP tasks. Mitigation strategies: (1) small learning rates (2e-5 to 5e-5 — the range widely validated for BERT fine-tuning); (2) early stopping to prevent over-fitting to task-specific patterns; (3) layer freezing for lower layers; (4) learning rate warmup to start with very small updates; (5) L2 regularization (weight decay) keeping weights close to pretrained initialization; (6) Elastic Weight Consolidation (EWC) which adds a penalty term proportional to the importance of each weight for previous tasks.

---

**Q14. ⭐⭐ When would you choose PEFT/LoRA over full fine-tuning?**

PEFT (Parameter-Efficient Fine-Tuning) methods like LoRA (Low-Rank Adaptation) are preferable when: (1) **GPU memory is constrained** — LoRA for a 7B model requires ~16GB vs ~80GB for full fine-tuning; (2) **serving multiple tasks** — you can maintain one base model and swap small LoRA adapters (typically <1% of base model size) per task; (3) **few-shot/small dataset** — LoRA's reduced parameter count acts as strong regularization; (4) **catastrophic forgetting is critical to avoid** — base model weights are frozen. LoRA adds rank-decomposition matrices to Q/K/V projections: `W' = W + BA` where B is d×r and A is r×k with rank r << d (typically r=4 to 32). For DistilBERT with r=16, you train ~1.8M params instead of 66M. Trade-off: LoRA may underperform full fine-tuning by 1-3% on tasks requiring deep representation adaptation.

**Follow-up A:** What does the rank r control in LoRA?
Rank r controls the expressivity of the adapter. Higher r allows more representational change but increases parameter count and overfitting risk. r=1 forces almost no change; r=64 approaches full fine-tuning expressivity. For classification tasks on standard benchmarks, r=8 to r=16 typically captures the necessary task-specific adaptation with minimal overhead.

**Follow-up B:** Can you combine LoRA with quantization?
Yes — this is QLoRA (Quantized LoRA). The base model is quantized to 4-bit NormalFloat (NF4) precision and kept frozen; LoRA adapters are trained in BF16. This enables fine-tuning 65B parameter models on a single 48GB GPU. The key innovation is that gradients flow through the quantized base model to the LoRA adapters without needing to dequantize.

**Follow-up C:** What other PEFT methods exist beyond LoRA?
Prefix Tuning prepends trainable soft tokens to the key-value pairs in each attention layer. Prompt Tuning prepends trainable tokens only to the input embedding layer (simpler but less powerful). Adapter modules insert small bottleneck FFN layers within each Transformer block. BitFit fine-tunes only bias terms. IA3 scales attention keys/values and FFN activations with learned vectors. Each has different trade-offs in parameter count, performance, and implementation complexity.

---

### 1.4 Training Dynamics (AdamW, Warmup, Clipping)

**Q15. ⭐ Why is AdamW preferred over Adam for fine-tuning BERT?**

The key difference is how weight decay is applied. Standard Adam applies L2 regularization in the gradient update step — this is mathematically equivalent to adding `λ * w` to the gradient, which means the effective weight decay is scaled by the adaptive learning rate and varies per-parameter. AdamW (Decoupled Weight Decay Regularization, Loshchilov & Hutter 2019) decouples weight decay from the gradient update: `w_t+1 = w_t - α * m̂_t/(√v̂_t + ε) - α * λ * w_t`. The weight decay term is independent of the gradient magnitude. For BERT's diverse layer magnitudes, this decoupled regularization is crucial — it consistently prevents weights from growing too large regardless of their gradient scale. Empirically, AdamW achieves better generalization and faster convergence than Adam on transformer fine-tuning tasks.

---

**Q16. ⭐ Explain the linear warmup + linear decay learning rate schedule. Why is warmup critical for transformers?**

The schedule has two phases. During **warmup** (typically first 6-10% of total training steps), the learning rate increases linearly from 0 to `lr_max`. During **decay**, the LR decreases linearly from `lr_max` back to 0 over the remaining steps. In this project: `get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)`. Warmup is critical because at the start of fine-tuning, Adam's second-moment estimates `v̂` are initialized to 0 and take many steps to converge to accurate variance estimates. Using the full learning rate immediately can produce very large, noisy gradient steps in early iterations, destabilizing the pretrained weights. Warmup keeps step sizes small while the optimizer's statistics stabilize.

```
LR Schedule Visualization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LR
  ^
  |          /\
5e-5|         /  \
  |        /    \
  |       /      \
  |      /        \
0 |_____/          \____________
  |--warmup--|------decay-------|
  0%        6%               100%
                 Training steps
```

---

**Q17. ⭐⭐ Why is gradient clipping (max_norm=1.0) particularly important for transformer fine-tuning?**

Transformers are susceptible to exploding gradients during fine-tuning because of their deep architecture and the interaction between attention weights and gradient flow. Without clipping, a single extreme gradient can update weights dramatically, overwriting the carefully learned pretrained representations. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` computes the global L2 norm of all gradients and if it exceeds 1.0, scales all gradients proportionally: `g_clipped = g * (max_norm / ||g||_2)`. This preserves gradient direction while bounding step size. The value 1.0 is the standard recommendation from the BERT paper. Unlike gradient clipping by value (which clips each gradient independently and can distort the update direction), norm clipping is more principled.

**Follow-up A:** How does gradient clipping interact with learning rate scheduling?
Gradient clipping and learning rate scheduling are complementary but distinct mechanisms. LR scheduling controls the magnitude of parameter updates over the training trajectory. Gradient clipping prevents any single step from being catastrophically large regardless of schedule. In practice, you need both — a high LR without clipping can still explode; clipping alone without schedule can train too slowly in later stages.

**Follow-up B:** What is gradient vanishing vs. gradient exploding, and which is more common in transformer fine-tuning?
Gradient vanishing: gradients become negligibly small during backpropagation through many layers, causing lower layers to not update. This is primarily a problem in RNNs without gating mechanisms. Gradient exploding: gradients grow exponentially large. Transformers are more prone to exploding gradients due to the multiplicative attention operations and the large number of layers, especially when learning rates are too high. Residual connections in transformers mitigate vanishing gradients by providing direct gradient paths.

**Follow-up C:** What would you observe in training if you removed gradient clipping?
You would see: (1) sudden loss spikes where the loss jumps dramatically in a single step then may recover or not; (2) NaN loss values if gradients overflow float16/32; (3) weight norm increasing rapidly (monitorable with wandb's gradient norm tracking); (4) final model performance worse than with clipping. The wandb integration in this project allows logging `gradient_norm` as a scalar every N steps, which is a critical diagnostic metric.

---

**Q18. ⭐ What learning rate range is appropriate for BERT fine-tuning, and what happens outside it?**

The empirically validated range is **2e-5 to 5e-5** based on the original BERT paper and extensive subsequent research. At **LR < 1e-5**: training is too slow, you may not converge within the typical 3-5 epochs used for fine-tuning. At **LR = 2e-5 to 3e-5**: safe, conservative choice, good for small datasets where catastrophic forgetting is a concern. At **LR = 5e-5**: faster convergence, works well with large datasets and warmup. At **LR > 1e-4**: near-certain catastrophic forgetting — the pretrained knowledge is overwritten. At **LR > 1e-3**: immediate divergence. The sweet spot depends on dataset size, warmup ratio, and number of epochs. For DistilBERT vs BERT-base, the optimal range is essentially the same.

---

### 1.5 Class Imbalance Handling

**Q19. ⭐ How do you handle class imbalance in text classification?**

Multiple complementary strategies exist. (1) **Weighted cross-entropy loss**: assign `weight = N_total / (N_classes * N_class_i)` to each class — minority classes get higher loss weight, forcing the model to pay more attention to them. PyTorch: `nn.CrossEntropyLoss(weight=class_weights)`. (2) **Oversampling**: duplicate or augment minority class examples (EDA: synonym replacement, random insertion). (3) **Undersampling**: remove majority class examples — risk losing information. (4) **Focal Loss**: `FL = -(1-p_t)^γ * log(p_t)`, downweights easy examples so the model focuses on hard, often minority-class examples. (5) **Threshold adjustment**: during inference, lower the classification threshold for minority classes based on precision-recall curve analysis.

---

**Q20. ⭐⭐ Explain focal loss and when it is preferable to weighted cross-entropy.**

Focal Loss adds a modulating factor `(1 - p_t)^γ` to standard cross-entropy: `FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)`. When the model correctly classifies an example with high confidence (p_t → 1), `(1-p_t)^γ → 0`, effectively down-weighting that example's contribution to the loss. Hard, misclassified examples retain their full loss weight since `(1-p_t)^γ → 1` when p_t is small. The `α_t` term provides class-frequency balancing similar to weighted CE. With γ=2, a well-classified example (p=0.9) contributes only `(0.1)^2 = 0.01` times its standard CE loss. Focal loss is preferable over weighted CE when: the class imbalance is moderate (not extreme), there are many easy examples drowning out the hard ones, and the model tends to be overconfident on the majority class. Weighted CE is simpler and works better for extreme imbalance (1:100+ ratio).

**Follow-up A:** How do you set the γ parameter in focal loss?
γ=0 reduces to standard cross-entropy. γ=2 is the default from the original paper (Lin et al., RetinaNet). Higher γ (3-5) focuses even more on hard examples but can cause instability. In practice, start with γ=2 and tune on validation set. For NLP tasks, γ=1 or 2 is typical. You can treat γ as a hyperparameter in your wandb sweep.

**Follow-up B:** How does the evaluation metric choice interact with class imbalance?
With imbalanced classes, accuracy is misleading — a model predicting the majority class always can achieve >90% accuracy on a 90/10 split. Macro F1 averages F1 score equally across all classes regardless of support, making it the correct metric for imbalanced evaluation. Weighted F1 (weighted by class frequency) can still be dominated by the majority class. For binary classification with severe imbalance, AUROC or AUCPR (area under precision-recall curve) are more informative.

**Follow-up C:** What is the difference between macro, micro, and weighted F1?
Macro F1: compute F1 for each class independently, then average unweighted — treats all classes equally regardless of support. Micro F1: aggregate TP/FP/FN across all classes then compute F1 — dominated by majority class, equals accuracy for multi-class. Weighted F1: average F1 weighted by class support — between macro and micro. In this project, macro F1 on validation set is used for checkpoint selection, which is appropriate for the multi-class AG News dataset (4 balanced classes) and IMDB (binary balanced).

---

### 1.6 Tokenization Deep Dive

**Q21. ⭐ How does WordPiece tokenization work?**

WordPiece is a subword tokenization algorithm that builds a vocabulary of 30,522 tokens for BERT. It starts with a character-level vocabulary and iteratively merges the pair of symbols that maximizes the likelihood of the training data (similar to BPE but using likelihood rather than frequency). During tokenization, each word is segmented into the longest matching subwords from the vocabulary: "tokenization" → ["token", "##ization"]. The "##" prefix indicates a continuation subword (not the start of a new word). This handles OOV (out-of-vocabulary) words by breaking them into known subwords. For example, the medical term "hypertriglyceridemia" would be split into several recognizable subword pieces even if the full word was never in training data.

---

**Q22. ⭐ What is the max_length parameter and what truncation strategies exist?**

`max_length=512` is the maximum number of tokens BERT/DistilBERT can process in one forward pass, set by the positional embedding table size. When an input exceeds this, truncation strategies include: (1) **truncate tail** (default): keep first 512 tokens, discard the rest — best when beginning contains most relevant info. (2) **truncate head**: discard first tokens, keep last 512 — rarely used. (3) **truncate middle** (first-128 + last-384 tokens): the "first-last-k" strategy found to work well for long document classification, keeping both introductory context and conclusion. (4) **Stride/sliding window**: process overlapping chunks, aggregate results. The `tokenizer(text, max_length=512, truncation=True, padding='max_length')` call in this project applies strategy 1 by default.

---

**Q23. ⭐⭐ What is the attention mask and why is it critical?**

The attention mask is a binary tensor of shape `(batch_size, seq_len)` where 1 indicates a real token and 0 indicates a padding token. Without the attention mask, the model would compute attention over padding tokens, which would corrupt the representations. In transformer self-attention, padding positions should receive attention weight of 0 — this is achieved by setting their pre-softmax logits to -10000 (effectively -inf) so softmax → 0. The attention mask is passed to `model(input_ids=..., attention_mask=..., ...)`. An incorrect or missing attention mask is one of the most common bugs in BERT fine-tuning implementations and can degrade performance significantly, especially with variable-length sequences.

**Follow-up A:** What is the difference between padding to max_length vs dynamic padding?
Padding to `max_length` (512) for all sequences wastes computation on padding tokens — a 50-token sequence padded to 512 has 462 padding tokens, but attention still processes all 512 positions (though attended tokens are masked). **Dynamic padding** (padding each batch to the maximum sequence length in that batch) dramatically reduces wasted computation — if your batch's longest sequence is 150 tokens, you pad to 150. This can reduce training time by 2-4x for datasets with short texts like IMDB reviews. The `DataCollatorWithPadding` from HuggingFace implements this.

**Follow-up B:** How does the tokenizer handle tokens not in the vocabulary?
Out-of-vocabulary characters are handled by the `[UNK]` token. For words composed entirely of known subwords (which covers most common words), no UNK is needed. True OOV (characters not in the character vocabulary, e.g., non-Latin scripts when using `uncased` model) map to `[UNK]`. The `uncased` model lowercases and strips accents before tokenization, which can collapse distinctions like "Résumé" → "resume."

---

### 1.7 Evaluation & Metrics

**Q24. ⭐ Why save the best checkpoint by validation F1 rather than validation loss?**

Validation loss and F1 can diverge. A model might achieve lower validation loss while F1 improves little or even decreases — particularly with class imbalance where the model learns to be confidently wrong on minority classes (low loss on easy majority examples). F1 directly measures the classification performance you care about in production. Additionally, cross-entropy loss is unbounded and can decrease even as predictions become worse calibrated. The business-relevant metric (accuracy, F1, AUC) should drive checkpoint selection. In this project, best checkpoint = argmax(val_macro_F1) over all epochs.

---

**Q25. ⭐ How do you compute and interpret a confusion matrix for multi-class classification?**

A confusion matrix is an `N×N` matrix (N = number of classes) where `C[i][j]` = number of examples truly belonging to class i but predicted as class j. The diagonal contains correct predictions. For AG News (4 classes: World, Sports, Business, Sci/Tech): off-diagonal elements reveal which classes are confused with which others. Common patterns: World vs. Business articles might get confused (economic news), or Sports vs. Sci/Tech if articles discuss sports technology. Precision for class i = `C[i,i] / sum_j C[j,i]`. Recall for class i = `C[i,i] / sum_j C[i,j]`. A normalized confusion matrix (divide each row by its sum) highlights where the model is making proportional errors.

---

**Q26. ⭐⭐ What is calibration and why does it matter for text classification?**

A classifier is well-calibrated if its predicted probabilities match empirical frequencies — when the model says 90% confidence, it should be correct 90% of the time. BERT-based classifiers often have poor calibration, being overconfident (predict 99% probability but only correct 85% of the time). Calibration matters when downstream decisions use probability thresholds or when the output feeds into a pipeline requiring uncertainty estimates. Calibration can be measured with Expected Calibration Error (ECE). **Post-hoc calibration methods**: Temperature Scaling (learn a single scalar T that divides logits), Platt Scaling (logistic regression on logits), Isotonic Regression. Temperature Scaling is the recommended method for neural networks — one parameter, trained on validation set.

**Follow-up A:** How does Temperature Scaling work mathematically?
Learn scalar T on the validation set by minimizing NLL: `p_calibrated = softmax(logits / T)`. T > 1 softens the distribution (reduces overconfidence), T < 1 sharpens it. Typically for fine-tuned transformers, T > 1 is learned because they tend to be overconfident. The scalar T is found via 1D optimization (L-BFGS or grid search).

**Follow-up B:** How would you monitor calibration drift in production?
Collect prediction probabilities for a sliding window of recent examples where ground truth is available (or can be obtained via delayed labels). Compute ECE weekly. Alert if ECE increases beyond 2x the baseline. Also monitor the distribution of max-probability predictions — if the histogram shifts toward more extreme probabilities (0.99+), that indicates increasing overconfidence.

---

## 2. System Design Discussions

**Q27. ⭐ Design a real-time text classification API serving DistilBERT with <100ms p99 latency.**

The pipeline has three key optimizations. (1) **Model optimization**: export DistilBERT to ONNX Runtime for 1.5-2x CPU speedup; apply INT8 quantization for another 2x speedup with <1% accuracy drop. (2) **Batching**: even for "real-time" use, micro-batching (batch size 4-8) improves GPU utilization dramatically. Use async request queuing (Ray Serve or Triton). (3) **Infrastructure**: deploy on GPU instance with the model pre-loaded in memory; use connection pooling; set max_length to minimum needed (e.g., 128 for short texts like tweets). Target hardware: NVIDIA T4 GPU can handle ~500 DistilBERT inferences/second at batch size 8.

```
Real-Time Serving Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Client
    |
    v
 [Load Balancer] (Azure Front Door / AWS ALB)
    |
    v
 [API Gateway]  →  rate limiting, auth
    |
    v
 [Inference Service]
 ┌─────────────────────────────────────────┐
 │  FastAPI / Triton Inference Server      │
 │  ┌─────────────────────────────────┐    │
 │  │  Request Queue (micro-batch)    │    │
 │  └──────────────┬──────────────────┘    │
 │                 v                       │
 │  ┌─────────────────────────────────┐    │
 │  │  Tokenizer (CPU, cached)        │    │
 │  └──────────────┬──────────────────┘    │
 │                 v                       │
 │  ┌─────────────────────────────────┐    │
 │  │  ONNX/TensorRT DistilBERT       │    │
 │  │  (GPU: T4 or A10G)              │    │
 │  └──────────────┬──────────────────┘    │
 │                 v                       │
 │  ┌─────────────────────────────────┐    │
 │  │  Softmax + Label Decode         │    │
 │  └─────────────────────────────────┘    │
 └─────────────────────────────────────────┘
    |
    v
  Response (label, confidence, latency_ms)
```

---

**Q28. ⭐⭐ How would you design a continual learning pipeline for text classification as new data arrives?**

The pipeline must balance adding new knowledge while preventing catastrophic forgetting. Architecture: (1) **Data flywheel**: collect production predictions → send low-confidence samples to human review → build labeled dataset. (2) **Scheduled retraining**: retrain on combined historical + new data weekly/monthly. (3) **Evaluation gate**: new model must exceed current production model's F1 by >0.5% on held-out test set AND not regress more than 0.3% on any individual class. (4) **Canary deployment**: route 5% traffic to new model for 48 hours, compare live accuracy. (5) **Rollback trigger**: auto-rollback if error rate increases >2x or confidence distribution shifts significantly.

**Follow-up A:** How do you detect data drift without ground truth labels?
Use unsupervised drift detection on input features: (1) monitor embedding distribution (cosine distance from training cluster centers using FAISS); (2) monitor input text statistics (avg token length, vocabulary OOV rate, language distribution); (3) monitor prediction distribution (KL divergence from historical class distribution). Alerts trigger at a configurable threshold, prompting human review and potential retraining.

**Follow-up B:** What is the Elastic Weight Consolidation approach to continual learning?
EWC adds a regularization term to the loss: `L_EWC = L_task + (λ/2) * Σ F_i * (θ_i - θ*_i)^2`. F_i is the Fisher information matrix diagonal (approximating the importance of parameter i for the old task), θ*_i are the old task parameters. This penalizes large changes to important parameters while allowing less important parameters to change freely for the new task. EWC has been applied to BERT fine-tuning for sequential multi-task learning.

---

**Q29. ⭐ How would you handle multilingual text classification with the same model?**

Replace DistilBERT with `distilbert-base-multilingual-cased` (104 languages, 66M params) or XLM-R-base (100 languages, 270M params). XLM-R significantly outperforms mBERT on low-resource languages. Training considerations: (1) ensure training data is language-balanced or use language-proportional sampling; (2) avoid the curse of multilinguality — adding languages dilutes capacity, so XLM-R-large (560M params) may be needed for many-language scenarios; (3) use language-specific evaluation sets. Zero-shot cross-lingual transfer is possible: fine-tune on English labeled data, evaluate on French/German without any target-language labeled data. XLM-R achieves ~80-85% of supervised performance via zero-shot transfer on NLI tasks.

---

## 3. Coding & Implementation Questions

**Q30. ⭐ Walk through the TextClassificationDataset implementation. What does __getitem__ return?**

The `TextClassificationDataset` inherits from `torch.utils.data.Dataset` and wraps the tokenizer logic. `__len__` returns the number of examples. `__getitem__(idx)` returns a dictionary with keys: `input_ids` (LongTensor of shape `[max_length]`), `attention_mask` (LongTensor of shape `[max_length]`), and `labels` (LongTensor scalar). The tokenizer call inside `__getitem__` uses `padding='max_length'`, `truncation=True`, `max_length=config.max_length`, and `return_tensors='pt'`, then `.squeeze(0)` to remove the batch dimension. This design makes the dataset compatible with PyTorch's `DataLoader` which handles batching automatically.

---

**Q31. ⭐ Explain the BERTClassifier model architecture in model.py.**

```python
class BERTClassifier(nn.Module):
    def __init__(self, config):
        # AutoModel loads DistilBERT encoder
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        # Linear head: 768 -> num_classes
        self.classifier = nn.Linear(self.encoder.config.hidden_size,
                                     config.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask)
        # [CLS] token: outputs.last_hidden_state[:, 0, :]
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
```

The `freeze_encoder_layers(model, n_layers)` function freezes the embedding layer and the first n_layers transformer blocks by setting `param.requires_grad = False` for those parameters, reducing trainable parameters and preventing catastrophic forgetting of lower-level features.

---

**Q32. ⭐ How is the AdamW optimizer configured in trainer.py? Why separate parameter groups?**

Best practice is to use two parameter groups: (1) parameters with weight decay (all weights except biases and LayerNorm parameters), (2) parameters without weight decay (biases and LayerNorm scale/shift parameters). Applying weight decay to LayerNorm parameters is harmful — LayerNorm normalizes activations and its parameters (γ and β) should not be shrunk toward zero. Biases similarly should not be regularized.

```python
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
```

---

**Q33. ⭐⭐ How does gradient accumulation work, and how do you implement it correctly?**

Gradient accumulation simulates a larger batch size by accumulating gradients over multiple forward/backward passes before calling `optimizer.step()`. This is critical when GPU VRAM can't fit the desired batch size. Implementation:

```python
ACCUM_STEPS = 4  # effective batch = batch_size * 4
for step, batch in enumerate(train_loader):
    outputs = model(**batch)
    loss = criterion(outputs, batch['labels'])
    loss = loss / ACCUM_STEPS        # scale loss
    loss.backward()                  # accumulate gradients

    if (step + 1) % ACCUM_STEPS == 0:
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

Key correctness details: (1) divide loss by ACCUM_STEPS before backward to maintain consistent gradient magnitude; (2) call `clip_grad_norm_` and `optimizer.step()` only after accumulating all micro-batches; (3) update the LR scheduler at the same cadence as optimizer.step(); (4) compute the learning rate warmup based on *optimizer steps*, not *gradient accumulation micro-steps*.

**Follow-up A:** What is the difference between `loss.backward()` with accumulated gradients vs. a true larger batch?
They are mathematically equivalent (mean gradient = sum of per-sample gradients / batch_size) *only* if BatchNorm is not involved. With BatchNorm, a true large batch computes statistics over all samples simultaneously while gradient accumulation computes over micro-batches, leading to different batch statistics. BERT uses LayerNorm (not BatchNorm), so gradient accumulation is exactly equivalent to larger batches.

**Follow-up B:** How does `accelerate` library simplify gradient accumulation?
Hugging Face's `accelerate` wraps the model with `accelerator.prepare(model, optimizer, dataloader)` and the gradient accumulation context `with accelerator.accumulate(model):` handles the loss scaling, zero_grad timing, and step timing automatically. You specify `gradient_accumulation_steps` in `AcceleratorConfig` and the library handles all the bookkeeping, including correct scheduler stepping.

---

**Q34. ⭐ How is wandb used in the trainer for experiment tracking?**

Wandb tracks the entire training run. At init: `wandb.init(project="bert-text-classification", config=asdict(train_config))` logs all hyperparameters. During training: `wandb.log({"train_loss": loss, "learning_rate": scheduler.get_last_lr()[0], "grad_norm": total_norm, "step": global_step})`. After each epoch: `wandb.log({"val_f1": val_f1, "val_loss": val_loss, "epoch": epoch})`. Wandb also logs the confusion matrix as a custom plot: `wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=class_names)})`. The sweep configuration in wandb allows hyperparameter search over lr, dropout, warmup_ratio.

---

## 4. Common Bugs & Issues

| # | Bug / Issue | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 1 | Tokenizer-model mismatch | Garbage predictions | Loading wrong tokenizer | Always load tokenizer with `AutoTokenizer.from_pretrained(model_name)` matching model |
| 2 | Missing attention mask | Poor performance esp. with variable lengths | Padding tokens attended to | Always pass `attention_mask` to model forward |
| 3 | Wrong [CLS] index | `last_hidden_state[:, 0, :]` gets wrong token | Position 0 is [CLS] for BERT but check for other models | Verify with `tokenizer.cls_token_id` |
| 4 | Label mismatch | High loss, random predictions | Dataset labels not matching model's num_labels | Check `num_labels` in config == unique labels in dataset |
| 5 | No warmup steps → NaN loss | NaN in first few steps | Optimizer variance estimates at 0, unstable updates | Add warmup: `num_warmup_steps = 0.06 * total_steps` |
| 6 | Loss not divided in gradient accumulation | Gradients 4x too large | Loss not scaled by ACCUM_STEPS | `loss = loss / accumulation_steps` |
| 7 | Scheduler stepping every micro-batch | LR decays 4x too fast with accumulation | `scheduler.step()` inside gradient accumulation loop | Step scheduler only after optimizer.step() |
| 8 | Mixed precision NaN | NaN loss after a few steps | Loss scale overflow in fp16 | Use `GradScaler` with `scaler.scale(loss).backward()` |
| 9 | Catastrophic forgetting | Val accuracy plummets during training | LR too high (>1e-4) | Reduce LR to 2e-5 to 5e-5 range |
| 10 | Padding to 512 always | Training 3x slower than needed | Using `padding='max_length'` | Use `DataCollatorWithPadding` for dynamic padding |
| 11 | Model in eval mode during training | No dropout, too confident, overfits | Missing `model.train()` before epoch | Call `model.train()` at start of each training epoch |
| 12 | No `model.eval()` + `torch.no_grad()` | Slow validation, OOM on large batch | Gradient graph built for val | Wrap eval in `model.eval()` and `torch.no_grad()` |
| 13 | Weight decay on LayerNorm | Instability, worse performance | Regularizing normalization params | Exclude `LayerNorm.weight` and `bias` from weight decay |
| 14 | Checkpoint saves every epoch | Disk fills up | No best-model logic | Track best val_f1, only save when improved |
| 15 | OOM during inference | CUDA out of memory | Batch size too large for inference | Reduce inference batch, use `torch.no_grad()`, dynamic padding |
| 16 | Tokenizer in DataLoader worker | Slow tokenization | Tokenizer not shared/pickled efficiently | Pre-tokenize dataset and cache with `dataset.map()` |
| 17 | Uncased model with cased input | Slight performance hit | Not using model's built-in lowercasing | Verify `tokenizer.do_lower_case=True` for uncased |
| 18 | Wrong train/val split seed | Inconsistent results | Different random splits each run | Set `random_state` in split, log seed to wandb |

---

## 5. Deployment — Azure

**Q35. ⭐ Describe the end-to-end Azure ML deployment pipeline for BERT text classification.**

```
Azure ML Deployment Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Azure Blob Storage]
  ├── training-data/  (AG News, IMDB, SST-2)
  └── model-artifacts/

         │ input data
         ▼
  [Azure ML Compute Cluster]
  ┌──────────────────────────────┐
  │  GPU Cluster (NC6s_v3 T4)    │
  │  DistilBERT Fine-tuning      │
  │  wandb experiment tracking   │
  └──────────────┬───────────────┘
                 │ trained model + tokenizer
                 ▼
  [Azure ML Model Registry]
  ├── distilbert-news-classifier v1.0
  ├── Metrics: val_f1=0.923
  └── Tags: dataset=ag_news, epochs=3

                 │ registered model
                 ▼
  [Azure ML Pipeline - Eval Step]
  ┌──────────────────────────────┐
  │  Evaluate on test set        │
  │  Compare vs. production model│
  │  Gate: F1 > prod_F1 + 0.005  │
  └──────────────┬───────────────┘
                 │ approval
                 ▼
  [Azure Container Registry]
  └── inference-image:latest
      (Python + FastAPI + model)

                 │ deploy
                 ▼
  [Azure Container Apps]         [Azure Cache for Redis]
  ┌─────────────────────┐              │
  │  Inference API       │◄────────────┤ cache hot predictions
  │  /predict endpoint   │             │
  │  Auto-scaling        │─────────────┘
  └──────────┬──────────┘
             │ logs + metrics
             ▼
  [Application Insights]
  ├── Latency p50/p95/p99
  ├── Prediction distribution
  └── Error rate alerts
```

---

**Q36. ⭐ What Azure ML Compute SKU would you choose for DistilBERT fine-tuning and why?**

For DistilBERT fine-tuning on AG News (120k examples): **Standard_NC6s_v3** (1x V100 16GB, $3.06/hr on Azure) is the minimum viable option — DistilBERT with batch_size=32 uses ~6GB VRAM on V100. For faster training with gradient accumulation: **Standard_NC12s_v3** (2x V100) using `accelerate` for multi-GPU. For cost-sensitive training: **Standard_NC4as_T4_v3** (1x T4 16GB, $0.53/hr) — T4 is ~50% slower than V100 but 6x cheaper. For large-scale experiments: **Standard_ND40rs_v2** (8x V100 32GB). Spot instances reduce cost by 70-90% but require checkpointing strategy to resume from interruptions.

---

**Q37. ⭐ How do you serve the model via Azure Container Apps?**

Steps: (1) Create a FastAPI application that loads the tokenizer and model at startup, applies ONNX Runtime optimization. (2) Build a Docker image: `FROM python:3.10-slim`, install torch+onnxruntime, copy model artifacts. (3) Push to Azure Container Registry (ACR): `az acr build`. (4) Deploy to Azure Container Apps: `az containerapp create --image acr.azurecr.io/bert-classifier:v1 --cpu 2 --memory 4Gi --min-replicas 1 --max-replicas 10 --ingress external`. (5) Configure autoscaling based on HTTP request rate with KEDA. (6) Add Application Insights SDK for telemetry. For GPU inference on Container Apps, use `--workload-profile-name GPU-NC4-A100`.

---

**Q38. ⭐⭐ How would you implement A/B testing for model versions in Azure?**

Use Azure API Management (APIM) with a custom policy to split traffic. Option 1: **Traffic splitting in APIM** — route X% of requests to Model A (current production), (100-X)% to Model B (challenger), log predictions and latency for both. Option 2: **Azure Container Apps traffic splitting** — `az containerapp revision set-mode --name bert-classifier --mode multiple` then set traffic weights: `--revision bert-classifier-v1 30 --revision bert-classifier-v2 70`. Option 3: **Feature flags in Azure App Configuration** — client-side assignment of model version per user_id. Metric comparison: collect prediction confidence, downstream task success metrics, and latency for both variants. Statistical significance test (t-test on latency, proportion test on accuracy) before promoting.

**Follow-up A:** How do you handle the fact that you don't have immediate ground truth in production?
Use proxy metrics: click-through rate for recommendations, downstream task completion for intent classification, escalation rate for support ticket classification. Alternatively, collect a sample of predictions for delayed human labeling (active learning style). For online evaluation, use bandit algorithms that optimize for reward rather than requiring ground truth.

**Follow-up B:** What is shadow deployment and when is it appropriate?
Shadow deployment runs the new model alongside production but discards its outputs — it only receives the same inputs. Used to: validate latency/throughput of new model without serving risk, compare output distributions between versions, warm up the new model's caches. Appropriate when the new model is architecturally different (different output schema) or when the change is high-risk (model for a critical financial decision).

---

## 6. Deployment — AWS

**Q39. ⭐ Describe the AWS SageMaker deployment pipeline for BERT text classification.**

```
AWS SageMaker Deployment Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Amazon S3]
  ├── s3://bucket/training-data/
  └── s3://bucket/model-artifacts/

         │
         ▼
  [SageMaker Training Job]
  ┌──────────────────────────────┐
  │  instance: ml.p3.2xlarge     │
  │  (1x V100 16GB, $3.06/hr)    │
  │  Script: train.py            │
  │  Hyperparams via SM env vars  │
  └──────────────┬───────────────┘
                 │
                 ▼
  [SageMaker Model Registry]
  ├── ModelPackage: bert-text-clf
  ├── Approval: PendingManualApproval
  └── Metrics logged to SM Experiments

                 │ approved
                 ▼
  [SageMaker Endpoint]            [CloudWatch]
  ┌──────────────────────────────┐      │
  │  instance: ml.g4dn.xlarge    │──────┤ metrics
  │  (T4 GPU, $0.736/hr)         │      │
  │  Container: HuggingFace DLC  │      ▼
  │  Auto-scaling: 1-10 instances│ [CloudWatch Alarms]
  └──────────┬───────────────────┘ Latency > 200ms alert
             │
             ▼
  [AWS Step Functions]
  └── Train → Evaluate → Register → Deploy pipeline

  [Optional: Batch Transform]
  └── For bulk inference on S3 files
```

---

**Q40. ⭐ What is SageMaker's HuggingFace DLC (Deep Learning Container) and how does it simplify deployment?**

AWS provides pre-built Docker containers with HuggingFace Transformers, PyTorch, and CUDA pre-installed — the HuggingFace DLCs. You specify the container via `image_uri = sagemaker.image_uris.retrieve("huggingface", region, version="4.26.0", py_version="py39", instance_type="ml.g4dn.xlarge", image_scope="inference")`. The DLC expects your model artifacts in a specific format: `model.tar.gz` containing `pytorch_model.bin`, `config.json`, `tokenizer_config.json`, and a custom `inference.py` with `model_fn()`, `input_fn()`, `predict_fn()`, `output_fn()` functions. This dramatically reduces containerization overhead versus building a custom Docker image.

---

**Q41. ⭐⭐ How do you configure SageMaker auto-scaling for a BERT inference endpoint?**

SageMaker endpoints support Application Auto Scaling via CloudWatch metrics. Register the endpoint with auto-scaling: `client.register_scalable_target(ServiceNamespace='sagemaker', ResourceId='endpoint/bert-clf/variant/AllTraffic', ScalableDimension='sagemaker:variant:DesiredInstanceCount', MinCapacity=1, MaxCapacity=10)`. Create a scaling policy: target tracking on `SageMakerVariantInvocationsPerInstance` (e.g., target=100 invocations/instance/minute). Add a scale-in cooldown (300s) to prevent thrashing, scale-out cooldown (60s) for quick response to traffic spikes. For GPU endpoints, also set a step scaling policy based on GPU utilization >70%. Cost consideration: with min=1 at ml.g4dn.xlarge, baseline cost is ~$527/month.

**Follow-up A:** What metric should you use for BERT endpoint autoscaling — requests/second or latency?
Both. Primary: `SageMakerVariantInvocationsPerInstance` (requests per instance). Secondary: P95 latency via CloudWatch — if latency spikes even at low request rates (e.g., due to longer inputs), scale out based on latency threshold. You can create a composite alarm: scale out if EITHER requests > 100/min/instance OR P95 latency > 150ms.

**Follow-up B:** How does SageMaker multi-model endpoint differ from a standard endpoint?
Multi-model endpoint (MME) hosts multiple models on one endpoint, loading/unloading models from S3 on demand with an LRU cache in memory. This is ideal for serving many customer-specific fine-tuned models (e.g., per-client text classifiers) without paying for N separate endpoints. Each inference request includes a `TargetModel` parameter specifying which model to load. The trade-off is cold start latency (1-5s) when a model isn't in the hot cache — implement a warmer that pings low-traffic models periodically.

---

**Q42. ⭐ How do you run batch inference on S3 data with SageMaker Batch Transform?**

Batch Transform is the right tool for classifying large datasets stored in S3 rather than real-time endpoints. Create a transform job: `transformer = model.transformer(instance_count=1, instance_type='ml.g4dn.xlarge', strategy='MultiRecord', assemble_with='Line', output_path='s3://bucket/output/')`. Input format: one JSON per line `{"inputs": "text to classify"}`. Output: one prediction per line. For DistilBERT on 1M texts at batch_size=128: ~2.5 hours on one ml.g4dn.xlarge. Cost: ~$1.85 total vs. $45/month for a continuously-running endpoint. Use batch transform for: overnight batch scoring, model evaluation on large test sets, creating feature embeddings for downstream ML.

---

## 7. Post-Production Issues

| # | Issue | Detection Method | Root Cause | Resolution |
|---|---|---|---|---|
| 1 | F1 decay on new news topics | Weekly eval on labeled sample | Model hasn't seen new topic vocabulary | Retrain with new topic examples |
| 2 | Tokenizer mismatch on deployment | `tokenizer(text)` returns unexpected tokens | Wrong tokenizer version deployed | Pin tokenizer version in requirements, test in CI |
| 3 | VRAM OOM during inference | CUDA OOM error in logs | Long articles (>512 tokens) with max padding | Enforce max_length=256 for inference, use dynamic padding |
| 4 | Prediction latency regression | CloudWatch p99 > 2x baseline | Model version upgrade added extra layer | ONNX re-export after model update |
| 5 | Confidence calibration drift | ECE increases from 0.05 to 0.15 | Distribution shift in input topics | Re-run temperature scaling on recent labeled data |
| 6 | OOD inputs (non-English text) | High `[UNK]` token rate | Users submitting non-English text | Add language detection pre-filter |
| 7 | Catastrophic forgetting in retraining | Old-category F1 drops after new-category training | New training loop doesn't include old data | Use rehearsal (include 10% old data in new training) |
| 8 | Silent failure: wrong model loaded | Predictions are plausible but wrong | Model artifact path misconfiguration | Add model metadata validation at startup |
| 9 | Slow cold starts on scaled-down endpoints | First request after 0-instances takes 30s+ | Container cold start + model load | Set min-replicas=1, use provisioned concurrency |
| 10 | Memory leak in tokenizer batching | RSS memory grows over time | tokenizer fast/slow mismatch creating retained tensors | Explicitly call `del` on large tensors, pin tokenizer type |
| 11 | Batch padding inconsistency | Accuracy lower in production than offline | Production uses `padding='max_length'`, training used dynamic | Standardize padding strategy across training/inference |
| 12 | Class distribution shift | Precision drops for specific class | One class becomes more/less prevalent in production | Monitor prediction distribution, adjust threshold |
| 13 | Logging PII in inference requests | GDPR/privacy violation | Logging raw input text | Hash/redact input text before logging |
| 14 | Stale model in cache | Predictions from old model version | CDN or application-layer cache not invalidated | Version model endpoint URLs, cache-bust on deployment |
| 15 | Quantized model accuracy regress | INT8 model F1 drops 3% on certain classes | Outlier activations collapse under INT8 | Use dynamic quantization only on FFN, keep attention in FP32 |
| 16 | Wandb metric mismatch | val_f1 in wandb ≠ actual evaluation | Logged intermediate F1, not end-of-epoch | Only log epoch-level metrics from evaluator.py |

---

## 8. General ML Interview Topics

**Q43. ⭐ Explain the bias-variance trade-off in the context of fine-tuning BERT.**

In BERT fine-tuning: **high bias** (underfitting) occurs when the model doesn't adapt enough to the target task — too few training epochs, frozen all layers, or LR too small. **High variance** (overfitting) occurs when the model memorizes training data — typically seen with small datasets (<1000 examples), too many epochs, or no regularization. BERT's large parameter count (66M for DistilBERT) means it has enormous capacity and can easily overfit. Regularization tools: dropout (0.1-0.3 in classifier head), weight decay (0.01), early stopping, layer freezing, data augmentation (EDA). The warmup + small LR is itself a form of regularization — it prevents the model from taking large steps that overfit to the first few batches.

---

**Q44. ⭐ How does cross-validation work for NLP fine-tuning, and what are its limitations?**

Standard k-fold CV splits the dataset into k folds, trains on k-1, evaluates on 1, rotates. For NLP fine-tuning, this means training k separate BERT models — expensive but provides more reliable performance estimates. Limitations: (1) k-fold for transformers is computationally expensive (5x training cost for 5-fold); (2) if training data has temporal ordering (e.g., news articles by date), random k-fold causes leakage — use time-based splits; (3) if data has duplicate or near-duplicate texts, splitting randomly may put near-duplicates in both train and val (data leakage). Practical alternative: single train/val/test split with multiple random seeds (3-5 runs), report mean ± std. For small datasets (<5000 examples), 5-fold CV is warranted.

---

**Q45. ⭐⭐ How do you determine if a benchmark improvement is statistically significant?**

A 0.5% F1 improvement on SST-2 could be noise. Approaches: (1) **Bootstrap confidence intervals**: resample predictions 1000 times, compute F1 for each resample, compare CIs of two models. If CIs don't overlap, improvement is significant. (2) **McNemar's test**: for paired binary predictions (is example correctly classified?), tests if the two models differ significantly on matched examples. Appropriate when test set is fixed. (3) **Permutation test**: randomly shuffle model assignments for predictions and compute null distribution of F1 difference. (4) **Multiple runs**: train both models with k different seeds, test if mean F1 of new model > old model using t-test. For NLP benchmarks, generally report significance at p<0.05. Rule of thumb: on 10,000-example test sets, differences > 0.3% F1 are typically significant; on 1,000-example sets, need >1% difference.

**Follow-up A:** Why is reporting a single-run result on BERT fine-tuning misleading?
BERT fine-tuning results have high variance due to random weight initialization of the classification head, random training data ordering, and the sensitive interaction between warmup schedule and early training dynamics. Devlin et al. themselves noted that BERT fine-tuning can fail for certain random seeds on small datasets. Dodge et al. (2020) found that reporting the best of N runs (without disclosing N) inflates benchmark performance. Best practice: report mean ± std over at least 5 runs.

---

**Q46. ⭐ What regularization techniques are available for BERT fine-tuning?**

| Technique | Mechanism | Where Applied |
|---|---|---|
| Dropout | Randomly zero out activations during training | After [CLS] embedding, in BERT's attention |
| Weight Decay (L2) | Penalizes large weights in optimizer | All non-bias, non-LN params via AdamW |
| Early Stopping | Stop when val metric plateaus | Monitor val F1, patience=3 epochs |
| Layer Freezing | Freeze lower layers | Embedding + layers 0-2 |
| Data Augmentation | Synonym replacement, back-translation | Training data only |
| Mixup (text) | Interpolate sentence embeddings | Post-encoder mixing |
| Label Smoothing | Soften one-hot targets to [ε/K, ..., 1-ε] | `CrossEntropyLoss(label_smoothing=0.1)` |
| LLRD | Lower LR for lower layers | Optimizer parameter groups |

---

## 9. Behavioral / Scenario Questions

**Q47. ⭐ Describe a time you had to debug a silent failure in an ML system.**

Structure answer with STAR method. Scenario: deployed BERT classifier shows excellent offline metrics (F1=0.92) but business reports classification quality is poor in production. Investigation: (1) Check tokenizer — confirm same tokenizer used in training vs. serving. (2) Check preprocessing — if production strips HTML but training data had HTML entities, mismatch exists. (3) Check label mapping — if training labels were 0-indexed but production expects 1-indexed, all predictions are shifted. (4) Check data drift — compare production input length distribution vs. training distribution using KS test. Resolution: found that production system applied a different text normalization (removed punctuation), changing the token distribution. Fix: align preprocessing pipelines.

---

**Q48. ⭐⭐ Your model achieves 92% accuracy in development but only 83% in production. What are your next steps?**

Systematic diagnosis: (1) **Data leakage check**: verify dev evaluation set has no overlap with training data. (2) **Distribution shift analysis**: compare token distributions, text length distributions, vocabulary OOV rates between dev and production using statistical tests (KS test, Jensen-Shannon divergence). (3) **Preprocessing pipeline audit**: log raw and preprocessed inputs in production, compare with training pipeline. (4) **Temporal shift**: if production data is newer than training data, newer vocabulary/topics may be OOV. (5) **Label noise**: production labels may have different annotation guidelines than training. (6) **Stratification**: ensure dev set is representative of production distribution (all topics, all source websites). Resolution prioritized by most likely cause.

**Follow-up A:** How do you quantify "distribution shift" objectively?
Embed production samples and training samples using a frozen sentence encoder. Compute Maximum Mean Discrepancy (MMD) between the two sets. Alternatively, train a binary classifier to distinguish production vs. training examples — if the classifier achieves >60% accuracy, significant shift exists. Monitor MMD over time as a drift metric.

---

**Q49. ⭐ How would you explain the choice of DistilBERT over BERT-base to a non-technical stakeholder?**

Frame it as a business trade-off: "We chose DistilBERT — a streamlined version of BERT that's 40% smaller and 60% faster, while retaining 97% of the accuracy. In concrete terms: our system can classify 600 articles per second instead of 250, reducing our cloud compute bill by approximately 40%. For our use case of news categorization where the accuracy difference is less than 1.5%, this trade-off strongly favors DistilBERT. If we were classifying medical diagnoses where each 0.1% accuracy improvement saves lives, we would use the larger model."

---

**Q50. ⭐⭐ You need to add 2 new classes to a 4-class classifier without retraining from scratch. How do you approach this?**

Options: (1) **Fine-tuning head only**: freeze encoder, add 2 neurons to the output layer (now 6 classes), fine-tune only the head on data for all 6 classes including original class data. Fast but may not fully adapt representations. (2) **Progressive net**: keep original model intact, train a new model for the 2 new classes, ensemble predictions. No interference with old model. (3) **Full retraining with all 6 classes**: most accurate but requires 4-class data + 2-class data. Typically best if combined dataset is available. (4) **Zero-shot with prompt**: reframe as NLI ("Is this text about [class]?") using a fine-tuned NLI model — works without any new labeled data. The correct choice depends on: available data for new classes, acceptable accuracy regression on old classes, and time/compute constraints.

---

## 10. Quick-Fire Questions

1. ⭐ What does BERT stand for? — Bidirectional Encoder Representations from Transformers
2. ⭐ How many transformer layers does DistilBERT have? — 6 (BERT-base has 12)
3. ⭐ What is the hidden dimension of DistilBERT? — 768
4. ⭐ What is the maximum sequence length for DistilBERT? — 512 tokens
5. ⭐ What tokenization algorithm does BERT use? — WordPiece
6. ⭐ What are the two BERT pretraining objectives? — MLM (Masked Language Model) and NSP (Next Sentence Prediction)
7. ⭐ What optimizer is used for BERT fine-tuning? — AdamW
8. ⭐ What is the recommended learning rate range for BERT fine-tuning? — 2e-5 to 5e-5
9. ⭐ What does gradient clipping prevent? — Exploding gradients / parameter update instability
10. ⭐ What is the [CLS] token used for? — Aggregate sequence representation for classification
11. ⭐ What is the [SEP] token? — Separator between sentence A and B / sequence terminator
12. ⭐ What does attention_mask do? — Tells the model to ignore padding tokens
13. ⭐ What is warmup in LR scheduling? — Gradual LR increase from 0 to lr_max over first N steps
14. ⭐ What metric is used for checkpoint selection in this project? — Validation macro F1
15. ⭐ Why macro F1 over accuracy for AG News? — Equal weighting of all 4 classes regardless of support
16. ⭐ What is weight decay in AdamW? — L2 regularization, keeps weights from growing large
17. ⭐ What is dropout and where is it applied in BERTClassifier? — Random neuron zeroing; applied after [CLS] embedding
18. ⭐ What is the vocabulary size of DistilBERT's tokenizer? — 30,522 tokens
19. ⭐ What is knowledge distillation? — Training a small student model to mimic a large teacher model
20. ⭐ How much smaller is DistilBERT vs BERT-base? — 40% fewer parameters (66M vs 110M)
21. ⭐ What is catastrophic forgetting? — Loss of pretrained knowledge when fine-tuning on a new task
22. ⭐ What is gradient accumulation? — Accumulating gradients over multiple mini-batches before optimizer step
23. ⭐ What does `freeze_encoder_layers()` do? — Sets `requires_grad=False` for specified encoder layers
24. ⭐ What is the confusion matrix diagonal? — Correctly classified examples per class
25. ⭐ What is mixed precision training? — Using FP16 for most computations, FP32 for gradients
26. ⭐⭐ What is PEFT? — Parameter-Efficient Fine-Tuning (LoRA, Adapters, Prefix Tuning)
27. ⭐⭐ What is LoRA? — Low-Rank Adaptation: add low-rank matrices to attention projections
28. ⭐⭐ What is the temperature parameter in distillation? — Controls softness of teacher's probability distribution
29. ⭐⭐ What is LLRD? — Layer-wise Learning Rate Decay: lower LR for lower transformer layers
30. ⭐⭐ What is focal loss? — Cross-entropy that down-weights easy/confident examples
31. ⭐⭐ What is calibration? — Alignment between predicted probabilities and actual frequencies
32. ⭐⭐ What is temperature scaling? — Post-hoc calibration via single scalar on logits
33. ⭐⭐ What is EWC? — Elastic Weight Consolidation: regularization for continual learning
34. ⭐⭐ What is data drift? — Change in input distribution between training and production
35. ⭐⭐ What is concept drift? — Change in relationship between inputs and labels over time
36. ⭐ What does wandb track in this project? — LR, loss, grad norm, val F1, confusion matrix, hyperparams
37. ⭐ What is the `evaluate` library used for? — Computing F1, accuracy, confusion matrix metrics
38. ⭐ What is the `accelerate` library? — HuggingFace distributed training abstraction
39. ⭐ What format does the Predictor return? — (predicted_label, confidence_score) tuple
40. ⭐ What datasets are used in this project? — AG News (4-class), IMDB (binary), GLUE/SST-2 (binary)
41. ⭐⭐ What is multi-label vs multi-class classification? — Multi-label: multiple labels per example (sigmoid); multi-class: one label (softmax)
42. ⭐⭐ What loss for multi-label classification? — `BCEWithLogitsLoss` (binary cross-entropy per label)
43. ⭐ What is the WordPiece ## prefix? — Indicates a continuation subword (not start of new word)
44. ⭐ What is dynamic padding? — Padding each batch only to max length in that batch (not global max)
45. ⭐⭐ What Azure service stores model artifacts? — Azure Blob Storage
46. ⭐⭐ What Azure service manages model versions? — Azure ML Model Registry
47. ⭐⭐ What Azure service hosts the inference API? — Azure Container Apps
48. ⭐⭐ What AWS service runs training jobs? — SageMaker Training Jobs
49. ⭐⭐ What AWS service hosts real-time inference? — SageMaker Endpoints
50. ⭐ What is a SageMaker HuggingFace DLC? — Pre-built Docker container with HuggingFace + PyTorch + CUDA

---

*End of Text Classification with BERT Interview Guide — 200+ questions covered.*
