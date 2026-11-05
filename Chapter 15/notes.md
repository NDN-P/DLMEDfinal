# Chapter 15 — Understanding Transformers for Computer Vision

> **Not on the final exam** (confirmed by the teacher). Kept for reference only.

A Transformer uses **attention** to (1) learn relationships between tokens and (2) exchange/aggregate information between tokens, which gives context-aware features. ViT applies this to images: split into patches → patch embedding → add positional information → stack of Encoder Blocks (MHSA + MLP + skip connections + LayerNorm) → one image vector → classification head.

> Source note: the title slide of the PDF says "Chapter 14", but this is the course's Chapter 15 topic. Formulas, shapes and diagrams were checked against the rendered slide images; page numbers are given as (p.N). Lines marked *(standard fact)* are general DL knowledge that is not stated in the slides or lecture.

---

## Key concepts

### 1. What is a Transformer / ViT (slides p.3–4)
- **Transformer** = a neural-network architecture that uses **Attention** to learn relationships and exchange information between **Tokens** ("รวมข้อมูลระหว่าง Tokens"), producing features that understand **context**. It works for text, images, and other data types.
  - Two jobs inside it: *between* tokens (attention) and *within* one token (the MLP "แปลงคุณลักษณะภายใน Token เดียวกัน").
- **Token** = a vector that represents *part* of the whole data.
  - NLP: a word, character, sentence, or sub-word (e.g. "ทำไม" split into "ทำ" + "ไม").
  - Image: a sub-region (patch) of the image, usually from a grid.
  - Video: a frame or part of a frame. Audio: a time segment (e.g. minute 1–2).
  - Teacher's point: if you can define tokens for your data, you can use a Transformer on it.
- **Vision Transformer (ViT)** = a Transformer for images: split the image into **Patches** → convert to **Tokens** → add **position info** → pass through the **Transformer Encoder** → use the output for the task.
  - The original Transformer has an **Encoder + Decoder**. Later models may use **only the Encoder (e.g. ViT)** or only the Decoder.
  - Vision Transformers usually keep only the encoder and do **not decode back**. The encoded features go straight to prediction (classification, detection, regression…).
- The lecture called ViT the "officially" first Transformer for images (the teacher was careful not to claim it was truly the first).

### 2. Image → Patches (slide p.8)
- **Image to Patches** = extracting sub-regions of the image. It is related to splitting an image into blocks and to the **sliding window** idea.
- **Stride** = the distance the window moves ("ระยะเลื่อนหน้าต่าง").
- **ViT uses non-overlapping patches**: **stride = patch size**, so patches share no pixels. With the same patch size this gives **fewer tokens** (p.8).
- **Overlapping patches** (later research, e.g. *Overlapping Patch Embedding in PVT v2*): **stride < patch size**, so neighbouring patches share some pixels. A smaller stride gives **more tokens** (p.8).
  - Pro: keeps continuity/transitions across patch borders. The teacher's example was a checkerboard: a non-overlapping cut can land exactly on a black/white border and lose continuity.
  - Con: more patches, so more memory and compute.
  - Which one is better? "Depends on dataset/problem". Usually just follow what the chosen model uses.
- **Patch vs Token**: a patch is the **2-D** view; a token is the **1-D vector** view. Papers often mix the two words.

### 3. Image Embedding vs Patch Embedding (slides p.11–12)
- **Image Embedding** = representing an image by a **feature vector**. It is **NOT** a way to shrink the image file. It comes from **Representation Learning**, which existed **before** Transformers.
  - The whole image is treated as one patch → Flatten → Linear → **one vector** = a **global feature**.
- **Patch Embedding** = converting each patch's pixels into a feature vector so the Transformer can process the image as a **sequence of tokens**. The idea comes from representing words as vectors in NLP.
  - Steps: split into patches (overlap or not) → **Flatten** (2-D → 1-D, still raw data/preprocessing) → **Linear Projection** (the actual embedding/encoding) → tokens.
  - **Number of output vectors = number of patches** (this is the *count*, not the vector length). You choose the output vector size in the Linear layer.
  - From slide p.12:
    - Flatten only rearranges the pixels into a vector and learns nothing ("ยังไม่เรียนรู้คุณลักษณะ").
    - Linear = **learnable parameters**, and the **same Linear (same weights) is used for every patch**. One patch gives one vector.
    - At this stage each patch is converted **independently**. It has **no context from other patches yet** (no attention yet).
    - Standard ViT **needs** this step. The method can differ: a **Conv2D/CNN** can also create the tokens.
    - A general CNN does **not** need a ViT-style patch-embedding step.
- Image vs Patch embedding:

| | Image Embedding | Patch Embedding |
|---|---|---|
| Input to Linear | whole image flattened | each patch flattened |
| Output | 1 vector (global feature) | N vectors (1 per patch) |
| Information kept | less, more information loss | more, less loss |
| Cost | low | higher memory and processing time |

### 4. Positional Encoding (slides p.14–22)
- **Why**: the feature vector tells **"what"** is in a patch ("มีอะไร") but not **"where"** it is ("อยู่ตรงไหน"). Positional encoding adds the **Where**. After adding, the token holds **What + Where**, which is richer information.
- Order matters (teacher: "แมวไล่หนู" vs "หนูไล่แมว" mean different things; sitting vs standing frames).
- **Why does a Transformer need it when a CNN doesn't?** A CNN filter only looks at a pixel and its **neighbours**, so locality is built in. Self-attention compares each token with **every** token, so on its own it has no idea which tokens are its neighbours or where it sits. Positional information is the guide.
- Positional encoding can be classified by **how it is created, where it is inserted, and data type**. A broader name is "วิธีใส่ข้อมูลตำแหน่ง" (ways of inserting position info), because some methods **adjust the attention scores** or **rotate Q/K** instead of adding to the embedding.
- **The slide (p.15) lists 6 types**, and the suggested use for each ("น่าทดลอง"):

| # | Type | Idea | Suggested use |
|---|---|---|---|
| 1 | Fixed Sinusoidal | vector from sine/cosine | baseline with no extra position parameters |
| 2 | Learned Absolute | learn a vector per position | images with a fixed grid |
| 3 | Relative Position Bias | adjust attention scores by distance and direction | image structure and Window Attention |
| 4 | RoPE / 2D RoPE | **rotate Q and K** by position | text/images whose length or size changes |
| 5 | Conditional Encoding | create position info from nearby context | images of different sizes, local structure |
| 6 | ALiBi | add a bias that **lowers the score with distance** | notable for text longer than the training length |

- The lecture explained **4 of them in detail** (Fixed, Learned, Relative, Conditional). That is why the teacher said "อันที่ 5 คือ Conditional": he was using the slide's numbering.
- Fixed Sinusoidal picture (p.17):
  - Each position (row) has its own code. Read the code **along the row**, not down the column.
  - Each pair of dimensions uses a sine and a cosine at a **different frequency**. The left dimensions change quickly with position; the right dimensions change slowly. Values run from −1 to +1.
  - **Feature + Position = Input**: added channel by channel, so the vector length is unchanged.
- The 4 methods taught in the lecture:

| Method | How | Learned? | Inserted where | Same position → same code? |
|---|---|---|---|---|
| **Fixed Sinusoidal** | Sine & Cosine formula (low frequency changes slowly, high frequency changes quickly) | No (fixed formula, independent of data) | **added element-wise** to patch embedding | Yes, always |
| **Learned Absolute** | one learnable vector per position, starts random and is updated in training | Yes | **added** to patch embedding before the Transformer | Yes (after training), so it fits the training dataset |
| **Relative Position Bias** | learnable **bias** by relative **distance + direction** between each pair of patches | Yes | **added to attention scores before Softmax** | Same distance/direction pair → same bias (within that head's table) |
| **Conditional (CPE)** | code generated from the patch **and its neighbouring patches** through a learned network (lecture: **depth-wise convolution**), then added to the features | Yes | added to features | **No**. It changes with the input image, even at the same position (the only one of the four) |

- Learned Absolute learns "**where is this patch in the image**". Relative Position Bias learns "**how far apart and in which direction are two patches**" (slide p.21).
- Data prep (patching → embedding → position) is done **only once** at the start ("หลายสถาปัตยกรรมจะดำเนินการเพียงครั้งเดียว", slide p.24). After that the tokens go through the encoder blocks.

### 5. Attention, Self-attention, Cross-attention (slide p.28)
- **Attention** = deciding which region of the data should matter, be highlighted, or be focused on.
- **Self-attention** is one form of attention: the relationships **within** the same data (which region of the image relates to which other region, e.g. cat ear ↔ eye ↔ leg).
- **Cross-attention** = relationships **between two data sources/types**, e.g. text and image in **Visual Language Models** (the reason ChatGPT can read an image directly, without OCR). It was mentioned only, not taught in detail.

### 6. Q, K, V (slides p.29–32)
- **Q = Query**: what I am looking for. The token takes itself as the reference and asks which other tokens relate to it.
- **K = Key**: what represents a token for matching. Queries are compared against it.
- **V = Value**: the actual information/content of that token.
- **Q is matched with K only. V is not involved in the matching** ("Q จะไป Finding กับ Key แต่จะไม่ยุ่งกับ V").
- Analogies from the lecture:
  - YouTube search: the typed keyword = **Query**, the video title = **Key**, the video content = **Value**.
  - Database: Student ID = **Key** (represents the person), name/birthday/ID card number/money = **Value**.
- **How Q, K, V are produced**: each token goes through **3 separate Linear layers (W_Q, W_K, W_V)**. These are **learnable parameters that must be trained**.
  - They are named by **how they are used**, not by what they "are". They are abstract features, and training makes them play those roles.
- **One token → one set of Q, K, V**, computed from **all channels of that token** (10 tokens → 10 Q, 10 K, 10 V).
- Slide p.32 steps: Image Patches → Flatten → Patch Embedding → Positional Encoding.
  - **In standard ViT the tokens go through LayerNorm before K, Q, V are made**, giving X.
  - X is then split into **3 paths**: Linear K, Linear Q, Linear V.
  - **Each Linear learns its own parameters**, and each path uses **the same parameters for every token**.
  - "ทุก patch สร้างครบทั้ง K, Q และ V". Q is compared with K to get the weights, and the V's are summed by those weights.
- Slide p.29 example (YouTube search "Vision Transformer"):
  - K = title/features of each item. V = actual content (video, sound, ideas…).
  - Scores s₁, s₂, s₃ → Softmax → a₁, a₂, a₃ → **Output = a₁V₁ + a₂V₂ + a₃V₃** (weighted sum).

### 7. Scaled dot-product self-attention (slides p.35–36)
- Every Q is compared with every K → similarity scores = relationships between patches.
- Dot product ≈ "like" cosine similarity, but **it is actually Scaled Dot-Product**. Both use the dot product. Only the divisor differs (slide p.36):

| | Cosine Similarity | Scaled Dot-Product |
|---|---|---|
| Formula | q·k / (‖q‖‖k‖) | q·k / √d_k |
| Divide by | product of the two vector lengths | square root of the **Key dimension** |
| Score depends on | **direction** only | **direction and length** |
| Range | −1 to 1 (non-zero vectors) | **no fixed bound** |
| Purpose of dividing | remove the effect of vector length | shrink scores when the dimension is large so **Softmax doesn't become too peaked** ("ไม่กระจุกตัวมากเกินไป") |

- Why does similarity mean relationship? In this context, two things are related when they are **similar**. A dot product measures similarity, so we read it as relationship.
- Slide p.35 "standard attention formula":
  - s_i = q·k_i / √d_k and a_i = exp(s_i) / Σ_j exp(s_j).
  - **d_k = number of Key dimensions per head**.
  - s_i is **not limited to −1…1**. After Softmax, Σ_j a_j = 1.
  - One query is compared with **every** K.
- **Softmax** (applied **per query**, i.e. per row, "Softmax แยกแต่ละแถว" on p.5) turns the scores into **attention weights A** in the range 0–1 that **sum to 1**. That is why Softmax and not Sigmoid/Tanh: those don't force the sum to 1.
- **Z = A·V** = weighted sum of the Values = **aggregating information between tokens/patches**. Highly related patches contribute more information.
- Mapping back to the definition (teacher at [01:01:05]):
  - Q·K (+Softmax) = **learning relationships between tokens** (point 1).
  - A·V = **exchanging/aggregating information between tokens** (point 2).

### 8. Multi-Head Self-Attention (MHSA)
- 1 self-attention = **1 head**. Similar to how a conv layer has many filters, an encoder block can use **many heads**, each able to attend to different things.
- Each head outputs Z_h with **the same number of tokens as the input** (token shape in = token shape out).
- **Each head has its own W_Q,h, W_K,h, W_V,h** (slide p.5: Q₁ = X̂W_Q,1, Q₂ = X̂W_Q,2, …). The heads split the model dimension: D = 8 with 2 heads gives d_k = 4 per head.
- Combining: **concatenate** the heads' outputs **per token** ("ต่อคุณลักษณะของ Token เดียวกัน") → **Linear W_O** ("W0", 8×8 in the slide example), which **merges the heads' information** ("กาดอกจัน" point).
- Slide p.51: "Multi-head: learns several ways of combining information". The heads in the picture are illustrative; no head is assigned a specific job.
- Teacher's CNN analogy: 1 head ≈ the result of 1 **kernel** (one channel). Concat + W_O ≈ combining kernels into a **filter**.

### 9. Encoder Block (slides p.26–27, 33, 37–45)
- Idea (p.40): **Attention exchanges information between Tokens**, then an **MLP adjusts features inside each Token**, and the result goes to the next Encoder Block.
- Components: **LayerNorm**, **Multi-Head Self-Attention**, **Dropout** (regularization), **Skip/Residual connection** ("อาศัยหลักการ Skip Connection", where arrows meet at "+"), **MLP**.
- Skip connection: old data + new (attended) data are added, so information is added gradually.
- **The slide's encoder block is Pre-LayerNorm** ("Encoder Block (Pre-LayerNorm)", p.5):
  - **Residual 1**: U = X₀ + Dropout(MHA(LN₁(X₀))). The skip goes **around LN1 + MHA + Dropout**.
  - **Residual 2**: Y = U + Dropout(MLP(LN₂(U))). The skip goes **around LN2 + MLP + Dropout**.
  - **The shape is the same everywhere in the block** (4×8 in the example), and Y goes on to the next encoder block.
  - Dropout (including the "Embedding Dropout" after X = E + P) is active **only during training**.
- **LayerNorm**: the lecture explained it as keeping values in range after the addition (adding can push the range too high). Note that in the slide diagram the LN sits **before** MHA and before the MLP (pre-norm).
- **MLP** = Linear → **GELU** → Dropout → Linear. Input and output have the **same size**, and it **expands in the middle** (slide: 8 → 32 → 8). **The same MLP is used for every token**, i.e. it processes each token separately ("Feed Forward Network"). Purpose: **blend information across all channels of each token**, because the attention operator works channel-wise per head, unlike a conv filter with many kernels.
- **Stacking blocks** (p.45): many blocks let the model refine token relationships in several stages, each using the previous block's features, to understand context and complex patterns. **More blocks ≠ guaranteed better**, and they add compute cost. Most blocks are identical; only the first stage differs because it converts the data.
- After the encoder you **cannot plot a feature map** like a CNN, but you can plot the **attention map** or Grad-CAM.

### 10. Using the output: Classification Head (slide p.46)
- After the last block, each token has a context-aware feature vector. Make **one image-representative vector** → **Classification Head** (usually **1 Linear layer**) → prediction.
- Two styles:
  1. **Global Average Pooling** over tokens (average each channel across all tokens T1, T2, T3…) → Linear. This style has **no CLS token**.
  2. **CLS token** (from NLP): a dummy/empty token is added at the start. After the encoder it holds the summary, and **only it** is used for classification. **ViT uses CLS**.
- The teacher couldn't recall exactly which later models (DeiT/MaxViT/Swin…) use GAP. Don't memorise that list.
- Slide p.46 example (4 patches, D = 8, 3 classes: cat/dog/bird):
  - **CLS**: add CLS before the first encoder → last encoder output **5×8** → pick CLS + **LayerNorm** → image vector **1×8** → Linear **8→3**. "CLS learns to gather the information for classifying the image."
  - **GAP**: no CLS → last output **4×8** → average across tokens, feature by feature → **1×8** → Linear **8→3**.
  - Where the LayerNorm goes depends on the model design.

### 11. CNN vs RNN vs Self-attention (slides p.47–48, from d2l.ai)
- **CNN**: **hierarchical**, **local** processing, supports **parallel** processing. It grows its receptive field by reducing spatial size (pooling). One filter is used on every pixel (weight sharing), which generalises and reduces parameters.
- **RNN**: **sequential** processing, **no parallelism**. Early steps are mostly local and later steps more global, because they have accumulated earlier information (only past, not future).
- **Self-attention**: processes nodes that are **directly related**, supports **parallel** processing, and is **global** (like fully connected), but its **weights change with the data**.

| | CNN | RNN | Self-Attention |
|---|---|---|---|
| Complexity per layer | O(k·n·d²) | O(n·d²) | **O(n²·d)** |
| Sequential operations | O(1) | O(n) | O(1) |
| Maximum path length | O(log_k(n)) | O(n) | **O(1)** |

n = input sequence length, d = number of input/output channels, k = kernel size.

### 12. CNN vs Transformer (slides p.49–51)
- A CNN uses **learned kernels** to combine information from a **fixed nearby region**. A Transformer uses self-attention to **compute the combination weights between patches from the image content**, so the relationships **change with the input**. In **both**, the **parameters stay fixed after training**.

| Compared item | CNN | Transformer |
|---|---|---|
| Parameters after training | kernel values fixed | **W_Q, W_K, W_V** and other parameters fixed |
| What changes per input image | feature map and prediction | **Q, K, V, attention map** and prediction |
| How info is combined across positions | learned kernel over a fixed region | weights between tokens computed from the tokens' data |
| Area connected in one layer | depends on **kernel size & dilation** | **global self-attention connects every patch** |
| Locality / position | built in (neighbourhood) | needs **positional encoding** |
| Processing style | local, hierarchical | global, content-dependent |

- Slide p.50 adds **Global vs Window attention**:
  - **Global**: the chosen token can connect with **every** patch in the image.
  - **Window**: it connects only with patches **inside a set window** (e.g. Swin-style models).
- **Slide p.51: 4 advantages of attention over CNN**, with the teacher's surgical example:
  1. **Weights adapt to the content**. A cat facing front vs side: the image changes, so the attention weights may change (CNN: same kernel weights; attention: weights between patches computed from the data).
  2. **Connects distant regions**. In a suture-practice image, the tip of the tool ↔ the simulated wound line. A CNN's small kernel needs many layers to see wider; global attention links far patches **in one layer**.
  3. **Uses context to tell apart similar things**. A small crop of suture thread, a tool edge, or a skin mark can look like the same line (CNN: a small region may lack information; attention: uses other regions to interpret it).
  4. **Combines many kinds of relationships**. Head A, Head B… (CNN: many filters, but positions are linked by the kernel; multi-head: learns several ways of combining).
  - Footer: these are **conceptual advantages**. "**ผลจริงขึ้นอยู่กับข้อมูล สถาปัตยกรรม และการฝึก**" (real results depend on the data, the architecture and the training).
- *(Standard fact, only implied in the lecture):* CNNs have strong **inductive biases** (locality, weight sharing/translation equivariance), while ViTs have weaker ones, so plain ViTs generally need **more data / pre-training** to match CNNs. The lecture itself only says the choice "depends on your dataset and your application".

---

## Formulas & numbers to remember

**Q/K/V projection** (X = matrix of tokens, one row per token; the W's are trainable Linear layers):

    Q = X·W_Q      K = X·W_K      V = X·W_V

**Scaled dot-product attention** (lecture's S → A → Z):

    S = Q·Kᵀ / √d_k            (similarity / relationship scores, N×N)
    A = softmax(S)  row-wise   (attention weights; each query row sums to 1)
    Z = A·V                    (aggregated info)
    ⇒ Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V

- Slide form (p.35): s_i = q·k_i/√d_k, a_i = exp(s_i)/Σ_j exp(s_j), output = Σ a_i V_i.
- d_k = number of Key dimensions **per head**. Lecture and slide p.5 example: **d_k = 4 → √d_k = 2**.
- Cosine similarity for comparison: q·k / (‖q‖‖k‖), range [−1, 1].
- With **Relative Position Bias**: softmax(Q·Kᵀ/√d_k **+ B**)·V (bias added **before** Softmax).

**Multi-head:**

    MHSA(X) = Concat(Z_1, …, Z_h) · W_O        (concat per token, W_O merges heads)

Head size: d_k = D / h (slide p.5: D = 8, h = 2 → d_k = 4; *(standard fact)* ViT-Base: D = 768, h = 12, d_k = 64).

**Encoder block, Pre-LayerNorm** (slide p.5 §3.3, citing Dosovitskiy et al. 2021 ViT §3.1 and Vaswani et al. 2017):

    U = X₀ + Dropout( MHA( LN₁(X₀) ) )
    Y = U  + Dropout( MLP( LN₂(U) ) )          shape unchanged (N × D)
    MLP = Linear(D→4D) → GELU → Dropout → Linear(4D→D)   (slide: 8→32→8)

**Fixed sinusoidal PE** *(standard formula; the slide only shows the heatmap and says "sine/cosine at many frequencies, added channel by channel")*:
PE(pos, 2i) = sin(pos / 10000^(2i/D)), PE(pos, 2i+1) = cos(pos / 10000^(2i/D)).

**Patch / token count** (non-overlapping, patch size P):

    N = (H/P) × (W/P)            sequence length = N (+1 if a CLS token is used)
    flattened patch length = P × P × C
    patch-embedding output = N × D ; with CLS = (N+1) × D ; attention matrix = (N+1)×(N+1) per head

*(Standard fact)* Overlapping with stride S < P: N = (⌊(H−P)/S⌋+1) × (⌊(W−P)/S⌋+1), so more patches.

**Worked example from the slides (p.5–7, 46). Learn every shape: this is the master diagram of the chapter.**

| Step | Shape |
|---|---|
| Image | 32×32×3 |
| Split into patches 16×16×3 | (32/16)² = **4 patches** |
| Flatten each patch (16·16·3 = 768) | X: **4 × 768** |
| Linear 768 → 8 (same weights for every patch) | E: **4 × 8** |
| + Positional embedding P (4 × 8) | X = E + P: **4 × 8** → Embedding Dropout → X₀: **4 × 8** |
| LayerNorm₁ | X̂: 4 × 8 |
| Per head (h = 1, 2): W_Q,h, W_K,h, W_V,h | **8 × 4** each → Q_h, K_h, V_h: **4 × 4** |
| A_h = Softmax(Q_h K_hᵀ / √4) (row-wise) | **4 × 4** (token × token) |
| Z_h = A_h V_h | **4 × 4** |
| Concat [Z₁; Z₂] (per token) | **4 × 8** |
| Linear W_O | **8 × 8** → MHA output 4 × 8 |
| U = X₀ + Dropout(MHA) | 4 × 8 |
| MLP: Linear 8→32, GELU, Dropout, Linear 32→8 | 4×8 → 4×32 → 4×8 |
| Y = U + Dropout(MLP) | **4 × 8** (same as input) → next encoder block |
| Head (p.46), with CLS: 5 tokens | 5 × 8 → CLS + LN → 1 × 8 → Linear 8→3 |
| Head (p.46), with GAP: 4 tokens | 4 × 8 → average → 1 × 8 → Linear 8→3 |

(The p.5 example has no CLS token: "ตัวอย่างนี้ไม่รวม CLS token".)

**Worked example at real size (standard ViT-B/16, not from the slides):**
- Image 224×224×3, P = 16 → 224/16 = 14 → **14 × 14 = 196 patches**.
- Each patch flattened: 16×16×3 = **768 values** → Linear → D (ViT-B: D = 768) → tokens **196 × 768**.
- Add CLS → **197 tokens** → **197 × 768**. Add positional embedding (also 197 × 768, **added**, so the shape is unchanged).
- Each head's attention matrix: **197 × 197**. The encoder output has the same shape, **197 × 768**. CLS output (1 × 768) → Linear → classes.
- Self-attention cost grows with **n²**: a smaller patch gives more tokens and much more compute/memory.

**Numbers from the slides/lecture:**
- Cat example: image split into a **3×3 grid = 9 patches** → 9 tokens (+1 if CLS).
- 10 tokens → 10 Q, 10 K, 10 V; each Q compared with all 10 K → a 10×10 score matrix.
- Token feature 1×3 → "3 channels". Q/K/V are computed from all 3 channels of that one token.
- Classification head: usually **1 Linear layer**.
- Positional encoding: **6 types on slide p.15** (Fixed Sinusoidal, Learned Absolute, Relative Position Bias, RoPE/2D RoPE, Conditional, ALiBi). **4 were explained** in the lecture (Fixed, Learned, Relative, Conditional).

---

## What the teacher emphasised

- **[00:02:23]** "คีย์ เส้นใต้ครับ … เป็น keyword" → Transformer = (1) **learn relationships** between tokens ("ขีด1"), (2) **exchange information** between tokens ("ขีด2"). [00:02:36] "อันนี้คือ concept สำคัญเลยครับ". This is what separates it from CNN.
- **[00:05:52]–[00:06:06]** If you can define/convert your data into tokens, you can use a Transformer. That is the "root concept".
- **[00:07:02]–[00:07:23]** CNN = **local processing** (only as far as the filter size). Attention = itself vs **every** token = **global**.
- **[00:09:12]** Patch = 2-D view, Token = 1-D vector. Flatten + Linear turns patch → token.
- **[00:10:03]** After positional embedding the data represents both **What and Where**. "นี่คือจุดต่างจุดหนึ่ง".
- **[00:13:40]–[00:14:17]** Flatten is only preprocessing (2-D→1-D). The **Linear** layer does the actual embedding.
- **[00:15:32]–[00:16:06]** "อยากให้เด็กขีดเส้นใต้ไว้" → the goal of embedding is a **representative feature vector**. A smaller dimension is a **by-product, not the objective**. "จำไว้นะครับ". Not for shrinking the image file.
- **[00:17:36]–[00:17:42]** Number of output vectors = **number of patches** ("จำนวน Vector นะ ไม่ใช่ขนาด Vector").
- **[00:18:48]–[00:19:24]** Patch embedding keeps more information / less loss, but costs more memory and time.
- **[00:20:06]–[00:20:18]** He can't say whether overlap or non-overlap is better. It depends on the dataset/problem.
- **[00:21:57]** "อันนี้สำคัญนะครับ สติดีๆ" → Feature vector = **What**, Positional Embedding = **Where**, "จดไปครับ" [00:22:03].
- **[00:25:13]–[00:25:29]** Fixed: patch 1 always gets the same positional vector. What and Where are **added element-wise**.
- **[00:26:45]–[00:27:08]** Learned absolute: after training, the same position uses the same vector, so it **fits the training dataset**.
- **[00:27:22]** "จุดนี้เป็นจุดที่เด็ก ๆ จะงงกัน" → Relative bias is also learned but is added at a **later step** (attention scores), not to the embedding.
- **[00:28:20]–[00:28:25]** Conditional is the **only** one where the same patch position in different images gets **different** vectors. It is common in newer models [00:28:39].
- **[00:29:35]–[00:30:59]** Class question: *why does a Transformer need positional info when a CNN doesn't?* Answer: CNN filters see neighbours. Attention sees every token, so it doesn't know who is near it.
- **[00:31:17]** Data prep is done **only once**. An encoder block ≈ one conv block [00:31:31].
- **[00:33:55]** "กาดอกจันทร์ไปเลยครับ" → **Cross-attention** combines more than one data type (Visual Language Models, e.g. image + text in ChatGPT, not OCR).
- **[00:37:52]–[00:38:06]** "จดไปครับ" → **Q matches with K, and does not touch V**.
- **[00:38:49]** "จดไปครับ" → Q×K gives the **relationship between patches**.
- **[00:40:47]–[00:40:55]** Each token has its own Q, K, V ("ของใครของมัน"). 10 tokens → 10 sets.
- **[00:44:14]–[00:44:22]** "How do you know it's really the Key?" "ไม่รู้" → abstract features are **named by how they are used**.
- **[00:44:29]–[00:44:53]** "กาดอกจันครับ จุดนี้จะเป็นจุดที่เด็ก ๆ ชอบผิด" → the **3 Linear layers for Q, K, V are learnable parameters and must be trained**.
- **[00:45:21]–[00:46:12]** "จดไปครับ … สิ่งสำคัญอย่างหนึ่งที่บางคนชอบผิด" → Q/K/V are computed **one token at a time ("โทเคนใครโทเคนมัน") but from all channels of that token**.
- **[00:51:27]–[00:51:50]** "เด็ก ๆ จำไว้เลยนะครับ ถ้าเกิดกรรมการถามคุณว่า…" why multiplying vectors shows a relationship: it is like cosine similarity (similarity), and in images similar things are likely related. "จดความเชื่อมโยงนี้ไว้".
- **[00:51:54]–[00:52:09]** But it is **not** cosine similarity (he admits he once thought so). It is **Scaled Dot-Product**, and only the divisor differs. [00:52:33] "จดไปครับ".
- **[00:53:16]–[00:53:52]** √d_k: d_k = number of channels of Q/K (4 → √4 = 2). Dot product depends on **magnitude and direction**, cosine on **direction (angle) only**.
- **[00:54:47]–[00:55:30]** Softmax normalises to 0–1 and the **sum must be 1** (why not Sigmoid/Tanh). It is applied **per query**.
- **[00:55:48]–[00:56:02]** Output of softmax = **Attention Weight A**. "จนถึงตอนนี้เรายังไม่ได้ใช้ V".
- **[00:57:29]–[00:57:33]** "จดไปครับ" → S/A steps = **considering relationships between patches**.
- **[00:58:47]** "จดไปเลยครับว่า Z เท่ากับ A คูณ V มันคือการรวมข้อมูลระหว่างแพทช์" (because it is multiply-then-sum, a weighted sum). [00:59:47] "จดไปครับ A คือตัวบ่งบอกความสำคัญ".
- **[01:01:05]–[01:01:41]** Ties back to the definition: Q vs K = learning relationships; Z = A·V = exchanging information between tokens. "นี่คือ Main Concept".
- **[01:04:23]–[01:04:35]** "กาดอกจันครับ Linear W0 สำคัญครับ" → **W_O merges the information of each head**.
- **[01:05:01]–[01:05:27]** "จดไปครับ" → the concat + Linear is done **per token**: combining each token's results from all heads.
- **[01:06:50]–[01:07:02]** LayerNorm is needed because the residual sum may go out of range. The slide diagram, however, places LN **before** MHA/MLP (Pre-LN).
- **[01:07:29]–[01:08:38]** MLP = Linear → GELU → Dropout → Linear, same size in/out, wider in the middle. It **blends all channels of each token** ("อันนี้สำคัญนะ"), because attention looks channel-wise.
- **[01:09:43]–[01:09:51]** You can't plot a feature map for a Transformer, but you can plot the attention map / Grad-CAM.
- **[01:10:22]–[01:10:27]** Deeper is not necessarily better, and shallower is not either. It depends on task complexity.
- **[01:11:02]–[01:12:35]** Two output styles: **GAP over tokens** vs **CLS token** (ViT uses CLS, from NLP). The GAP style has no CLS token.
- **[01:14:39]–[01:15:00]** RNN: early steps are local, later steps global (they accumulate earlier info). Self-attention is global, like fully connected, but its weights change with the data.
- **[01:15:40]–[01:15:54]** From review papers: nobody can say CNN or Transformer is better for classification. "Depends on your dataset and your application".
- **[01:15:59]–[01:18:24]** A misconception he once held himself: "CNN weights are fixed, Transformer weights change". This is **not 100% correct**. After training, **both** have fixed parameters (CNN kernels; Transformer's Linear W_Q/W_K/W_V). What changes with the input in a Transformer are **Q, K, V** (and the attention map), so the combination weights adapt to the content.
- **[01:18:43]–[01:19:49]** Use a Transformer when you want content-adaptive weighting and long-range context (stitches example). It helps separate similar-looking things by their context.
- **[01:20:55]–[01:22:22]** "ประเด็นนี้เป็นประเด็นที่มันจะทำให้งงได้" (the table is posted on jlearn). Three different "processing points":
  1. Q/K/V Linear: **one token, all channels**.
  2. A·V: **across many tokens, channel-wise** (1 head ≈ 1 kernel/channel).
  3. Concat + W_O: **one token, all heads' channels** (≈ kernels → filter).
  - Then the MLP processes each token separately as the final blend.
- **[01:22:47]–[01:23:10]** "สิ่งที่อาจารย์ย้ำเตือน" → **both CNN and Transformer can do image classification, regression, object detection, segmentation, and video analysis**. Which one fits depends on the **dataset and application**.

---

## Likely exam questions

1. **Define a Transformer. What are its two core ideas?**
   A neural-network architecture that uses attention to (1) learn relationships between tokens and (2) exchange/aggregate information between tokens, producing context-aware features. In self-attention, (1) = Q·Kᵀ/√d_k + Softmax and (2) = Z = A·V.

2. **What is a token in NLP vs images? How does a patch differ from a token?**
   A token is a vector representing part of the data. In NLP it is a word/sub-word/character; in images it is an image sub-region (patch). A patch is the 2-D region; a token is its 1-D embedded vector (Flatten → Linear).

3. **Describe ViT's pipeline from image to prediction.**
   Split into non-overlapping patches → flatten → linear projection (patch embedding) → (prepend CLS) → add positional encoding (+ embedding dropout) → N Pre-LN encoder blocks: U = X + MHA(LN₁(X)), Y = U + MLP(LN₂(U)) → CLS token (+LN) or GAP over tokens → classification head (usually 1 Linear) → output.

4. **A 224×224×3 image with 16×16 patches: how many patches/tokens? What is the flattened patch length?**
   (224/16)² = 14² = 196 patches. With CLS, 197 tokens. Each patch is 16·16·3 = 768 values. With D = 768 the input is 197×768, and the attention matrix per head is 197×197.

4b. **(Slide example) A 32×32×3 image, 16×16 patches, D = 8, 2 heads. Give the shapes.**
   - 4 patches → flatten 4×768 → Linear 768→8 → E 4×8; + P (4×8) → X₀ 4×8.
   - Per head: W_Q/W_K/W_V 8×4; Q, K, V 4×4; A = softmax(QKᵀ/√4) 4×4; Z = AV 4×4.
   - Concat → 4×8 → W_O 8×8 → 4×8. MLP 8→32→8. Block output 4×8.
   - With CLS: 5×8 → CLS 1×8 → Linear 8→3.

5. **Why is a positional encoding needed in ViT but not in a CNN?**
   Embeddings carry "what" but not "where". A CNN's filter only sees neighbours, so locality is built in. Self-attention compares every token with every token (position-agnostic), so position must be added to tell the model where each patch is.

6. **Compare Fixed Sinusoidal, Learned Absolute, Relative Position Bias, and Conditional PE.**
   Fixed: sin/cos formula, not learned, added to embeddings, same position → same code. Learned Absolute: a learnable vector per position, added to embeddings, fixed after training (fits the training dataset). Relative Bias: learnable bias by pairwise distance/direction, added to attention scores **before Softmax**. Conditional: generated from the patch and its neighbours (depth-wise conv), so it changes with the input image even at the same position. (Slide p.15 also lists RoPE/2D RoPE, which rotates Q and K by position, and ALiBi, which adds a bias that lowers the score with distance.)

7. **Explain Q, K, V and how they are obtained.**
   Q = what a token is looking for, K = what represents a token for matching, V = the token's information. Each token is passed through three separate trainable Linear layers W_Q, W_K, W_V (per token, using all its channels). Q is compared with K; V is only used afterwards, weighted by attention.

8. **Write the scaled dot-product attention formula and explain each step.**
   softmax(QKᵀ/√d_k)·V. QKᵀ = similarity/relationship between every query and every key. Dividing by √d_k keeps scores from growing with dimension so Softmax isn't too peaked. Softmax (per query row) turns scores into weights 0–1 that sum to 1 (A). A·V = weighted sum of values, which aggregates information from related tokens.

9. **Scaled dot-product vs cosine similarity?**
   Both use the dot product. Cosine divides by ‖q‖‖k‖ → depends only on direction, range [−1, 1]. Scaled dot-product divides by √d_k → depends on direction and magnitude, unbounded. Its purpose is to prevent overly peaked Softmax at large dimensions.

10. **Why Softmax instead of Sigmoid/Tanh for attention weights?**
    Softmax forces the weights for each query to sum to 1, giving a distribution of importance over tokens. Sigmoid/Tanh don't guarantee that.

11. **What is multi-head self-attention and what does W_O do?**
    Several self-attention heads run in parallel, each able to attend to different relations (like multiple conv filters). Each head outputs one Z per token. The heads' outputs are concatenated per token and passed through Linear W_O, which merges the heads' information.

12. **What are the roles of the skip connection, LayerNorm, and MLP in an encoder block?**
    Skip: adds the attended output to the original input (keeps old info, adds new). LayerNorm: normalises so values stay in range; in the slide's Pre-LN block it is applied **before** MHA and before the MLP. MLP (Linear–GELU–Dropout–Linear, expanded in the middle): mixes/blends all channels **within each token**. Attention mixes **between tokens**.

13. **CLS token vs Global Average Pooling for classification?**
    CLS: a learnable dummy token is prepended. After the encoder it summarises the image, and only it goes to the head (ViT). GAP: average each channel over all output tokens to get one vector, then the head. No CLS token is needed.

14. **"CNN weights are fixed after training but Transformer weights change with the input." True?**
    Not entirely. After training, both models' parameters are fixed (CNN kernels; Transformer's W_Q, W_K, W_V, W_O, MLP). What changes per input in a Transformer is Q, K, V and the attention map, so the mixing weights between tokens are content-dependent. In a CNN, the feature map changes but the kernel's mixing is fixed over a fixed local region.

15. **Compare CNN, RNN, and self-attention (complexity, parallelism, path length).**
    CNN: local, hierarchical, parallel, O(k·n·d²), path O(log_k n). RNN: sequential, not parallel, O(n·d²), O(n) sequential ops, path O(n). Self-attention: global, parallel, O(n²·d), O(1) sequential ops, path O(1).

16. **Overlapping vs non-overlapping patches?**
    ViT uses non-overlapping (stride = patch size). Overlapping preserves transitions/continuity across patch borders, but yields more patches, so more memory and compute. Which is better depends on the dataset/problem.

17. **Which should you choose for a medical imaging task, CNN or Transformer?**
    Nobody can say in general. Both can do classification, regression, detection, segmentation, and video. It depends on the dataset and application. Transformers help when long-range/global context and content-adaptive weighting matter (e.g. similar structures far apart, like stitches on both sides of a wound).

---

## Common traps

- **Embedding is not compression.** The goal is a representative feature vector. Smaller size is only a by-product.
- **Flatten ≠ embedding.** Flatten only reshapes 2-D→1-D (still raw pixels). The Linear projection does the embedding.
- **Number of tokens = number of patches**, not the vector length. The vector length (D) is set by the Linear layer.
- **Don't forget the CLS token**: 196 patches → **197** tokens in ViT. Models using GAP have no CLS.
- **Positional encoding is added, not concatenated.** The shape stays the same.
- **Relative Position Bias is not added to the embedding.** It is added to the **attention scores before Softmax**.
- **Only Conditional PE** changes with the image content at the same position. Fixed and Learned Absolute give the same code for the same position.
- **Q matches with K only. V is not used** until after Softmax (Z = A·V).
- **W_Q, W_K, W_V are learnable and must be trained**, not fixed formulas. Their "Q/K/V" names come from their use.
- **Q/K/V are per token but use all channels** of that token, not one channel.
- **The divisor is √d_k (key dimension)**, not d_k and not the vector norms. Don't call it cosine similarity: it is unbounded and magnitude-sensitive.
- **Softmax is applied per query (row)**. Each row sums to 1.
- **Q·K = relationship; A·V = aggregation/information exchange.** Don't swap them.
- **Multi-head outputs are concatenated per token, then W_O**. They are not averaged.
- **The MLP mixes channels within a token.** Attention mixes **between tokens**.
- **"Transformer weights change with the input" is wrong** if you mean parameters. The parameters are fixed after training. Q, K, V and attention maps change.
- **Self-attention cost is quadratic in the number of tokens (O(n²·d))**. Smaller patches mean many more tokens and much higher cost.
- **The original Transformer = encoder + decoder. ViT = encoder only.**
- **More encoder blocks ≠ always better**, and they cost more compute.
- **Transcript slips to ignore:** at [00:19:46] the teacher said "Non-Overlap จะเก็บข้อมูล Transition ได้ดีกว่า", but from context he means **Overlap**. At [00:23:34] he says "อันที่ 5" for Conditional. That is **not** a slip: slide p.15 numbers 6 types and Conditional is #5. He explained only 4 of them. He says "KQV" but textbooks say "QKV" (same thing). "W0" = **W_O**.
- **LayerNorm placement:** the slide's encoder block is explicitly **Pre-LayerNorm**: U = X₀ + Dropout(MHA(LN₁(X₀))), Y = U + Dropout(MLP(LN₂(U))). LN comes **before** MHA and the MLP, and the skip goes around LN + sublayer + Dropout. The teacher's spoken "add, then LayerNorm" wording ([01:06:50]) does not match the diagram. **Use the slide version in the exam.**
- **Each head has its own W_Q/W_K/W_V** (8×4 in the example). Heads are not just copies of one projection.
- **Same Linear for every patch, same W_Q/W_K/W_V for every token, same MLP for every token.** The weights are shared across tokens; only the token data differs.
- **Dropout is active only during training.**
