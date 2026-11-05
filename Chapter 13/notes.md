# Chapter 13 — Deep Learning for Video Processing

A video is time-series data, a sequence of frames, so it carries both **spatial** information (inside each frame) and **temporal** information (change between frames). The chapter builds the tools for both: RNN, then LSTM, then ConvLSTM (with PyTorch code), then 3D CNN / 3D max pooling, then I3D, two-stream and modern video architectures.

---

## Key concepts

### 1. Time series and video (slides 4–16)
- **Time series (ข้อมูลอนุกรมเวลา)**: data recorded over consecutive time, stored in time order. Used for **trend analysis**, **seasonal analysis** and **forecasting**. Examples on the slides: stock prices, daily temperature, sales volume.
- **Video as time series**: an ordered sequence of frames over time. It is more complex than an ordinary time series because it has both:
  - **Spatial feature (เชิงพื้นที่)**: what is *inside one frame*: edges, colour, shape, texture, position. Extracted with **CNNs (2D convolution filters)**.
  - **Temporal feature (เชิงเวลา)**: how a frame *changes compared with earlier frames*: motion, order, direction, speed, timing. Extracted with **RNN / LSTM**.
  - Teacher's wording: spatial looks at the **internal** content (within a frame), temporal looks at the **external** relation (between frames) [00:06:18].
- **Spatio-temporal**: spatial **and** temporal features extracted **at the same time (พร้อมกัน)**. If you extract spatial first and then feed the result to a temporal model, that is "spatial followed by temporal" (S→T), **not** spatio-temporal [00:06:59–00:07:23].
- Ways to combine the two: CNN + LSTM (CNN per frame, LSTM over time), or **3D CNN**, which uses 3D filters to capture spatial and temporal features in one operation (slide 8).
- **Application examples** (slide 5):
  - Action recognition (e.g., kicking, jumping, spinning in sports video)
  - Event detection (e.g., a crash on CCTV, an **elderly person falling**): CNN + RNN/LSTM
  - Behaviour analysis (e.g., how people walk in a train station): 3D CNN
- **Video data structure**: each frame is H×W×C (rank-3), and stacking frames makes a **rank-4 tensor** ("4-D data") (slides 15, 73). Tensor rank 0 is a scalar, 1 a vector, 2 a matrix, 3 or more a tensor.

### 2. Image tasks vs. video tasks (lecture [00:12:31]–[00:18:19])
Slide 10 ("ความเชื่อมโยงงานภาพในมุมมองงานวิดีโอ") sets this out as image = spatial dimension, video = temporal dimension.

| Image task | Video counterpart | Output / ground truth (slide 10 example) |
|---|---|---|
| Image classification: one label per image | **Video classification**: one label per clip that summarises the whole clip | one class for the whole video (slide: swimming clip gives "ว่ายน้ำ" (swimming); lecture: standing / sitting) |
| Image segmentation: a label for every pixel | **Temporal action segmentation**: a label for **every frame** | one class per frame (slide: cooking clip labelled prepare → cut → cook → plate; 100 frames give a 100-element vector) |
| Object detection: object class + bounding box | **Temporal action localization**: action class + **start–end time** | only the time intervals of the events of interest (slide: "ยิงประตู 00:15–00:20", a shot at goal, plus a foul segment); not every frame is labelled |
| Image regression | **Video regression** | a value (e.g., heart rate estimated from a video of skin) |

The slide footer reads: segmentation over time ≠ pixel segmentation in video.

### 3. Why RNN? Limits of NN/CNN (slides 18–20)
NN and CNN are designed for inputs that:
1. arrive **all at once** (a whole image or a whole vector; you cannot feed half an image),
2. have **no sequential relation** that affects the prediction,
3. have a **fixed, equal size**.

So NN/CNN do not suit ordered data or inputs of different sizes (videos have different lengths). A new architecture must (1) accept input piece by piece, (2) extract temporal/sequence information, and (3) accept inputs of different sizes. That architecture is the **RNN** [00:23:05–00:24:19].
- An RNN predicts from **two sources**: the **current input** and the **previous information/state** (the purple loop on the slide). It works like a human reading word by word.
- **Unfold**: there is only **one RNN cell** (hidden layer), reused at every time step. "A recurrent neural network can be seen as the repeated use of a single RNN Cell" (slide 21).
- At the very first step there is no previous information, so H₀ = 0 [00:29:34].
- **RNN Cell vs. RNN Cell Forward** (slide 25): the cell computes Hₜ (feature extraction). The "forward" part (Hₜ·W + b, then softmax) turns Hₜ into an output. It is **optional** and can be redesigned for the task.
- You can output at **every time step** (e.g., translation) or **only the last step** (e.g., video classification) [00:31:35–00:32:39].
- **Long-sequence problem** (slide 26): information from early steps fades as it passes through many cells, so late cells hold almost nothing from early cells. RNN is not suitable for very long series.

**RNN pros / cons (slide 29)**
| Pros | Cons |
|---|---|
| Can process inputs of different lengths | Slow computation |
| Model size does not grow with input size | Hard to access information from long ago |
| Computation considers past information | Cannot consider future inputs for the current state |
| Weights are shared across time steps | |

### 4. LSTM (slides 31–45)
- **LSTM** = a type of RNN that can learn **long-term dependencies**. Proposed by **Hochreiter et al., 1997**. Designed to avoid the **long-term dependency problem**.
- Structure (the teacher's breakdown): **2 states + 3 gates + 2 "modulates"**
  - **2 states**
    - **Cell state = long-term memory**: encodes information *accumulated (รวบรวม) from all previous steps*, from the start until now. It is only multiplied by f and added to, and is **never squashed by tanh**, which is why it is believed to "not forget".
    - **Hidden state = short-term memory**: encodes the "specific characteristics" of the data, kept **for predicting next** ("keep for predict"). It is mostly influenced by recent steps. It is **not** the prediction itself; it can be passed to a forward layer to produce an output.
  - **3 gates (sigmoid, range 0 to 1)**
    - **Forget gate (also called Remember gate) = "Delete"**: how much of the old cell state Cₜ₋₁ to keep. 0 means erase all, 1 means keep all.
    - **Input gate = "Write"**: how much new information flows into the cell state. 0 means none, 1 means all.
    - **Output gate = "Read"**: how much of the cell state is exposed to the outside as hₜ. It only reads; it does not change the cell state.
  - **2 modulates (tanh, range −1 to 1)**
    - **Input modulate gate** (zₜ, often called the "candidate" or g): prepares the new candidate content from xₜ and hₜ₋₁.
    - **Output modulate gate**: tanh(Cₜ) before it is multiplied by oₜ. **This name was made up by the teacher** ("ส่วนนี้ผมตั้งเอง"). The official parts are only the input, forget and output gates plus the input modulate gate.
- **Sigmoid vs. tanh roles**: a **sigmoid** produces a **weight / gate** (0 means ignore, 1 means keep everything). A **tanh** **normalises data/information** into −1…1 (zero-centred data learns faster).
- **Gate weights are adaptive**: every gate is computed from **xₜ and hₜ₋₁**, so its value changes with each input, even at inference. This differs from CNN weights, which are fixed after training. Each gate still has its **own learnable W and b**, which is why the gates differ [01:09:53–01:12:36].
- **Analogies the teacher used**: the input gate is like how much you actually understand and remember from reading a book. The forget gate is like how much of your old long-term memory survives as time passes. The output gate is like answering only what is asked (asked about addition, you don't explain multiplication).

### 5. RNN vs LSTM vs GRU
| | RNN | LSTM | GRU* |
|---|---|---|---|
| States | hidden Hₜ only | cell state Cₜ (long) + hidden hₜ (short) | hidden only |
| Gates | none (one tanh) | forget, input, output (+ tanh input modulate) | update, reset |
| Long-term memory | poor (early info fades) | good (additive cell-state path) | good |
| Cost | lowest | highest | between the two |

*GRU is **not on the slides**. The teacher mentions it only once, as an alternative to LSTM/RNN after I3D features [01:50:14]. The GRU column is a standard fact, not course material.

### 6. Pipelines for video and signal (slides 16, 46, 50, 97, 103, 104)
The teacher numbers four approaches [01:23:30–01:25:22]:
1. **CNN → feature vector per frame → LSTM → prediction**. The CNN usually stops at the **flatten** layer. This is **S→T (sequential)**, not spatio-temporal.
2. **Your own feature extraction** (hand-crafted filters, e.g., a −1 −1 −1 / 0 0 0 / 1 1 1 edge filter, MFCC/spectrogram for signals) → feature vector → LSTM, *or* feature map → CNN.
3. **ConvLSTM** → optional NN → prediction (spatio-temporal).
4. **3D CNN** → optional NN → prediction (spatio-temporal).
- For signals: cut the signal into segments, encode each segment as a feature vector, and feed them in order to RNN/LSTM. If the features look like an image (e.g., a spectrogram), you can feed them to a CNN instead [01:20:56–01:23:11].

### 7. ConvLSTM (slides 53–56; lecture [01:25:30–01:34:38])
- **Convolutional LSTM**, proposed by **Xingjian Shi et al., 2015**. It is a recurrent network for **spatial + temporal** prediction (e.g., movies, satellite images, security cameras). It has 2 main parts: **convolution layer + LSTM**.
- The base is **LSTM** ("เขาเอา LSTM เป็นพื้นฐาน… นี่คือจุดเน้น" [01:34:05]). Changes from LSTM:
  - the **matrix multiplications** on the input xₜ and hidden hₜ₋₁ are replaced by **convolution**,
  - so the states are **feature maps (2D)**, not 1D vectors,
  - **element-wise** products (∘) remain for f∘C, i∘g and o∘tanh(C).
- **How it works (PyTorch cell, slide 64)**: concatenate [xₜ, hₜ₋₁] along the channel axis. One `Conv2d` with `out_channels = 4 × hidden_dim` produces all four pre-activations, which are split into i, f, o, g. Then i, f, o = sigmoid and g = tanh, `c_next = f*c + i*g` and `h_next = o*tanh(c_next)`. In the teacher's whiteboard example, hidden_dim = 3 gives 12 filters, which are cut into 4 groups of 3 [01:27:37–01:28:49].
- **ConvLSTM is spatio-temporal (ST)**: the 2D filter gives spatial features, and because the channels it mixes come from **different times** (xₜ and hₜ₋₁), it is temporal at the same time [01:32:49–01:33:47].

### 8. Building a ConvLSTM in PyTorch (slides 58–69)
The usual 5 steps: (1) define the Dataset, (2) define the ConvLSTM model, (3) choose the loss and optimizer, (4) train, (5) test/evaluate.
- **Dataset**: video classification input is **5-D**: (Batch, Frame, Width, Height, Channel) as written on slide 59. There is no ready-made dataset, so you write a custom `VideoDataset(Dataset)`:
  - `__init__`: stores paths and labels for `.mp4/.avi/.mov` files found in each class folder.
  - `__getitem__`: `load_video` gives a numpy array of frames, the transform is applied **frame by frame**, then `torch.stack` makes a tensor and the label index is returned.
  - `load_video`: `cv2.VideoCapture` and `cap.read()` in a loop. `ret` is True on success and False when reading fails, which stops the loop. Each frame is resized, reading stops at `max_frames`, and `cap.release()` sits in `finally`.
- **Parameters on the slide**: classes `['Biking','Diving','HorseRiding']`, `batch_size=4`, `max_frames=30`, `resize=(112,112)`, `Normalize(mean=0.5, std=0.5)`.
- **num_workers**: 0 or 1 means a single CPU loading instance. More than 1 creates that many instances. It should not exceed the number of CPU cores.
- **Model**: PyTorch had no built-in ConvLSTM, so the class comes from the ndrplz/ConvLSTM_pytorch repo. The slide uses `ConvLSTM(input_dim=3, hidden_dim=[64,64], kernel_size=(3,3), num_layers=2, batch_first=True)`, takes the **last hidden state** of the last layer, flattens it, and passes it to `nn.Linear(112*112*64, num_classes)`.
- **num_layers** is the number of stacked ConvLSTM layers. The FC layer can take **all time steps** or **only the last** step [01:39:07–01:39:24].
- **Loss/optimizer**: `nn.CrossEntropyLoss()` with `optim.SGD(lr=0.001, momentum=0.9)`. Pick them according to the task: regression, multiclass, multilabel or multitask.
- **Training loop**: `zero_grad`, forward, loss, `backward`, `step`. Train for 10 epochs, then `torch.save(model.state_dict(), ...)`.
- **Testing**: `model.eval()`, `torch.no_grad()`, `torch.max(outputs, 1)` and `sklearn.metrics.accuracy_score`.
- **Inference on one video**: load the frames, transform each frame, `torch.stack`, then **`unsqueeze(0)`** to add the batch dimension. Run the model, `softmax`, then `torch.max` to get the predicted class and its %.

### 9. 3D CNN (slides 72–92)
- **3D CNN**: a CNN designed for 3-D data. It uses **more memory and time** than 2D but is usually **more accurate** than 2D on 3-D data.
- **2D vs. 3D convolution** (slide 75):
  - **2D**: the filter slides only along **x and y**, and at each position it uses **all channels** (the filter depth equals the input depth).
  - **3D**: the filter slides along **x, y and z (time)**. Its depth can be **smaller** than the input's.
- **3D conv is a spatio-temporal operation**: it extracts the spatial feature of each frame and the relation between consecutive frames at once (slide 77).
- **3D max pooling** usually comes paired with 3D conv. With stride 2 it halves every dimension (4×4×4 becomes 2×2×2).
- **Pros/cons** (slide 92): more accurate than 2D and gets spatial + temporal together, **but** it has **many parameters**, so there is a **high overfitting risk**, and **few pretrained models** exist.

### 10. Action recognition problems, Two-Stream and I3D (slides 98–102)
- **Problems in action recognition**:
  - common video datasets (**UCF-101, HMDB-51**) are small, so complex models overfit,
  - image ConvNets are advanced but hard to apply when time is involved,
  - 3D ConvNets have many parameters and are hard to train.
- **Two-Stream** processes "**RGB frames**" and "**Optical Flow** (motion)" separately, mostly with 2D ConvNets, and **fuses** the results at the end (late fusion). The teacher's simplified description of optical flow is "frame minus frame" (frame 1 − frame 2, 2 − 3, …) to expose changes between frames [01:49:04–01:49:41]. Strictly, optical flow is per-pixel motion estimation, but frame difference is the teacher's intuition.
- **I3D (Inflated 3D ConvNet)**: "inflates" a well-designed 2D model (**Inception-v1**) into 3D, so it sees spatial and temporal together.
  - **Inflation**: 2D N×N filters and pooling become **T×N×N** (T = temporal extent).
  - Key design choice: the **temporal stride/pooling**, which sets a suitable temporal receptive field.
  - **Two streams** (RGB + optical flow). Each goes through a 3D ConvNet, Average Pool 3D, a 1×1 conv and a score (1×nc). The two scores are averaged, then softmax.
  - I3D is pretrained on video, so it can be used as a feature extractor followed by LSTM/RNN/GRU [01:50:05].
- **1×1×1 convolution in I3D**: temporal size 1 and spatial size 1×1. It looks at neither space nor time; it is only a **linear combination of channels**. Roles:
  - **channel reduction** (less computation and memory),
  - **channel mixing**,
  - **non-linearity** (with ReLU + BatchNorm),
  - **flexibility** in Inception branches (short, light features).

### 11. Modern video architectures (slides 93–96; names only)
| Architecture | Year |
|---|---|
| SlowFast (Feichtenhofer et al., ICCV) | 2019 |
| X3D | 2020 |
| Temporal Pyramid Network | 2020 |
| MViTv2 | 2021 |
| Video Swin Transformer | 2021 |
| VideoMAE | 2022 |
| UniFormer | 2023 |
| TimeFormer | 2025 |

- The teacher describes SlowFast as combining a coarse view and a fine view [01:46:41].
- Most ready-made architectures use either ConvLSTM or 3D convolution inside. Otherwise they extract spatial features first and then process the temporal dimension [01:50:57].

### Summary: video architectures
| Approach | Spatial | Temporal | S→T or ST? | Notes |
|---|---|---|---|---|
| CNN (per frame) + LSTM/RNN | 2D CNN up to flatten | LSTM | **S→T** | simple, can use pretrained image CNN |
| Hand-crafted features + LSTM / CNN | custom | LSTM | S→T | e.g., MFCC, edge filters |
| ConvLSTM | conv inside the gates | LSTM recurrence | **ST** | states are feature maps |
| 3D CNN (+3D max pool) | 3D kernel | 3D kernel along time | **ST** | many params, overfit risk, few pretrained |
| Two-Stream | RGB stream | optical-flow stream | parallel, then late fusion | mostly 2D ConvNets |
| I3D | inflated Inception 3D | 3D + flow stream | ST (two-stream) | pretrained on video |

---

## Formulas & numbers to remember

### RNN (slides 22–24)
- **Hₜ = tanh(Xₜ·Wₓₜ + Hₜ₋₁·Wₕₜ + Bₜ)**: current input plus previous information. This is the feature-extraction part.
- **Yₜ = softmax(Hₜ·Wᵧₜ + Bᵧₜ)**: the output, which is the optional "RNN Cell Forward".
- H₀ = 0 at the first step.

### LSTM (slides 37–44)
- Forget: **fₜ = σ(W_xf·xₜ + W_hf·hₜ₋₁ + b_f)**, range 0…1
- Input: **iₜ = σ(W_xi·xₜ + W_hi·hₜ₋₁ + b_i)**, range 0…1
- Output: **oₜ = σ(W_xo·xₜ + W_ho·hₜ₋₁ + b_o)**, range 0…1
- Input modulate: **zₜ = tanh(W_xz·xₜ + W_hz·hₜ₋₁ + b_z)**, range −1…1
- Cell update: **Cₜ = fₜ ∘ Cₜ₋₁ + iₜ ∘ zₜ** (the kept old part plus the added new part)
- Hidden ("output modulate"): **hₜ = oₜ ∘ tanh(Cₜ)**
- 2 states, 3 gates, 2 modulates. Sigmoid is for gates/weights; tanh is for data. LSTM dates from 1997 (Hochreiter et al.).

### ConvLSTM (slide 56)
- iₜ = Sigmoid(Conv(xₜ; w_xi) + Conv(hₜ₋₁; w_hi) + b_i)
- fₜ = Sigmoid(Conv(xₜ; w_xf) + Conv(hₜ₋₁; w_hf) + b_f)
- oₜ = Sigmoid(Conv(xₜ; w_xo) + Conv(hₜ₋₁; w_ho) + b_o)
- gₜ = Tanh(Conv(xₜ; w_xg) + Conv(hₜ₋₁; w_hg) + b_g). Here gₜ is the input modulate / candidate, which is zₜ in the LSTM slides.
- cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ
- hₜ = oₜ ⊙ Tanh(cₜ)
- In short: the LSTM equations with W·x and W·h replaced by **Conv(x; w) and Conv(h; w)**. The ⊙ products stay element-wise.
- PyTorch cell: `Conv2d(in = input_dim + hidden_dim, out = 4·hidden_dim, padding = k//2)`. The output is split into i, f, o, g. Padding k//2 keeps H×W unchanged.
- Element-wise product = multiply position by position, e.g., [1,2,3,4] ∘ [1,0,0,−1] = [1,0,0,−4].
- ConvLSTM dates from 2015 (Shi et al.).

### Tensor shapes (watch the axis order)
| Where | Shape |
|---|---|
| Video in memory | rank-4: frames × H × W × C |
| Dataset text (slide 59) | (Batch, Frame, Width, Height, Channel) |
| ConvLSTM code, `batch_first=True` (slide 63) | **(B, T, C, H, W)** |
| `nn.Conv3d` / I3D slide | **(B, C, T, H, W)** = (N, C, D, H, W) |

- **ConvLSTM example from the slides**: input (4, 30, 3, 112, 112). The last hidden state is (4, 64, 112, 112), `.view(4, -1)` makes it (4, 802,816), and FC gives (4, 3).
- **One video at inference**: stacked frames (30, 3, 112, 112), then `unsqueeze(0)` gives (1, 30, 3, 112, 112).
- **I3D (slide 99)**:
  - Video (25×3×16×224×224) = (Batch×Channel×Time×H×W). The optical flow input has the same shape.
  - After the 3D ConvNet the feature map is (25×2048×2×7×7): time 16 becomes 2 (÷8), and space 224 becomes 7 (÷32).
  - Final feature vector (25×1×2048). Score per stream (1×nc), then average and softmax.

### 3D conv / pooling output size
Per dimension (D/T, H, W):  **O = ⌊(I + 2P − K) / S⌋ + 1**. This is the standard formula; the slides show it through examples.

- **Slide worked example (slides 78–83)**: data D is 4×4×4, filter F is 3×3×3, stride 1, no padding, so the output O is **2×2×2** ((4−3)/1+1 = 2).
  - Each filter slice is [[1,−1,−1],[−1,1,−1],[−1,−1,1]].
  - The data slices are:
    - D[0] = [[3,2,1,3],[0,7,3,3],[3,9,3,4],[2,3,8,3]]
    - D[1] = [[0,2,1,3],[0,4,3,2],[3,9,3,4],[2,6,8,1]]
    - D[2] = [[3,2,1,1],[0,3,3,3],[3,4,3,4],[2,3,6,3]]
    - D[3] = [[3,2,1,3],[0,3,6,3],[0,9,3,4],[2,3,1,3]]
  - Method: lay F[0], F[1], F[2] over 3 consecutive data slices (D[0..2] for O[0], D[1..3] for O[1]). Multiply element-wise, sum each slice, then **add the 3 slice sums**. Then slide along x, y **and** z (depth).
  - Final grid on slide 83: **O[0] = [[−20, −37], [−15, −55]]**, **O[1] = [[−24, −29], [−22, −55]]**.
  - **The slides contain arithmetic slips. I recomputed every value from the slide data:**
    - O[0][0,0]: slides 80/81 show **−21**. The first slice is written as "3+0−3−2+7−9−**2**−3+3 = −6", but the term is 1(−1), so the slice sum is **−5**. The correct value is −5 + (−11) + (−4) = **−20**, which matches the final grid.
    - O[0][0,1]: the slide shows −37 = −17 − 13 − 7. The third slice "2−3−4−1+3−3−1−3+4" actually equals **−6**, not −7, so the correct value is **−36**.
    - O[1][1,0]: the slide shows −22. Recomputed it is **−18**.
    - All other cells are correct: −15, −55, −24, −29, −55.
    - **Correct output: O[0] = [[−20, −36], [−15, −55]], O[1] = [[−24, −29], [−18, −55]].**
  - In an exam, show the method. If you quote the slide's grid, be aware these three cells differ.
  - The teacher's walkthrough [01:42:25–01:43:22]: slide the same filter along x, y **and z**.
- **3D max pooling (slides 84–89)**: 4×4×4 (same D) with a 2×2×2 window and stride 2 gives **2×2×2**. Take the max of each 2×2×2 block. Result **O[0] = [[7, 3], [9, 8]]**, **O[1] = [[3, 6], [9, 6]]**. I verified these; they are correct on the slides. The teacher says it usually halves every dimension (depth/channel, width, height) and uses stride 2.
- **`nn.Conv3d(16, 33, 3, stride=2)`** (slide 91) means 16 input channels, 33 output channels (= number of filters), kernel 3×3×3, and stride 2 in every dimension.
  - With the slide's input `torch.randn(20, 16, 10, 50, 100)` (N, C, D, H, W): D = ⌊(10−3)/2⌋+1 = 4, H = ⌊(50−3)/2⌋+1 = 24, W = ⌊(100−3)/2⌋+1 = 49, so the **output is (20, 33, 4, 24, 49)**. I computed this; the slide does not show it.
  - Second slide example: `Conv3d(16, 33, (3,5,2), stride=(2,1,1), padding=(4,2,0))` on the same input gives **(20, 33, 8, 50, 99)**. Also computed by me.
  - Parameter count of `Conv3d(16,33,3)`: 33·16·27 + 33 = 14,289. Computed by me.
- **Teacher's channel-stacking example [01:41:45]**: RGB frames stacked as channels and a 3D filter of depth 6 means the filter covers **2 frames at a time**. It then steps along the time/channel axis, so every window combines spatial information with information from more than one frame, which makes it spatio-temporal.

---

## What the teacher emphasised

- **[00:01:38]–[00:01:55] Video is a time series.** Each image in a video is a **frame**. In this class only the image part is covered, not audio.
- **[00:06:18]** Spatial means internal (within a frame); temporal means external (the relation between frames).
- **[00:06:59]–[00:07:23] Definition of spatio-temporal.** It means both are considered **at the same time**. "ถ้าเราพิจารณา spatial ก่อน… ค่อยเอาผลลัพธ์ที่ได้ไปพิจารณาต่อ temporal เราจะไม่เรียกเป็น spatial temporal" (if you do spatial first and then temporal, it is not called spatio-temporal).
- **[00:10:22]–[00:10:49] Rule of thumb**: spatial uses convolution; temporal uses the RNN/LSTM family; both together use 3D convolution or ConvLSTM. "**คีย์เวิร์ดก็คือพร้อมกัน**" (the keyword is *simultaneous*).
- **[00:11:35]–[00:12:21] Why temporal matters.** With only frame 1 and frame 5 and no order, you **cannot tell "standing up" from "sitting down"**. Spatial alone "ไม่เพียงพอ" (is not enough).
- **[00:12:31]–[00:18:19]** Image tasks map to video tasks (classification, temporal action segmentation, temporal action localization, video regression). See the table above. The examples are medically flavoured: checking elderly home physiotherapy, fall detection, analysing surgical steps, and heart rate from a skin video.
- **[00:19:17]–[00:19:55]** A CNN backbone per frame followed by LSTM is "spatial then temporal", **not** spatio-temporal.
- **[00:22:23] "ขนาด input เท่ากันอันเนี้ยสําคัญมาก"** (equal input size is very important). Videos have different lengths, so you either **trim/clip them to equal length** or use a tool that supports variable length, such as an RNN.
- **[00:23:14]–[00:24:15]** The 3 properties a new architecture needs: partial input, temporal features, variable size. RNN is the answer.
- **[00:26:53]–[00:27:41] Question: data vs. information** ("ไฮไลท์และกาดอกจัน", highlight and star it). Information is what remains after processing data; it is relevant and usable. Data can include noise and irrelevant content. xₜ is data; h and C are information.
- **[00:28:44]–[00:28:54]** The hidden output represents **all previous frames**, not only the last one, because it accumulates.
- **[00:29:34]** At the first step the previous information is 0.
- **[00:30:06]–[00:30:53] In-class question: which of tanh and sigmoid is 0…1 and which is −1…1?** Answer: **sigmoid is 0 to 1; tanh is −1 to 1.** It is repeated at [00:53:04]: "เด็กๆต้องจําได้ก่อน… หาที่จดครับ" (you must remember this, write it down).
- **[00:31:04]** In an RNN, tanh acts as **normalisation**, keeping values in a standard range.
- **[00:32:29]–[00:32:39]** Output at every step only if needed. For video classification, usually take **only the last** output.
- **[00:34:29]** The weights act as **filters** that select what to use from the current and past data.
- **[00:38:26]** "RNN Cell ไม่เหมาะสำหรับข้อมูลที่เป็น Time Series ในลักษณะที่ยาวมาก" (RNN cells are not suitable for very long time series).
- **[00:40:15]–[00:40:31]** The pro "can process different lengths" only means it **can**, not that it does it **well**.
- **[00:41:43]–[00:42:31]** LSTM sends data along **two paths**: long (guaranteed to hold info from the first to the last step) and short (mostly recent steps). LSTM came after RNN, in **1997**.
- **[00:53:22]–[00:54:10], [01:20:43]** **Sigmoid works as a weight/gate** (input, forget, output gates). **Tanh normalises data/information.**
- **[00:56:14]–[00:57:19]** The **cell state is never normalised (no tanh)**. It only accumulates by addition ("เหมือนเราเก็บเงินไปเรื่อยๆ", like saving money), so it does not forget. The hidden path is normalised because it is processed further.
- **[00:59:22]–[01:11:13] Gate weights are adaptive.** The input changes, so the weight changes, even at inference. Unlike fixed CNN weights (the teacher compares this to Transformers). All three gates are generated from **x and the previous hidden state**.
- **[01:11:22]–[01:12:36]** Each gate has its **own learnable W and b**, so the same formula gives different gates.
- **[01:00:16]–[01:00:31] and [01:14:27] "แต่ย้ำนะครับ"** (I stress this): the "**output modulate gate**" is the teacher's own name. Officially there are only the input, forget and output gates plus the input modulate gate.
- **[01:01:57] "ขีดเส้นใต้คํานี้กาดอกจันครับ — รวบรวม ทั้งหมด"** (underline and star this: "gathers everything"). The cell state *accumulates* all information.
- **[01:03:00]–[01:03:32]** The slide says cell state holds "มากกว่า" (*more*), not "ครบ" (*complete*). Data is lost at the forget gate, and new info is only partly admitted by the input gate.
- **[01:04:34]** Hidden state keyword: "**keep for predict**".
- **[01:07:42]–[01:08:02], [01:12:46]–[01:14:11]** Forget gate = remember gate = **Delete**. Input gate = **Write**. Output gate = **Read**. Write and delete change the cell state; read does not.
- **[01:21:02]–[01:23:11]** Signals (and in older work, video) are first turned into feature vectors (MFCC, spectrogram, …) and then fed to LSTM. If the features look like an image, a CNN can be used instead.
- **[01:23:34]–[01:23:48]** The CNN is usually cut at the **flatten** layer to get per-frame feature vectors.
- **[01:25:25] "เดี๋ยวจะเป็นคําถาม"** (this will be a question): whether ConvLSTM is S→T or ST.
- **[01:31:24]–[01:33:47] IN-CLASS QUIZ ("พิมพ์แค่หนึ่งกับสอง", just type 1 or 2)**
  - **Question**: ConvLSTM keeps the LSTM concept but replaces the internal operator with 2D convolution, so the data is 2D maps instead of 1D vectors. Is it **(1) S→T** (spatial then temporal) or **(2) ST** (spatio-temporal, both at once)?
  - **Correct answer: 2, ST (spatio-temporal).** Most of the class answered 2 and the teacher said "ถูก" (correct).
  - **Reason**: the 2D filter running over the maps gives spatial features. Because the concatenated channels come from **different times** (frame xₜ and hidden hₜ₋₁), the same convolution also captures temporal relations. Spatial and temporal are extracted at the same time.
- **[01:34:05] "เขาเอา LSTM เป็นพื้นฐาน… นี่คือจุดเน้น"** (LSTM is the base; this is the key point). Only the x/h products change to convolution; ∘ stays element-wise.
- **[01:36:20]–[01:36:57] "อันนี้คือจุดสําคัญมาก… สิ่งที่ต้องระวัง"** (this is very important… something to watch out for). Transforms and augmentation are applied **frame by frame independently**. A random transform (e.g., random erase) will be **different in each frame**, not consistent across the video.
- **[01:37:17]–[01:37:22]** Loading and transforming happens on the **CPU**. After conversion to a tensor, the data goes to the **GPU**.
- **[01:39:36]–[01:39:59]** PyTorch had no ready-made ConvLSTM, so the teacher used a GitHub implementation. A built-in may exist now, but the teacher has not reviewed it.
- **[01:40:27]–[01:42:22] 2D vs. 3D convolution.** 2D slides x and y and uses all channels. 3D slides x, y **and z**, and the filter can be shallower than the input.
- **[01:43:43]–[01:44:12]** 3D conv gives spatial features from each placement and temporal features because it covers more than one frame, so it is a **spatio-temporal feature** extractor.
- **[01:44:25]–[01:45:21]** 3D conv usually comes with **3D max pooling**, which halves each dimension with stride 2.
- **[01:46:04]–[01:46:23]** With these operators you can now do at least video classification and video regression. Approaches: S then T, or ST together.
- **[01:48:09]–[01:50:14]** **I3D** is a CNN pretrained for video features. One stream takes RGB video and the other takes **optical flow** ("frame minus frame"). It can be followed by linear layers, LSTM, RNN or GRU.
- **[01:51:39]–[01:52:16] Wrap-up.** What is new in video compared with images is **temporal information**. There are three strategies: (a) **parallel** spatial and temporal, fused later (two-stream), (b) **sequential** S→T, (c) **simultaneous** ST. Which works best depends on the task, the dataset and what you predict. Large frames and long videos cause resource problems.

---

## Likely exam questions

1. **What is the difference between spatial and temporal features in video, and which models extract each?**
   Spatial features are the content inside one frame (edges, colour, shape, texture, position) and are extracted by a 2D CNN. Temporal features are the changes or relations between frames (motion, order, speed) and are extracted by an RNN/LSTM.

2. **Define "spatio-temporal". Is CNN-per-frame followed by LSTM spatio-temporal?**
   Spatio-temporal means spatial and temporal features are extracted **at the same time**. CNN followed by LSTM is **not**: it is spatial, then temporal (S→T).

3. **Is ConvLSTM S→T or ST? Explain.** (This was the in-class quiz.)
   It is **ST**. Its convolution runs over the concatenation of xₜ (the current frame) and hₜ₋₁ (past information). The 2D kernel captures spatial structure, and because the channels come from different times it captures temporal relations in the same operation.

4. **List three limitations of NN/CNN that motivated the RNN.**
   NN/CNN need (a) the whole input at once, (b) inputs without sequential relations, and (c) a fixed, equal input size. The RNN accepts input step by step, models sequences, and handles variable length.

5. **Write the RNN equations and explain each part.**
   Hₜ = tanh(XₜWₓ + Hₜ₋₁Wₕ + B) extracts information from the current input and the previous state; tanh normalises it. Yₜ = softmax(HₜWᵧ + Bᵧ) is the optional forward/output part. At the first step H₀ = 0.

6. **Give the pros and cons of RNN.**
   Pros: handles variable length, model size does not grow with the input, uses past information, shares weights across time. Cons: slow, hard to access information from long ago (early information fades), cannot use future inputs.

7. **Name the components of an LSTM and the role of each.**
   - 2 states: the **cell state** (long-term memory, accumulates everything, not normalised) and the **hidden state** (short-term memory, "keep for predict").
   - 3 sigmoid gates: **forget** (delete; how much of Cₜ₋₁ to keep), **input** (write; how much new content enters), **output** (read; how much of C goes out as h).
   - 2 tanh modulates: **input modulate** zₜ (the candidate content) and the "output modulate" tanh(Cₜ). The second name is the teacher's own.

8. **Write the LSTM cell update and hidden update.**
   Cₜ = fₜ∘Cₜ₋₁ + iₜ∘zₜ and hₜ = oₜ∘tanh(Cₜ).

9. **Why does LSTM use sigmoid for gates and tanh elsewhere?**
   Sigmoid's range 0…1 works as a weight (0 blocks, 1 passes everything). Tanh's range −1…1 normalises the data, and zero-centred values learn faster.

10. **Why can the cell state keep long-term information while the RNN hidden state cannot?**
    The cell state is updated only by an element-wise forget scaling plus an addition. It is never squashed by tanh, so information accumulates. The RNN hidden state is recomputed through tanh at every step, so early information fades.

11. **How do LSTM gate weights differ from CNN filter weights?**
    CNN weights are fixed after training. LSTM gate values are **adaptive**: they are computed from xₜ and hₜ₋₁, so they change with every input, even at inference. The W and b used to compute each gate are learnable, and each gate has its own.

12. **How does 2D convolution differ from 3D convolution? Why is 3D convolution spatio-temporal?**
    A 2D filter slides along x and y and covers all channels. A 3D filter also slides along z/time and covers several frames, so each output mixes spatial and multi-frame (temporal) information.

13. **Compute the output size: 4×4×4 input, 3×3×3 kernel, stride 1, no padding. Then 3D max pooling 2×2×2 with stride 2 on 4×4×4.**
    The convolution gives 2×2×2: each output value is the sum over the 3 slices of the element-wise products. The pooling gives 2×2×2.

14. **Give the output shape of `nn.Conv3d(16, 33, 3, stride=2)` on input (20, 16, 10, 50, 100).**
    It is (20, 33, 4, 24, 49). Each dimension uses ⌊(I − 3)/2⌋ + 1.

15. **Give the pros and cons of 3D CNN, and the problems in action recognition.**
    3D CNN is more accurate than 2D on 3-D data and gets spatial and temporal features together. However, it has many parameters, a high overfitting risk, few pretrained models, and needs more memory and time. Action-recognition datasets (UCF-101, HMDB-51) are small, and 3D ConvNets are hard to train, which motivates **Two-Stream** and **I3D**.

16. **What is I3D? What does the 1×1×1 convolution do in it?**
    I3D inflates 2D Inception-v1 filters (N×N) to T×N×N and runs two streams, RGB and optical flow, whose scores are averaged. The 1×1×1 convolution does a channel-only linear combination: it reduces channels, mixes channels, adds non-linearity with ReLU/BN, and gives flexibility in Inception branches.

17. **Which architecture fits each medical video scenario?**
    - **Elderly fall detection from CCTV** is event detection or video classification. The slides pair it with **CNN + RNN/LSTM**; ConvLSTM or 3D CNN are alternatives if you need ST features.
    - **Recognising surgical phases frame by frame** is **temporal action segmentation** (one label per frame). Use a temporal model with an output at every time step.
    - **Finding only the time intervals of a specific surgical event** is **temporal action localization**.
    - **Estimating heart rate from a video of skin** is **video regression**, e.g., per-frame features into LSTM, then a linear output.
    - **Checking whether home physiotherapy is done correctly** is **video classification** (one label per video, take the last LSTM output).
    - **Small medical video dataset**: avoid training a large 3D CNN from scratch because of the overfitting risk. Prefer a pretrained CNN+LSTM or a pretrained I3D as the feature extractor.

18. **What is the input shape for a ConvLSTM classifier, and why `unsqueeze(0)` at inference?**
    The input is (B, T, C, H, W), e.g., (4, 30, 3, 112, 112). A single video stacks to (T, C, H, W), and `unsqueeze(0)` adds the batch dimension to make (1, T, C, H, W).

---

## Common traps

- **S→T is not ST.** CNN followed by LSTM is *not* spatio-temporal. ConvLSTM and 3D CNN *are*.
- **Sigmoid vs. tanh ranges.** Sigmoid is 0…1 (gates/weights); tanh is −1…1 (data normalisation). Do not swap them.
- **"Output modulate gate" is not an official term.** It is the teacher's own label for tanh(Cₜ). The official parts are the forget, input and output gates plus the input modulate (candidate) gate.
- **Forget gate = 1 means keep everything**, not forget everything. 0 means erase.
- **Hidden state ≠ prediction.** It still goes through the optional forward part (Linear/softmax) to become an output.
- **The cell state holds "more" information, not "complete" information.** Loss happens at the forget gate and at the input gate.
- **The RNN "variable length" advantage** means it *can* process such inputs, not that it does so well. Long sequences still lose early information.
- **There is only one RNN cell**, reused across time. The unfolded picture does not mean many different cells with different weights; weights are shared.
- **Gate values are input-dependent (adaptive)**, while the W and b that compute them are fixed after training. Don't say "the gate weights are learned constants".
- **ConvLSTM** replaces the **x·W and h·W matrix multiplications** with convolution. The f∘C, i∘g and o∘tanh(C) products remain **element-wise**.
- **Axis order differs.** ConvLSTM with `batch_first` uses (B, T, C, H, W). `nn.Conv3d` and the I3D slide use (B, C, T, H, W). Slide 59's text says (Batch, Frame, Width, Height, Channel). Permute before Conv3d.
- **A 3D conv filter is not the same as a 2D conv over all channels.** In 2D the filter depth equals the number of input channels and does not slide in depth. In 3D the filter also slides along depth/time.
- **3D conv output size uses the same formula in every dimension**, including time: 4×4×4 with a 3×3×3 kernel gives 2×2×2, not 2×2×4.
- **Per-frame transforms** mean random augmentation differs between frames of the same video, which the teacher says to watch out for.
- **3D CNN is not automatically better.** It has more parameters, overfits on small datasets (UCF-101, HMDB-51) and has few pretrained models.
- **1×1×1 convolution** sees neither space nor time. It only mixes channels.
- **The slides contain errors.** Slide 40's "Input Gate (Write)" text repeats the hidden-state definition; use slide 38's definition (it controls how much new data flows into the cell state). The 3D-conv worked example has arithmetic slips: the step slide shows −21 where the correct value is −20, the final grid's −37 should be −36, and its −22 should be −18. See the formulas section.
- **GRU** is only mentioned in passing in the lecture and is not on the slides. Don't expect detailed GRU equations from this chapter's material.
