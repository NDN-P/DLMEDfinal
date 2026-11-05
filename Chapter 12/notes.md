# Chapter 12 — Image Segmentation Using Autoencoder (Encoder–Decoder) Architectures

Segmentation answers **What + Where at pixel level**: an encoder (conv + pooling) extracts "what" but loses spatial "where", and a decoder (upsampling via interpolation / transposed conv / unpooling) plus skip connections rebuilds a same-size output map.
Covers autoencoder basics, the upsampling operators, dilated conv, U-Net / PSPNet / LinkNet / PAN / DeepLabV3(+), output-layer design (sigmoid/softmax/linear, #channels = #classes), label formats, and the `segmentation_models_pytorch` workflow (Dice/Jaccard loss, IoU metric).

> Scope note: the slides and lecture do **not** cover FCN, Mask R-CNN, or formulas for Dice/IoU/pixel accuracy. The formulas in section 2 are standard definitions added for completeness and are marked as such.

---

## Key concepts

### 1. Task types: Classification vs Detection vs Segmentation
| Task | Answers | Form of "Where" |
|---|---|---|
| Image classification (or image regression) | **What** (whole image) | none |
| Object detection | **What + Where** | smallest axis-aligned rectangle (bounding box) enclosing the object |
| Image segmentation | **What + Where** | shape that follows the object outline, i.e. a label **per pixel** |

- Teacher's framing: classification = **Image Prediction** (uses all info, FC layer → one answer per image); segmentation = **Pixel Prediction** (one answer per pixel). Detection = "what" of a rectangular area.
- Segmentation output can be **discrete (class per pixel)** or **continuous (regression per pixel)**, e.g. depth estimation.

### 2. Three kinds of segmentation (slides p.3–4)
| | Semantic | Instance | Panoptic |
|---|---|---|---|
| What is labelled | **Every pixel** (foreground + background) | **Foreground objects only** (objects of interest) | **Every pixel** |
| Same-class objects | Same label: car1 and car2 are both "car" (cannot separate objects) | Separate mask per object (car1 orange, car2 pink, car3 green) | Things separated, stuff not |
| Pixels left unlabelled? | No | Yes (background not segmented) | No |
| Typical use (teacher) | Image understanding, **calculating area / size / surface** | **Tracking** (e.g. a person moving) | Full scene understanding |

- **Panoptic = Semantic + Instance.**
  - **Stuff** = things without clear boundaries (sky, road, water) → segmented semantically (all sky pixels = class sky).
  - **Thing** = countable objects (person, car, chair) → segmented as instances (each gets its own mask).
- Note: by concept semantic segmentation labels the background too, but in practice you can run semantic segmentation that ignores the background.

### 3. Autoencoder fundamentals (slides p.7–18)
- **Definition:** a feedforward NN whose **input and output have the same size** (at least the same width × height). It compresses input into a lower-dimensional **code**, then reconstructs the output from that code.
- **3 parts:**
  - **Encoder**: compresses the data and creates the code. For images, this is feature extraction (the CNN part before flatten).
  - **Code**: compressed/"summary" representation = **latent-space representation** ("latent").
  - **Decoder**: reconstructs the input from the code (decompress / "unzip" / reconstruction).
- Goal: minimise ‖x − x̂‖² → 0. x and x̂ are the same size but not identical (lossy). The code is smaller than x and x̂. Weights w (encoder) and w′ (decoder) are learned by training (backpropagation, as with ANN/CNN).
- **3 key properties:**
  1. **Data-specific**: only compresses data similar to the training data (an AE trained on handwritten digits cannot compress landscapes), unlike zip/winrar/7zip.
  2. **Lossy**: output is close to the input but not identical; removed information is gone permanently.
  3. **Unsupervised**: for compression only the input is needed (input = target). For other tasks such as segmentation, **labels are still required**.
- **Deep autoencoder** (Hinton 2006): many layers, e.g. 784 → 1000 → 500 → 250 → **2 (code)** → 250 → 500 → 1000 → 784. Encoder and decoder are **usually symmetric, but they don't have to be**.
- With the same number of components, a deep AE reconstructs better than **PCA** (given good initial weights). Code size is comparable to the number of principal components in PCA.
- Why the encoder shrinks and the decoder grows layer by layer: feature learning, dimensionality reduction (also removes noise), gradual reconstruction, keeping spatial relationships (conv in the encoder, deconv/upsampling in the decoder), progressive learning.
- **4 hyperparameters**: Code size, Number of layers, Number of nodes per layer, Loss function (MSE or binary cross-entropy; use **cross-entropy if inputs are in [0,1]**).
- **Caution:** **Code size is the most important hyperparameter.** If it is too large, the AE just **copies input to output** without learning a useful code, which leads to **overfitting**. That is why the code is kept smaller than the input, giving a **bow-tie / hourglass** shape. Purely random input with no internal correlation cannot be reconstructed well.
- **Convolutional Autoencoder**: uses convolution instead of FC layers. Introduced by **Zeiler et al. 2013**. Uses: image/video compression, denoising, synthesis. Why conv: **flattening destroys spatial information** (neighbour context); convolution processes a pixel together with its neighbours.
- **Denoising Autoencoder** (Vincent et al. 2008): train with corrupted input and clean output. The code becomes robust to noise.
- Teacher's key idea: an encoder–decoder fits any task where **input is an image and output is an image of the same W×H with any number of channels**. Segmentation is "generate an image that matches the ground-truth mask". Depth estimation is the same thing with continuous values.

### 4. Encoder vs Decoder behaviour
| | Encoder (contraction path, downsampling, "pooling") | Decoder (expansion path, upsampling, "unpooling") |
|---|---|---|
| W × H | decreases | increases |
| Channels (depth) | increase | decrease |
| Learns | **WHAT** (rich information) | **WHERE** (recovers spatial info) |
| Loses | **WHERE** (spatial information lost) | — |
| Operators | Convolution, **Dilated convolution**, max pooling | **Interpolation, Transposed conv (Deconvolution), Unpooling** |

- Deep, small feature maps are **rich in information but poor in spatial detail**. Shallow, large maps are **rich in spatial detail but low in information**.

### 5. Upsampling operators (decoder) — slides p.21–48
1. **Interpolation**: nearest-neighbour, bilinear (no learnable weights).
   - Nearest neighbour gives discrete, copied values. Bilinear gives smoothly scaled values. The bilinear calculation was **not taught**, only shown.
2. **Transposed convolution** (= **Deconvolution**, same thing). The most popular approach.
   - Each input value is multiplied by the whole kernel. Each result is placed in the output shifted by **stride** positions, and **overlapping regions are summed**.
   - Example: input 1×1 with 3×3 kernel gives a 3×3 output. Input 4×4, k=3, s=2, p=0 gives **9×9**.
   - **The kernel is not flipped/rotated** (slide note "ไม่หมุนตัวกรอง").
   - Larger stride gives a larger output (stride 3 with k=3 means no overlap). Larger kernel gives a larger output. **Larger padding gives a smaller output** (see the trap below).
   - Used in AE decoders and in GANs.
3. **Unpooling (Max unpooling)**: reverses max pooling. During max pooling you must **store the pooling indices (mask / location)**. During unpooling, each value goes back to its original position and the **rest are filled with 0**. A 3×3 or 5×5 **convolution follows** to fill the gaps / smooth the result. Rarely used today.
- In practice a network uses **one, at most two** of these, never all three.

### 6. Padding, Stride, Dilation
- **Padding**: adds (usually zero) values at the border so a normal conv doesn't shrink the output. Zeros don't affect the dot product.
- **Stride**: step size. In normal conv, a higher stride means more compression. Usually stride = 1. In transposed conv, stride 2 takes a 3×3 input to a 5×5 output (slide p.51).
- **Dilated (atrous) convolution**: kernel taps are spaced apart, which **enlarges the receptive field without more parameters or compute**. This is **not** the morphological dilation from the image-processing chapter.
  - **Dilation = 1 means a normal conv (taps adjacent). Dilation = 2 means a gap of 1 pixel. Dilation = 3 means a gap of 2.** The teacher first said 0 and then corrected himself to this. The slide figure (p.53) uses the same labels: normal 3×3 = dilation 1, and "3×3, d=2" covers a 5×5 area.
  - Used mainly on the **encoder** side.

### 7. U-Net (Ronneberger et al., **2015**) — designed for **biomedical/medical image segmentation**
- Named for its **U shape**. Components: **Encoder (contraction path)**, **Decoder (expansion path, transposed conv / "up-conv")**, **Skip connections** (grey arrows, "copy and crop").
- Blocks in the slide diagram (p.54–55): blue = Conv 3×3 + ReLU. Red = **MaxPool 2×2** (W, H halve: 256² → 128² → 64² → 32²). Grey = **Copy**: encoder features are **concatenated** with the decoder features at the **same scale** (same W×H). Green = **Up-conv 2×2** (transposed conv). Yellow = Dropout, then conv 3×3 + ReLU. Light blue = **Conv 1×1, sigmoid**.
- Channel numbers in the diagram: input 1 channel (256²) → 112 → 224 → 448 → 448 at the bottleneck (32²). Concat after up-conv: 448 + 448 = **896** at 64². Output is **1 channel** (sigmoid binary mask). The diagram's example is not medical: a lunar elevation map ("Input DEM") goes in and a crater-rim "Predicted Mask" comes out. The teacher's "112, 224, 448" [00:56:31, 00:57:52] matches these numbers.
- **Why skip connections:** the encoder loses spatial info, so reconstruction could be distorted. Skip connections carry **spatial information** from encoder to decoder at the same scale (like ResNet's skip). At every upsampling step, spatial info and "what" info are merged first, then processed, then upsampled.
- **Final layer = 1×1 convolution** (pixel-wise prediction) + activation:

| Task | Meaning | # of 1×1 filters (output channels) | Activation |
|---|---|---|---|
| Binary (fg/bg) – option 1 | 1 pixel → 1 of 2 classes | **1** | **Sigmoid** (threshold 0.5) |
| Binary – option 2 | same | **2** | **Softmax** (pick the larger score) |
| Multiclass | 1 pixel → exactly 1 of N classes | **N** (e.g. 5) | **Softmax** (`softmax2d` in smp) |
| Multilabel | 1 pixel → can be several classes (e.g. finger overlapping a pencil) | **N** (e.g. 4) | **Sigmoid** (each channel > 0.5 → that class) |
| Regression (e.g. depth estimation) | continuous value per pixel | **# of values to estimate** (1 for depth only) | **Linear / none** |

- For binary tasks people usually use sigmoid with 1 channel rather than softmax.

### 8. Label (mask) formats (slides p.60–62)
- **Class label**: 1 channel, same H×W as the image. Each pixel holds a class index (0, 1, 2, …), so one value per pixel. Works for **multiclass and regression**.
- **One-hot label**: **#channels = #classes**. A pixel has 1 in the channel of its class. Works for **multiclass or multilabel**. **Multilabel must use one-hot**, because a pixel can be 1 in several channels.
- Example (3 classes: road = 0, car = 1, person = 2): the class-label map row `0 1 2` becomes one-hot channels where channel 0 has 1 where the label is 0, and so on.
- Which to use depends on (1) the task and (2) what the library supports.
- In smp the binary mask contains 0/1, has the **same size as the image**, and image and mask must use the **same transform**, which needs `torchvision.transforms.v2`.

### 9. Other architectures
| Architecture | Year / authors | Key idea |
|---|---|---|
| **U-Net** | 2015, Ronneberger | Encoder–decoder + skip connections by **concatenation**; medical images |
| **PSPNet** (Pyramid Scene Parsing) | 2017, Zhao et al. | CNN backbone, then a **Pyramid Pooling Module**: pool the feature map into 1×1 (global info), 2×2, 3×3, … (local info). Then 1×1 conv on each, upsample to the same size, **concatenate** with the backbone features, and 1×1 conv to predict. Multi-scale handles objects of different sizes. The slide also lists **Dice loss** (helps with **imbalanced data**). |
| **LinkNet** | 2017, Chaurasia | Basically U-Net, but skip connections use **element-wise addition** (needs the same channels and W×H; output keeps the same size) instead of concatenation. Light and fast, for resource-limited systems. 5 parts: Initial block, Encoder block, Decoder block (transposed conv), Final block (1×1 conv), Skip connection. |
| **PAN** (Pyramid Attention Network) | 2018, Li et al. | Encoder + **FPA** (Feature Pyramid Attention: multi-scale features using **different filter sizes** instead of resizing the image, multiplied element-wise with the backbone features, plus a global-pooling branch) + **GAU** (Global Attention Upsample: high-level features → **global average pooling** → 1×1 conv produces **channel weights**. These weight the low-level features, which are then added to the high-level features. It acts like a **weighted skip connection that weights channels, not locations**). Uses BN after each conv. |
| **DeepLabV3** | 2017, Chen et al. | Addresses the encoder's spatial-info loss: downsample only to a point, then use **atrous/dilated conv** instead of shrinking more. Uses **ASPP** (Atrous Spatial Pyramid Pooling: several dilation rates, run in **parallel** and concatenated, or in **sequence**). It predicts with 1×1 conv **first, then upsamples** to image size. Weakness: the **decoder is tiny** (a single upsample step, very asymmetric). |
| **DeepLabV3+** | 2018, Chen et al. | DeepLabV3 + a **Decoder module**: upsample, concatenate with backbone features, gradually enlarge, then classify. A true encoder–decoder. |

- The encoder **backbone is swappable** (ResNet, MobileNet, DenseNet, VGG, GoogLeNet, EfficientNet, …).
- smp architectures: Unet, Unet++, MAnet, FPN, DeepLabV3, DeepLabV3+.

### 10. `segmentation_models_pytorch` (smp) workflow
- Library by **Pavel Iakubovskii**, built on PyTorch: a high-level API with pretrained encoders, metrics, and losses.
- Install: `pip install segmentation-models-pytorch` (PyPI), or `pip install git+https://github.com/qubvel/segmentation_models.pytorch` (GitHub, sometimes a newer version).
- **5 steps:** (1) define the Dataset (image + mask) → (2) define the network (`smp.FPN` / `smp.Unet`, `encoder_name='resnet50'`, `encoder_weights='imagenet'`, `classes=len(CLASSES)`, `activation`) → (3) loss + optimizer (**DiceLoss** or **JaccardLoss**, both support binary, multiclass, and multilabel; Adam, lr = 1e-4) → (4) train (`TrainEpoch` / `ValidEpoch` only **configure**; training actually happens on `.run(loader)`; 40 epochs; lr drops to 1e-5 at epoch 25; `verbose=True` shows a progress bar) → (5) test/evaluate (`ValidEpoch` on the test loader, `best_model.predict(image)`).
- Metrics available: **F1, IoU (threshold = 0.5), Accuracy, Precision, Recall**.
- `aux_params` (pooling 'avg'/'max', dropout, activation, classes) adds a classification head: pooling flattens to 1-D, then an FC layer.
- Saving: either at the last epoch or keep the best epoch (lowest loss / best KPI such as IoU). The teacher saves both.
- Example dataset: **Retina Blood Vessel** (Kaggle), a binary task (`CLASSES=['blood_vessel']`, sigmoid).

---

## Formulas & numbers to remember

### Transposed convolution output size (PyTorch `ConvTranspose2d`, dilation 1)
```
H_out = (H_in − 1)·s − 2p + k
```
Slide demo (input 4×4 all ones, kernel 3×3 all ones):

| padding | stride | output |
|---|---|---|
| 0 | 1 | 6×6 |
| 0 | 2 | **9×9** (the worked example in class) |
| 0 | 3 | 12×12 (no overlap) |
| 1 | 1 | 4×4 |
| 2 | 1 | 2×2 |
| 1 | 2 | 7×7 |
| 2 | 2 | 5×5 (the slide's code: `padding=2, stride=2`) |

- Other slide examples: 1×1 input with a 3×3 kernel gives 3×3. With stride 2, a 3×3 input gives 5×5 (that corresponds to k=3, p=1).
- For comparison, normal conv: `H_out = ⌊(H_in + 2p − k)/s⌋ + 1`.
- Dilated kernel effective size: `k + (k−1)(d−1)` (3×3 with d=2 covers 5×5 but still uses 9 weights). Cost: 3×3 = 9 multiplies, 5×5 = 25, 7×7 = 49. That is why DeepLab uses dilated 3×3 instead of a large kernel.
- Max pool stride 2 halves W and H (to 50%).

### Nearest-neighbour interpolation (2×2 → 5×5, slides p.22–23)
1. Source coordinates spread evenly over [0,1] with step 1/(n−1) = 1/4, giving [0, 0.25, 0.5, 0.75, 1].
2. `nearest = ⌊src + 0.5⌋` gives [0, 0, 1, 1, 1].
3. Do this for both rows and columns. Output rows/cols 0–1 come from input row/col 0, and rows/cols 2–4 come from input row/col 1.

With X = [[10, 20], [30, 40]] (the teacher's example):
```
10 10 20 20 20
10 10 20 20 20
30 30 40 40 40
30 30 40 40 40
30 30 40 40 40
```
If H and W differ (e.g. 4 and 5), compute the steps separately for each axis.

### Bilinear interpolation (same X, 2×2 → 5×5, slide p.24, result only)
```
10   12.5 15   17.5 20
15   17.5 20   22.5 25
20   22.5 25   27.5 30
25   27.5 30   32.5 35
30   32.5 35   37.5 40
```
Values change smoothly (linear in both directions). Nearest neighbour copies discrete values. The calculation steps were not taught.

### Transposed conv worked example (slides p.29–40)
X = [[1,1,1,1],[1,1,1,1],[3,1,2,2],[3,1,2,2]], K = 3×3 all ones, p = 0, s = 2.
- Step 1: multiply each input value by the kernel, giving sixteen 3×3 blocks (e.g. X[3,1] = 3 gives a block of all 3s).
- Step 2: place the blocks 2 apart (the stride) and **add the values where they overlap**.
- Output is **9×9** = (4−1)·2 + 3:
```
1 1 2 1 2 1 2 1 1
1 1 2 1 2 1 2 1 1
2 2 4 2 4 2 4 2 2
1 1 2 1 2 1 2 1 1
4 4 6 2 5 3 6 3 3
3 3 4 1 3 2 4 2 2
6 6 8 2 6 4 8 4 4
3 3 4 1 3 2 4 2 2
3 3 4 1 3 2 4 2 2
```
- Positions where 2 blocks overlap are doubled, and where 4 overlap they are quadrupled. With an all-ones input (p.44), s = 2 gives values 1/2/4. With s = 1 the output is 6×6 with values up to 9.

### Overlap metrics and losses (standard definitions; the slides only name them)
Let A = predicted foreground pixels and B = ground-truth pixels.
- **IoU / Jaccard** = |A∩B| / |A∪B| = TP / (TP + FP + FN)
- **Dice (= F1 for masks)** = 2|A∩B| / (|A| + |B|) = 2TP / (2TP + FP + FN)
- Relationship: Dice = 2·IoU / (1 + IoU), so Dice ≥ IoU.
- **Pixel accuracy** = (TP + TN) / all pixels
- **Dice loss** = 1 − Dice. **Jaccard loss** = 1 − IoU. Soft versions use predicted probabilities.
- In smp, `IoU(threshold=0.5)` binarises the predicted probabilities at 0.5 before computing.

**Worked example (practice numbers, not from the slides):** 10×10 image = 100 px, GT = 20 px, Pred = 25 px, overlap = 15 px.
- TP = 15, FP = 10, FN = 5, TN = 100 − 30 = 70
- IoU = 15 / (20 + 25 − 15) = 15/30 = **0.50**
- Dice = 2·15 / (20 + 25) = 30/45 = **0.667** (check: 2·0.5/1.5 = 0.667)
- Pixel accuracy = (15 + 70)/100 = **0.85**. It looks good only because the background dominates.

### Dates / names
Hinton 2006 (deep AE, 784-1000-500-250-2) · Vincent 2008 (denoising AE) · Zeiler 2013 (conv AE) · Ronneberger 2015 (U-Net) · Zhao 2017 (PSPNet) · Chaurasia 2017 (LinkNet) · Chen 2017 (DeepLabV3) · Li 2018 (PAN) · Chen 2018 (DeepLabV3+).

---

## What the teacher emphasised

- **[00:00:06] / [00:09:06]** You must be able to explain how classification, detection, and segmentation differ: classification = What; detection and segmentation = What + Where. Detection's Where is the smallest bounding rectangle; segmentation's Where follows the object shape. "สิ่งที่เด็ก ๆ จะต้องตอบได้" — "what you must be able to answer".
- **[00:03:50]–[00:05:22]** Instance segmentation only covers the **foreground**. Semantic covers foreground and background but doesn't group pixels into objects.
- **[00:08:02]** In practice semantic segmentation *can* ignore the background, but "by concept" it labels every pixel.
- **[00:08:28]** Instance segmentation suits **tracking**. Semantic suits image understanding and **area/size calculation**.
- **[00:11:13]** First decide whether you need only What, or What + Where. Then decide whether Where should be a bounding box or an object shape.
- **[00:14:05]–[00:16:07]** The 3 AE properties: data-specific (unlike WinZip/WinRAR), lossy (the loss is often undetectable to the eye), unsupervised.
- **[00:18:16]** The "unsupervised" statement "อาจจะไม่จริงในบริบทของวิชานี้" ("may not be true in this course"): segmentation training needs ground-truth masks.
- **[00:18:42]** "สิ่งที่เด็กๆต้องเข้าใจ" ("what you need to understand"): what the 3 parts (encoder, code, decoder) are and what each does. Encoder = feature extraction (the CNN part before flatten). The feature map before flatten = code / latent.
- **[00:23:14]** x and x̂ have **the same size but don't look identical** (data loss). The code is usually smaller.
- **[00:26:11]** Encoder and decoder are usually symmetric, but symmetry is "ไม่จำเป็น" ("not necessary"). DeepLabV3 is the example [01:48:47].
- **[00:28:56]** "Keyword สำคัญ" ("important keyword"): the encoder is the feature-extraction part you already know. The decoder has a special operator, transposed conv / deconvolution, that enlarges small features.
- **[00:29:39]–[00:30:23]** Encoder: rich information, **lost spatial information**. "Encode ลด Width, Height เพิ่ม Channel" ("encode shrinks width and height, adds channels"). Decode does the opposite.
- **[00:31:27]–[00:34:20]** The big insight: an encoder–decoder is for **image in, image out (same W×H, any number of channels)**. Segmentation is just "generate this target image from the input". Depth estimation is the same idea.
- **[00:35:42]** Why conv instead of FC: flattening loses spatial and neighbour information.
- **[00:36:57]** 3 decoder upsampling approaches. **Deconvolution = transposed convolution** ("ให้รู้ว่ามันคือสิ่งเดียวกัน", "know that they are the same thing"), and it is the most popular.
- **[00:39:52]–[00:41:26]** Nearest-neighbour steps: compute the spacing first (n−1 rule, 1/4 = 0.25), add 0.5, floor, then map indices on both axes.
- **[00:42:35]** Bilinear: shown only, the calculation was not taught.
- **[00:44:09]–[00:46:17]** Transposed conv walkthrough: multiply each input by the kernel, shift by stride, **sum the overlaps**. 4×4 input with k=3, s=2 gives **9×9**.
- **[00:46:25]–[00:47:45]** Size depends on **stride, padding, kernel size**. A larger stride or kernel gives a bigger output. The kernel is not flipped.
- **[00:49:02]** "ยิ่ง Padding เพิ่ม ข้อมูลก็จะยิ่งหดลง" ("the more padding, the smaller the output"): padding is not what you use to enlarge.
- **[00:50:44]–[00:51:55]** Unpooling requires saving the **location/mask** during max pooling. Fill the other positions with 0, then smooth with a 3×3 or 5×5 conv.
- **[00:52:15]** In practice use one, at most two, upsampling methods, never all three.
- **[00:52:38]** Dilated conv ≠ the morphological dilation from local image processing. **[00:55:17] correction:** adjacent taps = dilation 1, gap of 1 = dilation 2. Dilated conv goes on the **encoder** side.
- **[00:55:40]** U-Net is the most popular, foundational segmentation architecture, built for **clinical/medical images**. The name comes from its U shape.
- **[00:57:07]** "หาที่จด" ("get something to write on"): encoder = feature extraction that reduces W×H and increases channels. Deep layers are rich in info but lose spatial data. Upper layers are rich in spatial data but low in info.
- **[00:59:17]–[01:01:38]** **The drawback of AE architectures is that the encoder loses spatial info**, so reconstruction can be distorted. **Skip connections** bring spatial info from the encoder to the decoder **at the same scale** (256↔256, 128↔128, …).
- **[01:02:14]–[01:03:21]** The skip connection doesn't "know" it is sending spatial info. Shallow features simply contain mostly spatial info, so it survives the merge.
- **[01:04:12]** Grey = copy + **concat**. Green = up-conv = transposed conv.
- **[01:05:30]–[01:13:16] Bonus quiz, "ใครพิมพ์ตอบมาได้ ผมบวกให้ 0.5 คะแนน"** ("whoever types the answer gets +0.5 points"; see below).
- **[01:13:16]–[01:14:02]** After the 1×1 conv, the activation sets the task type: sigmoid (binary/multilabel), softmax (multiclass), linear (regression), "ใช้มุกเดิมกับตอนที่เราเรียน CNN" ("same trick as when we studied CNNs").
- **[01:14:14]** Multilabel example: where a finger covers a pencil, the pixel is both finger and pencil.
- **[01:19:00]–[01:19:07]** "**จำนวน Channel ตรงนี้ เท่ากับจำนวนคลาส จดไปครับ**" ("the number of channels here equals the number of classes, write it down"), said twice.
- **[01:20:04]** "ประโยคเมื่อกี้เหมือนสั้น ๆ แต่สำคัญ" ("that sentence was short but important"): multiclass = 1 pixel, 1 class, many choices, softmax. Multilabel = 1 pixel, possibly many classes, sigmoid, channels = classes, threshold 0.5 per channel.
- **[01:21:05]–[01:24:11] Second bonus question (+0.5, for the first correct answer):** how do you modify the architecture for **regression** (per-pixel depth estimation)? At A (activation): **linear or no activation**. At B (number of output channels / 1×1 filters): **number of values to estimate** (1 for depth only).
- **[01:24:16]–[01:27:11]** Label formats: class label (1 channel) for regression and multiclass. One-hot (channels = classes) for multiclass or multilabel. **Multilabel must be one-hot** because pixels overlap. Which one depends on the task and the library.
- **[01:28:17]–[01:29:05]** Multi-task (e.g. wound tissue type per pixel plus wound depth) can share an encoder with parallel 1×1 heads. But "อย่าทำเลย แยก Architecture ไปเถอะ ง่ายกว่า" ("don't do it, use separate architectures, it's easier").
- **[01:29:20]–[01:30:23]** Stacked encoder–decoders need huge GPU RAM (an RTX 3090 was not enough for 2 stacks). **Segmentation uses the most memory and is the slowest to predict.** Memory grows with input size: classification < detection < segmentation.
- **[01:30:48]–[01:31:01]** "**การทำ Classification มันคือการ Predict จากภาพ แต่การ Segmentation มันคือการ predict ในระดับพิกเซล — ประโยคนี้จำดีๆ**" ("classification predicts from the image, segmentation predicts at pixel level; remember this sentence well").
- **[01:33:03]** AEs can also do denoising and sharp reconstruction, but they work well only on data specific to the training data.
- **[01:35:37]–[01:37:33]** PSPNet: 1×1 pooling = global info, finer grids = local info. Upsample, concat, 1×1 conv to predict.
- **[01:38:05]–[01:38:31]** **U-Net = concatenation skip. LinkNet = addition skip** (same channels and W×H required, size unchanged).
- **[01:41:33] "จดไป"** ("write it down"): PAN's GAU **weights which feature channels are important** (channel weighting, not spatial location), then adds the result to the high-level features. It acts like a weighted skip connection.
- **[01:43:47]–[01:44:24]** PAN's FPA builds the pyramid by changing **filter size**, not image size. It multiplies element-wise where PSPNet concatenates.
- **[01:45:14]–[01:46:59]** DeepLab fixes spatial loss with **dilated conv instead of further downsampling**. Why not 7×7: 9 vs 25 vs 49 multiplications. ASPP runs in parallel (concat, then 1×1) or in sequence. It predicts first and then upsamples, the opposite of the usual order.
- **[01:48:26]–[01:49:25]** DeepLabV3's weakness is a single-step decoder. V3+ adds a gradual decoder that uses backbone features.
- **[01:54:44]** The teacher admitted the multiclass code slide uses the wrong activation. Multiclass should be softmax(2d).
- **[01:55:14]–[01:55:50]** Training usually uses **Dice loss or Jaccard loss** with Adam. Metrics are IoU or accuracy.
- **[01:56:39]–[01:56:58]** `TrainEpoch` only sets the parameters. Training happens on `.run()`.
- **[01:57:17]–[01:57:45]** Save at the last epoch or the best epoch. The teacher saves both.

### In-class quiz [01:06:24]–[01:13:16], worth +0.5
**Question:** "ทำไมเขาต้องใช้ 1 คูณ 1 Convolution ในการ classify" ("why do they use 1×1 convolution to classify"). In U-Net, why is the **final prediction layer a 1×1 convolution**?

**Correct answer (teacher's):** because segmentation is **pixel prediction**. To decide which class a pixel belongs to, the network uses **that pixel's features across all channels** (each channel = one feature) at that position. A 1×1 conv acts like a **fully connected layer applied independently to each pixel**, so the output keeps the image's W×H.

A classification CNN instead uses an FC layer over **all** spatial positions to predict one label for the **whole image** (image prediction).

Hints the teacher gave: "**ไม่เกี่ยวกับ Softmax**" ("it's not about softmax") [01:09:53]. "Because the output is an image?" was "เกือบถูก" ("almost right") [01:10:46]. "Extract important info to weight that pixel" was "ใกล้เคียง แต่ยังไม่ใช่" ("close, but not it") [01:11:17]. The sigmoid/softmax choice is a separate issue: it sets the task type (binary/multiclass/multilabel/regression), not why the layer is 1×1.

---

## Likely exam questions

1. **Compare classification, detection and segmentation.** Classification gives What for the whole image. Detection gives What + Where as a bounding box. Segmentation gives What + Where as a per-pixel shape (pixel-level prediction).

2. **Semantic vs instance vs panoptic?** Semantic: every pixel (including background) gets a class, but same-class objects are not separated. Instance: only foreground objects, each with its own mask, background unlabelled. Panoptic: every pixel labelled; stuff (sky, road) handled semantically, things (people, cars) as instances.

3. **Medical scenario: measure the total tumour or lesion area in an MRI slice.** Semantic (binary) segmentation, since you need area. Counting and following individual cells/nuclei over time is instance segmentation (separate objects, tracking). Labelling every tissue type plus separating each individual lesion is panoptic.

4. **What are the 3 parts of an autoencoder and their roles?** Encoder compresses and extracts features. Code is the compressed latent-space representation. Decoder reconstructs the output from the code. Input and output have the same size.

5. **3 properties of autoencoders?** Data-specific, lossy, unsupervised (for compression only; segmentation still needs masks).

6. **Why must the code size be smaller than the input?** If the code is too large, the AE learns to copy input to output, which means overfitting and no useful features. The bottleneck (hourglass shape) forces it to learn important features. Code size is the most important hyperparameter.

7. **Why do segmentation networks need skip connections?** The encoder's downsampling loses spatial ("where") information. Skip connections pass high-resolution spatial features from encoder to decoder at the same scale, so reconstruction and boundaries are accurate.

8. **U-Net vs LinkNet skip connections?** U-Net concatenates (channels add up). LinkNet adds element-wise (needs the same shape, output size unchanged, lighter and faster).

9. **List 3 upsampling methods and describe unpooling.** Interpolation (nearest, bilinear), transposed conv, max unpooling. Unpooling needs the max-pool indices saved, puts values back in their original positions, fills zeros, then a conv smooths the result.

10. **Compute the output size of a transposed conv: input 4×4, k=3, s=2, p=0.** (4−1)·2 + 3 = **9×9**. With p=1: 7×7. With s=1, p=0: 6×6.

11. **Nearest-neighbour upsample [[10,20],[30,40]] to 5×5.** Index map [0,0,1,1,1] on both axes gives two rows of `10 10 20 20 20` and three rows of `30 30 40 40 40`.

12. **Why does U-Net use a 1×1 conv at the end?** Per-pixel prediction from all channel features at that pixel: an FC layer applied per pixel that keeps W×H. (This was the bonus quiz.)

13. **Design the output layer:**
    - (a) retinal vessel vs background: 1 channel + sigmoid (or 2 + softmax);
    - (b) 5 tissue classes, one per pixel: 5 channels + softmax;
    - (c) 4 overlapping structures: 4 channels + sigmoid, labels must be one-hot;
    - (d) per-pixel depth or temperature map: 1 channel (or # of quantities) + linear/no activation, class-label-style target.

14. **What is dilated (atrous) conv and why does DeepLab use it?** Kernel taps are spaced (dilation 2 means a gap of 1). It enlarges the receptive field with no extra parameters or compute, so DeepLab avoids further downsampling and keeps spatial resolution. ASPP combines several dilation rates for multi-scale context. V3+ adds a proper decoder because V3's decoder is just one upsample.

15. **Dice/IoU calculation: GT = 20 px, Pred = 25 px, overlap = 15 px (100 px image).** IoU = 15/30 = 0.50. Dice = 30/45 ≈ 0.67. Pixel accuracy = 0.85, which is misleading because the background dominates. Dice/Jaccard loss (1 − score) is preferred for imbalanced masks.

16. **PSPNet vs PAN?** PSPNet uses a Pyramid Pooling Module (pool to 1×1, 2×2, 3×3, … for global and local context), upsamples, and concatenates. PAN uses FPA (multi-scale by filter size, multiplied in) and GAU (high-level features produce channel weights for low-level features, then added): an attention-weighted skip connection.

---

## Common traps

- **Transposed conv ≠ inverse of convolution** in values. It only reverses the *shape* change. It is the same thing as "deconvolution" in this course, and the kernel is **not flipped**.
- **Padding in transposed conv shrinks the output** (−2p). I checked the slide image: p.41 really does say "ยิ่งค่า Padding มากขึ้น ขนาดของภาพผลลัพธ์จะยิ่งขยายใหญ่ขึ้น" ("the more padding, the bigger the output"). The slide deck's own PyTorch heatmaps contradict it. p.46 (s=1): p=1 gives 4×4, p=2 gives 2×2 (versus 6×6 at p=0). p.47 (s=2): p=1 gives 7×7, p=2 gives 5×5 (versus 9×9 at p=0). The teacher also said padding shrinks it [00:49:02]. If an exam question asks what padding does to output size in transposed conv, answer that it makes the output smaller. The stride bullet on p.41 is correct: a larger stride gives a larger output.
- **Dilation numbering:** a normal conv is **dilation = 1**, not 0 (teacher's correction at [00:55:17]).
- **Dilated convolution ≠ morphological dilation.**
- **Number of output channels = number of classes**. The exception is binary + sigmoid, which uses **1 channel**.
- **Multiclass ≠ multilabel.** Multiclass: one class per pixel, softmax. Multilabel: several classes per pixel, sigmoid, one-hot labels required. Some code slides (p.87–90) still pass `activation='sigmoid'` in `smp.FPN(...)` even when `ACTIVATION='softmax2d'`. The teacher admitted the mistake; multiclass should use softmax(2d).
- **Regression segmentation:** linear/no activation. Don't use sigmoid or softmax.
- **"Autoencoders are unsupervised"** is true for compression/reconstruction only. Segmentation needs ground-truth masks.
- **Encoder/decoder don't have to be symmetric** (DeepLabV3 has a one-step decoder).
- **Instance segmentation does not label background pixels.** Semantic does, by concept.
- **Max pooling loses the location of the max.** Unpooling only works if the indices were saved during pooling.
- **Pixel accuracy is inflated by large backgrounds.** Use IoU/Dice (Dice ≥ IoU for the same prediction).
- **Segmentation is the most memory-hungry task** (memory grows with input size). Stacking U-Nets can exceed 24 GB VRAM.
- In smp, `TrainEpoch(...)` does not train anything until `.run(loader)` is called.
- Slide p.81 code bug: `__len__` returns `len(self.img_dir)` (the length of the path string). It should be `len(self.img_id)`.
