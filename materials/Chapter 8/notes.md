# Chapter 8 — Pre-trained CNN Architectures (Transfer Learning) and Data Augmentation

Deep learning needs huge training sets. The chapter covers two fixes: **Data Augmentation** (synthesize more and more varied images: distortion, occlusion, multi-image mixing, and dynamic/progressive schedules) and **Transfer Learning** (reuse an ImageNet-pretrained backbone).
It then walks through the operators that make each well-known backbone special: AlexNet, VGG, SqueezeNet, GoogLeNet/Inception, ResNet, DenseNet and MobileNetV3, with a lot of attention on the **1×1 convolution**.

> Source: 89-page slide deck (ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์, IT KMITL) + ~2 h 8 min lecture recording. Figure-only slides (56 ImageNet error chart, 57 accuracy-vs-speed chart, 69 AlexNet/VGG/GoogLeNet layer diagrams, 15 MixUp formula, 72 ResNet-18 diagram, 81 growth rate) were read from the rendered slide images.

---

## Key concepts

### 1. Why we need these techniques
- DL works well for recognition, detection, segmentation and denoising, but its main limitation is that it **needs a huge number of training images**. Real production models use up to **1,000,000** training images (slide); the teacher said "อย่างน้อยก็ 10 ล้าน" depending on the problem.
- There are two fixes. **Transfer Learning** reuses knowledge. **Data Augmentation** synthesizes new samples from the ones you have.
- Teacher's analogy for transfer learning: don't train a doctor from kindergarten. Take an "ม.6 graduate" (the **base model** trained on ImageNet, good at general things) and train it 4–6 more years on the specialty (your dataset). That needs less data, less time and fewer resources.

### 2. Data Augmentation: purpose
- Slide: even with transfer learning (LeNet, AlexNet, ResNet, VGGNet, Inception), a dataset with **fewer than 1,000 images** is still not enough. Example: grow 500 images to 1,000 with 2-D geometric transforms (shift, rotate, scale, flip LR/UD, random crop).
- Augmentation forces the model to find **new features** instead of relying on only a few.
- The teacher's two uses ("ใช้สองประโยชน์"):
  1. **Increase the number of samples**: small data or heavy **class imbalance**. His example: a dam-motor tilt fault seen only once in 10 years.
  2. **Increase variety** even when you already have many samples, so the model is robust to real-world input ("ทนมือ ทนตีน User").

### 3. Augmentation families
The slides list Distortion, Image Occlusion and YOLOv4 techniques. The teacher groups them into **3 groups**:

| Group | What it does | Members | Data stays "on its own frame"? |
|---|---|---|---|
| 1. **Distortion** | *Adjusts* shape or colour; nothing is lost | **Photometric**: contrast, sharpness, blurring, brightness, colour/saturation, noise · **Geometric**: cropping, rotation, scaling, translation, flip | Yes |
| 2. **Image Occlusion** | *Hides/erases* part of the image | Random Erase, Cutout, Hide-and-Seek, GridMask | Yes (information partly removed) |
| 3. "Group 3" (teacher left it unnamed) | *Builds a new image from >1 source image* | MixUp, CutMix, Mosaic | No, it mixes 2–4 sources |

- **Photometric distortion** changes pixel values. **Geometric distortion** changes shape.
- Zoom-in = crop then resize up, which is fine. **Zoom-out** in practice fills the border with **black** because there is no real content outside the original image.
- Why occlusion works (Grad-CAM dog example): the model looked only at the dog's **head**, so a headless photo was misclassified ("ลักไก่", similar to overfitting). Randomly hiding a different region each epoch forces it to use other parts (for example the legs). This reduces overfitting and adds regularization. Each epoch hides a different region, so it acts like a "new exam question" each time and the model can't memorize.

#### Occlusion methods
| Method | Definition | What is random |
|---|---|---|
| **Random Erase (RE)** | Replace a region with random values or the training-set **mean** pixel; varied sizes and aspect ratios. Standard anti-overfitting method | **Number of boxes, position, size**, plus the **fill value** (0 / mean / random) |
| **Cutout** | Cut out a **fixed-size square** throughout training. An improvement on RE | **Only the position**. Size, count and fill value are fixed, so you can **guarantee what % of information remains** (RE could wipe out 80 %) |
| **Hide-and-Seek (HaS)** | Split the image into **N×N patches**; hide each patch with some probability | Which patches are hidden. Weakness: hidden patches can be **adjacent**, so the whole body may vanish (leaving only background + head) |
| **GridMask** | Hide parts in a regular **grid/mesh** pattern (like HaS but spread out) | You set the grid size. It guarantees every area keeps some information |

#### Multi-source methods
- **MixUp**: overlay two images with transparency weights. It was designed for classification but also works for detection. **Labels are mixed too**: 0.7 cat + 0.3 dog should make a good model predict cat 0.7 / dog 0.3. Best **α = 0.2, 0.3, 0.4** (slide).
- **CutMix**: cut a patch of one training image and paste it onto another (St. Bernard + Poodle example). It avoids the information loss of plain cut-and-drop and improves **object detection**.
- **Mosaic** (first used in **YOLOv4**): "รวมหลายภาพให้เป็นภาพเดียว", combining **4 images** into one. Steps:
  1. Resize all 4 images to **256×256**.
  2. Stitch them into one **512×512** image.
  3. Shift the bounding boxes: top-left unchanged; top-right x+256; bottom-left y+256; bottom-right x+256 and y+256.
  4. Randomly crop a **square** (slide: size range 16–240 px) that **must touch all 4 images**. The proportions don't have to be equal.
  5. Remove the boxes that were cropped away.
  6. Keep the non-cropped area and resize it to the model's input size.
- When to use group 3: **object detection**, or classification where **classes look very similar** (ข/ฃ-type Thai letters, orange vs lime). CutMix and Mosaic are used mostly in detection.
- Teacher's minimum today: use **Distortion + Occlusion**. Group 3 is only needed if the dataset is hard.

### 4. Static vs Dynamic augmentation
| Static (Traditional, Fixed Policy) | Dynamic (per epoch) |
|---|---|
| Same policy for the whole training run | Intensity/type changes with epoch |
| Fine for general tasks | Suits curriculum / robust learning |
| Does not know the model's state | Responds to loss, accuracy or sample difficulty |

**Problems with traditional augmentation**
- Same augmentation every epoch: *Static Configuration*, *One-Size-Fits-All* (for example rotation 30°, brightness ±0.3 from epoch 1 to the end), *no adaptation*.
- Ignores model state: no view of *Learning Progress*, ignores *Model Capacity*, *no Monitoring*.
- **Early training** (weak model, heavy distortion): *Overwhelming Complexity* ("teaching kindergarteners with university textbooks"), *Gradient Instability* (loss unstable, slow or no convergence), *Feature Learning Disruption* (basic edges/shapes/colours get disturbed).
- **Late training** (strong model, easy data, around epoch 80–100): *Insufficient Challenge* (leads to overfitting; "university students doing kindergarten exercises"), *Missed Generalization Opportunity*, *Resource Wastage*.
- Impact: **Performance** (slower convergence, lower final accuracy, poor generalization), **Resources** (longer training, compute waste, higher cost), **Stability** (unstable training, inconsistent results, many runs needed).
- Conclusion (slide): fixed-policy augmentation is losing popularity. Match augmentation to the model's learning level.

**Weak → Strong levels** (teacher: a rough "ใช้กันเองในครัวเรือน" guideline, not exact)
| Level | Examples | Effect |
|---|---|---|
| Very Mild | flip H/V, slight brightness/contrast | Semantics unchanged |
| Mild | rotate ±5–10°, translate ±5 %, low colour jitter, low-std Gaussian noise, random crop + resize | Main shape kept |
| Moderate | shear ±10–20°, small Cutout, AutoContrast/Equalize, medium brightness/contrast | Alters geometry and colour |
| Strong | rotate ±30–45°, strong colour jitter, CutMix / MixUp / large Random Erasing, Posterize/Solarize, translate ±30–50 % | Big change, harder to classify |
| Very Strong | rotate ±90°+, perspective warp, high-std noise, strong blur, JPEG artefacts, adversarial transforms | Far from real data. Only for models already confident |

The mild levels are pure Distortion. Moderate and Strong mix in Occlusion and group 3. The teacher says Very Strong is rarely used because it can change the image's meaning, and in practice you stop at Strong.

**Three dynamic methods**
| Method | How it works | Strength |
|---|---|---|
| **Progressive** | Increase strength **with time (epochs)**, ignoring actual model state | Easy-to-hard (curriculum learning) |
| **Adaptive Scheduling** | Adjust by **training loss / validation metric** | The model drives the adjustment |
| **Sample-Aware** | Adjust by **class or sample difficulty** | Fairness / customization |

- **Progressive** example: epochs 1–20 use rotation ±5°, brightness ±0.1. Epochs 21–50 use ±15°, ±0.2 and add Gaussian noise. Epoch 51+ uses full augmentation (±30°, Cutout, MixUp). Benefits: avoids **underfitting** early, and it follows curriculum learning. You need a trial run first to know how many epochs to split into stages.
- **Adaptive Scheduling**: if validation loss is still high, keep augmentation weak. Once loss drops, make it stronger to avoid overfitting. Signs the model is "too comfortable" ("สบายใจ"):
  - val loss has not decreased for **5 consecutive epochs**
  - accuracy has not increased beyond a set **threshold**
  - the gradient is too small

  Benefits: augmentation matches the learning stage and lowers the risk of over- or under-augmentation.
- **Sample-Aware**:
  - *Difficulty-aware per sample*: confidence **> 0.9** gets **Strong**. Wrong or confidence **< 0.6** gets **Weak**. In between gets Medium.
  - *Class-aware*: a class with **low accuracy** gets **Strong** (more data, more challenge). A class with **high accuracy** gets **Weak** (avoid noise).

### 5. Transfer Learning
- **Definition**: reuse part of an already-trained model by connecting it to a new model. Training from scratch takes days to months, so reusing a model trained on a similar task (a **pretrained network**) cuts training time a lot.
- Typical source is **ImageNet: 1.4 M images, 1,000 classes**. Core assumption: **general features learned on a large dataset can be shared across seemingly different datasets.** Public ImageNet models include AlexNet, VGG, ResNet, Inception and DenseNet.
- Vocabulary (teacher): the **Backbone/Architecture** is the model's layer layout. **Backbone + Pre-trained** means that architecture with ImageNet-trained weights.
- A CNN has 2 parts. The **convolutional base** extracts features. The **FC layers** classify.
- **Two ways to use a pretrained model** (Yamashita et al. 2018, radiology review):
  1. **Fixed feature extraction**: remove the FC layers and keep the conv base **frozen** as a feature extractor. Put any classifier on top (random forest, SVM, or new FC layers) and train only that classifier. This is **rarely used in medical imaging DL because ImageNet and medical images are dissimilar.**
  2. **Fine-tuning**: replace the FC layers and **also retrain all or some conv-base kernels via backprop**. Usually the early layers are frozen and the deeper layers trained. Early layers hold **general features (edges)**; later layers are **task-specific**. This is the method used more in radiology research.
- **Disadvantage**: an input-dimension constraint. Input must be **RGB (3-channel)** because ImageNet is RGB, while medical grayscale images have **1 channel**. Height and width can be arbitrary (not too small) if you add a **Global Pooling layer** between the conv base and the new FC layers.
- **Three patterns** (slides 51–53, plus the teacher's explanation):

| Type | Slide description | Teacher's interpretation |
|---|---|---|
| 1 | Use the **whole** pretrained model and train it further on new data, removing nothing | Use everything as is (all frozen). "แทบจะ Impossible" in practice, because the number of output nodes ≠ your number of classes unless it's the same task |
| 2 | Use only the **feature-extraction** part: remove the last Dense layer and build and train a **new classifier** on the smaller or more specific dataset | **Freeze** the feature extractor and train only the classifier. The **most common** approach, because image features (colour, shape, texture) are "common" across tasks |
| 3 | Use **some parts** (especially feature extraction) of a model from a similar task | Freeze the **early** conv layers and make the **later** conv layers trainable, plus the classifier. Use when your images **don't look like ImageNet natural photos**, for example **ultrasound or X-ray** |

- **Freeze = that part is NOT trained** (teacher: "กาดอกจันทร์คำว่า freeze"). The output layer must always be retrained, because its node count must equal your number of classes.
- Why freeze early layers in type 3: early layers extract low-level features (lines, colours) that every kind of image shares. Later layers combine them into objects, and that combination differs between ultrasound, X-ray and photographs. You *can* train everything, but this is the reasoning for freezing part.
- **Choosing a pretrained model**: consider **Accuracy, Speed, Size** (slide 54). Options: AlexNet, ResNet, VGG-19, GoogLeNet, MobileNet, Inception, DenseNet.
- **Slide 57 chart** plots ImageNet accuracy (y) against relative GPU prediction time (x, log scale), with bubble size = model size. Values below are approximate readings:

  | Model | Accuracy | Speed / size |
  |---|---|---|
  | SqueezeNet | ~55 % | fastest, tiny |
  | AlexNet | ~54 % | fast, large bubble |
  | ShuffleNet | ~64 % | small |
  | GoogLeNet | ~66 % | – |
  | ResNet-18 | ~69.5 % | – |
  | MobileNet-v2 | ~70.5 % | small, fast |
  | VGG-16/19 | ~70 % | **by far the biggest bubbles**, slow |
  | ResNet-50 | ~74.5 % | – |
  | ResNet-101 | ~76 % | – |
  | Inception-v3 | ~77 % | – |
  | DenseNet-201 | ~76 % | slow (~23× time) |
  | Inception-ResNet-v2 | ~80 % | – |
  | NASNet-Large | ~82 % | slowest, large |

  The models taught in class are underlined on the slide. Takeaways:
  - Higher accuracy usually costs speed and/or size.
  - VGG is huge for its accuracy.
  - SqueezeNet and MobileNet are small and fast.

### 6. History and naming (teacher)
- The ImageNet classification challenge ran 2010–2011 with ML methods at ~70+ % accuracy. In **2012 AlexNet** (CNN) cut the error from **25 % to 16 %** (~84 % accuracy) and started the boom. 2013 brought an improved AlexNet; **2014 VGG and GoogLeNet**; **2015 ResNet**. After ResNet the challenge stopped because accuracy reached ~96–97 %.
- **Slide 56 chart** (ILSVRC top-5 error %, from He et al. 2015):

  | Year | Model | Top-5 error | Depth |
  |---|---|---|---|
  | '10 | – | 28.2 | shallow |
  | '11 | – | 25.8 | shallow |
  | '12 | AlexNet | **16.4** | 8 layers |
  | '13 | – | 11.7 | 8 layers |
  | '14 | VGG | **7.3** | 19 layers |
  | '14 | GoogLeNet | **6.7** | 22 layers |
  | '15 | ResNet | **3.57** | **152 layers** |

  Deeper networks give lower error. The teacher's "25 % to 16 %" is this top-5 error.
- **LeNet** was the actual first CNN, but only for digits 0–9. AlexNet was the one that "boomed" because it worked on colour images. Google**L**eNet's capital **L** pays respect to LeNet.
- Naming eras:
  1. single names (AlexNet, SqueezeNet, ShuffleNet, GoogLeNet)
  2. layer count (ResNet-18, VGG-16/19)
  3. size letters (S/M/L, Base/Large/Tiny) because a number doesn't tell users how big the model is

  Current: EfficientNet; non-CNN: Transformers.
- Names for a group of layers: **Module** (SqueezeNet *Fire module*), then **Cell** (GoogLeNet *Inception cell*), then **Block/Stage** (ResNet onward). They all mean a group of layers working together.

### 7. Architectures

| Architecture | Year / by | Depth | Key ideas | Notes |
|---|---|---|---|---|
| **LeNet** | first CNN | – | Digit recognition 0–9 | Mentioned only verbally |
| **AlexNet** | **2012**, Alex Krizhevsky et al., ImageNet | **8** = 5 conv + 3 FC | **ReLU instead of Tanh**; **Dropout** against overfitting (but ~**2×** compute) | **60 M params**, so it suffered **overfitting** |
| **VGG** | **2014**, Simonyan et al., Oxford | 13 conv + 3 FC (VGG-16); VGG-19 = 19 | Studied **effect of depth on accuracy**; **all filters 3×3** (believed as good as 7×7); input **224×224 RGB**; preprocessing: **subtract ImageNet mean colour**; **1×1 conv to make the prediction** | Teacher: the first to normalize with (x−mean)/sd from ImageNet, and the first to classify with conv instead of an FC network |
| **SqueezeNet** | **2016** | – | **Fire module** = **squeeze** (1×1) + **expand** (1×1 and 3×3, concatenated) | Goal: save memory/resources |
| **GoogLeNet / Inception** | **2014**, Google | **22** | **9 Inception cells**; parallel **1×1, 3×3, 5×5 conv + 3×3 max-pool**, outputs concatenated (**multi-scale**); **1×1 conv before 3×3/5×5 and after the max-pool** to cut compute; **Global Average Pooling** before the output (7×7 to 1×1) | Saves compute: fewer params and operations |
| **ResNet (ResNet-18)** | **2015**, Microsoft Research | 18 | **Residual learning**: **Convolution Block, Identity Block, Skip Connection**; **element-wise addition** | Solves vanishing/exploding gradients so very deep nets can train. Its blocks were later reused in segmentation networks |
| **DenseNet** | **2016**, Gao Huang et al. | – | **Dense blocks** (each layer receives **all previous layers'** maps, **channel-wise concatenation**); **Transition layers**; **growth rate k** | Feature reuse; needs more memory |
| **MobileNetV3** | – | – | **Inverted residual block** (1×1 expansion, depthwise separable conv, 1×1 projection, skip) + **Squeeze-and-Excite** | For mobile / low-spec devices |

#### Layer stacks (slide 69 "เปรียบเทียบสถาปัตยกรรม AlexNet, VGG และ GoogLeNet")
- **AlexNet, 8 layers (ILSVRC 2012)**:
  1. 11×11 conv, 96, stride 4, pool/2
  2. 5×5 conv, 256, pool/2
  3. 3×3 conv, 384
  4. 3×3 conv, 384
  5. 3×3 conv, 256, pool/2
  6. fc 4096
  7. fc 4096
  8. fc 1000

  That is 5 conv + 3 FC with **large early filters** (11×11, 5×5).
- **VGG, 19 layers (ILSVRC 2014)**: only 3×3 convs, in stages of 64 ×2, 128 ×2, 256 ×4, 512 ×4, 512 ×4, each stage ending in pool/2. Then fc 4096, fc 4096, fc 1000. That is 16 conv + 3 FC = 19. VGG-16 (slide 59 text) has 13 conv + 3 FC.
- **GoogLeNet, 22 layers (ILSVRC 2014)**: a tall stack of repeated Inception cells. The diagram also shows two side branches.

#### 1×1 convolution (the teacher's favourite operator)
- It looks only at the pixel itself, not its neighbours, and **combines that pixel's features across all channels**.
- Uses:
  1. **Change (usually reduce) the number of channels**. "เมื่อไหร่ที่ต้องการปรับจำนวน channel เราใช้ 1×1 เสมอ". The teacher compares it to PCA.
  2. Fewer channels mean **less memory, fewer params and less compute**, which also means **less overfitting**.
  3. **Classification** (VGG: 1,000 1×1 filters give 1,000 class scores).
- History (slide 66): the 1×1 conv was introduced in **Network in Network (NIN)** to add **non-linearity** (used with ReLU). GoogLeNet used it to **shrink feature maps and cut compute**.

#### SqueezeNet Fire module
- Three design ideas:
  1. Replace 3×3 filters with **1×1 (point-wise)**, giving **9× fewer parameters**.
  2. 1×1 reduces feature-map depth and compute time.
  3. **Late (delayed) pooling** keeps a large feature map, which improves performance.
- **Squeeze** = 1×1 conv that **reduces channels** (example: 6 to 2) **without changing W×H**. It is dimensionality reduction by *combining* channels, not selecting them.
- **Expand** = 1×1 filters (**point features**) and 3×3 filters (**local features**: the pixel plus its neighbours) run separately and are **concatenated**.
- Order: squeeze, ReLU, expand, concat, ReLU.

#### GoogLeNet Inception cell
- **Multi-scale**: small filters capture small objects and fine detail; large filters capture large objects. Using several filter sizes in one layer and concatenating them gives both.
- 1×1 before 3×3/5×5 reduces channels so the parameter count doesn't "บวม" (bloat).
- **Why 1×1 after the 3×3 max-pool?** Max pooling works **per channel**: it does not mix or change the channel count. A 1×1 conv afterwards is needed to reduce channels. Conv filters (1×1/3×3/5×5) use **all channels**.
- **Global Average Pooling (GAP)**: every channel becomes one value (average of a 7×7 map to 1×1).
  - **Why**: conv layers accept any input size, but the **FC network has a fixed number of input nodes**. With GAP, a different input size changes W×H but not the channel count, so the FC input stays fixed.
  - It also reduces compute. Later generalized to **Adaptive** (global) average pooling with N×N output.

#### Connectivity patterns (slides 70–71)
| Pattern | How | Pros | Cons |
|---|---|---|---|
| **Standard** | Successive convolutions, output goes to the next layer only | Simple | Vanishing/exploding gradient in very deep nets |
| **ResNet** | **Skip/residual (shortcut) connection**; skipped output **added element-wise** (⊕) to a later layer | Reduces vanishing gradient, enables deeper nets | Slightly more complex; the shapes being added **must match** |
| **DenseNet** | Each layer gets input from **all previous layers**; **channel-wise concatenation** (Ⓒ) | More effective feature learning | Needs **more memory** to store all features |

- Teacher's motivation: a filter is *supposed* to discard noise and keep useful information, but **nothing guarantees** that important information wasn't filtered out early. Adding back the unprocessed input prevents **Information Loss** and enriches (re-weights) the useful features.
- Both ResNet and DenseNet connections exist **only inside a block**, not across the whole model.

#### ResNet-18 blocks (slides 72–73)
- **Overall structure (slide 72 figure)**:
  1. Input image 224×224
  2. 7×7 conv, 64, stride 2
  3. pool, stride 2
  4. 4× (3×3 conv, 64)
  5. 4× 3×3 conv, 128 (first one stride 2)
  6. 4× 3×3 conv, 256 (first one stride 2)
  7. 4× 3×3 conv, 512 (first one stride 2)
  8. Average pool, then fully connected, then softmax

  Count: 1 + 16 + 1 = **18 weighted layers**.
- Skip arrows span every 2 convs. **Solid** arrows are identity shortcuts. **Dashed** arrows are where the size changes (/2, more channels): those are the convolution-block shortcuts (1×1 stride-2 conv).
- **Convolution Block** is used when the output size must change. The first conv is **3×3, stride 2, padding 1**: (56×56×64) becomes (28×28×128). The second conv is 3×3, stride 1, padding 1, which keeps 28×28×128. The **shortcut needs a 1×1 conv with stride 2** (64 to 128 channels) so the old 56×56×64 output becomes 28×28×128 and can be added. Then ReLU.
- **Identity Block**: input size = output size, so the shortcut carries the input unchanged (*identity*). Add, then ReLU.
- Pattern: a Conv block (A) then Identity blocks (B, B, B). Only block A downsizes.
- Teacher's "spot the difference" game: there are **2 differences** between a Conv block and an Identity block: (1) the first 3×3 conv uses **stride 2** instead of 1; (2) the shortcut has a **1×1 conv with stride 2**.
- Design lessons:
  1. ResNet **downsamples with a stride-2 conv instead of pooling**.
  2. The skip connection reduces information loss.
  3. A 1×1 stride-2 shortcut samples every other pixel and mixes channels, so the "added back" data is **not the original data**: it is downsampled and recombined. The shortcut path carries **point** features (1×1); the main path carries **local** features (3×3).

#### DenseNet (slides 74–81)
- **Basic composition layer**: **pre-activation** BN, then ReLU, then **3×3 conv** that outputs **k** channels.
- **Bottleneck layer**: BN, ReLU, **1×1 conv** (reduce channels), then BN, ReLU, **3×3 conv** (extract new features). Fewer params and less compute.
- **Dense connectivity**: all layers in a block keep the same spatial size, so maps can be concatenated. Benefits:
  - **More diversified features**
  - The classifier can use features of every complexity level, which **maintains low-complexity features**, gives smoother decisions and **works well with limited training data**
  - **Feature reuse**, which means fewer parameters
- **Transition layer** (between dense blocks): BN, ReLU, **1×1 conv** (reduce the channels that piled up through concatenation), then **2×2 average pooling** (downsample W×H).
- The teacher notes this is the **first architecture to use average** rather than max pooling. His guess why: averaging suppresses noisy peaks.
- **Growth rate (k)**: the number of feature maps each layer **adds** inside a dense block. In the slide 81 figure:
  - x₀ goes through h₁ to give x₁.
  - [x₀, x₁] are concatenated and go through h₂ to give x₂.
  - [x₀, x₁, x₂] give x₃ through h₃, and so on to x₄.
  - The stack of feature maps grows by k with every layer.
- Teacher's experience: DenseNet doesn't perform especially well unless the task is truly complex.

#### MobileNetV3 (slides 82–89)
- The **bottleneck block** improves performance and reduces memory. Components:
  - **Inverted residual block**: **1×1 expansion conv** (increases channels so depthwise conv can capture more spatial features), **depthwise separable conv**, **1×1 projection conv** (reduces channels again, fewer params), **skip connection**.
  - **Squeeze-and-Excite (SE)** block.
- Slide example: M×N×3, then 1×1 expansion with 16 filters (each 1×1×3) gives M×N×16. Depthwise separable conv, then SE, keeps M×N×16. 1×1 projection with 3 filters (each 1×1×16) gives M×N×3. Add the skip connection.
- **Depthwise separable conv** replaces a standard conv to shrink the model. It has two parts:
  - **3×3 Depthwise conv + BN + H-swish**: one filter per channel, each working on **one channel only**. Fewer params and less compute, so faster training.
  - **1×1 Pointwise conv + BN + H-swish**: combines channels at each pixel to create new channels. It can **increase or decrease** the channel count.
- Teacher's key point: in a normal conv the per-channel results are simply **summed** (implicitly weight 1 each). In depthwise + pointwise the 1×1 step has **learnable weights per channel**, so channels are *weighted* before combining.
- **SE block** (Jie Hu et al., "Squeeze-and-Excitation Networks"): models relationships between channels so the network focuses on important feature maps. The idea: **channels are not equally important, so weight them.**
  1. **Squeeze** = GAP: H×W×C becomes **1×1×C**.
  2. **Excitation** = 2 FC layers: the first **reduces** dimension, followed by **ReLU**; the second **restores** it to C, followed by **sigmoid**. The result is per-channel weights.
  3. **Recalibration** = channel-wise multiplication of the weights with the original feature map.

---

## Formulas & numbers to remember

**Conv output size** (standard formula, consistent with the slide example): `O = floor((W − K + 2P) / S) + 1`
- ResNet conv block: W=56, K=3, P=1, S=2 gives (56 − 3 + 2)/2 + 1 = 27 + 1 = **28**, so 56×56×64 becomes **28×28×128** (128 filters).
- Same conv with S=1, P=1 gives **56**: a 3×3 conv with stride 1 and padding 1 keeps the size; with stride 2 it **halves** W×H.
- Shortcut 1×1, S=2, P=0: (56 − 1)/2 + 1 = **28**, with 128 filters, so it matches 28×28×128 and can be added.

**GoogLeNet 1×1 bottleneck** (slide 66; the slide calls these *number of operations*)
- Direct 5×5, 48 filters on 14×14×480: (14×14×48) × (5×5×480) = **112.9 M**
- With a 1×1 reduction to 16 channels first:
  - 1×1 step: (14×14×16) × (1×1×480) = **1.5 M**
  - 5×5 step: (14×14×48) × (5×5×16) = **3.8 M**
  - Total = **5.3 M**, about **20× less** (teacher: "ลดไป 20 เท่า")
- Each 5×5 filter has as many kernels as there are **input channels** (480), and that is what causes the overhead.

**VGG classification with conv** (teacher's example)
- 14×14×512, then max-pool to **7×7×512**.
- **4,096** filters of size 7×7 give **1×1×4096**.
- **1,000** filters of size 1×1 give **1×1×1000**, one score per class.

**Other numbers**

| Item | Value |
|---|---|
| ImageNet | 1.4 M images, 1,000 classes |
| AlexNet | 2012, 5 conv + 3 FC = 8 layers (layer counts include only learnable conv/FC layers, not pooling/activation), first conv 11×11 with 96 filters, stride 4; **60 M params**; Dropout costs ~2× compute; top-5 error 25.8 % to **16.4 %** |
| VGG | 2014, 224×224 input, all 3×3 filters; VGG-16 = 13 conv + 3 FC; VGG-19 = 16 conv + 3 FC; FC 4096, 4096, 1000; top-5 error 7.3 % |
| ResNet (ILSVRC'15) | 152 layers, top-5 error **3.57 %**; ResNet-18 = 7×7 conv + 16 3×3 convs + FC |
| GoogLeNet | 2014, 22 layers, 9 Inception cells, filters 1×1/3×3/5×5 + 3×3 max-pool, GAP 7×7 to 1×1 |
| SqueezeNet | 2016, 1×1 instead of 3×3 gives **9× fewer params** (3×3 = 9 weights vs 1) |
| ResNet | 2015 (teacher); ResNet-18 example 56×56×64 to 28×28×128 |
| DenseNet | 2016; transition = BN, ReLU, 1×1 conv, 2×2 avg-pool; growth rate k |
| SE block | H×W×C to 1×1×C (GAP), then FC(reduce)+ReLU, then FC(restore)+sigmoid, then multiply |
| MixUp | best α = 0.2, 0.3, 0.4 |
| Mosaic | 4 images, resize to 256×256, stitch to 512×512, box shift +256 in x and/or y, square crop that touches all 4 |
| Adaptive trigger | val loss flat for **5 consecutive epochs**; teacher: patience ≈ **5–10 % of total epochs** (5 of 50 = 10 %) |
| Sample-aware | confidence > **0.9** gets Strong, < **0.6** gets Weak, else Medium |
| Progressive (slide) | ep 1–20: rot ±5°, bright ±0.1; ep 21–50: ±15°, ±0.2, + Gaussian noise; ep 51+: ±30°, Cutout, MixUp |
| Late-training problem zone | epochs 80–100 |

**MixUp formula** (slide 15):
- `x̂ = λ·xᵢ + (1−λ)·xⱼ`
- `ŷ = λ·yᵢ + (1−λ)·yⱼ`

Here λ ∈ [0, 1] is a random number, xᵢ and xⱼ are raw input vectors, and yᵢ and yⱼ are **one-hot** labels. Best α = 0.2, 0.3, 0.4; in the original MixUp paper λ ~ Beta(α, α), which is where α comes in.

Worked examples on the slide:
- cat [1, 0] + dog [0, 1] with λ = 0.5 gives label **[0.5, 0.5]**
- 70 % cat image + 30 % dog image gives label **[0.7, 0.3]**

**Parameter counts** (standard clarification, not on the slides)
- Standard conv: `K·K·C_in·C_out`
- Depthwise separable: `K·K·C_in` (depthwise) + `C_in·C_out` (pointwise)
- DenseNet layer ℓ in a block receives `k₀ + k·(ℓ−1)` channels

**PyTorch code on the slides** (CIFAR-10, SimpleCNN; Linear(32·8·8) because 32×32 is max-pooled twice to 8×8; Adam lr = 0.001, 30 epochs, batch 64, CrossEntropyLoss)
- *Progressive* `get_transform(epoch)`:
  - epoch < 10: RandomHorizontalFlip
  - epoch < 20: + ColorJitter(brightness=0.2, contrast=0.2)
  - else: ColorJitter(0.3, 0.3, saturation=0.3) + RandomRotation(15) + RandomErasing(p=0.2)
- *Adaptive* `build_transform(mag)`: ColorJitter(mag), RandomRotation(mag·45), RandomAffine translate (mag·0.2), RandomErasing(p = mag·0.5).
  - Start `magnitude = 0.1`.
  - If `val_loss > last_val_loss`, the model got worse, so `magnitude = max(0.05, magnitude − 0.05)`.
  - Otherwise `magnitude = min(0.6, magnitude + 0.05)`.
- (Clarification: torchvision's `RandomErasing` works on tensors, so it must come **after** `ToTensor()`. The progressive slide lists it before.)

---

## What the teacher emphasised

- **[00:04:18–00:09:44] Wood-species case study.** K-fold accuracy was 90 %+, but at the live demo almost all ~20 photos were wrong, because the training images were "ideal" top-down (bird's-eye) shots and real users shot tilted, zoomed, with background, hands and bad focus. Adding augmentation (tilt, rotate, center crop, zoom in/out, colour) on the *same* data raised correct answers by ~60 %. Lesson: **observe user behaviour**; some issues can't be fixed in the model.
- **[00:11:40] Augmentation has two uses: increase *Variety* and increase *Sample count*** (the second especially for imbalance or rare events).
- **[00:12:14] "Distortion แปลว่าอะไร"**: degraded but not lost. Geometric changes shape; Photometric changes pixel intensity or colour.
- **[00:19:02] "จำไว้นะครับ… จดไปครับ"**: Random Erase randomizes **(1) number of boxes, (2) position, (3) box size**, plus **(4) the fill value** (0, mean or random).
- **[00:20:07–00:21:19]** Cutout fixes size, count and fill value, and randomizes **only the position**. That guarantees how much information remains.
- **[00:22:30–00:23:50]** Hide-and-Seek's weakness is that adjacent patches can be dropped, which GridMask fixes by spreading the mask.
- **[00:25:09–00:26:38, 00:30:13]** The teacher's **3 groups**: Distortion *adjusts*, Occlusion *hides* (both keep the data's own frame), and group 3 (MixUp, CutMix, Mosaic) *mixes >1 data source*. "แต่ละกลุ่มเล่นอะไรกับมัน ต้องแยกให้ออก".
- **[00:28:50] "ย้ำนะครับ สี่เหลี่ยมจัตุรัส"**: the Mosaic crop is a random-size **square** that must contain all 4 images (not in equal proportions).
- **[00:31:21]** Minimum practice today: use Distortion **and** Occlusion. Group 3 only for hard datasets.
- **[00:32:10] "ยุคนี้ไม่มีใครทำ Static Data Augmentation แล้ว"**: use Dynamic augmentation matched to epoch count, loss or accuracy.
- **[00:41:04] In-class question**: "For Thai character recognition, do you need colour adjustment? Flips?" Answer: **no colour jitter** (letters aren't distinguished by colour). **No left-right flip**; upside-down only if the real scenario has it. **Rotation/tilt: yes.** He criticised senior projects that used augmentations irrelevant to the data, which shows a lack of understanding.
- **[00:43:03] "ประโยคเมื่อกี้งงไหม คิดดี ๆ"**: **high loss means the model isn't smart yet, so use Weak; low loss means smart, so use Strong.** Repeated in the recap at **[02:05:31]**.
- **[00:49:26–00:49:59]** Class-aware is "ตรงกันข้าม" (counter-intuitive): a **low-accuracy class gets Strong**, a high-accuracy class gets Weak.
- **[00:52:41–00:53:01]** Transfer type 1 (use the pretrained model as is) is practically impossible because the number of output nodes ≠ your classes.
- **[00:53:12] "กาดอกจันทร์คำว่า freeze… จดไป freeze แปลว่าท่อนนั้นจะไม่ถูกเทรน."**
- **[00:55:02–00:56:51]** Type 3 (train the later conv layers too) is for images unlike ImageNet natural photos, for example **ultrasound or X-ray**. Early layers hold common low-level features; later layers combine them into objects differently per domain. Repeated in the recap at **[02:07:18]**.
- **[01:04:51–01:06:09] "จดไปครับ"**: a 1×1 filter combines a pixel's features **across all channels**. It is used to **reduce/change the number of channels**, and is always used when you need to resize channels.
- **[01:06:23–01:07:46] VGG trick**: 7×7×512 with 4,096 7×7 filters gives 1×1×4096, then 1,000 1×1 filters give 1,000 class scores. "Classification by conv, no neural network". The teacher calls this, plus (x−mean)/sd normalization, VGG's contribution.
- **[01:10:01] "จดไปครับ"**: Squeeze reduces channels (6 to 2) but **does not change W×H**.
- **[01:11:12] Question**: why does Expand use two filter sizes? **1×1 gives point features; 3×3 gives local features** (pixel plus neighbours).
- **[01:15:20] Question**: a 5×5 filter on a 480-channel input has how many kernels? **480.** That's why direct 5×5 needs 112.9 M; with a 1×1 bottleneck it's 5.3 M, "**ลดไป 20 เท่า**" [01:16:35].
- **[01:19:29–01:21:50] "คีย์เวิร์ดคือ ในทุกแชนแนล… จดไปครับ"**: conv (1×1/3×3/5×5) uses **all channels**; max-pool works **per channel**. Max-pool shrinks W×H but **channels stay the same**, so Inception puts a 1×1 after the max-pool to reduce channels. "Understand the *reason* for operator order and you can modify architectures."
- **[01:22:33] Question**: what problem does Global Average Pooling solve? **Different input sizes.** Conv is fine with any size; the **FC network** has a fixed number of input nodes. GAP leaves one value per channel, and the channel count doesn't change with input size.
- **[01:24:58] "หน้านี้หาที่จดหรือจำไปให้ดี ๆ"** (AlexNet/VGG/GoogLeNet comparison):
  - AlexNet started it.
  - VGG: 1×1 for classification and input normalization.
  - GoogLeNet: multi-filter-size features, 1×1 to cut compute, "cell".
  - Depth 8, 19, 22: everyone stacked more layers.
- **[01:25:03–01:26:19]** A newer or deeper model is **not** automatically better. **Match model complexity to dataset complexity**, measured by (1) number of classes and (2) similarity between classes (ก/ข is easy; ฃ/ข is hard). Anyone who says "just use the newest model" doesn't understand.
- **[01:26:46–01:27:33] What 1×1 does ("ลด ลด ลด")**: reduces channels, which (1) fuses channel features like PCA, (2) cuts memory, compute and params, so less overfitting, and (3) can classify.
- **[01:28:32–01:31:27]** The motivation for connectivity: filtering has **no guarantee** that important information isn't thrown away, so add the unprocessed input back.
- **[01:32:56–01:33:51] "กาดอกจัน operator"**: ResNet uses **⊕ (element-wise add)**, so shapes must match. DenseNet uses **Ⓒ (concatenate)** along channels.
- **[01:34:13] Question** (hint: "5 syllables, starts with I"): what problem do Res and Dense connections address? **Information Loss.**
- **[01:36:38–01:38:29] In-class game**: how many differences between the Conv block and the Identity block (operators only)? **2**: the first conv uses **stride 2** (vs 1), and the shortcut has a **1×1 conv, stride 2**.
- **[01:39:50] "จดไป"**: a 3×3 conv with **stride 2** halves W×H. With stride 1 and padding 1 the size is unchanged.
- **[01:41:21] "หาที่จดไปครับ"**: besides Max Pooling you can reduce W×H with a **conv stride > 1**. The difference: max-pool is per channel; conv uses all channels.
- **[01:43:32–01:45:12]** The shortcut 1×1 stride-2 conv picks every other pixel and mixes channels, so the added data is **not the original**: it is downsampled and transformed (point features vs the main path's local features).
- **[01:46:08–01:47:15]** "Identity" because the shortcut adds the input itself. "Convolution block" because the shortcut needs a conv. Design choices: no pooling (use conv to downsample) and the skip connection against information loss.
- **[01:47:36] In-class question**: "Max-pool and 3×3 stride-2 conv can both reduce W×H — true?" **True.** Differences:
  1. Conv affects/uses **all channels**; max-pool doesn't, so **max-pool has less overhead**.
  2. Max-pool considers each channel separately.
  3. Max-pool is a **fixed operation** (always takes the max); conv weights are **learned**.
- **[01:49:09–01:49:54] "อันนี้ต้องจดไปและต้องระวัง"**: Res and Dense connectivity exist **only within a block**, not across the whole model. The teacher misunderstood this himself at first.
- **[01:51:00–01:51:35]** In DenseNet's bottleneck, the 1×1 before the 3×3 is "the same trick", reducing dimension and memory. The 3×3 does the real feature extraction.
- **[01:53:03] Question**: why does the transition layer end with a 2×2 average pool? CNN shape logic: early layers have large W×H and few channels; later layers have small W×H and many channels. Concatenation grew the channels, so 1×1 sets the channel count and avg-pool shrinks W×H. This is the first use of **average** pooling (earlier networks used max).
- **[01:56:56] "จุดนี้เด็ก ๆ หลายคนเข้าใจผิด"**: depthwise separable ≠ normal conv. A normal conv **sums** per-channel results (implicit weight 1). Depthwise + 1×1 pointwise has **learnable per-channel weights** ("นี่คือจุดต่างกาดอกจันทร์").
- **[02:00:14–02:01:23]** SE block = **weight each channel**, because channels aren't equally important. Squeeze uses GAP; excitation uses an FC network sized to the channel count; then scalar × feature map.
- **[02:02:41–02:07:44] Recap**:
  - Backbone vs pretrained.
  - 3 augmentation groups.
  - Dynamic augmentation by epoch or by loss/accuracy.
  - The 3 transfer-learning types.
  - For projects: download every backbone, run them all, pick what fits, then analyse.

---

## Likely exam questions

1. **What is the main limitation of deep learning that motivates this chapter, and what are the two remedies?**
   It needs a huge amount of training data (up to ~10⁶ images). The remedies are Transfer Learning (reuse an ImageNet-pretrained backbone) and Data Augmentation (synthesize more and more varied samples).

2. **Give the two purposes of data augmentation, with a real-world example of each.**
   (1) Increase the sample count: small datasets or class imbalance, such as a fault seen once in 10 years. (2) Increase variety for robustness: the wood-species app failed on tilted, zoomed, cluttered user photos until rotation, crop, zoom and colour augmentation were added.

3. **Compare Photometric vs Geometric distortion and give 3 examples of each.**
   Photometric changes pixel values: brightness, contrast, sharpness/blur, colour/saturation, noise. Geometric changes shape or position: crop, rotation, scaling, translation, flip.

4. **Compare Random Erase, Cutout, Hide-and-Seek and GridMask.**
   - **Random Erase**: random number, position and size of boxes, and random fill value.
   - **Cutout**: a fixed-size square with a fixed fill value; only the position is random, so the remaining information is guaranteed.
   - **Hide-and-Seek**: N×N patches hidden with some probability; adjacent patches may vanish together.
   - **GridMask**: a regular grid mask, so every area keeps some information.

5. **What is the difference between MixUp, CutMix and Mosaic? Which tasks suit them?**
   - **MixUp**: blend 2 images by transparency; labels are mixed proportionally (0.7 cat / 0.3 dog); α = 0.2–0.4.
   - **CutMix**: paste a patch of one image onto another.
   - **Mosaic**: 4 images resized to 256, stitched to 512, boxes shifted, random square crop that touches all 4 (introduced in YOLOv4).

   They suit object detection and classification of very similar classes.

6. **Explain the problems of static (traditional) augmentation early and late in training.**
   - Early: a weak model hit with heavy augmentation suffers overwhelming complexity, gradient instability and disrupted basic-feature learning.
   - Late: a strong model given easy augmentation faces insufficient challenge (overfitting), misses generalization opportunities and wastes resources.

7. **Compare Progressive, Adaptive Scheduling and Sample-Aware augmentation.**
   - **Progressive**: strength increases by epoch (curriculum; avoids early underfitting).
   - **Adaptive**: driven by loss/metrics. High loss means weak augmentation; low loss or a plateau (for example val loss flat for 5 epochs) means stronger augmentation.
   - **Sample-aware**: per sample (confidence > 0.9 gets Strong, < 0.6 gets Weak) or per class (low-accuracy class gets Strong, high-accuracy class gets Weak).

8. **In the adaptive code, magnitude = 0.3 and validation loss went up this epoch. What is the new magnitude? What if it had gone down?**
   Loss up: max(0.05, 0.3 − 0.05) = **0.25**. Loss down: min(0.6, 0.3 + 0.05) = **0.35**.

9. **Fixed feature extraction vs fine-tuning: which is more common in radiology, and why?**
   Fixed feature extraction freezes the conv base and trains only a new classifier (FC, SVM or random forest). Fine-tuning also retrains some or all conv layers (early layers often frozen). Fine-tuning is more common in radiology because ImageNet photos and medical images are dissimilar. Early layers are generic (edges); later layers are task-specific.

10. **Scenario: 400 grayscale chest X-rays, 3 classes, an ImageNet-pretrained ResNet-18. Describe your transfer-learning setup.**
    - Replace the output layer with 3 nodes.
    - Since X-rays don't look like natural photos, use teacher-type 3: freeze the early conv layers, train the later conv layers plus the classifier.
    - Handle the input constraint: the pretrained net expects 3-channel RGB, but X-ray is 1-channel (for example replicate it to 3 channels). Use global pooling for flexible H×W.
    - Add augmentation that matches real variation (mild rotation/translation, brightness/contrast), ideally scheduled weak to strong.

11. **Calculation: input 14×14×480, 48 filters of 5×5 ("same" padding). Count the multiplications directly and with a 16-filter 1×1 bottleneck first.**
    Direct: (14·14·48)(5·5·480) = 112.9 M. Bottleneck: (14·14·16)(480) = 1.5 M plus (14·14·48)(5·5·16) = 3.8 M, total 5.3 M, about 20× fewer.

12. **Calculation: a ResNet-18 conv block takes 56×56×64. The main path's first conv is 3×3, stride 2, padding 1, 128 filters. What is the output size, and why does the shortcut need a 1×1 stride-2 conv?**
    floor((56 − 3 + 2)/2) + 1 = 28, so the output is 28×28×128. The input 56×56×64 can't be added element-wise to 28×28×128. A 1×1 conv with stride 2 and 128 filters maps it to 28×28×128. In an identity block the shapes already match, so the shortcut is the identity.

13. **Max pooling vs 3×3 stride-2 convolution for downsampling: similarities and differences.**
    Both reduce W×H. Max-pool works per channel, keeps the channel count, is a fixed operation and has low overhead. Conv uses all channels, can change the channel count and has learnable weights.

14. **List the uses of the 1×1 convolution and name the architectures that use each.**
    - Reduce or change channels, which cuts compute and params and lowers overfitting: GoogLeNet before 3×3/5×5 and after max-pool; SqueezeNet squeeze; DenseNet bottleneck and transition; ResNet shortcut; MobileNetV3 expansion/projection and pointwise.
    - Fuse cross-channel features (PCA-like): all of the above.
    - Classification: VGG.
    - Add non-linearity: NIN.

15. **Compare ResNet and DenseNet connectivity (operation, shape requirement, cost), and give the shared motivation.**
    ResNet adds the skip input element-wise to the output, so shapes must match. It eases vanishing gradients and enables very deep nets. DenseNet concatenates all previous layers' maps along channels, so channels grow by the growth rate k. That gives feature reuse, diverse features and good performance with limited data, but needs more memory. Transition layers (BN, ReLU, 1×1 conv, 2×2 avg-pool) keep the channel growth in check. Both work only inside a block. Shared motivation: avoid information loss and vanishing gradients.

16. **What does a Squeeze-and-Excitation block do? Describe its 3 steps.**
    It weights channels by importance.
    1. **Squeeze**: GAP from H×W×C to 1×1×C.
    2. **Excitation**: FC (reduce) + ReLU, then FC (restore to C) + sigmoid, giving per-channel weights.
    3. **Recalibration**: multiply the weights channel-wise with the feature map.

17. **Scenario: deploy a skin-lesion classifier on a clinic's mobile phones. Which backbone family, and why?**
    MobileNetV3. It was designed for low-spec or mobile devices. Depthwise separable convs and inverted residual blocks cut params and memory, and SE boosts accuracy cheaply. This fits the "Accuracy, Speed, Size" selection criteria.

---

## Common traps

- **"Newer/deeper model = better."** Wrong. Match model complexity to dataset complexity: number of classes and how similar the classes are [01:25:03].
- **Fixed-policy augmentation for every epoch.** Outdated. One size does not fit all; use dynamic augmentation [00:32:10].
- **High loss means use strong augmentation.** Backwards. High loss means the model isn't smart yet, so use **weak**. Low loss means **strong** [00:43:03].
- **Mixing up the two sample-aware rules.** Per-sample: a confident sample gets *Strong*, an unsure one *Weak*. Per-class: a **low-accuracy class gets Strong**, a high-accuracy class Weak. The class rule looks like the opposite and the teacher flagged it as counter-intuitive [00:49:35].
- **Applying augmentations that don't fit the task.** Colour jitter or left-right flips for Thai characters are wrong. Augment only variation the real scenario produces, without changing the meaning [00:41:04].
- **Zoom-out produces real content at the edges.** It doesn't. In practice the border is filled with black [00:13:17].
- **"Freeze" means the model is broken.** It means that part is **not trained** [00:53:28].
- **Using a pretrained model unchanged.** You must at least replace and train the output layer, because the node count must equal your classes [00:52:51].
- **Fixed feature extraction is the default in medical imaging.** It is rarely used because ImageNet ≠ medical images. Fine-tuning is preferred, and for X-ray or ultrasound later conv layers should be trainable too.
- **Pretrained ImageNet models accept 1-channel grayscale directly.** They don't. They expect 3-channel RGB (slide 50).
- **GAP exists because convolution can't handle variable input size.** Wrong. Conv is fine; the **FC layer's** fixed input size is the problem [01:22:33].
- **Max pooling changes the number of channels.** It doesn't. It works per channel, which is why Inception adds a 1×1 conv after the max-pool [01:21:23].
- **ResNet/DenseNet links connect across the whole model.** They connect only within a block [01:49:09].
- **ResNet merge = concatenation.** Wrong. ResNet uses element-wise **addition** (shapes must match). DenseNet uses channel **concatenation**.
- **The shortcut in a ResNet conv block carries the original input.** It doesn't. It passes through a 1×1 stride-2 conv, so it is downsampled and transformed [01:44:16].
- **Depthwise separable conv is just a normal conv.** In a normal conv the per-channel results are summed with equal weight. Depthwise + pointwise uses learnable channel weights and fewer params [01:56:56].
- **The first 1×1 in a MobileNetV3 block reduces channels.** The slide calls it a 1×1 **Expansion** (it *increases* channels, for example 3 to 16), and the 1×1 **Projection** reduces them back. In the recording the teacher said "ลด Dimension" at that step [01:55:49]; trust the slide.
- **112.9 M is a parameter count.** The slide labels the GoogLeNet 112.9 M / 5.3 M figures as **number of operations** (multiplications), although the teacher said "parameters". Write "operations/multiplications" in an answer.
- **Slide vs teacher grouping of augmentation.** The slides list Distortion / Occlusion (with MixUp under Occlusion) / YOLOv4 (CutMix, Mosaic). The teacher teaches **3 groups** and puts MixUp, CutMix and Mosaic together as the multi-source group. Know both.
- **AlexNet was the first CNN.** It was the first to win ImageNet and start the boom (2012, colour images). **LeNet** was earlier, for digits only, and GoogLeNet's capital L honours it.
