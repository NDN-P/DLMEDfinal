# Chapter 11 — Object Detection with Deep Learning Techniques

Object detection answers both **What** (class) and **Where** (bounding box). The chapter traces the multi-stage R-CNN family (R-CNN → Fast R-CNN → Faster R-CNN with RPN, anchors and NMS), then the single-stage YOLO, and ends with training YOLOv8 using Ultralytics.

Lecturer: ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์ ("อาจารย์แบงค์"), Faculty of IT, KMITL. Source files: 76-page slide PDF and an ~84-minute recording.

---

## Key concepts

### 1. Classification vs Detection vs Segmentation
- **Classification / regression** says *What* the image is.
- **Object detection** and **image segmentation** say both *What* and *Where*.
- They differ only in the **Where** part:
  - Detection: the **smallest rectangle that fully covers the object**. This is the *bounding box*; the teacher also calls it "Boundary".
  - Segmentation: the object's **actual shape** (its exact outline).
- Example in the slides: 4 objects (2 cats, 1 dog, 1 duck), each given a box.

### 2. Two families (slide outline)
- **Multi-stage detectors**: R-CNN, Fast R-CNN, Faster R-CNN. *Where* and *What* come from **separate modules/stages**.
- **Single-stage detectors**: YOLO. A **unified network** trained **end-to-end** finds *What* and *Where* **at the same time**.

### 3. R-CNN (Region-based CNN) — Ross Girshick et al., 2014
The first CNN-based detector. It has 4 steps:
1. **Region detection with Selective Search**
   - Selective Search groups adjacent pixels/regions that look similar. It works much like clustering.
   - 1.1 Start from **graph-based segmentation** (Felzenszwalb et al.). The output regions are called **Region Proposals** (also called *Candidate Regions*).
   - 1.2 Keep every proposal in a list. Repeatedly merge adjacent ones using a similarity built from **colour, texture, size and shape (fill) compatibility**:
     `s(ri, rj) = a1·s_colour + a2·s_texture + a3·s_size + a4·s_fill`
     - The slide says each `ai ∈ {0, 1}` turns a measure on or off.
     - The teacher said the weights are "normally 0.25 each".
     - The merges build up a hierarchy, like a tree.
   - Stop when the number of proposals is about **1,000–2,000**. A fixed number of iterations is not used, because complex and simple images need different numbers of merges.
   - Each region comes back as a list of (x, y) pixels. Its box is **(x_min, y_min)–(x_max, y_max)**.
2. **Warping**
   - Resize every proposal to the same square size, **224×224**. The result is called a **warped region**.
   - Why it is needed (the slide leaves this as a blank to fill in): CNNs of that era had **no Global Average Pooling or Adaptive Average Pooling**. Inputs of different sizes would give flattened feature vectors of different lengths, and the FC layer needs a fixed length.
   - Downside: warping **loses the object's proportion (aspect ratio)**. For example, a tall rider on a horse becomes square.
3. **Feature extraction**
   - Each warped region goes through a CNN (**AlexNet** with the **last FC layer removed**). The output is a **Feature Vector**.
4. **Classification with an SVM**, one SVM per class.
   - The hyperplane is `WX + b = 0`.
   - The two margin planes are H1: `WX + b ≥ +1` (for y = +1) and H2: `WX + b ≤ −1` (for y = −1).
   - Points that lie on H1 or H2 are the **support vectors**.
   - The margin is `M = d1 + d2`, the distances from H1 and H2 to the hyperplane. The SVM picks the hyperplane with the **largest M**. (Standard result: M = 2/‖W‖.)
- The teacher's lesson: R-CNN passes the flattened vector through FC layers before classifying, because a raw flattened vector "isn't ready" for prediction.
- **Weakness:** it is **slow**. It runs the CNN about 2,000 times per image, once per proposal.

### 4. Fast R-CNN — Ross Girshick et al.
The goal was to cut computation time. It has **8 steps**, and the steps that differ from R-CNN are shown in red on the slide:
1. Feed the **whole image** into the CNN feature extractor **once**, giving one **Feature Map** per image. Use the last feature map before flattening; any backbone works.
2. Run **Selective Search on the whole original image** (in parallel with step 1). This gives the Region Proposals.
3. **Map and crop (RoI projection)**
   - Place each proposal's (x, y, w, h) onto the feature map and crop that area. The crops come out in different sizes.
   - Mapping uses **relative position and scale**: treat coordinates as fractions of the image (0 → 0, 1000 px → 1), so they line up even though the feature map is smaller.
   - This is the slide's stated reason why Fast R-CNN needs a **fixed input image size**.
4. **RoI Pooling** turns every crop into the same fixed size:
   - 4.1 Crop the region from the feature map.
   - 4.2 Split it into n parts per side using the bracketed `[m/n]`. Here m is the height or width and n is the number of parts; the **last part takes the remainder**.
     - The slide's worked figure rounds **down**: a 5×7 region pooled to 2×2 splits the rows 2+3 and the columns 3+4.
     - The teacher said "round up" (8/3 → 3, 3, 2). See Formulas below.
   - 4.3 Take the **max** of each part (max pooling, **not average**).
5. **Flatten** into a feature vector.
6. Pass it through the **Fully Connected** layers.
7. Classify with **Softmax**. This is the *classifier* branch (*What*).
8. Refine the box with **Linear Regression** on x, y, w, h. This is the *bounding-box regressor* branch.
   - The regressor **adjusts** the *Where* that Selective Search already found. It does not create it: "ขยับยุกยิกนิดนึง ยืดหดนิดนึง" (just nudges and stretches the box a little).
- **Why it is faster:** it runs the CNN **once per image**, not once per proposal.
- **Results** (slide 27 table):
  - Training: 84 h → 9.5 h, **8.8×** faster.
  - Test time per image, excluding proposals: 47 s → 0.32 s, **146×** faster.
  - Test time per image **with Selective Search**: 50 s → **2 s**, only **25×** faster. Selective Search is now the bottleneck.
- **It is still multi-stage**, because it still uses Selective Search. At 0.32 s it is **not real-time**.

### 5. Faster R-CNN — Shaoqing Ren et al., 2015
- It fixes the slow, resource-hungry **Selective Search** by replacing it with a **Region Proposal Network (RPN)**. The RPN is trainable.
- Everything else matches Fast R-CNN: whole-image feature map → proposals → map and crop → RoI pooling → FC → softmax + box regressor.
- **The RPN has 6 steps:**
  - **2.1 Intermediate layer.** A 3×3 conv with padding and **512 filters** (input and output channels stay the same).
    - Teacher's view: the backbone feature map is "general" and good for *What*. The intermediate layer turns it into features suited to *Where*.
  - **2.2 Anchor box generation.** Anchors are **pre-defined** boxes, effectively a sliding window, set by two parameters: **Scale** and **Aspect Ratio**. They rest on two assumptions:
    1. **Every point on the feature map is a possible object centre.**
    2. An object there could take any of **3 scales × 3 aspect ratios = 9 anchor boxes** per location (square, tall, wide).
  - **2.3 Classification layer (score).** Decides **foreground vs background** for each anchor and gives its **objectness score**.
  - **2.4 Regression layer (coordinates).** Refines each anchor's [x, y, w, h].
  - **2.5 Candidate selection.** Keep only the anchors predicted as foreground, sort them by objectness score from high to low, and keep the **top n**. The teacher said this is typically the top 2,000 (or 6,000).
  - **2.6 Non-Maximum Suppression (NMS).** Removes spatially duplicate proposals and keeps the most likely one.
- **How it is implemented** (slide 35), with a = number of anchor types:
  - 3×3 conv (c filters) on the last feature map → w×h×c.
  - Score branch: a **1×1 conv with a filters** → w×h×a.
  - Coordinate branch: a **1×1 conv with 4·a filters** → w×h×4a.
  - **"Naming by use":** a 1×1 conv does not inherently "extract a score". The output is *used* as the score or offset, the loss trains it into that role, and so we name it after its use. The teacher called this "ตัวเปลี่ยนโลก" (the world-changer): **YOLO was born from this idea.**
- **Regressor outputs are adjustments, not new coordinates:** x and y offsets are **added**, while w and h are **multiplied** as a ratio.
- **Result** (slide 51, test time per image with proposals):

| | R-CNN | Fast R-CNN | Faster R-CNN |
|---|---|---|---|
| Time | 50 s | 2 s | **0.2 s** |
| Speedup vs R-CNN | 1× | 25× | **250×** |
| mAP (PASCAL VOC 07) | 66.0 | 66.9 | 66.9 |

  The teacher said Faster R-CNN is also more accurate. In this table its mAP equals Fast R-CNN's (66.9); the gain is in **speed**.

### 6. YOLO (You Only Look Once) — single-stage
- **History:** introduced by **Joseph Redmon et al. (University of Washington), June 2015**, according to the slide 64 timeline. The teacher wasn't sure of the origin. YOLOv8 is by Glenn Jocher et al. at **Ultralytics, January 2023**.
- **Unified network, trained end-to-end**: *What* and *Where* are predicted together, with no separate proposal module.
- **Steps:**
  1. Split the image into an **S×S grid**. The **grid cell that contains an object's centre** is responsible for that object.
  2. Each cell predicts **B bounding boxes** (YOLOv1 uses **B = 2**), each with **cx, cy, w, h and a confidence** (objectness), plus **one set of class probabilities per cell**: P(class | object).
- **Object–cell–box–class relationships** (slide 55):
  - Object → cell: the cell that holds the object's centre.
  - Object → box: of the B boxes in that cell, the one with the **highest IoU with the ground truth** "is" the object.
  - Cell → class: a cell can have only **one class**. YOLOv1 does poorly when several objects of different classes share a cell, and on **many small objects clustered together** (its cells are large).
- **Box encoding:**
  - w and h are normalised by the image width W and height H.
  - x and y are **offsets within the cell**.
  - So all four values lie in 0–1.
- **Scoring:** class score per box = P(class | object) × box confidence. Then **NMS** gives the final detections. NMS adds about **2–3% mAP**.
- **Implementation idea** (teacher): a backbone (AlexNet, ResNet, DenseNet, EfficientNet, DarkNet, ...) then 1×1 filters.
  - Each box needs 5 values, so 2 boxes need **10 filters** for *Where*.
  - Add **one filter per class** for *What*: 10 classes need 10 more filters.
- **YOLO vs Faster R-CNN, the same belief in different domains:**
  - Faster R-CNN: every **feature-map pixel** is an object centre.
  - YOLO: every **grid cell on the original image** holds an object centre.
  - One feature-map cell projected back onto the image covers many pixels, which is roughly a grid cell.
- **YOLOv1 architecture:** **24 convolutional layers + 2 fully connected layers**.
- **YOLOv1 output tensor** (slide 59): **7×7×30**, with S = 7, B = 2, C = 20. Per cell:
  - Channels 1–5: Box #1 (confidence, x, y, width, height).
  - Channels 6–10: Box #2 (the same five values).
  - Channels 11–30: class probabilities P(Cat | Object), ..., P(TV | Object).
- **Loss** has 5 parts, all sum-squared error (SSE):
  1. (x, y) position error.
  2. (w, h) error, using the **square root** of w and h to reduce the gap between large and small boxes.
  3. Confidence when an object is present (IoU-based).
  4. Confidence when no object is present, weighted by **λ_noobj = 0.5**.
  5. Class probabilities.
  - **λ_coord = 5** increases the localisation loss.
  - The full formula from slide 62:
    `λcoord Σi Σj 1ij_obj [(xi−x̂i)² + (yi−ŷi)²] + λcoord Σi Σj 1ij_obj [(√wi−√ŵi)² + (√hi−√ĥi)²] + Σi Σj 1ij_obj (Ci−Ĉi)² + λnoobj Σi Σj 1ij_noobj (Ci−Ĉi)² + Σi 1i_obj Σc (pi(c)−p̂i(c))²`
    - i runs over the S² cells and j over the B boxes.
    - `1ij_obj` is 1 when box j in cell i holds an object and 0 otherwise. `1ij_noobj` is the opposite.
    - The class term counts only cells that contain an object.
  - The teacher's point: the loss covers more than class prediction; it also covers objectness and confidence.
- **Versions:** YOLO evolves by combining strengths of earlier models, not by adding new operators. YOLOv8 is built from conv2d, max pooling, concat, batch norm, linear, SiLU, plus pyramid and upsampling, which come in the segmentation chapter.
- **Licensing:** newer YOLO versions are under licences. They are fine for study and research, but commercial use can get you in trouble ("โดนคาบตับ").

### 7. Detector family comparison

| | R-CNN (2014) | Fast R-CNN | Faster R-CNN (2015) | YOLO (v1) |
|---|---|---|---|---|
| Type | Multi-stage | Multi-stage | Multi-stage (two-stage) | **Single-stage**, unified, end-to-end |
| Where (proposals) | Selective Search on original image | Selective Search on original image | **RPN** (anchors + objectness + NMS) | Grid cells predict boxes directly |
| CNN runs per image | ~2,000 (one per warped region) | **1** (whole image) | 1 | 1 |
| Proposal coordinates placed on | Original image | **Feature map** (RoI projection) | Feature map | n/a |
| Standardise region size | **Warping** to 224×224 (loses proportion) | **RoI Pooling** (max) | RoI Pooling | n/a |
| Classifier (What) | **SVM** (CNN = AlexNet, last FC removed) | FC + **Softmax** | FC + Softmax | 1×1 filters, class probability per cell |
| Box refinement | None (per slides/teacher) | Linear regressor (x, y, w, h) | Regressor in RPN + head | Box regression in the loss |
| Test time with proposals | 50 s/img | 2 s (25×) | 0.2 s (250×) | Real-time |
| mAP (PASCAL VOC 07) | 66.0 | 66.9 | 66.9 | — |
| Main weakness | Slow; overhead from ~2,000 CNN passes | Selective Search still slow; not real-time | Still two stages | 1 class per cell; small clustered objects |

> SSD was not covered in this chapter. The single-stage example is YOLO only.

### 8. Ultralytics YOLOv8 (hands-on)
- Open source from Ultralytics Inc. It does **detection, instance segmentation, classification and pose estimation**.
- **Install:** `pip install ultralytics`. Needs **Python ≥ 3.8** and **PyTorch ≥ 1.8**.
- **CLI:** `yolo predict model=yolov8n.pt source='...bus.jpg'`
- **Training on custom data takes 6 steps:**
  1. **dataset.yaml** with keys `train`, `val`, `test` (image paths; absolute paths recommended), `nc` (number of classes) and `names` (list of class names).
  2. **YOLO-format labels:** one `.txt` per image, **one row per object**: `class x_center y_center width height`.
     - Values are normalised to 0–1. Top-left of the image is (0, 0) and bottom-right is (1, 1).
     - Example: class 0 is person, class 27 is tie (COCO).
  3. **Directory layout:** the label path is found automatically by replacing `/images/` with `/labels/`, for example `./datasets/train/images/im0.jpg` → `./datasets/train/labels/im0.txt`. Use the **same file name with a different extension**.
  4. **Choose a model:** 5 sizes, **n / s / m / l / x**, pretrained on COCO (80 classes). A bigger model uses more resources and is more accurate. The slide 72 table, all at 640 px:

| Model | mAP (val, IoU 50–95) | Params (M) | FLOPs (B) | A100 TensorRT (ms) |
|---|---|---|---|---|
| YOLOv8n | 37.3 | 3.2 | 8.7 | 0.99 |
| YOLOv8s | 44.9 | 11.2 | 28.6 | 1.20 |
| YOLOv8m | 50.2 | 25.9 | 78.9 | 1.83 |
| YOLOv8l | 52.9 | 43.7 | 165.2 | 2.39 |
| YOLOv8x | 53.9 | 68.2 | 257.8 | 3.53 |

     - n suits mobile and embedded devices; s is the "standard" middle option.
  5. **Train:** `YOLO("yolov8n.pt").train(data=..., epochs=5, batch=16, imgsz=256)`. `batch` is the number of images sent at once.
  6. **Validate:** `model.val(data=..., imgsz=256, batch=16, conf=0.25, iou=0.5)`.
     - `conf` is the confidence threshold that drops low-confidence objects. Its default is 0.001, which the slide says is not recommended.
     - `iou` is the **NMS overlap threshold**. Its default is **0.6**.
     - The best weights are saved at `runs/detect/train/weights/best.pt`.
- **Inference:** `results = model(img)` returns a **list**, because the library supports both detection and segmentation outputs. Call `result[0].save(path)` to save the detection image only.

---

## Formulas & numbers to remember

### Bounding-box formats
The teacher: "there are 3 formats on the market". Always check which one your library or model expects.

| Format | Values | Notes |
|---|---|---|
| xyxy | (x1, y1, x2, y2) = (x_min, y_min, x_max, y_max) | Top-left and bottom-right corners |
| xywh (corner) | (x, y, w, h), with (x, y) the top-left corner | |
| cxcywh (centre) | (x_center, y_center, w, h) | |
| **YOLO normalised** | `class xc/W yc/H w/W h/H`, each in 0–1 | Resizing the image does **not** change the labels |

**Conversions:**
- `w = x2 − x1`, `h = y2 − y1`
- `xc = x1 + w/2`, `yc = y1 + h/2`

**Worked example (made up).** Image 640×480, box xyxy = (100, 50, 300, 250):
- xywh = (100, 50, 200, 200)
- centre = (200, 150, 200, 200)
- YOLO = (200/640, 150/480, 200/640, 200/480) = **(0.3125, 0.3125, 0.3125, 0.4167)**

### IoU (Intersection over Union)
The slide writes it as "Intersection of Union".

`IoU = |R1 ∩ R2| / |R1 ∪ R2| = intersection / (area1 + area2 − intersection)`

For xyxy boxes:
- `ix1 = max(x1a, x1b)`, `iy1 = max(y1a, y1b)`, `ix2 = min(x2a, x2b)`, `iy2 = min(y2a, y2b)`
- `inter = max(0, ix2 − ix1) · max(0, iy2 − iy1)`

**Worked example.** A = (0, 0, 4, 4), B = (2, 2, 6, 6):
- Intersection = 2×2 = 4
- Union = 16 + 16 − 4 = 28
- **IoU = 4/28 ≈ 0.143**

Edge cases: boxes that do not overlap have IoU = 0 (slide example: 0.91 vs 0.89). Identical boxes have IoU = 1.

### NMS (Non-Maximum Suppression), slide steps 2.6.1–2.6.4
1. Sort the boxes by score (objectness or confidence) from high to low and store them in a list.
2. Take the top box and compute its IoU with each remaining box.
3. If IoU > threshold (the teacher's example used 0.5; Ultralytics' default `iou` is 0.6), it covers the same object: **keep the higher score and delete the lower one**.
4. Move to the next surviving box and repeat until every box has been compared.

**Slide example** (slides 42–49, a person and a dog):
- The person has boxes scored **0.91, 0.88 and 0.63**. The dog has boxes scored **0.89, 0.77 and 0.70**.
- Sorted list: 0.91, 0.89, 0.88, 0.77, 0.70, 0.63.
- Reference 0.91: vs 0.89, IoU = 0 (a different object), so keep it. Vs 0.88, high IoU, so remove it. Vs 0.63, overlapping, so remove it.
- Reference 0.89: removes 0.77 and 0.70.
- Final result: **0.91 (person) and 0.89 (dog)**.

### RoI Pooling cell sizes
Split each side into parts of size `[m/n]`. The last part always takes the remainder. The two sources round differently:
- **Slide 24 figure (round down):**
  - An 8×8 feature map contains a region of **5 rows × 7 columns**, pooled to 2×2.
  - Rows: 5/2 → 2, then 3 (remainder). Columns: 7/2 → 3, then 4 (remainder).
  - The block maxima are **0.85, 0.84, 0.97, 0.96**, flattened to `[0.85, 0.84, 0.97, 0.96]`.
  - This matches the teacher's first explanation [00:36:41–00:37:09]: "7/2 → 3 and 4; 5/2 = 2.5 → 2, remainder last".
- **Teacher's correction (round up)** [00:37:32]: 8/3 → 3, 3, 2.
- In the exam, follow the slide example and state your rounding rule. The pooling is always **max**.

### Anchor and filter counts (RPN)
- Anchors per location: k = scales × ratios = 3 × 3 = **9**.
- Total anchors = H_fm × W_fm × k.
  - 10×10 map: 100 × 9 = **900**.
  - 50×50 map: 2,500 × 9 = **22,500**. The teacher said "~30,000, about that"; the exact figure is 22,500.
- 1×1 filters for k anchor types: **k for scores** and **4k for coordinates**. With 9 anchors that is 9 and 36. Per anchor that is 5 values (1 score + x, y, w, h).
- Intermediate layer: a 3×3 conv with **512** filters.
- Proposals kept: the top **2,000** (or 6,000) by objectness, then NMS.
- (Clarification: the original Faster R-CNN paper uses 2k score outputs for a foreground/background softmax. These slides use k.)

### YOLO numbers
- Output per cell = **B×5 + C**.
  - Teacher's example: B = 2 and C = 10 → 10 + 10 = **20 filters**.
  - Slide 59 (YOLOv1): S = 7, B = 2, C = 20 → **7×7×30**.
- B = 2; architecture is 24 conv + 2 FC; λ_coord = 5; λ_noobj = 0.5; the loss uses √w and √h.
- NMS adds about 2–3% mAP.

### Speed numbers

| Model | Training | Test time (no proposals) | Test time with proposals | mAP VOC07 |
|---|---|---|---|---|
| R-CNN | 84 h | 47 s | 50 s | 66.0 |
| Fast R-CNN | 9.5 h (8.8×) | 0.32 s (146×) | 2 s (25×) | 66.9 |
| Faster R-CNN | — | — | 0.2 s (250×) | 66.9 |

### Precision / Recall / AP / mAP
*Background only. The slides report mAP numbers (PASCAL VOC 07 mAP, and YOLOv8's "mAP val 50–95") but never define mAP.*
- A detection is a **True Positive** if IoU with an unmatched ground-truth box ≥ threshold (e.g. 0.5) and the class is correct. Otherwise it is a **False Positive**. Missed ground-truth boxes are **False Negatives**.
- Precision = TP / (TP + FP). Recall = TP / (TP + FN).
- **AP** is the area under the precision–recall curve for one class, with detections sorted by confidence.
- **mAP** is the mean AP over all classes. mAP@0.5 uses an IoU threshold of 0.5; mAP@0.5:0.95 averages over IoU thresholds from 0.5 to 0.95.

---

## What the teacher emphasised

- **[00:03:07]–[00:05:18]** Detection and segmentation differ in **Where, not What**: "ไม่ได้ต่างกันที่ what ต่างกันที่ where" (they don't differ in what, they differ in where). A detection box is "the smallest rectangle that covers the object"; segmentation gives the actual shape.
- **[00:13:06]–[00:14:30]** Selective Search stops at a **target number of proposals (1,000–2,000)**, not at a fixed number of iterations. Image complexity and object sizes vary too much for a fixed count.
- **[00:15:28]** A region is turned into a box using **min/max of x and y**.
- **[00:17:30]–[00:18:27]** Fill-in-the-blank on slide 9: **why warping is needed.**
  - Q: "ที่เราต้องทำ Wrapping เพราะอะไร... จุดในช่องว่างนั้นคืออะไร" (why do we need warping? what goes in the blank?)
  - **A:** CNNs of that era had **no Global/Adaptive Average Pooling**, so different input sizes give different flattened vector lengths.
  - Also remember the downside: warping **loses proportion**.
- **[00:20:30]–[00:21:02]** In-class question: "การได้ What กับ Where มันได้มาพร้อมกันป่ะ" (are What and Where obtained at the same time?) **A: No.** R-CNN gets Where from proposals, then What from the CNN. That is why it is **multi-stage**; "end-to-end" means **single-stage**.
- **[00:22:37]** "จดไปครับ" (write this down): R-CNN and Fast R-CNN **both still use Selective Search**, so Fast R-CNN is **still multi-stage**.
- **[00:23:01]** "นี่คือ Pain Point" (this is the pain point): R-CNN extracts features 1,000–2,000 times per image, which is the overhead.
- **[00:24:29]** "จดไป" (write this down): R-CNN extracts features from **warped sub-images**; Fast R-CNN extracts them **once from the whole image**.
- **[00:26:12]** "จดไปครับ" (write this down): proposal coordinates are placed on the **original image** (R-CNN) versus the **feature map** (Fast R-CNN).
- **[00:26:51]–[00:27:28]** Question: which is bigger, the feature map or the original image? **A: the original image.** So how can the boxes be placed onto the feature map? The answer comes at **[00:35:01]**: treat coordinates as **percentages (0–1)** of the image and map them proportionally.
- **[00:28:20]–[00:29:02]** "จดไปครับ" (write this down): size standardisation is **Warping on the original image** (R-CNN) versus **RoI Pooling on the feature map** (Fast R-CNN).
- **[00:29:36]–[00:29:40]** "จดใบครับ" (write this down): prediction uses an **SVM** (R-CNN) versus a **neural network + Softmax** (Fast R-CNN).
- **[00:30:31]–[00:31:36]** The bounding-box regressor is "เอาไว้ปรับ Where" (there to *adjust* Where). It does not create Where. Question: does R-CNN adjust Where? **A: No.** Fast R-CNN does, which shows R-CNN was both slow *and* imprecise.
- **[00:31:54]–[00:32:20]** Speed numbers: training 84 h → 9.5 h; inference 47 s → 0.32 s (×146).
- **[00:36:41]–[00:37:49]** RoI pooling division.
  - The teacher first said round down with the remainder last (7/2 → 3, 4; 5/2 → 2, 3), which is what the slide 24 figure shows.
  - He then said "ผมพูดผิด มันต้องปัดขึ้น" (I misspoke, it should round up): 8/3 → 3, 3, 2.
  - Either way, use the **max, not the average**: "สูงสุดนะ ไม่ใช่เฉลี่ยนะ".
- **[00:40:27]–[00:41:22]** 0.32 s is still not real-time. The remaining bottleneck is **Selective Search**, and the authors also wanted a **trainable** proposal step. That leads to Faster R-CNN, the "สารตั้งต้น" (precursor) of the YOLO family.
- **[00:43:39]** The RPN is hard to understand because of **mindset, not operators**. His best advisee needed about 1.5 months to understand it.
- **[00:44:31]–[00:45:18]** The intermediate layer converts features that are good for *What* into features that are good for *Where*.
- **[00:47:04]** "ท่อนที่งงที่สุดของวันนี้" (the most confusing part today): **anchor box generation** rests on two beliefs: (1) every feature-map point is an object centre, and (2) 3 sizes × 3 ratios.
- **[00:49:39]–[00:51:16]** Class calculation: a 10×10 feature map with 9 anchors per point gives **900** candidate boxes. Creating them involves no computation yet; they are only "pre-defined".
- **[00:57:02]** "จดไปครับ อันนี้คือตัวเปลี่ยนโลก" (write this down, this is the world-changer): 1×1 conv outputs are **given meaning by how they are used**, and the loss trains them into that role. **YOLO came from this idea.**
- **[01:02:10]** "อันนี้คือสิ่งที่อยากเน้น" (this is what I want to stress): features are named after their context of use, unlike an edge filter whose function we already know.
- **[01:02:20]–[01:02:43]** In-class quiz: with **9 anchor types**, how many 1×1 filters are needed? **A: 9 for scores and 36 (9×4) for coordinates.**
- **[01:03:00]–[01:03:52]** A real feature map is about 50×50, giving 2,500 × 9 anchors, far too many. Sort by objectness and keep the **top 2,000 (or 6,000)**.
- **[01:04:38]–[01:06:39]** NMS walkthrough with 6 proposals on a person and a dog:
  - Sort by score, use the top box as the reference, and compare IoU with the rest.
  - If IoU > 0.5, drop the lower-scoring box.
  - Then move the reference to the next surviving box.
- **[01:08:16]** "จำไว้นะ" (remember this): **regressor outputs are adjustments, not new x and y**. x and y offsets are **added**; w and h are **multiplied** as a ratio.
- **[01:08:46]** Faster R-CNN: 2 s → 0.2 s, **250× faster than R-CNN**. The teacher said it is also more accurate; slide 51 shows the same mAP as Fast R-CNN (66.9).
- **[01:09:54]–[01:11:03]** Single-stage versus multi-stage, as questions and answers:
  - What does R-CNN use for Where? **Selective Search.** For What? **CNN + SVM.**
  - What does Faster R-CNN use for Where? **RPN.** For What? **Neural network.**
  - YOLO is an **end-to-end, unified network** that finds What and Where **simultaneously**.
- **[01:12:15]–[01:12:42]** YOLO's grid belief is the **same idea as Faster R-CNN's** "every feature-map pixel is a centre". The only difference is the **domain**: original-image grid versus feature map.
- **[01:14:45]–[01:15:15]** Filter counting: 2 boxes × (x, y, w, h, conf) = **10 filters** for Where, plus **one filter per class** for What. Where and What come out at once, so it is single-stage.
- **[01:15:52]–[01:17:41]** Seniors who used YOLO in their projects **could not explain it** at project exams. Employers want people who can "Verify / Understand / Optimize" AI output (*human in the loop*). Students fail interviews when they "ตอบ Why the reason ไม่ได้" (can't answer why). "**อันนี้คือสิ่งสำคัญ**" (this is the important thing): expect *why* questions.
- **[01:17:51]** The YOLO loss is computed from **several parts** (object, confidence, class), not only class prediction.
- **[01:21:01]–[01:21:44]** "สิ่งที่เด็ก ๆ ต้องรู้ก็คือ" (what you must know): there are **3 bounding-box formats**: (x, y, w, h) from a corner, (x_center, y_center, w, h), and (x1, y1, x2, y2). Check what your library expects.
- **[01:22:04]–[01:22:07]** YOLO labels are **normalised to 0–1**, so **resizing the image does not affect the ground truth**.
- **[01:22:30]** Image and label files have the **same name, with a different extension**.
- **[01:22:47]–[01:23:10]** Model sizes n/s/m/l/x: bigger means more resources and more accuracy. n has 3.2 M params and 8.7 GFLOPs, for mobile and embedded devices.

---

## Likely exam questions

1. **What is the difference between classification, object detection and segmentation?**
   Classification gives What only. Detection and segmentation give What and Where. Detection's Where is the smallest box covering the object; segmentation's Where is the actual shape.

2. **List R-CNN's 4 steps and explain why warping is needed.**
   The steps are Selective Search proposals → warp to 224×224 → AlexNet features (last FC removed) → SVM. Warping is needed because CNNs then had no GAP/adaptive pooling, so the FC layer needs a fixed-length flattened vector. The cost is lost aspect ratio.

3. **How does Selective Search group regions, and when does it stop?**
   It uses graph-based segmentation, then hierarchically merges adjacent regions by colour, texture, size and fill (shape) similarity. It stops when about 1,000–2,000 proposals remain, not after a fixed number of iterations.

4. **Why is Fast R-CNN faster than R-CNN? Give at least 3 differences.**
   - It runs the CNN once on the whole image instead of about 2,000 times on warped regions.
   - Proposals are projected onto the feature map, not cropped from the image.
   - It uses RoI pooling instead of warping.
   - It classifies with softmax instead of an SVM, and adds a bbox regressor.
   - Result: 47 s → 0.32 s (×146); training 84 h → 9.5 h.

5. **Why is Fast R-CNN still multi-stage and not real-time? What does Faster R-CNN change?**
   It still relies on Selective Search, which is slow and not trainable. Faster R-CNN replaces it with a trainable Region Proposal Network; the rest of the pipeline is the same.

6. **Compute the RoI pooling output for a 5-row × 7-column feature region pooled to 2×2 (slide 24).**
   - Split rows 2 + 3 and columns 3 + 4, with the remainder in the last part.
   - Take the max in each block: [[0.85, 0.84], [0.97, 0.96]].
   - Flatten to `[0.85, 0.84, 0.97, 0.96]`.

7. **The RPN feature map is 10×10 with 3 scales and 3 aspect ratios. How many anchors? How many 1×1 filters for scores and for coordinates?**
   9 anchors per point × 100 points = 900 anchors. Filters: 9 for scores and 36 for coordinates.

8. **List the 6 steps of the RPN.**
   1. Intermediate 3×3 conv (512 filters).
   2. Anchor generation (scale × ratio).
   3. Classification layer: foreground/background objectness score.
   4. Regression layer: x, y, w, h offsets.
   5. Sort by objectness and keep the top n.
   6. NMS.

9. **Compute IoU for A = (10, 10, 50, 50) and B = (30, 30, 70, 70) in xyxy.**
   - Intersection = (50−30) × (50−30) = 400.
   - Each box's area = 1,600, so union = 3,200 − 400 = 2,800.
   - IoU = 400/2,800 ≈ **0.143**.

10. **Apply NMS (threshold 0.5) to boxes with scores 0.9 (A), 0.8 (B), 0.7 (C), where IoU(A, B) = 0.6, IoU(A, C) = 0.1 and IoU(B, C) = 0.7.**
    - Sort: A, B, C.
    - Reference A: B has IoU 0.6 > 0.5, so remove B. C has 0.1, so keep C.
    - Reference C: nothing is left.
    - **Kept: A and C.** C survives even though it overlapped heavily with B, because B was already removed.

11. **How does YOLO(v1) predict boxes? What is the output size per cell?**
    - Split the image into an S×S grid; the cell containing the object's centre is responsible for it.
    - Each cell predicts B = 2 boxes (x, y, w, h, confidence) and one class distribution.
    - Per cell: B·5 + C values. For B = 2 and C = 10, that is 20. YOLOv1 on slide 59 gives 7×7×30.
    - Final score = P(class | object) × confidence, then NMS.

12. **What are YOLOv1's limitations?**
    - Only one class per cell, so different-class objects sharing a cell perform poorly.
    - It struggles with many small, clustered objects because the cells are large.

13. **Explain the YOLO loss terms and the λ values.**
    - SSE on (x, y); SSE on (√w, √h), where the square root reduces large- vs small-box imbalance.
    - Confidence with an object, and without an object (λ_noobj = 0.5).
    - Class probabilities.
    - λ_coord = 5 increases the localisation loss.

14. **Convert the box xyxy (100, 50, 300, 250) in a 640×480 image to YOLO label format for class 0.**
    `0 0.3125 0.3125 0.3125 0.4167`. Why normalise? The label stays valid after resizing.

15. **Medical scenarios: which approach would you use?**
    - (a) **Real-time polyp detection in colonoscopy video:** a single-stage detector (YOLO). Speed matters, and Faster R-CNN at about 0.2 s/frame is slower. A small YOLO variant (n/s) suits edge or embedded devices.
    - (b) **Exact tumour outline to measure area or volume:** segmentation, not detection. A detection box only approximates Where.
    - (c) **Many tiny, densely clustered cells or microcalcifications in one image region:** YOLOv1 is weak here (large cells, one class per cell). An anchor-based two-stage detector such as Faster R-CNN, which predicts several anchors per feature-map location, handles dense objects better.

---

## Common traps

- **"Wrapping"**: the slides and teacher spell it *Wrapping*, but the technique is **warping** (resizing). Either spelling describes the same step. The output is the **warped region/image**.
- **Detection vs segmentation** differ in **Where**, not What.
- **Fast R-CNN still uses Selective Search** and is therefore **still multi-stage**. Only Faster R-CNN replaces it, with the RPN.
- **Faster R-CNN is not single-stage**, even though it is fully trainable. What and Where still come from separate modules (RPN, then head). YOLO is the single-stage one.
- **Where the proposal boxes are placed:**
  - R-CNN: on the **original image**, then warped.
  - Fast/Faster R-CNN: on the **feature map** (RoI projection, proportional mapping), then **RoI pooling**.
- **RoI pooling uses MAX**, not average. The **remainder goes in the last part**: the slide 24 figure splits 5 → 2 + 3 and 7 → 3 + 4.
- **The R-CNN classifier is an SVM**, not softmax. Its CNN is **AlexNet with the last FC removed**.
- **R-CNN has no bounding-box regressor** in this course's version (slides/teacher). Fast R-CNN adds the linear regressor. (The original R-CNN paper did add a post-hoc box regressor, but follow the slides in the exam.)
- **Regressor outputs are offsets:** x and y are **added**, w and h are **multiplied** (ratio). They are not absolute coordinates.
- **Anchor counts:** k = scales × ratios (3×3 = 9 per location), and the total is H × W × k.
  - Score filters = k, coordinate filters = **4k** (9 → 36).
  - 50×50×9 is **22,500**; the teacher's "~30,000" was a rough figure.
- **NMS:**
  - It removes the **lower-score** box of an overlapping pair. It uses **IoU > threshold** to decide "same object".
  - Boxes with IoU = 0 are different objects; keep both.
  - Repeat with the next surviving box as the reference.
- **The Ultralytics `iou` argument is the NMS threshold** (default 0.6), not an mAP threshold. The default `conf` = 0.001 is "not recommended".
- **YOLO class probabilities are per cell, not per box.** The confidence is per box. Final score = class prob × box confidence.
- **YOLOv1 uses B = 2 boxes per cell** but only **one class per cell**.
- **YOLO label format** is `class x_center y_center w h`, normalised to 0–1. It is a **centre**, not a corner, and not xyxy.
  - Top-left of the image is (0, 0).
  - One `.txt` per image, one row per object.
  - Same file name as the image, under `labels/` instead of `images/`.
- **Three box formats exist** (corner xywh, centre xywh, xyxy). Always check what the library or model expects or returns.
- **The YOLO loss uses √w and √h**, with λ_coord = 5 and λ_noobj = 0.5. Don't swap the λ values.
- **Speed figures:**
  - Fast R-CNN is ×146 vs R-CNN *excluding* proposals (47 s → 0.32 s), but only **×25 including Selective Search** (50 s → 2 s).
  - Training is ×8.8 (84 h → 9.5 h).
  - Faster R-CNN is ×250 (0.2 s).
  - mAP is 66.0 → 66.9 → 66.9, so Faster R-CNN's gain is speed, not mAP.
