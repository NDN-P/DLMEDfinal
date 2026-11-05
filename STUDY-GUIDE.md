# Final exam study guide: everything you need

**One file to pass the exam.** Each question has five parts: what they will ask, what you must know, formulas with fully worked examples, **model answers you can write as they are**, and the traps that lose marks.
Every number was checked in Python, and every code block was run.
Deeper detail is in `materials/Chapter N/notes.md`. Practise with [mock-exam.md](mock-exam.md) and check with [mock-exam-answers.md](mock-exam-answers.md).

**Not on the exam:** Chapter 15 (Transformers) and Chapter 14 §6 RAG.

## Contents
| Exam Q | Topic | Marks (mock) | Chapter | Jump |
|---|---|---|---|---|
| Q1 | Project 1: Thai Character & Number Recognition | 15 | Project1, Ch 8–10 | [Q1](#q1--project-1-thai-character--number-recognition) |
| Q2 | Case study: apply the knowledge | 15 | Ch 8, 9, 10, 14 | [Q2](#q2--case-study-apply-the-knowledge-to-a-medical-scenario) |
| Q3 | Video | 14 | Ch 13 (+14 §4) | [Q3](#q3--video-chapter-13--ch-14-4-metrics--14-marks) |
| Q4 | Image segmentation | 14 | Ch 12 (+14 §3) | [Q4](#q4--image-segmentation) |
| Q5 | Transposed conv by hand | 14 | Ch 12 | [Q5](#q5--transposed-convolution-by-hand) |
| Q6 | Object detection | 14 | Ch 11 (+14 §2) | [Q6](#q6--object-detection) |
| Q7 | Coding (PyTorch) | 14 | Ch 10 | [Q7](#q7--coding-pytorch) |

## How to score high
1. **Be on time.** The teacher explains the answer format in the first 5 minutes. Late = wrong format = no marks.
2. **Answer all 7 questions.** A blank scores 0, and a partial answer with the right idea still gets marks.
3. **Always give the why.** "U-Net" alone is not an answer. "U-Net, because its skip connections bring back the spatial detail the encoder lost" is.
4. **Calculations: show every step.** Write the formula, put in the numbers, give the result and do the size check. In transposed conv, write each X·K block and where it goes.
5. **Use the recipe** for design questions. For Q2 and Q1: task type → output layer + loss → model + transfer learning → preprocessing/augmentation → split → metric → risks.
6. **Draw a small diagram** for architectures: U-Net, Faster R-CNN, CNN→LSTM, two-stream. It's quick and shows you understand.
7. **Don't copy slide errors.** They are listed in [exam-hints.md](exam-hints.md) and repeated in each Traps section below.
8. **"It depends on the dataset/application"** is the teacher's answer for "which is best?". Say it, then explain the trade-off.

## Study order
Do them in this order (easy marks first). After each one, do that question in the mock exam *before* reading its answer.

1. **Q5** Transposed conv: practise the procedure on paper 3 times.
2. **Q7** Coding: write the template program from memory once.
3. **Q6** Detection: one IoU and one NMS by hand.
4. **Q4** Segmentation: the output-layer table and Dice/IoU.
5. **Q3** Video: the comparison table and the LSTM gates.
6. **Q2** Case study: learn the 7-step recipe and write one model answer yourself.
7. **Q1** Project 1: fill in the blanks with your group's model and results, then say it out loud.
8. **Day before:** the full mock exam timed (3 h), then all the **Traps** sections again.

**Only 2 hours?** Read Q5 §3, Q7 §3 (the template program), Q6 §3 (IoU + NMS), the Q4 output-layer table + Dice, the Q3 comparison table, the Q2 recipe, and the Q1 skeleton.

---

## Q1 — Project 1: Thai Character & Number Recognition

### What they will ask
Mock Q1 (15 pts):
- 1.1 task type, input/output, output layer, what to do with the output before submitting
- 1.2 resize strategy, grayscale/3 channels, augmentation dos and **don'ts with Thai-letter reasons**
- 1.3 ≥ 3 confusable pairs of different types, how to find them, how to fix them
- 1.4 the metric the scoring equals, the inference procedure within 30 min, **your group's real model and results**

The exam asks about *what you did*, so fill in the blanks before the exam.

### Fill-in answer skeleton: "Explain your project"
1. **Task.** Classify one cropped image of a Thai character into **72 classes**: 41 consonants (101–141), 21 vowels/tone marks/symbols (201–221), 10 Thai digits (301–310). It is **multiclass** (หนึ่งภาพ หนึ่งคำตอบ), not multilabel and not segmentation.
2. **Input → output.**
   - Input: one `.jpg`, size about **7×11 to 180×177 px** (W×H), grayscale.
   - Output: a **3-digit class_id**. First digit = type, digits 2–3 = index, e.g. 205 = ี.
   - Test set: ~500 images, predictions entered in our group's column of a Google Sheet within **30 min**.
3. **Why this design.** Shape is the only information, and aspect ratio and tiny strokes separate classes, so we kept proportions and avoided augmentations that change meaning.
4. **Data prep.**
   - Grayscale → duplicated to 3 channels for the pretrained backbone.
   - **Pad to square, then resize** to ____ × ____ (e.g. 64×64).
   - Normalise with ____ (ImageNet mean/std if pretrained), identical for train/val/test.
   - Split train/val ____ % **before** augmenting, saved once.
   - Augmentation (train only): ____ (see the points below).
5. **Model.** Backbone ____ (pretrained on ImageNet? ____ ; frozen layers ____), last layer replaced by `Linear(____, 72)`. Why this model: ____ (match model complexity to the data: 72 classes, some nearly identical).
6. **Training.** Loss `CrossEntropyLoss`; optimizer ____, lr ____, batch ____, epochs ____ (early stopping? ____). Best model chosen by **validation accuracy**.
7. **Metric / result.** Validation accuracy ____ %. Test score ____ / ~500 (= ____ %).
8. **Errors.** Main confusions from the confusion matrix: ____ ↔ ____, ____ ↔ ____. What we tried: ____. Result after fix: ____.
9. **Who did what.** ____ (data/preprocessing), ____ (model/training), ____ (inference/submission), ____ (analysis).

### Key technical points (model answers)
- **Output layer.**
  - **72 nodes, softmax, cross-entropy**. `nn.CrossEntropyLoss` takes raw logits (softmax is inside the loss), and the targets are integer indices.
  - Prediction = **argmax**.
  - Why softmax: exactly one correct class, and the probabilities sum to 1.
- **index → class_id mapping.**
  - The model outputs an index 0–71. It must be **mapped back to class_id** with the class table; writing the raw index scores 0.
  - With classes sorted by folder (`ImageFolder` sorts folder names *(standard fact)*), folder order = class_id order: index 0 → 101 ก, 40 → 141 ฮ, 41 → 201 ฯ, 61 → 221 ์, 62 → 301 ๐, 71 → 310 ๙ (checked against the class table).
  - Folder names are TIS-620 codes (161 = ก, 249 = ๙). Codes with no folder (165 ฅ, 166 ฆ, 172 ฌ, 174 ฎ, 198 ฦ …) aren't in the dataset.
- **Pad-then-resize.**
  - The sizes and aspect ratios vary a lot. A direct resize stretches the character, and classes that differ by proportion (with/without a tail, flat upper vowels) start to look alike. The teacher used Thai characters as the example: "เราจะไม่ Resize ก่อน Padding".
  - Tiny marks (7×11, e.g. ่ ้) blur when upscaled, and the thin strokes are what tell them apart, so don't make the input too small.
- **Grayscale.** Colour carries no meaning. Use 1 channel, or duplicate it to 3 for an ImageNet backbone.
- **Augmentation (train only, Ch 8 in-class question).**
  - **Do:** small rotation/tilt (±10–15°), small shift and scale. Handwriting and cropping vary like this.
  - **No left-right flip:** real text is never mirrored, and ใ/ไ, ด/ต would change meaning.
  - **No up-down flip:** ุ (below) ↔ ิ (above) swap meaning.
  - **No colour jitter:** letters aren't distinguished by colour. The teacher criticised irrelevant augmentation as "not understanding the data".
  - **No big rotation:** ่/้ and ๑/๓ get mixed up.
- **Confusable pairs** (give one per type):
  - Consonants: ข/ฃ, ช/ซ, ด/ต, ถ/ภ, บ/ป, ผ/ฝ/พ/ฟ, ศ/ส, ห/ฬ, ท/ฑ
  - Vowels/tone marks: **า/ๅ**, ิ/ี/ึ/ื, ุ/ู, ่/้/๊, ั/็, ใ/ไ
  - Digits: ๔/๕, ๓/๑
  - They differ by a tail, dot or small stroke.
  - **Detect:** confusion matrix on validation (large off-diagonal cells) and look at the wrong images.
  - **Fix:**
    - Larger input so the detail survives, and no augmentation that erases it.
    - More data or class weights for those classes.
    - Pretrained backbone + fine-tune.
    - Class-aware augmentation (low-accuracy class gets **strong**).
    - Ensemble or light TTA if the 30 min allows.
- **Score = accuracy.** 1 point if the prediction = gt_class_id, else 0, so the score is **accuracy over ~500 images**. Blank or not finished in time = 0, the same as wrong, so always fill every row.
- **30-min inference.**
  1. Rebuild the same architecture (same `fc`), then `load_state_dict`.
  2. **`model.eval()` + `torch.no_grad()`.**
  3. **Same preprocessing as validation** (pad → resize → normalise, no augmentation).
  4. Batch on GPU → argmax → map to class_id.
  5. Write a CSV in `gt_path` order (ts_img_001 …) → paste into our column only.
  6. Rehearse the whole pipeline on a folder beforehand. The teacher gives a **folder path**, so one-image inference isn't enough.

### Traps
- Calling it multilabel or segmentation. It is **multiclass**, one character per image.
- Submitting the **index 0–71** instead of the **class_id**. Every row is wrong.
- Direct resize (distorts proportions), or resizing before padding.
- Flips / colour jitter "because everyone uses them". You must justify each augmentation by the Thai-letter reason.
- Augmentation or random crop at validation/inference. Different preprocessing at test than at validation.
- Forgetting `model.eval()` (Dropout/BN stay in train mode) or `no_grad()`.
- Leaving rows blank or running out of time: 0 points, the same as wrong.
- Augmenting before the train/val split: augmented copies leak into validation.
- Giving only generic answers in 1.4. The marks are for **your group's** model, numbers and errors.

---

## Q2 — Case study (apply the knowledge to a medical scenario)

### What they will ask
A hospital scenario (image type, size, number of images/patients, labels, class balance), then 3–4 sub-questions. The mock has 2.1 task + output layer, 2.2 architecture/transfer/augmentation, 2.3 data split, 2.4 metrics from a confusion matrix. Expect a twist such as "now they also want to measure X".

| Scenario wording | Task type | Typical medical example (from the notes) |
|---|---|---|
| "exactly one diagnosis per image" | **Multiclass** | skin lesion → one diagnosis |
| "may show several findings, **or none**" | **Multilabel** | chest X-ray: pneumonia / effusion / cardiomegaly |
| "yes/no", screening | Binary (2 softmax or 1 sigmoid) | COVID screening, sick / not sick |
| "measure / estimate a number" (continuous) | **Regression** | cardiothoracic ratio, blood pressure from arm/skin image, age |
| "measure the area / outline the lesion" | **Segmentation** (then pixels × pixel size) | effusion area, tumour on MRI (2 % of pixels) |
| "where is it / how many" (box is enough) | **Object detection** | locate lesions with box + class + confidence |
| "several *different* questions about one image" | **Multi-task** | ultrasound: lymphedema yes/no + SEG grade + SEFS grade |
| video, whole clip → one label | Video classification | exercise pose, activity |
| video, label every frame / steps | Temporal action segmentation | **surgery step analysis** |
| video, find when an event happens | Temporal action localization | **elderly falls on CCTV** (class, start, end) |
| video → a number | Video regression | heart rate from a video of skin |

### The answer recipe: write these 7 headings every time
1. **Task type + why.** Quote the scenario: "one image *may* have more than one finding, or none → multilabel" (หนึ่งภาพมีโอกาสที่จะเป็นไปได้มากกว่าหนึ่งคำตอบ). Saying only "1 ภาพหลายคำตอบ" gets half credit.
2. **Output layer:** number of nodes, activation, loss, and how to read the prediction (argmax / threshold / raw value). The activation goes outside the model: `CrossEntropyLoss` and `BCEWithLogitsLoss` take logits.
3. **Architecture + transfer learning.**
   - ImageNet-pretrained CNN (ResNet-50 / DenseNet-121 / EfficientNet; MobileNetV3 if it runs on a phone).
   - Replace the last layer (`model.fc = nn.Linear(2048, n)`).
   - **Type 3**: freeze the early layers (edges/lines, shared by every kind of image), train the later conv layers + the new head, because X-ray/ultrasound ≠ ImageNet photos. Fixed feature extraction is rarely used in medical imaging.
   - **1 → 3 channels**: duplicate the grayscale channel (don't change the first conv, or you lose the pretrained weights). Use ImageNet mean/std.
   - **Pad to square, then resize** to 224/256 (เราจะไม่ Resize ก่อน Padding) so anatomy proportions aren't stretched.
4. **Augmentation.** Train only (val/test get the same standardisation, no augmentation). Mild distortion: rotate ±5–10°, small shift/scale, small brightness/contrast. Name what NOT to use and say why (table E). Also use it to add samples of the rare class.
5. **Data split.** **By patient**: all images of one patient go in the same split (follow-ups look almost identical, so a random image split = data leakage = test score too high). Train/val/test, e.g. 70/15/15, stratified by label. Make the split **once, save it to files**, and load the same files for every model ("a test image must be test for every model"). Val picks the hyperparameters and threshold; touch test once.
6. **Metric + why.** Screening → **recall/sensitivity** (FN = missed patient is the worst error). Confirmatory test → **precision**. Both matter → **F1**. Imbalanced → **never accuracy alone**; add F1/AUC. Regression → MAE/RMSE **with unit and range** + Pearson r. Segmentation → Dice/IoU. Detection → IoU ≥ 0.5 + mAP.
7. **Risks / limitations.**
   - Leakage.
   - Class imbalance → lower the threshold, class weights / oversampling.
   - Domain shift: a different hospital or machine, or user behaviour (wood-species lesson: 90 % k-fold, near 0 at the live demo).
   - Small dataset → overfitting.
   - Thin/small findings lost by resizing.
   - The model is a *second reader*; a doctor confirms.

### Must-know tables

**A. Task → output layer → loss → metric**

| Task | Output nodes | Activation | Loss | Read prediction | Metric |
|---|---|---|---|---|---|
| Multiclass | = #classes | Softmax (sums to 1) | Cross-entropy | argmax | Accuracy (if balanced), per-class P/R/F1, confusion matrix |
| Binary | 2 (softmax) or 1 (sigmoid) | Softmax / Sigmoid | CE / BCE | larger / p ≥ threshold | Recall, precision, F1, specificity, AUC |
| Multilabel | = #labels | **Sigmoid per node** (independent) | **BCE** (`BCEWithLogitsLoss`; slide also allows CE) | every node ≥ threshold (0.5 diagram, 0.3 code; tunable) → can be **none** | per-label recall/precision/F1/AUC |
| Regression | = #values (1 per model recommended) | **Linear** (none) | L1/MAE or L2/MSE (no fixed rule, try both) | raw output | MAE, RMSE + Pearson r, state unit & range |
| Segmentation | 1×1 conv; channels = #classes (binary: 1 + sigmoid, or 2 + softmax; multilabel: N + sigmoid; per-pixel regression: linear) | per pixel | Dice / Jaccard loss (good for imbalance), BCE/CE | per-pixel argmax / threshold | Dice (= F1), IoU / mIoU; **not** pixel accuracy |
| Detection | per anchor/cell: class scores + 4 box values (9 anchors → 9 score + 36 coord filters) | softmax/sigmoid for class, linear for box | class loss + box regression (YOLO: SSE) | confidence threshold + NMS | IoU ≥ 0.5 **and** correct class = TP; AP per class, mAP |
| Video | CNN → LSTM (spatial then temporal), ConvLSTM or 3D CNN (spatio-temporal); head as for the task | as task | as task | clip / per-frame / (class, start, end) | Acc/F1 (clip); frame-wise acc, Edit, F1@k (segmentation); tIoU, mAP@tIoU (localization) |
| Multi-task | one head per task, each sized for its task | each head its own | sum of task losses | each head its own rule | per-task metric |

Multi-task options: (a) product of classes in one softmax (2 × 3 = 6), (b) shared extractor + task-specific heads, (c) one model per task. Use (b) when the tasks are related; unrelated tasks → **negative transfer**.

**B. Metric formulas** (Positive = has the disease)

| Metric | Formula | Focus |
|---|---|---|
| Accuracy | (TP+TN)/N | misleads when imbalanced |
| Precision | TP/(TP+FP) | hurt by FP (false alarm) |
| Recall = Sensitivity = TPR | TP/(TP+FN) | hurt by FN (missed patient) |
| Specificity | TN/(TN+FP) = 1 − FPR | recall of the healthy class *(standard fact)* |
| F1 | 2PR/(P+R) = **2TP/(2TP+FP+FN)** | harmonic mean, low if either is low |
| MAE / MSE / RMSE | (1/n)Σ\|e\| / (1/n)Σe² / √MSE | MAE robust to outliers; MSE in unit², punishes big errors; RMSE back in the original unit |
| ROC / AUC | TPR (y) vs FPR (x) over thresholds | 1 = perfect, 0.5 = random, < 0.5 = reversed |

*Worked example (verified with Python).* Diabetic-retinopathy screening, 500 test images, 50 sick. TP = 42, FN = 8, FP = 45, TN = 405.
- Accuracy = 447/500 = **0.894**
- Precision = 42/87 = **0.483**
- Recall = 42/50 = **0.840**
- Specificity = 405/450 = **0.900**
- F1 = 84/(84+45+8) = 84/137 = **0.613**
- A model that says "healthy" for everyone: accuracy = 450/500 = **0.900** (higher!) but recall = **0**. That is why accuracy lies on imbalanced data.

*Regression example (verified).* Cardiothoracic ratio, true [0.45, 0.50, 0.55, 0.60], predicted [0.47, 0.49, 0.60, 0.52]. |e| = 0.02, 0.01, 0.05, 0.08.
- MAE = **0.040**, MSE = **0.00235**, RMSE = **0.048**.
- MSE < MAE here because every error is < 1. "MAE is always smaller than MSE" is false.

**C. Which metric when**

| Situation | Pick | Why |
|---|---|---|
| Screening (don't miss anyone) | **Recall** + lower the threshold | FN is costly; an FP just gets a follow-up test |
| Confirm before treatment (chemo) | **Precision** | FP means harmful, unnecessary treatment |
| Cancer detection: both errors bad | **F1** | balances P and R |
| Compare models over all thresholds | **AUC** | threshold-independent, one number |
| Rare class / imbalanced | recall, precision, F1, AUC per class | accuracy dominated by the majority |
| Small lesion segmentation | Dice / IoU | pixel accuracy inflated by background (2 % tumour → 98 %) |
| Regression | MAE (equal penalty) or RMSE (big errors dangerous) **+ Pearson r**, with unit + possible range | low error ≠ follows the trend (MAE 2.4 with r = 0 vs MAE 8 with r = 1) |

**D. Transfer-learning patterns** (freeze = ท่อนนั้นจะไม่ถูกเทรน)

| Type | What | When |
|---|---|---|
| 1 | Use the whole pretrained model as is | almost impossible: 1000 output nodes ≠ your classes |
| 2 | Freeze the feature extractor, train only a new classifier | most common; data close to natural photos; very little data |
| 3 | Freeze the **early** conv layers, train the **later** conv layers + classifier | images unlike ImageNet: **X-ray, ultrasound** → the medical default |

Input constraint: pretrained nets expect **3-channel RGB**, so duplicate the grayscale channel. Use GAP or a fixed resize for H×W. Pick the backbone by **accuracy, speed, size**; "newest/deepest" is not automatically better. Match model complexity to the data (number of classes, how similar they are).

**E. Augmentation**

| Family | Members | Medical use |
|---|---|---|
| Distortion: geometric | rotate, shift, scale, crop, flip | small rotate/shift/scale: **yes**. Flip: **only if anatomy allows** |
| Distortion: photometric | brightness, contrast, blur, noise, colour | small brightness/contrast (scanner/exposure variation): yes |
| Occlusion | Random Erase, Cutout, Hide-and-Seek, GridMask | fights "looking at one spot only"; careful it doesn't erase the small lesion that *is* the label |
| Multi-source | MixUp, CutMix, Mosaic | only for hard datasets/detection; labels get mixed |

Medical **don'ts** (say why):
- **Horizontal flip on chest X-ray.** The heart moves to the right, so the anatomy/cardiomegaly meaning changes.
- **Large rotation, Very Strong augmentation** (perspective warp, heavy noise/blur). It changes the image's meaning, and real scans never look like that.
- **Colour jitter when colour carries no meaning** (grayscale X-ray). Pointless.
- **Any augmentation at val/test/inference.** Train only. Random crop at inference = different answers for the same image.
- Golden rule (teacher, Thai-character example): augment only the variation the real scenario produces.
- Dynamic augmentation: high loss → **weak**, low loss → strong. Per class: a low-accuracy class gets **strong**.

### Model answer 1: ER chest X-ray triage (multilabel + imbalance)
*Scenario:* 12,000 grayscale chest X-rays (≈2000×2500 px) from 4,000 patients (many have follow-ups). Labels: nodule, pneumothorax, effusion, cardiomegaly. Any combination, or none, can appear. Pneumothorax is only 3 %. The system flags urgent cases in the ER.

1. **Task:** **multilabel**. One image *may* have more than one finding, or none, and all labels answer the same question ("what findings are present?"). That's why it is not multi-task, and not multiclass, because findings co-occur.
2. **Output layer:** **4 nodes**, **sigmoid on each node** (independent 0–1; softmax would force them to compete and sum to 1). Loss **BCE** (`BCEWithLogitsLoss`). Prediction: every node ≥ threshold is a finding. All below = normal. Tune each label's threshold on validation; pneumothorax gets a **low** threshold for recall.
3. **Model:**
   - ImageNet-pretrained DenseNet-121 or ResNet-50, `fc` replaced by `Linear(in, 4)`.
   - **Type-3 fine-tuning**: freeze the early blocks (generic edges), train the later blocks + head, because X-rays don't look like ImageNet photos.
   - Duplicate 1 → 3 channels, ImageNet normalisation.
   - **Pad to square, then resize** to 512 (or at least 224), so the heart/lung proportions (needed for cardiomegaly) are not stretched. 512 is my choice *(standard fact: small findings such as a thin pneumothorax line or a small nodule vanish at low resolution)*.
4. **Augmentation (train only):** rotate ±5–10°, shift/scale ±5 %, small brightness/contrast (different machines and exposure). **No horizontal flip** (the heart would move to the right). No large rotation, no Very Strong augmentation. Extra augmented samples / oversampling for pneumothorax.
5. **Split:** **by patient ID**, 70/15/15, stratified so every split has pneumothorax cases. Saved once to files and used by every model. A random image split puts a patient's follow-ups in both train and test, so the model "remembers the patient" (data leakage).
6. **Metric:** per label **recall (sensitivity)**, precision, F1 and AUC. **Recall for pneumothorax is the main one**, because this is triage: missing it (FN) is dangerous, while a false alarm only costs a doctor's look. Example on 1,200 test images: TP 27, FN 9, FP 60, TN 1104.
   - Accuracy = 1131/1200 = 0.9425, which looks great, but "all normal" scores 1164/1200 = 0.970.
   - Recall = 27/36 = **0.75**, Precision = 27/87 = **0.31**, Specificity = **0.948**, F1 = 54/123 = **0.44**.
   - Conclusion: lower the threshold further to push recall up, and report recall + F1/AUC, not accuracy.
7. **Risks:**
   - Imbalance (3 %).
   - Different hospital/machine → validate externally.
   - Follow-up leakage.
   - Labels from reports may be noisy.
   - Use it as a second reader; a radiologist confirms.
   - *Twist:* "also measure effusion area" → **segmentation** (1×1 conv, 1 channel + sigmoid, Dice loss, Dice/IoU metric, area = pixels × pixel size). "Estimate the cardiothoracic ratio" → **regression** (1 node, linear, L1/L2, MAE/RMSE + Pearson). Both together in one network → multi-task heads.

### Model answer 2: Ultrasound lymphedema (multi-task, Ch 9 lecture case)
*Scenario:* ultrasound images of limbs. For each image predict (i) lymphedema yes/no, (ii) SEG grade 0/1/2, (iii) SEFS grade 0/1/2. About 1,500 images from 300 patients.

1. **Task:** **multi-task**. Three *different* questions with different meanings, so it is not multilabel (multilabel answers all share one meaning). Each task is itself **multiclass**. The grades are discrete levels, so "a number" doesn't automatically mean regression.
2. **Output layer:** option **(b)**, a shared feature extractor + 3 task-specific heads:
   - disease head **2 nodes**, softmax, CE
   - SEG head **3 nodes**, softmax, CE
   - SEFS head **3 nodes**, softmax, CE
   - Total loss = L_disease + L_SEG + L_SEFS. Each head is read with its own argmax.
   - Alternatives: (a) one softmax over the product of classes, 2 × 3 × 3 = 18 classes (many combinations, few images each). (c) 3 separate models: 3× the resources, no sharing.
   - (b) fits because the outputs are clinically related (doctors judge lymphedema from the SEG/SEFS grades) and the data is limited. If the tasks were unrelated → negative transfer.
3. **Model:** ImageNet-pretrained backbone (e.g. ResNet-50), **type 3** (freeze early, train later + heads), because ultrasound is the teacher's own example of non-ImageNet-like images. 1 → 3 channels, pad-then-resize to 224.
4. **Augmentation (train only):** small rotate/shift/scale (probe angle and position vary), small brightness/contrast (gain settings). Avoid large rotation and Very Strong augmentation. An up-down flip would put deep tissue on top, which a real scan never shows *(standard fact: the probe is always at the top of an ultrasound image)*.
5. **Split:** by patient (several images per patient), fixed files, the same for all compared models.
6. **Metric:** disease head → **recall** + F1 (screening, don't miss lymphedema). Grade heads → accuracy if the classes are balanced, else per-class F1 and the confusion matrix (do neighbouring grades 1/2 get confused?).
7. **Risks:** small data → overfitting (augmentation, freezing); the grades are subjective (label noise); operator/machine dependence; negative transfer if one head hurts the others (compare against option (c)).

### Traps
- **Softmax for multilabel.** Outputs sum to 1, so findings compete. Use **sigmoid + threshold**, and remember the answer can be **none**.
- **argmax on multilabel.** Every node ≥ threshold counts.
- **Multilabel vs multi-task.** Same question with many answers (multilabel) vs different questions (multi-task).
- **"It's a number → regression."** Integer grades/ages can be multiclass. Regression = continuous.
- **Regression with sigmoid/softmax, or CE loss.** Use linear + L1/L2. One node per value (1 model per value is recommended: regression needs ~10–20× more data).
- **Accuracy on imbalanced data.** The all-negative model wins on accuracy. Give recall/F1/AUC.
- **Mixing up precision and recall.** Recall's denominator = actual positives (FN). Precision's = predicted positives (FP). Screening → recall.
- **F1 is a harmonic mean.** P = 1.0, R = 0.1 → F1 = 0.18, not 0.55.
- **Slide softmax error.** Scores [−3.44, 1.66, 3.91] give softmax ≈ **[0.001, 0.095, 0.904]**, not the slide's [0.010, 0.059, 0.931]. The sigmoid values [0.03, 0.84, 0.98] are correct. Compute with the formula; don't copy the slide.
- **Augmenting val/test**, or random crop at inference.
- **Resizing before padding**; stretching the image directly.
- **Changing the first conv to 1 channel.** You lose the pretrained weights. Duplicate the channel instead.
- **Fixed feature extraction / "freeze everything" for X-ray.** The medical default is type 3.
- **Random image-level `train_test_split`** with follow-up images = leakage. A seed doesn't fix unfair comparisons; save the split.
- **Pixel accuracy for small lesions**, or comparing Dice from one paper with IoU from another (Dice ≥ IoU).
- **Reporting MAE without unit and range**, or MAE alone without Pearson r / a plot.

---

## Q3 — Video (Chapter 13 + Ch 14 §4 metrics) — 14 marks

### 1. What they will ask
Mock Q3 had 4 parts (3 + 4 + 4 + 3). Expect the same mix:
- **Concepts / classify (≈3)**: spatial vs temporal; define spatio-temporal; label CNN→LSTM / ConvLSTM / 3D CNN as **S→T or ST** (this was the in-class quiz).
- **Medical scenario / choose an architecture (≈4)**: small clinical video dataset, clips of different lengths → pick a model and justify it (dataset size, pretrained, resources); name the video task type for a few variations (per-frame / start–end / a number).
- **LSTM (≈4)**: role of cell state, hidden state, 3 gates, sigmoid vs tanh; **compute Cₜ and hₜ** by hand and interpret.
- **Shapes + 3D conv by hand (≈3)**: Conv3d/MaxPool3d output shape, parameter count, axis order (permute from ConvLSTM layout), a small 3D convolution on paper.
- Possible extras: limits of NN/CNN → why RNN; RNN pros/cons; Two-Stream / I3D / 1×1×1 conv; video metrics (Edit, F1@k, tIoU).
- Rule: "อธิบาย" = give the **why**, not just the name. "คำนวณ" = **show the method**.


### 2. Must-know

**Spatial vs temporal**
| | Spatial (เชิงพื้นที่) | Temporal (เชิงเวลา) |
|---|---|---|
| What | content **inside one frame** ("internal"): edge, colour, shape, texture, position | change **between frames** ("external"): motion, order, direction, speed, timing |
| Operator | 2D convolution (CNN) | RNN / LSTM |
- **Spatio-temporal (ST)** = both extracted **at the same time**. Keyword: **"พร้อมกัน"** (simultaneous). Done by **ConvLSTM** or **3D conv**.
- Spatial first, then temporal = **S→T** (sequential), *not* ST. Example: CNN per frame → LSTM.
- Why temporal matters: with frame 1 and frame 5 and no order you **cannot tell standing up from sitting down**. Spatial alone "ไม่เพียงพอ" (not enough).
- Video tensor: frame = H×W×C (rank 3); stacked frames = **rank-4**; with a batch = 5-D.
- Wrap-up strategies: (a) **parallel** S and T, fused later (Two-Stream), (b) **sequential** S→T, (c) **simultaneous** ST.

**Image task → video task**
| Image task | Video task | Ground truth | Input | Metric (Ch 14) |
|---|---|---|---|---|
| Classification | **Video classification** | 1 label per clip | trimmed clip | Acc, P, R, F1 |
| Segmentation (label per pixel) | **Temporal action segmentation** | 1 label **per frame** (100 frames → 100 labels) | long untrimmed | frame-wise acc, **Edit**, **Segmental F1@k** |
| Object detection (class + box) | **Temporal action localization** | (class, **start, end**) for actions of interest only | untrimmed | **tIoU**, AP, mAP@tIoU |
| Regression | **Video regression** | a number (e.g. heart rate from skin video) | clip | (regression metrics) |
- "Segmentation over time ≠ pixel segmentation in video" (slide 10 footer).

**Why RNN: limits of NN/CNN.** NN/CNN need input that (1) comes **all at once**, (2) has **no sequential relation**, (3) has a **fixed, equal size**. Videos are ordered and of different lengths. RNN: accepts input piece by piece, extracts sequence information, takes variable length.
- One RNN cell, **reused** at every step (unfold), weights shared. H₀ = 0. Output at every step (e.g. translation, per-frame labels) or **only the last** (video classification).
- RNN pros: variable length, model size doesn't grow with input, uses the past, shared weights. Cons: **slow**, early info fades (long sequence problem), cannot see the future.

**RNN vs LSTM vs GRU**
| | RNN | LSTM (Hochreiter 1997) | GRU *(not on slides; standard fact)* |
|---|---|---|---|
| States | hidden Hₜ | **cell Cₜ (long) + hidden hₜ (short)** | hidden only |
| Gates | none (one tanh) | forget, input, output + tanh input modulate | update, reset |
| Long memory | poor | good (additive cell path) | good |
| Cost | lowest | highest | between |

**LSTM = 2 states + 3 gates + 2 modulates**
| Part | Act. | Role (teacher's word) |
|---|---|---|
| Cell state Cₜ | — | **long-term memory**, "รวบรวม" (accumulates) info from all steps; only × f and + new; **no tanh on the path** → keeps info ("like saving money") |
| Hidden state hₜ | — | **short-term memory**, "**keep for predict**"; mostly recent steps; **not** the prediction itself |
| Forget gate fₜ | σ | **Delete**: how much of Cₜ₋₁ to keep (1 = keep all, 0 = erase) |
| Input gate iₜ | σ | **Write**: how much new candidate enters C |
| Output gate oₜ | σ | **Read**: how much of C goes out as hₜ (does not change C) |
| Input modulate zₜ (candidate g) | tanh | prepares new content from xₜ, hₜ₋₁ |
| "Output modulate" tanh(Cₜ) | tanh | **teacher's own name** — not official |
- **Sigmoid (0…1) = weight/gate**; **tanh (−1…1) = normalise data** (zero-centred → learns faster). Don't swap.
- Gate values are **adaptive**: computed from xₜ and hₜ₋₁, so they change with each input even at inference (unlike fixed CNN filters). Each gate has its **own learnable W, b**.
- Cell state holds **more** information, not **complete** (loss at forget gate and input gate).
- **Why LSTM fixes vanishing gradients** *(standard fact; notes say "additive cell-state path")*: Cₜ = fₜ∘Cₜ₋₁ + iₜ∘zₜ, so ∂Cₜ/∂Cₜ₋₁ = fₜ (element-wise). No tanh/weight-matrix squash on that path; with fₜ ≈ 1 the gradient flows back many steps nearly unchanged. In a plain RNN every step multiplies by W and tanh′ (≤ 1) → product shrinks → early steps get ~0 gradient.

**Pipelines (teacher's four)**
1. **CNN → flatten feature vector per frame → LSTM → prediction** — S→T. Can reuse a pretrained image CNN.
2. **Hand-crafted features** (edge filter, MFCC/spectrogram for signals) → LSTM (or CNN if it looks like an image) — S→T.
3. **ConvLSTM** → optional NN → prediction — **ST**.
4. **3D CNN** (+ 3D max pool) → optional NN → prediction — **ST**.

**ConvLSTM (Shi et al. 2015)** — base is LSTM ("นี่คือจุดเน้น"). What changes:
- **matrix multiplications** W·xₜ and W·hₜ₋₁ → **convolution** Conv(xₜ; w), Conv(hₜ₋₁; w);
- states become **2D feature maps**, not 1D vectors;
- f∘C, i∘g, o∘tanh(C) stay **element-wise**.
- Code: concat [xₜ, hₜ₋₁] on channels → one `Conv2d(in=input_dim+hidden_dim, out=4·hidden_dim, padding=k//2)` → split into i, f, o, g. (hidden_dim 3 → 12 filters, 4 groups of 3.)
- **Why ST**: the 2D filter gives spatial features, and the channels it mixes come from **different times** (xₜ and hₜ₋₁) → temporal in the same operation.

**3D CNN**
- 2D conv: filter slides **x, y** only; depth = all input channels. 3D conv: slides **x, y and z (time)**; filter depth can be **smaller** than input → each output covers several frames → ST.
- Paired with **3D max pooling** (2×2×2, stride 2 → halves every dim).
- Pros: more accurate than 2D on 3-D data; S+T together. Cons: **many parameters → overfit**, **few pretrained models**, more memory/time.

**Action recognition problems → Two-Stream / I3D.** UCF-101 / HMDB-51 are **small** → complex models overfit; 3D ConvNets have many params, hard to train.
- **Two-Stream**: RGB frames stream + **optical flow** (motion) stream, mostly **2D ConvNets**, **late fusion** at the end. Teacher's intuition: flow ≈ "frame minus frame" (strictly per-pixel motion estimation).
- **I3D (Inflated 3D)**: inflate 2D **Inception-v1** N×N filters/pools to **T×N×N**; key choice = temporal stride/pooling (temporal receptive field). Two streams (RGB + flow) → 3D ConvNet → AvgPool3D → 1×1 conv → score (1×nc) → **average** the 2 scores → softmax. Pretrained on video → use as feature extractor + LSTM/RNN/GRU/linear.
- **1×1×1 conv**: sees neither space nor time; linear combination of channels → channel reduction, channel mixing, non-linearity (with ReLU+BN), flexibility in Inception branches.
- Modern (names only): SlowFast 2019, X3D 2020, MViTv2 / Video Swin 2021, VideoMAE 2022, UniFormer 2023, TimeFormer 2025. Most use ConvLSTM or 3D conv inside, else S→T.

**One-table comparison**
| Model | Input | How time is modelled | S→T / ST | Pros | Cons | Choose when |
|---|---|---|---|---|---|---|
| CNN→LSTM | (B,T,C,H,W), CNN per frame | LSTM over per-frame vectors | S→T | pretrained ImageNet CNN, light, variable length | spatial detail lost at flatten before time is seen | small dataset, limited GPU, long clips |
| ConvLSTM | (B,T,C,H,W) | LSTM recurrence with conv gates, map states | ST | keeps spatial layout through time | train from scratch (no built-in in PyTorch), heavy memory | where-and-when motion matters, enough data |
| 3D CNN | (B,C,T,H,W) | 3D kernel slides along time | ST | accurate, S+T in one op | many params, overfit, few pretrained, memory | large dataset + GPU, short fixed clips |
| Two-Stream | RGB + optical flow | flow stream carries motion | parallel, late fusion | explicit motion, 2D pretrained nets | must compute flow, 2 networks | motion is the key cue |
| I3D | RGB + flow, (B,C,T,H,W) | inflated 3D kernels + flow | ST, two-stream | **pretrained on video**, strong features | heavy, needs flow | small dataset → use as pretrained extractor |
- **"It depends"**: no single best model. Which works best depends on the **task, the dataset and what you predict**; large frames and long videos cause resource problems.

**Tensor shapes / axis order**
| Where | Shape |
|---|---|
| Slide 59 text | (Batch, Frame, Width, Height, Channel) |
| ConvLSTM `batch_first=True` | **(B, T, C, H, W)** e.g. (4, 30, 3, 112, 112) |
| `nn.Conv3d`, I3D | **(N, C, T, H, W)** = (N, C, D, H, W) |
- ConvLSTM → Conv3d: `x.permute(0, 2, 1, 3, 4)`.
- One video at inference: stack → (30, 3, 112, 112) → `unsqueeze(0)` → (1, 30, 3, 112, 112).
- Slide ConvLSTM classifier: last hidden (4, 64, 112, 112) → `view(4, -1)` = (4, 802,816) → `Linear(112*112*64, 3)` → (4, 3).
- I3D: (25, 3, 16, 224, 224) → 3D ConvNet → (25, 2048, 2, 7, 7): time ÷8 (16→2), space ÷32 (224→7) → (25, 1, 2048).
- Practical: transforms are applied **frame by frame** → random augmentation differs per frame ("จุดสำคัญมาก"). Unequal lengths: trim/sample to fixed `max_frames`, or use an RNN.

**Video metrics (Ch 14 §4)** — verified examples
- Classification (TP 8, FP 2, FN 1, TN 9): Acc 0.85, P 0.80, R 0.889, F1 0.842.
- Frame-wise acc: correct frames / total (17/20 = 0.85).
- **Edit = 1 − D(P,G)/max(N_pred, N_gt)**, D = Levenshtein on collapsed segments; penalises **order errors and over-segmentation**. GT A-B-C-D, pred A-B-C-B-C-D → 1 − 2/6 = 0.667.
- **Segmental F1@k**: segment is TP if right class and IoU ≥ k; each GT matched once, extra fragments = FP. Example: F1@10 = F1@25 = 0.800, F1@50 = 0.600.
- **tIoU** = |pred ∩ GT| / |pred ∪ GT| in time: GT 10–20, pred 12–22 → 8/12 = 0.667; TP if class right and tIoU ≥ 0.3/0.5/0.7.


### 3. Formulas + worked examples (all checked with PyTorch)

**RNN**: Hₜ = tanh(XₜWₓ + Hₜ₋₁Wₕ + B); Yₜ = softmax(HₜWᵧ + Bᵧ) (optional "forward" part); H₀ = 0.

**LSTM**
```
f_t = σ(W_xf x_t + W_hf h_{t-1} + b_f)        # forget / Delete, 0..1
i_t = σ(W_xi x_t + W_hi h_{t-1} + b_i)        # input  / Write,  0..1
o_t = σ(W_xo x_t + W_ho h_{t-1} + b_o)        # output / Read,   0..1
z_t = tanh(W_xz x_t + W_hz h_{t-1} + b_z)     # input modulate (candidate), -1..1
C_t = f_t ∘ C_{t-1} + i_t ∘ z_t               # keep old part + add new part
h_t = o_t ∘ tanh(C_t)
```
ConvLSTM: same, with every `W·x` / `W·h` replaced by `Conv(x; w)` / `Conv(h; w)`; ∘ stays element-wise.

**Worked LSTM step (mock 3.3)**: f = [1, 0.5], Cₜ₋₁ = [2, −1], i = [0, 0.5], z = [0.4, 1], o = [0.5, 1]
- Cₜ = [1·2 + 0·0.4, 0.5·(−1) + 0.5·1] = **[2, 0]**
- hₜ = [0.5·tanh(2), 1·tanh(0)] = [0.5·0.964, 0] = **[0.482, 0]**
- Interpret: dim 1 keeps all old memory (f = 1), writes nothing (i = 0), reads out half (o = 0.5). Dim 2 deletes half of the old value and writes half of the new; they cancel to 0.

**Output size (each of T, H, W separately)**: **O = ⌊(I + 2P − K) / S⌋ + 1**. Channels out = number of filters.

**Worked shape (mock 3.4a)**: x (2, 3, 16, 112, 112), `Conv3d(3, 16, 3, stride=(1,2,2), padding=1)` then `MaxPool3d(2, 2)`
- T: ⌊(16 + 2 − 3)/1⌋ + 1 = 16; H = W: ⌊(112 + 2 − 3)/2⌋ + 1 = ⌊55.5⌋ + 1 = 56 → **(2, 16, 16, 56, 56)**
- MaxPool3d(2, 2): halves T, H, W → **(2, 16, 8, 28, 28)**
- Slide example: `Conv3d(16, 33, 3, stride=2)` on (20, 16, 10, 50, 100) → T ⌊7/2⌋+1 = 4, H ⌊47/2⌋+1 = 24, W ⌊97/2⌋+1 = 49 → **(20, 33, 4, 24, 49)**. With `kernel (3,5,2), stride (2,1,1), padding (4,2,0)` → **(20, 33, 8, 50, 99)**.

**Conv3d parameter count** = C_out · C_in · k_T · k_H · k_W + C_out (bias)
- `Conv3d(3, 16, 3)`: 16·3·27 + 16 = 1,296 + 16 = **1,312**
- `Conv3d(16, 33, 3)`: 33·16·27 + 33 = 14,256 + 33 = **14,289**
- (Extra, ConvLSTM cell, input 3, hidden 64, k = 3: `Conv2d(67, 256, 3)` = 256·67·9 + 256 = **154,624**.)

**3D conv by hand, method**: output depth = D_in − k_D + 1 (stride 1, no pad). For each output position: lay F[0], F[1], … over consecutive data slices, multiply element-wise, sum each slice, **add the slice sums**. Slide x, y **and z**. No filter flip (cross-correlation, like PyTorch).

**Small example (mock 3.4c)**: D depth 2, 3×3; F 2×2×2; stride 1 → output **1×2×2**
```
D[0] = 1 0 2   D[1] = 2 1 0     F[0] = 1  0   F[1] = 0 1
       3 1 0          0 1 3            0 -1          1 0
       0 2 1          1 0 2
```
| pos | D[0] patch ∘ F[0] | D[1] patch ∘ F[1] | sum |
|---|---|---|---|
| (0,0) | 1·1 + 1·(−1) = 0 | 1 + 0 = 1 | **1** |
| (0,1) | 0 − 0 = 0 | 0 + 1 = 1 | **1** |
| (1,0) | 3 − 2 = 1 | 1 + 1 = 2 | **3** |
| (1,1) | 1 − 1 = 0 | 3 + 0 = 3 | **3** |
**Output = [[1, 1], [3, 3]]**

**Slide example (slides 78–83)**: D 4×4×4, F 3×3×3 with every slice [[1,−1,−1],[−1,1,−1],[−1,−1,1]], stride 1 → **2×2×2**
- D[0] = [[3,2,1,3],[0,7,3,3],[3,9,3,4],[2,3,8,3]]; D[1] = [[0,2,1,3],[0,4,3,2],[3,9,3,4],[2,6,8,1]]
- D[2] = [[3,2,1,1],[0,3,3,3],[3,4,3,4],[2,3,6,3]]; D[3] = [[3,2,1,3],[0,3,6,3],[0,9,3,4],[2,3,1,3]]
- O[0][0,0], slice 0: [[3,2,1],[0,7,3],[3,9,3]] → 3 − 2 − 1 − 0 + 7 − 3 − 3 − 9 + 3 = **−5**; slice 1 = −11; slice 2 = −4 → **−20** (slide shows −21: it wrote −2 instead of −1·1).
- O[0][0,1]: −17 − 13 − **6** = **−36** (slide −37: third slice mis-added as −7).
- O[1][1,0]: −4 − 7 − 7 = **−18** (slide −22).
- **Correct: O[0] = [[−20, −36], [−15, −55]], O[1] = [[−24, −29], [−18, −55]]**
- 3D max pool 2×2×2, stride 2 on the same D: **O[0] = [[7, 3], [9, 8]], O[1] = [[3, 6], [9, 6]]** (slides correct).


### 4. Model answers

**Q: Spatial vs temporal; define spatio-temporal; classify CNN→LSTM, ConvLSTM, 3D CNN.**
Spatial features are the content inside one frame (edges, colour, shape, texture, position), extracted with 2D convolution. Temporal features are the changes between frames (motion, order, speed), extracted with RNN/LSTM. Spatio-temporal means both are extracted **at the same time (พร้อมกัน)**. (1) CNN→LSTM is **S→T**: the CNN processes each frame alone up to flatten, and only then does the LSTM see time. (2) ConvLSTM is **ST**: its convolution runs on the concat of xₜ and hₜ₋₁, so the 2D kernel sees spatial structure while the channels come from different times. (3) 3D CNN is **ST**: the kernel slides along x, y and time and covers several frames in one operation.

**Q: Choose an architecture for a small medical video dataset (e.g. 300 gait clips, normal/abnormal, unequal lengths).**
It depends on the dataset and the resources; here the dataset is small. I choose a **pretrained 2D CNN (ImageNet) per frame, cut at flatten/GAP → LSTM → last hidden state → FC (2-class softmax or 1 sigmoid)**. Why: 300 clips is too few for a 3D CNN, which has many parameters, overfits and has few pretrained models. A pretrained 2D CNN gives good spatial features through transfer learning, the LSTM adds the motion that one frame cannot show (sitting down vs standing up), and it needs less memory. Also acceptable with reasons: a **video-pretrained I3D** as feature extractor + LSTM/linear (no 3D training from scratch). Unequal lengths: sample a fixed number of frames across each clip (`max_frames`), or rely on the RNN's variable-length input (very long sequences still lose early info). Watch out: random augmentation is applied per frame, so it differs between frames.

**Q: Name the task type (compare to image tasks).**
Label the surgical phase of **every frame** → temporal action segmentation (like image segmentation; output at every time step; Edit / F1@k). Find the **start–end time** of bleeding → temporal action localization (like object detection; class + interval; tIoU / mAP). **Heart rate** from skin video → video regression (linear output). Normal/abnormal per clip, or fall vs no-fall per clip → video classification (take the last LSTM output). (Fall detection on long CCTV where you need *when* → localization.)

**Q: Why RNN? Limits of NN/CNN.**
NN/CNN assume the input arrives all at once, has no sequential relation, and has a fixed equal size. Video breaks all three: frames come in order, order changes meaning, and clips have different lengths. An RNN takes input step by step, keeps a hidden state that carries past information (H₀ = 0), reuses one cell with shared weights, and so handles sequences of any length. Its weakness is long sequences: early information fades, which motivates LSTM.

**Q: Roles in LSTM and why sigmoid for gates, tanh for data.**
Cell state = long-term memory that accumulates information from all steps; it is only scaled by f and added to, never squashed by tanh, so it keeps early information. Hidden state = short-term memory, "keep for predict", passed to the next step and to the output layer. Forget gate (Delete) decides how much of Cₜ₋₁ to keep; input gate (Write) how much new candidate zₜ enters; output gate (Read) how much of C goes out as hₜ. Gates use sigmoid because 0…1 acts as a weight (0 block, 1 pass all); data uses tanh because −1…1 normalises it around zero. Gate values are computed from xₜ and hₜ₋₁, so they adapt to every input.

**Q: Why does LSTM handle long sequences / vanishing gradient better than RNN?**
In an RNN the hidden state is recomputed through W and tanh at every step, so information and gradients shrink each step and early frames are lost. LSTM adds a cell state updated by Cₜ = fₜ∘Cₜ₋₁ + iₜ∘zₜ: an additive path with only element-wise scaling by f. When f ≈ 1, information and the gradient (∂Cₜ/∂Cₜ₋₁ = fₜ) pass back almost unchanged *(gradient part: standard fact)*.

**Q: What does ConvLSTM change from LSTM, and why?**
It keeps the LSTM equations (3 gates, cell and hidden states) but replaces the matrix multiplications on xₜ and hₜ₋₁ with convolutions, so inputs and states are 2D feature maps instead of flattened vectors. The gate products stay element-wise. Why: flattening a frame throws away spatial layout; with conv gates the memory keeps *where* things are while tracking *how* they change, giving spatio-temporal features.

**Q: 2D vs 3D conv; pros/cons of 3D CNN; why Two-Stream/I3D.**
A 2D filter slides over x, y and covers all channels at once; a 3D filter also slides along time and covers several frames, so each output mixes spatial and temporal information (ST). 3D CNN is more accurate on 3-D data but has many parameters, overfits small datasets (UCF-101, HMDB-51), has few pretrained models and costs memory/time. Two-Stream avoids heavy 3D training by using 2D ConvNets on RGB and on optical flow (motion) and fusing late. I3D inflates pretrained Inception-v1 filters to T×N×N, runs RGB + flow streams and averages the scores, giving a video-pretrained ST model.

**Q: Shapes (ConvLSTM loader → Conv3d).**
The ConvLSTM DataLoader (`batch_first=True`) gives (B, T, C, H, W), e.g. (2, 16, 3, 112, 112). `nn.Conv3d` expects (B, C, T, H, W), so call `x.permute(0, 2, 1, 3, 4)` first. Then compute each of T, H, W with ⌊(I + 2P − K)/S⌋ + 1; channels = number of filters.


### 5. Traps
- **CNN→LSTM is S→T, not ST.** Only ConvLSTM and 3D CNN are ST. Write "พร้อมกัน" in the definition.
- **Slide 3D-conv errors**: −21 → **−20**, −37 → **−36**, −22 → **−18**. Don't copy the slide grid; show the method.
- **Sigmoid 0…1 (gates), tanh −1…1 (data).** Swapping these loses marks; the teacher said "หาที่จด".
- **Forget gate = 1 means keep everything**, 0 means erase.
- **"Output modulate gate" is the teacher's own name**; officially forget, input, output gates + input modulate (candidate).
- **Slide 40's "Input Gate (Write)" text repeats the hidden-state definition**; use slide 38: it controls how much new data enters the cell state.
- **Hidden state ≠ prediction**; it goes through the optional forward (Linear/softmax) part.
- **Cell state holds "more", not "complete" info** (loss at forget and input gates).
- **Gate values are adaptive**; the W, b that compute them are fixed after training.
- **ConvLSTM**: only the x·W and h·W products become convolutions; ∘ stays element-wise.
- **Axis order**: ConvLSTM (B, T, C, H, W) vs Conv3d (B, C, T, H, W); permute. Slide 59's text order (B, Frame, W, H, C) is not PyTorch's.
- **Output-size formula applies to time too**: 4×4×4 with 3×3×3 → 2×2×2, not 2×2×4. Use the floor (55.5 → 55, +1 = 56).
- **Conv3d params include C_in and bias**: 16·3·27 + 16 = 1,312, not 16·27.
- **3D CNN is not automatically better**: small (medical) data → overfit; say "it depends on dataset/resources".
- **1×1×1 conv sees neither space nor time** — channel mixing only.
- **RNN "variable length" = it *can*, not that it does it well**; long sequences still fade.
- **Only one RNN cell**, reused; unfolded picture ≠ many different cells.
- **Segmentation (every frame) vs localization (only intervals of interest)** — don't mix them up; per-frame metrics vs tIoU.
- **GRU is not on the slides**: mention it as an alternative, don't build answers on it.

---

## Q4 — Image segmentation

### 1. What they will ask
- Compare classification / detection / segmentation (**What / Where**). Explain semantic vs instance vs panoptic, with a medical example of each.
- U-Net: encoder vs decoder (W×H, channels, What/Where), **why skip connections**, U-Net (concat) vs LinkNet (add) channel counts, **why the final layer is a 1×1 conv**.
- **Design the output layer** (number of 1×1 filters, activation, label format, loss) for binary / multiclass / multilabel / regression.
- Given a GT and a predicted mask: count TP/FP/FN/TN and compute **IoU, Dice, pixel accuracy**. Why pixel accuracy misleads, why Dice loss.
- Possibly: upsampling methods/unpooling, dilated conv + DeepLab (ASPP), PSPNet/PAN, AE properties.

### 2. Must-know
| Task | Answers | Where = |
|---|---|---|
| Classification | **What** (whole image, *image prediction*) | none |
| Detection | What + Where | smallest axis-aligned **bounding box** |
| Segmentation | What + Where | object **shape, per pixel** (*pixel prediction*) |

Teacher, "จำดีๆ" ("remember this well"): classification predicts from the image, segmentation predicts at **pixel level**.

| | Semantic | Instance | Panoptic (= semantic + instance) |
|---|---|---|---|
| Labels | every pixel, incl. background | **foreground only** | every pixel |
| Same-class objects | same label (can't separate) | separate mask each | **things** separated, **stuff** (sky, road, tissue) not |
| Use | **area/size** (tumour area in MRI) | **counting/tracking** (each nucleus/cell) | full scene (all tissue types + each gland/cell) |

- **Autoencoder (encoder–decoder)**: input and output have the **same W×H**, with any number of channels. Parts:
  - Encoder: compresses, does feature extraction.
  - **Code/latent**: the compressed representation.
  - Decoder: reconstructs.
- AE properties: data-specific, lossy, unsupervised (only for compression; **segmentation needs GT masks**). **Code size** is the most important hyperparameter: too big → copies input → overfitting.
- **Encoder** (conv + max pool): W×H ↓, channels ↑, learns **WHAT**, loses **WHERE**.
- **Decoder** (transposed conv / interpolation / unpooling): W×H ↑, channels ↓, recovers **WHERE**.
- Deep, small maps are rich in information but poor in spatial detail. Shallow, large maps are the reverse.
- **U-Net** (Ronneberger **2015**, built for **biomedical** images):
  - blocks: conv 3×3 + ReLU, maxpool 2×2, up-conv 2×2 (transposed conv);
  - **skip = copy + concatenate** at the same scale;
  - final layer: **1×1 conv** + sigmoid/softmax.
- **LinkNet** (2017): U-Net-like, but the skip is **element-wise addition** (same channels and W×H needed). Lighter and faster.
- **DeepLabV3** (2017):
  - **dilated/atrous conv** instead of more downsampling;
  - receptive field ↑ with no extra parameters: 3×3 at d=2 covers 5×5 with 9 weights, vs 25 for a real 5×5;
  - **ASPP** runs several dilation rates in parallel;
  - predicts first, then upsamples, so the **decoder is tiny**.
  - **V3+** (2018) adds a real decoder.
- Dilation numbering: **d = 1 is a normal conv**, d = 2 leaves a 1-pixel gap. Effective size = k + (k−1)(d−1).
- **PSPNet** (2017): pyramid pooling 1×1 (global), 2×2, 3×3… (local), then upsample + concat. **PAN** (2018): FPA (multi-scale by filter size) + GAU (channel weights; a weighted skip).
- Segmentation uses the **most GPU memory** and is the slowest of classification < detection < segmentation.

**Output layer (1×1 conv; "จำนวน Channel = จำนวนคลาส", "number of channels = number of classes"):**

| Task | 1×1 filters | Activation | Label | Loss |
|---|---|---|---|---|
| Binary (tumour vs bg) | **1** (or 2) | **sigmoid**, thr 0.5 (or softmax if 2) | 0/1 mask, same size as image | Dice / BCE+Dice / Jaccard |
| Multiclass (1 px → exactly 1 of N) | **N** | **softmax** (softmax2d) | class-label map (1 ch, 0..N−1) or one-hot | CE / multiclass Dice |
| Multilabel (1 px → several) | **N** | **sigmoid per channel**, > 0.5 | **must be one-hot** (N ch) | BCE / multilabel Dice |
| Regression (depth, thickness) | **# values** (1) | **linear / none** | continuous map | L1 / L2 |

### 3. Formulas + worked example
Pixel counts (unit = **pixels**): TP = object predicted object, FP = background predicted object, FN = object predicted background, TN = background predicted background.
```
IoU (Jaccard)  = |A∩B| / |A∪B|      = TP / (TP + FP + FN)
Dice (= F1)    = 2|A∩B| / (|A|+|B|) = 2TP / (2TP + FP + FN)
Pixel accuracy = (TP + TN) / all pixels
Precision = TP/(TP+FP)    Recall = TP/(TP+FN)
Dice = 2·IoU / (1 + IoU)   →   Dice ≥ IoU always;   IoU = Dice / (2 − Dice)
Dice loss = 1 − Dice       Jaccard loss = 1 − IoU
mIoU = mean of per-class IoU (one-vs-all per class)
```
(A = predicted foreground, B = GT foreground. The Ch 12 slides only name these metrics; the formulas are the standard ones used in the Ch 14 lecture.)

**Worked example (4×4 = 16 px):**
```
GT          Pred
0 1 1 0     0 0 1 1
0 1 1 0     0 1 1 1
0 1 1 0     0 0 1 0
0 0 0 0     0 0 0 0
```
- TP = 4 (overlap), FP = 2 (Pred=1, GT=0: (0,3), (1,3)), FN = 2 (GT=1, Pred=0: (0,1), (2,1)), TN = 16 − 8 = 8.
- IoU = 4/(4+2+2) = **0.50**
- Dice = 8/(8+2+2) = **0.667**. Check: 2·0.5/1.5 = 0.667 ✓
- Pixel accuracy = (4+8)/16 = **0.75**

**Mock 4.4 answer (6×6 tumour mask):** TP = 11, FP = 2, FN = 2, TN = 21. IoU = 11/15 = **0.733**, Dice = 22/26 = **0.846**, pixel accuracy = 32/36 = **0.889**.

**Ch 14 slide example (4×4):** TP = 5, FN = 2, FP = 1, TN = 8. Acc = 0.8125, Precision = 0.833, Recall = 0.714, Dice = 10/13 = 0.769, IoU = 5/8 = 0.625.

**Accuracy trap in numbers:** a tumour covering 2% of the image. Predicting "all background" gives pixel accuracy **0.98** but IoU = Dice = **0** (TP = 0).

### 4. Model answers
**Classification vs detection vs segmentation:**
> Classification answers only **What** for the whole image (image prediction: an FC layer over all features gives one label). Detection answers **What + Where**, with Where as the smallest bounding box around the object. Segmentation answers **What + Where at pixel level**: every pixel gets a label, so Where follows the true shape of the object. That is why segmentation suits measuring a lesion's area or boundary.

**Semantic / instance / panoptic + medical examples:**
> - **Semantic**: every pixel, including background, gets a class, but objects of the same class are merged. Good for measuring area/size, e.g. total tumour area in an MRI slice.
> - **Instance**: only foreground objects, each with its own mask, background unlabelled. Good for counting/tracking, e.g. counting each nucleus in histology or tracking cells in a video.
> - **Panoptic = semantic + instance**: every pixel is labelled. Stuff (no clear boundary, e.g. tissue background) is handled semantically and things (countable, e.g. each gland/cell) as instances.

**Encoder / decoder:**
> The encoder (contraction path: conv + max pooling) **reduces W×H and increases channels**. It learns *what* is in the image (rich features) but **loses spatial information (where)**. The decoder (expansion path: transposed conv / interpolation / unpooling) **increases W×H and reduces channels** until the output is the input's size, recovering *where*. The output is an image-sized map with one channel per class.

**Why skip connections; U-Net vs LinkNet (128 ch × 64² + 128 ch × 64²):**
> Downsampling in the encoder throws away spatial detail, so a decoder that works only from the bottleneck produces distorted, blurry boundaries. Skip connections pass high-resolution encoder features to the decoder **at the same scale** (64↔64, 128↔128). Shallow features are rich in spatial info, so object boundaries can be localised accurately. **U-Net concatenates**: 128 + 128 = **256 channels**, 64×64. **LinkNet adds element-wise**: shapes must match, giving **128 channels**, 64×64. Addition is lighter and faster.

**Why the final layer is a 1×1 conv (bonus quiz):**
> Segmentation is **pixel prediction**: the class of a pixel is decided from **that pixel's feature vector across all channels**. A 1×1 conv is a **fully connected layer applied to each pixel independently** with shared weights, so it maps C feature channels → N class channels while **keeping W×H**. An FC layer after flatten would mix all positions and give one answer per image. Number of 1×1 filters = number of classes. (The reason is not softmax; the activation is a separate choice that sets the task type.)

**Output-layer design (one line per case, give the why):**
> - (a) Liver tumour vs background: binary, **1 filter + sigmoid** (threshold 0.5), 0/1 mask, Dice or BCE+Dice loss because the tumour is small (imbalanced).
> - (b) 4 brain tissue classes, each pixel exactly one: multiclass, **4 filters + softmax**, class-label map (0–3) or one-hot, cross-entropy.
> - (c) Histology where a pixel can be several structures: multilabel, **4 filters + sigmoid per channel**, labels **must be one-hot** because a pixel can be 1 in several channels, BCE.
> - (d) Corneal thickness per pixel: regression, **1 filter + linear/no activation** (a sigmoid/softmax would squash the value), L1/L2 loss.

**Why not pixel accuracy; why Dice loss:**
> In medical images the lesion is tiny and **TN (background) dominates**, so predicting "all background" already gives about 98% pixel accuracy while finding no tumour (IoU = Dice = 0). IoU/Dice measure only the **overlap of the foreground** and ignore TN. **Dice loss = 1 − Dice** directly optimises that overlap and handles **class imbalance** (listed with PSPNet in the slides). Jaccard loss = 1 − IoU is the alternative in smp.

**Dilated conv / DeepLab (if asked):**
> Dilated (atrous) conv spaces out the kernel taps: d = 1 is normal, d = 2 leaves a 1-pixel gap. A 3×3 kernel at d = 2 covers 5×5 with only 9 weights instead of 25, so the **receptive field grows with no extra parameters or compute**. DeepLabV3 stops downsampling early and uses dilated conv instead, which keeps spatial resolution. ASPP runs several dilation rates in parallel for multi-scale context. Its decoder is a single upsample step, so V3+ adds a proper decoder.

### 5. Traps
- **Instance seg does NOT label background**; semantic does (by concept).
- U-Net skip = **concat** (channels add up), LinkNet = **add** (channels unchanged). Don't swap them.
- **Channels = classes**, except binary + sigmoid, which uses **1**.
- Multiclass → softmax. Multilabel → sigmoid + one-hot. The slide code p.87–90 wrongly passes `activation='sigmoid'` for multiclass; the teacher admitted it.
- Regression → **linear/no activation**, never sigmoid/softmax.
- 1×1 conv answer is about **per-pixel FC over channels**, not "because of softmax" or "because the output is an image" (that was "เกือบถูก", "almost right").
- "AE is unsupervised": true only for compression. Segmentation needs masks.
- Encoder/decoder **need not be symmetric** (DeepLabV3).
- **Dilation 1 = normal conv** (not 0). Dilated conv ≠ morphological dilation.
- Dice ≥ IoU for the same masks. If your Dice < IoU, you made an arithmetic error.
- TN counts **background pixels predicted background**; it is not in IoU/Dice. Count GT and Pred pixels first, then TN = total − (TP + FP + FN).
- Max unpooling only works if pooling **indices were saved**.

---

## Q5 — Transposed convolution by hand

### 1. What they will ask
- Given a small X (2×2 or 3×3) and K (2×2 or 3×3), compute the **full output matrix** for stride s, padding 0. Show the X[i,j]·K blocks, where each one goes, and the summed overlaps. (Mock: 6 of 14 marks.)
- Same X, K with **padding 1**: output size and values. What padding does in transposed conv.
- Fill an **output-size table** (input, k, s, p → output). Pick k, s, p to upsample e.g. 32×32 → 64×64 (U-Net decoder).
- True/false: slide p.41 "ยิ่งค่า padding มากขึ้น ขนาดของภาพผลลัพธ์จะยิ่งขยายใหญ่ขึ้น" ("more padding, bigger output"). Is transposed conv the inverse of conv?
- Possibly: nearest-neighbour upsampling by hand (2×2 → 5×5), and the 3 upsampling methods.

### 2. Must-know
- **Transposed convolution = Deconvolution** (same thing in this course; "ให้รู้ว่ามันคือสิ่งเดียวกัน", "know they are the same thing"). It is the most popular decoder upsampling operator, used in AE decoders, U-Net "up-conv 2×2" and GANs.
- **Kernel is NOT flipped/rotated** ("ไม่หมุนตัวกรอง"). Kernel weights are **learned** in training.
- Mechanism: multiply each input value by the **whole kernel**, place the block shifted by **stride**, **sum overlaps**.
- What changes the output size:

| Increase... | Output | Why |
|---|---|---|
| stride s | **bigger** | blocks are spaced further apart |
| kernel k | **bigger** | each block is bigger |
| padding p | **smaller** | p rows/cols are **cut off** each side (−2p) |

- Overlap only happens when **s < k**. If s = k the blocks tile with no overlap. If s > k, gaps of zeros appear *(standard fact)*.
- It reverses the **shape** change of a conv, not the **values**. It is **not** the inverse of convolution.
- The 3 decoder upsampling methods (a network uses one, at most two):
  1. **Interpolation** (nearest-neighbour, bilinear): no learnable weights.
  2. **Transposed conv**: learnable.
  3. **Max unpooling**: needs the **max-pool indices (mask/location) saved**, puts each value back at its old position, fills **0** elsewhere, then a 3×3/5×5 conv smooths it.

### 3. Formulas + worked examples

**Output size** (H = input height/width, k = kernel size, s = stride, p = padding):
```
Full (before crop)  = (H − 1)·s + k
Output              = (H − 1)·s − 2p + k        (+ output_padding, PyTorch)
```
- `output_padding` *(standard fact)*: adds 1 extra row/col on the bottom/right only, to hit an even size. Example: 32, k=3, s=2, p=1 → 63; with output_padding=1 → **64**.
- Compare normal conv: `⌊(H + 2p − k)/s⌋ + 1`. Here padding **grows** the output, the opposite of transposed conv.

**Procedure on paper (memorise):**
1. Write the size: full = (H−1)·s + k, final = full − 2p. Draw an empty full grid.
2. For each input X[i,j], write the block **X[i,j]·K** (kernel not flipped).
3. Put the block's top-left corner at **(i·s, j·s)** in the full grid.
4. **Add** all values that land on the same cell.
5. If p > 0, **cut p rows/cols from every side**. What is left is the output.
6. Self-check:
   - final size matches the formula;
   - top-left of the full grid = X[0,0]·K[0,0];
   - with p = 0, **sum of all output values = sum(X) × sum(K)** *(standard fact; every product appears exactly once)*.

**Example A: s = 1, p = 0.** X = [[1, −1], [2, 3]], K = [[1, 2], [0, 1]]. Size = (2−1)·1 + 2 = **3×3**.
```
1·K @(0,0)   -1·K @(0,1)    2·K @(1,0)   3·K @(1,1)
| 1 2 |      | -1 -2 |      | 2 4 |      | 3 6 |
| 0 1 |      |  0 -1 |      | 0 2 |      | 0 3 |
```
- Row 0: [1, 2 + (−1), −2] = [1, 1, −2]
- Row 1: [0 + 2, 1 + 0 + 4 + 3, −1 + 6] = [2, 8, 5]. The centre (1,1) gets all 4 blocks.
- Row 2: [0, 2 + 0, 3] = [0, 2, 3]
```
Output = |  1  1 -2 |
         |  2  8  5 |      check: sum = 20 = sum(X)·sum(K) = 5·4 ✓
         |  0  2  3 |
```

**Example B: s = 2, p = 1.** X = [[2, 1], [1, 3]], K = [[1, 2, 1], [0, 1, 0], [1, 0, 2]].
Full = (2−1)·2 + 3 = 5×5. Final = 5 − 2·1 = **3×3**.
```
2·K rows0-2,cols0-2   1·K rows0-2,cols2-4   1·K rows2-4,cols0-2   3·K rows2-4,cols2-4
| 2 4 2 |             | 1 2 1 |             | 1 2 1 |             | 3 6 3 |
| 0 2 0 |             | 0 1 0 |             | 0 1 0 |             | 0 3 0 |
| 2 0 4 |             | 1 0 2 |             | 1 0 2 |             | 3 0 6 |
```
- Column 2 is shared by the left and right blocks. Row 2 is shared by the top and bottom blocks. Cell (2,2) is shared by all 4.
- Row 0: [2, 4, 2+1, 2, 1] = [2, 4, 3, 2, 1]
- Row 1: [0, 2, 0+0, 1, 0] = [0, 2, 0, 1, 0]
- Row 2: [2+1, 0+2, 4+1+1+3, 0+6, 2+3] = [3, 2, 9, 6, 5]
- Row 3: [0, 1, 0+0, 3, 0] = [0, 1, 0, 3, 0]
- Row 4: [1, 0, 2+3, 0, 6] = [1, 0, 5, 0, 6]
```
Full 5×5 (p=0):          crop 1 each side → Output 3×3 (p=1):
| 2 4 3 2 1 |
| 0 2 0 1 0 |            | 2 0 1 |
| 3 2 9 6 5 |            | 2 9 6 |
| 0 1 0 3 0 |            | 1 0 3 |
| 1 0 5 0 6 |
```
Check (full): sum = 56 = sum(X)·sum(K) = 7·8 ✓. With output_padding=1 you get a 4×4 output: the same crop, but nothing is cut from the bottom/right.

**Slide example (p.29–40):** X = [[1,1,1,1],[1,1,1,1],[3,1,2,2],[3,1,2,2]], K = 3×3 all ones, s = 2, p = 0 → **9×9**:
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
With all-ones X and K, cells where 2 blocks overlap are ×2 and where 4 overlap are ×4. All-ones X, s = 1 gives 6×6 with max 9 (the centre 2×2).

**Size table** (all verified with `F.conv_transpose2d`):

| input | k | s | p | calc | output |
|---|---|---|---|---|---|
| 1×1 | 3 | 1 | 0 | 0 + 3 | 3×3 |
| 3×3 | 3 | 2 | 1 | 4 − 2 + 3 | 5×5 (slide p.51) |
| 4×4 | 3 | 1 | 0 / 1 / 2 | 3 − 2p + 3 | 6 / 4 / 2 |
| 4×4 | 3 | 2 | 0 / 1 / 2 | 6 − 2p + 3 | **9** / 7 / 5 |
| 4×4 | 3 | 3 | 0 | 9 + 3 | 12×12 (no overlap) |
| 16×16 | 2 | 2 | 0 | 30 + 2 | 32×32 |
| 8×8 | 4 | 2 | 1 | 14 − 2 + 4 | 16×16 |
| 5×5 | 3 | 1 | 1 | 4 − 2 + 3 | 5×5 (same size) |
| 32×32 | 2 | 2 | 0 | 62 + 2 | **64×64** (U-Net up-conv 2×2) |
| 32×32 | 4 | 2 | 1 | 62 − 2 + 4 | **64×64** |

**Rule of thumb:** k = s and p = 0 doubles the size exactly when s = 2. So does k = 4, s = 2, p = 1.

**Nearest-neighbour vs bilinear** (X = [[10,20],[30,40]] → 5×5):
- Nearest, steps:
  - source coordinates = 0, 1/4, 2/4, 3/4, 1 (step 1/(n−1) = 0.25);
  - index = ⌊src + 0.5⌋ = [0, 0, 1, 1, 1] on both axes;
  - copy the values: rows 0–1 = `10 10 20 20 20`, rows 2–4 = `30 30 40 40 40`.
  - Result: discrete, blocky, copied values.
- Bilinear: smooth, linear in both directions. First row `10 12.5 15 17.5 20`, last row `30 32.5 35 37.5 40`, centre 25. The calculation was **not taught**, so only describe it.
- Both have **no learnable weights**. Transposed conv **learns** its kernel.

### 4. Model answers
**Full matrix (e.g. mock 5.1):**
- Write all 4 blocks X[i,j]·K with their row/col ranges (as in Example B).
- Show the summed rows, write the full matrix, then verify the size with (H−1)·s − 2p + k.
- Mock answer: X = [[1,2],[3,−1]], K = [[1,0,1],[2,1,0],[0,1,1]], s = 2, p = 0. The result is
  `[[1,0,3,0,2],[2,1,4,2,0],[3,1,3,2,1],[6,3,−2,−1,0],[0,3,3,−1,−1]]` (5×5).
- With p = 1, take the centre 3×3: `[[1,4,2],[1,3,2],[3,−2,−1]]`.

**"What does padding do in transposed conv?"**
> Output = (H−1)·s − 2p + k, so padding p **removes p rows/columns from each border** of the full output and the output gets **smaller** by 2p. This is the opposite of normal convolution, where padding adds zeros at the border to stop the output from shrinking. Example: 4×4, k=3, s=2 gives 9×9 at p=0, 7×7 at p=1 and 5×5 at p=2. So padding is not used to enlarge. Stride and kernel size enlarge.

**"Slide: more padding → bigger output. True or false?"**
> **False.** From (H−1)·s − 2p + k, a larger p gives a smaller output. The deck's own PyTorch results (p.46–47: p = 0/1/2 → 9/7/5 at s=2) and the teacher ("ยิ่ง Padding เพิ่ม ข้อมูลก็จะยิ่งหดลง", "the more padding, the smaller the output") agree. The stride part of that slide is correct: larger stride, larger output.

**"Is transposed conv the inverse of conv?"**
> **No.** It only reverses the **shape** change: small → large W×H, the same way conv goes large → small. It does not recover the original pixel values. Its kernel weights are learned by backprop to produce whatever the decoder needs. It is also called deconvolution in this course, but it is not a mathematical inverse.

**"Choose k, s, p for 32×32 → 64×64":**
> - (1) k=2, s=2, p=0: 31·2 + 2 = 64 (U-Net's up-conv 2×2, no overlap).
> - (2) k=4, s=2, p=1: 31·2 − 2 + 4 = 64.
> - (3) k=3, s=2, p=1 gives 63, so it needs output_padding=1 to reach 64 *(standard fact)*.

**"Why does stride 3 with k=3 give no overlap?"**
> Blocks start every s = 3 cells and each block is k = 3 wide, so they sit side by side exactly. Overlap needs s < k.

### 5. Traps
- **Flipping the kernel.** Don't. Use K as given (ไม่หมุน).
- **Slide p.41 is wrong**: more padding gives a **smaller** output, not bigger.
- Forgetting the **crop** when p > 0. Compute the full (H−1)·s + k grid first, then cut p from **each** side (2p in total).
- Placing blocks at (i, j) instead of **(i·s, j·s)** when s > 1.
- Missing the 4-way overlap cell. With s = 2 and k = 3, cell (2,2) gets a contribution from all 4 blocks of a 2×2 input.
- Using the normal-conv formula ⌊(H+2p−k)/s⌋+1 by mistake.
- Showing only the answer. The exam says "แสดงวิธีทำ" (show your work): write the blocks, their positions, the per-row sums and the size check.
- Negative inputs: keep the signs through the sums (−1·K gives a negative block).

---

## Q6 — Object detection

Sources: Ch 11 notes (อาจารย์แบงค์), Ch 14 §2 (พี่ใหม่, metrics), mock Q6, exam-hints. All numbers below were checked with Python (`torchvision.ops.box_iou` / `nms`).

### 1. What they will ask

The mock Q6 (14 pts) had 4 parts, and the real one will probably look the same:
- **IoU by hand** from two boxes (often in different formats), then TP/FP/FN at a threshold, then **convert to a YOLO label**.
- **NMS step by step**: reference box, IoU per pair, which box is deleted, final boxes, and how the result changes if the threshold changes. There is usually a "why wasn't D deleted?" chain question.
- **R-CNN family comparison**: Where comes from? CNN runs per image? How is region size standardised? Which classifier? Why is Fast R-CNN still multi-stage? Count **anchors and 1×1 filters**.
- **YOLO**: output tensor size for given S, B, C; what one cell holds; the loss parts; one-stage vs two-stage for a medical case (real-time polyp video); YOLOv1's weaknesses.
- Possible extras: RoI pooling by hand, R-CNN's 4 steps and why warping is needed, the RPN's 6 steps, P/R/AP/mAP. The teacher stressed **"why"**: "ตอบ Why the reason ไม่ได้" is what makes students fail.

### 2. Must-know

**Task types.** The teacher's line: "ไม่ได้ต่างกันที่ what ต่างกันที่ where"

| | What | Where |
|---|---|---|
| Classification | yes | no |
| Detection | yes | **smallest rectangle that fully covers the object** (bounding box) |
| Segmentation | yes | the object's **actual shape** (exact outline) |

Detection output per object: box + class + confidence (e.g. "dog 0.98").

**Box formats** ("3 formats on the market", so check what the library expects):

| Format | Values |
|---|---|
| xyxy | (x1, y1, x2, y2) = top-left and bottom-right corners |
| xywh (corner) | (x, y, w, h) with (x, y) = top-left corner |
| cxcywh (centre) | (xc, yc, w, h) |
| **YOLO label** | `class xc/W yc/H w/W h/H`, all in 0–1, one row per object, one `.txt` per image |

Conversions: `w = x2−x1`, `h = y2−y1`, `xc = x1 + w/2`, `yc = y1 + h/2`. Going back: `x1 = xc − w/2`, `x2 = xc + w/2`.
YOLO labels are normalised so that **resizing the image does not change the label**.

**Two families**
- **Multi-stage** (R-CNN, Fast R-CNN, Faster R-CNN): Where and What come from **separate modules/stages**.
- **Single-stage** (YOLO): a **unified network, trained end-to-end**, gives What and Where **at the same time**.

**R-CNN (Girshick 2014): 4 steps**
1. **Selective Search** gives about 1,000–2,000 region proposals. It starts from graph-based segmentation, then merges adjacent regions hierarchically by similarity of **colour, texture, size and fill**. It stops at a **target number of proposals, not a fixed number of iterations**, because image complexity varies. Each region becomes a box via min/max of x and y.
2. **Warping**: resize every region to **224×224**. *Why:* CNNs of that time had **no Global/Adaptive Average Pooling**, so different input sizes would give different flattened lengths, and the FC layer needs a fixed length. *Cost:* the object **loses its aspect ratio**.
3. **Features**: AlexNet with the **last FC removed** gives a feature vector.
4. **SVM** per class (maximum-margin hyperplane `WX + b = 0`).
- No box regressor (course version). **Pain point:** the CNN runs **~2,000 times per image**, so it is slow.

**Fast R-CNN: fix = run the CNN once**
- The CNN runs on the **whole image once** and gives one feature map. Selective Search still runs on the original image.
- **RoI projection**: map each proposal onto the **feature map** using relative coordinates (0–1).
- **RoI Pooling** (max) turns crops of different sizes into a fixed size, then flatten → FC → **Softmax** (What) + **linear bbox regressor** (x, y, w, h), which *adjusts* Where ("ขยับยุกยิกนิดนึง ยืดหดนิดนึง").
- Training 84 h → 9.5 h (8.8×). Test without proposals 47 s → 0.32 s (146×). **With Selective Search 50 s → 2 s (only 25×)**.
- **Still multi-stage, not real-time:** Selective Search is now the bottleneck, and it is **not trainable**.

**Faster R-CNN (Ren 2015): fix = replace Selective Search with an RPN**
- The **Region Proposal Network** is trainable. Everything else is the same as Fast R-CNN.
- RPN, 6 steps:
  1. Intermediate **3×3 conv, 512 filters** (turns What-features into Where-features).
  2. **Anchors**: pre-defined boxes from **scale × aspect ratio = 3 × 3 = 9** per location. Belief: **every feature-map point is a possible object centre**.
  3. Classification layer: foreground/background **objectness score**.
  4. Regression layer: refines [x, y, w, h].
  5. Keep foreground anchors, sort by score, keep the **top n (~2,000 or 6,000)**.
  6. **NMS**.
- Implementation: 1×1 conv with **k filters → scores**, 1×1 conv with **4k filters → coordinates**. **"Naming by use"** ("ตัวเปลี่ยนโลก"): a 1×1 conv output becomes a "score" because the loss trains it into that role. **YOLO was born from this idea.**
- Regressor outputs are **adjustments**: x, y offsets are **added**, w, h are **multiplied** (ratio).
- 0.2 s/image, **250×** faster than R-CNN. mAP VOC07: 66.0 / 66.9 / **66.9**, so the gain is **speed**, not accuracy.
- Still **two-stage** even though it is fully trainable.

**YOLO (Redmon 2015): single-stage**
- The image is split into an **S×S grid**. The cell holding the object's **centre** is responsible for it. (This is the same belief as the RPN, but on the original-image grid instead of the feature map.)
- Each cell predicts **B boxes × (x, y, w, h, confidence)** plus **one set of class probabilities P(class|object) per cell**.
  - x, y are offsets within the cell. w, h are divided by the image W, H. All values are in 0–1.
  - Of the B boxes, the one with the **highest IoU with the GT** is responsible.
- Final score = P(class|object) × box confidence, then **NMS** (adds about 2–3% mAP).
- YOLOv1: **24 conv + 2 FC**, S = 7, B = 2, C = 20 → **7×7×30**.
- **Loss = 5 SSE terms**:
  1. (x, y) error, × λ_coord
  2. (√w, √h) error, × λ_coord. The square root reduces the large-box vs small-box imbalance.
  3. Confidence where there is an object
  4. Confidence where there is no object, × **λ_noobj = 0.5**
  5. Class probabilities (only in cells that contain an object)
  - **λ_coord = 5** boosts localisation. The loss covers objectness/confidence as well as class.
- **Weaknesses:** **one class per cell** (objects of different classes in one cell), and **many small objects clustered together** (cells are large).
- YOLOv8 (Ultralytics 2023): sizes n/s/m/l/x. Bigger = more accurate and more resources. n has 3.2 M params and suits mobile/embedded devices.

**Two-stage vs one-stage**

| | Two-stage (Faster R-CNN) | One-stage (YOLO) |
|---|---|---|
| Pipeline | RPN proposals → RoI pooling → head | one network, grid predicts boxes + classes directly |
| What & Where | separate stages | **simultaneously**, end-to-end |
| Speed | 0.2 s/img (not real-time) | real-time |
| Dense / small objects | better (9 anchors at every feature-map location) | YOLOv1 weak (1 class per cell, large cells) |
| Use when | accuracy on crowded small objects, offline | video, real-time, edge devices |

**Metrics (Ch 14, พี่ใหม่)**
- **TP**: correct class **and** IoU ≥ threshold (commonly 0.5). **FP**: wrong class, **or** IoU < threshold, **or** no real object (also duplicates). **FN**: a GT object that was missed. **TN is not used**, because there are too many background regions.
- Precision = TP/(TP+FP). Recall = TP/(TP+FN). The counting unit is **objects (boxes)**.
- **AP** = area under the P–R curve for **one class**, built by sweeping the **confidence threshold**. **mAP** = mean AP over classes. mAP@0.5 uses IoU 0.5. mAP@0.5:0.95 averages over IoU thresholds 0.5–0.95 (YOLOv8 "mAP 50–95").

### 3. Formulas + worked examples

**IoU** = intersection / (area1 + area2 − intersection).
For xyxy boxes: `ix1 = max(x1)`, `iy1 = max(y1)`, `ix2 = min(x2)`, `iy2 = min(y2)`, `inter = max(0, ix2−ix1) · max(0, iy2−iy1)`.

*Worked example (with conversion).* GT is given as corner xywh (20, 30, 60, 40). The prediction is given as cxcywh (60, 55, 50, 50).
1. Convert GT: x2 = 20 + 60 = 80, y2 = 30 + 40 = 70, so GT = (20, 30, 80, 70). Area = 60·40 = **2,400**.
2. Convert prediction: x1 = 60 − 25 = 35, x2 = 85, y1 = 55 − 25 = 30, y2 = 80, so P = (35, 30, 85, 80). Area = 50·50 = **2,500**.
3. Intersection: x from max(20, 35) = 35 to min(80, 85) = 80, width 45. y from max(30, 30) = 30 to min(70, 80) = 70, height 40. inter = **1,800**.
4. Union = 2,400 + 2,500 − 1,800 = **3,100**. **IoU = 1,800/3,100 = 0.581**.
5. At threshold 0.5 with the correct class, this is a **TP**. At 0.75 it is an **FP**, and the GT becomes an **FN**.

*Mock 6.1 check:* G = (40, 40, 100, 100), P = (60, 50, 120, 110). inter = 40·50 = 2,000, union = 3,600 + 3,600 − 2,000 = 5,200, **IoU = 0.385** < 0.5, so P is an FP and G is an FN. YOLO label for G in a 512×512 image: `0 0.1367 0.1367 0.1172 0.1172` (70/512, 70/512, 60/512, 60/512).

*Notes check:* (0, 0, 4, 4) vs (2, 2, 6, 6) gives 4/28 = **0.143**. The YOLO label for xyxy (100, 50, 300, 250) in a 640×480 image is `0 0.3125 0.3125 0.3125 0.4167`.

**NMS** (slide steps 2.6.1–2.6.4):
1. Sort boxes by score, high to low.
2. Take the top box as the reference and compute its IoU with every remaining box.
3. If IoU > threshold, it is the same object: **delete the lower-score box**.
4. Move to the next *surviving* box and repeat.

*Worked example (threshold 0.5).* Boxes A–C are 100×100 (area 10,000). D and E are 80×80 (area 6,400).

| Box | xyxy | score |
|---|---|---|
| A | (50, 50, 150, 150) | 0.92 |
| B | (70, 60, 170, 160) | 0.85 |
| C | (95, 70, 195, 170) | 0.80 |
| D | (300, 80, 380, 160) | 0.75 |
| E | (310, 90, 390, 170) | 0.60 |

| Pair | intersection | union | IoU |
|---|---|---|---|
| A, B | 80·90 = 7,200 | 20,000 − 7,200 = 12,800 | **0.563** |
| A, C | 55·80 = 4,400 | 15,600 | **0.282** |
| B, C | 75·90 = 6,750 | 13,250 | **0.509** |
| D, E | 70·70 = 4,900 | 12,800 − 4,900 = 7,900 | **0.620** |
| A or B or C vs D or E | no overlap | | 0 |

- Sorted: A, B, C, D, E.
- **Reference A:** B 0.563 > 0.5, so **delete B**. C 0.282, keep. D 0, keep. E 0, keep.
- **Reference C:** D 0, keep. E 0, keep.
- **Reference D:** E 0.620 > 0.5, so **delete E**.
- **Kept: A, C, D.**
- *Why does C survive although IoU(B, C) = 0.509 > 0.5?* B was already deleted by A. NMS compares only with **surviving higher-score** boxes, and IoU(A, C) = 0.282 is below the threshold.
- Threshold 0.25: C is also deleted (0.282 > 0.25), giving **A, D**. Threshold 0.6: B survives (0.563 < 0.6), giving **A, B, C, D**.
- Rule: a **lower threshold deletes more** (risk: you remove a real object lying nearby). A **higher threshold leaves more duplicates**. Ultralytics `iou` default = 0.6.
- (Mock 6.2 answer for reference: 0.5 gives A, C, D; 0.4 gives A, C.)

**Anchors and RPN filters**
- k = scales × ratios = 3 × 3 = **9** anchors per location. Total anchors = H_fm × W_fm × k.
  - 10×10 → **900**. 32×32 → **9,216**. 50×50 → **22,500** (not "~30,000").
- 1×1 conv filters, as the slides give it: **k for objectness scores, 4k for coordinates**. For k = 9 that is **9 and 36** (in-class quiz answer). Per anchor that is 5 values (1 score + x, y, w, h).
- *(The original paper uses **2k** scores, i.e. fg + bg softmax, so 18 for k = 9. Write k = 9 as the slides do, and mention 2k only as a side note.)*

**RoI pooling cell sizes**
- Split each side of length m into n parts of size [m/n], with the **remainder in the last part**. Then take the **max** of each part.
- Slide 24 (round down): a 5 × 7 region pooled to 2×2 splits rows **2 + 3** and columns **3 + 4**. Result `[0.85, 0.84, 0.97, 0.96]`.
- Teacher's spoken correction (round up): 8/3 → **3, 3, 2**. In the exam, **state your rounding rule**.
- *Worked example:* pool this 4×5 region to 2×2.
  ```
  1 5 2 0 3
  4 2 7 1 1
  0 6 3 8 2
  9 1 2 4 5
  ```
  - Round down: rows 2 + 2, columns 2 + 3. Max of each block gives **[[5, 7], [9, 8]]**, flattened to [5, 7, 9, 8].
  - Round up: columns 3 + 2. This gives **[[7, 3], [9, 8]]**.

**YOLO output size**: per cell = **B·5 + C**. Tensor = **S × S × (B·5 + C)**.
- S = 7, B = 2, C = 20 → 30 → **7×7×30** (1,470 values).
- C = 3 → 13 → **7×7×13**. C = 10 → **20 filters** (the teacher's example).
- Layout per cell: values 1–5 = box 1 (conf, x, y, w, h), 6–10 = box 2, then C class probabilities (**one set per cell**).

**Precision / Recall / AP** (Ch 14 slide example)
- Birds: 12 detections, 10 correct, 15 real birds. P = 10/12 = **0.833**, R = 10/15 = **0.667**, F1 = 2PR/(P+R) = **0.741**.
- AP(bird): sweep the confidence threshold (20 GT birds), e.g. at conf ≥ 0.85, TP 6 and FP 1 give P = 0.86, R = 0.30. The trapezoid area under the P–R curve, starting at (R 0, P 1), is **0.773**.
- mAP = (Bird 0.773 + Car 0.883 + Dog 0.842)/3 = **0.833**.
- A higher confidence threshold gives higher P and lower R. A lower threshold gives higher R, more FP and lower P.

### 4. Model answers

**Q: Difference between classification, detection and segmentation?**
Classification gives only *What* (the class of the image). Detection and segmentation give both *What* and *Where*, and they differ only in Where. Detection's Where is the smallest rectangle that fully covers each object (a bounding box with class and confidence). Segmentation's Where is the object's actual shape, pixel by pixel. So for measuring a tumour's area or volume use segmentation, and for finding and counting lesions a box is enough.

**Q: R-CNN's steps; why is warping needed?**
(1) Selective Search proposes ~2,000 regions by hierarchically merging similar adjacent regions (colour, texture, size, fill). (2) Each region is warped to 224×224. (3) AlexNet without its last FC layer extracts a feature vector. (4) One SVM per class classifies it. Warping is needed because CNNs then had no global/adaptive average pooling: different input sizes would give different flattened lengths, while the FC layer needs a fixed length. The cost is a distorted aspect ratio. R-CNN is slow because it runs the CNN about 2,000 times per image.

**Q: Why is Fast R-CNN faster, and why is it still multi-stage / not real-time?**
Fast R-CNN runs the CNN **once on the whole image**, projects the proposals onto the feature map (relative coordinates), and uses RoI max pooling instead of warping. It classifies with FC + softmax instead of an SVM and adds a bbox regressor. Test time drops from 47 s to 0.32 s (146×) excluding proposals. But Where still comes from **Selective Search**, a separate, untrainable stage, so it is multi-stage. With Selective Search included it takes 2 s/image (only 25×), which is not real-time.

**Q: What does Faster R-CNN change? Explain the RPN.**
It replaces Selective Search with a **trainable Region Proposal Network** that shares the backbone feature map. A 3×3 conv (512 filters) adapts the features for Where. At every feature-map location, 9 anchors (3 scales × 3 ratios) are assumed. A 1×1 conv with k filters gives objectness scores, and one with 4k filters gives box offsets (x, y added; w, h scaled). The top ~2,000 foreground anchors by score are kept and NMS removes duplicates. The rest is Fast R-CNN. The result is 0.2 s/image (250× faster than R-CNN) with the same mAP (66.9), so the gain is speed. It is still two-stage.

**Q: Feature map 32×32, 3 scales × 3 ratios: anchors and filters?**
k = 3 × 3 = 9 anchors per location. Total = 32 × 32 × 9 = **9,216 anchors**. 1×1 conv filters: **9** for objectness scores (one per anchor type) and **36 = 9 × 4** for coordinates (x, y, w, h per anchor).

**Q: How does YOLO detect? Output size for S = 7, B = 2, C = 3?**
YOLO is a single-stage, end-to-end network. It divides the image into a 7×7 grid, and the cell containing an object's centre is responsible for it. Each cell predicts B = 2 boxes, each (conf, x, y, w, h), plus one class distribution P(class|object). Per cell that is 2·5 + 3 = 13, so the output is **7×7×13**. x, y are offsets in the cell and w, h are normalised by the image size. Final score = class prob × confidence, followed by NMS. What and Where come out together, so it is fast.

**Q: Explain the YOLO loss.**
It is a sum of 5 squared-error terms. Localisation: (x, y) and (√w, √h), both weighted by λ_coord = 5 so box accuracy matters more. The square root makes an error on a small box count more than the same error on a big box. Confidence where there is an object, and confidence where there is no object, weighted by λ_noobj = 0.5 because most cells are empty and would otherwise dominate. Class probabilities count only in cells that contain an object.

**Q: Real-time polyp detection in colonoscopy video: one- or two-stage?**
One-stage (YOLO). It predicts What and Where together in one network, so it reaches video frame rates, while Faster R-CNN needs ~0.2 s/frame. A small variant (YOLOv8n/s) suits embedded devices next to the endoscope. Watch YOLOv1's weaknesses: only one class per cell, and poor results on many small polyps clustered close together. For a screening task, recall matters (a missed polyp is an FN), so tune the confidence threshold accordingly.

**Q: Is a prediction a TP? Why is TN not used? What is mAP?**
It is a TP only if the class is correct **and** IoU with an unmatched GT ≥ threshold (e.g. 0.5). Otherwise it is an FP, and any unmatched GT is an FN. TN is not used because there are countless background regions. AP is the area under the precision–recall curve of one class, obtained by sweeping the confidence threshold. mAP is the mean AP over classes, reported at an IoU threshold (mAP@0.5, or @0.5:0.95).

### 5. Traps

- **Union = A + B − intersection.** Do not divide by A + B. Convert every box to xyxy **before** computing IoU, and check the format given (corner xywh vs centre vs xyxy).
- If the boxes don't overlap, the intersection is **max(0, …)** = 0, so IoU = 0. Never use a negative width.
- **NMS deletes the lower-score box** when IoU **>** threshold, and compares only against **surviving** boxes. A box deleted earlier can't delete anything. Boxes with IoU = 0 are different objects, so keep both.
- A lower NMS threshold deletes **more** boxes, not fewer. Ultralytics `iou` = the **NMS** threshold (default 0.6), not an mAP threshold. `conf` default 0.001 is "not recommended".
- **Slide error:** "Faster R-CNN is more accurate". The table shows 66.0 / 66.9 / 66.9 mAP, so its gain is **speed** (250×).
- **Spoken error:** "~30,000 anchors" for 50×50. The correct figure is 50·50·9 = **22,500**.
- Filters: **k scores, 4k coordinates** (9 / 36). Don't write 9 × 5 = 45 as "the coordinate filters". If you mention the paper's 2k, label it as such.
- **Fast R-CNN still uses Selective Search**, so it is still multi-stage. **Faster R-CNN is still two-stage** (not single-stage) even though it is trainable. Only YOLO is single-stage.
- Fast R-CNN speed-up: 146× **excluding** proposals, only **25×** including Selective Search. Training 8.8×.
- R-CNN = **SVM** classifier, AlexNet without its last FC, **no box regressor** (course version). Boxes sit on the **original image** and are warped. Fast/Faster place them on the **feature map** and use RoI pooling.
- The spelling "Wrapping" on the slides means **warping** (resizing).
- RoI pooling = **MAX**, not average ("สูงสุดนะ ไม่ใช่เฉลี่ยนะ"). The remainder goes in the **last** part. State your rounding rule.
- Regressor outputs are **adjustments** (x, y added, w, h multiplied), not new coordinates. The regressor refines Where; it does not create it.
- YOLO: class probabilities are **per cell**, confidence is **per box**. B = 2 but only **one class per cell**. Per cell = B·5 + C (not B·(5 + C)).
- Don't swap λ: **λ_coord = 5**, **λ_noobj = 0.5**. The loss uses **√w, √h**.
- YOLO label = `class xc yc w h`, a **centre** normalised to 0–1, not a corner and not xyxy. The image origin (0, 0) is the top-left.
- AP is **per class**. mAP is the mean over classes. TN is **not** used in detection. A box with the wrong class but good IoU is an **FP**.

---

## Q7 — Coding (PyTorch)

Source: `materials/Chapter 10/notes.md` (sections 2–10 + teacher emphasis), mock Q7. Code is written **by hand**, so the templates below are short and every line has a job. Every code block here was run (torch 2.11 / torchvision 0.26) and every shape number checked. *(standard fact)* = general PyTorch knowledge, not from the slides.

### 1. What they will ask

| Type | Example | What earns the marks |
|---|---|---|
| Write the loop | "Write training + validation loop, save best weights" (mock 7.2) | correct order, `eval()` + `no_grad()`, `state_dict` saved when val improves |
| Order / fill blanks | put `backward`, `step`, `zero_grad`, `model(x)`, `criterion` in order | `zero_grad → forward → loss → backward → step` |
| Find bugs | buggy fine-tune script, "find ≥6 bugs, line + problem + fix" (mock 7.1) | name the line, say the *consequence*, give the fix |
| Explain a line | "which line trains?", "what does `is_available()` check?", "why `model.train()`?" | the teacher's exact points (§2) |
| Shapes | "fill `Linear(?, 10)`", output size of a conv | formula + each step written out |
| Modify a pretrained model | ResNet-50 for 4 classes / multilabel / regression | `weights=...`, `model.fc = nn.Linear(2048, n)`, right loss |
| Inference | `predict(path)` printing `Predicted: MEL 97.31%` (mock 7.3) | rebuild arch + `load_state_dict`, val transform, `unsqueeze(0)`, eval + no_grad, softmax + max |

### 2. Must-know

**2.1 Training loop: order and why each line exists** (the core of the exam)

| # | Line | Why |
|---|---|---|
| 0 | `model.train()` | sets the **state** (Dropout on, BatchNorm uses batch stats). Does not train anything. Teacher starred it: "ตายมานักต่อนักแล้ว" |
| 1 | `x, y = x.to(device), y.to(device)` | model and data must be on the **same device** |
| 2 | `optimizer.zero_grad()` | PyTorch **accumulates** gradients; without it the previous batch's gradients are added in |
| 3 | `out = model(x)` | **prediction (forward pass)**, *not* the training line ("อันนี้ผิด"). Needed because loss needs a prediction |
| 4 | `loss = criterion(out, y)` | one number for how wrong the model is |
| 5 | `loss.backward()` | **computes gradients** of the loss w.r.t. every parameter with `requires_grad=True` |
| 6 | `optimizer.step()` | **updates weights**: w_new = w_old − α·∂loss/∂w. "เวทถูกอัปเดตจริงๆที่คำสั่ง optimizer.step" |

- "Which commands train the model?" → the chain **predict → criterion → backward → step** (4 lines).
- 2 for-loops: outer = epochs, inner = batches. `step()` inside the batch loop = update every batch (usual). Not per image (batch_size=1): an outlier image pushes weights the wrong way; a batch averages the loss.
- Print only every N batches (`if i % 500 == 499`). Display I/O is slow: "10 นาที ผมโดนไป 1 วัน".
- Use Torch, not NumPy, inside training: NumPy = CPU, Torch = GPU, converting forces CPU↔GPU copies.

**2.2 Validation / inference: `model.eval()` + `torch.no_grad()`, why both**
- `model.eval()`: Dropout **off**, BatchNorm uses the **stored running mean/var**. Forgetting it gives distorted, unstable predictions (default mode is train).
- `torch.no_grad()`: no gradient tracking → faster, less memory, bigger batches. It does **not** change Dropout/BN.
- They do different jobs, so you need **both**. Put `eval()` **before** `with torch.no_grad():` (the teacher forgot it on his own slide).
- Validation has **no** `zero_grad / backward / step`. Weights change only on `optimizer.step()`.

**2.3 Device / GPU check: "ถูกครึ่งเดียว"**
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)                       # once, AFTER replacing model.fc
images, labels = images.to(device), labels.to(device)   # every batch
```
- `is_available()` checks **2 stages**: (1) is there a GPU, (2) is it **available** (not fully used by someone else). "Checks if the machine has a GPU" = **half credit**.
- If False, the code silently runs on CPU (3-day job took over a week). In real work on a shared server, force `.cuda()` so it errors instead of running slowly.

**2.4 Model class**
- `class Net(nn.Module)`; `__init__` **creates the layers as attributes** (call `super().__init__()` first); `forward(self, x)` **defines the data flow**.
- `forward` returns the **score function (logits)**. Softmax / sigmoid go **outside** the model.
- Channel matching: conv `out_channels` = next conv `in_channels` = `BatchNorm2d(n)`. Linear in = previous out. **Last Linear out = number of classes**.
- `nn.Sequential` groups layers in order (recommended, tidier; cannot branch). Flatten: `torch.flatten(x, 1)` = `x.view(x.size(0), -1)`.
- Dropout default p=0.5 (use 0.2–0.5), active only in train mode.

**2.5 Shape arithmetic** (tensor = N×C×H×W)
- Conv: **out = ⌊(W − K + 2P)/S⌋ + 1**. Floor when it doesn't divide *(standard fact)*. `Conv2d(in, out, kernel_size, stride=1, padding=0)` defaults.
- K=3, P=1, S=1 keeps size; K=5, P=2 keeps size. `MaxPool2d(2, 2)` halves H, W (same formula, P=0).
- Flatten size = C × H × W of the last feature map. Channels change only in conv; pooling keeps C.
- Params *(standard fact)*: Conv = out·(in·K·K) + out; Linear(in,out) = out·in + out (`Linear(3,2)` → weight 2×3 + bias 2 = 8).

**2.6 Loss per task** (training code otherwise "แทบจะเหมือนกัน 100%")

| Task | Output nodes | Loss | Target | Read-out (outside model) |
|---|---|---|---|---|
| Multiclass | n classes | `nn.CrossEntropyLoss()` (has softmax inside → give **logits**) | class index, `long` shape (N,) | `torch.softmax(out,1)`, `torch.max(...,1)` |
| Multilabel | n labels | `nn.BCEWithLogitsLoss()` (sigmoid + BCE) | multi-hot **float** (N, n) | `torch.sigmoid(out) > 0.5` (0.3 in slide code) |
| Regression | values to predict (1 per model recommended) | `nn.L1Loss()` (MAE) or `nn.MSELoss()` | float (N, 1) | the raw value, **no activation** |

Multilabel = "หนึ่งภาพมีโอกาสที่จะเป็นไปได้มากกว่าหนึ่งคำตอบ" (and possibly none). "1 ภาพหลายคำตอบ" = half credit.

**2.7 Transforms: 2 jobs** (strongest emphasis of Part 2, "รุ่นพี่คุณพลาดมาแล้ว")
1. **Standardisation** (ToImage, Resize, ToDtype, Normalize): **identical** for train / val / test / inference.
2. **Augmentation** (RandomHorizontalFlip, RandomRotation, RandomResizedCrop): **train only**. Augmenting at val/inference "ตลกมาก ไม่ควร": same image → different answers.
- `v2.ToDtype(torch.float32, scale=True)` → [0,1]. Old `ToTensor()` (deprecated in v2) = `ToImage()` + `ToDtype(float32, scale=True)`.
- `Normalize(mean, std)`: x′ = (x − μ)/σ. (0.5, 0.5) maps [0,1] → [−1,1]. Pretrained ImageNet models: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), same at inference.
- Pad to square first, then resize ("เราจะไม่ Resize ก่อน Padding"). Grayscale into a 3-channel pretrained model: duplicate to 3 channels, don't change the first conv.

**2.8 Dataset / DataLoader**
- Custom `Dataset` implements **3 methods**: `__init__` (runs once: csv, folder, transform), `__len__` (number of samples), `__getitem__(idx)` (return `(image, label)` for one index, transform applied here).
- `DataLoader(ds, batch_size=32, shuffle=True)` for train (reshuffled **every epoch**); `shuffle=False` for val/test. `num_workers` = CPU processes (slide default 0). Loading/augmentation run on CPU.

**2.9 Pretrained backbone**
- `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`. `resnet50()` = **random** weights ("อย่าทำแบบนี้") — except when you will `load_state_dict` your own weights right after.
- `print(model)` → last layer is `fc`, in_features **2048** → `model.fc = nn.Linear(2048, n_classes)` (1000 → n). "โค้ดบรรทัดนี้ สำคัญมาก".
- Freeze the backbone *(standard fact, not on the slides)*: `for p in model.parameters(): p.requires_grad = False`, **then** replace `fc` (a new layer has `requires_grad=True`), optimizer on `model.fc.parameters()`.
- Order: replace `fc` → `model.to(device)` → create the optimizer (else the new fc is on CPU / not in the optimizer).

**2.10 Save / load**
- `torch.save(model.state_dict(), 'best.pt')` = **weights only** (smaller). Load: **rebuild the same architecture (same fc)** → `model.load_state_dict(torch.load('best.pt'))` → "All keys matched successfully".
- `torch.save(model, path)` = architecture + weights, `model = torch.load(path)`.
- Keep 3 checkpoints: last epoch, early-stopping point, best validation.

**2.11 Data leakage in splitting**
- Teacher's definition: an image is **test for model A but train for model B** → unfair comparison. `train_test_split` / auto k-fold, even with a seed, may differ between runs.
- Fix: **split manually once**, save the lists (train.csv / val.csv / test.csv), every model loads the same files.
- Medical extension (from exam-hints): split **by patient**, so no patient is in both train and test.

### 3. Code templates to memorise

**3A. Full program: dataset → ResNet-50 fine-tune → train + validate → save best → predict one image**
```python
import os, torch, torch.nn as nn, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights

classes = ['MEL', 'NV', 'BCC', 'BKL']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # ImageNet
train_tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(10),          # augmentation: TRAIN ONLY
    v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
val_tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224)),           # same standardisation,
    v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])  # no augmentation

class ImgDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)            # columns: filename, label (0..3)
        self.img_dir, self.transform = img_dir, transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.df.iloc[idx, 0])).convert('RGB')
        label = int(self.df.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

train_loader = DataLoader(ImgDataset('train.csv', 'train', train_tf), batch_size=32, shuffle=True)
val_loader = DataLoader(ImgDataset('val.csv', 'val', val_tf), batch_size=32, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)        # pretrained
model.fc = nn.Linear(2048, len(classes))                          # 1000 -> 4 classes
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0.0
for epoch in range(20):
    model.train()                                  # inside the epoch loop (eval() below)
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                      # 1 reset gradients
        outputs = model(images)                    # 2 prediction (logits, N x 4)
        loss = criterion(outputs, labels)          # 3 loss
        loss.backward()                            # 4 gradients
        optimizer.step()                           # 5 update weights
        running_loss += loss.item()
    model.eval()                                   # Dropout off, BN running stats
    correct, total = 0, 0
    with torch.no_grad():                          # no gradient tracking
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f'epoch {epoch+1} loss {running_loss/len(train_loader):.4f} val_acc {val_acc:.4f}')
    if val_acc > best_acc:                         # keep the best-validation weights
        best_acc = val_acc
        torch.save(model.state_dict(), 'best.pt')

def predict(path):
    model = resnet50()                             # same architecture (weights come from file)
    model.fc = nn.Linear(2048, len(classes))       # same fc as in training
    model.load_state_dict(torch.load('best.pt', map_location=device))
    model.to(device)
    model.eval()
    image = Image.open(path).convert('RGB')
    x = val_tf(image).unsqueeze(0).to(device)      # (3,224,224) -> (1,3,224,224)
    with torch.no_grad():
        outputs = model(x).squeeze(0)              # (1,4) -> (4,)
    score = torch.softmax(outputs, 0)              # softmax OUTSIDE the model
    conf, idx = torch.max(score, 0)
    print(f'Predicted: {classes[idx]} {conf.item() * 100:.2f}%')
```

**3B. Small custom CNN** (slide's CIFAR net, 3×32×32 input; `Resize((32, 32))` in the transforms)
```python
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),   # 3x32x32 -> 6x28x28 -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2))  # -> 16x10x10 -> 16x5x5
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(120, num_classes))           # out = number of classes
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)                    # (N,16,5,5) -> (N,400)
        return self.classifier(x)                  # logits, no softmax

model = Net(num_classes=len(classes)).to(device)   # drop-in replacement for the ResNet
```

**3C. Shape worked examples** (write every step)

| Net | Layer | Calculation | Output (C×H×W) |
|---|---|---|---|
| CIFAR (3B) | input | | 3×32×32 |
| | Conv2d(3,6,5) | (32 − 5 + 0)/1 + 1 = 28 | 6×28×28 |
| | MaxPool(2,2) | (28 − 2)/2 + 1 = 14 | 6×14×14 |
| | Conv2d(6,16,5) | (14 − 5)/1 + 1 = 10 | 16×10×10 |
| | MaxPool(2,2) | 10/2 | 16×5×5 → flatten **400** |
| Slide `Net` (MNIST) | Conv2d(1,4,3,1,1) + BN(4) + ReLU + Pool, ×2 | (28 − 3 + 2)/1 + 1 = 28 → 14; again → 7 | 4×7×7 → `Linear(4*7*7 = 196, 10)` |
| Slide ConvNet | Conv(1,32,k5,p2)+Pool, Conv(32,64,k5,p2)+Pool | (28 − 5 + 4) + 1 = 28 → 14 → 14 → 7 | 64×7×7 → `Linear(7*7*64 = 3136, 1000)` |
| Stride 2 *(standard)* | Conv2d(3,16,3,stride=2,padding=1) on 224 | ⌊(224 − 3 + 2)/2⌋ + 1 = 111 + 1 | 16×112×112 |

Params of `Conv2d(3,6,5)` = 6·(3·5·5) + 6 = **456**; `Linear(400,120)` = 400·120 + 120 = **48,120**. ResNet-50 on 224×224 ends in a 2048-vector → fc *(standard: global avg pool)*.

### 4. Model answers

**(a) "Put in order / which line trains?"** → `model.train()`; per batch: `.to(device)` → `optimizer.zero_grad()` → `out = model(x)` → `loss = criterion(out, y)` → `loss.backward()` → `optimizer.step()`. `model(x)` only **predicts**; `backward()` computes gradients; `step()` actually updates the weights. Training = these 4 (predict → loss → backward → step).

**(b) "Why `zero_grad()`?"** → `.backward()` **adds** to `.grad`; without reset, gradients of all previous batches accumulate, so each update is too big and wrong.

**(c) "Why `model.train()` / `model.eval()`?"** → they set the mode for layers that behave differently: Dropout (on in train, off in eval) and BatchNorm (batch stats + update running stats in train; stored running stats in eval). Forgetting `eval()` at test → noisy, distorted predictions.

**(d) "`no_grad()` vs `eval()`?"** → `no_grad()` stops gradient tracking (speed, memory); `eval()` switches Dropout/BN. Different jobs → use both: `model.eval()` then `with torch.no_grad():`.

**(e) "What does `torch.cuda.is_available()` check? 'whether the machine has a GPU' — full marks?"** → No, half. It checks (1) a GPU exists **and** (2) it is available/free. If False the code silently runs on CPU (days → weeks), so on shared servers force `.cuda()` and let it error.

**(f) "Adapt ResNet-50 to 4 classes"** → `model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`; `model.fc = nn.Linear(2048, 4)`; then `.to(device)`, then the optimizer. Pretrained weights stay; the new fc is learned (fine-tuning).

**(g) "Change to multilabel / regression"** → only 3 places change:
```python
model.fc = nn.Linear(2048, n_labels); criterion = nn.BCEWithLogitsLoss()   # labels: float multi-hot (N, n_labels)
pred = torch.sigmoid(model(x)) > 0.5                                       # each label independent, may be none
model.fc = nn.Linear(2048, 1); criterion = nn.L1Loss()                     # regression: labels float (N, 1), no activation
age = model(x).squeeze().item()
```

**(h) "Find the bugs"** (practice; mock 7.1 has 8 of the same kind)
```python
model = resnet50()                                  # L1
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)   # L2
model.fc = nn.Linear(2048, 4)                       # L3
for images, labels in train_loader:                 # L4
    out = torch.softmax(model(images), 1)           # L5
    loss = nn.CrossEntropyLoss()(out, labels)       # L6
    loss.backward(); optimizer.step()               # L7
with torch.no_grad():                               # L8 (validation)
    acc = (model(val_x).argmax(1) == val_y).float().mean()
```
| Line | Problem | Fix |
|---|---|---|
| L1 | random weights, not pretrained | `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` |
| L2–L3 | optimizer built **before** fc replaced → holds the old fc; new fc never updated | replace `fc` first, then create the optimizer (and `.to(device)` model + data) |
| L5 | softmax before `CrossEntropyLoss` (already has log-softmax) → double softmax, weak/wrong gradients | `out = model(images)` (logits) |
| L7 | no `optimizer.zero_grad()` → gradients accumulate across batches | `optimizer.zero_grad()` before the forward pass |
| L8 | no `model.eval()` → Dropout on, BN uses batch stats and updates running stats with val data | `model.eval()` before `with torch.no_grad():`, and `model.train()` at the start of each epoch |

**(i) "Save the best model and load it for inference"** → in the epoch loop `if val_acc > best_acc: best_acc = val_acc; torch.save(model.state_dict(), 'best.pt')`. Load: rebuild `resnet50()` + same `fc`, `load_state_dict(torch.load('best.pt'))`, `eval()`. `state_dict` = weights only, so the architecture must be rebuilt identically.

**(j) "Write the custom Dataset"** → the `ImgDataset` class in 3A: `__init__` / `__len__` / `__getitem__`; `transform` applied to the image inside `__getitem__`; train loader `shuffle=True`, val/test `shuffle=False`.

### 5. Traps / common bugs

| Bug | Consequence | Fix |
|---|---|---|
| No `zero_grad()` | gradients accumulate (verified: grad doubles after 2 backwards) | `optimizer.zero_grad()` every batch |
| `step()` before `backward()` | first step has no gradients; later steps use the previous batch's | `backward()` then `step()` |
| `softmax`/`sigmoid` before `CrossEntropyLoss`/`BCEWithLogitsLoss` | applied twice → wrong, flattened gradients | feed raw logits |
| CE target `labels.float()` | error: float targets are read as class probabilities and must be (N, C) | `long` class indices, shape (N,) |
| BCEWithLogits with `long` targets | error (needs float) | `labels.float()` multi-hot (N, C) |
| L1/MSE with output (N,1) vs target (N,) | silent broadcasting to (N,N), wrong loss *(standard)* | `labels.float().view(-1, 1)` or `out.squeeze(1)` |
| No `eval()` in val/test | Dropout on, BN batch stats → distorted results | `model.eval()` |
| `model.train()` only before the epoch loop while validating each epoch | epochs 2+ train in eval mode *(standard)* | `model.train()` at the top of every epoch |
| No `no_grad()` in validation | memory/time wasted | `with torch.no_grad():` |
| `loss.backward()` / `step()` in validation | the model learns from validation data ("แอบจำเฉลย") | validation = forward + metric only |
| Augmentation in val/test/inference (`val_tf = train_tf`) | same image → different predictions, unreliable metrics | separate `val_tf` without random transforms |
| Different normalisation at inference | predictions break | copy the train standardisation exactly |
| `ToDtype(float32)` without `scale=True` | values stay 0–255 *(verified)* | `scale=True` → [0,1] |
| Wrong flatten size in `Linear` | `mat1 and mat2 shapes cannot be multiplied` error | compute C×H×W (§3C) |
| Model on GPU, data on CPU (or new fc after `.to`) | "expected all tensors on the same device" error | `.to(device)` model **after** fc change, and every batch |
| `resnet50()` for fine-tuning | random init, no transfer learning | `weights=ResNet50_Weights.IMAGENET1K_V2` |
| fc not replaced | still 1000 outputs | `model.fc = nn.Linear(2048, n)` |
| `load_state_dict` into a model without the new fc | key/shape mismatch error | rebuild identical architecture first |
| No `unsqueeze(0)` for one image | model expects (N,C,H,W) | `x.unsqueeze(0)` |
| `shuffle=True` on val/test | not wrong for accuracy, but pointless | `shuffle=False` |
| `train_test_split` per model | data leakage between models | fixed manual split from files (by patient) |
| Printing every batch | huge slowdown | `if i % N == N-1` |
| Torch ↔ NumPy inside the loop | CPU↔GPU copies, huge slowdown | stay in Torch |
