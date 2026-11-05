# Chapter 14 — Evaluation Metrics

Metrics for six task types: classification, object detection, image segmentation, video processing, regression and RAG (retrieval plus answering). The thread through the whole lecture is that every task asks a different question, so it needs its own ground truth and metric, and one number alone can mislead you (accuracy on imbalanced data, MAE on a shifted regression line).

> **Who presented what** (slides by Pitawas, Janya, Witchaya, Chetsadaporn, Chinnatip; advisor Asst. Prof. Dr. Taravichet Titijaroonroj)
> - **พี่บุ๊ค**: Classification, [00:01:44]–[00:15:44]
> - **พี่ใหม่**: Object Detection and Image Segmentation, [00:15:54]–[00:42:20]
> - **พี่ฟิล**: Video Processing, [00:43:00]–[01:08:12]
> - **อาจารย์** (covering พี่เจมส์'s part): Regression, [01:08:18]–[01:31:22]
> - **พี่ชิน**: RAG / retrieval and LLM-as-a-Judge, [01:31:56]–[01:56:36]
> - พี่ปอโท was introduced at [00:00:24], but no section is clearly attributed to them in the transcript.
>
> **Not covered in this lecture** (so don't expect them unless another chapter has them): Hausdorff distance, R², MAPE, and explicit "macro vs micro averaging". Specificity is only implied, through FPR on the ROC curve. Dice is never named, but it is mathematically the same as the segmentation F1 (see below).

---

## Key concepts

### 1. Classification (พี่บุ๊ค)

**Confusion matrix** is a table of *Actual vs Predicted*. In the slide's COVID example, Positive = has COVID and Negative = no COVID.
- **TP**: predicted COVID, really has COVID.
- **TN**: predicted no COVID, really doesn't have it.
- **FP**: predicted COVID, but actually healthy (false alarm).
- **FN**: predicted no COVID, but actually sick (missed case).
- Slide numbers: TP = 80, FN = 20, TN = 885, FP = 15.
- What it's for: (1) it shows predictions in detail; (2) it is the basis for Accuracy, Precision and Recall; (3) it shows **what kind of errors** the model makes.
  - Multi-class example (slide 6; rows = actual, columns = predicted):
    - Dog: 100 / 0 / 0
    - Cat: 0 / 70 / 30
    - Tiger: 0 / 50 / 50
  - Dog is never wrong, but cat and tiger get confused with each other. That suggests the two classes look alike [00:04:39]–[00:05:11].

**Accuracy** is the fraction of all samples predicted correctly. Range 0–1.
- **Misleads on imbalanced data.** Slide example: 1,000 patients, of whom 100 have COVID and 900 don't. A model that predicts "no COVID" for everyone gets TP = 0, FN = 100, TN = 900, FP = 0, so **Accuracy = 90%**, yet it finds **zero** COVID cases. "ถามว่า โมเดลนี้เป็นโมเดลที่ดีไหม ก็ต้องตอบว่าไม่ครับ" [00:06:53].

**Recall** (= Sensitivity = TPR) answers: of all real positives, how many did the model find?
- It focuses on **FN**. High recall means few missed cases.
- Use it when **missing a positive is costly**: malware detection, defect detection, **medical screening**. Example: COVID screening to quarantine everyone who might be infected, where false alarms are acceptable [00:09:06].

**Precision** answers: when the model says "positive", how often is it right? It is the trustworthiness of positive predictions.
- It focuses on **FP**. High precision means few false alarms.
- Use it when **a false alarm is costly**:
  - Email spam: an important email must not go to spam.
  - Face recognition: a stranger must not unlock your phone.
  - **Medical confirmatory test**: confirm that the patient really has the disease before treating [00:10:54].

**F1-score** is the **harmonic mean** of Precision and Recall. It measures the balance between them.
- F1 is low if *either* one is low.
- The medical example is cancer detection, where you want both:
  - **High recall**: don't miss patients, treat them in time, less severe disease, lower death risk.
  - **High precision**: avoid misdiagnosis, unnecessary treatment, patient stress and wasted medical resources [00:12:20]–[00:13:10].

**ROC curve** plots **TPR (y-axis) vs FPR (x-axis)** as the decision **threshold** changes.
- TPR is the same thing as recall.
- The top-left corner is the "perfect classifier", so a good curve bows toward it (high TPR, low FPR).
- The diagonal is a random classifier.
- In the slide figure: blue = best model, green = second, orange = worst [00:13:59]–[00:14:28].

**AUC** is the area under the ROC curve. It turns the curve into **one number**, which makes models easy to compare.
- AUC close to 1: separates the classes very well.
- AUC = 0.5: no better than random.
- AUC < 0.5: the model separates in the **wrong direction** (its predictions are inverted).

> Clarification (standard fact): **Specificity = TN/(TN+FP) = 1 − FPR**. It is the "recall of the negative class".

### 2. Object Detection (พี่ใหม่)

**Task:** find *where* each object is (roughly) and *what* it is.
- **Output:** bounding box, class label and confidence score (e.g. "dog 0.98", "cat 0.85").
- **Ground truth:** bounding box and class label.
- **Box formats:**
  - **Corner coordinates**: top-left and bottom-right corners.
  - **Center + width/height**.
  - **Top-left + width/height**: the format used in the slide. Example: Cat (x = 5, y = 40, w = 160, h = 300), Dog (140, 180, 280, 220).

**IoU (Intersection over Union)** = area of overlap ÷ area of union between the predicted box and the GT box.
- Range 0–1; higher means better localisation.
- An **IoU threshold** (commonly **0.5**) decides correctness. A detection is correct if **IoU ≥ threshold AND the class is correct**.
- The threshold reflects how strict the evaluation is [00:21:31].

**TP / FP / FN / TN in detection:**
- **TP**: object detected, correct class, and IoU ≥ threshold.
- **FP**: detection with the **wrong class**, **or IoU < threshold**, **or no real object there**.
- **FN**: a real object the model did not detect.
- **TN**: usually **not used**. The number of "no-object" regions is huge and can't be defined clearly [00:23:00].

**Precision, Recall and F1** work as in classification, but they **count objects (boxes)**.

**AP (Average Precision)** is the **area under the Precision–Recall curve for ONE class**. You build the curve by sweeping the **confidence threshold**.
- High threshold: only very confident predictions survive, so precision is high and recall is low.
- Lowering the threshold accepts more predictions, so recall rises, FP rises and precision falls [00:27:06]–[00:27:35].

**mAP** is the **mean of AP over all classes**.

### 3. Image Segmentation (พี่ใหม่)

**Task:** assign a class to **every pixel**. The output and the GT are both a **pixel mask**, aligned pixel-for-pixel with the image.

**Label settings:**
- **Binary**: 2 classes (object / background), e.g. giraffe vs background.
- **Multi-class**: many classes, but **each pixel belongs to exactly one** (car, road, tree…).
- **Multi-label**: a pixel can have **more than one class** (a hand holding a pen: some pixels are both "hand" and "pen"). Evaluate each class **separately, as binary segmentation**.

**Unit = pixels.** In detection the unit is objects; here TP/FP/FN/TN count pixels [00:33:07].
- **TP**: object pixel predicted as object.
- **FP**: background pixel predicted as object.
- **FN**: object pixel predicted as background.
- **TN**: background pixel predicted as background.

**Metrics** (all range 0–1, higher is better):
- **Pixel Accuracy**: fraction of all pixels predicted correctly.
- **Precision**: of the pixels predicted as object, the fraction that are correct.
- **Recall**: of the real object pixels, the fraction found.
- **F1**: balance of precision and recall.
- **IoU**: overlap of prediction with GT.

**Multi-class:** compute TP/FP/FN/TN **per class, one-vs-all**, then average to get **mPrecision, mRecall, mF1 and mIoU**.
- This is the same as macro-averaging (my clarification; the lecture calls it the "mean").

"เราก็ไม่ควรแค่พิจารณาค่าตัวชีวัตรค่าใดค่าหนึ่ง เราก็ต้องพิจารณาหลายๆมุมมอง" [00:37:54]: each metric sees the model from a different angle.

> Clarification (standard facts):
> - **Dice coefficient = 2|A∩B| / (|A|+|B|) = 2TP/(2TP+FP+FN) = F1** for pixels. Dice and IoU are linked: **Dice = 2·IoU/(1+IoU)**, and Dice ≥ IoU always.
> - Pixel accuracy is inflated by large background areas, because TN dominates. This is the same imbalance problem as classification accuracy, so for small lesions or tumours prefer IoU or Dice.

### 4. Video Processing (พี่ฟิล)

**Image vs video:**
- An image is spatial only, with no time.
- A video adds the **temporal dimension**: order matters, and the output is at clip, frame or time-interval level.
- Key example: the **same frames** shown in one order mean "sitting down" and in reverse order mean "standing up". "งานวิดีโอมันไม่ได้มีแค่เอาภาพมาต่อ ๆ กัน" [00:45:32].

| Video task | Image analogue | Input | Ground truth | Metrics |
|---|---|---|---|---|
| **Video Classification** | Image classification + time | **Trimmed** clip with one clear main content | 1 label per clip (swimming, archery, running) | Accuracy, Precision, Recall, F1 |
| **Temporal Action Segmentation** | Image segmentation + time | Long **untrimmed** video with many consecutive actions | Label for **every frame** (one label per frame) | Frame-wise Accuracy, **Edit score**, **Segmental F1@k** |
| **Temporal Action Localization** | Object detection + time | Untrimmed video including irrelevant parts | List of (class, start, end) for **actions of interest only** | **tIoU**, AP, **mAP@tIoU** |

**Example uses from the slides:**
- Video classification: sport type, human activity, exercise pose.
- Action segmentation: analysing the steps of **surgery**, cooking, or factory assembly.
- Action localization: goals and fouls in football, **elderly falls on CCTV**, accidents in traffic video.

**Frame-wise accuracy** = correctly predicted frames ÷ total frames.

**Edit score** measures whether the **order of actions** is correct, and it penalises **over-segmentation**, where the model splits an action into too many small pieces.
1. Collapse consecutive identical labels into segments.
2. Compute the **Levenshtein distance** D(P, G): the minimum number of **insertions, deletions and substitutions** needed to turn the predicted sequence into the GT sequence.
3. Normalise by the longer sequence.
- High score: the order matches and there are few extra segments.
- Low score: actions are missing, extra, swapped, or fragmented.

**Segmental F1@k** is like F1, but a predicted segment is TP only if its **IoU with the GT segment ≥ k** (k = 10%, 25%, 50%) and the class is right.
- **Each GT segment can be counted as TP only once.** Extra fragments become FP, which again penalises over-segmentation.
- @10 is the most lenient ("did it find the action at all?"); @50 is the strictest of the three.

**tIoU (temporal IoU)** is the IoU of time intervals.
- A localisation counts as TP if the **class is correct AND tIoU ≥ threshold**; otherwise it is FP.
- Common thresholds: **0.3 / 0.5 / 0.7**.
- **AP** per class is computed at a fixed tIoU (area under the PR curve over confidence).
- **mAP@tIoU** is the mean over classes, usually reported at several tIoU values.

### 5. Regression (อาจารย์)

**Classification vs regression:**
- Classification predicts **categorical / discrete** labels and looks for a **boundary that separates** classes.
- Regression predicts **numerical / continuous** values (room price, age, **heart rate**, test score, income) and looks for a **line that represents** the data.
- "อันหนึ่งแบ่ง อันหนึ่งเป็น(ตัวแทน)" [01:10:02].
- Slide dataset example: California housing (target = MedHouseVal, in units of 100k USD).
- Slide models: Linear, Lasso (L1), Ridge (L2), XGBoost, Neural Network.

**There are two families of regression metrics** [01:10:12]:

**1. Error-based** (MAE, MSE, RMSE) measures the gap between prediction and actual (the residual).
- Range **0 to ∞**; lower is better. Zero is ideal but impossible in practice.
- Absolute value and squaring both remove the **sign (direction)**, so these metrics measure only the **magnitude** of the error.
- **MAE** weights every error equally and stays in the **original unit**. It is **less sensitive to outliers**.
- **MSE** squares the error, which **amplifies large errors**: "ยิ่งผิดเยอะ ยิ่งโดนโทษเยอะ" [01:16:06]. Its unit is **not** the original unit (it is unit²). It is **sensitive to outliers**.
- **RMSE = √MSE**. It is designed to keep MSE's "punish big errors more" property *and* return to the original unit like MAE [01:19:50]. Like MSE, it is sensitive to outliers.
- **Rule of thumb from อาจารย์** [01:17:31]–[01:17:53]:
  - If big errors should be punished more than small ones, use **MSE (or RMSE)**.
  - If all errors should be punished equally, use **MAE**.
- **"MAE is always smaller than MSE"? Not necessarily** [01:22:58]–[01:23:27].
  - Slide 97 uses 4 images with errors [0, 0, 0, e], so MAE = |e|/4, MSE = e²/4 and RMSE = |e|/2.
  - **Target range [0, 1]** (max |e| = 1): MSE ≤ MAE everywhere.
  - **Target range [−1, 1]** (max |e| = 2): MAE > MSE while |e| < 1, and MSE > MAE once |e| > 1.
  - **Unbounded range** (e.g. |e| up to 10): "MSE grows fastest" (MSE = 25 vs MAE = 2.5 at |e| = 10).
  - The range of the output affects how you read these metrics.
- **Slide 98 (pros and cons):**

| Metric | Pros | Limitation | Use when |
|---|---|---|---|
| MAE | Easy to understand; same unit as the data; more robust to outliers than MSE/RMSE | Doesn't specially punish large errors | You want the typical average error |
| MSE | Punishes large errors hard; commonly used as a **loss** | Very sensitive to outliers; squared unit is hard to interpret | You want the model to focus on big-error cases |
| RMSE | Same unit as the data; still reflects large errors | Still sensitive to outliers; ranks models the same as MSE | **Reporting** results in an easy-to-read form |

  - Slide note: "MSE and RMSE are not interchangeable in interpretation". **MSE suits optimisation; RMSE suits reporting.**
- **Always state the unit and the "Possible Range"** [01:29:02]–[01:30:17].
  - An error of 10 years when predicting age is a big deal.
  - An error of 10 baht on a salary is not.
  - An error of 10 on a 0–100 range is 10%; on a range of tens of thousands it is negligible.

**2. Correlation-based (Pearson correlation)** does not measure the size of the gap. It asks whether prediction and ground truth **move in the same direction**.
- **Weakness of error-based metrics: "Error ต่ำกว่า แต่ Correlation กลับต่ำกว่า"** (slide 99). Ground-truth weights are 46, 48, 50, 52, 54 kg.
  - **Model A** predicts 50, 51, 50, 51, 50: **MAE 2.40 kg, RMSE 2.90 kg, Pearson r = 0.00**. It nearly always predicts a constant of about 50 kg, so it doesn't follow the trend.
  - **Model B** predicts 54, 56, 58, 60, 62: **MAE 8.00 kg, RMSE 8.00 kg, Pearson r = 1.00**. It is always 8 kg too high, but the trend matches perfectly (a **shift**).
  - On MAE alone you'd pick A, yet B captures the behaviour of the data [01:24:35]–[01:25:36].
  - Slide lesson: "MAE/RMSE measure *how far off*; Pearson r measures *how linearly related*… **Report Error and Correlation together**."
- **The reverse trap: high r ≠ accurate** (slide 103). Actual values are [10, 20, 30, 40].
  - **Model A** predicts [11, 19, 31, 39]: **MAE 1, RMSE 1, r ≈ 0.997**.
  - **Model B** predicts [20, 40, 60, 80] (scale ×2): **MAE 25, RMSE 27.4, r = 1.000**.
  - "Pearson r สูง ไม่ได้แปลว่าทำนายแม่นเสมอไป": r does not detect **bias or scale error**.
- A low error doesn't guarantee that the model follows the data trend, so **look at both families and plot the predictions** [01:25:36]–[01:28:22]. "เรากลัวโดนหลอก".
- **Pearson r (ρ)** measures **linear relationship**, from −1 to +1 (slides 101–102).
  - Positive r: X and Y move in the same direction. Negative r: they move in opposite directions. The closer to ±1, the stronger the relationship.
  - **+1**: perfect positive correlation. **0**: no *linear* relationship. **−1**: perfect negative (inverse) correlation.
  - อาจารย์ noted r = −1 is "easier to fix" than r = 0, because the relationship still exists, just reversed [01:26:31].
  - Slide 102 interpretation bands:

| ρ | Meaning |
|---|---|
| 1.0 | Perfect positive |
| 0.7–0.9 | Strong positive |
| 0.3–0.6 | Moderate positive |
| 0.0–0.2 | Weak positive |
| 0 | No linear relationship |
| −0.2–0.0 | Weak negative |
| −0.6 to −0.3 | Moderate negative |
| −0.9 to −0.7 | Strong negative |
| −1.0 | Perfect negative |

  - Slide 102 caution: ρ ≈ 0 does **not** mean "no relationship". There may be a **non-linear** one (U-shape, inverted U).
  - Slide 101 figure: ρ ≈ 0.75 with fitted slope ≈ 0.72. Loss alone measures accuracy, but not agreement between prediction and truth.
- **Best values** (slide 104): MAE = MSE = RMSE = **0**; Pearson r = **+1**. The two groups "measure different angles, so they should not replace each other; use them together".
- อาจารย์'s preference: a model with slightly higher MAE but Pearson close to 1 can be the better choice, because it balances both the error and the distribution [01:28:01].

### 6. RAG: Retrieval-Augmented Generation (พี่ชิน)

**RAG pipeline:**
1. **Query**: the user's question.
2. **Retrieve**: fetch the **Top-K** relevant chunks (K = 5 is common).
3. **Answer**: the LLM answers grounded on that evidence.
- "Grounding" means feeding the retrieved context to the LLM.

**Evaluate the two stages separately.** If you measure only the final answer, you can't tell where the system fails.
1. **Retrieval**: did we get relevant context? Use Precision@K, Recall@K, MRR, nDCG@K.
2. **Answering**: is the answer supported by the evidence? Use **LLM-as-a-Judge**.
- Failure modes: wrong / missing / adding extra.

**Google analogy:** people only look at the top 3–5 results, never page 2. So **rank matters**: the best document must be near the top [01:33:16]–[01:34:52].

**The four retrieval metrics:**
- **Recall@K**: of all relevant documents for the query, the fraction found in the top-K.
  - It **never decreases as K grows**, but a larger K also lets in more irrelevant documents, so read it together with Precision@K.
  - It ignores *where* in the top-K the hit is; only whether it was found [01:39:17].
- **Precision@K**: of the K retrieved documents, the fraction that are relevant. It measures "wasted" retrievals.
  - It usually **drops as K grows**, and tends to be much lower than Recall@K.
  - Smaller K can raise it, but if relevant documents aren't ranked at the top you get 0 for that query (a trade-off) [01:41:41]–[01:42:10].
- **MRR (Mean Reciprocal Rank)** cares only about **the rank of the FIRST relevant document** ("how fast does the first correct answer appear").
  - Score per query = 1/rank, or 0 if no relevant document is retrieved.
  - Average over queries.
  - Good for measuring how quickly search finds the answer; **not** for judging the whole retrieved context.
- **nDCG@K (normalised Discounted Cumulative Gain)** extends MRR.
  - It uses **graded relevance** (e.g. 3 = very relevant, 2 = partly, 1 = barely, 0 = not relevant) for **all** retrieved documents.
  - It discounts lower ranks, then normalises by the **ideal DCG** (the same documents sorted by relevance, best first).
  - Better than MRR when there are several relevant passages, because the order of all of them counts [01:48:01]–[01:48:18].
- **Tool:** the Python library **Rank-Eval** (spoken as "Rank Evolved") bundles DCG, MRR, Precision, Recall and more [01:48:35].

**Summary slide** (slide 123):

| Metric | What it tells you |
|---|---|
| Precision | How **on-target** the retrieved results are ("ตรงโจทย์") |
| Recall | How **completely** the system finds what it should ("ค้นพบครบ") |
| MRR | How **fast** the first correct answer appears ("มาเร็ว") |
| nDCG | How well results are **ordered by relevance** ("เรียงตามระดับความเกี่ยวข้อง") |
| LLM-as-a-Judge | Whether the final answer is **correct, on-point and helpful** |

**LLM-as-a-Judge:** use an LLM, ideally stronger than the model being tested, to grade answers.
- History: early ChatGPT used human eval (e.g. 100 questions × 100 people ≈ 10,000 ratings). There are now too many LLMs for that, so evaluation shifted to LLM eval [01:49:39]–[01:50:28].
- **Inputs:** question, model answer, ground-truth answer, **Criteria / Rubric** (what to look at, and what earns 0, 1, 2, 3 or 4 points) and **Evaluation Steps**.
- **Output:** a **score + reason** (e.g. why 0.6 out of 1).
- **พี่ชิน's own experiment:**
  - Judges: GPT-4.1 and Gemini 2.5 Flash. Model judged: a fine-tuned Qwen.
  - Criteria: **Correctness, Relevance, Helpfulness** on a 0–1 scale (based on **G-Eval**).
  - **Dual judges** reduce single-model bias.
  - Scale: 6 models × 226 questions.
  - Human experts rated a balanced subset of **40 questions**. The **average MAE between human and LLM ratings was 0.13** (low disagreement means practically reliable), with targeted human review for edge cases.
- The technique is "not yet a best practice", which is why it was presented for review [01:50:49].

---

## Formulas & numbers to remember

### Classification

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Recall = Sensitivity = TPR | TP / (TP + FN) |
| Precision | TP / (TP + FP) |
| F1 | 2·P·R / (P + R) = **2TP / (2TP + FP + FN)** |
| FPR | FP / (FP + TN) |
| Specificity *(clarification)* | TN / (TN + FP) = 1 − FPR |
| AUC | area under the ROC curve (TPR vs FPR); 1 = perfect, 0.5 = random, < 0.5 = reversed |

**Worked example: slide COVID matrix** (TP = 80, FN = 20, TN = 885, FP = 15, N = 1000). The slide shows the matrix only; these values are computed here.
- Accuracy = (80 + 885)/1000 = **0.965**
- Recall = 80/100 = **0.800**
- Precision = 80/95 = **0.842**
- F1 = 160/(160 + 15 + 20) = 160/195 = **0.821**
- Specificity = 885/900 = **0.983**; FPR = 15/900 = **0.017**

**Imbalanced trap** (TP = 0, FN = 100, TN = 900, FP = 0):
- Accuracy = 900/1000 = **0.90**
- Recall = 0/100 = **0**
- Precision = 0/0, undefined (the model never predicts positive).

### Object detection

- **IoU = Area of Overlap / Area of Union**. Union = A + B − Overlap.
  - Slide example: overlap 85 px and union 100 px give IoU = **0.85**, which is ≥ 0.5, so the detection is correct (if the class is right).
- **Precision = TP/(TP+FP)**. Birds: 12 detected, 10 correct, so 10/12 = **0.833**.
- **Recall = TP/(TP+FN)**. 15 real birds, 10 found, 5 missed, so 10/15 = **0.667**.
- **F1** = 2(0.833)(0.667)/(0.833 + 0.667) = 20/27 = **0.741**.
- **AP**: slide table for 20 real birds, sweeping the confidence threshold. Precision = TP/(TP+FP); Recall = TP/20.

| Conf ≥ | TP | FP | FN | P | R |
|---|---|---|---|---|---|
| 0.95 | 2 | 0 | 18 | 1.00 | 0.10 |
| 0.90 | 4 | 0 | 16 | 1.00 | 0.20 |
| 0.85 | 6 | 1 | 14 | 0.86 | 0.30 |
| 0.80 | 8 | 2 | 12 | 0.80 | 0.40 |
| 0.75 | 10 | 3 | 10 | 0.77 | 0.50 |
| 0.70 | 12 | 5 | 8 | 0.71 | 0.60 |
| 0.65 | 14 | 7 | 6 | 0.67 | 0.70 |
| 0.60 | 16 | 10 | 4 | 0.62 | 0.80 |
| 0.50 | 18 | 14 | 2 | 0.56 | 0.90 |
| 0.40 | 20 | 20 | 0 | 0.50 | 1.00 |

- Slide result: **AP(Bird) = 0.773**. I checked it: the trapezoidal area under the curve, starting from (R = 0, P = 1), gives 0.7727.
- **mAP** = mean AP over classes = (Bird 0.773 + Car 0.883 + Dog 0.842)/3 = 2.498/3 = **0.833** (the slide shows the table but not this final number).

### Segmentation

| Metric | Formula |
|---|---|
| Pixel Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 (= Dice) | 2TP / (2TP + FP + FN) |
| IoU (Jaccard) | TP / (TP + FP + FN) |
| mIoU, mF1, … | mean of the per-class value (one-vs-all) |

**Binary worked example** (slide, 4×4 = 16 px): TP = 5, FN = 2, FP = 1, TN = 8.
- Pixel Accuracy = 13/16 = **0.8125**
- Precision = 5/6 = **0.833**
- Recall = 5/7 = **0.714**
- F1 / Dice = 10/13 = **0.769**
- IoU = 5/8 = **0.625**. Check: Dice = 2(0.625)/1.625 = 0.769 ✓

**Multi-class worked example** (slide, 3×3 = 9 px; 0 = background, 1 = cat, 2 = dog):

| Class | TP | FP | FN | TN | IoU | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Background | 2 | 2 | 1 | 4 | 2/5 = 0.400 | 2/4 = 0.500 | 2/3 = 0.667 | 4/7 = 0.571 |
| Cat | 2 | 0 | 1 | 6 | 2/3 = 0.667 | 2/2 = 1.000 | 2/3 = 0.667 | 4/5 = 0.800 |
| Dog | 2 | 1 | 1 | 5 | 2/4 = 0.500 | 2/3 = 0.667 | 2/3 = 0.667 | 4/6 = 0.667 |

- Pixel Accuracy = 6/9 = **0.667**
- **mIoU** = (0.400 + 0.667 + 0.500)/3 = **0.522**
- mPrecision = **0.722**; mRecall = **0.667**; mF1 = **0.679**
- Sanity check: each class's TP + FP + FN + TN = 9.

### Video

- **Video classification** (20 videos, class "swimming"): TP = 8, FP = 2, FN = 1, TN = 9.
  - Accuracy = 17/20 = **0.85**
  - Precision = 8/10 = **0.80**
  - Recall = 8/9 = **0.889**
  - F1 = 16/19 = **0.842**
- **Frame-wise Accuracy** = correct frames / total frames.
  - Slide: GT is AAAAA BBBBB CCCCC DDDDD; Pred is AAAA BBBBBB CC BB C DDDDD.
  - 3 frames wrong (frame 5: A→B; frames 13–14: C→B), so 17/20 = **0.85**.
- **Edit = 1 − D(P,G) / max(N_pred, N_gt)** (slide 65; often reported as a percentage)
  - GT collapsed = A→B→C→D (N_gt = 4). Pred collapsed = A→B→C→**B→C**→D (N_pred = 6).
  - D = 2 deletions, so Edit = 1 − 2/6 = **0.667 (66.67%)**.
- **Segmental F1@k:**
  - GT segments: A 1–5, B 6–10, C 11–15, D 16–20.
  - Pred segments: A 1–4, B 5–10, C 11–12, B 13–14, C 15, D 16–20.
  - IoU with the GT segment of the same class: A 4/5 = 0.80; B 5/6 = 0.83; C(11–12) 2/5 = 0.40; B(13–14) 0; C(15) 1/5 = 0.20 (but GT C is already matched, so FP anyway); D 5/5 = 1.00.

| | F1@10 | F1@25 | F1@50 |
|---|---|---|---|
| TP | 4 | 4 | 3 |
| FP | 2 | 2 | 3 |
| FN | 0 | 0 | 1 (GT C has no match ≥ 50%) |
| Precision | 4/6 = 0.667 | 0.667 | 3/6 = 0.500 |
| Recall | 4/4 = 1.000 | 1.000 | 3/4 = 0.750 |
| **F1** | **0.800** | **0.800** | **0.600** |

- **tIoU = |pred ∩ GT| / |pred ∪ GT|** (in time).
  - Goal: GT 10–20 s, Pred 12–22 s. Intersection 12–20 = 8, union 10–22 = 12, so **0.667**.
  - Foul: GT 45–52 s, Pred 47–60 s. Intersection 47–52 = 5, union 45–60 = 15, so **0.333**.
  - At threshold 0.3: both TP. At 0.5: goal TP, foul FP. At 0.7: both FP.
- **mAP@0.5** = (AP_goal 0.611 + AP_foul 0.146)/2 = 0.757/2 = **0.379 (37.9%)**.

### Regression

| Metric | Formula | Unit / range |
|---|---|---|
| MAE | (1/n) Σ \|yᵢ − ŷᵢ\| | original unit, 0 to ∞ |
| MSE | (1/n) Σ (yᵢ − ŷᵢ)² | unit², 0 to ∞ |
| RMSE | √MSE | original unit, 0 to ∞ |
| Pearson ρ (slide 101) | Σ(xᵢ − x̄)(yᵢ − ȳ) / √[Σ(xᵢ − x̄)² · Σ(yᵢ − ȳ)²] | −1 to +1, best = +1 |

(The slide writes the means as x̂ and ŷ; they are the means of the actual and predicted values.)

**Worked example (slides 90–95):**

| | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| Actual | 50 | 55 | 60 | 65 | 70 | 75 | 80 |
| Predicted | 55 | 47 | 66 | 61 | 77 | 70 | 102 |
| \|e\| | 5 | 8 | 6 | 4 | 7 | 5 | 22 |
| e² | 25 | 64 | 36 | 16 | 49 | 25 | 484 |

- MAE = 57/7 = **8.14**
- MSE = 699/7 = **99.86**
- RMSE = √99.86 = **9.99**
- Point G is an **outlier**: one error of 22 contributes 484 of the 699 squared error.
  - My check: without G, MAE = 35/6 = 5.83 and MSE = 215/6 = 35.83.
  - So G moves MSE far more than MAE.

**Model A vs B** (slide 96: weight prediction from 4 images, total absolute error 4 for both) [01:20:15]–[01:21:28]:
- Model A errors 1, 1, 1, 1 kg: MAE = 1 kg, MSE = 1 kg², RMSE = 1 kg.
- Model B errors 0, 0, 0, 4 kg: MAE = 1 kg, MSE = 16/4 = 4 kg², RMSE = 2 kg.
- Slide: "MAE ของทั้งสองโมเดลเท่ากัน (1 kg) แต่ RMSE ของ Model B สูงกว่าเป็น 2 เท่า".
- MAE can't tell them apart. MSE strongly penalises B's single big error. RMSE (2) shows the difference in the original unit without blowing it up.

**Error vs correlation** (slide 99; GT = 46, 48, 50, 52, 54 kg):
- Model A predicts 50, 51, 50, 51, 50.
  - |e| = 4, 3, 0, 1, 4, so MAE = 12/5 = **2.40**.
  - e² = 16, 9, 0, 1, 16, so MSE = 42/5 = 8.4 and RMSE = **2.90**.
  - Pearson r = **0.00**: the prediction deviations (−0.4, 0.6, −0.4, 0.6, −0.4) have zero covariance with the GT deviations (−4, −2, 0, 2, 4).
- Model B predicts 54, 56, 58, 60, 62.
  - Every error is 8, so MAE = RMSE = **8.00**.
  - Pearson r = **1.00**: a pure shift.

**Scale error** (slide 103; actual = [10, 20, 30, 40]):
- Model B predicts [20, 40, 60, 80], so |e| = 10, 20, 30, 40.
- MAE = 100/4 = **25**; MSE = (100 + 400 + 900 + 1600)/4 = 750; RMSE = √750 = **27.4**.
- Yet r = **1.000**.

### RAG / Retrieval

| Metric | Formula |
|---|---|
| Precision@K | (# relevant in top-K) / **K**, averaged over queries |
| Recall@K | (# relevant in top-K) / **R** (all relevant for that query), averaged over queries |
| MRR | (1/\|Q\|) Σ 1/rankᵢ, where rankᵢ is the first relevant position (0 if none found) |
| DCG@K | Σᵢ (2^relᵢ − 1) / log₂(i + 1) |
| nDCG@K | DCG@K / IDCG@K (IDCG = DCG of the ideal descending ordering) |

**Shared example data** (slides 112–114):

| Query | Top-5 retrieved | Relevant (GT) | Found in top-5 | Recall@5 | Precision@5 |
|---|---|---|---|---|---|
| Q1 | D7, D2, **D3**, D10, D4 | D3 | D3 | 1/1 = 1.00 | 1/5 = 0.20 |
| Q2 | D12, **D4**, D8, D1, D9 | D4, D5 | D4 | 1/2 = 0.50 | 1/5 = 0.20 |
| Q3 | D6, D9, D13, D14, D15 | D20 | none | 0/1 = 0.00 | 0/5 = 0.00 |
| Q4 | **D1**, D8, **D2**, D9, D10 | D1, D2, D3 | D1, D2 | 2/3 = 0.67 | 2/5 = 0.40 |
| Q5 | D4, D6, D7, D8, **D5** | D5 | D5 | 1/1 = 1.00 | 1/5 = 0.20 |

- **Mean Recall@5** = (1.00 + 0.50 + 0.00 + 0.67 + 1.00)/5 = **0.634 ≈ 0.63**.
- **Mean Precision@5** = (0.20 + 0.20 + 0.00 + 0.40 + 0.20)/5 = **0.20**. It is much lower than Recall@5.
- พี่ชิน said the right-hand box diagram on the slide was drawn slightly wrong [01:39:08]. The table above is the authoritative data.

**MRR** (slide 114, Q1–Q3; in this table Q2's top-5 is D12, **D4**, D8, **D5**, D1):

| Query | First relevant at rank | Reciprocal rank |
|---|---|---|
| Q1 | D3 at rank 3 | 1/3 ≈ 0.333 |
| Q2 | D4 at rank 2 (D5 at rank 4 is ignored) | 1/2 = 0.5 |
| Q3 | not in top 5 | 0 |

- MRR = (0.333 + 0.5 + 0)/3 = 0.833/3 = **0.278**.
- If Q1's relevant doc were moved to rank 1, Q1 would score 1/1 instead of 1/3 [01:44:06].

**nDCG@5** (slide 115; relevance grades 3 = very, 2 = partly, 1 = barely, 0 = not relevant):
Retrieved ranking for Q1 (slides 115–116):

| Rank i | rel | 2^rel − 1 | log₂(i+1) | Gain |
|---|---|---|---|---|
| 1 | 3 | 7 | 1.0 | 7.00 |
| 2 | 0 | 0 | 1.58 | 0 |
| 3 | 2 | 3 | 2.0 | 1.50 |
| 4 | 0 | 0 | 2.32 | 0 |
| 5 | 1 | 1 | 2.58 | 0.39 |

- **DCG@5** ≈ 7 + 0 + 1.5 + 0 + 0.39 = **8.89**.
- Ideal ordering [3, 2, 1, 0, 0]: **IDCG@5** = 7/1 + 3/1.58496 + 1/2 = 7 + 1.8928 + 0.5 = **9.3928**.
- **nDCG@5 = 8.89/9.3928 = 0.946**.
- Slide 116's caption: "early ranks carry heavy weight; late ranks are discounted".
- Handy logs: log₂2 = 1, log₂3 = 1.585, log₂4 = 2, log₂5 = 2.322, log₂6 = 2.585.

---

## What the speakers emphasised

**พี่บุ๊ค: Classification**
- [00:06:42]–[00:07:17] The imbalanced COVID example: 90% accuracy while finding no COVID at all. "ถามว่า โมเดลนี้เป็นโมเดลที่ดีไหม ก็ต้องตอบว่าไม่ครับ"; this is "ข้อจำกัดอย่างหนึ่งของ accuracy", which is why other metrics exist.
- [00:07:57] Recall "ให้ความสำคัญกับ False Negative"; [00:09:54] Precision "ให้ความสำคัญกับ False Positive". **This is the key contrast.**
- [00:09:06] Screening (COVID quarantine) means recall matters: you accept some healthy people being flagged.
- [00:10:54] Confirmatory test means precision matters, so patients aren't treated wrongly.
- [00:12:20]–[00:13:10] Medical tasks such as cancer detection need **both**, hence F1.
- [00:13:40] TPR on the ROC curve is the same formula as recall. The top-left corner is the "perfect classifier".
- [00:14:47]–[00:15:06] You can't easily compare many ROC curves by eye, which is why AUC turns the curve into a single number.

**พี่ใหม่: Detection & Segmentation**
- [00:21:31]–[00:21:49] The IoU threshold is the minimum acceptable IoU and depends on how strict you want to be; typically **0.5**. A correct detection needs **IoU ≥ threshold AND the correct class**.
- [00:23:00] TN is not used in detection because there are too many background regions.
- [00:27:06]–[00:27:35] High confidence threshold gives high precision and low recall. Lowering it raises recall, raises FP and lowers precision.
- [00:33:07] In detection the unit is objects; in segmentation the unit is **pixels**.
- [00:37:54] "ไม่ควรแค่พิจารณาค่าตัวชีวัตรค่าใดค่าหนึ่ง… ต้องพิจารณาหลายๆมุมมอง".
- [00:38:40] Multi-class: compute one-vs-all per class, then average.

**พี่ฟิล: Video**
- [00:44:20]–[00:45:42] "สิ่งที่สำคัญของงานวีดีโอจริง ๆ ก็คือลำดับของเวลา" (temporal dimension). Same frames in reversed order means sitting down vs standing up.
- [00:49:07] Video classification uses **trimmed** clips; segmentation and localization use **untrimmed** video [00:53:01], [01:02:53].
- [00:56:00]–[00:56:11] Over-segmentation: the model splits an action into too many pieces. The Edit score catches it.
- [01:00:59]–[01:01:12] At 50%, GT C gets FN = 1 because the prediction split C into two pieces and neither overlaps ≥ 50%.
- [01:05:05] tIoU thresholds commonly used: 0.3 / 0.5 / 0.7. Results are reported as **mAP@tIoU** [01:06:56].
- [01:08:00]–[01:08:10] Localization cares only about the actions of interest, not every frame. "อันนี้คือความแตกต่างกัน".

**อาจารย์: Regression**
- [01:10:12] There are two groups: **Error-based** and **Correlation-based**.
- [01:16:30] "ในงาน Regression สิ่งที่สำคัญก็คือ E (Error)", meaning prediction minus actual.
- [01:17:07]–[01:17:22] Absolute value and squaring both **remove the sign**, so these metrics measure only magnitude, not direction.
- [01:17:31]–[01:17:53] **Guideline:** if big and small errors should be punished differently, use MSE; if equally, use MAE.
- [01:18:39] "เด็กๆต้องรู้ก่อนว่าข้อเสียของมันคืออะไร". MSE is not in the original unit, so RMSE exists to keep MSE's property *and* the original unit [01:19:50]–[01:20:03].
- [01:21:44] "สิ่งที่เด็ก ๆ ชอบพลาดกัน": regression outputs have units and ranges, unlike classification counts.
- [01:23:03]–[01:23:27] MAE is not always smaller than MSE; it depends on the value range. "อันนี้คือสิ่งที่เราต้องเข้าใจก่อน… ต้องระวังในการเลือกใช้".
- [01:23:41] The **weakness of MSE and RMSE is sensitivity to outliers**; MAE is less sensitive.
- [01:25:36] "เราไม่ควรดูแค่พวก Error Based Matrix" (the shifted-line example, MAE 2.4 vs 8). Also check Pearson correlation and **plot** the predictions: "เรากลัวโดนหลอก" [01:28:17].
- [01:29:26]–[01:30:17] An error of 10 means different things depending on the **Possible Range**. Always state the unit and the range.

**พี่ชิน: RAG**
- [01:34:12]–[01:34:52] Users only care about the first page of results, so rank matters.
- [01:36:13] Evaluate the RAG pipeline at **two points**: retrieval and answering.
- [01:40:07]–[01:40:27] Increasing K can only raise Recall@K.
- [01:41:28]–[01:42:10] Precision@K tends to be much lower than Recall and drops as K grows. Reducing K helps, but it is a trade-off.
- [01:42:24]–[01:42:32] "ลำดับของการได้เอกสารที่อยู่บนๆ… สำคัญหรือไม่ครับ? คำตอบของเรื่องนี้คือค่อนข้างสำคัญ". This leads to MRR and nDCG.
- [01:42:47] MRR doesn't care how many correct documents were found, only the rank of the first one.
- [01:48:01]–[01:48:18] nDCG compares better than MRR because it scores the order of all retrieved documents, using graded relevance.
- [01:51:35]–[01:52:46] LLM-as-a-Judge needs two key tools, a **Criteria / Rubric** and **Evaluation Steps**. It outputs a score with a reason.
- [01:52:51]–[01:55:27] พี่ชิน was sceptical at first, but in their test human and LLM-judge scores differed by only about 0.13 on average (slide: average MAE 0.13).

---

## Likely exam questions

**Q1.** A model's confusion matrix: TP = 80, FN = 20, TN = 885, FP = 15. Compute Accuracy, Precision, Recall and F1.
**A:**
- Accuracy = (80 + 885)/1000 = **0.965**
- Precision = 80/(80 + 15) = **0.842**
- Recall = 80/(80 + 20) = **0.80**
- F1 = 2·80/(2·80 + 15 + 20) = 160/195 = **0.821**

**Q2.** 1,000 patients, 100 with disease. The model predicts "healthy" for everyone. Give Accuracy and Recall, and say why accuracy misleads.
**A:**
- TN = 900, FN = 100, TP = FP = 0.
- Accuracy = 900/1000 = **90%**; Recall = 0/100 = **0**.
- The data is **imbalanced**, so accuracy is dominated by the majority (negative) class. The model never detects a single patient. Use recall, precision, F1 or AUC instead.

**Q3.** For each scenario, pick Precision or Recall: (a) COVID screening before quarantine; (b) confirmatory test before starting chemotherapy; (c) cancer detection where both missing a patient and over-treating are bad.
**A:**
- (a) **Recall**: missing a case (FN) is costly; false alarms are acceptable.
- (b) **Precision**: a false positive means unnecessary, harmful treatment.
- (c) **F1**: it balances both.

**Q4.** Model X has AUC = 0.93, model Y has AUC = 0.50 and model Z has AUC = 0.30. Interpret each.
**A:**
- X separates positives from negatives very well.
- Y is no better than random guessing.
- Z separates in the **wrong direction**: its scores are inverted, so flipping its decision would give AUC 0.70.
- AUC is threshold-independent: it summarises TPR vs FPR across all thresholds.

**Q5.** A predicted box and a GT box each have area 60 px², and they overlap by 40 px². Compute IoU. Is it a TP at threshold 0.5? At 0.75?
**A:**
- Union = 60 + 60 − 40 = 80, so IoU = 40/80 = **0.5**.
- At threshold 0.5: **TP**, since 0.5 ≥ 0.5 (if the class is correct).
- At 0.75: **FP**.

**Q6.** A detector outputs 12 "bird" boxes, of which 10 are correct. The image has 15 birds. Find Precision, Recall and F1.
**A:**
- TP = 10, FP = 2, FN = 5.
- P = 10/12 = **0.833**; R = 10/15 = **0.667**; F1 = 20/(20 + 2 + 5) = **0.741**.

**Q7.** The AP values per class are Bird 0.773, Car 0.883, Dog 0.842. What is mAP, and what does AP represent?
**A:**
- mAP = (0.773 + 0.883 + 0.842)/3 = **0.833**.
- AP is the area under the precision–recall curve for one class, obtained by sweeping the confidence threshold. mAP is the mean over classes.

**Q8.** A binary tumour segmentation has TP = 5, FN = 2, FP = 1, TN = 8 pixels. Compute Pixel Accuracy, IoU and Dice/F1.
**A:**
- Pixel Accuracy = 13/16 = **0.8125**
- IoU = 5/(5 + 1 + 2) = **0.625**
- Dice/F1 = 2·5/(2·5 + 1 + 2) = 10/13 = **0.769**

**Q9.** A tumour occupies 2% of an MRI slice. Model A predicts "no tumour" everywhere. What is its pixel accuracy? Which metric should you use instead?
**A:**
- Pixel accuracy is about **98%**, because every background pixel is a TN, yet IoU = Dice = 0 for the tumour class.
- Use **IoU or Dice (F1)** for the lesion class, or mIoU across classes. These ignore the huge TN count.

**Q10.** Multi-class segmentation per-class IoU: background 0.400, cat 0.667, dog 0.500. Compute mIoU and say how per-class values are obtained.
**A:**
- mIoU = (0.400 + 0.667 + 0.500)/3 = **0.522**.
- Per-class values come from a **one-vs-all** confusion matrix per class (that class vs everything else). IoU = TP/(TP + FP + FN).

**Q11.** Temporal action segmentation: GT (collapsed) is A→B→C→D; the prediction (collapsed) is A→B→C→B→C→D. Compute the Edit score and name the problem it reveals.
**A:**
- The Levenshtein distance is D = 2 (delete the extra B and C).
- Edit = 1 − 2/max(6, 4) = 1 − 1/3 = **0.667 (66.67%)**.
- It reveals **over-segmentation**: the model fragments one action into pieces.

**Q12.** Fall detection on CCTV. A GT fall is at 10–20 s and the prediction is 12–22 s. Compute tIoU. Is it a TP at tIoU 0.5 and at 0.7? Which task type is this?
**A:**
- Intersection = 12–20 = 8 s; union = 10–22 = 12 s; tIoU = 8/12 = **0.667**.
- At 0.5: **TP**. At 0.7: **FP**.
- This is **Temporal Action Localization**: untrimmed video, find (class, start, end) for actions of interest only. Report mAP@tIoU.

**Q13.** Heart-rate prediction. Actual = [50, 55, 60, 65, 70, 75, 80]; Predicted = [55, 47, 66, 61, 77, 70, 102]. Compute MAE, MSE and RMSE. Which metric is most affected by the last sample?
**A:**
- |e| = 5, 8, 6, 4, 7, 5, 22, so MAE = 57/7 = **8.14**.
- e² = 25, 64, 36, 16, 49, 25, 484, so MSE = 699/7 = **99.86**.
- RMSE = √99.86 = **9.99**.
- **MSE** (and RMSE) are most affected: the outlier error of 22 contributes 484 of the 699 squared error. MAE is the least sensitive.

**Q14.** Two models each have total absolute error 4 over 4 samples: A has errors (1, 1, 1, 1) and B has errors (0, 0, 0, 4). Compute MAE and RMSE for both. Which metric would you choose if one large error is clinically dangerous?
**A:**
- MAE: A = 1, B = 1, so MAE doesn't distinguish them.
- MSE: A = 1, B = 16/4 = 4. RMSE: A = **1**, B = **2**.
- Choose **RMSE (or MSE)**: it punishes large errors more, and RMSE stays in the original unit.

**Q15.** Retrieval with K = 5. For three queries, the first relevant document appears at rank 3, at rank 2, and not at all. Compute MRR. Also compute Precision@5 for five queries with 1, 1, 0, 2, 1 relevant documents in the top 5.
**A:**
- MRR = (1/3 + 1/2 + 0)/3 = 0.833/3 = **0.278**.
- Precision@5 = (1/5 + 1/5 + 0/5 + 2/5 + 1/5)/5 = 1.0/5 = **0.20**.

**Q16 (bonus, nDCG).** A retrieved list has graded relevance [2, 1, 0, 0, 2] (slide 111). Compute nDCG@5 using gain 2^rel − 1 and discount log₂(i + 1).
**A:**
- DCG = 3/1 + 1/1.585 + 0 + 0 + 3/2.585 = 3 + 0.631 + 1.161 = **4.791**.
- Ideal ordering [2, 2, 1, 0, 0]: IDCG = 3/1 + 3/1.585 + 1/2 = 3 + 1.893 + 0.5 = **5.393**.
- nDCG = 4.791/5.393 = **0.888**.

**Q17 (bonus, concept).** A medical RAG chatbot gives wrong answers. How would you locate the problem?
**A:** Evaluate the two stages separately.
- **Retrieval** with Precision@K, Recall@K, MRR and nDCG@K: was the right evidence retrieved, and ranked near the top?
- **Answering** with LLM-as-a-Judge, using a rubric of correctness, relevance and helpfulness, with a ground truth plus evaluation steps, giving a score and a reason.
- Using dual judges and spot-checking against human experts (as in พี่ชิน's test, average MAE 0.13) helps make the judge's results trustworthy.

---

## Common traps

1. **High accuracy on imbalanced data ≠ a good model.** Always check recall and precision, especially in medical data where positives are rare. The 90% accuracy, 0 recall example is the classic.
2. **Mixing up precision and recall.**
   - Recall's denominator is the *actual* positives (TP + FN), and it is hurt by **FN**.
   - Precision's denominator is the *predicted* positives (TP + FP), and it is hurt by **FP**.
   - Screening means recall; confirmation means precision.
3. **F1 is the harmonic mean, not the arithmetic mean.** With P = 1.0 and R = 0.1, F1 = 0.18, not 0.55. The shortcut **F1 = 2TP/(2TP + FP + FN)** avoids rounding errors.
4. **TPR = Recall.** ROC plots TPR (y) against **FPR** (x), not against precision. AUC < 0.5 means "reversed", not "terrible but random".
5. **A detection counts as TP only if IoU ≥ threshold AND the class is right.** A box with good IoU but the wrong class is an **FP**, and so is a duplicate or low-IoU box. TN is not used in detection.
6. **Union = A + B − overlap.** Don't divide by A + B (that gives half of Dice, not IoU).
7. **AP is per class; mAP is the mean over classes** (and, in video, it is reported at a specific tIoU, e.g. mAP@0.5).
8. **Pixel accuracy is inflated by background.** Use IoU or Dice for small structures. Dice (= F1) is always ≥ IoU, so don't compare a Dice from one paper with an IoU from another.
9. **Multi-class segmentation metrics are per-class one-vs-all, then averaged.** The pixel accuracy, by contrast, is computed once over all pixels.
10. **Segmental F1@k: each GT segment can be a TP only once.** Extra fragments are FP. Higher k is stricter: @50 is the strictest and @10 the most lenient.
11. **The Edit score uses collapsed segment sequences and divides by the LONGER sequence** (max of N_pred and N_gt), not by the GT length.
12. **Video task confusion:**
    - Classification: trimmed clip, one label.
    - Action segmentation: every frame gets a label.
    - Action localization: only the actions of interest, as (class, start, end).
13. **MSE is not in the original unit (it's unit²); RMSE is.** MSE and RMSE are sensitive to outliers; MAE is not as sensitive.
14. **"MAE < MSE always" is false.** When errors are below 1 (small output ranges such as 0–1), MSE < MAE.
15. **A low error doesn't mean a good model, and a high r doesn't mean an accurate one.**
    - Slide 99: MAE 2.40 with r = 0, against MAE 8 with r = 1 (a constant shift).
    - Slide 103: r = 1.000 with MAE 25 (scale ×2). r cannot see bias or scale error.
    - Report both families and plot the predictions.
16. **Pearson r = −1 is a perfect inverse correlation, not "no correlation".** r = 0 means no *linear* relationship; a non-linear relationship (e.g. a U-shape) may still exist.
17. **Error values have no meaning without the unit and the possible range.** Report something like "10 bpm on a 40–200 range", not just "error = 10".
18. **Recall@K never decreases as K grows**, while Precision@K usually falls. A big Recall@K at large K can hide poor ranking.
19. **MRR uses only the FIRST relevant document**; the others are ignored. It scores 0 if nothing relevant is retrieved. nDCG uses graded relevance for all positions.
20. **nDCG = DCG/IDCG**, which is ≤ 1. Don't invert it. (The spoken explanation at [01:47:40] said it the other way round; the slide shows 8.89/9.3928 = 0.946.) The ideal ordering sorts the *same* relevance grades in descending order.
21. **Evaluating a RAG system only by its final answer hides where it failed.** Score retrieval and answering separately.
