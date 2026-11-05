# Chapter 9 — Multilabel Classification & Image Regression & Multi-task Learning

You can take one CNN backbone (e.g. AlexNet) and point it at multiclass, multilabel or regression work by changing only three things: the **number of output nodes**, the **last activation function** and the **loss function**. The teacher said that in real code this is "แก้แค่ 2-3 บรรทัดเอง".
Multi-task learning is a *problem* with several different questions per image. Slide p.22 shows three ways to solve it: (a) one feature extractor + one classifier that outputs everything, (b) a shared feature extractor plus task-specific heads, or (c) one model per task.

> Source note: the slide examples (logits, softmax/sigmoid values, threshold, losses, the house-price regression, the fruit and lymphedema architectures) are taken from the slide images (p.9–11, 15–17, 19, 21, 22). Formulas marked *(standard)* are textbook definitions that do not appear on the slides.

---

## Key concepts

### The three task types (slides p.4–7)
- **Multiclass Image Classification**: classifies the image into classes that do **not overlap**, so each image gets exactly **one** class. Slide examples: dog/cat/horse, flower type, vehicle type. Picture example: "Dog".
  - Teacher's one-liner: **"หนึ่งภาพ หนึ่งคำตอบ"**, meaning many possible choices but only one answer.
- **Multilabel Image Classification**: classes **can overlap**, so one image can have **several labels**. Slide examples: a picture with both a cat and a dog, a scene with car + person + tree, movie genre from a poster ("Action, Sci-Fi").
  - Teacher's one-liner: **"หนึ่งภาพ มีโอกาสหลายคำตอบ"**, meaning many choices and possibly more than one answer.
  - (The Whisper transcript writes it as "Multi-Level". The correct term is **Multilabel**.)
- **Image Regression**: predicts a **continuous number** from an image. Slide examples: age from a face, fish length, temperature from aerial photos, distance, real-estate price from photos ("2,450,000"). Slide p.14 adds facial age estimation and house-price prediction.
  - Regression is used for **continuous/series data**. Classification is used for **discrete data**. Data handling, **loss function** and **evaluation metrics** all differ (slide p.14).
- Multiclass and multilabel are both **classification**: the output is **discrete**, a group name. Regression outputs a **continuous** number.

### Building the model from a multiclass CNN (slides p.9–11, p.15–17)
- **The common pipeline** on every slide: Input → Convolution → Pooling → Fully Connected → **Scoring Function** (raw output scores, i.e. logits) → output activation → prediction.
- **One output-layer neuron** (slides p.10, p.16):
  - Summation: z = (i1×w1) + (i2×w2) + … + b1.
  - Classification neurons then apply an activation, so output = **f(z)**, where f = softmax or sigmoid.
  - The regression neuron outputs **z itself**: "Linear Function" means no f(·).
- **Multilabel**: take the multiclass CNN and change the output-layer activation from **Softmax → Sigmoid**.
  - Softmax: the outputs **always sum to 1**.
  - Sigmoid: each output lies in **0–1** and the outputs are **independent of each other**.
  - Number of output nodes = **number of classes**.
  - Sigmoid is drawn as an "S" on each output node **separately**. Softmax is one box that takes all the scores together.
  - Slide p.11 then adds a **"Threshold at 0.5"** step.
- **Image Regression**: take the multiclass CNN and set the output-layer activation to **Linear**.
  - Linear output range is **−∞ to ∞**.
  - Number of output nodes = **1** on the slide. The lecture says it equals **the number of values you want to predict**.
  - "ใช้ Linear ก็คือไม่ใช้": the score-function value is used directly as the answer.
- **Computing the error** ("การพิจารณาผลลัพธ์ของแบบจำลองและการคำนวณข้อผิดพลาด", slides p.11, p.17):
  - The prediction (ผลทำนาย) and the ground truth (ผลเฉลย) both go into the loss box, which outputs "Loss, Error".
  - Multiclass and multilabel use the **Cross Entropy Loss** box. Regression uses the **L1 Loss** box.
  - Slide footnotes:
    - "นักศึกษาสามารถกำหนด Loss Function เป็น **Binary Cross Entropy Loss หรือ Cross Entropy Loss** ก็ได้ใน ณ ปัจจุบันนี้" (multilabel).
    - "…**Mean Absolute Error (MAE) หรือ L1 loss, Mean Squared Error (MSE) หรือ L2 loss และอื่น ๆ** ก็ได้" (regression).

### Master table

| Task type | Output neurons | Output activation | Loss | How to read the prediction |
|---|---|---|---|---|
| Binary classification (e.g. sick / not sick, fresh / rotten) | 2 (one per class, as in the slide p.21 "2 Class" head and the lecture's lymphedema head); a single sigmoid node is the common alternative | Softmax (2 nodes) / Sigmoid (1 node) | (Binary) Cross-Entropy | Highest probability, or p ≥ 0.5 for the single-node version |
| Multiclass | = number of classes | **Softmax** (sum = 1) | **Cross-Entropy** | **argmax**: the class with the largest value is the answer |
| Multilabel | = number of classes (labels) | **Sigmoid** (each 0–1, independent) | **Binary Cross-Entropy or Cross-Entropy** (slide p.11 footnote and lecture) | **Threshold 0.5**: every node with output **≥ 0.5** is part of the answer |
| Regression | = number of values to predict (slide: 1) | **Linear** (no activation, −∞ to ∞) | **L1 (MAE)** or **L2 (MSE)**, or others; try them | The raw output number *is* the prediction |
| Multi-task (approach b) | One head per task, and each head is sized for its own task type (slide p.21: 2-class head + 11-class head) | Each head uses its own task's activation | Each head uses its own task's loss (total = sum of task losses, *standard*) | Read each head separately with its own rule |

(Binary row: the single-sigmoid-node version is standard DL, not from the slides.)

### Multi-task Learning (slides p.19–22)
- **Definition**: train **one model** to predict **several things at once**, so each prediction answers several questions (tasks).
  - Slide example: fruit type (apple / banana / orange) **and** fruit state (rotten / not rotten).
- **Benefits** (slide p.19):
  - **Saves resources**: no separate model per task, so less time and compute.
  - **Joint learning**: tasks share information, which improves accuracy on each task.
- **Points to consider** (slide p.20):
  - **Task relatedness**: works best when the tasks are related or share something.
  - **Data limitation**: useful when data is limited, because the shared data across tasks improves the model.
  - **Shared Feature Extractor**: the feature-extraction part of the network is shared between tasks.
  - **Task-specific Heads**: one neural network per task. Each head connects to the shared feature extractor.
  - **Shared Decision-making Layer**: an alternative where the feature extractor *and* part of the neural network are shared. Only the **final output layer** is task-specific.
- **Negative Transfer** (slide p.21): if the tasks are very different or independent, multi-task learning can perform *worse* than single-task models. The architecture must make sure the shared features help **every** task.
- **Multi-Task Learning vs Multi-Model Learning** (slide p.21 diagram, fruit image):
  - **Multi-Task Learning**: Convolution block → Flatten → **two Dense heads**.
    - Head 1: "Fresh", **2 classes** (ความสดและเน่าเปื่อย, fresh / rotten).
    - Head 2: "Apple", **11 classes** (ชนิดผักและผลไม้, vegetable/fruit type).
  - **Multi-Model Learning**: **two complete, separate pipelines** (Convolution block → Flatten → Dense), one for the 2-class task and one for the 11-class task.
- **Slide p.19 diagram** (the lymphedema case): ultrasound image → **Feature extractor (Class/SEG/SEFS)**, shared → three classifiers: **Classifier (Class)** → "Class: Lymphedema", **Classifier (SEG)** → "SEG grade: 2", **Classifier (SEFS)** → "SEFS grade: 2".

### Multilabel vs Multi-task (the lecture's key distinction)
- Both give "more than one answer". The difference is whether the answers share a meaning.
  - **Multilabel**: all answers have the **same meaning**. Sci-Fi, Comedy and Action are all answers to one question, the movie's genre.
  - **Multi-task**: the answers have **different meanings**. "Which fruit?" and "rotten or not?" are different questions.
- Each task in a multi-task problem can itself be multiclass, multilabel or regression. In practice, each task is usually **multiclass**.

### Three ways to solve a multi-task problem (slide p.22 "ทางเลือกในการเรียนรู้หลายงาน"; lecture [00:16:19]–[00:19:30])
Running example: a master's student uses **ultrasound images** to predict three things: **lymphedema yes/no** (โรคบวมน้ำเหลือง), the **SEG grade** and the **SEFS grade** (severity gradings; slides show e.g. "SEG grade: 2", "SEFS grade: 2").
- **(a) One model, one combined output.**
  - Diagram: a single Feature extractor (Class/SEG/SEFS) → a single Classifier (Class/SEG/SEFS) → "Class: Lymphedema, SEG grade: 2, SEFS grade: 2".
  - The architecture detail shown is DenseNet-style: Conv 7×7 → Max pooling → Dense block 1 → Transition 1 → Dense block 2 → Transition 2 → Dense block 3 → Transition 3 → Dense block 4 → Global average pooling → Fully connected → **Softmax function**.
  - The lecture explains this as **converting multi-task to multiclass by the product of classes**. Example: sick/not sick (2) × SEG level 0/1/2 (3) = **6 classes** (sick-0, sick-1, sick-2, not-sick-0, not-sick-1, not-sick-2), then one softmax.
- **(b) True multi-task.**
  - **One shared Feature extractor (Class/SEG/SEFS)** produces the feature map.
  - The feature map is flattened and fed to **three separate classifiers**: Classifier (Class), Classifier (SEG), Classifier (SEFS). These are the **Task-Specific Heads**.
  - Lecture example of head sizes: the disease head has 2 output nodes; the SEG head (levels 0/1/2) has 3 output nodes.
  - Suitable here because the outputs are **clinically related**: doctors judge lymphedema from the SEG/SEFS levels.
- **(c) Multi-model.**
  - **Three separate Feature extractors and three Classifiers** (Class, SEG, SEFS). This is "1 โมเดลต่อ 1 Task".
- Key point: **"Multitask จริง ๆ แล้วมันไม่ใช่ Architecture มันเป็นปัญหา"**. It is a type of problem, and a, b and c are different solutions to it, each with its own pros and cons.

---

## Formulas & numbers to remember

**From the slides**
- Output neuron (p.10, p.16): z = (i1×w1) + (i2×w2) + … + b1.
  - Classification output = f(z).
  - Regression (linear) output = z.
- Softmax output: the outputs sum to 1. Sigmoid output: each value is in 0–1 and independent. Linear output: −∞ to ∞.
- Multilabel threshold = **0.5**.
- Regression loss: **MAE = L1**, **MSE = L2**.

**Formulas** *(standard; the slides give the properties above but not the equations)*
- **Softmax** (multiclass): pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}, so **Σ pᵢ = 1**.
- **Sigmoid** (multilabel): σ(z) = 1 / (1 + e^{−z}). Each output is in (0,1). The sum can be anything from 0 up to the number of classes.
- **Linear** (regression): ŷ = z.
- **Cross-Entropy** (multiclass, one-hot y): L = −Σᵢ yᵢ log pᵢ, which equals −log(p of the true class).
- **Binary Cross-Entropy** (multilabel, averaged over labels): L = −(1/C) Σᵢ [yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ)].
- **MAE / L1**: (1/N) Σ |yᵢ − ŷᵢ|. **MSE / L2**: (1/N) Σ (yᵢ − ŷᵢ)².
- **Combined multi-task loss** (approach b): L_total = Σ_t w_t · L_t. Each task t uses its own loss. The simplest choice is all w_t = 1.

**Worked examples from the slides** (p.9–11, p.17). The same scoring-function outputs **[−3.44, 1.66, 3.91]** are used for all of them.
- **Multiclass** (dog photo, classes Bird / Cat / Dog):
  - Softmax → Bird 0.010, Cat 0.059, **Dog 0.931**, so the prediction is **Dog**.
  - Prediction [0.010, 0.059, 0.931] vs ground truth [0, 0, 1] → Cross Entropy Loss.
  - CE = −ln 0.931 ≈ **0.07** (my calculation).
- **Multilabel** (Avengers poster, labels Drama / Sci-Fi / Action):
  - Sigmoid → **Drama 0.03, Sci-Fi 0.84, Action 0.98**.
  - Threshold at 0.5 → prediction **[0, 1, 1]**, i.e. "Action, Sci-Fi".
  - Ground truth on slide p.11 is **[0, 0, 1]**, so the model wrongly adds Sci-Fi (a false positive) → Cross Entropy Loss.
  - The sigmoid values check out exactly: σ(−3.44) = 0.031, σ(1.66) = 0.840, σ(3.91) = 0.980.
- **Regression** (house photo, price):
  - Linear output **2,450,000**; ground truth **2,500,000** → L1 Loss.
  - L1 error = |2,450,000 − 2,500,000| = **50,000** (my calculation). L2 would be 50,000² = 2.5 × 10⁹.

**Numbers from the lecture**
- Regression needs roughly **10–20× more training data** than classification ("หลาย 10-20 เท่า"). That is why people usually train **1 model per regressed value**.
- Approach (a) class count: (classes in task 1) × (classes in task 2) × …, e.g. **2 × 3 = 6**.
- Regression output nodes = number of values to predict. Example: systolic + diastolic blood pressure = **2 nodes**, or 1 node if you only predict systolic.
- Difficulty ranking: **Regression (hardest to optimise) > Multilabel > Multiclass (easiest)**.

---

## What the teacher emphasised

- **[00:00:08]–[00:00:26] Why this chapter exists.** Seniors doing projects said "there's no model" for regression or other tasks, because the course only taught multiclass CNN. The teacher's reply: the same model works, you only need to know what to change.
- **[00:01:55]–[00:02:10] In-class question:** "If you had to explain to a 1st-year how Multiclass and Multilabel differ, what's your short answer?"
  → **Multiclass = one image, one answer. Multilabel = one image, possibly many answers.**
- **[00:02:53]–[00:04:06] "คำถาม Classic":** Is predicting age from a face image Regression or Multiclass?
  - **Multilabel is ruled out for sure.** The naive answer is Regression.
  - However, age as normally predicted is an integer (nobody says age 10.5), so it is effectively **discrete**.
  - Answer: it can be treated as **either Regression or Multiclass**.
  - "คีย์เวิร์ด" for regression is a **continuous** number. Do not assume "number = regression" (see Traps).
- **[00:04:18]–[00:04:48] Question:** Three friends must all use the **same AlexNet backbone** for multiclass, multilabel and regression. What do they change? Answer: output nodes, final activation, loss.
- **[00:05:08]–[00:05:12] First thing to look at is the possible answers.** Output-layer nodes = number of classes.
- **[00:06:45]–[00:07:01] "จดไปครับ":** for **all three** task types, **output-layer nodes = number of classes/values being predicted**. He repeated it twice.
- **[00:05:46]–[00:07:39] Blood-pressure example.** Photograph the arm or skin and predict systolic and diastolic pressure. This is regression with **2 values, so 2 output nodes**. Predicting systolic only needs 1 node. "งาน Regression ยาก… หนึ่งโมเดลต่อหนึ่งค่า ก็โหดแล้ว".
- **[00:07:50]–[00:08:56] "จดไป":** **Multiclass uses Softmax; Multilabel uses Sigmoid.** He said the difference between them is in the **inference** part, not training.
  - Multiclass: pass through softmax, and the largest value is the answer.
  - Multilabel: sigmoid maps each score to 0–1, and **every node ≥ 0.5 is also an answer** (the transcript's "Test Showing" means a threshold).
- **[00:09:07]–[00:09:20] Loss for Multiclass and Multilabel:** the same family, "binary cross entropy หรือ cross entropy ก็ได้". This matches the slide p.11 footnote.
- **[00:09:39]–[00:10:13] Regression data reality.** Classification needs far less data than regression, which needs "หลาย 10-20 เท่า". So usually **1 model per regressed value**. Several values in one model is possible if you have enough data. "สิ่งที่อยากเน้นย้ำ": regression output nodes = number of values to predict.
- **[00:10:31]–[00:10:46] "จดไปครับ": Image Regression uses Linear**, meaning no activation; the score is the answer.
- **[00:11:07]–[00:11:49] Loss: Multiclass/Multilabel use Cross-Entropy; Regression uses L1 (MAE) or L2 (MSE).**
  - "จุดนี้กล้าดอกจัน" (put an asterisk here): **there is no fixed rule for L1 vs L2**. One student's project worked with L1, another's needed L2, so you must try both.
- **[00:11:52]–[00:13:24] In-class wrap-up quiz** (the class answered together):
  - Many choices, only one answer → **Multiclass**.
  - Many choices, possibly more than one answer → **Multilabel**.
  - Answer is a number → **Regression**.
  - Output node count depends on → **number of classes / answers**.
  - Final activation for multiclass at inference → **Softmax**. For multilabel → **Sigmoid, then values > 0.5 count as answers**. For regression → **Linear**.
  - Loss for multiclass/multilabel → **Cross-Entropy**. For regression → **L1 or L2**.
  - Code change is **only 2–3 lines**. Code on the internet is multiclass by default.
- **[00:13:44]–[00:15:02] Multi-task vs Multilabel (he warned "เด็ก ๆ จะสับสน").** The movie poster's answers all share **one meaning** (genre), so it is multilabel. Fruit type + rotten/not are **different meanings**, so it is multi-task.
- **[00:15:09]–[00:15:20]** Each task could be multiclass, multilabel or regression, but in practice each task is usually **multiclass**.
- **[00:15:26]–[00:15:36] Difficulty:** Regression is hardest to optimise, then Multilabel, and Multiclass is easiest.
- **[00:15:42]–[00:19:09] Lymphedema ultrasound case** (lymphedema yes/no + SEG grade + SEFS grade). Three ways to solve it:
  - (a) Product classes, 2×3 = 6.
  - (b) Shared feature extractor + task-specific heads.
  - (c) Multi-model, one CNN per task.
  - (b) fits because the outputs are clinically related.
- **[00:19:23]–[00:19:40] "Multitask จริง ๆ แล้วมันไม่ใช่ Architecture"**. It is a problem with several sub-problems. a, b and c each have different pros and cons.

---

## Likely exam questions

1. **Define and contrast multiclass vs multilabel image classification, with one medical example each.**
   - Multiclass: non-overlapping classes, one label per image. Example: classifying a skin lesion into exactly one diagnosis.
   - Multilabel: overlapping classes, several labels per image can be true. Example: a chest X-ray showing several findings at once.
   - (Medical examples are illustrative. The slides use animals, flowers and posters.)

2. **You have an AlexNet multiclass model. List exactly what you change to make it (a) multilabel, (b) an image regressor.**
   - (a) Output nodes = number of labels. Softmax → **Sigmoid**. Loss → **BCE/CE**. Inference: threshold each output at **0.5**.
   - (b) Output nodes = number of values to predict (1 on the slide). Softmax → **Linear**. Loss → **MAE (L1) or MSE (L2)**. Inference: read the raw output.

3. **Why must multilabel use sigmoid rather than softmax?**
   - Softmax outputs always sum to 1, so the classes compete. A high score for one class pushes the others down, which suits exactly one answer.
   - Sigmoid gives each class an **independent** 0–1 value, so several classes can be ≥ 0.5 at once.

4. **(Slide p.11) Scores [−3.44, 1.66, 3.91] for Drama / Sci-Fi / Action go through sigmoid → [0.03, 0.84, 0.98]. What is the prediction? The truth is [0, 0, 1]: is the model right?**
   - After the 0.5 threshold the prediction is [0, 1, 1] = **Action, Sci-Fi**.
   - Not fully correct: **Sci-Fi is a false positive**, so the loss is non-zero.

5. **(Slide p.9) The same scores through softmax give Bird 0.010, Cat 0.059, Dog 0.931. What is the prediction, and what is the CE loss if the truth is Dog?**
   - Prediction: **Dog** (argmax).
   - CE = −ln 0.931 ≈ **0.07**.

6. **Is "predicting a patient's age from a face/X-ray image" regression or multiclass? Justify.**
   - Either can be justified. Age is numeric, so the natural framing is regression.
   - Ages are normally reported as integers (discrete), so it can also be framed as multiclass.
   - It is **not** multilabel. Being a number does not automatically make it regression.

7. **Design the output layer: predict systolic and diastolic blood pressure from a skin/arm image.**
   - Regression with **2 output nodes**, **linear** activation and **MAE or MSE** loss.
   - Teacher's practical note: regression is hard and needs about 10–20× more data, so 2 separate single-output models are often preferred.

8. **(Slide p.17) A house-price CNN outputs 2,450,000; the true price is 2,500,000. Which activation and loss does the slide use, and what is the L1 error?**
   - Linear activation, L1 loss.
   - L1 error = **50,000**.

9. **Compute MAE and MSE.** True = [120, 80], predicted = [110, 84].
   - Errors: 10 and 4.
   - MAE = (10 + 4)/2 = **7**.
   - MSE = (100 + 16)/2 = **58**.

10. **Which is better for regression, L1 or L2?**
    - The teacher's answer: **no fixed rule, try both.** It depends on the task. The slide footnote says MAE/L1, MSE/L2 "และอื่น ๆ" are all allowed.
    - (Standard fact: L2 penalises large errors more and is more sensitive to outliers.)

11. **Distinguish multilabel from multi-task learning.**
    - Multilabel: many answers to **one** question with one meaning (e.g. all the genres of a film).
    - Multi-task: answers to **different** questions with different meanings (e.g. fruit freshness + fruit type, or lymphedema yes/no + SEG grade + SEFS grade).

12. **Ultrasound task: predict lymphedema (yes/no), SEG grade and SEFS grade. Describe options a, b, c from slide p.22.**
    - **a:** one feature extractor + one classifier whose single softmax output covers all tasks, i.e. the lecture's product of classes. For yes/no × SEG 0/1/2 that is a 6-class multiclass.
    - **b:** one shared feature extractor → three task-specific classifiers (Class, SEG, SEFS). Loss = sum of the three CE losses.
    - **c:** three separate feature extractors + three classifiers (multi-model).

13. **(Slide p.21) Draw / describe Multi-Task vs Multi-Model learning for a fruit image predicting freshness (2 classes) and fruit/vegetable type (11 classes).**
    - Multi-Task: one Convolution block → Flatten → two Dense heads, one with 2 output nodes and one with 11.
    - Multi-Model: two separate Conv → Flatten → Dense pipelines, with 2 and 11 outputs.

14. **List the benefits of multi-task learning and the conditions under which it works best. What is negative transfer?**
    - Benefits: saves resources (one model instead of several); joint learning that shares information to improve each task.
    - Works best when the tasks are **related** and when **data is limited**.
    - Negative transfer: when the tasks are very different or independent, a multi-task model can perform **worse** than single-task models.

15. **Shared feature extractor + task-specific heads vs shared decision-making layer: what's the difference?**
    - The first shares only the feature extractor, and each task has its own neural network (head).
    - The second shares the feature extractor **and** the neural-network layers. Only the final output layer is task-specific.

---

## Common traps

- **Using softmax for multilabel.** Softmax forces the outputs to sum to 1, so the classes compete and you effectively get one answer. Multilabel needs **sigmoid** (independent 0–1 outputs) with a **0.5 threshold**.
- **Reading multilabel with argmax.** Argmax is for multiclass. For multilabel, **every** node ≥ 0.5 is an answer. In the slide example both Sci-Fi (0.84) and Action (0.98) are output.
- **Thinking the same scores give the same answer.** The slides feed the **same** scores [−3.44, 1.66, 3.91] to softmax and to sigmoid. Softmax values sum to 1 and give one answer; sigmoid values (0.03 + 0.84 + 0.98 = 1.85) do not sum to 1 and give several answers.
- **Recomputing the slide's softmax values.** Softmax of exactly [−3.44, 1.66, 3.91] gives about [0.001, 0.095, 0.904], not the slide's [0.010, 0.059, 0.931]. Treat the slide's softmax numbers as illustrative; the sigmoid numbers do match. If you have to calculate, use the formula.
- **"It's a number, so it's regression."** The teacher called this the "เด็ก ๆ บางคนจะหลงทาง" mistake. Regression means **continuous** values. Discrete numbers such as integer ages or severity grades 0/1/2 can be treated as multiclass. SEG/SEFS grades are predicted by **classifier** heads on slides p.19 and p.22.
- **Putting an activation on the regression output.** Regression uses **Linear**, meaning the neuron outputs the summation directly with no f(·). Softmax or sigmoid would squash the output into 0–1 and could never produce "2,450,000".
- **Using cross-entropy for regression.** Regression uses L1/MAE or L2/MSE. CE is for classification.
- **Thinking regression always has 1 output node.** The slide shows 1, but the rule is **one node per value predicted**, e.g. 2 for systolic + diastolic.
- **Thinking there is a "correct" regression loss.** Neither L1 nor L2 is always right, so test both.
- **Confusing multilabel with multi-task.** Both give several answers. Multilabel answers share one meaning; multi-task answers are different questions.
- **Calling multi-task an architecture.** Multi-task is a *problem*. Shared extractor + heads (b) is one solution; a single combined classifier (a) and multi-model (c) are others.
- **Assuming multi-task always helps.** If the tasks are unrelated or independent, **negative transfer** can make it worse than single-task models.
- **Loss for multilabel.** The slides and the teacher allow "Binary Cross Entropy หรือ Cross Entropy". Strictly (standard practice), multilabel uses **binary** CE applied per label to the sigmoid probabilities. The slide draws the thresholded 0/1 vector going into the loss for illustration, but in real training the loss is computed on the probabilities.
- **Transcript spellings.** "Multi-Level" = **Multilabel**, "Multi-Card/Multicast" = **Multiclass**, "Signal" = **Sigmoid**, "Feature Attraction" = **Feature Extractor**, "TARG/ทาร์ท" = **Task**, "Test Showing" = **Threshold**, "SEG SEF" = **SEG / SEFS** grades.
