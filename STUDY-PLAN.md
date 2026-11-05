# What to read for the final exam

The exam is 7 written questions and you answer all of them. **Arrive on time**: in the first 5 minutes the teacher explains how to write the answers.
Ch 15 (Transformers) is **not** on the exam. Skip it. In Ch 14, skip **6. RAG**.

Go through the steps in order. Each step lists what to read, then a check: do the mock-exam question first, and only then open the answer key.
Mock exam = [mock-exam.md](mock-exam.md), answers = [mock-exam-answers.md](mock-exam-answers.md).

## 0. Start (15 min)
- [ ] [exam-structure.md](exam-structure.md): the 7 questions and where each one comes from
- [ ] [exam-hints.md](exam-hints.md): the whole file. It covers the in-class quiz answers, the **slide errors** (don't copy them) and the recurring themes.

## 1. Q5: Transposed conv by hand (do first; it's easy marks with practice)
Read [Chapter 12/notes.md](Chapter%2012/notes.md):
- [ ] 5. Upsampling operators and 6. Padding, Stride, Dilation
- [ ] Formulas: **Transposed convolution output size**, Output = (H−1)·s − 2p + k
- [ ] Formulas: **Transposed conv worked example** (slides p.29–40). Redo it on paper.
- [ ] Nearest-neighbour and bilinear interpolation
- [ ] Common traps. **Slide p.41 is wrong**: more padding gives a *smaller* output.

Check: mock Q5 plus the extra practice at the end of the answer key. Write out the full matrix every time.

## 2. Q7: Coding
Read [Chapter 10/notes.md](Chapter%2010/notes.md):
- [ ] 2. Defining the model (`class Net(Module)`)
- [ ] 4. Learning algorithm + GPU check
- [ ] **5. The training loop**: `train()` → `zero_grad` → forward → loss → `backward` → `step`
- [ ] 6. Saving the model (`state_dict`)
- [ ] 7. Validation / testing / inference: `eval()` + `torch.no_grad()`
- [ ] 8. Transforms, Dataset, DataLoader: augment the **train set only**
- [ ] 9. Pretrained backbone: replace `model.fc`
- [ ] What the teacher emphasised, then Common traps

Check: mock Q7. Write the code by hand, without looking.

## 3. Q6: Object detection
Read [Chapter 11/notes.md](Chapter%2011/notes.md):
- [ ] 1. Classification vs Detection vs Segmentation
- [ ] 3–5. R-CNN → Fast R-CNN → Faster R-CNN (RPN). Know what each one fixed. Faster R-CNN mainly made it **faster**.
- [ ] 6. YOLO (grid, loss) and 7. the detector family comparison (two-stage vs one-stage)
- [ ] Formulas: **IoU**, **NMS steps**, anchors and filters (k = 9 → 9 scores and 36 box coordinates, plus 50×50×9 = 22,500 anchors), Precision / Recall / AP / mAP
- [ ] What the teacher emphasised, then Common traps

Check: mock Q6. Calculate IoU and run NMS by hand.

## 4. Q4: Image segmentation
Read [Chapter 12/notes.md](Chapter%2012/notes.md):
- [ ] 1. Task types (What / Where)
- [ ] 2. Semantic vs instance vs panoptic segmentation
- [ ] 3–4. Autoencoder, encoder vs decoder
- [ ] 7. **U-Net**: skip connections **concatenate**, a 1×1 output conv, output channels = number of classes
- [ ] 8. Mask formats and 9. other architectures
- [ ] Formulas: **Dice / IoU** and their losses
- [ ] In-class quiz [01:06:24], What the teacher emphasised, Common traps
- [ ] [Chapter 14/notes.md](Chapter%2014/notes.md) → 3. Image Segmentation and its Formulas section

Check: mock Q4.

## 5. Q3: Video
Read [Chapter 13/notes.md](Chapter%2013/notes.md):
- [ ] 1–2. Video vs image tasks, spatial vs temporal information
- [ ] 3–5. Why RNN, **LSTM gates** (forget / input / output), RNN vs LSTM vs GRU
- [ ] 6–7. Pipelines: CNN → LSTM, **ConvLSTM**
- [ ] 9–10. **3D CNN**, Two-Stream, I3D
- [ ] **Summary: video architectures** table
- [ ] Formulas: tensor shapes (watch the axis order), **3D conv output size**
- [ ] What the teacher emphasised, then Common traps
- [ ] [Chapter 14/notes.md](Chapter%2014/notes.md) → 4. Video Processing

Check: mock Q3.

## 6. Q2: Case study (applying the knowledge to a medical problem)
Answer in this order: **task type → output layer / activation / loss → architecture + transfer learning → augmentation → split → metric**, and give the *why* for each step.
- [ ] [Chapter 9/notes.md](Chapter%209/notes.md): the three task types, **Master table**, multilabel vs multi-task
- [ ] [Chapter 8/notes.md](Chapter%208/notes.md): 2–4. augmentation (what *not* to use for medical images), 5. Transfer learning (the 3 patterns, freezing)
- [ ] [Chapter 10/notes.md](Chapter%2010/notes.md): 10. Multiclass vs multilabel vs regression, 11. Fitting images (pad before resize), **data leakage** (split by patient)
- [ ] [Chapter 14/notes.md](Chapter%2014/notes.md): 1. Classification metrics (screening → recall; accuracy misleads on imbalanced data) and 5. Regression (MAE / MSE)

Check: mock Q2.

## 7. Q1: Project 1 (Thai Character & Number Recognition)
- [ ] [Project1/notes.md](Project1/notes.md): read the whole file
- [ ] Be able to say from memory: the task, input → output, why 72 nodes + softmax + CE, mapping an index to a class_id, pad then resize, augmentation dos and don'ts, confused pairs, score = accuracy
- [ ] **Your group's** model, why you chose it, your accuracy or test score, one mistake and how you fixed it, and who did what

Check: mock Q1.

## 8. Final run (the day before)
- [ ] Do the whole [mock-exam.md](mock-exam.md) under time (3 h), then mark it with the answer key
- [ ] Re-read the **Common traps** sections of Ch 10–13 and the slide-errors table in [exam-hints.md](exam-hints.md)
- [ ] Redo one transposed-conv problem and one IoU/NMS problem on paper

## If you only have 2 hours
1. [exam-hints.md](exam-hints.md)
2. Transposed conv worked example (Ch 12)
3. Training loop + `eval` / `no_grad` (Ch 10 §5, §7)
4. IoU + NMS (Ch 11 Formulas)
5. U-Net + Dice (Ch 12 §7, Formulas)
6. Video summary table + LSTM gates (Ch 13)
7. [Project1/notes.md](Project1/notes.md) plus your group's results
