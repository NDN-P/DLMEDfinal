# Exam hints: quick sheet (Ch 8–15)

Full notes are in each folder's `notes.md`. **Exam = 7 written questions: project, case study, video, segmentation, transposed conv by hand, object detection, coding. See `exam-structure.md`.**

## In-class quiz questions (teacher asked these live, so likely testable)
| Ch | When | Question | Answer |
|---|---|---|---|
| 10 | P1 00:56 | What does `torch.cuda.is_available()` check? | Whether a GPU exists **and** is usable/free ("checks there's a GPU" was marked ถูกครึ่งเดียว) |
| 10 | P1 | Which line trains the model? | Not `model(x)` (that only predicts). `loss.backward()` computes gradients, `optimizer.step()` updates weights |
| 11 | — | 9 anchors per position: how many filters for score / coords? | 9 score, 36 (= 9×4) coordinate |
| 12 | 01:06 | Why does U-Net end with a 1×1 conv? | Per-pixel prediction, like an FC layer applied to each pixel's channel vector (not "because of softmax") |
| 12 | 01:21 | Adapt U-Net for regression? | Linear/no activation, output channels = number of values to estimate |
| 13 | 01:32 | ConvLSTM: (1) spatial then temporal, or (2) spatio-temporal? | **2**, because the conv runs on current frame + previous hidden state together (CNN→LSTM is option 1) |

## Errors on the slides (don't copy them in the exam)
| Ch | Slide | Problem | Correct |
|---|---|---|---|
| 9 | softmax example | scores [−3.44, 1.66, 3.91] do not give Dog 0.931 | ≈ [0.001, 0.095, 0.904]; compute with the formula |
| 11 | spoken | "Faster R-CNN is more accurate" | Slide 51: mAP 66.0 / 66.9 / 66.9, so the gain is **speed** |
| 11 | spoken | ~30,000 anchors on 50×50 | 50·50·9 = 22,500 |
| 12 | p.41 | "more padding → bigger output" (transposed conv) | Output = (H−1)·s − 2p + k, so more padding gives a **smaller** output (the deck's own p.46–47 agree) |
| 13 | 3D-conv steps | −21, −37, −22 | −20, −36, −18 |
| 15 | spoken | "add then LayerNorm" | Slide: **Pre-LN**: U = X + MHA(LN(X)), Y = U + MLP(LN(U)) |

## Recurring themes the teacher pushed in every chapter
- Output layer by task: multiclass → softmax + CE; multilabel → sigmoid per node + threshold + BCE; regression → linear + L1/L2. Output nodes = classes / values.
- "It depends on the dataset/application": no fixed best answer for L1 vs L2, CNN vs Transformer, or video architectures. Explain the trade-off.
- Medical data: recall matters for screening; accuracy misleads on imbalanced data; split by patient to avoid **data leakage**; augmentation on train only.
- Expect "why" questions, not just "what".

## Exam format
From friend-reminder.md: see `exam-structure.md`.
