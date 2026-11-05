# Final exam structure

## The teacher's briefing (from a friend's notes)
**7 written questions, answer all.** When you enter the exam, the teacher explains **within the first 5 minutes** how to write the answers. Come late and answer in the wrong format = no points.

| # | Question topic | Study from |
|---|---|---|
| 1 | Project 1: Thai Character & Number Recognition (โปรเจค 1) | [`Project1/notes.md`](Project1/notes.md): 72-class **multiclass classification** → 72 nodes + softmax + CE, map index → 3-digit class_id. Pad to square before resize (Ch 10 P1 `[01:17:20]`), augmentation for Thai characters: no colour change, no flips, slight rotate/tilt (Ch 8 `[00:41:04]`), confusable pairs + confusion matrix, score = accuracy, inference in 30 min. **Fill in your group's model and results there.** |
| 2 | Case study: how to apply the knowledge | Pick task type → output layer/activation/loss → architecture → metric for a medical scenario. Ch 9 (output layers), Ch 10 (pipeline, data leakage), Ch 8 (transfer learning, augmentation), Ch 14 (which metric) |
| 3 | Video | `materials/Chapter 13/notes.md`: spatial vs temporal, CNN→LSTM vs ConvLSTM vs 3D conv vs two-stream, LSTM gates, 3D-conv output size |
| 4 | Image segmentation | `materials/Chapter 12/notes.md`: semantic/instance/panoptic, encoder-decoder, U-Net (skip = concat), 1×1 output conv, output channels = classes, Dice/IoU |
| 5 | **Transposed conv by hand** (write and calculate manually) | `materials/Chapter 12/notes.md` worked example (slides p.29–47). Output = (H−1)·s − 2p + k. Practise the full matrix by hand. **Slide p.41 is wrong**: more padding → *smaller* output |
| 6 | Object detection | `materials/Chapter 11/notes.md`: IoU, NMS steps, anchors (k = 9, 9 score + 36 coord filters), R-CNN → Fast → Faster (RPN), YOLO grid/loss, two-stage vs one-stage |
| 7 | Coding | `materials/Chapter 10/notes.md`: training loop order (`train()` → `zero_grad` → forward → loss → `backward` → `step`), `eval()` + `no_grad()`, replacing `model.fc`, `state_dict` save/load, transforms (augment train only), `cuda.is_available()` |

No standalone question on Ch 8, 9 or 14 (metrics), but they feed Q2 (case study) and Q1 (project).

**Ch 15 (Transformers) is not on the exam.** Skip it.

## Did the recordings match?
The friend's list isn't in any of the 9 recordings: I searched the Whisper transcripts and YouTube captions for สอบ, ข้อสอบ, ปลายภาค, คะแนน, final and exam. The Ch 14 recording is stopped right after the TA talks (`[01:56:43]` "stop ให้หน่อยนะครับ"), so the briefing was most likely given after that and never recorded. Nothing in the recordings contradicts the friend's list. Its topics match chapters 10–13 exactly, and the teacher's comments in Ch 11 `[01:15:52]` about seniors' hands-on project exams fit the project + coding questions.
