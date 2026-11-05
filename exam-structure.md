# โครงสร้างข้อสอบ final

## สิ่งที่อาจารย์บอก (จากโน้ตของเพื่อน)
**ข้อสอบเขียน 7 ข้อ ต้องตอบทุกข้อ** พอเข้าห้องสอบ อาจารย์จะอธิบายวิธีเขียนคำตอบ **ภายใน 5 นาทีแรก** ถ้ามาสายแล้วตอบผิดรูปแบบ = ไม่ได้คะแนน

| # | หัวข้อคำถาม | อ่านจาก |
|---|---|---|
| 1 | Project 1: Thai Character & Number Recognition (โปรเจค 1) | [`Project1/notes.md`](Project1/notes.md): **multiclass classification** 72 คลาส → 72 nodes + softmax + CE แล้ว map index → class_id 3 หลัก ทำ pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อน resize (Ch 10 P1 `[01:17:20]`) augmentation สำหรับตัวอักษรไทย: ห้ามเปลี่ยนสี ห้าม flip หมุน/เอียงได้นิดหน่อย (Ch 8 `[00:41:04]`) คู่ตัวอักษรที่สับสนกัน + confusion matrix คะแนน = accuracy ทำ inference ใน 30 นาที **เติมโมเดลและผลของกลุ่มตัวเองไว้ในไฟล์นั้น** |
| 2 | Case study: เอาความรู้ไปใช้ยังไง | เลือกประเภทงาน → output layer/activation/loss → architecture → metric สำหรับโจทย์ทางการแพทย์ Ch 9 (output layers), Ch 10 (pipeline, data leakage), Ch 8 (transfer learning, augmentation), Ch 14 (ใช้ metric ไหน) |
| 3 | Video | `materials/Chapter 13/notes.md`: spatial vs temporal, CNN→LSTM vs ConvLSTM vs 3D conv vs two-stream, LSTM gates, ขนาด output ของ 3D-conv |
| 4 | Image segmentation | `materials/Chapter 12/notes.md`: semantic/instance/panoptic, encoder-decoder, U-Net (skip = concat), 1×1 output conv, output channels = จำนวนคลาส, Dice/IoU |
| 5 | **Transposed conv ด้วยมือ** (เขียนและคำนวณเอง) | ตัวอย่างใน `materials/Chapter 12/notes.md` (สไลด์ p.29–47) Output = (H−1)·s − 2p + k ฝึกทำทั้ง matrix ด้วยมือ **สไลด์ p.41 ผิด**: padding มากขึ้น → output *เล็กลง* |
| 6 | Object detection | `materials/Chapter 11/notes.md`: IoU, ขั้นตอน NMS, anchors (k = 9, 9 score + 36 coord filters), R-CNN → Fast → Faster (RPN), YOLO grid/loss, two-stage vs one-stage |
| 7 | Coding | `materials/Chapter 10/notes.md`: ลำดับ training loop (`train()` → `zero_grad` → forward → loss → `backward` → `step`), `eval()` + `no_grad()`, เปลี่ยน `model.fc`, save/load `state_dict`, transforms (augment แค่ train), `cuda.is_available()` |

ไม่มีข้อที่ถาม Ch 8, 9 หรือ 14 (metrics) แยกเดี่ยว ๆ แต่ใช้ตอบ Q2 (case study) และ Q1 (project)

**Ch 15 (Transformers) ไม่ออกสอบ** ข้ามไปได้เลย

## ตรงกับที่อัดไว้ไหม?
รายการของเพื่อนไม่มีอยู่ในคลิปที่อัดไว้ทั้ง 9 คลิป: ค้นใน transcript จาก Whisper และซับ YouTube ด้วยคำว่า สอบ, ข้อสอบ, ปลายภาค, คะแนน, final และ exam แล้ว คลิป Ch 14 หยุดอัดทันทีหลัง TA พูดจบ (`[01:56:43]` "stop ให้หน่อยนะครับ") เลยน่าจะบอกเรื่องข้อสอบหลังจากนั้นและไม่ได้อัดไว้ ไม่มีอะไรในคลิปที่ขัดกับรายการของเพื่อน หัวข้อตรงกับบท 10–13 พอดี และที่อาจารย์พูดใน Ch 11 `[01:15:52]` เรื่องรุ่นพี่สอบ project แบบลงมือทำ ก็เข้ากับข้อ project + coding
