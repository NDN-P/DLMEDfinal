# Exam hints: สรุปสั้น (Ch 8–14)

โน้ตเต็มอยู่ใน `notes.md` ของแต่ละ folder **ข้อสอบ = เขียน 7 ข้อ: project, case study, video, segmentation, transposed conv ด้วยมือ, object detection, coding ดู `exam-structure.md`**

## คำถามที่อาจารย์ถามในห้อง (ถามสด ๆ เลยน่าจะออกสอบ)
| Ch | ตอนไหน | คำถาม | คำตอบ |
|---|---|---|---|
| 10 | P1 00:56 | `torch.cuda.is_available()` เช็คอะไร? | เช็คว่ามี GPU **และ** ใช้ได้/ว่างอยู่ (ตอบว่า "เช็คว่ามี GPU" ได้แค่ ถูกครึ่งเดียว) |
| 10 | P1 | บรรทัดไหนที่ train โมเดล? | ไม่ใช่ `model(x)` (อันนั้นแค่ทำนาย) `loss.backward()` คำนวณ gradient ส่วน `optimizer.step()` อัปเดต weight |
| 11 | — | 9 anchors ต่อตำแหน่ง: ต้องใช้กี่ filter สำหรับ score / coords? | score 9, coordinate 36 (= 9×4) |
| 12 | 01:06 | ทำไม U-Net จบด้วย 1×1 conv? | เพื่อทำนายทีละ pixel เหมือนเอา FC layer ไปใช้กับ channel vector ของแต่ละ pixel (ไม่ใช่ "เพราะ softmax") |
| 12 | 01:21 | ปรับ U-Net ให้ทำ regression ยังไง? | ใช้ linear/ไม่มี activation, output channels = จำนวนค่าที่จะประมาณ |
| 13 | 01:32 | ConvLSTM: (1) spatial แล้วค่อย temporal หรือ (2) spatio-temporal? | **2** เพราะ conv ทำบน frame ปัจจุบัน + hidden state ก่อนหน้าไปพร้อมกัน (CNN→LSTM คือแบบ 1) |

## จุดที่สไลด์ผิด (อย่าลอกไปตอบในข้อสอบ)
| Ch | สไลด์ | ปัญหา | ที่ถูก |
|---|---|---|---|
| 9 | ตัวอย่าง softmax | score [−3.44, 1.66, 3.91] ไม่ได้ Dog 0.931 | ≈ [0.001, 0.095, 0.904] ให้คำนวณจากสูตร |
| 11 | ที่พูด | "Faster R-CNN แม่นกว่า" | สไลด์ 51: mAP 66.0 / 66.9 / 66.9 สิ่งที่ดีขึ้นคือ **ความเร็ว** |
| 11 | ที่พูด | ~30,000 anchors บน 50×50 | 50·50·9 = 22,500 |
| 12 | p.41 | "padding มากขึ้น → output ใหญ่ขึ้น" (transposed conv) | Output = (H−1)·s − 2p + k ดังนั้น padding มากขึ้นทำให้ output **เล็กลง** (p.46–47 ในสไลด์ชุดเดียวกันก็ตรงกับอันนี้) |
| 13 | ขั้นตอน 3D-conv | −21, −37, −22 | −20, −36, −18 |

## เรื่องที่อาจารย์ย้ำทุกบท
- Output layer ตามประเภทงาน: multiclass → softmax + CE; multilabel → sigmoid ทีละ node + threshold + BCE; regression → linear + L1/L2 จำนวน output nodes = จำนวนคลาส / จำนวนค่า
- "ขึ้นกับ dataset/application": ไม่มีคำตอบตายตัวว่า L1 หรือ L2 ดีกว่า หรือ architecture ไหนดีสุดสำหรับวิดีโอ ให้อธิบาย trade-off
- ข้อมูลการแพทย์: recall สำคัญสำหรับงานคัดกรอง; accuracy หลอกได้ถ้าข้อมูล imbalanced; แบ่งข้อมูลตามผู้ป่วยเพื่อกัน **data leakage**; ทำ augmentation แค่ train
- เตรียมตอบคำถาม "ทำไม" ไม่ใช่แค่ "คืออะไร"

## รูปแบบข้อสอบ
ดู `exam-structure.md`
