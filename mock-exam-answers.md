# เฉลยข้อสอบจำลอง: Deep Learning in Medicine

ใช้คู่กับ `mock-exam.md` ทุกข้อมี **จุดให้คะแนน (marking points)** และ **ที่มาในโน้ต** (ที่มา:)
ตัวเลขที่คำนวณทุกค่า (transposed conv, IoU, NMS, 3D conv, shape, metrics, LSTM) ตรวจด้วย Python แล้ว (เทียบ transposed conv กับ `torch.nn.functional.conv_transpose2d` และ NMS กับ `torchvision.ops.nms`)
ข้อ 1 อิงจาก spec ของโปรเจค 1 (`Project1/`) ส่วนโมเดลและผลของกลุ่มต้องเติมเอง

---

## ข้อ 1. โปรเจค 1: Thai Character & Number Recognition (15)

> ส่วนที่เป็น "ของกลุ่มเรา" (โมเดล, ผล) ให้เติมจาก `Project1/notes.md` → Our group (fill in) ก่อนสอบ

### 1.1 (4)
- **ประเภทงาน** (1): **multiclass classification** 72 คลาส ภาพหนึ่งมีคำตอบเดียว (ไม่ใช่ multilabel, ไม่ใช่ segmentation เพราะไม่ต้องบอกตำแหน่ง pixel)
- **Input / Output** (1): input = ภาพตัวอักษร 1 ตัว (crop มาแล้ว); output = class_id 3 หลัก
- **Output layer** (1): **72 output nodes**, **softmax**, loss = **cross-entropy** (`nn.CrossEntropyLoss` รับ logits, softmax อยู่ใน loss แล้ว); ผลทำนาย = argmax
- **Mapping** (1): โมเดลให้ index 0–71 ต้อง **แปลงกลับเป็น class_id** ตามตาราง (เช่น ถ้าเรียงคลาสตาม class_id: index 0 → 101 ก, index 41 → 201 ฯ, index 62 → 301 ๐) ถ้ากรอก index ตรง ๆ จะผิดทั้งหมด

### 1.2 (4)
- **ขนาด** (1.5): **pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อนแล้วค่อย resize** (เช่น 64×64) เพื่อคงสัดส่วน; resize ตรง ๆ ทำให้ตัวที่ต่างกันที่สัดส่วน (มีหาง/ไม่มีหาง, สระบนที่แบนยาว) ถูกยืดจนดูเหมือนกัน; ภาพเล็ก (วรรณยุกต์ 7×11) ขยายแล้วเบลอ จึงไม่ควรใช้ input เล็กเกินไป
- **Channel** (0.5): สีไม่มีความหมาย ใช้ grayscale; ถ้าใช้ ImageNet pretrained ให้ **duplicate เป็น 3 channels** และใช้ normalization เดียวกันทั้ง train/val/test
- **Augmentation** (2): ใช้ตอน train เท่านั้น
  - **ทำ:** หมุน/เอียงเล็กน้อย (±10–15°), เลื่อน/ย่อ-ขยายเล็กน้อย เพราะลายมือและการ crop จริงแปรปรวนแบบนี้
  - **ห้าม flip ซ้าย-ขวา:** ข้อความจริงไม่กลับด้าน และ ใ/ไ, ด/ต เปลี่ยนความหมาย
  - **ห้าม flip บน-ล่าง:** ุ กับ ิ (สระล่าง/สระบน) สลับตำแหน่งกัน
  - **ไม่ต้องปรับสี:** ตัวอักษรไม่ได้แยกกันด้วยสี
  - **ไม่หมุนมาก:** หมุนมากทำให้ ่ กับ ้ หรือ ๑ กับ ๓ ปนกัน

### 1.3 (4)
- **คู่ที่สับสน** (2):
  - พยัญชนะ: ข/ฃ, ช/ซ, ด/ต, ถ/ภ, บ/ป, ผ/ฝ, พ/ฟ, ศ/ส, ห/ฬ
  - สระ/วรรณยุกต์: **า/ๅ** (แทบเหมือนกัน), ิ/ี/ึ/ื, ุ/ู, ่/้/๊, ั/็, ใ/ไ
  - ตัวเลข: ๔/๕, ๑/๓
  - ต่างกันแค่หาง/จุด/ขีดเล็ก ๆ
- **ตรวจหา** (1): **confusion matrix** บน validation set (ดูช่องนอกแนวทแยงที่มีค่าสูง), ดูภาพที่ทายผิด
- **แก้** (1):
  - input ใหญ่ขึ้นพอให้เห็นรายละเอียดเล็ก และไม่ทำ augmentation ที่ลบรายละเอียดนั้น
  - เพิ่มข้อมูลหรือ class weight ให้คลาสที่ผิดบ่อย
  - ใช้ pretrained backbone + fine-tune
  - ensemble หรือ test-time augmentation แบบเบา ๆ (ถ้าเวลา 30 นาทีพอ)

### 1.4 (3)
- **Metric** (1): 1 คะแนนต่อภาพที่ถูก = **accuracy** บน test set (~500 ภาพ); ภาพที่ไม่ได้กรอกนับเป็นผิด จึงต้องกรอกให้ครบทุกภาพ
- **ขั้นตอน inference** (1):
  - โหลด weights → `model.eval()` → `torch.no_grad()`
  - preprocessing **เหมือน validation ทุกขั้น** (pad → resize → normalize, ไม่ augment)
  - ทำเป็น batch บน GPU → argmax → map เป็น class_id
  - เขียน CSV เรียงตาม `gt_path` → วางลงคอลัมน์ของกลุ่ม
  - ซ้อมรันทั้ง pipeline ล่วงหน้า
- **ของกลุ่มเรา** (1): โมเดลที่ใช้ + เหตุผล, validation accuracy, คะแนน test, ข้อผิดพลาดที่พบ (เติมจาก `Project1/notes.md`)

ที่มา: `Project1/notes.md` · `Chapter 8/notes.md` → Data Augmentation (ตัวอย่างตัวอักษรไทย `[00:41:04]`), Transfer Learning · `Chapter 9/notes.md` → Master table (multiclass) · `Chapter 10/notes.md` → 10. Multiclass vs multilabel, 11. Fitting images (pad ก่อน resize `[01:17:20]`), 7. Validation / testing / inference (`eval`, `no_grad`) · `Chapter 14/notes.md` → 1. Classification (accuracy, confusion matrix)

---

## ข้อ 2. Case study (15)

### 2.1 (4)
- **(ก) multilabel** (1): หนึ่งภาพ **มีโอกาส** มีได้มากกว่าหนึ่ง finding **หรือไม่มีเลย** ทุก label ตอบคำถามเดียวกัน ("ภาพนี้พบอะไรบ้าง") จึงไม่ใช่ multi-task (ตอบ "1 ภาพหลายคำตอบ" ได้แค่ครึ่ง)
- **(ข) output layer** (1.5): **3 output nodes**, **sigmoid แยกแต่ละ node** (ไม่ใช่ softmax เพราะไม่ต้องรวมเป็น 1), loss = **BCE** (`BCEWithLogitsLoss` รับ logits), อ่านผล: node ใด ≥ threshold (0.5 หรือปรับได้) = มี finding นั้น; activation ใส่นอกโมเดล
- **(ค)** (1.5):
  - วัดพื้นที่ effusion → **segmentation** (binary: 1 channel + sigmoid หรือ 2 + softmax, Dice/BCE loss) แล้วนับ pixel × ขนาด pixel; bounding box ไม่พอเพราะต้องการรูปร่างจริง
  - cardiothoracic ratio → **regression**: 1 output node, **linear (ไม่มี activation)**, loss **L1 (MAE) หรือ L2 (MSE)**; แนะนำ 1 โมเดลต่อ 1 ค่า
  - ถ้าทำพร้อมกันในโมเดลเดียวจะเป็น multi-task (head แยกต่อ task)

ที่มา: `Chapter 9/notes.md` → Master table, Multilabel vs Multi-task · `Chapter 10/notes.md` → 10. Multiclass vs multilabel vs regression · `Chapter 12/notes.md` → 1. Task types, 7. U-Net (ตาราง final layer) · `exam-hints.md` → Recurring themes

### 2.2 (4)
- **Backbone + transfer learning** (1.5): CNN pretrained ImageNet เช่น ResNet-50 / DenseNet-121 / EfficientNet; **เปลี่ยน output layer** เป็น 3 nodes เสมอ; ใช้ **fine-tuning แบบ type 3**: freeze ชั้นต้น (edge/สี ใช้ร่วมกันได้ทุกภาพ) และ train ชั้นหลัง + classifier เพราะ X-ray ไม่เหมือนภาพธรรมชาติใน ImageNet; fixed feature extraction ไม่ค่อยใช้ในงานแพทย์
- **Grayscale / ขนาด** (1): **duplicate 1 channel → 3 channels** (อย่าแก้ conv แรก ไม่งั้นเสีย pretrained weights); ใช้ normalisation แบบ ImageNet; **pad เป็นสี่เหลี่ยมจัตุรัสก่อนแล้วค่อย resize** (เช่น 224 หรือ 256) เพื่อไม่ให้สัดส่วนอวัยวะเพี้ยน
- **Augmentation** (1.5): ใช้ **ตอน train เท่านั้น** (val/test ใช้แค่ standardisation เดียวกัน); เลือกระดับ mild: rotate ±5–10°, translate/scale เล็กน้อย, brightness/contrast เล็กน้อย; **ไม่ควร**: horizontal flip (หัวใจย้ายไปขวา ทำให้ cardiomegaly/กายวิภาคผิดความหมาย), rotate มาก ๆ, very strong augmentation ที่เปลี่ยนความหมายภาพ; label มีน้อย (effusion 6%) ใช้ augmentation เพิ่มตัวอย่าง class น้อยได้

ที่มา: `Chapter 8/notes.md` → 2. Data Augmentation: purpose, 4. Weak → Strong levels, 5. Transfer Learning (ตาราง 3 patterns, Freeze) · `Chapter 10/notes.md` → 9. Pretrained backbone, 11. Fitting images (pad ก่อน resize, grayscale), Transforms emphasis

### 2.3 (3)
- **ปัญหา** (1.5): สุ่มรายภาพ → ภาพ follow-up ของผู้ป่วยคนเดียวกันไปอยู่ทั้ง train และ test (หน้าตาคล้ายมาก) โมเดลจำผู้ป่วยได้ → คะแนน test สูงเกินจริง = **data leakage**
- **วิธีที่ถูก** (1): แบ่ง **by patient** (group = patient ID, เช่น `GroupKFold` / `StratifiedGroupKFold` ให้สัดส่วน label ใกล้กัน) เป็น train / validation / test เช่น 70/15/15; ใช้ validation เลือก hyperparameter/threshold, test ใช้ครั้งเดียวตอนจบ
- **เทียบโมเดลให้ยุติธรรม** (0.5): แบ่ง split ครั้งเดียว **บันทึกเป็นไฟล์** แล้วทุกโมเดลโหลดไฟล์เดียวกัน (ภาพ test ต้องเป็น test ของทุกโมเดล) ไม่ใช้ `train_test_split` สุ่มใหม่แต่ละครั้ง แม้ใส่ seed

ที่มา: `Chapter 10/notes.md` → What the teacher emphasised → Data leakage and the train/val split · Likely exam question 7 · `exam-hints.md` → Recurring themes (split by patient)

### 2.4 (4)
TP = 45, FN = 15, FP = 30, TN = 910 (รวม 1,000; มี effusion จริง 60 ภาพ)

| Metric | สูตร | ค่า |
|---|---|---|
| Accuracy | (TP+TN)/N = 955/1000 | **0.955** |
| Precision | TP/(TP+FP) = 45/75 | **0.600** |
| Recall (Sensitivity) | TP/(TP+FN) = 45/60 | **0.750** |
| Specificity | TN/(TN+FP) = 910/940 | **0.968** |
| F1 | 2·P·R/(P+R) = 2·0.6·0.75/1.35 | **0.667** |

(2 คะแนนสำหรับการคำนวณ)
- โมเดลทาย "no effusion" ทุกภาพ: TN = 940 → accuracy = **940/1000 = 0.94** แต่ recall = 0 (พลาดผู้ป่วยทุกคน) → accuracy หลอกได้เมื่อข้อมูลไม่สมดุล (1)
- **Screening → Recall (sensitivity)** สำคัญที่สุด เพราะการพลาดผู้ป่วย (FN) อันตรายกว่า false alarm (FP ไปตรวจยืนยันต่อได้); ปรับโดย **ลด threshold ของ sigmoid** ของ effusion (ดูจาก ROC/validation), ใช้ class weight / oversampling / augmentation กับ class น้อย; รายงาน F1 และ AUC คู่ด้วย (1)

ที่มา: `Chapter 14/notes.md` → 1. Classification (Accuracy, Recall, Precision, F1, ROC/AUC, Specificity) · `exam-hints.md` → Recurring themes

---

## ข้อ 3. Video (14)

### 3.1 (3)
- **Spatial** = ข้อมูล *ภายใน* 1 เฟรม (edge, สี, รูปร่าง, texture, ตำแหน่ง) สกัดด้วย **2D CNN**; **Temporal** = การเปลี่ยนแปลง/ความสัมพันธ์ *ระหว่าง* เฟรม (motion, ลำดับ, ทิศทาง, ความเร็ว) สกัดด้วย **RNN/LSTM** (1)
- **Spatio-temporal** = สกัด spatial และ temporal **พร้อมกัน** (คีย์เวิร์ด "พร้อมกัน") (0.5)
- (1) CNN→LSTM = **S→T** (CNN สกัดทีละเฟรมจนถึง flatten แล้วค่อยส่งให้ LSTM) (0.5)
- (2) ConvLSTM = **ST**: conv ทำงานบน concat ของ xₜ (เฟรมปัจจุบัน) กับ hₜ₋₁ (ข้อมูลอดีต) channel มาจากต่างเวลา จึงได้ทั้ง spatial และ temporal ใน operation เดียว (in-class quiz ตอบ 2) (0.5)
- (3) 3D CNN = **ST**: filter เลื่อนตาม x, y และ z (เวลา) ครอบหลายเฟรมพร้อมกัน (0.5)

ที่มา: `Chapter 13/notes.md` → 1. Time series and video, 7. ConvLSTM, 9. 3D CNN, Summary table, emphasis [00:06:59], [01:31:24] quiz · `exam-hints.md` → quiz Ch 13

### 3.2 (4)
- **(ก) คำตอบแนะนำ: CNN (pretrained ImageNet, ตัดที่ flatten/GAP) ต่อเฟรม → LSTM → เอา hidden state สุดท้าย → FC 2 classes (softmax) หรือ 1 node sigmoid** (2)
  - เหตุผล: มีแค่ 300 คลิป (เล็ก) → 3D CNN มี parameter มาก **overfit ง่ายและ pretrained น้อย**; CNN 2D มี pretrained ดี ๆ ใช้ transfer learning ได้; LSTM จับ motion ของการเดิน (เฟรมเดียวแยกท่าทางที่ต่างกันแค่ลำดับไม่ได้ เช่น ลุก vs นั่ง); ใช้ resource น้อยกว่า
  - คำตอบอื่นที่ได้คะแนนถ้ามีเหตุผล: **I3D pretrained บนวิดีโอ** เป็น feature extractor (+LSTM/linear) เพราะ pretrained แล้วไม่ต้อง train 3D จากศูนย์; ConvLSTM ถ้าต้องการ ST จริง แต่ต้อง train จากศูนย์และกิน memory
  - ข้อควรระวัง: random transform ใน augmentation ทำทีละเฟรม ค่าสุ่มจะต่างกันแต่ละเฟรม
- **(ข) ความยาวไม่เท่ากัน** (0.5): ตัด/สุ่มเลือกจำนวนเฟรมคงที่ (เช่น sample 30 เฟรมกระจายทั้งคลิป, `max_frames`) หรือใช้ RNN/LSTM ซึ่งรับความยาวต่างกันได้ (แต่ยาวมากข้อมูลต้นเรื่องจะจางใน RNN ธรรมดา)
- **(ค)** (1.5):
  - (1) ขั้นตอนผ่าตัดทุกเฟรม → **temporal action segmentation** (เทียบ image segmentation; label ต่อเฟรม, output ทุก time step)
  - (2) ช่วงเวลาเลือดออก → **temporal action localization** (เทียบ object detection; class + start–end time)
  - (3) heart rate → **video regression** (เทียบ image regression; output linear)

ที่มา: `Chapter 13/notes.md` → 2. Image tasks vs video tasks, 6. Pipelines, 9. 3D CNN pros/cons, 10. Two-Stream and I3D, Likely exam question 17, emphasis [00:22:23], [01:36:20]

### 3.3 (4)
- **บทบาท** (2):
  - **cell state** = long-term memory สะสม (รวบรวม) ข้อมูลตั้งแต่ต้น ถูกแค่คูณด้วย f และบวกเพิ่ม ไม่ผ่าน tanh จึงไม่ค่อยลืม
  - **hidden state** = short-term memory "keep for predict" ส่งต่อไป step ถัดไปและไป forward layer (ไม่ใช่ prediction เอง)
  - **forget gate (Delete)** = เก็บ Cₜ₋₁ ไว้เท่าไร (1 = เก็บหมด, 0 = ลบหมด)
  - **input gate (Write)** = ให้ข้อมูลใหม่ zₜ เข้า cell state เท่าไร
  - **output gate (Read)** = เปิดเผย cell state ออกเป็น hₜ เท่าไร (อ่านอย่างเดียว ไม่แก้ C)
  - **sigmoid (0…1)** ทำหน้าที่เป็นน้ำหนัก/ประตู; **tanh (−1…1)** normalise ข้อมูลให้อยู่ช่วงมาตรฐาน zero-centred; ค่า gate เปลี่ยนตาม xₜ และ hₜ₋₁ (adaptive)
- **คำนวณ** (1.5):
  - `C_t = f ∘ C_{t-1} + i ∘ z`
    - มิติ 1: 1·2 + 0·0.4 = **2**
    - มิติ 2: 0.5·(−1) + 0.5·1 = −0.5 + 0.5 = **0**
    - **C_t = [2, 0]**
  - `h_t = o ∘ tanh(C_t)` = [0.5·tanh(2), 1·tanh(0)] = [0.5·0.964, 0] = **[0.482, 0]**
- **ตีความ** (0.5): มิติ 1 forget = 1, input = 0 → เก็บความจำเก่าไว้ทั้งหมด ไม่เขียนข้อมูลใหม่; output = 0.5 → อ่านออกไปครึ่งเดียว; มิติ 2 ลบความจำเก่าครึ่งหนึ่งแล้วเขียนข้อมูลใหม่ครึ่งหนึ่ง ซึ่งหักล้างกันพอดีเป็น 0

ที่มา: `Chapter 13/notes.md` → 4. LSTM, Formulas → LSTM, Likely exam questions 7–9

### 3.4 (3)
- **(ก)** (1.5) ใช้ `O = ⌊(I + 2P − K)/S⌋ + 1` ทีละมิติ
  - T: ⌊(16 + 2 − 3)/1⌋ + 1 = **16**; H, W: ⌊(112 + 2 − 3)/2⌋ + 1 = ⌊55.5⌋ + 1 = **56**
  - หลัง Conv3d: **(2, 16, 16, 56, 56)** (channels = จำนวน filters = 16)
  - หลัง MaxPool3d(2, 2): หารครึ่งทุกมิติ → **(2, 16, 8, 28, 28)**
  - parameters = 16 · 3 · (3·3·3) + 16 (bias) = 1,296 + 16 = **1,312**
- **(ข)** (0.5) ConvLSTM `batch_first=True` ใช้ **(B, T, C, H, W)** = (2, 16, 3, 112, 112); Conv3d ต้องการ **(B, C, T, H, W)** → `x.permute(0, 2, 1, 3, 4)`
- **(ค)** (1) output ขนาด (2−2+1) × (3−2+1) × (3−2+1) = **1 × 2 × 2**; วาง F[0] บน D[0] และ F[1] บน D[1] คูณ element-wise แล้วรวมทั้งสองชั้น

| ตำแหน่ง | D[0]·F[0] | D[1]·F[1] | รวม |
|---|---|---|---|
| (0,0) | [[1,0],[3,1]] → 1 − 1 = 0 | [[2,1],[0,1]] → 1 + 0 = 1 | **1** |
| (0,1) | [[0,2],[1,0]] → 0 − 0 = 0 | [[1,0],[1,3]] → 0 + 1 = 1 | **1** |
| (1,0) | [[3,1],[0,2]] → 3 − 2 = 1 | [[0,1],[1,0]] → 1 + 1 = 2 | **3** |
| (1,1) | [[1,0],[2,1]] → 1 − 1 = 0 | [[1,3],[0,2]] → 3 + 0 = 3 | **3** |

  **Output = [[1, 1], [3, 3]]** (เป็น cross-correlation ไม่หมุน filter เหมือนตัวอย่างในสไลด์ และเหมือน PyTorch)

ที่มา: `Chapter 13/notes.md` → Formulas → Tensor shapes, 3D conv / pooling output size (ตัวอย่าง `nn.Conv3d(16, 33, 3, stride=2)` และ parameter count) · `exam-hints.md` → slide errors Ch 13 (ตัวอย่างในสไลด์มีเลขผิด)

---

## ข้อ 4. Image segmentation (14)

### 4.1 (3)
- **What / Where** (1): classification = What ของทั้งภาพ (image prediction); detection = What + Where เป็น **กรอบสี่เหลี่ยมเล็กสุดที่ครอบวัตถุ**; segmentation = What + Where เป็น **รูปร่างจริงทีละ pixel** (pixel prediction)
- **Semantic / instance / panoptic** (2):
  - **Semantic:** ทุก pixel (รวม background) ได้ class; วัตถุ class เดียวกันแยกกันไม่ได้ → เหมาะกับ **วัดพื้นที่/ขนาด** เช่น พื้นที่ tumor หรือ lesion ใน MRI
  - **Instance:** เฉพาะ foreground, แต่ละวัตถุมี mask ของตัวเอง, background ไม่ label → เหมาะกับ **นับ/ติดตาม (tracking)** เช่น นับ nuclei/เซลล์แต่ละเซลล์ใน histology หรือติดตามเซลล์ในวิดีโอ
  - **Panoptic = semantic + instance:** ทุก pixel มี label; stuff (ไม่มีขอบชัด) แบบ semantic, things (นับได้) แบบ instance → เช่น ภาพ histology ที่ต้อง label ทุกชนิดเนื้อเยื่อ (stuff) และแยกต่อมหรือเซลล์แต่ละอัน (things)

ที่มา: `Chapter 12/notes.md` → 1. Task types, 2. Three kinds of segmentation, Likely exam questions 1–3 · `Chapter 11/notes.md` → 1.

### 4.2 (4)
- **Encoder vs decoder** (1): encoder (conv + max pooling) **ลด W×H เพิ่ม channels**, เรียนรู้ **What** แต่เสีย **Where** (spatial info); decoder (transposed conv / interpolation / unpooling) **เพิ่ม W×H ลด channels**, กู้คืน **Where** จนได้ output ขนาดเท่าภาพ
- **Skip connections** (1): encoder เสีย spatial info การ reconstruct จึงอาจเพี้ยน skip ส่ง feature ความละเอียดสูง (spatial info) จาก encoder ไปยัง decoder **ที่ scale เดียวกัน** ทำให้ขอบวัตถุแม่น; **U-Net = concatenation**, **LinkNet = element-wise addition** (ต้อง shape เท่ากัน, เบาและเร็วกว่า)
- **Channels** (1): U-Net concat → 128 + 128 = **256 channels** (64×64); LinkNet add → **128 channels** (64×64)
- **1×1 conv** (1): segmentation คือ **pixel prediction**; การตัดสินว่า pixel หนึ่งเป็น class ใดใช้ feature ทุก channel ของ pixel นั้น 1×1 conv เท่ากับ **fully connected layer ที่ apply กับแต่ละ pixel แยกกัน** โดยคง W×H ไว้; จำนวน filters = จำนวน output channels = จำนวน class (ไม่เกี่ยวกับ softmax)

ที่มา: `Chapter 12/notes.md` → 4. Encoder vs Decoder behaviour, 7. U-Net, 9. Other architectures (LinkNet), In-class quiz [01:06:24] · `exam-hints.md` → quiz Ch 12

### 4.3 (4) (ข้อละ 1)

| | งาน | 1×1 filters (output channels) | activation | label | loss |
|---|---|---|---|---|---|
| (ก) liver tumor vs background | binary | **1** (หรือ 2) | **sigmoid**, threshold 0.5 (หรือ softmax ถ้า 2) | binary mask 0/1 ขนาดเท่าภาพ | Dice / BCE + Dice / Jaccard |
| (ข) 4 ชนิดเนื้อสมอง | multiclass (1 pixel 1 class) | **4** | **softmax** (softmax2d) | class-label map 1 channel (0–3) หรือ one-hot 4 channels | cross-entropy / multiclass Dice |
| (ค) histology หลาย label ต่อ pixel | multilabel | **4** | **sigmoid แยกทุก channel**, > 0.5 = มี class นั้น | **ต้องเป็น one-hot** 4 channels (pixel เป็น 1 ได้หลาย channel) | BCE / multilabel Dice |
| (ง) ความหนากระจกตา | regression ต่อ pixel | **1** (= จำนวนค่าที่ประมาณ) | **linear / ไม่มี activation** | map ค่าต่อเนื่อง 1 channel | L1 (MAE) / L2 (MSE) |

ที่มา: `Chapter 12/notes.md` → 7. U-Net (ตาราง final layer), 8. Label (mask) formats, Likely exam question 13 · `exam-hints.md` → quiz Ch 12 (regression)

### 4.4 (3)
- นับ (1): GT = 13 pixels, Pred = 13 pixels
  - **TP = 11** (ทับกัน), **FP = 2** (Pred แถว 1 คอลัมน์ 4 และแถว 3 คอลัมน์ 5; นับจาก 0), **FN = 2** (GT แถว 1 คอลัมน์ 1 และแถว 3 คอลัมน์ 1), **TN = 36 − 15 = 21**
- คำนวณ (1.5):
  - **IoU** = TP/(TP+FP+FN) = 11/15 = **0.733**
  - **Dice** = 2TP/(2TP+FP+FN) = 22/26 = **0.846** (ตรวจ: 2·0.733/1.733 = 0.846; Dice ≥ IoU)
  - **Pixel accuracy** = (TP+TN)/36 = 32/36 = **0.889**
- เหตุผล (0.5): pixel accuracy ถูก background ที่มีจำนวนมากดันให้สูง (ภาพจริง tumor เล็กมาก ทาย background ทั้งภาพก็ได้ accuracy สูง); Dice/IoU วัดเฉพาะการทับซ้อนของ foreground; **Dice loss = 1 − Dice** ช่วยเรื่อง imbalanced data (tumor เล็ก vs background ใหญ่) และ optimise overlap โดยตรง

ที่มา: `Chapter 12/notes.md` → Formulas → Overlap metrics and losses (worked example), 9. PSPNet (Dice loss ช่วย imbalanced)

---

## ข้อ 5. Transposed convolution (14)

หลักการ (ทุกข้อ): (1) คูณ input แต่ละค่าด้วย kernel ทั้งอัน **ไม่หมุน kernel** (2) วาง block ที่ตำแหน่ง (i·s, j·s) (3) **รวมค่าที่ทับกัน** ได้ full output ขนาด (H−1)·s + k (4) ถ้ามี padding p **ตัดขอบออกข้างละ p** → `H_out = (H − 1)·s − 2p + k`

### 5.1 (6)
**Blocks** (2):
```
X[0,0]=1  → 1·K, วางที่แถว 0–2, คอลัมน์ 0–2     X[0,1]=2  → 2·K, วางที่แถว 0–2, คอลัมน์ 2–4
| 1 0 1 |                                         | 2 0 2 |
| 2 1 0 |                                         | 4 2 0 |
| 0 1 1 |                                         | 0 2 2 |

X[1,0]=3  → 3·K, วางที่แถว 2–4, คอลัมน์ 0–2     X[1,1]=-1 → -1·K, วางที่แถว 2–4, คอลัมน์ 2–4
| 3 0 3 |                                         | -1  0 -1 |
| 6 3 0 |                                         | -2 -1  0 |
| 0 3 3 |                                         |  0 -1 -1 |
```
**รวมค่าที่ทับกัน** (3): คอลัมน์ 2 ทับกันระหว่าง block ซ้าย–ขวา, แถว 2 ทับกันระหว่าง block บน–ล่าง, ตำแหน่ง (2,2) ทับ 4 block
- แถว 0: [1, 0, 1+2, 0, 2] = [1, 0, 3, 0, 2]
- แถว 1: [2, 1, 0+4, 2, 0] = [2, 1, 4, 2, 0]
- แถว 2: [0+3, 1+0, 1+2+3−1, 2+0, 2−1] = [3, 1, 3, 2, 1]
- แถว 3: [6, 3, 0−2, −1, 0] = [6, 3, −2, −1, 0]
- แถว 4: [0, 3, 3+0, −1, −1] = [0, 3, 3, −1, −1]

```
Output (5×5) = | 1  0  3  0  2 |
               | 2  1  4  2  0 |
               | 3  1  3  2  1 |
               | 6  3 -2 -1  0 |
               | 0  3  3 -1 -1 |
```
**ขนาด** (1): (2 − 1)·2 − 0 + 3 = **5** → 5×5 ✓

### 5.2 (3)
- ขนาด = (2 − 1)·2 − 2·1 + 3 = **3** → 3×3 (1)
- ตัดแถว/คอลัมน์ขอบออกข้างละ 1 จาก full output 5×5 ข้างบน (เหลือแถว 1–3, คอลัมน์ 1–3) (1)
```
Output (3×3) = | 1  4  2 |
               | 1  3  2 |
               | 3 -2 -1 |
```
- padding ใน transposed conv **ตัดขอบ output ออก** ทำให้ output **เล็กลง** (−2p) ตรงข้ามกับ conv ปกติที่ padding ช่วยไม่ให้ output หด (1)

### 5.3 (3)
| # | คำนวณ (H−1)·s − 2p + k | output |
|---|---|---|
| 1 | 3·2 − 0 + 3 | **9×9** |
| 2 | 3·2 − 2 + 3 | **7×7** |
| 3 | 3·2 − 4 + 3 | **5×5** |
| 4 | 15·2 − 0 + 2 | **32×32** |
| 5 | 7·2 − 2 + 4 | **16×16** |
| 6 | 4·1 − 2 + 3 | **5×5** (ขนาดเท่าเดิม) |

(ข้อละ 0.33, ปัดรวม 2)
- 32 → 64 (1): **k = 2, s = 2, p = 0** → 31·2 + 2 = 64 (แบบ up-conv 2×2 ของ U-Net) หรือ **k = 4, s = 2, p = 1** → 31·2 − 2 + 4 = 64

### 5.4 (2)
- **ผิด** (1): จากสูตร (H−1)·s − 2p + k เพิ่ม p ทำให้ output **เล็กลง** (เช่น 4×4, k=3, s=2: p = 0, 1, 2 → 9, 7, 5) สไลด์ของรายวิชาเองหน้า p.46–47 และที่อาจารย์พูด "ยิ่ง Padding เพิ่ม ข้อมูลก็จะยิ่งหดลง" ก็ยืนยัน; ส่วนที่ว่า stride มากขึ้น output ใหญ่ขึ้นนั้นถูก
- **ไม่ใช่ inverse** (1): transposed conv (= deconvolution ในวิชานี้) กลับแค่ **การเปลี่ยนขนาด (shape)** ไม่ได้คืนค่า pixel เดิม; weights ของ kernel เรียนรู้จากการ train

ที่มา: `Chapter 12/notes.md` → 5. Upsampling operators, Formulas → Transposed convolution output size (ตาราง), Transposed conv worked example, Common traps (padding) · `exam-hints.md` → Errors on the slides (Ch 12 p.41)

---

## ข้อ 6. Object detection (14)

### 6.1 (3)
- **IoU** (1.5):
  - intersection: x = max(40, 60) … min(100, 120) = 60 … 100 → 40; y = max(40, 50) … min(100, 110) = 50 … 100 → 50; inter = 40 × 50 = **2,000**
  - area G = 60 × 60 = 3,600; area P = 60 × 60 = 3,600
  - union = 3,600 + 3,600 − 2,000 = **5,200**
  - **IoU = 2,000 / 5,200 = 0.385**
- **TP/FP** (0.5): 0.385 < 0.5 → P เป็น **FP**; G ไม่มี prediction ที่ match → G เป็น **FN** (พลาด nodule)
- **YOLO format** (1): w = h = 60, xc = yc = 70 → `0 0.1367 0.1367 0.1172 0.1172` (70/512, 70/512, 60/512, 60/512) รูปแบบคือ `class x_center y_center width height`; normalise เป็น 0–1 เพื่อให้ **resize ภาพแล้ว label ไม่เปลี่ยน**

ที่มา: `Chapter 11/notes.md` → Formulas → IoU, Bounding-box formats (YOLO normalised), Precision / Recall / AP / mAP

### 6.2 (4)
ทุกกล่อง 50×50 → area = 2,500, union = 5,000 − inter

| คู่ | intersection | union | IoU |
|---|---|---|---|
| A, B | (60−15)·(60−12) = 45·48 = 2,160 | 2,840 | **0.761** |
| A, D | (60−22)·(60−20) = 38·40 = 1,520 | 3,480 | **0.437** |
| A, F | (60−12)·(58−10) = 48·48 = 2,304 | 2,696 | **0.855** |
| A, C / A, E | ไม่ทับ | | 0 |
| C, D | ไม่ทับ | | 0 |
| C, E | (150−105)·(148−100) = 45·48 = 2,160 | 2,840 | **0.761** |
| B, D | (65−22)·(62−20) = 43·42 = 1,806 | 3,194 | **0.565** |

**ขั้นตอน (threshold 0.5)** (2.5):
1. เรียง score: A 0.95, B 0.90, C 0.85, D 0.80, E 0.75, F 0.60
2. **Reference A:** B (0.761 > 0.5) → **ลบ**; C (0) เก็บ; D (0.437 ≤ 0.5) เก็บ; E (0) เก็บ; F (0.855) → **ลบ**; เหลือ C, D, E
3. **Reference C:** D (0) เก็บ; E (0.761) → **ลบ**
4. **Reference D:** ไม่มีกล่องเหลือให้เทียบ → จบ
- **ผลสุดท้าย: A, C, D**
- **IoU(B, D) = 0.565** (0.5) > 0.5 แต่ D ไม่ถูกลบเพราะ **B ถูก A ลบไปแล้วในรอบแรก** NMS เทียบเฉพาะกับกล่องที่ยังเหลือและมี score สูงกว่า และ IoU(A, D) = 0.437 ไม่ถึง threshold
- **threshold 0.4** (1): IoU(A, D) = 0.437 > 0.4 → D ถูกลบด้วย → เหลือ **A, C**; threshold ต่ำ = ลบมากขึ้น (เสี่ยงลบวัตถุจริงที่อยู่ใกล้กัน), threshold สูง = เหลือกล่องซ้ำมากขึ้น (Ultralytics ใช้ `iou` default 0.6)

ที่มา: `Chapter 11/notes.md` → Formulas → NMS (slide steps 2.6.1–2.6.4), Likely exam question 10, Common traps (NMS)

### 6.3 (4)
- **ตารางเปรียบเทียบ** (2):

| | R-CNN | Fast R-CNN | Faster R-CNN |
|---|---|---|---|
| หา Where | Selective Search บนภาพจริง | Selective Search บนภาพจริง | **RPN** (anchors + objectness + NMS, trainable) |
| CNN ต่อภาพ | ~2,000 ครั้ง (ทุก region) | **1 ครั้ง** (ทั้งภาพ) | 1 ครั้ง |
| ปรับขนาด region | **Warping** 224×224 บนภาพ (เสียสัดส่วน) | **RoI pooling (max)** บน feature map | RoI pooling |
| Classifier | **SVM** (AlexNet ตัด FC สุดท้าย) | FC + **softmax** + bbox regressor | FC + softmax + regressor |
| เวลาทดสอบรวม proposal | 50 s | 2 s (25×) | 0.2 s (250×) |

- **Fast R-CNN ยัง multi-stage** (1): ยังใช้ Selective Search หา Where แยกจากส่วน What (และ Selective Search train ไม่ได้) ซึ่งกลายเป็นคอขวด: 0.32 s ไม่รวม proposal แต่รวมแล้ว 2 s ต่อภาพ → ไม่ real-time
- **Anchors** (1): k = 3 × 3 = 9 ต่อตำแหน่ง → 32 × 32 × 9 = **9,216 anchors**; 1×1 conv **9 filters** สำหรับ objectness score และ **36 filters (9 × 4)** สำหรับ coordinates (x, y, w, h)

ที่มา: `Chapter 11/notes.md` → 3–5 (R-CNN, Fast R-CNN, Faster R-CNN), 7. Detector family comparison, Formulas → Anchor and filter counts · `exam-hints.md` → quiz Ch 11 และ slide errors (Faster R-CNN ได้ความเร็ว ไม่ใช่ mAP: 66.0 / 66.9 / 66.9)

### 6.4 (3)
- **Output tensor** (1.5): ต่อ cell = B·5 + C = 2·5 + 3 = 13 → **7 × 7 × 13**
  - ค่า 1–5: box 1 (confidence, x, y, w, h); ค่า 6–10: box 2; ค่า 11–13: P(class | object) 3 classes (**1 ชุดต่อ cell**)
  - x, y = offset ภายใน cell; w, h normalise ด้วยขนาดภาพ; cell ที่มีจุดศูนย์กลางวัตถุเป็นผู้รับผิดชอบ; score สุดท้าย = class prob × confidence แล้วทำ NMS
- **Polyp detection real-time** (1.5): เลือก **one-stage (YOLO)** เพราะหา What + Where พร้อมกันใน network เดียว (end-to-end) เร็วพอสำหรับวิดีโอ; Faster R-CNN ~0.2 s/frame ยังช้า; ใช้รุ่นเล็ก (n/s) ถ้ารันบนอุปกรณ์จำกัด; จุดอ่อน YOLOv1: **1 class ต่อ cell** และ **วัตถุเล็กอยู่ติดกันหลายอัน** ทำได้ไม่ดี (polyp เล็กหลายอันใกล้กัน)

ที่มา: `Chapter 11/notes.md` → 6. YOLO, Formulas → YOLO numbers, Likely exam questions 11, 12, 15(a)

---

## ข้อ 7. Coding (14)

### 7.1 (5): bugs (ครบ 6 จุดได้เต็ม, ข้อละ ~0.8)

| # | บรรทัด | ปัญหา | วิธีแก้ |
|---|---|---|---|
| 1 | 9 | `val_tf = train_tf` ทำ augmentation (flip, rotate) ตอน validation ผลไม่คงที่และวัดผลไม่น่าเชื่อถือ | สร้าง `val_tf` ที่มี standardisation เดียวกัน (Resize, ToDtype, Normalize) แต่ **ไม่มี** random transforms |
| 2 | 11 | `models.resnet50()` ได้ **random weights** ไม่ใช่ pretrained | `models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)` |
| 3 | หลัง 11 | ไม่ได้เปลี่ยน output layer โมเดลยังให้ **1000 classes** | `model.fc = nn.Linear(2048, 4)` ก่อน `model.to(device)` และก่อนสร้าง optimizer (ไม่งั้น optimizer ไม่รู้จัก parameter ของ fc ใหม่) |
| 4 | 21 | ใส่ `softmax` ก่อน `CrossEntropyLoss` ซึ่งมี (log-)softmax อยู่แล้ว → softmax ซ้ำ gradient ผิดเพี้ยน | ส่ง raw scores (logits): `outputs = model(images)` |
| 5 | 22 | `labels.float()` แต่ CrossEntropyLoss ต้องการ class index ชนิด **long (int64)** (float จะถูกตีเป็น probability และ shape ไม่ตรง → error) | `criterion(outputs, labels.long())` หรือ `labels` เฉย ๆ |
| 6 | ก่อน 21 | ไม่มี `optimizer.zero_grad()` → gradient **สะสม** ข้าม batch | เรียก `optimizer.zero_grad()` ทุก batch |
| 7 | 23–24 | `step()` ก่อน `backward()` → อัปเดตด้วย gradient เก่า/ไม่มี gradient | ลำดับ: forward → loss → `loss.backward()` → `optimizer.step()` |
| 8 | 27 | validation ไม่มี `model.eval()` (Dropout/BatchNorm ยังเป็นโหมด train, BN ยังอัปเดต running stats ด้วยข้อมูล val) และไม่มี `torch.no_grad()` (เปลือง memory/ช้า) | `model.eval()` แล้ว `with torch.no_grad():` |
| (เสริม) | — | ไม่ได้ save model | `torch.save(model.state_dict(), ...)` ที่ best epoch (ข้อ 7.2) |

### 7.2 (4): โค้ดที่ถูกต้อง
```python
val_tf = v2.Compose([
    v2.ToImage(), v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

classes = ["MEL", "NV", "BCC", "BKL"]
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(classes))     # 2048 -> 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0.0
for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)                 # prediction (forward)
        loss = criterion(outputs, labels)       # loss
        loss.backward()                         # gradients
        optimizer.step()                        # update weights

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(1)
            correct += (preds == labels.to(device)).sum().item()
    val_acc = correct / len(val_loader.dataset)
    print(epoch, round(loss.item(), 4), round(val_acc, 4))
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best.pt")   # weights only
```
**Marking:** head ถูก + pretrained (1), ลำดับ zero_grad → forward → loss → backward → step (1), eval + no_grad ใน validation (1), save `state_dict` เมื่อ val ดีขึ้น (1)
(โค้ดนี้ทดสอบรันบนข้อมูลจำลองแล้ว)

### 7.3 (3)
```python
from PIL import Image

def predict(path):
    model = models.resnet50()                       # สร้าง architecture เดิม
    model.fc = nn.Linear(2048, len(classes))        # ต้องแก้ fc เหมือนตอน train
    model.load_state_dict(torch.load("best.pt", map_location=device))
    model.to(device).eval()
    image = Image.open(path).convert("RGB")
    x = val_tf(image).unsqueeze(0).to(device)       # (1, 3, 224, 224)
    with torch.no_grad():
        score = torch.softmax(model(x).squeeze(0), dim=0)
    conf, idx = torch.max(score, 0)
    print(f"Predicted: {classes[idx]} {conf.item() * 100:.2f}%")
```
**Marking:** rebuild architecture เดิม + `load_state_dict` (1), ใช้ transform **แบบ validation (ไม่มี augmentation)** + `unsqueeze(0)` เพิ่ม batch dimension (1), `eval()` + `no_grad()` + softmax + max (1)
(ตรงนี้ไม่ต้องโหลด pretrained ImageNet weights เพราะจะถูกเขียนทับด้วย `best.pt` อยู่แล้ว)

### 7.4 (2)
- **Multilabel** (1): output nodes = จำนวน labels; loss → `nn.BCEWithLogitsLoss()` และ label เป็น multi-hot **float** เช่น `[1, 0, 1, 0]`; inference ใช้ `torch.sigmoid(out) > threshold` (0.5 ในไดอะแกรม, 0.3 ในโค้ดสไลด์) แทน softmax + argmax; validation metric เปลี่ยนเป็นต่อ label; ส่วน training loop เหมือนเดิม
- **`torch.cuda.is_available()`** (1): **ได้ครึ่งเดียว** ตรวจ 2 อย่าง: (1) มี GPU หรือไม่ และ (2) GPU นั้น **ใช้งานได้ (available)** หรือไม่ เช่น ไม่ถูกคนอื่นใช้เต็ม; ถ้า False โค้ดจะไปรันบน CPU แบบเงียบ ๆ (อาจารย์เล่าว่างาน 3 วันกลายเป็นเกินสัปดาห์) งานจริงอาจบังคับ `.cuda()` ให้ error แทน

ที่มา: `Chapter 10/notes.md` → 5. The training loop (ตารางหน้าที่แต่ละคำสั่ง), 6. Saving the model, 7. Validation / testing / inference, 8. Transforms emphasis, 9. Pretrained backbone, 10. Multiclass vs multilabel vs regression, Common traps · `exam-hints.md` → quiz Ch 10

---

## เฉลยแบบฝึกหัดเสริม: Transposed convolution

### ฝึก 1 (s = 1, p = 0): ขนาด (2−1)·1 + 2 = **3×3**
Blocks: 2K = [[2,4],[6,8]] ที่ (0,0); 1K = [[1,2],[3,4]] ที่ (0,1); 0K ที่ (1,0); 3K = [[3,6],[9,12]] ที่ (1,1)
- แถว 0: [2, 4+1, 2] = [2, 5, 2]
- แถว 1: [6, 8+3+0+3, 4+6] = [6, 14, 10]
- แถว 2: [0, 0+9, 12] = [0, 9, 12]
```
| 2  5  2 |
| 6 14 10 |
| 0  9 12 |
```
(ตำแหน่งกลาง (1,1) ทับกันครบ 4 block)

### ฝึก 2 (s = 2, p = 1): full = (3−1)·2 + 3 = 7×7, output = 7 − 2 = **5×5**
มีเพียง X[0,0]=1, X[0,2]=2, X[1,1]=1, X[2,0]=2, X[2,2]=1 ที่ไม่เป็น 0 → วาง K (plus shape) × ค่าเหล่านั้นที่ (0,0), (0,4), (2,2), (4,0), (4,4)
```
Full 7×7:                      ตัดขอบข้างละ 1 → Output 5×5:
| 0 1 0 0 0 2 0 |
| 1 2 1 0 2 4 2 |              | 2 1 0 2 4 |
| 0 1 0 1 0 2 0 |              | 1 0 1 0 2 |
| 0 0 1 2 1 0 0 |              | 0 1 2 1 0 |
| 0 2 0 1 0 1 0 |              | 2 0 1 0 1 |
| 2 4 2 0 1 2 1 |              | 4 2 0 1 2 |
| 0 2 0 0 0 1 0 |
```
(block ที่อยู่ติดกันทับกันแค่ที่มุมของ K เช่น full (2,2) = มุมขวาล่างของ block X[0,0] + มุมซ้ายบนของ block X[1,1] = 0 + 0 เพราะมุมของ K เป็น 0 ผลจึงดูเหมือนไม่ทับ; ขนาดตรงกับสไลด์ที่ 3×3 input + stride 2 ได้ 5×5)

### ฝึก 3 (s = 3)
**(ก) p = 0:** ขนาด (2−1)·3 + 3 = **6×6**; stride = kernel size = 3 → block วางชิดกันพอดี **ไม่มีการทับซ้อน** (ต้องมี s < k จึงจะทับ)
```
|  1  2  1 -1 -2 -1 |
|  0  1  0  0 -1  0 |
|  1  0  1 -1  0 -1 |
|  2  4  2  0  0  0 |
|  0  2  0  0  0  0 |
|  2  0  2  0  0  0 |
```
(block ซ้ายบน = 1·K, ขวาบน = −1·K, ซ้ายล่าง = 2·K, ขวาล่าง = 0·K)

**(ข) p = 1:** ขนาด 6 − 2 = **4×4** (ตัดขอบข้างละ 1)
```
| 1  0  0 -1 |
| 0  1 -1  0 |
| 4  2  0  0 |
| 2  0  0  0 |
```

ที่มา: `Chapter 12/notes.md` → 5. Upsampling operators (transposed conv), Formulas → Transposed convolution output size, Transposed conv worked example (slides p.29–40, p.44–47)
