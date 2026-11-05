# คู่มืออ่านสอบ Final: ทุกอย่างที่ต้องรู้

**ไฟล์เดียวจบ อ่านแล้วสอบผ่าน** แต่ละข้อมี 5 ส่วน: เขาจะถามอะไร, ต้องรู้อะไร, สูตรพร้อมตัวอย่างคิดเลขครบทุกขั้น, **คำตอบตัวอย่างที่ลอกไปเขียนได้เลย**, และกับดักที่ทำให้เสียคะแนน
ตัวเลขทุกตัวเช็กด้วย Python แล้ว และ code ทุก block ก็รันมาแล้ว
รายละเอียดเชิงลึกอยู่ใน `materials/Chapter N/notes.md` ฝึกทำได้ที่ [mock-exam.md](mock-exam.md) แล้วตรวจคำตอบกับ [mock-exam-answers.md](mock-exam-answers.md)

**ไม่ออกสอบ:** Chapter 15 (Transformers) และ Chapter 14 §6 RAG

## สารบัญ
| ข้อสอบ | หัวข้อ | คะแนน (mock) | Chapter | ไปที่ |
|---|---|---|---|---|
| Q1 | Project 1: Thai Character & Number Recognition | 15 | Project1, Ch 8–10 | [Q1](#q1) |
| Q2 | Case study: เอาความรู้ไปใช้ | 15 | Ch 8, 9, 10, 14 | [Q2](#q2) |
| Q3 | Video | 14 | Ch 13 (+14 §4) | [Q3](#q3) |
| Q4 | Image segmentation | 14 | Ch 12 (+14 §3) | [Q4](#q4) |
| Q5 | Transposed conv คิดมือ | 14 | Ch 12 | [Q5](#q5) |
| Q6 | Object detection | 14 | Ch 11 (+14 §2) | [Q6](#q6) |
| Q7 | เขียน code (PyTorch) | 14 | Ch 10 | [Q7](#q7) |

## ทำยังไงให้ได้คะแนนเยอะ
1. **มาให้ตรงเวลา** อาจารย์จะอธิบายรูปแบบการตอบใน 5 นาทีแรก มาสาย = ตอบผิดรูปแบบ = ไม่ได้คะแนน
2. **ตอบให้ครบทั้ง 7 ข้อ** เว้นว่างได้ 0 แต่ตอบไม่ครบแต่มาถูกทางก็ยังได้คะแนน
3. **บอกเหตุผลเสมอ** เขียนแค่ "U-Net" ไม่นับเป็นคำตอบ ต้องเขียนว่า "U-Net เพราะ skip connection ช่วยดึงรายละเอียดเชิงตำแหน่งที่ encoder ทำหายไปกลับมา" ถึงจะใช่
4. **ข้อคำนวณ: เขียนทุกขั้น** เขียนสูตร แทนตัวเลข ให้ผลลัพธ์ แล้วเช็กขนาด ข้อ transposed conv ให้เขียนทุก block X·K และบอกว่ามันไปวางตรงไหน
5. **ใช้สูตรสำเร็จ (recipe)** กับข้อออกแบบ Q2 กับ Q1: task type → output layer + loss → model + transfer learning → preprocessing/augmentation → split → metric → ความเสี่ยง
6. **วาดรูปเล็กๆ** ประกอบ architecture: U-Net, Faster R-CNN, CNN→LSTM, two-stream ใช้เวลาไม่นานและแสดงว่าเราเข้าใจ
7. **อย่าลอกจุดที่สไลด์ผิด** มีรายการอยู่ใน [exam-hints.md](exam-hints.md) และเขียนซ้ำไว้ในหัวข้อกับดักของแต่ละข้อด้านล่าง
8. **"ขึ้นอยู่กับ dataset/application"** คือคำตอบของอาจารย์เวลาถามว่า "อันไหนดีที่สุด?" ให้พูดประโยคนี้ แล้วอธิบาย trade-off ต่อ

## ลำดับการอ่าน
อ่านตามลำดับนี้ (เก็บคะแนนง่ายก่อน) อ่านจบแต่ละข้อแล้วให้ลองทำข้อนั้นใน mock exam *ก่อน* เปิดดูเฉลย

1. **Q5** Transposed conv: ฝึกทำขั้นตอนบนกระดาษ 3 รอบ
2. **Q7** Coding: เขียนโปรแกรม template จากความจำ 1 รอบ
3. **Q6** Detection: คิด IoU 1 ข้อและ NMS 1 ข้อด้วยมือ
4. **Q4** Segmentation: ตาราง output layer และ Dice/IoU
5. **Q3** Video: ตารางเปรียบเทียบและ gate ของ LSTM
6. **Q2** Case study: จำ recipe 7 ขั้น แล้วลองเขียนคำตอบตัวอย่างเองสัก 1 ข้อ
7. **Q1** Project 1: เติมช่องว่างด้วย model และผลลัพธ์ของกลุ่มเรา แล้วลองพูดออกเสียง
8. **วันก่อนสอบ:** ทำ mock exam เต็มชุดจับเวลา (3 ชม.) แล้วอ่านหัวข้อ **กับดัก** ทั้งหมดอีกรอบ

**มีแค่ 2 ชั่วโมง?** อ่าน Q5 §3, Q7 §3 (โปรแกรม template), Q6 §3 (IoU + NMS), ตาราง output layer ของ Q4 + Dice, ตารางเปรียบเทียบของ Q3, recipe ของ Q2 และโครงคำตอบของ Q1

---

<a id="q1"></a>

## Q1 — Project 1: Thai Character & Number Recognition

### เขาจะถามอะไร
Mock Q1 (15 คะแนน):
- 1.1 task type, input/output, output layer, ต้องทำอะไรกับ output ก่อนส่ง
- 1.2 วิธี resize, grayscale/3 channels, augmentation ที่ควรใช้และ **ไม่ควรใช้ พร้อมเหตุผลจากตัวอักษรไทย**
- 1.3 คู่ที่สับสนกัน ≥ 3 คู่ คนละประเภท, หาเจอได้ยังไง, แก้ยังไง
- 1.4 metric ที่ตรงกับวิธีให้คะแนน, ขั้นตอน inference ภายใน 30 นาที, **model และผลลัพธ์จริงของกลุ่มเรา**

ข้อสอบถามเรื่อง *สิ่งที่เราทำจริง* ดังนั้นเติมช่องว่างให้เสร็จก่อนสอบ

### โครงคำตอบให้เติม: "อธิบายโปรเจกต์ของคุณ"
1. **Task** จำแนกภาพตัวอักษรไทยที่ crop มาแล้ว 1 ภาพ ออกเป็น **72 classes**: พยัญชนะ 41 ตัว (101–141), สระ/วรรณยุกต์/สัญลักษณ์ 21 ตัว (201–221), เลขไทย 10 ตัว (301–310) เป็น **multiclass** (หนึ่งภาพ หนึ่งคำตอบ) ไม่ใช่ multilabel และไม่ใช่ segmentation
2. **Input → output**
   - Input: ไฟล์ `.jpg` 1 ไฟล์ ขนาดประมาณ **7×11 ถึง 180×177 px** (W×H) เป็น grayscale
   - Output: **class_id 3 หลัก** หลักแรก = ประเภท หลักที่ 2–3 = ลำดับ เช่น 205 = ี
   - Test set: ประมาณ 500 ภาพ กรอกคำตอบลงคอลัมน์ของกลุ่มเราใน Google Sheet ภายใน **30 นาที**
3. **ทำไมออกแบบแบบนี้** ข้อมูลที่มีคือรูปร่างอย่างเดียว และสิ่งที่แยก class ออกจากกันคือสัดส่วนภาพ (aspect ratio) กับเส้นเล็กๆ เราเลยรักษาสัดส่วนไว้ และไม่ใช้ augmentation ที่ทำให้ความหมายเปลี่ยน
4. **เตรียมข้อมูล**
   - Grayscale → copy ให้เป็น 3 channels เพื่อใช้กับ pretrained backbone
   - **Pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize** เป็น ____ × ____ (เช่น 64×64)
   - Normalise ด้วย ____ (ImageNet mean/std ถ้าใช้ pretrained) ใช้เหมือนกันทั้ง train/val/test
   - แบ่ง train/val ____ % **ก่อน** ทำ augmentation และ save ไว้ครั้งเดียว
   - Augmentation (เฉพาะ train): ____ (ดูหัวข้อด้านล่าง)
5. **Model** Backbone ____ (pretrained บน ImageNet? ____ ; freeze layer ไหน ____) เปลี่ยน layer สุดท้ายเป็น `Linear(____, 72)` เหตุผลที่เลือก model นี้: ____ (เลือกความซับซ้อนของ model ให้เหมาะกับข้อมูล: 72 classes บางตัวแทบเหมือนกัน)
6. **Training** Loss `CrossEntropyLoss`; optimizer ____, lr ____, batch ____, epochs ____ (early stopping? ____) เลือก model ที่ดีที่สุดจาก **validation accuracy**
7. **Metric / ผลลัพธ์** Validation accuracy ____ % คะแนน test ____ / ~500 (= ____ %)
8. **Error** คู่ที่สับสนหลักๆ จาก confusion matrix: ____ ↔ ____, ____ ↔ ____ สิ่งที่ลองแก้: ____ ผลหลังแก้: ____
9. **ใครทำอะไร** ____ (data/preprocessing), ____ (model/training), ____ (inference/ส่งคำตอบ), ____ (วิเคราะห์ผล)

### ประเด็นเทคนิคสำคัญ (คำตอบตัวอย่าง)
- **Output layer**
  - **72 nodes, softmax, cross-entropy** `nn.CrossEntropyLoss` รับ logits ดิบ (softmax อยู่ในตัว loss แล้ว) และ target เป็นเลข index จำนวนเต็ม
  - คำทำนาย = **argmax**
  - ทำไมใช้ softmax: มีคำตอบถูกแค่ class เดียว และความน่าจะเป็นรวมกันได้ 1
- **แปลง index → class_id**
  - Model ให้ index 0–71 ออกมา ต้อง **แปลงกลับเป็น class_id** ด้วยตาราง class ถ้าเขียน index ดิบลงไปจะได้ 0
  - ถ้าเรียง class ตามชื่อโฟลเดอร์ (`ImageFolder` เรียงชื่อโฟลเดอร์ให้ *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*) ลำดับโฟลเดอร์ = ลำดับ class_id: index 0 → 101 ก, 40 → 141 ฮ, 41 → 201 ฯ, 61 → 221 ์, 62 → 301 ๐, 71 → 310 ๙ (เช็กกับตาราง class แล้ว)
  - ชื่อโฟลเดอร์เป็นรหัส TIS-620 (161 = ก, 249 = ๙) รหัสที่ไม่มีโฟลเดอร์ (165 ฅ, 166 ฆ, 172 ฌ, 174 ฎ, 198 ฦ …) คือไม่มีใน dataset
- **Pad ก่อนแล้วค่อย resize**
  - ขนาดและสัดส่วนภาพต่างกันมาก ถ้า resize ตรงๆ ตัวอักษรจะโดนยืด แล้ว class ที่ต่างกันแค่สัดส่วน (มีหาง/ไม่มีหาง, สระบนแบนๆ) จะเริ่มดูเหมือนกัน อาจารย์ยกตัวอักษรไทยเป็นตัวอย่าง: "เราจะไม่ Resize ก่อน Padding"
  - เครื่องหมายเล็กๆ (7×11 เช่น ่ ้) จะเบลอเวลาขยาย และเส้นบางๆ คือตัวที่ใช้แยกมันออกจากกัน เพราะฉะนั้นอย่าทำ input เล็กเกินไป
- **Grayscale** สีไม่ได้มีความหมายอะไร ใช้ 1 channel หรือ copy เป็น 3 channels สำหรับ ImageNet backbone
- **Augmentation (เฉพาะ train, คำถามในห้อง Ch 8)**
  - **ควรใช้:** หมุน/เอียงนิดหน่อย (±10–15°), เลื่อนและย่อขยายนิดหน่อย เพราะลายมือและการ crop ก็ต่างกันประมาณนี้
  - **ห้าม flip ซ้าย-ขวา:** ตัวหนังสือจริงไม่มีทางกลับด้าน และ ใ/ไ, ด/ต จะเปลี่ยนความหมาย
  - **ห้าม flip บน-ล่าง:** ุ (ข้างล่าง) ↔ ิ (ข้างบน) ความหมายสลับกัน
  - **ห้าม colour jitter:** ตัวอักษรไม่ได้แยกกันด้วยสี อาจารย์ติว่าการใช้ augmentation ที่ไม่เกี่ยวข้องคือ "ไม่เข้าใจข้อมูล"
  - **ห้ามหมุนเยอะ:** ่/้ และ ๑/๓ จะปนกัน
- **คู่ที่สับสนกัน** (ยกตัวอย่างประเภทละ 1 คู่):
  - พยัญชนะ: ข/ฃ, ช/ซ, ด/ต, ถ/ภ, บ/ป, ผ/ฝ/พ/ฟ, ศ/ส, ห/ฬ, ท/ฑ
  - สระ/วรรณยุกต์: **า/ๅ**, ิ/ี/ึ/ื, ุ/ู, ่/้/๊, ั/็, ใ/ไ
  - ตัวเลข: ๔/๕, ๓/๑
  - ต่างกันแค่หาง จุด หรือขีดเล็กๆ
  - **หาเจอยังไง:** ดู confusion matrix บน validation (ช่องนอกเส้นทแยงที่ค่าสูง) แล้วเปิดดูภาพที่ทายผิด
  - **แก้ยังไง:**
    - ใช้ input ใหญ่ขึ้นให้รายละเอียดยังอยู่ และไม่ใช้ augmentation ที่ลบรายละเอียดนั้นทิ้ง
    - เพิ่มข้อมูล หรือใช้ class weights กับ class พวกนั้น
    - ใช้ pretrained backbone + fine-tune
    - Augmentation ตาม class (class ที่ accuracy ต่ำให้ใช้แบบ **strong**)
    - Ensemble หรือ TTA เบาๆ ถ้าเวลา 30 นาทีพอ
- **คะแนน = accuracy** ได้ 1 คะแนนถ้าคำทำนาย = gt_class_id ไม่งั้นได้ 0 คะแนนจึงเท่ากับ **accuracy บน ~500 ภาพ** เว้นว่างหรือทำไม่ทันก็ได้ 0 เท่ากับตอบผิด ดังนั้นกรอกให้ครบทุกแถวเสมอ
- **Inference 30 นาที**
  1. สร้าง architecture เดิมขึ้นมาใหม่ (`fc` เหมือนเดิม) แล้ว `load_state_dict`
  2. **`model.eval()` + `torch.no_grad()`**
  3. **Preprocessing เหมือนตอน validation** (pad → resize → normalise, ไม่มี augmentation)
  4. รันเป็น batch บน GPU → argmax → แปลงเป็น class_id
  5. เขียน CSV ตามลำดับใน `gt_path` (ts_img_001 …) → วางลงคอลัมน์ของกลุ่มเราเท่านั้น
  6. ซ้อมรันทั้ง pipeline กับโฟลเดอร์ล่วงหน้า อาจารย์ให้มาเป็น **folder path** ดังนั้นทำ inference ได้ทีละภาพอย่างเดียวไม่พอ

### กับดัก
- เรียกว่า multilabel หรือ segmentation จริงๆ คือ **multiclass** หนึ่งภาพมีตัวอักษรเดียว
- ส่ง **index 0–71** แทน **class_id** ผิดทุกแถว
- Resize ตรงๆ (สัดส่วนเพี้ยน) หรือ resize ก่อน pad
- ใช้ flip / colour jitter "เพราะใครๆ ก็ใช้" ต้องให้เหตุผลทุก augmentation โดยอิงกับตัวอักษรไทย
- ใช้ augmentation หรือ random crop ตอน validation/inference หรือ preprocessing ตอน test ไม่เหมือนตอน validation
- ลืม `model.eval()` (Dropout/BN ยังอยู่ในโหมด train) หรือลืม `no_grad()`
- เว้นแถวว่างหรือทำไม่ทันเวลา: ได้ 0 เท่ากับตอบผิด
- ทำ augmentation ก่อนแบ่ง train/val: ภาพที่ augment แล้วจะหลุดเข้าไปอยู่ใน validation
- ตอบข้อ 1.4 แบบกว้างๆ ทั่วไป คะแนนอยู่ที่ model ตัวเลข และ error ของ **กลุ่มเราเอง**

---

<a id="q2"></a>

## Q2 — Case study (เอาความรู้ไปใช้กับโจทย์การแพทย์)

### เขาจะถามอะไร
โจทย์สถานการณ์ในโรงพยาบาล (ชนิดภาพ, ขนาด, จำนวนภาพ/คนไข้, label, ความสมดุลของ class) แล้วมีข้อย่อย 3–4 ข้อ ใน mock มี 2.1 task + output layer, 2.2 architecture/transfer/augmentation, 2.3 การแบ่งข้อมูล, 2.4 metric จาก confusion matrix ให้เตรียมรับมือกับโจทย์พลิก เช่น "ตอนนี้เขาอยากวัด X เพิ่มด้วย"

| คำในโจทย์ | Task type | ตัวอย่างทางการแพทย์ที่เจอบ่อย (จาก notes) |
|---|---|---|
| "หนึ่งภาพมีการวินิจฉัยเดียวเท่านั้น" | **Multiclass** | รอยโรคผิวหนัง → วินิจฉัย 1 โรค |
| "อาจเจอหลาย finding **หรือไม่เจอเลย**" | **Multilabel** | chest X-ray: pneumonia / effusion / cardiomegaly |
| "ใช่/ไม่ใช่", คัดกรอง | Binary (2 softmax หรือ 1 sigmoid) | คัดกรอง COVID, ป่วย / ไม่ป่วย |
| "วัด / ประมาณค่าเป็นตัวเลข" (ค่าต่อเนื่อง) | **Regression** | cardiothoracic ratio, ความดันจากภาพแขน/ผิวหนัง, อายุ |
| "วัดพื้นที่ / วาดขอบรอยโรค" | **Segmentation** (แล้วเอาจำนวน pixel × ขนาด pixel) | พื้นที่ effusion, เนื้องอกใน MRI (2 % ของ pixel) |
| "อยู่ตรงไหน / มีกี่อัน" (แค่ box ก็พอ) | **Object detection** | หาตำแหน่งรอยโรคด้วย box + class + confidence |
| "ถามหลายคำถาม *คนละเรื่อง* กับภาพเดียว" | **Multi-task** | ultrasound: lymphedema ใช่/ไม่ใช่ + เกรด SEG + เกรด SEFS |
| video, ทั้งคลิป → 1 label | Video classification | ท่าออกกำลังกาย, กิจกรรม |
| video, ใส่ label ทุก frame / ทุกขั้นตอน | Temporal action segmentation | **วิเคราะห์ขั้นตอนการผ่าตัด** |
| video, หาว่าเหตุการณ์เกิดตอนไหน | Temporal action localization | **ผู้สูงอายุล้มในกล้อง CCTV** (class, start, end) |
| video → ตัวเลข | Video regression | อัตราการเต้นหัวใจจากวิดีโอผิวหนัง |

### Recipe การตอบ: เขียน 7 หัวข้อนี้ทุกครั้ง
1. **Task type + เหตุผล** ยกคำจากโจทย์มาเลย: "หนึ่งภาพ *อาจ* มีมากกว่าหนึ่ง finding หรือไม่มีเลย → multilabel" (หนึ่งภาพมีโอกาสที่จะเป็นไปได้มากกว่าหนึ่งคำตอบ) ถ้าเขียนแค่ "1 ภาพหลายคำตอบ" ได้ครึ่งเดียว
2. **Output layer:** จำนวน node, activation, loss และวิธีอ่านคำทำนาย (argmax / threshold / ค่าดิบ) activation อยู่นอก model: `CrossEntropyLoss` กับ `BCEWithLogitsLoss` รับ logits
3. **Architecture + transfer learning**
   - CNN ที่ pretrained บน ImageNet (ResNet-50 / DenseNet-121 / EfficientNet; MobileNetV3 ถ้าต้องรันบนมือถือ)
   - เปลี่ยน layer สุดท้าย (`model.fc = nn.Linear(2048, n)`)
   - **Type 3**: freeze layer ต้นๆ (ขอบ/เส้น ซึ่งภาพทุกแบบใช้ร่วมกัน) แล้ว train conv layer หลังๆ + head ใหม่ เพราะ X-ray/ultrasound ≠ ภาพถ่ายใน ImageNet แบบ fixed feature extraction แทบไม่ใช้ในงานภาพทางการแพทย์
   - **1 → 3 channels**: copy channel grayscale (อย่าไปแก้ conv แรก ไม่งั้น pretrained weights หาย) ใช้ ImageNet mean/std
   - **Pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize** เป็น 224/256 (เราจะไม่ Resize ก่อน Padding) เพื่อไม่ให้สัดส่วนอวัยวะโดนยืด
4. **Augmentation** ใช้เฉพาะ train (val/test ใช้ standardisation แบบเดียวกัน ไม่มี augmentation) บิดภาพเบาๆ: หมุน ±5–10°, เลื่อน/ย่อขยายนิดหน่อย, ปรับ brightness/contrast นิดหน่อย บอกด้วยว่าอันไหนห้ามใช้และเพราะอะไร (ตาราง E) ใช้เพิ่มตัวอย่างของ class ที่มีน้อยได้ด้วย
5. **การแบ่งข้อมูล** **แบ่งตามคนไข้**: ภาพทุกภาพของคนไข้คนเดียวต้องอยู่ใน split เดียวกัน (ภาพ follow-up แทบเหมือนกัน ถ้าแบ่งแบบสุ่มรายภาพ = data leakage = คะแนน test สูงเกินจริง) แบ่ง train/val/test เช่น 70/15/15 แบบ stratified ตาม label แบ่ง **ครั้งเดียวแล้ว save เป็นไฟล์** แล้วโหลดไฟล์เดิมให้ทุก model ("ภาพ test ต้องเป็น test สำหรับทุก model") Val ใช้เลือก hyperparameter และ threshold ส่วน test แตะแค่ครั้งเดียว
6. **Metric + เหตุผล** คัดกรอง → **recall/sensitivity** (FN = พลาดคนไข้ คือ error ที่แย่ที่สุด) ตรวจยืนยัน → **precision** สำคัญทั้งคู่ → **F1** ข้อมูลไม่สมดุล → **ห้ามใช้ accuracy อย่างเดียว** ให้เพิ่ม F1/AUC Regression → MAE/RMSE **พร้อมหน่วยและช่วงค่า** + Pearson r Segmentation → Dice/IoU Detection → IoU ≥ 0.5 + mAP
7. **ความเสี่ยง / ข้อจำกัด**
   - Leakage
   - Class imbalance → ลด threshold, ใช้ class weights / oversampling
   - Domain shift: ต่างโรงพยาบาล ต่างเครื่อง หรือพฤติกรรมผู้ใช้ต่างกัน (บทเรียนเรื่องพันธุ์ไม้: k-fold ได้ 90 % แต่ตอน demo จริงเกือบ 0)
   - Dataset เล็ก → overfitting
   - Finding ที่บาง/เล็กหายไปตอน resize
   - Model เป็นแค่ *ผู้อ่านคนที่สอง* หมอต้องเป็นคนยืนยัน

### ตารางที่ต้องรู้

**A. Task → output layer → loss → metric**

| Task | Output nodes | Activation | Loss | อ่านคำทำนาย | Metric |
|---|---|---|---|---|---|
| Multiclass | = จำนวน class | Softmax (รวมกันได้ 1) | Cross-entropy | argmax | Accuracy (ถ้าสมดุล), P/R/F1 ราย class, confusion matrix |
| Binary | 2 (softmax) หรือ 1 (sigmoid) | Softmax / Sigmoid | CE / BCE | ตัวที่มากกว่า / p ≥ threshold | Recall, precision, F1, specificity, AUC |
| Multilabel | = จำนวน label | **Sigmoid แยกแต่ละ node** (อิสระต่อกัน) | **BCE** (`BCEWithLogitsLoss`; ในสไลด์บอกว่าใช้ CE ก็ได้) | ทุก node ที่ ≥ threshold (0.5 ในรูป, 0.3 ใน code; ปรับได้) → อาจเป็น **ไม่มีเลย** ก็ได้ | recall/precision/F1/AUC ราย label |
| Regression | = จำนวนค่า (แนะนำ 1 ค่าต่อ 1 model) | **Linear** (ไม่มี) | L1/MAE หรือ L2/MSE (ไม่มีกฎตายตัว ลองทั้งคู่) | output ดิบ | MAE, RMSE + Pearson r, บอกหน่วยและช่วงค่า |
| Segmentation | 1×1 conv; channels = จำนวน class (binary: 1 + sigmoid หรือ 2 + softmax; multilabel: N + sigmoid; regression ราย pixel: linear) | ราย pixel | Dice / Jaccard loss (ดีกับข้อมูลไม่สมดุล), BCE/CE | argmax / threshold ราย pixel | Dice (= F1), IoU / mIoU; **ไม่ใช่** pixel accuracy |
| Detection | ต่อ anchor/cell: class scores + ค่า box 4 ค่า (9 anchors → 9 score + 36 coord filters) | softmax/sigmoid สำหรับ class, linear สำหรับ box | class loss + box regression (YOLO: SSE) | confidence threshold + NMS | IoU ≥ 0.5 **และ** class ถูก = TP; AP ราย class, mAP |
| Video | CNN → LSTM (spatial แล้วค่อย temporal), ConvLSTM หรือ 3D CNN (spatio-temporal); head ตาม task | ตาม task | ตาม task | ราย clip / ราย frame / (class, start, end) | Acc/F1 (clip); frame-wise acc, Edit, F1@k (segmentation); tIoU, mAP@tIoU (localization) |
| Multi-task | 1 head ต่อ 1 task ขนาดตาม task นั้น | แต่ละ head ของใครของมัน | ผลรวม loss ของทุก task | แต่ละ head ใช้กฎของตัวเอง | metric ราย task |

ทางเลือกของ multi-task: (a) เอา class มาคูณกันแล้วใช้ softmax เดียว (2 × 3 = 6), (b) ใช้ feature extractor ร่วมกัน + head แยกตาม task, (c) 1 model ต่อ 1 task ใช้ (b) ถ้า task เกี่ยวข้องกัน ถ้า task ไม่เกี่ยวกัน → **negative transfer**

**B. สูตร metric** (Positive = เป็นโรค)

| Metric | สูตร | เน้นอะไร |
|---|---|---|
| Accuracy | (TP+TN)/N | หลอกตาเวลาข้อมูลไม่สมดุล |
| Precision | TP/(TP+FP) | โดน FP กด (เตือนผิด) |
| Recall = Sensitivity = TPR | TP/(TP+FN) | โดน FN กด (พลาดคนไข้) |
| Specificity | TN/(TN+FP) = 1 − FPR | recall ของ class คนปกติ *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* |
| F1 | 2PR/(P+R) = **2TP/(2TP+FP+FN)** | harmonic mean ต่ำถ้าตัวใดตัวหนึ่งต่ำ |
| MAE / MSE / RMSE | (1/n)Σ\|e\| / (1/n)Σe² / √MSE | MAE ทนต่อ outlier; MSE หน่วยเป็น unit² ลงโทษ error ใหญ่หนัก; RMSE กลับมาเป็นหน่วยเดิม |
| ROC / AUC | TPR (แกน y) vs FPR (แกน x) ในทุก threshold | 1 = สมบูรณ์แบบ, 0.5 = เดาสุ่ม, < 0.5 = กลับด้าน |

*ตัวอย่างคิดเลข (เช็กด้วย Python แล้ว)* คัดกรองเบาหวานขึ้นจอตา ภาพ test 500 ภาพ ป่วย 50 คน TP = 42, FN = 8, FP = 45, TN = 405
- Accuracy = 447/500 = **0.894**
- Precision = 42/87 = **0.483**
- Recall = 42/50 = **0.840**
- Specificity = 405/450 = **0.900**
- F1 = 84/(84+45+8) = 84/137 = **0.613**
- Model ที่ตอบว่า "ปกติ" ทุกคน: accuracy = 450/500 = **0.900** (สูงกว่าอีก!) แต่ recall = **0** นี่คือเหตุผลที่ accuracy หลอกเราเวลาข้อมูลไม่สมดุล

*ตัวอย่าง regression (เช็กแล้ว)* Cardiothoracic ratio ค่าจริง [0.45, 0.50, 0.55, 0.60] ค่าทำนาย [0.47, 0.49, 0.60, 0.52] |e| = 0.02, 0.01, 0.05, 0.08
- MAE = **0.040**, MSE = **0.00235**, RMSE = **0.048**
- ตรงนี้ MSE < MAE เพราะ error ทุกตัว < 1 ประโยค "MAE น้อยกว่า MSE เสมอ" จึงผิด

**C. ใช้ metric ไหนเมื่อไหร่**

| สถานการณ์ | เลือก | เพราะอะไร |
|---|---|---|
| คัดกรอง (ห้ามพลาดใคร) | **Recall** + ลด threshold | FN มีราคาแพง ส่วน FP แค่ต้องไปตรวจเพิ่ม |
| ยืนยันก่อนรักษา (เคมีบำบัด) | **Precision** | FP แปลว่าได้รับการรักษาที่ไม่จำเป็นและเป็นอันตราย |
| ตรวจหามะเร็ง: ผิดแบบไหนก็แย่ | **F1** | balance ระหว่าง P กับ R |
| เทียบ model ในทุก threshold | **AUC** | ไม่ขึ้นกับ threshold ได้ตัวเลขเดียว |
| class หายาก / ไม่สมดุล | recall, precision, F1, AUC ราย class | accuracy โดน class ใหญ่ครอบงำ |
| Segmentation รอยโรคเล็ก | Dice / IoU | pixel accuracy โดน background ดันให้สูง (เนื้องอก 2 % → 98 %) |
| Regression | MAE (ลงโทษเท่ากัน) หรือ RMSE (error ใหญ่อันตราย) **+ Pearson r** พร้อมหน่วย + ช่วงค่าที่เป็นไปได้ | error ต่ำ ≠ ตามแนวโน้มถูก (MAE 2.4 กับ r = 0 vs MAE 8 กับ r = 1) |

**D. รูปแบบ transfer learning** (freeze = ท่อนนั้นจะไม่ถูกเทรน)

| Type | ทำอะไร | ใช้เมื่อไหร่ |
|---|---|---|
| 1 | ใช้ pretrained model ทั้งตัวตามเดิม | แทบเป็นไปไม่ได้: output 1000 nodes ≠ class ของเรา |
| 2 | Freeze feature extractor แล้ว train แค่ classifier ใหม่ | ใช้บ่อยสุด; ข้อมูลใกล้เคียงภาพถ่ายทั่วไป; ข้อมูลน้อยมาก |
| 3 | Freeze conv layer **ต้นๆ** แล้ว train conv layer **หลังๆ** + classifier | ภาพไม่เหมือน ImageNet: **X-ray, ultrasound** → ค่า default ของงานการแพทย์ |

ข้อจำกัดของ input: pretrained net ต้องการ **RGB 3 channels** เลยต้อง copy channel grayscale ใช้ GAP หรือ resize ขนาดตายตัวสำหรับ H×W เลือก backbone จาก **accuracy, ความเร็ว, ขนาด** "ใหม่สุด/ลึกสุด" ไม่ได้แปลว่าดีกว่าเสมอ เลือกความซับซ้อนของ model ให้เหมาะกับข้อมูล (จำนวน class, แต่ละ class คล้ายกันแค่ไหน)

**E. Augmentation**

| กลุ่ม | สมาชิก | ใช้กับงานการแพทย์ |
|---|---|---|
| Distortion: geometric | rotate, shift, scale, crop, flip | rotate/shift/scale เล็กน้อย: **ได้** Flip: **ได้เฉพาะถ้า anatomy ยอม** |
| Distortion: photometric | brightness, contrast, blur, noise, colour | brightness/contrast เล็กน้อย (เครื่อง scan/การ expose ต่างกัน): ได้ |
| Occlusion | Random Erase, Cutout, Hide-and-Seek, GridMask | แก้ปัญหา "ดูแค่จุดเดียว"; ระวังอย่าไปลบรอยโรคเล็กๆ ที่ *เป็น* label ทิ้ง |
| Multi-source | MixUp, CutMix, Mosaic | ใช้เฉพาะ dataset ยากๆ/detection; label จะปนกัน |

**สิ่งที่ห้ามทำ** ในงานการแพทย์ (บอกเหตุผลด้วย):
- **Horizontal flip กับ chest X-ray** หัวใจจะย้ายไปอยู่ด้านขวา ความหมายทาง anatomy/cardiomegaly เลยเปลี่ยน
- **หมุนเยอะ, augmentation แบบ Very Strong** (perspective warp, noise/blur หนักๆ) ทำให้ความหมายของภาพเปลี่ยน และภาพ scan จริงไม่มีทางหน้าตาแบบนั้น
- **Colour jitter ในงานที่สีไม่มีความหมาย** (X-ray grayscale) ไม่มีประโยชน์
- **Augmentation ใดๆ ตอน val/test/inference** ใช้เฉพาะ train random crop ตอน inference = ภาพเดียวกันได้คำตอบไม่เหมือนกัน
- กฎทอง (อาจารย์ ยกตัวอย่างตัวอักษรไทย): augment เฉพาะความหลากหลายที่เกิดขึ้นได้จริงในสถานการณ์จริง
- Dynamic augmentation: loss สูง → **weak**, loss ต่ำ → strong ราย class: class ที่ accuracy ต่ำได้แบบ **strong**

### คำตอบตัวอย่าง 1: คัดแยกผู้ป่วย chest X-ray ในห้องฉุกเฉิน (multilabel + imbalance)
*โจทย์:* chest X-ray grayscale 12,000 ภาพ (≈2000×2500 px) จากคนไข้ 4,000 คน (หลายคนมีภาพ follow-up) Label: nodule, pneumothorax, effusion, cardiomegaly เจอพร้อมกันกี่อันก็ได้ หรือไม่เจอเลยก็ได้ Pneumothorax มีแค่ 3 % ระบบใช้ flag เคสด่วนในห้อง ER

1. **Task:** **multilabel** หนึ่งภาพ *อาจ* มีมากกว่าหนึ่ง finding หรือไม่มีเลย และทุก label ตอบคำถามเดียวกัน ("เจอ finding อะไรบ้าง?") จึงไม่ใช่ multi-task และไม่ใช่ multiclass เพราะ finding เกิดร่วมกันได้
2. **Output layer:** **4 nodes**, **sigmoid แยกแต่ละ node** (อิสระต่อกัน ค่า 0–1; ถ้าใช้ softmax จะบังคับให้แย่งกันและรวมกันเป็น 1) Loss **BCE** (`BCEWithLogitsLoss`) คำทำนาย: ทุก node ที่ ≥ threshold คือเจอ finding นั้น ถ้าต่ำกว่าทั้งหมด = ปกติ ปรับ threshold ของแต่ละ label บน validation โดย pneumothorax ใช้ threshold **ต่ำ** เพื่อให้ได้ recall
3. **Model:**
   - DenseNet-121 หรือ ResNet-50 ที่ pretrained บน ImageNet เปลี่ยน `fc` เป็น `Linear(in, 4)`
   - **Fine-tune แบบ type 3**: freeze block ต้นๆ (ขอบทั่วไป) แล้ว train block หลังๆ + head เพราะ X-ray ไม่เหมือนภาพถ่ายใน ImageNet
   - Copy 1 → 3 channels, normalise แบบ ImageNet
   - **Pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize** เป็น 512 (หรืออย่างน้อย 224) เพื่อไม่ให้สัดส่วนหัวใจ/ปอด (ที่ต้องใช้ดู cardiomegaly) โดนยืด 512 เป็นตัวเลขที่ผมเลือกเอง *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์: finding เล็กๆ อย่างเส้น pneumothorax บางๆ หรือ nodule เล็กๆ จะหายไปที่ resolution ต่ำ)*
4. **Augmentation (เฉพาะ train):** หมุน ±5–10°, shift/scale ±5 %, brightness/contrast เล็กน้อย (เครื่องและการ expose ต่างกัน) **ห้าม horizontal flip** (หัวใจจะย้ายไปด้านขวา) ห้ามหมุนเยอะ ห้าม augmentation แบบ Very Strong เพิ่มตัวอย่างที่ augment / oversampling ให้ pneumothorax
5. **Split:** **แบ่งตาม patient ID** 70/15/15 แบบ stratified ให้ทุก split มีเคส pneumothorax save เป็นไฟล์ครั้งเดียวแล้วใช้กับทุก model ถ้าแบ่งสุ่มรายภาพ ภาพ follow-up ของคนไข้คนเดียวจะไปอยู่ทั้งใน train และ test ทำให้ model "จำคนไข้ได้" (data leakage)
6. **Metric:** ราย label ใช้ **recall (sensitivity)**, precision, F1 และ AUC **ตัวหลักคือ recall ของ pneumothorax** เพราะงานนี้คือ triage: พลาด (FN) อันตราย แต่เตือนผิดแค่เสียเวลาหมอดูเพิ่ม ตัวอย่างบนภาพ test 1,200 ภาพ: TP 27, FN 9, FP 60, TN 1104
   - Accuracy = 1131/1200 = 0.9425 ดูดีมาก แต่ตอบ "ปกติหมด" ได้ 1164/1200 = 0.970
   - Recall = 27/36 = **0.75**, Precision = 27/87 = **0.31**, Specificity = **0.948**, F1 = 54/123 = **0.44**
   - สรุป: ลด threshold ลงอีกเพื่อดัน recall ให้สูงขึ้น และรายงาน recall + F1/AUC ไม่ใช่ accuracy
7. **ความเสี่ยง:**
   - Imbalance (3 %)
   - ต่างโรงพยาบาล/ต่างเครื่อง → ต้อง validate กับข้อมูลภายนอก
   - Leakage จากภาพ follow-up
   - Label ที่ดึงมาจาก report อาจมี noise
   - ใช้เป็นผู้อ่านคนที่สอง ให้รังสีแพทย์เป็นคนยืนยัน
   - *โจทย์พลิก:* "วัดพื้นที่ effusion ด้วย" → **segmentation** (1×1 conv, 1 channel + sigmoid, Dice loss, metric Dice/IoU, พื้นที่ = จำนวน pixel × ขนาด pixel) "ประมาณค่า cardiothoracic ratio" → **regression** (1 node, linear, L1/L2, MAE/RMSE + Pearson) ถ้าต้องทำทั้งสองอย่างใน network เดียว → ใช้ head แบบ multi-task

### คำตอบตัวอย่าง 2: Ultrasound lymphedema (multi-task, เคสในคาบ Ch 9)
*โจทย์:* ภาพ ultrasound ของแขนขา แต่ละภาพให้ทำนาย (i) lymphedema ใช่/ไม่ใช่, (ii) เกรด SEG 0/1/2, (iii) เกรด SEFS 0/1/2 มีประมาณ 1,500 ภาพ จากคนไข้ 300 คน

1. **Task:** **multi-task** เป็น 3 คำถาม *คนละเรื่อง* ความหมายต่างกัน จึงไม่ใช่ multilabel (multilabel ทุกคำตอบมีความหมายแบบเดียวกัน) แต่ละ task เองเป็น **multiclass** เกรดเป็นระดับที่แยกขั้นกัน ดังนั้น "เป็นตัวเลข" ไม่ได้แปลว่าเป็น regression เสมอไป
2. **Output layer:** ทางเลือก **(b)** ใช้ feature extractor ร่วมกัน + head แยกตาม task 3 อัน:
   - head โรค **2 nodes**, softmax, CE
   - head SEG **3 nodes**, softmax, CE
   - head SEFS **3 nodes**, softmax, CE
   - Loss รวม = L_disease + L_SEG + L_SEFS แต่ละ head อ่านผลด้วย argmax ของตัวเอง
   - ทางเลือกอื่น: (a) softmax เดียวกับ class ที่คูณกัน 2 × 3 × 3 = 18 classes (combination เยอะ แต่ละอันมีภาพน้อย) (c) แยก 3 model: ใช้ทรัพยากร 3 เท่า ไม่ได้แชร์อะไรกันเลย
   - (b) เหมาะเพราะ output เกี่ยวข้องกันทางคลินิก (หมอดู lymphedema จากเกรด SEG/SEFS) และข้อมูลมีจำกัด ถ้า task ไม่เกี่ยวกัน → negative transfer
3. **Model:** backbone ที่ pretrained บน ImageNet (เช่น ResNet-50) แบบ **type 3** (freeze ต้นๆ, train หลังๆ + heads) เพราะ ultrasound คือตัวอย่างที่อาจารย์ยกเองว่าเป็นภาพที่ไม่เหมือน ImageNet copy 1 → 3 channels, pad ก่อนแล้ว resize เป็น 224
4. **Augmentation (เฉพาะ train):** rotate/shift/scale เล็กน้อย (มุมและตำแหน่ง probe ต่างกัน), brightness/contrast เล็กน้อย (ค่า gain ต่างกัน) เลี่ยงการหมุนเยอะและ augmentation แบบ Very Strong ถ้า flip บน-ล่าง เนื้อเยื่อชั้นลึกจะขึ้นไปอยู่ข้างบน ซึ่งภาพ scan จริงไม่มีทางเป็นแบบนั้น *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์: probe อยู่ด้านบนของภาพ ultrasound เสมอ)*
5. **Split:** แบ่งตามคนไข้ (คนไข้หนึ่งคนมีหลายภาพ) ใช้ไฟล์ตายตัว เหมือนกันทุก model ที่เอามาเทียบ
6. **Metric:** head โรค → **recall** + F1 (คัดกรอง ห้ามพลาด lymphedema) head เกรด → accuracy ถ้า class สมดุล ไม่งั้นใช้ F1 ราย class และ confusion matrix (เกรด 1/2 ที่ติดกันสับสนกันไหม?)
7. **ความเสี่ยง:** ข้อมูลน้อย → overfitting (ใช้ augmentation, freeze) เกรดขึ้นกับความเห็นคน (label noise) ผลขึ้นกับผู้ใช้เครื่อง/ตัวเครื่อง negative transfer ถ้า head หนึ่งไปทำให้ head อื่นแย่ลง (ลองเทียบกับทางเลือก (c))

### กับดัก
- **ใช้ softmax กับ multilabel** output รวมกันเป็น 1 finding เลยแย่งกัน ให้ใช้ **sigmoid + threshold** และจำไว้ว่าคำตอบอาจเป็น **ไม่มีเลย** ก็ได้
- **ใช้ argmax กับ multilabel** ทุก node ที่ ≥ threshold นับหมด
- **Multilabel vs multi-task** คำถามเดียวหลายคำตอบ (multilabel) vs คนละคำถาม (multi-task)
- **"เป็นตัวเลข → regression"** เกรด/อายุที่เป็นจำนวนเต็มอาจเป็น multiclass ได้ Regression = ค่าต่อเนื่อง
- **Regression ใช้ sigmoid/softmax หรือ CE loss** ต้องใช้ linear + L1/L2 1 node ต่อ 1 ค่า (แนะนำ 1 model ต่อ 1 ค่า: regression ต้องใช้ข้อมูลมากกว่า ~10–20 เท่า)
- **ใช้ accuracy กับข้อมูลไม่สมดุล** model ที่ตอบ negative หมดชนะเรื่อง accuracy ให้ตอบ recall/F1/AUC
- **สลับ precision กับ recall** ตัวส่วนของ recall = positive จริง (มี FN) ของ precision = ที่ทายว่า positive (มี FP) คัดกรอง → recall
- **F1 คือ harmonic mean** P = 1.0, R = 0.1 → F1 = 0.18 ไม่ใช่ 0.55
- **Softmax ในสไลด์ผิด** score [−3.44, 1.66, 3.91] ได้ softmax ≈ **[0.001, 0.095, 0.904]** ไม่ใช่ [0.010, 0.059, 0.931] ตามสไลด์ ค่า sigmoid [0.03, 0.84, 0.98] ถูกแล้ว คิดจากสูตรเอง อย่าลอกสไลด์
- **Augment val/test** หรือ random crop ตอน inference
- **Resize ก่อน pad** หรือยืดภาพตรงๆ
- **แก้ conv แรกเป็น 1 channel** pretrained weights จะหาย ให้ copy channel แทน
- **Fixed feature extraction / "freeze ทั้งหมด" กับ X-ray** ค่า default ของงานการแพทย์คือ type 3
- **แบ่ง `train_test_split` สุ่มรายภาพ** ทั้งที่มีภาพ follow-up = leakage การตั้ง seed ไม่ได้แก้ปัญหาเทียบไม่แฟร์ ต้อง save split ไว้
- **ใช้ pixel accuracy กับรอยโรคเล็ก** หรือเอา Dice จาก paper หนึ่งไปเทียบกับ IoU จากอีก paper (Dice ≥ IoU)
- **รายงาน MAE โดยไม่บอกหน่วยและช่วงค่า** หรือรายงาน MAE อย่างเดียวไม่มี Pearson r / กราฟ

---
<a id="q3"></a>

## Q3 — Video (Chapter 13 + Ch 14 §4 metrics) — 14 คะแนน

### 1. ข้อสอบจะถามอะไร
Mock Q3 มี 4 ข้อย่อย (3 + 4 + 4 + 3) คาดว่าข้อจริงจะคล้ายกัน:
- **Concept / จัดประเภท (≈3)**: spatial vs temporal; นิยาม spatio-temporal; บอกว่า CNN→LSTM / ConvLSTM / 3D CNN เป็น **S→T หรือ ST** (อันนี้คือ quiz ในห้อง)
- **โจทย์ทางการแพทย์ / เลือก architecture (≈4)**: dataset วิดีโอทางคลินิกขนาดเล็ก คลิปยาวไม่เท่ากัน → เลือกโมเดลแล้วให้เหตุผล (ขนาด dataset, pretrained, resource); บอกชนิดของ video task ในแต่ละแบบ (ต่อ frame / start–end / เป็นตัวเลข)
- **LSTM (≈4)**: หน้าที่ของ cell state, hidden state, 3 gate, sigmoid vs tanh; **คำนวณ Cₜ และ hₜ** ด้วยมือ แล้วแปลความหมาย
- **Shape + 3D conv ด้วยมือ (≈3)**: output shape ของ Conv3d/MaxPool3d, จำนวน parameter, ลำดับแกน (permute จาก layout ของ ConvLSTM), ทำ 3D convolution เล็กๆ บนกระดาษ
- อาจมีเพิ่ม: ข้อจำกัดของ NN/CNN → ทำไมต้อง RNN; ข้อดี/ข้อเสียของ RNN; Two-Stream / I3D / 1×1×1 conv; video metric (Edit, F1@k, tIoU)
- กฎ: "อธิบาย" = ต้องบอก **ว่าทำไม** ไม่ใช่แค่บอกชื่อ "คำนวณ" = **ต้องแสดงวิธีทำ**


### 2. ต้องรู้

**Spatial vs temporal**
| | Spatial (เชิงพื้นที่) | Temporal (เชิงเวลา) |
|---|---|---|
| คืออะไร | ข้อมูล **ภายในเฟรมเดียว** ("internal"): edge, สี, รูปร่าง, texture, ตำแหน่ง | การเปลี่ยนแปลง **ระหว่างเฟรม** ("external"): การเคลื่อนไหว, ลำดับ, ทิศทาง, ความเร็ว, จังหวะเวลา |
| ใช้อะไรสกัด | 2D convolution (CNN) | RNN / LSTM |
- **Spatio-temporal (ST)** = สกัดทั้งสองอย่าง **ในเวลาเดียวกัน** คีย์เวิร์ด: **"พร้อมกัน"** (simultaneous) ทำได้ด้วย **ConvLSTM** หรือ **3D conv**
- สกัด spatial ก่อน แล้วค่อย temporal = **S→T** (ทำตามลำดับ) *ไม่ใช่* ST ตัวอย่าง: CNN ทีละเฟรม → LSTM
- ทำไม temporal สำคัญ: ถ้ามีแค่เฟรม 1 กับเฟรม 5 แต่ไม่รู้ลำดับ จะ **แยกไม่ออกว่ากำลังลุกขึ้นหรือนั่งลง** ใช้ spatial อย่างเดียว "ไม่เพียงพอ"
- Video tensor: 1 เฟรม = H×W×C (rank 3); เอาหลายเฟรมมาซ้อนกัน = **rank-4**; ใส่ batch ด้วย = 5-D
- สรุปกลยุทธ์: (a) S กับ T แบบ **ขนาน** แล้วค่อย fuse ทีหลัง (Two-Stream), (b) แบบ **ตามลำดับ** S→T, (c) แบบ **พร้อมกัน** ST

**Image task → video task**
| Image task | Video task | Ground truth | Input | Metric (Ch 14) |
|---|---|---|---|---|
| Classification | **Video classification** | 1 label ต่อคลิป | คลิปที่ตัดมาแล้ว (trimmed) | Acc, P, R, F1 |
| Segmentation (label ทุก pixel) | **Temporal action segmentation** | 1 label **ต่อเฟรม** (100 เฟรม → 100 label) | วิดีโอยาว ไม่ได้ตัด (untrimmed) | frame-wise acc, **Edit**, **Segmental F1@k** |
| Object detection (class + box) | **Temporal action localization** | (class, **start, end**) เฉพาะ action ที่สนใจ | untrimmed | **tIoU**, AP, mAP@tIoU |
| Regression | **Video regression** | เป็นตัวเลข (เช่น heart rate จากวิดีโอผิวหนัง) | คลิป | (regression metric) |
- "Segmentation ตามเวลา ≠ pixel segmentation ในวิดีโอ" (footer สไลด์ 10)

**ทำไมต้อง RNN: ข้อจำกัดของ NN/CNN** NN/CNN ต้องการ input ที่ (1) มา **ทีเดียวทั้งหมด**, (2) **ไม่มีความสัมพันธ์เชิงลำดับ**, (3) มี **ขนาดคงที่และเท่ากัน** แต่วิดีโอมีลำดับและยาวไม่เท่ากัน RNN: รับ input ทีละส่วน, สกัดข้อมูลเชิงลำดับได้, รับความยาวไม่คงที่ได้
- มี RNN cell แค่ตัวเดียว แล้ว **ใช้ซ้ำ** ทุก step (unfold), weight แชร์กัน H₀ = 0 ให้ output ได้ทุก step (เช่น การแปลภาษา, label ต่อเฟรม) หรือ **เฉพาะ step สุดท้าย** (video classification)
- ข้อดีของ RNN: รับความยาวไม่คงที่ได้, ขนาดโมเดลไม่โตตาม input, ใช้ข้อมูลในอดีต, weight แชร์กัน ข้อเสีย: **ช้า**, ข้อมูลช่วงต้นจางหายไป (long sequence problem), มองไม่เห็นอนาคต

**RNN vs LSTM vs GRU**
| | RNN | LSTM (Hochreiter 1997) | GRU *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* |
|---|---|---|---|
| State | hidden Hₜ | **cell Cₜ (ระยะยาว) + hidden hₜ (ระยะสั้น)** | มีแค่ hidden |
| Gate | ไม่มี (มี tanh ตัวเดียว) | forget, input, output + tanh input modulate | update, reset |
| ความจำระยะยาว | แย่ | ดี (มี additive cell path) | ดี |
| ต้นทุน | ต่ำสุด | สูงสุด | อยู่ตรงกลาง |

**LSTM = 2 state + 3 gate + 2 modulate**
| ส่วน | Act. | หน้าที่ (คำของอาจารย์) |
|---|---|---|
| Cell state Cₜ | — | **ความจำระยะยาว** "รวบรวม" ข้อมูลจากทุก step; ทำได้แค่ × f กับ + ของใหม่; **ไม่มี tanh บนเส้นทางนี้** → เก็บข้อมูลไว้ได้ ("เหมือนเก็บเงิน") |
| Hidden state hₜ | — | **ความจำระยะสั้น** "**keep for predict**"; ส่วนใหญ่เป็นข้อมูลจาก step ล่าสุด; **ไม่ใช่** ตัว prediction เอง |
| Forget gate fₜ | σ | **Delete**: เก็บ Cₜ₋₁ ไว้เท่าไร (1 = เก็บหมด, 0 = ลบทิ้ง) |
| Input gate iₜ | σ | **Write**: ให้ candidate ใหม่เข้า C เท่าไร |
| Output gate oₜ | σ | **Read**: ให้ C ออกไปเป็น hₜ เท่าไร (ไม่ได้เปลี่ยน C) |
| Input modulate zₜ (candidate g) | tanh | เตรียมข้อมูลใหม่จาก xₜ, hₜ₋₁ |
| "Output modulate" tanh(Cₜ) | tanh | **ชื่อที่อาจารย์ตั้งเอง** — ไม่ใช่ชื่อทางการ |
- **Sigmoid (0…1) = weight/gate**; **tanh (−1…1) = normalise ข้อมูล** (อยู่รอบศูนย์ → เรียนรู้เร็วขึ้น) ห้ามสลับกัน
- ค่าของ gate **ปรับตัวได้ (adaptive)**: คำนวณจาก xₜ และ hₜ₋₁ เลยเปลี่ยนไปตาม input ทุกครั้ง แม้ตอน inference (ต่างจาก filter ของ CNN ที่คงที่) แต่ละ gate มี **W, b ของตัวเอง** ที่เรียนรู้ได้
- Cell state เก็บข้อมูลได้ **มากกว่า** แต่ไม่ได้ **ครบทั้งหมด** (เสียข้อมูลไปที่ forget gate กับ input gate)
- **ทำไม LSTM แก้ vanishing gradient ได้** *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์; ในโน้ตเขียนว่า "additive cell-state path")*: Cₜ = fₜ∘Cₜ₋₁ + iₜ∘zₜ ดังนั้น ∂Cₜ/∂Cₜ₋₁ = fₜ (element-wise) บนเส้นทางนี้ไม่มี tanh หรือ weight matrix มาบีบค่า; ถ้า fₜ ≈ 1 gradient จะไหลย้อนกลับไปได้หลาย step แทบไม่เปลี่ยน ส่วน RNN ธรรมดา ทุก step ต้องคูณด้วย W และ tanh′ (≤ 1) → ผลคูณเล็กลงเรื่อยๆ → step แรกๆ ได้ gradient ~0

**Pipeline (4 แบบของอาจารย์)**
1. **CNN → flatten เป็น feature vector ต่อเฟรม → LSTM → prediction** — S→T ใช้ image CNN ที่ pretrained มาแล้วได้
2. **Hand-crafted feature** (edge filter, MFCC/spectrogram สำหรับสัญญาณ) → LSTM (หรือ CNN ถ้าหน้าตาเหมือนภาพ) — S→T
3. **ConvLSTM** → NN (มีหรือไม่มีก็ได้) → prediction — **ST**
4. **3D CNN** (+ 3D max pool) → NN (มีหรือไม่มีก็ได้) → prediction — **ST**

**ConvLSTM (Shi et al. 2015)** — พื้นฐานคือ LSTM ("นี่คือจุดเน้น") สิ่งที่เปลี่ยนไป:
- **matrix multiplication** W·xₜ และ W·hₜ₋₁ → เปลี่ยนเป็น **convolution** Conv(xₜ; w), Conv(hₜ₋₁; w);
- state กลายเป็น **2D feature map** ไม่ใช่ vector 1D;
- f∘C, i∘g, o∘tanh(C) ยังคงเป็น **element-wise** เหมือนเดิม
- Code: concat [xₜ, hₜ₋₁] ตาม channel → ผ่าน `Conv2d(in=input_dim+hidden_dim, out=4·hidden_dim, padding=k//2)` ตัวเดียว → แบ่งเป็น i, f, o, g (hidden_dim 3 → 12 filter แบ่งเป็น 4 กลุ่ม กลุ่มละ 3)
- **ทำไมเป็น ST**: filter 2D ให้ spatial feature และ channel ที่มันเอามารวมกันมาจาก **คนละเวลา** (xₜ และ hₜ₋₁) → ได้ temporal ใน operation เดียวกัน

**3D CNN**
- 2D conv: filter เลื่อนแค่แกน **x, y**; ความลึกของ filter = ทุก channel ของ input ส่วน 3D conv: เลื่อนทั้ง **x, y และ z (เวลา)**; ความลึกของ filter **เล็กกว่า** input ได้ → แต่ละ output ครอบคลุมหลายเฟรม → ST
- ใช้คู่กับ **3D max pooling** (2×2×2, stride 2 → ทุกมิติลดลงครึ่งหนึ่ง)
- ข้อดี: แม่นกว่า 2D บนข้อมูล 3 มิติ; ได้ S+T พร้อมกัน ข้อเสีย: **parameter เยอะ → overfit**, **มี pretrained model น้อย**, ใช้ memory/เวลามากกว่า

**ปัญหาของ action recognition → Two-Stream / I3D** UCF-101 / HMDB-51 มีขนาด **เล็ก** → โมเดลซับซ้อนจะ overfit; 3D ConvNet มี parameter เยอะ เทรนยาก
- **Two-Stream**: stream ของเฟรม RGB + stream ของ **optical flow** (การเคลื่อนไหว) ส่วนใหญ่ใช้ **2D ConvNet** แล้วทำ **late fusion** ตอนท้าย อาจารย์อธิบายแบบเข้าใจง่ายว่า flow ≈ "เฟรมลบเฟรม" (ถ้าพูดให้เป๊ะคือการประมาณการเคลื่อนที่ของแต่ละ pixel)
- **I3D (Inflated 3D)**: ขยาย (inflate) filter/pool ขนาด N×N ของ 2D **Inception-v1** ให้เป็น **T×N×N**; จุดสำคัญคือเลือก temporal stride/pooling (temporal receptive field) มีสอง stream (RGB + flow) → 3D ConvNet → AvgPool3D → 1×1 conv → score (1×nc) → **เฉลี่ย** score ทั้ง 2 → softmax เพราะ pretrained บนวิดีโอมาแล้ว → ใช้เป็น feature extractor + LSTM/RNN/GRU/linear ได้
- **1×1×1 conv**: ไม่เห็นทั้ง space และ time; เป็นแค่ linear combination ของ channel → ใช้ลด channel, ผสม channel, เพิ่ม non-linearity (ใช้กับ ReLU+BN), ช่วยให้ branch ใน Inception ยืดหยุ่น
- โมเดลใหม่ๆ (รู้แค่ชื่อพอ): SlowFast 2019, X3D 2020, MViTv2 / Video Swin 2021, VideoMAE 2022, UniFormer 2023, TimeFormer 2025 ส่วนใหญ่ข้างในใช้ ConvLSTM หรือ 3D conv ไม่งั้นก็เป็น S→T

**ตารางเปรียบเทียบรวม**
| Model | Input | จัดการเรื่องเวลายังไง | S→T / ST | ข้อดี | ข้อเสีย | เลือกเมื่อ |
|---|---|---|---|---|---|---|
| CNN→LSTM | (B,T,C,H,W), CNN ทีละเฟรม | LSTM รันบน vector ของแต่ละเฟรม | S→T | ใช้ ImageNet CNN ที่ pretrained ได้, เบา, รับความยาวไม่คงที่ได้ | รายละเอียด spatial หายไปตอน flatten ก่อนจะได้เห็นเวลา | dataset เล็ก, GPU จำกัด, คลิปยาว |
| ConvLSTM | (B,T,C,H,W) | LSTM recurrence ที่ gate เป็น conv, state เป็น map | ST | เก็บ spatial layout ไว้ได้ตลอดเวลา | ต้องเทรนเองตั้งแต่ศูนย์ (PyTorch ไม่มีให้ในตัว), กิน memory เยอะ | สำคัญว่าเคลื่อนไหว "ตรงไหนและเมื่อไร", มีข้อมูลพอ |
| 3D CNN | (B,C,T,H,W) | 3D kernel เลื่อนไปตามแกนเวลา | ST | แม่น, ได้ S+T ใน op เดียว | parameter เยอะ, overfit, pretrained น้อย, กิน memory | dataset ใหญ่ + มี GPU, คลิปสั้นและยาวเท่ากัน |
| Two-Stream | RGB + optical flow | stream ของ flow เป็นตัวพาข้อมูลการเคลื่อนไหว | ขนาน, late fusion | มีข้อมูลการเคลื่อนไหวชัดเจน, ใช้ 2D pretrained net ได้ | ต้องคำนวณ flow, ต้องมี 2 network | การเคลื่อนไหวคือ cue หลัก |
| I3D | RGB + flow, (B,C,T,H,W) | inflated 3D kernel + flow | ST, two-stream | **pretrained บนวิดีโอ**, feature ดี | หนัก, ต้องใช้ flow | dataset เล็ก → ใช้เป็น pretrained extractor |
- **"It depends"**: ไม่มีโมเดลไหนดีที่สุดเสมอ อันไหนดีที่สุดขึ้นกับ **task, dataset และสิ่งที่จะ predict**; เฟรมใหญ่และวิดีโอยาวจะมีปัญหาเรื่อง resource

**Tensor shape / ลำดับแกน**
| ที่ไหน | Shape |
|---|---|
| ข้อความในสไลด์ 59 | (Batch, Frame, Width, Height, Channel) |
| ConvLSTM `batch_first=True` | **(B, T, C, H, W)** เช่น (4, 30, 3, 112, 112) |
| `nn.Conv3d`, I3D | **(N, C, T, H, W)** = (N, C, D, H, W) |
- ConvLSTM → Conv3d: `x.permute(0, 2, 1, 3, 4)`
- วิดีโอเดียวตอน inference: stack → (30, 3, 112, 112) → `unsqueeze(0)` → (1, 30, 3, 112, 112)
- ConvLSTM classifier ในสไลด์: hidden สุดท้าย (4, 64, 112, 112) → `view(4, -1)` = (4, 802,816) → `Linear(112*112*64, 3)` → (4, 3)
- I3D: (25, 3, 16, 224, 224) → 3D ConvNet → (25, 2048, 2, 7, 7): เวลา ÷8 (16→2), space ÷32 (224→7) → (25, 1, 2048)
- เรื่องใช้งานจริง: transform จะทำ **ทีละเฟรม** → random augmentation จะต่างกันในแต่ละเฟรม ("จุดสำคัญมาก") ถ้าความยาวไม่เท่ากัน: ตัด/sample ให้เหลือ `max_frames` คงที่ หรือใช้ RNN

**Video metric (Ch 14 §4)** — ตัวอย่างที่ตรวจแล้ว
- Classification (TP 8, FP 2, FN 1, TN 9): Acc 0.85, P 0.80, R 0.889, F1 0.842
- Frame-wise acc: จำนวนเฟรมที่ถูก / ทั้งหมด (17/20 = 0.85)
- **Edit = 1 − D(P,G)/max(N_pred, N_gt)**, D = Levenshtein บน segment ที่ยุบรวมแล้ว; ลงโทษ **ลำดับผิดและการแบ่ง segment มากเกินไป (over-segmentation)** GT A-B-C-D, pred A-B-C-B-C-D → 1 − 2/6 = 0.667
- **Segmental F1@k**: segment นับเป็น TP ถ้า class ถูกและ IoU ≥ k; GT แต่ละอัน match ได้ครั้งเดียว, ชิ้นที่เกินมา = FP ตัวอย่าง: F1@10 = F1@25 = 0.800, F1@50 = 0.600
- **tIoU** = |pred ∩ GT| / |pred ∪ GT| ในแกนเวลา: GT 10–20, pred 12–22 → 8/12 = 0.667; เป็น TP ถ้า class ถูกและ tIoU ≥ 0.3/0.5/0.7


### 3. สูตร + ตัวอย่างที่ทำให้ดู (ตรวจกับ PyTorch แล้วทั้งหมด)

**RNN**: Hₜ = tanh(XₜWₓ + Hₜ₋₁Wₕ + B); Yₜ = softmax(HₜWᵧ + Bᵧ) (ส่วน "forward" มีหรือไม่มีก็ได้); H₀ = 0

**LSTM**
```
f_t = σ(W_xf x_t + W_hf h_{t-1} + b_f)        # forget / Delete, 0..1
i_t = σ(W_xi x_t + W_hi h_{t-1} + b_i)        # input  / Write,  0..1
o_t = σ(W_xo x_t + W_ho h_{t-1} + b_o)        # output / Read,   0..1
z_t = tanh(W_xz x_t + W_hz h_{t-1} + b_z)     # input modulate (candidate), -1..1
C_t = f_t ∘ C_{t-1} + i_t ∘ z_t               # keep old part + add new part
h_t = o_t ∘ tanh(C_t)
```
ConvLSTM: เหมือนกันเลย แค่เปลี่ยนทุก `W·x` / `W·h` เป็น `Conv(x; w)` / `Conv(h; w)`; ∘ ยังเป็น element-wise

**ตัวอย่างคำนวณ LSTM 1 step (mock 3.3)**: f = [1, 0.5], Cₜ₋₁ = [2, −1], i = [0, 0.5], z = [0.4, 1], o = [0.5, 1]
- Cₜ = [1·2 + 0·0.4, 0.5·(−1) + 0.5·1] = **[2, 0]**
- hₜ = [0.5·tanh(2), 1·tanh(0)] = [0.5·0.964, 0] = **[0.482, 0]**
- แปลความหมาย: มิติที่ 1 เก็บความจำเก่าไว้ทั้งหมด (f = 1), ไม่เขียนอะไรใหม่เลย (i = 0), อ่านออกไปครึ่งเดียว (o = 0.5) มิติที่ 2 ลบค่าเก่าไปครึ่งหนึ่งและเขียนค่าใหม่เข้าไปครึ่งหนึ่ง; สองอันหักล้างกันเหลือ 0

**Output size (คิดแยกทีละแกน T, H, W)**: **O = ⌊(I + 2P − K) / S⌋ + 1** จำนวน channel ที่ออก = จำนวน filter

**ตัวอย่างคำนวณ shape (mock 3.4a)**: x (2, 3, 16, 112, 112), `Conv3d(3, 16, 3, stride=(1,2,2), padding=1)` แล้วตามด้วย `MaxPool3d(2, 2)`
- T: ⌊(16 + 2 − 3)/1⌋ + 1 = 16; H = W: ⌊(112 + 2 − 3)/2⌋ + 1 = ⌊55.5⌋ + 1 = 56 → **(2, 16, 16, 56, 56)**
- MaxPool3d(2, 2): T, H, W ลดลงครึ่งหนึ่ง → **(2, 16, 8, 28, 28)**
- ตัวอย่างในสไลด์: `Conv3d(16, 33, 3, stride=2)` บน (20, 16, 10, 50, 100) → T ⌊7/2⌋+1 = 4, H ⌊47/2⌋+1 = 24, W ⌊97/2⌋+1 = 49 → **(20, 33, 4, 24, 49)** ถ้าใช้ `kernel (3,5,2), stride (2,1,1), padding (4,2,0)` → **(20, 33, 8, 50, 99)**

**จำนวน parameter ของ Conv3d** = C_out · C_in · k_T · k_H · k_W + C_out (bias)
- `Conv3d(3, 16, 3)`: 16·3·27 + 16 = 1,296 + 16 = **1,312**
- `Conv3d(16, 33, 3)`: 33·16·27 + 33 = 14,256 + 33 = **14,289**
- (เพิ่มเติม, ConvLSTM cell, input 3, hidden 64, k = 3: `Conv2d(67, 256, 3)` = 256·67·9 + 256 = **154,624**)

**วิธีทำ 3D conv ด้วยมือ**: output depth = D_in − k_D + 1 (stride 1, ไม่มี pad) สำหรับแต่ละตำแหน่ง output: วาง F[0], F[1], … ทับบน slice ของข้อมูลที่ติดกัน, คูณแบบ element-wise, รวมผลของแต่ละ slice, แล้ว **เอาผลรวมของทุก slice มาบวกกัน** เลื่อนทั้ง x, y **และ z** ไม่ต้องกลับ filter (เป็น cross-correlation เหมือน PyTorch)

**ตัวอย่างเล็ก (mock 3.4c)**: D ลึก 2, ขนาด 3×3; F 2×2×2; stride 1 → output **1×2×2**
```
D[0] = 1 0 2   D[1] = 2 1 0     F[0] = 1  0   F[1] = 0 1
       3 1 0          0 1 3            0 -1          1 0
       0 2 1          1 0 2
```
| ตำแหน่ง | D[0] patch ∘ F[0] | D[1] patch ∘ F[1] | ผลรวม |
|---|---|---|---|
| (0,0) | 1·1 + 1·(−1) = 0 | 1 + 0 = 1 | **1** |
| (0,1) | 0 − 0 = 0 | 0 + 1 = 1 | **1** |
| (1,0) | 3 − 2 = 1 | 1 + 1 = 2 | **3** |
| (1,1) | 1 − 1 = 0 | 3 + 0 = 3 | **3** |
**Output = [[1, 1], [3, 3]]**

**ตัวอย่างในสไลด์ (สไลด์ 78–83)**: D 4×4×4, F 3×3×3 ทุก slice เป็น [[1,−1,−1],[−1,1,−1],[−1,−1,1]], stride 1 → **2×2×2**
- D[0] = [[3,2,1,3],[0,7,3,3],[3,9,3,4],[2,3,8,3]]; D[1] = [[0,2,1,3],[0,4,3,2],[3,9,3,4],[2,6,8,1]]
- D[2] = [[3,2,1,1],[0,3,3,3],[3,4,3,4],[2,3,6,3]]; D[3] = [[3,2,1,3],[0,3,6,3],[0,9,3,4],[2,3,1,3]]
- O[0][0,0], slice 0: [[3,2,1],[0,7,3],[3,9,3]] → 3 − 2 − 1 − 0 + 7 − 3 − 3 − 9 + 3 = **−5**; slice 1 = −11; slice 2 = −4 → **−20** (สไลด์เขียน −21: เพราะเขียน −2 แทนที่จะเป็น −1·1)
- O[0][0,1]: −17 − 13 − **6** = **−36** (สไลด์ −37: slice ที่สามบวกผิดเป็น −7)
- O[1][1,0]: −4 − 7 − 7 = **−18** (สไลด์ −22)
- **ที่ถูก: O[0] = [[−20, −36], [−15, −55]], O[1] = [[−24, −29], [−18, −55]]**
- 3D max pool 2×2×2, stride 2 บน D ตัวเดียวกัน: **O[0] = [[7, 3], [9, 8]], O[1] = [[3, 6], [9, 6]]** (สไลด์ถูกต้อง)


### 4. คำตอบตัวอย่าง

**Q: Spatial vs temporal; นิยาม spatio-temporal; จัดประเภท CNN→LSTM, ConvLSTM, 3D CNN**
Spatial feature คือข้อมูลภายในเฟรมเดียว (edge, สี, รูปร่าง, texture, ตำแหน่ง) สกัดด้วย 2D convolution ส่วน temporal feature คือการเปลี่ยนแปลงระหว่างเฟรม (การเคลื่อนไหว, ลำดับ, ความเร็ว) สกัดด้วย RNN/LSTM Spatio-temporal คือการสกัดทั้งสองอย่าง **ในเวลาเดียวกัน (พร้อมกัน)** (1) CNN→LSTM เป็น **S→T**: CNN ประมวลผลแต่ละเฟรมแยกกันจนถึง flatten แล้ว LSTM ถึงจะได้เห็นเรื่องเวลา (2) ConvLSTM เป็น **ST**: convolution ของมันทำบน concat ของ xₜ กับ hₜ₋₁ ดังนั้น 2D kernel เห็นโครงสร้าง spatial ในขณะที่ channel มาจากคนละเวลา (3) 3D CNN เป็น **ST**: kernel เลื่อนไปตาม x, y และเวลา ครอบคลุมหลายเฟรมใน operation เดียว

**Q: เลือก architecture สำหรับ dataset วิดีโอทางการแพทย์ขนาดเล็ก (เช่น คลิปการเดิน 300 คลิป, ปกติ/ผิดปกติ, ยาวไม่เท่ากัน)**
ขึ้นกับ dataset และ resource; ในข้อนี้ dataset เล็ก ผมเลือก **pretrained 2D CNN (ImageNet) ทีละเฟรม, ตัดที่ flatten/GAP → LSTM → hidden state สุดท้าย → FC (softmax 2 class หรือ sigmoid 1 ตัว)** เหตุผล: 300 คลิปน้อยเกินไปสำหรับ 3D CNN ซึ่งมี parameter เยอะ, overfit ง่าย และมี pretrained model น้อย ส่วน pretrained 2D CNN ให้ spatial feature ที่ดีผ่าน transfer learning, LSTM เพิ่มข้อมูลการเคลื่อนไหวที่เฟรมเดียวบอกไม่ได้ (นั่งลง vs ลุกขึ้น) และใช้ memory น้อยกว่า อีกทางที่ตอบได้ถ้าให้เหตุผล: ใช้ **I3D ที่ pretrained บนวิดีโอ** เป็น feature extractor + LSTM/linear (ไม่ต้องเทรน 3D ตั้งแต่ศูนย์) เรื่องความยาวไม่เท่ากัน: sample จำนวนเฟรมให้คงที่ตลอดคลิป (`max_frames`) หรือพึ่ง RNN ที่รับ input ความยาวไม่คงที่ได้ (แต่ sequence ยาวมากๆ ข้อมูลช่วงต้นก็ยังหายอยู่ดี) ระวัง: random augmentation ทำทีละเฟรม เลยต่างกันในแต่ละเฟรม

**Q: บอกชนิดของ task (เทียบกับ image task)**
ติด label ของ surgical phase **ทุกเฟรม** → temporal action segmentation (เหมือน image segmentation; มี output ทุก time step; Edit / F1@k) หา **เวลาเริ่ม–จบ** ของการเลือดออก → temporal action localization (เหมือน object detection; class + ช่วงเวลา; tIoU / mAP) **Heart rate** จากวิดีโอผิวหนัง → video regression (output แบบ linear) ปกติ/ผิดปกติต่อคลิป หรือล้ม/ไม่ล้มต่อคลิป → video classification (ใช้ output สุดท้ายของ LSTM) (ถ้าตรวจจับการล้มจาก CCTV ยาวๆ ที่ต้องรู้ว่า *ล้มตอนไหน* → localization)

**Q: ทำไมต้อง RNN? ข้อจำกัดของ NN/CNN**
NN/CNN สมมติว่า input มาทีเดียวทั้งหมด, ไม่มีความสัมพันธ์เชิงลำดับ, และมีขนาดคงที่เท่ากัน วิดีโอขัดทั้งสามข้อ: เฟรมมาตามลำดับ, ลำดับเปลี่ยนความหมาย, และคลิปยาวไม่เท่ากัน RNN รับ input ทีละ step, มี hidden state ที่พาข้อมูลในอดีตไปด้วย (H₀ = 0), ใช้ cell ตัวเดียวซ้ำโดยแชร์ weight เลยรับ sequence ยาวเท่าไรก็ได้ จุดอ่อนคือ sequence ยาว: ข้อมูลช่วงต้นจะจางหายไป ซึ่งเป็นเหตุผลที่มี LSTM

**Q: หน้าที่ของแต่ละส่วนใน LSTM และทำไม gate ใช้ sigmoid ส่วนข้อมูลใช้ tanh**
Cell state = ความจำระยะยาว รวบรวมข้อมูลจากทุก step; ถูกแค่คูณด้วย f แล้วบวกของใหม่เข้าไป ไม่เคยถูก tanh บีบ เลยเก็บข้อมูลช่วงต้นไว้ได้ Hidden state = ความจำระยะสั้น "keep for predict" ส่งต่อไป step ถัดไปและไปที่ output layer Forget gate (Delete) ตัดสินว่าจะเก็บ Cₜ₋₁ ไว้เท่าไร; input gate (Write) ตัดสินว่าจะให้ candidate ใหม่ zₜ เข้าไปเท่าไร; output gate (Read) ตัดสินว่าจะให้ C ออกไปเป็น hₜ เท่าไร Gate ใช้ sigmoid เพราะค่า 0…1 ทำหน้าที่เป็น weight (0 ปิดกั้น, 1 ให้ผ่านหมด); ข้อมูลใช้ tanh เพราะ −1…1 ช่วย normalise ให้อยู่รอบศูนย์ ค่าของ gate คำนวณจาก xₜ และ hₜ₋₁ จึงปรับตามทุก input

**Q: ทำไม LSTM รับมือกับ sequence ยาว / vanishing gradient ได้ดีกว่า RNN?**
ใน RNN hidden state ถูกคำนวณใหม่ผ่าน W และ tanh ทุก step ทำให้ข้อมูลและ gradient เล็กลงทุก step และเฟรมช่วงต้นหายไป LSTM เพิ่ม cell state ที่อัปเดตด้วย Cₜ = fₜ∘Cₜ₋₁ + iₜ∘zₜ: เป็นเส้นทางแบบบวก (additive) ที่ถูกแค่ scale แบบ element-wise ด้วย f ถ้า f ≈ 1 ข้อมูลและ gradient (∂Cₜ/∂Cₜ₋₁ = fₜ) จะผ่านย้อนกลับไปได้แทบไม่เปลี่ยน *(ส่วน gradient: ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*

**Q: ConvLSTM เปลี่ยนอะไรจาก LSTM และทำไม?**
ยังใช้สมการ LSTM เหมือนเดิม (3 gate, cell และ hidden state) แต่เปลี่ยน matrix multiplication บน xₜ และ hₜ₋₁ เป็น convolution ทำให้ input และ state เป็น 2D feature map แทนที่จะเป็น vector ที่ flatten แล้ว ส่วนการคูณของ gate ยังเป็น element-wise ทำไม: การ flatten เฟรมทำให้ spatial layout หายไป; พอ gate เป็น conv ความจำจะเก็บได้ว่าของอยู่ *ตรงไหน* ไปพร้อมกับติดตามว่ามันเปลี่ยน *ยังไง* เลยได้ spatio-temporal feature

**Q: 2D vs 3D conv; ข้อดี/ข้อเสียของ 3D CNN; ทำไมต้อง Two-Stream/I3D**
Filter 2D เลื่อนบน x, y และครอบคลุมทุก channel ในครั้งเดียว; filter 3D เลื่อนไปตามแกนเวลาด้วยและครอบคลุมหลายเฟรม ทำให้แต่ละ output ผสมข้อมูล spatial และ temporal เข้าด้วยกัน (ST) 3D CNN แม่นกว่าบนข้อมูล 3 มิติ แต่มี parameter เยอะ, overfit กับ dataset เล็ก (UCF-101, HMDB-51), มี pretrained model น้อย และกิน memory/เวลา Two-Stream เลี่ยงการเทรน 3D ที่หนักโดยใช้ 2D ConvNet บน RGB และบน optical flow (การเคลื่อนไหว) แล้ว fuse ตอนท้าย I3D inflate filter ของ Inception-v1 ที่ pretrained แล้วให้เป็น T×N×N, รัน stream RGB + flow แล้วเฉลี่ย score ได้เป็นโมเดล ST ที่ pretrained บนวิดีโอ

**Q: Shape (ConvLSTM loader → Conv3d)**
DataLoader ของ ConvLSTM (`batch_first=True`) ให้ (B, T, C, H, W) เช่น (2, 16, 3, 112, 112) แต่ `nn.Conv3d` ต้องการ (B, C, T, H, W) เลยต้องเรียก `x.permute(0, 2, 1, 3, 4)` ก่อน จากนั้นคำนวณ T, H, W แยกกันทีละแกนด้วย ⌊(I + 2P − K)/S⌋ + 1; channel = จำนวน filter


### 5. กับดัก
- **CNN→LSTM เป็น S→T ไม่ใช่ ST** มีแค่ ConvLSTM กับ 3D CNN ที่เป็น ST เขียนคำว่า "พร้อมกัน" ในนิยามด้วย
- **สไลด์ 3D-conv คำนวณผิด**: −21 → **−20**, −37 → **−36**, −22 → **−18** อย่าลอกตารางจากสไลด์; ให้แสดงวิธีทำ
- **Sigmoid 0…1 (gate), tanh −1…1 (ข้อมูล)** ถ้าสลับกันจะเสียคะแนน; อาจารย์บอกว่า "หาที่จด"
- **Forget gate = 1 คือเก็บไว้ทั้งหมด** 0 คือลบทิ้ง
- **"Output modulate gate" เป็นชื่อที่อาจารย์ตั้งเอง**; ชื่อทางการคือ forget, input, output gate + input modulate (candidate)
- **ข้อความ "Input Gate (Write)" ในสไลด์ 40 ไปซ้ำกับนิยามของ hidden state**; ให้ใช้สไลด์ 38: มันคุมว่าข้อมูลใหม่จะเข้า cell state เท่าไร
- **Hidden state ≠ prediction**; ต้องผ่านส่วน forward (Linear/softmax) ที่มีหรือไม่มีก็ได้ก่อน
- **Cell state เก็บข้อมูลได้ "มากกว่า" ไม่ใช่ "ครบทั้งหมด"** (เสียข้อมูลไปที่ forget และ input gate)
- **ค่าของ gate เป็น adaptive**; แต่ W, b ที่ใช้คำนวณมันคงที่หลังเทรนเสร็จ
- **ConvLSTM**: มีแค่ผลคูณ x·W และ h·W ที่กลายเป็น convolution; ∘ ยังเป็น element-wise
- **ลำดับแกน**: ConvLSTM (B, T, C, H, W) vs Conv3d (B, C, T, H, W); ต้อง permute ลำดับในข้อความสไลด์ 59 (B, Frame, W, H, C) ไม่ใช่ของ PyTorch
- **สูตร output size ใช้กับแกนเวลาด้วย**: 4×4×4 กับ 3×3×3 → 2×2×2 ไม่ใช่ 2×2×4 ใช้ floor (55.5 → 55, +1 = 56)
- **Parameter ของ Conv3d ต้องรวม C_in และ bias**: 16·3·27 + 16 = 1,312 ไม่ใช่ 16·27
- **3D CNN ไม่ได้ดีกว่าเสมอไป**: ข้อมูล (การแพทย์) น้อย → overfit; ให้ตอบว่า "ขึ้นกับ dataset/resource"
- **1×1×1 conv ไม่เห็นทั้ง space และ time** — ทำได้แค่ผสม channel
- **RNN "รับความยาวไม่คงที่" = มัน *ทำได้* ไม่ได้แปลว่าทำได้ดี**; sequence ยาวข้อมูลก็ยังจางหายไป
- **มี RNN cell แค่ตัวเดียว** ใช้ซ้ำ; ภาพที่ unfold ออกมา ≠ มีหลาย cell ที่ต่างกัน
- **Segmentation (ทุกเฟรม) vs localization (เฉพาะช่วงที่สนใจ)** — อย่าสับสน; metric ต่อเฟรม vs tIoU
- **GRU ไม่ได้อยู่ในสไลด์**: พูดถึงเป็นทางเลือกได้ แต่อย่าใช้เป็นแกนของคำตอบ

---

<a id="q4"></a>

## Q4 — Image segmentation

### 1. ข้อสอบจะถามอะไร
- เปรียบเทียบ classification / detection / segmentation (**What / Where**) อธิบาย semantic vs instance vs panoptic พร้อมยกตัวอย่างทางการแพทย์อย่างละหนึ่ง
- U-Net: encoder vs decoder (W×H, channel, What/Where), **ทำไมต้องมี skip connection**, จำนวน channel ของ U-Net (concat) vs LinkNet (add), **ทำไม layer สุดท้ายเป็น 1×1 conv**
- **ออกแบบ output layer** (จำนวน filter 1×1, activation, รูปแบบ label, loss) สำหรับ binary / multiclass / multilabel / regression
- ให้ GT กับ mask ที่ predict มา: นับ TP/FP/FN/TN แล้วคำนวณ **IoU, Dice, pixel accuracy** ทำไม pixel accuracy ทำให้เข้าใจผิด, ทำไมใช้ Dice loss
- อาจมี: วิธี upsampling/unpooling, dilated conv + DeepLab (ASPP), PSPNet/PAN, คุณสมบัติของ AE

### 2. ต้องรู้
| Task | ตอบอะไร | Where = |
|---|---|---|
| Classification | **What** (ทั้งภาพ, *image prediction*) | ไม่มี |
| Detection | What + Where | **bounding box** ที่เล็กที่สุดและขนานกับแกน |
| Segmentation | What + Where | **รูปร่าง** ของวัตถุ **ทีละ pixel** (*pixel prediction*) |

อาจารย์บอกว่า "จำดีๆ": classification predict จากทั้งภาพ, segmentation predict ที่ **ระดับ pixel**

| | Semantic | Instance | Panoptic (= semantic + instance) |
|---|---|---|---|
| Label | ทุก pixel รวม background | **เฉพาะ foreground** | ทุก pixel |
| วัตถุ class เดียวกัน | label เดียวกัน (แยกไม่ได้) | แยก mask ให้แต่ละชิ้น | **things** แยกกัน, **stuff** (ท้องฟ้า, ถนน, เนื้อเยื่อ) ไม่แยก |
| ใช้ทำอะไร | **วัดพื้นที่/ขนาด** (พื้นที่ tumour ใน MRI) | **นับ/ติดตาม** (แต่ละ nucleus/cell) | ทั้งฉาก (เนื้อเยื่อทุกชนิด + แต่ละ gland/cell) |

- **Autoencoder (encoder–decoder)**: input กับ output มี **W×H เท่ากัน** จะมีกี่ channel ก็ได้ ประกอบด้วย:
  - Encoder: บีบอัดข้อมูล ทำ feature extraction
  - **Code/latent**: ตัวแทนข้อมูลที่ถูกบีบอัดแล้ว
  - Decoder: สร้างกลับคืน
- คุณสมบัติของ AE: เฉพาะกับข้อมูลชุดนั้น (data-specific), เสียข้อมูลบางส่วน (lossy), unsupervised (แค่ตอนใช้บีบอัด; **segmentation ต้องมี GT mask**) **Code size** เป็น hyperparameter ที่สำคัญที่สุด: ใหญ่ไป → copy input ไปเลย → overfitting
- **Encoder** (conv + max pool): W×H ↓, channel ↑, เรียนรู้ **WHAT** แต่เสีย **WHERE**
- **Decoder** (transposed conv / interpolation / unpooling): W×H ↑, channel ↓, กู้ **WHERE** คืนมา
- Map ที่อยู่ลึกและเล็ก มีข้อมูลเยอะแต่รายละเอียดเชิงตำแหน่งน้อย ส่วน map ตื้นและใหญ่จะกลับกัน
- **U-Net** (Ronneberger **2015**, สร้างมาเพื่อภาพ **biomedical**):
  - block: conv 3×3 + ReLU, maxpool 2×2, up-conv 2×2 (transposed conv);
  - **skip = copy + concatenate** ที่ scale เดียวกัน;
  - layer สุดท้าย: **1×1 conv** + sigmoid/softmax
- **LinkNet** (2017): คล้าย U-Net แต่ skip เป็น **element-wise addition** (ต้องมี channel และ W×H เท่ากัน) เบากว่าและเร็วกว่า
- **DeepLabV3** (2017):
  - ใช้ **dilated/atrous conv** แทนการ downsample เพิ่ม;
  - receptive field ↑ โดยไม่เพิ่ม parameter: 3×3 ที่ d=2 ครอบคลุม 5×5 ด้วย weight 9 ตัว เทียบกับ 25 ตัวถ้าเป็น 5×5 จริง;
  - **ASPP** รันหลาย dilation rate แบบขนาน;
  - predict ก่อนแล้วค่อย upsample เลยมี **decoder เล็กมาก**
  - **V3+** (2018) เพิ่ม decoder จริงๆ เข้ามา
- การนับ dilation: **d = 1 คือ conv ธรรมดา**, d = 2 เว้นช่อง 1 pixel ขนาดจริงที่ครอบคลุม = k + (k−1)(d−1)
- **PSPNet** (2017): pyramid pooling 1×1 (global), 2×2, 3×3… (local) แล้ว upsample + concat **PAN** (2018): FPA (หลาย scale ด้วยขนาด filter ต่างกัน) + GAU (ให้น้ำหนัก channel; เป็น skip แบบมีน้ำหนัก)
- Segmentation ใช้ **GPU memory มากที่สุด** และช้าที่สุด เรียงเป็น classification < detection < segmentation

**Output layer (1×1 conv; "จำนวน Channel = จำนวนคลาส"):**

| Task | จำนวน filter 1×1 | Activation | Label | Loss |
|---|---|---|---|---|
| Binary (tumour vs bg) | **1** (หรือ 2) | **sigmoid**, thr 0.5 (หรือ softmax ถ้าใช้ 2) | mask 0/1 ขนาดเท่าภาพ | Dice / BCE+Dice / Jaccard |
| Multiclass (1 px → เป็นได้ 1 ใน N พอดี) | **N** | **softmax** (softmax2d) | class-label map (1 ch, 0..N−1) หรือ one-hot | CE / multiclass Dice |
| Multilabel (1 px → เป็นได้หลายอัน) | **N** | **sigmoid แยกแต่ละ channel**, > 0.5 | **ต้องเป็น one-hot** (N ch) | BCE / multilabel Dice |
| Regression (ความลึก, ความหนา) | **จำนวนค่า** (1) | **linear / ไม่มี** | map ค่าต่อเนื่อง | L1 / L2 |

### 3. สูตร + ตัวอย่างที่ทำให้ดู
นับ pixel (หน่วย = **pixel**): TP = วัตถุที่ predict เป็นวัตถุ, FP = background ที่ predict เป็นวัตถุ, FN = วัตถุที่ predict เป็น background, TN = background ที่ predict เป็น background
```
IoU (Jaccard)  = |A∩B| / |A∪B|      = TP / (TP + FP + FN)
Dice (= F1)    = 2|A∩B| / (|A|+|B|) = 2TP / (2TP + FP + FN)
Pixel accuracy = (TP + TN) / all pixels
Precision = TP/(TP+FP)    Recall = TP/(TP+FN)
Dice = 2·IoU / (1 + IoU)   →   Dice ≥ IoU always;   IoU = Dice / (2 − Dice)
Dice loss = 1 − Dice       Jaccard loss = 1 − IoU
mIoU = mean of per-class IoU (one-vs-all per class)
```
(A = foreground ที่ predict, B = foreground ของ GT สไลด์ Ch 12 แค่บอกชื่อ metric พวกนี้; สูตรเป็นสูตรมาตรฐานที่ใช้ในคาบ Ch 14)

**ตัวอย่าง (4×4 = 16 px):**
```
GT          Pred
0 1 1 0     0 0 1 1
0 1 1 0     0 1 1 1
0 1 1 0     0 0 1 0
0 0 0 0     0 0 0 0
```
- TP = 4 (ส่วนที่ทับกัน), FP = 2 (Pred=1, GT=0: (0,3), (1,3)), FN = 2 (GT=1, Pred=0: (0,1), (2,1)), TN = 16 − 8 = 8
- IoU = 4/(4+2+2) = **0.50**
- Dice = 8/(8+2+2) = **0.667** ตรวจ: 2·0.5/1.5 = 0.667 ✓
- Pixel accuracy = (4+8)/16 = **0.75**

**คำตอบ Mock 4.4 (tumour mask 6×6):** TP = 11, FP = 2, FN = 2, TN = 21 IoU = 11/15 = **0.733**, Dice = 22/26 = **0.846**, pixel accuracy = 32/36 = **0.889**

**ตัวอย่างในสไลด์ Ch 14 (4×4):** TP = 5, FN = 2, FP = 1, TN = 8 Acc = 0.8125, Precision = 0.833, Recall = 0.714, Dice = 10/13 = 0.769, IoU = 5/8 = 0.625

**กับดัก accuracy เป็นตัวเลข:** tumour กินพื้นที่ 2% ของภาพ ถ้า predict ว่า "เป็น background ทั้งหมด" จะได้ pixel accuracy **0.98** แต่ IoU = Dice = **0** (TP = 0)

### 4. คำตอบตัวอย่าง
**Classification vs detection vs segmentation:**
> Classification ตอบแค่ **What** ของทั้งภาพ (image prediction: FC layer รวม feature ทั้งหมดแล้วให้ label เดียว) Detection ตอบ **What + Where** โดย Where คือ bounding box ที่เล็กที่สุดที่ครอบวัตถุ Segmentation ตอบ **What + Where ที่ระดับ pixel**: ทุก pixel ได้ label ทำให้ Where ตามรูปร่างจริงของวัตถุ เพราะแบบนี้ segmentation จึงเหมาะกับการวัดพื้นที่หรือขอบเขตของรอยโรค

**Semantic / instance / panoptic + ตัวอย่างทางการแพทย์:**
> - **Semantic**: ทุก pixel รวม background ได้ class แต่วัตถุ class เดียวกันจะถูกรวมเป็นอันเดียว เหมาะกับการวัดพื้นที่/ขนาด เช่น พื้นที่ tumour ทั้งหมดใน MRI slice หนึ่ง
> - **Instance**: เฉพาะวัตถุ foreground แต่ละชิ้นมี mask ของตัวเอง background ไม่มี label เหมาะกับการนับ/ติดตาม เช่น นับ nucleus แต่ละตัวในภาพ histology หรือติดตาม cell ในวิดีโอ
> - **Panoptic = semantic + instance**: ทุก pixel มี label Stuff (ไม่มีขอบเขตชัด เช่น เนื้อเยื่อพื้นหลัง) จัดการแบบ semantic ส่วน things (นับได้ เช่น แต่ละ gland/cell) จัดการแบบ instance

**Encoder / decoder:**
> Encoder (contraction path: conv + max pooling) **ลด W×H และเพิ่ม channel** มันเรียนรู้ว่า *อะไร* อยู่ในภาพ (feature เยอะ) แต่ **เสียข้อมูลเชิงตำแหน่ง (where)** ส่วน decoder (expansion path: transposed conv / interpolation / unpooling) **เพิ่ม W×H และลด channel** จนได้ output ขนาดเท่า input เพื่อกู้ *where* คืนมา output เป็น map ขนาดเท่าภาพ มี 1 channel ต่อ 1 class

**ทำไมต้องมี skip connection; U-Net vs LinkNet (128 ch × 64² + 128 ch × 64²):**
> การ downsample ใน encoder ทำให้รายละเอียดเชิงตำแหน่งหายไป ถ้า decoder ทำงานจาก bottleneck อย่างเดียว ขอบจะเพี้ยนและเบลอ Skip connection ส่ง feature ความละเอียดสูงจาก encoder ไปให้ decoder **ที่ scale เดียวกัน** (64↔64, 128↔128) feature ชั้นตื้นมีข้อมูลตำแหน่งเยอะ เลยหาขอบของวัตถุได้แม่นยำ **U-Net ใช้ concatenate**: 128 + 128 = **256 channel**, 64×64 **LinkNet ใช้บวกแบบ element-wise**: shape ต้องเท่ากัน ได้ **128 channel**, 64×64 การบวกเบากว่าและเร็วกว่า

**ทำไม layer สุดท้ายเป็น 1×1 conv (bonus quiz):**
> Segmentation คือ **pixel prediction**: class ของ pixel หนึ่งตัดสินจาก **feature vector ของ pixel นั้นข้ามทุก channel** 1×1 conv ก็คือ **fully connected layer ที่ใช้กับแต่ละ pixel แยกกัน** โดยแชร์ weight เลยแปลง C feature channel → N class channel ได้โดย **ยังคง W×H ไว้** ถ้าใช้ FC layer หลัง flatten จะผสมทุกตำแหน่งเข้าด้วยกันและได้คำตอบเดียวต่อภาพ จำนวน filter 1×1 = จำนวน class (เหตุผลไม่ใช่เพราะ softmax; activation เป็นอีกเรื่องที่เลือกแยกตามชนิดของ task)

**ออกแบบ output layer (บรรทัดละหนึ่งกรณี, บอกเหตุผลด้วย):**
> - (a) Liver tumour vs background: binary, **1 filter + sigmoid** (threshold 0.5), mask 0/1, ใช้ Dice หรือ BCE+Dice loss เพราะ tumour เล็ก (imbalanced)
> - (b) เนื้อสมอง 4 class แต่ละ pixel เป็นได้แค่ 1 class: multiclass, **4 filter + softmax**, class-label map (0–3) หรือ one-hot, cross-entropy
> - (c) Histology ที่ pixel หนึ่งเป็นได้หลายโครงสร้าง: multilabel, **4 filter + sigmoid แยกแต่ละ channel**, label **ต้องเป็น one-hot** เพราะ pixel หนึ่งเป็น 1 ได้หลาย channel, BCE
> - (d) ความหนาของกระจกตาต่อ pixel: regression, **1 filter + linear/ไม่มี activation** (ถ้าใช้ sigmoid/softmax จะบีบค่าไปด้วย), L1/L2 loss

**ทำไมไม่ใช้ pixel accuracy; ทำไมใช้ Dice loss:**
> ในภาพทางการแพทย์ รอยโรคเล็กมากและ **TN (background) มีเยอะกว่ามาก** ถ้า predict ว่า "เป็น background ทั้งหมด" ก็ได้ pixel accuracy ประมาณ 98% แล้วทั้งที่หา tumour ไม่เจอเลย (IoU = Dice = 0) IoU/Dice วัดแค่ **ส่วนที่ทับกันของ foreground** และไม่สนใจ TN **Dice loss = 1 − Dice** optimise ส่วนที่ทับกันนั้นโดยตรงและรับมือ **class imbalance** ได้ (สไลด์เขียนไว้คู่กับ PSPNet) Jaccard loss = 1 − IoU เป็นอีกทางเลือกใน smp

**Dilated conv / DeepLab (ถ้าถาม):**
> Dilated (atrous) conv เว้นระยะจุดของ kernel ออกจากกัน: d = 1 คือแบบปกติ, d = 2 เว้นช่อง 1 pixel kernel 3×3 ที่ d = 2 ครอบคลุม 5×5 โดยใช้ weight แค่ 9 ตัวแทนที่จะเป็น 25 ตัว ดังนั้น **receptive field โตขึ้นโดยไม่เพิ่ม parameter หรือการคำนวณ** DeepLabV3 หยุด downsample ตั้งแต่เนิ่นๆ แล้วใช้ dilated conv แทน เลยรักษา spatial resolution ไว้ได้ ASPP รันหลาย dilation rate แบบขนานเพื่อให้ได้ context หลาย scale decoder ของมันเป็นแค่ upsample ขั้นเดียว V3+ เลยเพิ่ม decoder ที่ครบเข้ามา

### 5. กับดัก
- **Instance seg ไม่ได้ label background**; semantic label (ตามหลักการ)
- U-Net skip = **concat** (channel บวกกัน), LinkNet = **add** (channel เท่าเดิม) อย่าสลับกัน
- **Channel = จำนวน class** ยกเว้น binary + sigmoid ที่ใช้ **1**
- Multiclass → softmax Multilabel → sigmoid + one-hot code ในสไลด์หน้า 87–90 ใส่ `activation='sigmoid'` ให้ multiclass ซึ่งผิด; อาจารย์ยอมรับแล้ว
- Regression → **linear/ไม่มี activation** ห้ามใช้ sigmoid/softmax
- คำตอบเรื่อง 1×1 conv ต้องพูดถึง **FC ต่อ pixel ข้าม channel** ไม่ใช่ "เพราะ softmax" หรือ "เพราะ output เป็นภาพ" (แบบนั้นได้แค่ "เกือบถูก")
- "AE เป็น unsupervised": จริงเฉพาะตอนใช้บีบอัด segmentation ต้องมี mask
- Encoder/decoder **ไม่จำเป็นต้องสมมาตร** (DeepLabV3)
- **Dilation 1 = conv ธรรมดา** (ไม่ใช่ 0) Dilated conv ≠ morphological dilation
- Dice ≥ IoU เสมอสำหรับ mask คู่เดียวกัน ถ้าได้ Dice < IoU แปลว่าคำนวณผิด
- TN คือ **pixel background ที่ predict เป็น background**; ไม่อยู่ใน IoU/Dice นับ pixel ของ GT และ Pred ก่อน แล้วค่อยหา TN = ทั้งหมด − (TP + FP + FN)
- Max unpooling ใช้ได้เฉพาะเมื่อ **เก็บ index ของ pooling ไว้**

---
<a id="q5"></a>

## Q5 — Transposed convolution คิดด้วยมือ

### 1. ข้อสอบจะถามอะไร
- ให้ X เล็กๆ (2×2 หรือ 3×3) กับ K (2×2 หรือ 3×3) มา แล้วให้หา **output matrix เต็มๆ** ที่ stride s, padding 0 ต้องโชว์ block X[i,j]·K ว่าแต่ละอันไปวางตรงไหน แล้วรวมค่าที่ซ้อนกัน (Mock: 6 จาก 14 คะแนน)
- X, K ชุดเดิม แต่ **padding 1**: ได้ output ขนาดเท่าไหร่ ค่าอะไรบ้าง แล้ว padding ทำอะไรใน transposed conv
- เติม **ตารางขนาด output** (input, k, s, p → output) เลือก k, s, p เพื่อ upsample เช่น 32×32 → 64×64 (U-Net decoder)
- ถูก/ผิด: สไลด์หน้า 41 "ยิ่งค่า padding มากขึ้น ขนาดของภาพผลลัพธ์จะยิ่งขยายใหญ่ขึ้น" ("more padding, bigger output") และ transposed conv เป็น inverse ของ conv ไหม
- อาจจะมี: ทำ nearest-neighbour upsampling ด้วยมือ (2×2 → 5×5) และวิธี upsampling 3 แบบ

### 2. ต้องรู้
- **Transposed convolution = Deconvolution** (ในวิชานี้คือตัวเดียวกัน "ให้รู้ว่ามันคือสิ่งเดียวกัน", "know they are the same thing") เป็นตัว upsampling ใน decoder ที่นิยมที่สุด ใช้ใน AE decoder, U-Net "up-conv 2×2" และ GAN
- **ไม่ flip/rotate kernel** ("ไม่หมุนตัวกรอง") ค่า weight ของ kernel ได้จากการ **learn** ตอน training
- หลักการ: เอาค่า input แต่ละตัวไปคูณกับ **kernel ทั้งอัน** แล้ววาง block นั้นโดยเลื่อนไปทีละ **stride** จากนั้น **บวกค่าที่ซ้อนกัน**
- อะไรทำให้ขนาด output เปลี่ยน:

| เพิ่ม... | Output | ทำไม |
|---|---|---|
| stride s | **ใหญ่ขึ้น** | block วางห่างกันมากขึ้น |
| kernel k | **ใหญ่ขึ้น** | แต่ละ block ใหญ่ขึ้น |
| padding p | **เล็กลง** | โดน **ตัดออก** p แถว/คอลัมน์ ทุกด้าน (−2p) |

- จะซ้อนกันก็ต่อเมื่อ **s < k** ถ้า s = k block จะวางต่อกันพอดีไม่ซ้อน ถ้า s > k จะมีช่องว่างเป็น 0 *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*
- มันย้อนแค่ **shape** ที่ conv เปลี่ยนไป ไม่ได้ย้อน **ค่า** กลับ มัน **ไม่ใช่** inverse ของ convolution
- วิธี upsampling ใน decoder 3 แบบ (network หนึ่งใช้แบบเดียว มากสุดสองแบบ):
  1. **Interpolation** (nearest-neighbour, bilinear): ไม่มี weight ให้ learn
  2. **Transposed conv**: learn ได้
  3. **Max unpooling**: ต้อง **เก็บ index ของ max-pool ไว้ (mask/location)** แล้วเอาค่าแต่ละตัวกลับไปใส่ตำแหน่งเดิม ที่เหลือเติม **0** จากนั้นใช้ conv 3×3/5×5 ทำให้เรียบ

### 3. สูตร + ตัวอย่างทำจริง

**ขนาด Output** (H = ความสูง/กว้างของ input, k = ขนาด kernel, s = stride, p = padding):
```
Full (before crop)  = (H − 1)·s + k
Output              = (H − 1)·s − 2p + k        (+ output_padding, PyTorch)
```
- `output_padding` *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*: เพิ่ม 1 แถว/คอลัมน์ แค่ด้านล่าง/ขวา เพื่อให้ได้ขนาดเลขคู่ ตัวอย่าง: 32, k=3, s=2, p=1 → 63 ถ้าใส่ output_padding=1 → **64**
- เทียบกับ conv ปกติ: `⌊(H + 2p − k)/s⌋ + 1` ตรงนี้ padding ทำให้ output **ใหญ่ขึ้น** ตรงข้ามกับ transposed conv

**ขั้นตอนทำบนกระดาษ (จำให้ได้):**
1. เขียนขนาดก่อน: full = (H−1)·s + k, final = full − 2p แล้ววาด grid full เปล่าๆ
2. สำหรับ input X[i,j] แต่ละตัว เขียน block **X[i,j]·K** (ไม่ flip kernel)
3. วางมุมซ้ายบนของ block ที่ **(i·s, j·s)** ใน grid full
4. **บวก** ทุกค่าที่ตกช่องเดียวกัน
5. ถ้า p > 0 ให้ **ตัดออก p แถว/คอลัมน์ ทุกด้าน** ที่เหลือคือ output
6. เช็กตัวเอง:
   - ขนาดสุดท้ายตรงกับสูตร
   - มุมซ้ายบนของ grid full = X[0,0]·K[0,0]
   - ถ้า p = 0, **ผลรวมค่าทั้งหมดใน output = sum(X) × sum(K)** *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์; ผลคูณทุกคู่โผล่มาครั้งเดียวพอดี)*

**ตัวอย่าง A: s = 1, p = 0.** X = [[1, −1], [2, 3]], K = [[1, 2], [0, 1]]. ขนาด = (2−1)·1 + 2 = **3×3**
```
1·K @(0,0)   -1·K @(0,1)    2·K @(1,0)   3·K @(1,1)
| 1 2 |      | -1 -2 |      | 2 4 |      | 3 6 |
| 0 1 |      |  0 -1 |      | 0 2 |      | 0 3 |
```
- แถว 0: [1, 2 + (−1), −2] = [1, 1, −2]
- แถว 1: [0 + 2, 1 + 0 + 4 + 3, −1 + 6] = [2, 8, 5] ช่องกลาง (1,1) ได้จากทั้ง 4 block
- แถว 2: [0, 2 + 0, 3] = [0, 2, 3]
```
Output = |  1  1 -2 |
         |  2  8  5 |      check: sum = 20 = sum(X)·sum(K) = 5·4 ✓
         |  0  2  3 |
```

**ตัวอย่าง B: s = 2, p = 1.** X = [[2, 1], [1, 3]], K = [[1, 2, 1], [0, 1, 0], [1, 0, 2]].
Full = (2−1)·2 + 3 = 5×5. Final = 5 − 2·1 = **3×3**
```
2·K rows0-2,cols0-2   1·K rows0-2,cols2-4   1·K rows2-4,cols0-2   3·K rows2-4,cols2-4
| 2 4 2 |             | 1 2 1 |             | 1 2 1 |             | 3 6 3 |
| 0 2 0 |             | 0 1 0 |             | 0 1 0 |             | 0 3 0 |
| 2 0 4 |             | 1 0 2 |             | 1 0 2 |             | 3 0 6 |
```
- คอลัมน์ 2 ใช้ร่วมกันระหว่าง block ซ้ายกับขวา แถว 2 ใช้ร่วมกันระหว่าง block บนกับล่าง ช่อง (2,2) ใช้ร่วมกันทั้ง 4 block
- แถว 0: [2, 4, 2+1, 2, 1] = [2, 4, 3, 2, 1]
- แถว 1: [0, 2, 0+0, 1, 0] = [0, 2, 0, 1, 0]
- แถว 2: [2+1, 0+2, 4+1+1+3, 0+6, 2+3] = [3, 2, 9, 6, 5]
- แถว 3: [0, 1, 0+0, 3, 0] = [0, 1, 0, 3, 0]
- แถว 4: [1, 0, 2+3, 0, 6] = [1, 0, 5, 0, 6]
```
Full 5×5 (p=0):          crop 1 each side → Output 3×3 (p=1):
| 2 4 3 2 1 |
| 0 2 0 1 0 |            | 2 0 1 |
| 3 2 9 6 5 |            | 2 9 6 |
| 0 1 0 3 0 |            | 1 0 3 |
| 1 0 5 0 6 |
```
เช็ก (full): sum = 56 = sum(X)·sum(K) = 7·8 ✓ ถ้าใส่ output_padding=1 จะได้ output 4×4: crop เหมือนเดิม แต่ด้านล่าง/ขวาไม่โดนตัด

**ตัวอย่างในสไลด์ (หน้า 29–40):** X = [[1,1,1,1],[1,1,1,1],[3,1,2,2],[3,1,2,2]], K = 3×3 เป็น 1 ทั้งหมด, s = 2, p = 0 → **9×9**:
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
ถ้า X กับ K เป็น 1 หมด ช่องที่ 2 block ซ้อนกันจะ ×2 และช่องที่ 4 block ซ้อนกันจะ ×4 ถ้า X เป็น 1 หมด s = 1 จะได้ 6×6 ค่าสูงสุดคือ 9 (ตรงกลาง 2×2)

**ตารางขนาด** (เช็กทุกแถวด้วย `F.conv_transpose2d` แล้ว):

| input | k | s | p | calc | output |
|---|---|---|---|---|---|
| 1×1 | 3 | 1 | 0 | 0 + 3 | 3×3 |
| 3×3 | 3 | 2 | 1 | 4 − 2 + 3 | 5×5 (สไลด์หน้า 51) |
| 4×4 | 3 | 1 | 0 / 1 / 2 | 3 − 2p + 3 | 6 / 4 / 2 |
| 4×4 | 3 | 2 | 0 / 1 / 2 | 6 − 2p + 3 | **9** / 7 / 5 |
| 4×4 | 3 | 3 | 0 | 9 + 3 | 12×12 (ไม่ซ้อนกัน) |
| 16×16 | 2 | 2 | 0 | 30 + 2 | 32×32 |
| 8×8 | 4 | 2 | 1 | 14 − 2 + 4 | 16×16 |
| 5×5 | 3 | 1 | 1 | 4 − 2 + 3 | 5×5 (ขนาดเท่าเดิม) |
| 32×32 | 2 | 2 | 0 | 62 + 2 | **64×64** (U-Net up-conv 2×2) |
| 32×32 | 4 | 2 | 1 | 62 − 2 + 4 | **64×64** |

**จำง่ายๆ:** k = s และ p = 0 จะขยายขนาดเป็น 2 เท่าพอดีเมื่อ s = 2 ส่วน k = 4, s = 2, p = 1 ก็ได้ 2 เท่าเหมือนกัน

**Nearest-neighbour vs bilinear** (X = [[10,20],[30,40]] → 5×5):
- Nearest ทำตามนี้:
  - พิกัดต้นทาง = 0, 1/4, 2/4, 3/4, 1 (ก้าวทีละ 1/(n−1) = 0.25)
  - index = ⌊src + 0.5⌋ = [0, 0, 1, 1, 1] ทั้งสองแกน
  - copy ค่ามา: แถว 0–1 = `10 10 20 20 20`, แถว 2–4 = `30 30 40 40 40`
  - ผลที่ได้: ค่าเป็นขั้นๆ เป็นบล็อก แค่ copy ค่าเดิมมา
- Bilinear: เรียบ เปลี่ยนแบบเส้นตรงทั้งสองทิศ แถวแรก `10 12.5 15 17.5 20`, แถวสุดท้าย `30 32.5 35 37.5 40`, ตรงกลาง 25 วิธีคำนวณ **ไม่ได้สอน** เลยแค่อธิบายได้พอ
- ทั้งสองแบบ **ไม่มี weight ให้ learn** ส่วน transposed conv **learn** kernel ของตัวเอง

### 4. คำตอบตัวอย่าง
**Full matrix (เช่น mock 5.1):**
- เขียน block X[i,j]·K ทั้ง 4 อัน พร้อมช่วงแถว/คอลัมน์ที่วาง (เหมือนตัวอย่าง B)
- โชว์ผลรวมแต่ละแถว เขียน full matrix แล้วเช็กขนาดด้วย (H−1)·s − 2p + k
- คำตอบ mock: X = [[1,2],[3,−1]], K = [[1,0,1],[2,1,0],[0,1,1]], s = 2, p = 0 ผลคือ
  `[[1,0,3,0,2],[2,1,4,2,0],[3,1,3,2,1],[6,3,−2,−1,0],[0,3,3,−1,−1]]` (5×5)
- ถ้า p = 1 ให้เอาตรงกลาง 3×3: `[[1,4,2],[1,3,2],[3,−2,−1]]`

**"padding ทำอะไรใน transposed conv?"**
> Output = (H−1)·s − 2p + k ดังนั้น padding p จะ **ตัดออก p แถว/คอลัมน์ จากทุกขอบ** ของ full output ทำให้ output **เล็กลง** 2p ซึ่งตรงข้ามกับ convolution ปกติ ที่ padding คือการเติม 0 ที่ขอบเพื่อไม่ให้ output หด ตัวอย่าง: 4×4, k=3, s=2 ได้ 9×9 ที่ p=0, 7×7 ที่ p=1 และ 5×5 ที่ p=2 สรุปคือ padding ไม่ได้ใช้ขยายภาพ ตัวที่ขยายคือ stride กับขนาด kernel

**"สไลด์บอกว่า padding มาก → output ใหญ่ขึ้น ถูกหรือผิด?"**
> **ผิด** จาก (H−1)·s − 2p + k ถ้า p มากขึ้น output จะเล็กลง ผลจาก PyTorch ในสไลด์เอง (หน้า 46–47: p = 0/1/2 → 9/7/5 ที่ s=2) และอาจารย์ก็พูดตรงกัน ("ยิ่ง Padding เพิ่ม ข้อมูลก็จะยิ่งหดลง", "the more padding, the smaller the output") ส่วนเรื่อง stride ในสไลด์นั้นถูก: stride มากขึ้น output ใหญ่ขึ้น

**"transposed conv เป็น inverse ของ conv ไหม?"**
> **ไม่ใช่** มันย้อนแค่ **shape** ที่เปลี่ยนไป: จาก W×H เล็ก → ใหญ่ แบบเดียวกับที่ conv ทำจากใหญ่ → เล็ก มันไม่ได้กู้ค่า pixel เดิมกลับมา weight ของ kernel ได้จากการ learn ด้วย backprop ให้ออกมาเป็นอะไรก็ได้ที่ decoder ต้องการ ในวิชานี้เรียกมันว่า deconvolution ด้วย แต่มันไม่ใช่ inverse ทางคณิตศาสตร์

**"เลือก k, s, p สำหรับ 32×32 → 64×64":**
> - (1) k=2, s=2, p=0: 31·2 + 2 = 64 (up-conv 2×2 ของ U-Net, ไม่ซ้อนกัน)
> - (2) k=4, s=2, p=1: 31·2 − 2 + 4 = 64
> - (3) k=3, s=2, p=1 ได้ 63 เลยต้องใส่ output_padding=1 ถึงจะได้ 64 *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*

**"ทำไม stride 3 กับ k=3 ถึงไม่ซ้อนกัน?"**
> block เริ่มทุกๆ s = 3 ช่อง และแต่ละ block กว้าง k = 3 พอดี เลยวางต่อกันพอดีเป๊ะ จะซ้อนได้ต้อง s < k

### 5. กับดัก
- **Flip kernel** อย่าทำ ใช้ K ตามที่โจทย์ให้มาเลย (ไม่หมุน)
- **สไลด์หน้า 41 ผิด**: padding มากขึ้น output **เล็กลง** ไม่ใช่ใหญ่ขึ้น
- ลืม **crop** ตอน p > 0 ต้องคำนวณ grid full (H−1)·s + k ก่อน แล้วค่อยตัด p ออกจาก **ทุก** ด้าน (รวม 2p)
- วาง block ที่ (i, j) แทนที่จะเป็น **(i·s, j·s)** ตอน s > 1
- ลืมช่องที่ 4 block ซ้อนกัน ถ้า s = 2 และ k = 3 ช่อง (2,2) จะได้ค่าจากทั้ง 4 block ของ input 2×2
- เผลอใช้สูตร conv ปกติ ⌊(H+2p−k)/s⌋+1
- เขียนแค่คำตอบ ข้อสอบบอกว่า "แสดงวิธีทำ" (show your work): ต้องเขียน block, ตำแหน่งที่วาง, ผลรวมแต่ละแถว และเช็กขนาด
- input ติดลบ: เก็บเครื่องหมายไว้ตลอดตอนบวก (−1·K ได้ block ติดลบ)

---

<a id="q6"></a>

## Q6 — Object detection

ที่มา: Ch 11 notes (อาจารย์แบงค์), Ch 14 §2 (พี่ใหม่, metrics), mock Q6, exam-hints ตัวเลขทั้งหมดข้างล่างเช็กด้วย Python แล้ว (`torchvision.ops.box_iou` / `nms`)

### 1. ข้อสอบจะถามอะไร

Mock Q6 (14 คะแนน) มี 4 ข้อย่อย ของจริงน่าจะออกมาหน้าตาคล้ายๆ กัน:
- **IoU คิดด้วยมือ** จากกล่องสองกล่อง (มักให้มาคนละ format) แล้วหา TP/FP/FN ที่ threshold หนึ่ง จากนั้น **แปลงเป็น YOLO label**
- **NMS ทีละขั้น**: กล่องอ้างอิง, IoU ของแต่ละคู่, กล่องไหนโดนลบ, กล่องที่เหลือ และผลเปลี่ยนยังไงถ้าเปลี่ยน threshold มักมีคำถามต่อเนื่องแบบ "ทำไม D ไม่โดนลบ?"
- **เทียบตระกูล R-CNN**: Where มาจากไหน? CNN รันกี่ครั้งต่อภาพ? ทำให้ขนาด region เท่ากันยังไง? ใช้ classifier อะไร? ทำไม Fast R-CNN ยังเป็น multi-stage? นับ **anchor และ filter 1×1**
- **YOLO**: ขนาด output tensor สำหรับ S, B, C ที่ให้มา, cell หนึ่งเก็บอะไร, ส่วนต่างๆ ของ loss, one-stage vs two-stage ในเคสทางการแพทย์ (วิดีโอ polyp แบบ real-time), จุดอ่อนของ YOLOv1
- อาจมีเพิ่ม: RoI pooling ด้วยมือ, 4 ขั้นของ R-CNN และทำไมต้อง warp, 6 ขั้นของ RPN, P/R/AP/mAP อาจารย์เน้น **"why"**: "ตอบ Why the reason ไม่ได้" คือสาเหตุที่ทำให้นักศึกษาตก

### 2. ต้องรู้

**ประเภทงาน** ประโยคของอาจารย์: "ไม่ได้ต่างกันที่ what ต่างกันที่ where"

| | What | Where |
|---|---|---|
| Classification | มี | ไม่มี |
| Detection | มี | **สี่เหลี่ยมที่เล็กที่สุดที่ครอบวัตถุได้ทั้งหมด** (bounding box) |
| Segmentation | มี | **รูปร่างจริง** ของวัตถุ (ขอบเป๊ะๆ) |

Output ของ detection ต่อวัตถุหนึ่งชิ้น: box + class + confidence (เช่น "dog 0.98")

**Format ของ box** ("ในตลาดมี 3 format" เลยต้องเช็กว่า library ต้องการแบบไหน):

| Format | ค่า |
|---|---|
| xyxy | (x1, y1, x2, y2) = มุมซ้ายบนและมุมขวาล่าง |
| xywh (corner) | (x, y, w, h) โดย (x, y) = มุมซ้ายบน |
| cxcywh (centre) | (xc, yc, w, h) |
| **YOLO label** | `class xc/W yc/H w/W h/H` ทุกค่าอยู่ใน 0–1, วัตถุละหนึ่งแถว, ภาพละหนึ่งไฟล์ `.txt` |

การแปลง: `w = x2−x1`, `h = y2−y1`, `xc = x1 + w/2`, `yc = y1 + h/2` แปลงกลับ: `x1 = xc − w/2`, `x2 = xc + w/2`
YOLO label ถูก normalise ไว้ เพื่อให้ **resize ภาพแล้ว label ไม่เปลี่ยน**

**สองตระกูล**
- **Multi-stage** (R-CNN, Fast R-CNN, Faster R-CNN): Where กับ What มาจาก **module/stage แยกกัน**
- **Single-stage** (YOLO): **network เดียว train แบบ end-to-end** ได้ What และ Where **พร้อมกัน**

**R-CNN (Girshick 2014): 4 ขั้น**
1. **Selective Search** ให้ region proposal ประมาณ 1,000–2,000 อัน เริ่มจาก graph-based segmentation แล้วรวม region ที่ติดกันแบบเป็นลำดับชั้นตามความคล้ายของ **สี, texture, ขนาด และ fill** มันหยุดเมื่อได้ **จำนวน proposal ตามเป้า ไม่ใช่จำนวนรอบที่ตายตัว** เพราะแต่ละภาพซับซ้อนไม่เท่ากัน แต่ละ region แปลงเป็น box ด้วย min/max ของ x และ y
2. **Warping**: resize ทุก region เป็น **224×224** *ทำไม:* CNN สมัยนั้น **ไม่มี Global/Adaptive Average Pooling** ถ้า input ขนาดไม่เท่ากัน ตอน flatten จะได้ความยาวไม่เท่ากัน แต่ FC layer ต้องการความยาวคงที่ *ข้อเสีย:* วัตถุ **เสีย aspect ratio**
3. **Features**: AlexNet ที่ **ตัด FC ตัวสุดท้ายออก** ให้ feature vector
4. **SVM** หนึ่งตัวต่อ class (maximum-margin hyperplane `WX + b = 0`)
- ไม่มี box regressor (ตามเวอร์ชันในวิชา) **จุดเจ็บ:** CNN รัน **~2,000 ครั้งต่อภาพ** เลยช้า

**Fast R-CNN: แก้ด้วยการรัน CNN ครั้งเดียว**
- CNN รันบน **ภาพทั้งภาพครั้งเดียว** ได้ feature map อันเดียว ส่วน Selective Search ยังรันบนภาพต้นฉบับอยู่
- **RoI projection**: map proposal แต่ละอันลงบน **feature map** โดยใช้พิกัดสัมพัทธ์ (0–1)
- **RoI Pooling** (max) เปลี่ยน crop ที่ขนาดต่างกันให้เป็นขนาดคงที่ แล้ว flatten → FC → **Softmax** (What) + **linear bbox regressor** (x, y, w, h) ที่คอย *ปรับ* Where ("ขยับยุกยิกนิดนึง ยืดหดนิดนึง")
- Training 84 h → 9.5 h (8.8×) Test ไม่นับ proposal 47 s → 0.32 s (146×) **ถ้านับ Selective Search ด้วย 50 s → 2 s (แค่ 25×)**
- **ยังเป็น multi-stage และไม่ real-time:** ตอนนี้ Selective Search กลายเป็นคอขวด และมัน **train ไม่ได้**

**Faster R-CNN (Ren 2015): แก้ด้วยการเอา RPN มาแทน Selective Search**
- **Region Proposal Network** train ได้ ส่วนอื่นเหมือน Fast R-CNN ทั้งหมด
- RPN มี 6 ขั้น:
  1. **conv 3×3, 512 filters** ตรงกลาง (เปลี่ยน feature แบบ What ให้เป็น feature แบบ Where)
  2. **Anchors**: กล่องที่กำหนดไว้ล่วงหน้าจาก **scale × aspect ratio = 3 × 3 = 9** ต่อตำแหน่ง ความเชื่อคือ: **ทุกจุดบน feature map อาจเป็นจุดกึ่งกลางของวัตถุได้**
  3. Classification layer: **objectness score** ว่าเป็น foreground/background
  4. Regression layer: ปรับ [x, y, w, h] ให้แม่นขึ้น
  5. เก็บ anchor ที่เป็น foreground เรียงตาม score แล้วเก็บ **top n (~2,000 หรือ 6,000)**
  6. **NMS**
- Implementation: conv 1×1 ที่มี **k filters → scores**, conv 1×1 ที่มี **4k filters → coordinates** **"ตั้งชื่อตามการใช้งาน"** ("ตัวเปลี่ยนโลก"): output ของ conv 1×1 กลายเป็น "score" ได้เพราะ loss train มันให้ทำหน้าที่นั้น **YOLO เกิดมาจากไอเดียนี้**
- Output ของ regressor คือ **ค่าปรับ (adjustment)**: offset x, y เอาไป **บวก**, w, h เอาไป **คูณ** (เป็นอัตราส่วน)
- 0.2 s/ภาพ เร็วกว่า R-CNN **250×** mAP VOC07: 66.0 / 66.9 / **66.9** ดังนั้นที่ได้มาคือ **ความเร็ว** ไม่ใช่ความแม่น
- ยังเป็น **two-stage** ถึงจะ train ได้ทั้งหมดแล้วก็ตาม

**YOLO (Redmon 2015): single-stage**
- แบ่งภาพเป็น **grid S×S** cell ที่มี **จุดกึ่งกลาง** ของวัตถุอยู่จะรับผิดชอบวัตถุนั้น (เป็นความเชื่อเดียวกับ RPN แต่ใช้ grid บนภาพต้นฉบับแทน feature map)
- แต่ละ cell ทำนาย **B boxes × (x, y, w, h, confidence)** บวก **class probability P(class|object) หนึ่งชุดต่อ cell**
  - x, y คือ offset ภายใน cell ส่วน w, h หารด้วย W, H ของภาพ ทุกค่าอยู่ใน 0–1
  - ใน B boxes ตัวที่ **IoU กับ GT สูงสุด** คือตัวที่รับผิดชอบ
- Score สุดท้าย = P(class|object) × box confidence แล้วตามด้วย **NMS** (เพิ่ม mAP ประมาณ 2–3%)
- YOLOv1: **24 conv + 2 FC**, S = 7, B = 2, C = 20 → **7×7×30**
- **Loss = 5 เทอม SSE**:
  1. error ของ (x, y), × λ_coord
  2. error ของ (√w, √h), × λ_coord รากที่สองช่วยลดความไม่สมดุลระหว่างกล่องใหญ่กับกล่องเล็ก
  3. Confidence ตรงที่มีวัตถุ
  4. Confidence ตรงที่ไม่มีวัตถุ, × **λ_noobj = 0.5**
  5. Class probability (เฉพาะ cell ที่มีวัตถุ)
  - **λ_coord = 5** เพิ่มน้ำหนักให้ localisation loss ครอบคลุมทั้ง objectness/confidence และ class
- **จุดอ่อน:** **หนึ่ง class ต่อ cell** (วัตถุต่าง class อยู่ใน cell เดียวกันไม่ได้) และ **วัตถุเล็กๆ จำนวนมากที่อยู่ติดกัน** (เพราะ cell ใหญ่)
- YOLOv8 (Ultralytics 2023): ขนาด n/s/m/l/x ใหญ่ขึ้น = แม่นขึ้นแต่ใช้ทรัพยากรมากขึ้น n มี 3.2 M params เหมาะกับมือถือ/อุปกรณ์ embedded

**Two-stage vs one-stage**

| | Two-stage (Faster R-CNN) | One-stage (YOLO) |
|---|---|---|
| Pipeline | RPN proposals → RoI pooling → head | network เดียว grid ทำนาย box + class ตรงๆ |
| What & Where | แยก stage | **พร้อมกัน**, end-to-end |
| ความเร็ว | 0.2 s/img (ไม่ real-time) | real-time |
| วัตถุหนาแน่น / เล็ก | ดีกว่า (9 anchors ทุกตำแหน่งบน feature map) | YOLOv1 อ่อน (1 class ต่อ cell, cell ใหญ่) |
| ใช้ตอนไหน | ต้องการความแม่นกับวัตถุเล็กที่อยู่เบียดกัน, งาน offline | วิดีโอ, real-time, edge device |

**Metrics (Ch 14, พี่ใหม่)**
- **TP**: class ถูก **และ** IoU ≥ threshold (ปกติ 0.5) **FP**: class ผิด **หรือ** IoU < threshold **หรือ** ไม่มีวัตถุจริง (รวมกล่องซ้ำด้วย) **FN**: วัตถุ GT ที่หาไม่เจอ **ไม่ใช้ TN** เพราะ region ที่เป็น background มีเยอะเกินไป
- Precision = TP/(TP+FP). Recall = TP/(TP+FN) หน่วยที่นับคือ **วัตถุ (box)**
- **AP** = พื้นที่ใต้ P–R curve ของ **class เดียว** ได้จากการไล่ปรับ **confidence threshold** **mAP** = ค่าเฉลี่ย AP ของทุก class mAP@0.5 ใช้ IoU 0.5 ส่วน mAP@0.5:0.95 เฉลี่ยจาก IoU threshold 0.5–0.95 (YOLOv8 "mAP 50–95")

### 3. สูตร + ตัวอย่างทำจริง

**IoU** = intersection / (area1 + area2 − intersection)
สำหรับกล่อง xyxy: `ix1 = max(x1)`, `iy1 = max(y1)`, `ix2 = min(x2)`, `iy2 = min(y2)`, `inter = max(0, ix2−ix1) · max(0, iy2−iy1)`

*ตัวอย่าง (มีแปลง format).* GT ให้มาเป็น corner xywh (20, 30, 60, 40) prediction ให้มาเป็น cxcywh (60, 55, 50, 50)
1. แปลง GT: x2 = 20 + 60 = 80, y2 = 30 + 40 = 70 ดังนั้น GT = (20, 30, 80, 70) Area = 60·40 = **2,400**
2. แปลง prediction: x1 = 60 − 25 = 35, x2 = 85, y1 = 55 − 25 = 30, y2 = 80 ดังนั้น P = (35, 30, 85, 80) Area = 50·50 = **2,500**
3. Intersection: x จาก max(20, 35) = 35 ถึง min(80, 85) = 80 กว้าง 45 ส่วน y จาก max(30, 30) = 30 ถึง min(70, 80) = 70 สูง 40 inter = **1,800**
4. Union = 2,400 + 2,500 − 1,800 = **3,100** **IoU = 1,800/3,100 = 0.581**
5. ที่ threshold 0.5 และ class ถูก อันนี้เป็น **TP** แต่ที่ 0.75 จะเป็น **FP** และ GT กลายเป็น **FN**

*เช็ก Mock 6.1:* G = (40, 40, 100, 100), P = (60, 50, 120, 110). inter = 40·50 = 2,000, union = 3,600 + 3,600 − 2,000 = 5,200, **IoU = 0.385** < 0.5 ดังนั้น P เป็น FP และ G เป็น FN YOLO label ของ G ในภาพ 512×512: `0 0.1367 0.1367 0.1172 0.1172` (70/512, 70/512, 60/512, 60/512)

*เช็กจาก notes:* (0, 0, 4, 4) กับ (2, 2, 6, 6) ได้ 4/28 = **0.143** YOLO label ของ xyxy (100, 50, 300, 250) ในภาพ 640×480 คือ `0 0.3125 0.3125 0.3125 0.4167`

**NMS** (ขั้นตอนในสไลด์ 2.6.1–2.6.4):
1. เรียงกล่องตาม score จากมากไปน้อย
2. เอากล่องบนสุดเป็นกล่องอ้างอิง แล้วคำนวณ IoU กับกล่องที่เหลือทุกกล่อง
3. ถ้า IoU > threshold แปลว่าเป็นวัตถุเดียวกัน: **ลบกล่องที่ score ต่ำกว่า**
4. ไปกล่องถัดไปที่ *ยังรอดอยู่* แล้วทำซ้ำ

*ตัวอย่าง (threshold 0.5).* กล่อง A–C ขนาด 100×100 (area 10,000) D กับ E ขนาด 80×80 (area 6,400)

| Box | xyxy | score |
|---|---|---|
| A | (50, 50, 150, 150) | 0.92 |
| B | (70, 60, 170, 160) | 0.85 |
| C | (95, 70, 195, 170) | 0.80 |
| D | (300, 80, 380, 160) | 0.75 |
| E | (310, 90, 390, 170) | 0.60 |

| คู่ | intersection | union | IoU |
|---|---|---|---|
| A, B | 80·90 = 7,200 | 20,000 − 7,200 = 12,800 | **0.563** |
| A, C | 55·80 = 4,400 | 15,600 | **0.282** |
| B, C | 75·90 = 6,750 | 13,250 | **0.509** |
| D, E | 70·70 = 4,900 | 12,800 − 4,900 = 7,900 | **0.620** |
| A หรือ B หรือ C กับ D หรือ E | ไม่ซ้อนกัน | | 0 |

- เรียงแล้ว: A, B, C, D, E
- **อ้างอิง A:** B 0.563 > 0.5 เลย **ลบ B** C 0.282 เก็บไว้ D 0 เก็บไว้ E 0 เก็บไว้
- **อ้างอิง C:** D 0 เก็บไว้ E 0 เก็บไว้
- **อ้างอิง D:** E 0.620 > 0.5 เลย **ลบ E**
- **เหลือ: A, C, D**
- *ทำไม C รอด ทั้งที่ IoU(B, C) = 0.509 > 0.5?* เพราะ B โดน A ลบไปก่อนแล้ว NMS เทียบแค่กับกล่องที่ **ยังรอดและ score สูงกว่า** และ IoU(A, C) = 0.282 ต่ำกว่า threshold
- Threshold 0.25: C โดนลบด้วย (0.282 > 0.25) เหลือ **A, D** Threshold 0.6: B รอด (0.563 < 0.6) เหลือ **A, B, C, D**
- กฎ: **threshold ต่ำ ลบเยอะขึ้น** (เสี่ยงลบวัตถุจริงที่อยู่ใกล้ๆ) **threshold สูง เหลือกล่องซ้ำเยอะขึ้น** Ultralytics `iou` ค่า default = 0.6
- (คำตอบ Mock 6.2 ไว้เทียบ: 0.5 ได้ A, C, D; 0.4 ได้ A, C)

**Anchors และ filter ของ RPN**
- k = scales × ratios = 3 × 3 = **9** anchors ต่อตำแหน่ง Anchor ทั้งหมด = H_fm × W_fm × k
  - 10×10 → **900** 32×32 → **9,216** 50×50 → **22,500** (ไม่ใช่ "~30,000")
- filter ของ conv 1×1 ตามที่สไลด์ให้มา: **k สำหรับ objectness score, 4k สำหรับ coordinates** ถ้า k = 9 ก็คือ **9 และ 36** (คำตอบ quiz ในห้อง) ต่อ anchor หนึ่งตัวคือ 5 ค่า (1 score + x, y, w, h)
- *(paper ต้นฉบับใช้ **2k** scores คือ softmax fg + bg เลยได้ 18 สำหรับ k = 9 ให้เขียน k = 9 ตามสไลด์ แล้วค่อยพูดถึง 2k เป็นหมายเหตุเสริม)*

**ขนาด cell ของ RoI pooling**
- แบ่งแต่ละด้านที่ยาว m ออกเป็น n ส่วน ส่วนละ [m/n] โดย **เศษไปอยู่ส่วนสุดท้าย** แล้วเอา **max** ของแต่ละส่วน
- สไลด์ 24 (ปัดลง): region 5 × 7 pool เป็น 2×2 แบ่งแถวเป็น **2 + 3** และคอลัมน์เป็น **3 + 4** ผลคือ `[0.85, 0.84, 0.97, 0.96]`
- อาจารย์พูดแก้ในห้อง (ปัดขึ้น): 8/3 → **3, 3, 2** ในข้อสอบให้ **บอกด้วยว่าใช้กฎปัดแบบไหน**
- *ตัวอย่าง:* pool region 4×5 นี้ให้เป็น 2×2
  ```
  1 5 2 0 3
  4 2 7 1 1
  0 6 3 8 2
  9 1 2 4 5
  ```
  - ปัดลง: แถว 2 + 2 คอลัมน์ 2 + 3 เอา max ของแต่ละ block ได้ **[[5, 7], [9, 8]]** flatten เป็น [5, 7, 9, 8]
  - ปัดขึ้น: คอลัมน์ 3 + 2 ได้ **[[7, 3], [9, 8]]**

**ขนาด output ของ YOLO**: ต่อ cell = **B·5 + C** Tensor = **S × S × (B·5 + C)**
- S = 7, B = 2, C = 20 → 30 → **7×7×30** (1,470 ค่า)
- C = 3 → 13 → **7×7×13** C = 10 → **20 filters** (ตัวอย่างของอาจารย์)
- การเรียงค่าต่อ cell: ค่าที่ 1–5 = box 1 (conf, x, y, w, h), 6–10 = box 2 แล้วตามด้วย class probability C ค่า (**หนึ่งชุดต่อ cell**)

**Precision / Recall / AP** (ตัวอย่างในสไลด์ Ch 14)
- นก: detect ได้ 12 อัน ถูก 10 อัน มีนกจริง 15 ตัว P = 10/12 = **0.833**, R = 10/15 = **0.667**, F1 = 2PR/(P+R) = **0.741**
- AP(bird): ไล่ปรับ confidence threshold (GT นก 20 ตัว) เช่นที่ conf ≥ 0.85 ได้ TP 6 และ FP 1 จะได้ P = 0.86, R = 0.30 พื้นที่ใต้ P–R curve แบบ trapezoid เริ่มที่ (R 0, P 1) คือ **0.773**
- mAP = (Bird 0.773 + Car 0.883 + Dog 0.842)/3 = **0.833**
- confidence threshold สูงขึ้น P สูงขึ้นแต่ R ต่ำลง threshold ต่ำลง R สูงขึ้น FP เยอะขึ้น และ P ต่ำลง

### 4. คำตอบตัวอย่าง

**Q: classification, detection และ segmentation ต่างกันยังไง?**
Classification ให้แค่ *What* (class ของภาพ) ส่วน detection กับ segmentation ให้ทั้ง *What* และ *Where* และต่างกันแค่ที่ Where Where ของ detection คือสี่เหลี่ยมที่เล็กที่สุดที่ครอบวัตถุแต่ละชิ้นได้ทั้งหมด (bounding box พร้อม class และ confidence) ส่วน Where ของ segmentation คือรูปร่างจริงของวัตถุ ระดับ pixel ต่อ pixel ดังนั้นถ้าจะวัดพื้นที่หรือปริมาตรของเนื้องอกให้ใช้ segmentation แต่ถ้าแค่หาและนับรอยโรค ใช้ box ก็พอ

**Q: ขั้นตอนของ R-CNN; ทำไมต้อง warp?**
(1) Selective Search เสนอ region ~2,000 อัน โดยรวม region ที่ติดกันและคล้ายกันแบบเป็นลำดับชั้น (สี, texture, ขนาด, fill) (2) warp แต่ละ region เป็น 224×224 (3) AlexNet ที่ตัด FC ตัวสุดท้ายออก ดึง feature vector ออกมา (4) ใช้ SVM หนึ่งตัวต่อ class ในการ classify ที่ต้อง warp เพราะ CNN สมัยนั้นไม่มี global/adaptive average pooling ถ้า input ขนาดต่างกัน ตอน flatten จะได้ความยาวต่างกัน แต่ FC layer ต้องการความยาวคงที่ ข้อเสียคือ aspect ratio เพี้ยน R-CNN ช้าเพราะต้องรัน CNN ประมาณ 2,000 ครั้งต่อภาพ

**Q: ทำไม Fast R-CNN เร็วกว่า และทำไมยังเป็น multi-stage / ไม่ real-time?**
Fast R-CNN รัน CNN **ครั้งเดียวบนภาพทั้งภาพ** แล้ว project proposal ลงบน feature map (ใช้พิกัดสัมพัทธ์) และใช้ RoI max pooling แทนการ warp classify ด้วย FC + softmax แทน SVM และเพิ่ม bbox regressor เวลา test ลดจาก 47 s เหลือ 0.32 s (146×) ถ้าไม่นับ proposal แต่ Where ยังมาจาก **Selective Search** ซึ่งเป็น stage แยกที่ train ไม่ได้ เลยยังเป็น multi-stage ถ้านับ Selective Search ด้วยจะใช้ 2 s/ภาพ (แค่ 25×) ซึ่งไม่ real-time

**Q: Faster R-CNN เปลี่ยนอะไร? อธิบาย RPN**
มันเอา **Region Proposal Network ที่ train ได้** มาแทน Selective Search โดยใช้ feature map จาก backbone ร่วมกัน conv 3×3 (512 filters) ปรับ feature ให้เหมาะกับ Where ทุกตำแหน่งบน feature map จะสมมติ anchor ไว้ 9 อัน (3 scales × 3 ratios) conv 1×1 ที่มี k filters ให้ objectness score และอีกอันที่มี 4k filters ให้ box offset (x, y บวก; w, h scale) เก็บ anchor ที่เป็น foreground ~2,000 อันที่ score สูงสุด แล้วใช้ NMS ลบตัวซ้ำ ที่เหลือคือ Fast R-CNN ผลคือ 0.2 s/ภาพ (เร็วกว่า R-CNN 250×) โดย mAP เท่าเดิม (66.9) ดังนั้นที่ได้คือความเร็ว และมันยังเป็น two-stage

**Q: Feature map 32×32, 3 scales × 3 ratios: มีกี่ anchor และกี่ filter?**
k = 3 × 3 = 9 anchors ต่อตำแหน่ง รวม = 32 × 32 × 9 = **9,216 anchors** filter ของ conv 1×1: **9** สำหรับ objectness score (หนึ่งอันต่อ anchor แต่ละแบบ) และ **36 = 9 × 4** สำหรับ coordinates (x, y, w, h ต่อ anchor)

**Q: YOLO detect ยังไง? ขนาด output สำหรับ S = 7, B = 2, C = 3?**
YOLO เป็น network แบบ single-stage, end-to-end แบ่งภาพเป็น grid 7×7 และ cell ที่มีจุดกึ่งกลางของวัตถุจะรับผิดชอบวัตถุนั้น แต่ละ cell ทำนาย B = 2 boxes แต่ละอันมี (conf, x, y, w, h) บวก class distribution P(class|object) หนึ่งชุด ต่อ cell คือ 2·5 + 3 = 13 ดังนั้น output คือ **7×7×13** x, y เป็น offset ใน cell ส่วน w, h normalise ด้วยขนาดภาพ Score สุดท้าย = class prob × confidence แล้วตามด้วย NMS What กับ Where ออกมาพร้อมกัน เลยเร็ว

**Q: อธิบาย loss ของ YOLO**
เป็นผลรวมของ squared-error 5 เทอม Localisation: (x, y) และ (√w, √h) ทั้งคู่ถ่วงด้วย λ_coord = 5 เพื่อให้ความแม่นของ box สำคัญขึ้น รากที่สองทำให้ error บนกล่องเล็กมีน้ำหนักมากกว่า error ขนาดเท่ากันบนกล่องใหญ่ Confidence ตรงที่มีวัตถุ และ confidence ตรงที่ไม่มีวัตถุ ถ่วงด้วย λ_noobj = 0.5 เพราะ cell ส่วนใหญ่ว่าง ถ้าไม่ถ่วงจะครอบงำ loss Class probability นับเฉพาะ cell ที่มีวัตถุ

**Q: Detect polyp แบบ real-time ในวิดีโอ colonoscopy: one- หรือ two-stage?**
One-stage (YOLO) เพราะมันทำนาย What และ Where พร้อมกันใน network เดียว เลยทำได้ทัน frame rate ของวิดีโอ ขณะที่ Faster R-CNN ใช้ ~0.2 s/frame รุ่นเล็ก (YOLOv8n/s) เหมาะกับอุปกรณ์ embedded ที่ติดกับกล้อง endoscope ต้องระวังจุดอ่อนของ YOLOv1: ได้แค่หนึ่ง class ต่อ cell และทำได้ไม่ดีกับ polyp เล็กๆ จำนวนมากที่อยู่ใกล้กัน สำหรับงาน screening เรื่อง recall สำคัญ (polyp ที่พลาดไปคือ FN) เลยต้องปรับ confidence threshold ให้เหมาะ

**Q: prediction นี้เป็น TP ไหม? ทำไมไม่ใช้ TN? mAP คืออะไร?**
เป็น TP ก็ต่อเมื่อ class ถูก **และ** IoU กับ GT ที่ยังไม่ถูก match ≥ threshold (เช่น 0.5) ไม่งั้นเป็น FP และ GT ที่ไม่ถูก match คือ FN ไม่ใช้ TN เพราะ region ที่เป็น background มีนับไม่ถ้วน AP คือพื้นที่ใต้ precision–recall curve ของ class เดียว ได้จากการไล่ปรับ confidence threshold ส่วน mAP คือค่าเฉลี่ย AP ของทุก class รายงานที่ IoU threshold หนึ่ง (mAP@0.5 หรือ @0.5:0.95)

### 5. กับดัก

- **Union = A + B − intersection** อย่าหารด้วย A + B แปลงทุกกล่องเป็น xyxy **ก่อน** คำนวณ IoU และเช็ก format ที่โจทย์ให้ (corner xywh vs centre vs xyxy)
- ถ้ากล่องไม่ซ้อนกัน intersection คือ **max(0, …)** = 0 ดังนั้น IoU = 0 ห้ามใช้ความกว้างติดลบ
- **NMS ลบกล่องที่ score ต่ำกว่า** เมื่อ IoU **>** threshold และเทียบแค่กับกล่องที่ **ยังรอด** กล่องที่โดนลบไปแล้วจะลบใครไม่ได้ กล่องที่ IoU = 0 คือวัตถุคนละชิ้น เก็บไว้ทั้งคู่
- NMS threshold ต่ำลงจะลบกล่อง **มากขึ้น** ไม่ใช่น้อยลง `iou` ของ Ultralytics = threshold ของ **NMS** (default 0.6) ไม่ใช่ threshold ของ mAP ส่วน `conf` default 0.001 คือ "not recommended"
- **สไลด์ผิด:** "Faster R-CNN แม่นกว่า" ตารางแสดง mAP 66.0 / 66.9 / 66.9 ดังนั้นที่ได้คือ **ความเร็ว** (250×)
- **พูดผิดในห้อง:** "~30,000 anchors" สำหรับ 50×50 ที่ถูกคือ 50·50·9 = **22,500**
- Filters: **k scores, 4k coordinates** (9 / 36) อย่าเขียน 9 × 5 = 45 ว่าเป็น "filter ของ coordinates" ถ้าจะพูดถึง 2k ของ paper ให้บอกว่ามาจาก paper
- **Fast R-CNN ยังใช้ Selective Search** เลยยังเป็น multi-stage **Faster R-CNN ยังเป็น two-stage** (ไม่ใช่ single-stage) ถึงจะ train ได้ มีแค่ YOLO ที่เป็น single-stage
- Fast R-CNN เร็วขึ้น 146× **ไม่นับ** proposal ถ้านับ Selective Search ด้วยแค่ **25×** Training 8.8×
- R-CNN = classifier เป็น **SVM**, AlexNet ที่ตัด FC ตัวสุดท้ายออก, **ไม่มี box regressor** (ตามเวอร์ชันในวิชา) กล่องอยู่บน **ภาพต้นฉบับ** และโดน warp ส่วน Fast/Faster วางกล่องบน **feature map** และใช้ RoI pooling
- คำว่า "Wrapping" ในสไลด์หมายถึง **warping** (การ resize)
- RoI pooling = **MAX** ไม่ใช่ average ("สูงสุดนะ ไม่ใช่เฉลี่ยนะ") เศษไปอยู่ส่วน **สุดท้าย** และบอกกฎการปัดของตัวเองด้วย
- Output ของ regressor คือ **ค่าปรับ** (x, y บวก, w, h คูณ) ไม่ใช่พิกัดใหม่ regressor แค่ปรับ Where ให้แม่นขึ้น ไม่ได้สร้าง Where ขึ้นมาเอง
- YOLO: class probability เป็น **ต่อ cell**, confidence เป็น **ต่อ box** B = 2 แต่ได้แค่ **หนึ่ง class ต่อ cell** ต่อ cell = B·5 + C (ไม่ใช่ B·(5 + C))
- อย่าสลับ λ: **λ_coord = 5**, **λ_noobj = 0.5** loss ใช้ **√w, √h**
- YOLO label = `class xc yc w h` เป็น **จุดกึ่งกลาง** ที่ normalise เป็น 0–1 ไม่ใช่มุม และไม่ใช่ xyxy จุด origin (0, 0) ของภาพอยู่มุมซ้ายบน
- AP เป็น **ต่อ class** mAP คือค่าเฉลี่ยของทุก class detection **ไม่ใช้** TN กล่องที่ class ผิดแต่ IoU ดีถือเป็น **FP**

---
<a id="q7"></a>

## Q7 — เขียนโค้ด (PyTorch)

ที่มา: `materials/Chapter 10/notes.md` (หัวข้อ 2–10 + จุดที่อาจารย์เน้น), mock Q7. ในห้องสอบต้อง**เขียนโค้ดด้วยมือ** เทมเพลตข้างล่างเลยทำให้สั้น และทุกบรรทัดมีหน้าที่ของมัน โค้ดทุกบล็อกในนี้รันจริงแล้ว (torch 2.11 / torchvision 0.26) และเช็กตัวเลข shape ครบทุกตัว *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* = ความรู้ PyTorch ทั่วไป ไม่ได้มาจากสไลด์

### 1. ข้อสอบจะถามอะไร

| แบบ | ตัวอย่าง | ทำยังไงถึงได้คะแนน |
|---|---|---|
| เขียน loop | "เขียน training + validation loop แล้วเซฟ weights ที่ดีที่สุด" (mock 7.2) | ลำดับถูก, มี `eval()` + `no_grad()`, เซฟ `state_dict` ตอน val ดีขึ้น |
| เรียงลำดับ / เติมช่องว่าง | เอา `backward`, `step`, `zero_grad`, `model(x)`, `criterion` มาเรียงให้ถูก | `zero_grad → forward → loss → backward → step` |
| หาบั๊ก | สคริปต์ fine-tune ที่มีบั๊ก "หาให้ได้ ≥6 จุด บอกบรรทัด + ปัญหา + วิธีแก้" (mock 7.1) | บอกบรรทัด, บอก*ผลที่ตามมา*, ให้วิธีแก้ |
| อธิบายบรรทัด | "บรรทัดไหนที่เทรน?", "`is_available()` เช็กอะไร?", "ทำไมต้อง `model.train()`?" | ตอบตามประเด็นที่อาจารย์พูดเป๊ะๆ (§2) |
| Shapes | "เติม `Linear(?, 10)`", ขนาด output ของ conv | เขียนสูตร + แสดงทุกขั้น |
| แก้ pretrained model | ResNet-50 สำหรับ 4 คลาส / multilabel / regression | `weights=...`, `model.fc = nn.Linear(2048, n)`, เลือก loss ให้ถูก |
| Inference | `predict(path)` ที่พิมพ์ `Predicted: MEL 97.31%` (mock 7.3) | สร้าง arch ใหม่ + `load_state_dict`, ใช้ val transform, `unsqueeze(0)`, eval + no_grad, softmax + max |

### 2. ต้องรู้

**2.1 Training loop: ลำดับ และแต่ละบรรทัดมีไว้ทำไม** (หัวใจของข้อสอบ)

| # | บรรทัด | ทำไม |
|---|---|---|
| 0 | `model.train()` | ตั้ง **state** (เปิด Dropout, BatchNorm ใช้ค่าสถิติของ batch) ตัวมันเองไม่ได้เทรนอะไรเลย อาจารย์ติดดาวไว้: "ตายมานักต่อนักแล้ว" |
| 1 | `x, y = x.to(device), y.to(device)` | model กับข้อมูลต้องอยู่บน **device เดียวกัน** |
| 2 | `optimizer.zero_grad()` | PyTorch **สะสม** gradient ไว้ ถ้าไม่ล้าง gradient ของ batch ก่อนจะถูกบวกเข้ามาด้วย |
| 3 | `out = model(x)` | **การทำนาย (forward pass)** *ไม่ใช่*บรรทัดที่เทรน ("อันนี้ผิด") ต้องมีเพราะ loss ต้องใช้ค่าทำนาย |
| 4 | `loss = criterion(out, y)` | ตัวเลขตัวเดียวที่บอกว่า model ผิดมากแค่ไหน |
| 5 | `loss.backward()` | **คำนวณ gradient** ของ loss เทียบกับทุก parameter ที่ `requires_grad=True` |
| 6 | `optimizer.step()` | **อัปเดต weights**: w_new = w_old − α·∂loss/∂w "เวทถูกอัปเดตจริงๆที่คำสั่ง optimizer.step" |

- "คำสั่งไหนที่เทรน model?" → ต้องตอบเป็นชุด **predict → criterion → backward → step** (4 บรรทัด)
- มี for-loop 2 ชั้น: ชั้นนอก = epochs, ชั้นใน = batches ใส่ `step()` ไว้ใน loop ของ batch = อัปเดตทุก batch (แบบที่ใช้กันปกติ) ไม่อัปเดตทีละภาพ (batch_size=1) เพราะภาพ outlier ภาพเดียวจะดัน weights ไปผิดทาง ส่วน batch จะเฉลี่ย loss ให้
- พิมพ์ผลแค่ทุกๆ N batches (`if i % 500 == 499`) เพราะการแสดงผล (I/O) ช้า: "10 นาที ผมโดนไป 1 วัน"
- ใน training ให้ใช้ Torch อย่าใช้ NumPy: NumPy = CPU, Torch = GPU ถ้าแปลงไปมาจะต้อง copy ข้าม CPU↔GPU

**2.2 Validation / inference: `model.eval()` + `torch.no_grad()` ทำไมต้องใช้ทั้งคู่**
- `model.eval()`: **ปิด** Dropout, BatchNorm ใช้ **running mean/var ที่เก็บไว้** ถ้าลืม ผลทำนายจะเพี้ยนและไม่นิ่ง (โหมด default คือ train)
- `torch.no_grad()`: ไม่เก็บ gradient → เร็วขึ้น, กินเมมน้อยลง, ใช้ batch ใหญ่ขึ้นได้ แต่มัน**ไม่ได้**เปลี่ยน Dropout/BN
- สองตัวนี้ทำคนละหน้าที่ เลยต้องใช้**ทั้งคู่** ใส่ `eval()` **ก่อน** `with torch.no_grad():` (อาจารย์ลืมใส่ในสไลด์ตัวเอง)
- ใน validation **ไม่มี** `zero_grad / backward / step` weights จะเปลี่ยนก็ตอน `optimizer.step()` เท่านั้น

**2.3 เช็ก Device / GPU: "ถูกครึ่งเดียว"**
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)                       # ทำครั้งเดียว หลังเปลี่ยน model.fc แล้ว
images, labels = images.to(device), labels.to(device)   # ทำทุก batch
```
- `is_available()` เช็ก **2 ขั้น**: (1) มี GPU ไหม (2) GPU นั้น**ว่าง**ไหม (ไม่ได้ถูกคนอื่นใช้เต็ม) ถ้าตอบแค่ "เช็กว่าเครื่องมี GPU ไหม" = **ได้ครึ่งเดียว**
- ถ้าเป็น False โค้ดจะไปรันบน CPU เงียบๆ (งาน 3 วันกลายเป็นเกินอาทิตย์) ถ้าทำงานจริงบน server ที่ใช้ร่วมกัน ให้บังคับ `.cuda()` ไปเลย จะได้ error ออกมา แทนที่จะรันช้าๆ

**2.4 Model class**
- `class Net(nn.Module)`; `__init__` **สร้าง layers เก็บเป็น attributes** (เรียก `super().__init__()` ก่อน); `forward(self, x)` **กำหนดว่าข้อมูลไหลผ่านยังไง**
- `forward` คืนค่า **score function (logits)** ส่วน softmax / sigmoid ให้ทำ**นอก** model
- จับคู่ channel: conv `out_channels` = `in_channels` ของ conv ถัดไป = `BatchNorm2d(n)` Linear in = out ของตัวก่อนหน้า **Linear ตัวสุดท้าย out = จำนวนคลาส**
- `nn.Sequential` รวม layers ตามลำดับ (แนะนำ เพราะเป็นระเบียบกว่า แต่แตกกิ่งไม่ได้) Flatten: `torch.flatten(x, 1)` = `x.view(x.size(0), -1)`
- Dropout ค่า default p=0.5 (ใช้ 0.2–0.5) ทำงานแค่ใน train mode

**2.5 คำนวณ Shape** (tensor = N×C×H×W)
- Conv: **out = ⌊(W − K + 2P)/S⌋ + 1** ถ้าหารไม่ลงตัวให้ปัดลง *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* ค่า default ของ `Conv2d(in, out, kernel_size, stride=1, padding=0)`
- K=3, P=1, S=1 ขนาดเท่าเดิม; K=5, P=2 ขนาดเท่าเดิม `MaxPool2d(2, 2)` ลด H, W ลงครึ่งหนึ่ง (สูตรเดียวกัน, P=0)
- ขนาด Flatten = C × H × W ของ feature map สุดท้าย channel เปลี่ยนแค่ใน conv ส่วน pooling C เท่าเดิม
- Params *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*: Conv = out·(in·K·K) + out; Linear(in,out) = out·in + out (`Linear(3,2)` → weight 2×3 + bias 2 = 8)

**2.6 Loss ตามแต่ละงาน** (โค้ด training ส่วนอื่น "แทบจะเหมือนกัน 100%")

| งาน | Output nodes | Loss | Target | อ่านผล (นอก model) |
|---|---|---|---|---|
| Multiclass | n คลาส | `nn.CrossEntropyLoss()` (มี softmax ในตัว → ให้ป้อน **logits**) | class index เป็น `long` shape (N,) | `torch.softmax(out,1)`, `torch.max(...,1)` |
| Multilabel | n labels | `nn.BCEWithLogitsLoss()` (sigmoid + BCE) | multi-hot เป็น **float** (N, n) | `torch.sigmoid(out) > 0.5` (โค้ดในสไลด์ใช้ 0.3) |
| Regression | ค่าที่จะทำนาย (แนะนำ 1 ค่าต่อ 1 model) | `nn.L1Loss()` (MAE) หรือ `nn.MSELoss()` | float (N, 1) | ใช้ค่าดิบเลย **ไม่มี activation** |

Multilabel = "หนึ่งภาพมีโอกาสที่จะเป็นไปได้มากกว่าหนึ่งคำตอบ" (และอาจไม่มีคำตอบเลยก็ได้) ถ้าตอบแค่ "1 ภาพหลายคำตอบ" = ได้ครึ่งเดียว

**2.7 Transforms: มี 2 หน้าที่** (ส่วนที่อาจารย์เน้นหนักสุดใน Part 2, "รุ่นพี่คุณพลาดมาแล้ว")
1. **Standardisation** (ToImage, Resize, ToDtype, Normalize): ต้อง**เหมือนกันเป๊ะ**ทั้ง train / val / test / inference
2. **Augmentation** (RandomHorizontalFlip, RandomRotation, RandomResizedCrop): **ใช้ตอน train เท่านั้น** ถ้า augment ตอน val/inference "ตลกมาก ไม่ควร": ภาพเดิมแต่ได้คำตอบไม่เหมือนเดิม
- `v2.ToDtype(torch.float32, scale=True)` → [0,1] ส่วน `ToTensor()` แบบเก่า (deprecated ใน v2) = `ToImage()` + `ToDtype(float32, scale=True)`
- `Normalize(mean, std)`: x′ = (x − μ)/σ ค่า (0.5, 0.5) แปลง [0,1] → [−1,1] ถ้าใช้ pretrained ImageNet models: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) และตอน inference ก็ต้องใช้ค่าเดียวกัน
- Pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize ("เราจะไม่ Resize ก่อน Padding") ภาพ grayscale ที่จะเข้า pretrained model แบบ 3 channel: ก๊อปให้เป็น 3 channels ไม่ต้องไปแก้ conv ตัวแรก

**2.8 Dataset / DataLoader**
- `Dataset` ที่เขียนเองต้องมี **3 methods**: `__init__` (รันครั้งเดียว: โหลด csv, folder, transform), `__len__` (จำนวน samples), `__getitem__(idx)` (คืน `(image, label)` ของ index นั้น และใส่ transform ตรงนี้)
- `DataLoader(ds, batch_size=32, shuffle=True)` สำหรับ train (สลับใหม่**ทุก epoch**); `shuffle=False` สำหรับ val/test `num_workers` = จำนวน process ของ CPU (ในสไลด์ใช้ default 0) การโหลดข้อมูล/augmentation รันบน CPU

**2.9 Pretrained backbone**
- `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` ส่วน `resnet50()` = weights **สุ่ม** ("อย่าทำแบบนี้") ยกเว้นกรณีที่จะ `load_state_dict` weights ของเราเองต่อทันที
- `print(model)` → layer สุดท้ายชื่อ `fc`, in_features **2048** → `model.fc = nn.Linear(2048, n_classes)` (1000 → n) "โค้ดบรรทัดนี้ สำคัญมาก"
- Freeze backbone *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*: `for p in model.parameters(): p.requires_grad = False` **แล้วค่อย**เปลี่ยน `fc` (layer ใหม่จะมี `requires_grad=True`) และให้ optimizer ใช้ `model.fc.parameters()`
- ลำดับ: เปลี่ยน `fc` → `model.to(device)` → สร้าง optimizer (ไม่งั้น fc ตัวใหม่จะอยู่บน CPU / ไม่อยู่ใน optimizer)

**2.10 Save / load**
- `torch.save(model.state_dict(), 'best.pt')` = **เก็บแค่ weights** (ไฟล์เล็กกว่า) ตอนโหลด: **สร้าง architecture เดิมขึ้นมาใหม่ (fc เหมือนเดิม)** → `model.load_state_dict(torch.load('best.pt'))` → "All keys matched successfully"
- `torch.save(model, path)` = architecture + weights, `model = torch.load(path)`
- เก็บ checkpoint ไว้ 3 ตัว: epoch สุดท้าย, จุด early-stopping, validation ดีที่สุด

**2.11 Data leakage ตอนแบ่งข้อมูล**
- นิยามของอาจารย์: ภาพเดียวกันเป็น **test ของ model A แต่เป็น train ของ model B** → เทียบกันไม่แฟร์ `train_test_split` / k-fold อัตโนมัติ ต่อให้ตั้ง seed ไว้ ก็อาจแบ่งไม่เหมือนกันในแต่ละรอบ
- วิธีแก้: **แบ่งเองครั้งเดียว** แล้วเซฟรายชื่อไว้ (train.csv / val.csv / test.csv) ทุก model โหลดไฟล์ชุดเดียวกัน
- ต่อยอดสำหรับงานแพทย์ (จาก exam-hints): แบ่ง**ตามคนไข้** คนไข้คนเดียวกันต้องไม่อยู่ทั้งใน train และ test

### 3. เทมเพลตโค้ดที่ต้องจำ

**3A. โปรแกรมเต็ม: dataset → fine-tune ResNet-50 → train + validate → เซฟตัวที่ดีสุด → predict ภาพเดียว**
```python
import os, torch, torch.nn as nn, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights

classes = ['MEL', 'NV', 'BCC', 'BKL']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # ImageNet
train_tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(10),          # augmentation: ใช้แค่ตอน TRAIN
    v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
val_tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224)),           # standardisation เหมือนเดิม
    v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])  # ไม่มี augmentation

class ImgDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)            # คอลัมน์: filename, label (0..3)
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
model.fc = nn.Linear(2048, len(classes))                          # 1000 -> 4 คลาส
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0.0
for epoch in range(20):
    model.train()                                  # ไว้ใน loop ของ epoch (เพราะข้างล่างมี eval())
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                      # 1 ล้าง gradient
        outputs = model(images)                    # 2 ทำนาย (logits, N x 4)
        loss = criterion(outputs, labels)          # 3 loss
        loss.backward()                            # 4 คำนวณ gradient
        optimizer.step()                           # 5 อัปเดต weights
        running_loss += loss.item()
    model.eval()                                   # ปิด Dropout, BN ใช้ running stats
    correct, total = 0, 0
    with torch.no_grad():                          # ไม่เก็บ gradient
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f'epoch {epoch+1} loss {running_loss/len(train_loader):.4f} val_acc {val_acc:.4f}')
    if val_acc > best_acc:                         # เก็บ weights ที่ validation ดีที่สุด
        best_acc = val_acc
        torch.save(model.state_dict(), 'best.pt')

def predict(path):
    model = resnet50()                             # architecture เดิม (weights มาจากไฟล์)
    model.fc = nn.Linear(2048, len(classes))       # fc เหมือนตอน train
    model.load_state_dict(torch.load('best.pt', map_location=device))
    model.to(device)
    model.eval()
    image = Image.open(path).convert('RGB')
    x = val_tf(image).unsqueeze(0).to(device)      # (3,224,224) -> (1,3,224,224)
    with torch.no_grad():
        outputs = model(x).squeeze(0)              # (1,4) -> (4,)
    score = torch.softmax(outputs, 0)              # softmax อยู่นอก model
    conf, idx = torch.max(score, 0)
    print(f'Predicted: {classes[idx]} {conf.item() * 100:.2f}%')
```

**3B. CNN เล็กๆ ที่เขียนเอง** (CIFAR net จากสไลด์, input 3×32×32; ใส่ `Resize((32, 32))` ใน transforms)
```python
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),   # 3x32x32 -> 6x28x28 -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2))  # -> 16x10x10 -> 16x5x5
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(120, num_classes))           # out = จำนวนคลาส
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)                    # (N,16,5,5) -> (N,400)
        return self.classifier(x)                  # logits, ไม่มี softmax

model = Net(num_classes=len(classes)).to(device)   # ใช้แทน ResNet ได้เลย
```

**3C. ตัวอย่างคำนวณ Shape** (เขียนทุกขั้น)

| Net | Layer | การคำนวณ | Output (C×H×W) |
|---|---|---|---|
| CIFAR (3B) | input | | 3×32×32 |
| | Conv2d(3,6,5) | (32 − 5 + 0)/1 + 1 = 28 | 6×28×28 |
| | MaxPool(2,2) | (28 − 2)/2 + 1 = 14 | 6×14×14 |
| | Conv2d(6,16,5) | (14 − 5)/1 + 1 = 10 | 16×10×10 |
| | MaxPool(2,2) | 10/2 | 16×5×5 → flatten **400** |
| `Net` ในสไลด์ (MNIST) | Conv2d(1,4,3,1,1) + BN(4) + ReLU + Pool, ×2 | (28 − 3 + 2)/1 + 1 = 28 → 14; ทำซ้ำ → 7 | 4×7×7 → `Linear(4*7*7 = 196, 10)` |
| ConvNet ในสไลด์ | Conv(1,32,k5,p2)+Pool, Conv(32,64,k5,p2)+Pool | (28 − 5 + 4) + 1 = 28 → 14 → 14 → 7 | 64×7×7 → `Linear(7*7*64 = 3136, 1000)` |
| Stride 2 *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* | Conv2d(3,16,3,stride=2,padding=1) บนภาพ 224 | ⌊(224 − 3 + 2)/2⌋ + 1 = 111 + 1 | 16×112×112 |

Params ของ `Conv2d(3,6,5)` = 6·(3·5·5) + 6 = **456**; `Linear(400,120)` = 400·120 + 120 = **48,120** ResNet-50 บนภาพ 224×224 จะจบด้วย vector ขนาด 2048 → fc *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์: global avg pool)*

### 4. ตัวอย่างคำตอบ

**(a) "เรียงลำดับ / บรรทัดไหนที่เทรน?"** → `model.train()`; ในแต่ละ batch: `.to(device)` → `optimizer.zero_grad()` → `out = model(x)` → `loss = criterion(out, y)` → `loss.backward()` → `optimizer.step()` โดย `model(x)` แค่**ทำนาย**; `backward()` คำนวณ gradient; `step()` คือตัวที่อัปเดต weights จริงๆ การเทรน = 4 บรรทัดนี้รวมกัน (predict → loss → backward → step)

**(b) "ทำไมต้อง `zero_grad()`?"** → `.backward()` จะ**บวกเพิ่ม**เข้าไปใน `.grad` ถ้าไม่ล้าง gradient ของทุก batch ก่อนหน้าจะสะสมกันไปเรื่อยๆ ทำให้แต่ละรอบอัปเดตแรงเกินและผิดทิศ

**(c) "ทำไมต้อง `model.train()` / `model.eval()`?"** → ใช้ตั้งโหมดให้ layers ที่ทำงานต่างกันในแต่ละโหมด: Dropout (เปิดตอน train, ปิดตอน eval) และ BatchNorm (ตอน train ใช้สถิติของ batch + อัปเดต running stats; ตอน eval ใช้ running stats ที่เก็บไว้) ถ้าลืม `eval()` ตอน test → ผลทำนายจะแกว่งและเพี้ยน

**(d) "`no_grad()` ต่างจาก `eval()` ยังไง?"** → `no_grad()` หยุดเก็บ gradient (เร็วขึ้น ประหยัดเมม); `eval()` สลับโหมด Dropout/BN ทำคนละหน้าที่ → ต้องใช้ทั้งคู่: `model.eval()` แล้วตามด้วย `with torch.no_grad():`

**(e) "`torch.cuda.is_available()` เช็กอะไร? ตอบว่า 'เช็กว่าเครื่องมี GPU ไหม' ได้เต็มไหม?"** → ไม่เต็ม ได้ครึ่งเดียว มันเช็ก (1) มี GPU **และ** (2) GPU นั้นว่าง/ใช้ได้ ถ้าเป็น False โค้ดจะไปรันบน CPU เงียบๆ (จากไม่กี่วันกลายเป็นหลายอาทิตย์) ดังนั้นบน server ที่ใช้ร่วมกัน ให้บังคับ `.cuda()` แล้วปล่อยให้มัน error

**(f) "ปรับ ResNet-50 ให้ใช้กับ 4 คลาส"** → `model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`; `model.fc = nn.Linear(2048, 4)`; แล้วค่อย `.to(device)` แล้วค่อยสร้าง optimizer weights ที่ pretrained มายังอยู่ ส่วน fc ตัวใหม่จะถูกเรียนรู้ใหม่ (fine-tuning)

**(g) "เปลี่ยนเป็น multilabel / regression"** → แก้แค่ 3 จุด:
```python
model.fc = nn.Linear(2048, n_labels); criterion = nn.BCEWithLogitsLoss()   # labels: float multi-hot (N, n_labels)
pred = torch.sigmoid(model(x)) > 0.5                                       # แต่ละ label แยกกัน อาจไม่มีเลยก็ได้
model.fc = nn.Linear(2048, 1); criterion = nn.L1Loss()                     # regression: labels เป็น float (N, 1), ไม่มี activation
age = model(x).squeeze().item()
```

**(h) "หาบั๊ก"** (ไว้ฝึก; mock 7.1 มี 8 จุดแบบเดียวกันนี้)
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
| บรรทัด | ปัญหา | วิธีแก้ |
|---|---|---|
| L1 | weights สุ่ม ไม่ใช่ pretrained | `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` |
| L2–L3 | สร้าง optimizer **ก่อน**เปลี่ยน fc → optimizer ถือ fc ตัวเก่าไว้ fc ตัวใหม่เลยไม่เคยถูกอัปเดต | เปลี่ยน `fc` ก่อน แล้วค่อยสร้าง optimizer (และ `.to(device)` ทั้ง model + data) |
| L5 | ใส่ softmax ก่อน `CrossEntropyLoss` (ซึ่งมี log-softmax ในตัวแล้ว) → softmax ซ้ำสองรอบ gradient อ่อน/ผิด | `out = model(images)` (logits) |
| L7 | ไม่มี `optimizer.zero_grad()` → gradient สะสมข้าม batch | ใส่ `optimizer.zero_grad()` ก่อน forward pass |
| L8 | ไม่มี `model.eval()` → Dropout ยังเปิด, BN ใช้สถิติของ batch และเอาข้อมูล val ไปอัปเดต running stats | ใส่ `model.eval()` ก่อน `with torch.no_grad():` และใส่ `model.train()` ตอนต้นทุก epoch |

**(i) "เซฟ model ที่ดีที่สุดแล้วโหลดมาทำ inference"** → ใน loop ของ epoch: `if val_acc > best_acc: best_acc = val_acc; torch.save(model.state_dict(), 'best.pt')` ตอนโหลด: สร้าง `resnet50()` + `fc` ตัวเดิมขึ้นมาใหม่, `load_state_dict(torch.load('best.pt'))`, `eval()` เพราะ `state_dict` = เก็บแค่ weights เลยต้องสร้าง architecture ให้เหมือนเดิมเป๊ะ

**(j) "เขียน Dataset เอง"** → ใช้ class `ImgDataset` ใน 3A: `__init__` / `__len__` / `__getitem__`; ใส่ `transform` ให้ภาพข้างใน `__getitem__`; train loader ใช้ `shuffle=True`, val/test ใช้ `shuffle=False`

### 5. กับดัก / บั๊กที่เจอบ่อย

| บั๊ก | ผลที่ตามมา | วิธีแก้ |
|---|---|---|
| ไม่มี `zero_grad()` | gradient สะสม (ลองแล้ว: backward 2 รอบ grad เพิ่มเป็น 2 เท่า) | `optimizer.zero_grad()` ทุก batch |
| `step()` ก่อน `backward()` | step แรกไม่มี gradient; step ต่อๆ ไปใช้ gradient ของ batch ก่อนหน้า | `backward()` แล้วค่อย `step()` |
| `softmax`/`sigmoid` ก่อน `CrossEntropyLoss`/`BCEWithLogitsLoss` | ทำซ้ำสองรอบ → gradient ผิดและแบนลง | ป้อน logits ดิบๆ |
| target ของ CE เป็น `labels.float()` | error: target ที่เป็น float จะถูกมองว่าเป็น class probabilities และต้องมี shape (N, C) | ใช้ class index แบบ `long`, shape (N,) |
| BCEWithLogits กับ target แบบ `long` | error (ต้องเป็น float) | `labels.float()` แบบ multi-hot (N, C) |
| L1/MSE ที่ output (N,1) แต่ target (N,) | broadcast เงียบๆ เป็น (N,N) loss ผิด *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* | `labels.float().view(-1, 1)` หรือ `out.squeeze(1)` |
| ไม่มี `eval()` ตอน val/test | Dropout ยังเปิด, BN ใช้สถิติของ batch → ผลเพี้ยน | `model.eval()` |
| ใส่ `model.train()` แค่ก่อน loop ของ epoch แต่ validate ทุก epoch | ตั้งแต่ epoch 2 เป็นต้นไปจะเทรนในโหมด eval *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* | ใส่ `model.train()` ไว้บนสุดของทุก epoch |
| ไม่มี `no_grad()` ใน validation | เปลืองเมม/เวลา | `with torch.no_grad():` |
| มี `loss.backward()` / `step()` ใน validation | model ได้เรียนจากข้อมูล validation ("แอบจำเฉลย") | validation = forward + คำนวณ metric เท่านั้น |
| Augmentation ตอน val/test/inference (`val_tf = train_tf`) | ภาพเดิมแต่ทำนายไม่เหมือนเดิม, metric เชื่อไม่ได้ | แยก `val_tf` ที่ไม่มี random transforms |
| Normalisation ตอน inference ไม่เหมือนตอน train | ผลทำนายพัง | ก๊อป standardisation ของ train มาเป๊ะๆ |
| `ToDtype(float32)` ไม่ใส่ `scale=True` | ค่ายังเป็น 0–255 *(ลองรันแล้ว)* | `scale=True` → [0,1] |
| ขนาด flatten ใน `Linear` ผิด | error `mat1 and mat2 shapes cannot be multiplied` | คำนวณ C×H×W (§3C) |
| Model อยู่บน GPU แต่ data อยู่บน CPU (หรือเปลี่ยน fc หลัง `.to`) | error "expected all tensors on the same device" | `.to(device)` model **หลัง**เปลี่ยน fc และทำกับข้อมูลทุก batch |
| ใช้ `resnet50()` ตอน fine-tune | init แบบสุ่ม ไม่ได้ transfer learning | `weights=ResNet50_Weights.IMAGENET1K_V2` |
| ไม่ได้เปลี่ยน fc | output ยังเป็น 1000 | `model.fc = nn.Linear(2048, n)` |
| `load_state_dict` เข้า model ที่ยังไม่มี fc ตัวใหม่ | error key/shape ไม่ตรง | สร้าง architecture ให้เหมือนเดิมก่อน |
| ไม่ `unsqueeze(0)` ตอนป้อนภาพเดียว | model ต้องการ (N,C,H,W) | `x.unsqueeze(0)` |
| `shuffle=True` ใน val/test | ไม่ได้ทำให้ accuracy ผิด แต่ไม่มีประโยชน์ | `shuffle=False` |
| `train_test_split` แยกกันในแต่ละ model | data leakage ระหว่าง models | แบ่งเองครั้งเดียวแล้วโหลดจากไฟล์ (แบ่งตามคนไข้) |
| พิมพ์ผลทุก batch | ช้าลงมาก | `if i % N == N-1` |
| แปลง Torch ↔ NumPy ใน loop | copy ข้าม CPU↔GPU ช้าลงมาก | ใช้ Torch อย่างเดียว |
