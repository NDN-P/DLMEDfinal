# ข้อสอบจำลอง (Mock Final Exam): Deep Learning in Medicine

**คณะเทคโนโลยีสารสนเทศ สจล. (KMITL)**   เวลา **3 ชั่วโมง**   คะแนนเต็ม **100 คะแนน**

**คำสั่ง**
- ข้อสอบมี **7 ข้อ ต้องทำทุกข้อ** เป็นข้อเขียนทั้งหมด
- ตอบเป็นภาษาไทยได้ ใช้คำศัพท์เทคนิคเป็นภาษาอังกฤษได้
- ข้อที่ให้ "คำนวณ" ต้อง **แสดงวิธีทำ** ข้อที่ให้ "อธิบาย" ต้องตอบ **เหตุผล (why)** ด้วย ไม่ใช่แค่ชื่อ
- อนุญาตให้ใช้เครื่องคิดเลข
- (ข้อสอบจริง: อาจารย์จะอธิบายรูปแบบการเขียนคำตอบใน 5 นาทีแรก ห้ามมาสาย)

| ข้อ | หัวข้อ | คะแนน |
|---|---|---|
| 1 | โปรเจค 1: Segmentation ตัวอักษรไทย (รอข้อมูล) | 15 |
| 2 | Case study: ประยุกต์ใช้ความรู้กับโจทย์การแพทย์ | 15 |
| 3 | Video (Chapter 13) | 14 |
| 4 | Image segmentation (Chapter 12) | 14 |
| 5 | Transposed convolution คำนวณด้วยมือ (Chapter 12) | 14 |
| 6 | Object detection (Chapter 11) | 14 |
| 7 | Coding ด้วย PyTorch (Chapter 10) | 14 |

---

## ข้อ 1. โปรเจค 1: Segmentation ตัวอักษรไทย (15 คะแนน)

> **ยังไม่ได้เขียนข้อนี้** รอข้อมูลโปรเจค 1 (โจทย์, dataset, input/output, โมเดล, ผลลัพธ์) แล้วจะออกข้อสอบจำลองให้

ระหว่างนี้ ฝึกอธิบายโปรเจค 1 ของตัวเองให้ได้: โจทย์คืออะไร, input/output, ทำไปเพื่ออะไร, เตรียมข้อมูลอย่างไร,
ใช้โมเดลอะไรและทำไม, วัดผลด้วยอะไร ได้ผลเท่าไร, ข้อผิดพลาดและข้อจำกัด

เนื้อหาจากคาบเรียนที่เกี่ยวกับงานตัวอักษรไทย (ใช้ได้เลย):
- **Data augmentation** (Chapter 8 `[00:41:04]`): งานรู้จำตัวอักษรไทย **ไม่ต้องปรับสี** (ตัวอักษรไม่ได้แยกกันด้วยสี)
  **ไม่ flip ซ้าย-ขวา** (ภาพจริงแทบไม่มี) flip บน-ล่างเฉพาะถ้าภาพจริงกลับหัวได้ แต่ **หมุนและเอียงควรทำ**
  หลักคือ augment เฉพาะความแปรปรวนที่เกิดจริงและไม่เปลี่ยนความหมายของตัวอักษร
- **Resize** (Chapter 10 P1 `[01:17:20]`–`[01:18:20]`): resize ตรง ๆ ทำให้สัดส่วน (proportion) เพี้ยน
  ตัวอักษรที่ต่างกันที่สัดส่วน (เช่น ตัวที่มีหางยาวกับตัวที่ไม่มี) จะดูคล้ายกันจน classify ผิดเพิ่มขึ้น
  ให้ **pad ด้วย 0 จนเป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize** เพื่อคงสัดส่วน
- Segmentation ตัวอักษรใช้ความรู้ Chapter 12: output channels = จำนวนคลาส, U-Net + skip connection, วัดผลด้วย Dice/IoU (Chapter 14)

---

## ข้อ 2. Case study: ประยุกต์ใช้ความรู้ (15 คะแนน)

โรงพยาบาลแห่งหนึ่งต้องการระบบช่วยอ่าน **ภาพ chest X-ray** (grayscale, ขนาดต่างกัน ประมาณ 2000×2500 px) จำนวน **8,000 ภาพ จากผู้ป่วย 2,500 คน** (ผู้ป่วยหลายคนมีภาพ follow-up หลายภาพ) แต่ละภาพมี label 3 อย่าง: **pneumonia, pleural effusion, cardiomegaly** ซึ่ง **อาจพบพร้อมกันได้ หรือไม่พบเลยก็ได้** โดย pleural effusion พบเพียงประมาณ 6% ของภาพ

**2.1 (4 คะแนน)**
- (ก) งานนี้เป็นงานประเภทใด (multiclass / multilabel / regression / multi-task) ให้เหตุผล
- (ข) ออกแบบ **output layer**: จำนวน output nodes, activation, loss function และวิธีอ่านผลทำนาย
- (ค) ถ้าโรงพยาบาลต้องการให้ระบบ **วัดพื้นที่ของ effusion** ด้วย หรือต้องการ **ประมาณค่า cardiothoracic ratio (ตัวเลขต่อเนื่อง)** งานจะเปลี่ยนเป็นงานประเภทใด และ output layer/loss ต้องเปลี่ยนอย่างไร

**2.2 (4 คะแนน)** เลือก **architecture และ training strategy**:
- จะใช้ backbone อะไร ใช้ **transfer learning** แบบใด (freeze ส่วนไหน train ส่วนไหน) และเพราะอะไร
- ภาพเป็น grayscale 1 channel แต่ pretrained model รับ 3 channels จะทำอย่างไร ภาพมีขนาด/สัดส่วนไม่เท่ากันจะปรับขนาดอย่างไร
- เลือก **data augmentation** อะไรบ้าง ใช้ตอนไหน และ augmentation อะไรที่ **ไม่ควรใช้** กับ chest X-ray (ให้เหตุผล)
- เพื่อนเสนอให้ใช้ Vision Transformer (ViT) แทน CNN นักศึกษาจะตอบอย่างไร

**2.3 (3 คะแนน)** การแบ่งข้อมูล:
- นักศึกษาคนหนึ่งใช้ `train_test_split(images, labels, test_size=0.2)` แบบสุ่มภาพ บอกปัญหาของวิธีนี้ และวิธีที่ถูกต้อง
- ถ้าต้องการเปรียบเทียบโมเดล 3 แบบให้ยุติธรรม ต้องทำอย่างไรกับ split

**2.4 (4 คะแนน)** การประเมินผล: บน test set 1,000 ภาพ ผลของ label **pleural effusion** เป็นดังนี้

| | Predicted effusion | Predicted no effusion |
|---|---|---|
| **Actual effusion** | 45 | 15 |
| **Actual no effusion** | 30 | 910 |

- คำนวณ **Accuracy, Precision, Recall (Sensitivity), Specificity และ F1-score**
- โมเดลที่ทำนาย "no effusion" ทุกภาพจะได้ accuracy เท่าไร สรุปว่าทำไม accuracy จึงหลอกได้
- ถ้าระบบนี้ใช้สำหรับ **screening** ควรให้ความสำคัญกับ metric ใดมากที่สุด เพราะอะไร และจะปรับโมเดล/threshold อย่างไร

---

## ข้อ 3. Video (14 คะแนน)

**3.1 (3 คะแนน)**
- อธิบายความแตกต่างของ **spatial feature** และ **temporal feature** ในวิดีโอ และ operator ที่ใช้สกัดแต่ละแบบ
- นิยามคำว่า **spatio-temporal** แล้วจัดกลุ่มสถาปัตยกรรมต่อไปนี้ว่าเป็น S→T (spatial แล้วค่อย temporal) หรือ ST (พร้อมกัน) พร้อมเหตุผล: (1) CNN ต่อเฟรม → LSTM, (2) ConvLSTM, (3) 3D CNN

**3.2 (4 คะแนน)** คลินิกกายภาพบำบัดมี **วิดีโอการเดิน (gait) ของผู้ป่วย 300 คลิป** คลิปละประมาณ 10 วินาที (ความยาวไม่เท่ากัน) ต้องการจำแนกว่า **การเดินปกติ / ผิดปกติ**
- (ก) เลือกสถาปัตยกรรม 1 แบบจาก CNN→LSTM, ConvLSTM, 3D CNN, Two-stream/I3D พร้อมให้เหตุผล โดยคำนึงถึงขนาด dataset, pretrained model และ resource
- (ข) คลิปมีความยาวไม่เท่ากัน จะจัดการอย่างไร
- (ค) ถ้าเปลี่ยนโจทย์เป็น (1) ระบุ **ขั้นตอนการผ่าตัดทุกเฟรม** ในวิดีโอผ่าตัด, (2) หา **ช่วงเวลาเริ่ม–สิ้นสุด** ที่มีเลือดออก, (3) ประมาณ **อัตราการเต้นของหัวใจ** จากวิดีโอผิวหนัง แต่ละโจทย์เป็นงานประเภทใด (เทียบกับงานภาพ)

**3.3 (4 คะแนน)** LSTM
- อธิบายหน้าที่ของ **cell state, hidden state, forget gate, input gate, output gate** และทำไม gate ใช้ sigmoid แต่ข้อมูลใช้ tanh
- กำหนด (element-wise, เวกเตอร์ 2 มิติ):
  `f_t = [1, 0.5]`, `C_{t-1} = [2, -1]`, `i_t = [0, 0.5]`, `z_t = [0.4, 1]` (input modulate), `o_t = [0.5, 1]`
  **คำนวณ C_t และ h_t** (ทศนิยม 3 ตำแหน่ง) แล้วตีความว่าเกิดอะไรกับ memory ในแต่ละมิติ

**3.4 (3 คะแนน)** Shape และ 3D convolution
- (ก) input วิดีโอ `x` shape `(2, 3, 16, 112, 112)` = (B, C, T, H, W) ผ่าน
  `nn.Conv3d(3, 16, kernel_size=3, stride=(1, 2, 2), padding=1)` แล้วตามด้วย `nn.MaxPool3d(2, 2)`
  output shape หลังแต่ละ layer เป็นเท่าใด และ Conv3d นี้มีกี่ parameters
- (ข) ถ้าข้อมูลมาจาก DataLoader ของ ConvLSTM (`batch_first=True`) จะมี shape แบบใด และต้องทำอะไรก่อนส่งเข้า Conv3d
- (ค) คำนวณ 3D convolution ด้วยมือ (stride 1, no padding): data D มี depth 2 ขนาด 3×3, filter F ขนาด 2×2×2

```
D[0] = | 1 0 2 |    D[1] = | 2 1 0 |        F[0] = | 1  0 |    F[1] = | 0 1 |
       | 3 1 0 |           | 0 1 3 |               | 0 -1 |           | 1 0 |
       | 0 2 1 |           | 1 0 2 |
```
  ระบุขนาด output และค่าทุกตำแหน่ง

---

## ข้อ 4. Image segmentation (14 คะแนน)

**4.1 (3 คะแนน)**
- เปรียบเทียบ classification, object detection และ segmentation ในแง่ **What / Where**
- อธิบาย **semantic, instance และ panoptic segmentation** และยกตัวอย่างงานการแพทย์ที่เหมาะกับแต่ละแบบ 1 ตัวอย่าง

**4.2 (4 คะแนน)** U-Net
- อธิบายหน้าที่ของ encoder (contraction path) และ decoder (expansion path) ในแง่ W×H, channels และ What/Where
- ทำไม U-Net ต้องมี **skip connections** และ skip ของ U-Net ต่างจาก LinkNet อย่างไร
- ใน decoder มี feature จาก up-conv ขนาด 128 channels × 64×64 รวมกับ feature จาก encoder ที่ scale เดียวกัน 128 channels × 64×64 หลัง skip connection ของ **U-Net** และของ **LinkNet** จะได้กี่ channels
- ทำไม layer สุดท้ายของ U-Net เป็น **1×1 convolution**

**4.3 (4 คะแนน)** ออกแบบ output layer (จำนวน output channels ของ 1×1 conv, activation, รูปแบบ label, loss ที่เหมาะสม) สำหรับ:
- (ก) แยก liver tumor ออกจาก background ใน CT
- (ข) แบ่งเนื้อสมองทุก pixel เป็น 4 ชนิด: background, gray matter, white matter, CSF
- (ค) ภาพ histology ที่ต้องหา nucleus, cytoplasm, mitosis และ necrosis โดยหนึ่ง pixel อาจเป็นได้มากกว่าหนึ่งอย่าง
- (ง) ประมาณ **ความหนาของกระจกตา (ตัวเลขต่อเนื่อง) ทุก pixel**

**4.4 (3 คะแนน)** ground truth (GT) และ prediction (Pred) ของ mask ขนาด 6×6 (1 = tumor)

```
GT                  Pred
0 0 0 0 0 0         0 0 0 0 0 0
0 1 1 1 0 0         0 0 1 1 1 0
0 1 1 1 1 0         0 1 1 1 1 0
0 1 1 1 1 0         0 0 1 1 1 1
0 0 1 1 0 0         0 0 1 1 0 0
0 0 0 0 0 0         0 0 0 0 0 0
```
- หา TP, FP, FN, TN แล้วคำนวณ **IoU, Dice และ pixel accuracy**
- ทำไม pixel accuracy จึงไม่เหมาะ และทำไมนิยมใช้ Dice loss กับ tumor segmentation

---

## ข้อ 5. Transposed convolution: คำนวณด้วยมือ (14 คะแนน)

กำหนด input X ขนาด 2×2 และ kernel K ขนาด 3×3 (ไม่ต้องหมุน kernel)

```
X = |  1   2 |        K = | 1 0 1 |
    |  3  -1 |            | 2 1 0 |
                          | 0 1 1 |
```

**5.1 (6 คะแนน)** คำนวณ transposed convolution ด้วย **stride = 2, padding = 0**
- เขียน block ที่ได้จาก X[i,j]·K ทั้ง 4 block และตำแหน่งที่วางแต่ละ block
- รวมค่าที่ทับกัน แล้วเขียน **output matrix ทั้งหมด**
- ตรวจขนาด output ด้วยสูตร

**5.2 (3 คะแนน)** ใช้ X และ K เดิม แต่ **stride = 2, padding = 1** output มีขนาดเท่าใด และมีค่าอะไรบ้าง อธิบายว่า padding ใน transposed convolution ทำอะไรกับ output

**5.3 (3 คะแนน)** เติมขนาด output (กว้าง = สูง) ในตาราง และตอบคำถาม

| # | input | kernel k | stride s | padding p | output |
|---|---|---|---|---|---|
| 1 | 4×4 | 3 | 2 | 0 | ? |
| 2 | 4×4 | 3 | 2 | 1 | ? |
| 3 | 4×4 | 3 | 2 | 2 | ? |
| 4 | 16×16 | 2 | 2 | 0 | ? |
| 5 | 8×8 | 4 | 2 | 1 | ? |
| 6 | 5×5 | 3 | 1 | 1 | ? |

ใน U-Net decoder ต้องการขยาย feature map 32×32 เป็น 64×64 ด้วย `nn.ConvTranspose2d` จะเลือก k, s, p อย่างไร (ตอบอย่างน้อย 2 ชุด)

**5.4 (2 คะแนน)** สไลด์หน้าหนึ่งเขียนว่า "ยิ่งค่า padding มากขึ้น ขนาดของภาพผลลัพธ์จะยิ่งขยายใหญ่ขึ้น" ข้อความนี้ถูกหรือผิด พร้อมเหตุผล และบอกว่า transposed convolution เป็น "inverse ของ convolution" หรือไม่

---

## ข้อ 6. Object detection (14 คะแนน)

**6.1 (3 คะแนน)** ในภาพ chest X-ray ขนาด 512×512 มี ground truth ของ nodule G และ prediction P (รูปแบบ xyxy = x1, y1, x2, y2)

```
G = (40, 40, 100, 100)        P = (60, 50, 120, 110)
```
- คำนวณ **IoU** (แสดงวิธีทำ)
- ถ้าใช้เกณฑ์ IoU ≥ 0.5 ในการประเมิน P นับเป็น TP หรือ FP และ G นับเป็นอะไร
- แปลง G เป็น **YOLO label format** (class 0) และบอกว่าทำไม YOLO ใช้ค่า normalised

**6.2 (4 คะแนน)** Non-Maximum Suppression ระบบตรวจหา polyp ให้ผล 6 boxes (xyxy ทุกกล่องขนาด 50×50)

| Box | (x1, y1, x2, y2) | score |
|---|---|---|
| A | (10, 10, 60, 60) | 0.95 |
| B | (15, 12, 65, 62) | 0.90 |
| C | (100, 100, 150, 150) | 0.85 |
| D | (22, 20, 72, 70) | 0.80 |
| E | (105, 98, 155, 148) | 0.75 |
| F | (12, 8, 62, 58) | 0.60 |

- ทำ NMS ด้วย **IoU threshold = 0.5** แสดงทุกรอบ: กล่อง reference, IoU ที่คำนวณ (เฉพาะคู่ที่ต้องใช้), กล่องที่ถูกลบ
- กล่องใดเหลือเป็นผลสุดท้าย
- คำนวณ IoU(B, D) แล้วอธิบายว่าทำไม D ไม่ถูกลบทั้งที่ทับกับ B มาก
- ถ้าเปลี่ยน threshold เป็น **0.4** ผลเปลี่ยนอย่างไร

**6.3 (4 คะแนน)** R-CNN family
- เปรียบเทียบ R-CNN, Fast R-CNN และ Faster R-CNN ในแง่: หา Where ด้วยอะไร, CNN รันกี่ครั้งต่อภาพ, ปรับขนาด region ด้วยอะไร, classifier คืออะไร
- ทำไม Fast R-CNN ยังเป็น multi-stage และยังไม่ real-time
- Faster R-CNN ใช้ feature map ขนาด **32×32** มี anchor 3 scales × 3 aspect ratios จะมี anchors ทั้งหมดกี่กล่อง ต้องใช้ 1×1 conv กี่ filters สำหรับ objectness score และกี่ filters สำหรับ coordinates

**6.4 (3 คะแนน)** YOLO
- YOLOv1 แบ่งภาพเป็น S = 7, B = 2 boxes ต่อ cell และมี C = 3 classes output tensor มี shape เท่าใด อธิบายความหมายของค่าใน 1 cell
- งาน **real-time polyp detection ใน colonoscopy video** ควรเลือก one-stage หรือ two-stage detector เพราะอะไร และ YOLOv1 มีจุดอ่อนอะไรที่ต้องระวัง

---

## ข้อ 7. Coding: PyTorch (14 คะแนน)

นักศึกษาคนหนึ่งเขียนโค้ด fine-tune **ResNet-50 (ImageNet)** เพื่อจำแนก **ภาพ dermoscopy (RGB) 4 classes** (MEL, NV, BCC, BKL) โดย `train_loader` และ `val_loader` ถูกสร้างไว้แล้ว (label เป็น class index 0–3)

```python
import torch, torch.nn as nn
from torchvision import models
from torchvision.transforms import v2

train_tf = v2.Compose([
    v2.ToImage(), v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(), v2.RandomRotation(10),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
val_tf = train_tf                                         # line 9

model = models.resnet50()                                 # line 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = torch.softmax(model(images), dim=1)     # line 21
        loss = criterion(outputs, labels.float())         # line 22
        optimizer.step()                                  # line 23
        loss.backward()                                   # line 24

    correct = 0
    for images, labels in val_loader:                     # line 27
        outputs = model(images.to(device))
        preds = outputs.argmax(1)
        correct += (preds == labels.to(device)).sum().item()
    print(epoch, correct / len(val_loader.dataset))
```

**7.1 (5 คะแนน)** หา **bugs อย่างน้อย 6 จุด** บอกบรรทัด, ปัญหาที่เกิด และวิธีแก้

**7.2 (4 คะแนน)** เขียน training + validation loop ที่ถูกต้องใหม่ทั้งหมด (รวมการเปลี่ยน classifier head) และ **บันทึกเฉพาะ weights ของ epoch ที่ validation accuracy ดีที่สุด** ด้วย `state_dict`

**7.3 (3 คะแนน)** เขียนฟังก์ชัน `predict(path)` ที่ **โหลด weights ที่บันทึกไว้** แล้วทำนายภาพ 1 ภาพ พิมพ์ผลแบบ `Predicted: MEL 97.31%`

**7.4 (2 คะแนน)**
- ถ้าเปลี่ยนเป็นงาน **multilabel** (ภาพหนึ่งอาจมีหลาย findings) ต้องเปลี่ยนโค้ดส่วนใดบ้าง
- `torch.cuda.is_available()` ตรวจสอบอะไร (นักศึกษาตอบว่า "ตรวจว่าเครื่องมีการ์ดจอหรือไม่" ได้คะแนนเต็มหรือไม่)

---

## แบบฝึกหัดเสริม: Transposed convolution (ไม่คิดคะแนน)

ทุกข้อ: ไม่หมุน kernel, คูณ input แต่ละค่ากับ kernel ทั้งอัน, วางเลื่อนตาม stride, รวมค่าที่ทับกัน, แล้วตัดขอบออกข้างละ p

**ฝึก 1** (stride 1, padding 0)
```
X = | 2 1 |      K = | 1 2 |
    | 0 3 |          | 3 4 |
```
หา output ทั้งหมดและขนาด

**ฝึก 2** (stride 2, padding 1)
```
X = | 1 0 2 |      K = | 0 1 0 |
    | 0 1 0 |          | 1 2 1 |
    | 2 0 1 |          | 0 1 0 |
```
หา output ก่อนตัดขอบ (full) และ output สุดท้าย

**ฝึก 3** (stride 3)
```
X = |  1 -1 |      K = | 1 2 1 |
    |  2  0 |          | 0 1 0 |
                       | 1 0 1 |
```
- (ก) padding 0: หา output ทั้งหมด ทำไมไม่มีค่าใดทับกันเลย
- (ข) padding 1: หา output สุดท้าย
