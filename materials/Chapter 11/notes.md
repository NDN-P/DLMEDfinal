# Chapter 11 — Object Detection ด้วยเทคนิค Deep Learning

Object detection ตอบได้ทั้ง **What** (เป็น class อะไร) และ **Where** (bounding box อยู่ตรงไหน) บทนี้เล่าตระกูล R-CNN แบบ multi-stage (R-CNN → Fast R-CNN → Faster R-CNN ที่มี RPN, anchor และ NMS) ต่อด้วย YOLO แบบ single-stage แล้วปิดท้ายด้วยการเทรน YOLOv8 ด้วย Ultralytics

ผู้สอน: ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์ ("อาจารย์แบงค์"), คณะเทคโนโลยีสารสนเทศ สจล. (KMITL) ไฟล์ต้นทาง: สไลด์ PDF 76 หน้า และคลิปบันทึกการสอนยาวประมาณ 84 นาที

---

## Key concepts

### 1. Classification vs Detection vs Segmentation
- **Classification / regression** บอกแค่ว่าภาพนี้คือ *What*
- **Object detection** กับ **image segmentation** บอกได้ทั้ง *What* และ *Where*
- สองอย่างนี้ต่างกันแค่ตรงส่วน **Where**:
  - Detection: ให้ **สี่เหลี่ยมที่เล็กที่สุดที่ครอบวัตถุได้ทั้งหมด** เรียกว่า *bounding box* อาจารย์เรียกอีกชื่อว่า "Boundary"
  - Segmentation: ให้ **รูปร่างจริง** ของวัตถุ (เส้นขอบเป๊ะ ๆ)
- ตัวอย่างในสไลด์: มีวัตถุ 4 ตัว (แมว 2 ตัว หมา 1 ตัว เป็ด 1 ตัว) แต่ละตัวได้กล่องของตัวเอง

### 2. สองตระกูล (โครงของสไลด์)
- **Multi-stage detectors**: R-CNN, Fast R-CNN, Faster R-CNN ได้ *Where* กับ *What* มาจาก **คนละ module/stage กัน**
- **Single-stage detectors**: YOLO ใช้ **unified network** ตัวเดียว เทรนแบบ **end-to-end** แล้วหา *What* กับ *Where* ได้ **พร้อมกัน**

### 3. R-CNN (Region-based CNN) — Ross Girshick et al., 2014
เป็น detector ตัวแรกที่ใช้ CNN มี 4 ขั้น:
1. **หา region ด้วย Selective Search**
   - Selective Search จับ pixel/region ที่อยู่ติดกันและหน้าตาคล้ายกันมารวมกัน คล้าย ๆ กับการทำ clustering
   - 1.1 เริ่มจาก **graph-based segmentation** (Felzenszwalb et al.) region ที่ได้ออกมาเรียกว่า **Region Proposals** (หรือ *Candidate Regions*)
   - 1.2 เก็บ proposal ทุกอันไว้ใน list แล้วรวมอันที่อยู่ติดกันไปเรื่อย ๆ โดยใช้ความคล้ายจาก **สี, texture, ขนาด และรูปร่าง (fill) ที่เข้ากันได้**:
     `s(ri, rj) = a1·s_colour + a2·s_texture + a3·s_size + a4·s_fill`
     - สไลด์บอกว่า `ai ∈ {0, 1}` แต่ละตัวใช้เปิด/ปิดการวัดแต่ละแบบ
     - อาจารย์บอกว่า weight "ปกติให้ 0.25 เท่ากันทุกตัว"
     - การรวมแต่ละรอบจะค่อย ๆ สร้างเป็นลำดับชั้น คล้ายต้นไม้
   - หยุดเมื่อจำนวน proposal เหลือประมาณ **1,000–2,000** ไม่ใช้จำนวนรอบที่ตายตัว เพราะภาพซับซ้อนกับภาพง่ายต้องรวมกันไม่เท่ากัน
   - แต่ละ region ได้ออกมาเป็น list ของ pixel (x, y) กล่องของมันคือ **(x_min, y_min)–(x_max, y_max)**
2. **Warping**
   - resize ทุก proposal ให้เป็นสี่เหลี่ยมจัตุรัสขนาดเดียวกัน คือ **224×224** ผลที่ได้เรียกว่า **warped region**
   - ทำไมต้องทำ (สไลด์เว้นช่องว่างให้เติม): CNN ยุคนั้น **ยังไม่มี Global Average Pooling หรือ Adaptive Average Pooling** ถ้า input ขนาดไม่เท่ากัน vector ที่ flatten ออกมาก็จะยาวไม่เท่ากัน แต่ FC layer ต้องการความยาวคงที่
   - ข้อเสีย: warping ทำให้ **สัดส่วนของวัตถุ (aspect ratio) เสียไป** เช่น คนขี่ม้าที่สูง ๆ ถูกบีบจนกลายเป็นสี่เหลี่ยมจัตุรัส
3. **Feature extraction**
   - เอา warped region แต่ละอันเข้า CNN (**AlexNet** ที่ **ตัด FC layer ตัวสุดท้ายออก**) ได้ออกมาเป็น **Feature Vector**
4. **Classify ด้วย SVM** ใช้ SVM หนึ่งตัวต่อหนึ่ง class
   - hyperplane คือ `WX + b = 0`
   - ระนาบขอบ margin สองอันคือ H1: `WX + b ≥ +1` (สำหรับ y = +1) และ H2: `WX + b ≤ −1` (สำหรับ y = −1)
   - จุดที่อยู่บน H1 หรือ H2 เรียกว่า **support vectors**
   - margin คือ `M = d1 + d2` ซึ่งเป็นระยะจาก H1 และ H2 ไปถึง hyperplane SVM จะเลือก hyperplane ที่ได้ **M มากที่สุด** (ผลมาตรฐาน: M = 2/‖W‖)
- บทเรียนจากอาจารย์: R-CNN เอา vector ที่ flatten แล้วผ่าน FC layers ก่อนค่อย classify เพราะ vector ดิบ ๆ ที่เพิ่ง flatten ยัง "ไม่พร้อม" จะเอาไป predict
- **จุดอ่อน:** **ช้า** เพราะต้องรัน CNN ประมาณ 2,000 รอบต่อภาพ (รอบละหนึ่ง proposal)

### 4. Fast R-CNN — Ross Girshick et al.
เป้าหมายคือลดเวลาคำนวณ มี **8 ขั้น** ขั้นที่ต่างจาก R-CNN สไลด์ทำเป็นสีแดงไว้:
1. ใส่ **ทั้งภาพ** เข้า CNN feature extractor **ครั้งเดียว** ได้ **Feature Map** หนึ่งอันต่อภาพ ใช้ feature map ตัวสุดท้ายก่อน flatten จะใช้ backbone ตัวไหนก็ได้
2. รัน **Selective Search บนภาพต้นฉบับทั้งภาพ** (ทำคู่ขนานกับขั้นที่ 1) ได้ Region Proposals ออกมา
3. **Map แล้ว crop (RoI projection)**
   - เอา (x, y, w, h) ของแต่ละ proposal ไปวางบน feature map แล้ว crop ตรงนั้นออกมา ชิ้นที่ crop ได้จะมีขนาดไม่เท่ากัน
   - การ map ใช้ **ตำแหน่งและสเกลแบบสัดส่วน**: มองพิกัดเป็นสัดส่วนของภาพ (0 → 0, 1000 px → 1) เลยวางตรงกันได้ทั้งที่ feature map เล็กกว่า
   - นี่คือเหตุผลที่สไลด์บอกว่าทำไม Fast R-CNN ต้องใช้ **ภาพ input ขนาดคงที่**
4. **RoI Pooling** ทำให้ทุกชิ้นที่ crop มามีขนาดคงที่เท่ากัน:
   - 4.1 crop region ออกมาจาก feature map
   - 4.2 แบ่งแต่ละด้านเป็น n ส่วน โดยใช้ `[m/n]` ในวงเล็บ m คือความสูงหรือความกว้าง n คือจำนวนส่วน **ส่วนสุดท้ายรับเศษที่เหลือ**
     - รูปตัวอย่างในสไลด์ปัด **ลง**: region 5×7 ที่ pool เป็น 2×2 แบ่งแถวเป็น 2+3 และแบ่งคอลัมน์เป็น 3+4
     - อาจารย์บอกว่า "ปัดขึ้น" (8/3 → 3, 3, 2) ดูในหัวข้อ Formulas ด้านล่าง
   - 4.3 เอาค่า **max** ของแต่ละส่วน (max pooling **ไม่ใช่ average**)
5. **Flatten** ให้เป็น feature vector
6. ผ่าน **Fully Connected** layers
7. Classify ด้วย **Softmax** นี่คือกิ่ง *classifier* (*What*)
8. ปรับกล่องด้วย **Linear Regression** บน x, y, w, h นี่คือกิ่ง *bounding-box regressor*
   - regressor แค่ **ปรับ** *Where* ที่ Selective Search หามาแล้ว ไม่ได้สร้าง Where ขึ้นมาเอง: "ขยับยุกยิกนิดนึง ยืดหดนิดนึง" (แค่ขยับและยืดหดกล่องนิดหน่อย)
- **ทำไมเร็วกว่า:** รัน CNN **ครั้งเดียวต่อภาพ** ไม่ใช่ครั้งเดียวต่อ proposal
- **ผลลัพธ์** (ตารางสไลด์ 27):
  - Training: 84 h → 9.5 h เร็วขึ้น **8.8×**
  - Test time ต่อภาพ ไม่นับ proposal: 47 s → 0.32 s เร็วขึ้น **146×**
  - Test time ต่อภาพ **รวม Selective Search**: 50 s → **2 s** เร็วขึ้นแค่ **25×** ตอนนี้ Selective Search กลายเป็นคอขวด
- **ยังเป็น multi-stage อยู่** เพราะยังใช้ Selective Search และที่ 0.32 s ก็ยัง **ไม่ใช่ real-time**

### 5. Faster R-CNN — Shaoqing Ren et al., 2015
- แก้ปัญหา **Selective Search** ที่ช้าและกินทรัพยากร โดยเปลี่ยนไปใช้ **Region Proposal Network (RPN)** แทน ซึ่ง RPN เทรนได้
- ส่วนอื่นเหมือน Fast R-CNN หมด: feature map ทั้งภาพ → proposals → map แล้ว crop → RoI pooling → FC → softmax + box regressor
- **RPN มี 6 ขั้น:**
  - **2.1 Intermediate layer** เป็น conv 3×3 มี padding และ **512 filters** (จำนวน channel เข้าและออกเท่าเดิม)
    - มุมมองของอาจารย์: feature map จาก backbone เป็นแบบ "ทั่วไป" เหมาะกับ *What* ส่วน intermediate layer เปลี่ยนให้เป็น feature ที่เหมาะกับ *Where*
  - **2.2 Anchor box generation** anchor คือกล่องที่ **กำหนดไว้ล่วงหน้า** ใช้งานคล้าย sliding window กำหนดด้วยสองค่า: **Scale** และ **Aspect Ratio** ตั้งอยู่บนสองความเชื่อ:
    1. **ทุกจุดบน feature map อาจเป็นจุดศูนย์กลางของวัตถุได้**
    2. วัตถุตรงนั้นอาจมีได้ **3 scales × 3 aspect ratios = 9 anchor boxes** ต่อหนึ่งตำแหน่ง (จัตุรัส, สูง, กว้าง)
  - **2.3 Classification layer (score)** ตัดสินว่า anchor แต่ละอันเป็น **foreground หรือ background** และให้ **objectness score** (คะแนนว่ามีวัตถุไหม)
  - **2.4 Regression layer (พิกัด)** ปรับ [x, y, w, h] ของแต่ละ anchor
  - **2.5 Candidate selection** เก็บไว้เฉพาะ anchor ที่ทายว่าเป็น foreground เรียงตาม objectness score จากมากไปน้อย แล้วเก็บ **top n** อาจารย์บอกว่าปกติคือ top 2,000 (หรือ 6,000)
  - **2.6 Non-Maximum Suppression (NMS)** ลบ proposal ที่ซ้ำกันในพื้นที่เดียวกัน เก็บไว้แค่อันที่น่าจะใช่ที่สุด
- **วิธี implement** (สไลด์ 35) ให้ a = จำนวนแบบของ anchor:
  - conv 3×3 (c filters) บน feature map ตัวสุดท้าย → w×h×c
  - กิ่ง score: **conv 1×1 จำนวน a filters** → w×h×a
  - กิ่งพิกัด: **conv 1×1 จำนวน 4·a filters** → w×h×4a
  - **"ตั้งชื่อตามการใช้งาน":** conv 1×1 ไม่ได้ "ดึง score ออกมา" ได้เองโดยธรรมชาติ แต่เราเอา output ไป *ใช้* เป็น score หรือ offset แล้ว loss ก็เทรนให้มันทำหน้าที่นั้น เราเลยตั้งชื่อตามที่ใช้ อาจารย์เรียกจุดนี้ว่า "ตัวเปลี่ยนโลก" (the world-changer): **YOLO เกิดมาจากไอเดียนี้**
- **output ของ regressor คือค่าที่ใช้ปรับ ไม่ใช่พิกัดใหม่:** offset ของ x และ y เอาไป **บวก** ส่วน w และ h เอาไป **คูณ** เป็นอัตราส่วน
- **ผลลัพธ์** (สไลด์ 51, test time ต่อภาพรวม proposal):

| | R-CNN | Fast R-CNN | Faster R-CNN |
|---|---|---|---|
| เวลา | 50 s | 2 s | **0.2 s** |
| เร็วขึ้นเทียบกับ R-CNN | 1× | 25× | **250×** |
| mAP (PASCAL VOC 07) | 66.0 | 66.9 | 66.9 |

  อาจารย์บอกว่า Faster R-CNN แม่นกว่าด้วย แต่ในตารางนี้ mAP เท่ากับ Fast R-CNN (66.9) สิ่งที่ได้เพิ่มคือ **ความเร็ว**

### 6. YOLO (You Only Look Once) — single-stage
- **ประวัติ:** เปิดตัวโดย **Joseph Redmon et al. (University of Washington), June 2015** ตาม timeline ในสไลด์ 64 อาจารย์เองก็ไม่แน่ใจเรื่องที่มา ส่วน YOLOv8 เป็นของ Glenn Jocher et al. ที่ **Ultralytics, January 2023**
- **Unified network เทรนแบบ end-to-end**: ทาย *What* และ *Where* ไปพร้อมกัน ไม่มี module หา proposal แยกต่างหาก
- **ขั้นตอน:**
  1. แบ่งภาพเป็น **grid S×S** **grid cell ที่มีจุดศูนย์กลางของวัตถุอยู่** จะรับผิดชอบวัตถุนั้น
  2. แต่ละ cell ทาย **B bounding boxes** (YOLOv1 ใช้ **B = 2**) แต่ละกล่องมี **cx, cy, w, h และ confidence** (objectness) และมี **class probability หนึ่งชุดต่อ cell**: P(class | object)
- **ความสัมพันธ์ object–cell–box–class** (สไลด์ 55):
  - Object → cell: cell ที่มีจุดศูนย์กลางของวัตถุ
  - Object → box: ใน B กล่องของ cell นั้น กล่องที่ **IoU กับ ground truth สูงที่สุด** "คือ" วัตถุนั้น
  - Cell → class: หนึ่ง cell มีได้แค่ **class เดียว** YOLOv1 จึงทำได้ไม่ดีเวลามีวัตถุหลายตัวต่าง class อยู่ใน cell เดียวกัน และกับ **วัตถุเล็ก ๆ จำนวนมากที่อยู่รวมกันเป็นกลุ่ม** (เพราะ cell ใหญ่)
- **การเข้ารหัสกล่อง:**
  - w และ h ถูก normalise ด้วยความกว้างภาพ W และความสูงภาพ H
  - x และ y เป็น **offset ภายใน cell**
  - ค่าทั้งสี่จึงอยู่ในช่วง 0–1
- **การให้คะแนน:** class score ต่อกล่อง = P(class | object) × box confidence จากนั้นใช้ **NMS** ได้ผล detection สุดท้าย NMS ช่วยเพิ่ม mAP ได้ประมาณ **2–3%**
- **ไอเดียการ implement** (อาจารย์): backbone (AlexNet, ResNet, DenseNet, EfficientNet, DarkNet, ...) แล้วตามด้วย filter 1×1
  - กล่องหนึ่งต้องใช้ 5 ค่า ดังนั้น 2 กล่องต้องใช้ **10 filters** สำหรับ *Where*
  - เพิ่ม **หนึ่ง filter ต่อหนึ่ง class** สำหรับ *What*: 10 class ก็เพิ่มอีก 10 filters
- **YOLO vs Faster R-CNN ความเชื่อเดียวกันแต่อยู่คนละ domain:**
  - Faster R-CNN: ทุก **pixel บน feature map** เป็นจุดศูนย์กลางของวัตถุ
  - YOLO: ทุก **grid cell บนภาพต้นฉบับ** มีจุดศูนย์กลางของวัตถุ
  - feature map หนึ่ง cell ถ้าฉายกลับไปบนภาพจะครอบหลาย pixel ซึ่งก็ประมาณหนึ่ง grid cell
- **สถาปัตยกรรม YOLOv1:** **24 convolutional layers + 2 fully connected layers**
- **output tensor ของ YOLOv1** (สไลด์ 59): **7×7×30** โดย S = 7, B = 2, C = 20 ต่อหนึ่ง cell:
  - Channels 1–5: Box #1 (confidence, x, y, width, height)
  - Channels 6–10: Box #2 (ห้าค่าเหมือนกัน)
  - Channels 11–30: class probabilities P(Cat | Object), ..., P(TV | Object)
- **Loss** มี 5 ส่วน ทั้งหมดเป็น sum-squared error (SSE):
  1. error ของตำแหน่ง (x, y)
  2. error ของ (w, h) โดยใช้ **รากที่สอง** ของ w และ h เพื่อลดความต่างระหว่างกล่องใหญ่กับกล่องเล็ก
  3. Confidence ตอนที่มีวัตถุ (อิงจาก IoU)
  4. Confidence ตอนที่ไม่มีวัตถุ ถ่วงด้วย **λ_noobj = 0.5**
  5. Class probabilities
  - **λ_coord = 5** ใช้เพิ่มน้ำหนัก localisation loss
  - สูตรเต็มจากสไลด์ 62:
    `λcoord Σi Σj 1ij_obj [(xi−x̂i)² + (yi−ŷi)²] + λcoord Σi Σj 1ij_obj [(√wi−√ŵi)² + (√hi−√ĥi)²] + Σi Σj 1ij_obj (Ci−Ĉi)² + λnoobj Σi Σj 1ij_noobj (Ci−Ĉi)² + Σi 1i_obj Σc (pi(c)−p̂i(c))²`
    - i วิ่งไปตาม S² cells และ j วิ่งไปตาม B boxes
    - `1ij_obj` เป็น 1 เมื่อกล่อง j ใน cell i มีวัตถุ ไม่งั้นเป็น 0 ส่วน `1ij_noobj` ตรงกันข้าม
    - เทอม class นับเฉพาะ cell ที่มีวัตถุ
  - ประเด็นของอาจารย์: loss ไม่ได้ครอบแค่การทาย class แต่ครอบ objectness และ confidence ด้วย
- **เวอร์ชันต่าง ๆ:** YOLO พัฒนาโดยเอาจุดเด่นของโมเดลก่อน ๆ มารวมกัน ไม่ได้เพิ่ม operator ใหม่ YOLOv8 สร้างจาก conv2d, max pooling, concat, batch norm, linear, SiLU รวมกับ pyramid และ upsampling ซึ่งจะได้เรียนในบท segmentation
- **License:** YOLO เวอร์ชันใหม่ ๆ มี license กำกับ ใช้เรียนหรือทำวิจัยได้ แต่ถ้าเอาไปใช้เชิงพาณิชย์อาจเจอปัญหา ("โดนคาบตับ")

### 7. เทียบตระกูล detector

| | R-CNN (2014) | Fast R-CNN | Faster R-CNN (2015) | YOLO (v1) |
|---|---|---|---|---|
| ประเภท | Multi-stage | Multi-stage | Multi-stage (two-stage) | **Single-stage**, unified, end-to-end |
| Where (proposals) | Selective Search บนภาพต้นฉบับ | Selective Search บนภาพต้นฉบับ | **RPN** (anchors + objectness + NMS) | grid cell ทายกล่องออกมาตรง ๆ |
| จำนวนครั้งที่รัน CNN ต่อภาพ | ~2,000 (หนึ่งครั้งต่อ warped region) | **1** (ทั้งภาพ) | 1 | 1 |
| วางพิกัด proposal บน | ภาพต้นฉบับ | **Feature map** (RoI projection) | Feature map | n/a |
| ทำให้ region ขนาดเท่ากัน | **Warping** เป็น 224×224 (สัดส่วนเสีย) | **RoI Pooling** (max) | RoI Pooling | n/a |
| Classifier (What) | **SVM** (CNN = AlexNet ตัด FC สุดท้ายออก) | FC + **Softmax** | FC + Softmax | filter 1×1, class probability ต่อ cell |
| ปรับกล่อง | ไม่มี (ตามสไลด์/อาจารย์) | Linear regressor (x, y, w, h) | Regressor ใน RPN + head | Box regression อยู่ใน loss |
| Test time รวม proposal | 50 s/img | 2 s (25×) | 0.2 s (250×) | Real-time |
| mAP (PASCAL VOC 07) | 66.0 | 66.9 | 66.9 | — |
| จุดอ่อนหลัก | ช้า; overhead จากการรัน CNN ~2,000 รอบ | Selective Search ยังช้า; ไม่ real-time | ยังเป็นสอง stage | 1 class ต่อ cell; วัตถุเล็กที่อยู่กันเป็นกลุ่ม |

> บทนี้ไม่ได้สอน SSD ตัวอย่าง single-stage มีแค่ YOLO

### 8. Ultralytics YOLOv8 (ลงมือทำ)
- เป็น open source ของ Ultralytics Inc. ทำได้ทั้ง **detection, instance segmentation, classification และ pose estimation**
- **ติดตั้ง:** `pip install ultralytics` ต้องใช้ **Python ≥ 3.8** และ **PyTorch ≥ 1.8**
- **CLI:** `yolo predict model=yolov8n.pt source='...bus.jpg'`
- **เทรนกับข้อมูลของเราเองมี 6 ขั้น:**
  1. **dataset.yaml** มี key `train`, `val`, `test` (path ของภาพ แนะนำให้ใช้ absolute path), `nc` (จำนวน class) และ `names` (list ชื่อ class)
  2. **label แบบ YOLO format:** ไฟล์ `.txt` หนึ่งไฟล์ต่อหนึ่งภาพ **หนึ่งแถวต่อหนึ่งวัตถุ**: `class x_center y_center width height`
     - ค่าถูก normalise ให้อยู่ใน 0–1 มุมซ้ายบนของภาพคือ (0, 0) มุมขวาล่างคือ (1, 1)
     - ตัวอย่าง: class 0 คือ person, class 27 คือ tie (COCO)
  3. **โครงสร้างโฟลเดอร์:** หา path ของ label ได้อัตโนมัติโดยเปลี่ยน `/images/` เป็น `/labels/` เช่น `./datasets/train/images/im0.jpg` → `./datasets/train/labels/im0.txt` ใช้ **ชื่อไฟล์เดียวกัน แค่นามสกุลต่างกัน**
  4. **เลือกโมเดล:** มี 5 ขนาด **n / s / m / l / x** pretrained บน COCO (80 classes) โมเดลใหญ่กินทรัพยากรมากขึ้นและแม่นขึ้น ตารางสไลด์ 72 ทั้งหมดที่ 640 px:

| Model | mAP (val, IoU 50–95) | Params (M) | FLOPs (B) | A100 TensorRT (ms) |
|---|---|---|---|---|
| YOLOv8n | 37.3 | 3.2 | 8.7 | 0.99 |
| YOLOv8s | 44.9 | 11.2 | 28.6 | 1.20 |
| YOLOv8m | 50.2 | 25.9 | 78.9 | 1.83 |
| YOLOv8l | 52.9 | 43.7 | 165.2 | 2.39 |
| YOLOv8x | 53.9 | 68.2 | 257.8 | 3.53 |

     - n เหมาะกับมือถือและอุปกรณ์ embedded ส่วน s เป็นตัวเลือกกลาง ๆ แบบ "มาตรฐาน"
  5. **Train:** `YOLO("yolov8n.pt").train(data=..., epochs=5, batch=16, imgsz=256)` `batch` คือจำนวนภาพที่ส่งเข้าไปพร้อมกันในหนึ่งครั้ง
  6. **Validate:** `model.val(data=..., imgsz=256, batch=16, conf=0.25, iou=0.5)`
     - `conf` คือ confidence threshold ใช้ตัดวัตถุที่ confidence ต่ำทิ้ง ค่า default คือ 0.001 ซึ่งสไลด์บอกว่าไม่แนะนำ
     - `iou` คือ **threshold การซ้อนทับของ NMS** ค่า default คือ **0.6**
     - weight ที่ดีที่สุดถูกเซฟไว้ที่ `runs/detect/train/weights/best.pt`
- **Inference:** `results = model(img)` คืนค่ามาเป็น **list** เพราะ library รองรับทั้ง output แบบ detection และ segmentation เรียก `result[0].save(path)` เพื่อเซฟเฉพาะภาพผล detection

---

## สูตรและตัวเลขที่ต้องจำ

### รูปแบบของ Bounding-box
อาจารย์: "ในตลาดมีอยู่ 3 format" ต้องเช็กทุกครั้งว่า library หรือโมเดลที่ใช้ต้องการแบบไหน

| Format | ค่า | หมายเหตุ |
|---|---|---|
| xyxy | (x1, y1, x2, y2) = (x_min, y_min, x_max, y_max) | มุมซ้ายบนและมุมขวาล่าง |
| xywh (corner) | (x, y, w, h) โดย (x, y) คือมุมซ้ายบน | |
| cxcywh (centre) | (x_center, y_center, w, h) | |
| **YOLO normalised** | `class xc/W yc/H w/W h/H` แต่ละค่าอยู่ใน 0–1 | resize ภาพแล้ว label **ไม่** เปลี่ยน |

**การแปลง:**
- `w = x2 − x1`, `h = y2 − y1`
- `xc = x1 + w/2`, `yc = y1 + h/2`

**ตัวอย่างคำนวณ (สมมติขึ้นเอง)** ภาพ 640×480 กล่อง xyxy = (100, 50, 300, 250):
- xywh = (100, 50, 200, 200)
- centre = (200, 150, 200, 200)
- YOLO = (200/640, 150/480, 200/640, 200/480) = **(0.3125, 0.3125, 0.3125, 0.4167)**

### IoU (Intersection over Union)
สไลด์เขียนว่า "Intersection of Union"

`IoU = |R1 ∩ R2| / |R1 ∪ R2| = intersection / (area1 + area2 − intersection)`

สำหรับกล่องแบบ xyxy:
- `ix1 = max(x1a, x1b)`, `iy1 = max(y1a, y1b)`, `ix2 = min(x2a, x2b)`, `iy2 = min(y2a, y2b)`
- `inter = max(0, ix2 − ix1) · max(0, iy2 − iy1)`

**ตัวอย่างคำนวณ** A = (0, 0, 4, 4), B = (2, 2, 6, 6):
- Intersection = 2×2 = 4
- Union = 16 + 16 − 4 = 28
- **IoU = 4/28 ≈ 0.143**

กรณีขอบ: กล่องที่ไม่ซ้อนกันเลยมี IoU = 0 (ตัวอย่างในสไลด์: 0.91 vs 0.89) กล่องที่เหมือนกันเป๊ะมี IoU = 1

### NMS (Non-Maximum Suppression) ขั้นตอนในสไลด์ 2.6.1–2.6.4
1. เรียงกล่องตาม score (objectness หรือ confidence) จากมากไปน้อย แล้วเก็บไว้ใน list
2. หยิบกล่องบนสุดมา คำนวณ IoU กับกล่องที่เหลือทีละกล่อง
3. ถ้า IoU > threshold (ตัวอย่างของอาจารย์ใช้ 0.5; ค่า default ของ `iou` ใน Ultralytics คือ 0.6) แปลว่าครอบวัตถุเดียวกัน: **เก็บอันที่ score สูงกว่า ลบอันที่ต่ำกว่าทิ้ง**
4. ขยับไปกล่องถัดไปที่ยังรอดอยู่ แล้วทำซ้ำจนเทียบครบทุกกล่อง

**ตัวอย่างในสไลด์** (สไลด์ 42–49 คนกับหมา):
- คนมีกล่อง score **0.91, 0.88 และ 0.63** หมามีกล่อง score **0.89, 0.77 และ 0.70**
- list ที่เรียงแล้ว: 0.91, 0.89, 0.88, 0.77, 0.70, 0.63
- ตัวอ้างอิง 0.91: เทียบกับ 0.89 ได้ IoU = 0 (คนละวัตถุ) เลยเก็บไว้ เทียบกับ 0.88 IoU สูง เลยลบทิ้ง เทียบกับ 0.63 ซ้อนกัน เลยลบทิ้ง
- ตัวอ้างอิง 0.89: ลบ 0.77 และ 0.70 ทิ้ง
- ผลสุดท้าย: **0.91 (คน) และ 0.89 (หมา)**

### ขนาดช่องของ RoI Pooling
แบ่งแต่ละด้านเป็นส่วนขนาด `[m/n]` ส่วนสุดท้ายรับเศษที่เหลือเสมอ สองแหล่งปัดเศษไม่เหมือนกัน:
- **รูปในสไลด์ 24 (ปัดลง):**
  - feature map 8×8 มี region ขนาด **5 แถว × 7 คอลัมน์** pool เป็น 2×2
  - แถว: 5/2 → 2 แล้วตามด้วย 3 (เศษ) คอลัมน์: 7/2 → 3 แล้วตามด้วย 4 (เศษ)
  - ค่า max ของแต่ละบล็อกคือ **0.85, 0.84, 0.97, 0.96** flatten ได้ `[0.85, 0.84, 0.97, 0.96]`
  - ตรงกับที่อาจารย์อธิบายรอบแรก [00:36:41–00:37:09]: "7/2 → 3 and 4; 5/2 = 2.5 → 2, remainder last"
- **อาจารย์แก้คำพูด (ปัดขึ้น)** [00:37:32]: 8/3 → 3, 3, 2
- ในข้อสอบ ให้ทำตามตัวอย่างในสไลด์ และเขียนบอกด้วยว่าใช้กฎปัดเศษแบบไหน การ pool ใช้ **max** เสมอ

### จำนวน anchor และ filter (RPN)
- anchor ต่อตำแหน่ง: k = scales × ratios = 3 × 3 = **9**
- anchor ทั้งหมด = H_fm × W_fm × k
  - map 10×10: 100 × 9 = **900**
  - map 50×50: 2,500 × 9 = **22,500** อาจารย์พูดว่า "~30,000 ประมาณนั้น" แต่ตัวเลขจริงคือ 22,500
- filter 1×1 สำหรับ anchor k แบบ: **k สำหรับ score** และ **4k สำหรับพิกัด** ถ้ามี 9 anchors ก็คือ 9 และ 36 คิดต่อ anchor คือ 5 ค่า (1 score + x, y, w, h)
- Intermediate layer: conv 3×3 จำนวน **512** filters
- proposal ที่เก็บไว้: top **2,000** (หรือ 6,000) ตาม objectness แล้วค่อยทำ NMS
- (ขยายความ: paper ต้นฉบับของ Faster R-CNN ใช้ score output 2k ตัวสำหรับ softmax foreground/background แต่สไลด์นี้ใช้ k)

### ตัวเลขของ YOLO
- output ต่อ cell = **B×5 + C**
  - ตัวอย่างของอาจารย์: B = 2 และ C = 10 → 10 + 10 = **20 filters**
  - สไลด์ 59 (YOLOv1): S = 7, B = 2, C = 20 → **7×7×30**
- B = 2; สถาปัตยกรรมคือ 24 conv + 2 FC; λ_coord = 5; λ_noobj = 0.5; loss ใช้ √w และ √h
- NMS เพิ่ม mAP ได้ประมาณ 2–3%

### ตัวเลขความเร็ว

| Model | Training | Test time (ไม่รวม proposal) | Test time รวม proposal | mAP VOC07 |
|---|---|---|---|---|
| R-CNN | 84 h | 47 s | 50 s | 66.0 |
| Fast R-CNN | 9.5 h (8.8×) | 0.32 s (146×) | 2 s (25×) | 66.9 |
| Faster R-CNN | — | — | 0.2 s (250×) | 66.9 |

### Precision / Recall / AP / mAP
*เป็นแค่ความรู้พื้นฐาน สไลด์รายงานตัวเลข mAP (PASCAL VOC 07 mAP และ "mAP val 50–95" ของ YOLOv8) แต่ไม่เคยนิยาม mAP*
- detection หนึ่งอันจะเป็น **True Positive** ถ้า IoU กับกล่อง ground truth ที่ยังไม่ถูกจับคู่ ≥ threshold (เช่น 0.5) และ class ถูก ไม่งั้นเป็น **False Positive** กล่อง ground truth ที่หาไม่เจอเป็น **False Negative**
- Precision = TP / (TP + FP). Recall = TP / (TP + FN).
- **AP** คือพื้นที่ใต้กราฟ precision–recall ของหนึ่ง class โดยเรียง detection ตาม confidence
- **mAP** คือค่าเฉลี่ย AP ของทุก class mAP@0.5 ใช้ IoU threshold 0.5; mAP@0.5:0.95 เฉลี่ยข้าม IoU threshold ตั้งแต่ 0.5 ถึง 0.95

---

## สิ่งที่อาจารย์เน้น

- **[00:03:07]–[00:05:18]** Detection กับ segmentation ต่างกันที่ **Where ไม่ใช่ What**: "ไม่ได้ต่างกันที่ what ต่างกันที่ where" (they don't differ in what, they differ in where) กล่องของ detection คือ "สี่เหลี่ยมที่เล็กที่สุดที่ครอบวัตถุ" ส่วน segmentation ให้รูปร่างจริง
- **[00:13:06]–[00:14:30]** Selective Search หยุดเมื่อได้ **จำนวน proposal ตามเป้า (1,000–2,000)** ไม่ได้หยุดตามจำนวนรอบที่ตายตัว เพราะความซับซ้อนของภาพและขนาดวัตถุต่างกันมากเกินกว่าจะใช้จำนวนรอบคงที่
- **[00:15:28]** เปลี่ยน region เป็นกล่องโดยใช้ **min/max ของ x และ y**
- **[00:17:30]–[00:18:27]** ข้อเติมคำในช่องว่างในสไลด์ 9: **ทำไมต้อง warping**
  - Q: "ที่เราต้องทำ Wrapping เพราะอะไร... จุดในช่องว่างนั้นคืออะไร" (why do we need warping? what goes in the blank?)
  - **A:** CNN ยุคนั้น **ไม่มี Global/Adaptive Average Pooling** input ขนาดต่างกันเลยได้ vector ที่ flatten แล้วยาวไม่เท่ากัน
  - จำข้อเสียไว้ด้วย: warping ทำให้ **สัดส่วนเสีย**
- **[00:20:30]–[00:21:02]** คำถามในห้อง: "การได้ What กับ Where มันได้มาพร้อมกันป่ะ" (are What and Where obtained at the same time?) **A: ไม่** R-CNN ได้ Where จาก proposal ก่อน แล้วค่อยได้ What จาก CNN นี่คือเหตุผลที่มันเป็น **multi-stage** ส่วน "end-to-end" หมายถึง **single-stage**
- **[00:22:37]** "จดไปครับ" (write this down): R-CNN และ Fast R-CNN **ยังใช้ Selective Search ทั้งคู่** Fast R-CNN จึง **ยังเป็น multi-stage**
- **[00:23:01]** "นี่คือ Pain Point" (this is the pain point): R-CNN ดึง feature 1,000–2,000 ครั้งต่อภาพ ตรงนี้คือ overhead
- **[00:24:29]** "จดไป" (write this down): R-CNN ดึง feature จาก **ภาพย่อยที่ warp แล้ว** ส่วน Fast R-CNN ดึง **ครั้งเดียวจากทั้งภาพ**
- **[00:26:12]** "จดไปครับ" (write this down): พิกัดของ proposal วางบน **ภาพต้นฉบับ** (R-CNN) เทียบกับวางบน **feature map** (Fast R-CNN)
- **[00:26:51]–[00:27:28]** คำถาม: อะไรใหญ่กว่ากัน feature map หรือภาพต้นฉบับ? **A: ภาพต้นฉบับ** แล้วจะวางกล่องลงบน feature map ได้ยังไง? คำตอบมาที่ **[00:35:01]**: มองพิกัดเป็น **เปอร์เซ็นต์ (0–1)** ของภาพ แล้ว map ตามสัดส่วน
- **[00:28:20]–[00:29:02]** "จดไปครับ" (write this down): การทำให้ขนาดเท่ากันคือ **Warping บนภาพต้นฉบับ** (R-CNN) เทียบกับ **RoI Pooling บน feature map** (Fast R-CNN)
- **[00:29:36]–[00:29:40]** "จดใบครับ" (write this down): การ predict ใช้ **SVM** (R-CNN) เทียบกับ **neural network + Softmax** (Fast R-CNN)
- **[00:30:31]–[00:31:36]** bounding-box regressor "เอาไว้ปรับ Where" (there to *adjust* Where) ไม่ได้สร้าง Where ขึ้นมา คำถาม: R-CNN ปรับ Where ไหม? **A: ไม่** Fast R-CNN ปรับ แปลว่า R-CNN ทั้ง *ช้า และ* ไม่แม่น
- **[00:31:54]–[00:32:20]** ตัวเลขความเร็ว: training 84 h → 9.5 h; inference 47 s → 0.32 s (×146)
- **[00:36:41]–[00:37:49]** การแบ่งช่องของ RoI pooling
  - อาจารย์พูดตอนแรกว่าปัดลงและให้เศษไปอยู่ส่วนสุดท้าย (7/2 → 3, 4; 5/2 → 2, 3) ตรงกับรูปในสไลด์ 24
  - แล้วก็พูดว่า "ผมพูดผิด มันต้องปัดขึ้น" (I misspoke, it should round up): 8/3 → 3, 3, 2
  - ไม่ว่าแบบไหน ใช้ **max ไม่ใช่ average**: "สูงสุดนะ ไม่ใช่เฉลี่ยนะ"
- **[00:40:27]–[00:41:22]** 0.32 s ก็ยังไม่ real-time คอขวดที่เหลือคือ **Selective Search** และผู้เขียนก็อยากให้ขั้นหา proposal **เทรนได้** ด้วย นำไปสู่ Faster R-CNN ซึ่งเป็น "สารตั้งต้น" (precursor) ของตระกูล YOLO
- **[00:43:39]** RPN เข้าใจยากเพราะ **mindset ไม่ใช่เพราะ operator** นักศึกษาในที่ปรึกษาที่เก่งที่สุดของอาจารย์ยังใช้เวลาประมาณ 1.5 เดือนกว่าจะเข้าใจ
- **[00:44:31]–[00:45:18]** intermediate layer เปลี่ยน feature ที่เหมาะกับ *What* ให้เป็น feature ที่เหมาะกับ *Where*
- **[00:47:04]** "ท่อนที่งงที่สุดของวันนี้" (the most confusing part today): **anchor box generation** ตั้งอยู่บนสองความเชื่อ: (1) ทุกจุดบน feature map เป็นจุดศูนย์กลางของวัตถุ และ (2) 3 ขนาด × 3 อัตราส่วน
- **[00:49:39]–[00:51:16]** คำนวณในห้อง: feature map 10×10 มี 9 anchors ต่อจุด ได้กล่องผู้สมัคร **900** กล่อง การสร้างกล่องพวกนี้ยังไม่มีการคำนวณอะไร แค่ "กำหนดไว้ล่วงหน้า"
- **[00:57:02]** "จดไปครับ อันนี้คือตัวเปลี่ยนโลก" (write this down, this is the world-changer): output ของ conv 1×1 **มีความหมายตามที่เราเอาไปใช้** แล้ว loss ก็เทรนให้มันทำหน้าที่นั้น **YOLO มาจากไอเดียนี้**
- **[01:02:10]** "อันนี้คือสิ่งที่อยากเน้น" (this is what I want to stress): feature ถูกตั้งชื่อตามบริบทที่เอาไปใช้ ต่างจาก edge filter ที่เรารู้หน้าที่อยู่แล้ว
- **[01:02:20]–[01:02:43]** quiz ในห้อง: ถ้ามี **anchor 9 แบบ** ต้องใช้ filter 1×1 กี่ตัว? **A: 9 สำหรับ score และ 36 (9×4) สำหรับพิกัด**
- **[01:03:00]–[01:03:52]** feature map จริงประมาณ 50×50 ได้ anchor 2,500 × 9 ซึ่งเยอะเกินไป เรียงตาม objectness แล้วเก็บ **top 2,000 (หรือ 6,000)**
- **[01:04:38]–[01:06:39]** ไล่ NMS ด้วย proposal 6 อันบนคนกับหมา:
  - เรียงตาม score ใช้กล่องบนสุดเป็นตัวอ้างอิง แล้วเทียบ IoU กับที่เหลือ
  - ถ้า IoU > 0.5 ลบกล่องที่ score ต่ำกว่าทิ้ง
  - แล้วขยับตัวอ้างอิงไปกล่องถัดไปที่ยังรอด
- **[01:08:16]** "จำไว้นะ" (remember this): **output ของ regressor คือค่าที่ใช้ปรับ ไม่ใช่ x และ y ใหม่** offset ของ x และ y เอาไป **บวก** ส่วน w และ h เอาไป **คูณ** เป็นอัตราส่วน
- **[01:08:46]** Faster R-CNN: 2 s → 0.2 s **เร็วกว่า R-CNN 250×** อาจารย์บอกว่าแม่นกว่าด้วย แต่สไลด์ 51 แสดง mAP เท่ากับ Fast R-CNN (66.9)
- **[01:09:54]–[01:11:03]** Single-stage กับ multi-stage แบบถาม-ตอบ:
  - R-CNN ใช้อะไรหา Where? **Selective Search** หา What? **CNN + SVM**
  - Faster R-CNN ใช้อะไรหา Where? **RPN** หา What? **Neural network**
  - YOLO เป็น **end-to-end, unified network** ที่หา What และ Where **พร้อมกัน**
- **[01:12:15]–[01:12:42]** ความเชื่อเรื่อง grid ของ YOLO คือ **ไอเดียเดียวกับ Faster R-CNN** ที่ว่า "ทุก pixel บน feature map เป็นจุดศูนย์กลาง" ต่างกันแค่ **domain**: grid บนภาพต้นฉบับ กับ feature map
- **[01:14:45]–[01:15:15]** นับ filter: 2 กล่อง × (x, y, w, h, conf) = **10 filters** สำหรับ Where บวก **หนึ่ง filter ต่อหนึ่ง class** สำหรับ What ได้ Where และ What ออกมาพร้อมกัน จึงเป็น single-stage
- **[01:15:52]–[01:17:41]** รุ่นพี่ที่ใช้ YOLO ในโปรเจกต์ **อธิบายมันไม่ได้** ตอนสอบโปรเจกต์ บริษัทอยากได้คนที่ "Verify / Understand / Optimize" ผลจาก AI ได้ (*human in the loop*) นักศึกษาสอบสัมภาษณ์ไม่ผ่านเพราะ "ตอบ Why the reason ไม่ได้" (can't answer why) "**อันนี้คือสิ่งสำคัญ**" (this is the important thing): เตรียมเจอคำถาม *why*
- **[01:17:51]** loss ของ YOLO คำนวณจาก **หลายส่วน** (object, confidence, class) ไม่ใช่แค่การทาย class
- **[01:21:01]–[01:21:44]** "สิ่งที่เด็ก ๆ ต้องรู้ก็คือ" (what you must know): bounding box มี **3 format**: (x, y, w, h) จากมุม, (x_center, y_center, w, h) และ (x1, y1, x2, y2) เช็กว่า library ที่ใช้ต้องการแบบไหน
- **[01:22:04]–[01:22:07]** label ของ YOLO ถูก **normalise ให้อยู่ใน 0–1** ดังนั้น **resize ภาพแล้วไม่กระทบ ground truth**
- **[01:22:30]** ไฟล์ภาพกับไฟล์ label ใช้ **ชื่อเดียวกัน แค่นามสกุลต่างกัน**
- **[01:22:47]–[01:23:10]** ขนาดโมเดล n/s/m/l/x: ใหญ่ขึ้นก็กินทรัพยากรมากขึ้นและแม่นขึ้น n มี 3.2 M params และ 8.7 GFLOPs ใช้กับมือถือและอุปกรณ์ embedded

---

## ข้อสอบที่น่าจะออก

1. **Classification, object detection และ segmentation ต่างกันยังไง?**
   Classification ให้แค่ What ส่วน detection และ segmentation ให้ทั้ง What และ Where Where ของ detection คือกล่องเล็กที่สุดที่ครอบวัตถุ Where ของ segmentation คือรูปร่างจริง

2. **บอก 4 ขั้นของ R-CNN และอธิบายว่าทำไมต้อง warping**
   ขั้นตอนคือ Selective Search หา proposal → warp เป็น 224×224 → ดึง feature ด้วย AlexNet (ตัด FC สุดท้ายออก) → SVM ต้อง warping เพราะ CNN ตอนนั้นไม่มี GAP/adaptive pooling FC layer จึงต้องการ vector ที่ flatten แล้วยาวคงที่ แลกมากับการเสีย aspect ratio

3. **Selective Search รวม region ยังไง และหยุดเมื่อไหร่?**
   ใช้ graph-based segmentation แล้วรวม region ที่อยู่ติดกันเป็นลำดับชั้นตามความคล้ายของสี, texture, ขนาด และ fill (รูปร่าง) หยุดเมื่อเหลือประมาณ 1,000–2,000 proposals ไม่ได้หยุดตามจำนวนรอบที่ตายตัว

4. **ทำไม Fast R-CNN เร็วกว่า R-CNN? บอกความต่างอย่างน้อย 3 ข้อ**
   - รัน CNN ครั้งเดียวบนทั้งภาพ แทนที่จะรันประมาณ 2,000 ครั้งบน warped region
   - ฉาย proposal ลงบน feature map ไม่ได้ crop จากภาพ
   - ใช้ RoI pooling แทน warping
   - classify ด้วย softmax แทน SVM และเพิ่ม bbox regressor
   - ผลคือ: 47 s → 0.32 s (×146); training 84 h → 9.5 h

5. **ทำไม Fast R-CNN ยังเป็น multi-stage และไม่ real-time? Faster R-CNN เปลี่ยนอะไร?**
   เพราะยังพึ่ง Selective Search ซึ่งช้าและเทรนไม่ได้ Faster R-CNN เปลี่ยนไปใช้ Region Proposal Network ที่เทรนได้แทน ส่วนอื่นของ pipeline เหมือนเดิม

6. **คำนวณ output ของ RoI pooling สำหรับ feature region ขนาด 5 แถว × 7 คอลัมน์ ที่ pool เป็น 2×2 (สไลด์ 24)**
   - แบ่งแถวเป็น 2 + 3 และคอลัมน์เป็น 3 + 4 โดยเศษอยู่ส่วนสุดท้าย
   - เอาค่า max ของแต่ละบล็อก: [[0.85, 0.84], [0.97, 0.96]]
   - Flatten เป็น `[0.85, 0.84, 0.97, 0.96]`

7. **feature map ของ RPN ขนาด 10×10 มี 3 scales และ 3 aspect ratios มี anchor กี่อัน? ต้องใช้ filter 1×1 กี่ตัวสำหรับ score และสำหรับพิกัด?**
   9 anchors ต่อจุด × 100 จุด = 900 anchors Filters: 9 สำหรับ score และ 36 สำหรับพิกัด

8. **บอก 6 ขั้นของ RPN**
   1. Intermediate conv 3×3 (512 filters)
   2. สร้าง anchor (scale × ratio)
   3. Classification layer: objectness score แบบ foreground/background
   4. Regression layer: offset ของ x, y, w, h
   5. เรียงตาม objectness แล้วเก็บ top n
   6. NMS

9. **คำนวณ IoU ของ A = (10, 10, 50, 50) และ B = (30, 30, 70, 70) แบบ xyxy**
   - Intersection = (50−30) × (50−30) = 400
   - พื้นที่แต่ละกล่อง = 1,600 ดังนั้น union = 3,200 − 400 = 2,800
   - IoU = 400/2,800 ≈ **0.143**

10. **ทำ NMS (threshold 0.5) กับกล่องที่มี score 0.9 (A), 0.8 (B), 0.7 (C) โดย IoU(A, B) = 0.6, IoU(A, C) = 0.1 และ IoU(B, C) = 0.7**
    - เรียง: A, B, C
    - ตัวอ้างอิง A: B มี IoU 0.6 > 0.5 เลยลบ B ทิ้ง C มี 0.1 เลยเก็บ C ไว้
    - ตัวอ้างอิง C: ไม่เหลืออะไรให้เทียบแล้ว
    - **ที่เก็บไว้: A และ C** C รอดทั้งที่ซ้อนกับ B เยอะ เพราะ B ถูกลบไปก่อนแล้ว

11. **YOLO(v1) ทายกล่องยังไง? output ต่อ cell มีขนาดเท่าไหร่?**
    - แบ่งภาพเป็น grid S×S cell ที่มีจุดศูนย์กลางของวัตถุจะรับผิดชอบวัตถุนั้น
    - แต่ละ cell ทาย B = 2 กล่อง (x, y, w, h, confidence) และ class distribution หนึ่งชุด
    - ต่อ cell: B·5 + C ค่า ถ้า B = 2 และ C = 10 จะได้ 20 YOLOv1 ในสไลด์ 59 ได้ 7×7×30
    - score สุดท้าย = P(class | object) × confidence แล้วทำ NMS

12. **ข้อจำกัดของ YOLOv1 มีอะไรบ้าง?**
    - หนึ่ง cell มีได้แค่ class เดียว วัตถุต่าง class ที่อยู่ cell เดียวกันจึงทำได้ไม่ดี
    - มีปัญหากับวัตถุเล็ก ๆ จำนวนมากที่อยู่รวมกัน เพราะ cell ใหญ่

13. **อธิบายเทอมต่าง ๆ ใน loss ของ YOLO และค่า λ**
    - SSE บน (x, y); SSE บน (√w, √h) โดยรากที่สองช่วยลดความไม่สมดุลระหว่างกล่องใหญ่กับกล่องเล็ก
    - Confidence ตอนมีวัตถุ และตอนไม่มีวัตถุ (λ_noobj = 0.5)
    - Class probabilities
    - λ_coord = 5 เพิ่มน้ำหนัก localisation loss

14. **แปลงกล่อง xyxy (100, 50, 300, 250) ในภาพ 640×480 เป็น label แบบ YOLO สำหรับ class 0**
    `0 0.3125 0.3125 0.3125 0.4167` ทำไมต้อง normalise? เพราะ label ยังใช้ได้หลัง resize ภาพ

15. **สถานการณ์ทางการแพทย์: จะเลือกใช้วิธีไหน?**
    - (a) **ตรวจหา polyp แบบ real-time ในวิดีโอส่องกล้องลำไส้ (colonoscopy):** ใช้ single-stage detector (YOLO) เพราะความเร็วสำคัญ และ Faster R-CNN ที่ประมาณ 0.2 s/frame ช้ากว่า YOLO ขนาดเล็ก (n/s) เหมาะกับอุปกรณ์ edge หรือ embedded
    - (b) **ต้องการเส้นขอบเนื้องอกที่แม่นยำเพื่อวัดพื้นที่หรือปริมาตร:** ใช้ segmentation ไม่ใช่ detection เพราะกล่องของ detection บอก Where ได้แค่คร่าว ๆ
    - (c) **เซลล์หรือ microcalcification เล็ก ๆ จำนวนมากที่อยู่กันแน่นในบริเวณเดียวของภาพ:** YOLOv1 อ่อนตรงนี้ (cell ใหญ่, หนึ่ง class ต่อ cell) detector แบบ two-stage ที่ใช้ anchor อย่าง Faster R-CNN ซึ่งทายหลาย anchor ต่อหนึ่งตำแหน่งบน feature map รับมือกับวัตถุที่อยู่หนาแน่นได้ดีกว่า

---

## กับดักที่เจอบ่อย

- **"Wrapping"**: สไลด์และอาจารย์สะกดว่า *Wrapping* แต่เทคนิคจริงคือ **warping** (การ resize) สะกดแบบไหนก็หมายถึงขั้นเดียวกัน ผลที่ได้คือ **warped region/image**
- **Detection vs segmentation** ต่างกันที่ **Where** ไม่ใช่ What
- **Fast R-CNN ยังใช้ Selective Search** จึง **ยังเป็น multi-stage** มีแค่ Faster R-CNN ที่เปลี่ยนไปใช้ RPN แทน
- **Faster R-CNN ไม่ใช่ single-stage** ถึงจะเทรนได้ทั้งหมดก็ตาม What กับ Where ยังมาจากคนละ module (RPN แล้วค่อย head) YOLO ต่างหากที่เป็น single-stage
- **กล่อง proposal ถูกวางไว้ที่ไหน:**
  - R-CNN: บน **ภาพต้นฉบับ** แล้วค่อย warp
  - Fast/Faster R-CNN: บน **feature map** (RoI projection, map ตามสัดส่วน) แล้วค่อยทำ **RoI pooling**
- **RoI pooling ใช้ MAX** ไม่ใช่ average **เศษไปอยู่ส่วนสุดท้าย**: รูปในสไลด์ 24 แบ่ง 5 → 2 + 3 และ 7 → 3 + 4
- **Classifier ของ R-CNN คือ SVM** ไม่ใช่ softmax CNN ที่ใช้คือ **AlexNet ที่ตัด FC สุดท้ายออก**
- **R-CNN ไม่มี bounding-box regressor** ในเวอร์ชันที่สอนในวิชานี้ (ตามสไลด์/อาจารย์) Fast R-CNN เป็นตัวที่เพิ่ม linear regressor เข้ามา (paper ต้นฉบับของ R-CNN จริง ๆ มี box regressor ต่อท้าย แต่ในข้อสอบให้ตอบตามสไลด์)
- **output ของ regressor คือ offset:** x และ y เอาไป **บวก** w และ h เอาไป **คูณ** (อัตราส่วน) ไม่ใช่พิกัดสัมบูรณ์
- **จำนวน anchor:** k = scales × ratios (3×3 = 9 ต่อตำแหน่ง) และทั้งหมดคือ H × W × k
  - filter สำหรับ score = k, filter สำหรับพิกัด = **4k** (9 → 36)
  - 50×50×9 คือ **22,500** ที่อาจารย์พูดว่า "~30,000" เป็นแค่ตัวเลขคร่าว ๆ
- **NMS:**
  - ลบกล่องที่ **score ต่ำกว่า** ในคู่ที่ซ้อนกัน ใช้ **IoU > threshold** ตัดสินว่า "เป็นวัตถุเดียวกัน"
  - กล่องที่ IoU = 0 เป็นคนละวัตถุ เก็บไว้ทั้งคู่
  - ทำซ้ำโดยใช้กล่องถัดไปที่ยังรอดเป็นตัวอ้างอิง
- **argument `iou` ของ Ultralytics คือ threshold ของ NMS** (default 0.6) ไม่ใช่ threshold ของ mAP ส่วน `conf` default = 0.001 นั้น "ไม่แนะนำ"
- **class probability ของ YOLO เป็นต่อ cell ไม่ใช่ต่อกล่อง** ส่วน confidence เป็นต่อกล่อง score สุดท้าย = class prob × box confidence
- **YOLOv1 ใช้ B = 2 กล่องต่อ cell** แต่มีแค่ **หนึ่ง class ต่อ cell**
- **format ของ label YOLO** คือ `class x_center y_center w h` normalise ให้อยู่ใน 0–1 เป็น **จุดศูนย์กลาง** ไม่ใช่มุม และไม่ใช่ xyxy
  - มุมซ้ายบนของภาพคือ (0, 0)
  - ไฟล์ `.txt` หนึ่งไฟล์ต่อภาพ หนึ่งแถวต่อวัตถุ
  - ชื่อไฟล์เดียวกับภาพ อยู่ใต้ `labels/` แทน `images/`
- **box มีสาม format** (corner xywh, centre xywh, xyxy) เช็กทุกครั้งว่า library หรือโมเดลต้องการหรือคืนค่าแบบไหน
- **loss ของ YOLO ใช้ √w และ √h** โดย λ_coord = 5 และ λ_noobj = 0.5 อย่าสลับค่า λ กัน
- **ตัวเลขความเร็ว:**
  - Fast R-CNN เร็วกว่า R-CNN ×146 แบบ *ไม่นับ* proposal (47 s → 0.32 s) แต่แค่ **×25 ถ้านับ Selective Search ด้วย** (50 s → 2 s)
  - Training เร็วขึ้น ×8.8 (84 h → 9.5 h)
  - Faster R-CNN เร็วขึ้น ×250 (0.2 s)
  - mAP คือ 66.0 → 66.9 → 66.9 ดังนั้นสิ่งที่ Faster R-CNN ได้เพิ่มคือความเร็ว ไม่ใช่ mAP
