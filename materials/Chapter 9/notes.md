# Chapter 9 — Multilabel Classification & Image Regression & Multi-task Learning

เอา CNN backbone ตัวเดียว (เช่น AlexNet) ไปทำงาน multiclass, multilabel หรือ regression ได้ โดยเปลี่ยนแค่ 3 อย่าง: **จำนวน output nodes**, **activation function ตัวสุดท้าย** และ **loss function** อาจารย์บอกว่าในโค้ดจริงคือ "แก้แค่ 2-3 บรรทัดเอง"
Multi-task learning คือ *ปัญหา* ที่มีหลายคำถามต่อภาพ สไลด์ p.22 แสดง 3 วิธีแก้: (a) feature extractor 1 ตัว + classifier 1 ตัวที่ให้ output ทุกอย่าง, (b) feature extractor ที่ใช้ร่วมกัน + head แยกของแต่ละ task, หรือ (c) โมเดลละ 1 task

> หมายเหตุที่มา: ตัวอย่างในสไลด์ (logits, ค่า softmax/sigmoid, threshold, loss, regression ราคาบ้าน, architecture ผลไม้และ lymphedema) เอามาจากรูปในสไลด์ (p.9–11, 15–17, 19, 21, 22) สูตรที่มีป้าย *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)* เป็นนิยามจากตำรา ไม่มีในสไลด์

---

## Key concepts

### งาน 3 ประเภท (สไลด์ p.4–7)
- **Multiclass Image Classification**: แบ่งภาพเป็นคลาสที่ **ไม่ทับซ้อนกัน** ภาพหนึ่งได้คลาสเดียว **เท่านั้น** ตัวอย่างในสไลด์: หมา/แมว/ม้า, ชนิดดอกไม้, ชนิดรถ ตัวอย่างรูป: "Dog"
  - สรุปสั้น ๆ ของอาจารย์: **"หนึ่งภาพ หนึ่งคำตอบ"** คือมีตัวเลือกหลายตัวแต่คำตอบมีอันเดียว
- **Multilabel Image Classification**: คลาส **ทับซ้อนกันได้** ภาพหนึ่งมี **หลาย label** ได้ ตัวอย่างในสไลด์: รูปที่มีทั้งแมวและหมา, ฉากที่มี รถ + คน + ต้นไม้, แนวหนังจากโปสเตอร์ ("Action, Sci-Fi")
  - สรุปสั้น ๆ ของอาจารย์: **"หนึ่งภาพ มีโอกาสหลายคำตอบ"** คือมีหลายตัวเลือกและอาจตอบได้มากกว่า 1
  - (transcript จาก Whisper เขียนว่า "Multi-Level" คำที่ถูกคือ **Multilabel**)
- **Image Regression**: ทำนาย **ตัวเลขต่อเนื่อง** จากภาพ ตัวอย่างในสไลด์: อายุจากหน้า, ความยาวปลา, อุณหภูมิจากภาพถ่ายทางอากาศ, ระยะทาง, ราคาอสังหาฯ จากรูป ("2,450,000") สไลด์ p.14 เพิ่มการประมาณอายุจากหน้าและการทำนายราคาบ้าน
  - Regression ใช้กับ **ข้อมูลต่อเนื่อง/อนุกรม** ส่วน classification ใช้กับ **ข้อมูลไม่ต่อเนื่อง (discrete)** วิธีจัดการข้อมูล **loss function** และ **evaluation metrics** ต่างกันหมด (สไลด์ p.14)
- Multiclass และ multilabel เป็น **classification** ทั้งคู่: output เป็น **discrete** คือชื่อกลุ่ม ส่วน regression ให้ output เป็นตัวเลข **ต่อเนื่อง**

### สร้างโมเดลจาก multiclass CNN (สไลด์ p.9–11, p.15–17)
- **Pipeline ที่เหมือนกัน** ทุกสไลด์: Input → Convolution → Pooling → Fully Connected → **Scoring Function** (คะแนนดิบ หรือ logits) → output activation → ผลทำนาย
- **neuron 1 ตัวใน output layer** (สไลด์ p.10, p.16):
  - ผลรวม: z = (i1×w1) + (i2×w2) + … + b1
  - neuron ของ classification จะผ่าน activation ต่อ output = **f(z)** โดย f = softmax หรือ sigmoid
  - neuron ของ regression ให้ output เป็น **z เลย**: "Linear Function" แปลว่าไม่มี f(·)
- **Multilabel**: เอา multiclass CNN มาเปลี่ยน activation ของ output layer จาก **Softmax → Sigmoid**
  - Softmax: output **รวมกันได้ 1 เสมอ**
  - Sigmoid: output แต่ละตัวอยู่ใน **0–1** และ **ไม่ขึ้นกับกัน**
  - จำนวน output nodes = **จำนวนคลาส**
  - Sigmoid วาดเป็นรูป "S" บน output node แต่ละตัว **แยกกัน** ส่วน Softmax เป็นกล่องเดียวที่รับ score ทั้งหมดพร้อมกัน
  - สไลด์ p.11 เพิ่มขั้น **"Threshold at 0.5"** ต่อท้าย
- **Image Regression**: เอา multiclass CNN มาตั้ง activation ของ output layer เป็น **Linear**
  - ช่วงของ output แบบ linear คือ **−∞ ถึง ∞**
  - จำนวน output nodes = **1** ในสไลด์ ใน lecture บอกว่าเท่ากับ **จำนวนค่าที่อยากทำนาย**
  - "ใช้ Linear ก็คือไม่ใช้": เอาค่าจาก score function มาเป็นคำตอบเลย
- **การคำนวณ error** ("การพิจารณาผลลัพธ์ของแบบจำลองและการคำนวณข้อผิดพลาด", สไลด์ p.11, p.17):
  - ผลทำนาย (ผลทำนาย) กับคำตอบจริง (ผลเฉลย) เข้ากล่อง loss ทั้งคู่ ได้ output เป็น "Loss, Error"
  - Multiclass และ multilabel ใช้กล่อง **Cross Entropy Loss** ส่วน regression ใช้กล่อง **L1 Loss**
  - เชิงอรรถในสไลด์:
    - "นักศึกษาสามารถกำหนด Loss Function เป็น **Binary Cross Entropy Loss หรือ Cross Entropy Loss** ก็ได้ใน ณ ปัจจุบันนี้" (multilabel)
    - "…**Mean Absolute Error (MAE) หรือ L1 loss, Mean Squared Error (MSE) หรือ L2 loss และอื่น ๆ** ก็ได้" (regression)

### ตารางหลัก

| ประเภทงาน | Output neurons | Output activation | Loss | อ่านผลทำนายยังไง |
|---|---|---|---|---|
| Binary classification (เช่น ป่วย / ไม่ป่วย, สด / เน่า) | 2 (คลาสละ 1 ตัว เหมือน head "2 Class" ในสไลด์ p.21 และ head lymphedema ใน lecture); อีกแบบที่นิยมคือ sigmoid node เดียว | Softmax (2 nodes) / Sigmoid (1 node) | (Binary) Cross-Entropy | ความน่าจะเป็นสูงสุด หรือ p ≥ 0.5 ถ้าเป็นแบบ node เดียว |
| Multiclass | = จำนวนคลาส | **Softmax** (รวม = 1) | **Cross-Entropy** | **argmax**: คลาสที่ค่ามากสุดคือคำตอบ |
| Multilabel | = จำนวนคลาส (labels) | **Sigmoid** (แต่ละตัว 0–1 ไม่ขึ้นกับกัน) | **Binary Cross-Entropy หรือ Cross-Entropy** (เชิงอรรถสไลด์ p.11 และ lecture) | **Threshold 0.5**: ทุก node ที่ output **≥ 0.5** เป็นคำตอบ |
| Regression | = จำนวนค่าที่จะทำนาย (สไลด์: 1) | **Linear** (ไม่มี activation, −∞ ถึง ∞) | **L1 (MAE)** หรือ **L2 (MSE)** หรืออื่น ๆ ต้องลองดู | ตัวเลข output ดิบ ๆ *คือ* ผลทำนาย |
| Multi-task (วิธี b) | task ละ 1 head และแต่ละ head มีขนาดตามประเภทงานของตัวเอง (สไลด์ p.21: head 2 คลาส + head 11 คลาส) | แต่ละ head ใช้ activation ของ task ตัวเอง | แต่ละ head ใช้ loss ของ task ตัวเอง (รวม = ผลรวม loss ทุก task, *ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์*) | อ่านแต่ละ head แยกกันตามกฎของมัน |

(แถว Binary: แบบ sigmoid node เดียวเป็นความรู้ DL ทั่วไป ไม่ได้มาจากสไลด์)

### Multi-task Learning (สไลด์ p.19–22)
- **นิยาม**: train **โมเดลเดียว** ให้ทำนาย **หลายอย่างพร้อมกัน** ผลทำนายแต่ละครั้งตอบหลายคำถาม (tasks)
  - ตัวอย่างในสไลด์: ชนิดผลไม้ (แอปเปิ้ล / กล้วย / ส้ม) **และ** สภาพผลไม้ (เน่า / ไม่เน่า)
- **ข้อดี** (สไลด์ p.19):
  - **ประหยัดทรัพยากร**: ไม่ต้องมีโมเดลแยกทุก task ใช้เวลาและ compute น้อยลง
  - **Joint learning**: task แชร์ข้อมูลกัน ทำให้แต่ละ task แม่นขึ้น
- **เรื่องที่ต้องคิด** (สไลด์ p.20):
  - **Task relatedness**: ได้ผลดีสุดเมื่อ task เกี่ยวข้องกันหรือมีอะไรร่วมกัน
  - **Data limitation**: มีประโยชน์ตอนข้อมูลน้อย เพราะข้อมูลที่แชร์ข้าม task ช่วยให้โมเดลดีขึ้น
  - **Shared Feature Extractor**: ส่วน feature extraction ของ network ใช้ร่วมกันทุก task
  - **Task-specific Heads**: task ละ 1 neural network แต่ละ head ต่อกับ shared feature extractor
  - **Shared Decision-making Layer**: อีกทางเลือกที่แชร์ทั้ง feature extractor *และ* บางส่วนของ neural network มีแค่ **output layer สุดท้าย** ที่แยกตาม task
- **Negative Transfer** (สไลด์ p.21): ถ้า task ต่างกันมากหรือไม่เกี่ยวกันเลย multi-task learning อาจได้ผล *แย่กว่า* โมเดล single-task architecture ต้องทำให้ feature ที่แชร์ช่วย **ทุก** task
- **Multi-Task Learning vs Multi-Model Learning** (diagram สไลด์ p.21, รูปผลไม้):
  - **Multi-Task Learning**: Convolution block → Flatten → **Dense 2 heads**
    - Head 1: "Fresh", **2 คลาส** (ความสดและเน่าเปื่อย, สด / เน่า)
    - Head 2: "Apple", **11 คลาส** (ชนิดผักและผลไม้, ชนิดผัก/ผลไม้)
  - **Multi-Model Learning**: **pipeline แยกกันครบ 2 ชุด** (Convolution block → Flatten → Dense) ชุดหนึ่งสำหรับงาน 2 คลาส อีกชุดสำหรับงาน 11 คลาส
- **Diagram สไลด์ p.19** (เคส lymphedema): ภาพ ultrasound → **Feature extractor (Class/SEG/SEFS)** ที่ใช้ร่วมกัน → classifier 3 ตัว: **Classifier (Class)** → "Class: Lymphedema", **Classifier (SEG)** → "SEG grade: 2", **Classifier (SEFS)** → "SEFS grade: 2"

### Multilabel vs Multi-task (จุดแยกสำคัญใน lecture)
- ทั้งคู่ให้ "คำตอบมากกว่า 1" ต่างกันตรงที่คำตอบมีความหมายเดียวกันหรือเปล่า
  - **Multilabel**: คำตอบทั้งหมด **ความหมายเดียวกัน** Sci-Fi, Comedy และ Action ล้วนเป็นคำตอบของคำถามเดียว คือแนวหนัง
  - **Multi-task**: คำตอบ **ความหมายต่างกัน** "ผลไม้อะไร?" กับ "เน่าไหม?" เป็นคนละคำถาม
- แต่ละ task ในปัญหา multi-task เป็น multiclass, multilabel หรือ regression ก็ได้ แต่ในทางปฏิบัติ ส่วนใหญ่แต่ละ task เป็น **multiclass**

### 3 วิธีแก้ปัญหา multi-task (สไลด์ p.22 "ทางเลือกในการเรียนรู้หลายงาน"; lecture [00:16:19]–[00:19:30])
ตัวอย่างที่ใช้ตลอด: นักศึกษาป.โท ใช้ **ภาพ ultrasound** ทำนาย 3 อย่าง: **เป็น lymphedema หรือไม่** (โรคบวมน้ำเหลือง), **SEG grade** และ **SEFS grade** (ระดับความรุนแรง; สไลด์แสดงเช่น "SEG grade: 2", "SEFS grade: 2")
- **(a) โมเดลเดียว output รวมอันเดียว**
  - Diagram: Feature extractor (Class/SEG/SEFS) ตัวเดียว → Classifier (Class/SEG/SEFS) ตัวเดียว → "Class: Lymphedema, SEG grade: 2, SEFS grade: 2"
  - architecture ที่แสดงเป็นแบบ DenseNet: Conv 7×7 → Max pooling → Dense block 1 → Transition 1 → Dense block 2 → Transition 2 → Dense block 3 → Transition 3 → Dense block 4 → Global average pooling → Fully connected → **Softmax function**
  - ใน lecture อธิบายว่าเป็นการ **แปลง multi-task เป็น multiclass ด้วยการเอาจำนวนคลาสมาคูณกัน** ตัวอย่าง: ป่วย/ไม่ป่วย (2) × SEG ระดับ 0/1/2 (3) = **6 คลาส** (ป่วย-0, ป่วย-1, ป่วย-2, ไม่ป่วย-0, ไม่ป่วย-1, ไม่ป่วย-2) แล้วใช้ softmax ตัวเดียว
- **(b) Multi-task จริง ๆ**
  - **Feature extractor (Class/SEG/SEFS) ตัวเดียวที่ใช้ร่วมกัน** สร้าง feature map
  - feature map ถูก flatten แล้วส่งเข้า **classifier แยก 3 ตัว**: Classifier (Class), Classifier (SEG), Classifier (SEFS) พวกนี้คือ **Task-Specific Heads**
  - ตัวอย่างขนาด head ใน lecture: head โรคมี 2 output nodes; head SEG (ระดับ 0/1/2) มี 3 output nodes
  - เหมาะกับเคสนี้เพราะ output **เกี่ยวข้องกันทางคลินิก**: หมอดู lymphedema จากระดับ SEG/SEFS
- **(c) Multi-model**
  - **Feature extractor แยก 3 ตัวและ Classifier 3 ตัว** (Class, SEG, SEFS) คือ "1 โมเดลต่อ 1 Task"
- ประเด็นสำคัญ: **"Multitask จริง ๆ แล้วมันไม่ใช่ Architecture มันเป็นปัญหา"** มันคือประเภทของปัญหา ส่วน a, b และ c เป็นวิธีแก้คนละแบบ แต่ละแบบมีข้อดีข้อเสียต่างกัน

---

## สูตรและตัวเลขที่ต้องจำ

**จากสไลด์**
- Output neuron (p.10, p.16): z = (i1×w1) + (i2×w2) + … + b1
  - Classification output = f(z)
  - Regression (linear) output = z
- Softmax output: รวมกันได้ 1 Sigmoid output: แต่ละค่าอยู่ใน 0–1 และไม่ขึ้นกับกัน Linear output: −∞ ถึง ∞
- Multilabel threshold = **0.5**
- Regression loss: **MAE = L1**, **MSE = L2**

**สูตร** *(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์; สไลด์บอกแค่คุณสมบัติข้างบน ไม่ได้ให้สมการ)*
- **Softmax** (multiclass): pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ} ดังนั้น **Σ pᵢ = 1**
- **Sigmoid** (multilabel): σ(z) = 1 / (1 + e^{−z}) แต่ละ output อยู่ใน (0,1) ผลรวมเป็นอะไรก็ได้ตั้งแต่ 0 ถึงจำนวนคลาส
- **Linear** (regression): ŷ = z
- **Cross-Entropy** (multiclass, y แบบ one-hot): L = −Σᵢ yᵢ log pᵢ ซึ่งเท่ากับ −log(p ของคลาสที่ถูก)
- **Binary Cross-Entropy** (multilabel, เฉลี่ยทุก label): L = −(1/C) Σᵢ [yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ)]
- **MAE / L1**: (1/N) Σ |yᵢ − ŷᵢ| **MSE / L2**: (1/N) Σ (yᵢ − ŷᵢ)²
- **Loss รวมของ multi-task** (วิธี b): L_total = Σ_t w_t · L_t แต่ละ task t ใช้ loss ของตัวเอง แบบง่ายสุดคือ w_t = 1 ทั้งหมด

**ตัวอย่างคำนวณจากสไลด์** (p.9–11, p.17) ทุกตัวอย่างใช้ output จาก scoring function ชุดเดียวกัน **[−3.44, 1.66, 3.91]**
- **Multiclass** (รูปหมา, คลาส Bird / Cat / Dog):
  - Softmax → Bird 0.010, Cat 0.059, **Dog 0.931** ผลทำนายคือ **Dog**
  - ผลทำนาย [0.010, 0.059, 0.931] เทียบกับคำตอบจริง [0, 0, 1] → Cross Entropy Loss
  - CE = −ln 0.931 ≈ **0.07** (คำนวณเอง)
- **Multilabel** (โปสเตอร์ Avengers, label Drama / Sci-Fi / Action):
  - Sigmoid → **Drama 0.03, Sci-Fi 0.84, Action 0.98**
  - Threshold ที่ 0.5 → ผลทำนาย **[0, 1, 1]** คือ "Action, Sci-Fi"
  - คำตอบจริงในสไลด์ p.11 คือ **[0, 0, 1]** โมเดลเลยเติม Sci-Fi มาผิด (false positive) → Cross Entropy Loss
  - ค่า sigmoid ตรงเป๊ะ: σ(−3.44) = 0.031, σ(1.66) = 0.840, σ(3.91) = 0.980
- **Regression** (รูปบ้าน, ราคา):
  - Linear output **2,450,000**; คำตอบจริง **2,500,000** → L1 Loss
  - L1 error = |2,450,000 − 2,500,000| = **50,000** (คำนวณเอง) ถ้าเป็น L2 จะได้ 50,000² = 2.5 × 10⁹

**ตัวเลขจาก lecture**
- Regression ต้องใช้ **ข้อมูล train มากกว่า classification ประมาณ 10–20 เท่า** ("หลาย 10-20 เท่า") คนเลยมักจะ train **1 โมเดลต่อ 1 ค่าที่ทำ regression**
- จำนวนคลาสของวิธี (a): (คลาสของ task 1) × (คลาสของ task 2) × … เช่น **2 × 3 = 6**
- Regression output nodes = จำนวนค่าที่จะทำนาย ตัวอย่าง: ความดัน systolic + diastolic = **2 nodes** หรือ 1 node ถ้าทำนายแค่ systolic
- ลำดับความยาก: **Regression (optimise ยากสุด) > Multilabel > Multiclass (ง่ายสุด)**

---

## เรื่องที่อาจารย์เน้น

- **[00:00:08]–[00:00:26] ทำไมต้องมีบทนี้** รุ่นพี่ที่ทำ project บอกว่า "ไม่มีโมเดล" สำหรับ regression หรืองานอื่น เพราะวิชาสอนแค่ multiclass CNN อาจารย์ตอบว่า: ใช้โมเดลเดิมได้เลย แค่ต้องรู้ว่าต้องเปลี่ยนอะไร
- **[00:01:55]–[00:02:10] คำถามในห้อง:** "ถ้าต้องอธิบายให้น้องปี 1 ฟังว่า Multiclass กับ Multilabel ต่างกันยังไง จะตอบสั้น ๆ ว่าอะไร?"
  → **Multiclass = หนึ่งภาพ หนึ่งคำตอบ Multilabel = หนึ่งภาพ อาจมีหลายคำตอบ**
- **[00:02:53]–[00:04:06] "คำถาม Classic":** ทำนายอายุจากรูปหน้าเป็น Regression หรือ Multiclass?
  - **Multilabel ตัดทิ้งได้เลย** คำตอบแบบคิดเร็ว ๆ คือ Regression
  - แต่ปกติอายุที่ทำนายเป็นจำนวนเต็ม (ไม่มีใครบอกว่าอายุ 10.5) เลยเป็น **discrete** จริง ๆ
  - คำตอบ: มองเป็น **Regression หรือ Multiclass ก็ได้**
  - "คีย์เวิร์ด" ของ regression คือตัวเลข **ต่อเนื่อง** อย่าคิดว่า "เป็นตัวเลข = regression" (ดู Traps)
- **[00:04:18]–[00:04:48] คำถาม:** เพื่อน 3 คนต้องใช้ **AlexNet backbone ตัวเดียวกัน** ทำ multiclass, multilabel และ regression ต้องเปลี่ยนอะไร? คำตอบ: output nodes, activation ตัวสุดท้าย, loss
- **[00:05:08]–[00:05:12] สิ่งแรกที่ต้องดูคือคำตอบที่เป็นไปได้** Output-layer nodes = จำนวนคลาส
- **[00:06:45]–[00:07:01] "จดไปครับ":** สำหรับงาน **ทั้ง 3 ประเภท** **output-layer nodes = จำนวนคลาส/ค่าที่ทำนาย** อาจารย์พูดซ้ำ 2 รอบ
- **[00:05:46]–[00:07:39] ตัวอย่างความดันเลือด** ถ่ายรูปแขนหรือผิวแล้วทำนายความดัน systolic และ diastolic อันนี้คือ regression **2 ค่า เลยมี 2 output nodes** ถ้าทำนายแค่ systolic ใช้ 1 node "งาน Regression ยาก… หนึ่งโมเดลต่อหนึ่งค่า ก็โหดแล้ว"
- **[00:07:50]–[00:08:56] "จดไป":** **Multiclass ใช้ Softmax; Multilabel ใช้ Sigmoid** อาจารย์บอกว่าสองอันนี้ต่างกันตรงส่วน **inference** ไม่ใช่ตอน training
  - Multiclass: ผ่าน softmax แล้วค่ามากสุดคือคำตอบ
  - Multilabel: sigmoid แปลงแต่ละ score ให้อยู่ใน 0–1 และ **ทุก node ที่ ≥ 0.5 ก็เป็นคำตอบด้วย** ("Test Showing" ใน transcript หมายถึง threshold)
- **[00:09:07]–[00:09:20] Loss ของ Multiclass และ Multilabel:** ตระกูลเดียวกัน "binary cross entropy หรือ cross entropy ก็ได้" ตรงกับเชิงอรรถสไลด์ p.11
- **[00:09:39]–[00:10:13] ความจริงเรื่องข้อมูลของ regression** Classification ใช้ข้อมูลน้อยกว่า regression มาก regression ต้องใช้ "หลาย 10-20 เท่า" ปกติเลยทำ **1 โมเดลต่อ 1 ค่าที่ทำ regression** จะทำหลายค่าในโมเดลเดียวก็ได้ถ้าข้อมูลพอ "สิ่งที่อยากเน้นย้ำ": regression output nodes = จำนวนค่าที่จะทำนาย
- **[00:10:31]–[00:10:46] "จดไปครับ": Image Regression ใช้ Linear** คือไม่มี activation; score คือคำตอบ
- **[00:11:07]–[00:11:49] Loss: Multiclass/Multilabel ใช้ Cross-Entropy; Regression ใช้ L1 (MAE) หรือ L2 (MSE)**
  - "จุดนี้กล้าดอกจัน" (ใส่ดอกจันไว้ตรงนี้): **ไม่มีกฎตายตัวว่า L1 หรือ L2** project ของนักศึกษาคนหนึ่งใช้ L1 ได้ผล อีกคนต้องใช้ L2 เลยต้องลองทั้งคู่
- **[00:11:52]–[00:13:24] Quiz สรุปท้ายคาบ** (ทั้งห้องตอบพร้อมกัน):
  - หลายตัวเลือก คำตอบเดียว → **Multiclass**
  - หลายตัวเลือก อาจมีมากกว่า 1 คำตอบ → **Multilabel**
  - คำตอบเป็นตัวเลข → **Regression**
  - จำนวน output node ขึ้นกับ → **จำนวนคลาส / คำตอบ**
  - Activation ตัวสุดท้ายของ multiclass ตอน inference → **Softmax** ของ multilabel → **Sigmoid แล้วค่าที่ > 0.5 นับเป็นคำตอบ** ของ regression → **Linear**
  - Loss ของ multiclass/multilabel → **Cross-Entropy** ของ regression → **L1 หรือ L2**
  - แก้โค้ด **แค่ 2–3 บรรทัด** โค้ดในเน็ตเป็น multiclass เป็นค่าเริ่มต้น
- **[00:13:44]–[00:15:02] Multi-task vs Multilabel (อาจารย์เตือนว่า "เด็ก ๆ จะสับสน")** คำตอบของโปสเตอร์หนังมี **ความหมายเดียวกัน** (แนวหนัง) เลยเป็น multilabel ส่วนชนิดผลไม้ + เน่า/ไม่เน่า เป็น **คนละความหมาย** เลยเป็น multi-task
- **[00:15:09]–[00:15:20]** แต่ละ task เป็น multiclass, multilabel หรือ regression ก็ได้ แต่ในทางปฏิบัติส่วนใหญ่แต่ละ task เป็น **multiclass**
- **[00:15:26]–[00:15:36] ความยาก:** Regression optimise ยากสุด รองลงมาคือ Multilabel และ Multiclass ง่ายสุด
- **[00:15:42]–[00:19:09] เคส ultrasound lymphedema** (เป็น lymphedema ไหม + SEG grade + SEFS grade) แก้ได้ 3 วิธี:
  - (a) เอาคลาสมาคูณกัน 2×3 = 6
  - (b) Shared feature extractor + task-specific heads
  - (c) Multi-model, CNN 1 ตัวต่อ 1 task
  - (b) เหมาะเพราะ output เกี่ยวข้องกันทางคลินิก
- **[00:19:23]–[00:19:40] "Multitask จริง ๆ แล้วมันไม่ใช่ Architecture"** มันคือปัญหาที่มีปัญหาย่อยหลายข้อ a, b และ c มีข้อดีข้อเสียต่างกัน

---

## ข้อสอบที่น่าจะออก

1. **นิยามและเปรียบเทียบ multiclass กับ multilabel image classification พร้อมตัวอย่างทางการแพทย์อย่างละ 1 ตัวอย่าง**
   - Multiclass: คลาสไม่ทับซ้อนกัน ภาพละ 1 label ตัวอย่าง: แยกรอยโรคผิวหนังเป็นการวินิจฉัยเดียวพอดี
   - Multilabel: คลาสทับซ้อนกันได้ ภาพหนึ่งมีหลาย label ที่จริงได้ ตัวอย่าง: X-ray ปอดที่เจอหลายความผิดปกติพร้อมกัน
   - (ตัวอย่างทางการแพทย์ยกมาเพื่ออธิบาย ในสไลด์ใช้สัตว์ ดอกไม้ และโปสเตอร์)

2. **มีโมเดล AlexNet แบบ multiclass บอกให้ครบว่าต้องเปลี่ยนอะไรให้เป็น (a) multilabel, (b) image regressor**
   - (a) Output nodes = จำนวน label Softmax → **Sigmoid** Loss → **BCE/CE** Inference: threshold แต่ละ output ที่ **0.5**
   - (b) Output nodes = จำนวนค่าที่จะทำนาย (สไลด์ใช้ 1) Softmax → **Linear** Loss → **MAE (L1) หรือ MSE (L2)** Inference: อ่าน output ดิบ ๆ

3. **ทำไม multilabel ต้องใช้ sigmoid ไม่ใช่ softmax?**
   - Softmax ให้ output รวมได้ 1 เสมอ คลาสเลยแย่งกัน score สูงของคลาสหนึ่งดันคลาสอื่นให้ต่ำลง เหมาะกับคำตอบเดียวพอดี
   - Sigmoid ให้แต่ละคลาสมีค่า 0–1 ที่ **ไม่ขึ้นกับกัน** หลายคลาสเลย ≥ 0.5 พร้อมกันได้

4. **(สไลด์ p.11) Score [−3.44, 1.66, 3.91] ของ Drama / Sci-Fi / Action ผ่าน sigmoid → [0.03, 0.84, 0.98] ผลทำนายคืออะไร? คำตอบจริงคือ [0, 0, 1]: โมเดลถูกไหม?**
   - หลัง threshold 0.5 ผลทำนายคือ [0, 1, 1] = **Action, Sci-Fi**
   - ไม่ถูกทั้งหมด: **Sci-Fi เป็น false positive** loss เลยไม่เป็นศูนย์

5. **(สไลด์ p.9) Score ชุดเดียวกันผ่าน softmax ได้ Bird 0.010, Cat 0.059, Dog 0.931 ผลทำนายคืออะไร และ CE loss เท่าไหร่ถ้าคำตอบจริงคือ Dog?**
   - ผลทำนาย: **Dog** (argmax)
   - CE = −ln 0.931 ≈ **0.07**

6. **"ทำนายอายุผู้ป่วยจากภาพหน้า/X-ray" เป็น regression หรือ multiclass? ให้เหตุผล**
   - ตอบได้ทั้งคู่ อายุเป็นตัวเลข เลยมองเป็น regression ได้ตามธรรมชาติ
   - ปกติอายุบอกเป็นจำนวนเต็ม (discrete) เลยมองเป็น multiclass ก็ได้
   - **ไม่ใช่** multilabel เป็นตัวเลขไม่ได้แปลว่าต้องเป็น regression เสมอไป

7. **ออกแบบ output layer: ทำนายความดัน systolic และ diastolic จากภาพผิว/แขน**
   - Regression ที่มี **2 output nodes**, activation แบบ **linear** และ loss **MAE หรือ MSE**
   - ข้อสังเกตจากอาจารย์: regression ยากและต้องใช้ข้อมูลมากกว่าประมาณ 10–20 เท่า เลยมักเลือกทำ 2 โมเดลแยก โมเดลละ output เดียว

8. **(สไลด์ p.17) CNN ราคาบ้านให้ output 2,450,000; ราคาจริงคือ 2,500,000 สไลด์ใช้ activation และ loss อะไร และ L1 error เท่าไหร่?**
   - Linear activation, L1 loss
   - L1 error = **50,000**

9. **คำนวณ MAE และ MSE** ค่าจริง = [120, 80], ค่าทำนาย = [110, 84]
   - Error: 10 และ 4
   - MAE = (10 + 4)/2 = **7**
   - MSE = (100 + 16)/2 = **58**

10. **สำหรับ regression ใช้ L1 หรือ L2 ดีกว่า?**
    - คำตอบของอาจารย์: **ไม่มีกฎตายตัว ลองทั้งคู่** ขึ้นกับงาน เชิงอรรถสไลด์บอกว่า MAE/L1, MSE/L2 "และอื่น ๆ" ใช้ได้หมด
    - (ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์: L2 ลงโทษ error ใหญ่ ๆ หนักกว่า และไวต่อ outlier มากกว่า)

11. **แยกความต่างระหว่าง multilabel กับ multi-task learning**
    - Multilabel: หลายคำตอบของ **คำถามเดียว** ที่มีความหมายเดียว (เช่น ทุกแนวของหนังเรื่องหนึ่ง)
    - Multi-task: คำตอบของ **คนละคำถาม** ความหมายต่างกัน (เช่น ความสดของผลไม้ + ชนิดผลไม้ หรือ เป็น lymphedema ไหม + SEG grade + SEFS grade)

12. **งาน ultrasound: ทำนาย lymphedema (ใช่/ไม่ใช่), SEG grade และ SEFS grade อธิบายวิธี a, b, c จากสไลด์ p.22**
    - **a:** feature extractor 1 ตัว + classifier 1 ตัวที่มี softmax output เดียวครอบทุก task คือการเอาคลาสมาคูณกันแบบใน lecture สำหรับ ใช่/ไม่ใช่ × SEG 0/1/2 ได้เป็น multiclass 6 คลาส
    - **b:** shared feature extractor 1 ตัว → task-specific classifier 3 ตัว (Class, SEG, SEFS) Loss = ผลรวม CE loss ทั้ง 3
    - **c:** feature extractor แยก 3 ตัว + classifier 3 ตัว (multi-model)

13. **(สไลด์ p.21) วาด / อธิบาย Multi-Task vs Multi-Model learning สำหรับรูปผลไม้ที่ทำนายความสด (2 คลาส) และชนิดผัก/ผลไม้ (11 คลาส)**
    - Multi-Task: Convolution block 1 ชุด → Flatten → Dense 2 heads อันหนึ่งมี 2 output nodes อีกอันมี 11
    - Multi-Model: pipeline Conv → Flatten → Dense แยกกัน 2 ชุด มี output 2 และ 11

14. **บอกข้อดีของ multi-task learning และเงื่อนไขที่ทำให้ได้ผลดีที่สุด negative transfer คืออะไร?**
    - ข้อดี: ประหยัดทรัพยากร (โมเดลเดียวแทนหลายตัว); joint learning ที่แชร์ข้อมูลกันทำให้แต่ละ task ดีขึ้น
    - ได้ผลดีสุดเมื่อ task **เกี่ยวข้องกัน** และเมื่อ **ข้อมูลมีน้อย**
    - Negative transfer: เมื่อ task ต่างกันมากหรือไม่เกี่ยวกัน โมเดล multi-task อาจได้ผล **แย่กว่า** โมเดล single-task

15. **Shared feature extractor + task-specific heads vs shared decision-making layer: ต่างกันยังไง?**
    - แบบแรกแชร์แค่ feature extractor และแต่ละ task มี neural network (head) ของตัวเอง
    - แบบที่สองแชร์ feature extractor **และ** layer ของ neural network มีแค่ output layer สุดท้ายที่แยกตาม task

---

## จุดหลอกที่เจอบ่อย

- **ใช้ softmax กับ multilabel** Softmax บังคับให้ output รวมได้ 1 คลาสเลยแย่งกัน สุดท้ายได้คำตอบเดียว Multilabel ต้องใช้ **sigmoid** (output 0–1 ไม่ขึ้นกับกัน) กับ **threshold 0.5**
- **อ่าน multilabel ด้วย argmax** Argmax ใช้กับ multiclass สำหรับ multilabel **ทุก** node ที่ ≥ 0.5 คือคำตอบ ในตัวอย่างสไลด์ได้ทั้ง Sci-Fi (0.84) และ Action (0.98)
- **คิดว่า score เดียวกันให้คำตอบเดียวกัน** สไลด์เอา score **ชุดเดียวกัน** [−3.44, 1.66, 3.91] เข้าทั้ง softmax และ sigmoid ค่า softmax รวมได้ 1 และให้คำตอบเดียว ค่า sigmoid (0.03 + 0.84 + 0.98 = 1.85) รวมไม่ได้ 1 และให้หลายคำตอบ
- **คำนวณค่า softmax ในสไลด์ใหม่** Softmax ของ [−3.44, 1.66, 3.91] จริง ๆ ได้ประมาณ [0.001, 0.095, 0.904] ไม่ใช่ [0.010, 0.059, 0.931] แบบในสไลด์ ให้มองตัวเลข softmax ในสไลด์เป็นแค่ตัวอย่าง; ตัวเลข sigmoid ตรง ถ้าต้องคำนวณ ให้ใช้สูตร
- **"เป็นตัวเลข เลยเป็น regression"** อาจารย์เรียกข้อผิดพลาดนี้ว่า "เด็ก ๆ บางคนจะหลงทาง" Regression หมายถึงค่า **ต่อเนื่อง** ตัวเลขแบบ discrete เช่น อายุจำนวนเต็ม หรือ grade ความรุนแรง 0/1/2 มองเป็น multiclass ได้ SEG/SEFS grade ทำนายด้วย head แบบ **classifier** ในสไลด์ p.19 และ p.22
- **ใส่ activation ที่ output ของ regression** Regression ใช้ **Linear** คือ neuron ให้ผลรวมออกมาตรง ๆ ไม่มี f(·) ถ้าใช้ softmax หรือ sigmoid output จะถูกบีบให้อยู่ใน 0–1 และไม่มีทางได้ "2,450,000"
- **ใช้ cross-entropy กับ regression** Regression ใช้ L1/MAE หรือ L2/MSE CE ใช้กับ classification
- **คิดว่า regression มี output node 1 ตัวเสมอ** สไลด์แสดง 1 แต่กฎคือ **ค่าที่ทำนาย 1 ค่าต่อ 1 node** เช่น 2 สำหรับ systolic + diastolic
- **คิดว่ามี regression loss ที่ "ถูก"** ทั้ง L1 และ L2 ไม่ได้ถูกเสมอ ต้องลองทั้งคู่
- **สับสน multilabel กับ multi-task** ทั้งคู่ให้หลายคำตอบ คำตอบของ multilabel มีความหมายเดียวกัน; คำตอบของ multi-task เป็นคนละคำถาม
- **เรียก multi-task ว่า architecture** Multi-task คือ *ปัญหา* Shared extractor + heads (b) เป็นวิธีแก้แบบหนึ่ง; classifier รวมตัวเดียว (a) และ multi-model (c) เป็นอีกแบบ
- **คิดว่า multi-task ช่วยเสมอ** ถ้า task ไม่เกี่ยวกันหรือไม่ขึ้นกัน **negative transfer** ทำให้แย่กว่าโมเดล single-task ได้
- **Loss ของ multilabel** สไลด์และอาจารย์ยอมให้ใช้ "Binary Cross Entropy หรือ Cross Entropy" ถ้าพูดให้เป๊ะ (ตามที่ใช้กันทั่วไป) multilabel ใช้ **binary** CE ทีละ label กับความน่าจะเป็นจาก sigmoid สไลด์วาด vector 0/1 หลัง threshold เข้า loss เพื่ออธิบายเฉย ๆ แต่ตอน train จริง loss คำนวณจากความน่าจะเป็น
- **การสะกดใน transcript** "Multi-Level" = **Multilabel**, "Multi-Card/Multicast" = **Multiclass**, "Signal" = **Sigmoid**, "Feature Attraction" = **Feature Extractor**, "TARG/ทาร์ท" = **Task**, "Test Showing" = **Threshold**, "SEG SEF" = **SEG / SEFS** grades
