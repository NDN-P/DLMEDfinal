# บทที่ 8 — สถาปัตยกรรม CNN แบบ Pre-trained (Transfer Learning) และ Data Augmentation

Deep learning ต้องใช้ training set ใหญ่มาก บทนี้พูดถึงทางแก้ 2 ทาง คือ **Data Augmentation** (สร้างภาพเพิ่มให้เยอะขึ้นและหลากหลายขึ้น: distortion, occlusion, ผสมหลายภาพ และตารางแบบ dynamic/progressive) กับ **Transfer Learning** (เอา backbone ที่ pretrain บน ImageNet มาใช้ต่อ)
จากนั้นไล่ดูว่า operator อะไรที่ทำให้ backbone ดัง ๆ แต่ละตัวพิเศษ: AlexNet, VGG, SqueezeNet, GoogLeNet/Inception, ResNet, DenseNet และ MobileNetV3 โดยเน้นมากเป็นพิเศษที่ **1×1 convolution**

> ที่มา: สไลด์ 89 หน้า (ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์, IT KMITL) + คลิปบรรยายประมาณ 2 ชม. 8 นาที สไลด์ที่มีแต่รูป (56 กราฟ error ของ ImageNet, 57 กราฟ accuracy เทียบ speed, 69 แผนผัง layer ของ AlexNet/VGG/GoogLeNet, 15 สูตร MixUp, 72 แผนผัง ResNet-18, 81 growth rate) อ่านค่าจากภาพสไลด์

---

## แนวคิดหลัก

### 1. ทำไมต้องใช้เทคนิคพวกนี้
- DL ทำงานได้ดีกับ recognition, detection, segmentation และ denoising แต่ข้อจำกัดหลักคือ **ต้องใช้ภาพสำหรับเทรนเยอะมาก** โมเดลที่ใช้งานจริงใช้ภาพเทรนถึง **1,000,000** ภาพ (สไลด์) ส่วนอาจารย์บอกว่า "อย่างน้อยก็ 10 ล้าน" แล้วแต่โจทย์
- ทางแก้มี 2 ทาง **Transfer Learning** คือเอาความรู้เดิมมาใช้ซ้ำ **Data Augmentation** คือสร้างตัวอย่างใหม่จากที่มีอยู่
- อาจารย์เปรียบ transfer learning ว่า: ไม่ต้องฝึกหมอตั้งแต่อนุบาล เอา "เด็กจบ ม.6" (คือ **base model** ที่เทรนบน ImageNet เก่งเรื่องทั่ว ๆ ไป) มาเรียนต่ออีก 4–6 ปีเฉพาะทาง (dataset ของเรา) แบบนี้ใช้ข้อมูลน้อยกว่า เวลาน้อยกว่า และทรัพยากรน้อยกว่า

### 2. Data Augmentation: ใช้ทำอะไร
- สไลด์: ถึงจะใช้ transfer learning (LeNet, AlexNet, ResNet, VGGNet, Inception) แล้ว ถ้า dataset มี **น้อยกว่า 1,000 ภาพ** ก็ยังไม่พอ ตัวอย่าง: เพิ่ม 500 ภาพเป็น 1,000 ด้วยการแปลงเรขาคณิต 2 มิติ (shift, rotate, scale, flip LR/UD, random crop)
- Augmentation บังคับให้โมเดลหา **feature ใหม่ ๆ** แทนที่จะพึ่งแค่ไม่กี่ตัว
- อาจารย์บอกว่ามีประโยชน์ 2 อย่าง ("ใช้สองประโยชน์"):
  1. **เพิ่มจำนวนตัวอย่าง**: ข้อมูลน้อย หรือ **class imbalance** หนัก ๆ ตัวอย่างของอาจารย์: มอเตอร์เขื่อนเอียงผิดปกติ เจอแค่ครั้งเดียวใน 10 ปี
  2. **เพิ่มความหลากหลาย** ถึงจะมีตัวอย่างเยอะอยู่แล้ว เพื่อให้โมเดลรับ input จริงได้ ("ทนมือ ทนตีน User")

### 3. กลุ่มของ Augmentation
ในสไลด์แบ่งเป็น Distortion, Image Occlusion และเทคนิคของ YOLOv4 ส่วนอาจารย์จัดเป็น **3 กลุ่ม**:

| กลุ่ม | ทำอะไร | สมาชิก | ข้อมูลยังอยู่ "ในกรอบของตัวเอง" ไหม? |
|---|---|---|---|
| 1. **Distortion** | *ปรับ* รูปร่างหรือสี ไม่มีอะไรหาย | **Photometric**: contrast, sharpness, blurring, brightness, สี/saturation, noise · **Geometric**: cropping, rotation, scaling, translation, flip | ใช่ |
| 2. **Image Occlusion** | *ซ่อน/ลบ* บางส่วนของภาพ | Random Erase, Cutout, Hide-and-Seek, GridMask | ใช่ (ข้อมูลหายไปบางส่วน) |
| 3. "กลุ่ม 3" (อาจารย์ไม่ได้ตั้งชื่อ) | *สร้างภาพใหม่จากภาพต้นทาง >1 ภาพ* | MixUp, CutMix, Mosaic | ไม่ใช่ ผสมจาก 2–4 ภาพ |

- **Photometric distortion** เปลี่ยนค่า pixel **Geometric distortion** เปลี่ยนรูปร่าง
- Zoom-in = crop แล้ว resize ขยาย แบบนี้โอเค **Zoom-out** ในทางปฏิบัติขอบจะถูกเติมเป็น **สีดำ** เพราะนอกภาพเดิมไม่มีข้อมูลจริง
- ทำไม occlusion ได้ผล (ตัวอย่างหมาด้วย Grad-CAM): โมเดลดูแค่ **หัว** หมา พอเจอรูปไม่มีหัวก็ทายผิด ("ลักไก่" คล้าย overfitting) ถ้าสุ่มซ่อนคนละที่ในแต่ละ epoch จะบังคับให้มันไปใช้ส่วนอื่น (เช่นขา) ช่วยลด overfitting และเพิ่ม regularization แต่ละ epoch ซ่อนคนละที่ เหมือนได้ "ข้อสอบใหม่" ทุกครั้ง โมเดลเลยจำไม่ได้

#### วิธี Occlusion
| วิธี | คืออะไร | อะไรที่สุ่ม |
|---|---|---|
| **Random Erase (RE)** | แทนที่บางบริเวณด้วยค่าสุ่ม หรือค่า pixel **mean** ของ training set ขนาดและสัดส่วนหลากหลาย เป็นวิธีมาตรฐานกัน overfitting | **จำนวนกล่อง, ตำแหน่ง, ขนาด** และ **ค่าที่เติม** (0 / mean / random) |
| **Cutout** | ตัด **สี่เหลี่ยมจัตุรัสขนาดคงที่** ตลอดการเทรน เป็นตัวปรับปรุงจาก RE | **สุ่มแค่ตำแหน่ง** ขนาด จำนวน และค่าที่เติมคงที่ เลย **รับประกันได้ว่าข้อมูลเหลือกี่ %** (RE อาจลบไปถึง 80 %) |
| **Hide-and-Seek (HaS)** | แบ่งภาพเป็น **N×N patch** แล้วซ่อนแต่ละ patch ตามความน่าจะเป็น | patch ไหนถูกซ่อน จุดอ่อน: patch ที่ถูกซ่อนอาจ **ติดกัน** ตัวทั้งตัวเลยหายได้ (เหลือแค่พื้นหลัง + หัว) |
| **GridMask** | ซ่อนเป็นลาย **grid/ตาข่าย** สม่ำเสมอ (คล้าย HaS แต่กระจายออก) | เรากำหนดขนาด grid เอง รับประกันว่าทุกบริเวณยังมีข้อมูลเหลืออยู่บ้าง |

#### วิธีที่ใช้หลายภาพ
- **MixUp**: ซ้อนสองภาพด้วยน้ำหนักความโปร่งใส ออกแบบมาสำหรับ classification แต่ใช้กับ detection ได้ด้วย **label ก็ผสมด้วย**: แมว 0.7 + หมา 0.3 โมเดลที่ดีควรทายแมว 0.7 / หมา 0.3 ค่าที่ดีที่สุด **α = 0.2, 0.3, 0.4** (สไลด์)
- **CutMix**: ตัด patch จากภาพเทรนหนึ่งไปแปะบนอีกภาพ (ตัวอย่าง St. Bernard + Poodle) ช่วยเลี่ยงการเสียข้อมูลแบบตัดทิ้งเฉย ๆ และช่วยให้ **object detection** ดีขึ้น
- **Mosaic** (ใช้ครั้งแรกใน **YOLOv4**): "รวมหลายภาพให้เป็นภาพเดียว" รวม **4 ภาพ** เป็นภาพเดียว ขั้นตอน:
  1. Resize ทั้ง 4 ภาพเป็น **256×256**
  2. ต่อกันเป็นภาพเดียวขนาด **512×512**
  3. เลื่อน bounding box: บนซ้ายไม่เปลี่ยน; บนขวา x+256; ล่างซ้าย y+256; ล่างขวา x+256 และ y+256
  4. สุ่ม crop เป็น **สี่เหลี่ยมจัตุรัส** (สไลด์: ขนาด 16–240 px) ที่ **ต้องโดนทั้ง 4 ภาพ** สัดส่วนไม่ต้องเท่ากัน
  5. ลบ box ที่ถูก crop ทิ้งไป
  6. เก็บส่วนที่ไม่ถูก crop แล้ว resize ให้เท่าขนาด input ของโมเดล
- ใช้กลุ่ม 3 ตอนไหน: **object detection** หรือ classification ที่ **class หน้าตาคล้ายกันมาก** (ตัวอักษรไทยแบบ ข/ฃ, ส้มกับมะนาว) CutMix กับ Mosaic ใช้กับ detection เป็นหลัก
- ขั้นต่ำที่อาจารย์แนะนำตอนนี้: ใช้ **Distortion + Occlusion** กลุ่ม 3 ใช้เฉพาะตอน dataset ยาก

### 4. Augmentation แบบ Static กับ Dynamic
| Static (แบบดั้งเดิม, Fixed Policy) | Dynamic (ต่อ epoch) |
|---|---|
| ใช้ policy เดียวตลอดการเทรน | ความแรง/ชนิดเปลี่ยนตาม epoch |
| ใช้กับงานทั่วไปได้ | เหมาะกับ curriculum / robust learning |
| ไม่รู้สถานะของโมเดล | ปรับตาม loss, accuracy หรือความยากของ sample |

**ปัญหาของ augmentation แบบดั้งเดิม**
- ใช้ augmentation เหมือนเดิมทุก epoch: *Static Configuration*, *One-Size-Fits-All* (เช่น rotation 30°, brightness ±0.3 ตั้งแต่ epoch 1 จนจบ), *ไม่ปรับตัว*
- ไม่สนสถานะโมเดล: ไม่เห็น *Learning Progress*, ไม่สน *Model Capacity*, *ไม่มี Monitoring*
- **ช่วงต้นการเทรน** (โมเดลยังอ่อน แต่ distortion หนัก): *Overwhelming Complexity* ("สอนเด็กอนุบาลด้วยตำรามหาลัย"), *Gradient Instability* (loss ไม่นิ่ง converge ช้าหรือไม่ converge เลย), *Feature Learning Disruption* (ขอบ/รูปทรง/สีพื้นฐานถูกรบกวน)
- **ช่วงท้ายการเทรน** (โมเดลเก่งแล้ว แต่ข้อมูลง่าย ประมาณ epoch 80–100): *Insufficient Challenge* (ทำให้ overfitting; "นักศึกษามหาลัยทำแบบฝึกหัดอนุบาล"), *Missed Generalization Opportunity*, *Resource Wastage*
- ผลกระทบ: **Performance** (converge ช้า, accuracy สุดท้ายต่ำ, generalize ไม่ดี), **Resources** (เทรนนานขึ้น, เปลือง compute, ต้นทุนสูง), **Stability** (เทรนไม่นิ่ง, ผลไม่คงที่, ต้องรันหลายรอบ)
- สรุป (สไลด์): augmentation แบบ fixed policy เริ่มไม่นิยมแล้ว ควรปรับ augmentation ให้เข้ากับระดับการเรียนรู้ของโมเดล

**ระดับ Weak → Strong** (อาจารย์: เป็นแนวทางคร่าว ๆ "ใช้กันเองในครัวเรือน" ไม่ใช่ค่าตายตัว)
| ระดับ | ตัวอย่าง | ผล |
|---|---|---|
| Very Mild | flip H/V, ปรับ brightness/contrast นิดหน่อย | ความหมายไม่เปลี่ยน |
| Mild | rotate ±5–10°, translate ±5 %, colour jitter ต่ำ, Gaussian noise std ต่ำ, random crop + resize | รูปทรงหลักยังอยู่ |
| Moderate | shear ±10–20°, Cutout เล็ก ๆ, AutoContrast/Equalize, brightness/contrast ปานกลาง | เปลี่ยนรูปทรงและสี |
| Strong | rotate ±30–45°, colour jitter แรง, CutMix / MixUp / Random Erasing ขนาดใหญ่, Posterize/Solarize, translate ±30–50 % | เปลี่ยนเยอะ จำแนกยากขึ้น |
| Very Strong | rotate ±90°+, perspective warp, noise std สูง, blur แรง, JPEG artefacts, adversarial transforms | ห่างจากข้อมูลจริงมาก ใช้กับโมเดลที่มั่นใจแล้วเท่านั้น |

ระดับ mild เป็น Distortion ล้วน ๆ ส่วน Moderate กับ Strong มี Occlusion และกลุ่ม 3 ปนเข้ามา อาจารย์บอกว่า Very Strong แทบไม่ได้ใช้ เพราะอาจเปลี่ยนความหมายของภาพ ในทางปฏิบัติหยุดแค่ Strong

**วิธี dynamic 3 แบบ**
| วิธี | ทำงานยังไง | จุดเด่น |
|---|---|---|
| **Progressive** | เพิ่มความแรง **ตามเวลา (epoch)** ไม่สนสถานะจริงของโมเดล | ง่ายไปยาก (curriculum learning) |
| **Adaptive Scheduling** | ปรับตาม **training loss / validation metric** | โมเดลเป็นตัวกำหนดการปรับ |
| **Sample-Aware** | ปรับตาม **ความยากของ class หรือ sample** | ความยุติธรรม / ปรับเฉพาะตัว |

- ตัวอย่าง **Progressive**: epoch 1–20 ใช้ rotation ±5°, brightness ±0.1 epoch 21–50 ใช้ ±15°, ±0.2 และเพิ่ม Gaussian noise epoch 51+ ใช้ augmentation เต็มที่ (±30°, Cutout, MixUp) ข้อดี: เลี่ยง **underfitting** ช่วงต้น และเป็นไปตาม curriculum learning ต้องลองรันก่อนรอบหนึ่งถึงจะรู้ว่าควรแบ่ง stage กี่ epoch
- **Adaptive Scheduling**: ถ้า validation loss ยังสูง ให้ augmentation อ่อนไว้ก่อน พอ loss ลดลงค่อยเพิ่มความแรงเพื่อกัน overfitting สัญญาณว่าโมเดล "สบายใจ" เกินไป:
  - val loss ไม่ลดลง **5 epoch ติดกัน**
  - accuracy ไม่เพิ่มเกิน **threshold** ที่ตั้งไว้
  - gradient เล็กเกินไป

  ข้อดี: augmentation เข้ากับช่วงการเรียนรู้ และลดความเสี่ยง augment มากหรือน้อยเกินไป
- **Sample-Aware**:
  - *Difficulty-aware ต่อ sample*: confidence **> 0.9** ได้ **Strong** ทายผิดหรือ confidence **< 0.6** ได้ **Weak** ตรงกลางได้ Medium
  - *Class-aware*: class ที่ **accuracy ต่ำ** ได้ **Strong** (ข้อมูลเพิ่ม ท้าทายมากขึ้น) class ที่ **accuracy สูง** ได้ **Weak** (กัน noise)

### 5. Transfer Learning
- **นิยาม**: เอาบางส่วนของโมเดลที่เทรนแล้วมาต่อกับโมเดลใหม่ การเทรนจากศูนย์ใช้เวลาเป็นวันถึงเป็นเดือน การเอาโมเดลที่เทรนกับงานคล้าย ๆ กัน (**pretrained network**) มาใช้ซ้ำจึงลดเวลาเทรนได้มาก
- แหล่งที่ใช้กันทั่วไปคือ **ImageNet: 1.4 M ภาพ, 1,000 class** สมมติฐานหลัก: **feature ทั่วไปที่เรียนจาก dataset ใหญ่ ใช้ร่วมกับ dataset ที่ดูต่างกันได้** โมเดล ImageNet ที่เปิดให้ใช้ เช่น AlexNet, VGG, ResNet, Inception และ DenseNet
- คำศัพท์ (อาจารย์): **Backbone/Architecture** คือโครงสร้าง layer ของโมเดล **Backbone + Pre-trained** คือโครงสร้างนั้นพร้อม weight ที่เทรนบน ImageNet แล้ว
- CNN มี 2 ส่วน **convolutional base** ใช้ดึง feature **FC layers** ใช้จำแนก
- **วิธีใช้ pretrained model 2 แบบ** (Yamashita et al. 2018, review ด้าน radiology):
  1. **Fixed feature extraction**: ตัด FC layers ออก เก็บ conv base ไว้แบบ **frozen** เป็นตัวดึง feature เอา classifier อะไรก็ได้มาวางบน (random forest, SVM หรือ FC layers ใหม่) แล้วเทรนเฉพาะ classifier นั้น วิธีนี้ **ไม่ค่อยใช้ใน DL ภาพการแพทย์ เพราะ ImageNet กับภาพการแพทย์ต่างกันมาก**
  2. **Fine-tuning**: เปลี่ยน FC layers และ **เทรน kernel ใน conv base ใหม่ด้วย ทั้งหมดหรือบางส่วน ผ่าน backprop** ปกติจะ freeze layer ต้น ๆ แล้วเทรน layer ลึก ๆ layer ต้นเก็บ **feature ทั่วไป (ขอบ)** ส่วน layer หลังเป็น **เฉพาะงาน** วิธีนี้ใช้มากกว่าในงานวิจัย radiology
- **ข้อเสีย**: มีข้อจำกัดเรื่องขนาด input ต้องเป็น **RGB (3-channel)** เพราะ ImageNet เป็น RGB แต่ภาพการแพทย์ grayscale มี **1 channel** ส่วนความสูงกับความกว้างจะเป็นเท่าไหร่ก็ได้ (ไม่เล็กเกินไป) ถ้าใส่ **Global Pooling layer** ไว้ระหว่าง conv base กับ FC layers ใหม่
- **3 รูปแบบ** (สไลด์ 51–53 บวกคำอธิบายของอาจารย์):

| แบบ | คำอธิบายในสไลด์ | อาจารย์ตีความว่า |
|---|---|---|
| 1 | ใช้ pretrained model **ทั้งตัว** แล้วเทรนต่อกับข้อมูลใหม่ ไม่ตัดอะไรออก | ใช้ทุกอย่างตามเดิม (frozen ทั้งหมด) ในทางปฏิบัติ "แทบจะ Impossible" เพราะจำนวน output node ≠ จำนวน class ของเรา ยกเว้นเป็นงานเดียวกัน |
| 2 | ใช้แค่ส่วน **feature-extraction**: ตัด Dense layer สุดท้ายออก แล้วสร้างและเทรน **classifier ใหม่** บน dataset ที่เล็กกว่าหรือเฉพาะกว่า | **Freeze** ตัวดึง feature แล้วเทรนแค่ classifier เป็นวิธีที่ **ใช้บ่อยที่สุด** เพราะ feature ภาพ (สี รูปทรง texture) "common" กันทุกงาน |
| 3 | ใช้ **บางส่วน** (โดยเฉพาะ feature extraction) ของโมเดลจากงานที่คล้ายกัน | Freeze conv layer **ต้น ๆ** แล้วให้ conv layer **หลัง ๆ** เทรนได้ บวก classifier ใช้ตอนภาพของเรา **ไม่เหมือนภาพธรรมชาติใน ImageNet** เช่น **ultrasound หรือ X-ray** |

- **Freeze = ส่วนนั้นไม่ถูกเทรน** (อาจารย์: "กาดอกจันทร์คำว่า freeze") output layer ต้องเทรนใหม่เสมอ เพราะจำนวน node ต้องเท่ากับจำนวน class ของเรา
- ทำไมแบบ 3 ถึง freeze layer ต้น ๆ: layer ต้นดึง feature ระดับต่ำ (เส้น สี) ที่ภาพทุกชนิดมีเหมือนกัน layer หลังเอามันมารวมเป็นวัตถุ ซึ่งการรวมนี้ต่างกันระหว่าง ultrasound, X-ray และภาพถ่าย จะเทรนทั้งหมดก็ *ได้* แต่นี่คือเหตุผลที่ freeze บางส่วน
- **การเลือก pretrained model**: ดู **Accuracy, Speed, Size** (สไลด์ 54) ตัวเลือก: AlexNet, ResNet, VGG-19, GoogLeNet, MobileNet, Inception, DenseNet
- **กราฟสไลด์ 57** พล็อต accuracy บน ImageNet (แกน y) เทียบกับเวลาทำนายบน GPU แบบเทียบกัน (แกน x, log scale) ขนาดฟอง = ขนาดโมเดล ค่าข้างล่างอ่านจากกราฟแบบประมาณ:

  | Model | Accuracy | Speed / size |
  |---|---|---|
  | SqueezeNet | ~55 % | เร็วที่สุด, เล็กมาก |
  | AlexNet | ~54 % | เร็ว, ฟองใหญ่ |
  | ShuffleNet | ~64 % | เล็ก |
  | GoogLeNet | ~66 % | – |
  | ResNet-18 | ~69.5 % | – |
  | MobileNet-v2 | ~70.5 % | เล็ก, เร็ว |
  | VGG-16/19 | ~70 % | **ฟองใหญ่ที่สุดแบบทิ้งห่าง**, ช้า |
  | ResNet-50 | ~74.5 % | – |
  | ResNet-101 | ~76 % | – |
  | Inception-v3 | ~77 % | – |
  | DenseNet-201 | ~76 % | ช้า (~23× time) |
  | Inception-ResNet-v2 | ~80 % | – |
  | NASNet-Large | ~82 % | ช้าที่สุด, ใหญ่ |

  โมเดลที่สอนในห้องจะขีดเส้นใต้ไว้ในสไลด์ ข้อสรุป:
  - accuracy สูงขึ้นมักแลกกับความเร็วและ/หรือขนาด
  - VGG ใหญ่มากเมื่อเทียบกับ accuracy ที่ได้
  - SqueezeNet กับ MobileNet เล็กและเร็ว

### 6. ประวัติและการตั้งชื่อ (อาจารย์)
- การแข่ง ImageNet classification ปี 2010–2011 ใช้วิธี ML ได้ accuracy ~70+ % ปี **2012 AlexNet** (CNN) ลด error จาก **25 % เหลือ 16 %** (accuracy ~84 %) และเริ่มยุคบูม ปี 2013 มี AlexNet ที่ปรับปรุงแล้ว **2014 VGG และ GoogLeNet** **2015 ResNet** หลัง ResNet ก็เลิกแข่ง เพราะ accuracy ถึง ~96–97 % แล้ว
- **กราฟสไลด์ 56** (ILSVRC top-5 error %, จาก He et al. 2015):

  | ปี | Model | Top-5 error | ความลึก |
  |---|---|---|---|
  | '10 | – | 28.2 | ตื้น |
  | '11 | – | 25.8 | ตื้น |
  | '12 | AlexNet | **16.4** | 8 layers |
  | '13 | – | 11.7 | 8 layers |
  | '14 | VGG | **7.3** | 19 layers |
  | '14 | GoogLeNet | **6.7** | 22 layers |
  | '15 | ResNet | **3.57** | **152 layers** |

  network ยิ่งลึก error ยิ่งต่ำ ที่อาจารย์พูดว่า "25 % เหลือ 16 %" คือ top-5 error ตัวนี้
- **LeNet** คือ CNN ตัวแรกจริง ๆ แต่ใช้แค่กับตัวเลข 0–9 AlexNet เป็นตัวที่ "บูม" เพราะใช้กับภาพสีได้ ตัว **L** ใหญ่ใน Google**L**eNet ตั้งไว้เพื่อให้เกียรติ LeNet
- ยุคของการตั้งชื่อ:
  1. ชื่อเดี่ยว (AlexNet, SqueezeNet, ShuffleNet, GoogLeNet)
  2. จำนวน layer (ResNet-18, VGG-16/19)
  3. ตัวอักษรบอกขนาด (S/M/L, Base/Large/Tiny) เพราะตัวเลขไม่ได้บอกผู้ใช้ว่าโมเดลใหญ่แค่ไหน

  ปัจจุบัน: EfficientNet; ที่ไม่ใช่ CNN: Transformers
- ชื่อเรียกกลุ่มของ layer: **Module** (SqueezeNet *Fire module*) ต่อมาเป็น **Cell** (GoogLeNet *Inception cell*) แล้วเป็น **Block/Stage** (ตั้งแต่ ResNet) ทั้งหมดหมายถึงกลุ่ม layer ที่ทำงานด้วยกัน

### 7. สถาปัตยกรรม

| Architecture | ปี / ผู้สร้าง | ความลึก | ไอเดียหลัก | หมายเหตุ |
|---|---|---|---|---|
| **LeNet** | CNN ตัวแรก | – | จำตัวเลข 0–9 | อาจารย์พูดถึงแค่ปากเปล่า |
| **AlexNet** | **2012**, Alex Krizhevsky et al., ImageNet | **8** = 5 conv + 3 FC | **ใช้ ReLU แทน Tanh**; **Dropout** กัน overfitting (แต่ใช้ compute ~**2×**) | **60 M params** เลยเจอปัญหา **overfitting** |
| **VGG** | **2014**, Simonyan et al., Oxford | 13 conv + 3 FC (VGG-16); VGG-19 = 19 | ศึกษา **ผลของความลึกต่อ accuracy**; **filter ทั้งหมด 3×3** (เชื่อว่าดีเท่า 7×7); input **224×224 RGB**; preprocessing: **ลบด้วยสีเฉลี่ยของ ImageNet**; **ใช้ 1×1 conv ทำ prediction** | อาจารย์: เป็นตัวแรกที่ normalize ด้วย (x−mean)/sd จาก ImageNet และตัวแรกที่ classify ด้วย conv แทน FC network |
| **SqueezeNet** | **2016** | – | **Fire module** = **squeeze** (1×1) + **expand** (1×1 และ 3×3 แล้ว concat) | เป้าหมาย: ประหยัด memory/ทรัพยากร |
| **GoogLeNet / Inception** | **2014**, Google | **22** | **Inception cell 9 ตัว**; **1×1, 3×3, 5×5 conv + 3×3 max-pool** วางขนานกัน แล้ว concat output (**multi-scale**); **1×1 conv ก่อน 3×3/5×5 และหลัง max-pool** เพื่อลด compute; **Global Average Pooling** ก่อน output (7×7 เป็น 1×1) | ประหยัด compute: params และ operation น้อยลง |
| **ResNet (ResNet-18)** | **2015**, Microsoft Research | 18 | **Residual learning**: **Convolution Block, Identity Block, Skip Connection**; **element-wise addition** | แก้ vanishing/exploding gradient ทำให้เทรน net ลึกมาก ๆ ได้ block ของมันถูกเอาไปใช้ต่อใน segmentation network ภายหลัง |
| **DenseNet** | **2016**, Gao Huang et al. | – | **Dense blocks** (แต่ละ layer รับ map จาก **ทุก layer ก่อนหน้า**, **channel-wise concatenation**); **Transition layers**; **growth rate k** | Feature reuse; ใช้ memory มากขึ้น |
| **MobileNetV3** | – | – | **Inverted residual block** (1×1 expansion, depthwise separable conv, 1×1 projection, skip) + **Squeeze-and-Excite** | สำหรับมือถือ / เครื่องสเปกต่ำ |

#### ลำดับ layer (สไลด์ 69 "เปรียบเทียบสถาปัตยกรรม AlexNet, VGG และ GoogLeNet")
- **AlexNet, 8 layers (ILSVRC 2012)**:
  1. 11×11 conv, 96, stride 4, pool/2
  2. 5×5 conv, 256, pool/2
  3. 3×3 conv, 384
  4. 3×3 conv, 384
  5. 3×3 conv, 256, pool/2
  6. fc 4096
  7. fc 4096
  8. fc 1000

  รวมเป็น 5 conv + 3 FC โดยมี **filter ใหญ่ช่วงต้น** (11×11, 5×5)
- **VGG, 19 layers (ILSVRC 2014)**: ใช้แค่ 3×3 conv แบ่งเป็น stage 64 ×2, 128 ×2, 256 ×4, 512 ×4, 512 ×4 แต่ละ stage จบด้วย pool/2 จากนั้น fc 4096, fc 4096, fc 1000 รวม 16 conv + 3 FC = 19 ส่วน VGG-16 (ข้อความสไลด์ 59) มี 13 conv + 3 FC
- **GoogLeNet, 22 layers (ILSVRC 2014)**: Inception cell ซ้อนกันสูง ๆ ในแผนผังมีกิ่งข้างอีก 2 กิ่งด้วย

#### 1×1 convolution (operator ตัวโปรดของอาจารย์)
- ดูแค่ pixel ตัวเอง ไม่ดูเพื่อนข้าง ๆ และ **รวม feature ของ pixel นั้นจากทุก channel**
- ใช้ทำอะไร:
  1. **เปลี่ยน (ส่วนใหญ่คือลด) จำนวน channel** "เมื่อไหร่ที่ต้องการปรับจำนวน channel เราใช้ 1×1 เสมอ" อาจารย์เทียบกับ PCA
  2. channel น้อยลง = **memory น้อยลง params น้อยลง compute น้อยลง** ซึ่งก็คือ **overfitting น้อยลง** ด้วย
  3. **Classification** (VGG: 1×1 filter 1,000 ตัว ให้คะแนน 1,000 class)
- ประวัติ (สไลด์ 66): 1×1 conv เริ่มใน **Network in Network (NIN)** เพื่อเพิ่ม **non-linearity** (ใช้คู่กับ ReLU) GoogLeNet เอามาใช้ **ย่อ feature map และลด compute**

#### SqueezeNet Fire module
- ไอเดียการออกแบบ 3 ข้อ:
  1. เปลี่ยน filter 3×3 เป็น **1×1 (point-wise)** ได้ **parameter น้อยลง 9×**
  2. 1×1 ลดความลึกของ feature map และเวลาคำนวณ
  3. **Late (delayed) pooling** ทำให้ feature map ใหญ่อยู่นานขึ้น ช่วยให้ performance ดีขึ้น
- **Squeeze** = 1×1 conv ที่ **ลด channel** (ตัวอย่าง: 6 เหลือ 2) **โดยไม่เปลี่ยน W×H** เป็นการลดมิติด้วยการ *รวม* channel ไม่ใช่เลือก channel
- **Expand** = filter 1×1 (**point features**) กับ filter 3×3 (**local features**: pixel บวกเพื่อนข้าง ๆ) รันแยกกันแล้ว **concat**
- ลำดับ: squeeze, ReLU, expand, concat, ReLU

#### GoogLeNet Inception cell
- **Multi-scale**: filter เล็กจับวัตถุเล็กและรายละเอียด filter ใหญ่จับวัตถุใหญ่ ใช้หลายขนาด filter ใน layer เดียวแล้ว concat เลยได้ทั้งสองอย่าง
- 1×1 ก่อน 3×3/5×5 ลด channel เพื่อไม่ให้จำนวน parameter "บวม" (bloat)
- **ทำไมต้องมี 1×1 หลัง 3×3 max-pool?** Max pooling ทำ **ทีละ channel**: ไม่ผสม channel และไม่เปลี่ยนจำนวน channel เลยต้องใช้ 1×1 conv ตามหลังเพื่อลด channel ส่วน conv filter (1×1/3×3/5×5) ใช้ **ทุก channel**
- **Global Average Pooling (GAP)**: ทุก channel เหลือค่าเดียว (เฉลี่ย map 7×7 เป็น 1×1)
  - **ทำไม**: conv layer รับ input ขนาดไหนก็ได้ แต่ **FC network มีจำนวน input node ตายตัว** ถ้ามี GAP ขนาด input เปลี่ยนก็แค่ W×H เปลี่ยน จำนวน channel ไม่เปลี่ยน input ของ FC เลยคงที่
  - ช่วยลด compute ด้วย ต่อมาขยายเป็น **Adaptive** (global) average pooling ที่ output เป็น N×N

#### รูปแบบการเชื่อมต่อ (สไลด์ 70–71)
| รูปแบบ | ยังไง | ข้อดี | ข้อเสีย |
|---|---|---|---|
| **Standard** | conv ต่อกันเรื่อย ๆ output ไปแค่ layer ถัดไป | ง่าย | Vanishing/exploding gradient ใน net ที่ลึกมาก |
| **ResNet** | **Skip/residual (shortcut) connection**; output ที่ข้ามไป **บวกแบบ element-wise** (⊕) กับ layer ถัดไปข้างหลัง | ลด vanishing gradient ทำ net ลึกขึ้นได้ | ซับซ้อนขึ้นนิดหน่อย; shape ที่จะบวกกัน **ต้องเท่ากัน** |
| **DenseNet** | แต่ละ layer รับ input จาก **ทุก layer ก่อนหน้า**; **channel-wise concatenation** (Ⓒ) | เรียน feature ได้มีประสิทธิภาพกว่า | ใช้ **memory มากขึ้น** เพื่อเก็บ feature ทั้งหมด |

- เหตุผลของอาจารย์: filter *ควรจะ* ทิ้ง noise และเก็บข้อมูลที่มีประโยชน์ แต่ **ไม่มีอะไรรับประกัน** ว่าข้อมูลสำคัญไม่ได้ถูกกรองทิ้งไปตั้งแต่ต้น การเอา input ที่ยังไม่ผ่านการประมวลผลบวกกลับเข้าไปช่วยกัน **Information Loss** และเสริม (re-weight) feature ที่มีประโยชน์
- การเชื่อมของ ResNet และ DenseNet มี **แค่ภายใน block** ไม่ได้ข้ามทั้งโมเดล

#### Block ของ ResNet-18 (สไลด์ 72–73)
- **โครงสร้างรวม (รูปสไลด์ 72)**:
  1. ภาพ input 224×224
  2. 7×7 conv, 64, stride 2
  3. pool, stride 2
  4. 4× (3×3 conv, 64)
  5. 4× 3×3 conv, 128 (ตัวแรก stride 2)
  6. 4× 3×3 conv, 256 (ตัวแรก stride 2)
  7. 4× 3×3 conv, 512 (ตัวแรก stride 2)
  8. Average pool แล้ว fully connected แล้ว softmax

  นับได้: 1 + 16 + 1 = **18 layer ที่มี weight**
- ลูกศร skip คร่อมทุก 2 conv ลูกศร **เส้นทึบ** คือ identity shortcut ลูกศร **เส้นประ** คือจุดที่ขนาดเปลี่ยน (/2, channel มากขึ้น) ซึ่งเป็น shortcut ของ convolution block (1×1 conv stride 2)
- **Convolution Block** ใช้ตอนที่ขนาด output ต้องเปลี่ยน conv ตัวแรกเป็น **3×3, stride 2, padding 1**: (56×56×64) กลายเป็น (28×28×128) conv ตัวที่สองเป็น 3×3, stride 1, padding 1 ซึ่งคงขนาด 28×28×128 ไว้ **shortcut ต้องมี 1×1 conv stride 2** (64 เป็น 128 channel) เพื่อให้ output เดิม 56×56×64 กลายเป็น 28×28×128 แล้วบวกกันได้ จากนั้น ReLU
- **Identity Block**: ขนาด input = ขนาด output shortcut เลยส่ง input ไปตรง ๆ ไม่เปลี่ยน (*identity*) บวก แล้ว ReLU
- รูปแบบ: Conv block (A) ตามด้วย Identity blocks (B, B, B) มีแค่ block A ที่ลดขนาด
- เกม "จับผิดภาพ" ของอาจารย์: Conv block กับ Identity block ต่างกัน **2 จุด**: (1) 3×3 conv ตัวแรกใช้ **stride 2** แทน 1; (2) shortcut มี **1×1 conv stride 2**
- บทเรียนด้านการออกแบบ:
  1. ResNet **ลดขนาดด้วย conv stride 2 แทน pooling**
  2. Skip connection ลดการสูญเสียข้อมูล
  3. shortcut แบบ 1×1 stride 2 หยิบ pixel เว้น pixel และผสม channel ข้อมูลที่ "บวกกลับ" เลย **ไม่ใช่ข้อมูลเดิม**: ถูกลดขนาดและรวมใหม่แล้ว ทาง shortcut พา feature แบบ **point** (1×1) ทางหลักพา feature แบบ **local** (3×3)

#### DenseNet (สไลด์ 74–81)
- **Basic composition layer**: **pre-activation** BN แล้ว ReLU แล้ว **3×3 conv** ที่ให้ output **k** channel
- **Bottleneck layer**: BN, ReLU, **1×1 conv** (ลด channel) แล้ว BN, ReLU, **3×3 conv** (ดึง feature ใหม่) params และ compute น้อยลง
- **Dense connectivity**: ทุก layer ใน block มีขนาดเชิงพื้นที่เท่ากัน map เลย concat กันได้ ข้อดี:
  - **Feature หลากหลายขึ้น**
  - classifier ใช้ feature ได้ทุกระดับความซับซ้อน ช่วย **เก็บ feature ความซับซ้อนต่ำไว้** ได้ decision ที่เรียบขึ้น และ **ทำงานได้ดีกับข้อมูลเทรนน้อย**
  - **Feature reuse** ทำให้ parameter น้อยลง
- **Transition layer** (อยู่ระหว่าง dense block): BN, ReLU, **1×1 conv** (ลด channel ที่พอกขึ้นจากการ concat) แล้ว **2×2 average pooling** (ลด W×H)
- อาจารย์บอกว่านี่เป็น **สถาปัตยกรรมแรกที่ใช้ average** แทน max pooling อาจารย์เดาว่าเพราะการเฉลี่ยช่วยกดค่าพีคที่เป็น noise
- **Growth rate (k)**: จำนวน feature map ที่แต่ละ layer **เพิ่ม** เข้าไปใน dense block ในรูปสไลด์ 81:
  - x₀ ผ่าน h₁ ได้ x₁
  - [x₀, x₁] concat กันแล้วผ่าน h₂ ได้ x₂
  - [x₀, x₁, x₂] ผ่าน h₃ ได้ x₃ แบบนี้ไปจนถึง x₄
  - กอง feature map โตขึ้นทีละ k ทุก layer
- ประสบการณ์ของอาจารย์: DenseNet ไม่ได้เด่นอะไรนัก ยกเว้นงานที่ซับซ้อนจริง ๆ

#### MobileNetV3 (สไลด์ 82–89)
- **bottleneck block** ช่วยเพิ่ม performance และลด memory ส่วนประกอบ:
  - **Inverted residual block**: **1×1 expansion conv** (เพิ่ม channel ให้ depthwise conv จับ feature เชิงพื้นที่ได้มากขึ้น), **depthwise separable conv**, **1×1 projection conv** (ลด channel กลับ params น้อยลง), **skip connection**
  - block **Squeeze-and-Excite (SE)**
- ตัวอย่างในสไลด์: M×N×3 แล้ว 1×1 expansion ด้วย 16 filter (แต่ละตัว 1×1×3) ได้ M×N×16 Depthwise separable conv แล้ว SE ยังคง M×N×16 1×1 projection ด้วย 3 filter (แต่ละตัว 1×1×16) ได้ M×N×3 แล้วบวก skip connection
- **Depthwise separable conv** ใช้แทน conv ปกติเพื่อย่อโมเดล มี 2 ส่วน:
  - **3×3 Depthwise conv + BN + H-swish**: filter หนึ่งตัวต่อหนึ่ง channel แต่ละตัวทำงานกับ **channel เดียว** params และ compute น้อยลง เทรนเร็วขึ้น
  - **1×1 Pointwise conv + BN + H-swish**: รวม channel ที่แต่ละ pixel ให้เกิด channel ใหม่ **เพิ่มหรือลด** จำนวน channel ได้
- ประเด็นสำคัญของอาจารย์: ใน conv ปกติ ผลของแต่ละ channel แค่ถูก **บวกกัน** (เหมือนให้ weight 1 ทุกตัว) แต่ใน depthwise + pointwise ขั้น 1×1 มี **weight ต่อ channel ที่เรียนได้** channel เลยถูก *ถ่วงน้ำหนัก* ก่อนรวม
- **SE block** (Jie Hu et al., "Squeeze-and-Excitation Networks"): จำลองความสัมพันธ์ระหว่าง channel ให้ network โฟกัส feature map ที่สำคัญ ไอเดียคือ **channel แต่ละตัวสำคัญไม่เท่ากัน เลยต้องให้น้ำหนัก**
  1. **Squeeze** = GAP: H×W×C กลายเป็น **1×1×C**
  2. **Excitation** = FC 2 layer: ตัวแรก **ลด** มิติ ตามด้วย **ReLU** ตัวที่สอง **คืน** มิติกลับเป็น C ตามด้วย **sigmoid** ผลที่ได้คือ weight ต่อ channel
  3. **Recalibration** = เอา weight คูณกับ feature map เดิมแบบ channel-wise

---

## สูตรและตัวเลขที่ต้องจำ

**ขนาด output ของ conv** (สูตรมาตรฐาน ตรงกับตัวอย่างในสไลด์): `O = floor((W − K + 2P) / S) + 1`
- ResNet conv block: W=56, K=3, P=1, S=2 ได้ (56 − 3 + 2)/2 + 1 = 27 + 1 = **28** ดังนั้น 56×56×64 กลายเป็น **28×28×128** (128 filters)
- conv เดียวกันแต่ S=1, P=1 ได้ **56**: 3×3 conv stride 1 padding 1 คงขนาดเดิม ถ้า stride 2 จะ **ลด W×H ลงครึ่งหนึ่ง**
- Shortcut 1×1, S=2, P=0: (56 − 1)/2 + 1 = **28** กับ 128 filters เลยตรงกับ 28×28×128 และบวกกันได้

**GoogLeNet 1×1 bottleneck** (สไลด์ 66; สไลด์เรียกตัวเลขพวกนี้ว่า *number of operations*)
- 5×5 ตรง ๆ 48 filters บน 14×14×480: (14×14×48) × (5×5×480) = **112.9 M**
- ถ้าลดด้วย 1×1 เหลือ 16 channel ก่อน:
  - ขั้น 1×1: (14×14×16) × (1×1×480) = **1.5 M**
  - ขั้น 5×5: (14×14×48) × (5×5×16) = **3.8 M**
  - รวม = **5.3 M** น้อยกว่าประมาณ **20×** (อาจารย์: "ลดไป 20 เท่า")
- filter 5×5 แต่ละตัวมี kernel เท่ากับจำนวน **input channel** (480) ตรงนี้แหละที่ทำให้หนัก

**VGG classification ด้วย conv** (ตัวอย่างของอาจารย์)
- 14×14×512 แล้ว max-pool เป็น **7×7×512**
- filter 7×7 จำนวน **4,096** ตัว ได้ **1×1×4096**
- filter 1×1 จำนวน **1,000** ตัว ได้ **1×1×1000** คะแนนละหนึ่ง class

**ตัวเลขอื่น ๆ**

| หัวข้อ | ค่า |
|---|---|
| ImageNet | 1.4 M ภาพ, 1,000 class |
| AlexNet | 2012, 5 conv + 3 FC = 8 layers (นับแค่ conv/FC ที่เรียนได้ ไม่นับ pooling/activation), conv แรก 11×11 จำนวน 96 filters, stride 4; **60 M params**; Dropout ใช้ compute ~2×; top-5 error 25.8 % เหลือ **16.4 %** |
| VGG | 2014, input 224×224, filter 3×3 ทั้งหมด; VGG-16 = 13 conv + 3 FC; VGG-19 = 16 conv + 3 FC; FC 4096, 4096, 1000; top-5 error 7.3 % |
| ResNet (ILSVRC'15) | 152 layers, top-5 error **3.57 %**; ResNet-18 = 7×7 conv + 3×3 conv 16 ตัว + FC |
| GoogLeNet | 2014, 22 layers, Inception cell 9 ตัว, filter 1×1/3×3/5×5 + 3×3 max-pool, GAP 7×7 เป็น 1×1 |
| SqueezeNet | 2016, ใช้ 1×1 แทน 3×3 ได้ **params น้อยลง 9×** (3×3 = 9 weights เทียบกับ 1) |
| ResNet | 2015 (อาจารย์); ตัวอย่าง ResNet-18 56×56×64 เป็น 28×28×128 |
| DenseNet | 2016; transition = BN, ReLU, 1×1 conv, 2×2 avg-pool; growth rate k |
| SE block | H×W×C เป็น 1×1×C (GAP) แล้ว FC(ลด)+ReLU แล้ว FC(คืน)+sigmoid แล้วคูณ |
| MixUp | α ที่ดีที่สุด = 0.2, 0.3, 0.4 |
| Mosaic | 4 ภาพ, resize เป็น 256×256, ต่อเป็น 512×512, เลื่อน box +256 ใน x และ/หรือ y, crop สี่เหลี่ยมจัตุรัสที่โดนทั้ง 4 ภาพ |
| Adaptive trigger | val loss ไม่ขยับ **5 epoch ติดกัน**; อาจารย์: patience ≈ **5–10 % ของ epoch ทั้งหมด** (5 จาก 50 = 10 %) |
| Sample-aware | confidence > **0.9** ได้ Strong, < **0.6** ได้ Weak, นอกนั้น Medium |
| Progressive (สไลด์) | ep 1–20: rot ±5°, bright ±0.1; ep 21–50: ±15°, ±0.2, + Gaussian noise; ep 51+: ±30°, Cutout, MixUp |
| ช่วงปัญหาตอนท้ายการเทรน | epoch 80–100 |

**สูตร MixUp** (สไลด์ 15):
- `x̂ = λ·xᵢ + (1−λ)·xⱼ`
- `ŷ = λ·yᵢ + (1−λ)·yⱼ`

โดย λ ∈ [0, 1] เป็นค่าสุ่ม xᵢ กับ xⱼ คือ input vector ดิบ yᵢ กับ yⱼ คือ label แบบ **one-hot** α ที่ดีที่สุด = 0.2, 0.3, 0.4; ใน paper MixUp ต้นฉบับ λ ~ Beta(α, α) α มาจากตรงนี้

ตัวอย่างคำนวณในสไลด์:
- แมว [1, 0] + หมา [0, 1] ที่ λ = 0.5 ได้ label **[0.5, 0.5]**
- ภาพแมว 70 % + ภาพหมา 30 % ได้ label **[0.7, 0.3]**

**จำนวน parameter** (*(ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์)*)
- Standard conv: `K·K·C_in·C_out`
- Depthwise separable: `K·K·C_in` (depthwise) + `C_in·C_out` (pointwise)
- DenseNet layer ℓ ใน block รับ `k₀ + k·(ℓ−1)` channel

**โค้ด PyTorch ในสไลด์** (CIFAR-10, SimpleCNN; Linear(32·8·8) เพราะ 32×32 ผ่าน max-pool สองครั้งเหลือ 8×8; Adam lr = 0.001, 30 epochs, batch 64, CrossEntropyLoss)
- *Progressive* `get_transform(epoch)`:
  - epoch < 10: RandomHorizontalFlip
  - epoch < 20: + ColorJitter(brightness=0.2, contrast=0.2)
  - else: ColorJitter(0.3, 0.3, saturation=0.3) + RandomRotation(15) + RandomErasing(p=0.2)
- *Adaptive* `build_transform(mag)`: ColorJitter(mag), RandomRotation(mag·45), RandomAffine translate (mag·0.2), RandomErasing(p = mag·0.5)
  - เริ่ม `magnitude = 0.1`
  - ถ้า `val_loss > last_val_loss` แปลว่าโมเดลแย่ลง ให้ `magnitude = max(0.05, magnitude − 0.05)`
  - ไม่งั้น `magnitude = min(0.6, magnitude + 0.05)`
- (หมายเหตุ: `RandomErasing` ของ torchvision ทำงานกับ tensor เลยต้องวาง **หลัง** `ToTensor()` แต่ในสไลด์ progressive วางไว้ก่อน)

---

## สิ่งที่อาจารย์เน้น

- **[00:04:18–00:09:44] กรณีศึกษาจำแนกชนิดไม้** K-fold accuracy ได้ 90 %+ แต่ตอนเดโมจริง ภาพ ~20 ภาพผิดเกือบหมด เพราะภาพเทรนเป็นภาพ "ในอุดมคติ" ถ่ายจากด้านบน (มุมนก) แต่ผู้ใช้จริงถ่ายเอียง ซูม มีพื้นหลัง มีมือ และโฟกัสไม่ชัด พอเพิ่ม augmentation (เอียง หมุน center crop ซูมเข้า/ออก สี) บนข้อมูล *ชุดเดิม* ตอบถูกเพิ่มขึ้น ~60 % บทเรียน: **สังเกตพฤติกรรมผู้ใช้** บางปัญหาแก้ที่โมเดลไม่ได้
- **[00:11:40] Augmentation มีประโยชน์ 2 อย่าง: เพิ่ม *Variety* และเพิ่ม *จำนวน Sample*** (อย่างหลังสำคัญมากตอน imbalance หรือเหตุการณ์ที่เกิดยาก)
- **[00:12:14] "Distortion แปลว่าอะไร"**: เสื่อมแต่ไม่หาย Geometric เปลี่ยนรูปทรง Photometric เปลี่ยนความเข้มหรือสีของ pixel
- **[00:19:02] "จำไว้นะครับ… จดไปครับ"**: Random Erase สุ่ม **(1) จำนวนกล่อง (2) ตำแหน่ง (3) ขนาดกล่อง** และ **(4) ค่าที่เติม** (0, mean หรือ random)
- **[00:20:07–00:21:19]** Cutout ล็อกขนาด จำนวน และค่าที่เติม แล้วสุ่ม **แค่ตำแหน่ง** เลยรับประกันได้ว่าข้อมูลเหลือเท่าไหร่
- **[00:22:30–00:23:50]** จุดอ่อนของ Hide-and-Seek คือ patch ที่ติดกันอาจถูกลบไปพร้อมกัน GridMask แก้โดยกระจาย mask ออก
- **[00:25:09–00:26:38, 00:30:13]** **3 กลุ่ม** ของอาจารย์: Distortion *ปรับ*, Occlusion *ซ่อน* (ทั้งสองยังอยู่ในกรอบของข้อมูลตัวเอง) และกลุ่ม 3 (MixUp, CutMix, Mosaic) *ผสมข้อมูล >1 แหล่ง* "แต่ละกลุ่มเล่นอะไรกับมัน ต้องแยกให้ออก"
- **[00:28:50] "ย้ำนะครับ สี่เหลี่ยมจัตุรัส"**: crop ของ Mosaic เป็น **สี่เหลี่ยมจัตุรัส** ขนาดสุ่ม ที่ต้องมีครบทั้ง 4 ภาพ (สัดส่วนไม่ต้องเท่ากัน)
- **[00:31:21]** ขั้นต่ำที่ควรทำตอนนี้: ใช้ Distortion **และ** Occlusion กลุ่ม 3 ใช้เฉพาะ dataset ที่ยาก
- **[00:32:10] "ยุคนี้ไม่มีใครทำ Static Data Augmentation แล้ว"**: ใช้ Dynamic augmentation ปรับตามจำนวน epoch, loss หรือ accuracy
- **[00:41:04] คำถามในห้อง**: "จำแนกตัวอักษรไทย ต้องปรับสีไหม? Flip ไหม?" คำตอบ: **ไม่ต้อง colour jitter** (ตัวอักษรไม่ได้แยกกันด้วยสี) **ไม่ flip ซ้าย-ขวา** กลับหัวเฉพาะถ้าสถานการณ์จริงมี **หมุน/เอียง: ใช้** อาจารย์ติโปรเจกต์รุ่นพี่ที่ใช้ augmentation ไม่เกี่ยวกับข้อมูล ซึ่งแสดงว่าไม่เข้าใจ
- **[00:43:03] "ประโยคเมื่อกี้งงไหม คิดดี ๆ"**: **loss สูง แปลว่าโมเดลยังไม่ฉลาด ให้ใช้ Weak; loss ต่ำ แปลว่าฉลาดแล้ว ให้ใช้ Strong** พูดซ้ำตอนสรุปที่ **[02:05:31]**
- **[00:49:26–00:49:59]** Class-aware เป็นแบบ "ตรงกันข้าม" (ขัดกับที่คิด): **class ที่ accuracy ต่ำได้ Strong** class ที่ accuracy สูงได้ Weak
- **[00:52:41–00:53:01]** Transfer แบบ 1 (ใช้ pretrained model ตามเดิม) แทบเป็นไปไม่ได้ เพราะจำนวน output node ≠ จำนวน class ของเรา
- **[00:53:12] "กาดอกจันทร์คำว่า freeze… จดไป freeze แปลว่าท่อนนั้นจะไม่ถูกเทรน."**
- **[00:55:02–00:56:51]** แบบ 3 (เทรน conv layer หลัง ๆ ด้วย) ใช้กับภาพที่ไม่เหมือนภาพธรรมชาติใน ImageNet เช่น **ultrasound หรือ X-ray** layer ต้นเก็บ feature ระดับต่ำที่ใช้ร่วมกัน layer หลังรวมเป็นวัตถุต่างกันไปในแต่ละ domain พูดซ้ำตอนสรุปที่ **[02:07:18]**
- **[01:04:51–01:06:09] "จดไปครับ"**: filter 1×1 รวม feature ของ pixel **จากทุก channel** ใช้ **ลด/เปลี่ยนจำนวน channel** และใช้เสมอเมื่ออยากปรับขนาด channel
- **[01:06:23–01:07:46] ทริกของ VGG**: 7×7×512 กับ filter 7×7 จำนวน 4,096 ตัว ได้ 1×1×4096 แล้ว filter 1×1 จำนวน 1,000 ตัว ได้คะแนน 1,000 class "Classification ด้วย conv ไม่มี neural network" อาจารย์บอกว่านี่ บวกกับการ normalize ด้วย (x−mean)/sd คือผลงานของ VGG
- **[01:10:01] "จดไปครับ"**: Squeeze ลด channel (6 เหลือ 2) แต่ **ไม่เปลี่ยน W×H**
- **[01:11:12] คำถาม**: ทำไม Expand ใช้ filter สองขนาด? **1×1 ได้ point features; 3×3 ได้ local features** (pixel บวกเพื่อนข้าง ๆ)
- **[01:15:20] คำถาม**: filter 5×5 บน input 480 channel มีกี่ kernel? **480** นี่แหละที่ทำให้ 5×5 ตรง ๆ ต้องใช้ 112.9 M ถ้ามี 1×1 bottleneck เหลือ 5.3 M "**ลดไป 20 เท่า**" [01:16:35]
- **[01:19:29–01:21:50] "คีย์เวิร์ดคือ ในทุกแชนแนล… จดไปครับ"**: conv (1×1/3×3/5×5) ใช้ **ทุก channel** ส่วน max-pool ทำ **ทีละ channel** Max-pool ลด W×H แต่ **จำนวน channel เท่าเดิม** Inception เลยวาง 1×1 ไว้หลัง max-pool เพื่อลด channel "เข้าใจ *เหตุผล* ของลำดับ operator แล้วจะแก้สถาปัตยกรรมเองได้"
- **[01:22:33] คำถาม**: Global Average Pooling แก้ปัญหาอะไร? **ขนาด input ที่ต่างกัน** Conv รับขนาดไหนก็ได้ แต่ **FC network** มีจำนวน input node ตายตัว GAP เหลือค่าเดียวต่อ channel และจำนวน channel ไม่เปลี่ยนตามขนาด input
- **[01:24:58] "หน้านี้หาที่จดหรือจำไปให้ดี ๆ"** (เปรียบเทียบ AlexNet/VGG/GoogLeNet):
  - AlexNet เป็นตัวเริ่ม
  - VGG: 1×1 สำหรับ classification และ normalize input
  - GoogLeNet: feature จากหลายขนาด filter, 1×1 ลด compute, "cell"
  - ความลึก 8, 19, 22: ทุกคนซ้อน layer เพิ่มขึ้นเรื่อย ๆ
- **[01:25:03–01:26:19]** โมเดลใหม่กว่าหรือลึกกว่า **ไม่ได้** ดีกว่าเสมอ **ให้ความซับซ้อนของโมเดลเข้ากับความซับซ้อนของ dataset** วัดจาก (1) จำนวน class และ (2) ความคล้ายกันระหว่าง class (ก/ข ง่าย; ฃ/ข ยาก) ใครบอกว่า "ใช้โมเดลใหม่สุดไปเลย" คือไม่เข้าใจ
- **[01:26:46–01:27:33] 1×1 ทำอะไร ("ลด ลด ลด")**: ลด channel ซึ่ง (1) รวม feature ของ channel แบบ PCA (2) ลด memory, compute และ params เลย overfitting น้อยลง และ (3) ใช้ classify ได้
- **[01:28:32–01:31:27]** เหตุผลของ connectivity: การกรองไม่มี **อะไรรับประกัน** ว่าข้อมูลสำคัญไม่ถูกทิ้ง เลยบวก input ที่ยังไม่ผ่านการประมวลผลกลับเข้าไป
- **[01:32:56–01:33:51] "กาดอกจัน operator"**: ResNet ใช้ **⊕ (element-wise add)** shape เลยต้องเท่ากัน DenseNet ใช้ **Ⓒ (concatenate)** ตามแนว channel
- **[01:34:13] คำถาม** (คำใบ้: "5 พยางค์ ขึ้นต้นด้วย I"): การเชื่อมแบบ Res และ Dense แก้ปัญหาอะไร? **Information Loss**
- **[01:36:38–01:38:29] เกมในห้อง**: Conv block กับ Identity block ต่างกันกี่จุด (ดูแค่ operator)? **2**: conv ตัวแรกใช้ **stride 2** (แทน 1) และ shortcut มี **1×1 conv, stride 2**
- **[01:39:50] "จดไป"**: 3×3 conv ที่ **stride 2** ลด W×H ลงครึ่งหนึ่ง ถ้า stride 1 และ padding 1 ขนาดไม่เปลี่ยน
- **[01:41:21] "หาที่จดไปครับ"**: นอกจาก Max Pooling ยังลด W×H ได้ด้วย **conv stride > 1** ต่างกันตรงที่ max-pool ทำทีละ channel ส่วน conv ใช้ทุก channel
- **[01:43:32–01:45:12]** shortcut ที่เป็น 1×1 stride-2 conv หยิบ pixel เว้น pixel และผสม channel ข้อมูลที่บวกเข้าไปเลย **ไม่ใช่ของเดิม**: ถูกลดขนาดและแปลงแล้ว (point features เทียบกับ local features ของทางหลัก)
- **[01:46:08–01:47:15]** เรียก "Identity" เพราะ shortcut บวกตัว input เองเข้าไป เรียก "Convolution block" เพราะ shortcut ต้องมี conv การออกแบบที่เลือก: ไม่ใช้ pooling (ใช้ conv ลดขนาดแทน) และใช้ skip connection กัน information loss
- **[01:47:36] คำถามในห้อง**: "Max-pool กับ 3×3 stride-2 conv ลด W×H ได้ทั้งคู่ — จริงไหม?" **จริง** ข้อแตกต่าง:
  1. Conv ส่งผล/ใช้ **ทุก channel** แต่ max-pool ไม่ใช่ เลย **max-pool มี overhead น้อยกว่า**
  2. Max-pool พิจารณาแต่ละ channel แยกกัน
  3. Max-pool เป็น **operation ตายตัว** (เอาค่า max เสมอ) ส่วน weight ของ conv **เรียนได้**
- **[01:49:09–01:49:54] "อันนี้ต้องจดไปและต้องระวัง"**: connectivity แบบ Res และ Dense มี **แค่ภายใน block** ไม่ได้ข้ามทั้งโมเดล ตอนแรกอาจารย์เองก็เข้าใจผิด
- **[01:51:00–01:51:35]** ใน bottleneck ของ DenseNet 1×1 ก่อน 3×3 คือ "ทริกเดิม" ลดมิติและ memory ส่วน 3×3 ทำหน้าที่ดึง feature จริง ๆ
- **[01:53:03] คำถาม**: ทำไม transition layer จบด้วย 2×2 average pool? ตรรกะรูปทรงของ CNN: layer ต้นมี W×H ใหญ่ channel น้อย layer หลังมี W×H เล็ก channel เยอะ การ concat ทำให้ channel เพิ่ม 1×1 เลยกำหนดจำนวน channel แล้ว avg-pool ลด W×H นี่เป็นครั้งแรกที่ใช้ **average** pooling (network ก่อนหน้าใช้ max)
- **[01:56:56] "จุดนี้เด็ก ๆ หลายคนเข้าใจผิด"**: depthwise separable ≠ conv ปกติ conv ปกติ **บวก** ผลของแต่ละ channel (weight 1 โดยปริยาย) Depthwise + 1×1 pointwise มี **weight ต่อ channel ที่เรียนได้** ("นี่คือจุดต่างกาดอกจันทร์")
- **[02:00:14–02:01:23]** SE block = **ให้น้ำหนักแต่ละ channel** เพราะ channel สำคัญไม่เท่ากัน Squeeze ใช้ GAP; excitation ใช้ FC network ขนาดตามจำนวน channel; แล้วเอา scalar × feature map
- **[02:02:41–02:07:44] สรุปท้ายคาบ**:
  - Backbone กับ pretrained ต่างกันยังไง
  - Augmentation 3 กลุ่ม
  - Dynamic augmentation ตาม epoch หรือตาม loss/accuracy
  - Transfer learning 3 แบบ
  - สำหรับโปรเจกต์: โหลด backbone มาทุกตัว รันให้หมด เลือกตัวที่เหมาะ แล้วค่อยวิเคราะห์

---

## ข้อสอบที่น่าจะออก

1. **ข้อจำกัดหลักของ deep learning ที่เป็นที่มาของบทนี้คืออะไร และทางแก้ 2 ทางคืออะไร?**
   ต้องใช้ข้อมูลเทรนเยอะมาก (ถึง ~10⁶ ภาพ) ทางแก้คือ Transfer Learning (เอา backbone ที่ pretrain บน ImageNet มาใช้ต่อ) และ Data Augmentation (สร้างตัวอย่างเพิ่มให้เยอะและหลากหลายขึ้น)

2. **บอกจุดประสงค์ 2 ข้อของ data augmentation พร้อมตัวอย่างจากโลกจริงข้อละหนึ่ง**
   (1) เพิ่มจำนวนตัวอย่าง: dataset เล็กหรือ class imbalance เช่น ความผิดปกติที่เจอครั้งเดียวใน 10 ปี (2) เพิ่มความหลากหลายให้ทนทาน: แอปจำแนกชนิดไม้ทายผิดกับภาพที่ผู้ใช้ถ่ายเอียง ซูม รก ๆ จนกระทั่งเพิ่ม augmentation แบบ rotation, crop, zoom และสี

3. **เปรียบเทียบ Photometric กับ Geometric distortion และยกตัวอย่างอย่างละ 3**
   Photometric เปลี่ยนค่า pixel: brightness, contrast, sharpness/blur, สี/saturation, noise Geometric เปลี่ยนรูปทรงหรือตำแหน่ง: crop, rotation, scaling, translation, flip

4. **เปรียบเทียบ Random Erase, Cutout, Hide-and-Seek และ GridMask**
   - **Random Erase**: สุ่มจำนวน ตำแหน่ง และขนาดกล่อง และสุ่มค่าที่เติม
   - **Cutout**: สี่เหลี่ยมจัตุรัสขนาดคงที่ ค่าที่เติมคงที่ สุ่มแค่ตำแหน่ง เลยรับประกันข้อมูลที่เหลือได้
   - **Hide-and-Seek**: patch N×N ถูกซ่อนตามความน่าจะเป็น patch ที่ติดกันอาจหายไปพร้อมกัน
   - **GridMask**: mask เป็น grid สม่ำเสมอ ทุกบริเวณเลยยังมีข้อมูลเหลือ

5. **MixUp, CutMix และ Mosaic ต่างกันยังไง? เหมาะกับงานไหน?**
   - **MixUp**: ผสม 2 ภาพด้วยความโปร่งใส label ผสมตามสัดส่วน (แมว 0.7 / หมา 0.3); α = 0.2–0.4
   - **CutMix**: แปะ patch จากภาพหนึ่งลงอีกภาพ
   - **Mosaic**: 4 ภาพ resize เป็น 256 ต่อเป็น 512 เลื่อน box สุ่ม crop สี่เหลี่ยมจัตุรัสที่โดนทั้ง 4 ภาพ (เริ่มใน YOLOv4)

   เหมาะกับ object detection และ classification ที่ class คล้ายกันมาก

6. **อธิบายปัญหาของ augmentation แบบ static (ดั้งเดิม) ช่วงต้นและช่วงท้ายการเทรน**
   - ช่วงต้น: โมเดลยังอ่อนแต่โดน augmentation หนัก เจอ overwhelming complexity, gradient instability และการเรียน feature พื้นฐานถูกรบกวน
   - ช่วงท้าย: โมเดลเก่งแล้วแต่ได้ augmentation ง่าย เจอ insufficient challenge (overfitting) พลาดโอกาส generalize และเปลืองทรัพยากร

7. **เปรียบเทียบ augmentation แบบ Progressive, Adaptive Scheduling และ Sample-Aware**
   - **Progressive**: ความแรงเพิ่มตาม epoch (curriculum; เลี่ยง underfitting ช่วงต้น)
   - **Adaptive**: ปรับตาม loss/metric loss สูงใช้ augmentation อ่อน loss ต่ำหรือนิ่ง (เช่น val loss ไม่ขยับ 5 epoch) ใช้ augmentation แรงขึ้น
   - **Sample-aware**: ต่อ sample (confidence > 0.9 ได้ Strong, < 0.6 ได้ Weak) หรือต่อ class (class accuracy ต่ำได้ Strong, class accuracy สูงได้ Weak)

8. **ในโค้ด adaptive, magnitude = 0.3 และ validation loss เพิ่มขึ้นใน epoch นี้ magnitude ใหม่เป็นเท่าไหร่? ถ้า loss ลดลงล่ะ?**
   Loss เพิ่ม: max(0.05, 0.3 − 0.05) = **0.25** Loss ลด: min(0.6, 0.3 + 0.05) = **0.35**

9. **Fixed feature extraction กับ fine-tuning: radiology ใช้แบบไหนมากกว่า เพราะอะไร?**
   Fixed feature extraction freeze conv base แล้วเทรนแค่ classifier ใหม่ (FC, SVM หรือ random forest) Fine-tuning เทรน conv layer บางส่วนหรือทั้งหมดใหม่ด้วย (มักจะ freeze layer ต้น ๆ) radiology ใช้ fine-tuning มากกว่า เพราะภาพ ImageNet กับภาพการแพทย์ต่างกันมาก layer ต้นเป็นแบบทั่วไป (ขอบ) layer หลังเป็นแบบเฉพาะงาน

10. **โจทย์: X-ray ทรวงอก grayscale 400 ภาพ, 3 class, ResNet-18 ที่ pretrain บน ImageNet จงอธิบายการตั้งค่า transfer learning**
    - เปลี่ยน output layer เป็น 3 node
    - X-ray ไม่เหมือนภาพธรรมชาติ เลยใช้แบบ 3 ของอาจารย์: freeze conv layer ต้น ๆ เทรน conv layer หลัง ๆ บวก classifier
    - จัดการข้อจำกัดของ input: pretrained net ต้องการ RGB 3 channel แต่ X-ray มี 1 channel (เช่น copy ให้เป็น 3 channel) ใช้ global pooling เพื่อให้ H×W ยืดหยุ่น
    - เพิ่ม augmentation ที่ตรงกับความแปรผันจริง (rotation/translation เล็กน้อย, brightness/contrast) ดีที่สุดคือตั้งตารางจาก weak ไป strong

11. **คำนวณ: input 14×14×480, filter 5×5 จำนวน 48 ตัว (padding แบบ "same") นับจำนวนการคูณแบบตรง ๆ และแบบมี 1×1 bottleneck 16 filter ก่อน**
    ตรง ๆ: (14·14·48)(5·5·480) = 112.9 M Bottleneck: (14·14·16)(480) = 1.5 M บวก (14·14·48)(5·5·16) = 3.8 M รวม 5.3 M น้อยกว่าประมาณ 20×

12. **คำนวณ: conv block ของ ResNet-18 รับ 56×56×64 conv ตัวแรกของทางหลักเป็น 3×3, stride 2, padding 1, 128 filters ขนาด output เป็นเท่าไหร่ และทำไม shortcut ต้องมี 1×1 stride-2 conv?**
    floor((56 − 3 + 2)/2) + 1 = 28 ดังนั้น output คือ 28×28×128 input 56×56×64 บวกแบบ element-wise กับ 28×28×128 ไม่ได้ 1×1 conv stride 2 กับ 128 filters จะแปลงมันเป็น 28×28×128 ส่วนใน identity block shape เท่ากันอยู่แล้ว shortcut เลยเป็น identity

13. **Max pooling กับ 3×3 stride-2 convolution สำหรับการลดขนาด: เหมือนและต่างกันยังไง**
    ทั้งคู่ลด W×H ได้ Max-pool ทำทีละ channel จำนวน channel เท่าเดิม เป็น operation ตายตัว และ overhead ต่ำ Conv ใช้ทุก channel เปลี่ยนจำนวน channel ได้ และมี weight ที่เรียนได้

14. **บอกการใช้งานของ 1×1 convolution และบอกว่าสถาปัตยกรรมไหนใช้แต่ละแบบ**
    - ลดหรือเปลี่ยน channel ซึ่งลด compute และ params และลด overfitting: GoogLeNet ก่อน 3×3/5×5 และหลัง max-pool; SqueezeNet squeeze; DenseNet bottleneck และ transition; ResNet shortcut; MobileNetV3 expansion/projection และ pointwise
    - รวม feature ข้าม channel (แบบ PCA): ทุกตัวข้างบน
    - Classification: VGG
    - เพิ่ม non-linearity: NIN

15. **เปรียบเทียบ connectivity ของ ResNet กับ DenseNet (operation, เงื่อนไข shape, ต้นทุน) และบอกเหตุผลร่วม**
    ResNet บวก input ที่ skip มาเข้ากับ output แบบ element-wise shape เลยต้องเท่ากัน ช่วยเรื่อง vanishing gradient ทำให้ net ลึกมาก ๆ ได้ DenseNet concat map ของทุก layer ก่อนหน้าตามแนว channel channel เลยโตขึ้นตาม growth rate k ได้ feature reuse, feature หลากหลาย และทำงานดีกับข้อมูลน้อย แต่ใช้ memory มากขึ้น Transition layer (BN, ReLU, 1×1 conv, 2×2 avg-pool) คุมการโตของ channel ทั้งคู่ทำงานแค่ภายใน block เหตุผลร่วม: กัน information loss และ vanishing gradient

16. **Squeeze-and-Excitation block ทำอะไร? อธิบาย 3 ขั้นตอน**
    ให้น้ำหนัก channel ตามความสำคัญ
    1. **Squeeze**: GAP จาก H×W×C เป็น 1×1×C
    2. **Excitation**: FC (ลด) + ReLU แล้ว FC (คืนเป็น C) + sigmoid ได้ weight ต่อ channel
    3. **Recalibration**: เอา weight คูณกับ feature map แบบ channel-wise

17. **โจทย์: จะ deploy ตัวจำแนกรอยโรคผิวหนังบนมือถือของคลินิก ใช้ backbone ตระกูลไหน เพราะอะไร?**
    MobileNetV3 ออกแบบมาสำหรับเครื่องสเปกต่ำหรือมือถือ Depthwise separable conv และ inverted residual block ลด params และ memory ส่วน SE เพิ่ม accuracy ได้แบบถูก ๆ ตรงกับเกณฑ์การเลือก "Accuracy, Speed, Size"

---

## กับดักที่เจอบ่อย

- **"โมเดลใหม่กว่า/ลึกกว่า = ดีกว่า"** ผิด ให้ความซับซ้อนของโมเดลเข้ากับความซับซ้อนของ dataset: จำนวน class และความคล้ายกันของ class [01:25:03]
- **ใช้ augmentation แบบ fixed policy ทุก epoch** ล้าสมัยแล้ว แบบเดียวใช้กับทุกช่วงไม่ได้ ให้ใช้ dynamic augmentation [00:32:10]
- **loss สูงต้องใช้ augmentation แรง** กลับด้าน loss สูงแปลว่าโมเดลยังไม่ฉลาด ให้ใช้ **weak** loss ต่ำใช้ **strong** [00:43:03]
- **สับสนกฎ sample-aware สองแบบ** ต่อ sample: sample ที่มั่นใจได้ *Strong* ที่ไม่มั่นใจได้ *Weak* ต่อ class: **class ที่ accuracy ต่ำได้ Strong** class ที่ accuracy สูงได้ Weak กฎของ class ดูเหมือนตรงข้าม และอาจารย์ก็บอกว่ามันขัดกับที่คิด [00:49:35]
- **ใช้ augmentation ที่ไม่เข้ากับงาน** colour jitter หรือ flip ซ้าย-ขวากับตัวอักษรไทยเป็นสิ่งที่ผิด augment เฉพาะความแปรผันที่เกิดในสถานการณ์จริง โดยไม่เปลี่ยนความหมาย [00:41:04]
- **Zoom-out ได้ข้อมูลจริงที่ขอบ** ไม่ใช่ ในทางปฏิบัติขอบจะเติมเป็นสีดำ [00:13:17]
- **"Freeze" แปลว่าโมเดลพัง** ไม่ใช่ แปลว่าส่วนนั้น **ไม่ถูกเทรน** [00:53:28]
- **ใช้ pretrained model แบบไม่แก้อะไรเลย** อย่างน้อยต้องเปลี่ยนและเทรน output layer เพราะจำนวน node ต้องเท่ากับจำนวน class ของเรา [00:52:51]
- **Fixed feature extraction เป็นค่าตั้งต้นของภาพการแพทย์** ไม่ค่อยใช้ เพราะ ImageNet ≠ ภาพการแพทย์ fine-tuning นิยมกว่า และสำหรับ X-ray หรือ ultrasound ควรให้ conv layer หลัง ๆ เทรนได้ด้วย
- **Pretrained ImageNet model รับ grayscale 1 channel ได้ตรง ๆ** ไม่ได้ ต้องเป็น RGB 3 channel (สไลด์ 50)
- **GAP มีไว้เพราะ convolution รับ input หลายขนาดไม่ได้** ผิด Conv รับได้ ปัญหาอยู่ที่ขนาด input ตายตัวของ **FC layer** [01:22:33]
- **Max pooling เปลี่ยนจำนวน channel** ไม่เปลี่ยน มันทำทีละ channel เลยต้องมี 1×1 conv หลัง max-pool ใน Inception [01:21:23]
- **ลิงก์ของ ResNet/DenseNet เชื่อมข้ามทั้งโมเดล** เชื่อมแค่ภายใน block [01:49:09]
- **การรวมของ ResNet = concatenation** ผิด ResNet ใช้ **การบวก** แบบ element-wise (shape ต้องเท่ากัน) DenseNet ใช้ **concatenation** ตามแนว channel
- **Shortcut ใน conv block ของ ResNet พา input เดิมไป** ไม่ใช่ มันผ่าน 1×1 stride-2 conv เลยถูกลดขนาดและแปลงแล้ว [01:44:16]
- **Depthwise separable conv ก็แค่ conv ปกติ** ใน conv ปกติ ผลของแต่ละ channel บวกกันด้วยน้ำหนักเท่ากัน Depthwise + pointwise ใช้ weight ของ channel ที่เรียนได้ และ params น้อยกว่า [01:56:56]
- **1×1 ตัวแรกใน block ของ MobileNetV3 ลด channel** สไลด์เรียกมันว่า 1×1 **Expansion** (มัน *เพิ่ม* channel เช่น 3 เป็น 16) ส่วน 1×1 **Projection** ลดกลับ ในคลิปอาจารย์พูดว่า "ลด Dimension" ตรงขั้นนั้น [01:55:49]; ให้เชื่อสไลด์
- **112.9 M คือจำนวน parameter** สไลด์ระบุตัวเลข 112.9 M / 5.3 M ของ GoogLeNet ว่าเป็น **number of operations** (จำนวนการคูณ) ถึงอาจารย์จะพูดว่า "parameters" ตอนตอบให้เขียนว่า "operations/multiplications"
- **การจัดกลุ่ม augmentation ในสไลด์กับของอาจารย์ต่างกัน** สไลด์แบ่งเป็น Distortion / Occlusion (มี MixUp อยู่ใต้ Occlusion) / YOLOv4 (CutMix, Mosaic) อาจารย์สอนเป็น **3 กลุ่ม** และจับ MixUp, CutMix และ Mosaic ไว้ด้วยกันเป็นกลุ่มหลายแหล่ง ต้องรู้ทั้งสองแบบ
- **AlexNet เป็น CNN ตัวแรก** มันเป็นตัวแรกที่ชนะ ImageNet และเริ่มยุคบูม (2012, ภาพสี) **LeNet** มาก่อน ใช้กับตัวเลขอย่างเดียว และตัว L ใหญ่ใน GoogLeNet ก็ให้เกียรติมัน
