# Chapter 12 — Image Segmentation ด้วยสถาปัตยกรรม Autoencoder (Encoder–Decoder)

Segmentation ตอบ **What + Where ในระดับ pixel**: encoder (conv + pooling) ดึงข้อมูล "what" ออกมาได้ แต่ทำ "where" (ตำแหน่งในภาพ) หายไป ส่วน decoder (upsampling ด้วย interpolation / transposed conv / unpooling) รวมกับ skip connection จะสร้าง output map ขนาดเท่าเดิมกลับมา
บทนี้มี: พื้นฐาน autoencoder, ตัวดำเนินการ upsampling, dilated conv, U-Net / PSPNet / LinkNet / PAN / DeepLabV3(+), การออกแบบ output layer (sigmoid/softmax/linear, #channels = #classes), รูปแบบ label และ workflow ของ `segmentation_models_pytorch` (Dice/Jaccard loss, IoU metric)

> หมายเหตุขอบเขต: ในสไลด์และในคลาส **ไม่ได้** สอน FCN, Mask R-CNN และไม่ได้ให้สูตร Dice/IoU/pixel accuracy สูตรในข้อ 2 เป็นนิยามมาตรฐานที่เติมเข้ามาให้ครบ และมีบอกไว้ตรงนั้น

---

## Key concepts

### 1. ประเภทงาน: Classification vs Detection vs Segmentation
| งาน | ตอบอะไร | "Where" อยู่ในรูปไหน |
|---|---|---|
| Image classification (หรือ image regression) | **What** (ทั้งภาพ) | ไม่มี |
| Object detection | **What + Where** | สี่เหลี่ยมเล็กที่สุดที่ขนานแกนและครอบวัตถุ (bounding box) |
| Image segmentation | **What + Where** | รูปร่างที่ตามขอบวัตถุ คือให้ label **ทีละ pixel** |

- อาจารย์สรุปแบบนี้: classification = **Image Prediction** (ใช้ข้อมูลทั้งหมด, FC layer → ได้คำตอบเดียวต่อภาพ); segmentation = **Pixel Prediction** (ได้คำตอบหนึ่งอันต่อ pixel) ส่วน Detection = "what" ของพื้นที่สี่เหลี่ยม
- Output ของ segmentation เป็นได้ทั้ง **แบบ discrete (class ต่อ pixel)** หรือ **แบบ continuous (regression ต่อ pixel)** เช่น depth estimation

### 2. Segmentation 3 แบบ (slides p.3–4)
| | Semantic | Instance | Panoptic |
|---|---|---|---|
| อะไรถูกติด label | **ทุก pixel** (foreground + background) | **เฉพาะวัตถุ foreground** (วัตถุที่เราสนใจ) | **ทุก pixel** |
| วัตถุ class เดียวกัน | ได้ label เดียวกัน: car1 กับ car2 เป็น "car" ทั้งคู่ (แยกวัตถุไม่ได้) | แต่ละวัตถุมี mask ของตัวเอง (car1 สีส้ม, car2 สีชมพู, car3 สีเขียว) | Things แยกกัน, stuff ไม่แยก |
| มี pixel ที่ไม่ได้ label ไหม? | ไม่มี | มี (ไม่ segment background) | ไม่มี |
| ใช้ทำอะไร (ตามอาจารย์) | เข้าใจภาพ, **คำนวณพื้นที่ / ขนาด / พื้นผิว** | **Tracking** (เช่น ติดตามคนที่เดินไปมา) | เข้าใจทั้งฉาก |

- **Panoptic = Semantic + Instance**
  - **Stuff** = ของที่ไม่มีขอบชัด (ท้องฟ้า, ถนน, น้ำ) → segment แบบ semantic (pixel ท้องฟ้าทั้งหมด = class sky)
  - **Thing** = วัตถุที่นับได้ (คน, รถ, เก้าอี้) → segment แบบ instance (แต่ละอันมี mask ของตัวเอง)
- หมายเหตุ: ตามคอนเซ็ปต์ semantic segmentation ต้อง label background ด้วย แต่ในทางปฏิบัติจะทำ semantic segmentation แบบไม่สน background ก็ได้

### 3. พื้นฐาน Autoencoder (slides p.7–18)
- **นิยาม:** feedforward NN ที่ **input กับ output ขนาดเท่ากัน** (อย่างน้อย width × height เท่ากัน) มันบีบ input ให้เป็น **code** ที่มีมิติต่ำลง แล้วสร้าง output กลับมาจาก code นั้น
- **มี 3 ส่วน:**
  - **Encoder**: บีบข้อมูลและสร้าง code สำหรับภาพ ส่วนนี้คือ feature extraction (ส่วน CNN ก่อน flatten)
  - **Code**: ตัวแทนข้อมูลที่ถูกบีบ/"สรุป" = **latent-space representation** ("latent")
  - **Decoder**: สร้าง input กลับมาจาก code (decompress / "unzip" / reconstruction)
- เป้าหมาย: ทำให้ ‖x − x̂‖² → 0 โดย x กับ x̂ ขนาดเท่ากันแต่ไม่เหมือนกันเป๊ะ (lossy) code เล็กกว่า x และ x̂ ส่วน weight w (encoder) และ w′ (decoder) ได้มาจากการ train (backpropagation เหมือน ANN/CNN)
- **คุณสมบัติหลัก 3 ข้อ:**
  1. **Data-specific**: บีบได้เฉพาะข้อมูลที่คล้ายกับข้อมูลที่ใช้ train (AE ที่ train กับตัวเลขเขียนมือ เอาไปบีบภาพวิวไม่ได้) ต่างจาก zip/winrar/7zip
  2. **Lossy**: output ใกล้เคียง input แต่ไม่เหมือนเป๊ะ ข้อมูลที่หายไปแล้วจะหายไปเลย
  3. **Unsupervised**: ถ้าทำแค่บีบข้อมูล ใช้แค่ input ก็พอ (input = target) แต่ถ้าเป็นงานอื่นอย่าง segmentation **ยังต้องมี label อยู่ดี**
- **Deep autoencoder** (Hinton 2006): มีหลาย layer เช่น 784 → 1000 → 500 → 250 → **2 (code)** → 250 → 500 → 1000 → 784 โดย encoder กับ decoder **ส่วนใหญ่จะสมมาตรกัน แต่ไม่จำเป็นต้องสมมาตร**
- ถ้าใช้จำนวน component เท่ากัน deep AE สร้างข้อมูลกลับได้ดีกว่า **PCA** (ถ้า weight เริ่มต้นดี) ขนาด code เทียบได้กับจำนวน principal component ใน PCA
- ทำไม encoder ต้องเล็กลงและ decoder ต้องใหญ่ขึ้นทีละ layer: เพื่อเรียนรู้ feature, ลดมิติ (ช่วยตัด noise ด้วย), สร้างกลับทีละขั้น, รักษาความสัมพันธ์เชิงตำแหน่ง (ใช้ conv ใน encoder, deconv/upsampling ใน decoder), เรียนรู้ทีละขั้น
- **Hyperparameter 4 ตัว**: Code size, จำนวน layer, จำนวน node ต่อ layer, Loss function (MSE หรือ binary cross-entropy; **ใช้ cross-entropy ถ้า input อยู่ในช่วง [0,1]**)
- **ระวัง:** **Code size เป็น hyperparameter ที่สำคัญที่สุด** ถ้าใหญ่เกินไป AE จะแค่ **copy input ไปเป็น output** โดยไม่ได้เรียนรู้ code ที่มีประโยชน์ ทำให้ **overfitting** เลยต้องให้ code เล็กกว่า input จนได้รูปทรง **โบว์ไท / นาฬิกาทราย (bow-tie / hourglass)** ส่วน input ที่สุ่มล้วน ๆ ไม่มีความสัมพันธ์ภายใน จะสร้างกลับได้ไม่ดี
- **Convolutional Autoencoder**: ใช้ convolution แทน FC layer เสนอโดย **Zeiler et al. 2013** ใช้ทำ: บีบภาพ/วิดีโอ, denoising, สังเคราะห์ภาพ ทำไมต้อง conv: **การ flatten ทำลายข้อมูลตำแหน่ง** (บริบทของ pixel ข้าง ๆ) แต่ convolution จะดู pixel พร้อมกับ pixel รอบ ๆ
- **Denoising Autoencoder** (Vincent et al. 2008): train โดยให้ input ที่มี noise และ output ที่สะอาด ทำให้ code ทนต่อ noise
- ไอเดียหลักของอาจารย์: encoder–decoder ใช้ได้กับทุกงานที่ **input เป็นภาพ และ output เป็นภาพที่ W×H เท่ากัน จะกี่ channel ก็ได้** Segmentation คือ "สร้างภาพที่ตรงกับ ground-truth mask" ส่วน depth estimation ก็คือแบบเดียวกันแต่ค่าเป็น continuous

### 4. พฤติกรรมของ Encoder vs Decoder
| | Encoder (contraction path, downsampling, "pooling") | Decoder (expansion path, upsampling, "unpooling") |
|---|---|---|
| W × H | ลดลง | เพิ่มขึ้น |
| Channels (depth) | เพิ่มขึ้น | ลดลง |
| เรียนรู้ | **WHAT** (ข้อมูลเยอะ) | **WHERE** (กู้ข้อมูลตำแหน่งกลับมา) |
| สูญเสีย | **WHERE** (ข้อมูลตำแหน่งหาย) | — |
| ตัวดำเนินการ | Convolution, **Dilated convolution**, max pooling | **Interpolation, Transposed conv (Deconvolution), Unpooling** |

- feature map ที่ลึกและเล็ก **มีข้อมูลเยอะแต่รายละเอียดตำแหน่งน้อย** ส่วน map ที่ตื้นและใหญ่ **มีรายละเอียดตำแหน่งเยอะแต่ข้อมูลน้อย**

### 5. ตัวดำเนินการ upsampling (decoder) — slides p.21–48
1. **Interpolation**: nearest-neighbour, bilinear (ไม่มี weight ให้เรียนรู้)
   - Nearest neighbour ให้ค่าแบบ discrete คือ copy ค่ามา ส่วน bilinear ให้ค่าที่ไล่ระดับอย่างนุ่มนวล วิธีคำนวณ bilinear **ไม่ได้สอน** แค่เอามาให้ดู
2. **Transposed convolution** (= **Deconvolution** คืออันเดียวกัน) เป็นวิธีที่นิยมที่สุด
   - เอาค่า input แต่ละตัวไปคูณกับ kernel ทั้งอัน แล้ววางผลลงใน output โดยเลื่อนไปทีละ **stride** ตำแหน่ง และ **ส่วนที่ซ้อนทับกันให้เอามาบวกกัน**
   - ตัวอย่าง: input 1×1 กับ kernel 3×3 ได้ output 3×3 ส่วน input 4×4, k=3, s=2, p=0 ได้ **9×9**
   - **ไม่ต้องกลับ/หมุน kernel** (สไลด์เขียนว่า "ไม่หมุนตัวกรอง")
   - stride ยิ่งมาก output ยิ่งใหญ่ (stride 3 กับ k=3 คือไม่ซ้อนกันเลย) kernel ยิ่งใหญ่ output ยิ่งใหญ่ **padding ยิ่งมาก output ยิ่งเล็ก** (ดูกับดักข้างล่าง)
   - ใช้ใน decoder ของ AE และใน GAN
3. **Unpooling (Max unpooling)**: ทำย้อนกลับ max pooling ตอนทำ max pooling ต้อง **เก็บ pooling indices (mask / ตำแหน่ง) ไว้** ตอน unpooling ค่าแต่ละตัวจะกลับไปอยู่ตำแหน่งเดิม และ **ที่เหลือเติม 0** จากนั้น **ตามด้วย convolution** 3×3 หรือ 5×5 เพื่อเติมช่องว่าง / ทำให้ผลเรียบขึ้น ปัจจุบันไม่ค่อยใช้แล้ว
- ในทางปฏิบัติ network หนึ่งจะใช้ **วิธีเดียว อย่างมากสองวิธี** ไม่ใช้ทั้งสามพร้อมกัน

### 6. Padding, Stride, Dilation
- **Padding**: เติมค่า (ปกติเป็น 0) ที่ขอบ เพื่อไม่ให้ conv ปกติทำให้ output หดลง ค่า 0 ไม่มีผลกับ dot product
- **Stride**: ระยะก้าว ใน conv ปกติ stride ยิ่งมากยิ่งบีบมาก ปกติ stride = 1 ใน transposed conv, stride 2 ทำให้ input 3×3 กลายเป็น output 5×5 (slide p.51)
- **Dilated (atrous) convolution**: จุดของ kernel ถูกเว้นระยะห่างกัน ทำให้ **receptive field ใหญ่ขึ้นโดยไม่เพิ่ม parameter หรือการคำนวณ** อันนี้ **ไม่ใช่** morphological dilation จากบท image processing
  - **Dilation = 1 คือ conv ปกติ (จุดติดกัน) Dilation = 2 คือเว้น 1 pixel Dilation = 3 คือเว้น 2** ตอนแรกอาจารย์พูดว่า 0 แล้วแก้เป็นแบบนี้ รูปในสไลด์ (p.53) ก็ใช้ label แบบเดียวกัน: 3×3 ปกติ = dilation 1 และ "3×3, d=2" ครอบพื้นที่ 5×5
  - ใช้ฝั่ง **encoder** เป็นหลัก

### 7. U-Net (Ronneberger et al., **2015**) — ออกแบบมาเพื่อ **biomedical/medical image segmentation**
- ชื่อมาจาก **รูปตัว U** ส่วนประกอบ: **Encoder (contraction path)**, **Decoder (expansion path, transposed conv / "up-conv")**, **Skip connections** (ลูกศรสีเทา, "copy and crop")
- บล็อกในรูปสไลด์ (p.54–55): สีน้ำเงิน = Conv 3×3 + ReLU สีแดง = **MaxPool 2×2** (W, H ลดลงครึ่งหนึ่ง: 256² → 128² → 64² → 32²) สีเทา = **Copy**: feature ของ encoder ถูก **concatenate** กับ feature ของ decoder ที่ **scale เดียวกัน** (W×H เท่ากัน) สีเขียว = **Up-conv 2×2** (transposed conv) สีเหลือง = Dropout แล้วตามด้วย conv 3×3 + ReLU สีฟ้าอ่อน = **Conv 1×1, sigmoid**
- จำนวน channel ในรูป: input 1 channel (256²) → 112 → 224 → 448 → 448 ที่ bottleneck (32²) Concat หลัง up-conv: 448 + 448 = **896** ที่ 64² Output เป็น **1 channel** (binary mask ผ่าน sigmoid) ตัวอย่างในรูปไม่ใช่ภาพทางการแพทย์: ใส่แผนที่ความสูงของดวงจันทร์ ("Input DEM") แล้วได้ "Predicted Mask" ของขอบหลุมอุกกาบาตออกมา ตัวเลข "112, 224, 448" ที่อาจารย์พูด [00:56:31, 00:57:52] ตรงกับตัวเลขนี้
- **ทำไมต้องมี skip connection:** encoder ทำข้อมูลตำแหน่งหาย การสร้างกลับเลยอาจบิดเบี้ยว skip connection จะส่ง **ข้อมูลตำแหน่ง** จาก encoder ไป decoder ที่ scale เดียวกัน (คล้าย skip ของ ResNet) ทุกขั้นของ upsampling จะรวมข้อมูลตำแหน่งกับข้อมูล "what" ก่อน แล้วประมวลผล แล้วค่อย upsample
- **Layer สุดท้าย = 1×1 convolution** (ทำนายทีละ pixel) + activation:

| งาน | ความหมาย | จำนวน filter 1×1 (output channels) | Activation |
|---|---|---|---|
| Binary (fg/bg) – แบบที่ 1 | 1 pixel → 1 ใน 2 class | **1** | **Sigmoid** (threshold 0.5) |
| Binary – แบบที่ 2 | เหมือนกัน | **2** | **Softmax** (เลือกอันที่คะแนนมากกว่า) |
| Multiclass | 1 pixel → ได้ 1 class พอดีจาก N class | **N** (เช่น 5) | **Softmax** (`softmax2d` ใน smp) |
| Multilabel | 1 pixel → เป็นได้หลาย class (เช่น นิ้วทับดินสอ) | **N** (เช่น 4) | **Sigmoid** (channel ไหน > 0.5 → เป็น class นั้น) |
| Regression (เช่น depth estimation) | ค่า continuous ต่อ pixel | **จำนวนค่าที่ต้องการประมาณ** (1 ถ้าทำแค่ depth) | **Linear / ไม่ใส่** |

- งาน binary ส่วนใหญ่จะใช้ sigmoid กับ 1 channel มากกว่า softmax

### 8. รูปแบบ Label (mask) (slides p.60–62)
- **Class label**: 1 channel, H×W เท่ากับภาพ แต่ละ pixel เก็บเลข class (0, 1, 2, …) คือหนึ่งค่าต่อ pixel ใช้ได้กับ **multiclass และ regression**
- **One-hot label**: **#channels = #classes** pixel จะมีค่า 1 ใน channel ของ class ตัวเอง ใช้ได้กับ **multiclass หรือ multilabel** **Multilabel ต้องใช้ one-hot เท่านั้น** เพราะ pixel หนึ่งอาจเป็น 1 ได้หลาย channel
- ตัวอย่าง (3 class: ถนน = 0, รถ = 1, คน = 2): แถว `0 1 2` ใน class-label map จะกลายเป็น one-hot channel โดย channel 0 มีค่า 1 ตรงที่ label เป็น 0 และ channel อื่นก็ทำแบบเดียวกัน
- จะใช้แบบไหนขึ้นกับ (1) งาน และ (2) library รองรับแบบไหน
- ใน smp binary mask มีค่า 0/1, **ขนาดเท่ากับภาพ** และภาพกับ mask ต้องผ่าน **transform เดียวกัน** ซึ่งต้องใช้ `torchvision.transforms.v2`

### 9. สถาปัตยกรรมอื่น ๆ
| สถาปัตยกรรม | ปี / ผู้เขียน | ไอเดียหลัก |
|---|---|---|
| **U-Net** | 2015, Ronneberger | Encoder–decoder + skip connection แบบ **concatenation**; ภาพทางการแพทย์ |
| **PSPNet** (Pyramid Scene Parsing) | 2017, Zhao et al. | CNN backbone แล้วต่อด้วย **Pyramid Pooling Module**: pool feature map เป็น 1×1 (ข้อมูลภาพรวม), 2×2, 3×3, … (ข้อมูลเฉพาะจุด) จากนั้นทำ 1×1 conv แต่ละอัน, upsample ให้ขนาดเท่ากัน, **concatenate** กับ feature ของ backbone แล้วทำ 1×1 conv เพื่อทำนาย Multi-scale ช่วยจัดการวัตถุหลายขนาด สไลด์ยังพูดถึง **Dice loss** (ช่วยเรื่อง **imbalanced data**) |
| **LinkNet** | 2017, Chaurasia | โดยพื้นฐานคือ U-Net แต่ skip connection ใช้ **element-wise addition** (ต้อง channel และ W×H เท่ากัน; output ขนาดเท่าเดิม) แทน concatenation เบาและเร็ว เหมาะกับระบบที่ทรัพยากรน้อย มี 5 ส่วน: Initial block, Encoder block, Decoder block (transposed conv), Final block (1×1 conv), Skip connection |
| **PAN** (Pyramid Attention Network) | 2018, Li et al. | Encoder + **FPA** (Feature Pyramid Attention: ได้ feature หลาย scale โดยใช้ **filter หลายขนาด** แทนการย่อขยายภาพ แล้วคูณแบบ element-wise กับ feature ของ backbone และมีกิ่ง global-pooling อีกกิ่ง) + **GAU** (Global Attention Upsample: high-level feature → **global average pooling** → 1×1 conv ได้ **channel weights** ออกมา เอาไปถ่วงน้ำหนัก low-level feature แล้วบวกกับ high-level feature มันทำงานเหมือน **skip connection แบบถ่วงน้ำหนักที่ถ่วงตาม channel ไม่ใช่ตามตำแหน่ง**) ใช้ BN หลัง conv ทุกตัว |
| **DeepLabV3** | 2017, Chen et al. | แก้ปัญหา encoder ทำข้อมูลตำแหน่งหาย: downsample ถึงจุดหนึ่งแล้วหยุด แล้วใช้ **atrous/dilated conv** แทนการย่อต่อ ใช้ **ASPP** (Atrous Spatial Pyramid Pooling: ใช้ dilation rate หลายค่า รันแบบ **ขนาน** แล้ว concatenate หรือรันแบบ **ต่อกัน**) มันทำนายด้วย 1×1 conv **ก่อน แล้วค่อย upsample** ให้เท่าขนาดภาพ จุดอ่อน: **decoder เล็กมาก** (upsample ขั้นเดียว ไม่สมมาตรเลย) |
| **DeepLabV3+** | 2018, Chen et al. | DeepLabV3 + **Decoder module**: upsample, concatenate กับ feature ของ backbone, ขยายทีละขั้น แล้วค่อย classify เป็น encoder–decoder ของจริง |

- **Backbone ของ encoder เปลี่ยนได้** (ResNet, MobileNet, DenseNet, VGG, GoogLeNet, EfficientNet, …)
- สถาปัตยกรรมใน smp: Unet, Unet++, MAnet, FPN, DeepLabV3, DeepLabV3+

### 10. Workflow ของ `segmentation_models_pytorch` (smp)
- Library ของ **Pavel Iakubovskii** สร้างบน PyTorch: เป็น API ระดับสูงที่มี pretrained encoder, metric และ loss ให้ใช้
- ติดตั้ง: `pip install segmentation-models-pytorch` (PyPI) หรือ `pip install git+https://github.com/qubvel/segmentation_models.pytorch` (GitHub บางทีได้เวอร์ชันใหม่กว่า)
- **5 ขั้นตอน:** (1) กำหนด Dataset (ภาพ + mask) → (2) กำหนด network (`smp.FPN` / `smp.Unet`, `encoder_name='resnet50'`, `encoder_weights='imagenet'`, `classes=len(CLASSES)`, `activation`) → (3) loss + optimizer (**DiceLoss** หรือ **JaccardLoss** ทั้งคู่รองรับ binary, multiclass และ multilabel; Adam, lr = 1e-4) → (4) train (`TrainEpoch` / `ValidEpoch` แค่ **ตั้งค่า**; train จริงตอนเรียก `.run(loader)`; 40 epochs; lr ลดเป็น 1e-5 ที่ epoch 25; `verbose=True` แสดง progress bar) → (5) test/evaluate (`ValidEpoch` กับ test loader, `best_model.predict(image)`)
- Metric ที่มี: **F1, IoU (threshold = 0.5), Accuracy, Precision, Recall**
- `aux_params` (pooling 'avg'/'max', dropout, activation, classes) เพิ่มหัว classification: pooling ทำให้เป็น 1-D แล้วต่อด้วย FC layer
- การ save: save ที่ epoch สุดท้าย หรือเก็บ epoch ที่ดีที่สุด (loss ต่ำสุด / KPI ดีสุดเช่น IoU) อาจารย์ save ทั้งสองแบบ
- Dataset ตัวอย่าง: **Retina Blood Vessel** (Kaggle) เป็นงาน binary (`CLASSES=['blood_vessel']`, sigmoid)

---

## สูตรและตัวเลขที่ต้องจำ

### ขนาด output ของ transposed convolution (PyTorch `ConvTranspose2d`, dilation 1)
```
H_out = (H_in − 1)·s − 2p + k
```
Demo ในสไลด์ (input 4×4 เป็น 1 ทั้งหมด, kernel 3×3 เป็น 1 ทั้งหมด):

| padding | stride | output |
|---|---|---|
| 0 | 1 | 6×6 |
| 0 | 2 | **9×9** (ตัวอย่างที่ทำในคลาส) |
| 0 | 3 | 12×12 (ไม่ซ้อนกัน) |
| 1 | 1 | 4×4 |
| 2 | 1 | 2×2 |
| 1 | 2 | 7×7 |
| 2 | 2 | 5×5 (โค้ดในสไลด์: `padding=2, stride=2`) |

- ตัวอย่างอื่นในสไลด์: input 1×1 กับ kernel 3×3 ได้ 3×3 ถ้า stride 2, input 3×3 ได้ 5×5 (ตรงกับ k=3, p=1)
- เทียบกับ conv ปกติ: `H_out = ⌊(H_in + 2p − k)/s⌋ + 1`
- ขนาดจริงของ dilated kernel: `k + (k−1)(d−1)` (3×3 ที่ d=2 ครอบ 5×5 แต่ยังใช้ weight แค่ 9 ตัว) ต้นทุน: 3×3 = คูณ 9 ครั้ง, 5×5 = 25, 7×7 = 49 นี่คือเหตุผลที่ DeepLab ใช้ dilated 3×3 แทน kernel ใหญ่
- Max pool stride 2 ทำให้ W และ H เหลือครึ่งเดียว (เหลือ 50%)

### Nearest-neighbour interpolation (2×2 → 5×5, slides p.22–23)
1. พิกัดต้นทางกระจายเท่า ๆ กันในช่วง [0,1] ทีละ 1/(n−1) = 1/4 ได้ [0, 0.25, 0.5, 0.75, 1]
2. `nearest = ⌊src + 0.5⌋` ได้ [0, 0, 1, 1, 1]
3. ทำแบบนี้ทั้งแถวและคอลัมน์ แถว/คอลัมน์ 0–1 ของ output มาจากแถว/คอลัมน์ 0 ของ input และแถว/คอลัมน์ 2–4 มาจากแถว/คอลัมน์ 1 ของ input

ถ้า X = [[10, 20], [30, 40]] (ตัวอย่างของอาจารย์):
```
10 10 20 20 20
10 10 20 20 20
30 30 40 40 40
30 30 40 40 40
30 30 40 40 40
```
ถ้า H กับ W ไม่เท่ากัน (เช่น 4 กับ 5) ให้คำนวณระยะก้าวแยกกันในแต่ละแกน

### Bilinear interpolation (X เดิม, 2×2 → 5×5, slide p.24, ให้แค่ผลลัพธ์)
```
10   12.5 15   17.5 20
15   17.5 20   22.5 25
20   22.5 25   27.5 30
25   27.5 30   32.5 35
30   32.5 35   37.5 40
```
ค่าเปลี่ยนอย่างนุ่มนวล (เป็นเส้นตรงทั้งสองทิศ) ส่วน nearest neighbour จะ copy ค่าแบบ discrete ขั้นตอนการคำนวณไม่ได้สอน

### ตัวอย่างการคำนวณ transposed conv (slides p.29–40)
X = [[1,1,1,1],[1,1,1,1],[3,1,2,2],[3,1,2,2]], K = 3×3 เป็น 1 ทั้งหมด, p = 0, s = 2
- ขั้นที่ 1: คูณค่า input แต่ละตัวกับ kernel ได้บล็อก 3×3 ทั้งหมด 16 บล็อก (เช่น X[3,1] = 3 ได้บล็อกที่เป็น 3 ทั้งหมด)
- ขั้นที่ 2: วางบล็อกห่างกัน 2 (ตาม stride) และ **บวกค่าตรงที่ซ้อนกัน**
- Output เป็น **9×9** = (4−1)·2 + 3:
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
- ตำแหน่งที่ 2 บล็อกซ้อนกันค่าจะเป็น 2 เท่า และที่ 4 บล็อกซ้อนกันค่าจะเป็น 4 เท่า ถ้า input เป็น 1 ทั้งหมด (p.44), s = 2 ได้ค่า 1/2/4 ถ้า s = 1 output เป็น 6×6 และค่าสูงสุดถึง 9

### Metric และ loss แบบวัดการซ้อนทับ (นิยามมาตรฐาน; สไลด์แค่บอกชื่อ)
ให้ A = pixel ที่ทำนายว่าเป็น foreground และ B = pixel ของ ground truth
- **IoU / Jaccard** = |A∩B| / |A∪B| = TP / (TP + FP + FN)
- **Dice (= F1 สำหรับ mask)** = 2|A∩B| / (|A| + |B|) = 2TP / (2TP + FP + FN)
- ความสัมพันธ์: Dice = 2·IoU / (1 + IoU) ดังนั้น Dice ≥ IoU
- **Pixel accuracy** = (TP + TN) / pixel ทั้งหมด
- **Dice loss** = 1 − Dice **Jaccard loss** = 1 − IoU แบบ soft จะใช้ probability ที่ทำนายได้
- ใน smp, `IoU(threshold=0.5)` จะตัด probability ที่ทำนายได้ที่ 0.5 ให้เป็น 0/1 ก่อนคำนวณ

**ตัวอย่างคำนวณ (ตัวเลขฝึกทำ ไม่ได้มาจากสไลด์):** ภาพ 10×10 = 100 px, GT = 20 px, Pred = 25 px, ซ้อนกัน = 15 px
- TP = 15, FP = 10, FN = 5, TN = 100 − 30 = 70
- IoU = 15 / (20 + 25 − 15) = 15/30 = **0.50**
- Dice = 2·15 / (20 + 25) = 30/45 = **0.667** (เช็ก: 2·0.5/1.5 = 0.667)
- Pixel accuracy = (15 + 70)/100 = **0.85** ดูดีแค่เพราะ background มีเยอะ

### ปี / ชื่อ
Hinton 2006 (deep AE, 784-1000-500-250-2) · Vincent 2008 (denoising AE) · Zeiler 2013 (conv AE) · Ronneberger 2015 (U-Net) · Zhao 2017 (PSPNet) · Chaurasia 2017 (LinkNet) · Chen 2017 (DeepLabV3) · Li 2018 (PAN) · Chen 2018 (DeepLabV3+)

---

## สิ่งที่อาจารย์เน้น

- **[00:00:06] / [00:09:06]** ต้องอธิบายได้ว่า classification, detection และ segmentation ต่างกันยังไง: classification = What; detection และ segmentation = What + Where ส่วน Where ของ detection คือสี่เหลี่ยมเล็กที่สุดที่ครอบวัตถุ ส่วน Where ของ segmentation จะตามรูปร่างวัตถุ "สิ่งที่เด็ก ๆ จะต้องตอบได้" — "what you must be able to answer"
- **[00:03:50]–[00:05:22]** Instance segmentation ครอบคลุมแค่ **foreground** ส่วน semantic ครอบคลุมทั้ง foreground และ background แต่ไม่ได้จัดกลุ่ม pixel เป็นวัตถุ
- **[00:08:02]** ในทางปฏิบัติ semantic segmentation *ไม่สน* background ก็ได้ แต่ "by concept" มันต้อง label ทุก pixel
- **[00:08:28]** Instance segmentation เหมาะกับ **tracking** ส่วน semantic เหมาะกับการเข้าใจภาพและ **คำนวณพื้นที่/ขนาด**
- **[00:11:13]** อันดับแรกต้องตัดสินใจว่าต้องการแค่ What หรือ What + Where จากนั้นค่อยเลือกว่า Where ควรเป็น bounding box หรือรูปร่างวัตถุ
- **[00:14:05]–[00:16:07]** คุณสมบัติ 3 ข้อของ AE: data-specific (ไม่เหมือน WinZip/WinRAR), lossy (ส่วนที่หายไปตาเรามักมองไม่ออก), unsupervised
- **[00:18:16]** คำว่า "unsupervised" นั้น "อาจจะไม่จริงในบริบทของวิชานี้" ("may not be true in this course"): การ train segmentation ต้องมี ground-truth mask
- **[00:18:42]** "สิ่งที่เด็กๆต้องเข้าใจ" ("what you need to understand"): 3 ส่วน (encoder, code, decoder) คืออะไรและแต่ละส่วนทำอะไร Encoder = feature extraction (ส่วน CNN ก่อน flatten) feature map ก่อน flatten = code / latent
- **[00:23:14]** x กับ x̂ **ขนาดเท่ากันแต่หน้าตาไม่เหมือนกันเป๊ะ** (ข้อมูลหาย) code มักจะเล็กกว่า
- **[00:26:11]** Encoder กับ decoder ส่วนใหญ่สมมาตรกัน แต่ความสมมาตร "ไม่จำเป็น" ("not necessary") ตัวอย่างคือ DeepLabV3 [01:48:47]
- **[00:28:56]** "Keyword สำคัญ" ("important keyword"): encoder คือส่วน feature extraction ที่เรารู้จักอยู่แล้ว ส่วน decoder มีตัวดำเนินการพิเศษคือ transposed conv / deconvolution ที่ขยาย feature เล็ก ๆ ให้ใหญ่ขึ้น
- **[00:29:39]–[00:30:23]** Encoder: ข้อมูลเยอะ, **ข้อมูลตำแหน่งหาย** "Encode ลด Width, Height เพิ่ม Channel" ("encode shrinks width and height, adds channels") ส่วน Decode ทำตรงข้าม
- **[00:31:27]–[00:34:20]** ไอเดียใหญ่: encoder–decoder ใช้กับงาน **ภาพเข้า ภาพออก (W×H เท่ากัน, กี่ channel ก็ได้)** Segmentation ก็แค่ "สร้างภาพเป้าหมายนี้จาก input" depth estimation ก็ไอเดียเดียวกัน
- **[00:35:42]** ทำไมใช้ conv แทน FC: การ flatten ทำข้อมูลตำแหน่งและข้อมูลของ pixel ข้าง ๆ หาย
- **[00:36:57]** วิธี upsampling ของ decoder มี 3 แบบ **Deconvolution = transposed convolution** ("ให้รู้ว่ามันคือสิ่งเดียวกัน", "know that they are the same thing") และเป็นแบบที่นิยมที่สุด
- **[00:39:52]–[00:41:26]** ขั้นตอน nearest-neighbour: หาระยะห่างก่อน (กฎ n−1, 1/4 = 0.25), บวก 0.5, ปัดลง แล้ว map index ทั้งสองแกน
- **[00:42:35]** Bilinear: แค่เอามาให้ดู ไม่ได้สอนวิธีคำนวณ
- **[00:44:09]–[00:46:17]** ทำ transposed conv ทีละขั้น: คูณ input แต่ละตัวกับ kernel, เลื่อนตาม stride, **บวกส่วนที่ซ้อนกัน** input 4×4 กับ k=3, s=2 ได้ **9×9**
- **[00:46:25]–[00:47:45]** ขนาดขึ้นกับ **stride, padding, kernel size** stride หรือ kernel ยิ่งใหญ่ output ยิ่งใหญ่ kernel ไม่ต้องกลับด้าน
- **[00:49:02]** "ยิ่ง Padding เพิ่ม ข้อมูลก็จะยิ่งหดลง" ("the more padding, the smaller the output"): padding ไม่ใช่ตัวที่ใช้ขยายขนาด
- **[00:50:44]–[00:51:55]** Unpooling ต้องเก็บ **location/mask** ไว้ตอนทำ max pooling ตำแหน่งอื่นเติม 0 แล้วทำให้เรียบด้วย conv 3×3 หรือ 5×5
- **[00:52:15]** ในทางปฏิบัติใช้วิธี upsampling วิธีเดียว อย่างมากสองวิธี ไม่ใช้ทั้งสาม
- **[00:52:38]** Dilated conv ≠ morphological dilation จากบท local image processing **[00:55:17] แก้ใหม่:** จุดติดกัน = dilation 1, เว้น 1 = dilation 2 Dilated conv ใช้ฝั่ง **encoder**
- **[00:55:40]** U-Net เป็นสถาปัตยกรรม segmentation พื้นฐานที่นิยมที่สุด สร้างมาเพื่อ **ภาพทางคลินิก/การแพทย์** ชื่อมาจากรูปตัว U
- **[00:57:07]** "หาที่จด" ("get something to write on"): encoder = feature extraction ที่ลด W×H และเพิ่ม channel Layer ลึก ๆ มีข้อมูลเยอะแต่ข้อมูลตำแหน่งหาย Layer บน ๆ มีข้อมูลตำแหน่งเยอะแต่ข้อมูลน้อย
- **[00:59:17]–[01:01:38]** **ข้อเสียของสถาปัตยกรรม AE คือ encoder ทำข้อมูลตำแหน่งหาย** การสร้างกลับเลยอาจบิดเบี้ยว **Skip connection** นำข้อมูลตำแหน่งจาก encoder ไป decoder **ที่ scale เดียวกัน** (256↔256, 128↔128, …)
- **[01:02:14]–[01:03:21]** Skip connection ไม่ได้ "รู้" ว่าตัวเองส่งข้อมูลตำแหน่ง แค่ feature ตื้น ๆ มีข้อมูลตำแหน่งเป็นส่วนใหญ่ มันเลยรอดมาถึงตอนรวมกัน
- **[01:04:12]** สีเทา = copy + **concat** สีเขียว = up-conv = transposed conv
- **[01:05:30]–[01:13:16] Quiz โบนัส "ใครพิมพ์ตอบมาได้ ผมบวกให้ 0.5 คะแนน"** ("whoever types the answer gets +0.5 points"; ดูข้างล่าง)
- **[01:13:16]–[01:14:02]** หลัง 1×1 conv, activation เป็นตัวกำหนดประเภทงาน: sigmoid (binary/multilabel), softmax (multiclass), linear (regression) "ใช้มุกเดิมกับตอนที่เราเรียน CNN" ("same trick as when we studied CNNs")
- **[01:14:14]** ตัวอย่าง multilabel: ตรงที่นิ้วทับดินสอ pixel นั้นเป็นทั้งนิ้วและดินสอ
- **[01:19:00]–[01:19:07]** "**จำนวน Channel ตรงนี้ เท่ากับจำนวนคลาส จดไปครับ**" ("the number of channels here equals the number of classes, write it down") พูดสองรอบ
- **[01:20:04]** "ประโยคเมื่อกี้เหมือนสั้น ๆ แต่สำคัญ" ("that sentence was short but important"): multiclass = 1 pixel, 1 class, มีหลายตัวเลือก, softmax ส่วน Multilabel = 1 pixel, อาจเป็นหลาย class, sigmoid, channels = classes, threshold 0.5 ต่อ channel
- **[01:21:05]–[01:24:11] คำถามโบนัสข้อที่ 2 (+0.5 สำหรับคนแรกที่ตอบถูก):** จะแก้สถาปัตยกรรมยังไงสำหรับ **regression** (depth estimation ทีละ pixel)? ที่ A (activation): **linear หรือไม่ใส่ activation** ที่ B (จำนวน output channel / filter 1×1): **จำนวนค่าที่ต้องการประมาณ** (1 ถ้าทำแค่ depth)
- **[01:24:16]–[01:27:11]** รูปแบบ label: class label (1 channel) สำหรับ regression และ multiclass ส่วน One-hot (channels = classes) สำหรับ multiclass หรือ multilabel **Multilabel ต้องเป็น one-hot** เพราะ pixel ซ้อนกันได้ ใช้แบบไหนขึ้นกับงานและ library
- **[01:28:17]–[01:29:05]** Multi-task (เช่น ชนิดเนื้อเยื่อแผลทีละ pixel บวกความลึกของแผล) ใช้ encoder ร่วมกันแล้วแตกเป็นหัว 1×1 ขนานกันได้ แต่ "อย่าทำเลย แยก Architecture ไปเถอะ ง่ายกว่า" ("don't do it, use separate architectures, it's easier")
- **[01:29:20]–[01:30:23]** Encoder–decoder ที่ซ้อนกันหลายชั้นใช้ GPU RAM เยอะมาก (RTX 3090 ไม่พอสำหรับ 2 stack) **Segmentation ใช้ memory มากที่สุดและ predict ช้าที่สุด** Memory โตตามขนาด input: classification < detection < segmentation
- **[01:30:48]–[01:31:01]** "**การทำ Classification มันคือการ Predict จากภาพ แต่การ Segmentation มันคือการ predict ในระดับพิกเซล — ประโยคนี้จำดีๆ**" ("classification predicts from the image, segmentation predicts at pixel level; remember this sentence well")
- **[01:33:03]** AE ยังใช้ทำ denoising และสร้างภาพกลับให้คมได้ แต่ใช้ได้ดีเฉพาะกับข้อมูลที่คล้ายข้อมูลที่ใช้ train
- **[01:35:37]–[01:37:33]** PSPNet: pooling 1×1 = ข้อมูลภาพรวม, grid ที่ละเอียดขึ้น = ข้อมูลเฉพาะจุด แล้ว Upsample, concat, 1×1 conv เพื่อทำนาย
- **[01:38:05]–[01:38:31]** **U-Net = skip แบบ concatenation ส่วน LinkNet = skip แบบ addition** (ต้อง channel และ W×H เท่ากัน, ขนาดไม่เปลี่ยน)
- **[01:41:33] "จดไป"** ("write it down"): GAU ของ PAN **ให้น้ำหนักว่า feature channel ไหนสำคัญ** (ถ่วงน้ำหนักตาม channel ไม่ใช่ตามตำแหน่ง) แล้วเอาผลไปบวกกับ high-level feature มันทำงานเหมือน skip connection แบบถ่วงน้ำหนัก
- **[01:43:47]–[01:44:24]** FPA ของ PAN สร้าง pyramid โดยเปลี่ยน **ขนาด filter** ไม่ใช่ขนาดภาพ มันคูณแบบ element-wise ในขณะที่ PSPNet ใช้ concatenate
- **[01:45:14]–[01:46:59]** DeepLab แก้ปัญหาตำแหน่งหายด้วย **dilated conv แทนการ downsample ต่อ** ทำไมไม่ใช้ 7×7: คูณ 9 vs 25 vs 49 ครั้ง ASPP รันแบบขนาน (concat แล้ว 1×1) หรือแบบต่อกัน มันทำนายก่อนแล้วค่อย upsample ซึ่งกลับกับลำดับปกติ
- **[01:48:26]–[01:49:25]** จุดอ่อนของ DeepLabV3 คือ decoder มีขั้นเดียว V3+ เพิ่ม decoder ที่ค่อย ๆ ขยายและใช้ feature จาก backbone
- **[01:54:44]** อาจารย์ยอมรับว่าสไลด์โค้ด multiclass ใช้ activation ผิด multiclass ควรเป็น softmax(2d)
- **[01:55:14]–[01:55:50]** การ train ส่วนใหญ่ใช้ **Dice loss หรือ Jaccard loss** กับ Adam ส่วน Metric ใช้ IoU หรือ accuracy
- **[01:56:39]–[01:56:58]** `TrainEpoch` แค่ตั้งค่า parameter การ train เกิดตอนเรียก `.run()`
- **[01:57:17]–[01:57:45]** Save ที่ epoch สุดท้ายหรือ epoch ที่ดีที่สุด อาจารย์ save ทั้งสองแบบ

### Quiz ในคลาส [01:06:24]–[01:13:16] ได้ +0.5
**คำถาม:** "ทำไมเขาต้องใช้ 1 คูณ 1 Convolution ในการ classify" ("why do they use 1×1 convolution to classify") ใน U-Net ทำไม **layer ทำนายตัวสุดท้ายถึงเป็น 1×1 convolution**?

**คำตอบที่ถูก (ของอาจารย์):** เพราะ segmentation คือ **pixel prediction** การจะตัดสินว่า pixel หนึ่งเป็น class อะไร network ใช้ **feature ของ pixel นั้นจากทุก channel** (แต่ละ channel = หนึ่ง feature) ที่ตำแหน่งนั้น 1×1 conv ทำงานเหมือน **fully connected layer ที่ใช้กับแต่ละ pixel แยกกัน** output เลยยังมี W×H เท่ากับภาพ

ส่วน CNN สำหรับ classification จะใช้ FC layer กับตำแหน่ง **ทั้งหมด** เพื่อทำนาย label เดียวสำหรับ **ทั้งภาพ** (image prediction)

คำใบ้ที่อาจารย์ให้: "**ไม่เกี่ยวกับ Softmax**" ("it's not about softmax") [01:09:53] คำตอบ "เพราะ output เป็นภาพ?" ได้ "เกือบถูก" ("almost right") [01:10:46] คำตอบ "ดึงข้อมูลสำคัญมาถ่วงน้ำหนัก pixel นั้น" ได้ "ใกล้เคียง แต่ยังไม่ใช่" ("close, but not it") [01:11:17] การเลือก sigmoid/softmax เป็นอีกเรื่องหนึ่ง: มันกำหนดประเภทงาน (binary/multiclass/multilabel/regression) ไม่ใช่เหตุผลที่ layer เป็น 1×1

---

## ข้อสอบที่น่าจะออก

1. **เปรียบเทียบ classification, detection และ segmentation** Classification ให้ What ของทั้งภาพ Detection ให้ What + Where เป็น bounding box ส่วน Segmentation ให้ What + Where เป็นรูปร่างทีละ pixel (ทำนายระดับ pixel)

2. **Semantic vs instance vs panoptic?** Semantic: ทุก pixel (รวม background) ได้ class แต่วัตถุ class เดียวกันไม่ถูกแยก Instance: เฉพาะวัตถุ foreground แต่ละอันมี mask ของตัวเอง background ไม่ได้ label Panoptic: label ทุก pixel; stuff (ท้องฟ้า, ถนน) ทำแบบ semantic, things (คน, รถ) ทำแบบ instance

3. **โจทย์ทางการแพทย์: วัดพื้นที่รวมของ tumour หรือรอยโรคใน MRI slice หนึ่ง** ใช้ Semantic (binary) segmentation เพราะต้องการพื้นที่ ถ้าต้องนับและติดตามเซลล์/นิวเคลียสแต่ละตัวตามเวลา คือ instance segmentation (แยกวัตถุ, tracking) ถ้าต้อง label เนื้อเยื่อทุกชนิดพร้อมแยกรอยโรคแต่ละอัน คือ panoptic

4. **Autoencoder มี 3 ส่วนอะไรบ้างและแต่ละส่วนทำอะไร?** Encoder บีบข้อมูลและดึง feature Code คือ latent-space representation ที่ถูกบีบ Decoder สร้าง output กลับจาก code Input และ output ขนาดเท่ากัน

5. **คุณสมบัติ 3 ข้อของ autoencoder?** Data-specific, lossy, unsupervised (เฉพาะงานบีบข้อมูล; segmentation ยังต้องมี mask)

6. **ทำไม code size ต้องเล็กกว่า input?** ถ้า code ใหญ่เกินไป AE จะเรียนแค่ copy input ไป output ซึ่งคือ overfitting และไม่ได้ feature ที่มีประโยชน์ Bottleneck (รูปนาฬิกาทราย) บังคับให้มันเรียนรู้ feature ที่สำคัญ Code size เป็น hyperparameter ที่สำคัญที่สุด

7. **ทำไม network segmentation ต้องมี skip connection?** การ downsample ของ encoder ทำข้อมูลตำแหน่ง ("where") หาย skip connection ส่ง feature ตำแหน่งความละเอียดสูงจาก encoder ไป decoder ที่ scale เดียวกัน ทำให้การสร้างกลับและขอบวัตถุแม่นยำ

8. **Skip connection ของ U-Net vs LinkNet?** U-Net ใช้ concatenate (channel บวกเพิ่มกัน) LinkNet ใช้บวกแบบ element-wise (ต้อง shape เท่ากัน, ขนาด output ไม่เปลี่ยน, เบาและเร็วกว่า)

9. **บอกวิธี upsampling 3 วิธีและอธิบาย unpooling** Interpolation (nearest, bilinear), transposed conv, max unpooling ส่วน Unpooling ต้องเก็บ index ของ max-pool ไว้, เอาค่ากลับไปวางตำแหน่งเดิม, เติม 0 แล้วใช้ conv ทำให้ผลเรียบ

10. **คำนวณขนาด output ของ transposed conv: input 4×4, k=3, s=2, p=0** (4−1)·2 + 3 = **9×9** ถ้า p=1: 7×7 ถ้า s=1, p=0: 6×6

11. **Nearest-neighbour upsample [[10,20],[30,40]] เป็น 5×5** index map [0,0,1,1,1] ทั้งสองแกน ได้ `10 10 20 20 20` สองแถว และ `30 30 40 40 40` สามแถว

12. **ทำไม U-Net ใช้ 1×1 conv ตอนท้าย?** ทำนายทีละ pixel จาก feature ทุก channel ที่ pixel นั้น: เหมือน FC layer ที่ใช้ทีละ pixel และรักษา W×H ไว้ (นี่คือ quiz โบนัส)

13. **ออกแบบ output layer:**
    - (a) เส้นเลือดจอตา vs background: 1 channel + sigmoid (หรือ 2 + softmax);
    - (b) เนื้อเยื่อ 5 class, หนึ่ง class ต่อ pixel: 5 channels + softmax;
    - (c) โครงสร้าง 4 อย่างที่ซ้อนกันได้: 4 channels + sigmoid, label ต้องเป็น one-hot;
    - (d) map ความลึกหรืออุณหภูมิทีละ pixel: 1 channel (หรือจำนวนปริมาณที่ต้องการ) + linear/ไม่ใส่ activation, target แบบ class-label

14. **Dilated (atrous) conv คืออะไร และทำไม DeepLab ใช้?** จุดของ kernel เว้นระยะห่างกัน (dilation 2 คือเว้น 1) ทำให้ receptive field ใหญ่ขึ้นโดยไม่เพิ่ม parameter หรือการคำนวณ DeepLab เลยไม่ต้อง downsample ต่อและรักษาความละเอียดเชิงตำแหน่งไว้ได้ ASPP รวม dilation rate หลายค่าเพื่อได้บริบทหลาย scale V3+ เพิ่ม decoder ที่ใช้ได้จริงเพราะ decoder ของ V3 มีแค่ upsample ครั้งเดียว

15. **คำนวณ Dice/IoU: GT = 20 px, Pred = 25 px, ซ้อนกัน = 15 px (ภาพ 100 px)** IoU = 15/30 = 0.50 Dice = 30/45 ≈ 0.67 Pixel accuracy = 0.85 ซึ่งหลอกตาเพราะ background มีเยอะ สำหรับ mask ที่ไม่สมดุล ควรใช้ Dice/Jaccard loss (1 − score)

16. **PSPNet vs PAN?** PSPNet ใช้ Pyramid Pooling Module (pool เป็น 1×1, 2×2, 3×3, … เพื่อได้บริบททั้งภาพรวมและเฉพาะจุด), upsample แล้ว concatenate PAN ใช้ FPA (หลาย scale ด้วยขนาด filter แล้วเอาไปคูณ) และ GAU (high-level feature สร้าง channel weight ให้ low-level feature แล้วบวกเข้าไป): เป็น skip connection ที่ถ่วงน้ำหนักด้วย attention

---

## กับดักที่เจอบ่อย

- **Transposed conv ≠ inverse ของ convolution** ในแง่ค่า มันแค่ย้อนการเปลี่ยน *shape* ในวิชานี้มันคือสิ่งเดียวกับ "deconvolution" และ kernel **ไม่ต้องกลับด้าน**
- **Padding ใน transposed conv ทำให้ output เล็กลง** (−2p) ผมเช็กภาพสไลด์แล้ว: p.41 เขียนจริง ๆ ว่า "ยิ่งค่า Padding มากขึ้น ขนาดของภาพผลลัพธ์จะยิ่งขยายใหญ่ขึ้น" ("the more padding, the bigger the output") แต่ heatmap จาก PyTorch ในสไลด์ชุดเดียวกันขัดกับประโยคนี้ p.46 (s=1): p=1 ได้ 4×4, p=2 ได้ 2×2 (เทียบกับ 6×6 ที่ p=0) p.47 (s=2): p=1 ได้ 7×7, p=2 ได้ 5×5 (เทียบกับ 9×9 ที่ p=0) อาจารย์ก็พูดว่า padding ทำให้หดลง [00:49:02] ถ้าข้อสอบถามว่า padding มีผลกับขนาด output ใน transposed conv ยังไง ให้ตอบว่าทำให้ output เล็กลง ส่วนข้อ stride ใน p.41 ถูกต้อง: stride ยิ่งมาก output ยิ่งใหญ่
- **การนับ dilation:** conv ปกติคือ **dilation = 1** ไม่ใช่ 0 (อาจารย์แก้ที่ [00:55:17])
- **Dilated convolution ≠ morphological dilation**
- **จำนวน output channel = จำนวน class** ยกเว้น binary + sigmoid ที่ใช้ **1 channel**
- **Multiclass ≠ multilabel** Multiclass: หนึ่ง class ต่อ pixel, softmax Multilabel: หลาย class ต่อ pixel, sigmoid, ต้องใช้ label แบบ one-hot สไลด์โค้ดบางหน้า (p.87–90) ยังใส่ `activation='sigmoid'` ใน `smp.FPN(...)` ทั้งที่ `ACTIVATION='softmax2d'` อาจารย์ยอมรับว่าผิด multiclass ควรใช้ softmax(2d)
- **Regression segmentation:** ใช้ linear/ไม่ใส่ activation อย่าใช้ sigmoid หรือ softmax
- **"Autoencoder เป็น unsupervised"** จริงเฉพาะงานบีบ/สร้างข้อมูลกลับ Segmentation ต้องมี ground-truth mask
- **Encoder/decoder ไม่จำเป็นต้องสมมาตร** (DeepLabV3 มี decoder ขั้นเดียว)
- **Instance segmentation ไม่ label pixel ของ background** ส่วน Semantic ต้อง label ตามคอนเซ็ปต์
- **Max pooling ทำตำแหน่งของค่า max หาย** Unpooling ทำได้ก็ต่อเมื่อเก็บ index ไว้ตอน pooling
- **Pixel accuracy ถูกดันให้สูงเกินจริงเมื่อ background ใหญ่** ให้ใช้ IoU/Dice (Dice ≥ IoU สำหรับการทำนายเดียวกัน)
- **Segmentation เป็นงานที่กิน memory มากที่สุด** (memory โตตามขนาด input) การ stack U-Net หลายตัวอาจเกิน 24 GB VRAM
- ใน smp, `TrainEpoch(...)` ยังไม่ได้ train อะไรจนกว่าจะเรียก `.run(loader)`
- บั๊กโค้ดในสไลด์ p.81: `__len__` คืนค่า `len(self.img_dir)` (ความยาวของ string path) ที่ถูกควรเป็น `len(self.img_id)`
