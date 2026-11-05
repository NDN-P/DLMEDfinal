# Chapter 13 — Deep Learning สำหรับประมวลผลวิดีโอ

วิดีโอคือข้อมูล time series คือมีเฟรมเรียงต่อกันไปตามเวลา เลยมีทั้งข้อมูล **spatial** (สิ่งที่อยู่ในแต่ละเฟรม) และข้อมูล **temporal** (การเปลี่ยนแปลงระหว่างเฟรม) บทนี้สร้างเครื่องมือสำหรับทั้งสองอย่างทีละขั้น: RNN ต่อด้วย LSTM ต่อด้วย ConvLSTM (มีโค้ด PyTorch) ต่อด้วย 3D CNN / 3D max pooling แล้วจบที่ I3D, two-stream และ architecture วิดีโอสมัยใหม่

---

## แนวคิดหลัก

### 1. Time series กับวิดีโอ (สไลด์ 4–16)
- **Time series (ข้อมูลอนุกรมเวลา)**: ข้อมูลที่เก็บต่อเนื่องตามเวลา เรียงตามลำดับเวลา ใช้ทำ **trend analysis** (ดูแนวโน้ม), **seasonal analysis** (ดูฤดูกาล) และ **forecasting** (พยากรณ์) ตัวอย่างในสไลด์: ราคาหุ้น, อุณหภูมิรายวัน, ยอดขาย
- **วิดีโอในฐานะ time series**: คือเฟรมที่เรียงลำดับตามเวลา ซับซ้อนกว่า time series ทั่วไป เพราะมีทั้ง:
  - **Spatial feature (เชิงพื้นที่)**: สิ่งที่อยู่ *ในเฟรมเดียว* เช่น ขอบ, สี, รูปร่าง, texture, ตำแหน่ง ดึงออกมาด้วย **CNN (2D convolution filter)**
  - **Temporal feature (เชิงเวลา)**: เฟรมนี้ *เปลี่ยนไปยังไงเมื่อเทียบกับเฟรมก่อนๆ* เช่น การเคลื่อนไหว, ลำดับ, ทิศทาง, ความเร็ว, จังหวะเวลา ดึงออกมาด้วย **RNN / LSTM**
  - อาจารย์พูดว่า spatial ดูเนื้อหา **ภายใน** (ในเฟรม) ส่วน temporal ดูความสัมพันธ์ **ภายนอก** (ระหว่างเฟรม) [00:06:18]
- **Spatio-temporal**: ดึง feature spatial **และ** temporal **พร้อมกัน (พร้อมกัน)** ถ้าดึง spatial ก่อนแล้วค่อยเอาผลไปเข้าโมเดล temporal แบบนี้เรียกว่า "spatial ตามด้วย temporal" (S→T) **ไม่ใช่** spatio-temporal [00:06:59–00:07:23]
- วิธีรวมสองอย่างนี้: CNN + LSTM (CNN ทีละเฟรม แล้ว LSTM ตามเวลา) หรือ **3D CNN** ที่ใช้ filter 3 มิติจับ spatial และ temporal ได้ในการคำนวณครั้งเดียว (สไลด์ 8)
- **ตัวอย่างการใช้งาน** (สไลด์ 5):
  - Action recognition (เช่น เตะ, กระโดด, หมุนตัว ในวิดีโอกีฬา)
  - Event detection (เช่น รถชนในกล้อง CCTV, **ผู้สูงอายุล้ม**): CNN + RNN/LSTM
  - Behaviour analysis (เช่น ดูว่าคนเดินกันยังไงในสถานีรถไฟ): 3D CNN
- **โครงสร้างข้อมูลวิดีโอ**: แต่ละเฟรมเป็น H×W×C (rank-3) พอเอาเฟรมมาซ้อนกันจะได้ **rank-4 tensor** ("ข้อมูล 4-D") (สไลด์ 15, 73) ทบทวน: tensor rank 0 คือ scalar, 1 คือ vector, 2 คือ matrix, 3 ขึ้นไปคือ tensor

### 2. งานภาพ vs. งานวิดีโอ (lecture [00:12:31]–[00:18:19])
สไลด์ 10 ("ความเชื่อมโยงงานภาพในมุมมองงานวิดีโอ") แบ่งไว้ว่า ภาพ = มิติ spatial ส่วนวิดีโอ = มิติ temporal

| งานภาพ | งานวิดีโอที่คู่กัน | Output / ground truth (ตัวอย่างสไลด์ 10) |
|---|---|---|
| Image classification: หนึ่ง label ต่อภาพ | **Video classification**: หนึ่ง label ต่อคลิป สรุปทั้งคลิป | หนึ่ง class ต่อทั้งวิดีโอ (สไลด์: คลิปว่ายน้ำได้ "ว่ายน้ำ" (swimming); lecture: ยืน / นั่ง) |
| Image segmentation: label ทุก pixel | **Temporal action segmentation**: label **ทุกเฟรม** | หนึ่ง class ต่อเฟรม (สไลด์: คลิปทำอาหารติด label เตรียม → หั่น → ปรุง → จัดจาน; 100 เฟรมได้ vector 100 ตัว) |
| Object detection: class ของวัตถุ + bounding box | **Temporal action localization**: class ของ action + **เวลาเริ่ม–จบ** | เฉพาะช่วงเวลาของเหตุการณ์ที่สนใจ (สไลด์: "ยิงประตู 00:15–00:20" คือการยิงประตู และมีช่วงฟาวล์ด้วย); ไม่ได้ติด label ทุกเฟรม |
| Image regression | **Video regression** | ค่าตัวเลข (เช่น ประมาณ heart rate จากวิดีโอผิวหนัง) |

ท้ายสไลด์เขียนว่า: segmentation ตามเวลา ≠ segmentation ระดับ pixel ในวิดีโอ

### 3. ทำไมต้อง RNN? ข้อจำกัดของ NN/CNN (สไลด์ 18–20)
NN และ CNN ถูกออกแบบมาสำหรับ input ที่:
1. เข้ามา **ทีเดียวทั้งหมด** (ทั้งภาพหรือทั้ง vector; ป้อนครึ่งภาพไม่ได้),
2. **ไม่มีความสัมพันธ์ตามลำดับ** ที่มีผลต่อการทำนาย,
3. มี **ขนาดคงที่และเท่ากัน**

เพราะฉะนั้น NN/CNN ไม่เหมาะกับข้อมูลที่มีลำดับ หรือ input ที่ขนาดไม่เท่ากัน (วิดีโอยาวไม่เท่ากัน) architecture ใหม่ต้อง (1) รับ input ทีละส่วนได้ (2) ดึงข้อมูล temporal/ลำดับได้ และ (3) รับ input ขนาดต่างกันได้ คำตอบคือ **RNN** [00:23:05–00:24:19]
- RNN ทำนายจาก **สองแหล่ง**: **input ปัจจุบัน** และ **ข้อมูล/state ก่อนหน้า** (วงสีม่วงในสไลด์) ทำงานเหมือนคนอ่านหนังสือทีละคำ
- **Unfold**: จริงๆ มี **RNN cell แค่ตัวเดียว** (hidden layer) ที่ถูกใช้ซ้ำทุก time step "A recurrent neural network can be seen as the repeated use of a single RNN Cell" (สไลด์ 21)
- ตอนเริ่ม step แรกยังไม่มีข้อมูลก่อนหน้า เลยให้ H₀ = 0 [00:29:34]
- **RNN Cell vs. RNN Cell Forward** (สไลด์ 25): cell คำนวณ Hₜ (ส่วน feature extraction) ส่วน "forward" (Hₜ·W + b แล้ว softmax) แปลง Hₜ ให้เป็น output ส่วนนี้ **ใส่หรือไม่ใส่ก็ได้** และออกแบบใหม่ตามงานได้
- จะเอา output **ทุก time step** ก็ได้ (เช่น งานแปลภาษา) หรือ **เฉพาะ step สุดท้าย** ก็ได้ (เช่น video classification) [00:31:35–00:32:39]
- **ปัญหาลำดับยาว** (สไลด์ 26): ข้อมูลจาก step แรกๆ จะจางลงเรื่อยๆ เมื่อผ่านหลาย cell พอถึง cell ท้ายๆ แทบไม่เหลือข้อมูลจาก cell แรกเลย RNN เลยไม่เหมาะกับข้อมูลที่ยาวมาก

**ข้อดี / ข้อเสียของ RNN (สไลด์ 29)**
| ข้อดี | ข้อเสีย |
|---|---|
| รับ input ความยาวต่างกันได้ | คำนวณช้า |
| ขนาดโมเดลไม่โตตามขนาด input | ดึงข้อมูลจากอดีตไกลๆ ได้ยาก |
| การคำนวณคิดถึงข้อมูลในอดีต | ใช้ input ในอนาคตกับ state ปัจจุบันไม่ได้ |
| ใช้ weight ร่วมกันทุก time step | |

### 4. LSTM (สไลด์ 31–45)
- **LSTM** = RNN ชนิดหนึ่งที่เรียนรู้ **long-term dependency** (ความสัมพันธ์ระยะยาว) ได้ เสนอโดย **Hochreiter et al., 1997** ออกแบบมาเพื่อแก้ **long-term dependency problem**
- โครงสร้าง (แบบที่อาจารย์แบ่ง): **2 states + 3 gates + 2 "modulates"**
  - **2 states**
    - **Cell state = ความจำระยะยาว**: เก็บข้อมูลที่ *รวบรวม (รวบรวม) มาจากทุก step ก่อนหน้า* ตั้งแต่เริ่มจนถึงตอนนี้ มันแค่ถูกคูณด้วย f แล้วบวกเพิ่ม **ไม่เคยถูกบีบด้วย tanh** เลยเชื่อกันว่ามัน "ไม่ลืม"
    - **Hidden state = ความจำระยะสั้น**: เก็บ "ลักษณะเฉพาะ" ของข้อมูล เก็บไว้ **เพื่อทำนายต่อ** ("keep for predict") ส่วนใหญ่ได้อิทธิพลจาก step ล่าสุดๆ มัน **ไม่ใช่** ผลทำนายเอง แต่ส่งต่อไป forward layer เพื่อสร้าง output ได้
  - **3 gates (sigmoid, ช่วง 0 ถึง 1)**
    - **Forget gate (เรียกอีกชื่อว่า Remember gate) = "Delete"**: เก็บ cell state เก่า Cₜ₋₁ ไว้เท่าไหร่ 0 คือลบหมด, 1 คือเก็บหมด
    - **Input gate = "Write"**: ให้ข้อมูลใหม่ไหลเข้า cell state เท่าไหร่ 0 คือไม่ให้เข้าเลย, 1 คือให้เข้าหมด
    - **Output gate = "Read"**: เปิดให้ cell state ออกไปข้างนอกเป็น hₜ เท่าไหร่ แค่อ่านอย่างเดียว ไม่ได้แก้ cell state
  - **2 modulates (tanh, ช่วง −1 ถึง 1)**
    - **Input modulate gate** (zₜ มักเรียกว่า "candidate" หรือ g): เตรียมเนื้อหาใหม่ที่จะเข้าไป จาก xₜ และ hₜ₋₁
    - **Output modulate gate**: tanh(Cₜ) ก่อนเอาไปคูณกับ oₜ **ชื่อนี้อาจารย์ตั้งเอง** ("ส่วนนี้ผมตั้งเอง") ส่วนที่เป็นทางการมีแค่ input, forget, output gate กับ input modulate gate
- **หน้าที่ของ sigmoid vs. tanh**: **sigmoid** ใช้สร้าง **weight / gate** (0 คือไม่สนใจ, 1 คือเอาหมด) ส่วน **tanh** ใช้ **normalise ข้อมูล/information** ให้อยู่ในช่วง −1…1 (ข้อมูลที่มีศูนย์กลางที่ 0 เรียนรู้ได้เร็วกว่า)
- **Weight ของ gate ปรับตาม input (adaptive)**: ทุก gate คำนวณจาก **xₜ และ hₜ₋₁** ค่าเลยเปลี่ยนไปตาม input ทุกครั้ง แม้ตอน inference ต่างจาก weight ของ CNN ที่คงที่หลัง train เสร็จ แต่ละ gate ยังมี **W และ b ที่ learn ได้เป็นของตัวเอง** gate แต่ละตัวเลยไม่เหมือนกัน [01:09:53–01:12:36]
- **ตัวอย่างเปรียบเทียบที่อาจารย์ใช้**: input gate เหมือนอ่านหนังสือแล้วเข้าใจและจำได้จริงแค่ไหน forget gate เหมือนความจำระยะยาวเก่าๆ เหลืออยู่แค่ไหนเมื่อเวลาผ่านไป output gate เหมือนตอบเฉพาะที่ถูกถาม (ถามเรื่องบวก ก็ไม่ต้องอธิบายเรื่องคูณ)

### 5. RNN vs LSTM vs GRU
| | RNN | LSTM | GRU* |
|---|---|---|---|
| States | มีแค่ hidden Hₜ | cell state Cₜ (ยาว) + hidden hₜ (สั้น) | มีแค่ hidden |
| Gates | ไม่มี (มี tanh ตัวเดียว) | forget, input, output (+ tanh input modulate) | update, reset |
| ความจำระยะยาว | แย่ (ข้อมูลแรกๆ จางหาย) | ดี (เส้นทาง cell state แบบบวก) | ดี |
| ต้นทุน | ต่ำสุด | สูงสุด | อยู่ระหว่างสองตัว |

*GRU **ไม่มีในสไลด์** อาจารย์พูดถึงแค่ครั้งเดียว ว่าเป็นทางเลือกแทน LSTM/RNN หลังได้ feature จาก I3D [01:50:14] คอลัมน์ GRU เป็นความรู้ทั่วไป ไม่ใช่เนื้อหาวิชา

### 6. Pipeline สำหรับวิดีโอและสัญญาณ (สไลด์ 16, 46, 50, 97, 103, 104)
อาจารย์แบ่งเป็น 4 แบบ [01:23:30–01:25:22]:
1. **CNN → feature vector ต่อเฟรม → LSTM → prediction** ปกติ CNN จะตัดที่ layer **flatten** แบบนี้เป็น **S→T (ทำตามลำดับ)** ไม่ใช่ spatio-temporal
2. **ทำ feature extraction เอง** (filter ที่ออกแบบเอง เช่น edge filter −1 −1 −1 / 0 0 0 / 1 1 1, MFCC/spectrogram สำหรับสัญญาณ) → feature vector → LSTM *หรือ* feature map → CNN
3. **ConvLSTM** → NN (ใส่หรือไม่ใส่ก็ได้) → prediction (spatio-temporal)
4. **3D CNN** → NN (ใส่หรือไม่ใส่ก็ได้) → prediction (spatio-temporal)
- สำหรับสัญญาณ: ตัดสัญญาณเป็นท่อนๆ แปลงแต่ละท่อนเป็น feature vector แล้วป้อนเข้า RNN/LSTM ตามลำดับ ถ้า feature หน้าตาเหมือนภาพ (เช่น spectrogram) จะป้อนเข้า CNN แทนก็ได้ [01:20:56–01:23:11]

### 7. ConvLSTM (สไลด์ 53–56; lecture [01:25:30–01:34:38])
- **Convolutional LSTM** เสนอโดย **Xingjian Shi et al., 2015** เป็น recurrent network สำหรับทำนายแบบ **spatial + temporal** (เช่น หนัง, ภาพดาวเทียม, กล้องวงจรปิด) มี 2 ส่วนหลัก: **convolution layer + LSTM**
- ฐานคือ **LSTM** ("เขาเอา LSTM เป็นพื้นฐาน… นี่คือจุดเน้น" [01:34:05]) สิ่งที่เปลี่ยนจาก LSTM:
  - **การคูณ matrix** กับ input xₜ และ hidden hₜ₋₁ ถูกเปลี่ยนเป็น **convolution**,
  - state เลยเป็น **feature map (2D)** ไม่ใช่ vector 1D,
  - การคูณแบบ **element-wise** (∘) ยังเหมือนเดิมใน f∘C, i∘g และ o∘tanh(C)
- **ทำงานยังไง (PyTorch cell, สไลด์ 64)**: เอา [xₜ, hₜ₋₁] มาต่อกันตามแกน channel ใช้ `Conv2d` ตัวเดียวที่ `out_channels = 4 × hidden_dim` ได้ pre-activation ครบทั้ง 4 ตัว แล้วแบ่งเป็น i, f, o, g จากนั้น i, f, o = sigmoid และ g = tanh, `c_next = f*c + i*g` และ `h_next = o*tanh(c_next)` ตัวอย่างบนกระดานของอาจารย์: hidden_dim = 3 ได้ 12 filter แบ่งเป็น 4 กลุ่ม กลุ่มละ 3 [01:27:37–01:28:49]
- **ConvLSTM เป็น spatio-temporal (ST)**: filter 2D ให้ feature spatial และเพราะ channel ที่มันผสมกันมาจาก **คนละเวลา** (xₜ และ hₜ₋₁) มันเลยเป็น temporal ไปพร้อมกัน [01:32:49–01:33:47]

### 8. สร้าง ConvLSTM ใน PyTorch (สไลด์ 58–69)
5 ขั้นตอนปกติ: (1) สร้าง Dataset (2) สร้างโมเดล ConvLSTM (3) เลือก loss และ optimizer (4) train (5) test/evaluate
- **Dataset**: input ของ video classification เป็น **5-D**: (Batch, Frame, Width, Height, Channel) ตามที่เขียนในสไลด์ 59 ไม่มี dataset สำเร็จรูป เลยต้องเขียน `VideoDataset(Dataset)` เอง:
  - `__init__`: เก็บ path และ label ของไฟล์ `.mp4/.avi/.mov` ที่เจอในแต่ละโฟลเดอร์ class
  - `__getitem__`: `load_video` ได้ numpy array ของเฟรม แล้ว transform **ทีละเฟรม** จากนั้น `torch.stack` เป็น tensor แล้วคืน label index
  - `load_video`: `cv2.VideoCapture` แล้ว `cap.read()` วนลูป `ret` เป็น True ถ้าอ่านได้ และ False ถ้าอ่านไม่ได้ ซึ่งจะหยุดลูป resize ทุกเฟรม หยุดอ่านที่ `max_frames` และ `cap.release()` อยู่ใน `finally`
- **Parameter ในสไลด์**: classes `['Biking','Diving','HorseRiding']`, `batch_size=4`, `max_frames=30`, `resize=(112,112)`, `Normalize(mean=0.5, std=0.5)`
- **num_workers**: 0 หรือ 1 คือใช้ CPU โหลดข้อมูลแค่ instance เดียว มากกว่า 1 ก็สร้าง instance ตามจำนวนนั้น ไม่ควรเกินจำนวน core ของ CPU
- **โมเดล**: PyTorch ไม่มี ConvLSTM ในตัว เลยเอา class มาจาก repo ndrplz/ConvLSTM_pytorch สไลด์ใช้ `ConvLSTM(input_dim=3, hidden_dim=[64,64], kernel_size=(3,3), num_layers=2, batch_first=True)` เอา **hidden state ตัวสุดท้าย** ของ layer สุดท้าย มา flatten แล้วส่งเข้า `nn.Linear(112*112*64, num_classes)`
- **num_layers** คือจำนวน ConvLSTM layer ที่ซ้อนกัน FC layer จะรับ **ทุก time step** หรือ **แค่ step สุดท้าย** ก็ได้ [01:39:07–01:39:24]
- **Loss/optimizer**: `nn.CrossEntropyLoss()` กับ `optim.SGD(lr=0.001, momentum=0.9)` เลือกตามงาน: regression, multiclass, multilabel หรือ multitask
- **Training loop**: `zero_grad`, forward, loss, `backward`, `step` train 10 epoch แล้ว `torch.save(model.state_dict(), ...)`
- **Testing**: `model.eval()`, `torch.no_grad()`, `torch.max(outputs, 1)` และ `sklearn.metrics.accuracy_score`
- **Inference วิดีโอเดียว**: โหลดเฟรม transform ทีละเฟรม `torch.stack` แล้ว **`unsqueeze(0)`** เพื่อเพิ่มมิติ batch รันโมเดล `softmax` แล้ว `torch.max` เพื่อเอา class ที่ทำนายและ % ของมัน

### 9. 3D CNN (สไลด์ 72–92)
- **3D CNN**: CNN ที่ออกแบบมาสำหรับข้อมูล 3-D ใช้ **memory และเวลามากกว่า** 2D แต่ปกติ **แม่นกว่า** 2D เมื่อใช้กับข้อมูล 3-D
- **2D vs. 3D convolution** (สไลด์ 75):
  - **2D**: filter เลื่อนแค่ตาม **x และ y** และทุกตำแหน่งใช้ **ทุก channel** (ความลึกของ filter เท่ากับความลึกของ input)
  - **3D**: filter เลื่อนตาม **x, y และ z (เวลา)** ความลึกของ filter **น้อยกว่า** ของ input ได้
- **3D conv เป็นการคำนวณแบบ spatio-temporal**: ดึง feature spatial ของแต่ละเฟรม และความสัมพันธ์ระหว่างเฟรมที่ติดกันได้ในทีเดียว (สไลด์ 77)
- **3D max pooling** มักมาคู่กับ 3D conv ถ้า stride 2 จะลดทุกมิติลงครึ่งหนึ่ง (4×4×4 กลายเป็น 2×2×2)
- **ข้อดี/ข้อเสีย** (สไลด์ 92): แม่นกว่า 2D และได้ spatial + temporal พร้อมกัน **แต่** มี **parameter เยอะ** เลย **เสี่ยง overfitting สูง** และมี **pretrained model น้อย**

### 10. ปัญหาของ action recognition, Two-Stream และ I3D (สไลด์ 98–102)
- **ปัญหาใน action recognition**:
  - dataset วิดีโอที่ใช้กันบ่อย (**UCF-101, HMDB-51**) มีขนาดเล็ก โมเดลซับซ้อนเลย overfit,
  - ConvNet สำหรับภาพเก่งแล้ว แต่เอามาใช้กับเรื่องเวลายาก,
  - 3D ConvNet มี parameter เยอะ train ยาก
- **Two-Stream** ประมวลผล "**RGB frames**" กับ "**Optical Flow** (การเคลื่อนไหว)" แยกกัน ส่วนใหญ่ใช้ 2D ConvNet แล้ว **รวม (fuse)** ผลตอนท้าย (late fusion) อาจารย์อธิบาย optical flow แบบง่ายๆ ว่า "เฟรมลบเฟรม" (เฟรม 1 − เฟรม 2, 2 − 3, …) เพื่อให้เห็นการเปลี่ยนแปลงระหว่างเฟรม [01:49:04–01:49:41] จริงๆ แล้ว optical flow คือการประมาณการเคลื่อนที่ราย pixel แต่การลบเฟรมคือภาพในหัวที่อาจารย์ให้ไว้
- **I3D (Inflated 3D ConvNet)**: "พอง" (inflate) โมเดล 2D ที่ออกแบบมาดีแล้ว (**Inception-v1**) ให้เป็น 3D เลยมองเห็น spatial และ temporal พร้อมกัน
  - **Inflation**: filter และ pooling 2D ขนาด N×N กลายเป็น **T×N×N** (T = ขนาดตามเวลา)
  - จุดออกแบบสำคัญ: **temporal stride/pooling** ซึ่งกำหนด temporal receptive field ให้พอดี
  - **Two streams** (RGB + optical flow) แต่ละ stream ผ่าน 3D ConvNet, Average Pool 3D, 1×1 conv และได้ score (1×nc) เอา score สองตัวมาเฉลี่ย แล้ว softmax
  - I3D ถูก pretrain มากับวิดีโอ เลยใช้เป็น feature extractor แล้วต่อด้วย LSTM/RNN/GRU ได้ [01:50:05]
- **1×1×1 convolution ใน I3D**: ขนาดเวลา 1 และขนาดพื้นที่ 1×1 ไม่ได้ดูทั้งพื้นที่และเวลา แค่เป็น **linear combination ของ channel** หน้าที่:
  - **ลด channel** (ลดการคำนวณและ memory),
  - **ผสม channel**,
  - **เพิ่ม non-linearity** (เมื่อใช้กับ ReLU + BatchNorm),
  - **ความยืดหยุ่น** ใน branch ของ Inception (feature สั้นๆ เบาๆ)

### 11. Architecture วิดีโอสมัยใหม่ (สไลด์ 93–96; แค่ชื่อ)
| Architecture | ปี |
|---|---|
| SlowFast (Feichtenhofer et al., ICCV) | 2019 |
| X3D | 2020 |
| Temporal Pyramid Network | 2020 |
| MViTv2 | 2021 |
| Video Swin Transformer | 2021 |
| VideoMAE | 2022 |
| UniFormer | 2023 |
| TimeFormer | 2025 |

- อาจารย์อธิบาย SlowFast ว่าเป็นการรวมมุมมองแบบหยาบกับแบบละเอียดเข้าด้วยกัน [01:46:41]
- architecture สำเร็จรูปส่วนใหญ่ข้างในใช้ ConvLSTM หรือ 3D convolution ไม่อย่างนั้นก็ดึง feature spatial ก่อนแล้วค่อยจัดการมิติ temporal [01:50:57]

### สรุป: architecture สำหรับวิดีโอ
| วิธี | Spatial | Temporal | S→T หรือ ST? | หมายเหตุ |
|---|---|---|---|---|
| CNN (ทีละเฟรม) + LSTM/RNN | 2D CNN ถึง flatten | LSTM | **S→T** | ง่าย ใช้ CNN ภาพที่ pretrained ได้ |
| Feature ที่ออกแบบเอง + LSTM / CNN | ทำเอง | LSTM | S→T | เช่น MFCC, edge filter |
| ConvLSTM | conv อยู่ใน gate | LSTM recurrence | **ST** | state เป็น feature map |
| 3D CNN (+3D max pool) | 3D kernel | 3D kernel ตามเวลา | **ST** | param เยอะ เสี่ยง overfit pretrained น้อย |
| Two-Stream | RGB stream | optical-flow stream | ขนานกัน แล้ว late fusion | ส่วนใหญ่ใช้ 2D ConvNet |
| I3D | Inception 3D ที่ inflate แล้ว | 3D + flow stream | ST (two-stream) | pretrained กับวิดีโอ |

---

## สูตรและตัวเลขที่ต้องจำ

### RNN (สไลด์ 22–24)
- **Hₜ = tanh(Xₜ·Wₓₜ + Hₜ₋₁·Wₕₜ + Bₜ)**: input ปัจจุบันบวกข้อมูลก่อนหน้า นี่คือส่วน feature extraction
- **Yₜ = softmax(Hₜ·Wᵧₜ + Bᵧₜ)**: output คือส่วน "RNN Cell Forward" ที่ใส่หรือไม่ใส่ก็ได้
- H₀ = 0 ที่ step แรก

### LSTM (สไลด์ 37–44)
- Forget: **fₜ = σ(W_xf·xₜ + W_hf·hₜ₋₁ + b_f)**, ช่วง 0…1
- Input: **iₜ = σ(W_xi·xₜ + W_hi·hₜ₋₁ + b_i)**, ช่วง 0…1
- Output: **oₜ = σ(W_xo·xₜ + W_ho·hₜ₋₁ + b_o)**, ช่วง 0…1
- Input modulate: **zₜ = tanh(W_xz·xₜ + W_hz·hₜ₋₁ + b_z)**, ช่วง −1…1
- Cell update: **Cₜ = fₜ ∘ Cₜ₋₁ + iₜ ∘ zₜ** (ส่วนเก่าที่เก็บไว้ บวกส่วนใหม่ที่เพิ่มเข้ามา)
- Hidden ("output modulate"): **hₜ = oₜ ∘ tanh(Cₜ)**
- 2 states, 3 gates, 2 modulates sigmoid ใช้กับ gate/weight; tanh ใช้กับข้อมูล LSTM มาจากปี 1997 (Hochreiter et al.)

### ConvLSTM (สไลด์ 56)
- iₜ = Sigmoid(Conv(xₜ; w_xi) + Conv(hₜ₋₁; w_hi) + b_i)
- fₜ = Sigmoid(Conv(xₜ; w_xf) + Conv(hₜ₋₁; w_hf) + b_f)
- oₜ = Sigmoid(Conv(xₜ; w_xo) + Conv(hₜ₋₁; w_ho) + b_o)
- gₜ = Tanh(Conv(xₜ; w_xg) + Conv(hₜ₋₁; w_hg) + b_g) ตรงนี้ gₜ คือ input modulate / candidate ซึ่งก็คือ zₜ ในสไลด์ LSTM
- cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ
- hₜ = oₜ ⊙ Tanh(cₜ)
- สรุปสั้นๆ: คือสมการ LSTM ที่เปลี่ยน W·x และ W·h เป็น **Conv(x; w) และ Conv(h; w)** ส่วนการคูณ ⊙ ยังเป็น element-wise
- PyTorch cell: `Conv2d(in = input_dim + hidden_dim, out = 4·hidden_dim, padding = k//2)` แบ่ง output เป็น i, f, o, g ใช้ padding k//2 เพื่อให้ H×W เท่าเดิม
- Element-wise product = คูณตำแหน่งต่อตำแหน่ง เช่น [1,2,3,4] ∘ [1,0,0,−1] = [1,0,0,−4]
- ConvLSTM มาจากปี 2015 (Shi et al.)

### Tensor shape (ระวังลำดับแกน)
| ที่ไหน | Shape |
|---|---|
| วิดีโอใน memory | rank-4: frames × H × W × C |
| ข้อความใน Dataset (สไลด์ 59) | (Batch, Frame, Width, Height, Channel) |
| โค้ด ConvLSTM, `batch_first=True` (สไลด์ 63) | **(B, T, C, H, W)** |
| `nn.Conv3d` / สไลด์ I3D | **(B, C, T, H, W)** = (N, C, D, H, W) |

- **ตัวอย่าง ConvLSTM จากสไลด์**: input (4, 30, 3, 112, 112) hidden state ตัวสุดท้ายเป็น (4, 64, 112, 112) `.view(4, -1)` ได้ (4, 802,816) และ FC ได้ (4, 3)
- **วิดีโอเดียวตอน inference**: stack เฟรมได้ (30, 3, 112, 112) แล้ว `unsqueeze(0)` ได้ (1, 30, 3, 112, 112)
- **I3D (สไลด์ 99)**:
  - วิดีโอ (25×3×16×224×224) = (Batch×Channel×Time×H×W) input optical flow ก็ shape เดียวกัน
  - หลังผ่าน 3D ConvNet ได้ feature map (25×2048×2×7×7): เวลา 16 เหลือ 2 (÷8) และพื้นที่ 224 เหลือ 7 (÷32)
  - Feature vector สุดท้าย (25×1×2048) score ต่อ stream (1×nc) แล้วเฉลี่ยและ softmax

### ขนาด output ของ 3D conv / pooling
ต่อมิติ (D/T, H, W):  **O = ⌊(I + 2P − K) / S⌋ + 1** นี่คือสูตรมาตรฐาน สไลด์แสดงผ่านตัวอย่าง

- **ตัวอย่างในสไลด์ (สไลด์ 78–83)**: ข้อมูล D ขนาด 4×4×4, filter F ขนาด 3×3×3, stride 1, ไม่มี padding ได้ output O เป็น **2×2×2** ((4−3)/1+1 = 2)
  - filter แต่ละแผ่นคือ [[1,−1,−1],[−1,1,−1],[−1,−1,1]]
  - ข้อมูลแต่ละแผ่นคือ:
    - D[0] = [[3,2,1,3],[0,7,3,3],[3,9,3,4],[2,3,8,3]]
    - D[1] = [[0,2,1,3],[0,4,3,2],[3,9,3,4],[2,6,8,1]]
    - D[2] = [[3,2,1,1],[0,3,3,3],[3,4,3,4],[2,3,6,3]]
    - D[3] = [[3,2,1,3],[0,3,6,3],[0,9,3,4],[2,3,1,3]]
  - วิธีทำ: วาง F[0], F[1], F[2] ทับข้อมูล 3 แผ่นที่ติดกัน (D[0..2] สำหรับ O[0], D[1..3] สำหรับ O[1]) คูณ element-wise รวมแต่ละแผ่น แล้ว **บวกผลรวมของ 3 แผ่นเข้าด้วยกัน** จากนั้นเลื่อนตาม x, y **และ** z (ความลึก)
  - ตารางสุดท้ายในสไลด์ 83: **O[0] = [[−20, −37], [−15, −55]]**, **O[1] = [[−24, −29], [−22, −55]]**
  - **สไลด์คิดเลขผิดบางจุด ผมคำนวณใหม่ทุกค่าจากข้อมูลในสไลด์:**
    - O[0][0,0]: สไลด์ 80/81 แสดง **−21** แผ่นแรกเขียนว่า "3+0−3−2+7−9−**2**−3+3 = −6" แต่พจน์นั้นคือ 1(−1) ผลรวมแผ่นเลยเป็น **−5** ค่าที่ถูกคือ −5 + (−11) + (−4) = **−20** ซึ่งตรงกับตารางสุดท้าย
    - O[0][0,1]: สไลด์แสดง −37 = −17 − 13 − 7 แผ่นที่สาม "2−3−4−1+3−3−1−3+4" จริงๆ ได้ **−6** ไม่ใช่ −7 ค่าที่ถูกเลยเป็น **−36**
    - O[1][1,0]: สไลด์แสดง −22 คำนวณใหม่ได้ **−18**
    - ช่องอื่นถูกหมด: −15, −55, −24, −29, −55
    - **Output ที่ถูก: O[0] = [[−20, −36], [−15, −55]], O[1] = [[−24, −29], [−18, −55]]**
  - ในข้อสอบให้แสดงวิธีทำ ถ้าจะอ้างตารางในสไลด์ ให้รู้ไว้ว่ามี 3 ช่องนี้ที่ไม่ตรง
  - อาจารย์อธิบายไว้ [01:42:25–01:43:22]: เลื่อน filter ตัวเดิมไปตาม x, y **และ z**
- **3D max pooling (สไลด์ 84–89)**: 4×4×4 (D ตัวเดิม) ใช้ window 2×2×2 และ stride 2 ได้ **2×2×2** เอาค่า max ของแต่ละก้อน 2×2×2 ผลคือ **O[0] = [[7, 3], [9, 8]]**, **O[1] = [[3, 6], [9, 6]]** ผมตรวจแล้ว ในสไลด์ถูกต้อง อาจารย์บอกว่าปกติจะลดทุกมิติลงครึ่ง (depth/channel, width, height) และใช้ stride 2
- **`nn.Conv3d(16, 33, 3, stride=2)`** (สไลด์ 91) หมายถึง input 16 channel, output 33 channel (= จำนวน filter), kernel 3×3×3 และ stride 2 ทุกมิติ
  - ใช้ input จากสไลด์ `torch.randn(20, 16, 10, 50, 100)` (N, C, D, H, W): D = ⌊(10−3)/2⌋+1 = 4, H = ⌊(50−3)/2⌋+1 = 24, W = ⌊(100−3)/2⌋+1 = 49 เลยได้ **output เป็น (20, 33, 4, 24, 49)** ผมคำนวณเอง สไลด์ไม่ได้แสดง
  - ตัวอย่างที่สองในสไลด์: `Conv3d(16, 33, (3,5,2), stride=(2,1,1), padding=(4,2,0))` กับ input เดิมได้ **(20, 33, 8, 50, 99)** อันนี้ผมคำนวณเองเหมือนกัน
  - จำนวน parameter ของ `Conv3d(16,33,3)`: 33·16·27 + 33 = 14,289 ผมคำนวณเอง
- **ตัวอย่างซ้อน channel ของอาจารย์ [01:41:45]**: เอาเฟรม RGB มาซ้อนเป็น channel และใช้ filter 3D ลึก 6 แปลว่า filter ครอบ **ทีละ 2 เฟรม** แล้วเลื่อนไปตามแกนเวลา/channel ทุก window เลยรวมข้อมูล spatial กับข้อมูลจากมากกว่าหนึ่งเฟรม จึงเป็น spatio-temporal

---

## สิ่งที่อาจารย์เน้น

- **[00:01:38]–[00:01:55] วิดีโอคือ time series** ภาพแต่ละภาพในวิดีโอเรียกว่า **frame** ในคลาสนี้เรียนแค่ส่วนภาพ ไม่เรียนเสียง
- **[00:06:18]** Spatial คือภายใน (ในเฟรม); temporal คือภายนอก (ความสัมพันธ์ระหว่างเฟรม)
- **[00:06:59]–[00:07:23] นิยามของ spatio-temporal** คือพิจารณาทั้งสองอย่าง **พร้อมกัน** "ถ้าเราพิจารณา spatial ก่อน… ค่อยเอาผลลัพธ์ที่ได้ไปพิจารณาต่อ temporal เราจะไม่เรียกเป็น spatial temporal" (ถ้าทำ spatial ก่อนแล้วค่อย temporal จะไม่เรียกว่า spatio-temporal)
- **[00:10:22]–[00:10:49] หลักจำง่ายๆ**: spatial ใช้ convolution; temporal ใช้ตระกูล RNN/LSTM; ทั้งสองพร้อมกันใช้ 3D convolution หรือ ConvLSTM "**คีย์เวิร์ดก็คือพร้อมกัน**" (คำสำคัญคือ *พร้อมกัน*)
- **[00:11:35]–[00:12:21] ทำไม temporal สำคัญ** ถ้ามีแค่เฟรม 1 กับเฟรม 5 แบบไม่รู้ลำดับ จะ **แยกไม่ออกว่า "ลุกขึ้นยืน" หรือ "นั่งลง"** spatial อย่างเดียว "ไม่เพียงพอ" (ไม่พอ)
- **[00:12:31]–[00:18:19]** งานภาพจับคู่กับงานวิดีโอ (classification, temporal action segmentation, temporal action localization, video regression) ดูตารางด้านบน ตัวอย่างออกแนวการแพทย์: ตรวจการทำกายภาพบำบัดที่บ้านของผู้สูงอายุ, ตรวจจับการล้ม, วิเคราะห์ขั้นตอนการผ่าตัด, และวัด heart rate จากวิดีโอผิวหนัง
- **[00:19:17]–[00:19:55]** CNN backbone ทีละเฟรมแล้วตามด้วย LSTM คือ "spatial แล้วค่อย temporal" **ไม่ใช่** spatio-temporal
- **[00:22:23] "ขนาด input เท่ากันอันเนี้ยสําคัญมาก"** (input ขนาดเท่ากันสำคัญมาก) วิดีโอยาวไม่เท่ากัน เลยต้อง **ตัด/คลิปให้ยาวเท่ากัน** หรือใช้เครื่องมือที่รับความยาวไม่เท่ากันได้ เช่น RNN
- **[00:23:14]–[00:24:15]** คุณสมบัติ 3 ข้อที่ architecture ใหม่ต้องมี: รับ input ทีละส่วน, ได้ feature temporal, รับขนาดต่างกันได้ คำตอบคือ RNN
- **[00:26:53]–[00:27:41] คำถาม: data vs. information** ("ไฮไลท์และกาดอกจัน" ให้ไฮไลท์และติดดาว) Information คือสิ่งที่เหลือหลังประมวลผล data แล้ว เป็นสิ่งที่เกี่ยวข้องและใช้ได้ ส่วน data อาจมี noise และของที่ไม่เกี่ยวปนอยู่ xₜ คือ data; h และ C คือ information
- **[00:28:44]–[00:28:54]** Hidden output แทน **ทุกเฟรมก่อนหน้า** ไม่ใช่แค่เฟรมสุดท้าย เพราะมันสะสมมาเรื่อยๆ
- **[00:29:34]** ที่ step แรก ข้อมูลก่อนหน้าเป็น 0
- **[00:30:06]–[00:30:53] คำถามในห้อง: tanh กับ sigmoid ตัวไหนอยู่ช่วง 0…1 ตัวไหน −1…1?** คำตอบ: **sigmoid คือ 0 ถึง 1; tanh คือ −1 ถึง 1** อาจารย์ย้ำอีกที่ [00:53:04]: "เด็กๆต้องจําได้ก่อน… หาที่จดครับ" (ต้องจำให้ได้ จดไว้)
- **[00:31:04]** ใน RNN tanh ทำหน้าที่ **normalisation** คุมค่าให้อยู่ในช่วงมาตรฐาน
- **[00:32:29]–[00:32:39]** เอา output ทุก step เฉพาะเมื่อจำเป็น สำหรับ video classification ปกติเอา **แค่ตัวสุดท้าย**
- **[00:34:29]** Weight ทำหน้าที่เป็น **filter** ที่เลือกว่าจะใช้อะไรจากข้อมูลปัจจุบันและข้อมูลอดีต
- **[00:38:26]** "RNN Cell ไม่เหมาะสำหรับข้อมูลที่เป็น Time Series ในลักษณะที่ยาวมาก" (RNN cell ไม่เหมาะกับ time series ที่ยาวมาก)
- **[00:40:15]–[00:40:31]** ข้อดี "รับความยาวต่างกันได้" หมายถึงแค่ว่ามัน **ทำได้** ไม่ได้แปลว่ามันทำได้ **ดี**
- **[00:41:43]–[00:42:31]** LSTM ส่งข้อมูลไป **สองเส้นทาง**: เส้นยาว (รับประกันว่ามีข้อมูลตั้งแต่ step แรกถึง step สุดท้าย) และเส้นสั้น (ส่วนใหญ่เป็น step ล่าสุดๆ) LSTM มาหลัง RNN ในปี **1997**
- **[00:53:22]–[00:54:10], [01:20:43]** **Sigmoid ทำหน้าที่เป็น weight/gate** (input, forget, output gate) **Tanh ใช้ normalise data/information**
- **[00:56:14]–[00:57:19]** **Cell state ไม่เคยถูก normalise (ไม่มี tanh)** มันแค่สะสมด้วยการบวก ("เหมือนเราเก็บเงินไปเรื่อยๆ" เหมือนเก็บเงินออม) เลยไม่ลืม ส่วนเส้นทาง hidden ถูก normalise เพราะต้องเอาไปประมวลผลต่อ
- **[00:59:22]–[01:11:13] Weight ของ gate ปรับตาม input (adaptive)** input เปลี่ยน weight ก็เปลี่ยน แม้ตอน inference ไม่เหมือน weight ของ CNN ที่คงที่ (อาจารย์เทียบกับ Transformer) gate ทั้งสามสร้างจาก **x และ hidden state ก่อนหน้า**
- **[01:11:22]–[01:12:36]** แต่ละ gate มี **W และ b ที่ learn ได้เป็นของตัวเอง** สูตรเดียวกันเลยได้ gate ต่างกัน
- **[01:00:16]–[01:00:31] และ [01:14:27] "แต่ย้ำนะครับ"** (ขอย้ำ): "**output modulate gate**" เป็นชื่อที่อาจารย์ตั้งเอง ที่เป็นทางการมีแค่ input, forget, output gate กับ input modulate gate
- **[01:01:57] "ขีดเส้นใต้คํานี้กาดอกจันครับ — รวบรวม ทั้งหมด"** (ขีดเส้นใต้และติดดาวคำนี้: "รวบรวมทั้งหมด") cell state *สะสม* ข้อมูลทั้งหมด
- **[01:03:00]–[01:03:32]** สไลด์บอกว่า cell state เก็บ "มากกว่า" (*มากกว่า*) ไม่ใช่ "ครบ" (*ครบถ้วน*) ข้อมูลหายไปที่ forget gate และข้อมูลใหม่ก็เข้ามาได้แค่บางส่วนผ่าน input gate
- **[01:04:34]** คำสำคัญของ hidden state: "**keep for predict**"
- **[01:07:42]–[01:08:02], [01:12:46]–[01:14:11]** Forget gate = remember gate = **Delete** Input gate = **Write** Output gate = **Read** write และ delete เปลี่ยน cell state ส่วน read ไม่เปลี่ยน
- **[01:21:02]–[01:23:11]** สัญญาณ (และวิดีโอในงานยุคก่อน) ถูกแปลงเป็น feature vector ก่อน (MFCC, spectrogram, …) แล้วค่อยป้อนเข้า LSTM ถ้า feature หน้าตาเหมือนภาพ ใช้ CNN แทนได้
- **[01:23:34]–[01:23:48]** ปกติตัด CNN ที่ layer **flatten** เพื่อเอา feature vector ต่อเฟรม
- **[01:25:25] "เดี๋ยวจะเป็นคําถาม"** (เดี๋ยวจะเป็นคำถาม): ConvLSTM เป็น S→T หรือ ST
- **[01:31:24]–[01:33:47] QUIZ ในห้อง ("พิมพ์แค่หนึ่งกับสอง" พิมพ์แค่ 1 หรือ 2)**
  - **คำถาม**: ConvLSTM ใช้แนวคิด LSTM เหมือนเดิม แต่เปลี่ยน operator ข้างในเป็น 2D convolution ข้อมูลเลยเป็น map 2D แทน vector 1D มันเป็น **(1) S→T** (spatial แล้วค่อย temporal) หรือ **(2) ST** (spatio-temporal ทั้งสองพร้อมกัน)?
  - **คำตอบที่ถูก: 2, ST (spatio-temporal)** คนส่วนใหญ่ในห้องตอบ 2 และอาจารย์บอกว่า "ถูก" (ถูกต้อง)
  - **เหตุผล**: filter 2D ที่วิ่งบน map ให้ feature spatial และเพราะ channel ที่เอามาต่อกันมาจาก **คนละเวลา** (เฟรม xₜ และ hidden hₜ₋₁) convolution ตัวเดียวกันก็จับความสัมพันธ์ temporal ได้ด้วย spatial และ temporal ถูกดึงออกมาพร้อมกัน
- **[01:34:05] "เขาเอา LSTM เป็นพื้นฐาน… นี่คือจุดเน้น"** (LSTM เป็นฐาน นี่คือประเด็นหลัก) เปลี่ยนแค่การคูณกับ x/h เป็น convolution; ∘ ยังเป็น element-wise
- **[01:36:20]–[01:36:57] "อันนี้คือจุดสําคัญมาก… สิ่งที่ต้องระวัง"** (อันนี้สำคัญมาก… ต้องระวัง) transform และ augmentation ทำ **ทีละเฟรมแยกกัน** transform แบบสุ่ม (เช่น random erase) จะ **ต่างกันในแต่ละเฟรม** ไม่ได้เหมือนกันทั้งวิดีโอ
- **[01:37:17]–[01:37:22]** การโหลดและ transform ทำบน **CPU** พอแปลงเป็น tensor แล้ว ข้อมูลจะไปที่ **GPU**
- **[01:39:36]–[01:39:59]** PyTorch ไม่มี ConvLSTM สำเร็จรูป อาจารย์เลยใช้ implementation จาก GitHub ตอนนี้อาจมีในตัวแล้ว แต่อาจารย์ยังไม่ได้ดู
- **[01:40:27]–[01:42:22] 2D vs. 3D convolution** 2D เลื่อน x และ y และใช้ทุก channel 3D เลื่อน x, y **และ z** และ filter ตื้นกว่า input ได้
- **[01:43:43]–[01:44:12]** 3D conv ได้ feature spatial จากแต่ละตำแหน่งที่วาง และได้ feature temporal เพราะครอบมากกว่าหนึ่งเฟรม เลยเป็นตัวดึง **spatio-temporal feature**
- **[01:44:25]–[01:45:21]** 3D conv มักมาคู่กับ **3D max pooling** ซึ่งลดทุกมิติลงครึ่งด้วย stride 2
- **[01:46:04]–[01:46:23]** มี operator พวกนี้แล้ว อย่างน้อยก็ทำ video classification และ video regression ได้ วิธีทำ: S แล้วค่อย T หรือ ST พร้อมกัน
- **[01:48:09]–[01:50:14]** **I3D** คือ CNN ที่ pretrain มาเพื่อดึง feature วิดีโอ stream หนึ่งรับวิดีโอ RGB อีก stream รับ **optical flow** ("เฟรมลบเฟรม") ต่อด้วย linear layer, LSTM, RNN หรือ GRU ได้
- **[01:51:39]–[01:52:16] สรุปท้ายคาบ** สิ่งที่ใหม่ในวิดีโอเมื่อเทียบกับภาพคือ **ข้อมูล temporal** มี 3 กลยุทธ์: (a) spatial และ temporal **ขนานกัน** แล้วค่อย fuse (two-stream) (b) **ตามลำดับ** S→T (c) **พร้อมกัน** ST แบบไหนดีที่สุดขึ้นกับงาน, dataset และสิ่งที่จะทำนาย เฟรมใหญ่และวิดีโอยาวทำให้มีปัญหาเรื่องทรัพยากร

---

## ข้อสอบที่น่าจะออก

1. **feature spatial กับ temporal ในวิดีโอต่างกันยังไง และโมเดลไหนดึงแต่ละอย่าง?**
   Spatial feature คือเนื้อหาในเฟรมเดียว (ขอบ, สี, รูปร่าง, texture, ตำแหน่ง) ดึงด้วย 2D CNN ส่วน temporal feature คือการเปลี่ยนแปลงหรือความสัมพันธ์ระหว่างเฟรม (การเคลื่อนไหว, ลำดับ, ความเร็ว) ดึงด้วย RNN/LSTM

2. **นิยาม "spatio-temporal" CNN ทีละเฟรมแล้วตามด้วย LSTM เป็น spatio-temporal ไหม?**
   Spatio-temporal คือดึง feature spatial และ temporal **พร้อมกัน** CNN แล้วตามด้วย LSTM **ไม่ใช่**: มันคือ spatial แล้วค่อย temporal (S→T)

3. **ConvLSTM เป็น S→T หรือ ST? อธิบาย** (ข้อนี้คือ quiz ในห้อง)
   เป็น **ST** convolution ของมันวิ่งบน xₜ (เฟรมปัจจุบัน) กับ hₜ₋₁ (ข้อมูลอดีต) ที่ต่อกันอยู่ kernel 2D จับโครงสร้าง spatial และเพราะ channel มาจากคนละเวลา มันเลยจับความสัมพันธ์ temporal ในการคำนวณเดียวกันด้วย

4. **บอกข้อจำกัด 3 ข้อของ NN/CNN ที่ทำให้เกิด RNN**
   NN/CNN ต้องการ (a) input ทั้งหมดทีเดียว (b) input ที่ไม่มีความสัมพันธ์ตามลำดับ และ (c) input ขนาดคงที่เท่ากัน RNN รับ input ทีละ step, จัดการลำดับได้, และรับความยาวต่างกันได้

5. **เขียนสมการ RNN และอธิบายแต่ละส่วน**
   Hₜ = tanh(XₜWₓ + Hₜ₋₁Wₕ + B) ดึงข้อมูลจาก input ปัจจุบันกับ state ก่อนหน้า; tanh ทำ normalise Yₜ = softmax(HₜWᵧ + Bᵧ) คือส่วน forward/output ที่ใส่หรือไม่ใส่ก็ได้ ที่ step แรก H₀ = 0

6. **บอกข้อดีและข้อเสียของ RNN**
   ข้อดี: รับความยาวต่างกันได้, ขนาดโมเดลไม่โตตาม input, ใช้ข้อมูลอดีต, ใช้ weight ร่วมกันทุกเวลา ข้อเสีย: ช้า, ดึงข้อมูลจากอดีตไกลๆ ยาก (ข้อมูลแรกๆ จางหาย), ใช้ input ในอนาคตไม่ได้

7. **บอกส่วนประกอบของ LSTM และหน้าที่ของแต่ละส่วน**
   - 2 states: **cell state** (ความจำระยะยาว สะสมทุกอย่าง ไม่ถูก normalise) และ **hidden state** (ความจำระยะสั้น "keep for predict")
   - 3 sigmoid gates: **forget** (delete; เก็บ Cₜ₋₁ ไว้เท่าไหร่), **input** (write; ให้เนื้อหาใหม่เข้ามาเท่าไหร่), **output** (read; ให้ C ออกไปเป็น h เท่าไหร่)
   - 2 tanh modulates: **input modulate** zₜ (เนื้อหา candidate) และ "output modulate" tanh(Cₜ) ชื่อที่สองอาจารย์ตั้งเอง

8. **เขียนสมการ cell update และ hidden update ของ LSTM**
   Cₜ = fₜ∘Cₜ₋₁ + iₜ∘zₜ และ hₜ = oₜ∘tanh(Cₜ)

9. **ทำไม LSTM ใช้ sigmoid กับ gate และใช้ tanh ที่อื่น?**
   ช่วง 0…1 ของ sigmoid ใช้เป็น weight ได้ (0 กั้น, 1 ปล่อยผ่านหมด) ช่วง −1…1 ของ tanh ใช้ normalise ข้อมูล และค่าที่มีศูนย์กลางที่ 0 เรียนรู้ได้เร็วกว่า

10. **ทำไม cell state เก็บข้อมูลระยะยาวได้ แต่ hidden state ของ RNN เก็บไม่ได้?**
    Cell state ถูกอัปเดตแค่ด้วยการคูณ forget แบบ element-wise บวกกับการบวกเพิ่ม ไม่เคยถูกบีบด้วย tanh ข้อมูลเลยสะสมได้ ส่วน hidden state ของ RNN ถูกคำนวณใหม่ผ่าน tanh ทุก step ข้อมูลแรกๆ เลยจางหาย

11. **weight ของ gate ใน LSTM ต่างจาก weight ของ filter ใน CNN ยังไง?**
    Weight ของ CNN คงที่หลัง train เสร็จ ค่าของ gate ใน LSTM **adaptive**: คำนวณจาก xₜ และ hₜ₋₁ เลยเปลี่ยนไปตาม input ทุกครั้ง แม้ตอน inference ส่วน W และ b ที่ใช้คำนวณแต่ละ gate เป็นค่าที่ learn ได้ และแต่ละ gate มีของตัวเอง

12. **2D convolution ต่างจาก 3D convolution ยังไง? ทำไม 3D convolution เป็น spatio-temporal?**
    filter 2D เลื่อนตาม x และ y และครอบทุก channel filter 3D เลื่อนตาม z/เวลาด้วย และครอบหลายเฟรม output แต่ละค่าเลยผสมข้อมูล spatial กับข้อมูลหลายเฟรม (temporal)

13. **คำนวณขนาด output: input 4×4×4, kernel 3×3×3, stride 1, ไม่มี padding แล้ว 3D max pooling 2×2×2 stride 2 บน 4×4×4**
    convolution ได้ 2×2×2: output แต่ละค่าคือผลรวมของผลคูณ element-wise ทั้ง 3 แผ่น pooling ได้ 2×2×2

14. **บอก output shape ของ `nn.Conv3d(16, 33, 3, stride=2)` เมื่อ input เป็น (20, 16, 10, 50, 100)**
    ได้ (20, 33, 4, 24, 49) แต่ละมิติใช้ ⌊(I − 3)/2⌋ + 1

15. **บอกข้อดีข้อเสียของ 3D CNN และปัญหาใน action recognition**
    3D CNN แม่นกว่า 2D กับข้อมูล 3-D และได้ feature spatial กับ temporal พร้อมกัน แต่มี parameter เยอะ เสี่ยง overfitting สูง pretrained model น้อย และใช้ memory กับเวลามากกว่า dataset action recognition (UCF-101, HMDB-51) มีขนาดเล็ก และ 3D ConvNet train ยาก ซึ่งเป็นเหตุผลที่มี **Two-Stream** และ **I3D**

16. **I3D คืออะไร? 1×1×1 convolution ใน I3D ทำอะไร?**
    I3D พอง filter ของ Inception-v1 แบบ 2D (N×N) ให้เป็น T×N×N และรันสอง stream คือ RGB กับ optical flow แล้วเอา score มาเฉลี่ย 1×1×1 convolution ทำ linear combination ของ channel อย่างเดียว: ลด channel, ผสม channel, เพิ่ม non-linearity ด้วย ReLU/BN, และให้ความยืดหยุ่นใน branch ของ Inception

17. **สถานการณ์วิดีโอทางการแพทย์แต่ละแบบเหมาะกับ architecture ไหน?**
    - **ตรวจจับผู้สูงอายุล้มจาก CCTV** คือ event detection หรือ video classification สไลด์จับคู่ไว้กับ **CNN + RNN/LSTM**; ถ้าต้องการ feature แบบ ST ใช้ ConvLSTM หรือ 3D CNN แทนได้
    - **แยกขั้นตอนการผ่าตัดทีละเฟรม** คือ **temporal action segmentation** (หนึ่ง label ต่อเฟรม) ใช้โมเดล temporal ที่มี output ทุก time step
    - **หาเฉพาะช่วงเวลาของเหตุการณ์ผ่าตัดที่สนใจ** คือ **temporal action localization**
    - **ประมาณ heart rate จากวิดีโอผิวหนัง** คือ **video regression** เช่น feature ต่อเฟรมเข้า LSTM แล้ว output เป็น linear
    - **ตรวจว่าทำกายภาพบำบัดที่บ้านถูกไหม** คือ **video classification** (หนึ่ง label ต่อวิดีโอ เอา output ตัวสุดท้ายของ LSTM)
    - **dataset วิดีโอทางการแพทย์ที่เล็ก**: อย่า train 3D CNN ใหญ่ๆ ตั้งแต่ศูนย์ เพราะเสี่ยง overfitting ควรใช้ CNN+LSTM ที่ pretrained หรือ I3D ที่ pretrained เป็น feature extractor

18. **input shape ของ ConvLSTM classifier เป็นยังไง และทำไมต้อง `unsqueeze(0)` ตอน inference?**
    input เป็น (B, T, C, H, W) เช่น (4, 30, 3, 112, 112) วิดีโอเดียว stack แล้วได้ (T, C, H, W) และ `unsqueeze(0)` เพิ่มมิติ batch ให้เป็น (1, T, C, H, W)

---

## กับดักที่เจอบ่อย

- **S→T ไม่ใช่ ST** CNN แล้วตามด้วย LSTM *ไม่ใช่* spatio-temporal ส่วน ConvLSTM กับ 3D CNN *ใช่*
- **ช่วงของ sigmoid vs. tanh** Sigmoid คือ 0…1 (gate/weight); tanh คือ −1…1 (normalise ข้อมูล) อย่าสลับกัน
- **"Output modulate gate" ไม่ใช่ศัพท์ทางการ** เป็นชื่อที่อาจารย์ตั้งเองให้ tanh(Cₜ) ส่วนที่เป็นทางการคือ forget, input, output gate กับ input modulate (candidate) gate
- **Forget gate = 1 แปลว่าเก็บทุกอย่าง** ไม่ใช่ลืมทุกอย่าง 0 คือลบ
- **Hidden state ≠ prediction** ยังต้องผ่านส่วน forward (Linear/softmax) ที่ใส่หรือไม่ใส่ก็ได้ ถึงจะเป็น output
- **Cell state เก็บข้อมูล "มากกว่า" ไม่ใช่ "ครบ"** ข้อมูลหายที่ forget gate และที่ input gate
- **ข้อดี "รับความยาวต่างกันได้" ของ RNN** แปลว่า *ทำได้* ไม่ได้แปลว่าทำได้ดี ลำดับยาวๆ ยังทำข้อมูลแรกๆ หายอยู่
- **RNN cell มีตัวเดียว** ใช้ซ้ำทุกเวลา ภาพที่ unfold ไม่ได้แปลว่ามีหลาย cell ที่ weight ต่างกัน; weight ใช้ร่วมกัน
- **ค่าของ gate ขึ้นกับ input (adaptive)** ส่วน W และ b ที่ใช้คำนวณมันคงที่หลัง train อย่าพูดว่า "weight ของ gate เป็นค่าคงที่ที่ learn มา"
- **ConvLSTM** เปลี่ยน **การคูณ matrix x·W และ h·W** เป็น convolution การคูณ f∘C, i∘g และ o∘tanh(C) ยังเป็น **element-wise**
- **ลำดับแกนไม่เหมือนกัน** ConvLSTM ที่ใช้ `batch_first` เป็น (B, T, C, H, W) `nn.Conv3d` และสไลด์ I3D เป็น (B, C, T, H, W) ข้อความในสไลด์ 59 บอก (Batch, Frame, Width, Height, Channel) ต้อง permute ก่อนเข้า Conv3d
- **filter ของ 3D conv ไม่เหมือน 2D conv ที่ครอบทุก channel** ใน 2D ความลึกของ filter เท่ากับจำนวน channel ของ input และไม่เลื่อนตามความลึก ใน 3D filter เลื่อนตามความลึก/เวลาด้วย
- **ขนาด output ของ 3D conv ใช้สูตรเดียวกันทุกมิติ** รวมถึงเวลา: 4×4×4 กับ kernel 3×3×3 ได้ 2×2×2 ไม่ใช่ 2×2×4
- **Transform ทีละเฟรม** ทำให้ augmentation แบบสุ่มต่างกันในแต่ละเฟรมของวิดีโอเดียวกัน อาจารย์บอกให้ระวัง
- **3D CNN ไม่ได้ดีกว่าเสมอ** มี parameter มากกว่า overfit กับ dataset เล็ก (UCF-101, HMDB-51) และมี pretrained model น้อย
- **1×1×1 convolution** ไม่ได้ดูทั้งพื้นที่และเวลา แค่ผสม channel
- **สไลด์มีจุดผิด** ข้อความ "Input Gate (Write)" ในสไลด์ 40 ซ้ำกับนิยามของ hidden state; ให้ใช้นิยามในสไลด์ 38 (คุมว่าข้อมูลใหม่ไหลเข้า cell state เท่าไหร่) ตัวอย่าง 3D conv มีจุดคิดเลขผิด: สไลด์ขั้นตอนแสดง −21 แต่ค่าที่ถูกคือ −20, −37 ในตารางสุดท้ายควรเป็น −36 และ −22 ควรเป็น −18 ดูในส่วนสูตร
- **GRU** แค่ถูกพูดถึงผ่านๆ ใน lecture และไม่มีในสไลด์ อย่าคาดหวังสมการ GRU ละเอียดจากเนื้อหาบทนี้
