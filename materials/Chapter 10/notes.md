# บทที่ 10 — ลงมือเขียน Convolutional Neural Networks จริง (PyTorch)

บทนี้พาไล่ workflow ของ CNN ใน PyTorch ครบทั้งชุด (model class, data, training loop, การ save, inference, evaluation) สำหรับงานภาพแบบ multiclass, multilabel และ regression เริ่มจากโค้ดแบบเก่าสไตล์ MNIST ก่อน แล้วค่อยเป็นเวอร์ชันใหม่ที่ใช้ Dataset, DataLoader, transforms.v2, ResNet-50 แบบ pretrained และ `state_dict`
ผู้สอน: ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์ ("อาจารย์แบงค์") Notebook ของวิชา: `classification_train.ipynb` (เทรน), `classification_inference.ipynb` (ทีละภาพ), `classification_evaluation.ipynb` (หลายภาพพร้อมกัน) Dataset ที่ใช้: multiclass, multi-label และ image-regression

---

## แนวคิดหลัก

### 0. Framework ต่าง ๆ (สไลด์ช่วงแนะนำ และ P1 00:00–00:06)
- **PyTorch** (Facebook): วิชานี้เลือกใช้ตัวนี้ เพราะปรับแต่งได้ทุกอย่าง ทั้ง training loop, inference และ architecture งานวิจัยนิยมใช้ ("ประมาณ 1,000 กว่า Paper" ใช้ PyTorch เทียบกับ "400 กว่า" ที่ใช้ TensorFlow ช่วง 2021–2025)
- **TensorFlow / Keras** (Google) Keras เป็น *high-level API* ที่ไปเรียก TF/PyTorch อยู่ข้างใต้ อ่านง่าย แต่แก้ไขข้างในได้ไม่ค่อยได้
- **OpenVINO** เป็น toolkit สำหรับ *deploy* โมเดลที่เทรนเสร็จแล้วลงบนฮาร์ดแวร์สเปกจำกัด เป็นตัวเดียวในกลุ่มนี้ที่ **ไม่ได้** ออกแบบมาไว้เทรน
- **JAX** เหมาะกับงานคำนวณขนาดใหญ่
- **Inference** คือเอาโมเดลที่เสร็จแล้วมาใช้งาน
- สไลด์ p.5–6 (ตารางเปรียบเทียบที่เป็นรูปอย่างเดียว):
  - **Keras 3** เป็น high-level API แบบ multi-backend (โดย François Chollet, 2015) รันบน backend TensorFlow, JAX หรือ PyTorch ได้ และมี OpenVINO สำหรับ inference อย่างเดียว Keras ไม่ได้ห่อ TF/PyTorch ไว้ และสองตัวนั้นก็ไม่ต้องพึ่ง Keras
  - **TensorFlow** เป็นแพลตฟอร์ม ML แบบครบวงจร (Google Brain, 2015) ใช้ได้ทั้งเทรน, inference และ production
  - **JAX** เป็นการคำนวณแบบ array (Google): สไตล์ NumPy, autodiff, JIT, CPU/GPU/TPU
  - **OpenVINO** (Intel) เป็น toolkit สำหรับ inference อย่างเดียว ใช้ optimise และ deploy โมเดล
  - **PyTorch** เน้น Python เป็นหลัก เริ่มที่ FAIR (ตอนนี้อยู่กับ PyTorch Foundation) ทำได้ทั้งเทรน + inference, training loop ยืดหยุ่น เหมาะทั้งงานวิจัย + production
- สไลด์ p.7 (กราฟความนิยม):
  - Paper บน arXiv ปี 2021–2025: PyTorch **1,163**, TensorFlow 428, JAX 226, Keras 103, OpenVINO 23
  - GitHub stars: TensorFlow **199.7K**, PyTorch 102.9K, Keras 64.3K, JAX 36.3K, OpenVINO 10.8K
  - ข้อควรระวังในสไลด์: จำนวน star สะท้อนความสนใจที่สะสมมา ไม่ได้บอกจำนวนคนใช้ตอนนี้หรือประสิทธิภาพ

### 1. ไฟล์ในโปรเจกต์ CNN (สไลด์ p.16, P1 00:08)
| ไฟล์ | หน้าที่ |
|---|---|
| `Net.py` | กำหนด architecture (ตัว class) ไม่ต้องมีถ้าโหลด architecture ที่มีอยู่แล้ว เช่น ResNet |
| `TrainingCNN.py` | import Net มาเทรน ได้ไฟล์ `model.pt` ออกมา |
| `TestingCNN.py` | import Net / โหลด `model.pt` มาทดสอบ |
| `model.pt` / `.pth` | โมเดลที่เทรนแล้ว |

- Training Phase ใช้ภาพ + label เพื่อสร้างโมเดล Testing Phase ส่งภาพ test เข้าโมเดลเพื่อเอาคำตอบ
- Forward propagation = จำแนกภาพ (score แล้ว softmax ได้ probability แล้วได้ class) Backward propagation = ใช้ loss ไปอัปเดต weight (loss แล้ว optimizer)
- ตัวอย่างในสไลด์: score [1.16, −0.81, −3.44, 3.91] ผ่าน softmax ได้ probability [0.0596, 0.0083, 0.0006, 0.9315]

### 2. การสร้างโมเดล (`class Net(Module)`) (สไลด์ p.18–33, P1 00:11–00:42)
```python
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(            # ดึง feature
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4), ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4), ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        self.linear_layers = Sequential(Linear(4 * 7 * 7, 10))   # จำแนก class
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)      # flatten
        x = self.linear_layers(x)
        return x                       # score ดิบ (ไม่มี softmax)
```
- class ต้อง **inherit จาก `nn.Module`** อาจารย์บอกด้วยว่าชื่อ class ควรตรงกับชื่อไฟล์ (`Net` ใน `net.py`) ไม่งั้นอาจ import ไม่ได้
- method ที่ต้องมี:
  - `__init__` (constructor) **สร้าง layer ให้เป็น attribute** ("หน้าที่ของ Init คือการสร้าง Layer ให้เป็น Attribute") `super().__init__()` คือเรียก constructor ของคลาสแม่ ต้องใส่ทุกครั้ง
  - `forward(self, x)` **กำหนดการไหลของข้อมูลในโมเดล** `x` คือ input จากผู้ใช้
- `self.xxx` ที่ไม่มีวงเล็บคือ attribute ส่วน `self.xxx()` ที่มีวงเล็บคือ method
- เก็บ layer ได้ 2 แบบ คือ 1 attribute ต่อ 1 layer หรือเก็บเป็นกลุ่ม (block) ใน `nn.Sequential` ได้ทั้งคู่ Sequential จะรัน layer ตามลำดับจากบนลงล่าง ข้อเสียคือแตกกิ่ง (branch) ไม่ได้
- สไลด์ p.18 เทียบ `TinyConv` (ใช้ Sequential, ยกนิ้วโป้ง) กับ `TinyConvLessOrganized` (1 attribute ต่อ layer, คว่ำนิ้ว) ไม่บังคับต้องใช้ Sequential แต่ *แนะนำ* เพราะโค้ดเป็นระเบียบกว่าและจัด layer เป็น block ได้
- การ register parameter: ค่า `nn.Parameter` (weight และ bias) ที่สร้างใน `__init__` จะถูก register ใน `self.parameters()` ให้อัตโนมัติ ตัวอย่าง: `Linear(3,2)` มี weight ขนาด 2×3 และ bias ขนาด 2 (`requires_grad=True`)
- **Channel ต้องตรงกัน**: `out_channels` ของ conv ตัวหนึ่งต้องเท่ากับ `in_channels` ของตัวถัดไป `BatchNorm2d(n)` ใช้ n = จำนวน channel ที่เข้ามา แต่ละ Linear ค่า in = ค่า out ของ layer ก่อนหน้า
- `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, …, padding_mode='zeros')` `in_channels`=1 คือภาพ grayscale และ 3 คือภาพสี `kernel_size=3` คือ filter 3×3 ถ้าใส่เป็น tuple จะได้ filter ที่ไม่ใช่สี่เหลี่ยมจัตุรัส padding ค่าเริ่มต้นเป็น zero-padding เปลี่ยนได้ด้วย `padding_mode`
- `ReLU(inplace=True)` เขียนผลทับลงบน tensor input เลยเพื่อประหยัด memory ค่าเริ่มต้นคือ `False` ซึ่งจะสร้าง tensor ใหม่
- `MaxPool2d(kernel_size=2, stride=2)` ลด H และ W ลงครึ่งหนึ่ง เป็นค่าที่ตั้งกันเป็นปกติ
- `Linear(in_features, out_features)` คือ fully connected layer `4*7*7` = 4 channel × map ขนาด 7×7 หลังผ่าน pooling 2 ครั้งจาก input 28×28 (28→14→7) **output ของ Linear ตัวสุดท้ายต้องเท่ากับจำนวน class** ("สิ่งที่เราซีเรียสจริง ๆ")
- `Dropout(p=0.5)`: p คือความน่าจะเป็นที่ค่าจะถูกทำให้เป็นศูนย์ ค่าเริ่มต้นคือ 0.5 ช่วงที่แนะนำคือ 0.2–0.5 ทำงาน **เฉพาะตอนเทรน** เท่านั้น
- Flatten ทำได้ 3 แบบ: `x.view(x.size(0), -1)`, `out.reshape(out.size(0), -1)` หรือ `torch.flatten(x, 1)` / `nn.Flatten`
- `forward` คืนค่าเป็น **score function** ส่วน softmax หรือ sigmoid ไปใส่ **นอก** โมเดล
- Output แบบ multi-task (P1 00:30): แตก head 2 อันออกจาก feature map เดียวกันใน `forward` แล้ว return y1 กับ y2 พร้อมกัน
- AlexNet (torchvision, สไลด์ p.32): `Conv2d(3,64,…)` แปลว่า input เป็นภาพสี มี `num_classes=1000` และ `dropout=0.5` เป็นค่า default ของ constructor `AdaptiveAvgPool2d((6,6))` บังคับแต่ละ channel ให้เป็น 6×6 โดยไม่เปลี่ยน batch หรือ channel เช่น (1,3,10,14) → (1,3,6,6) Global Average Pooling คือ `AdaptiveAvgPool2d((1,1))` ได้ 1 ค่าต่อ channel pooling พวกนี้ทำให้ภาพหลายขนาดเข้าถึง FC layer ได้
- อยากดูภาพโมเดล ใช้ https://netron.app/

### 3. สคริปต์เทรนแบบเก่า: 7 ขั้น (สไลด์ p.35)
1. **Data Loader**: โหลดภาพ + label
2. **Training & Validating Set Generation**: แบ่งเป็น train set กับ validation set
3. **Model Loader**: โหลด architecture
4. **Defining Learning Algorithm**: optimizer, loss, GPU
5. **Training Model**
6. **Evaluating Performance**: loss / accuracy
7. **Is it accepted?** ถ้าใช่ save แล้วจบ ถ้าไม่ใช่ ปรับแก้

Flowchart ในสไลด์ p.33–56 (เป็นรูปอย่างเดียว):
- Start → Data Loader (ภาพ + label)
- จากนั้นมี 3 block ส่งเข้า **Training Model**: Defining Learning Algo, Training & Validating Set Generation และ Model Loader (จาก `Net.py`)
- Training Model → Evaluating Performance → "Is it accepted?"
  - **No:** "Feedback error to update weight model" แล้ววนกลับไปที่ Training Model
  - **Yes:** End
- โครงสร้างโฟลเดอร์ในสไลด์ p.33: `train/` (1.png, 2.png, … + `train.csv`), `test/`, `out.csv`, `TrainingCNN.py`, `TestingCNN.py`
- `train.csv` มี 2 คอลัมน์ `id` เก็บเลขภาพแบบไม่มี path และนามสกุล (1, 2, 3 …) `label` เก็บเลข class (9, 0, 0, 3 …)

อาจารย์สรุปง่าย ๆ (P1 00:43): การเทรนต้องมี **input 3 อย่าง**: (1) โมเดล, (2) ข้อมูลเทรน, (3) วิธีเทรนที่กำหนดไว้ เลยทำให้โค้ดเทรนยาว

**การโหลดข้อมูล (แบบเก่า):** อ่าน `train.csv` (คอลัมน์ `id` = ชื่อภาพ, `label` = **เลข** class ไม่ใช่ชื่อ) แล้ว `imread(path, as_gray=True)`, `img /= 255.0` (ช่วง 0–1), `astype('float32')` เก็บใส่ list แล้วแปลงเป็น `np.array` ได้ `(60000, 28, 28)`
**การแบ่งข้อมูล:** `train_test_split(train_x, train_y, test_size=0.1)` ได้ 54000 / 6000 จากนั้น `reshape(54000, 1, 28, 28)` ให้เป็น N×C×H×W แล้ว `torch.from_numpy(...).to(torch.float32)`
**Model loader:** `from Net import Net` แล้ว `model = Net()` การโหลดโมเดลมี 2 ขั้น: import แล้วสร้าง object

### 4. กำหนด learning algorithm + เช็ก GPU (สไลด์ p.42)
```python
model = Net()
optimizer = SGD(model.parameters(), lr=0.07, momentum=0.9)   # Adam(model.parameters(), lr=0.07) commented out
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
```
- `model.parameters()` ส่ง weight ให้ optimizer ทำให้โมเดลกับ optimizer "รู้จักกัน"
- วิธีใช้ `torch.optim`: สร้าง object ของ optimizer, ตั้ง hyperparameter (lr, …) แล้วอัปเดตด้วย gradient
- **Normalization** (สไลด์ p.45): x′ = (x − μ)/σ ได้ mean 0 และ SD 1 ถ้า feature มีช่วงค่าต่างกันมาก (0–1 กับ 0–255) gradient descent จะ converge ช้ากว่า
- **Loss function** (สไลด์ p.47): ตัวเลขตัวเดียวที่บอกว่าโมเดลผิดมากแค่ไหน ยิ่งต่ำยิ่งดี รับค่า probability จาก softmax
  - ตอบถูก + มั่นใจ ได้ loss ต่ำ
  - ตอบถูก + ไม่มั่นใจ ได้ loss สูง
  - ตอบผิด + มั่นใจ ได้ loss สูง
  - Loss สำหรับ classification: Hinge, Cross Entropy การ optimise คือการทำให้ loss ต่ำที่สุด
- วิธีเขียน device แบบใหม่ (handout):
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device); inputs, labels = inputs.to(device), labels.to(device)
```

### 5. Training loop: หัวใจของข้อสอบ
เวอร์ชันเก่า (สไลด์ p.50–56) เป็นแบบ full-batch มีแค่ loop ของ epoch:
```python
for epoch in tqdm(range(n_epochs)):          # n_epochs = 25
    model.train()
    optimizer.zero_grad()                    # ล้าง gradient ที่สะสมไว้
    output_train = model(x_train)            # ทำนาย (forward pass)
    loss_train = criterion(output_train, y_train)   # loss
    loss_train.backward()                    # คำนวณ gradient
    optimizer.step()                         # อัปเดต weight
    tr_loss = loss_train.item()
```
เวอร์ชันใหม่ (สไลด์ p.86, handout) มี **for-loop 2 ชั้น**: ชั้นนอกวนตาม epoch ชั้นในวนตาม batch
```python
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:                 # print ทุก ๆ N batch เท่านั้น
            print(...); running_loss = 0.0
```
แต่ละคำสั่งทำอะไร:
| คำสั่ง | ทำอะไร |
|---|---|
| `model.train()` | ตั้ง **state** เป็นโหมดเทรน: เปิด Dropout, BatchNorm ใช้ mean/var ของ batch และอัปเดต running stats ตัวมัน *ไม่ได้* เทรนอะไรเอง |
| `optimizer.zero_grad()` | รีเซ็ต gradient เป็น 0 PyTorch **สะสม** gradient ไว้ ถ้าไม่ทำ gradient ของรอบก่อนจะถูกบวกเข้ามาด้วย |
| `model(x)` | **ทำนาย (forward pass)** ไม่ใช่ "บรรทัดที่เทรน" ที่ต้องมีเพราะ loss ต้องใช้ผลทำนาย |
| `criterion(output, y)` | คำนวณ loss (ในที่นี้คือ CrossEntropy) target ต้องเป็น `long` (int64) |
| `loss.backward()` | **คำนวณ gradient** ของ loss เทียบกับทุก parameter ที่ `requires_grad=True` |
| `optimizer.step()` | **อัปเดต weight** ด้วย gradient พวกนั้น: w_new = w_old − α·∂loss/∂w |
| `loss.item()` | ดึงค่า loss ออกมาเป็นตัวเลข Python เอาไว้ log |

ลำดับคือ **predict แล้ว criterion (loss) แล้ว backward (gradient) แล้ว step (update)** ถ้าโดนถามว่า "คำสั่งไหนใช้เทรนโมเดล" คำตอบคือ 4 ตัวนี้ [P1 01:12:00]
- ตำแหน่งที่วาง `step()` กำหนดว่าอัปเดต weight บ่อยแค่ไหน: อยู่ใน loop ของ batch คืออัปเดต **ทุก batch** (ที่ใช้กันปกติ) อยู่นอก loop คืออัปเดตทุก epoch และ batch_size=1 คืออัปเดตทุกภาพ (ไม่แนะนำ ดูรายการที่อาจารย์เน้น)
- คำสั่งเสริม: `Variable(...)` ต้องใช้แค่ใน PyTorch < 0.4 ตอนนี้ tensor เก็บทั้งข้อมูลและ gradient ได้เลย `squeeze()` ตัดมิติที่มีขนาด 1 ออก เช่น (10,1,28,28) → (10,28,28); `squeeze_()` คือเวอร์ชัน in-place (มี `_` ต่อท้าย) `.long()` เหมือนกับ `.to(torch.int64)`; `.to()` ยืดหยุ่นกว่า (เปลี่ยน dtype หรือ device ได้หมด)
- **Epoch** = วนครบทั้ง dataset 1 รอบ **Batch** = ข้อมูล 1 ก้อนย่อย ตัวอย่าง: 100 ภาพ batch size 20 จะได้ 5 batch ต่อ epoch

### 6. การ save โมเดล (checkpoint)
- แบบเก่า: `torch.save(model, 'C:/model04.pt')` save **architecture + weight** โหลดกลับด้วย `model = torch.load(...)`
- แบบใหม่: `torch.save(model.state_dict(), 'model.pt')` save **weight อย่างเดียว** ไฟล์เลยเล็กกว่า ตอนโหลดต้องสร้าง architecture เดิมขึ้นมาใหม่ (แก้ `fc` แบบเดียวกัน) แล้ว `model.load_state_dict(torch.load('./model.pt'))` จะขึ้นว่า `<All keys matched successfully>`
- นามสกุลไฟล์คือ `.pt` หรือ `.pth`
- Checkpoint 3 ตัวที่ควรเก็บจากการรัน 1 ครั้ง [P2 00:10:35]: (1) epoch สุดท้าย, (2) จุดที่ early stopping, (3) ตัวที่ผล validation ดีที่สุด

### 7. Validation / testing / inference
```python
model.eval()                 # ปิด Dropout, BatchNorm ใช้ running mean/var ที่เก็บไว้
with torch.no_grad():        # ไม่เก็บ gradient: เร็วขึ้น, ใช้ memory น้อยลง, ใช้ batch ใหญ่ขึ้นได้
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
```
- แบบเก่า: `softmax = torch.exp(output).cpu()`, `np.argmax(prob, axis=1)`, `accuracy_score(val_y, predictions)` แล้วเขียนผลทำนายลง `out.csv`
- Inference ทีละภาพ (handout): เปิดภาพด้วย PIL แล้ว `.convert('RGB')` ใช้ **transform สำหรับ standardisation ตัวเดียวกัน** `image.unsqueeze(0)` (เพิ่มมิติ batch), `model.eval()`, `outputs.squeeze(0)`, `torch.softmax(outputs, 0)`, `torch.max(score, 0)` ผลที่ได้: "Predicted: kiwi 99.89%"
- Inference ของ classifier เบา รันบน CPU ได้ ตรงนั้นเลยไม่ต้องบังคับใช้ CUDA [P1 01:13:58]
- สคริปต์ test = โหลดโมเดล + โหลดข้อมูล + รัน ไม่มีส่วน learning algorithm หรือส่วนเทรน
- ขยายความ (PyTorch มาตรฐาน): weight เปลี่ยนเฉพาะตอน `optimizer.step()` `no_grad()` หยุดเก็บ gradient (ประหยัด memory/เร็วขึ้น) ส่วน `eval()` สลับพฤติกรรมของ Dropout/BN เลยต้องใช้ทั้งคู่ถึงจะ evaluate ได้ถูกต้องและมีประสิทธิภาพ

### 8. Pipeline แบบใหม่: transforms, Dataset, DataLoader (สไลด์ p.79–112, P2)
**ขั้นพื้นฐานของ classification (สไลด์ p.79):** (1) dataset, (2) CNN, (3) loss + optimizer, (4) เทรน, (5) test/evaluate

**Transforms** มี 2 หน้าที่ [P2 00:01]: **(1) data standardisation / pre-processing** (แปลงเป็น tensor, dtype, resize/crop, normalise) และ **(2) data augmentation** ต่อกันเป็นสายด้วย `Compose`:
```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToImage(),                          # แปลงเป็นรูปแบบ image/tensor
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])   # ได้ช่วง -1..1
```
ตัวอย่าง augmentation (`torchvision.transforms.v2`, สไลด์ p.107):
```python
preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),         # float ในช่วง [0,1]
    v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])  # ImageNet
```
- transform อื่น ๆ ใน v2:
  - เรขาคณิต: Resize, RandomResize, RandomCrop, RandomHorizontalFlip, RandomZoomOut, RandomRotation, Pad, CenterCrop
  - สี: Grayscale, RGB, GaussianBlur, RandomAdjustSharpness
  - การประกอบ: RandomApply
  - การแปลง: PILToTensor, ToDtype, ToPILImage `ToTensor()` **เลิกใช้แล้ว (deprecated)** ตอนนี้เท่ากับ `Compose([ToImage(), ToDtype(float32, scale=True)])`
  - Auto-augmentation: AugMix, CutMix
- ภาพที่ normalise ด้วย mean/SD ของ ImageNet พอเอามาแสดงจะดูดำไปบางส่วน เพราะค่าติดลบ ข้อมูลไม่ได้เสีย แค่ช่วงค่าเปลี่ยน [P2 00:03:27]

**Dataset กับ DataLoader ต่างกันยังไง (สไลด์ p.100):**
- `torch.utils.data.Dataset` เก็บตัวอย่าง + label (เป็นแหล่งข้อมูล)
- `torch.utils.data.DataLoader` ป้อนข้อมูลจาก Dataset ทีละ **batch** ให้ตอนเทรน/ตอน test และคืนค่าเป็น iterator
- Dataset สำเร็จรูป: `torchvision.datasets.CIFAR10/FashionMNIST/ImageNet(root=..., train=True/False, download=True, transform=...)` `train=True` โหลดชุด train ส่วน `False` โหลดชุด test `download=True` จะดาวน์โหลดก็ต่อเมื่อยังไม่มีข้อมูลในเครื่อง

**Custom Dataset** ต้อง inherit `Dataset` และเขียน **3 method**:
```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file); self.img_dir = img_dir
        self.transform = transform; self.target_transform = target_transform
    def __len__(self):  return len(self.img_labels)          # จำนวนตัวอย่าง
    def __getitem__(self, idx):                               # คืน (image, label) ของ index idx
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path); label = self.img_labels.iloc[idx, 1]
        if self.transform: image = self.transform(image)
        if self.target_transform: label = self.target_transform(label)
        return image, label
```
- `__init__` รันครั้งเดียวตอนสร้าง object (root path, transforms, ไฟล์ label) `transform` ทำกับ **ภาพ** ส่วน `target_transform` ทำกับ **label**
- handout ผลไม้ map ชื่อ class เป็น index ด้วย `class_to_idx` และโครงโฟลเดอร์มี 1 โฟลเดอร์ต่อ 1 class บวกไฟล์ `annotation.csv`

**Parameter ของ DataLoader:**
```python
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader  = DataLoader(testset,  batch_size=4, shuffle=False, num_workers=2)
```
- `batch_size` คือจำนวนตัวอย่างที่ส่งเข้าไปพร้อมกัน
- `shuffle=True` สลับลำดับใหม่ **ทุก epoch**
- `num_workers` คือจำนวน process ของ CPU ที่ใช้โหลดข้อมูล สไลด์บอกว่าค่าเริ่มต้นคือ 0
- การโหลดข้อมูล, transforms และ augmentation รันบน **CPU + RAM** ส่วนการเทรนรันบน **GPU** ถ้า CPU อ่อน การเทรนก็ช้าได้ ถึงจะมี GPU ดีก็ตาม

### 9. ใช้ backbone แบบ pretrained (สไลด์ p.96–98, P2 00:17–00:33)
```python
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)   # pretrained บน ImageNet
model.fc = nn.Linear(2048, len(classes))                   # เปลี่ยน output layer
```
- `resnet50()` แบบไม่ใส่อะไรเลยจะได้ architecture ที่มี **weight เริ่มต้นแบบสุ่ม** อาจารย์บอกว่าอย่าโหลดแบบนี้ถ้าต้องการ weight แบบ pretrained
- โมเดล pretrained ให้ output 1000 class ของ ImageNet เลยต้อง **เปลี่ยน layer สุดท้าย**:
  1. `print(model)` เพื่อหาชื่อ attribute ของ layer สุดท้าย (`fc` สำหรับ ResNet) และขนาด input (2048)
  2. ใส่ `nn.Linear(in, n_classes)` ตัวใหม่ลงไป
- weight ทุกตัวยกเว้น layer ใหม่ยังเป็นแบบ pretrained อยู่ การเทรนเลยไป optimise (fine-tune) layer นั้นเป็นหลัก
- โมเดลอื่นในสไลด์: googlenet, squeezenet1_0, densenet121 (`*_Weights.IMAGENET1K_V2`), VGG, EfficientNet, MobileNet
- เลือกโมเดลสำหรับ embedded / IoT / มือถือ: เลือกตัวที่ **parameter count** ต่ำและ **GFLOPS** ต่ำ สองค่านี้ไม่ได้ไปด้วยกันเสมอ
- สไลด์ p.92 (ภาพหน้าจอตาราง weight ของ torchvision):
  - คอลัมน์: Weight, **Acc@1, Acc@5, Params, GFLOPS**
  - ตัวอย่าง: AlexNet 56.5% / 61.1M / 0.71 GFLOPS. EfficientNet_B0 77.7% / 5.3M / 0.39. ConvNeXt_Large 84.4% / 197.8M / 34.36. DenseNet121 74.4% / 8.0M / 2.83. GoogLeNet 69.8% / 6.6M / 1.5.
  - เห็นว่า AlexNet มี parameter เยอะแต่ GFLOPS ต่ำ แปลว่าสองค่านี้ไม่ได้แปรผันตามกัน

### 10. Multiclass vs multilabel vs regression (สไลด์ p.110–123, P2 00:37–00:43)
| | Multiclass | Multilabel | Regression |
|---|---|---|---|
| ความหมาย | 1 ภาพ 1 class | 1 ภาพ **อาจ** มีมากกว่า 1 class (หรือไม่มีเลย) | ทำนายค่าต่อเนื่อง |
| จำนวน output node | เท่าจำนวน class | เท่าจำนวน class (เช่น `Linear(2048, 24)`) | เท่าจำนวนค่าที่จะทำนาย (แนะนำ **1 ค่าต่อ 1 โมเดล**: `Linear(2048, 1)`) |
| Activation สุดท้าย (ใส่นอกโมเดล) | Softmax | **Sigmoid** + threshold (สไลด์ใช้ `> 0.3`) | **Linear** (ไม่ใส่) ช่วงค่าจะได้ไม่เพี้ยน |
| Loss ในสไลด์ | `CrossEntropyLoss` | ในโค้ดใช้ `BCEWithLogitsLoss` (Sigmoid + BCE ในตัวเดียว) เชิงอรรถในแผนภาพบอกว่าใช้ BCE **หรือ** Cross Entropy ก็ได้ | `L1Loss` (MAE) ใช้ L2 (MSE) ก็ได้ |
| การอ่านผลตอน inference | `torch.max(softmax)` | `torch.nonzero(sigmoid(out) > 0.3)` | `outputs.squeeze().item()` เช่น "… years old" |

โค้ดเทรนของทั้ง 3 แบบแทบเหมือนกัน ("แทบจะเหมือนกัน 100%") ที่เปลี่ยนมีแค่ output layer, loss และการอ่านผลตอน inference

แผนภาพในสไลด์ p.110 และ p.114 (เป็นรูปอย่างเดียว), "การพิจารณาผลลัพธ์ของแบบจำลองและการคำนวณข้อผิดพลาด":
- **Multiclass:** score (−3.44, 1.66, 3.91) → Softmax → Bird 0.010, Cat 0.059, Dog 0.931 เทียบกับเฉลยแบบ one-hot (0, 0, 1) ด้วย Cross Entropy Loss
- **Multilabel:** score ชุดเดียวกัน → **sigmoid แยกกันทีละ node** → Drama 0.03, Sci-Fi 0.84, Action 0.98 → **threshold ที่ 0.5** → ทำนายได้ (0, 1, 1) เทียบกับเฉลย (0, 0, 1) ด้วย Cross Entropy Loss เชิงอรรถบอกว่า loss ใช้ Binary Cross Entropy หรือ Cross Entropy ก็ได้
- **Regression:** output node เดียวใช้ **Linear function** → 2,450,000 เทียบกับค่าจริง 2,500,000 → **L1 Loss** เชิงอรรถบอกว่าใช้ MAE/L1, MSE/L2 หรืออื่น ๆ ได้

### 11. ปรับภาพให้เข้ากับขนาด input ที่โมเดลกำหนด (P1 01:16–01:22)
โมเดล pretrained ต้องการขนาดตายตัว ปกติคือ **224×224 หรือ 256×256** มี 6 วิธี:
1. **Resize ตรง ๆ** สัดส่วนของวัตถุจะเพี้ยน class ที่ดูจากรูปร่าง (เช่น การอ่านตัวอักษร) จะสับสน
2. **Pad ให้เป็นสี่เหลี่ยมจัตุรัส (เติมศูนย์) แล้ว resize** สัดส่วนไม่เพี้ยน **pad ก่อน แล้วค่อย resize**
3. **Center crop แล้ว resize** สมมติว่าวัตถุอยู่ตรงกลาง เหมือนรูปถ่ายจากมือถือหรือกล้องทั่วไป
4. **Random crop แล้ว resize** **ใช้ตอนเทรนเท่านั้น** (เพิ่มความหลากหลายทุก epoch) ห้ามใช้ตอน inference เด็ดขาด ไม่งั้นรูปเดิมจะได้คำตอบต่างกัน
5. **ROI crop แล้ว resize** ใช้โมเดลหรืออัลกอริทึม (เช่น face detection) หาบริเวณที่สนใจก่อน
6. **Tiling / sliding window** + majority vote มี overhead

ในงานจริงอาจารย์ใช้ข้อ 1–3 เป็นหลัก

สรุปในสไลด์ p.67 (เป็นรูปอย่างเดียว ตัวอย่าง: ภาพ 16:10 เป็น 224×224×3; "ขนาดที่ต้องใช้จริงขึ้นกับ architecture และ pretrained weights"):

| # | ชื่อในสไลด์ | เหมาะกับ | ข้อควรระวัง |
|---|---|---|---|
| 1 | Direct Resize (Stretch) | ภาพที่สัดส่วนใกล้ ๆ กัน | วัตถุถูกยืดหรือบีบ เร็วและง่าย แต่เลี่ยงถ้ารูปร่างเป็น feature สำคัญ |
| 2 | Resize + Padding (Letterbox) | Detection, Segmentation, OCR | padding อาจกลายเป็น feature ที่โมเดลจำ ใช้เมื่อภาพมี aspect ratio ปนกันและต้องรักษารูปร่าง |
| 3 | Center Crop + Resize | สินค้า, ใบหน้า, วัตถุที่อยู่ตรงกลาง | วัตถุที่อยู่ริมขอบจะโดนตัด ไม่มี padding และไม่เพี้ยน |
| 4 | Random Resized Crop | data augmentation สำหรับ classification | อาจตัด feature สำคัญทิ้ง **สุ่มเฉพาะตอนเทรน ส่วน validation/test ใช้ transform แบบตายตัว** |
| 5 | ROI / Object-Centric Crop | **ภาพทางการแพทย์**, โรคใบพืช, วัตถุเล็กมาก | ต้องมี bounding box, mask หรือ ROI detector ลด background และทำให้วัตถุเล็กชัดขึ้น |
| 6 | Tiling / Sliding Window | **histology**, ภาพดาวเทียม, segmentation | ช้ากว่า และต้องรวมผลของแต่ละ patch ใช้ overlap เพื่อไม่ให้วัตถุโดนตัดที่ขอบ patch |

- กล่องในสไลด์เขียนว่า "Resize + Padding" (รักษาสัดส่วนไว้ แล้วเติมพื้นที่ว่างให้ได้ขนาดเป้าหมาย) อาจารย์พูดว่า "pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize" ทั้งสองแบบรักษา aspect ratio ไว้ สิ่งที่ห้ามคือ resize เฉย ๆ จนภาพถูกยืด
- สิ่งที่สไลด์เตือน:
  - ภาพปกติใช้ interpolation แบบ **Bilinear/Bicubic** ส่วน **mask ของ segmentation ใช้ Nearest-neighbour**
  - **ภาพกับ label (mask) ต้อง transform ด้วย parameter เดียวกัน**
  - แปลงสีและ normalise (ให้ตรงกับโมเดล pretrained) **หลัง** resize และ pad

**ภาพ grayscale กับโมเดล pretrained แบบ 3 channel:** ก๊อป channel เดียวให้เป็น 3 channel ดีกว่าไปแก้ conv ตัวแรก ซึ่งจะทำให้เสีย weight ที่ pretrained มา ไฟล์บางประเภทโหลดภาพ grayscale มาเป็น 3 channel ที่เหมือนกัน ถ้าต้องการ 1 channel ก็หยิบอันไหนก็ได้

**ข้อเตือนในสไลด์ p.73:** AlexNet ต้องการ 227×227×3 (3 plane) input ที่เป็น grayscale หรือ binary เลยต้องจัดการตามที่บอกข้างบน

### 12. เรื่องเสริม (สไลด์ p.74)
- **Regularization** ใช้แก้หรือบรรเทา over/underfitting มี 3 วิธีที่นิยม: Data Augmentation, Batch Normalization, Dropout
- **BatchNorm2d(num_features)** กันไม่ให้ feature ตัวเดียวครอบงำ ทำให้เทรนเร็วขึ้นและ loss ต่ำลง

---

## สูตรและตัวเลขที่ต้องจำ

| เรื่อง | ค่า (ที่มา) |
|---|---|
| Normalisation | x′ = (x − μ)/σ ได้ mean 0 และ SD 1 (สไลด์ p.45) |
| Gradient descent | w_new = w_old − α · ∂loss/∂w_old. α = learning rate gradient ชี้ขึ้นเนิน เลยต้องเดินไปทางตรงข้าม (p.53) |
| การ scale pixel (โค้ดเก่า) | `img /= 255.0` ได้ 0–1 เป็น float32 |
| Normalize(0.5, 0.5) ต่อ channel | แปลง [0,1] เป็น **[−1, 1]** (p.80) ถ้าจะแสดงภาพให้ un-normalise: `img/2 + 0.5` |
| mean / std ของ ImageNet (RGB) | **(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)** |
| ข้อมูลสไตล์ MNIST | 60000 ภาพ grayscale ขนาด 28×28 `test_size=0.1` ได้ **54000 train / 6000 val** shape ของ tensor **(N, 1, 28, 28)** = (Batch, Channel, Height, Width) test set 10000 |
| input ของ FC ใน Net | `4*7*7` (28 → 14 → 7 หลัง MaxPool(2,2) สองครั้ง), output 10 |
| ตัวอย่าง ConvNet | Conv(1,32,k5,p2), Conv(32,64,k5,p2), FC `7*7*64` → 1000 → 10 |
| โมเดล CIFAR | Conv(3,6,5), Pool(2,2), Conv(6,16,5), FC 16·5·5 → 120 → 84 → 10 เวอร์ชัน regression: 16·5·5 → 128 → 64 → 1 |
| AlexNet | input 227×227×3 AdaptiveAvgPool → 6×6 FC 256·6·6 → 4096 → 4096 → 1000 Dropout 0.5 |
| ResNet-50 | `fc` in_features = **2048** ผลไม้ = 10 class, multilabel = 24, regression = 1 weight IMAGENET1K_V2 ≈ 80.858% accuracy |
| Optimizer แบบเก่า | `SGD(lr=0.07, momentum=0.9)` (Adam lr=0.07 ถูก comment ไว้) ตัวอย่างในสไลด์ optim: SGD lr=0.01 m=0.9, Adam lr=0.0001 |
| Optimizer แบบใหม่ | `optim.SGD(model.parameters(), lr=0.001, momentum=0.9)` momentum "ปกติใช้ 0.9" |
| Epochs | แบบเก่า 25, CIFAR 10, handout ผลไม้ 4 งานจริง: **50–100 + early stopping** [P2 00:10:32] |
| Batch size | CIFAR / ผลไม้ **4**, ตัวอย่าง custom **64** อาจารย์: ใช้เลขยกกำลังของ 2 Classification 16/32/64, segmentation 4/8, video 2 |
| ความถี่ในการ print | ทุก 2000 mini-batch (CIFAR), ทุก 500 (ผลไม้) |
| Dropout | default p=0.5, แนะนำ 0.2–0.5 |
| ค่า default ของ BatchNorm2d | eps=1e-05, momentum=0.1, affine=True, track_running_stats=True |
| ค่า default ของ Conv2d | stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros' |
| MaxPool | kernel 2, stride 2 ลด H และ W ลงครึ่ง |
| ขนาด input ของโมเดล | 224×224 หรือ 256×256 handout ใช้ Resize (256,256) RandomResizedCrop 224 |
| ความน่าจะเป็นของการ flip | RandomHorizontalFlip p=0.5 |
| Threshold ของ multilabel | โค้ด inference (p.117): sigmoid > **0.3** แผนภาพ (p.110) และอาจารย์: **0.5** เป็นค่าที่ปรับเลือกเองได้ |
| num_workers ของ DataLoader | สไลด์: default **0** ตัวอย่าง CIFAR ใช้ 2 |
| Seed | `torch.manual_seed(0)` ใน handout ตอนเทรน |
| Linear(3,2) | weight 2×3 + bias 2 |

---

## สิ่งที่อาจารย์เน้น

### Data leakage กับการแบ่ง train/val (P1 00:49–00:51)
- **[P1 00:49:52–00:50:07]** เรื่องบรรทัด `train_test_split`: ถ้าเป็นโปรเจกต์จริง (โปรเจกต์ปี 4 ไม่ใช่วิชานี้) *อย่า* ใช้ `train_test_split` หรือ k-fold อัตโนมัติ "เราควรจะแบ่ง manual และให้มันโหลดมาเอง เพื่อกันการ data leak" (แบ่งเองแล้วโหลด split ที่ตายตัวมาใช้)
- **[P1 00:50:13–00:50:34]** ตอนสอบป้องกันโปรเจกต์ Data Science / AI กรรมการจะทักว่าการทดลองของเรา bias ต่อให้ตั้ง seed ไว้ ก็ไม่ได้รับประกันว่าทุกโมเดลจะได้ภาพชุดเดียวกัน ("ต่อให้ใส่ Seed บางครั้งมันก็มีความคาดเคลื่อน")
- **[P1 00:50:38–00:50:59]** นิยาม leak ของอาจารย์: "สมมุติภาพผมถูกใช้เป็นเทสต์ ณ Model A แต่ภาพผมอาจจะใช้เป็นเทรนด์ ณ Model B ซึ่งมันไม่แฟร์" ภาพ test ต้องเป็น test สำหรับ *ทุก* โมเดล และภาพ train ก็ต้องเป็น train สำหรับทุกโมเดล เลยต้องแบ่ง train/validate/test เองแล้วโหลดจากไฟล์

### เรื่องเช็ก GPU, "ถูกครึ่งเดียว" (P1 00:56–01:00)
หมายเหตุ: ที่อาจารย์แก้ว่า "ผิด / ถูกครึ่งเดียว" ในช่วงนี้เป็นเรื่อง `torch.cuda.is_available()` ไม่ใช่เรื่อง data leakage
- **[P1 00:56:04]** "บรรทัดนี้ กาดอกจันทร์ครับ" (ติดดาวบรรทัดนี้): `if torch.cuda.is_available(): model = model.cuda(); criterion = criterion.cuda()`
- **[P1 00:56:22–00:56:31]** อาจารย์ถามว่าบรรทัดนี้มีไว้ทำอะไร นักศึกษาคนหนึ่งตอบแล้วโดน "ผิด" อีกคนบอกว่าไว้เช็ก อาจารย์ตอบ "ใช่ แต่ประโยคที่เพื่อนพูดไม่ถูก" (ใช่ แต่พูดไม่ถูก)
- **[P1 00:56:44]** นักศึกษาตอบ "ตรวจสอบว่าเครื่องนั้นมีการ์ดจอหรือไม่" อาจารย์ตอบ **"อันนี้ถูกครึ่งเดียว"**
- **[P1 00:56:50–00:57:06]** อีกครึ่งคือ "มีแล้วมัน available ไหม" GPU อาจมีอยู่แต่คนอื่นใช้อยู่ **"is available มันเช็ค 2 stage"**: (1) มี GPU ไหม, (2) GPU ว่างไหม
- **[P1 00:57:14–00:58:45]** เรื่องเล่า: บน server กลางของคณะ GPU เต็มหมด check เลยคืน False แล้วโค้ดก็แอบไปรันบน CPU เงียบ ๆ งานที่ปกติใช้ 3 วันผ่านไปกว่าสัปดาห์ก็ยังไม่เสร็จ ต้อง debug อยู่สัปดาห์หนึ่งกว่าจะเจอสาเหตุ
- **[P1 00:58:45–00:59:56]** "ถ้าการ์ดจอเต็ม ควรรันไหม ไม่ควร" ในงานจริงให้ตัด `if` ออกแล้วบังคับ `model.cuda()` / `criterion.cuda()` ไปเลย ถ้า GPU ไม่ว่างมันจะ error ก็แค่ยังไม่ต้องรัน โค้ดในอินเทอร์เน็ตใส่ `if` ไว้เพราะเขาไม่รู้ว่าเราใช้ฮาร์ดแวร์อะไร

### Training loop
- **[P1 01:02:29–01:02:41]** ติดดาว `model.train()`: "ตายมานักต่อนักแล้ว…หลายๆคนทำงานเสร็จปุ๊บต้องรันใหม่เพราะบรรทัดนี้" (หลายคนต้องรันการทดลองใหม่ทั้งหมดเพราะบรรทัดนี้)
- **[P1 01:03:11–01:05:06]** "จดไปครับ": `model.train()` ประกาศว่า state ตอนนี้คือการเทรน "ทำไมต้องบอก นี่แหละคือคำถามสำคัญ" บาง layer ทำงานต่างกัน: Dropout ทำงานเฉพาะตอนเทรน ส่วน BatchNorm สะสม mean/SD ตอนเทรน แต่ใช้ค่าที่เก็บไว้ตอน eval train/eval เลยเป็นตัวเปิดปิด layer บางตัว
- **[P1 01:05:07–01:05:41]** bug ไม่ได้อยู่ตอนเทรน (ค่า default เป็น train mode อยู่แล้ว) แต่อยู่ที่ **inference โดยไม่มี `model.eval()`** ซึ่งให้ค่าเพี้ยน อีกกรณี: ทำ k-fold validation โดยไม่มี `eval()` fold แรก ๆ ผลจะแย่ fold หลัง ๆ ผลจะดี
- **[P1 01:06:37–01:06:46]** `zero_grad` ล้าง gradient ของรอบก่อน ("Set Gradient เป็น 0")
- **[P1 01:07:03–01:07:24]** "หลาย ๆ คนเข้าใจผิด" พอถามว่าบรรทัดไหนเทรนโมเดล นักศึกษาชี้ไปที่ `model(x_train)` อาจารย์บอก **"อันนี้ผิด"** บรรทัดนั้นคือ **การทำนาย (prediction)** ที่ต้องมีเพราะการเทรนคืออัปเดต weight ด้วย gradient, gradient มาจาก loss และ loss ต้องใช้ผลทำนาย
- **[P1 01:09:39–01:09:58]** "จดไปครับ": `loss_train.backward()` คำนวณ gradient จาก loss
- **[P1 01:10:25–01:10:39]** `optimizer.step()` อัปเดต weight "เวทถูกอัปเดตจริงๆที่คำสั่งไหน: optimizer.step"
- **[P1 01:11:34–01:12:05]** สรุป: model ทำนาย แล้ว criterion ให้ loss แล้ว backward ให้ gradient แล้ว step อัปเดต ถ้าถามว่าคำสั่งไหนใช้เทรน คำตอบคือ 4 ตัวนี้
- **[P2 00:12:13]** โค้ด CIFAR ไม่มี `model.train()` ซึ่งไม่เป็นไรเพราะ default คือ train อยู่แล้ว
- **[P2 00:12:44–00:13:19]** ตำแหน่งของโค้ดอัปเดตกำหนดความถี่ในการอัปเดต: อยู่ใน loop ของ batch คืออัปเดตทุก batch (ที่ใช้กันปกติ) อยู่นอก loop คืออัปเดตทุก epoch
- **[P2 00:13:25–00:14:00]** ทำไมไม่อัปเดตทุกภาพ (batch_size=1)? นอกจากเรื่องความเร็วแล้ว ภาพที่เป็น **outlier** จะดัน weight ไปผิดทาง การอัปเดตทีละ batch เฉลี่ย loss ไว้ เลยช่วยกลบ noise
- **[P2 00:06:46–00:06:51]** batch size เล็กจะแกว่งมากกว่า batch size ใหญ่ช่วย regularise มากกว่า
- **[P2 00:14:05–00:16:11]** ติดดาวบรรทัด print ทุก N การ print ทุก batch หรือทุก epoch ทำให้เทรนช้า เพราะ I/O ของการแสดงผลช้ากว่า CPU หรือ RAM มาก "Quote ที่ปกติรัน 10 นาที ผมโดนไป 1 วัน" (โค้ด 10 นาทีกลายเป็นทั้งวัน) ให้ใช้เงื่อนไข `if i % N`

### Torch vs NumPy (P1 00:52–00:54)
- **[P1 00:52:51–00:53:50]** "สิ่งแรกที่เด็กต้องรู้: NumPy process on CPU, Torch process on GPU" รุ่นพี่คนหนึ่งแปลง Torch เป็น NumPy แล้วแปลงกลับใน custom layer การส่งข้อมูล CPU↔GPU ทำให้งานที่ใช้ ~25 นาทีกลายเป็นสัปดาห์ "ตรงนี้คือสิ่งสำคัญครับ… ในโค้ดที่มีหน้าที่เทรน ใช้ Torch ให้หมด อย่าไปใช้ NumPy" ใช้ NumPy ตอนเตรียมข้อมูลได้ไม่มีปัญหา

### การสร้างโมเดล
- **[P1 00:15:12–00:15:18]** "จดไปครับ": หน้าที่ของ `__init__` คือสร้าง layer ให้เป็น attribute
- **[P1 00:20:36–00:20:49]** output channel ของ layer หนึ่งต้องเท่ากับ input ของ layer ถัดไป ("เลขมันต้องซิงค์กัน… อันนี้คือประเด็นหนึ่งที่ต้องระวัง")
- **[P1 00:27:28–00:27:39]** "จดไปครับ": `forward` คือ method ที่กำหนดการไหลของข้อมูลในโมเดล
- **[P1 00:29:20–00:29:47]** `forward` คืนค่าเป็น **score function** activation (softmax ฯลฯ) ทำนอกโมเดล
- **[P1 00:41:29–00:41:50]** output ของ Linear ตัวสุดท้ายต้องเท่ากับจำนวน class "สิ่งที่เราซีเรียสจริง ๆ ก็คือตัวสุดท้ายตัวนี้แหละ"
- **[P1 00:38:02–00:38:06]** ถ้า input เป็น grayscale แต่โมเดลรับ 3 channel ให้ก๊อปเป็น 3 channel อย่าไปแก้ layer "ไม่งั้นเดี๋ยว Pre-train จะเสีย"
- **[P1 01:22:18–01:23:28]** "อย่างต่อมาที่ต้องระวัง": ภาพ grayscale อาจถูกโหลดมาเป็น 3 channel ใน PyTorch เลือกหยิบมา channel เดียวได้ (1 จาก 3) หรือก๊อปเพิ่ม (1 เป็น 3) สำหรับโมเดลสี pretrained

### Inference, evaluation และการ save
- **[P1 01:13:44–01:13:58]** ตอน validation ใช้ `model.eval()` + `torch.no_grad()` ("ไม่ต้องแอบจำ ไม่งั้นมันจะแอบจำเฉลยไป")
- **[P1 01:15:41–01:15:52]** อาจารย์ลืมใส่ `model.eval()` ในสไลด์โค้ด test ให้เติม **ก่อน** `with torch.no_grad()`
- **[P2 00:16:40–00:17:07]** ตอน validation `model.eval()` ปิด Dropout และตั้ง BatchNorm เป็นโหมด eval ส่วน `torch.no_grad()` ไม่เก็บ gradient
- **[P2 00:33:05–00:33:34]** ติดดาวบรรทัด save "จดไปครับ": `model.state_dict()` save **แค่ weight ไม่มี architecture** ไฟล์เลยเล็กกว่า
- **[P2 00:34:03–00:34:18]** ตอน inference ต้องสร้างโมเดลให้เหมือนตอนเทรนเป๊ะ (เปลี่ยน `fc` แบบเดียวกัน) แล้ว `load_state_dict` เพื่อโหลด weight ที่เทรนแล้ว
- **[P2 00:36:32]** "อย่าลืม Eval ท่อนนี้"
- **[P2 00:28:49–00:29:11]** Inference = ภาพเดียว Evaluation = ทำแบบเดียวกันกับหลายภาพ "วันแข่ง" อาจารย์จะให้ path ของโฟลเดอร์ inference ทีละภาพเลยไม่พอ

### Transforms (P2 00:34–00:36) เรื่องที่เน้นหนักสุดของ Part 2
- **[P2 00:34:29–00:34:44]** ติดดาว block ของ transform: "จุดนี้มีรุ่นพี่คุณพลาดมาแล้ว"
- **[P2 00:34:50–00:35:14]** train, validate และ test ต้องมี **data standardisation เหมือนกัน**: ขนาดภาพเท่ากัน normalise เหมือนกัน crop เหมือนกัน
- **[P2 00:35:26–00:35:45]** data augmentation ต้องใช้ตอนเทรน แต่ไม่ใช้ตอน validation หรือ test ตอน inference ให้ก๊อป transform ของตอนเทรนมาแล้ว **เอา augmentation ออก**
- **[P2 00:35:50–00:36:22]** "ย้ำนะครับ… จำไว้เลย": transform มี 2 หน้าที่ (1) standardisation (เหมือนกันทั้ง train/val/test) และ (2) augmentation (**ตอนเทรนเท่านั้น**) "เด็กบางคนไปใส่ Data Augmentation ตอน Validate กับตอน Inference อันเนี้ยตลกมาก ไม่ควรนะครับ"
- **[P2 00:02:08]** `transforms.ToImage()`: "อันนี้สำคัญนะครับ"
- **[P2 00:02:46–00:03:15]** ตอน fine-tune โมเดล pretrained ให้ normalise ทุกภาพด้วย **mean/SD ของ ImageNet ชุดเดียวกัน**
- **[P1 01:18:28–01:18:33]** "เราจะไม่ Resize ก่อน Padding จำไว้นะครับ": pad ให้เป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize
- **[P1 01:21:14–01:21:34]** Random crop ใช้ตอนเทรนเท่านั้น ถ้าใช้ตอน inference จะพัง ("ไม่งั้นจะวิบัติ"): รูปเดิมได้ผลไม่เหมือนกัน

### โมเดล pretrained และประเภทงาน
- **[P2 00:21:03–00:21:22]** `squeezenet1_0()` แบบวงเล็บว่างคือได้ weight เริ่มต้น (ยังไม่เทรน) "อย่าทำแบบนี้"; ให้ใส่ `weights=...IMAGENET1K_V2`
- **[P2 00:31:20–00:32:16]** "โค้ดบรรทัดนี้นะครับ สำคัญมาก ตั้งใจฟังนะครับ": `model.fc = nn.Linear(2048, n_classes)` ก่อนอื่น `print(model)` เพื่อหาชื่อ attribute ของ layer สุดท้าย (`fc`) และขนาด input แล้วตั้ง output ให้เท่าจำนวน class ของเรา ส่วนที่เหลือยังเป็น weight pretrained การเทรนจะไป fine-tune layer นี้เป็นหลัก
- **[P2 00:37:29–00:37:53]** Multiclass vs multilabel ถ้าตอบว่า "1 ภาพหลายคำตอบ" ได้ **แค่ครึ่งคะแนน** ("อาจารย์จะให้แค่ครึ่งเดียว") คำตอบที่ถูก: "หนึ่งภาพมีโอกาสที่จะเป็นไปได้มากกว่าหนึ่งคำตอบ" และยัง **ไม่มีเลย** ก็ได้ ถ้า sigmoid ทุกตัวต่ำกว่า threshold
- **[P2 00:38:02–00:38:20]** multiclass กับ multilabel ต่างกันแค่ activation สุดท้ายหลัง score function: Softmax กับ Sigmoid
- **[P2 00:39:05]** ปกติ **ไม่** ใส่ Softmax/Sigmoid ไว้ในโมเดล แต่ใส่ข้างนอก
- **[P2 00:40:04–00:41:17]** Regression: จำนวน output node = จำนวนค่าที่จะทำนาย regression ยาก เลยแนะนำ 1 โมเดลต่อ 1 ค่า ใช้ output activation แบบ **Linear** (ไม่ใส่ activation) เพราะตัวอื่นจะทำให้ช่วงค่าเพี้ยน ใช้ loss แบบ L1 (MAE) หรือ L2 (MSE)
- **[P2 00:42:01]** โค้ด sigmoid / threshold ในสไลด์ inference ของ regression (p.121) เป็นสไลด์ที่ลืมลบ ("สไลด์เกินมา ผมลืมตัด") ให้ขีดฆ่าทิ้ง
- **[P2 00:42:39–00:43:02]** โค้ดเทรนของ multiclass, multilabel และ regression แทบเหมือนกัน 100% ต่างกันแค่ตอน inference อย่าสรุปว่า "ไม่มีโค้ดสำหรับ multi-label"
- **[P2 00:10:23–00:11:22]** ตั้ง epoch 50–100 พร้อม early stopping save epoch สุดท้าย จุด early-stop และโมเดลที่ผลดีที่สุด แล้วค่อยวิเคราะห์ว่าจะใช้ตัวไหน
- **[P2 00:07:41–00:08:12]** "ถ้า CPU ไม่แรง Training ก็ช้าได้": การโหลดข้อมูลและ transform รันบน CPU และ RAM num_workers = จำนวน core ของ CPU ที่ใช้โหลดข้อมูล

---

## ข้อสอบที่น่าจะออก

1. **เรียงลำดับขั้นใน training loop:** `loss.backward()`, `optimizer.step()`, `model(x)`, `optimizer.zero_grad()`, `criterion(out, y)`
   **ตอบ:** `zero_grad()` แล้ว `model(x)` แล้ว `criterion(...)` แล้ว `loss.backward()` แล้ว `optimizer.step()` ทั้งหมดอยู่ใน loop ของ epoch (และ batch) หลัง `model.train()`

2. **บรรทัดไหนอัปเดต weight จริง? บรรทัดไหนคำนวณ gradient? `output = model(x_train)` คือบรรทัดที่เทรนใช่ไหม?**
   **ตอบ:** `optimizer.step()` อัปเดต `loss.backward()` คำนวณ gradient `model(x_train)` เป็นแค่ **การทำนาย (forward pass)** ที่ต้องมีเพราะ loss และ gradient ต้องใช้ผลทำนาย

3. **ทำไมต้องเรียก `optimizer.zero_grad()` ทุกรอบ?**
   **ตอบ:** PyTorch *สะสม* gradient ทุกครั้งที่ `.backward()` ถ้าไม่รีเซ็ต gradient ของรอบก่อนจะบวกเข้ากับรอบนี้ แล้วการอัปเดตจะผิด

4. **`model.train()` กับ `model.eval()` ทำอะไร? ถ้าลืม `eval()` ตอน test จะเป็นยังไง?**
   **ตอบ:** ตั้งโหมดและสลับพฤติกรรมของ layer ในโหมด train Dropout ทำงาน BatchNorm ใช้สถิติของ batch และอัปเดต running mean/var ในโหมด eval Dropout ปิด BatchNorm ใช้สถิติ running ที่เก็บไว้ ค่า default คือ train ถ้าลืม `eval()` ตอน inference ผลทำนายจะแกว่ง เพี้ยน และไม่คงที่ (เช่น ผล k-fold ที่ fold แรก ๆ ดูแย่)

5. **`torch.no_grad()` ทำอะไร และทำไมใช้คู่กับ `model.eval()`?**
   **ตอบ:** ปิดการเก็บ gradient ใน block นั้น ทำให้ inference เร็วขึ้น ใช้ memory น้อยลง และใช้ batch ใหญ่ขึ้นได้ `eval()` ล็อกพฤติกรรม Dropout/BN ส่วน `no_grad()` ประหยัดทรัพยากร ใช้ทั้งคู่ตอน validation, test และ inference

6. **อธิบาย `if torch.cuda.is_available():` นักศึกษาตอบว่า "เช็กว่าเครื่องมี GPU ไหม" ถูกไหม?**
   **ตอบ:** ถูกแค่ครึ่งเดียว มันเช็ก 2 อย่าง: (1) มี GPU ไหม และ (2) GPU *ว่าง* ไหม (ไม่ได้ถูกใช้เต็ม) ถ้าคืน False โค้ดจะแอบรันบน CPU เงียบ ๆ ซึ่งอาจใช้เวลาเป็นวันหรือเป็นสัปดาห์ ในงานจริงบน server ที่ใช้ร่วมกัน อาจารย์บังคับ `.cuda()` ไปเลยเพื่อให้มัน error แทน

7. **Data leakage ในความหมายของ lecture นี้คืออะไร และป้องกันยังไง?**
   **ตอบ:** การเทียบโมเดลที่ชุด train/test ไม่เหมือนกัน: ภาพที่เป็น test ของโมเดล A กลับเป็น train ของโมเดล B ซึ่งไม่แฟร์และ bias การใช้ `train_test_split` หรือ k-fold อัตโนมัติ ต่อให้ตั้ง seed ก็อาจได้ split ไม่เหมือนกัน ป้องกันโดย **แบ่งเองครั้งเดียว** save split ไว้ แล้วโหลดไฟล์ชุดเดียวกันให้ทุกโมเดล
   *ต่อยอดทางการแพทย์ (ความรู้ทั่วไป ไม่ได้อยู่ในสไลด์):* ภาพทุกภาพของผู้ป่วยคนเดียวกันต้องอยู่ใน split เดียวกันด้วย ผู้ป่วยคนไหนก็ไม่ควรโผล่ทั้งใน train และ test

8. **pipeline นี้ผิดตรงไหน?** `transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToDtype(...), Normalize(...)])` ใช้กับทั้ง train, validation และ test
   **ตอบ:** augmentation (random crop, flip) ต้องใช้ **ตอนเทรนเท่านั้น** validation และ test ต้องใช้ *standardisation* เดียวกัน (ขนาด, crop, normalise เหมือนกัน) **แต่ไม่มี** augmentation ไม่งั้นภาพเดิมจะได้ผลทำนายต่างกันและ metric เชื่อไม่ได้

9. **โค้ดนี้ผิดตรงไหน?** (a) `model = resnet50()` แล้ว fine-tune สำหรับ 10 class (b) ลืม `model.fc = nn.Linear(2048, 10)`
   **ตอบ:** (a) ไม่มี weight pretrained (เริ่มแบบสุ่ม) ให้ใช้ `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` (b) output layer ยังมี 1000 node ให้ print โมเดล หา layer สุดท้าย (`fc`, in_features 2048) แล้วเปลี่ยนเป็น `Linear(2048, n_classes)`

10. **`torch.save(model, ...)` กับ `torch.save(model.state_dict(), ...)`: โหลดแต่ละแบบยังไง?**
    **ตอบ:** แบบแรก save architecture + weight (ไฟล์ใหญ่กว่า) โหลดด้วย `torch.load(path)` แบบที่สอง save แค่ weight (เล็กกว่า) ต้องสร้าง architecture เดิมขึ้นมาใหม่ (รวม `fc` ที่แก้แล้ว) แล้วเรียก `model.load_state_dict(torch.load(path))`

11. **Multiclass vs multilabel vs regression: ในโค้ดต้องเปลี่ยนอะไร?**
    **ตอบ:**
    - **Multiclass:** output node = จำนวน class, Softmax, CrossEntropyLoss, argmax
    - **Multilabel:** output node = จำนวน class, **Sigmoid** แยกทีละ node + threshold (0.5 ในแผนภาพ, 0.3 ในโค้ด), `BCEWithLogitsLoss` ภาพเดียว *อาจ* มีหลาย label หรือไม่มีเลยก็ได้
    - **Regression:** output node = จำนวนค่าที่จะทำนาย (แนะนำ 1 ค่าต่อ 1 โมเดล), activation แบบ **linear** (ไม่ใส่), loss แบบ L1/L2 (MAE/MSE), อ่านค่าออกมาตรง ๆ
    - นอกนั้นโค้ดเทรนแทบเหมือนกันหมด

12. **โมเดล pretrained รับ 224×224 แต่ภาพของเราเป็นสี่เหลี่ยมผืนผ้า มีทางเลือกอะไรบ้าง และแต่ละทางเสี่ยงอะไร?**
    **ตอบ:**
    1. Resize ตรง ๆ สัดส่วนเพี้ยน
    2. Pad ให้เป็นสี่เหลี่ยมจัตุรัส *แล้วค่อย* resize สัดส่วนไม่เพี้ยน
    3. Center crop แล้ว resize สมมติว่าวัตถุอยู่ตรงกลาง
    4. Random crop ใช้ตอนเทรนเท่านั้น
    5. ROI crop ต้องมี detector
    6. Tiling + majority vote มี overhead

13. **ภาพเป็น grayscale (1 channel) แต่ network pretrained ต้องการ 3 channel ทำยังไง?**
    **ตอบ:** ก๊อป channel ให้เป็น 3 อย่าไปแก้ conv ตัวแรก ไม่งั้นเสีย weight pretrained ถ้าไฟล์ grayscale โหลดมาเป็น 3 channel ที่เหมือนกันแล้วต้องการแค่ 1 ก็หยิบ channel ไหนก็ได้

14. **เติมตัวเลข:** `Conv2d(1,4,3,1,1)`, BN, ReLU, MaxPool(2,2) สองรอบกับ input 28×28 แล้ว `Linear(?, 10)` เลข `1`, `4` และ `BatchNorm2d(4)` หมายถึงอะไร?
    **ตอบ:** 28 → 14 → 7 เลยได้ `Linear(4*7*7=196, 10)` 1 = input channel (grayscale), 4 = output channel (feature map) BatchNorm2d(4) = จำนวน channel ที่จะ normalise ต้องตรงกับ output ของ conv

15. **ทำไมอัปเดต weight ทีละ mini-batch แทนที่จะทีละภาพหรือทีละ epoch? ทำไม print loss แค่ทุก N batch? ทำไมเลี่ยง NumPy ในโค้ดเทรน?**
    **ตอบ:**
    - **ทีละ batch:** loss ถูกเฉลี่ยทั้ง batch ภาพ outlier หรือภาพที่มี noise ภาพเดียวเลยไม่ลาก weight ไป และเร็วกว่าอัปเดตทีละภาพ ทีละ batch คือแบบที่ใช้กันทั่วไป
    - **การ print:** I/O ของการแสดงผลช้ากว่า CPU หรือ RAM มาก print บ่อย ๆ ทำให้งานหลักนาทีกลายเป็นวันได้
    - **NumPy:** รันบน CPU ส่วน Torch รันบน GPU แปลงไปมาเลยต้องส่งข้อมูล CPU↔GPU และช้าลงมหาศาล

---

## กับดักที่เจอบ่อย

- **เรียก `model(x)` ว่า "ขั้นเทรน"** มันคือการทำนาย การเทรน = loss แล้ว `backward()` แล้ว `step()`
- **สับสน `backward()` กับ `step()`** `backward()` คำนวณ gradient `step()` อัปเดต weight
- **ลืม `zero_grad()`** gradient จะสะสมข้ามรอบ
- **ลืม `model.eval()` ตอน validation, test หรือ inference** Dropout ยังเปิดอยู่และ BatchNorm ใช้สถิติของ batch โหมด default คือ *train* สไลด์ test ของอาจารย์เองก็ขาด `model.eval()` ให้เติมก่อน `with torch.no_grad()`
- **Loop แบบเก่าในสไลด์ p.50–55 (แหล่งข้อสอบแนว "ผิดตรงไหน")** มันคำนวณ `output_val = model(x_val)` ในโหมด train และยังเก็บ gradient อยู่ validation ที่ถูกต้องต้องใช้ `model.eval()` + `torch.no_grad()`
- **"`is_available()` เช็กว่ามี GPU ไหม"** นั่นแค่ครึ่งเดียว มันเช็กด้วยว่า GPU ว่างไหม
- **ใส่ augmentation ตอน val/test** ผิด standardisation ต้องเหมือนกันทั้ง train, val และ test ส่วน augmentation ใช้ตอนเทรนเท่านั้น
- **normalise ไม่เหมือนกันระหว่างเทรนกับ inference** (เช่น ตอนเทรนใช้ mean/std ของ ImageNet ตอน test ใช้ 0.5/0.5) ผลทำนายจะพัง ตอน fine-tune โมเดล ImageNet ให้ใช้ (0.485, 0.456, 0.406)/(0.229, 0.224, 0.225)
- **Resize ก่อน pad** ต้อง pad เป็นสี่เหลี่ยมจัตุรัสก่อน แล้วค่อย resize
- **Random crop ตอน inference** ผลจะไม่คงที่
- **โหลดโมเดล torchvision แบบวงเล็บว่าง** ทั้งที่อยากได้ weight pretrained จะได้ weight เริ่มต้นแบบสุ่ม
- **ไม่เปลี่ยน layer สุดท้าย (`model.fc`)** ของโมเดล pretrained มันยังให้ output 1000 class `Linear` ตัวใหม่ต้องมี in_features = 2048 สำหรับ ResNet-50 (เช็กด้วย `print(model)`)
- **โหลด `state_dict` โดยไม่ได้สร้าง architecture ให้เหมือนเดิม** (รวมถึง `fc` ที่แก้แล้ว) key และ shape จะไม่ตรงกัน
- **Channel / feature ไม่ตรงกัน** out_channels ของ conv ต้องเท่ากับ in_channels ของตัวถัดไปและ BatchNorm2d(num_features) out ของ Linear ก่อนหน้าต้องเท่ากับ in ของ Linear ถัดไป
- **ใส่ Softmax หรือ Sigmoid ไว้ในโมเดล** โมเดลในวิชานี้คืน score ดิบแล้วค่อยใส่ activation ข้างนอก `CrossEntropyLoss` และ `BCEWithLogitsLoss` มี softmax / sigmoid อยู่ในตัวแล้ว
- **target ของ `CrossEntropyLoss` ต้องเป็นเลข index ของ class แบบจำนวนเต็ม (`.long()`)** โค้ดเก่าแปลง label ที่เป็น float32 ด้วย `.long()` และ `.squeeze()`
- **inference ทีละภาพโดยไม่ `unsqueeze(0)`** โมเดลต้องการมิติ batch (N, C, H, W)
- **นิยาม multilabel ว่า "1 ภาพ หลายคำตอบ"** ได้ครึ่งคะแนน ต้องเป็น "อาจมีมากกว่าหนึ่งคำตอบ" และไม่มีเลยก็ได้
- **Regression ที่ output เป็น sigmoid หรือ softmax** ช่วงค่าจะเพี้ยน ใช้ output แบบ linear sigmoid ในสไลด์ inference ของ regression คือสไลด์ผิด
- **ใช้ `train_test_split` / k-fold อัตโนมัติเพื่อเทียบโมเดล** เสี่ยง leakage หรือเทียบไม่แฟร์ ให้ล็อก split เอง
- **แปลง Torch↔NumPy ใน training loop** ทำให้ต้องส่งข้อมูล CPU↔GPU และช้าลงมหาศาล
- **print ทุก batch** ทำให้เทรนช้าลงมาก
- **`Variable(...)`** เป็นของเก่าก่อนเวอร์ชัน 0.4 **`ToTensor()`** เลิกใช้แล้วใน v2; ให้ใช้ `ToImage()` + `ToDtype(float32, scale=True)`
- **`squeeze_()` vs `squeeze()`** `_` ต่อท้ายแปลว่า in-place
- **จุดที่ lecture กับสไลด์ไม่ตรงกัน** (ตอนสอบให้เชื่อสไลด์):
  - อาจารย์บอกว่า default ของ num_workers คือ 1 สไลด์บอกว่า **0**
  - Loss ของ multilabel: อาจารย์บอกว่า "เหมือน multiclass" โค้ดในสไลด์ใช้ **`BCEWithLogitsLoss`** และเชิงอรรถแผนภาพ p.110 บอกว่าใช้ BCE *หรือ* Cross Entropy ก็ได้ คำตอบที่ปลอดภัยสุดตอนสอบคือ Sigmoid + BCE (`BCEWithLogitsLoss`)
  - Threshold ของ multilabel: 0.5 ในแผนภาพและใน lecture, **0.3** ในโค้ด inference threshold เป็นค่าที่เราเลือกเอง
- **`test_dataloader` ในสไลด์ใช้ `shuffle=True`** ตัวอย่าง CIFAR ใช้ `shuffle=False` ตอน test การ shuffle ชุด test ไม่ได้ทำให้ accuracy เปลี่ยน แต่ก็ไม่จำเป็นต้องทำ
