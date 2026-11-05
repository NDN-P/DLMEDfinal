# Chapter 10 — Practical Implementation of Convolutional Neural Networks (PyTorch)

This chapter walks through a full PyTorch CNN workflow (model class, data, training loop, saving, inference, evaluation) for multiclass, multilabel and regression image tasks. The old MNIST-style code is shown first, then the modern version with Dataset, DataLoader, transforms.v2, pretrained ResNet-50 and `state_dict`.
Lecturer: ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์ ("อาจารย์แบงค์"). Course notebooks: `classification_train.ipynb` (training), `classification_inference.ipynb` (one image at a time), `classification_evaluation.ipynb` (many images at once). Datasets: multiclass, multi-label and image-regression.

---

## Key concepts

### 0. Frameworks (intro slides and P1 00:00–00:06)
- **PyTorch** (Facebook): chosen for this course because you can customise everything: training loop, inference and architecture. Research papers favour it ("ประมาณ 1,000 กว่า Paper" use PyTorch vs "400 กว่า" for TensorFlow, 2021–2025).
- **TensorFlow / Keras** (Google). Keras is a *high-level API* that calls TF/PyTorch underneath. It is easy to read, but you can't really modify it.
- **OpenVINO** is a toolkit for *deploying* a trained model on limited hardware. It is the only one of these **not** designed for training.
- **JAX** is suited to large-scale computing.
- **Inference** means taking a finished model and using it.
- Slide p.5–6 (an image-only comparison table):
  - **Keras 3** is a high-level, multi-backend API (by François Chollet, 2015). It runs on a TensorFlow, JAX or PyTorch backend, plus OpenVINO for inference only. Keras doesn't wrap TF/PyTorch, and they don't need Keras.
  - **TensorFlow** is an end-to-end ML platform (Google Brain, 2015) for training, inference and production.
  - **JAX** is array computing (Google): NumPy-style, autodiff, JIT, CPU/GPU/TPU.
  - **OpenVINO** (Intel) is an inference-only toolkit for optimising and deploying models.
  - **PyTorch** is Python-first and started at FAIR (now the PyTorch Foundation). It offers training + inference and flexible training loops, and suits research + production.
- Slide p.7 (popularity chart):
  - arXiv papers 2021–2025: PyTorch **1,163**, TensorFlow 428, JAX 226, Keras 103, OpenVINO 23.
  - GitHub stars: TensorFlow **199.7K**, PyTorch 102.9K, Keras 64.3K, JAX 36.3K, OpenVINO 10.8K.
  - The slide's caveat: stars reflect accumulated interest, not current users or performance.

### 1. Files in a CNN project (slides p.16, P1 00:08)
| File | Role |
|---|---|
| `Net.py` | defines the architecture (the class). Not needed if you load an existing architecture such as ResNet |
| `TrainingCNN.py` | imports Net and trains it, producing `model.pt` |
| `TestingCNN.py` | imports Net / loads `model.pt` and tests it |
| `model.pt` / `.pth` | the trained model |

- Training Phase uses images + labels to produce the model. Testing Phase sends test images through the model to get the answer.
- Forward propagation = recognise the image (score, then softmax probabilities, then class). Backward propagation = use the loss to update the weights (loss, then optimizer).
- Slide example: scores [1.16, −0.81, −3.44, 3.91] give softmax probabilities [0.0596, 0.0083, 0.0006, 0.9315].

### 2. Defining the model (`class Net(Module)`) (slides p.18–33, P1 00:11–00:42)
```python
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(            # feature extraction
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4), ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4), ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        self.linear_layers = Sequential(Linear(4 * 7 * 7, 10))   # classification
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)      # flatten
        x = self.linear_layers(x)
        return x                       # raw scores (no softmax)
```
- The class must **inherit from `nn.Module`**. The teacher also says the class name should match the file name (`Net` in `net.py`), because otherwise the import may fail.
- Required methods:
  - `__init__` (the constructor) **creates the layers as attributes** ("หน้าที่ของ Init คือการสร้าง Layer ให้เป็น Attribute"). `super().__init__()` calls the parent constructor, and you always do this.
  - `forward(self, x)` **defines the data flow inside the model**. `x` is the user's input.
- `self.xxx` with no parentheses is an attribute. `self.xxx()` with parentheses is a method.
- Layers can be stored as 1 attribute per layer, or as a group of layers (a block) inside `nn.Sequential`. Both are allowed. Sequential runs the layers in order, top to bottom. Its drawback is that it can't branch.
- Slide p.18 compares `TinyConv` (with Sequential, thumbs up) against `TinyConvLessOrganized` (one attribute per layer, thumbs down). Sequential isn't mandatory, but it is *recommended* because it makes the code tidier and groups layers into a block.
- Parameter registration: the `nn.Parameter` values (weights and biases) created in `__init__` are registered automatically in `self.parameters()`. Example: `Linear(3,2)` has a weight of shape 2×3 and a bias of shape 2 (`requires_grad=True`).
- **Channel matching**: the `out_channels` of one conv must equal the `in_channels` of the next. `BatchNorm2d(n)` takes n = number of channels coming in. For each Linear, in = the previous layer's out.
- `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, …, padding_mode='zeros')`. `in_channels`=1 means grayscale and 3 means colour. `kernel_size=3` means a 3×3 filter, and a tuple gives a non-square filter. Padding defaults to zero-padding, and `padding_mode` changes that.
- `ReLU(inplace=True)` writes the result over the input tensor to save memory. The default is `False`, which creates a new tensor.
- `MaxPool2d(kernel_size=2, stride=2)` halves H and W. This is the usual fixed setting.
- `Linear(in_features, out_features)` is a fully connected layer. `4*7*7` = 4 channels × 7×7 map after two poolings of a 28×28 input (28→14→7). **The last Linear's output must equal the number of classes** ("สิ่งที่เราซีเรียสจริง ๆ").
- `Dropout(p=0.5)`: p is the probability of zeroing a value. The default is 0.5 and the recommended range is 0.2–0.5. It is active **only in training**.
- Flatten can be done 3 ways: `x.view(x.size(0), -1)`, `out.reshape(out.size(0), -1)`, or `torch.flatten(x, 1)` / `nn.Flatten`.
- `forward` returns the **score function**. Softmax or sigmoid is applied **outside** the model.
- Multi-task output (P1 00:30): branch two heads off the same feature map in `forward` and return y1 and y2 together.
- AlexNet (torchvision, slide p.32): `Conv2d(3,64,…)` means colour input. It has `num_classes=1000` and `dropout=0.5` as default constructor arguments. `AdaptiveAvgPool2d((6,6))` fixes each channel to 6×6 without changing batch or channels, e.g. (1,3,10,14) → (1,3,6,6). Global Average Pooling is `AdaptiveAvgPool2d((1,1))`, giving 1 value per channel. These poolings let images of different sizes reach the FC layer.
- To visualise a model, use https://netron.app/.

### 3. Old-style training script: 7 steps (slide p.35)
1. **Data Loader**: load images + labels
2. **Training & Validating Set Generation**: split into train and validation sets
3. **Model Loader**: load the architecture
4. **Defining Learning Algorithm**: optimizer, loss, GPU
5. **Training Model**
6. **Evaluating Performance**: loss / accuracy
7. **Is it accepted?** If yes, save and stop. If not, adjust.

Flowchart on slides p.33–56 (image-only):
- Start → Data Loader (images + labels).
- Then three blocks feed **Training Model**: Defining Learning Algo, Training & Validating Set Generation, and Model Loader (from `Net.py`).
- Training Model → Evaluating Performance → "Is it accepted?"
  - **No:** "Feedback error to update weight model", which loops back to Training Model.
  - **Yes:** End.
- The folder layout on slide p.33: `train/` (1.png, 2.png, … + `train.csv`), `test/`, `out.csv`, `TrainingCNN.py`, `TestingCNN.py`.
- `train.csv` has two columns. `id` holds the image number without path or extension (1, 2, 3 …). `label` holds the class number (9, 0, 0, 3 …).

The teacher's simplification (P1 00:43): training needs **3 inputs**: (1) a model, (2) training data, (3) a defined training method. That is why the training code is long.

**Data loading (old):** read `train.csv` (column `id` = image name, `label` = class **number**, not name), then `imread(path, as_gray=True)`, `img /= 255.0` (0–1 range), `astype('float32')`, then a list, then `np.array`, giving `(60000, 28, 28)`.
**Split:** `train_test_split(train_x, train_y, test_size=0.1)` gives 54000 / 6000. Then `reshape(54000, 1, 28, 28)` to N×C×H×W, and `torch.from_numpy(...).to(torch.float32)`.
**Model loader:** `from Net import Net`, then `model = Net()`. Loading a model takes 2 steps: import it, then create the object.

### 4. Defining the learning algorithm + GPU check (slide p.42)
```python
model = Net()
optimizer = SGD(model.parameters(), lr=0.07, momentum=0.9)   # Adam(model.parameters(), lr=0.07) commented out
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
```
- `model.parameters()` passes the weights to the optimizer, so the model and the optimizer "know each other".
- `torch.optim` usage: create the optimizer object, set hyperparameters (lr, …), then update with the gradients.
- **Normalization** (slide p.45): x′ = (x − μ)/σ gives mean 0 and SD 1. If features have very different ranges (0–1 vs 0–255), gradient descent takes longer to converge.
- **Loss function** (slide p.47): a single number for how wrong the model is. Lower is better. It takes softmax probabilities.
  - correct + confident gives a low loss
  - correct + not confident gives a high loss
  - wrong + confident gives a high loss
  - Classification losses: Hinge, Cross Entropy. Optimization means minimising the loss.
- Modern device idiom (handouts):
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device); inputs, labels = inputs.to(device), labels.to(device)
```

### 5. The training loop: the core of the exam
Old version (slides p.50–56). It is full-batch and has only an epoch loop:
```python
for epoch in tqdm(range(n_epochs)):          # n_epochs = 25
    model.train()
    optimizer.zero_grad()                    # reset accumulated gradients
    output_train = model(x_train)            # PREDICTION (forward pass)
    loss_train = criterion(output_train, y_train)   # loss
    loss_train.backward()                    # compute gradients
    optimizer.step()                         # update weights
    tr_loss = loss_train.item()
```
Modern version (slide p.86, handout). It has **2 for-loops**: the outer one over epochs, the inner one over batches.
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
        if i % 2000 == 1999:                 # print only every N batches
            print(...); running_loss = 0.0
```
Role of each call:
| Call | What it does |
|---|---|
| `model.train()` | sets the **state** to training: Dropout on, BatchNorm uses batch mean/var and updates its running stats. It does *not* train anything by itself |
| `optimizer.zero_grad()` | resets gradients to 0. PyTorch **accumulates** gradients, so without this the previous iteration's gradients get added in |
| `model(x)` | **prediction (forward pass)**, not "the training line". It is needed because the loss requires a prediction |
| `criterion(output, y)` | computes the loss (here CrossEntropy). Targets must be `long` (int64) |
| `loss.backward()` | **computes the gradients** of the loss with respect to every parameter with `requires_grad=True` |
| `optimizer.step()` | **updates the weights** using those gradients: w_new = w_old − α·∂loss/∂w |
| `loss.item()` | the loss as a Python number, for logging |

Chain: **predict, then criterion (loss), then backward (gradient), then step (update)**. If asked "which commands train the model", the answer is these 4 [P1 01:12:00].
- Where `step()` sits sets how often the weights update: inside the batch loop means **every batch** (the usual choice), outside it means every epoch, and batch_size=1 means every image (not recommended, see emphasis list).
- Helper calls: `Variable(...)` is only needed in PyTorch < 0.4. Tensors now store data and gradients. `squeeze()` removes size-1 dims, e.g. (10,1,28,28) → (10,28,28); `squeeze_()` is the in-place version (the `_` suffix). `.long()` is the same as `.to(torch.int64)`; `.to()` is more flexible (any dtype or device).
- **Epoch** = one pass over the whole dataset. **Batch** = a chunk of it. Example: 100 images with batch size 20 gives 5 batches per epoch.

### 6. Saving the model (checkpoints)
- Old: `torch.save(model, 'C:/model04.pt')` saves **architecture + weights**. Load it with `model = torch.load(...)`.
- New: `torch.save(model.state_dict(), 'model.pt')` saves **weights only**, so the file is smaller. To load it, rebuild the same architecture (with the same `fc` change), then `model.load_state_dict(torch.load('./model.pt'))`. This prints `<All keys matched successfully>`.
- Extensions are `.pt` or `.pth`.
- Three checkpoints to keep from one run [P2 00:10:35]: (1) the last epoch, (2) the early-stopping point, (3) the best validation performance.

### 7. Validation / testing / inference
```python
model.eval()                 # Dropout off, BatchNorm uses stored running mean/var
with torch.no_grad():        # no gradient tracking: faster, less memory, larger batches possible
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
```
- Old style: `softmax = torch.exp(output).cpu()`, `np.argmax(prob, axis=1)`, `accuracy_score(val_y, predictions)`, then predictions are written to `out.csv`.
- Single-image inference (handout): open the image with PIL and `.convert('RGB')`, apply the **same standardisation transform**, `image.unsqueeze(0)` (add the batch dimension), `model.eval()`, `outputs.squeeze(0)`, `torch.softmax(outputs, 0)`, `torch.max(score, 0)`. Result: "Predicted: kiwi 99.89%".
- Inference of a classifier is light and can run on CPU, so forcing CUDA isn't required there [P1 01:13:58].
- Test script = load model + load data + execute. There is no learning-algorithm or training block.
- Clarification (standard PyTorch): weights only change on `optimizer.step()`. `no_grad()` stops gradient tracking (memory/speed) and `eval()` switches Dropout/BN behaviour, so you need both for correct, efficient evaluation.

### 8. Modern pipeline: transforms, Dataset, DataLoader (slides p.79–112, P2)
**Basic classification steps (slide p.79):** (1) dataset, (2) CNN, (3) loss + optimizer, (4) train, (5) test/evaluate.

**Transforms** have 2 jobs [P2 00:01]: **(1) data standardisation / pre-processing** (to tensor, dtype, resize/crop, normalise) and **(2) data augmentation**. Chain them with `Compose`:
```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToImage(),                          # convert to image/tensor format
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])   # gives range -1..1
```
Augmentation example (`torchvision.transforms.v2`, slide p.107):
```python
preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),         # float in [0,1]
    v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])  # ImageNet
```
- Other v2 transforms:
  - Geometry: Resize, RandomResize, RandomCrop, RandomHorizontalFlip, RandomZoomOut, RandomRotation, Pad, CenterCrop
  - Colour: Grayscale, RGB, GaussianBlur, RandomAdjustSharpness
  - Composition: RandomApply
  - Conversion: PILToTensor, ToDtype, ToPILImage. `ToTensor()` is **deprecated** and now equals `Compose([ToImage(), ToDtype(float32, scale=True)])`.
  - Auto-augmentation: AugMix, CutMix
- An image normalised with ImageNet mean/SD looks partly black when displayed because the values are negative. The data is fine; only the range changed [P2 00:03:27].

**Dataset vs DataLoader (slide p.100):**
- `torch.utils.data.Dataset` stores the samples + labels (the data source).
- `torch.utils.data.DataLoader` feeds the Dataset in **batches** to training/testing and returns an iterator.
- Built-in datasets: `torchvision.datasets.CIFAR10/FashionMNIST/ImageNet(root=..., train=True/False, download=True, transform=...)`. `train=True` loads the training split and `False` the test split. `download=True` downloads only if the data isn't already there.

**Custom Dataset** must inherit `Dataset` and implement **3 methods**:
```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file); self.img_dir = img_dir
        self.transform = transform; self.target_transform = target_transform
    def __len__(self):  return len(self.img_labels)          # number of samples
    def __getitem__(self, idx):                               # return (image, label) for index idx
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path); label = self.img_labels.iloc[idx, 1]
        if self.transform: image = self.transform(image)
        if self.target_transform: label = self.target_transform(label)
        return image, label
```
- `__init__` runs once when the object is created (root path, transforms, label file). `transform` acts on the **image**, `target_transform` on the **label**.
- The fruit handout maps class names to indices with `class_to_idx`, and the folder layout has one folder per class plus `annotation.csv`.

**DataLoader parameters:**
```python
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader  = DataLoader(testset,  batch_size=4, shuffle=False, num_workers=2)
```
- `batch_size` is how many samples go in at once.
- `shuffle=True` reshuffles the order **every epoch**.
- `num_workers` is the number of CPU processes loading data. The slide says the default is 0.
- Data loading, transforms and augmentation run on **CPU + RAM**. Training runs on the **GPU**, so a weak CPU slows training even with a good GPU.

### 9. Using a pretrained backbone (slides p.96–98, P2 00:17–00:33)
```python
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)   # pretrained on ImageNet
model.fc = nn.Linear(2048, len(classes))                   # replace the output layer
```
- `resnet50()` with no arguments gives an architecture with **random initial weights**. The teacher says not to load it that way when you want pretrained weights.
- Pretrained models output 1000 ImageNet classes, so **replace the last layer**:
  1. `print(model)` to find the attribute name of the last layer (`fc` for ResNet) and its input size (2048)
  2. assign a new `nn.Linear(in, n_classes)`
- All weights except the new layer stay pretrained, so training mainly optimises (fine-tunes) that layer.
- Other models on the slide: googlenet, squeezenet1_0, densenet121 (`*_Weights.IMAGENET1K_V2`), VGG, EfficientNet, MobileNet.
- Choosing a model for embedded / IoT / mobile: pick one with a low **parameter count** and low **GFLOPS**. The two don't always go together.
- Slide p.92 (a screenshot of the torchvision weights table):
  - Columns: Weight, **Acc@1, Acc@5, Params, GFLOPS**.
  - Examples: AlexNet 56.5% / 61.1M / 0.71 GFLOPS. EfficientNet_B0 77.7% / 5.3M / 0.39. ConvNeXt_Large 84.4% / 197.8M / 34.36. DenseNet121 74.4% / 8.0M / 2.83. GoogLeNet 69.8% / 6.6M / 1.5.
  - So AlexNet is large in parameters yet low in GFLOPS, which shows the two measures are not proportional.

### 10. Multiclass vs multilabel vs regression (slides p.110–123, P2 00:37–00:43)
| | Multiclass | Multilabel | Regression |
|---|---|---|---|
| Meaning | 1 image, 1 class | 1 image **may** have more than 1 class (or none) | predict a continuous value |
| Output nodes | number of classes | number of classes (e.g. `Linear(2048, 24)`) | number of values to predict (recommended **1 per model**: `Linear(2048, 1)`) |
| Final activation (applied outside the model) | Softmax | **Sigmoid** + threshold (slide uses `> 0.3`) | **Linear** (none), so the range isn't distorted |
| Loss on the slide | `CrossEntropyLoss` | `BCEWithLogitsLoss` (Sigmoid + BCE in one) in the code. The diagram footnote says BCE **or** Cross Entropy is allowed | `L1Loss` (MAE). L2 (MSE) is also fine |
| Inference read-out | `torch.max(softmax)` | `torch.nonzero(sigmoid(out) > 0.3)` | `outputs.squeeze().item()`, e.g. "… years old" |

The training code is almost the same in all three ("แทบจะเหมือนกัน 100%"). Only the output layer, the loss and the inference read-out change.

Diagrams on slides p.110 and p.114 (image-only), "การพิจารณาผลลัพธ์ของแบบจำลองและการคำนวณข้อผิดพลาด":
- **Multiclass:** scores (−3.44, 1.66, 3.91) → Softmax → Bird 0.010, Cat 0.059, Dog 0.931. Compared with the one-hot answer (0, 0, 1) via Cross Entropy Loss.
- **Multilabel:** the same scores → a **separate sigmoid on each node** → Drama 0.03, Sci-Fi 0.84, Action 0.98 → **threshold at 0.5** → prediction (0, 1, 1). Compared with the answer (0, 0, 1) via Cross Entropy Loss. The footnote says the loss can be Binary Cross Entropy or Cross Entropy.
- **Regression:** one output node with a **Linear function** → 2,450,000 vs the true 2,500,000 → **L1 Loss**. The footnote allows MAE/L1, MSE/L2 and others.

### 11. Fitting images to a fixed model input size (P1 01:16–01:22)
Pretrained models expect a fixed size, usually **224×224 or 256×256**. The 6 techniques:
1. **Direct resize.** Distorts the object's proportions, so shape-based classes (e.g. character recognition) get confused.
2. **Pad to square (zeros), then resize.** Keeps proportions. **Pad first, then resize.**
3. **Center crop, then resize.** Assumes the object is in the centre, as in normal phone or camera photos.
4. **Random crop, then resize.** **Training only** (adds variety each epoch). Never at inference, or the same photo gives different answers.
5. **ROI crop, then resize.** A model or algorithm (e.g. face detection) finds the region first.
6. **Tiling / sliding window** + majority vote. Has overhead.

In practice the teacher mostly uses 1–3.

Slide p.67 summary (image-only; example: 16:10 image to 224×224×3; "the size actually needed depends on the architecture and pretrained weights"):

| # | Slide name | Suits | Caution |
|---|---|---|---|
| 1 | Direct Resize (Stretch) | images with similar proportions | objects stretched or squashed. Fast and easy, but avoid it when shape is a key feature |
| 2 | Resize + Padding (Letterbox) | Detection, Segmentation, OCR | padding may become a feature the model memorises. Use it when images have mixed aspect ratios and shape must be kept |
| 3 | Center Crop + Resize | products, faces, centred objects | objects near the edge get cut off. No padding and no distortion |
| 4 | Random Resized Crop | data augmentation for classification | may cut out key features. **Random in training only; validation/test use a fixed transform** |
| 5 | ROI / Object-Centric Crop | **medical images**, leaf disease, tiny objects | needs a bounding box, mask or ROI detector. Reduces background and makes small objects clearer |
| 6 | Tiling / Sliding Window | **histology**, satellite, segmentation | slower, and the patch results must be combined. Use overlap so objects aren't cut at patch borders |

- The slide's box says "Resize + Padding" (keep the ratio, then fill the empty area to the target size). The teacher said "pad to square first, then resize". Both keep the aspect ratio; what's forbidden is a plain resize that stretches the image.
- Slide reminders:
  - Normal images use **Bilinear/Bicubic** interpolation. Segmentation **masks use Nearest-neighbour**.
  - The **image and its label (mask) must be transformed with the same parameters**.
  - Do colour conversion and normalisation (matching the pretrained model) **after** resizing and padding.

**Grayscale with a 3-channel pretrained model:** duplicate the single channel to 3 rather than changing the first conv, which would lose the pretrained weights. Some file types load a grayscale image as 3 identical channels. If you need 1 channel, take any one of them.

**Slide p.73 caution:** AlexNet expects 227×227×3 (3 planes), so grayscale or binary input must be handled as above.

### 12. Extras (slide p.74)
- **Regularization** fixes or eases over/underfitting. 3 popular methods: Data Augmentation, Batch Normalization, Dropout.
- **BatchNorm2d(num_features)** keeps one feature from dominating, speeds up training and lowers the loss.

---

## Formulas & numbers to remember

| Item | Value (source) |
|---|---|
| Normalisation | x′ = (x − μ)/σ, giving mean 0 and SD 1 (slide p.45) |
| Gradient descent | w_new = w_old − α · ∂loss/∂w_old. α = learning rate. Gradient points uphill, so move the opposite way (p.53) |
| Pixel scaling (old code) | `img /= 255.0`, giving 0–1, as float32 |
| Normalize(0.5, 0.5) per channel | maps [0,1] to **[−1, 1]** (p.80). Un-normalise for display: `img/2 + 0.5` |
| ImageNet mean / std (RGB) | **(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)** |
| MNIST-style data | 60000 images of 28×28 grayscale. `test_size=0.1` gives **54000 train / 6000 val**. Tensor shape **(N, 1, 28, 28)** = (Batch, Channel, Height, Width). Test set 10000 |
| Net FC input | `4*7*7` (28 → 14 → 7 after two MaxPool(2,2)), 10 outputs |
| ConvNet example | Conv(1,32,k5,p2), Conv(32,64,k5,p2), FC `7*7*64` → 1000 → 10 |
| CIFAR model | Conv(3,6,5), Pool(2,2), Conv(6,16,5), FC 16·5·5 → 120 → 84 → 10. Regression version: 16·5·5 → 128 → 64 → 1 |
| AlexNet | input 227×227×3. AdaptiveAvgPool → 6×6. FC 256·6·6 → 4096 → 4096 → 1000. Dropout 0.5 |
| ResNet-50 | `fc` in_features = **2048**. Fruit = 10 classes, multilabel = 24, regression = 1. IMAGENET1K_V2 weights ≈ 80.858% accuracy |
| Old optimizer | `SGD(lr=0.07, momentum=0.9)` (Adam lr=0.07 commented out). Optim slide example: SGD lr=0.01 m=0.9, Adam lr=0.0001 |
| Modern optimizer | `optim.SGD(model.parameters(), lr=0.001, momentum=0.9)`. Momentum is "usually 0.9" |
| Epochs | old 25, CIFAR 10, fruit handout 4. Real work: **50–100 + early stopping** [P2 00:10:32] |
| Batch size | CIFAR / fruit **4**, custom example **64**. Teacher: use powers of 2. Classification 16/32/64, segmentation 4/8, video 2 |
| Print frequency | every 2000 mini-batches (CIFAR), every 500 (fruit) |
| Dropout | default p=0.5, recommended 0.2–0.5 |
| BatchNorm2d defaults | eps=1e-05, momentum=0.1, affine=True, track_running_stats=True |
| Conv2d defaults | stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros' |
| MaxPool | kernel 2, stride 2 halves H and W |
| Model input sizes | 224×224 or 256×256. Handout Resize (256,256). RandomResizedCrop 224 |
| Flip probability | RandomHorizontalFlip p=0.5 |
| Multilabel threshold | inference code (p.117): sigmoid > **0.3**. Diagram (p.110) and the teacher: **0.5**. It is a tunable choice |
| DataLoader num_workers | slide: default **0**. CIFAR example uses 2 |
| Seed | `torch.manual_seed(0)` in the training handout |
| Linear(3,2) | weight 2×3 + bias 2 |

---

## What the teacher emphasised

### Data leakage and the train/val split (P1 00:49–00:51)
- **[P1 00:49:52–00:50:07]** About the `train_test_split` line: for a real project (Year-4 project, not this course), *don't* use `train_test_split` or automatic k-fold. "เราควรจะแบ่ง manual และให้มันโหลดมาเอง เพื่อกันการ data leak" (split manually and load the fixed splits).
- **[P1 00:50:13–00:50:34]** In a Data Science / AI project defence, examiners will flag your experiment as biased. Even setting a seed doesn't guarantee each model gets the same images ("ต่อให้ใส่ Seed บางครั้งมันก็มีความคาดเคลื่อน").
- **[P1 00:50:38–00:50:59]** His definition of the leak: "สมมุติภาพผมถูกใช้เป็นเทสต์ ณ Model A แต่ภาพผมอาจจะใช้เป็นเทรนด์ ณ Model B ซึ่งมันไม่แฟร์". A test image must be test for *every* model, and a train image must be train for every model. So build the train/validate/test split by hand and load it from files.

### The GPU-check exchange, "ถูกครึ่งเดียว" (P1 00:56–01:00)
Note: the "ผิด / ถูกครึ่งเดียว" correction in this window is about `torch.cuda.is_available()`, not data leakage.
- **[P1 00:56:04]** "บรรทัดนี้ กาดอกจันทร์ครับ" (star this line): `if torch.cuda.is_available(): model = model.cuda(); criterion = criterion.cuda()`.
- **[P1 00:56:22–00:56:31]** He asked what the line is for. One student's answer got "ผิด". Another said it is for checking, and he replied "ใช่ แต่ประโยคที่เพื่อนพูดไม่ถูก" (yes, but the wording isn't right).
- **[P1 00:56:44]** A student answered "ตรวจสอบว่าเครื่องนั้นมีการ์ดจอหรือไม่" (checks whether the machine has a GPU). He replied **"อันนี้ถูกครึ่งเดียว"** (only half right).
- **[P1 00:56:50–00:57:06]** The other half: "มีแล้วมัน available ไหม". The GPU may exist but be in use by someone else. **"is available มันเช็ค 2 stage"**: (1) is there a GPU, (2) is it available.
- **[P1 00:57:14–00:58:45]** The story: on the shared faculty server all GPUs were full, so the check returned False and the code silently ran on CPU. A job that took 3 days still wasn't finished after more than a week, and it took a week of debugging to find the cause.
- **[P1 00:58:45–00:59:56]** "ถ้าการ์ดจอเต็ม ควรรันไหม ไม่ควร". In real work, drop the `if` and force `model.cuda()` / `criterion.cuda()`. If the GPU isn't available it errors out, and you just don't run yet. Internet code keeps the `if` only because it doesn't know your hardware.

### Training loop
- **[P1 01:02:29–01:02:41]** Star `model.train()`: "ตายมานักต่อนักแล้ว…หลายๆคนทำงานเสร็จปุ๊บต้องรันใหม่เพราะบรรทัดนี้" (many people had to rerun their whole experiment because of this line).
- **[P1 01:03:11–01:05:06]** "จดไปครับ": `model.train()` declares that the current state is training. "ทำไมต้องบอก นี่แหละคือคำถามสำคัญ". Some layers behave differently: Dropout is active only in training, and BatchNorm accumulates mean/SD in training but uses the stored values in eval. So train/eval switches certain layers on and off.
- **[P1 01:05:07–01:05:41]** The bug is not in training (the default is train mode anyway). It is in **inference without `model.eval()`**, which gives distorted values. Another case: k-fold validation without `eval()`, where early folds come out bad and later folds good.
- **[P1 01:06:37–01:06:46]** `zero_grad` clears the previous gradients ("Set Gradient เป็น 0").
- **[P1 01:07:03–01:07:24]** "หลาย ๆ คนเข้าใจผิด". Asked which line trains the model, students pointed at `model(x_train)`, and he said **"อันนี้ผิด"**. That line is **prediction**. It is needed because training updates weights by gradient, the gradient comes from the loss, and the loss needs a prediction.
- **[P1 01:09:39–01:09:58]** "จดไปครับ": `loss_train.backward()` computes the gradient from the loss.
- **[P1 01:10:25–01:10:39]** `optimizer.step()` updates the weights. "เวทถูกอัปเดตจริงๆที่คำสั่งไหน: optimizer.step".
- **[P1 01:11:34–01:12:05]** Summary: model predicts, then criterion gives the loss, then backward gives the gradient, then step updates. If asked which commands are used to train, the answer is these 4.
- **[P2 00:12:13]** The CIFAR code has no `model.train()`, which is fine because train is the default.
- **[P2 00:12:44–00:13:19]** Where the update code sits sets the update frequency: inside the batch loop means every batch (the usual choice), outside it means every epoch.
- **[P2 00:13:25–00:14:00]** Why not update on every image (batch_size=1)? Besides speed, an **outlier** image would push the weights the wrong way. Batch updates average the loss, which absorbs noise.
- **[P2 00:06:46–00:06:51]** Small batch size swings more. Large batch size is more regularising.
- **[P2 00:14:05–00:16:11]** Star the print-every-N line. Printing every batch or epoch slows training because display I/O is far slower than CPU or RAM. "Quote ที่ปกติรัน 10 นาที ผมโดนไป 1 วัน" (10-minute code took a whole day). Use an `if i % N` check.

### Torch vs NumPy (P1 00:52–00:54)
- **[P1 00:52:51–00:53:50]** "สิ่งแรกที่เด็กต้องรู้: NumPy process on CPU, Torch process on GPU". A former student converted Torch to NumPy and back inside a custom layer, and the CPU↔GPU transfer turned a ~25-minute run into a week. "ตรงนี้คือสิ่งสำคัญครับ… ในโค้ดที่มีหน้าที่เทรน ใช้ Torch ให้หมด อย่าไปใช้ NumPy". Using NumPy during data preparation is fine.

### Model definition
- **[P1 00:15:12–00:15:18]** "จดไปครับ": the job of `__init__` is to create layers as attributes.
- **[P1 00:20:36–00:20:49]** The output channels of one layer must equal the input of the next ("เลขมันต้องซิงค์กัน… อันนี้คือประเด็นหนึ่งที่ต้องระวัง").
- **[P1 00:27:28–00:27:39]** "จดไปครับ": `forward` is the method that defines the data flow in the model.
- **[P1 00:29:20–00:29:47]** `forward` returns the **score function**. The activation (softmax etc.) happens outside the model.
- **[P1 00:41:29–00:41:50]** The output of the last Linear must equal the number of classes. "สิ่งที่เราซีเรียสจริง ๆ ก็คือตัวสุดท้ายตัวนี้แหละ".
- **[P1 00:38:02–00:38:06]** For grayscale input to a 3-channel model, duplicate to 3 channels. Don't modify the layer, "ไม่งั้นเดี๋ยว Pre-train จะเสีย".
- **[P1 01:22:18–01:23:28]** "อย่างต่อมาที่ต้องระวัง": a grayscale image may load as 3 channels in PyTorch. You can pick one channel (1 from 3), or duplicate (1 to 3) for pretrained colour models.

### Inference, evaluation and saving
- **[P1 01:13:44–01:13:58]** For validation, use `model.eval()` + `torch.no_grad()` ("ไม่ต้องแอบจำ ไม่งั้นมันจะแอบจำเฉลยไป").
- **[P1 01:15:41–01:15:52]** He forgot `model.eval()` on the test-code slide. Add it **before** `with torch.no_grad()`.
- **[P2 00:16:40–00:17:07]** For validation, `model.eval()` disables Dropout and sets BatchNorm to eval mode, and `torch.no_grad()` drops gradient storage.
- **[P2 00:33:05–00:33:34]** Star the save line. "จดไปครับ": `model.state_dict()` saves **only the weights, not the architecture**, so the file is smaller.
- **[P2 00:34:03–00:34:18]** At inference, rebuild the model exactly as in training (same `fc` replacement), then `load_state_dict` to load the trained weights.
- **[P2 00:36:32]** "อย่าลืม Eval ท่อนนี้".
- **[P2 00:28:49–00:29:11]** Inference = one image. Evaluation = the same thing on many images. "On competition day" he will give a folder path, so one-image inference won't be enough.

### Transforms (P2 00:34–00:36), the strongest emphasis of Part 2
- **[P2 00:34:29–00:34:44]** Star the transform block: "จุดนี้มีรุ่นพี่คุณพลาดมาแล้ว" (seniors have made this mistake).
- **[P2 00:34:50–00:35:14]** Train, validate and test must all have the **same data standardisation**: same image size, same normalisation, same crop.
- **[P2 00:35:26–00:35:45]** Data augmentation is needed in training but not in validation or test. For inference, copy the training transform and **remove the augmentation**.
- **[P2 00:35:50–00:36:22]** "ย้ำนะครับ… จำไว้เลย": the transform has 2 jobs, (1) standardisation (identical in train/val/test) and (2) augmentation (**training only**). "เด็กบางคนไปใส่ Data Augmentation ตอน Validate กับตอน Inference อันเนี้ยตลกมาก ไม่ควรนะครับ".
- **[P2 00:02:08]** `transforms.ToImage()`: "อันนี้สำคัญนะครับ".
- **[P2 00:02:46–00:03:15]** When fine-tuning a pretrained model, normalise every image with the **same ImageNet mean/SD**.
- **[P1 01:18:28–01:18:33]** "เราจะไม่ Resize ก่อน Padding จำไว้นะครับ": pad to square first, then resize.
- **[P1 01:21:14–01:21:34]** Random crop is for training only. At inference it is a disaster ("ไม่งั้นจะวิบัติ"): the same photo gives different results.

### Pretrained models and task types
- **[P2 00:21:03–00:21:22]** `squeezenet1_0()` with empty parentheses means initial (untrained) weights. "อย่าทำแบบนี้"; pass `weights=...IMAGENET1K_V2`.
- **[P2 00:31:20–00:32:16]** "โค้ดบรรทัดนี้นะครับ สำคัญมาก ตั้งใจฟังนะครับ": `model.fc = nn.Linear(2048, n_classes)`. First `print(model)` to find the last layer's attribute name (`fc`) and its input size, then set the output to your class count. The rest keeps the pretrained weights, and training fine-tunes mostly this layer.
- **[P2 00:37:29–00:37:53]** Multiclass vs multilabel. Answering "1 ภาพหลายคำตอบ" gets **only half credit** ("อาจารย์จะให้แค่ครึ่งเดียว"). The correct answer: "หนึ่งภาพมีโอกาสที่จะเป็นไปได้มากกว่าหนึ่งคำตอบ", and it can also be **none** if all sigmoids fall below the threshold.
- **[P2 00:38:02–00:38:20]** The only difference between multiclass and multilabel is the final activation after the score function: Softmax vs Sigmoid.
- **[P2 00:39:05]** Softmax/Sigmoid is usually **not** put inside the model; it is applied outside.
- **[P2 00:40:04–00:41:17]** Regression: output nodes = the number of values to predict. Regression is hard, so 1 model per value is recommended. Use **Linear** output activation (no activation) because others would distort the range. Use L1 (MAE) or L2 (MSE) loss.
- **[P2 00:42:01]** The sigmoid / threshold code on the regression-inference slide (p.121) is a leftover slide error ("สไลด์เกินมา ผมลืมตัด"). Cross it out.
- **[P2 00:42:39–00:43:02]** Multiclass, multilabel and regression training code is almost 100% the same; only inference differs. Don't conclude "there's no code for multi-label".
- **[P2 00:10:23–00:11:22]** Set epochs to 50–100 with early stopping. Save the last epoch, the early-stop point and the best-performance model, then analyse which to use.
- **[P2 00:07:41–00:08:12]** "ถ้า CPU ไม่แรง Training ก็ช้าได้": data loading and transforms run on CPU and RAM. num_workers = number of CPU cores used for loading.

---

## Likely exam questions

1. **Put the training-loop steps in order:** `loss.backward()`, `optimizer.step()`, `model(x)`, `optimizer.zero_grad()`, `criterion(out, y)`.
   **A:** `zero_grad()`, then `model(x)`, then `criterion(...)`, then `loss.backward()`, then `optimizer.step()`, all inside the epoch (and batch) loop after `model.train()`.

2. **Which line actually updates the weights? Which computes the gradients? Is `output = model(x_train)` the training line?**
   **A:** `optimizer.step()` updates. `loss.backward()` computes the gradients. `model(x_train)` is only a **prediction (forward pass)**. It is needed because the loss, and therefore the gradient, requires a prediction.

3. **Why call `optimizer.zero_grad()` every iteration?**
   **A:** PyTorch *accumulates* gradients on `.backward()`. Without resetting, the previous iteration's gradients add to the current ones and the update is wrong.

4. **What do `model.train()` and `model.eval()` do? What if you forget `eval()` at test time?**
   **A:** They set the mode and switch layer behaviour. In train mode Dropout is active and BatchNorm uses batch statistics and updates its running mean/var. In eval mode Dropout is off and BatchNorm uses the stored running statistics. The default is train, so forgetting `eval()` at inference gives noisy, distorted, non-deterministic predictions (e.g. k-fold results that look bad for early folds).

5. **What does `torch.no_grad()` do, and why use it with `model.eval()`?**
   **A:** It turns off gradient tracking inside the block, which means faster inference, less memory and room for larger batches. `eval()` fixes Dropout/BN behaviour while `no_grad()` saves resources. Use both for validation, test and inference.

6. **Explain `if torch.cuda.is_available():`. A student says "it checks whether the machine has a GPU". Is that right?**
   **A:** Only half right. It checks 2 things: (1) a GPU exists, and (2) it is *available* (not fully occupied). If it returns False the code silently runs on CPU and can take days or weeks. In real work on shared servers the teacher forces `.cuda()` so it errors instead.

7. **What is data leakage in the sense used in this lecture, and how do you prevent it?**
   **A:** Comparing models whose train/test membership differs: an image that is test data for model A is training data for model B, which is unfair and biased. Automatic `train_test_split` or k-fold, even with a seed, may give different splits. Prevent it by **splitting manually once**, saving the splits, and loading the same files for every model.
   *Medical extension (general knowledge, not from the slides):* also keep all images of one patient in the same split, so that no patient appears in both train and test.

8. **What's wrong with this pipeline?** `transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToDtype(...), Normalize(...)])` is used for train, validation and test.
   **A:** Augmentation (random crop, flip) must be **training only**. Validation and test need the same *standardisation* (same size, crop, normalisation) **without** augmentation. Otherwise the same image gives different predictions and the metrics are unreliable.

9. **What's wrong with this code?** (a) `model = resnet50()` then fine-tune for 10 classes. (b) Forgetting `model.fc = nn.Linear(2048, 10)`.
   **A:** (a) There are no pretrained weights (random initialisation); use `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`. (b) The output layer still has 1000 nodes. Print the model, find the last layer (`fc`, in_features 2048) and replace it with `Linear(2048, n_classes)`.

10. **`torch.save(model, ...)` vs `torch.save(model.state_dict(), ...)`: how do you load each?**
    **A:** The first saves architecture + weights (bigger) and loads with `torch.load(path)`. The second saves weights only (smaller). For it, recreate the same architecture (including the modified `fc`) and call `model.load_state_dict(torch.load(path))`.

11. **Multiclass vs multilabel vs regression: what changes in the code?**
    **A:**
    - **Multiclass:** output nodes = classes, Softmax, CrossEntropyLoss, argmax.
    - **Multilabel:** output nodes = classes, **Sigmoid** on each node + threshold (0.5 in the diagram, 0.3 in the code), `BCEWithLogitsLoss`. One image *may* have several labels, or none.
    - **Regression:** output nodes = values to predict (recommended 1 per model), **linear** (no) activation, L1/L2 (MAE/MSE) loss, read the value directly.
    - Training code is otherwise almost identical.

12. **The pretrained model takes 224×224 but your images are rectangular. What are your options, and what's the risk of each?**
    **A:**
    1. Direct resize distorts proportions.
    2. Pad to a square and *then* resize; this keeps proportions.
    3. Center crop then resize, which assumes a centred object.
    4. Random crop, for training only.
    5. ROI crop, which needs a detector.
    6. Tiling + majority vote, which has overhead.

13. **Your images are grayscale (1 channel) but the pretrained network expects 3 channels. What do you do?**
    **A:** Duplicate the channel to 3. Don't change the first conv, or you lose the pretrained weights. If a grayscale file loads as 3 identical channels and you need 1, take any one channel.

14. **Fill in the numbers:** `Conv2d(1,4,3,1,1)`, BN, ReLU, MaxPool(2,2) twice on a 28×28 input, then `Linear(?, 10)`. What do `1`, `4` and `BatchNorm2d(4)` mean?
    **A:** 28 → 14 → 7, so `Linear(4*7*7=196, 10)`. 1 = input channels (grayscale), 4 = output channels (feature maps). BatchNorm2d(4) = number of channels to normalise, which must match the conv output.

15. **Why update the weights per mini-batch instead of per image or per epoch? Why print the loss only every N batches? Why avoid NumPy inside the training code?**
    **A:**
    - **Per batch:** the loss is averaged over the batch, so a single outlier or noisy image doesn't steer the weights, and it is faster than per-image updates. Per batch is the common choice.
    - **Printing:** display I/O is far slower than CPU or RAM, and frequent prints can turn minutes into a day.
    - **NumPy:** it runs on CPU and Torch on GPU, so converting back and forth forces CPU↔GPU transfers and huge slowdowns.

---

## Common traps

- **Calling `model(x)` "the training step".** It is prediction. Training = loss, then `backward()`, then `step()`.
- **Mixing up `backward()` and `step()`.** `backward()` computes gradients. `step()` updates the weights.
- **Forgetting `zero_grad()`.** Gradients accumulate across iterations.
- **Forgetting `model.eval()` at validation, test or inference.** Dropout stays on and BatchNorm uses batch statistics. The default mode is *train*. The teacher's own test slide is missing `model.eval()`; add it before `with torch.no_grad()`.
- **The old loop on slides p.50–55 (a "what's wrong" source).** It computes `output_val = model(x_val)` in train mode with gradient tracking. Proper validation uses `model.eval()` + `torch.no_grad()`.
- **"`is_available()` checks if a GPU exists."** That is only half of it; it also checks that the GPU is free.
- **Augmentation at val/test time.** Wrong. Standardisation must be identical across train, val and test, and augmentation is for training only.
- **Different normalisation between training and inference** (e.g. ImageNet mean/std in training, 0.5/0.5 at test). The predictions break. When fine-tuning ImageNet models, use (0.485, 0.456, 0.406)/(0.229, 0.224, 0.225).
- **Resizing before padding.** Pad to square first, then resize.
- **Random crop at inference.** It gives non-deterministic results.
- **Loading a torchvision model with empty parentheses** when you want pretrained weights. You get random initialisation.
- **Not replacing the last layer (`model.fc`)** of a pretrained model. It still outputs 1000 classes. The new `Linear` must have in_features = 2048 for ResNet-50 (check with `print(model)`).
- **Loading a `state_dict` without rebuilding the identical architecture** (including the modified `fc`). The keys and shapes won't match.
- **Channel / feature mismatch.** Conv out_channels must equal the next in_channels and BatchNorm2d(num_features). The previous Linear's out must equal the next Linear's in.
- **Putting Softmax or Sigmoid inside the model.** The course's models return raw scores and apply the activation outside. `CrossEntropyLoss` and `BCEWithLogitsLoss` already include softmax / sigmoid.
- **Targets for `CrossEntropyLoss` must be integer class indices (`.long()`).** The old code converts float32 labels with `.long()` and `.squeeze()`.
- **Single-image inference without `unsqueeze(0)`.** The model expects a batch dimension (N, C, H, W).
- **Defining multilabel as "1 image, many answers".** That gets half credit. It is "may be more than one answer", and it can also be none.
- **Regression with a sigmoid or softmax output.** It distorts the range. Use linear output. The sigmoid on the regression-inference slide is a slide error.
- **Automatic `train_test_split` / k-fold for comparing models.** It risks leakage or unfair comparison. Fix the split manually.
- **Torch↔NumPy conversion inside the training loop.** It causes CPU↔GPU transfers and massive slowdowns.
- **Printing every batch.** It slows training dramatically.
- **`Variable(...)`** is a pre-0.4 artifact. **`ToTensor()`** is deprecated in v2; use `ToImage()` + `ToDtype(float32, scale=True)`.
- **`squeeze_()` vs `squeeze()`.** The trailing `_` means in-place.
- **Discrepancies between lecture and slides** (trust the slides in the exam):
  - The teacher said num_workers' default is 1; the slide says **0**.
  - Multilabel loss: the teacher said it is "the same as multiclass". The slide code uses **`BCEWithLogitsLoss`**, and the p.110 diagram footnote says BCE *or* Cross Entropy is acceptable. The safest exam answer is Sigmoid + BCE (`BCEWithLogitsLoss`).
  - Multilabel threshold: 0.5 in the diagram and the lecture, **0.3** in the inference code. The threshold is something you choose.
- **The slide's `test_dataloader` uses `shuffle=True`.** The CIFAR example uses `shuffle=False` for testing. Shuffling the test set doesn't change accuracy, but there's no need for it.
