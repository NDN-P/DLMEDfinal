# Project 1 — Thai Character & Number Recognition

Source: the two spec images in this folder (`download.jpg` = data dictionary, `download (1).jpg` = test set & submission guide)
and the two CSV files (`6ab4…0aaf.txt` = class table, `6ab4…0ab0.txt` = submission sheet template, encoded in Windows-874).
This is exam **Q1** ("โปรเจค 1 ที่ทำ"). Our group finished and passed it.

> **Fill in your group's own details** before the exam: model, input size, augmentation, validation accuracy, test score.
> The exam asks about *what you did*, so these matter more than anything generic below.

## The task
- **Image classification, 72 classes, one label per image** → multiclass (not multilabel, not segmentation).
- Input: one cropped image of a single Thai character, vowel/tone mark or Thai digit (`.jpg`).
- Output: a 3-digit **class_id**.

## class_id structure
First digit = type, digits 2–3 = index within the type (example: 205 = type 2, item 05 = ี).

| Type | Classes | class_id | Training folders |
|---|---|---|---|
| 1 Thai consonants (พยัญชนะ, incl. ฤ) | 41 | 101–141 | `./161` … `./206` |
| 2 Vowels, tone marks, symbols (สระ วรรณยุกต์ เครื่องหมาย) | 21 | 201–221 | `./207` … `./236` |
| 3 Thai digits ๐–๙ | 10 | 301–310 | `./240` … `./249` |

- Folder names are the character's TIS-620 code (161 = ก, 249 = ๙), so codes with no folder (165 ฅ, 166 ฆ, 172 ฌ, 174 ฎ, 198 ฦ …) are not in the dataset.
- The model outputs an index 0–71, which must be **mapped back to class_id** with the table. Writing the raw index is scored as wrong.

## Test and submission
| Item | Value |
|---|---|
| Test set | about 500 images, `./test_dataset/ts_img_001` … |
| Format | all `.jpg` |
| Image size | about **7×11 px up to 180×177 px** (W×H), so sizes and aspect ratios vary a lot |
| Time limit | **30 minutes** for inference |
| Submission | Google Sheet, one column per group (10 groups); fill only your own column |
| Scoring | prediction = gt_class_id → 1 point; wrong → 0; **blank or not finished in time → 0** |

So the score is simply **accuracy over ~500 images**, and an unfinished run costs as much as a wrong answer.

## Points to be able to explain (with the lecture they come from)
- **Output layer:** 72 nodes, softmax, cross-entropy (`nn.CrossEntropyLoss` takes logits; prediction = argmax). Ch 9 and Ch 10.
- **Resize:** pad to a square first, then resize, so proportions are kept. Ch 10 P1 `[01:17:20]` uses Thai characters as the example: characters that differ by proportion (with or without a tail) get confused if stretched.
  Tiny images (7×11 marks such as ่ ้) get blurry when upscaled. The thin strokes are what tell them apart.
- **Grayscale:** colour carries no meaning. Use 1 channel, or duplicate to 3 channels for an ImageNet-pretrained backbone.
- **Augmentation** (train only). Ch 8 `[00:41:04]` uses Thai characters as the example:
  - **no colour change**: letters aren't distinguished by colour
  - **no left-right flip**: real text is never mirrored; ใ/ไ and ด/ต would change meaning
  - **no up-down flip**: ุ ↔ ิ positions would swap meaning
  - **do rotate/tilt slightly and shift/scale slightly**: handwriting and scanning vary like this
- **Similar-looking classes** (check them in the confusion matrix):
  - ข/ฃ, ช/ซ, ด/ต, ถ/ภ, บ/ป, ผ/ฝ/พ/ฟ, ศ/ส, ห/ฬ, ท/ฑ
  - า/ๅ, ิ/ี/ึ/ื, ุ/ู, ่/้/๊, ั/็, ใ/ไ
  - ๔/๕ and ๓/๑
- **Inference:** `model.eval()` and `torch.no_grad()`. Use exactly the same preprocessing as validation. Batch the images, and write a CSV with the class_id column ready to paste into the sheet. Test the full pipeline before the 30-minute window. Ch 10.
- **Validation:** split before augmenting. Choose the model on validation accuracy, and use the confusion matrix to find which pairs to fix. Ch 10 and Ch 14.

## Our group (fill in)
- Model / backbone:
- Input size and preprocessing:
- Augmentation used:
- Training (epochs, optimizer, lr):
- Validation accuracy:
- Test score (out of ~500):
- Main confusions and what we tried:
- Who did what:
