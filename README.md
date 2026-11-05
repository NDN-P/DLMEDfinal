# DL in MED: เตรียมสอบ final (Ch 8–14)

เนื้อหาอ่านสอบ final วิชา *Deep Learning in Medicine* (KMITL)

## เริ่มตรงนี้
**อ่านก่อน: [EASY-SUMMARY.md](EASY-SUMMARY.md) สรุปง่าย ข้อละไม่กี่บรรทัด**

**[STUDY-GUIDE.md](STUDY-GUIDE.md) (ภาษาไทย, ละเอียด): ทุกอย่างที่ต้องใช้สอบรวมไว้ไฟล์เดียว แยกตามข้อ มีเรื่องที่ต้องรู้ วิธีคิดเลขทีละขั้น ตัวอย่างคำตอบ จุดหลอก และลำดับการอ่าน**

1. [exam-structure.md](exam-structure.md): ข้อสอบเขียน 7 ข้อ แต่ละข้อบอกว่าต้องอ่านอะไร
2. [exam-hints.md](exam-hints.md): เฉลยคำถามที่อาจารย์ถามในห้อง จุดที่สไลด์ผิด และเรื่องที่อาจารย์ย้ำบ่อย
3. `materials/Chapter N/notes.md` (Chapters 8–14): โน้ตแต่ละบท มี concept สูตร เรื่องที่อาจารย์เน้น (พร้อม timestamp) ข้อสอบที่น่าจะออก และจุดหลอก
4. [Project1/notes.md](Project1/notes.md): รายละเอียด Project 1 และสิ่งที่ต้องอธิบายได้ (ข้อสอบ Q1)
5. [mock-exam.md](mock-exam.md) → [mock-exam-answers.md](mock-exam-answers.md): ข้อสอบจำลองรูปแบบเดียวกับของจริง ตัวเลขทุกตัวเช็คด้วย Python แล้ว

ข้อสอบข้อ 3–7 มาจาก Chapters 10–13 ส่วน Chapters 8, 9 และ 14 ใช้ตอบข้อ case study **Chapter 15 (Transformers) ไม่ออกสอบ**

## บทเรียนและวิดีโอ lecture
Chapters 8–14 อยู่ใน [`materials/`](materials) ส่วน Chapter 15 อยู่ชั้นนอกสุดเพราะไม่ออกสอบ

| Folder | หัวข้อ | Lecture |
|---|---|---|
| Chapter 8 | Pre-trained CNN architectures, data augmentation | [YouTube](https://www.youtube.com/watch?v=3VpmHr2Hxdo) |
| Chapter 9 | Multi-label classification, image regression, multi-task learning | [YouTube](https://www.youtube.com/watch?v=lupWlZlnPaQ) |
| Chapter 10 | การเขียน CNN ใช้งานจริง (PyTorch) | [Part 1](https://www.youtube.com/watch?v=WfHaKE28gco), [Part 2](https://www.youtube.com/watch?v=3ntxugJKLco) |
| Chapter 11 | Object detection | [YouTube](https://www.youtube.com/watch?v=zzEWOhmlUxw) |
| Chapter 12 | Image segmentation | [YouTube](https://www.youtube.com/watch?v=l9Zrj877P9E) |
| Chapter 13 | Deep learning สำหรับประมวลผลวิดีโอ | [YouTube](https://www.youtube.com/watch?v=Aul0nwvG_9I) |
| Chapter 14 | Evaluation metrics (คาบ TA) | [YouTube](https://www.youtube.com/watch?v=2_O0rgWbvxg) |
| Chapter 15 | Transformers สำหรับ computer vision (**ไม่ออกสอบ**) | [YouTube](https://www.youtube.com/watch?v=NT0VGFoJxi0) |

## ในแต่ละ folder บทมีอะไรบ้าง
| ไฟล์ | เนื้อหา |
|---|---|
| `*.pdf` | สไลด์ lecture |
| `slides.txt` | ข้อความจากสไลด์ มีบอกเลขหน้า (ค้นหาได้) |
| `lecture*.txt` | transcript ของ lecture จาก Whisper บรรทัดละช่วง มี timestamp |
| `lecture*.th-orig.srt` | ซับไทยอัตโนมัติของ YouTube (หยาบกว่า transcript จาก Whisper) |
| `notes.md` | โน้ตอ่านสอบ |
| `instruction` | โน้ตของวิชา (Chapter 10: ลิงก์ไป notebook และ dataset ของวิชา) |

`Project1/` มีรูปรายละเอียด Project 1, ตารางคลาส (`6ab4…0aaf.txt`) และแบบฟอร์มส่งคำตอบ (`6ab4…0ab0.txt`, encoding Windows-874)

ไฟล์เสียง lecture ไม่ได้อยู่ใน repo (734 MB, อยู่บน YouTube) ถ้าจะทำ transcript ใหม่ ให้โหลดเสียงด้วย
`yt-dlp --js-runtimes node -f "bestaudio[ext=m4a]" -o "materials/Chapter N/lecture.m4a" <url>` แล้วรัน
[`transcribe.py`](transcribe.py) (faster-whisper large-v3 บน GPU ที่มี CUDA ใช้เวลาประมาณ 2.5 ชม. สำหรับทุก lecture บน RTX 5070 Laptop)
