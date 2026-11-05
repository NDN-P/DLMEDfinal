# DL in MED: final exam prep (Ch 8–14)

Study material for the final exam of *Deep Learning in Medicine* (KMITL).

## Start here
**[STUDY-GUIDE.md](STUDY-GUIDE.md) (ภาษาไทย): everything you need for the exam in one file: per question, the must-know facts, worked calculations, model answers and traps, plus the study order.**

1. [exam-structure.md](exam-structure.md): the exam's 7 written questions, each mapped to what to study
2. [exam-hints.md](exam-hints.md): in-class quiz answers, errors on the slides, recurring themes
3. `materials/Chapter N/notes.md` (Chapters 8–14): per-chapter notes with concepts, formulas, what the teacher emphasised (with timestamps), likely questions and traps
4. [Project1/notes.md](Project1/notes.md): Project 1 spec and what to explain (exam Q1)
5. [mock-exam.md](mock-exam.md) → [mock-exam-answers.md](mock-exam-answers.md): practice exam in the real format; all numbers checked in Python

Exam questions 3–7 come from Chapters 10–13; Chapters 8, 9 and 14 feed the case-study question. **Chapter 15 (Transformers) is not on the exam.**

## Chapters and lecture videos
Chapters 8–14 are in [`materials/`](materials). Chapter 15 stays at the top level because it is not on the exam.

| Folder | Topic | Lecture |
|---|---|---|
| Chapter 8 | Pre-trained CNN architectures, data augmentation | [YouTube](https://www.youtube.com/watch?v=3VpmHr2Hxdo) |
| Chapter 9 | Multi-label classification, image regression, multi-task learning | [YouTube](https://www.youtube.com/watch?v=lupWlZlnPaQ) |
| Chapter 10 | Practical implementation of CNNs (PyTorch) | [Part 1](https://www.youtube.com/watch?v=WfHaKE28gco), [Part 2](https://www.youtube.com/watch?v=3ntxugJKLco) |
| Chapter 11 | Object detection | [YouTube](https://www.youtube.com/watch?v=zzEWOhmlUxw) |
| Chapter 12 | Image segmentation | [YouTube](https://www.youtube.com/watch?v=l9Zrj877P9E) |
| Chapter 13 | Deep learning for video processing | [YouTube](https://www.youtube.com/watch?v=Aul0nwvG_9I) |
| Chapter 14 | Evaluation metrics (TA session) | [YouTube](https://www.youtube.com/watch?v=2_O0rgWbvxg) |
| Chapter 15 | Transformers for computer vision (**not on the exam**) | [YouTube](https://www.youtube.com/watch?v=NT0VGFoJxi0) |

## What each chapter folder contains
| File | Content |
|---|---|
| `*.pdf` | lecture slides |
| `slides.txt` | text of the slides, with page markers (searchable) |
| `lecture*.txt` | Whisper transcript of the lecture, one timestamped line per segment |
| `lecture*.th-orig.srt` | YouTube's Thai auto-captions (rougher than the Whisper transcript) |
| `notes.md` | study notes |
| `instruction` | course notes (Chapter 10: link to the course notebooks and datasets) |

`Project1/` holds the Project 1 spec images, the class table (`6ab4…0aaf.txt`) and the submission sheet template (`6ab4…0ab0.txt`, Windows-874 encoding).

Lecture audio is not in the repo (734 MB, on YouTube). To re-create the transcripts, download the audio with
`yt-dlp --js-runtimes node -f "bestaudio[ext=m4a]" -o "materials/Chapter N/lecture.m4a" <url>`, then run
[`transcribe.py`](transcribe.py) (faster-whisper large-v3 on a CUDA GPU; ~2.5 h for all lectures on an RTX 5070 Laptop).
