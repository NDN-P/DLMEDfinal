# DL in MED: final exam prep (Ch 8–15)

Study material for the final exam of *Deep Learning in Medicine* (KMITL).

## Start here
1. [exam-structure.md](exam-structure.md): the exam's 7 written questions, each mapped to what to study
2. [exam-hints.md](exam-hints.md): in-class quiz answers, errors on the slides, recurring themes
3. `Chapter N/notes.md`: per-chapter notes with concepts, formulas, what the teacher emphasised (with timestamps), likely questions and traps
4. [mock-exam.md](mock-exam.md) → [mock-exam-answers.md](mock-exam-answers.md): practice exam in the real format; all numbers checked in Python

Exam questions 3–7 come from Chapters 10–13; Chapters 8, 9, 14 and 15 feed the case-study question.

## Chapters and lecture videos
| Folder | Topic | Lecture |
|---|---|---|
| Chapter 8 | Pre-trained CNN architectures, data augmentation | [YouTube](https://www.youtube.com/watch?v=3VpmHr2Hxdo) |
| Chapter 9 | Multi-label classification, image regression, multi-task learning | [YouTube](https://www.youtube.com/watch?v=lupWlZlnPaQ) |
| Chapter 10 | Practical implementation of CNNs (PyTorch) | [Part 1](https://www.youtube.com/watch?v=WfHaKE28gco), [Part 2](https://www.youtube.com/watch?v=3ntxugJKLco) |
| Chapter 11 | Object detection | [YouTube](https://www.youtube.com/watch?v=zzEWOhmlUxw) |
| Chapter 12 | Image segmentation | [YouTube](https://www.youtube.com/watch?v=l9Zrj877P9E) |
| Chapter 13 | Deep learning for video processing | [YouTube](https://www.youtube.com/watch?v=Aul0nwvG_9I) |
| Chapter 14 | Evaluation metrics (TA session) | [YouTube](https://www.youtube.com/watch?v=2_O0rgWbvxg) |
| Chapter 15 | Transformers for computer vision | [YouTube](https://www.youtube.com/watch?v=NT0VGFoJxi0) |

## Not in this repo
Lecture audio, slide PDFs and full transcripts stay local (size, and they are the teacher's material).
To re-create the transcripts: download audio with `yt-dlp --js-runtimes node -f "bestaudio[ext=m4a]" -o "Chapter N/lecture.m4a" <url>`,
then run [`transcribe.py`](transcribe.py) (faster-whisper large-v3 on a CUDA GPU; ~2.5 h for all lectures on an RTX 5070 Laptop).

Related project repo: [NDN-P/DLMED](https://github.com/NDN-P/DLMED) (exam question 1).
