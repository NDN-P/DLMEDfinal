"""Transcribe every Chapter*/lecture*.m4a to a timestamped .txt next to it (skips ones already done).

Usage: C:/Users/guymy/.venvs/dlmed/Scripts/python transcribe.py
"""
import sys
from pathlib import Path

import av
import numpy as np
import torch  # noqa: F401  loads the cuBLAS/cuDNN DLLs that ship with PyTorch, which CTranslate2 needs on Windows
from faster_whisper import WhisperModel

# Lectures are Thai with English technical terms; the prompt nudges Whisper to keep those terms in English.
PROMPT = ("บรรยายวิชา Deep Learning in Medicine: CNN, ResNet, data augmentation, segmentation, U-Net, "
          "object detection, YOLO, bounding box, IoU, mAP, Dice, loss function, transformer, attention.")


def ts(sec):
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_audio(path):
    """16 kHz mono float32. Own decoder because faster-whisper's passes an argument PyAV >= 15 removed."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(str(path)) as container:
        chunks = [f.to_ndarray() for frame in container.decode(audio=0) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray() for f in resampler.resample(None)]
    return np.concatenate(chunks, axis=1).ravel().astype(np.float32) / 32768.0


def main():
    root = Path(__file__).parent
    files = sorted(root.glob("Chapter*/lecture*.m4a"))
    todo = [f for f in files if not f.with_suffix(".txt").exists()]
    print(f"{len(files)} audio files, {len(todo)} to transcribe")
    if not todo:
        return
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for f in todo:
        print(f"-> {f.relative_to(root)}", flush=True)
        segments, info = model.transcribe(
            load_audio(f), language="th", initial_prompt=PROMPT, vad_filter=True, beam_size=5,
            condition_on_previous_text=False,  # avoids Whisper getting stuck repeating a phrase
        )
        out = f.with_suffix(".txt.part")
        with out.open("w", encoding="utf-8") as fh:
            for s in segments:
                fh.write(f"[{ts(s.start)}] {s.text.strip()}\n")
                if int(s.start) % 600 < 10:
                    print(f"   {ts(s.start)} / {ts(info.duration)}", flush=True)
        out.rename(f.with_suffix(".txt"))  # only finished transcripts get the .txt name


if __name__ == "__main__":
    sys.exit(main())
