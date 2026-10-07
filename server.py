#!/usr/bin/env python3
"""KittenTTS 2 HTTP API for speech synthesis and voice cloning."""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from kittenml import KittenTTS
from kittenml.kittentts2.text import join_chunks, split_for_synthesis

DEFAULT_CHECKPOINT = "KittenML/kitten-tts-2"
MAX_REFERENCE_BYTES = 50 * 1024 * 1024
app = FastAPI(title="KittenTTS 2 API")
logger = logging.getLogger("kitten_tts_api")
model_lock = threading.Lock()
model = None
model_checkpoint = None
model_load_args = {}
generation_defaults = dict(voice="Bruno", preset="stable", max_new_tokens=1000,
                           seed=42, normalize=True, split_sentences=False,
                           pause_ms=160, chunk_chars=380, chunk_min_chars=130,
                           output_format="pcm")


def _ensure_model():
    if model is None:
        raise HTTPException(503, "Model is not initialized. Start with `python server.py`.")
    return model


def _audio_to_pcm16le_bytes(audio):
    pcm = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1, 1)
    return (pcm * 32767).astype("<i2").tobytes()


def _decode_reference(upload):
    if upload.filename and not upload.filename.lower().endswith(".wav"):
        raise HTTPException(400, "`reference_wav` must be a .wav file.")
    data = upload.file.read(MAX_REFERENCE_BYTES + 1)
    if not data:
        raise HTTPException(400, "`reference_wav` was provided but is empty.")
    if len(data) > MAX_REFERENCE_BYTES:
        raise HTTPException(413, "`reference_wav` exceeds the 50 MiB limit.")
    try:
        with sf.SoundFile(io.BytesIO(data)) as source:
            if source.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("not WAV")
            if not 5 <= source.frames / source.samplerate <= 30:
                raise HTTPException(400, "Reference audio must be between 5 and 30 seconds.")
            audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
            rate = source.samplerate
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, "`reference_wav` is not a valid WAV file.") from exc
    if not np.isfinite(audio).all() or not np.any(audio):
        raise HTTPException(400, "Reference audio must contain finite, non-silent samples.")
    return audio, rate


def _forget_reference(tts_model, path):
    # Upload paths are unique: discard cached conditioning before deleting them.
    for cache in (tts_model._reference_cache, tts_model.codec._ref_cache):
        for key in list(cache):
            if path in key:
                del cache[key]


def _generate_audio(tts_model, text, reference, reference_text, voice, options,
                    advanced, split_sentences):
    segments = [text]
    if split_sentences:
        segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = sum(len(split_for_synthesis(tts_model.normalize_text(segment)
                                        if options["normalize"] else segment,
                                        advanced["chunk_chars"], advanced["chunk_min_chars"]))
                 for segment in segments)
    speaker = (dict(reference=reference, reference_text=reference_text)
               if reference else dict(voice=voice or "Bruno"))
    started = perf_counter()
    audios = [tts_model.generate(segment, **speaker, **options, advanced=advanced)
              for segment in segments]
    audio = (audios[0] if len(audios) == 1 else
             join_chunks(audios, tts_model.sample_rate, gap_s=advanced["chunk_gap_s"]))
    if audio is None or not np.isfinite(audio).all():
        raise RuntimeError("Model produced invalid audio.")
    return audio, chunks, perf_counter() - started


@app.get("/health")
def health():
    tts_model = _ensure_model()
    return dict(status="ok", checkpoint=model_checkpoint, sample_rate=tts_model.sample_rate)


@app.get("/model")
@app.get("/config")
def get_model_info():
    tts_model = _ensure_model()
    return JSONResponse(dict(checkpoint=model_checkpoint, load_args=model_load_args,
                             sample_rate=tts_model.sample_rate,
                             available_voices=tts_model.available_voices,
                             decode_presets=list(tts_model.decode_presets),
                             generation_defaults=generation_defaults,
                             voice_clone=dict(enabled=True, reference_text_required=True),
                             long_form=dict(enabled=True, automatic=True,
                                            strategy="sentence_aware_chunk_and_stitch")))


@app.post("/tts")
def tts(
    text: str = Form(...),
    ref_text: str | None = Form(None),
    reference_text: str | None = Form(None),
    reference_wav: UploadFile | None = File(None),
    voice: str | None = Form(None),
    preset: str = Form("stable"),
    max_new_tokens: int = Form(1000, ge=1),
    temperature: float | None = Form(None, gt=0),
    top_p: float | None = Form(None, gt=0, le=1),
    top_k: int | None = Form(None, ge=1),
    min_p: float | None = Form(None, ge=0, le=1),
    seed: int = Form(42, ge=0),
    normalize: bool = Form(True),
    split_sentences: bool = Form(False),
    pause_ms: int = Form(160, ge=0),
    chunk_chars: int = Form(380, ge=130),
    chunk_min_chars: int = Form(130, ge=1),
    output_format: str = Form("pcm"),
):
    tts_model = _ensure_model()
    text = text.strip()
    if not text:
        raise HTTPException(400, "`text` must be non-empty.")
    ref_text = (ref_text or "").strip() or None
    reference_text = (reference_text or "").strip() or None
    if ref_text and reference_text and ref_text != reference_text:
        raise HTTPException(400, "`ref_text` and `reference_text` disagree.")
    transcript = ref_text or reference_text
    if reference_wav is not None and transcript is None:
        raise HTTPException(400, "`ref_text` or `reference_text` is required with `reference_wav`.")
    if reference_wav is None and transcript is not None:
        raise HTTPException(400, "`reference_wav` is required with reference text.")
    if reference_wav is not None and voice is not None:
        raise HTTPException(400, "Provide either `voice` or `reference_wav`.")
    if voice is not None and voice not in tts_model.available_voices:
        raise HTTPException(400, "Unknown voice. See /model for available_voices.")
    if preset not in tts_model.decode_presets:
        raise HTTPException(400, "Unknown preset. See /model for decode_presets.")
    if output_format not in {"pcm", "wav"}:
        raise HTTPException(400, "`output_format` must be pcm or wav.")
    if chunk_min_chars > chunk_chars:
        raise HTTPException(400, "`chunk_min_chars` must not exceed `chunk_chars`.")
    if any(value is not None and not math.isfinite(value)
           for value in (temperature, top_p, min_p)):
        raise HTTPException(400, "Sampling parameters must be finite.")
    decoded = _decode_reference(reference_wav) if reference_wav is not None else None
    advanced = dict(seed=seed, chunk_chars=chunk_chars, chunk_min_chars=chunk_min_chars,
                    chunk_gap_s=pause_ms / 1000)
    options = dict(preset=preset, max_new_tokens=max_new_tokens, normalize=normalize,
                   temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
    try:
        with model_lock, tempfile.TemporaryDirectory(prefix="kitten-reference-") as tmp:
            path = None
            if decoded is not None:
                path = str(Path(tmp) / "reference.wav")
                sf.write(path, *decoded, subtype="FLOAT")
            try:
                audio, segments, seconds = _generate_audio(
                    tts_model, text, path, transcript, voice, options, advanced, split_sentences)
            finally:
                if path is not None:
                    _forget_reference(tts_model, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(500, "TTS generation failed; see server logs.") from exc
    if output_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, audio, tts_model.sample_rate, format="WAV", subtype="PCM_16")
        content = buffer.getvalue()
    else:
        content = _audio_to_pcm16le_bytes(audio)
    logger.info("Generated characters=%d segments=%d seconds=%.3f", len(text), segments, seconds)
    return Response(content, media_type="audio/wav" if output_format == "wav" else "audio/pcm",
                    headers={"X-Audio-Format": "wav_pcm_s16le" if output_format == "wav" else "pcm_s16le",
                             "X-Sample-Rate": str(tts_model.sample_rate), "X-Channels": "1",
                             "X-Segments": str(segments), "X-Generation-Seconds": f"{seconds:.3f}"})


def main(argv=None):
    global model, model_checkpoint, model_load_args
    parser = argparse.ArgumentParser(description="Launch the KittenTTS 2 API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None, help="Auto-select CUDA or CPU.")
    parser.add_argument("--weights", choices=["packed", "emb4", "full"], default="packed")
    parser.add_argument("--decoder", default="default")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(args.threads)
    model = KittenTTS(args.checkpoint, device=args.device, weights=args.weights, decoder=args.decoder)
    if not hasattr(model, "decode_presets"):
        raise ValueError("This server requires a KittenTTS 2 checkpoint.")
    model_checkpoint = args.checkpoint
    model_load_args = dict(device=model.device, dtype=str(model.dtype),
                           weights=args.weights, decoder=args.decoder)
    logger.info("Model ready: %s", model_load_args)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
