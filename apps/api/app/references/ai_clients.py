from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.core.llm import DEFAULT_CLAUDE_MODEL, claude_json, create_claude_client
from app.references.files import write_reference_raw_stt, write_reference_script


class TranscriptionClient(Protocol):
    def transcribe(self, audio_path: Path) -> dict:
        ...


class LocalWhisperTranscriptionClient:
    """faster-whisper 로컬 STT. OpenAI Whisper verbose_json과 같은 형태를 반환한다.

    반환: {"text": str, "language": str, "duration": float,
           "segments": [{"id", "start", "end", "text"}]}
    """

    def __init__(self, model: str = "small", device: str = "cpu"):
        self.model_name = model or "small"
        self.device = device or "cpu"
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            compute_type = "int8" if self.device == "cpu" else "float16"
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=compute_type)
        return self._model

    def transcribe(self, audio_path: Path) -> dict:
        # condition_on_previous_text=True(기본값)이면 BGM이 깔린 쇼츠에서 첫 문장 뒤
        # 디코딩이 멈춰 세그먼트가 1개만 나온다 (58초 음원 → 1문장). 끄면 전체가 전사된다.
        segments_iter, info = self._load().transcribe(
            str(audio_path), vad_filter=True, condition_on_previous_text=False
        )
        segments = [
            {"id": i, "start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for i, s in enumerate(segments_iter)
        ]
        return {
            "text": " ".join(s["text"] for s in segments).strip(),
            "language": getattr(info, "language", ""),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "segments": segments,
        }


class AnalysisClient(Protocol):
    def analyze(self, prompt: str) -> dict:
        ...


class ClaudeAnalysisClient:
    def __init__(self, api_key: str, model: str = DEFAULT_CLAUDE_MODEL):
        self.client = create_claude_client(api_key)
        self.model = model or DEFAULT_CLAUDE_MODEL

    def analyze(self, prompt: str) -> dict:
        result = claude_json(self.client, prompt, model=self.model)
        if not isinstance(result, dict):
            raise ValueError("Claude analysis response must be a JSON object")
        return result


def write_reference_stt_outputs(project_dir: Path, raw_stt: dict) -> str:
    text = str(raw_stt.get("text", "")).strip()
    write_reference_raw_stt(project_dir, raw_stt)
    write_reference_script(project_dir, f"{text}\n")
    return text
