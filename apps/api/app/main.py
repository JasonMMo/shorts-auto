from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows 콘솔/리다이렉트 기본 인코딩(cp949)은 로그의 이모지(🎬 등)를 인코딩하지 못해
# print 한 줄 때문에 요청 전체가 UnicodeEncodeError로 실패한다. 출력을 UTF-8로 고정한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# `uv run`은 PYTHONHOME을 자신의 파이썬으로 지정한다. 서버가 띄우는 다른 파이썬
# 기반 CLI(yt-dlp 등)가 이를 상속하면 표준 라이브러리 버전이 어긋나 즉시 죽으므로,
# uv가 넣은 값일 때만 제거한다. 이미 실행 중인 서버 인터프리터에는 영향이 없다.
if os.environ.get("PYTHONHOME") and os.environ.get("PYTHONHOME") == os.environ.get(
    "UV_INTERNAL__PYTHONHOME"
):
    del os.environ["PYTHONHOME"]

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.paths import find_repo_root
from app.core.settings import load_settings
from app.jobs.router import create_jobs_router
from app.projects.router import create_projects_router
from app.references.audio import extract_reference_audio
from app.references.ai_clients import ClaudeAnalysisClient, LocalWhisperTranscriptionClient
from app.references.router import create_references_router
from app.production.router import create_production_router
from app.references.youtube import YouTubeClient
from app.scripts.router import create_scripts_router


def create_app(
    repo_root: Path | None = None,
    youtube_client_factory=YouTubeClient,
    audio_extractor=extract_reference_audio,
    transcription_client_factory=LocalWhisperTranscriptionClient,
    analysis_client_factory=ClaudeAnalysisClient,
) -> FastAPI:
    root = repo_root or find_repo_root()
    settings = load_settings(root)
    outputs_dir = root / settings.output_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Shorts Auto API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def outputs_dir_provider() -> Path:
        return outputs_dir

    def settings_provider():
        return settings

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/settings/status")
    def settings_status():
        return {
            "claude_api_key": bool(settings.claude_api_key),
            "youtube_api_key": bool(settings.youtube_api_key),
            "higgsfield_api_key": bool(settings.higgsfield_api_key),
            "qwen3_tts_base_url": settings.qwen3_tts_base_url,
            "reference_voice_path": settings.reference_voice_path,
            "reference_voice_text": bool(settings.reference_voice_text),
            "bgm_path": settings.bgm_path,
            "bgm_volume": settings.bgm_volume,
            "output_dir": settings.output_dir,
        }

    app.include_router(create_projects_router(outputs_dir_provider))
    app.include_router(create_jobs_router(outputs_dir_provider))
    app.include_router(
        create_references_router(
            outputs_dir_provider,
            settings_provider,
            youtube_client_factory,
            audio_extractor,
            transcription_client_factory,
            analysis_client_factory,
        )
    )
    app.include_router(create_scripts_router(outputs_dir_provider, settings_provider, analysis_client_factory))
    app.include_router(create_production_router(outputs_dir_provider, settings_provider))
    return app


app = create_app()
