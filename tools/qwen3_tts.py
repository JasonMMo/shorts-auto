"""Local Qwen3-TTS (Pinokio, Gradio) client. Standard library only.

Usage:
  python tools/qwen3_tts.py status
  python tools/qwen3_tts.py custom "안녕하세요" --speaker Sohee [--instruct "밝게"] [-o out.wav]
  python tools/qwen3_tts.py clone  "대본" [--ref voice.wav --ref-text "레퍼런스 대사"] [-o out.wav]
  python tools/qwen3_tts.py design "대본" --desc "차분한 30대 여성 내레이터" [-o out.wav]

Base URL / reference voice come from tools/.env (QWEN3_TTS_BASE_URL, REFERENCE_VOICE_PATH).
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"]
LANGS = ["Auto", "Korean", "English", "Japanese", "Chinese", "French", "German",
         "Spanish", "Portuguese", "Russian", "Italian"]
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def load_env():
    env = {}
    p = ROOT / "tools" / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = load_env()
BASE = ENV.get("QWEN3_TTS_BASE_URL", "http://127.0.0.1:7860").rstrip("/")


def call(api, data, timeout=900):
    """Run a Gradio endpoint via the REST queue API and return its output list."""
    req = urllib.request.Request(
        f"{BASE}/gradio_api/call/{api}",
        data=json.dumps({"data": data}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        event_id = json.load(r)["event_id"]
    event = None
    with urllib.request.urlopen(f"{BASE}/gradio_api/call/{api}/{event_id}", timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event in ("complete", "error"):
                payload = line[5:].strip()
                if event == "error":
                    raise RuntimeError(f"{api} failed: {payload}")
                return json.loads(payload)
    raise RuntimeError(f"{api}: stream ended without result")


def upload(path):
    """Upload a local file to the Gradio server; returns a FileData dict."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
        f"filename=\"{Path(path).name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    ).encode() + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/gradio_api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        server_path = json.load(r)[0]
    return {"path": server_path, "meta": {"_type": "gradio.FileData"}}


def ensure_loaded(model_type, size):
    """Load the model explicitly before generating (generating with nothing loaded
    has crashed the server on this CPU-only machine)."""
    if f"{model_type} ({size})" in call("get_loaded_models_status", [])[0]:
        return
    print(f"loading {model_type} {size}...")
    print(call("load_model_manual", [model_type, size])[0])


def save_audio(result, out):
    audio, status = result[0], result[1] if len(result) > 1 else ""
    if not audio:
        raise RuntimeError(f"no audio returned: {status}")
    out = Path(out) if out else ROOT / "outputs" / "tts" / f"tts_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = audio.get("url") or f"{BASE}/gradio_api/file={audio['path']}"
    with urllib.request.urlopen(src, timeout=120) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"saved: {out}")
    if status:
        print(f"status: {status}")
    return out


def resolve_ref(ref):
    p = Path(ref or ENV.get("REFERENCE_VOICE_PATH", ""))
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXT)
        if not files:
            sys.exit(f"no reference audio in {p}")
        p = files[0]
    elif not p.exists():
        cands = [p.with_suffix(e) for e in AUDIO_EXT if p.with_suffix(e).exists()]
        if not cands:
            sys.exit(f"reference audio not found: {p}")
        p = cands[0]
    return p


def main():
    ap = argparse.ArgumentParser(description="Local Qwen3-TTS client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")

    def common(sp):
        sp.add_argument("text")
        sp.add_argument("--lang", default="Korean", choices=LANGS)
        sp.add_argument("--seed", type=int, default=-1)
        sp.add_argument("-o", "--out")

    c = sub.add_parser("custom", help="preset speaker")
    common(c)
    c.add_argument("--speaker", default="Sohee", choices=SPEAKERS)
    c.add_argument("--instruct", default="", help="style instruction")
    c.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B"])

    k = sub.add_parser("clone", help="clone reference voice")
    common(k)
    k.add_argument("--ref", help="reference audio (default: REFERENCE_VOICE_PATH)")
    k.add_argument("--ref-text", default="", help="transcript of reference audio")
    k.add_argument("--xvector-only", action="store_true")
    k.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B"])
    k.add_argument("--chunk", type=float, default=200)

    d = sub.add_parser("design", help="voice from description")
    common(d)
    d.add_argument("--desc", required=True, help="voice description")

    a = ap.parse_args()
    if a.cmd == "status":
        print(call("get_downloaded_models_status", [])[0])
        print(call("get_loaded_models_status", [])[0])
    elif a.cmd == "custom":
        ensure_loaded("CustomVoice", a.size)
        save_audio(call("generate_custom_voice",
                        [a.text, a.lang, a.speaker, a.instruct, a.size, a.seed]), a.out)
    elif a.cmd == "clone":
        ref = upload(resolve_ref(a.ref))
        ensure_loaded("Base", a.size)
        save_audio(call("generate_voice_clone",
                        [ref, a.ref_text, a.text, a.lang, a.xvector_only or not a.ref_text,
                         a.size, a.chunk, 0, a.seed]), a.out)
    elif a.cmd == "design":
        ensure_loaded("VoiceDesign", "1.7B")
        save_audio(call("generate_voice_design", [a.text, a.lang, a.desc, a.seed]), a.out)


if __name__ == "__main__":
    main()
