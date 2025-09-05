
# RUNPOD_QWEN_OMNI_SETUP.md

This runbook captures exactly how we stood up **Qwen 2.5‑Omni 7B** on a Runpod workspace with a FastAPI server, local model snapshot, short English replies, and CLI testing (including saving audio).

---

## 1) Persistent caches and model location

Add these to your shell init (or export per session) **inside the pod**:

```bash
export HF_HOME=/workspace/.hf-cache
export HF_HUB_CACHE=/workspace/.hf-cache/hub
export TRANSFORMERS_CACHE=/workspace/.hf-cache/transformers   # optional; HF_HOME is preferred
```

## 2) Install core deps (do NOT reinstall torch)

```bash
# (optional) faster hub downloads
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Transformers preview that includes Omni
pip uninstall -y transformers
pip install "git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"

# Acceleration + Qwen helpers + audio stack
pip install -U accelerate "qwen-omni-utils[decord]" soundfile librosa pydub ffmpeg-python

# System libs for audio/video when needed
apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1
```

## 3) Pre‑download the model into /workspace (no network at import)

```bash
mkdir -p /workspace/models/Qwen2.5-Omni-7B

huggingface-cli download Qwen/Qwen2.5-Omni-7B   --local-dir /workspace/models/Qwen2.5-Omni-7B   --local-dir-use-symlinks False
```

## 4) App notes (voice_agent.py / main.py)

- Load from the local directory (e.g., `/workspace/models/Qwen2.5-Omni-7B`).
- Use a concise **system prompt** to force English + brevity:
  ```python
  self.system_prompt = {
      "role": "system",
      "content": "You are Qwen. Always reply in English. Keep answers under one sentence. Generate both text and speech."
  }
  ```
- For **text-only** requests, build tensors directly with the processor and call `model.generate(**inputs)`.
- Keep generation short & non-repetitive (on supported builds you may prefix with `thinker_` to scope to text):
  ```python
  text_ids, audio = self.model.generate(
      **inputs,
      max_new_tokens=48,
      no_repeat_ngram_size=3,
      repetition_penalty=1.1,
      temperature=0.7,
      top_p=0.8,
  )
  reply = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
  ```
- In FastAPI, create the agent in a **startup** hook (not at import).
- Return a JSON with `text` and `audio_base64` (24 kHz WAV).

## 5) Start the server

```bash
cd /workspace
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> Keep a **single worker** so the large model is only loaded once.

## 6) Test from **inside the pod** (bypasses proxy timeout)

```bash
# Hit endpoint and save full JSON (give it up to 300s)
curl -s --max-time 300   -H "Content-Type: application/json"   -d '{"text":"Say something short and clear"}'   http://127.0.0.1:8000/respond_to_text > /workspace/response.json

# Extract audio and save to WAV
jq -r '.audio_base64' /workspace/response.json | base64 -d > /workspace/reply.wav
```

Play (if tools installed): `aplay /workspace/reply.wav` or `ffplay -autoexit /workspace/reply.wav`

## 7) Test from your laptop

- Runpod proxy URL works for quick tests but can 524 if **time‑to‑first‑byte > ~100s**.
- Prefer an **SSH tunnel** for long calls, or keep responses short with `max_new_tokens` etc.

## 8) Handy one‑shot helper (included below)

Use `make_request.sh` (created alongside this runbook) to hit the pod locally or through the proxy, save `response.json`, and decode `reply.wav` automatically.

```bash
# inside the pod (localhost) or on your laptop (proxy/tunnel)
bash make_request.sh   --url http://127.0.0.1:8000/respond_to_text   --text "Say something short and clear"   --timeout 300   --out /workspace/response.json   --wav /workspace/reply.wav
```

---

## Appendix: Quick references

- curl timeouts (`--max-time`, `--connect-timeout`)
- jq raw extraction (`-r`)
- base64 decode (`-d/--decode`)
- HF Hub CLI `download --local-dir`
- HF cache env vars (`HF_HOME`, `HF_HUB_CACHE`)
- Runpod port exposure & proxy notes
- Uvicorn workers note

Save this file somewhere persistent (e.g., `/workspace/RUNPOD_QWEN_OMNI_SETUP.md`).
