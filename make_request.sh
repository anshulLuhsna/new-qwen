
#!/usr/bin/env bash
set -euo pipefail

# Defaults
URL="http://127.0.0.1:8000/respond_to_text"
TEXT="Say something short and clear"
TIMEOUT=300
OUT_JSON="response.json"
OUT_WAV="reply.wav"

usage() {
  cat <<USAGE
Usage: $0 [--url URL] [--text TEXT] [--timeout SECS] [--out FILE.json] [--wav FILE.wav]

Examples:
  $0 --url http://127.0.0.1:8000/respond_to_text --text "Hello there" --timeout 300 --out /workspace/response.json --wav /workspace/reply.wav
  $0 --url https://<podid>-8000.proxy.runpod.net/respond_to_text --text "Short reply" --timeout 120
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --out) OUT_JSON="$2"; shift 2;;
    --wav) OUT_WAV="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "[make_request] URL: $URL"
echo "[make_request] TEXT: $TEXT"
echo "[make_request] TIMEOUT: ${TIMEOUT}s"
echo "[make_request] OUT_JSON: $OUT_JSON"
echo "[make_request] OUT_WAV:  $OUT_WAV"

# Call API
curl -s --max-time "${TIMEOUT}" -H "Content-Type: application/json"   -d "$(printf '{"text":%s}' "$(printf %s "$TEXT" | python -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")"   "$URL" > "$OUT_JSON"

# Extract audio -> WAV
if command -v jq >/dev/null 2>&1; then
  jq -r '.audio_base64' "$OUT_JSON" | base64 -d > "$OUT_WAV"
else
  # Fallback without jq (python stdlib)
  python - <<PY
import base64, json, sys, pathlib
j = json.load(open("$OUT_JSON","r"))
b64 = j.get("audio_base64","")
pathlib.Path("$OUT_WAV").write_bytes(base64.b64decode(b64))
print("Wrote: $OUT_WAV")
PY
fi

echo "[make_request] Done. JSON: $OUT_JSON  WAV: $OUT_WAV"
