"""Local results dashboard for the Right-of-Way project.

Run from the ev_recognition folder:

    python -m evr.dashboard

Then open http://127.0.0.1:8765 in a browser. The dashboard starts the existing
test scripts as subprocesses and shows their output, logs and the live annotated
video. It uses only the Python standard library and listens on localhost only.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path.cwd()
MEDIA_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".wav", ".mp3", ".flac"}
MAX_LOG_ROWS = 1500
MAX_OUTPUT_LINES = 400
FRAME_PATH = "logs/live_frame.jpg"


def read_config_summary(config_name: str = "config.yaml") -> dict:
    """Pick out the values worth showing. Needs PyYAML, which the project already uses."""
    try:
        import yaml
        data = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    fusion = data.get("fusion", {})
    return {
        "Emergency threshold": fusion.get("emergency_threshold"),
        "Clear threshold": fusion.get("clear_threshold"),
        "Frames needed to trigger": fusion.get("trigger_frames"),
        "Seconds below clear to resume": fusion.get("clear_seconds"),
        "Vision weight / audio weight": f"{fusion.get('visual_weight')} / {fusion.get('audio_weight')}",
        "Vision runs every N frames": data.get("vision_every_n_frames"),
        "Audio window (s)": data.get("audio_window_seconds"),
        "Controller": (data.get("controller") or {}).get("mode"),
    }


def read_model_info(config_name: str = "config.yaml") -> list[dict]:
    try:
        import yaml
        data = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    models = []
    for label, key in (("Vision model", "vision_model"), ("Siren model", "audio_model")):
        rel = data.get(key)
        if not rel:
            continue
        path = ROOT / rel
        if path.exists():
            info = path.stat()
            models.append({"name": label, "file": str(rel), "kb": round(info.st_size / 1024),
                           "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(info.st_mtime))})
        else:
            models.append({"name": label, "file": str(rel), "kb": None, "modified": "missing"})
    return models


class JobRunner:
    """Runs one test script at a time and keeps its console output."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.name = ""
        self.lines: list[str] = []
        self.returncode: int | None = None
        self.log_path = ""
        self.report_path = ""
        self.details: dict = {}
        self.started = 0.0
        self.finished = 0.0

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, name: str, command: list[str], log_path: str = "", report_path: str = "",
              inputs: dict | None = None, config_name: str = "config.yaml") -> bool:
        with self.lock:
            if self.running():
                return False
            self.name = name
            self.lines = []
            self.returncode = None
            self.log_path = log_path
            self.report_path = report_path
            self.started = time.time()
            self.finished = 0.0
            shown = ["python"] + command[1:]
            self.details = {
                "command": " ".join(shown),
                "inputs": inputs or {},
                "config": read_config_summary(config_name) if log_path else {},
                "models": read_model_info(config_name) if log_path else [],
            }
            frame = ROOT / FRAME_PATH
            if log_path and frame.exists():
                frame.unlink()
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._read, args=(self.process,), daemon=True).start()
            return True

    def _read(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\n"))
                if len(self.lines) > MAX_OUTPUT_LINES:
                    del self.lines[: len(self.lines) - MAX_OUTPUT_LINES]
        code = process.wait()
        with self.lock:
            self.returncode = code
            self.finished = time.time()

    def stop(self) -> None:
        with self.lock:
            if self.running():
                self.process.terminate()

    def status(self) -> dict:
        with self.lock:
            return {
                "running": self.running(),
                "job": self.name,
                "returncode": self.returncode,
                "output": "\n".join(self.lines),
                "log_path": self.log_path,
                "report_path": self.report_path,
                "details": self.details,
                "started": self.started,
                "finished": self.finished,
            }


RUNNER = JobRunner()


def inside_root(relative: str, allowed_dirs: tuple[str, ...]) -> Path | None:
    """Return a safe path under one of the allowed folders, or None."""
    candidate = (ROOT / relative).resolve()
    for folder in allowed_dirs:
        base = (ROOT / folder).resolve()
        if candidate == base or base in candidate.parents:
            return candidate
    return None


def list_media() -> dict:
    demo = ROOT / "demo"
    files = sorted(p.name for p in demo.glob("*") if p.suffix.lower() in MEDIA_SUFFIXES) if demo.exists() else []
    videos = [f for f in files if Path(f).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    audio = [f for f in files if Path(f).suffix.lower() in {".wav", ".mp3", ".flac"}]
    logs_dir = ROOT / "logs"
    logs = sorted((p.name for p in logs_dir.glob("*.csv")), reverse=True) if logs_dir.exists() else []
    return {"videos": videos, "audio": audio, "logs": logs, "esc50_ready": (ROOT / "demo" / "esc50_test").exists()}


def read_log(relative: str) -> dict:
    path = inside_root(relative, ("logs",))
    if path is None or not path.exists():
        return {"rows": [], "error": "Log not found"}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "t": float(row["timestamp"]),
                        "v": float(row["vision_score"]),
                        "a": float(row["audio_score"]),
                        "f": float(row["fused"]),
                        "state": row["state"],
                        "command": row["command"],
                        "side": row.get("side", ""),
                    }
                )
            except (KeyError, ValueError):
                continue
    if len(rows) > MAX_LOG_ROWS:
        step = len(rows) / MAX_LOG_ROWS
        rows = [rows[int(i * step)] for i in range(MAX_LOG_ROWS)]
    return {"rows": rows}


def read_report(relative: str) -> dict:
    path = inside_root(relative, ("reports",))
    if path is None or not path.exists():
        return {"columns": [], "rows": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return {"columns": reader.fieldnames or [], "rows": rows}


def build_job(payload: dict) -> tuple | None:
    job = payload.get("job")
    python = sys.executable
    if job == "demo":
        log_rel = "logs/dashboard_run.csv"
        config_name = payload.get("config") or "config.yaml"
        if config_name not in {"config.yaml", "config_esp32.yaml"}:
            return None
        command = [python, "-m", "evr.realtime", "--config", config_name, "--headless",
                   "--log-csv", log_rel, "--frame-out", FRAME_PATH]
        inputs = {"Rover output": "ESP32 over serial (drives the rover)" if config_name == "config_esp32.yaml"
                  else "print only (nothing is sent to hardware)"}
        video = payload.get("video") or ""
        audio = payload.get("audio") or ""
        if video:
            safe = inside_root(f"demo/{video}", ("demo",))
            if safe is None or not safe.exists():
                return None
            command += ["--video", f"demo/{video}"]
        inputs["Video"] = video or "camera"
        if audio:
            safe = inside_root(f"demo/{audio}", ("demo",))
            if safe is None or not safe.exists():
                return None
            command += ["--audio", f"demo/{audio}"]
        inputs["Audio"] = audio or "microphone"
        if payload.get("no_vision"):
            command.append("--no-vision")
            inputs["Vision"] = "off"
        if payload.get("no_audio"):
            command.append("--no-audio")
            inputs["Siren model"] = "off"
        if payload.get("boxes"):
            command.append("--boxes")
            inputs["Vehicle boxes"] = "on (general COCO detector)"
            if payload.get("box_classify"):
                command.append("--box-classify")
                inputs["Per-box emergency score"] = "on (experimental)"
        return "Live run", command, log_rel, "", inputs, config_name
    if job == "esc50":
        try:
            threshold = float(payload.get("threshold", 0.80))
        except (TypeError, ValueError):
            return None
        command = [python, "-m", "evr.evaluate_esc50", "--test-dir", "demo/esc50_test",
                   "--threshold", f"{threshold:.2f}", "--output", "reports/esc50_results.csv"]
        return ("Non-siren audio test (ESC-50 fold 5)", command, "", "reports/esc50_results.csv",
                {"Test clips": "demo/esc50_test", "Threshold": f"{threshold:.2f}"})
    if job == "sweep":
        return ("Audio threshold sweep",
                [python, "-m", "evr.sweep_audio_threshold", "--test-dir", "demo/esc50_test"], "", "",
                {"Test clips": "demo/esc50_test"})
    if job == "simulate":
        return ("Fusion simulation",
                [python, "-m", "evr.simulate_fusion_audio", "--test-dir", "demo/esc50_test"], "", "",
                {"Test clips": "demo/esc50_test"})
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # keep the terminal quiet
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        self._send(json.dumps(data).encode("utf-8"), "application/json", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/favicon.ico":
            self._send(b"", "image/x-icon", 204)
        elif parsed.path == "/api/files":
            self._json(list_media())
        elif parsed.path == "/api/status":
            self._json(RUNNER.status())
        elif parsed.path == "/api/log":
            self._json(read_log(query.get("path", [""])[0]))
        elif parsed.path == "/api/report":
            self._json(read_report(query.get("path", [""])[0]))
        elif parsed.path == "/api/frame":
            frame = ROOT / FRAME_PATH
            try:
                self._send(frame.read_bytes(), "image/jpeg")
            except OSError:
                self._send(b"", "text/plain", 404)
        else:
            self._send(b"Not found", "text/plain", 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            payload = {}
        if self.path == "/api/run":
            job = build_job(payload)
            if job is None:
                self._json({"ok": False, "error": "Invalid job or file"}, 400)
                return
            started = RUNNER.start(*job)
            self._json({"ok": started, "error": "" if started else "Another job is still running"})
        elif self.path == "/api/stop":
            RUNNER.stop()
            self._json({"ok": True})
        else:
            self._send(b"Not found", "text/plain", 404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dashboard for running tests and viewing results.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not (ROOT / "config.yaml").exists():
        raise SystemExit("Run this from the ev_recognition folder (config.yaml not found).")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard running at {url} (press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        RUNNER.stop()
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Right-of-Way Dashboard</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--mute:#5d6672;--line:#dfe3e8;--accent:#1d5fd1;
--vis:#1d5fd1;--aud:#d97a00;--fus:#1a1f26;--ok:#1b7f45;--warn:#b42318;
--s-normal:#d7f0e0;--s-slow:#fff1c2;--s-pull:#ffd9a8;--s-stop:#f8c9c4}
@media (prefers-color-scheme:dark){:root{--bg:#13161a;--card:#1c2026;--ink:#e8ebef;--mute:#9aa4b0;--line:#2c323a;--accent:#6da2ff;
--vis:#6da2ff;--aud:#f0a23a;--fus:#f1f3f5;--ok:#53c585;--warn:#ff8a7e;
--s-normal:#1d3a2a;--s-slow:#443b17;--s-pull:#4a3118;--s-stop:#4a2220}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 48px}
.top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}
h1{font-size:22px;margin:0 0 2px}
.sub{color:var(--mute);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.card h2{font-size:15px;margin:0 0 10px}
.grid2{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:16px}
@media (max-width:860px){.grid2{grid-template-columns:1fr}}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:end}
label{display:flex;flex-direction:column;font-size:12px;color:var(--mute);gap:3px}
select,input[type=number]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);min-width:150px}
.check{flex-direction:row;align-items:center;gap:6px;font-size:13px;color:var(--ink)}
button{font:inherit;padding:7px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.5;cursor:not-allowed}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile .k{font-size:12px;color:var(--mute)}
.tile .v{font-size:20px;font-weight:600}
canvas{width:100%;height:300px;display:block}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--mute);margin-top:6px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
.chip{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle;border:1px solid var(--line)}
pre{margin:0;padding:12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;max-height:320px;overflow:auto;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mute);font-weight:500}
.status{font-size:13px;color:var(--mute);margin-left:6px;font-weight:400}
.status b.run{color:var(--accent)}.status b.ok{color:var(--ok)}.status b.bad{color:var(--warn)}
.hint{font-size:12px;color:var(--mute);margin-top:8px}
.acc li{margin:4px 0;list-style:none}
.acc ul{padding:0;margin:0}
.live{position:relative;background:#0c0e11;border-radius:8px;overflow:hidden;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}
.live img{width:100%;height:100%;object-fit:contain;display:none}
.live .ph{color:#9aa4b0;font-size:13px;padding:12px;text-align:center}
.verdict{border-radius:8px;padding:12px 14px;margin-bottom:12px;font-size:15px;border:1px solid var(--line)}
.verdict.yes{background:var(--s-stop)}
.verdict.no{background:var(--s-normal)}
.verdict.none{color:var(--mute)}
.verdict b{display:block;font-size:16px;margin-bottom:2px}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0;font-size:13px}
dt{color:var(--mute)}
dd{margin:0;word-break:break-word}
code{font:12.5px ui-monospace,Menlo,Consolas,monospace;background:var(--bg);padding:2px 5px;border-radius:4px}
body.present .dev{display:none}
@media print{.noprint{display:none!important}.dev{display:none!important}body{background:#fff}.card{break-inside:avoid}}
</style>
</head>
<body>
<div class="wrap">
<div class="top">
  <div>
    <h1>Right-of-Way: Emergency Vehicle Recognition</h1>
    <p class="sub">Vision and siren detection fused into a yield decision for the rover. Runs on this computer, no hardware needed.</p>
  </div>
  <div class="row noprint">
    <button id="present">Presentation mode</button>
    <button id="print">Save report (PDF)</button>
  </div>
</div>

<div class="card noprint">
  <h2>Run</h2>
  <div class="row">
    <label>Video<select id="video"></select></label>
    <label>Audio<select id="audio"></select></label>
    <label>Output<select id="config"><option value="config.yaml">Print only (safe)</option><option value="config_esp32.yaml">Drive the rover (ESP32)</option></select></label>
    <label class="check"><input type="checkbox" id="boxes"> vehicle boxes</label>
    <label class="check"><input type="checkbox" id="boxclass"> score each box (experimental)</label>
    <label class="check"><input type="checkbox" id="novision"> audio only</label>
    <label class="check"><input type="checkbox" id="noaudio"> video only</label>
    <button class="primary" id="runDemo">Run demo</button>
    <button id="stop" disabled>Stop</button>
  </div>
  <div class="hint">Video "(camera)" and Audio "(microphone)" run live from the connected camera and mic. Choosing a file plays that file instead (files come from the demo/ folder). "Print only" sends nothing to hardware. Vehicle boxes need <code>pip install ultralytics</code> on this computer.</div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Live view</h2>
    <div class="live"><img id="live" alt="Annotated video"><div class="ph" id="liveph">Run the demo to see the annotated video here.</div></div>
    <div class="hint">The banner shows what the project's own models decide. Boxes come from a separate general detector that only finds cars, buses, trucks and motorcycles; the emergency decision does not come from the boxes.</div>
  </div>
  <div class="card">
    <h2>What happened</h2>
    <div class="verdict none" id="verdict">No run yet.</div>
    <dl id="details"></dl>
  </div>
</div>

<div class="card">
  <h2>Scores over time</h2>
  <div class="row noprint" style="margin-bottom:10px">
    <label>Or open a saved log<select id="logSelect"></select></label>
    <button id="loadLog">Load</button>
  </div>
  <div class="tiles" id="tiles"></div>
  <canvas id="chart"></canvas>
  <div class="legend">
    <span><i style="background:var(--vis)"></i>Vision score</span>
    <span><i style="background:var(--aud)"></i>Siren score</span>
    <span><i style="background:var(--fus)"></i>Fused confidence</span>
    <span>Dashed lines: emergency threshold 0.80, clear threshold 0.45</span>
    <span><span class="chip" style="background:var(--s-normal)"></span>NORMAL</span>
    <span><span class="chip" style="background:var(--s-slow)"></span>SLOW</span>
    <span><span class="chip" style="background:var(--s-pull)"></span>PULL_OVER</span>
    <span><span class="chip" style="background:var(--s-stop)"></span>STOPPED</span>
  </div>
</div>

<div class="card">
  <h2>Decision timeline</h2>
  <div id="timeline"><div class="hint">No run yet.</div></div>
</div>

<div class="card dev noprint">
  <h2>Audio tests on non-siren sounds (ESC-50 fold 5, never used in training)</h2>
  <div class="row">
    <label>Siren score threshold<input type="number" id="threshold" value="0.80" min="0.3" max="0.95" step="0.05"></label>
    <button class="primary" data-job="esc50">Run non-siren test</button>
    <button data-job="sweep">Threshold sweep</button>
    <button data-job="simulate">Fusion simulation</button>
  </div>
  <div class="hint" id="esc50hint"></div>
  <div id="reportBox" style="margin-top:12px"></div>
</div>

<div class="card dev">
  <h2>Console output <span class="status" id="status"></span></h2>
  <pre id="output">Nothing run yet.</pre>
</div>

<div class="card acc dev">
  <h2>Course acceptance checklist (tick when measured)</h2>
  <ul id="acc"></ul>
  <div class="hint">Ticks are remembered in this browser only. The hardware items need the rover.</div>
</div>
</div>

<script>
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let lastRows = [];
let lastStatus = null;
let wasRunning = false;
const esc = x => String(x).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

async function api(path, body){
  const r = await fetch(path, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : undefined);
  return r.json();
}
function fill(select, items, blank){
  select.innerHTML = "";
  if (blank){ const o=document.createElement("option"); o.value=""; o.textContent=blank; select.appendChild(o); }
  items.forEach(x=>{ const o=document.createElement("option"); o.value=x; o.textContent=x; select.appendChild(o); });
}
async function loadFiles(){
  const f = await api("/api/files");
  fill($("video"), f.videos, "(camera)");
  fill($("audio"), f.audio, "(microphone)");
  fill($("logSelect"), f.logs, f.logs.length ? null : "(no logs yet)");
  $("esc50hint").textContent = f.esc50_ready ? "" :
    "demo/esc50_test not found. Run: python prepare_esc50_split.py   to build the held-out test clips first.";
  if (f.videos.includes("traffic.mp4")) $("video").value = "traffic.mp4";
  if (f.audio.includes("siren.wav")) $("audio").value = "siren.wav";
}

function analyse(rows){
  if (!rows.length) return null;
  const t0 = rows[0].t;
  const first = rows.find(r=>r.state!=="NORMAL");
  const stop = rows.find(r=>r.state==="STOPPED");
  const back = stop ? rows.find(r=>r.t>stop.t && r.state==="NORMAL") : null;
  return {t0, dur: rows[rows.length-1].t - t0, maxF: Math.max(...rows.map(r=>r.f)),
          maxV: Math.max(...rows.map(r=>r.v)), maxA: Math.max(...rows.map(r=>r.a)), first, stop, back};
}

function summarize(rows){
  const a = analyse(rows);
  if (!a){ $("tiles").innerHTML = '<div class="hint">No rows yet.</div>'; return; }
  const fmt = r => r ? (r.t-a.t0).toFixed(1)+" s" : null;
  const tiles = [
    ["Duration", a.dur.toFixed(1)+" s"],
    ["Peak vision score", a.maxV.toFixed(2)],
    ["Peak siren score", a.maxA.toFixed(2)],
    ["Peak fused confidence", a.maxF.toFixed(2)],
    ["First reaction (SLOW)", fmt(a.first) || "none"],
    ["Reached STOPPED", fmt(a.stop) || "never"],
    ["Back to NORMAL", fmt(a.back) || (a.stop ? "not within run" : "n/a")],
  ];
  $("tiles").innerHTML = tiles.map(([k,v])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

function verdict(rows, running){
  const el = $("verdict"), a = analyse(rows);
  if (!a){ el.className="verdict none"; el.textContent = running ? "Starting..." : "No run yet."; return; }
  if (a.stop || a.first){
    const parts = [`Slowed down at ${(a.first.t-a.t0).toFixed(1)} s`];
    if (a.stop) parts.push(`stopped at ${(a.stop.t-a.t0).toFixed(1)} s`);
    if (a.back) parts.push(`resumed normal driving at ${(a.back.t-a.t0).toFixed(1)} s`);
    el.className = "verdict yes";
    el.innerHTML = `<b>Emergency vehicle detected${running ? " (run in progress)" : ""}</b>${parts.join(", ")}. Peak confidence ${a.maxF.toFixed(2)}.`;
  } else {
    el.className = "verdict no";
    el.innerHTML = `<b>No emergency detected${running ? " so far" : ""}</b>Peak fused confidence ${a.maxF.toFixed(2)}, below the 0.80 threshold. The rover kept driving normally.`;
  }
}

function details(s){
  const d = s && s.details; const box = $("details");
  if (!d || !d.command){ box.innerHTML = ""; return; }
  const items = [["Job", s.job], ["Command", `<code>${esc(d.command)}</code>`]];
  Object.entries(d.inputs||{}).forEach(([k,v])=>items.push([k, esc(v)]));
  (d.models||[]).forEach(m=>items.push([m.name, esc(m.file)+(m.kb!=null ? ` (${m.kb} KB, ${esc(m.modified)})` : " (missing)")]));
  Object.entries(d.config||{}).forEach(([k,v])=>{ if (v!=null) items.push([k, esc(v)]); });
  if (s.started){
    items.push(["Started", new Date(s.started*1000).toLocaleTimeString()]);
    const end = s.finished || (Date.now()/1000);
    items.push(["Wall time", (end - s.started).toFixed(1)+" s" + (s.running ? " (running)" : "")]);
  }
  if (!s.running && s.job) items.push(["Exit status", s.returncode===0 ? "completed" : `stopped or failed (code ${s.returncode})`]);
  if (/Vehicle boxes disabled/.test(s.output||"")) items.push(["Vehicle boxes", "disabled, see console output"]);
  box.innerHTML = items.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join("");
}

const STATE_TEXT = {SLOW:"Emergency confirmed, slowing down", PULL_OVER:"Pulling over to the right", STOPPED:"Stopped and waiting", NORMAL:"Cleared, resuming normal driving"};
function why(r){
  if (r.state==="NORMAL" || r.state==="PULL_OVER" || r.state==="STOPPED") return "";
  if (r.v>=0.8 && r.a>=0.8) return "Vision and siren agree";
  if (r.v>=0.8) return "Vision cue";
  if (r.a>=0.8) return "Siren cue";
  return "Combined score above threshold";
}
function timeline(rows){
  const a = analyse(rows), box = $("timeline");
  if (!a){ box.innerHTML = '<div class="hint">No run yet.</div>'; return; }
  const events = [{r:rows[0], from:null}];
  for (let i=1;i<rows.length;i++) if (rows[i].state !== rows[i-1].state) events.push({r:rows[i], from:rows[i-1].state});
  const body = events.map(e=>{
    const r = e.r;
    const label = e.from===null ? `Run starts in ${r.state}` : (STATE_TEXT[r.state] || r.state);
    return `<tr><td>${(r.t-a.t0).toFixed(1)} s</td><td><b>${r.state}</b><br><span class="status" style="margin:0">${label}</span></td>
      <td>${why(r)}</td><td>${r.v.toFixed(2)}</td><td>${r.a.toFixed(2)}</td><td>${r.f.toFixed(2)}</td><td>${esc(r.command)}</td></tr>`;
  }).join("");
  box.innerHTML = `<table><thead><tr><th>Time</th><th>State</th><th>Trigger</th><th>Vision</th><th>Siren</th><th>Fused</th><th>Command to rover</th></tr></thead><tbody>${body}</tbody></table>
    <div class="hint">One row each time the rover's state changes. Scores are the values at that moment.</div>`;
}

function draw(rows){
  const cv = $("chart"), dpr = window.devicePixelRatio||1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const g = cv.getContext("2d"); g.scale(dpr,dpr);
  g.clearRect(0,0,w,h);
  const L=40,R=10,T=10,B=34, pw=w-L-R, ph=h-T-B;
  const ink=css("--mute"), line=css("--line");
  g.font="11px system-ui"; g.fillStyle=ink; g.strokeStyle=line; g.lineWidth=1;
  for (let i=0;i<=5;i++){ const y=T+ph*(1-i/5); g.beginPath(); g.moveTo(L,y); g.lineTo(L+pw,y); g.stroke(); g.fillText((i/5).toFixed(1),6,y+4); }
  if (!rows.length){ g.fillText("No data yet. Run the demo or load a log.", L+10, T+20); return; }
  const t0=rows[0].t, span=Math.max(rows[rows.length-1].t-t0,0.001);
  const X=t=>L+pw*(t-t0)/span, Y=v=>T+ph*(1-Math.min(Math.max(v,0),1));
  const stateColor={NORMAL:css("--s-normal"),SLOW:css("--s-slow"),PULL_OVER:css("--s-pull"),STOPPED:css("--s-stop")};
  for (let i=0;i<rows.length;i++){
    const x1=X(rows[i].t), x2=i+1<rows.length ? X(rows[i+1].t) : L+pw;
    g.fillStyle=stateColor[rows[i].state]||line; g.fillRect(x1,T+ph+6,Math.max(x2-x1,1),12);
  }
  g.fillStyle=ink;
  const ticks=Math.min(8,Math.max(2,Math.floor(span)));
  for (let i=0;i<=ticks;i++){ const s=span*i/ticks; g.fillText(s.toFixed(0)+"s", X(t0+s)-8, h-4); }
  const dash=(v,c)=>{ g.setLineDash([5,4]); g.strokeStyle=c; g.beginPath(); g.moveTo(L,Y(v)); g.lineTo(L+pw,Y(v)); g.stroke(); g.setLineDash([]); };
  dash(0.80,ink); dash(0.45,ink);
  const series=(key,color,width)=>{ g.strokeStyle=color; g.lineWidth=width; g.beginPath();
    rows.forEach((r,i)=>{ const x=X(r.t), y=Y(r[key]); i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke(); };
  series("v",css("--vis"),1.5); series("a",css("--aud"),1.5); series("f",css("--fus"),2.5);
}

async function showLog(path, running){
  const d = await api("/api/log?path="+encodeURIComponent(path));
  lastRows = d.rows||[];
  summarize(lastRows); draw(lastRows); verdict(lastRows, running); timeline(lastRows);
}

function renderReport(rep){
  const box=$("reportBox");
  if (!rep.rows.length){ box.innerHTML=""; return; }
  const cols=rep.columns;
  box.innerHTML = "<table><thead><tr>"+cols.map(c=>`<th>${c.replace(/_/g," ")}</th>`).join("")+"</tr></thead><tbody>"+
    rep.rows.map(r=>"<tr>"+cols.map(c=>`<td>${esc(r[c])}</td>`).join("")+"</tr>").join("")+"</tbody></table>";
}

let frameBusy = false;
function refreshFrame(){
  if (frameBusy) return;
  frameBusy = true;
  const img = new Image();
  img.onload = ()=>{ $("live").src = img.src; $("live").style.display="block"; $("liveph").style.display="none"; frameBusy=false; };
  img.onerror = ()=>{ frameBusy=false; };
  img.src = "/api/frame?t="+Date.now();
}

async function poll(){
  const s = await api("/api/status");
  lastStatus = s;
  $("output").textContent = s.output || (s.job ? "Starting..." : "Nothing run yet.");
  $("output").scrollTop = $("output").scrollHeight;
  $("stop").disabled = !s.running;
  document.querySelectorAll("#runDemo,[data-job]").forEach(b=>b.disabled = s.running);
  let label = "";
  if (s.running) label = `<b class="run">Running: ${esc(s.job)}</b>`;
  else if (s.job) label = s.returncode===0 ? `<b class="ok">Finished: ${esc(s.job)}</b>` : `<b class="bad">Stopped or failed: ${esc(s.job)} (exit ${s.returncode})</b>`;
  $("status").innerHTML = label;
  details(s);
  if (s.running && s.log_path) await showLog(s.log_path, true);
  if (wasRunning && !s.running){
    await loadFiles();
    if (s.log_path){ await showLog(s.log_path, false); refreshFrame(); }
    if (s.report_path && s.returncode===0) renderReport(await api("/api/report?path="+encodeURIComponent(s.report_path)));
  }
  wasRunning = s.running;
}

async function run(body){
  const r = await api("/api/run", body);
  if (!r.ok) alert(r.error || "Could not start");
  poll();
}
$("runDemo").onclick = ()=>{
  if ($("config").value === "config_esp32.yaml" && !confirm("Drive the rover: this sends real drive commands to the ESP32 as soon as the run starts. Are the wheels raised or the area clear?")) return;
  run({job:"demo", config:$("config").value, video:$("video").value, audio:$("audio").value,
  no_vision:$("novision").checked, no_audio:$("noaudio").checked, boxes:$("boxes").checked, box_classify:$("boxclass").checked});
};
document.querySelectorAll("[data-job]").forEach(b=>b.onclick=()=>run({job:b.dataset.job, threshold:parseFloat($("threshold").value)}));
$("stop").onclick = ()=>api("/api/stop",{}).then(poll);
$("loadLog").onclick = ()=>{ const v=$("logSelect").value; if (v) showLog("logs/"+v, false); };
$("print").onclick = ()=>window.print();
$("present").onclick = ()=>{
  const on = document.body.classList.toggle("present");
  $("present").textContent = on ? "Show test panels" : "Presentation mode";
  try{ localStorage.setItem("evr_present", on ? "1" : "0"); }catch(e){}
};
window.addEventListener("resize", ()=>draw(lastRows));

const ACC = [
  "Vision accuracy at least 93% (and at least 80 held-out project-camera images)",
  "At least 6 FPS on the Raspberry Pi",
  "Siren detection at least 90% with the motors running",
  "At least 18 of 20 yield scenarios correct",
  "At most 1 false yield in 10 normal runs",
  "Rover resumes within 3 seconds after the vehicle passes"
];
function loadAcc(){
  let saved={}; try{ saved=JSON.parse(localStorage.getItem("evr_acc")||"{}"); }catch(e){}
  $("acc").innerHTML = ACC.map((t,i)=>`<li><label class="check"><input type="checkbox" data-i="${i}" ${saved[i]?"checked":""}> ${t}</label></li>`).join("");
  document.querySelectorAll("#acc input").forEach(c=>c.onchange=()=>{
    saved[c.dataset.i]=c.checked; try{ localStorage.setItem("evr_acc", JSON.stringify(saved)); }catch(e){}
  });
}
try{ if (localStorage.getItem("evr_present")==="1") $("present").click(); }catch(e){}
loadAcc(); loadFiles().then(()=>{ draw([]); summarize([]); });
setInterval(poll,1000);
setInterval(()=>{ if (lastStatus && lastStatus.running) refreshFrame(); },200);
poll(); refreshFrame();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()