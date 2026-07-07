from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adaptive Bin Pool Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #111418; color: #e9ecef; }
    header { padding: 14px 18px; border-bottom: 1px solid #2b3036; display: flex; justify-content: space-between; gap: 16px; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px; padding: 14px; }
    .panel { background: #171b20; border: 1px solid #2b3036; border-radius: 8px; padding: 12px; }
    h1 { font-size: 18px; margin: 0; }
    h2 { font-size: 14px; margin: 0 0 8px; color: #ced4da; }
    canvas { width: 100%; height: 300px; image-rendering: pixelated; background: #0b0d10; border: 1px solid #343a40; border-radius: 4px; }
    .stack { display: grid; gap: 14px; }
    .meta, .hint { color: #adb5bd; font-size: 12px; }
    .details { white-space: pre-wrap; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.45; }
    .error { color: #ffa8a8; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Adaptive Bin Pool Viewer</h1>
      <div class="meta" id="summary">Waiting for snapshot...</div>
    </div>
    <div class="meta">Failure rate and access count are compressed global bucket views.</div>
  </header>
  <main>
    <div class="stack">
      <section class="panel">
        <h2>Failure Rate</h2>
        <canvas id="failure"></canvas>
      </section>
      <section class="panel">
        <h2>Access Count (log scale)</h2>
        <canvas id="access"></canvas>
      </section>
    </div>
    <aside class="panel">
      <h2>Hover Details</h2>
      <div class="details" id="details">Move over a heatmap cell.</div>
      <p class="hint">A cell is one motion-id bucket and one time bin. Paths are representative first/last paths for the bucket.</p>
    </aside>
  </main>
<script>
const state = { meta: null, access: null, failure: null, valid: null, lastKey: "" };
const summary = document.getElementById("summary");
const details = document.getElementById("details");
const failureCanvas = document.getElementById("failure");
const accessCanvas = document.getElementById("access");

function colorFailure(v, valid) {
  if (!valid) return [36, 40, 44, 255];
  const x = Math.max(0, Math.min(1, v));
  return [
    Math.floor(40 + 210 * x),
    Math.floor(180 - 120 * x),
    Math.floor(170 - 140 * x),
    255,
  ];
}

function colorAccess(v, vmax, valid) {
  if (!valid) return [36, 40, 44, 255];
  const x = vmax > 0 ? Math.log1p(v) / Math.log1p(vmax) : 0;
  return [
    Math.floor(30 + 60 * x),
    Math.floor(80 + 120 * x),
    Math.floor(120 + 120 * x),
    255,
  ];
}

function drawCanvas(canvas, mode) {
  const meta = state.meta;
  if (!meta || !state.access || !state.failure || !state.valid) return;
  const w = meta.bucket_count;
  const h = meta.bin_count;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  let maxAccess = 0;
  for (let i = 0; i < state.access.length; i++) maxAccess = Math.max(maxAccess, state.access[i]);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const src = x * h + y;
      const dst = ((h - 1 - y) * w + x) * 4;
      const valid = state.valid[src] > 0;
      const access = state.access[src];
      const failure = state.failure[src];
      const color = mode === "failure"
        ? colorFailure(access > 0 ? failure / access : 0, valid)
        : colorAccess(access, maxAccess, valid);
      image.data[dst] = color[0];
      image.data[dst + 1] = color[1];
      image.data[dst + 2] = color[2];
      image.data[dst + 3] = color[3];
    }
  }
  ctx.putImageData(image, 0, 0);
}

async function fetchBinary(name, type, key) {
  const res = await fetch(`${name}?v=${key}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`failed to fetch ${name}: ${res.status}`);
  return new type(await res.arrayBuffer());
}

async function refresh() {
  try {
    const res = await fetch(`latest.json?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error("latest.json not found");
    const meta = await res.json();
    const key = `${meta.iteration}-${meta.updated_at_unix}`;
    if (key === state.lastKey) return;
    state.meta = meta;
    state.access = await fetchBinary(meta.access_file, Float32Array, key);
    state.failure = await fetchBinary(meta.failure_file, Float32Array, key);
    state.valid = await fetchBinary(meta.valid_file, Int32Array, key);
    state.lastKey = key;
    drawCanvas(failureCanvas, "failure");
    drawCanvas(accessCanvas, "access");
    summary.textContent = `iteration ${meta.iteration} | motions ${meta.num_files} | buckets ${meta.bucket_count} | bins ${meta.bin_count}`;
    summary.className = "meta";
  } catch (err) {
    summary.textContent = String(err);
    summary.className = "meta error";
  }
}

function showHover(evt) {
  const meta = state.meta;
  if (!meta || !state.access || !state.failure || !state.valid) return;
  const rect = evt.currentTarget.getBoundingClientRect();
  const x = Math.max(0, Math.min(meta.bucket_count - 1, Math.floor((evt.clientX - rect.left) * meta.bucket_count / rect.width)));
  const drawY = Math.max(0, Math.min(meta.bin_count - 1, Math.floor((evt.clientY - rect.top) * meta.bin_count / rect.height)));
  const y = meta.bin_count - 1 - drawY;
  const idx = x * meta.bin_count + y;
  const access = state.access[idx];
  const failure = state.failure[idx];
  const rate = access > 0 ? failure / access : 0;
  const step0 = y * meta.bin_width_steps;
  const step1 = (y + 1) * meta.bin_width_steps;
  details.textContent =
    `bucket: ${x}\n` +
    `motion ids: [${meta.bucket_start_motion_ids[x]}, ${meta.bucket_end_motion_ids[x]})\n` +
    `bin: ${y}\n` +
    `step range: [${step0}, ${step1})\n` +
    `valid motions in cell: ${state.valid[idx]}\n` +
    `access count: ${access.toFixed(4)}\n` +
    `failure count: ${failure.toFixed(4)}\n` +
    `failure rate: ${rate.toFixed(4)}\n\n` +
    `first path:\n${meta.bucket_first_paths[x] || "(none)"}\n\n` +
    `last path:\n${meta.bucket_last_paths[x] || "(none)"}`;
}

failureCanvas.addEventListener("mousemove", showHover);
accessCanvas.addEventListener("mousemove", showHover);
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


class ViewerRequestHandler(SimpleHTTPRequestHandler):
  def do_GET(self) -> None:
    if self.path in {"/", "/index.html"}:
      body = INDEX_HTML.encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    super().do_GET()


def main() -> None:
  parser = argparse.ArgumentParser(description="Serve adaptive bin pool snapshots.")
  parser.add_argument("snapshot_dir", type=Path)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()

  snapshot_dir = args.snapshot_dir.expanduser().resolve()
  snapshot_dir.mkdir(parents=True, exist_ok=True)
  handler = functools.partial(ViewerRequestHandler, directory=str(snapshot_dir))
  server = ThreadingHTTPServer((args.host, args.port), handler)
  print(f"Adaptive bin viewer: http://{args.host}:{args.port}")
  print(f"Serving snapshots from: {snapshot_dir}")
  server.serve_forever()


if __name__ == "__main__":
  main()
