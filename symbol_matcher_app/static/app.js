const ridSelect = document.getElementById("rid-select");
const widInput = document.getElementById("wid-input");
const widList = document.getElementById("wid-list");
const loadBtn = document.getElementById("load-btn");
const rematchBtn = document.getElementById("rematch-btn");
const clearBtn = document.getElementById("clear-btn");
const selectBtn = document.getElementById("select-btn");
const samBtn = document.getElementById("sam-btn");
const hqsamBtn = document.getElementById("hqsam-btn");
const sam2Btn = document.getElementById("sam2-btn");
const pcToggle = document.getElementById("pc-toggle");
const sharpenToggle = document.getElementById("sharpen-toggle");
const thrSlider = document.getElementById("thr-slider");
const thrVal = document.getElementById("thr-val");
const statusEl = document.getElementById("status");
const img = document.getElementById("worksheet-img");
const overlay = document.getElementById("overlay");
const wrap = document.getElementById("stage-wrap");
const stage = document.getElementById("stage");
const zoomGroup = document.getElementById("zoom-group");
const zoomVal = document.getElementById("zoom-val");
const layersPanel = document.getElementById("layers");

const SVGNS = "http://www.w3.org/2000/svg";
// Distinct color per selection group (cycles if more are added).
const PALETTE = [
  "#2ecc71", // green
  "#e74c3c", // red
  "#3498db", // blue
  "#e67e22", // orange
  "#9b59b6", // purple
  "#1abc9c", // teal
  "#f1c40f", // yellow
  "#ff6bcb", // pink
];
const SAM_COLORS = { fastsam: "#00e5ff", hqsam: "#ff3df0", sam2: "#ffb300" }; // cyan / magenta / amber
const POINT_COLOR = "#ff1744"; // ground-truth reference points (red crosshair)
const PATCH_COLOR = "#76ff03"; // ground-truth polygon patches (green box)
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 12;

let state = {
  rid: null,
  wid: null,
  natW: 0, // natural (original) image width in px
  natH: 0,
  baseWidth: 0, // displayed width (px) at zoom = 1 (fit-to-view)
  zoom: 1,
  // Each entry: {selection:{x,y,w,h}, template:{x,y,w,h}, matches:[], color}
  selections: [],
  sam: { fastsam: [], hqsam: [], sam2: [] }, // SAM boxes per model from reference points
  refPoints: [], // ground-truth reference points (electrical Point features)
  refPolygons: [], // ground-truth polygon patches (electrical Polygon features)
  layers: { points: true, patches: true, matches: true, fastsam: true, hqsam: true, sam2: true }, // visibility toggles
  wids: new Set(), // valid wids for the current request
};

function colorForIndex(i) {
  return PALETTE[i % PALETTE.length];
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.style.color = isError ? "#ff6b6b" : "var(--accent)";
}

// ---- zoom -----------------------------------------------------------------
function displayedWidth() {
  return state.baseWidth * state.zoom;
}

function applyZoom() {
  stage.style.width = `${displayedWidth()}px`;
  // Show zoom relative to actual pixels (1:1 = displayedWidth == natW).
  const pct = state.natW ? Math.round((displayedWidth() / state.natW) * 100) : 100;
  zoomVal.textContent = `${pct}%`;
}

function setZoom(z, anchorClientX, anchorClientY) {
  const old = displayedWidth();
  state.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
  const next = displayedWidth();
  if (old <= 0) {
    applyZoom();
    return;
  }
  const r = next / old;
  // Keep the point under the cursor (or viewport center) stationary.
  const rect = wrap.getBoundingClientRect();
  const ax = anchorClientX == null ? rect.width / 2 : anchorClientX - rect.left;
  const ay = anchorClientY == null ? rect.height / 2 : anchorClientY - rect.top;
  const contentX = wrap.scrollLeft + ax;
  const contentY = wrap.scrollTop + ay;
  applyZoom();
  wrap.scrollLeft = contentX * r - ax;
  wrap.scrollTop = contentY * r - ay;
}

function fitZoom() {
  setZoom(1);
  wrap.scrollLeft = 0;
  wrap.scrollTop = 0;
}

function initZoomForImage() {
  // Fit the image width to the viewport (never upscale beyond natural width).
  const avail = wrap.clientWidth - 2;
  state.baseWidth = Math.min(avail, state.natW) || avail;
  state.zoom = 1;
  applyZoom();
}

// ---- displayed <-> natural pixel conversion -------------------------------
function dispScale() {
  // natural px per displayed px
  const dw = img.clientWidth;
  if (!dw) return 1;
  return state.natW / dw;
}

function clientToNat(clientX, clientY) {
  const rect = overlay.getBoundingClientRect();
  const s = dispScale();
  return {
    x: Math.round((clientX - rect.left) * s),
    y: Math.round((clientY - rect.top) * s),
  };
}

function syncOverlaySize() {
  overlay.setAttribute("viewBox", `0 0 ${state.natW} ${state.natH}`);
  overlay.setAttribute("preserveAspectRatio", "none");
}

// ---- API ------------------------------------------------------------------
async function loadRequests() {
  setStatus("Loading requests…");
  const r = await fetch("/api/requests");
  const data = await r.json();
  ridSelect.innerHTML = "";
  if (!data.requests.length) {
    ridSelect.innerHTML = '<option value="">No cached requests found</option>';
    setStatus("No requests in data/requests/", true);
    return;
  }
  ridSelect.appendChild(new Option("— select request —", ""));
  data.requests.forEach((rid) => ridSelect.appendChild(new Option(rid, rid)));
  setStatus("");
}

async function loadWorksheets(rid) {
  widList.innerHTML = "";
  widInput.value = "";
  widInput.disabled = true;
  loadBtn.disabled = true;
  state.wid = null;
  state.wids = new Set();
  const r = await fetch(`/api/requests/${rid}/worksheets`);
  const data = await r.json();
  if (!data.worksheets.length) {
    widInput.placeholder = "No worksheets with images";
    return;
  }
  // Option value = wid (shown + searchable); label carries page/geometry info.
  data.worksheets.forEach((w) => {
    const pg = w.page_no != null ? `p${w.page_no} · ` : "";
    const flag = w.has_geometry ? "● " : "○ "; // ● = has geometries
    const label = `${flag}${pg}${w.name || "(no name)"}${w.title ? " — " + w.title : ""}`;
    const opt = document.createElement("option");
    opt.value = w.wid;
    opt.label = label; // datalist shows this next to the wid
    opt.textContent = label;
    widList.appendChild(opt);
    state.wids.add(w.wid);
  });
  const withGeom = data.worksheets.filter((w) => w.has_geometry).length;
  widInput.placeholder = `search ${data.worksheets.length} wids (${withGeom} ● with geometries)…`;
  widInput.disabled = false;
}

function resolveWid() {
  const v = widInput.value.trim();
  state.wid = state.wids.has(v) ? v : null;
  loadBtn.disabled = !state.wid;
}

async function loadImage() {
  clearOverlay();
  state.selections = [];
  state.sam = { fastsam: [], hqsam: [], sam2: [] };
  state.refPoints = [];
  state.refPolygons = [];
  rematchBtn.disabled = true;
  clearBtn.disabled = true;
  setStatus("Downloading worksheet image (first load may take a few seconds)…");

  const { rid, wid } = state;
  try {
    const metaR = await fetch(`/api/worksheet/${rid}/${wid}/meta`);
    if (!metaR.ok) throw new Error((await metaR.json()).detail || metaR.statusText);
    const meta = await metaR.json();
    state.natW = meta.width;
    state.natH = meta.height;
  } catch (e) {
    setStatus(`Failed to load image: ${e.message}`, true);
    return;
  }

  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("image element failed to load"));
    img.src = `/api/worksheet/${rid}/${wid}/image?t=${Date.now()}`;
  });
  syncOverlaySize();
  initZoomForImage();
  zoomGroup.hidden = false;
  selectBtn.disabled = false;
  samBtn.disabled = false;
  hqsamBtn.disabled = false;
  sam2Btn.disabled = false;
  pcToggle.disabled = false;
  sharpenToggle.disabled = false;
  layersPanel.hidden = false;
  updateLayerCounts();
  setTool("pan");
  loadRefPoints();
  loadRefPolygons();
  setStatus(`Loaded ${state.natW}×${state.natH}px. Drag to pan; click "Select symbol" to pick one.`);
}

async function matchOne(sel, threshold) {
  const r = await fetch(`/api/worksheet/${state.rid}/${state.wid}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...sel.selection, threshold }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  const data = await r.json();
  sel.matches = data.matches;
  sel.template = data.template;
  return data.count;
}

function summary() {
  return state.selections
    .map((s, i) => `#${i + 1} ${s.matches.length}`)
    .join("  ·  ");
}

// Match just the newest selection (called right after drawing a box).
async function runMatchLatest() {
  const sel = state.selections[state.selections.length - 1];
  if (!sel) return;
  const threshold = parseFloat(thrSlider.value);
  setStatus("Matching…");
  rematchBtn.disabled = true;
  try {
    const count = await matchOne(sel, threshold);
    redrawAll();
    setStatus(`Selection #${state.selections.length}: ${count} match${count === 1 ? "" : "es"} (threshold ${threshold.toFixed(2)}).  Totals — ${summary()}`);
  } catch (e) {
    setStatus(`Match failed: ${e.message}`, true);
  } finally {
    rematchBtn.disabled = false;
  }
}

// Re-run every selection (e.g. after changing the threshold).
async function rematchAll() {
  if (!state.selections.length) return;
  const threshold = parseFloat(thrSlider.value);
  setStatus("Re-matching all selections…");
  rematchBtn.disabled = true;
  try {
    for (const sel of state.selections) {
      await matchOne(sel, threshold);
      redrawAll();
    }
    setStatus(`Re-matched ${state.selections.length} selection(s) at threshold ${threshold.toFixed(2)}.  Totals — ${summary()}`);
  } catch (e) {
    setStatus(`Match failed: ${e.message}`, true);
  } finally {
    rematchBtn.disabled = false;
  }
}

// ---- overlay drawing ------------------------------------------------------
function clearOverlay() {
  while (overlay.firstChild) overlay.removeChild(overlay.firstChild);
}

function addRect(x, y, w, h, color, dashed = false) {
  const rect = document.createElementNS(SVGNS, "rect");
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  rect.setAttribute("width", w);
  rect.setAttribute("height", h);
  rect.setAttribute("fill", "none"); // transparent — outline only
  rect.setAttribute("stroke", color);
  rect.setAttribute("stroke-width", dashed ? 2.5 : 2);
  if (dashed) rect.setAttribute("stroke-dasharray", "6 4");
  rect.setAttribute("vector-effect", "non-scaling-stroke"); // constant px at any zoom
  overlay.appendChild(rect);
  return rect;
}

// A small crosshair marker at (x, y) in image pixels.
function addPoint(x, y, color) {
  const L = 5; // half-length in image units; stroke stays constant px via vector-effect
  const p = document.createElementNS(SVGNS, "path");
  p.setAttribute("d", `M ${x - L} ${y} H ${x + L} M ${x} ${y - L} V ${y + L}`);
  p.setAttribute("stroke", color);
  p.setAttribute("stroke-width", 1.5);
  p.setAttribute("vector-effect", "non-scaling-stroke");
  overlay.appendChild(p);
}

// Redraw every visible layer: selection matches + each SAM model's boxes.
function redrawAll(extra) {
  clearOverlay();
  Object.entries(state.sam).forEach(([model, boxes]) => {
    if (!state.layers[model]) return;
    boxes.forEach((b) => addRect(b.x, b.y, b.w, b.h, SAM_COLORS[model]));
  });
  if (state.layers.matches) {
    state.selections.forEach((sel) => {
      sel.matches.forEach((m) => addRect(m.x, m.y, m.w, m.h, sel.color));
      const t = sel.template || sel.selection;
      addRect(t.x, t.y, t.w, t.h, sel.color, true); // template box (dashed)
    });
  }
  if (state.layers.patches) {
    state.refPolygons.forEach((p) => addRect(p.x, p.y, p.w, p.h, PATCH_COLOR));
  }
  if (state.layers.points) {
    state.refPoints.forEach((p) => addPoint(p.x, p.y, POINT_COLOR));
  }
  if (extra) addRect(extra.x, extra.y, extra.w, extra.h, extra.color, true);
  updateLayerCounts();
}

// Update the per-layer box counts shown in the Layers panel.
function updateLayerCounts() {
  const m = state.selections.reduce((n, s) => n + s.matches.length, 0);
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  };
  set("cnt-points", state.refPoints.length);
  set("cnt-patches", state.refPolygons.length);
  set("cnt-matches", m);
  set("cnt-fastsam", state.sam.fastsam.length);
  set("cnt-hqsam", state.sam.hqsam.length);
  set("cnt-sam2", state.sam.sam2.length);
}

// Fetch the ground-truth reference points and show them as their own layer.
async function loadRefPoints() {
  try {
    const r = await fetch(`/api/worksheet/${state.rid}/${state.wid}/ref_points`);
    if (!r.ok) return;
    const data = await r.json();
    state.refPoints = data.points || [];
    redrawAll();
  } catch (e) {
    /* non-fatal: the points layer just stays empty */
  }
}

// Fetch the ground-truth polygon patches (electrical Polygons) as a layer.
async function loadRefPolygons() {
  try {
    const r = await fetch(`/api/worksheet/${state.rid}/${state.wid}/ref_polygons?electrical=0`);
    if (!r.ok) return;
    const data = await r.json();
    state.refPolygons = data.polygons || [];
    redrawAll();
  } catch (e) {
    /* non-fatal: the patches layer just stays empty */
  }
}

// Segment symbols from the worksheet's reference points with the chosen SAM model.
const SAM_LABEL = { fastsam: "FastSAM (cyan)", hqsam: "HQ-SAM (magenta)", sam2: "SAM 2.1 (amber)" };
const SAM_BTN = { fastsam: () => samBtn, hqsam: () => hqsamBtn, sam2: () => sam2Btn };
async function runSam(model) {
  const btn = SAM_BTN[model]();
  const pc = pcToggle.checked;
  const sharp = sharpenToggle.checked;
  const tag = [pc && "pc", sharp && "sharpen"].filter(Boolean).join("+");
  const preTag = tag ? ` (${tag})` : "";
  setStatus(`Running ${SAM_LABEL[model]}${preTag} on reference points… (first run loads the model)`);
  btn.disabled = true;
  try {
    const r = await fetch(
      `/api/worksheet/${state.rid}/${state.wid}/sam_points?model=${model}` +
        `${pc ? "&pseudocolor=1" : ""}${sharp ? "&sharpen=1" : ""}`
    );
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    state.sam[model] = data.boxes;
    redrawAll();
    clearBtn.disabled = false;
    if (!data.total_points) {
      setStatus("No reference points found in worksheet geometries for this sheet.", true);
    } else {
      setStatus(`${SAM_LABEL[model]}${preTag}: ${data.count} boxes from ${data.used_points} reference points.`);
    }
  } catch (e) {
    setStatus(`SAM failed: ${e.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

// ---- pointer: drag-select vs pan ------------------------------------------
let mode = null; // active drag: "select" | "pan"
let tool = "pan"; // armed tool for a plain left-drag: "pan" | "select"
let dragStart = null; // {clientX, clientY, scrollLeft, scrollTop}
let spaceHeld = false;

function setTool(t) {
  tool = t;
  selectBtn.classList.toggle("active", t === "select");
  overlay.classList.toggle("selecting", t === "select");
}

function isPanIntent(e) {
  // Middle/right mouse or Space always pans, regardless of the armed tool.
  return spaceHeld || e.button === 1 || e.button === 2;
}

overlay.addEventListener("contextmenu", (e) => {
  // Allow right-drag panning without the context menu popping up.
  if (state.natW) e.preventDefault();
});

overlay.addEventListener("mousedown", (e) => {
  if (!state.natW) return;
  e.preventDefault();
  const wantSelect = tool === "select" && e.button === 0 && !isPanIntent(e);
  if (!wantSelect) {
    mode = "pan";
    overlay.classList.add("panning");
    dragStart = {
      clientX: e.clientX,
      clientY: e.clientY,
      scrollLeft: wrap.scrollLeft,
      scrollTop: wrap.scrollTop,
    };
    return;
  }
  mode = "select";
  dragStart = { clientX: e.clientX, clientY: e.clientY };
  // keep existing match boxes visible until a new box is drawn
});

window.addEventListener("mousemove", (e) => {
  if (!mode) return;
  if (mode === "pan") {
    wrap.scrollLeft = dragStart.scrollLeft - (e.clientX - dragStart.clientX);
    wrap.scrollTop = dragStart.scrollTop - (e.clientY - dragStart.clientY);
    return;
  }
  // select: redraw existing groups, plus the in-progress box in the next color
  const a = clientToNat(Math.min(dragStart.clientX, e.clientX), Math.min(dragStart.clientY, e.clientY));
  const b = clientToNat(Math.max(dragStart.clientX, e.clientX), Math.max(dragStart.clientY, e.clientY));
  redrawAll({
    x: a.x,
    y: a.y,
    w: b.x - a.x,
    h: b.y - a.y,
    color: colorForIndex(state.selections.length),
  });
});

window.addEventListener("mouseup", (e) => {
  if (!mode) return;
  if (mode === "pan") {
    mode = null;
    overlay.classList.remove("panning");
    return;
  }
  // select
  const a = clientToNat(Math.min(dragStart.clientX, e.clientX), Math.min(dragStart.clientY, e.clientY));
  const b = clientToNat(Math.max(dragStart.clientX, e.clientX), Math.max(dragStart.clientY, e.clientY));
  mode = null;
  const w = b.x - a.x;
  const h = b.y - a.y;
  if (w < 5 || h < 5) {
    setStatus("Selection too small — drag a larger box.", true);
    redrawAll();
    return;
  }
  state.selections.push({
    selection: { x: a.x, y: a.y, w, h },
    template: { x: a.x, y: a.y, w, h },
    matches: [],
    color: colorForIndex(state.selections.length),
  });
  rematchBtn.disabled = false;
  clearBtn.disabled = false;
  setTool("pan"); // back to panning after drawing a box
  runMatchLatest();
});

// wheel to zoom (centered on cursor)
const ZOOM_SENSITIVITY = 0.002; // smaller = slower wheel zoom
wrap.addEventListener(
  "wheel",
  (e) => {
    if (!state.natW) return;
    e.preventDefault();
    // Smoothly scale with wheel delta; clamp so a single big tick is gentle.
    const delta = Math.max(-60, Math.min(60, e.deltaY));
    const factor = Math.exp(-delta * ZOOM_SENSITIVITY);
    setZoom(state.zoom * factor, e.clientX, e.clientY);
  },
  { passive: false }
);

// Space toggles pan-ready cursor
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && state.natW) {
    spaceHeld = true;
    overlay.classList.add("pan-ready");
    e.preventDefault();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space") {
    spaceHeld = false;
    overlay.classList.remove("pan-ready");
  }
});

// ---- wiring ---------------------------------------------------------------
ridSelect.addEventListener("change", () => {
  state.rid = ridSelect.value || null;
  widInput.disabled = true;
  widInput.value = "";
  loadBtn.disabled = true;
  if (state.rid) loadWorksheets(state.rid);
});

widInput.addEventListener("input", resolveWid);
widInput.addEventListener("change", resolveWid);

loadBtn.addEventListener("click", () => state.wid && loadImage());
selectBtn.addEventListener("click", () => setTool(tool === "select" ? "pan" : "select"));
samBtn.addEventListener("click", () => runSam("fastsam"));
hqsamBtn.addEventListener("click", () => runSam("hqsam"));
sam2Btn.addEventListener("click", () => runSam("sam2"));

layersPanel.querySelectorAll("input[data-layer]").forEach((cb) => {
  cb.addEventListener("change", () => {
    state.layers[cb.dataset.layer] = cb.checked;
    redrawAll();
  });
});
rematchBtn.addEventListener("click", rematchAll);
clearBtn.addEventListener("click", () => {
  state.selections = [];
  state.sam = { fastsam: [], hqsam: [], sam2: [] };
  redrawAll(); // keeps the ground-truth points/patches layers visible
  rematchBtn.disabled = true;
  clearBtn.disabled = true;
  setStatus('Cleared. Click "Select symbol" and drag a box to match.');
});
thrSlider.addEventListener("input", () => {
  thrVal.textContent = parseFloat(thrSlider.value).toFixed(2);
});

document.getElementById("zoom-in").addEventListener("click", () => setZoom(state.zoom * 1.25));
document.getElementById("zoom-out").addEventListener("click", () => setZoom(state.zoom / 1.25));
document.getElementById("zoom-fit").addEventListener("click", fitZoom);
document.getElementById("zoom-100").addEventListener("click", () => {
  if (state.baseWidth) setZoom(state.natW / state.baseWidth);
});

window.addEventListener("resize", () => {
  if (!state.natW) return;
  syncOverlaySize();
});

loadRequests();
