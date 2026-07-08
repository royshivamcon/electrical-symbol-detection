const ridSelect = document.getElementById("rid-select");
const widInput = document.getElementById("wid-input");
const widList = document.getElementById("wid-list");
const patchesOnly = document.getElementById("patches-only");
const loadBtn = document.getElementById("load-btn");
const rematchBtn = document.getElementById("rematch-btn");
const clearBtn = document.getElementById("clear-btn");
const selectBtn = document.getElementById("select-btn");
const samBtn = document.getElementById("sam-btn");
const fastsamxBtn = document.getElementById("fastsamx-btn");
const hqsamBtn = document.getElementById("hqsam-btn");
const toolsRow = document.getElementById("tools");
const filterSelect = document.getElementById("filter-select");
const ksizeInput = document.getElementById("ksize");
const ksizeField = document.getElementById("ksize-field");
const channelsField = document.getElementById("channels-field");
const krInput = document.getElementById("kr");
const kgInput = document.getElementById("kg");
const kbInput = document.getElementById("kb");
const tileSelect = document.getElementById("tile-select");
const zoomProcSelect = document.getElementById("zoom-proc-select");
const removeText = document.getElementById("remove-text");
const procViewSelect = document.getElementById("proc-view-select");
const showTileGrid = document.getElementById("show-tile-grid");
const tileHighlight = document.getElementById("tile-highlight");
const postprocCheck = document.getElementById("postproc");
const hullCheck = document.getElementById("hull");
const maskLayer = document.getElementById("mask-layer");
const maskModelSelect = document.getElementById("mask-model-select");
const maskMinScore = document.getElementById("mask-min-score");
const maskMinVal = document.getElementById("mask-min-val");
const maskScanBtn = document.getElementById("mask-scan-btn");
const anRemoveText = document.getElementById("an-remove-text");
const evalModel = document.getElementById("eval-model");
const evalGt = document.getElementById("eval-gt");
const evalIou = document.getElementById("eval-iou");
const evalIouVal = document.getElementById("eval-iou-val");
const evalBtn = document.getElementById("eval-btn");
const evalMetrics = document.getElementById("eval-metrics");
const anIou = document.getElementById("an-iou");
const anIouVal = document.getElementById("an-iou-val");
const anMax = document.getElementById("an-max");
const anTile = document.getElementById("an-tile");
const anZoom = document.getElementById("an-zoom");
const anRun = document.getElementById("an-run");
const anResults = document.getElementById("an-results");
const thrSlider = document.getElementById("thr-slider");
const thrVal = document.getElementById("thr-val");
const matchMethod = document.getElementById("match-method");
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
const SAM_MODELS = {
  fastsam:  { label: "FastSAM-S (cyan)",   color: "#00e5ff", btn: () => samBtn },
  fastsamx: { label: "FastSAM-X (orange)", color: "#ff9100", btn: () => fastsamxBtn },
  hqsam:    { label: "HQ-SAM (magenta)",   color: "#ff3df0", btn: () => hqsamBtn },
};
const SAM_COLORS = Object.fromEntries(Object.entries(SAM_MODELS).map(([k, v]) => [k, v.color]));
const POINT_COLOR = "#ff1744"; // ground-truth reference points (red crosshair)
const PATCH_COLOR = "#76ff03"; // ground-truth polygon patches (green box)
const EVAL_TP_COLOR = "#00e676"; // matched prediction (green)
const EVAL_FP_COLOR = "#ff5252"; // false-positive prediction (red)
const EVAL_MISS_COLOR = "#ffd600"; // missed GT (yellow)
const TILE_GRID_COLOR = "#ff8c00"; // orange tile outlines (notebook BGR 0,140,255)
const TILE_HIGHLIGHT_COLOR = "#ff0000"; // walkthrough / selected tile
const SAM_CROP_PX = 100; // matches api_sam_points default crop
const TILE_OVERLAP_BASE = 96; // boxes_from_points tile_overlap default
const PROC_VIEWS = ["original", "binary", "suppressed"];
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 12;

function emptyViewState() {
  return {
    sam: { fastsam: [], fastsamx: [], hqsam: [] },
    eval: { pred: [], gt: [] },
    layers: { masks: false, segment: false },
    lastSamModel: "fastsam",
    lastSegmentModel: "fastsam",
  };
}

let state = {
  rid: null,
  wid: null,
  natW: 0, // natural (original) image width in px
  natH: 0,
  baseWidth: 0, // displayed width (px) at zoom = 1 (fit-to-view)
  zoom: 1,
  // Each entry: {selection:{x,y,w,h}, template:{x,y,w,h}, matches:[], color}
  selections: [],
  sam: { fastsam: [], fastsamx: [], hqsam: [] }, // SAM boxes per model from reference points
  refPoints: [], // ground-truth reference points (electrical Point features)
  refPolygons: [], // ground-truth polygon patches (electrical Polygon features)
  eval: { pred: [], gt: [] }, // last evaluation: tagged prediction + GT boxes
  layers: {
    points: true, patches: true, matches: true, fastsam: true, fastsamx: true, hqsam: true,
    masks: false, segment: false, "eval-tp": true,
    "eval-fp": true, "eval-missed": true,
  }, // visibility toggles
  // Per processing-view cache: SAM boxes, eval, mask layers survive view switches.
  byView: Object.fromEntries(PROC_VIEWS.map((v) => [v, emptyViewState()])),
  // Processing options: background view (original/binary/suppressed) + whether
  // SAM masks are symbol_det post-processed. lastSamModel drives the ink-mask layer.
  detect: { view: "original", postproc: true },
  lastSamModel: "fastsam",
  wids: new Set(), // valid wids for the current request
  tiles: [], // [{x0,y0,x1,y1}, ...] SAM tiling grid in image px
  highlightTile: -1, // -1 = all orange, >=0 = that tile red
};

function colorForIndex(i) {
  return PALETTE[i % PALETTE.length];
}

function currentProcView() {
  return procViewSelect ? procViewSelect.value : "original";
}

// Persist the active view's detection overlays before switching backgrounds.
function saveCurrentViewState() {
  const view = state.detect.view || "original";
  state.byView[view] = {
    sam: {
      fastsam: [...state.sam.fastsam],
      fastsamx: [...state.sam.fastsamx],
      hqsam: [...state.sam.hqsam],
    },
    eval: {
      pred: [...state.eval.pred],
      gt: [...state.eval.gt],
    },
    layers: { masks: state.layers.masks, segment: state.layers.segment },
    lastSamModel: state.lastSamModel,
    lastSegmentModel: maskModelSelect ? maskModelSelect.value : "fastsam",
  };
}

function restoreViewState(view) {
  const vs = state.byView[view] || emptyViewState();
  state.sam = {
    fastsam: [...vs.sam.fastsam],
    fastsamx: [...vs.sam.fastsamx],
    hqsam: [...vs.sam.hqsam],
  };
  state.eval = { pred: [...vs.eval.pred], gt: [...vs.eval.gt] };
  state.layers.masks = vs.layers.masks;
  state.layers.segment = vs.layers.segment;
  state.lastSamModel = vs.lastSamModel || "fastsam";
  if (maskModelSelect) maskModelSelect.value = vs.lastSegmentModel || "fastsam";
  const maskCb = layersPanel.querySelector('input[data-layer="masks"]');
  const segCb = layersPanel.querySelector('input[data-layer="segment"]');
  if (maskCb) maskCb.checked = state.layers.masks;
  if (segCb) segCb.checked = state.layers.segment;
  evalMetrics.hidden = !state.eval.pred.length && !state.eval.gt.length;
  if (state.eval.pred.length || state.eval.gt.length) {
    // metrics table is rebuilt on evaluate; hide if we only restored boxes
    if (!evalMetrics.querySelector(".metric-tbl")) evalMetrics.hidden = true;
  }
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
  const q = patchesOnly.checked ? "?patches_only=1" : "";
  if (patchesOnly.checked) setStatus("Scanning worksheets for ground-truth patches…");
  const r = await fetch(`/api/requests/${rid}/worksheets${q}`);
  const data = await r.json();
  if (patchesOnly.checked) setStatus("");
  if (!data.worksheets.length) {
    widInput.placeholder = patchesOnly.checked
      ? "No sheets with patches — uncheck the filter"
      : "No worksheets with images";
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
  widInput.placeholder = patchesOnly.checked
    ? `search ${data.worksheets.length} wids with patches…`
    : `search ${data.worksheets.length} wids (${withGeom} ● with geometries)…`;
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
  state.sam = { fastsam: [], fastsamx: [], hqsam: [] };
  state.refPoints = [];
  state.refPolygons = [];
  state.eval = { pred: [], gt: [] };
  state.byView = Object.fromEntries(PROC_VIEWS.map((v) => [v, emptyViewState()]));
  evalMetrics.hidden = true;
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
  toolsRow.hidden = false;
  selectBtn.disabled = false;
  if (matchMethod) matchMethod.disabled = false;
  samBtn.disabled = false;
  hqsamBtn.disabled = false;
  fastsamxBtn.disabled = false;
  filterSelect.disabled = false;
  tileSelect.disabled = false;
  zoomProcSelect.disabled = false;
  if (removeText) removeText.disabled = false;
  if (procViewSelect) { procViewSelect.disabled = false; procViewSelect.value = "original"; }
  if (showTileGrid) { showTileGrid.disabled = false; showTileGrid.checked = false; }
  if (tileHighlight) { tileHighlight.disabled = true; tileHighlight.innerHTML = '<option value="-1">All</option>'; }
  state.tiles = [];
  state.highlightTile = -1;
  if (postprocCheck) postprocCheck.disabled = false;
  if (hullCheck) hullCheck.disabled = false;
  state.detect.view = "original";
  state.layers.masks = false;
  state.layers.segment = false;
  const maskCb = layersPanel.querySelector('input[data-layer="masks"]');
  const segCb = layersPanel.querySelector('input[data-layer="segment"]');
  if (maskCb) maskCb.checked = false;
  if (segCb) segCb.checked = false;
  if (maskLayer) { maskLayer.hidden = true; maskLayer.removeAttribute("src"); }
  if (maskModelSelect) maskModelSelect.disabled = false;
  if (maskMinScore) maskMinScore.disabled = false;
  if (maskScanBtn) maskScanBtn.disabled = false;
  evalModel.disabled = false;
  evalGt.disabled = false;
  evalIou.disabled = false;
  evalBtn.disabled = false;
  updateFilterFields();
  layersPanel.hidden = false;
  updateLayerCounts();
  setTool("pan");
  loadRefPoints();
  loadRefPolygons();
  refreshTileGrid();
  setStatus(`Loaded ${state.natW}×${state.natH}px. Drag to pan; click "Select symbol" to pick one.`);
}

async function matchOne(sel, threshold) {
  const method = matchMethod ? matchMethod.value : "classical";
  const r = await fetch(`/api/worksheet/${state.rid}/${state.wid}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...sel.selection, threshold, method }),
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
  const method = matchMethod ? matchMethod.value : "classical";
  setStatus(method === "classical" ? "Matching…" : `Matching with ${method.toUpperCase()} (encoding the sheet, this can take a bit)…`);
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

function addRect(x, y, w, h, color, dashed = false, fill = false) {
  const rect = document.createElementNS(SVGNS, "rect");
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  rect.setAttribute("width", w);
  rect.setAttribute("height", h);
  // Optional translucent fill so tiny boxes stay visible when the whole sheet
  // is zoomed out (a hollow ~5px outline on a 7000px sheet is easy to miss).
  rect.setAttribute("fill", fill ? color : "none");
  if (fill) rect.setAttribute("fill-opacity", "0.25");
  rect.setAttribute("stroke", color);
  rect.setAttribute("stroke-width", dashed ? 2.5 : 2);
  if (dashed) rect.setAttribute("stroke-dasharray", "6 4");
  rect.setAttribute("vector-effect", "non-scaling-stroke"); // constant px at any zoom
  overlay.appendChild(rect);
  return rect;
}

// Draw an arbitrary polygon (e.g. a convex-hull / rotated box) from [[x,y], …]
// image-pixel corners. Mirrors addRect's fill/stroke so hull boxes match the
// look of the axis-aligned ones.
function addPolygon(pts, color, fill = false) {
  const poly = document.createElementNS(SVGNS, "polygon");
  poly.setAttribute("points", pts.map(([x, y]) => `${x},${y}`).join(" "));
  poly.setAttribute("fill", fill ? color : "none");
  if (fill) poly.setAttribute("fill-opacity", "0.25");
  poly.setAttribute("stroke", color);
  poly.setAttribute("stroke-width", 2);
  poly.setAttribute("vector-effect", "non-scaling-stroke");
  overlay.appendChild(poly);
  return poly;
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
  drawTileGrid();
  Object.entries(state.sam).forEach(([model, boxes]) => {
    if (!state.layers[model]) return;
    boxes.forEach((b) => {
      if (b.hull && b.hull.length >= 3) addPolygon(b.hull, SAM_COLORS[model], true);
      else addRect(b.x, b.y, b.w, b.h, SAM_COLORS[model], false, true);
    });
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
  // Evaluation overlay: tagged predictions (tp/fp) + missed GT patches.
  state.eval.pred.forEach((b) => {
    if (b.status === "tp" && state.layers["eval-tp"]) addRect(b.x, b.y, b.w, b.h, EVAL_TP_COLOR, false, true);
    if (b.status === "fp" && state.layers["eval-fp"]) addRect(b.x, b.y, b.w, b.h, EVAL_FP_COLOR, false, true);
  });
  if (state.layers["eval-missed"]) {
    state.eval.gt.forEach((g) => {
      if (g.status !== "missed") return;
      // points-mode GT has no w/h -> mark the missed point with a cross
      if (g.w && g.h) addRect(g.x, g.y, g.w, g.h, EVAL_MISS_COLOR, true);
      else addPoint(g.x, g.y, EVAL_MISS_COLOR);
    });
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
  set("cnt-fastsamx", state.sam.fastsamx.length);
  set("cnt-masks", state.layers.masks ? "on" : "off");
  set("cnt-segment", state.layers.segment ? "on" : "off");
  set("cnt-eval-tp", state.eval.pred.filter((b) => b.status === "tp").length);
  set("cnt-eval-fp", state.eval.pred.filter((b) => b.status === "fp").length);
  set("cnt-eval-missed", state.eval.gt.filter((g) => g.status === "missed").length);
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
    const r = await fetch(`/api/worksheet/${state.rid}/${state.wid}/ref_polygons?electrical=1`);
    if (!r.ok) return;
    const data = await r.json();
    state.refPolygons = data.polygons || [];
    redrawAll();
  } catch (e) {
    /* non-fatal: the patches layer just stays empty */
  }
}

// Show only the kernel field(s) relevant to the selected preprocessing filter.
const KSIZE_FILTERS = new Set([
  "gaussian", "laplace", "sharpen", "median", "bilateral", "clahe", "threshold",
]);
function updateFilterFields() {
  const f = filterSelect.value;
  const isChannels = f === "channels";
  const usesKsize = KSIZE_FILTERS.has(f);
  ksizeField.hidden = !usesKsize;
  channelsField.hidden = !isChannels;
  ksizeInput.disabled = !usesKsize;
  krInput.disabled = kgInput.disabled = kbInput.disabled = !isChannels;
}

// Build the ?filt=… query fragment + a short human-readable tag for the status.
// Channel / kernel sizes always come from the SAM preprocessing filter group.
function filterQueryFor(f) {
  if (!f || f === "none") return { qs: "", tag: "" };
  if (f === "channels") {
    const r = +krInput.value || 1, g = +kgInput.value || 1, b = +kbInput.value || 1;
    return { qs: `&filt=channels&kr=${r}&kg=${g}&kb=${b}`, tag: `channels ${r}/${g}/${b}` };
  }
  const k = +ksizeInput.value || 1;
  return { qs: `&filt=${f}&ksize=${k}`, tag: `${f} k${k}` };
}
function filterQuery() {
  return filterQueryFor(filterSelect.value);
}

// Hybrid-tiling query fragment (FastSAM / HQ-SAM). tile=0 -> crop-per-point.
function tileQuery() {
  const t = +tileSelect.value || 0;
  return t > 0 ? { qs: `&tile=${t}`, tag: `tiled@${t}` } : { qs: "", tag: "" };
}

// Port of tiling.tile_grid — must stay in sync with symbol_matcher_app/tiling.py.
function computeTileGrid(w, h, tile, overlap) {
  tile = Math.max(64, tile | 0);
  overlap = Math.max(0, Math.min(overlap | 0, tile - 1));
  const step = Math.max(1, tile - overlap);
  const starts = (total) => {
    if (total <= tile) return [0];
    const s = [];
    for (let v = 0; v <= total - tile; v += step) s.push(v);
    if (s[s.length - 1] !== total - tile) s.push(total - tile);
    return s;
  };
  const tiles = [];
  for (const y0 of starts(h)) {
    for (const x0 of starts(w)) {
      tiles.push({ x0, y0, x1: Math.min(x0 + tile, w), y1: Math.min(y0 + tile, h) });
    }
  }
  return tiles;
}

// Recompute the SAM tile grid from the Tiling select (overlap matches boxes_from_points).
function refreshTileGrid() {
  const tile = +tileSelect.value || 0;
  const tilingOn = tile > 0 && state.natW > 0;
  if (tileHighlight) tileHighlight.disabled = !tilingOn || !(showTileGrid && showTileGrid.checked);
  if (!tilingOn) {
    state.tiles = [];
    state.highlightTile = -1;
    if (tileHighlight) {
      tileHighlight.innerHTML = '<option value="-1">All</option>';
      tileHighlight.value = "-1";
    }
    redrawAll();
    return;
  }
  const overlap = Math.max(TILE_OVERLAP_BASE, 2 * SAM_CROP_PX);
  state.tiles = computeTileGrid(state.natW, state.natH, tile, overlap);
  if (tileHighlight) {
    const prev = state.highlightTile;
    tileHighlight.innerHTML = '<option value="-1">All</option>';
    state.tiles.forEach((_, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `Tile #${i}`;
      tileHighlight.appendChild(opt);
    });
    const valid = prev >= 0 && prev < state.tiles.length;
    state.highlightTile = valid ? prev : -1;
    tileHighlight.value = String(state.highlightTile);
    tileHighlight.disabled = !(showTileGrid && showTileGrid.checked);
  }
  redrawAll();
}

function drawTileGrid() {
  if (!showTileGrid || !showTileGrid.checked || !state.tiles.length) return;
  const hi = state.highlightTile;
  state.tiles.forEach((t, i) => {
    const w = t.x1 - t.x0;
    const h = t.y1 - t.y0;
    const highlighted = hi >= 0 && i === hi;
    const rect = document.createElementNS(SVGNS, "rect");
    rect.setAttribute("x", t.x0);
    rect.setAttribute("y", t.y0);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", highlighted ? TILE_HIGHLIGHT_COLOR : TILE_GRID_COLOR);
    rect.setAttribute("stroke-width", highlighted ? 3 : 2);
    if (!highlighted) rect.setAttribute("stroke-dasharray", "6 4");
    rect.setAttribute("vector-effect", "non-scaling-stroke");
    overlay.appendChild(rect);
  });
}

// Render-zoom query fragment (quill_forge / fitz PDF render). zoom=1 -> off.
function zoomQuery() {
  const z = +zoomProcSelect.value || 1;
  const nt = removeText && removeText.checked;
  let qs = z > 1 ? `&zoom=${z}` : "";
  if (nt) qs += "&remove_text=1";
  const tag = [z > 1 ? `zoom${z}x` : "", nt ? "no-text" : ""].filter(Boolean).join(", ");
  return { qs, tag };
}

// symbol_det mask post-processing: intersect each SAM mask with the ink and drop
// small / sparse / line-like ones. Applies to hqsam / fastsam / fastsamx.
function postprocQuery() {
  const on = postprocCheck && postprocCheck.checked;
  return on ? { qs: "&postproc=1", tag: "postproc" } : { qs: "", tag: "" };
}

// Box each SAM detection as the convex-hull-based rotated rectangle (4 corners)
// instead of the axis-aligned rect. Applies to hqsam / fastsam / fastsamx.
function hullQuery() {
  const on = hullCheck && hullCheck.checked;
  return on ? { qs: "&hull=1", tag: "hull" } : { qs: "", tag: "" };
}

// Swap the worksheet background between the original raster and a symbol_det
// processed view (binary / line-suppressed). Saves/restores per-view overlays.
function applyProcView() {
  if (!state.natW) return;
  const view = currentProcView();
  saveCurrentViewState();
  state.detect.view = view;
  restoreViewState(view);
  const { rid, wid } = state;
  if (view === "original") {
    img.src = `/api/worksheet/${rid}/${wid}/image?t=${Date.now()}`;
  } else {
    setStatus(`Building ${view === "binary" ? "binary ink" : "line-suppressed"} view…`);
    const zq = zoomQuery();
    img.src = `/api/worksheet/${rid}/${wid}/processed?view=${view}${zq.qs}`;
  }
  img.onload = () => {
    refreshMaskLayers();
    redrawAll();
  };
}

function procViewQuery() {
  const v = currentProcView();
  return v === "original" ? { qs: "", tag: "" } : { qs: `&proc_view=${v}`, tag: v };
}

// (Re)load mask overlay(s): segment-everything scan and/or per-point ink masks.
function refreshMaskLayers({ filterOnly = false } = {}) {
  if (!maskLayer) return;
  if (!state.natW) {
    maskLayer.hidden = true;
    maskLayer.removeAttribute("src");
    return;
  }
  const showSegment = state.layers.segment;
  const showInk = state.layers.masks;
  if (!showSegment && !showInk) {
    maskLayer.hidden = true;
    maskLayer.removeAttribute("src");
    return;
  }
  const { rid, wid } = state;
  const { qs: fqs } = filterQuery();
  const tq = tileQuery();
  const zq = zoomQuery();
  const pv = procViewQuery();
  const tile = +tileSelect.value || 1024;
  const minConf = maskMinScore ? parseFloat(maskMinScore.value) : 0.25;
  let url;
  if (showSegment) {
    const model = maskModelSelect ? maskModelSelect.value : "fastsam";
    if (!filterOnly) {
      setStatus(`Building segment mask overlay (${model})… (first run may take a few minutes)`);
    }
    url =
      `/api/worksheet/${rid}/${wid}/segment_masks?model=${model}${fqs}${tq.qs}${zq.qs}${pv.qs}` +
      `&tile=${tile}&min_score=${minConf}`;
  } else {
    const model = state.lastSamModel || "fastsam";
    setStatus(`Building per-point mask overlay (${model})…`);
    url =
      `/api/worksheet/${rid}/${wid}/sam_masks?model=${model}${fqs}${tq.qs}${zq.qs}${pv.qs}` +
      `&postproc=1&tile=${tile}`;
  }
  maskLayer.onload = () => {
    maskLayer.hidden = false;
    setStatus("");
  };
  maskLayer.onerror = () => setStatus("Mask overlay failed to load", true);
  maskLayer.src = url;
}

function refreshMaskLayer() {
  refreshMaskLayers();
}

// Segment symbols from the worksheet's reference points with the chosen SAM model.
const SAM_LABEL = Object.fromEntries(Object.entries(SAM_MODELS).map(([k, v]) => [k, v.label]));
const SAM_BTN = Object.fromEntries(Object.entries(SAM_MODELS).map(([k, v]) => [k, v.btn]));
async function runSam(model) {
  const btn = SAM_BTN[model]();
  const { qs, tag } = filterQuery();
  const tq = tileQuery();
  const zq = zoomQuery();
  const pq = postprocQuery();
  const hq = hullQuery();
  const pv = procViewQuery();
  const tags = [tag, tq.tag, zq.tag, pq.tag, hq.tag, pv.tag].filter(Boolean).join(", ");
  const preTag = tags ? ` (${tags})` : "";
  setStatus(`Running ${SAM_LABEL[model]}${preTag} on reference points… (first run loads the model)`);
  btn.disabled = true;
  state.lastSamModel = model;
  const wantMasks = state.layers.masks;
  try {
    const masksQs = wantMasks ? "&masks=1" : "";
    const r = await fetch(
      `/api/worksheet/${state.rid}/${state.wid}/sam_points?model=${model}${qs}${tq.qs}${zq.qs}${pq.qs}${hq.qs}${pv.qs}${masksQs}`
    );
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    state.sam[model] = data.boxes;
    saveCurrentViewState();
    redrawAll();
    if (state.layers.masks || state.layers.segment) refreshMaskLayers();
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

async function runMaskScan() {
  const model = maskModelSelect ? maskModelSelect.value : "fastsam";
  const { qs, tag } = filterQuery();
  const tq = tileQuery();
  const zq = zoomQuery();
  const pv = procViewQuery();
  const minConf = maskMinScore ? parseFloat(maskMinScore.value) : 0.25;
  const tile = +tileSelect.value || 1024;
  const tags = [tag, tq.tag, zq.tag, pv.tag].filter(Boolean).join(", ");
  const preTag = tags ? ` (${tags})` : "";
  setStatus(`Running segment-everything mask scan (${model})${preTag}… (first run loads the model)`);
  if (maskScanBtn) maskScanBtn.disabled = true;
  state.layers.segment = true;
  const segCb = layersPanel.querySelector('input[data-layer="segment"]');
  if (segCb) segCb.checked = true;
  saveCurrentViewState();
  try {
    const r = await fetch(
      `/api/worksheet/${state.rid}/${state.wid}/segment_masks?model=${model}${qs}${tq.qs}${zq.qs}${pv.qs}` +
      `&tile=${tile}&min_score=${minConf}&t=${Date.now()}`
    );
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const blob = await r.blob();
    if (maskLayer) {
      maskLayer.onload = () => { maskLayer.hidden = false; };
      maskLayer.src = URL.createObjectURL(blob);
    }
    saveCurrentViewState();
    updateLayerCounts();
    clearBtn.disabled = false;
    setStatus(`Mask scan (${model})${preTag}: overlay ready (min conf ${minConf.toFixed(2)}). Toggle "Segment masks" layer.`);
  } catch (e) {
    setStatus(`Mask scan failed: ${e.message}`, true);
  } finally {
    if (maskScanBtn) maskScanBtn.disabled = false;
  }
}

// ---- evaluation against GT patches ----------------------------------------
function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

function renderMetrics(m) {
  const c = m.center, i = m.iou;
  if (m.mode === "points") {
    evalMetrics.innerHTML = `
      <div class="metric-head">Prompted at ${m.n_gt} GT points → ${m.n_pred} boxes · scored by center-hit</div>
      <table class="metric-tbl">
        <tr><th></th><th>Precision</th><th>Recall</th><th>F1</th><th>extra</th></tr>
        <tr><td>Center hit</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>${c.gt_found}/${m.n_gt} points found</td></tr>
      </table>`;
  } else {
    evalMetrics.innerHTML = `
      <div class="metric-head">Prompted at ${m.n_gt} GT centers → ${m.n_pred} boxes · matched by IoU ≥ ${m.iou_thr}</div>
      <table class="metric-tbl">
        <tr><th></th><th>Precision</th><th>Recall</th><th>F1</th><th>extra</th></tr>
        <tr><td>BBox IoU</td><td>${pct(i.precision)}</td><td>${pct(i.recall)}</td><td>${pct(i.f1)}</td><td>mean IoU ${i.mean_iou}</td></tr>
        <tr class="muted-row"><td>Center hit</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>${c.gt_found} GT found (info)</td></tr>
      </table>`;
  }
  evalMetrics.hidden = false;
}

async function runEvaluate() {
  const model = evalModel.value;
  const mode = evalGt.value;
  const iou = parseFloat(evalIou.value);
  // Detection config mirrors the interactive Segment path (api_sam_points):
  // same shared controls (filter / tile / zoom / post-process) and the same backend
  // box-fit defaults, so evaluate scores the exact boxes the app produces.
  const { qs, tag } = filterQuery();
  const tq = tileQuery();
  const zq = zoomQuery();
  const pq = postprocQuery();
  const pv = procViewQuery();
  const tags = [tag, tq.tag, zq.tag, pq.tag, pv.tag].filter(Boolean).join(", ");
  const preTag = tags ? ` [${tags}]` : "";
  const how = mode === "points"
    ? "prompting at GT points, scoring center-hit"
    : `prompting at GT centers, matching by IoU ≥ ${iou.toFixed(2)}`;
  setStatus(`Evaluating ${model}${preTag}: ${how}…`);
  evalBtn.disabled = true;
  try {
    const r = await fetch(
      `/api/worksheet/${state.rid}/${state.wid}/evaluate?model=${model}&gt=${mode}&iou_thr=${iou}${qs}${tq.qs}${zq.qs}${pq.qs}${pv.qs}`
    );
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    state.eval = { pred: data.boxes || [], gt: data.gt || [] };
    renderMetrics(data.metrics);
    saveCurrentViewState();
    redrawAll();
    clearBtn.disabled = false;
    const c = data.metrics.center, i = data.metrics.iou;
    if (data.metrics.mode === "points") {
      setStatus(`${model}: center-hit F1 ${pct(c.f1)} · ${c.gt_found}/${data.metrics.n_gt} GT points found.`);
    } else {
      setStatus(`${model}: IoU F1 ${pct(i.f1)} (mean IoU ${i.mean_iou}) · center F1 ${pct(c.f1)} over ${data.metrics.n_gt} GT patches.`);
    }
  } catch (e) {
    setStatus(`Evaluation failed: ${e.message}`, true);
  } finally {
    evalBtn.disabled = false;
  }
}

async function runAnalysis() {
  const models = [...document.querySelectorAll(".an-model:checked")].map((c) => c.value);
  if (!models.length) {
    anResults.innerHTML = '<p class="an-empty">Pick at least one model.</p>';
    return;
  }
  if (!state.rid) {
    anResults.innerHTML = '<p class="an-empty">Select a request (rid) first.</p>';
    return;
  }
  const gts = [...document.querySelectorAll(".an-gt:checked")].map((c) => c.value);
  const gtCsv = gts.length ? gts.join(",") : "bboxes";
  const filts = [...document.querySelectorAll(".an-filter:checked")].map((c) => c.value);
  const filtCsv = filts.length ? filts.join(",") : "none";
  const iou = parseFloat(anIou.value);
  const maxSheets = +anMax.value || 6;
  const tile = +anTile.value || 0;
  const zoom = +anZoom.value || 1;
  const nt = anRemoveText && anRemoveText.checked ? "&remove_text=1" : "";
  // Kernel sizes are shared across all selected filters (from the SAM group).
  const k = `&ksize=${+ksizeInput.value || 5}&kr=${+krInput.value || 6}&kg=${+kgInput.value || 8}&kb=${+kbInput.value || 3}${nt}`;
  anRun.disabled = true;
  anResults.innerHTML = '<p class="an-empty">Running… this sweeps every model × filter × GT over several sheets and can take a while.</p>';
  try {
    const r = await fetch(
      `/api/analysis/${state.rid}?models=${models.join(",")}&gt=${gtCsv}&filt=${filtCsv}&iou_thr=${iou}&max_sheets=${maxSheets}&tile=${tile}&zoom=${zoom}${k}`
    );
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    renderAnalysis(data);
  } catch (e) {
    anResults.innerHTML = `<p class="an-empty" style="color:#ff6b6b">Analysis failed: ${e.message}</p>`;
  } finally {
    anRun.disabled = false;
  }
}

function renderAnalysis(data) {
  const results = data.results || [];
  let html = `<div class="an-head">Request ${data.rid} · ${data.n_sheets} sheet(s) · IoU ≥ ${data.iou_thr} · ${results.length} combination(s)</div>`;
  for (const res of results) {
    const c = res.center, i = res.iou;
    const isPoints = res.gt_mode === "points";
    let primary, sheetHead, rows;
    if (isPoints) {
      primary = `<tr><td>Center hit</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td></td></tr>`;
      sheetHead = `<tr><th>Sheet</th><th>pred</th><th>GT</th><th>center F1</th></tr>`;
      rows = res.sheets
        .map((s) => `<tr><td class="wid">${s.wid.slice(0, 8)}…</td><td>${s.n_pred}</td><td>${s.n_gt}</td><td>${pct(s.center_f1)}</td></tr>`)
        .join("");
    } else {
      primary =
        `<tr><td>BBox IoU</td><td>${pct(i.precision)}</td><td>${pct(i.recall)}</td><td>${pct(i.f1)}</td><td>mean IoU ${i.mean_iou}</td></tr>` +
        `<tr class="muted-row"><td>Center hit</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>(info)</td></tr>`;
      sheetHead = `<tr><th>Sheet</th><th>pred</th><th>GT</th><th>center F1</th><th>IoU F1</th><th>mean IoU</th></tr>`;
      rows = res.sheets
        .map((s) => `<tr><td class="wid">${s.wid.slice(0, 8)}…</td><td>${s.n_pred}</td><td>${s.n_gt}</td><td>${pct(s.center_f1)}</td><td>${pct(s.iou_f1)}</td><td>${s.mean_iou}</td></tr>`)
        .join("");
    }
    html += `
      <div class="an-model-block">
        <h3>${res.label} — ${res.n_pred} preds vs ${res.n_gt} GT</h3>
        <table class="metric-tbl">
          <tr><th></th><th>Precision</th><th>Recall</th><th>F1</th><th>extra</th></tr>
          ${primary}
        </table>
        <table class="metric-tbl sheets">
          ${sheetHead}
          ${rows}
        </table>
      </div>`;
  }
  anResults.innerHTML = html;
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

// Reload the worksheet list when the patches filter is toggled.
patchesOnly.addEventListener("change", () => {
  if (state.rid) loadWorksheets(state.rid);
});

loadBtn.addEventListener("click", () => state.wid && loadImage());
selectBtn.addEventListener("click", () => setTool(tool === "select" ? "pan" : "select"));
samBtn.addEventListener("click", () => runSam("fastsam"));
fastsamxBtn.addEventListener("click", () => runSam("fastsamx"));
hqsamBtn.addEventListener("click", () => runSam("hqsam"));
filterSelect.addEventListener("change", updateFilterFields);
evalBtn.addEventListener("click", runEvaluate);
evalIou.addEventListener("input", () => {
  evalIouVal.textContent = parseFloat(evalIou.value).toFixed(2);
});
anIou.addEventListener("input", () => {
  anIouVal.textContent = parseFloat(anIou.value).toFixed(2);
});
anRun.addEventListener("click", runAnalysis);

layersPanel.querySelectorAll("input[data-layer]").forEach((cb) => {
  cb.addEventListener("change", () => {
    state.layers[cb.dataset.layer] = cb.checked;
    if (cb.dataset.layer === "masks" || cb.dataset.layer === "segment") {
      saveCurrentViewState();
      refreshMaskLayers();
      updateLayerCounts();
    } else {
      redrawAll();
    }
  });
});

// Processing-view select swaps the background; post-process re-fetches the ink
// masks if their layer is showing (its effect on boxes is applied on the next run).
if (procViewSelect) procViewSelect.addEventListener("change", applyProcView);
if (showTileGrid) {
  showTileGrid.addEventListener("change", () => {
    if (tileHighlight) tileHighlight.disabled = !showTileGrid.checked || !(+tileSelect.value > 0);
    redrawAll();
  });
}
if (tileHighlight) {
  tileHighlight.addEventListener("change", () => {
    state.highlightTile = parseInt(tileHighlight.value, 10);
    redrawAll();
  });
}
if (tileSelect) tileSelect.addEventListener("change", refreshTileGrid);
if (postprocCheck) {
  postprocCheck.addEventListener("change", () => {
    state.detect.postproc = postprocCheck.checked;
    if (state.layers.masks) refreshMaskLayers();
  });
}
// Keep the active processed view / mask overlay in sync with render zoom + text.
[zoomProcSelect, removeText, filterSelect, ksizeInput, krInput, kgInput, kbInput].forEach((el) => {
  if (!el) return;
  el.addEventListener("change", () => {
    if (state.detect.view !== "original") applyProcView();
    else if (state.layers.masks || state.layers.segment) refreshMaskLayers();
  });
});
if (maskMinScore) {
  let maskScoreTimer = null;
  maskMinScore.addEventListener("input", () => {
    if (maskMinVal) maskMinVal.textContent = parseFloat(maskMinScore.value).toFixed(2);
    if (!state.layers.segment) return;
    setStatus("Updating mask filter…");
    clearTimeout(maskScoreTimer);
    maskScoreTimer = setTimeout(() => refreshMaskLayers({ filterOnly: true }), 150);
  });
}
if (maskScanBtn) maskScanBtn.addEventListener("click", runMaskScan);
rematchBtn.addEventListener("click", rematchAll);
clearBtn.addEventListener("click", () => {
  state.selections = [];
  state.sam = { fastsam: [], fastsamx: [], hqsam: [] };
  state.eval = { pred: [], gt: [] };
  state.layers.masks = false;
  state.layers.segment = false;
  evalMetrics.hidden = true;
  const maskCb = layersPanel.querySelector('input[data-layer="masks"]');
  const segCb = layersPanel.querySelector('input[data-layer="segment"]');
  if (maskCb) maskCb.checked = false;
  if (segCb) segCb.checked = false;
  if (maskLayer) { maskLayer.hidden = true; maskLayer.removeAttribute("src"); }
  saveCurrentViewState();
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

// Header page switcher: "Detect boxes" (this page) <-> "Visual Quill" (/quill).
const modeSelect = document.getElementById("mode-select");
if (modeSelect) {
  modeSelect.addEventListener("change", () => {
    window.location.href = modeSelect.value;
  });
}

loadRequests();
