// Visual Quill — preview a bbox "dryrun" JSON over its worksheet image.
// Self-contained viewer; the pan/zoom/overlay machinery mirrors static/app.js
// (kept separate so the detection page stays untouched).

const modeSelect = document.getElementById("mode-select");
const dryrunSelect = document.getElementById("dryrun-select");
const dryrunInfo = document.getElementById("dryrun-info");
const statusEl = document.getElementById("status");
const img = document.getElementById("worksheet-img");
const overlay = document.getElementById("overlay");
const wrap = document.getElementById("stage-wrap");
const stage = document.getElementById("stage");
const zoomGroup = document.getElementById("zoom-group");
const zoomVal = document.getElementById("zoom-val");
const layersPanel = document.getElementById("layers");
const layersList = document.getElementById("layers-list");

const SVGNS = "http://www.w3.org/2000/svg";
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 12;

let state = {
  rid: null,
  wid: null,
  natW: 0, // raster (image) width in px
  natH: 0,
  baseWidth: 0, // displayed width (px) at zoom = 1 (fit-to-view)
  zoom: 1,
  layersData: [], // [{name, description, style, polygons:[[[x,y]…]…]}]
  layers: {}, // name -> bool (visible)
};

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.style.color = isError ? "#ff6b6b" : "var(--accent)";
}

// ---- zoom (ported from app.js) --------------------------------------------
function displayedWidth() {
  return state.baseWidth * state.zoom;
}

function applyZoom() {
  stage.style.width = `${displayedWidth()}px`;
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
  const avail = wrap.clientWidth - 2;
  state.baseWidth = Math.min(avail, state.natW) || avail;
  state.zoom = 1;
  applyZoom();
}

function syncOverlaySize() {
  overlay.setAttribute("viewBox", `0 0 ${state.natW} ${state.natH}`);
  overlay.setAttribute("preserveAspectRatio", "none");
}

// ---- drawing --------------------------------------------------------------
function clearOverlay() {
  while (overlay.firstChild) overlay.removeChild(overlay.firstChild);
}

// Draw a quill polygon [[x,y]…] (image px) honoring its layer style: color,
// stroke width, border/fill opacity. Fill is a translucent tint so the symbol
// underneath stays visible; stroke stays a constant px at any zoom.
function addPolygonStyled(pts, style) {
  const poly = document.createElementNS(SVGNS, "polygon");
  poly.setAttribute("points", pts.map(([x, y]) => `${x},${y}`).join(" "));
  const color = style.color || "#00e5ff";
  poly.setAttribute("fill", color);
  poly.setAttribute("fill-opacity", String((style.opacity ?? 1) * 0.2));
  poly.setAttribute("stroke", color);
  poly.setAttribute("stroke-width", String(Math.max(1, style.width || 2)));
  poly.setAttribute("stroke-opacity", String(style.border_opacity ?? 1));
  poly.setAttribute("vector-effect", "non-scaling-stroke");
  overlay.appendChild(poly);
}

function redraw() {
  clearOverlay();
  state.layersData.forEach((L) => {
    if (!state.layers[L.name]) return;
    const style = L.style || {};
    if (style.is_visible === false) return;
    L.polygons.forEach((pts) => {
      if (pts && pts.length >= 3) addPolygonStyled(pts, style);
    });
  });
}

// ---- layers panel ---------------------------------------------------------
function buildLayersPanel() {
  layersList.innerHTML = "";
  state.layers = {};
  state.layersData.forEach((L) => {
    const style = L.style || {};
    const visible = style.is_visible !== false;
    state.layers[L.name] = visible;

    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = visible;
    cb.dataset.layer = L.name;
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = style.color || "#888";
    const cnt = document.createElement("span");
    cnt.className = "cnt";
    cnt.textContent = L.polygons.length;

    label.appendChild(cb);
    label.appendChild(sw);
    label.appendChild(document.createTextNode(` ${L.name} `));
    label.appendChild(cnt);
    layersList.appendChild(label);
  });
  layersPanel.hidden = state.layersData.length === 0;
}

// ---- data load ------------------------------------------------------------
async function loadDryruns() {
  setStatus("Loading dryrun list…");
  try {
    const r = await fetch("/api/quill/dryruns");
    if (!r.ok) throw new Error(r.statusText);
    const { dryruns } = await r.json();
    dryrunSelect.innerHTML = "";
    if (!dryruns.length) {
      dryrunSelect.innerHTML = '<option value="">(no dryruns found)</option>';
      setStatus("No *_bbox_dryrun.json files found in srcs/symbol_det_steps_out/.", true);
      return;
    }
    dryruns.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.file;
      const bits = [];
      if (d.n_boxes != null) bits.push(`${d.n_boxes} boxes`);
      if (d.n_layers != null) bits.push(`${d.n_layers} layers`);
      opt.textContent = bits.length ? `${d.file} — ${bits.join(", ")}` : d.file;
      if (i === 0) opt.selected = true;
      dryrunSelect.appendChild(opt);
    });
    await loadDryrun(dryrunSelect.value);
  } catch (e) {
    setStatus(`Failed to list dryruns: ${e.message}`, true);
  }
}

async function loadDryrun(file) {
  if (!file) return;
  clearOverlay();
  setStatus(`Loading ${file}…`);
  let data;
  try {
    const r = await fetch(`/api/quill/dryrun?file=${encodeURIComponent(file)}`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    data = await r.json();
  } catch (e) {
    setStatus(`Failed to load ${file}: ${e.message}`, true);
    return;
  }
  state.rid = data.rid;
  state.wid = data.wid;
  state.layersData = data.layers || [];
  const [rw, rh] = data.raster_wh || [0, 0];
  state.natW = rw;
  state.natH = rh;

  buildLayersPanel();

  try {
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("image element failed to load"));
      img.src = `/api/worksheet/${state.rid}/${state.wid}/image?t=${Date.now()}`;
    });
  } catch (e) {
    setStatus(`Boxes loaded, but the worksheet image failed: ${e.message}`, true);
  }
  syncOverlaySize();
  initZoomForImage();
  zoomGroup.hidden = false;
  redraw();

  const totalBoxes = state.layersData.reduce((n, L) => n + L.polygons.length, 0);
  dryrunInfo.textContent =
    `wid ${state.wid} · ${state.natW}×${state.natH}px · ${totalBoxes} boxes · ${state.layersData.length} layers`;
  setStatus(
    `Loaded ${file}: ${totalBoxes} boxes across ${state.layersData.length} layers. Drag to pan; scroll to zoom.`
  );
}

// ---- pan + zoom interactions (ported from app.js) -------------------------
let panStart = null;
overlay.addEventListener("mousedown", (e) => {
  if (!state.natW || e.button !== 0) return;
  e.preventDefault();
  overlay.classList.add("panning");
  panStart = {
    clientX: e.clientX,
    clientY: e.clientY,
    scrollLeft: wrap.scrollLeft,
    scrollTop: wrap.scrollTop,
  };
});
window.addEventListener("mousemove", (e) => {
  if (!panStart) return;
  wrap.scrollLeft = panStart.scrollLeft - (e.clientX - panStart.clientX);
  wrap.scrollTop = panStart.scrollTop - (e.clientY - panStart.clientY);
});
window.addEventListener("mouseup", () => {
  if (!panStart) return;
  panStart = null;
  overlay.classList.remove("panning");
});

const ZOOM_SENSITIVITY = 0.002;
wrap.addEventListener(
  "wheel",
  (e) => {
    if (!state.natW) return;
    e.preventDefault();
    const delta = Math.max(-60, Math.min(60, e.deltaY));
    const factor = Math.exp(-delta * ZOOM_SENSITIVITY);
    setZoom(state.zoom * factor, e.clientX, e.clientY);
  },
  { passive: false }
);

document.getElementById("zoom-in").addEventListener("click", () => setZoom(state.zoom * 1.25));
document.getElementById("zoom-out").addEventListener("click", () => setZoom(state.zoom / 1.25));
document.getElementById("zoom-fit").addEventListener("click", fitZoom);
document.getElementById("zoom-100").addEventListener("click", () => {
  // actual size: displayedWidth == natW  =>  zoom = natW / baseWidth
  if (state.baseWidth) setZoom(state.natW / state.baseWidth);
});

// ---- init -----------------------------------------------------------------
if (modeSelect) {
  modeSelect.addEventListener("change", () => {
    window.location.href = modeSelect.value;
  });
}
dryrunSelect.addEventListener("change", () => loadDryrun(dryrunSelect.value));
layersList.addEventListener("change", (e) => {
  const cb = e.target;
  if (cb && cb.dataset && cb.dataset.layer != null) {
    state.layers[cb.dataset.layer] = cb.checked;
    redraw();
  }
});

loadDryruns();
