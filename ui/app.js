/* PathSnap evaluator UI: all behaviour.

   What this file does, in order:
     1. read the settings from the URL (?api=... and ?mock=1)
     2. talk to the API (GET /health, GET /trips, POST /predict)
     3. draw the GPS points and routes on a Leaflet map
     4. show the metrics the API returned

   Two rules from api/CONTRACT.md shape the code:
     - The UI never receives, requests or shows the reference ("ground truth")
       route. It has no code for it at all.
     - The UI never computes accuracy. It only displays the numbers in
       `metrics`. The one bit of arithmetic is the HMM-vs-classifier difference,
       which is a subtraction of two numbers the API already gave us.
*/
'use strict';

// ---------------------------------------------------------------------------
// 1. Settings
// ---------------------------------------------------------------------------

const params = new URLSearchParams(window.location.search);
const MOCK = params.get('mock') === '1';              // dev mode: read ui/mock/ files
const DEFAULT_API = 'http://localhost:8000';
const REQUEST_TIMEOUT_MS = 120000;                    // give a slow pipeline 2 minutes

// Route colours. Okabe-Ito palette, so they stay distinguishable for
// colour-blind viewers. Line style differs too (solid vs dashed), so colour is
// never the only cue. Keep in sync with the CSS variables in style.css.
const COLOUR_GPS = '#0072B2';
const COLOUR_FINAL = '#D55E00';
const COLOUR_CLASSIFIER = '#CC79A7';

// The four numbers the evaluator cares about, in display order.
// `key` is the field name inside metrics.final / metrics.classifier_only.
const METRICS = [
  { key: 'top1_accuracy',    label: 'Top-1 accuracy' },
  { key: 'macro_f1',         label: 'Macro F1' },
  { key: 'route_continuity', label: 'Route continuity' },
  { key: 'route_f1',         label: 'Route F1' },
];

// ---------------------------------------------------------------------------
// Page elements
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const apiForm = $('api-form');
const apiInput = $('api-url');
const connectBtn = $('connect-btn');
const statusBox = $('status');
const statusIcon = $('status-icon');
const statusText = $('status-text');
const tripSelect = $('trip-select');
const runBtn = $('run-btn');
const mapMessage = $('map-message');
const tileNotice = $('tile-notice');
const errorBox = $('error-box');
const resultsBox = $('results');

// ---------------------------------------------------------------------------
// 2. Talking to the API
// ---------------------------------------------------------------------------

/** Read the API address box and tidy it ("192.168.1.5:8000/" -> "http://192.168.1.5:8000"). */
function apiBase() {
  let url = apiInput.value.trim().replace(/\/+$/, '');
  if (url && !/^https?:\/\//i.test(url)) url = 'http://' + url;
  return url;
}

/**
 * fetch() wrapper that always fails with a readable Error message, so the
 * caller can put e.message straight on screen. Handles three failure kinds:
 * the server cannot be reached, it answered with an HTTP error, or it
 * answered with something that is not JSON.
 */
async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response;
  try {
    response = await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(`The API did not answer within ${REQUEST_TIMEOUT_MS / 1000} seconds.`);
    }
    if (MOCK) {
      throw new Error('Could not read the mock files. In mock mode serve the page with: python -m http.server (run it inside ui/).');
    }
    throw new Error(`Cannot reach the API at ${apiBase()}. Check the address, that the API is running, and that both laptops are on the same network.`);
  } finally {
    clearTimeout(timer);
  }

  // Try to read a JSON body even for errors: the API sends {"detail": "..."}.
  let body = null;
  try { body = await response.json(); } catch (_) { /* not JSON */ }

  if (!response.ok) {
    let detail = body && body.detail;
    if (detail && typeof detail !== 'string') detail = JSON.stringify(detail);
    throw new Error(detail ? `${detail} (HTTP ${response.status})` : `The server answered HTTP ${response.status}.`);
  }
  if (body === null) throw new Error('The server answered with something that is not JSON.');
  return body;
}

/** GET /health or GET /trips (or the matching mock file). */
function apiGet(path) {
  const url = MOCK ? `mock${path}.json` : apiBase() + path;
  return fetchJson(url);
}

/** POST /predict for one trip (or read mock/predict_<id>.json). */
async function apiPredict(tripId) {
  if (MOCK) {
    try {
      return await fetchJson(`mock/predict_${encodeURIComponent(tripId)}.json`);
    } catch (err) {
      // http.server answers 404 for a missing file; tell the human plainly.
      if (/HTTP 404/.test(err.message) || /not JSON/.test(err.message)) {
        throw new Error(`No mock data for trip ${tripId}. Only 00000052 and 00000065 have mock files.`);
      }
      throw err;
    }
  }
  return fetchJson(apiBase() + '/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trip_id: tripId }),   // trip_id stays a string (leading zeros)
  });
}

// ---------------------------------------------------------------------------
// Connection indicator and trip list
// ---------------------------------------------------------------------------

function setStatus(kind, icon, text) {
  statusBox.className = `status status-${kind}`;   // checking | ok | bad
  statusIcon.textContent = icon;
  statusText.textContent = text;
}

/** Fill the dropdown. `trips` is [{trip_id, n_points}, ...] from GET /trips. */
function populateTrips(trips) {
  tripSelect.replaceChildren();

  const first = document.createElement('option');
  first.value = '';
  first.textContent = trips.length ? 'Select a trip…' : 'No trips available';
  tripSelect.appendChild(first);

  for (const t of trips) {
    const opt = document.createElement('option');
    opt.value = t.trip_id;
    opt.textContent = `${t.trip_id}  (${Number(t.n_points).toLocaleString()} points)`;
    tripSelect.appendChild(opt);
  }
  tripSelect.disabled = trips.length === 0;
  updateRunButton();
}

/** Check /health, then load /trips. Called on page load and by the Connect button. */
async function connect() {
  setStatus('checking', '…', 'Checking connection…');
  connectBtn.disabled = true;
  tripSelect.disabled = true;
  runBtn.disabled = true;

  try {
    const health = await apiGet('/health');
    if (health.status !== 'ok') throw new Error(`API reports status "${health.status}".`);
    const list = await apiGet('/trips');
    populateTrips(list.trips || []);
    setStatus('ok', '✓', `Connected · model ${health.model} · ${health.n_trips} trips`);
  } catch (err) {
    populateTrips([]);
    setStatus('bad', '✗', `Not connected: ${err.message}`);
  } finally {
    connectBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 3. The map
// ---------------------------------------------------------------------------

// preferCanvas draws vectors on a <canvas>, which stays fast with thousands of GPS points.
const map = L.map('map', { preferCanvas: true }).setView([20, 0], 2);

// Panes fix the stacking order no matter which layer is switched on last.
// Bottom to top: GPS points < final route < classifier route < Start/End markers.
// - GPS points are dense (about one every 15 m), so they stay at the bottom or
//   they would hide the routes completely.
// - The final route is the thick line. The classifier route is thinner and
//   dashed, drawn on top of it, so where the two differ the dashes stand out
//   and where they agree the dashes sit on the thick line.
map.createPane('gpsPane').style.zIndex = 410;
map.createPane('finalPane').style.zIndex = 420;
map.createPane('classifierPane').style.zIndex = 430;
map.createPane('endpointPane').style.zIndex = 440;

// Base maps. If the tiles cannot load (no internet) the data still draws on
// the plain grey background, and the notice below tells the viewer why.
const streets = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);
streets.on('tileerror', () => { tileNotice.hidden = false; });
streets.on('tileload', () => { tileNotice.hidden = true; });
const plain = L.layerGroup();    // an empty base layer = "no map" (good for clean screenshots)

const layerControl = L.control.layers(
  { 'Street map': streets, 'Plain background': plain },
  {},                      // overlays are added per run
  { collapsed: false }
).addTo(map);

// Legend. The text here is fixed (nothing from the server), so innerHTML is safe.
const legend = L.control({ position: 'bottomleft' });
legend.onAdd = () => {
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML = `
    <h4>Legend</h4>
    <div class="legend-row">
      <svg width="34" height="14"><circle cx="17" cy="7" r="4" fill="${COLOUR_GPS}" stroke="#fff" stroke-width="1"/></svg>
      Raw GPS point
    </div>
    <div class="legend-row">
      <svg width="34" height="14"><line x1="1" y1="7" x2="33" y2="7" stroke="${COLOUR_FINAL}" stroke-width="6"/></svg>
      Final route (classifier + HMM)
    </div>
    <div class="legend-row">
      <svg width="34" height="14"><line x1="1" y1="7" x2="33" y2="7" stroke="${COLOUR_CLASSIFIER}" stroke-width="3" stroke-dasharray="7 5"/></svg>
      Classifier-only route
    </div>
    <div class="legend-note">A gap in a route = the route is disconnected there.</div>`;
  return div;
};
legend.addTo(map);

// The layers belonging to the current result, so the next run can remove them.
let currentLayers = [];

function clearMap() {
  for (const layer of currentLayers) {
    layerControl.removeLayer(layer);
    map.removeLayer(layer);
  }
  currentLayers = [];
}

/** Draw one run's GPS points and routes, then zoom to fit. */
function drawResult(data) {
  clearMap();

  const gps = data.gps || [];
  const finalPaths = (data.predicted_route && data.predicted_route.paths) || [];
  const classifierPaths = (data.classifier_route && data.classifier_route.paths) || [];

  // Raw GPS points: small circle markers. First and last are larger and labelled.
  const gpsLayer = L.featureGroup();
  for (const p of gps) {
    L.circleMarker(p, {
      pane: 'gpsPane', radius: 3, weight: 1,
      color: '#ffffff', fillColor: COLOUR_GPS, fillOpacity: 0.9,
    }).addTo(gpsLayer);
  }
  if (gps.length) {
    const endpoint = (point, label) => L.circleMarker(point, {
      pane: 'endpointPane', radius: 8, weight: 3,
      color: '#14213d', fillColor: '#ffffff', fillOpacity: 1,
    }).bindTooltip(label).addTo(gpsLayer);
    endpoint(gps[0], 'Start');
    endpoint(gps[gps.length - 1], 'End');
  }

  // Routes. `paths` is a LIST of polylines and L.polyline draws it as one
  // multi-polyline, so the gaps in the route stay real gaps. Never join them.
  const finalLayer = L.polyline(finalPaths, {
    pane: 'finalPane', color: COLOUR_FINAL, weight: 6, opacity: 0.9,
  });
  const classifierLayer = L.polyline(classifierPaths, {
    pane: 'classifierPane', color: COLOUR_CLASSIFIER, weight: 3, opacity: 0.95, dashArray: '8 8',
  });

  // Final route and GPS are on by default; the classifier route is a toggle.
  gpsLayer.addTo(map);
  finalLayer.addTo(map);
  layerControl.addOverlay(gpsLayer, 'Raw GPS points');
  layerControl.addOverlay(finalLayer, 'Final route (classifier + HMM)');
  layerControl.addOverlay(classifierLayer, 'Classifier-only route');
  currentLayers = [gpsLayer, finalLayer, classifierLayer];

  // Zoom to everything we drew.
  const bounds = L.latLngBounds([]);
  for (const layer of [gpsLayer, finalLayer, classifierLayer]) {
    const b = layer.getBounds();
    if (b.isValid()) bounds.extend(b);
  }
  // Generous padding keeps the route ends clear of the legend and layer control.
  if (bounds.isValid()) map.fitBounds(bounds, { padding: [60, 60] });

  mapMessage.hidden = true;
}

// ---------------------------------------------------------------------------
// 4. Results panel
// ---------------------------------------------------------------------------

/** Make an element with optional class and text. textContent (never innerHTML) so server text cannot inject markup. */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const isNumber = (v) => typeof v === 'number' && Number.isFinite(v);

/** 0.9655 -> "96.6%". Anything missing -> "n/a". */
function pct(v) { return isNumber(v) ? `${(v * 100).toFixed(1)}%` : 'n/a'; }

/** milliseconds -> "181 ms". */
function ms(v) { return isNumber(v) ? `${Math.round(v)} ms` : 'n/a'; }

/** Build the "difference" label shown under a metric: arrow + signed percentage points. */
function deltaLabel(finalValue, classifierValue) {
  if (!isNumber(finalValue) || !isNumber(classifierValue)) return null;
  const diff = (finalValue - classifierValue) * 100;           // percentage points
  if (Math.abs(diff) < 0.05) return { text: '＝ no change', cls: 'delta-flat' };
  return diff > 0
    ? { text: `▲ +${diff.toFixed(1)} pts`, cls: 'delta-up' }
    : { text: `▼ ${diff.toFixed(1)} pts`, cls: 'delta-down' };   // toFixed keeps the minus sign
}

/** One metric card: big final value, classifier-only beside it, and the difference. */
function metricCard(def, finalMetrics, classifierMetrics) {
  const finalValue = finalMetrics ? finalMetrics[def.key] : null;
  const classifierValue = classifierMetrics ? classifierMetrics[def.key] : null;

  const card = el('div', 'metric');
  card.appendChild(el('div', 'metric-name', def.label));
  card.appendChild(el('div', isNumber(finalValue) ? 'metric-final' : 'metric-final na', pct(finalValue)));

  const compare = el('div', 'metric-compare', `Classifier only: ${pct(classifierValue)}`);
  const delta = deltaLabel(finalValue, classifierValue);
  if (delta) {
    const badge = el('span', `metric-delta ${delta.cls}`, delta.text);
    badge.title = 'Final (classifier + HMM) minus classifier-only, in percentage points.';
    compare.appendChild(badge);
  }
  card.appendChild(compare);
  return card;
}

/** A small two-column table of label/value rows. */
function keyValueTable(rows) {
  const table = el('table', 'kv');
  for (const [label, value] of rows) {
    const tr = el('tr');
    tr.appendChild(el('th', null, label));
    tr.appendChild(el('td', null, value));
    table.appendChild(tr);
  }
  return table;
}

function clearResults() {
  resultsBox.replaceChildren(el('p', 'placeholder', 'Results appear here after you run a trip.'));
}

/** Show everything the API returned for one trip. `roundTripMs` is measured in the browser. */
function renderResults(data, roundTripMs) {
  resultsBox.replaceChildren();

  resultsBox.appendChild(el('h2', null, `Trip ${data.trip_id}`));
  const nArcs = data.predicted_route && data.predicted_route.arcs ? data.predicted_route.arcs.length : 'n/a';
  const nPieces = data.predicted_route && data.predicted_route.paths ? data.predicted_route.paths.length : 'n/a';
  resultsBox.appendChild(el('p', 'trip-meta',
    `${Number(data.n_points).toLocaleString()} GPS points · model ${data.model} · decoder ${data.decoder}`));
  // Route shape: how many road segments, and in how many unbroken pieces.
  // Several pieces means the route has gaps where it is disconnected.
  resultsBox.appendChild(el('p', 'trip-meta',
    `Predicted route: ${nArcs} road segments in ${nPieces} piece${nPieces === 1 ? '' : 's'}`));

  // Metrics. `metrics` is null when the server has no reference route for the
  // trip; show n/a instead of crashing.
  resultsBox.appendChild(el('h3', null, 'Accuracy'));
  const metrics = data.metrics || null;
  const grid = el('div', 'metric-grid');
  for (const def of METRICS) {
    grid.appendChild(metricCard(def, metrics && metrics.final, metrics && metrics.classifier_only));
  }
  resultsBox.appendChild(grid);

  if (!metrics) {
    resultsBox.appendChild(el('p', 'na-note',
      'Accuracy is n/a: the server has no reference route for this trip, so it could not score the prediction.'));
  } else if (metrics.final && isNumber(metrics.final.n_scored_points)) {
    resultsBox.appendChild(el('p', 'fine-print',
      `Scored on ${metrics.final.n_scored_points.toLocaleString()} GPS points.`));
  }

  // Timing: server-side steps from the API, plus the browser's own round trip.
  const t = data.timing_ms || {};
  resultsBox.appendChild(el('h3', null, 'Timing'));
  resultsBox.appendChild(keyValueTable([
    ['Server total', ms(t.total)],
    ['Browser round trip', ms(roundTripMs)],
  ]));
  resultsBox.appendChild(el('p', 'fine-print',
    `Server steps: load ${ms(t.load)} · features + classifier ${ms(t.features_and_classifier)} · HMM ${ms(t.hmm)}`));

  resultsBox.appendChild(el('p', 'fine-print',
    'Final route = classifier followed by HMM/Viterbi decoding. All numbers are computed by the API; this page only displays them.'));
}

// ---------------------------------------------------------------------------
// Errors and the Run button
// ---------------------------------------------------------------------------

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
function hideError() { errorBox.hidden = true; errorBox.textContent = ''; }

let running = false;

/** Run is allowed when a trip is chosen and nothing is already running. */
function updateRunButton() {
  runBtn.disabled = running || !tripSelect.value;
}

function setRunning(isRunning) {
  running = isRunning;
  runBtn.textContent = isRunning ? 'Running…' : 'Run';
  tripSelect.disabled = isRunning || tripSelect.options.length <= 1;
  connectBtn.disabled = isRunning;
  if (isRunning) {
    mapMessage.textContent = 'Running the pipeline…';
    mapMessage.classList.add('busy');
    mapMessage.hidden = false;
  }
  updateRunButton();
}

/** The main action: send the chosen trip id to the API and show what comes back. */
async function runPrediction() {
  const tripId = tripSelect.value;
  if (!tripId || running) return;

  setRunning(true);
  hideError();
  clearResults();      // never leave the previous trip's numbers next to a new run
  clearMap();

  const started = performance.now();
  try {
    const data = await apiPredict(tripId);
    const roundTripMs = performance.now() - started;

    if (!data || !Array.isArray(data.gps)) {
      throw new Error('The API response is missing the "gps" list, so there is nothing to draw.');
    }
    drawResult(data);
    renderResults(data, roundTripMs);
  } catch (err) {
    showError(err.message);                       // on screen, not just the console
    mapMessage.textContent = 'No result. See the error on the right.';
    mapMessage.classList.remove('busy');
    mapMessage.hidden = false;
    console.error(err);
  } finally {
    setRunning(false);
  }
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

apiInput.value = params.get('api') || DEFAULT_API;

if (MOCK) {
  $('mock-banner').hidden = false;
  apiInput.disabled = true;                       // the address is ignored in mock mode
}

apiForm.addEventListener('submit', (event) => { event.preventDefault(); connect(); });
tripSelect.addEventListener('change', updateRunButton);
runBtn.addEventListener('click', runPrediction);

connect();
