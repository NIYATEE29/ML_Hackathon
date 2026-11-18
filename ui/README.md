# PathSnap evaluator UI

A single-page browser UI for the evaluator. It lists the reserved trips, sends the
chosen trip id to the API, and draws the predicted route and the accuracy the API
returns. Plain HTML + CSS + JavaScript with Leaflet; no npm, no build step.

The API it talks to is described in [`api/CONTRACT.md`](../api/CONTRACT.md).

## Run it

Serve the `ui/` folder and open it in a browser:

```bash
cd ui
python -m http.server 8080
# then open http://localhost:8080
```

Opening `ui/index.html` directly also works for the real API. Mock mode needs the
page to be served (see below), because browsers block reading local files.

## Point it at the API

The API runs on the other laptop. Type its address into the **API address** box
and press **Connect**, or set it in the URL:

```
http://localhost:8080/?api=http://192.168.1.20:8000
```

The default is `http://localhost:8000`. The pill in the header shows whether
`GET /health` succeeded.

## Use it

1. Pick a trip in the dropdown (loaded from `GET /trips`).
2. Press **Run**. The UI sends `POST /predict {"trip_id": ...}`.
3. The map shows the GPS points, the final route (thick, solid) and, if you tick
   it in the layer control, the classifier-only route (thin, dashed).
4. The panel shows Top-1 accuracy, Macro F1, Route continuity and Route F1 for the
   final route, with the classifier-only value and the difference beside each, plus
   server timing and the browser's own round-trip time.

Errors (unknown trip, API unreachable, timeout) appear in a red box on the page.
If the API returns `metrics: null` the numbers show as n/a.

## Dev mode: mock data

```
http://localhost:8080/?mock=1
```

Reads `mock/health.json`, `mock/trips.json` and `mock/predict_<trip_id>.json`
instead of calling the API. Only trips `00000052` and `00000065` have mock files.
This is for development only; the real mode always calls the API.

## Rules this UI follows

- It never receives, requests or shows the reference ("ground truth") route.
- It does not compute accuracy. It displays the numbers in `metrics`; the only
  arithmetic is the difference between the final and classifier-only values.
- Nothing is hardcoded for specific trips.

## Files

| File | What it is |
|---|---|
| `index.html` | Page structure |
| `style.css` | Layout and colours |
| `app.js` | All behaviour: API calls, map drawing, results panel |
| `vendor/` | Leaflet 1.9.4 (JS, CSS, images), so the page needs no CDN |
| `mock/` | Sample API responses for development |

If the map tiles cannot load (no internet) a notice appears and the GPS points and
routes still draw on a plain background. The layer control also has a
"Plain background" option for clean screenshots.

## Colours

Okabe-Ito palette, chosen to stay distinguishable for colour-blind viewers. Line
style differs as well as colour, so colour is never the only cue.

| Layer | Colour | Style |
|---|---|---|
| GPS points | blue `#0072B2` | small dots |
| Final route (classifier + HMM) | vermillion `#D55E00` | thick, solid |
| Classifier-only route | purple `#CC79A7` | thin, dashed |
