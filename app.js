'use strict';

/* =============================================================================
   Photo → Drawing  —  app.js

   Everything here runs in the browser. The photo is read from the file picker,
   processed with plain JavaScript on <canvas> pixel arrays, and never sent
   anywhere.

   TABLE OF CONTENTS
     0. Settings & small helpers
     1. SHARED PIPELINE      (resize → gray → contrast stretch → blur → Sobel
                               → non-max suppression → hysteresis threshold
                               → contours → smooth → simplify → curves
                               → stroke ordering; the tracing + curve steps
                               turn every style's lines into pen strokes)
     2. MODE 1: LINE ART     (flowing XDoG lines, colored from the photo, glowing on black)
     3. PAINTING HELPERS     (paper, the "sketch first, then color it in"
                               live drawing, and the result object for
                               Cartoon and Illustration)
     4. WEB WORKER HELPER    (runs the heavy k-means step off the main thread)
     4b. FACE & BODY FINDER  (MediaPipe person/hair/skin segmentation and the
                               478 face landmarks, plus small geometry helpers)
     4c. MODES 2 & 3: CARTOON AND ILLUSTRATION
                             (one shared engine with two presets: edge-aware
                               smoothing, bold color, shading bands, and
                               pencil or ink lines)
     5. LIVE DRAWING         (requestAnimationFrame animation "pen")
     6. EXPORT & SHARING     (PNG, video recording, Web Share API)
     6b. MUSIC               (built-in songs made with Web Audio, or your own)
     7. UI WIRING            (DOM events, view switching)

   Sections 1–4c only work on numbers/arrays and canvases. They don't touch the
   page's buttons or layout, so they can later be moved to their own files
   (for example shared-pipeline.js, line-art.js, stylized.js).
   ============================================================================= */


/* =============================================================================
   0. SETTINGS & SMALL HELPERS
   ============================================================================= */

const SETTINGS = {
  maxSideLineArt: 1400,     // processing size (longest side) for Line art: full output size, for fine detail
  illustrationSide: 1400,   // Cartoon and Illustration work at full output size too, so edges stay sharp
  maxUpscale: 2.5,          // small photos are enlarged (up to 2.5x) before processing, so lines come out finer
  outputLongSide: 1400,     // the result canvas is always about this big, whatever the processing size
  rdpEpsilon: 0.6,          // Ramer–Douglas–Peucker tolerance, in pixels (lower = keeps more detail)
  maxStrokes: 12000,        // safety cap so extremely busy photos stay fast
  baseDurationMs: 8000,     // live drawing takes about 8 seconds at 1x speed
  holdFinalFrameMs: 1000,   // video keeps the finished picture on screen for 1 second
  paperColor: [245, 238, 222], // the blank paper shown before the colors are painted in
};

/** Keep a number inside [lo, hi]. */
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Smooth 0→1 ramp as x goes from a to b (flat at both ends, so masks built
 * from it have soft edges). Works "backwards" too, when a > b.
 */
const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

/** Create an off-screen canvas of a given size. */
function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** A promise that resolves on the next animation frame. Awaiting it lets the browser repaint (e.g. show a spinner). */
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/**
 * Convert RGB (0–255) to HSL. Hue is 0–360 degrees, saturation and lightness 0–1.
 * HSL is handy because "make it more colorful" = raise S, "make it brighter" = raise L.
 */
function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  let h = 0, s = 0;
  if (d > 1e-6) {
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
  }
  return [h, s, l];
}

/** Convert HSL back to RGB (0–255). */
function hslToRgb(h, s, l) {
  h = (((h % 360) + 360) % 360) / 360;
  if (s <= 0) { const v = l * 255; return [v, v, v]; }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const channel = (t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [channel(h + 1 / 3) * 255, channel(h) * 255, channel(h - 1 / 3) * 255];
}

/**
 * Smooth "value noise": random numbers on a coarse grid, blended smoothly in between.
 * Returns one value in [0, 1] per pixel. `cell` = grid spacing in pixels (bigger = blotchier).
 * Used for the paper texture and to make paint washes spread with ragged, natural edges.
 */
function makeValueNoise(w, h, cell) {
  const gw = Math.ceil(w / cell) + 2;
  const gh = Math.ceil(h / cell) + 2;
  const grid = new Float32Array(gw * gh);
  for (let i = 0; i < grid.length; i++) grid[i] = Math.random();

  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const gy = y / cell, iy = gy | 0, fy = gy - iy;
    const sy = fy * fy * (3 - 2 * fy); // "smoothstep" easing hides the grid
    for (let x = 0; x < w; x++) {
      const gx = x / cell, ix = gx | 0, fx = gx - ix;
      const sx = fx * fx * (3 - 2 * fx);
      const a = grid[iy * gw + ix], b = grid[iy * gw + ix + 1];
      const c = grid[(iy + 1) * gw + ix], d = grid[(iy + 1) * gw + ix + 1];
      out[y * w + x] = a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
    }
  }
  return out;
}


/* =============================================================================
   1. SHARED PIPELINE  (the stroke toolkit every style uses)
   -----------------------------------------------------------------------------
   Goal: turn a photo into a list of pen strokes. Each stroke is an ordered list
   of points that a "pen" can follow.

   photo → downscale → grayscale → blur → Sobel → non-max suppression
         → threshold → contour tracing → drop short / simplify → order strokes
         → smooth quadratic curve segments

   extractStrokes() runs all of it in one call (a classic "Canny" edge
   detector plus tracing). The three styles find their lines a more artistic
   way, with XDoG (section 4c), but they still use the downscale, blur and
   Sobel steps and then the tracing → simplify → order → curves steps here to
   turn those lines into pen strokes for the live drawing.
   ============================================================================= */

/**
 * STEP 1 — Downscale.
 * Big phone photos can be 12+ megapixels. Every step below loops over every
 * pixel, so we shrink the image first so the longest side is at most `maxSide`.
 * Shrinking in several halving steps (instead of one big jump) gives a cleaner,
 * less jagged result, like a mini "mipmap".
 * SMALL photos (e.g. a 400px profile picture) are gently ENLARGED toward
 * `maxSide` instead. That doesn't invent detail, but it gives edge detection
 * more room: lines can follow curves at a finer, sub-pixel-like precision,
 * and the result doesn't look chunky.
 * Returns { width, height, data } where data is RGBA bytes: [r,g,b,a, r,g,b,a, ...].
 */
function downscaleImage(source, maxSide) {
  const srcW = source.naturalWidth || source.width;
  const srcH = source.naturalHeight || source.height;
  const scale = Math.min(SETTINGS.maxUpscale, maxSide / Math.max(srcW, srcH));
  const w = Math.max(1, Math.round(srcW * scale));
  const h = Math.max(1, Math.round(srcH * scale));

  let current = source, curW = srcW, curH = srcH;
  while (curW / 2 > w && curH / 2 > h) {
    // Halve each step, but never make an intermediate canvas bigger than 4096px
    // (some phones refuse to create huge canvases).
    let nextW = Math.round(curW / 2), nextH = Math.round(curH / 2);
    const tooBig = Math.max(nextW, nextH) / 4096;
    if (tooBig > 1) { nextW = Math.round(nextW / tooBig); nextH = Math.round(nextH / tooBig); }
    const step = makeCanvas(nextW, nextH);
    const sctx = step.getContext('2d');
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(current, 0, 0, nextW, nextH);
    current = step; curW = nextW; curH = nextH;
  }

  const out = makeCanvas(w, h);
  const ctx = out.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(current, 0, 0, w, h);
  return { width: w, height: h, data: ctx.getImageData(0, 0, w, h).data };
}

/**
 * STEP 2 — Grayscale with weighted luminance.
 * Edges are about brightness changes, so we only need one number per pixel.
 * Our eyes are most sensitive to green and least to blue, so a plain average
 * looks wrong. The standard (Rec. 601) weights are 0.299 R + 0.587 G + 0.114 B.
 */
function toGrayscale(rgba, w, h) {
  const gray = new Float32Array(w * h);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  }
  return gray;
}

/**
 * STEP 2b — Contrast stretch (with a gentle shadow lift).
 * Night photos or dim indoor shots use only part of the 0–255 range, so their
 * edges are weak and get thrown away. We find the darkest 1% and brightest 1%
 * levels and stretch that range to fill 0–255 (like "auto levels" in a photo
 * editor). A mild gamma (power 0.8) then opens up the shadows, so details in
 * dark clothes and hair produce stronger edges. The gain is capped at 3x so
 * that a nearly flat image doesn't turn its noise into "edges".
 */
function stretchContrast(gray) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < gray.length; i++) hist[clamp(gray[i] | 0, 0, 255)]++;
  const cut = gray.length * 0.01;
  let lo = 0, hi = 255, acc = 0;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > cut) { lo = v; break; } }
  acc = 0;
  for (let v = 255; v >= 0; v--) { acc += hist[v]; if (acc > cut) { hi = v; break; } }
  const range = Math.max(hi - lo, 255 / 3);
  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    const n = clamp((gray[i] - lo) / range, 0, 1);
    out[i] = 255 * Math.pow(n, 0.8);
  }
  return out;
}

/**
 * STEP 3 — Light Gaussian blur.
 * Photos have sensor noise and fine texture (skin pores, grass). Without a
 * blur, Sobel would detect thousands of tiny fake edges. A Gaussian blur
 * replaces each pixel with a weighted average of its neighbors (closer
 * neighbors count more), using the 5-tap kernel [1 4 6 4 1] / 16.
 * It's "separable": blurring rows then columns gives the same result as a full
 * 2D 5×5 kernel but with 10 multiplications per pixel instead of 25.
 */
function gaussianBlur(src, w, h) {
  const k = [1, 4, 6, 4, 1];
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  // Horizontal pass
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -2; j <= 2; j++) s += src[row + clamp(x + j, 0, w - 1)] * k[j + 2];
      tmp[row + x] = s / 16;
    }
  }
  // Vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -2; j <= 2; j++) s += tmp[clamp(y + j, 0, h - 1) * w + x] * k[j + 2];
      out[y * w + x] = s / 16;
    }
  }
  return out;
}

/**
 * STEP 4 — Sobel edge detection.
 * An "edge" is where brightness changes quickly. Sobel estimates how fast it
 * changes in x and y using two small 3×3 kernels:
 *
 *        Gx:  -1  0 +1        Gy:  -1 -2 -1
 *             -2  0 +2              0  0  0
 *             -1  0 +1             +1 +2 +1
 *
 * Gx compares the right column with the left one; Gy compares bottom with top.
 * The middle row/column counts double, which also gives a little smoothing.
 * Together (gx, gy) is the *gradient*, an arrow pointing toward "brighter":
 *   magnitude = sqrt(gx² + gy²)  → how strong the edge is
 *   direction = atan2(gy, gx)    → which way the arrow points (perpendicular to the edge line)
 */
function sobelEdges(gray, w, h) {
  const magnitude = new Float32Array(w * h);
  const direction = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const tl = gray[i - w - 1], t = gray[i - w], tr = gray[i - w + 1];
      const l = gray[i - 1], r = gray[i + 1];
      const bl = gray[i + w - 1], b = gray[i + w], br = gray[i + w + 1];
      const gx = (tr + 2 * r + br) - (tl + 2 * l + bl);
      const gy = (bl + 2 * b + br) - (tl + 2 * t + tr);
      magnitude[i] = Math.sqrt(gx * gx + gy * gy);
      direction[i] = Math.atan2(gy, gx);
    }
  }
  return { magnitude, direction };
}

/**
 * STEP 5 — Non-maximum suppression (NMS).
 * Sobel edges are thick: a single boundary lights up as a band 3–5 pixels wide.
 * A pen line should be 1 pixel wide, so we keep only the *peak* of each band.
 *
 * For each pixel, look along the gradient direction (straight across the
 * edge) at the two neighbors on either side. If this pixel isn't at least as
 * strong as both, it's on the slope of the band, not the ridge, so we zero it.
 * The direction is rounded to one of 4 cases: horizontal, vertical, or one of
 * the two diagonals, because a pixel only has 8 neighbors.
 */
function nonMaxSuppression(mag, dir, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m === 0) continue;
      let deg = dir[i] * (180 / Math.PI);
      if (deg < 0) deg += 180; // a gradient pointing left is the same edge as one pointing right
      let n1, n2;
      if (deg < 22.5 || deg >= 157.5) { n1 = mag[i - 1]; n2 = mag[i + 1]; }                // gradient → : compare left/right
      else if (deg < 67.5)            { n1 = mag[i - w - 1]; n2 = mag[i + w + 1]; }        // gradient ↘ : compare diagonal
      else if (deg < 112.5)           { n1 = mag[i - w]; n2 = mag[i + w]; }                // gradient ↓ : compare up/down
      else                            { n1 = mag[i - w + 1]; n2 = mag[i + w - 1]; }        // gradient ↙ : other diagonal
      // ">=" on one side and ">" on the other so flat plateaus keep exactly one pixel
      if (m >= n1 && m > n2) out[i] = m;
    }
  }
  return out;
}

/**
 * STEP 6 — Double threshold + hysteresis → binary edge map (1 = edge, 0 = not).
 * This is the last step of the classic Canny edge detector.
 *
 * Picking the cutoff: a fixed number would behave very differently on a dim
 * photo vs. a high-contrast one, so the HIGH cutoff is a fraction of this
 * photo's own "strong edge" level (`ref`, the strength only the top 0.5% of
 * edge pixels reach). The "Edge detail" slider picks the fraction: more
 * detail → smaller fraction → more lines. An absolute floor always applies,
 * so a nearly blank photo gives almost no edges instead of amplified noise.
 *
 * Why two thresholds? With one cutoff you must choose between losing faint
 * but real lines (the soft edge of a cheek) and keeping lots of noisy specks.
 * Hysteresis gets both right:
 *   - pixels above HIGH are definitely edges ("strong");
 *   - pixels between LOW and HIGH ("weak") are kept ONLY if they connect,
 *     through other kept pixels, to a strong one.
 * We "flood fill" outward from every strong pixel into neighboring weak ones.
 * Real contours are continuous, so they get followed even where they fade;
 * isolated noise has no strong pixel to grow from and disappears. The result
 * is longer, unbroken lines with far fewer fragments.
 * `strictness` > 1 raises both cutoffs (for cleaner, calmer outlines).
 */
function thresholdEdges(thin, w, h, detail, strictness = 1, gain = 1, lowRatio = 0.4) {
  const BINS = 2048; // Sobel magnitudes on 0–255 input stay below ~1450
  const hist = new Uint32Array(BINS);
  let ridgePixels = 0;
  for (let i = 0; i < thin.length; i++) {
    if (thin[i] > 0) { hist[Math.min(BINS - 1, thin[i] | 0)]++; ridgePixels++; }
  }
  // Walk down from the strongest bin until 0.5% of ridge pixels are covered.
  let ref = 0, acc = 0;
  for (let b = BINS - 1; b >= 0; b--) {
    acc += hist[b];
    if (acc >= ridgePixels * 0.005) { ref = b; break; }
  }

  const d = clamp(detail, 1, 100) / 100;
  // Fraction of `ref` needed: 0.6 at detail 1 down to 0.05 at detail 100 (exponential, so the slider feels even).
  const fraction = 0.6 * Math.pow(0.05 / 0.6, d);
  // Floor: ~50 at low detail, ~20 at high (a Sobel value of 40 ≈ a 10-gray-level step).
  // `gain` > 1 when the photo was enlarged: enlarging spreads each brightness
  // step over more pixels, so per-pixel gradients shrink by about that factor.
  // Lowering the floor by the same factor keeps real edges from being dropped.
  const floor = (20 + (1 - d) * 30) / gain;
  const high = Math.max(floor, ref * fraction) * strictness;
  const low = Math.max(high * lowRatio, 12 / gain);

  const edges = new Uint8Array(w * h);
  const stack = new Int32Array(w * h); // pixels waiting to spread to their neighbors
  let top = 0, count = 0;

  // Seed with every strong pixel.
  for (let i = 0; i < thin.length; i++) {
    if (thin[i] >= high) { edges[i] = 1; stack[top++] = i; count++; }
  }
  // Grow into connected weak pixels (8 neighbors).
  while (top > 0) {
    const i = stack[--top];
    const x = i % w, y = (i / w) | 0;
    for (let dy = -1; dy <= 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        if ((dx === 0 && dy === 0) || nx < 0 || nx >= w) continue;
        const j = ny * w + nx;
        if (edges[j] === 0 && thin[j] >= low) { edges[j] = 1; stack[top++] = j; count++; }
      }
    }
  }
  return { edges, count };
}

// The 8 neighbors of a pixel, in clockwise order starting at "east".
// Keeping them in circular order lets us say "try the direction I was already
// going first, then the ones just left/right of it" by adding small offsets.
const NEIGHBOR_DX = [1, 1, 0, -1, -1, -1, 0, 1];
const NEIGHBOR_DY = [0, 1, 1, 1, 0, -1, -1, -1];
const TURN_PREFERENCE = [0, 1, -1, 2, -2, 3, -3];

/**
 * STEP 7 — Contour tracing.
 * The edge map is just scattered "on" pixels. A pen needs *ordered paths*.
 * We scan the image; when we find an edge pixel we haven't used yet, we walk
 * from it to a connected (8-neighbor) unvisited edge pixel, then from that one
 * to the next, and so on, marking each pixel visited so it's used once.
 * While walking we prefer to keep going straight (then turn slightly, then
 * more), which keeps lines smooth at junctions.
 * The starting pixel might be in the *middle* of a line, so we walk both ways
 * from it and join the two halves: reverse(backward) + start + forward.
 * Returns an array of paths, each an array of {x, y}.
 */
function traceContours(edges, w, h) {
  const visited = new Uint8Array(w * h);
  const paths = [];

  const isFree = (x, y) =>
    x >= 0 && y >= 0 && x < w && y < h && edges[y * w + x] === 1 && visited[y * w + x] === 0;

  // Follow the line from (x, y) until it ends; returns the points found (not including the start).
  function walk(x, y) {
    const pts = [];
    let prevDir = -1;
    for (;;) {
      let found = -1;
      if (prevDir === -1) {
        for (let d = 0; d < 8; d++) {
          if (isFree(x + NEIGHBOR_DX[d], y + NEIGHBOR_DY[d])) { found = d; break; }
        }
      } else {
        for (const turn of TURN_PREFERENCE) {
          const d = (prevDir + turn + 8) % 8;
          if (isFree(x + NEIGHBOR_DX[d], y + NEIGHBOR_DY[d])) { found = d; break; }
        }
      }
      if (found === -1) return pts; // dead end: the line is finished
      x += NEIGHBOR_DX[found];
      y += NEIGHBOR_DY[found];
      visited[y * w + x] = 1;
      pts.push({ x, y });
      prevDir = found;
    }
  }

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (edges[i] !== 1 || visited[i]) continue;
      visited[i] = 1;
      const forward = walk(x, y);
      const backward = walk(x, y);
      paths.push(backward.reverse().concat([{ x, y }], forward));
    }
  }
  return paths;
}

/**
 * STEP 8a — Ramer–Douglas–Peucker (RDP) simplification.
 * A traced path has one point per pixel, which is far more than needed and
 * looks jaggy. RDP keeps only the "important" corner points:
 *   1. Draw a straight line from the first point to the last.
 *   2. Find the point farthest from that line.
 *   3. If it's farther than `epsilon`, keep it and repeat on both halves;
 *      otherwise every point in between can be dropped.
 * Written with an explicit stack instead of recursion so very long paths can't
 * overflow the call stack.
 */
function rdpSimplify(points, epsilon) {
  const n = points.length;
  if (n < 3) return points.slice();
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const stack = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const ax = points[a].x, ay = points[a].y;
    const dx = points[b].x - ax, dy = points[b].y - ay;
    const len = Math.hypot(dx, dy);
    let maxD = -1, idx = -1;
    for (let i = a + 1; i < b; i++) {
      const px = points[i].x - ax, py = points[i].y - ay;
      // distance from point to the line a→b (or to point a if a and b coincide, e.g. a closed loop)
      const d = len < 1e-9 ? Math.hypot(px, py) : Math.abs(dy * px - dx * py) / len;
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (maxD > epsilon) {
      keep[idx] = 1;
      stack.push([a, idx], [idx, b]);
    }
  }
  return points.filter((_, i) => keep[i] === 1);
}

/**
 * STEP 8a' — Smooth a traced path (remove the pixel "staircase").
 * Traced points sit exactly on the pixel grid, so a gentle diagonal line
 * becomes a jittery staircase: right, right, down, right, right, down...
 * Replacing each point with a weighted average of itself and its neighbors
 * (weights 1-2-1, done twice) turns the steps into the smooth line the edge
 * really follows. Points become fractional ("sub-pixel") coordinates. The two
 * end points stay put so strokes still meet where they should.
 */
function smoothPath(points, passes = 2) {
  let pts = points;
  for (let p = 0; p < passes; p++) {
    const n = pts.length;
    if (n < 3) return pts;
    const next = new Array(n);
    next[0] = pts[0];
    next[n - 1] = pts[n - 1];
    for (let i = 1; i < n - 1; i++) {
      next[i] = {
        x: (pts[i - 1].x + 2 * pts[i].x + pts[i + 1].x) / 4,
        y: (pts[i - 1].y + 2 * pts[i].y + pts[i + 1].y) / 4,
      };
    }
    pts = next;
  }
  return pts;
}

/**
 * STEP 8b — Discard very short paths, then smooth and simplify the rest.
 * Tiny paths (a few pixels) are almost always noise or texture specks.
 * `rawLength` (the pixel count before simplifying) is kept for ranking later.
 */
function filterAndSimplify(paths, minPoints, epsilon) {
  const strokes = [];
  for (const path of paths) {
    if (path.length < minPoints) continue;
    strokes.push({ points: rdpSimplify(smoothPath(path), epsilon), rawLength: path.length });
  }
  return strokes;
}

/**
 * STEP 8c' — Texture suppression.
 * Patterned things (carpet, polka dots, knitted fabric, grass, gravel)
 * produce hundreds of short edge fragments that turn a drawing into
 * scribbles. Real contours (a jaw, an arm, a couch edge) are LONG, while
 * texture is lots of SHORT lines crammed close together.
 * So we split the image into a grid of small cells and measure how much
 * traced line falls in each cell (the "line density"). A short stroke that
 * lives mostly in dense cells is texture, and is dropped. Long strokes always
 * survive, even when they run through a busy area.
 */
function suppressTexture(strokes, w, h, { maxDensity = 0.09, longLen = 0 } = {}) {
  const cell = Math.max(12, Math.round(Math.max(w, h) / 60));
  const gw = Math.ceil(w / cell), gh = Math.ceil(h / cell);
  const count = new Float32Array(gw * gh);
  for (const s of strokes) {
    const perPoint = s.rawLength / s.points.length; // simplified points stand for this many traced pixels each
    for (const p of s.points) count[clamp(p.y / cell | 0, 0, gh - 1) * gw + clamp(p.x / cell | 0, 0, gw - 1)] += perPoint;
  }
  // Density = traced pixels per pixel of cell area, averaged over each cell's 3×3 neighborhood.
  const density = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      let sum = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || xx < 0 || yy >= gh || xx >= gw) continue;
        sum += count[yy * gw + xx]; n++;
      }
      density[y * gw + x] = sum / n / (cell * cell);
    }
  }
  const minLong = longLen || Math.max(w, h) * 0.06;
  return strokes.filter((s) => {
    if (s.rawLength >= minLong || s.feature) return true;
    let d = 0;
    for (const p of s.points) d += density[clamp(p.y / cell | 0, 0, gh - 1) * gw + clamp(p.x / cell | 0, 0, gw - 1)];
    return d / s.points.length < maxDensity;
  });
}

/**
 * STEP 9 — Order strokes naturally.
 * An artist blocks in the big shapes first, then fills in details near where
 * the pen already is. We copy that:
 *   1. The longest ~8% of strokes ("prominent" ones) go first.
 *   2. Within each group, a greedy nearest-neighbor tour: after finishing a
 *      stroke, pick the remaining stroke whose start OR end is closest to the
 *      pen (reversing it if its end is closer). This minimizes pen jumps.
 * Greedy nearest-neighbor isn't the perfect shortest tour, but it's simple,
 * fast enough for thousands of strokes, and looks natural.
 */
function orderStrokes(strokes) {
  if (strokes.length < 2) return strokes.slice();
  const sorted = strokes.slice().sort((a, b) => b.rawLength - a.rawLength);
  const prominentCount = Math.min(60, Math.max(1, Math.round(sorted.length * 0.08)));
  const result = [];

  function greedyTour(group, penX, penY) {
    const remaining = group.slice();
    while (remaining.length) {
      let best = 0, bestD = Infinity, bestReverse = false;
      for (let i = 0; i < remaining.length; i++) {
        const pts = remaining[i].points;
        const s = pts[0], e = pts[pts.length - 1];
        const ds = (s.x - penX) ** 2 + (s.y - penY) ** 2;
        const de = (e.x - penX) ** 2 + (e.y - penY) ** 2;
        if (ds < bestD) { bestD = ds; best = i; bestReverse = false; }
        if (de < bestD) { bestD = de; best = i; bestReverse = true; }
      }
      const stroke = remaining[best];
      remaining[best] = remaining[remaining.length - 1]; // O(1) removal: swap with last, then pop
      remaining.pop();
      if (bestReverse) stroke.points.reverse();
      result.push(stroke);
      const end = stroke.points[stroke.points.length - 1];
      penX = end.x; penY = end.y;
    }
    return [penX, penY];
  }

  const start = sorted[0].points[0];
  const pen = greedyTour(sorted.slice(0, prominentCount), start.x, start.y);
  greedyTour(sorted.slice(prominentCount), pen[0], pen[1]);
  return result;
}

/**
 * STEP 8c — Turn simplified points into smooth quadratic Bézier segments.
 * Connecting points with straight lines looks robotic. The classic trick:
 * curve through the *midpoints* between points, using each actual point as the
 * curve's control ("pull") point. The result passes smoothly around corners.
 * Each segment stores its start (x0,y0), control (cx,cy), end (x1,y1) and an
 * approximate length (so the live pen can move at a constant speed).
 */
function buildSegments(points) {
  const segs = [];
  const add = (x0, y0, cx, cy, x1, y1) => {
    // A quadratic curve's length lies between its chord and its control polygon; averaging is a good estimate.
    const chord = Math.hypot(x1 - x0, y1 - y0);
    const poly = Math.hypot(cx - x0, cy - y0) + Math.hypot(x1 - cx, y1 - cy);
    segs.push({ x0, y0, cx, cy, x1, y1, len: Math.max(0.01, (chord + poly) / 2) });
  };
  const n = points.length;
  if (n === 2) {
    const [a, b] = points;
    add(a.x, a.y, (a.x + b.x) / 2, (a.y + b.y) / 2, b.x, b.y); // a straight line is a curve whose control sits in the middle
    return segs;
  }
  let sx = points[0].x, sy = points[0].y;
  for (let i = 1; i < n - 1; i++) {
    const p = points[i], q = points[i + 1];
    const mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
    add(sx, sy, p.x, p.y, mx, my);
    sx = mx; sy = my;
  }
  const last = points[n - 1];
  add(sx, sy, (sx + last.x) / 2, (sy + last.y) / 2, last.x, last.y);
  return segs;
}

/** Add a stroke's smooth curve to the current canvas path (caller does beginPath/stroke). */
function drawSmoothPath(ctx, segs) {
  ctx.moveTo(segs[0].x0, segs[0].y0);
  for (const s of segs) ctx.quadraticCurveTo(s.cx, s.cy, s.x1, s.y1);
}

/**
 * Run the whole shared pipeline (steps 2–9) on an RGBA image.
 * opts: { detail (1–100), strictness (1 = normal, higher = fewer lines), minPoints }
 * Returns strokes: [{ points, rawLength, segs, length }]
 */
function extractStrokes(rgba, w, h, opts) {
  return finalizeStrokes(extractRawStrokes(rgba, w, h, opts));
}

/** Steps 2–8: image → simplified point lists (not yet ordered or turned into curves). */
function extractRawStrokes(rgba, w, h, opts) {
  const gray = stretchContrast(toGrayscale(rgba, w, h));
  const blurred = gaussianBlur(gray, w, h);
  const { magnitude, direction } = sobelEdges(blurred, w, h);
  const thin = nonMaxSuppression(magnitude, direction, w, h);
  const { edges } = thresholdEdges(thin, w, h, opts.detail, opts.strictness, opts.gain || 1, opts.lowRatio || 0.4);
  const paths = traceContours(edges, w, h);
  return filterAndSimplify(paths, opts.minPoints, SETTINGS.rdpEpsilon);
}

/** Step 9 + curves: cap the count, order for natural drawing, build smooth segments. */
function finalizeStrokes(strokes) {
  if (strokes.length > SETTINGS.maxStrokes) {
    strokes.sort((a, b) => b.rawLength - a.rawLength);
    strokes.length = SETTINGS.maxStrokes;
  }
  strokes = orderStrokes(strokes);

  for (const s of strokes) {
    s.segs = buildSegments(s.points);
    s.length = s.segs.reduce((sum, seg) => sum + seg.len, 0);
  }
  return strokes;
}

/**
 * Canvas transform for drawing strokes: scale processing-pixel coordinates up
 * to the output size, and shift by half a pixel so lines sit on pixel centers.
 */
function setStrokeTransform(ctx, scale) {
  ctx.setTransform(scale, 0, 0, scale, 0.5 * scale, 0.5 * scale);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
}

/** How much to enlarge processing coordinates so the result canvas is ~SETTINGS.outputLongSide px. */
const outputScale = (w, h) => SETTINGS.outputLongSide / Math.max(w, h);

/**
 * How "important" a stroke is, from 0 (tiny detail) to 1 (a major contour),
 * based on its traced length. Used to vary line weight like a real artist:
 * bold outer contours, light and thin inner details.
 */
const strokeProminence = (stroke) => Math.sqrt(Math.min(1, stroke.rawLength / 240));

/** Reset the canvas state we change (transform, glow, transparency). */
function resetCtx(ctx) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.shadowBlur = 0;
  ctx.shadowColor = 'transparent';
  ctx.globalAlpha = 1;
}


/* =============================================================================
   2. MODE 1: LINE ART  —  glowing, colorful lines on black
   ============================================================================= */

/**
 * Make a color "neon": convert to HSL, multiply saturation and keep lightness
 * in a bright band, so every stroke glows on black while still hinting at the
 * photo's real color. Saturation is multiplied (not shifted up), so colorful
 * areas get vivid while gray areas (dark clothes, concrete) stay a soft,
 * almost-white glow instead of all turning the same orange.
 */
function vividColor(r, g, b) {
  const [R, G, B] = vividRGB(r, g, b);
  return `rgb(${R | 0}, ${G | 0}, ${B | 0})`;
}

/** The same "neon" color as vividColor, as an [r, g, b] array. */
function vividRGB(r, g, b) {
  const [hue, s, l] = rgbToHsl(r, g, b);
  const s2 = Math.min(1, s * 2.2);
  const l2 = clamp(0.56 + l * 0.3, 0.58, 0.85);
  return hslToRgb(hue, s2, l2);
}

/**
 * The finished Line art, rendered straight from the flow-based ink (so every
 * fine line and its natural thick-to-thin weight is kept, which traced pen
 * strokes lose). Each ink pixel takes a vivid version of the photo's color
 * there; a glow is made by shrinking the lines to 1/4 and 1/10 size and
 * stretching them back up (a cheap, soft blur), then adding it underneath
 * with the "lighter" blend, which makes light add up like real light.
 */
function renderGlowInk(ink, photo, w, h) {
  const scale = Math.max(w, h) / 1000;
  const n = w * h;
  // Color source: a slightly blurred photo, so a line takes the color of
  // its surroundings rather than one noisy pixel.
  const chan = (c) => gaussianBlurSigma(Float32Array.from({ length: n }, (_, i) => photo[i * 4 + c]), w, h, 2 * scale);
  const R = chan(0), G = chan(1), B = chan(2);
  const lines = new Uint8ClampedArray(n * 4);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const a = 1 - ink[i];
    if (a < 0.03) continue;
    const [r, g, b] = vividRGB(R[i], G[i], B[i]);
    lines[j] = r; lines[j + 1] = g; lines[j + 2] = b; lines[j + 3] = 255 * a;
  }
  const lineCanvas = makeCanvas(w, h);
  lineCanvas.getContext('2d').putImageData(new ImageData(lines, w, h), 0, 0);

  const out = makeCanvas(w, h);
  const octx = out.getContext('2d');
  octx.fillStyle = '#000';
  octx.fillRect(0, 0, w, h);
  octx.globalCompositeOperation = 'lighter';
  octx.imageSmoothingQuality = 'high';
  for (const [div, alpha] of [[4, 0.55], [10, 0.45]]) {
    const small = makeCanvas(Math.max(1, Math.round(w / div)), Math.max(1, Math.round(h / div)));
    const sctx = small.getContext('2d');
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(lineCanvas, 0, 0, small.width, small.height);
    octx.globalAlpha = alpha;
    octx.drawImage(small, 0, 0, w, h);
  }
  octx.globalAlpha = 1;
  octx.drawImage(lineCanvas, 0, 0);
  return out;
}

/**
 * Pick a stroke's color from the ORIGINAL photo along the stroke, then make it vivid.
 * An edge sits between two regions (e.g. a red roof and a gray sky), so at
 * each point we look at a small 5×5 neighborhood and weight colorful pixels
 * more (weight = 0.35 + saturation). That makes the line lean toward the
 * more interesting side instead of a muddy average.
 */
function sampleStrokeColor(stroke, rgba, w, h) {
  let r = 0, g = 0, b = 0, total = 0;
  for (const p of stroke.points) {
    const px = Math.round(p.x), py = Math.round(p.y); // smoothed points are fractional
    for (let dy = -2; dy <= 2; dy += 2) {
      for (let dx = -2; dx <= 2; dx += 2) {
        const x = clamp(px + dx, 0, w - 1), y = clamp(py + dy, 0, h - 1);
        const i = (y * w + x) * 4;
        const pr = rgba[i], pg = rgba[i + 1], pb = rgba[i + 2];
        const sat = (Math.max(pr, pg, pb) - Math.min(pr, pg, pb)) / 255; // quick saturation estimate
        const weight = 0.35 + sat;
        r += pr * weight; g += pg * weight; b += pb * weight; total += weight;
      }
    }
  }
  return vividColor(r / total, g / total, b / total);
}

/**
 * Canvas style for one glowing line. Major contours are thicker, brighter and
 * glow more; small details are fine, fainter hairlines.
 * lineWidth is in processing pixels (the canvas is scaled), so we divide by
 * `scale` to think in output pixels. shadowBlur (the glow) ignores the canvas
 * scale and is always in output pixels.
 */
function styleLineArtStroke(ctx, stroke, scale) {
  const p = stroke.prominence;
  ctx.strokeStyle = stroke.color;
  ctx.shadowColor = stroke.color;
  ctx.shadowBlur = 2 + 6 * p;
  ctx.lineWidth = (1.2 + 2.2 * p) / scale; // 1.2–3.4 px on the 1400px output: fine details, bold contours
  ctx.globalAlpha = 0.7 + 0.3 * p;
}

/** Draw the complete line-art picture at once. */
function drawLineArtFinal(ctx, art) {
  resetCtx(ctx);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (art.inkCanvas) {
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(art.inkCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }
  // Pen strokes: all of them without the ink picture (older pipeline), or
  // just the precise face-feature outlines on top of it.
  setStrokeTransform(ctx, art.scale);
  ctx.globalCompositeOperation = art.inkCanvas ? 'lighter' : 'source-over';
  for (const s of art.strokes) {
    if (art.inkCanvas && !s.feature) continue;
    styleLineArtStroke(ctx, s, art.scale);
    ctx.beginPath();
    drawSmoothPath(ctx, s.segs);
    ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';
  resetCtx(ctx);
}

/**
 * Live version: returns drawAt(progress) where progress goes 0 → 1.
 * The pen moves at a constant speed along the total length of all strokes.
 * With the ink picture, the pen finishes at 88% of the time and the last
 * 12% cross-fades into the finished, full-detail ink.
 */
function createLineArtDrawer(ctx, art) {
  resetCtx(ctx);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const pen = createStrokePen(art.strokes, (c, s) => styleLineArtStroke(c, s, art.scale));
  const penShare = art.inkCanvas ? 0.88 : 1;
  let sketch = null; // snapshot of the finished pen sketch, for the cross-fade
  return (progress) => {
    if (progress <= penShare || !art.inkCanvas) {
      setStrokeTransform(ctx, art.scale);
      pen.advanceTo(ctx, Math.min(1, progress / penShare) * pen.totalLength);
      resetCtx(ctx);
      return;
    }
    if (!sketch) {
      setStrokeTransform(ctx, art.scale);
      pen.advanceTo(ctx, Infinity);
      resetCtx(ctx);
      sketch = makeCanvas(ctx.canvas.width, ctx.canvas.height);
      sketch.getContext('2d').drawImage(ctx.canvas, 0, 0);
    }
    const q = clamp((progress - penShare) / (1 - penShare), 0, 1);
    resetCtx(ctx);
    ctx.drawImage(sketch, 0, 0);
    ctx.globalAlpha = q;
    const finalCanvas = art.finalCanvas || (art.finalCanvas = (() => {
      const c = makeCanvas(ctx.canvas.width, ctx.canvas.height);
      drawLineArtFinal(c.getContext('2d'), art);
      return c;
    })());
    ctx.drawImage(finalCanvas, 0, 0);
    ctx.globalAlpha = 1;
  };
}

/** Build the Line art result from an <img>. */
async function buildLineArt(image, detail, onStatus = () => {}) {
  const { width: w, height: h, data } = downscaleImage(image, SETTINGS.maxSideLineArt);
  const d = clamp(detail, 1, 100) / 100;
  const sizeFactor = Math.max(w, h) / 800;
  const scale = Math.max(w, h) / 1000;

  // How the lines are found (the same ink method as the Illustration mode,
  // section 4c, which captures far more real detail than edge tracing):
  // 1. Edge-aware smoothing (domain transform) removes noise and fine
  //    texture but keeps every real edge, including thin ones like hair
  //    strands and lashes. Color edges count too (the guide includes color).
  // 2. XDoG ink finds lines: bold on strong edges, lighter on soft ones.
  // 3. Specks are removed, lines are thinned to 1-pixel center lines, and
  //    those are traced into pen strokes (steps 7–9 of the shared pipeline).
  onStatus('Finding the lines…');
  await nextFrame();
  // Dark photos: brighten a copy for line finding only, so faint edges in
  // near-black areas (dark clothes, hair at night) become visible.
  let lineSource = data;
  const avgLight = averageLightness(data);
  if (avgLight < 0.38) {
    lineSource = new Uint8ClampedArray(data);
    const lut = makeLevelsLut(lineSource);
    const strength = clamp((0.38 - avgLight) / 0.18, 0, 1);
    for (let v = 0; v < 256; v++) lut[v] = v + (lut[v] - v) * strength;
    applyLut(lineSource, lut);
  }
  const lab = rgbaToLab(lineSource);
  const guide = [new Float32Array(lab.L), lab.A, lab.B];
  domainTransformSmooth([lab.L], guide, w, h, 16 * scale, 3 + (1 - d) * 6, 3);
  const flow = edgeTangentFlow(lab.L, w, h, 3 * scale);
  // tau = 1 ("pure" difference of Gaussians): only real edges become lines.
  // (The Illustration uses tau < 1, which also inks very dark areas: good
  // shading for a painting, but it would flood dark clothes in a line drawing.)
  const main = xdogLines(lab.L, w, h, 1.0 * scale, { tau: 1, eps: -1.1 + d * 0.8, phi: 2.2, flow: { ...flow, length: 14 * scale } });
  // A second, finer layer (smaller blur, shorter flow) catches small details:
  // lashes, creases, small folds and texture edges. It's drawn a bit lighter,
  // so the main flowing lines still lead.
  const fine = xdogLines(lab.L, w, h, 0.6 * scale, { tau: 1, eps: -1.0 + d * 0.7, phi: 2.2, flow: { ...flow, length: 10 * scale } });
  const ink = new Float32Array(w * h);
  for (let i = 0; i < ink.length; i++) ink[i] = Math.min(main[i], 1 - (1 - fine[i]) * 0.75);
  removeInkSpecks(ink, w, h, Math.round(6 * scale * scale));
  // Grow the ink by 1 pixel before thinning ("dilation"): it bridges tiny
  // 1–2 px gaps, so a hair strand that the ink broke into dashes becomes one
  // continuous line again.
  const inkBin = new Uint8Array(w * h);
  for (let i = 0; i < inkBin.length; i++) inkBin[i] = ink[i] < 0.5 ? 1 : 0;
  const bin = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      bin[i] = inkBin[i] | inkBin[i - 1] | inkBin[i + 1] | inkBin[i - w] | inkBin[i + w] |
        inkBin[i - w - 1] | inkBin[i - w + 1] | inkBin[i + w - 1] | inkBin[i + w + 1];
    }
  }
  thinLines(bin, w, h);
  let raw = filterAndSimplify(traceContours(bin, w, h), Math.max(4, Math.round(3.5 * sizeFactor)), SETTINGS.rdpEpsilon);
  raw = suppressTexture(raw, w, h, { maxDensity: 0.15 }); // looser than before, so more detail survives
  // How strong each line is (how dark its ink was), for line weight later.
  const inkDark = blurTimes(Float32Array.from(ink, (v) => 1 - v), w, h, 1);
  for (const s of raw) {
    let sum = 0;
    for (const p of s.points) sum += inkDark[clamp(Math.round(p.y), 0, h - 1) * w + clamp(Math.round(p.x), 0, w - 1)];
    s.strength = sum / s.points.length;
  }

  // Faces deserve extra care: find them, re-trace each one from a zoomed-in
  // crop, then add precise feature outlines from the face landmarks.
  let faces = [], landmarkFaces = [];
  try {
    onStatus('Looking for faces…');
    const c = makeCanvas(w, h);
    c.getContext('2d').putImageData(new ImageData(data, w, h), 0, 0);
    landmarkFaces = await withTimeout(findFaceLandmarks(c), 20000);
    const k = (image.naturalWidth || image.width) / w; // processing pixels → source pixels
    faces = landmarkFaces.map((pts) => {
      const xs = pts.map((p) => p.x * k), ys = pts.map((p) => p.y * k);
      return { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) };
    });
  } catch (err) {
    console.warn('Face finder unavailable, drawing without the face pass:', err);
  }
  if (faces.length) {
    // The ink already captures the face's detail at full size; the landmarks
    // add clean, complete outlines of the eyes, irises, brows, nose and lips.
    onStatus('Sketching facial features…');
    raw = addLandmarkFeatureLines(raw, landmarkFaces);
  }

  const strokes = finalizeStrokes(raw);
  // Line weight: half from how dark the ink was, half from the stroke's
  // length (relative to an 800px image, so weights look the same at any size).
  // Bold major contours, fine detail lines.
  for (const s of strokes) {
    s.color = sampleStrokeColor(s, data, w, h);
    const lengthPart = Math.sqrt(Math.min(1, s.rawLength / (240 * sizeFactor)));
    s.prominence = s.feature ? 0.7 : clamp(0.5 * lengthPart + 0.6 * (s.strength ?? 0.5), 0, 1);
  }
  const totalLength = strokes.reduce((sum, s) => sum + s.length, 0);
  onStatus('Adding the glow…');
  await nextFrame();
  const inkCanvas = renderGlowInk(ink, data, w, h);

  return {
    mode: 'lineart',
    width: w,
    height: h,
    scale: outputScale(w, h),
    strokes,
    inkCanvas,
    isEmpty: strokes.length === 0 || totalLength < 40,
    faceCount: faces.length,
    note: strokes.length < 15 ? 'Only a few lines were found. Try raising "Edge detail".' : '',
    drawFinal(ctx) { drawLineArtFinal(ctx, this); },
    createLiveDrawer(ctx) { return createLineArtDrawer(ctx, this); },
  };
}


/* =============================================================================
   3. PAINTING HELPERS  (used by Cartoon and Illustration, section 4c)
   -----------------------------------------------------------------------------
   Section 4c makes the finished picture. This section turns it into a live
   drawing, the way an artist works: the pen sketches the lines on blank
   paper first, then the colors are washed in, region by region.
     - k-means groups the finished picture's pixels into 12 color regions,
       and computeRevealOrder() decides when each pixel gets its color.
     - makePaperGrain() / composePaper(): the textured blank paper.
     - makePaintedArt() packages everything into the result object.
   (Kuwahara is the "oil paint" smoother. paintStage() can run it before
   k-means; the current styles skip it because 4c already smooths the photo.)

   NOTE: kuwaharaFilter, kmeansPlusPlusInit, kmeansQuantize and paintStage are
   copied into a Web Worker as text (see section 4), so they must be fully
   self-contained: they can't use helpers or constants from outside themselves.
   ============================================================================= */

/**
 * Kuwahara filter — the "oil paint" smoother.
 * A normal blur smears across edges. Kuwahara flattens texture but keeps
 * edges sharp, which is exactly how painted regions look.
 *
 * For each pixel, look at 4 overlapping square windows around it (top-left,
 * top-right, bottom-left, bottom-right). Measure how "busy" each window is
 * (the variance of brightness). Take the average color of the CALMEST window.
 * Near an edge, the calmest window is the one entirely on one side of the
 * edge, so the pixel copies that side and the edge stays crisp.
 *
 * Speed trick: "summed-area tables" (integral images). S[y][x] holds the sum of
 * all pixels above-left of (x, y), so the sum of ANY rectangle takes just 4
 * lookups. That makes the window size free: O(1) per window instead of O(r²).
 */
function kuwaharaFilter(rgba, w, h, radius) {
  const W1 = w + 1, size = W1 * (h + 1);
  const sR = new Float64Array(size), sG = new Float64Array(size), sB = new Float64Array(size);
  const sL = new Float64Array(size), sL2 = new Float64Array(size);

  // Build the summed-area tables (one row at a time with running row sums).
  for (let y = 0; y < h; y++) {
    let rR = 0, rG = 0, rB = 0, rL = 0, rL2 = 0;
    for (let x = 0; x < w; x++) {
      const j = (y * w + x) * 4;
      const r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      rR += r; rG += g; rB += b; rL += lum; rL2 += lum * lum;
      const k = (y + 1) * W1 + (x + 1), up = k - W1;
      sR[k] = sR[up] + rR; sG[k] = sG[up] + rG; sB[k] = sB[up] + rB;
      sL[k] = sL[up] + rL; sL2[k] = sL2[up] + rL2;
    }
  }

  // Sum of table S over the inclusive rectangle x0..x1, y0..y1.
  const rect = (S, x0, y0, x1, y1) =>
    S[(y1 + 1) * W1 + (x1 + 1)] - S[y0 * W1 + (x1 + 1)] - S[(y1 + 1) * W1 + x0] + S[y0 * W1 + x0];

  const out = new Uint8ClampedArray(rgba.length);
  for (let y = 0; y < h; y++) {
    const yTop = Math.max(0, y - radius), yBot = Math.min(h - 1, y + radius);
    for (let x = 0; x < w; x++) {
      const xL = Math.max(0, x - radius), xR = Math.min(w - 1, x + radius);
      // The 4 windows as [x0, y0, x1, y1]
      const windows = [
        [xL, yTop, x, y], [x, yTop, xR, y],
        [xL, y, x, yBot], [x, y, xR, yBot],
      ];
      let bestVar = Infinity, best = windows[0], bestN = 1;
      for (const win of windows) {
        const n = (win[2] - win[0] + 1) * (win[3] - win[1] + 1);
        const mean = rect(sL, win[0], win[1], win[2], win[3]) / n;
        const variance = rect(sL2, win[0], win[1], win[2], win[3]) / n - mean * mean; // Var = E[X²] − E[X]²
        if (variance < bestVar) { bestVar = variance; best = win; bestN = n; }
      }
      const j = (y * w + x) * 4;
      out[j] = rect(sR, best[0], best[1], best[2], best[3]) / bestN;
      out[j + 1] = rect(sG, best[0], best[1], best[2], best[3]) / bestN;
      out[j + 2] = rect(sB, best[0], best[1], best[2], best[3]) / bestN;
      out[j + 3] = 255;
    }
  }
  return out;
}

/**
 * k-means++ initialization.
 * k-means needs k starting "guess" colors. Random picks often land several
 * guesses in the same area (e.g. all in the sky), giving a poor palette.
 * k-means++ spreads them out: the first is random; each next one is chosen
 * with probability proportional to its squared distance from the nearest
 * already-chosen center, so far-away (different) colors are favored.
 * `samples` is a flat array [r,g,b, r,g,b, ...] of n colors.
 */
function kmeansPlusPlusInit(samples, n, k) {
  const centers = new Float64Array(k * 3);
  const first = Math.floor(Math.random() * n);
  centers[0] = samples[first * 3]; centers[1] = samples[first * 3 + 1]; centers[2] = samples[first * 3 + 2];

  const nearest = new Float64Array(n).fill(Infinity); // squared distance to the closest chosen center
  for (let c = 1; c < k; c++) {
    const pr = centers[(c - 1) * 3], pg = centers[(c - 1) * 3 + 1], pb = centers[(c - 1) * 3 + 2];
    let total = 0;
    for (let i = 0; i < n; i++) {
      const dr = samples[i * 3] - pr, dg = samples[i * 3 + 1] - pg, db = samples[i * 3 + 2] - pb;
      const d = dr * dr + dg * dg + db * db;
      if (d < nearest[i]) nearest[i] = d;
      total += nearest[i];
    }
    let pick = Math.floor(Math.random() * n);
    if (total > 0) {
      let target = Math.random() * total;
      for (let i = 0; i < n; i++) {
        target -= nearest[i];
        if (target <= 0) { pick = i; break; }
      }
    }
    centers[c * 3] = samples[pick * 3];
    centers[c * 3 + 1] = samples[pick * 3 + 1];
    centers[c * 3 + 2] = samples[pick * 3 + 2];
  }
  return centers;
}

/**
 * k-means color quantization — choose a small "paint box" of k colors.
 * Real illustrators use a limited palette; flat areas of shared color are a
 * big part of the anime-film look.
 *
 * k-means repeats two steps until the colors stop moving (or maxIter):
 *   1. ASSIGN: each pixel joins the cluster whose center color is nearest
 *      (distance in RGB space, like distance between points in 3D).
 *   2. UPDATE: each center moves to the average color of its members.
 * For speed we learn the palette from ~20,000 sample pixels, then map EVERY
 * pixel to its nearest palette color at the end.
 * Returns { labels (cluster index per pixel), centers (flat k*3 array) }.
 */
function kmeansQuantize(rgba, w, h, k, maxIter) {
  const total = w * h;
  const step = Math.max(1, Math.floor(total / 20000));
  const n = Math.floor((total - 1) / step) + 1;
  const samples = new Float64Array(n * 3);
  for (let s = 0, p = 0; s < n; s++, p += step) {
    samples[s * 3] = rgba[p * 4]; samples[s * 3 + 1] = rgba[p * 4 + 1]; samples[s * 3 + 2] = rgba[p * 4 + 2];
  }

  const centers = kmeansPlusPlusInit(samples, n, k);
  const sums = new Float64Array(k * 3);
  const counts = new Uint32Array(k);

  for (let iter = 0; iter < maxIter; iter++) {
    sums.fill(0);
    counts.fill(0);
    // 1. Assign each sample to its nearest center
    for (let i = 0; i < n; i++) {
      const r = samples[i * 3], g = samples[i * 3 + 1], b = samples[i * 3 + 2];
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const dr = r - centers[c * 3], dg = g - centers[c * 3 + 1], db = b - centers[c * 3 + 2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bestD) { bestD = d; best = c; }
      }
      sums[best * 3] += r; sums[best * 3 + 1] += g; sums[best * 3 + 2] += b;
      counts[best]++;
    }
    // 2. Move each center to the average of its members
    let maxShift = 0;
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        // An empty cluster is useless: restart it at a random sample.
        const p = Math.floor(Math.random() * n);
        centers[c * 3] = samples[p * 3]; centers[c * 3 + 1] = samples[p * 3 + 1]; centers[c * 3 + 2] = samples[p * 3 + 2];
        maxShift = Infinity;
        continue;
      }
      const nr = sums[c * 3] / counts[c], ng = sums[c * 3 + 1] / counts[c], nb = sums[c * 3 + 2] / counts[c];
      const shift = Math.hypot(nr - centers[c * 3], ng - centers[c * 3 + 1], nb - centers[c * 3 + 2]);
      if (shift > maxShift) maxShift = shift;
      centers[c * 3] = nr; centers[c * 3 + 1] = ng; centers[c * 3 + 2] = nb;
    }
    if (maxShift < 1) break; // converged: colors moved less than 1 level
  }

  // Map every pixel to its nearest palette color
  const labels = new Uint8Array(total);
  for (let p = 0; p < total; p++) {
    const r = rgba[p * 4], g = rgba[p * 4 + 1], b = rgba[p * 4 + 2];
    let best = 0, bestD = Infinity;
    for (let c = 0; c < k; c++) {
      const dr = r - centers[c * 3], dg = g - centers[c * 3 + 1], db = b - centers[c * 3 + 2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestD) { bestD = d; best = c; }
    }
    labels[p] = best;
  }
  return { labels, centers };
}

/**
 * The heavy painting step, grouped so it can run in a Web Worker:
 * optional Kuwahara passes (opts.radii), then k-means on the result.
 */
function paintStage(rgba, w, h, opts) {
  let smoothed = rgba;
  for (let i = 0; i < opts.radii.length; i++) smoothed = kuwaharaFilter(smoothed, w, h, opts.radii[i]);
  const km = kmeansQuantize(smoothed, w, h, opts.k, opts.maxIter);
  return { smoothed, labels: km.labels, centers: km.centers };
}

/** Apply a 256-entry look-up table to the R, G and B of every pixel (in place). */
function applyLut(rgba, lut) {
  for (let j = 0; j < rgba.length; j += 4) {
    rgba[j] = lut[rgba[j]]; rgba[j + 1] = lut[rgba[j + 1]]; rgba[j + 2] = lut[rgba[j + 2]];
  }
}

/**
 * "Auto levels" curve for brightening dark photos, as a 256-entry look-up
 * table (old value → new value). Like auto levels in a photo editor:
 *   1. find the darkest and brightest 0.5% brightness levels and stretch that
 *      range to 0–255 (gain capped at 2.5x so noise isn't blown up);
 *   2. if the picture is still dark on average, apply a gamma curve (power
 *      < 1) that lifts the shadows and midtones until the average brightness
 *      is about 45%.
 * All three channels get the same curve, so colors keep their hue. Callers
 * can blend it with "no change" to lift only partway (see buildStylized).
 */
function makeLevelsLut(rgba) {
  const n = rgba.length / 4;
  const hist = new Uint32Array(256);
  for (let j = 0; j < rgba.length; j += 4) {
    hist[(0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]) | 0]++;
  }
  const cut = n * 0.005;
  let lo = 0, hi = 255, acc = 0;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > cut) { lo = v; break; } }
  acc = 0;
  for (let v = 255; v >= 0; v--) { acc += hist[v]; if (acc > cut) { hi = v; break; } }
  const range = Math.max(hi - lo, 255 / 2.5);

  // Average brightness after stretching, to decide how much to lift.
  let sum = 0;
  for (let v = 0; v < 256; v++) sum += hist[v] * clamp((v - lo) / range, 0, 1);
  const mean = Math.max(0.02, sum / n);
  const gamma = clamp(Math.log(0.45) / Math.log(mean), 0.55, 1); // only ever brightens

  const lut = new Uint8ClampedArray(256); // look-up table: old value → new value
  for (let v = 0; v < 256; v++) lut[v] = 255 * Math.pow(clamp((v - lo) / range, 0, 1), gamma);
  return lut;
}

/**
 * Paper grain: a per-pixel brightness multiplier around 1.0 (±6%), mixing
 * large soft blotches, medium fibers and fine random speckle.
 */
function makePaperGrain(w, h) {
  const coarse = makeValueNoise(w, h, 40);
  const medium = makeValueNoise(w, h, 6);
  const grain = new Float32Array(w * h);
  for (let i = 0; i < grain.length; i++) {
    const v = 0.4 * coarse[i] + 0.35 * medium[i] + 0.25 * Math.random();
    grain[i] = 1 + (v - 0.5) * 0.12;
  }
  return grain;
}

/** The blank textured paper shown before the paint arrives. */
function composePaper(grain, w, h) {
  const [pr, pg, pb] = SETTINGS.paperColor;
  const paper = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < grain.length; i++, j += 4) {
    paper[j] = pr * grain[i]; paper[j + 1] = pg * grain[i]; paper[j + 2] = pb * grain[i]; paper[j + 3] = 255;
  }
  return paper;
}

/**
 * For the live animation: WHEN each pixel gets painted, as a number 0–1.
 * Clusters are painted lightest first (like watercolor: light washes, then
 * darker layers). Inside a cluster, paint spreads outward from the cluster's
 * center, with noise added so the wash edge looks ragged and organic.
 * Clusters overlap in time so it flows instead of stepping.
 */
function computeRevealOrder(labels, palette, w, h) {
  const k = palette.length;
  const lum = palette.map(([r, g, b]) => 0.299 * r + 0.587 * g + 0.114 * b);
  const order = lum.map((_, i) => i).sort((a, b) => lum[b] - lum[a]);
  const rank = new Float32Array(k);
  order.forEach((c, r) => { rank[c] = r; });

  // Center of mass of each cluster
  const sx = new Float64Array(k), sy = new Float64Array(k), cnt = new Float64Array(k);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = labels[y * w + x];
      sx[c] += x; sy[c] += y; cnt[c]++;
    }
  }
  for (let c = 0; c < k; c++) { if (cnt[c]) { sx[c] /= cnt[c]; sy[c] /= cnt[c]; } }

  // Distance of each pixel from its cluster's center, and the max per cluster
  const dist = new Float32Array(w * h);
  const maxD = new Float32Array(k);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, c = labels[i];
      const d = Math.hypot(x - sx[c], y - sy[c]);
      dist[i] = d;
      if (d > maxD[c]) maxD[c] = d;
    }
  }

  const noise = makeValueNoise(w, h, 24);
  const span = 0.3; // each cluster's wash takes 30% of the paint phase
  const spacing = k > 1 ? (1 - span) / (k - 1) : 0;
  const reveal = new Float32Array(w * h);
  for (let i = 0; i < reveal.length; i++) {
    const c = labels[i];
    const local = 0.65 * (dist[i] / (maxD[c] || 1)) + 0.35 * noise[i];
    reveal[i] = rank[c] * spacing + local * span;
  }
  return reveal;
}

/**
 * Paint the "wash" frame for paint-phase progress q (0–1): each pixel fades
 * from paper to its final color over a short window around its reveal time.
 */
const WASH_FADE = 0.06;
function renderPaintWash(frame, painted, paper, reveal, q) {
  const p = q * (1 + WASH_FADE);
  const d = frame.data;
  for (let i = 0, j = 0; i < reveal.length; i++, j += 4) {
    let a = (p - reveal[i]) / WASH_FADE;
    a = a < 0 ? 0 : a > 1 ? 1 : a;
    d[j] = paper[j] + (painted[j] - paper[j]) * a;
    d[j + 1] = paper[j + 1] + (painted[j + 1] - paper[j + 1]) * a;
    d[j + 2] = paper[j + 2] + (painted[j + 2] - paper[j + 2]) * a;
    d[j + 3] = 255;
  }
}

/**
 * Outline style. Major contours get thicker lines (like pressing harder),
 * small details stay fine. `outlineWidth` (in processing pixels) and the
 * optional `outlineColor` / `outlineAlpha` are set when the strokes are built;
 * the default is a soft dark brown.
 */
function styleOutline(ctx, stroke) {
  ctx.strokeStyle = stroke.outlineColor || '#3d2518';
  ctx.lineWidth = stroke.outlineWidth;
  ctx.globalAlpha = stroke.outlineAlpha ?? 1;
}

/**
 * Put the painting and the outline layer together on the visible canvas.
 * Outlines are drawn on their own layer at full strength, then the whole layer
 * is blended at reduced opacity. (Drawing each line semi-transparent directly
 * would create darker dots wherever line pieces overlap.)
 */
function compositePainted(ctx, paintSource, outlineLayer, opacity) {
  resetCtx(ctx);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(paintSource, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.globalAlpha = opacity;
  ctx.drawImage(outlineLayer, 0, 0);
  ctx.globalAlpha = 1;
}

function drawPaintedFinal(ctx, art) {
  if (art.linesBaked) {
    // Cartoon and Illustration already have their lines inside the painting.
    resetCtx(ctx);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(art.paintCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
    return;
  }
  const layer = makeCanvas(ctx.canvas.width, ctx.canvas.height);
  const lctx = layer.getContext('2d');
  setStrokeTransform(lctx, art.scale);
  for (const s of art.strokes) {
    styleOutline(lctx, s);
    lctx.beginPath();
    drawSmoothPath(lctx, s.segs);
    lctx.stroke();
  }
  compositePainted(ctx, art.paintCanvas, layer, art.outlineOpacity);
}

/**
 * Live version. With art.linesFirst (Cartoon and Illustration), the pen
 * sketches the lines for the first 45% of the time and then the colors wash
 * in. Otherwise the paint washes come first (~55%) and the outlines are drawn
 * stroke by stroke on top.
 */
function createPaintedDrawer(ctx, art) {
  const work = makeCanvas(art.width, art.height);
  const wctx = work.getContext('2d');
  const frame = wctx.createImageData(art.width, art.height);
  const layer = makeCanvas(ctx.canvas.width, ctx.canvas.height);
  const lctx = layer.getContext('2d');
  setStrokeTransform(lctx, art.scale);
  const pen = createStrokePen(art.strokes, styleOutline);

  if (art.linesFirst) {
    // Ink first, then color (how an illustrator works): the pen sketches the
    // line art on blank paper, then the colors wash in underneath while the
    // sketch lines fade into the finished, inked painting.
    const paper = makeCanvas(art.width, art.height);
    paper.getContext('2d').putImageData(new ImageData(art.paper, art.width, art.height), 0, 0);
    const lineShare = art.strokes.length ? 0.45 : 0;
    let penDone = lineShare === 0;
    return (progress) => {
      if (!penDone && progress < lineShare) {
        pen.advanceTo(lctx, (progress / lineShare) * pen.totalLength);
        compositePainted(ctx, paper, layer, 1);
        return;
      }
      if (!penDone) { pen.advanceTo(lctx, Infinity); penDone = true; }
      const q = lineShare ? Math.min(1, (progress - lineShare) / (1 - lineShare)) : progress;
      renderPaintWash(frame, art.painted, art.paper, art.revealAt, q);
      wctx.putImageData(frame, 0, 0);
      compositePainted(ctx, q >= 1 ? art.paintCanvas : work, layer, Math.max(0, 1 - q * 1.3));
    };
  }

  const paintShare = art.strokes.length ? 0.55 : 1;
  let paintDone = false;

  return (progress) => {
    if (!paintDone) {
      const q = Math.min(1, progress / paintShare);
      renderPaintWash(frame, art.painted, art.paper, art.revealAt, q);
      wctx.putImageData(frame, 0, 0);
      if (q >= 1) paintDone = true;
    }
    if (progress > paintShare) {
      pen.advanceTo(lctx, ((progress - paintShare) / (1 - paintShare)) * pen.totalLength);
    }
    compositePainted(ctx, paintDone ? art.paintCanvas : work, layer, art.outlineOpacity);
  };
}

/**
 * Shared finishing steps for a painted result: paper, reveal plan, pen
 * strokes and the result object that the UI draws.
 */
function makePaintedArt({ w, h, painted, grain, labels, centers, edgeSource, detail, strictness, minPoints, outlineOpacity, faces = 0, customStrokes = null }) {
  const palette = [];
  for (let c = 0; c < centers.length / 3; c++) palette.push([centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2]]);
  const paper = composePaper(grain, w, h);
  const revealAt = computeRevealOrder(labels, palette, w, h);
  const scale = outputScale(w, h);

  // Either strokes built by the caller (Cartoon / Illustration), or soft outlines
  // traced from `edgeSource` with the classic pipeline from section 1.
  let strokes = customStrokes;
  if (!strokes) {
    strokes = extractStrokes(edgeSource, w, h, { detail, strictness, minPoints });
    for (const s of strokes) s.outlineWidth = (1.2 + 1.8 * strokeProminence(s)) / scale; // 1.2–3 px on the 1400px output
  }

  const paintCanvas = makeCanvas(w, h);
  paintCanvas.getContext('2d').putImageData(new ImageData(painted, w, h), 0, 0);

  return {
    mode: 'painted', // replaced by the caller's style name
    width: w,
    height: h,
    scale,
    strokes,
    painted,
    paper,
    revealAt,
    paintCanvas,
    outlineOpacity,
    faceCount: faces,
    isEmpty: false, // there's always a painting, even with no outlines
    note: '',
    drawFinal(ctx) { drawPaintedFinal(ctx, this); },
    createLiveDrawer(ctx) { return createPaintedDrawer(ctx, this); },
  };
}


/* =============================================================================
   4. WEB WORKER HELPER
   -----------------------------------------------------------------------------
   A Web Worker is a background thread. While it crunches Kuwahara + k-means,
   the page stays responsive (the spinner keeps spinning).
   To keep this a no-build static site, we don't ship a separate worker file:
   we turn the self-contained functions above into source text with
   .toString(), put it in a Blob, and start the worker from that Blob's URL.
   If workers aren't available, we just run the same functions on the main thread.
   ============================================================================= */

const PAINT_WORKER_FUNCTIONS = [kuwaharaFilter, kmeansPlusPlusInit, kmeansQuantize, paintStage];

function createPaintWorker() {
  const source =
    PAINT_WORKER_FUNCTIONS.map((fn) => fn.toString()).join('\n\n') +
    `
self.onmessage = function (e) {
  var d = e.data;
  try {
    var r = paintStage(d.rgba, d.w, d.h, d.opts);
    self.postMessage({ ok: true, result: r }, [r.smoothed.buffer, r.labels.buffer]);
  } catch (err) {
    self.postMessage({ ok: false, error: String(err) });
  }
};`;
  const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  const worker = new Worker(url);
  worker.blobUrl = url;
  return worker;
}

function runPaintStage(rgba, w, h, opts) {
  const onMainThread = async () => {
    await nextFrame();
    return paintStage(rgba, w, h, opts);
  };

  let worker;
  try {
    worker = createPaintWorker();
  } catch (err) {
    return onMainThread();
  }

  return new Promise((resolve, reject) => {
    const cleanup = () => { worker.terminate(); URL.revokeObjectURL(worker.blobUrl); };
    worker.onmessage = (e) => {
      cleanup();
      if (e.data.ok) resolve(e.data.result);
      else onMainThread().then(resolve, reject);
    };
    worker.onerror = (e) => {
      e.preventDefault();
      cleanup();
      onMainThread().then(resolve, reject);
    };
    const copy = new Uint8ClampedArray(rgba); // send a copy; we may still need the original for the fallback
    worker.postMessage({ rgba: copy, w, h, opts }, [copy.buffer]);
  });
}


/* =============================================================================
   4b. FACE & BODY FINDER  (MediaPipe segmentation + face landmarks)
   -----------------------------------------------------------------------------
   Two small neural networks from Google's MediaPipe tell the styles WHERE
   things are, so each part of the picture can be treated the right way:

     1. The "multiclass selfie" segmenter labels every pixel: background,
        hair, body skin, face skin, clothes or accessories. Cartoon and
        Illustration use it to smooth skin more than hair, keep clothes
        detailed, and outline the person.
     2. The face landmarker finds 478 points on each face: outlines of the
        eyes, irises, eyebrows, lips, nose and jaw. Line art uses them for
        crisp feature lines; Cartoon and Illustration use them to refine the
        eyes, brows and lips (refineFace in 4c).

   Both are optional: if they can't load (offline, blocked, old browser),
   every style still works, just without the extra face/body care.

   Both models are small (~20 MB together), run on your device with
   WebAssembly, and are cached after the first download. The photo itself
   never leaves the browser.
   ============================================================================= */

const MP = {
  base: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1',
  segmenterUrl: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
  landmarkerUrl: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  cacheName: 'photo-to-drawing-models-v2',
};

// The segmenter's classes.
const SEG = { BG: 0, HAIR: 1, BODY: 2, FACE: 3, CLOTHES: 4, OTHER: 5 };

// Face-landmark indices (MediaPipe face mesh numbering).
const LM = {
  // Eye outlines start at the outer corner, run along the upper lid to the
  // inner corner (the first 9 points), then back along the lower lid.
  rightEye: [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
  leftEye: [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
  rightIris: 468, rightIrisEdge: [469, 470, 471, 472],
  leftIris: 473, leftIrisEdge: [474, 475, 476, 477],
  rightBrow: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],   // upper edge, then lower edge back
  leftBrow: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
  lipsOuter: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
  lipsInner: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
  lipLine: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  noseBottom: 2, noseRight: 98, noseLeft: 327,
  cheekRight: 205, cheekLeft: 425,
  chin: 152,
  faceOval: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
};

let visionPromise = null, segmenterPromise = null, landmarkerPromise = null;

/** Reject if `promise` takes longer than `ms`. */
function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('timed out')), ms)),
  ]);
}

/**
 * Download a model file, reporting progress (0–1). Files are kept in the
 * browser's Cache Storage, so later visits load them instantly, even offline.
 */
async function fetchModel(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(MP.cacheName);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (e) { cache = null; } // Cache Storage needs HTTPS (or localhost)

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  let bytes;
  if (res.body && total && onProgress) {
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(Math.min(1, received / total));
    }
    bytes = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
  }
  if (cache) { try { await cache.put(url, new Response(bytes)); } catch (e) { /* storage full: fine */ } }
  return bytes;
}

/** Load MediaPipe's vision library (an ES module) and its WebAssembly files, once. */
function loadVision() {
  if (!visionPromise) {
    visionPromise = (async () => {
      const vision = await import(`${MP.base}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP.base}/wasm`);
      return { vision, fileset };
    })();
    visionPromise.catch(() => { visionPromise = null; }); // allow a retry later
  }
  return visionPromise;
}

function getSegmenter(onProgress) {
  if (!segmenterPromise) {
    segmenterPromise = (async () => {
      const { vision, fileset } = await loadVision();
      const model = await fetchModel(MP.segmenterUrl, onProgress);
      return vision.ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: model, delegate: 'CPU' },
        runningMode: 'IMAGE',
        outputCategoryMask: false,
        outputConfidenceMasks: true, // one "how likely is this class" map per class
      });
    })();
    segmenterPromise.catch(() => { segmenterPromise = null; });
  }
  return segmenterPromise;
}

function getLandmarker(onProgress) {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const { vision, fileset } = await loadVision();
      const model = await fetchModel(MP.landmarkerUrl, onProgress);
      return vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: model, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numFaces: 4,
        minFaceDetectionConfidence: 0.4,
        minFacePresenceConfidence: 0.4,
      });
    })();
    landmarkerPromise.catch(() => { landmarkerPromise = null; });
  }
  return landmarkerPromise;
}

/** Face landmarks for every face on a canvas, as arrays of {x, y} in canvas pixels. */
async function findFaceLandmarks(canvas) {
  const landmarker = await getLandmarker();
  const result = landmarker.detect(canvas);
  return result.faceLandmarks.map((pts) => pts.map((p) => ({ x: p.x * canvas.width, y: p.y * canvas.height })));
}

/** Fraction of a stroke's points that lie inside a circle. */
function fractionInside(points, cx, cy, radius) {
  let inside = 0;
  for (const p of points) if ((p.x - cx) ** 2 + (p.y - cy) ** 2 <= radius * radius) inside++;
  return inside / points.length;
}

/**
 * Line-art feature outlines from the face landmarks: the exact contours of
 * the eyes, irises, eyebrows, nose, lips and face shape. Edge tracing can
 * miss or break these on small or dim faces; the landmarks always give
 * clean, complete shapes. Traced strokes that sit on the eyes or lips are
 * removed first, so features aren't drawn twice.
 * `faces` are landmark arrays in processing-image pixels.
 */
function addLandmarkFeatureLines(raw, faces) {
  let result = raw;
  const ring = (idx, pts) => { const p = pick(pts, idx); return p.concat([p[0]]); }; // closed loop
  for (const pts of faces) {
    const unit = (dist(pts[33], pts[133]) + dist(pts[263], pts[362])) / 2;
    if (unit < 4) continue; // too small for clean features

    // Remove traced strokes on the eyes and lips (they're redrawn precisely below).
    const zones = [LM.rightEye, LM.leftEye, LM.lipsOuter].map((idx) => {
      const p = pick(pts, idx);
      const c = centroid(p);
      return { c, r: Math.max(...p.map((q) => dist(q, c))) * 1.25 };
    });
    result = result.filter((s) => !zones.some((z) => fractionInside(s.points, z.c.x, z.c.y, z.r) > 0.6));

    const lines = [
      ring(LM.rightEye, pts), ring(LM.leftEye, pts),
      pick(pts, LM.rightBrow.slice(0, 5)), pick(pts, LM.leftBrow.slice(0, 5)),      // top edge of each brow
      pick(pts, LM.rightBrow.slice(5)), pick(pts, LM.leftBrow.slice(5)),            // bottom edge
      ring(LM.lipsOuter, pts), ring(LM.lipsInner, pts),
      pick(pts, [168, 6, 197, 195, 5]),                                             // nose bridge
      pick(pts, [98, 97, 2, 326, 327]),                                             // underside of the nose
      ring(LM.faceOval, pts),
    ];
    // Irises: circles through the iris landmarks.
    for (const [center, edge] of [[LM.rightIris, LM.rightIrisEdge], [LM.leftIris, LM.leftIrisEdge]]) {
      const c = pts[center];
      const r = pick(pts, edge).reduce((sum, p) => sum + dist(p, c), 0) / 4;
      const circle = [];
      for (let a = 0; a <= 16; a++) circle.push({ x: c.x + r * Math.cos((a / 16) * Math.PI * 2), y: c.y + r * Math.sin((a / 16) * Math.PI * 2) });
      lines.push(circle);
    }
    for (const points of lines) result.push({ points, rawLength: 120, feature: true });
  }
  return result;
}

/* ---------- Small geometry and color helpers for drawing ---------- */

const pick = (pts, idx) => idx.map((i) => pts[i]);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const lerpPt = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

function centroid(points) {
  let x = 0, y = 0;
  for (const p of points) { x += p.x; y += p.y; }
  return { x: x / points.length, y: y / points.length };
}

/** Stretch points away from (cx, cy) by sx horizontally and sy vertically. */
function scaleAbout(points, cx, cy, sx, sy) {
  return points.map((p) => ({ x: cx + (p.x - cx) * sx, y: cy + (p.y - cy) * sy }));
}

/** A smooth closed curve through a ring of points (curves through midpoints, like buildSegments). */
function smoothClosedPath(ctx, pts) {
  const n = pts.length;
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const start = mid(pts[n - 1], pts[0]);
  ctx.moveTo(start.x, start.y);
  for (let i = 0; i < n; i++) {
    const m = mid(pts[i], pts[(i + 1) % n]);
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, m.x, m.y);
  }
  ctx.closePath();
}

/** A smooth open curve through a list of points. */
function smoothOpenPath(ctx, pts) {
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const m = { x: (pts[i].x + pts[i + 1].x) / 2, y: (pts[i].y + pts[i + 1].y) / 2 };
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, m.x, m.y);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last.x, last.y);
}

/** Blur a float mask several times (each pass is the 5-tap Gaussian from step 3). */
function blurTimes(mask, w, h, passes) {
  let m = mask;
  for (let p = 0; p < passes; p++) m = gaussianBlur(m, w, h);
  return m;
}

/** Average brightness (0–1) of RGBA pixels. */
function averageLightness(rgba) {
  let sum = 0;
  for (let j = 0; j < rgba.length; j += 4) sum += 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return sum / (rgba.length / 4) / 255;
}

/** Nearest-neighbor resize of a float mask (if the segmenter returns a different size). */
function resizeMask(mask, mw, mh, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const my = Math.min(mh - 1, Math.floor((y * mh) / h));
    for (let x = 0; x < w; x++) out[y * w + x] = mask[my * mw + Math.min(mw - 1, Math.floor((x * mw) / w))];
  }
  return out;
}


/* =============================================================================
   4c. MODES 2 & 3: CARTOON AND ILLUSTRATION  (one shared engine, on-device)
   -----------------------------------------------------------------------------
   Both styles keep the photo's REAL shapes, lighting and detail and re-render
   them the way an artist would, so they keep your likeness and work on any
   picture. They run the same steps (buildStylized); STYLE_PRESETS decides
   the look:
     - Cartoon: a pencil sketch of the photo as it is, colored in with clean,
       flat cartoon colors (strong flattening, a few clear shading tones).
       Like an artist, it tells busy TEXTURE (carpet, lace: flattened, with
       just a pencil hint of the pattern) from real STRUCTURE (wall streaks,
       wood grain: smoothed along its grain and kept). See textureMap.
     - Illustration: a detailed digital painting with soft shading and ink.
   Both keep the eyes and the inside of the mouth true to the photo, so eye
   color and a smile's teeth survive (see faceMasks and irisColor).
   The recipe comes from computer-graphics research on "image abstraction"
   (Winnemöller et al., "Real-Time Video Abstraction"):

     1. Lab color: split each pixel into lightness (L) and two color axes
        (a = green↔red, b = blue↔yellow), so shading and color can be
        treated separately.
     2. Edge-aware smoothing (the "domain transform" filter): flattens skin
        pores, fabric weave and noise into clean surfaces while every real
        edge stays sharp. It's region-aware (using MediaPipe's person
        segmentation): skin is smoothed most, hair and clothes keep their
        detail, and the background is simplified more.
     3. Shading bands: lightness is pulled toward a few levels, the stepped
        shading artists use (the Cartoon uses fewer, crisper bands).
     4. Lines with XDoG (eXtended Difference of Gaussians), guided by the
        "edge flow" so they follow shapes smoothly, like an artist's hand
        (Kang et al., "Coherent Line Drawing"). The Illustration inks them;
        the Cartoon draws them as graphite pencil with streaky texture.
     5. Face refinement from the 478 face landmarks: defined lash lines,
        iris ring and sparkle, cleaner brows, glossy lips.
   ============================================================================= */

// sRGB byte → linear light (0–1), precomputed for all 256 values.
const SRGB_TO_LINEAR = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return t;
})();

/**
 * RGBA → CIE Lab (D65). L is lightness 0–100; a and b are the two color
 * axes (roughly −100…100). Lab is "perceptually uniform": a difference of
 * 10 looks about equally big anywhere, which makes edge detection and
 * smoothing behave evenly on dark and bright areas.
 */
function rgbaToLab(rgba) {
  const n = rgba.length / 4;
  const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const r = SRGB_TO_LINEAR[rgba[j]], g = SRGB_TO_LINEAR[rgba[j + 1]], b = SRGB_TO_LINEAR[rgba[j + 2]];
    const fx = f((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047);
    const fy = f(0.2126 * r + 0.7152 * g + 0.0722 * b);
    const fz = f((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
    L[i] = 116 * fy - 16;
    A[i] = 500 * (fx - fy);
    B[i] = 200 * (fy - fz);
  }
  return { L, A, B };
}

/** CIE Lab → RGBA (written into `out`). */
function labToRgba(L, A, B, out) {
  const finv = (t) => { const t3 = t * t * t; return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787; };
  const enc = (v) => {
    v = v <= 0 ? 0 : v >= 1 ? 1 : v;
    return 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
  };
  for (let i = 0, j = 0; i < L.length; i++, j += 4) {
    const fy = (L[i] + 16) / 116, fx = fy + A[i] / 500, fz = fy - B[i] / 200;
    const x = finv(fx) * 0.95047, y = finv(fy), z = finv(fz) * 1.08883;
    out[j] = enc(3.2406 * x - 1.5372 * y - 0.4986 * z);
    out[j + 1] = enc(-0.9689 * x + 1.8758 * y + 0.0415 * z);
    out[j + 2] = enc(0.0557 * x - 0.204 * y + 1.057 * z);
    out[j + 3] = 255;
  }
  return out;
}

/**
 * Gaussian blur with any sigma (separable; kernel radius 3·sigma).
 * Wide blurs use a shortcut: three box blurs in a row look the same as a
 * Gaussian (the "central limit theorem" at work), and a box blur costs the
 * same for any width, using a running sum.
 */
function gaussianBlurSigma(src, w, h, sigma) {
  if (sigma > 3) return boxBlurGauss(src, w, h, sigma);
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++) s += src[row + clamp(x + j, 0, w - 1)] * k[j + r];
      tmp[row + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++) s += tmp[clamp(y + j, 0, h - 1) * w + x] * k[j + r];
      out[y * w + x] = s;
    }
  }
  return out;
}

/**
 * Three box blurs whose combined spread matches a Gaussian of `sigma`
 * (box widths from the classic "boxes for Gauss" formula).
 */
function boxBlurGauss(src, w, h, sigma) {
  const passes = 3;
  const ideal = Math.sqrt((12 * sigma * sigma) / passes + 1);
  let lower = Math.floor(ideal);
  if (lower % 2 === 0) lower--;
  const narrow = Math.round((12 * sigma * sigma - passes * lower * lower - 4 * passes * lower - 3 * passes) / (-4 * lower - 4));
  let a = Float32Array.from(src), b = new Float32Array(w * h);
  for (let p = 0; p < passes; p++) {
    const r = ((p < narrow ? lower : lower + 2) - 1) / 2;
    for (let y = 0; y < h; y++) boxBlurLine(a, b, y * w, 1, w, r);
    for (let x = 0; x < w; x++) boxBlurLine(b, a, x, w, h, r);
  }
  return a;
}

/** One box blur of radius r along a line of `len` pixels (start, start+stride, …), edges repeated. */
function boxBlurLine(src, dst, start, stride, len, r) {
  const at = (k) => src[start + clamp(k, 0, len - 1) * stride];
  let sum = 0;
  for (let k = -r; k <= r; k++) sum += at(k);
  const inv = 1 / (2 * r + 1);
  for (let k = 0; k < len; k++) {
    dst[start + k * stride] = sum * inv;
    sum += at(k + r + 1) - at(k - r);
  }
}

/**
 * Edge-aware smoothing with the "domain transform" recursive filter
 * (Gastal & Oliveira, 2011). Smooths `chans` in place.
 *
 * The idea: imagine each image row as a rubber band. Where the image is
 * flat, neighboring pixels stay close together; across an edge, the band
 * is stretched, so pixels on the two sides end up "far apart". A plain
 * blur in that stretched space mixes neighbors that are close, so flat areas
 * get smoothed while pixels across an edge barely mix.
 *   stretch between neighbors = 1 + (sigmaS / sigmaR) · |color difference|
 * The blur itself is a cheap running average (each pixel blends toward its
 * neighbor by a weight that shrinks with the stretch), run left→right,
 * right→left, top→bottom and bottom→top, repeated 3 times with shrinking
 * strength. Cost is O(pixels), so it's fast even at full size.
 *
 * sigmaS: how far smoothing spreads (pixels). sigmaR: how big a color
 * difference counts as an edge. It can be one number, or a per-pixel array
 * (e.g. skin smoothed more than hair).
 */
function domainTransformSmooth(chans, guide, w, h, sigmaS, sigmaR, iterations = 3) {
  const n = w * h;
  const perPixel = typeof sigmaR !== 'number';
  const dH = new Float32Array(n), dV = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const ratio = sigmaS / (perPixel ? sigmaR[i] : sigmaR);
      if (x > 0) { let s = 0; for (const g of guide) s += Math.abs(g[i] - g[i - 1]); dH[i] = 1 + ratio * s; }
      if (y > 0) { let s = 0; for (const g of guide) s += Math.abs(g[i] - g[i - w]); dV[i] = 1 + ratio * s; }
    }
  }
  const wH = new Float32Array(n), wV = new Float32Array(n);
  for (let it = 0; it < iterations; it++) {
    const sigmaH = (sigmaS * Math.sqrt(3) * Math.pow(2, iterations - it - 1)) / Math.sqrt(Math.pow(4, iterations) - 1);
    const c = -Math.SQRT2 / sigmaH;
    for (let i = 0; i < n; i++) { wH[i] = Math.exp(c * dH[i]); wV[i] = Math.exp(c * dV[i]); }
    for (const J of chans) {
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 1; x < w; x++) { const i = row + x; J[i] += wH[i] * (J[i - 1] - J[i]); }
        for (let x = w - 2; x >= 0; x--) { const i = row + x; J[i] += wH[i + 1] * (J[i + 1] - J[i]); }
      }
      for (let y = 1; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) { const i = row + x; J[i] += wV[i] * (J[i - w] - J[i]); }
      }
      for (let y = h - 2; y >= 0; y--) {
        const row = y * w;
        for (let x = 0; x < w; x++) { const i = row + x; J[i] += wV[i + w] * (J[i + w] - J[i]); }
      }
    }
  }
}

/**
 * Soft shading bands ("soft quantization"): pull lightness toward `levels`
 * evenly spaced steps with a smooth tanh curve instead of hard rounding, so
 * shading looks painted in a few tones but without jagged banding. `mix`
 * (0–1, or a per-pixel array) sets how strongly each pixel is stepped.
 */
function softQuantize(L, levels, sharpness, mix) {
  const dq = 100 / levels;
  const perPixel = typeof mix !== 'number';
  for (let i = 0; i < L.length; i++) {
    const v = L[i];
    const q = Math.round(v / dq) * dq;
    const stepped = q + (dq / 2) * Math.tanh((sharpness * (v - q)) / (dq / 2));
    L[i] = v + (stepped - v) * (perPixel ? mix[i] : mix);
  }
}

/**
 * XDoG ink lines (Winnemöller, Kyprianidis & Olsen, 2012).
 * A Difference of Gaussians (a sharp blur minus a wider blur) is strongly
 * negative on the dark side of an edge and near zero in flat areas. XDoG
 * turns it into ink with a soft threshold:
 *   D = G_σ(L) − τ·G_kσ(L)
 *   ink = 1 (white paper) if D ≥ ε, else 1 + tanh(φ·(D − ε))
 * Strong edges give solid dark strokes, weaker edges thinner and lighter
 * ones, and flat regions stay clean. That's what makes it look drawn instead
 * of traced. Returns 0 (full ink) … 1 (no ink) per pixel.
 */
function xdogLines(L, w, h, sigma, { tau = 0.98, eps = -0.6, phi = 1.3, k = 1.6, flow = null } = {}) {
  const g1 = gaussianBlurSigma(L, w, h, sigma);
  const g2 = gaussianBlurSigma(L, w, h, sigma * k);
  let dog = new Float32Array(w * h);
  for (let i = 0; i < dog.length; i++) dog[i] = g1[i] - tau * g2[i];
  // Flow-based version: smooth the DoG signal ALONG the direction the lines
  // run, so on-and-off detections (a faint hair strand) join into one line.
  if (flow) dog = flowSmooth(dog, flow.tx, flow.ty, w, h, flow.length);
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i++) {
    const d = dog[i];
    out[i] = d >= eps ? 1 : Math.max(0, 1 + Math.tanh(phi * (d - eps)));
  }
  return out;
}

/**
 * Edge tangent flow: the direction lines "run" at every pixel (along an
 * edge, not across it), from the smoothed structure tensor.
 * The gradient (gx, gy) points across an edge. Averaging the tensor
 * [gx² gx·gy; gx·gy gy²] over a neighborhood (instead of averaging the raw
 * gradients, which cancel out on thin lines where the two sides point in
 * opposite directions) gives a stable direction even along hair strands.
 * Its main eigenvector is the average "across" direction; turning it 90°
 * gives the "along" direction. Returns unit vectors (tx, ty) per pixel.
 */
function edgeTangentFlow(L, w, h, sigma) {
  const { magnitude, direction } = sobelEdges(L, w, h);
  const n = w * h;
  let E = new Float32Array(n), F = new Float32Array(n), G = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const gx = magnitude[i] * Math.cos(direction[i]), gy = magnitude[i] * Math.sin(direction[i]);
    E[i] = gx * gx; F[i] = gx * gy; G[i] = gy * gy;
  }
  E = gaussianBlurSigma(E, w, h, sigma); F = gaussianBlurSigma(F, w, h, sigma); G = gaussianBlurSigma(G, w, h, sigma);
  const tx = new Float32Array(n), ty = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const lambda = (E[i] + G[i] + Math.sqrt((E[i] - G[i]) ** 2 + 4 * F[i] * F[i])) / 2; // larger eigenvalue
    let vx = F[i], vy = lambda - E[i];                                                 // its eigenvector (across)
    if (Math.abs(vx) + Math.abs(vy) < 1e-9) { vx = lambda - G[i]; vy = F[i]; }
    const len = Math.hypot(vx, vy);
    if (len < 1e-9) { tx[i] = 1; ty[i] = 0; continue; }
    tx[i] = -vy / len; ty[i] = vx / len;                                              // rotate 90° → along
  }
  return { tx, ty };
}

/**
 * Where is the photo busy TEXTURE rather than real STRUCTURE? (0–1 per pixel)
 * Uses the same structure tensor as edgeTangentFlow, averaged over a wider
 * window (sigma):
 *   - energy: how strong the gradients are there (≈ local contrast);
 *   - coherence: how much they agree on one direction (0 = every which way,
 *     1 = all parallel).
 * Real structure (an edge, hair strands, wood grain, stripes) is coherent.
 * Busy texture (carpet, lace, gravel, leaves) is strong but incoherent. An
 * artist simplifies texture into a clean fill with just a hint of pattern,
 * but draws structure. Very strong, bold patterns (a printed pillow) count
 * as design, not texture, and are kept. Returns { texture, coherence, energy }.
 */
function textureMap(L, w, h, sigma) {
  const { magnitude, direction } = sobelEdges(L, w, h);
  const n = w * h;
  let E = new Float32Array(n), F = new Float32Array(n), G = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const gx = magnitude[i] * Math.cos(direction[i]), gy = magnitude[i] * Math.sin(direction[i]);
    E[i] = gx * gx; F[i] = gx * gy; G[i] = gy * gy;
  }
  E = gaussianBlurSigma(E, w, h, sigma); F = gaussianBlurSigma(F, w, h, sigma); G = gaussianBlurSigma(G, w, h, sigma);
  const texture = new Float32Array(n), coherence = new Float32Array(n), energy = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const trace = E[i] + G[i];
    coherence[i] = trace > 1e-6 ? Math.sqrt((E[i] - G[i]) ** 2 + 4 * F[i] * F[i]) / trace : 0;
    energy[i] = Math.sqrt(trace);
    texture[i] = smoothstep(0.5, 0.25, coherence[i]) * smoothstep(3, 7, energy[i]) * smoothstep(36, 22, energy[i]);
  }
  return { texture, coherence, energy };
}

/**
 * Smooth a signal along the flow field: from each pixel, take small steps
 * forward and backward along the tangent direction (a "streamline", like
 * following a current), and average the values met on the way with Gaussian
 * weights. This is line integral convolution (LIC).
 */
function flowSmooth(src, tx, ty, w, h, length) {
  const out = new Float32Array(w * h);
  const steps = Math.max(2, Math.round(length));
  const sigma = steps / 2;
  const wts = [];
  for (let k = 0; k <= steps; k++) wts.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i0 = y * w + x;
      let sum = src[i0] * wts[0], wsum = wts[0];
      for (const dir of [1, -1]) {
        let px = x, py = y, vx = tx[i0] * dir, vy = ty[i0] * dir;
        for (let k = 1; k <= steps; k++) {
          px += vx; py += vy;
          const ix = Math.round(px), iy = Math.round(py);
          if (ix < 0 || iy < 0 || ix >= w || iy >= h) break;
          const j = iy * w + ix;
          let nx = tx[j], ny = ty[j];
          if (nx * vx + ny * vy < 0) { nx = -nx; ny = -ny; } // keep walking the same way
          vx = nx; vy = ny;
          sum += src[j] * wts[k]; wsum += wts[k];
        }
      }
      out[i0] = sum / wsum;
    }
  }
  return out;
}

/**
 * flowSmooth for several channels at once (they share each streamline walk),
 * changing them in place: every pixel moves toward its smoothed value by its
 * `weights` value (0–1). Pixels with weight 0 are skipped, which keeps this
 * fast when only part of the picture needs it.
 * It's also edge-aware: values along the streamline count less the more they
 * differ from the starting pixel in the first channel (by about `edge`), so
 * a streamline that wanders across an edge (out of a white lamp into a dark
 * wall) doesn't drag the other side's color in.
 */
function flowSmoothBlend(chans, tx, ty, w, h, length, weights, edge = 8) {
  const steps = Math.max(2, Math.round(length));
  const sigma = steps / 2;
  const wts = [];
  for (let k = 0; k <= steps; k++) wts.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
  const src = chans.map((c) => Float32Array.from(c)); // read from unchanged copies
  const sums = new Float64Array(chans.length);
  const key = src[0], inv2e2 = 1 / (2 * edge * edge);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i0 = y * w + x;
      const blend = weights[i0];
      if (blend < 0.01) continue;
      for (let c = 0; c < src.length; c++) sums[c] = src[c][i0] * wts[0];
      let wsum = wts[0];
      for (const dir of [1, -1]) {
        let px = x, py = y, vx = tx[i0] * dir, vy = ty[i0] * dir;
        for (let k = 1; k <= steps; k++) {
          px += vx; py += vy;
          const ix = Math.round(px), iy = Math.round(py);
          if (ix < 0 || iy < 0 || ix >= w || iy >= h) break;
          const j = iy * w + ix;
          let nx = tx[j], ny = ty[j];
          if (nx * vx + ny * vy < 0) { nx = -nx; ny = -ny; }
          vx = nx; vy = ny;
          const diff = key[j] - key[i0];
          const wk = wts[k] * Math.exp(-diff * diff * inv2e2);
          for (let c = 0; c < src.length; c++) sums[c] += src[c][j] * wk;
          wsum += wk;
        }
      }
      for (let c = 0; c < src.length; c++) chans[c][i0] += (sums[c] / wsum - chans[c][i0]) * blend;
    }
  }
}

/**
 * Remove ink specks: tiny isolated dots of ink (from hair texture, fabric
 * dots or noise) that read as dirt rather than lines. Each connected blob of
 * ink is found with a flood fill; blobs smaller than `minArea` pixels are
 * erased. Real lines are long, connected blobs and are kept.
 */
function removeInkSpecks(ink, w, h, minArea) {
  const n = w * h;
  const seen = new Uint8Array(n);
  const stack = new Int32Array(n);
  const blob = [];
  for (let s = 0; s < n; s++) {
    if (seen[s] || ink[s] >= 0.5) continue;
    let top = 0;
    blob.length = 0;
    stack[top++] = s;
    seen[s] = 1;
    while (top > 0) {
      const i = stack[--top];
      blob.push(i);
      const x = i % w, y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (!seen[j] && ink[j] < 0.5) { seen[j] = 1; stack[top++] = j; }
        }
      }
    }
    if (blob.length < minArea) for (const i of blob) ink[i] = 1;
  }
}

/**
 * Zhang–Suen thinning: peel pixels off the outside of thick lines, pass after
 * pass, until only a 1-pixel "skeleton" is left down each line's middle.
 * Used to turn the ink into center lines the live-drawing pen can follow.
 * A pixel is removed if it's on the boundary (2–6 filled neighbors), removing
 * it can't split the line (exactly one empty→filled change around it), and
 * it passes the directional test for that sub-pass.
 */
function thinLines(bin, w, h) {
  const del = [];
  let changed = true;
  while (changed) {
    changed = false;
    for (let pass = 0; pass < 2; pass++) {
      del.length = 0;
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          if (!bin[i]) continue;
          const p2 = bin[i - w], p3 = bin[i - w + 1], p4 = bin[i + 1], p5 = bin[i + w + 1];
          const p6 = bin[i + w], p7 = bin[i + w - 1], p8 = bin[i - 1], p9 = bin[i - w - 1];
          const filled = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (filled < 2 || filled > 6) continue;
          const ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
          let transitions = 0;
          for (let k = 0; k < 8; k++) if (ring[k] === 0 && ring[k + 1] === 1) transitions++;
          if (transitions !== 1) continue;
          if (pass === 0 ? (p2 * p4 * p6 || p4 * p6 * p8) : (p2 * p4 * p8 || p2 * p6 * p8)) continue;
          del.push(i);
        }
      }
      for (const i of del) bin[i] = 0;
      if (del.length) changed = true;
    }
  }
  return bin;
}

/**
 * Soft masks (0–1 per pixel) from the face landmarks:
 *   features:  the eyes and the inside of the mouth. They're kept true to
 *              the photo (no heavy smoothing, glare correction or color
 *              banding), so eye color and a smile's teeth survive.
 *   faceInner: the inside of the face, away from the eyes, brows, lower
 *              nose and lips. Stray sketch lines there (like a shadow edge
 *              across a cheek) are softened; the features keep theirs.
 */
function faceMasks(faces, w, h) {
  const c = makeCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const shape = (pts, idx, sx, sy) => {
    const p = pick(pts, idx), mid = centroid(p);
    ctx.beginPath();
    smoothClosedPath(ctx, scaleAbout(p, mid.x, mid.y, sx, sy));
    ctx.fill();
  };
  const read = () => {
    const a = ctx.getImageData(0, 0, w, h).data;
    return Float32Array.from({ length: w * h }, (_, i) => a[i * 4 + 3] / 255);
  };
  for (const pts of faces) {
    shape(pts, LM.rightEye, 1.3, 1.7);
    shape(pts, LM.leftEye, 1.3, 1.7);
    shape(pts, LM.lipsInner, 1.1, 1.3);
  }
  const features = blurTimes(read(), w, h, 2);

  ctx.clearRect(0, 0, w, h);
  for (const pts of faces) shape(pts, LM.faceOval, 0.86, 0.86);
  ctx.globalCompositeOperation = 'destination-out'; // cut the features back out
  for (const pts of faces) {
    shape(pts, LM.rightEye, 1.8, 2.8);
    shape(pts, LM.leftEye, 1.8, 2.8);
    shape(pts, LM.rightBrow, 1.3, 2.2);
    shape(pts, LM.leftBrow, 1.3, 2.2);
    shape(pts, LM.lipsOuter, 1.3, 1.6);
    // The lower half of the nose (tip, nostrils and the bottom of its sides):
    // an ellipse from mid-bridge to just under the nose, tilted with the face.
    const top = lerpPt(pts[6], pts[4], 0.45), bottom = pts[LM.noseBottom];
    const len = dist(top, bottom);
    ctx.beginPath();
    ctx.ellipse((top.x + bottom.x) / 2, (top.y + bottom.y) / 2, len * 0.62, dist(pts[LM.noseRight], pts[LM.noseLeft]) * 0.72,
      Math.atan2(bottom.y - top.y, bottom.x - top.x), 0, Math.PI * 2);
    ctx.fill();
  }
  const faceInner = blurTimes(read(), w, h, 3);
  return { features, faceInner };
}

/**
 * The iris color in the photo: the average of a ring between the pupil and
 * the iris edge, using only its darker pixels (that skips the catchlight and
 * any eyelid or eye white that overlaps the ring). Returns [r, g, b] or null.
 */
function irisColor(rgba, w, h, pts, center, edge) {
  const c = pts[center], r = pick(pts, edge).reduce((s, p) => s + dist(p, c), 0) / 4;
  const px = [];
  for (let y = Math.floor(c.y - r); y <= c.y + r; y++) {
    for (let x = Math.floor(c.x - r); x <= c.x + r; x++) {
      const d = Math.hypot(x - c.x, y - c.y);
      if (d < r * 0.3 || d > r * 0.85 || x < 0 || y < 0 || x >= w || y >= h) continue;
      const j = (y * w + x) * 4;
      px.push([rgba[j], rgba[j + 1], rgba[j + 2]]);
    }
  }
  if (px.length < 6) return null;
  const lum = (p) => 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  px.sort((a, b) => lum(a) - lum(b));
  const part = px.slice(Math.floor(px.length * 0.15), Math.ceil(px.length * 0.5));
  return [0, 1, 2].map((ch) => Math.round(part.reduce((s, p) => s + p[ch], 0) / part.length));
}

/**
 * Illustration face refinement, drawn on top with the face landmarks: the
 * details a digital illustrator emphasizes. Everything is blended gently
 * (multiply for darkening, screen for highlights) so the person's own eyes,
 * brows and lips still show through. It sharpens them without replacing them.
 */
function refineFace(ctx, pts, opts = {}) {
  const unit = (dist(pts[33], pts[133]) + dist(pts[263], pts[362])) / 2;
  if (unit < 5) return;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // Face shading ("contouring"): a soft warm shadow toward the edges of the
  // face and under the cheekbones gives the face shape and depth, the way
  // illustrators shade even a flatly lit (e.g. flash) photo.
  // (Along the face's own left–right axis, so it follows a tilted head, and
  // only at the sides: shading across the forehead just looks like a band.)
  const oval = pick(pts, LM.faceOval);
  const fc = centroid(oval);
  const left = pts[234], right = pts[454]; // the face's left and right edges
  ctx.save();
  ctx.beginPath();
  smoothClosedPath(ctx, oval);
  ctx.clip();
  ctx.globalCompositeOperation = 'multiply';
  const cg = ctx.createLinearGradient(left.x, left.y, right.x, right.y);
  cg.addColorStop(0, 'rgba(150, 92, 72, 0.42)');
  cg.addColorStop(0.24, 'rgba(150, 92, 72, 0)');
  cg.addColorStop(0.76, 'rgba(150, 92, 72, 0)');
  cg.addColorStop(1, 'rgba(150, 92, 72, 0.42)');
  ctx.fillStyle = cg;
  const fr = Math.max(...oval.map((p) => dist(p, fc)));
  ctx.fillRect(fc.x - fr * 1.3, fc.y - fr * 1.3, fr * 2.6, fr * 2.6);
  ctx.restore();

  // Brows: deepen them a little, like a clean painted brow.
  ctx.globalCompositeOperation = 'multiply';
  for (const brow of [LM.rightBrow, LM.leftBrow]) {
    const poly = pick(pts, brow), c = centroid(poly);
    ctx.beginPath();
    smoothClosedPath(ctx, scaleAbout(poly, c.x, c.y, 1.02, 1.12));
    ctx.fillStyle = 'rgba(70, 48, 38, 0.38)';
    ctx.fill();
  }

  const eyes = [
    { ring: LM.rightEye, iris: LM.rightIris, edge: LM.rightIrisEdge },
    { ring: LM.leftEye, iris: LM.leftIris, edge: LM.leftIrisEdge },
  ];
  eyes.forEach((e, k) => { e.color = opts.irisColors ? opts.irisColors[k] : null; });
  for (const e of eyes) {
    const ring = pick(pts, e.ring);
    const ew = dist(ring[0], ring[8]);
    const ys = ring.map((p) => p.y);
    const eh = Math.max(...ys) - Math.min(...ys);
    const upper = pick(pts, e.ring.slice(0, 9));
    const lower = pick(pts, [e.ring[8], ...e.ring.slice(9), e.ring[0]]);
    const open = eh / ew > 0.14;

    if (open) {
      // Brighten the eye whites a touch, and give the iris a defined ring,
      // a deeper pupil and a catchlight (the sparkle that makes eyes look alive).
      const ic = pts[e.iris];
      const ir = pick(pts, e.edge).reduce((s, p) => s + dist(p, ic), 0) / 4;
      ctx.save();
      ctx.beginPath();
      smoothClosedPath(ctx, ring);
      ctx.clip();
      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.fillRect(ic.x - ew, ic.y - ew, ew * 2, ew * 2);
      ctx.globalCompositeOperation = 'multiply';
      if (e.color) {
        // The person's own iris color from the photo, deepened (multiplying a
        // color by itself makes it darker and richer): brown eyes stay brown,
        // blue eyes stay blue.
        ctx.fillStyle = `rgba(${e.color[0]}, ${e.color[1]}, ${e.color[2]}, 0.6)`;
        ctx.beginPath(); ctx.arc(ic.x, ic.y, ir * 0.95, 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = 'rgba(40, 25, 20, 0.45)';
      ctx.beginPath(); ctx.arc(ic.x, ic.y, ir * 0.45, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(30, 18, 14, 0.55)';
      ctx.lineWidth = Math.max(0.8, ir * 0.14);
      ctx.beginPath(); ctx.arc(ic.x, ic.y, ir * 0.97, 0, Math.PI * 2); ctx.stroke();
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.beginPath(); ctx.arc(ic.x - ir * 0.32, ic.y - ir * 0.35, Math.max(0.8, ir * 0.24), 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.beginPath(); ctx.arc(ic.x + ir * 0.3, ic.y + ir * 0.25, Math.max(0.5, ir * 0.1), 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    // Lash line: a bold, tapered upper lid, plus a few lashes toward the outer corner.
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = 'rgba(22, 14, 11, 0.82)';
    ctx.lineWidth = unit * 0.075;
    ctx.beginPath();
    smoothOpenPath(ctx, upper);
    ctx.stroke();
    ctx.lineWidth = unit * 0.045;
    for (let k = 0; k < 4; k++) {
      const p = upper[k], q = upper[k + 1];
      const len = Math.max(1e-3, dist(p, q));
      const nx = (q.y - p.y) / len, ny = -(q.x - p.x) / len; // normal to the lid
      const dir = ny > 0 ? -1 : 1;                            // point it upward
      const l = unit * (0.16 - k * 0.025);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.quadraticCurveTo(p.x + nx * dir * l * 0.6 - (q.x - p.x) * 0.3, p.y + ny * dir * l * 0.6, p.x + nx * dir * l - (q.x - p.x) * 0.5, p.y + ny * dir * l);
      ctx.stroke();
    }
    // Soft lower lid.
    ctx.strokeStyle = 'rgba(60, 38, 30, 0.35)';
    ctx.lineWidth = unit * 0.035;
    ctx.beginPath();
    smoothOpenPath(ctx, lower);
    ctx.stroke();
  }

  // Lips: a little richer color, a defined parting line, and a gloss highlight.
  // Only the lips themselves are tinted (outer outline minus the inner one,
  // the "evenodd" fill rule), so the teeth in a smile stay white.
  const outer = pick(pts, LM.lipsOuter);
  ctx.globalCompositeOperation = 'multiply';
  ctx.fillStyle = 'rgba(200, 95, 100, 0.32)';
  ctx.beginPath();
  smoothClosedPath(ctx, outer);
  smoothClosedPath(ctx, pick(pts, LM.lipsInner));
  ctx.fill('evenodd');
  ctx.globalCompositeOperation = 'source-over';
  ctx.strokeStyle = 'rgba(70, 28, 28, 0.45)';
  ctx.lineWidth = unit * 0.04;
  ctx.beginPath();
  smoothOpenPath(ctx, pick(pts, LM.lipLine));
  ctx.stroke();
  const shine = lerpPt(pts[17], pts[14], 0.45);
  const g = ctx.createRadialGradient(shine.x, shine.y, 0, shine.x, shine.y, unit * 0.28);
  g.addColorStop(0, 'rgba(255, 255, 255, 0.45)');
  g.addColorStop(1, 'rgba(255, 255, 255, 0)');
  ctx.globalCompositeOperation = 'screen';
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.ellipse(shine.x, shine.y, unit * 0.28, unit * 0.1, 0, 0, Math.PI * 2);
  ctx.fill();

  // Pencil sketches (the Cartoon) suggest the nose the way artists do: two
  // light curves for the nostril wings, from the side of the nose curling in
  // under it. (The nose is soft in most photos, so the traced lines often
  // miss it.)
  if (opts.noseSketch) {
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = 'rgba(58, 50, 52, 0.55)';
    ctx.lineWidth = unit * 0.045;
    for (const wing of [[48, 64, 98, 97], [278, 294, 327, 326]]) {
      ctx.beginPath();
      smoothOpenPath(ctx, pick(pts, wing));
      ctx.stroke();
    }
  }

  // Nose: a soft highlight on the tip.
  const tip = pts[4];
  const ng = ctx.createRadialGradient(tip.x, tip.y - unit * 0.05, 0, tip.x, tip.y - unit * 0.05, unit * 0.18);
  ng.addColorStop(0, 'rgba(255, 255, 255, 0.28)');
  ng.addColorStop(1, 'rgba(255, 255, 255, 0)');
  ctx.fillStyle = ng;
  ctx.beginPath();
  ctx.arc(tip.x, tip.y - unit * 0.05, unit * 0.18, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * Pencil texture: graphite on paper isn't solid. It's made of fine streaks
 * that follow the direction of the stroke. We make that texture the way
 * "line integral convolution" pencil-drawing methods do: start with random
 * noise (like paper grain), then smear it ALONG the edge flow. Every pixel
 * becomes the average of the noise along a short streamline, so the noise
 * turns into tiny parallel streaks that follow the drawing's lines.
 * Returns values around 0.5 (0 = light, 1 = dark streak).
 */
function pencilStreaks(flow, w, h, length) {
  const n = w * h;
  const noise = new Float32Array(n);
  for (let i = 0; i < n; i++) noise[i] = Math.random();
  const lic = flowSmooth(noise, flow.tx, flow.ty, w, h, length);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += lic[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (lic[i] - mean) ** 2;
  const std = Math.sqrt(variance / n) || 1;
  for (let i = 0; i < n; i++) lic[i] = clamp(0.5 + (lic[i] - mean) / (3.5 * std), 0, 1);
  return lic;
}

/**
 * Style presets for the shared engine below (buildStylized). Both styles
 * run the same steps; these numbers decide the look.
 *   Illustration: gentle smoothing, soft many-level shading, colored ink.
 *   Cartoon: stronger flattening, 2–3 clear shading tones per area ("cel
 *   shading"), pencil-sketch lines with a graphite texture, on paper.
 * Skin settings: skinRich / skinWarm = how much richer (×) and warmer (+)
 * the evened skin tone gets; skinCurve = how much of the contrast curve skin
 * gets (1 = all). The Cartoon keeps skin close to the person's real tone.
 * flattenTexture / keepStructure (Cartoon only): how strongly busy texture
 * is flattened, and how much directional background structure (streaks,
 * wood grain) is protected from smearing. See textureMap.
 */
const STYLE_PRESETS = {
  illustration: {
    mode: 'illustration',
    sigmaS: 32, sigmaR: { bg: 26, skin: 20, hair: 8, other: 11, even: 16 },
    chromaBlur: 1.2, vibrance: 1.3, vibranceMuted: 0.55, skinVibrance: 1.15, warm: 1,
    skinRich: 1.1, skinWarm: 2, skinCurve: 1,
    levels: 12, sharpness: 2.2, mixPerson: 0.45, mixBg: 0.8, mixEven: 0.55,
    sCurve: 0.42, gamma: 1.07, lighten: 0,
    flattenTexture: 0, keepStructure: 0,
    lines: 'ink',
  },
  cartoon: {
    mode: 'cartoon',
    sigmaS: 44, sigmaR: { bg: 34, skin: 28, hair: 12, other: 18, even: 24 },
    chromaBlur: 2.4, vibrance: 1.25, vibranceMuted: 0.3, skinVibrance: 1.02, warm: 0.6,
    skinRich: 1, skinWarm: 1, skinCurve: 0.5,
    levels: 6, sharpness: 3.4, mixPerson: 0.85, mixBg: 0.9, mixEven: 0.88,
    sCurve: 0.25, gamma: 1.0, lighten: 0.14,
    flattenTexture: 4, keepStructure: 0.8,
    lines: 'pencil',
  },
};

/** Build the Illustration result from an <img>. */
function buildIllustration(image, detail, onStatus = () => {}) {
  return buildStylized(image, detail, onStatus, STYLE_PRESETS.illustration);
}

/**
 * Build the Cartoon result from an <img>: a pencil sketch of the photo as it
 * is, colored in with clean cartoon colors (see STYLE_PRESETS.cartoon).
 */
function buildSketchCartoon(image, detail, onStatus = () => {}) {
  return buildStylized(image, detail, onStatus, STYLE_PRESETS.cartoon);
}

/**
 * The shared engine for the Illustration and Cartoon styles.
 * Photo → (optional) person/face finding → Lab → region-aware edge-aware
 * smoothing → even skin tone → bold color → lines (ink or pencil) →
 * shading bands → contrast → compose → face refinement → pen strokes for
 * the live drawing.
 */
async function buildStylized(image, detail, onStatus, preset) {
  const { width: w, height: h, data } = downscaleImage(image, SETTINGS.illustrationSide);
  const n = w * h;
  const scale = Math.max(w, h) / 1000; // sizes below are tuned for a 1000px image
  const d = clamp(detail, 1, 100) / 100;

  // Only genuinely dark photos get a lift (up to 60% of full auto-levels).
  // Brightening normally lit photos washed the colors out; bold styles keep
  // their deep darks.
  const photo = new Uint8ClampedArray(data);
  const avgLight = averageLightness(photo);
  const lift = clamp((0.38 - avgLight) / 0.18, 0, 1) * 0.6;
  if (lift > 0) {
    const lut = makeLevelsLut(photo);
    for (let v = 0; v < 256; v++) lut[v] = v + (lut[v] - v) * lift;
    applyLut(photo, lut);
  }
  const photoCanvas = makeCanvas(w, h);
  photoCanvas.getContext('2d').putImageData(new ImageData(photo, w, h), 0, 0);

  // Regions and faces (optional: if MediaPipe can't load, the whole photo is treated evenly).
  let person = null, skin = null, hair = null, clothes = null, faces = [];
  try {
    onStatus('Finding the person and their face…');
    await nextFrame();
    const segmenter = await withTimeout(getSegmenter(), 30000);
    const seg = segmenter.segment(photoCanvas);
    const conf = seg.confidenceMasks.map((m) => {
      const a = new Float32Array(m.getAsFloat32Array());
      return m.width === w && m.height === h ? a : resizeMask(a, m.width, m.height, w, h);
    });
    seg.close();
    const soft = (arr) => blurTimes(arr, w, h, 2);
    person = soft(Float32Array.from(conf[SEG.BG], (v) => 1 - v));
    skin = soft(Float32Array.from(conf[SEG.FACE], (v, i) => v + conf[SEG.BODY][i]));
    hair = soft(conf[SEG.HAIR]);
    clothes = soft(Float32Array.from(conf[SEG.CLOTHES], (v, i) => v + (conf[SEG.OTHER] ? conf[SEG.OTHER][i] : 0)));
    faces = await findFaceLandmarks(photoCanvas);
  } catch (err) {
    console.warn('Person/face finder unavailable; styling the whole photo evenly:', err);
  }

  onStatus('Smoothing into clean surfaces…');
  await nextFrame();
  const lab = rgbaToLab(photo);
  const guide = [new Float32Array(lab.L), new Float32Array(lab.A), new Float32Array(lab.B)];
  const { features, faceInner } = faces.length ? faceMasks(faces, w, h) : { features: null, faceInner: null };

  // The pencil sketch is traced from a LIGHTLY smoothed copy of the photo (so
  // fine detail survives). Its edge flow also shows where the photo has real
  // structure and where it's only busy texture (textureMap).
  let lineL = null, flow = null, tex = null, along = null, lineSourceL = null;
  if (preset.lines === 'pencil') {
    lineSourceL = new Float32Array(guide[0]); // the photo's own lightness, untouched
    lineL = new Float32Array(guide[0]);
    domainTransformSmooth([lineL], guide, w, h, 16 * scale, 3 + (1 - d) * 6, 3);
    flow = edgeTangentFlow(lineL, w, h, 3 * scale);
    tex = textureMap(lineL, w, h, 8 * scale);

    // Directional texture in the background (wall streaks, wood grain):
    // smooth it ALONG its own direction first, so it turns into clean,
    // simple streaks of color, and then smooth it less in the main pass
    // below (which would otherwise smear the streaks into blotches).
    if (preset.keepStructure > 0) {
      along = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        along[i] = smoothstep(0.6, 0.8, tex.coherence[i]) * smoothstep(1.2, 2.2, tex.energy[i]) * (person ? 1 - clamp(person[i], 0, 1) : 1);
      }
      along = blurTimes(along, w, h, 3);
      flowSmoothBlend([lab.L, lab.A, lab.B], flow.tx, flow.ty, w, h, 14 * scale, along);
      for (let c = 0; c < 3; c++) guide[c].set([lab.L, lab.A, lab.B][c]);
    }
  }

  // Per-pixel "edge sensitivity" (sigmaR, in Lab units): bigger = smoother.
  // The Edge detail slider scales it: more detail → less smoothing.
  const detailFactor = 1.5 - d * 0.9; // 1.5 … 0.6
  const R = preset.sigmaR;
  const sigmaR = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let r = R.even, notSkinOrHair = 1;
    if (person) {
      const p = clamp(person[i], 0, 1), s = clamp(skin[i], 0, 1), hr = clamp(hair[i], 0, 1);
      const other = Math.max(0, 1 - s - hr);
      const personR = R.skin * s + R.hair * hr + R.other * other; // skin smooth, hair detailed, clothes in between
      r = (1 - p) * R.bg + p * personR;                             // background more painterly
      notSkinOrHair = 1 - clamp(p * (s + hr), 0, 1);
    }
    // Busy texture (carpet, lace) is flattened into a clean fill, the way an
    // artist colors it in; streaks (above) are smoothed less, so they stay.
    if (tex) r *= 1 + preset.flattenTexture * tex.texture[i] * notSkinOrHair;
    if (along) r *= 1 - preset.keepStructure * along[i];
    sigmaR[i] = r * detailFactor;
    if (features) sigmaR[i] += (4 - sigmaR[i]) * features[i]; // eyes and teeth stay crisp
  }
  domainTransformSmooth([lab.L, lab.A, lab.B], guide, w, h, preset.sigmaS * scale, sigmaR, 3);

  // Clean, richer color: blur the color axes a little (removes blotchy color
  // noise, like a painter's even color fills). More blur = flatter fills.
  const A = gaussianBlurSigma(lab.A, w, h, preset.chromaBlur * scale), B = gaussianBlurSigma(lab.B, w, h, preset.chromaBlur * scale);
  if (features) {
    // …but not in the eyes: blurring a small iris with the eye white turns it gray.
    for (let i = 0; i < n; i++) { A[i] += (lab.A[i] - A[i]) * features[i]; B[i] += (lab.B[i] - B[i]) * features[i]; }
  }
  const L = lab.L;
  const skinInner = skin ? blurTimes(Float32Array.from(skin, (v) => (v > 0.5 ? 1 : 0)), w, h, 3) : null;

  // Even skin tone. Camera flash and strong light leave glare that makes skin
  // look pale and washed out; artists paint skin as an even, warm tone with
  // soft shading. So in skin areas we (1) compress highlights above the
  // skin's typical lightness, and (2) pull the glare's washed-out color back
  // toward the skin's real mid-tone color.
  // (The eyes and teeth are left out: their whites are bright on purpose.)
  if (skin) {
    const vals = [];
    for (let i = 0; i < n; i += 3) if (skin[i] > 0.6 && !(features && features[i] > 0.1)) vals.push(i);
    if (vals.length > 200) {
      vals.sort((a, b) => L[a] - L[b]);
      // Reference = the 35th-percentile skin lightness. (Not the median: with
      // flash, most of the face IS glare, so the median is glare-bright too.)
      const ref = L[vals[Math.floor(vals.length * 0.35)]];
      let sa = 0, sb = 0, cnt = 0;
      for (let k = Math.floor(vals.length * 0.2); k < Math.floor(vals.length * 0.5); k++) { sa += A[vals[k]]; sb += B[vals[k]]; cnt++; }
      // The real skin color (optionally a touch richer and warmer, see the presets).
      sa = (sa / cnt) * preset.skinRich; sb = (sb / cnt) * preset.skinRich + preset.skinWarm;
      for (let i = 0; i < n; i++) {
        const s = clamp(skin[i], 0, 1) * (features ? 1 - features[i] : 1);
        if (s < 0.05 || L[i] <= ref) continue;
        const over = L[i] - ref;
        const t = clamp(over / 20, 0, 1) * 0.9 * s; // how much of this pixel is glare
        L[i] -= over * 0.6 * s;
        A[i] += (sa - A[i]) * t;
        B[i] += (sb - B[i]) * t;
      }
    }
  }
  // Bold color with "vibrance": muted colors get the biggest boost (a
  // grayish navy couch becomes navy, a gray-brown wall becomes warm brown),
  // already-vivid colors a smaller one, and true neutrals (white, gray,
  // black; chroma under ~3) are left alone so they don't pick up a tint.
  // Skin is boosted less so it stays natural instead of turning orange.
  for (let i = 0; i < n; i++) {
    const chroma = Math.hypot(A[i], B[i]);
    if (chroma < 1.5) continue;
    const neutralGuard = clamp((chroma - 1.5) / 3, 0, 1);
    let k = preset.vibrance + preset.vibranceMuted * Math.exp(-chroma / 16);
    if (skin) k -= (k - preset.skinVibrance) * clamp(skin[i], 0, 1);
    k = 1 + (k - 1) * neutralGuard * (features ? 1 - features[i] : 1); // eye color stays true
    A[i] *= k; B[i] *= k;
  }
  // A warm grade: nudge the midtones toward warm (a little more yellow/red),
  // the golden light illustrations often have. Skin gets less (it's warm
  // already); deep shadows and highlights are untouched.
  for (let i = 0; i < n; i++) {
    const mid = clamp(1 - Math.abs(L[i] - 55) / 40, 0, 1);
    if (mid <= 0) continue;
    const wgt = mid * preset.warm * (skin ? 1 - 0.6 * clamp(skin[i], 0, 1) : 1);
    B[i] += 4 * wgt;
    A[i] += 1.2 * wgt;
  }

  // Lines. The Edge detail slider sets how many.
  onStatus(preset.lines === 'pencil' ? 'Sketching in pencil…' : 'Inking…');
  await nextFrame();
  let ink;
  if (preset.lines === 'pencil') {
    // Pencil sketch of the photo as it is, traced from lineL (above) with the
    // flow-based ink method from Line art: a main layer of flowing lines plus
    // a finer detail layer.
    const main = xdogLines(lineL, w, h, 1.0 * scale, { tau: 1, eps: -1.3 + d * 0.8, phi: 2.2, flow: { ...flow, length: 14 * scale } });
    const fine = xdogLines(lineL, w, h, 0.6 * scale, { tau: 1, eps: -1.0 + d * 0.7, phi: 2.2, flow: { ...flow, length: 10 * scale } });
    const combined = new Float32Array(n);
    for (let i = 0; i < n; i++) combined[i] = Math.min(main[i], 1 - (1 - fine[i]) * 0.6);
    removeInkSpecks(combined, w, h, Math.round(8 * scale * scale));
    // Pencil lines are a touch thicker than ink: each pixel takes the darkest
    // of itself and its 4 neighbors ("dilation"), so the sketch reads clearly
    // under the colors. Skin keeps most of its lines (a sketch shows the face).
    ink = new Float32Array(n);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let v = combined[i];
        if (x > 0) v = Math.min(v, combined[i - 1] * 0.5 + 0.5 * combined[i]);
        if (x < w - 1) v = Math.min(v, combined[i + 1] * 0.5 + 0.5 * combined[i]);
        if (y > 0) v = Math.min(v, combined[i - w] * 0.5 + 0.5 * combined[i]);
        if (y < h - 1) v = Math.min(v, combined[i + w] * 0.5 + 0.5 * combined[i]);
        ink[i] = v;
      }
    }
    // Keep the lines an artist would draw, and only hint at the rest:
    //  - busy texture (carpet, lace) gets a light hint of its pattern, not
    //    scribbles (hair strands are kept: they ARE the hair);
    //  - faint marks where the photo has almost no contrast fade out, unless
    //    they follow a clear direction (wall streaks, wood grain);
    //  - inside the face, stray lines like a shadow edge across a cheek are
    //    softened, while the eyes, brows, nose tip and lips keep theirs;
    //  - smooth skin keeps most of its lines (a sketch shows the face).
    for (let i = 0; i < n; i++) {
      const hairOrSkin = person ? clamp((hair[i] + skin[i]) * 1.5, 0, 1) : 0;
      let keep = 1 - 0.45 * tex.texture[i] * (1 - hairOrSkin);
      keep *= 0.3 + 0.7 * Math.max(smoothstep(1.5, 3.5, tex.energy[i]), 0.7 * smoothstep(0.65, 0.85, tex.coherence[i]));
      if (faceInner) keep *= 1 - 0.5 * faceInner[i];
      if (skinInner) keep *= 1 - 0.3 * clamp(skinInner[i] * 1.2 - 0.2, 0, 1);
      ink[i] = 1 - (1 - ink[i]) * keep;
    }
  } else {
    // Ink traced from the smoothed lightness BEFORE sharpening (so hair
    // texture doesn't become speckles); softened inside smooth skin.
    flow = edgeTangentFlow(L, w, h, 3 * scale);
    ink = xdogLines(L, w, h, 1.0 * scale, { eps: -0.6 + d * 0.9, phi: 1.9, flow: { ...flow, length: 10 * scale } });
    removeInkSpecks(ink, w, h, Math.round(16 * scale * scale));
    if (skinInner) {
      for (let i = 0; i < n; i++) ink[i] = 1 - (1 - ink[i]) * (1 - 0.85 * clamp(skinInner[i] * 1.2 - 0.2, 0, 1));
    }
  }

  // Crisp edges everywhere except smooth skin, plus extra crispness for hair
  // strands and fabric folds: an "unsharp mask" (add back the difference
  // between the image and a blurred copy).
  const Lb = gaussianBlurSigma(L, w, h, 1.6 * scale);
  for (let i = 0; i < n; i++) {
    let amount = person
      ? 0.35 * (1 - (skinInner ? skinInner[i] : 0)) + 0.5 * clamp(hair[i], 0, 1) + 0.25 * clamp(clothes[i], 0, 1)
      : 0.35;
    if (features) amount += 0.3 * features[i]; // crisp eyes and teeth
    L[i] = clamp(L[i] + (L[i] - Lb[i]) * amount, 0, 100);
  }

  // Shading bands: a few clear tones (cartoon) or soft many-level shading
  // (illustration); the background is stepped more than the person. The
  // eyes and teeth keep their own real shading.
  const mix = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    mix[i] = person ? preset.mixPerson + (preset.mixBg - preset.mixPerson) * (1 - clamp(person[i], 0, 1)) : preset.mixEven;
    if (features) mix[i] *= 1 - features[i];
  }
  softQuantize(L, preset.levels, preset.sharpness, mix);

  // Bolder light and shadow: an S-curve on lightness (done in Lab, so it
  // adds contrast without shifting colors), plus the preset's gamma and a
  // slight lightening for the Cartoon's colored-in-on-paper feel.
  for (let i = 0; i < n; i++) {
    const x = clamp(L[i] / 100, 0, 1);
    const sk = skin ? clamp(skin[i], 0, 1) : 0;
    const curve = preset.sCurve * (1 - (1 - preset.skinCurve) * sk); // skin can get a gentler curve
    const s = Math.pow(x - (curve * Math.sin(2 * Math.PI * x)) / (2 * Math.PI), preset.gamma);
    // Lighten midtones and highlights only (a smooth ramp from 15% to 45%
    // lightness), so dark hair and clothes stay dark while walls and fabrics
    // get the lighter "colored in" look. Skin, eyes and teeth are not
    // lightened: skin keeps the person's real tone.
    const t = clamp((s - 0.15) / 0.3, 0, 1);
    const lighten = preset.lighten * (1 - sk) * (features ? 1 - features[i] : 1);
    L[i] = 100 * (s + lighten * (1 - s) * t * t * (3 - 2 * t));
  }

  onStatus('Coloring…');
  await nextFrame();
  const painted = labToRgba(L, A, B, new Uint8ClampedArray(n * 4));

  // A thin, crisp outline along the person's edge (threshold the soft
  // segmentation mask, blur it just 1 pass, and use its gradient).
  let outline = null;
  if (person) {
    const edge = gaussianBlur(Float32Array.from(person, (v) => (v > 0.5 ? 1 : 0)), w, h);
    const { magnitude } = sobelEdges(edge, w, h);
    outline = Float32Array.from(magnitude, (m) => clamp((m - 0.6) * 0.45, 0, 0.55));
  }

  if (preset.lines === 'pencil') {
    // Compose the pencil sketch over the colors, on paper:
    //  - line darkness is broken up by the pencil streak texture, so lines
    //    look like graphite, not ink;
    //  - darker areas get a light layer of pencil shading that follows the
    //    form (the same streaks), like an artist shading with the side of
    //    the pencil;
    //  - busy texture that was flattened (carpet, lace) gets a light pencil
    //    hint of its pattern: the darker bits of the photo's fine detail,
    //    drawn in graphite;
    //  - the whole picture sits on a subtle paper grain.
    const streak = pencilStreaks(flow, w, h, 9 * scale);
    const grain = makePaperGrain(w, h);
    const fineDetail = gaussianBlurSigma(lineSourceL, w, h, 2.5 * scale);
    for (let i = 0; i < n; i++) fineDetail[i] = lineSourceL[i] - fineDetail[i]; // negative = darker than around it
    const graphite = [44, 41, 46];
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      let dark = 1 - ink[i];
      if (outline) dark = Math.max(dark, outline[i] * 1.2);
      const texture = clamp(0.8 + (streak[i] - 0.5) * 1.1, 0.45, 1);
      let pd = clamp(dark * texture * 1.05, 0, 0.95);
      const tone = clamp((45 - L[i]) / 35, 0, 1); // how dark this area is
      const shade = tone * clamp((streak[i] - 0.45) * 2.5, 0, 1) * 0.22;
      pd = 1 - (1 - pd) * (1 - shade);
      const hairOrSkin = person ? clamp((hair[i] + skin[i]) * 1.5, 0, 1) : 0;
      const hint = tex.texture[i] * (1 - hairOrSkin) * clamp(-fineDetail[i] / 8, 0, 1) * 0.4;
      pd = 1 - (1 - pd) * (1 - hint);
      const g = 1 + (grain[i] - 1) * 0.7;
      for (let c = 0; c < 3; c++) painted[j + c] = (painted[j + c] * (1 - pd) + graphite[c] * pd) * g;
    }
  } else {
    // Ink is colored by what's underneath (dark brown on skin, near-black on
    // hair), like an illustrator's colored line art, never flat gray.
    const contrast = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v++) {
      // A gentle S-curve: y = x − a·sin(2πx)/2π is steeper in the midtones
      // (more punch) and flatter at the ends (no crushed blacks or blown whites).
      const x = v / 255;
      contrast[v] = 255 * (x - (0.15 * Math.sin(2 * Math.PI * x)) / (2 * Math.PI));
    }
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      let e = ink[i];
      if (outline) e = Math.min(e, 1 - outline[i] * 1.2);
      const t = e + (1 - e) * 0.1; // ink never goes below 10% of the color: bold, but colored, lines
      painted[j] = contrast[(painted[j] * t) | 0];
      painted[j + 1] = contrast[(painted[j + 1] * t) | 0];
      painted[j + 2] = contrast[(painted[j + 2] * t) | 0];
    }
  }

  // Faces: refine eyes, brows, lips and nose from the landmarks.
  if (faces.length) {
    onStatus(faces.length > 1 ? `Refining ${faces.length} faces…` : 'Refining the face…');
    await nextFrame();
    const fc = makeCanvas(w, h);
    const fctx = fc.getContext('2d', { willReadFrequently: true });
    fctx.putImageData(new ImageData(painted, w, h), 0, 0);
    for (const pts of faces) {
      refineFace(fctx, pts, {
        irisColors: [irisColor(photo, w, h, pts, LM.rightIris, LM.rightIrisEdge), irisColor(photo, w, h, pts, LM.leftIris, LM.leftIrisEdge)],
        noseSketch: preset.lines === 'pencil',
      });
    }
    painted.set(fctx.getImageData(0, 0, w, h).data);
  }

  // Pen strokes for the live drawing: the lines' center lines.
  onStatus('Preparing the sketch…');
  await nextFrame();
  const lineBin = new Uint8Array(n);
  for (let i = 0; i < n; i++) lineBin[i] = ink[i] < 0.45 || (outline && outline[i] > 0.3) ? 1 : 0;
  // Grow by 1 pixel to bridge tiny gaps (see Line art), then thin to center lines.
  const bin = new Uint8Array(n);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      bin[i] = lineBin[i] | lineBin[i - 1] | lineBin[i + 1] | lineBin[i - w] | lineBin[i + w];
    }
  }
  thinLines(bin, w, h);
  let raw = filterAndSimplify(traceContours(bin, w, h), Math.round((preset.lines === 'pencil' ? 5 : 10) * scale), SETTINGS.rdpEpsilon);
  raw = suppressTexture(raw, w, h, { maxDensity: preset.lines === 'pencil' ? 0.15 : 0.12 });
  const strokes = finalizeStrokes(raw);
  const outScale = outputScale(w, h);
  for (const s of strokes) {
    if (preset.lines === 'pencil') {
      s.outlineWidth = (1.0 + 0.9 * strokeProminence(s)) / outScale;
      s.outlineColor = '#3a363a';
      s.outlineAlpha = 0.8;
    } else {
      s.outlineWidth = (1.1 + 0.9 * strokeProminence(s)) / outScale;
      s.outlineColor = '#2a1c16';
      s.outlineAlpha = 0.9;
    }
  }

  const { labels: washLabels, centers } = await runPaintStage(painted, w, h, { radii: [], k: 12, maxIter: 10 });
  const art = makePaintedArt({
    w, h, painted, grain: makePaperGrain(w, h), labels: washLabels, centers,
    customStrokes: strokes, outlineOpacity: 1, faces: faces.length,
  });
  art.mode = preset.mode;
  art.linesFirst = true; // sketch first, then color in
  art.linesBaked = true;
  return art;
}


/* =============================================================================
   5. LIVE DRAWING
   ============================================================================= */

/** Point on a quadratic Bézier with "blossoming": b(u, v). b(t, t) is the point at t. */
function blossom(p0, p1, p2, u, v) {
  return (1 - u) * (1 - v) * p0 + ((1 - u) * v + u * (1 - v)) * p1 + u * v * p2;
}

/**
 * A "pen" that walks along all strokes in order. advanceTo(ctx, target)
 * draws everything from where it stopped last time up to `target` pixels of
 * total length, including partial curve segments. The piece of a curve
 * between t0 and t1 is itself a quadratic curve with start b(t0,t0),
 * control b(t0,t1) and end b(t1,t1).
 * Because progress is measured in *length*, the pen moves at an even speed
 * no matter how many or how few strokes the picture has.
 */
function createStrokePen(strokes, applyStyle) {
  const totalLength = strokes.reduce((sum, s) => sum + s.length, 0);
  let si = 0, gi = 0, t = 0, drawn = 0; // stroke index, segment index, position within segment, length drawn

  function advanceTo(ctx, target) {
    if (target >= totalLength) target = Infinity; // finish everything, immune to rounding error
    let pathOpen = false;
    while (si < strokes.length && drawn < target) {
      const stroke = strokes[si];
      const seg = stroke.segs[gi];
      if (!pathOpen) {
        applyStyle(ctx, stroke);
        ctx.beginPath();
        ctx.moveTo(blossom(seg.x0, seg.cx, seg.x1, t, t), blossom(seg.y0, seg.cy, seg.y1, t, t));
        pathOpen = true;
      }
      const remaining = seg.len * (1 - t);
      const budget = target - drawn;
      let t1;
      if (budget >= remaining) { t1 = 1; drawn += remaining; }
      else { t1 = t + budget / seg.len; drawn = target; }
      ctx.quadraticCurveTo(
        blossom(seg.x0, seg.cx, seg.x1, t, t1), blossom(seg.y0, seg.cy, seg.y1, t, t1),
        blossom(seg.x0, seg.cx, seg.x1, t1, t1), blossom(seg.y0, seg.cy, seg.y1, t1, t1)
      );
      t = t1;
      if (t >= 1) {
        t = 0;
        gi++;
        if (gi >= stroke.segs.length) {
          gi = 0; si++;
          ctx.stroke();
          pathOpen = false;
        }
      }
    }
    if (pathOpen) ctx.stroke();
  }

  return { totalLength, advanceTo };
}

/**
 * Drive drawAt(progress) with requestAnimationFrame for `durationMs`.
 * Resolves true when finished, false if cancelled (user pressed Back).
 */
function animateDrawing(drawAt, durationMs, onProgress, isCancelled) {
  return new Promise((resolve) => {
    let start = null;
    const tick = (now) => {
      if (isCancelled()) return resolve(false);
      if (start === null) start = now;
      const p = Math.min(1, (now - start) / durationMs);
      drawAt(p);
      onProgress(p);
      if (p < 1) requestAnimationFrame(tick);
      else resolve(true);
    };
    requestAnimationFrame(tick);
  });
}

/**
 * Keep the finished picture on screen for the video. A captured canvas stream
 * may only send frames when the canvas changes, so we redraw a snapshot of the
 * final image every frame to make sure the video actually contains the hold.
 */
function holdFinalFrame(canvas, ms, isCancelled) {
  const snap = makeCanvas(canvas.width, canvas.height);
  snap.getContext('2d').drawImage(canvas, 0, 0);
  const ctx = canvas.getContext('2d');
  return new Promise((resolve) => {
    const start = performance.now();
    const tick = (now) => {
      if (isCancelled()) return resolve();
      ctx.drawImage(snap, 0, 0);
      if (now - start < ms) requestAnimationFrame(tick);
      else resolve();
    };
    requestAnimationFrame(tick);
  });
}


/* =============================================================================
   6. EXPORT & SHARING
   ============================================================================= */

/**
 * Pick a video format this browser can record. Chrome/Firefox/Edge support
 * WebM (VP9 or VP8). Safari only records MP4, so that's the last fallback.
 * With `withAudio`, formats that name an audio codec (Opus for WebM, AAC for
 * MP4) are tried first, so the music ends up in the video.
 * Returns { mime, ext } or null if recording isn't possible at all.
 */
function pickVideoFormat(withAudio = false) {
  if (typeof MediaRecorder === 'undefined' || typeof HTMLCanvasElement.prototype.captureStream !== 'function') return null;
  const candidates = [
    ...(withAudio ? [
      { mime: 'video/webm;codecs=vp9,opus', ext: 'webm' },
      { mime: 'video/webm;codecs=vp8,opus', ext: 'webm' },
      { mime: 'video/mp4;codecs=avc1,mp4a.40.2', ext: 'mp4' },
    ] : []),
    { mime: 'video/webm;codecs=vp9', ext: 'webm' },
    { mime: 'video/webm;codecs=vp8', ext: 'webm' },
    { mime: 'video/webm', ext: 'webm' },
    { mime: 'video/mp4;codecs=avc1', ext: 'mp4' },
    { mime: 'video/mp4', ext: 'mp4' },
  ];
  for (const c of candidates) {
    try { if (MediaRecorder.isTypeSupported(c.mime)) return c; } catch (e) { /* keep trying */ }
  }
  return null;
}

/**
 * Start recording a canvas (plus, optionally, the audio tracks of `audioStream`).
 * Returns { stop(): Promise<Blob>, cancel() }.
 */
function startRecording(canvas, format, audioStream = null) {
  const canvasStream = canvas.captureStream(30);
  const tracks = [...canvasStream.getVideoTracks()];
  if (audioStream) tracks.push(...audioStream.getAudioTracks());
  const stream = new MediaStream(tracks);
  const recorder = new MediaRecorder(stream, { mimeType: format.mime, videoBitsPerSecond: 6000000, audioBitsPerSecond: 128000 });
  const chunks = [];
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  recorder.start(250); // hand us data every 250ms
  // Only stop the canvas track: the music's audio track is shared and reused by later recordings.
  const endTracks = () => canvasStream.getTracks().forEach((t) => t.stop());

  return {
    stop: () => new Promise((resolve) => {
      const finish = () => { endTracks(); resolve(new Blob(chunks, { type: format.mime.split(';')[0] })); };
      if (recorder.state === 'inactive') return finish();
      recorder.onstop = finish;
      recorder.stop();
    }),
    cancel: () => {
      try { if (recorder.state !== 'inactive') recorder.stop(); } catch (e) { /* ignore */ }
      endTracks();
    },
  };
}

const canvasToBlob = (canvas) => new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));

/** Trigger a file download for a Blob. */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/** Copy this site's address (without ?query or #hash) to the clipboard. */
async function copySiteLink() {
  const url = location.href.split('#')[0].split('?')[0];
  try {
    await navigator.clipboard.writeText(url);
    return true;
  } catch (e) {
    // Older browsers / non-HTTPS pages: the classic hidden-textarea trick
    const ta = document.createElement('textarea');
    ta.value = url;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
    ta.remove();
    return ok;
  }
}


/* =============================================================================
   6b. MUSIC
   -----------------------------------------------------------------------------
   Background music for the live drawing, made with the Web Audio API.
   The built-in songs aren't audio files: they're COMPOSED IN CODE. A tiny
   "sequencer" schedules notes bar by bar, and each note is an oscillator
   (a tone generator: sine, triangle, square or sawtooth wave) shaped by a
   gain "envelope" (quick fade in, then a decay), sometimes through a
   filter. Drums are a pitch-dropping sine (kick) and filtered noise (snare,
   hi-hat). So there's nothing to download and no copyright to worry about.
   A song file the user picks is decoded and played through the same output.
   Everything goes into one "master" volume node, which feeds both the
   speakers and a MediaStream, so the recorder can put the music in the video.
   ============================================================================= */

const music = {
  ctx: null,          // the AudioContext (created on first use)
  master: null,       // master volume
  recordDest: null,   // MediaStream output for the video recorder
  noise: null,        // 1 second of white noise, reused for drums
  userBuffer: null,   // decoded song the user picked
  userName: '',
  preview: null,      // song playing from the Preview button
};

/**
 * Create (or wake up) the audio system. Browsers only allow sound to start
 * after a user action, so this is called directly inside click handlers.
 */
function ensureAudio() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;
  if (!music.ctx) {
    const ctx = new AC();
    music.ctx = ctx;
    music.master = ctx.createGain();
    music.master.gain.value = 0.8;
    music.master.connect(ctx.destination);
    if (ctx.createMediaStreamDestination) {
      music.recordDest = ctx.createMediaStreamDestination();
      music.master.connect(music.recordDest);
    }
    const len = ctx.sampleRate;
    music.noise = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = music.noise.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  }
  if (music.ctx.state === 'suspended') music.ctx.resume();
  return music.ctx;
}

/** MIDI note number → frequency in Hz (A4 = note 69 = 440 Hz; each semitone is ×2^(1/12)). */
const midiHz = (n) => 440 * 2 ** ((n - 69) / 12);

/** A small seeded random generator, so each song's melody is the same every time it loops. */
function seededRandom(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One note: oscillator → (optional low-pass filter) → envelope → `dest`. */
function playNote(ctx, dest, { note, start, dur, type = 'sine', gain = 0.15, attack = 0.01, release = 0.25, cutoff = 0, detune = 0 }) {
  const osc = ctx.createOscillator();
  osc.type = type;
  osc.frequency.value = midiHz(note);
  osc.detune.value = detune;
  const env = ctx.createGain();
  env.gain.setValueAtTime(0.0001, start);
  env.gain.exponentialRampToValueAtTime(gain, start + attack);
  env.gain.setValueAtTime(gain, start + Math.max(attack, dur * 0.6));
  env.gain.exponentialRampToValueAtTime(0.0001, start + dur + release);
  let node = osc;
  if (cutoff) {
    const f = ctx.createBiquadFilter();
    f.type = 'lowpass';
    f.frequency.value = cutoff;
    osc.connect(f);
    node = f;
  }
  node.connect(env).connect(dest);
  osc.start(start);
  osc.stop(start + dur + release + 0.05);
}

/** Kick drum: a sine wave whose pitch drops fast from 130 Hz to 45 Hz. */
function playKick(ctx, dest, t, gain = 0.8) {
  const osc = ctx.createOscillator();
  const env = ctx.createGain();
  osc.frequency.setValueAtTime(130, t);
  osc.frequency.exponentialRampToValueAtTime(45, t + 0.12);
  env.gain.setValueAtTime(gain, t);
  env.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
  osc.connect(env).connect(dest);
  osc.start(t);
  osc.stop(t + 0.4);
}

/** Snare / hi-hat: a burst of white noise through a filter (band-pass for snare, high-pass for hats). */
function playNoise(ctx, dest, t, { dur = 0.1, gain = 0.2, type = 'highpass', freq = 7000 } = {}) {
  const src = ctx.createBufferSource();
  src.buffer = music.noise;
  const f = ctx.createBiquadFilter();
  f.type = type;
  f.frequency.value = freq;
  const env = ctx.createGain();
  env.gain.setValueAtTime(gain, t);
  env.gain.exponentialRampToValueAtTime(0.001, t + dur);
  src.connect(f).connect(env).connect(dest);
  src.start(t, Math.random() * 0.5);
  src.stop(t + dur + 0.02);
}

/**
 * The built-in songs. Each has a tempo, a chord progression (lists of MIDI
 * notes, one chord per bar), and a bar(ctx, out, i, t, beat) function that
 * schedules everything in bar number i, starting at time t (seconds), where
 * `beat` is the length of one beat. out.dry goes straight to the speakers,
 * out.echo goes through an echo effect first.
 */
const SONGS = {
  lofi: {
    bpm: 78, volume: 0.9,
    chords: [[65, 69, 72, 76], [64, 67, 71, 74], [62, 65, 69, 72], [60, 64, 67, 71]], // Fmaj7 Em7 Dm7 Cmaj7
    scale: [72, 74, 76, 79, 81, 84],                                                     // C major pentatonic
    bar(ctx, out, i, t, beat) {
      const chord = this.chords[i % 4];
      for (const n of chord) playNote(ctx, out.dry, { note: n, start: t, dur: beat * 3.6, type: 'triangle', gain: 0.045, attack: 0.08, release: 0.6, cutoff: 900 });
      playNote(ctx, out.dry, { note: chord[0] - 24, start: t, dur: beat * 1.5, gain: 0.22, release: 0.2 });
      playNote(ctx, out.dry, { note: chord[0] - 24, start: t + beat * 2.5, dur: beat, gain: 0.18, release: 0.2 });
      playKick(ctx, out.dry, t, 0.55); playKick(ctx, out.dry, t + beat * 2.5, 0.45);
      playNoise(ctx, out.dry, t + beat, { dur: 0.18, gain: 0.12, type: 'bandpass', freq: 1800 });
      playNoise(ctx, out.dry, t + beat * 3, { dur: 0.18, gain: 0.12, type: 'bandpass', freq: 1800 });
      for (let k = 0; k < 8; k++) playNoise(ctx, out.dry, t + beat * (k / 2 + (k % 2 ? 0.08 : 0)), { dur: 0.04, gain: 0.025 }); // swung hats
      const rnd = seededRandom(1000 + (i % 8));
      for (let k = 0; k < 4; k++) {
        if (rnd() < 0.45) continue;
        playNote(ctx, out.echo, { note: this.scale[Math.floor(rnd() * this.scale.length)], start: t + beat * k + (rnd() < 0.5 ? beat / 2 : 0), dur: beat * 0.6, gain: 0.07, release: 0.5 });
      }
    },
  },
  piano: {
    bpm: 68, volume: 1,
    chords: [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]], // C G Am F
    bar(ctx, out, i, t, beat) {
      const chord = this.chords[i % 4];
      const arp = [chord[0], chord[1], chord[2], chord[0] + 12, chord[1] + 12, chord[2] + 12, chord[0] + 12, chord[2]];
      arp.forEach((n, k) => {
        // Piano-ish: a sine plus a quieter octave-up triangle, fast attack, long ring.
        playNote(ctx, out.echo, { note: n, start: t + k * beat / 2, dur: beat * 0.3, gain: 0.08, attack: 0.005, release: 1.2 });
        playNote(ctx, out.echo, { note: n + 12, start: t + k * beat / 2, dur: beat * 0.2, type: 'triangle', gain: 0.02, attack: 0.005, release: 0.8 });
      });
      playNote(ctx, out.dry, { note: chord[0] - 12, start: t, dur: beat * 3.5, gain: 0.12, attack: 0.02, release: 1.5 });
      if (i % 2 === 1) {
        const rnd = seededRandom(2000 + (i % 8));
        playNote(ctx, out.echo, { note: chord[Math.floor(rnd() * 3)] + 24, start: t + beat * 2, dur: beat, gain: 0.05, release: 1.5 });
      }
    },
  },
  chip: {
    bpm: 132, volume: 0.55,
    chords: [[57, 60, 64], [53, 57, 60], [60, 64, 67], [55, 59, 62]], // Am F C G
    scale: [69, 72, 74, 76, 79, 81],                                  // A minor pentatonic
    bar(ctx, out, i, t, beat) {
      const chord = this.chords[i % 4];
      for (let k = 0; k < 8; k++) playNote(ctx, out.dry, { note: chord[0] - 12, start: t + k * beat / 2, dur: beat * 0.35, type: 'square', gain: 0.05, release: 0.02, cutoff: 1400 });
      for (let k = 0; k < 16; k++) playNote(ctx, out.dry, { note: chord[k % 3] + 12, start: t + k * beat / 4, dur: beat * 0.15, type: 'triangle', gain: 0.03, release: 0.02 });
      const rnd = seededRandom(3000 + (i % 8));
      for (let k = 0; k < 8; k++) {
        if (rnd() < 0.3) continue;
        playNote(ctx, out.echo, { note: this.scale[Math.floor(rnd() * this.scale.length)], start: t + k * beat / 2, dur: beat * 0.4, type: 'square', gain: 0.045, release: 0.05, cutoff: 3000 });
      }
      for (let k = 0; k < 4; k++) playKick(ctx, out.dry, t + k * beat, 0.5);
      playNoise(ctx, out.dry, t + beat, { dur: 0.1, gain: 0.15, type: 'bandpass', freq: 2500 });
      playNoise(ctx, out.dry, t + beat * 3, { dur: 0.1, gain: 0.15, type: 'bandpass', freq: 2500 });
    },
  },
  synth: {
    bpm: 96, volume: 0.7,
    chords: [[57, 60, 64], [53, 57, 60], [60, 64, 67], [55, 59, 62]], // Am F C G
    bar(ctx, out, i, t, beat) {
      const chord = this.chords[i % 4];
      for (const n of chord) {
        // Pad: two slightly detuned sawtooth waves per note, softened by a low-pass filter.
        playNote(ctx, out.dry, { note: n, start: t, dur: beat * 3.8, type: 'sawtooth', gain: 0.025, attack: 0.3, release: 0.5, cutoff: 1100, detune: -7 });
        playNote(ctx, out.dry, { note: n, start: t, dur: beat * 3.8, type: 'sawtooth', gain: 0.025, attack: 0.3, release: 0.5, cutoff: 1100, detune: 7 });
      }
      for (let k = 0; k < 8; k++) playNote(ctx, out.dry, { note: chord[0] - 24, start: t + k * beat / 2, dur: beat * 0.4, type: 'sawtooth', gain: 0.06, release: 0.05, cutoff: 500 });
      for (let k = 0; k < 16; k++) playNote(ctx, out.echo, { note: chord[k % 3] + 12 + (k % 8 >= 4 ? 12 : 0), start: t + k * beat / 4, dur: beat * 0.18, type: 'sawtooth', gain: 0.022, release: 0.05, cutoff: 2400 });
      for (let k = 0; k < 4; k++) playKick(ctx, out.dry, t + k * beat, 0.6);
      playNoise(ctx, out.echo, t + beat, { dur: 0.22, gain: 0.14, type: 'bandpass', freq: 1500 });
      playNoise(ctx, out.echo, t + beat * 3, { dur: 0.22, gain: 0.14, type: 'bandpass', freq: 1500 });
      for (let k = 0; k < 8; k++) playNoise(ctx, out.dry, t + k * beat / 2 + beat / 4, { dur: 0.03, gain: 0.03 });
    },
  },
};

/**
 * Start a built-in song. A "lookahead scheduler" runs every 200 ms and
 * schedules any bars that begin within the next 1.5 seconds on the audio
 * clock. Timing stays precise because the audio hardware plays scheduled
 * notes at exact times, even if the page's timers are a little late.
 * Returns { stop(fadeSeconds) }.
 */
function startBuiltInSong(id) {
  const ctx = ensureAudio();
  const song = SONGS[id];
  if (!ctx || !song) return null;
  const beat = 60 / song.bpm, barLen = beat * 4;

  const out = ctx.createGain();
  out.gain.setValueAtTime(0.0001, ctx.currentTime);
  out.gain.exponentialRampToValueAtTime(song.volume, ctx.currentTime + 0.6); // gentle fade in
  out.connect(music.master);
  // Echo effect: a delay line that feeds a little of itself back in.
  const echoIn = ctx.createGain();
  const delay = ctx.createDelay(2);
  delay.delayTime.value = beat * 0.75;
  const feedback = ctx.createGain(); feedback.gain.value = 0.32;
  const wet = ctx.createGain(); wet.gain.value = 0.35;
  echoIn.connect(out);
  echoIn.connect(delay);
  delay.connect(feedback).connect(delay);
  delay.connect(wet).connect(out);
  const dest = { dry: out, echo: echoIn };

  let bar = 0, nextBar = ctx.currentTime + 0.08;
  const schedule = () => {
    while (nextBar < ctx.currentTime + 1.5) {
      song.bar(ctx, dest, bar, nextBar, beat);
      bar++;
      nextBar += barLen;
    }
  };
  schedule();
  const timer = setInterval(schedule, 200);
  return {
    stop(fade = 0.6) {
      clearInterval(timer);
      const t = ctx.currentTime;
      out.gain.cancelScheduledValues(t);
      out.gain.setValueAtTime(Math.max(0.0001, out.gain.value), t);
      out.gain.exponentialRampToValueAtTime(0.0001, t + Math.max(0.05, fade));
      setTimeout(() => out.disconnect(), (fade + 2) * 1000);
    },
  };
}

/** Play the user's own song (already decoded), looping, with fade in/out. */
function startUserSong() {
  const ctx = ensureAudio();
  if (!ctx || !music.userBuffer) return null;
  const src = ctx.createBufferSource();
  src.buffer = music.userBuffer;
  src.loop = true;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.0001, ctx.currentTime);
  g.gain.exponentialRampToValueAtTime(1, ctx.currentTime + 0.4);
  src.connect(g).connect(music.master);
  src.start();
  return {
    stop(fade = 0.6) {
      const t = ctx.currentTime;
      g.gain.cancelScheduledValues(t);
      g.gain.setValueAtTime(Math.max(0.0001, g.gain.value), t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + Math.max(0.05, fade));
      src.stop(t + fade + 0.05);
    },
  };
}

/** Start whatever is chosen in the Music menu (or return null for "No music"). */
function startSelectedSong(choice) {
  if (choice === 'file') return startUserSong();
  if (SONGS[choice]) return startBuiltInSong(choice);
  return null;
}

/** Read and decode an audio file the user picked (MP3, M4A/AAC, WAV, OGG…, whatever the browser supports). */
async function loadUserSong(file) {
  const ctx = ensureAudio();
  if (!ctx) throw new Error('Web Audio is not supported');
  const bytes = await file.arrayBuffer();
  music.userBuffer = await new Promise((resolve, reject) => {
    // The callback form works in older Safari too; newer browsers also return a promise.
    const p = ctx.decodeAudioData(bytes, resolve, reject);
    if (p && p.catch) p.catch(reject);
  });
  music.userName = file.name;
}


/* =============================================================================
   7. UI WIRING
   ============================================================================= */

const $ = (id) => document.getElementById(id);
const ui = {
  mainView: $('main-view'),
  resultView: $('result-view'),
  fileInput: $('file-input'),
  dropZone: $('drop-zone'),
  dropEmpty: $('drop-empty'),
  preview: $('preview'),
  uploadMsg: $('upload-msg'),
  detail: $('detail'),
  detailValue: $('detail-value'),
  liveToggle: $('live-toggle'),
  speedField: $('speed-field'),
  speed: $('speed'),
  speedValue: $('speed-value'),
  generateBtn: $('generate-btn'),
  backBtn: $('back-btn'),
  resultTitle: $('result-title'),
  canvas: $('result-canvas'),
  loading: $('loading'),
  loadingText: $('loading-text'),
  emptyMsg: $('empty-msg'),
  emptyText: $('empty-text'),
  emptyBackBtn: $('empty-back-btn'),
  progress: $('progress'),
  progressBar: $('progress-bar'),
  status: $('result-status'),
  saveVideoBtn: $('save-video-btn'),
  saveImageBtn: $('save-image-btn'),
  shareBtn: $('share-btn'),
  videoNote: $('video-note'),
  musicField: $('music-field'),
  musicSelect: $('music'),
  musicFile: $('music-file'),
  musicPreview: $('music-preview'),
  musicMsg: $('music-msg'),
  toast: $('toast'),
};

const state = {
  image: null,        // the loaded HTMLImageElement
  previewUrl: null,
  runId: 0,           // bumps on every Generate/Back so stale async work knows to stop
  recorder: null,
  song: null,         // the music playing during the live drawing
  mode: 'lineart',
  imageBlob: null,
  videoBlob: null,
  videoExt: 'webm',
};

let toastTimer = 0;
function showToast(text) {
  ui.toast.textContent = text;
  ui.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove('show'), 3200);
}

function setUploadMsg(text, isError = false) {
  ui.uploadMsg.textContent = text;
  ui.uploadMsg.classList.toggle('error', isError);
}

/* ---------- Upload ---------- */

async function loadFile(file) {
  if (!file) return;
  const looksLikeImage = (file.type && file.type.startsWith('image/')) || /\.(jpe?g|png|gif|webp|bmp|avif)$/i.test(file.name);
  if (!looksLikeImage) {
    setUploadMsg("That file doesn't look like an image. Please choose a JPG, PNG, WebP or GIF.", true);
    return;
  }

  const url = URL.createObjectURL(file);
  const img = new Image();
  try {
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });
    if (!img.naturalWidth || !img.naturalHeight) throw new Error('empty image');
  } catch (err) {
    URL.revokeObjectURL(url);
    setUploadMsg("Couldn't read that image. Some formats (like HEIC) aren't supported by every browser. Try a JPG or PNG.", true);
    return;
  }

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = url;
  state.image = img;
  ui.preview.src = url;
  ui.preview.hidden = false;
  ui.dropEmpty.hidden = true;
  ui.generateBtn.disabled = false;

  const w = img.naturalWidth, h = img.naturalHeight;
  const big = Math.max(w, h) > 2000 || file.size > 8 * 1024 * 1024;
  setUploadMsg(`${w} × ${h}px${big ? ', a big photo, so it will be scaled down to keep things fast' : ''}. Tap the photo to change it.`);
}

ui.fileInput.addEventListener('change', () => {
  loadFile(ui.fileInput.files[0]);
  ui.fileInput.value = ''; // allow re-selecting the same file
});

// Drag & drop (desktop nicety)
['dragenter', 'dragover'].forEach((type) =>
  ui.dropZone.addEventListener(type, (e) => { e.preventDefault(); ui.dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((type) =>
  ui.dropZone.addEventListener(type, (e) => { e.preventDefault(); ui.dropZone.classList.remove('dragging'); }));
ui.dropZone.addEventListener('drop', (e) => loadFile(e.dataTransfer.files[0]));

/* ---------- Options ---------- */

ui.detail.addEventListener('input', () => { ui.detailValue.textContent = ui.detail.value; });
ui.speed.addEventListener('input', () => { ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(1)}x`; });
ui.liveToggle.addEventListener('change', () => {
  ui.speedField.hidden = !ui.liveToggle.checked;
  ui.musicField.hidden = !ui.liveToggle.checked;
  if (!ui.liveToggle.checked) stopPreview();
});

/* ---------- Music menu ---------- */

function stopPreview() {
  if (music.preview) { music.preview.stop(0.3); music.preview = null; }
  clearTimeout(stopPreview.timer);
  ui.musicPreview.textContent = '▶ Preview';
}

function updateMusicUI() {
  const choice = ui.musicSelect.value;
  const ready = choice !== 'none' && (choice !== 'file' || !!music.userBuffer);
  ui.musicPreview.disabled = !ready;
  if (choice === 'file') {
    ui.musicMsg.textContent = music.userBuffer
      ? `🎵 ${music.userName}. It plays from your device and is never uploaded.`
      : 'Pick an audio file (MP3, M4A, WAV…) from your device.';
  } else {
    ui.musicMsg.textContent = choice === 'none' ? '' : 'The music is recorded into "Save as video" too.';
  }
}

ui.musicSelect.addEventListener('change', () => {
  stopPreview();
  if (ui.musicSelect.value === 'file') ui.musicFile.click(); // opens the file picker (this is a user action, so it's allowed)
  updateMusicUI();
});

ui.musicFile.addEventListener('change', async () => {
  const file = ui.musicFile.files[0];
  ui.musicFile.value = '';
  if (!file) { updateMusicUI(); return; }
  ui.musicMsg.textContent = 'Loading your song…';
  try {
    await loadUserSong(file);
  } catch (err) {
    console.warn(err);
    music.userBuffer = null;
    ui.musicMsg.textContent = "Couldn't read that audio file. Try an MP3, M4A or WAV.";
    ui.musicPreview.disabled = true;
    return;
  }
  updateMusicUI();
});

// Clicking the menu again while "Your own song" is already selected lets you pick a different file.
ui.musicMsg.addEventListener('click', () => { if (ui.musicSelect.value === 'file') ui.musicFile.click(); });

ui.musicPreview.addEventListener('click', () => {
  if (music.preview) { stopPreview(); return; }
  ensureAudio();
  music.preview = startSelectedSong(ui.musicSelect.value);
  if (!music.preview) return;
  ui.musicPreview.textContent = '■ Stop';
  stopPreview.timer = setTimeout(stopPreview, 15000); // previews play for 15 seconds
});

updateMusicUI();

const selectedMode = () => document.querySelector('input[name="mode"]:checked').value;

/* ---------- View switching ---------- */

function showView(name) {
  ui.mainView.hidden = name !== 'main';
  ui.resultView.hidden = name !== 'result';
  window.scrollTo(0, 0);
}

function cancelRun() {
  state.runId++;
  if (state.recorder) { state.recorder.cancel(); state.recorder = null; }
  if (state.song) { state.song.stop(0.3); state.song = null; }
}

function goBack() {
  // If we pushed a history entry for the result view, go back through history
  // so the phone's own back button/gesture works the same as ours.
  if (history.state && history.state.view === 'result') history.back();
  else { cancelRun(); showView('main'); }
}

window.addEventListener('popstate', () => {
  if (!ui.resultView.hidden) { cancelRun(); showView('main'); }
});
if (history.state && history.state.view === 'result') history.replaceState(null, '');

ui.backBtn.addEventListener('click', goBack);
ui.emptyBackBtn.addEventListener('click', goBack);

/* ---------- Generate ---------- */

function setResultButtons({ image = false, share = false, video = false }) {
  ui.saveImageBtn.disabled = !image;
  ui.shareBtn.disabled = !share;
  ui.saveVideoBtn.disabled = !video;
}

function showEmpty(text) {
  ui.canvas.hidden = true;
  ui.loading.hidden = true;
  ui.progress.hidden = true;
  ui.emptyText.textContent = text;
  ui.emptyMsg.hidden = false;
}

async function generate() {
  if (!state.image) { setUploadMsg('Please choose a photo first.', true); return; }

  cancelRun();
  const runId = state.runId;
  const cancelled = () => runId !== state.runId;

  const mode = selectedMode();
  const live = ui.liveToggle.checked;
  const speed = Number(ui.speed.value);
  const detail = Number(ui.detail.value);

  // Music: wake up the audio system right now, inside the click. Browsers
  // only allow sound to start from a user action, and the drawing (and its
  // music) begins a few seconds later, after the image is processed.
  stopPreview();
  let musicChoice = live ? ui.musicSelect.value : 'none';
  if (musicChoice === 'file' && !music.userBuffer) musicChoice = 'none';
  if (musicChoice !== 'none' && !ensureAudio()) musicChoice = 'none';

  // Reset the result view
  state.mode = mode;
  state.imageBlob = null;
  state.videoBlob = null;
  ui.resultTitle.textContent = { lineart: 'Line art', cartoon: 'Cartoon', illustration: 'Illustration' }[mode] || 'Drawing';
  setResultButtons({});
  ui.saveVideoBtn.hidden = !live;
  ui.videoNote.hidden = true;
  ui.emptyMsg.hidden = true;
  ui.canvas.hidden = true;
  ui.progress.hidden = true;
  ui.progressBar.style.width = '0%';
  ui.status.textContent = '';
  ui.loadingText.textContent = mode === 'lineart' ? 'Sketching…' : 'Drawing…';
  ui.loading.hidden = false;

  showView('result');
  history.pushState({ view: 'result' }, '');
  await nextFrame(); // two frames: make sure the spinner is actually on screen
  await nextFrame();  // before any heavy main-thread work starts

  let art;
  try {
    const onStatus = (text) => { if (!cancelled()) ui.loadingText.textContent = text; };
    art = mode === 'lineart'
      ? await buildLineArt(state.image, detail, onStatus)
      : mode === 'illustration'
        ? await buildIllustration(state.image, detail, onStatus)
        : await buildSketchCartoon(state.image, detail, onStatus);
  } catch (err) {
    console.error(err);
    if (!cancelled()) showEmpty('Something went wrong while processing this photo. Please try a different one.');
    return;
  }
  if (cancelled()) return;
  ui.loading.hidden = true;

  if (art.isEmpty) {
    showEmpty("We couldn't find enough edges to draw. The photo may be very smooth or low-contrast. Try raising \"Edge detail\" or use a photo with clearer shapes.");
    return;
  }

  ui.canvas.width = Math.round(art.width * art.scale);
  ui.canvas.height = Math.round(art.height * art.scale);
  ui.canvas.hidden = false;
  const ctx = ui.canvas.getContext('2d');

  if (!live) {
    art.drawFinal(ctx);
  } else {
    const withMusic = musicChoice !== 'none' && !!music.recordDest;
    const format = pickVideoFormat(withMusic);
    let recorder = null;
    if (format) {
      try {
        recorder = startRecording(ui.canvas, format, withMusic ? music.recordDest.stream : null);
      } catch (err) { console.warn('Recording failed to start', err); }
    }
    state.recorder = recorder;

    const drawAt = art.createLiveDrawer(ctx);
    drawAt(0);
    ui.progress.hidden = false;
    ui.status.textContent = 'Drawing…';
    state.song = musicChoice !== 'none' ? startSelectedSong(musicChoice) : null;

    // ~8 s at 1x; 2x speed → 4 s, 0.5x → 16 s. The pen's pace is derived from the total work inside drawAt.
    const duration = SETTINGS.baseDurationMs / speed;
    const finished = await animateDrawing(drawAt, duration, (p) => { ui.progressBar.style.width = `${p * 100}%`; }, cancelled);
    if (!finished) return;

    // The music fades out while the finished picture is held on screen.
    if (state.song) { state.song.stop(recorder ? SETTINGS.holdFinalFrameMs / 1000 : 1.2); state.song = null; }

    if (recorder) {
      ui.status.textContent = 'Finishing video…';
      await holdFinalFrame(ui.canvas, SETTINGS.holdFinalFrameMs, cancelled);
      if (cancelled()) return;
      const blob = await recorder.stop();
      state.recorder = null;
      if (cancelled()) return;
      if (blob.size > 0) { state.videoBlob = blob; state.videoExt = format.ext; }
    }
    ui.progress.hidden = true;

    if (!state.videoBlob) {
      ui.videoNote.textContent = format
        ? "The video couldn't be recorded this time, but you can still save the image."
        : "Saving video isn't supported in this browser (it needs MediaRecorder and canvas capture). You can still save the image.";
      ui.videoNote.hidden = false;
    }
  }

  // Prepare the PNG now, so Share can run instantly on tap (mobile browsers
  // only allow sharing directly inside a tap, not after a slow await).
  state.imageBlob = await canvasToBlob(ui.canvas);
  if (cancelled()) return;
  ui.status.textContent = art.note || 'Done!';
  setResultButtons({ image: !!state.imageBlob, share: !!state.imageBlob, video: !!state.videoBlob });
}

ui.generateBtn.addEventListener('click', generate);

/* ---------- Save & share ---------- */

ui.saveImageBtn.addEventListener('click', () => {
  if (state.imageBlob) downloadBlob(state.imageBlob, `drawing-${state.mode}.png`);
});

ui.saveVideoBtn.addEventListener('click', () => {
  if (state.videoBlob) downloadBlob(state.videoBlob, `drawing-${state.mode}.${state.videoExt}`);
});

ui.shareBtn.addEventListener('click', async () => {
  if (!state.imageBlob) return;
  const base = `drawing-${state.mode}`;
  const files = [];
  if (state.videoBlob) files.push(new File([state.videoBlob], `${base}.${state.videoExt}`, { type: state.videoBlob.type }));
  files.push(new File([state.imageBlob], `${base}.png`, { type: 'image/png' }));

  // 1) Native share sheet with the file itself (mostly phones). Try the video first, then the image.
  if (navigator.share && navigator.canShare) {
    for (const file of files) {
      let can = false;
      try { can = navigator.canShare({ files: [file] }); } catch (e) { can = false; }
      if (!can) continue;
      try {
        await navigator.share({ files: [file], title: 'My drawing' });
        return;
      } catch (err) {
        if (err.name === 'AbortError') return; // user closed the share sheet
        // otherwise try the next file type, then fall back
      }
    }
  }

  // 2) Fallback: download the file and copy the site link.
  const file = files[0];
  downloadBlob(file, file.name);
  const copied = await copySiteLink();
  showToast(copied ? 'Saved! Site link copied. Paste it anywhere to share.' : 'Saved to your downloads.');
});
