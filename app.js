'use strict';

/* =============================================================================
   Photo → Drawing  —  app.js

   Everything here runs in the browser. The photo is read from the file picker,
   processed with plain JavaScript on <canvas> pixel arrays, and never sent
   anywhere.

   TABLE OF CONTENTS
     0. Settings & small helpers
     1. SHARED PIPELINE      (used by both styles: resize → gray → contrast
                               stretch → blur → Sobel → non-max suppression
                               → hysteresis threshold → contours → smooth
                               → simplify → curves → stroke ordering)
     2. MODE 1: LINE ART     (vivid colors sampled from the photo, glow on black)
     3. PAINTED FALLBACK     (Kuwahara → k-means palette → color grading
                               → paper grain → soft outlines; used if the
                               Cartoon tools can't load) + shared painting helpers
     4. WEB WORKER HELPER    (runs the heavy filters off the main thread)
     4b. MODE 2: CARTOON     (MediaPipe hair/skin/clothes segmentation + face
                               landmarks → flat colors, drawn face, bold outlines)
     5. LIVE DRAWING         (requestAnimationFrame animation "pen")
     6. EXPORT & SHARING     (PNG, video recording, Web Share API)
     7. UI WIRING            (DOM events, view switching)

   Sections 1–3 only work on numbers/arrays and canvases. They don't touch the
   page's buttons or layout, so they can later be moved to their own files
   (for example shared-pipeline.js, line-art.js, storybook.js).
   ============================================================================= */


/* =============================================================================
   0. SETTINGS & SMALL HELPERS
   ============================================================================= */

const SETTINGS = {
  maxSideLineArt: 1400,     // processing size (longest side) for Line art: full output size, for fine detail
  maxSideStorybook: 600,    // Storybook is heavier, so it works on a smaller image
  cartoonSide: 1400,        // Cartoon mode works at full output size, so edges stay razor sharp (no upscaling blur)
  maxUpscale: 2.5,          // small photos are enlarged (up to 2.5x) before processing, so lines come out finer
  outputLongSide: 1400,     // the result canvas is always about this big, whatever the processing size
  rdpEpsilon: 0.6,          // Ramer–Douglas–Peucker tolerance, in pixels (lower = keeps more detail)
  maxStrokes: 9000,         // safety cap so extremely busy photos stay fast
  baseDurationMs: 8000,     // live drawing takes about 8 seconds at 1x speed
  holdFinalFrameMs: 1000,   // video keeps the finished picture on screen for 1 second
  kmeansK: 16,              // number of paint colors in Storybook mode
  kmeansMaxIter: 20,        // k-means gives up after this many rounds
  kuwaharaRadii: [3, 2],    // two Kuwahara passes: a medium one, then a gentle one (bigger = blockier)
  outlineOpacity: 0.6,      // Storybook outlines are drawn semi-transparent
  paperColor: [245, 238, 222],
};

/** Keep a number inside [lo, hi]. */
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

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
   1. SHARED PIPELINE  (used by both Line art and Storybook anime)
   -----------------------------------------------------------------------------
   Goal: turn a photo into a list of pen strokes. Each stroke is an ordered list
   of points that a "pen" can follow.

   photo → downscale → grayscale → blur → Sobel → non-max suppression
         → threshold → contour tracing → drop short / simplify → order strokes
         → smooth quadratic curve segments
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
 * `strictness` > 1 raises both cutoffs (Storybook wants cleaner outlines).
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
 * A copy of the image with texture smoothed away but edges kept (the
 * Kuwahara filter from section 3), used as the source for tracing lines.
 * Tracing this instead of the raw photo means fabric weave, skin pores,
 * carpet and noise don't become lines, while the outlines of real shapes do.
 */
function textureFree(rgba, w, h, radius) {
  return radius > 0 ? kuwaharaFilter(rgba, w, h, radius) : rgba;
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
  const [hue, s, l] = rgbToHsl(r, g, b);
  const s2 = Math.min(1, s * 2.2);
  const l2 = clamp(0.56 + l * 0.3, 0.58, 0.85);
  const [R, G, B] = hslToRgb(hue, s2, l2);
  return `rgb(${R | 0}, ${G | 0}, ${B | 0})`;
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
  ctx.shadowBlur = 3 + 6 * p;
  ctx.lineWidth = (1.0 + 1.6 * p) / scale; // 1.0–2.6 px on the 1400px output: fine, crisp lines
  ctx.globalAlpha = 0.7 + 0.3 * p;
}

/** Draw the complete line-art picture at once. */
function drawLineArtFinal(ctx, art) {
  resetCtx(ctx);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  setStrokeTransform(ctx, art.scale);
  for (const s of art.strokes) {
    styleLineArtStroke(ctx, s, art.scale);
    ctx.beginPath();
    drawSmoothPath(ctx, s.segs);
    ctx.stroke();
  }
  resetCtx(ctx);
}

/**
 * Live version: returns drawAt(progress) where progress goes 0 → 1.
 * The pen moves at a constant speed along the total length of all strokes.
 */
function createLineArtDrawer(ctx, art) {
  resetCtx(ctx);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const pen = createStrokePen(art.strokes, (c, s) => styleLineArtStroke(c, s, art.scale));
  return (progress) => {
    setStrokeTransform(ctx, art.scale);
    pen.advanceTo(ctx, progress * pen.totalLength);
    resetCtx(ctx);
  };
}

/** Build the Line art result from an <img>. */
async function buildLineArt(image, detail, onStatus = () => {}) {
  const { width: w, height: h, data } = downscaleImage(image, SETTINGS.maxSideLineArt);
  // The slider is shifted a little toward "more detail": 50 on the slider traces like 65.
  const effectiveDetail = Math.min(100, 35 + detail * 0.6);
  const gain = Math.max(1, w / (image.naturalWidth || image.width)); // how much a small photo was enlarged
  const sizeFactor = Math.max(w, h) / 800;
  // Trace a texture-free copy (edges kept, fabric/carpet/skin texture
  // smoothed away), then drop leftover texture scribbles. This is what keeps
  // busy photos clean. lowRatio 0.35: hysteresis follows fainter edges along
  // real contours, so lines continue further.
  const smooth = textureFree(data, w, h, Math.max(2, Math.round(Math.max(w, h) / 450)));
  let raw = extractRawStrokes(smooth, w, h, { detail: effectiveDetail, strictness: 1, minPoints: Math.round(8 * sizeFactor), gain, lowRatio: 0.35 });
  raw = suppressTexture(raw, w, h);

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
    onStatus('Sketching facial features…');
    raw = addFaceDetailLines(raw, w, h, image, faces, effectiveDetail);
    raw = addLandmarkFeatureLines(raw, landmarkFaces);
  }

  const strokes = finalizeStrokes(raw);
  // Line weight is based on stroke length; measure it relative to an 800px
  // image (sizeFactor) so weights look the same at any processing size.
  for (const s of strokes) {
    s.color = sampleStrokeColor(s, data, w, h);
    s.prominence = s.feature ? 0.7 : Math.sqrt(Math.min(1, s.rawLength / (240 * sizeFactor)));
  }
  const totalLength = strokes.reduce((sum, s) => sum + s.length, 0);

  return {
    mode: 'lineart',
    width: w,
    height: h,
    scale: outputScale(w, h),
    strokes,
    isEmpty: strokes.length === 0 || totalLength < 40,
    faceCount: faces.length,
    note: strokes.length < 15 ? 'Only a few lines were found. Try raising "Edge detail".' : '',
    drawFinal(ctx) { drawLineArtFinal(ctx, this); },
    createLiveDrawer(ctx) { return createLineArtDrawer(ctx, this); },
  };
}


/* =============================================================================
   3. PAINTED FALLBACK + SHARED PAINTING HELPERS
      (the classic filter "storybook" look, used when the Cartoon tools in
      section 4b can't load; its reveal animation and sky/glow helpers are
      shared with the Cartoon)
   -----------------------------------------------------------------------------
   Kuwahara (painterly smoothing) → k-means (limited paint palette) → color
   grading (warm, soft, lifted) → soften + paper grain → thin brown outlines.

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
 * The heavy part of Storybook mode, grouped so it can run in a Web Worker:
 * Kuwahara passes, then k-means on the smoothed result.
 */
function paintStage(rgba, w, h, opts) {
  let smoothed = rgba;
  for (let i = 0; i < opts.radii.length; i++) smoothed = kuwaharaFilter(smoothed, w, h, opts.radii[i]);
  const km = kmeansQuantize(smoothed, w, h, opts.k, opts.maxIter);
  return { smoothed, labels: km.labels, centers: km.centers };
}

/**
 * Auto levels + brightening (changes `rgba` in place).
 * Storybook films are bright and airy; a night or indoor photo would come out
 * muddy. Like "auto levels" in a photo editor:
 *   1. find the darkest and brightest 0.5% brightness levels and stretch that
 *      range to 0–255 (gain capped at 2.5x so noise isn't blown up);
 *   2. if the picture is still dark on average, apply a gamma curve (power
 *      < 1) that lifts the shadows and midtones until the average brightness
 *      is about 45%.
 * All three channels get the same curve, so colors keep their hue.
 * makeLevelsLut() computes the curve as a 256-entry look-up table so the SAME
 * curve can also be applied to a face close-up (keeping colors consistent).
 */
function autoLevels(rgba) {
  applyLut(rgba, makeLevelsLut(rgba));
}

/** Apply a 256-entry look-up table to the R, G and B of every pixel (in place). */
function applyLut(rgba, lut) {
  for (let j = 0; j < rgba.length; j += 4) {
    rgba[j] = lut[rgba[j]]; rgba[j + 1] = lut[rgba[j + 1]]; rgba[j + 2] = lut[rgba[j + 2]];
  }
}

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
 * Soft color grading, applied to each palette color and to the shading layer.
 *  - Lift shadows: nothing is pitch black, like light bouncing in a painting.
 *  - Gently boost saturation.
 *  - Sky: blue, fairly bright colors get lighter and slightly softer.
 *  - Warm midtones: add a touch of red/yellow where lightness is near 50%.
 */
function gradeColor(r, g, b) {
  let [hue, s, l] = rgbToHsl(r, g, b);
  l = 0.08 + 0.92 * l;
  s = Math.min(1, s * 1.2 + 0.04);
  if (hue > 185 && hue < 250 && s > 0.12 && l > 0.35) {
    l += (1 - l) * 0.3;
    s *= 0.85;
    hue -= 6; // nudge toward a clean cyan-blue
  }
  let [R, G, B] = hslToRgb(hue, s, l);
  const mid = 1 - Math.abs(l - 0.5) * 2; // 1 at mid-gray, 0 at black/white
  R += 10 * mid; G += 4 * mid; B -= 6 * mid;
  return [clamp(R, 0, 255), clamp(G, 0, 255), clamp(B, 0, 255)];
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

/** Light blur on each color channel, to soften the hard pixel steps between palette regions (like wet paint edges). */
function softenColors(rgba, w, h) {
  const n = w * h;
  const out = new Uint8ClampedArray(rgba.length);
  const channel = new Float32Array(n);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) channel[i] = rgba[i * 4 + c];
    const blurred = gaussianBlur(channel, w, h);
    for (let i = 0; i < n; i++) out[i * 4 + c] = blurred[i];
  }
  for (let i = 0; i < n; i++) out[i * 4 + 3] = 255;
  return out;
}

/**
 * Final painted pixels: palette color per pixel, blended with a little of the
 * smoothed photo's own (graded) color → soften → paper grain.
 * Pure palette colors look like flat cut-paper, and a face can collapse into a
 * single blob. Mixing back ~35% of the smooth shading keeps the form (cheeks,
 * nose, folds in clothes) while the palette still gives the flat,
 * illustrated look. Painters work the same way: flat base washes plus soft
 * shading.
 */
const SHADING_MIX = 0.35;
function composePainting(labels, palette, smoothed, grain, w, h) {
  const flat = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < labels.length; i++, j += 4) {
    const c = palette[labels[i]];
    const s = gradeColor(smoothed[j], smoothed[j + 1], smoothed[j + 2]);
    flat[j] = c[0] + (s[0] - c[0]) * SHADING_MIX;
    flat[j + 1] = c[1] + (s[1] - c[1]) * SHADING_MIX;
    flat[j + 2] = c[2] + (s[2] - c[2]) * SHADING_MIX;
    flat[j + 3] = 255;
  }
  const soft = softenColors(flat, w, h);
  for (let i = 0, j = 0; i < grain.length; i++, j += 4) {
    soft[j] *= grain[i]; soft[j + 1] *= grain[i]; soft[j + 2] *= grain[i];
  }
  return soft;
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
function compositeStorybook(ctx, paintSource, outlineLayer, opacity) {
  resetCtx(ctx);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(paintSource, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.globalAlpha = opacity;
  ctx.drawImage(outlineLayer, 0, 0);
  ctx.globalAlpha = 1;
}

function drawStorybookFinal(ctx, art) {
  const layer = makeCanvas(ctx.canvas.width, ctx.canvas.height);
  const lctx = layer.getContext('2d');
  setStrokeTransform(lctx, art.scale);
  for (const s of art.strokes) {
    styleOutline(lctx, s);
    lctx.beginPath();
    drawSmoothPath(lctx, s.segs);
    lctx.stroke();
  }
  compositeStorybook(ctx, art.paintCanvas, layer, art.outlineOpacity);
}

/**
 * Live version: the first ~55% of the time spreads the paint washes, the rest
 * draws outlines stroke by stroke on top.
 */
function createStorybookDrawer(ctx, art) {
  const work = makeCanvas(art.width, art.height);
  const wctx = work.getContext('2d');
  const frame = wctx.createImageData(art.width, art.height);
  const layer = makeCanvas(ctx.canvas.width, ctx.canvas.height);
  const lctx = layer.getContext('2d');
  setStrokeTransform(lctx, art.scale);
  const pen = createStrokePen(art.strokes, styleOutline);
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
    compositeStorybook(ctx, paintDone ? art.paintCanvas : work, layer, art.outlineOpacity);
  };
}

/**
 * Mode 2 entry point: the Cartoon (section 4b), which draws the person as a
 * flat, bold-outlined cartoon using MediaPipe body-part segmentation and face
 * landmarks. If those can't load (offline, very old browser, download
 * blocked), fall back to the classic storybook filters so the button still works.
 */
async function buildCartoonOrFallback(image, detail, onStatus = () => {}) {
  try {
    return await buildCartoon(image, detail, onStatus);
  } catch (err) {
    console.warn('Cartoon tools unavailable, using the classic filters instead:', err);
    onStatus('Painting…');
    const art = await buildStorybookClassic(image, detail);
    art.note = "The cartoon tools couldn't load (offline or unsupported browser), so this is a simpler painted version.";
    return art;
  }
}

/** Classic Storybook (no AI): Kuwahara → k-means palette → grading → grain. */
async function buildStorybookClassic(image, detail) {
  const { width: w, height: h, data } = downscaleImage(image, SETTINGS.maxSideStorybook);
  autoLevels(data); // storybook scenes are bright and airy, even when the photo was taken at night

  const { smoothed, labels, centers } = await runPaintStage(data, w, h, {
    radii: SETTINGS.kuwaharaRadii,
    k: SETTINGS.kmeansK,
    maxIter: SETTINGS.kmeansMaxIter,
  });
  await nextFrame(); // let the spinner breathe between heavy steps

  const palette = [];
  for (let c = 0; c < centers.length / 3; c++) {
    palette.push(gradeColor(centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2]));
  }
  const grain = makePaperGrain(w, h);
  const painted = composePainting(labels, palette, smoothed, grain, w, h);

  // Outlines come from the smoothed image: cleaner lines that follow the painted shapes.
  return makeStorybookArt({
    w, h, painted, grain, labels, centers,
    edgeSource: smoothed, detail, strictness: 1.4, minPoints: 12, outlineOpacity: SETTINGS.outlineOpacity,
  });
}

/** Shared finishing steps for both Storybook versions: paper, reveal plan, outlines, result object. */
function makeStorybookArt({ w, h, painted, grain, labels, centers, edgeSource, detail, strictness, minPoints, outlineOpacity, faces = 0, customStrokes = null }) {
  const palette = [];
  for (let c = 0; c < centers.length / 3; c++) palette.push([centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2]]);
  const paper = composePaper(grain, w, h);
  const revealAt = computeRevealOrder(labels, palette, w, h);
  const scale = outputScale(w, h);

  // Either strokes built by the caller (Cartoon), or soft outlines traced from `edgeSource`.
  let strokes = customStrokes;
  if (!strokes) {
    strokes = extractStrokes(edgeSource, w, h, { detail, strictness, minPoints });
    for (const s of strokes) s.outlineWidth = (1.2 + 1.8 * strokeProminence(s)) / scale; // 1.2–3 px on the 1400px output
  }

  const paintCanvas = makeCanvas(w, h);
  paintCanvas.getContext('2d').putImageData(new ImageData(painted, w, h), 0, 0);

  return {
    mode: 'storybook',
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
    drawFinal(ctx) { drawStorybookFinal(ctx, this); },
    createLiveDrawer(ctx) { return createStorybookDrawer(ctx, this); },
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
   4b. CARTOON MODE  (MediaPipe body-part segmentation + face landmarks)
   -----------------------------------------------------------------------------
   Instead of filtering the photo, this mode DRAWS a cartoon of it, the way an
   illustrator works from a reference photo:

     1. MediaPipe's "multiclass selfie" segmenter (a small neural network)
        labels every pixel: background, hair, body skin, face skin, clothes
        or accessories.
     2. MediaPipe's face landmarker finds 478 points on each face: outlines of
        the eyes, irises, eyebrows, lips, nose and jaw.
     3. Every region gets ONE flat color taken from the photo, plus one
        shadow tone ("cel shading"). Clothes get a few flat colors (so folds,
        pockets and logos survive), and the background is flattened into big
        simple shapes.
     4. Cartoon eyes (whites, iris, pupil, sparkle), filled eyebrows, a small
        nose, lips and blush are drawn as vector shapes exactly where the
        landmarks are.
     5. Bold outlines are traced along the borders between regions with the
        same contour pipeline Line art uses.

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

/**
 * Face boxes in source-image pixels, biggest first (used by Line art's face
 * pass). Small photos are enlarged first: the detector finds small faces
 * more reliably when they're a bit bigger.
 */
async function detectFaces(source) {
  const srcW = source.naturalWidth || source.width, srcH = source.naturalHeight || source.height;
  const scale = clamp(1024 / Math.max(srcW, srcH), 0.25, 3);
  const c = makeCanvas(Math.round(srcW * scale), Math.round(srcH * scale));
  const cctx = c.getContext('2d');
  cctx.imageSmoothingQuality = 'high';
  cctx.drawImage(source, 0, 0, c.width, c.height);
  const faces = await findFaceLandmarks(c);
  const minSide = Math.max(12, 0.02 * Math.max(srcW, srcH));
  return faces
    .map((pts) => {
      const xs = pts.map((p) => p.x / scale), ys = pts.map((p) => p.y / scale);
      return { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) };
    })
    .filter((f) => f.x2 - f.x1 >= minSide && f.y2 - f.y1 >= minSide)
    .sort((a, b) => (b.x2 - b.x1) * (b.y2 - b.y1) - (a.x2 - a.x1) * (a.y2 - a.y1))
    .slice(0, 4);
}

/**
 * A square crop around a face box, grown by `grow` so it includes hair, ears
 * and chin, and nudged up by `lift` × face height. Kept inside the image.
 */
function faceRegion(face, imgW, imgH, grow, lift) {
  const fw = face.x2 - face.x1, fh = face.y2 - face.y1;
  const size = Math.min(grow * Math.max(fw, fh), imgW, imgH);
  const cx = (face.x1 + face.x2) / 2, cy = (face.y1 + face.y2) / 2 - lift * fh;
  return {
    x: clamp(cx - size / 2, 0, imgW - size),
    y: clamp(cy - size / 2, 0, imgH - size),
    size,
  };
}

/** Resize RGBA pixels with the browser's high-quality scaler. */
function resizeRGBA(rgba, w, h, nw, nh) {
  const src = makeCanvas(w, h);
  src.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  const dst = makeCanvas(nw, nh);
  const dctx = dst.getContext('2d', { willReadFrequently: true });
  dctx.imageSmoothingQuality = 'high';
  dctx.drawImage(src, 0, 0, nw, nh);
  return dctx.getImageData(0, 0, nw, nh).data;
}

/** Fraction of a stroke's points that lie inside a circle. */
function fractionInside(points, cx, cy, radius) {
  let inside = 0;
  for (const p of points) if ((p.x - cx) ** 2 + (p.y - cy) ** 2 <= radius * radius) inside++;
  return inside / points.length;
}

/**
 * Line-art face pass: re-trace each face from a zoomed-in crop.
 * In the full photo, eyes and lips are only a few pixels tall and their edges
 * are weak next to strong edges like a jacket against the sky, so the global
 * threshold drops them. Tracing a close-up of just the face gives two wins:
 * (1) features become big enough to trace smoothly, and (2) the edge
 * threshold adapts to the face's own contrast, so soft features survive.
 * Base strokes inside the face are then replaced by the detailed ones.
 */
function addFaceDetailLines(raw, w, h, source, faces, detail) {
  const srcW = source.naturalWidth || source.width, srcH = source.naturalHeight || source.height;
  const k = w / srcW;
  let result = raw;

  for (const face of faces) {
    const r = faceRegion(face, srcW, srcH, 1.9, 0.12); // wide enough to include the hair
    const size = r.size * k;
    // Close-up size: enlarge the face ~3x, but not more. Enlarging a tiny JPEG
    // face too far turns skin texture and compression blocks into fake lines.
    const C = Math.round(clamp(r.size * 3, 240, 600));
    const cx = (r.x + r.size / 2) * k, cy = (r.y + r.size / 2) * k, radius = size / 2;

    const crop = makeCanvas(C, C);
    const cctx = crop.getContext('2d', { willReadFrequently: true });
    cctx.imageSmoothingQuality = 'high';
    cctx.drawImage(source, r.x, r.y, r.size, r.size, 0, 0, C, C);
    // Smooth away skin texture first (Kuwahara), and trace a little stricter
    // than the main pass with longer minimum strokes: we want clean eyes,
    // brows, nose, lips and hair shapes, not every pore.
    const data = textureFree(cctx.getImageData(0, 0, C, C).data, C, C, 2);
    const faceStrokes = suppressTexture(
      extractRawStrokes(data, C, C, { detail, strictness: 1.3, minPoints: 14 }), C, C, { maxDensity: 0.12 });
    const f = size / C; // crop pixels → processing pixels
    const mapped = [];
    for (const s of faceStrokes) {
      const points = s.points.map((p) => ({ x: r.x * k + p.x * f, y: r.y * k + p.y * f }));
      if (fractionInside(points, cx, cy, radius * 0.92) < 0.7) continue; // stay inside the face circle
      // Features are short, but they matter: give them a bit more line weight than their length alone would.
      mapped.push({ points, rawLength: s.rawLength * f * 1.6 });
    }

    result = result.filter((s) => fractionInside(s.points, cx, cy, radius * 0.8) < 0.5).concat(mapped);
  }
  return result;
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
const css = (c, a = 1) => `rgba(${c[0] | 0}, ${c[1] | 0}, ${c[2] | 0}, ${a})`;
const mixColor = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

function centroid(points) {
  let x = 0, y = 0;
  for (const p of points) { x += p.x; y += p.y; }
  return { x: x / points.length, y: y / points.length };
}

/** Stretch points away from (cx, cy) by sx horizontally and sy vertically. */
function scaleAbout(points, cx, cy, sx, sy) {
  return points.map((p) => ({ x: cx + (p.x - cx) * sx, y: cy + (p.y - cy) * sy }));
}

/** Shoelace formula: the area inside a polygon. */
function polygonArea(points) {
  let a = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i], q = points[(i + 1) % points.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a) / 2;
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

/** Pixels inside a polygon, as a 0/1 mask (the browser's canvas does the filling). */
function polygonMask(points, w, h) {
  const c = makeCanvas(w, h);
  const cctx = c.getContext('2d', { willReadFrequently: true });
  cctx.beginPath();
  smoothClosedPath(cctx, points);
  cctx.fill();
  const d = cctx.getImageData(0, 0, w, h).data;
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < mask.length; i++) mask[i] = d[i * 4 + 3] > 127 ? 1 : 0;
  return mask;
}

/** Blur a float mask several times (each pass is the 5-tap Gaussian from step 3). */
function blurTimes(mask, w, h, passes) {
  let m = mask;
  for (let p = 0; p < passes; p++) m = gaussianBlur(m, w, h);
  return m;
}

/** Median color of the pixels where labels === label (null if there are none). Medians ignore stray outliers. */
function medianColor(rgba, labels, label) {
  const hist = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  let n = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== label) continue;
    hist[0][rgba[i * 4]]++; hist[1][rgba[i * 4 + 1]]++; hist[2][rgba[i * 4 + 2]]++;
    n++;
  }
  if (n < 20) return null;
  return hist.map((hst) => {
    let acc = 0;
    for (let v = 0; v < 256; v++) { acc += hst[v]; if (acc >= n / 2) return v; }
    return 255;
  });
}

/**
 * A robust "true color" of a region: the average of its pixels in a middle
 * brightness band (between the `lo` and `hi` brightness percentiles),
 * ignoring near-white glare and colorless pixels (below `minSat` saturation).
 * A plain median fails with camera flash or strong light: the lit parts of a
 * face can be almost white, which made darker skin tones come out far too
 * pale. The mid-tones hold the real color. For hair, a lower band ignores
 * the shine so black hair stays black.
 */
function toneSample(rgba, labels, classes, lo = 0.25, hi = 0.5, minSat = 0) {
  const px = [];
  for (let i = 0; i < labels.length; i += 2) {
    if (!classes.includes(labels[i])) continue;
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    if (mx > 248) continue;                                  // blown-out glare
    if (minSat && (mx - mn) / Math.max(1, mx) < minSat) continue; // colorless (shine, gray)
    px.push([0.299 * r + 0.587 * g + 0.114 * b, r, g, b]);
  }
  if (px.length < 30) return null;
  px.sort((a, b) => a[0] - b[0]);
  const a = Math.floor(px.length * lo), z = Math.max(a + 1, Math.floor(px.length * hi));
  let r = 0, g = 0, b = 0;
  for (let k = a; k < z; k++) { r += px[k][1]; g += px[k][2]; b += px[k][3]; }
  const n = z - a;
  return [r / n, g / n, b / n];
}

/**
 * Majority ("mode") filter: each pixel takes the most common label in its
 * (2r+1)×(2r+1) neighborhood. It erases specks and smooths jagged borders
 * between flat color regions, which is key to the clean "vector" look.
 * Pixels labelled `ignore` are skipped and never counted.
 */
function majorityFilter(labels, w, h, k, r, ignore = 255) {
  const out = new Uint8Array(labels);
  const counts = new Uint16Array(k);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (labels[i] === ignore) continue;
      counts.fill(0);
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const l = labels[yy * w + xx];
          if (l !== ignore) counts[l]++;
        }
      }
      let best = labels[i], bestN = 0;
      for (let c = 0; c < k; c++) if (counts[c] > bestN) { bestN = counts[c]; best = c; }
      out[i] = best;
    }
  }
  return out;
}

/* ---------- Cartoon color styling ---------- */

const hsl = (h, s, l) => hslToRgb(h, clamp(s, 0, 1), clamp(l, 0, 1));

/** Skin: keep the person's real skin tone, but clean and a little brighter, plus a shadow, blush and line tone. */
function styleSkin(c) {
  const [h, s0, l0] = rgbToHsl(c[0], c[1], c[2]);
  // Only a small cleanup: a touch more color and brightness. Big boosts wash out darker skin tones.
  const s = clamp(s0 * 1.08, 0.18, 0.62), l = clamp(l0 + 0.03, 0.18, 0.86);
  return {
    base: hsl(h, s, l),
    shadow: hsl(h - 3, s + 0.02, l - 0.08),
    line: hsl(h - 8, s + 0.12, l * 0.5),
    blush: mixColor(hsl(h, s, l), [236, 110, 115], 0.55),
    lips: mixColor(hsl(h - 5, s + 0.07, l - 0.12), [196, 88, 92], 0.5),
  };
}

/** Hair: flat base, a shadow and a glossy highlight band, plus eyebrow color. */
function styleHair(c) {
  const [h, s0, l0] = rgbToHsl(c[0], c[1], c[2]);
  // Lift very dark hair a little so it isn't a black hole, and keep dark hair
  // neutral (a photo's warm light cast would otherwise turn black hair brown).
  const l = clamp(l0 * 1.15, 0.11, 0.55);
  const s = l0 < 0.2 ? Math.min(s0, 0.18) : clamp(s0 * 1.15, 0.08, 0.6);
  return {
    base: hsl(h, s, l),
    shadow: hsl(h, s, l * 0.68),
    highlight: hsl(h, s * 0.9, l + 0.09),
    brow: hsl(h, s, Math.min(l * 0.8, 0.25)),
  };
}

/** Clothes and background: flatter, cleaner, more saturated; nothing pure black. */
function styleFlat(c, lift = 0.05) {
  const [h, s, l] = rgbToHsl(c[0], c[1], c[2]);
  // Boost real colors, but leave near-grays neutral (boosting them invents
  // odd olive/brown tints), and pull very dark colors toward neutral: in dim
  // light, black and charcoal fabric picks up the light's color cast.
  const s2 = l < 0.25 ? s * 0.35 : s < 0.12 ? s : s * 1.45; // cartoons love bold color
  return hsl(h, s2, clamp(lift + l * (1 - lift), 0.12, 0.96));
}

/* ---------- Drawing the face ---------- */

/**
 * Draw cartoon facial features for one face (landmarks `pts`, in canvas
 * pixels) with colors `skin`, `hair` and `iris`.
 * Sizes are measured in "eye widths" (`unit`) so everything scales with the face.
 */
function drawCartoonFace(ctx, pts, skin, hair, iris, shadeSide = 1) {
  const unit = (dist(pts[33], pts[133]) + dist(pts[263], pts[362])) / 2;
  if (unit < 3) return; // too small to draw features
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // Cel-shaded side shadow: one crisp shadow shape on the side of the face
  // away from the light (`shadeSide`: -1 = left, 1 = right, from the photo).
  const oval = pick(pts, LM.faceOval);
  const fc = centroid(oval);
  const fxs = oval.map((p) => p.x), fys = oval.map((p) => p.y);
  const faceW = Math.max(...fxs) - Math.min(...fxs), faceH = Math.max(...fys) - Math.min(...fys);
  ctx.save();
  ctx.beginPath();
  smoothClosedPath(ctx, oval);
  ctx.clip();
  ctx.fillStyle = css(skin.shadow, 0.75);
  ctx.beginPath();
  ctx.ellipse(fc.x + shadeSide * faceW * 0.66, fc.y + faceH * 0.06, faceW * 0.34, faceH * 0.66, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Nose: a soft shadow line down the shaded side of the nose bridge.
  const bridge = pick(pts, [168, 197, 195, 5]).map((p) => ({ x: p.x + shadeSide * unit * 0.2, y: p.y }));
  ctx.beginPath();
  smoothOpenPath(ctx, bridge);
  ctx.strokeStyle = css(skin.line, 0.35);
  ctx.lineWidth = unit * 0.06;
  ctx.stroke();

  // Blush: soft pink circles on the cheeks.
  for (const idx of [LM.cheekRight, LM.cheekLeft]) {
    const p = pts[idx], r = unit * 0.75;
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r);
    g.addColorStop(0, css(skin.blush, 0.55));
    g.addColorStop(1, css(skin.blush, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(p.x, p.y, r, r * 0.7, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // Eyebrows: the landmark outline, slightly thickened, filled solid.
  for (const brow of [LM.rightBrow, LM.leftBrow]) {
    const poly = pick(pts, brow);
    const c = centroid(poly);
    ctx.beginPath();
    smoothClosedPath(ctx, scaleAbout(poly, c.x, c.y, 1.04, 1.3));
    ctx.fillStyle = css(hair.brow);
    ctx.fill();
  }

  // Eyes.
  const eyes = [
    { ring: LM.rightEye, iris: LM.rightIris, edge: LM.rightIrisEdge },
    { ring: LM.leftEye, iris: LM.leftIris, edge: LM.leftIrisEdge },
  ];
  const lineColor = '#1f1411';
  for (const e of eyes) {
    const ring = pick(pts, e.ring);
    const c = centroid(ring);
    const ew = dist(ring[0], ring[8]);
    const ys = ring.map((p) => p.y);
    const eh = Math.max(0.5, Math.max(...ys) - Math.min(...ys));
    const upperIdx = e.ring.slice(0, 9);
    const lowerIdx = [e.ring[8], ...e.ring.slice(9), e.ring[0]];

    if (eh / ew < 0.14) {
      // Squinting or laughing: draw a happy closed-eye curve instead.
      ctx.beginPath();
      smoothOpenPath(ctx, scaleAbout(pick(pts, upperIdx), c.x, c.y, 1.1, 1.6));
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = unit * 0.1;
      ctx.stroke();
      continue;
    }

    // Cartoon eyes are a bit bigger and rounder than real ones.
    const sy = clamp(Math.max(1.25, (ew * 0.42) / eh), 1, 2.2), sx = 1.12;
    const shape = scaleAbout(ring, c.x, c.y, sx, sy);

    ctx.beginPath();
    smoothClosedPath(ctx, shape);
    ctx.fillStyle = '#fbf8f3';
    ctx.fill();

    // Iris, pupil and sparkles, clipped to the eye shape.
    ctx.save();
    ctx.beginPath();
    smoothClosedPath(ctx, shape);
    ctx.clip();
    const sys = shape.map((p) => p.y), eyeTop = Math.min(...sys), eyeBottom = Math.max(...sys);
    const ic = pts[e.iris];
    const icx = c.x + (ic.x - c.x) * sx, icy = c.y + (ic.y - c.y) * sy;
    const measured = pick(pts, e.edge).reduce((sum, p) => sum + dist(p, ic), 0) / 4;
    const ir = clamp(measured * 1.2, ew * 0.22, ew * 0.36);
    const grad = ctx.createRadialGradient(icx, icy + ir * 0.3, ir * 0.1, icx, icy, ir);
    grad.addColorStop(0, css(mixColor(iris, [255, 255, 255], 0.25)));
    grad.addColorStop(1, css(mixColor(iris, [0, 0, 0], 0.35)));
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(icx, icy, ir, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#140c09';
    ctx.beginPath(); ctx.arc(icx, icy, ir * 0.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(icx - ir * 0.35, icy - ir * 0.4, ir * 0.3, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(icx + ir * 0.32, icy + ir * 0.28, ir * 0.12, 0, Math.PI * 2); ctx.fill();
    // Thin darker ring around the iris, and the upper lid's shadow across the top of the eye.
    ctx.strokeStyle = css(mixColor(iris, [0, 0, 0], 0.6), 0.8);
    ctx.lineWidth = Math.max(0.6, ir * 0.12);
    ctx.beginPath(); ctx.arc(icx, icy, ir, 0, Math.PI * 2); ctx.stroke();
    const lidShadow = ctx.createLinearGradient(0, eyeTop, 0, eyeTop + (eyeBottom - eyeTop) * 0.45);
    lidShadow.addColorStop(0, css(skin.line, 0.35));
    lidShadow.addColorStop(1, css(skin.line, 0));
    ctx.fillStyle = lidShadow;
    ctx.fillRect(c.x - ew, eyeTop, ew * 2, eyeBottom - eyeTop);
    ctx.restore();

    // Bold upper lid with a little lash flick at the outer corner; thin lower lid.
    const upper = scaleAbout(pick(pts, upperIdx), c.x, c.y, sx, sy);
    ctx.beginPath();
    smoothOpenPath(ctx, upper);
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = unit * 0.11;
    ctx.stroke();
    const out = upper[0], nextPt = upper[1];
    const len = Math.max(1e-3, dist(out, nextPt));
    const dx = (out.x - nextPt.x) / len, dy = (out.y - nextPt.y) / len;
    ctx.beginPath();
    ctx.moveTo(out.x, out.y);
    ctx.lineTo(out.x + dx * unit * 0.2, out.y + dy * unit * 0.2 - unit * 0.1);
    ctx.lineWidth = unit * 0.08;
    ctx.stroke();

    // Eyelid crease: a thin curve above the middle of the upper lid.
    const crease = upper.slice(2, 8).map((p) => ({ x: p.x, y: p.y - unit * 0.2 }));
    ctx.beginPath();
    smoothOpenPath(ctx, crease);
    ctx.strokeStyle = css(skin.line, 0.5);
    ctx.lineWidth = unit * 0.045;
    ctx.stroke();

    ctx.beginPath();
    smoothOpenPath(ctx, scaleAbout(pick(pts, lowerIdx), c.x, c.y, sx, sy));
    ctx.strokeStyle = css(skin.line, 0.55);
    ctx.lineWidth = unit * 0.045;
    ctx.stroke();
  }

  // Nose: just a small curve under the tip, the classic cartoon nose.
  const nb = pts[LM.noseBottom];
  const nr = lerpPt(pts[LM.noseRight], nb, 0.35), nl = lerpPt(pts[LM.noseLeft], nb, 0.35);
  ctx.beginPath();
  ctx.moveTo(nr.x, nr.y);
  ctx.quadraticCurveTo(nb.x, nb.y + unit * 0.12, nl.x, nl.y);
  ctx.strokeStyle = css(skin.line, 0.85);
  ctx.lineWidth = unit * 0.075;
  ctx.stroke();
  // Nostril wings: small curves on each side of the nose.
  for (const idx of [LM.noseRight, LM.noseLeft]) {
    const wing = pts[idx];
    const side = Math.sign(wing.x - nb.x) || 1;
    const end = lerpPt(wing, nb, 0.3);
    ctx.beginPath();
    ctx.moveTo(wing.x, wing.y - unit * 0.14);
    ctx.quadraticCurveTo(wing.x + side * unit * 0.07, wing.y + unit * 0.04, end.x, end.y + unit * 0.02);
    ctx.strokeStyle = css(skin.line, 0.6);
    ctx.lineWidth = unit * 0.05;
    ctx.stroke();
  }

  // Mouth: filled lips; open mouths get a dark inside with teeth, closed ones a lip line.
  const outer = pick(pts, LM.lipsOuter), inner = pick(pts, LM.lipsInner);
  const open = polygonArea(inner) / Math.max(1, polygonArea(outer)) > 0.18;

  // Smile lines: when smiling, short soft curves from the nose wings toward the mouth corners.
  if (open) {
    for (const [wingIdx, cornerIdx] of [[LM.noseRight, 61], [LM.noseLeft, 291]]) {
      const wing = pts[wingIdx], corner = pts[cornerIdx];
      const side = Math.sign(wing.x - nb.x) || 1;
      const a = { x: wing.x + side * unit * 0.28, y: wing.y + unit * 0.05 };
      const b = { x: corner.x + side * unit * 0.22, y: corner.y - unit * 0.05 };
      const ctrl = { x: (a.x + b.x) / 2 + side * unit * 0.12, y: (a.y + b.y) / 2 }; // bows outward
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(ctrl.x, ctrl.y, lerpPt(a, b, 0.8).x, lerpPt(a, b, 0.8).y);
      ctx.strokeStyle = css(skin.line, 0.4);
      ctx.lineWidth = unit * 0.05;
      ctx.stroke();
    }
  }

  ctx.beginPath();
  smoothClosedPath(ctx, outer);
  ctx.fillStyle = css(skin.lips, 0.9);
  ctx.fill();
  // Upper lip a shade darker than the lower lip (light comes from above), and a small shine on the lower lip.
  ctx.beginPath();
  smoothClosedPath(ctx, pick(pts, LM.lipsOuter.slice(0, 11)).concat(pick(pts, LM.lipLine).reverse()));
  ctx.fillStyle = css(mixColor(skin.lips, [60, 20, 20], 0.25), 0.9);
  ctx.fill();
  const shine = lerpPt(pts[17], pts[14], 0.45);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
  ctx.beginPath();
  ctx.ellipse(shine.x, shine.y, unit * 0.18, unit * 0.06, 0, 0, Math.PI * 2);
  ctx.fill();
  if (open) {
    ctx.beginPath();
    smoothClosedPath(ctx, inner);
    ctx.fillStyle = '#5b2323';
    ctx.fill();
    ctx.save();
    ctx.beginPath();
    smoothClosedPath(ctx, inner);
    ctx.clip();
    const iys = inner.map((p) => p.y), top = Math.min(...iys), bottom = Math.max(...iys);
    const ixs = inner.map((p) => p.x);
    ctx.fillStyle = '#fbf7f0';
    ctx.fillRect(Math.min(...ixs), top, Math.max(...ixs) - Math.min(...ixs), (bottom - top) * 0.5);
    ctx.restore();
    ctx.beginPath();
    smoothClosedPath(ctx, inner);
    ctx.strokeStyle = css(skin.line);
    ctx.lineWidth = unit * 0.06;
    ctx.stroke();
  } else {
    ctx.beginPath();
    smoothOpenPath(ctx, pick(pts, LM.lipLine));
    ctx.strokeStyle = css(skin.line);
    ctx.lineWidth = unit * 0.075;
    ctx.stroke();
  }
  ctx.restore();
}

/* ---------- The cartoon pipeline ---------- */

/**
 * Cartoon background: the scene flattened into big simple color shapes.
 * Kuwahara smoothing + k-means (10 colors) + a majority filter give clean
 * "vector" regions. Night skies are repainted and lights get a soft glow.
 * Returns flat colors plus the region labels (used for thin background lines).
 */
async function cartoonBackground(photo, w, h, night) {
  // 10 colors and light smoothing: enough simplification to look drawn, but
  // buildings, windows and railings stay recognizable.
  // Strong Kuwahara passes and a wide majority filter, so textured walls,
  // carpets and foliage become calm shapes instead of blotches.
  const K = 8;
  const { labels, centers } = await runPaintStage(photo, w, h, { radii: [7, 5], k: K, maxIter: 12 });
  let clean = majorityFilter(labels, w, h, K, 4);
  clean = majorityFilter(clean, w, h, K, 4);
  const palette = [];
  for (let c = 0; c < K; c++) {
    let col = styleFlat([centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2]], 0.08);
    if (night) {
      // Cartoon nights are blue: tint the darker shapes toward twilight blue (lights stay warm).
      const l = rgbToHsl(col[0], col[1], col[2])[2];
      col = mixColor(col, [44, 58, 112], clamp((0.85 - l) / 0.6, 0, 1) * 0.55);
    }
    palette.push(col);
  }
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < clean.length; i++) {
    const c = palette[clean[i]];
    rgba[i * 4] = c[0]; rgba[i * 4 + 1] = c[1]; rgba[i * 4 + 2] = c[2]; rgba[i * 4 + 3] = 255;
  }
  // No blur and no glow here: soft edges and haze are what make an image read
  // as a *painting*. Cartoons use crisp, flat shapes with hard edges.
  const sky = detectSkyMask(photo, w, h, night);
  if (sky) paintAnimeSky(rgba, w, h, sky, w, h, night, true);
  return { rgba, labels: clean, sky };
}

/**
 * Build the Cartoon result from an <img>.
 */
async function buildCartoon(image, detail, onStatus) {
  onStatus('Loading the cartoon tools…');
  const progress = [0, 0];
  const report = () => onStatus(`Downloading the cartoon tools (first time only)… ${Math.round(((progress[0] + progress[1]) / 2) * 100)}%`);
  await Promise.all([
    getSegmenter((p) => { progress[0] = p; report(); }),
    getLandmarker((p) => { progress[1] = p; report(); }),
  ]);

  const { width: w, height: h, data } = downscaleImage(image, SETTINGS.cartoonSide);
  const night = averageLightness(data) < 0.28;
  const photo = new Uint8ClampedArray(data);
  applyLut(photo, makeLevelsLut(photo)); // brighten dark photos so colors read clearly
  const photoCanvas = makeCanvas(w, h);
  photoCanvas.getContext('2d').putImageData(new ImageData(photo, w, h), 0, 0);

  // 1. Which pixel is hair / skin / clothes / background?
  onStatus('Finding hair, skin and clothes…');
  await nextFrame();
  const segmenter = await getSegmenter();
  const seg = segmenter.segment(photoCanvas);
  const conf = seg.confidenceMasks.map((m) => {
    const a = new Float32Array(m.getAsFloat32Array());
    return m.width === w && m.height === h ? a : resizeMask(a, m.width, m.height, w, h);
  });
  seg.close();

  // Blur each class's confidence map, then give every pixel its most likely
  // class. Blurring first rounds off jagged edges into smooth, drawable shapes.
  const soft = conf.map((m) => blurTimes(m, w, h, 3));
  soft[SEG.CLOTHES] = soft[SEG.CLOTHES].map((v, i) => v + (soft[SEG.OTHER] ? soft[SEG.OTHER][i] : 0)); // accessories count as clothes
  const labels = new Uint8Array(w * h);
  for (let i = 0; i < labels.length; i++) {
    let best = 0, bestV = -1;
    for (let c = 0; c < 5; c++) if (soft[c][i] > bestV) { bestV = soft[c][i]; best = c; }
    labels[i] = best;
  }
  let personCount = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] !== SEG.BG) personCount++;

  // 2. Face landmarks.
  onStatus('Finding eyes, nose and mouth…');
  await nextFrame();
  const faces = await findFaceLandmarks(photoCanvas);
  // Inside each face, everything becomes face skin: we draw the eyes, brows
  // and mouth ourselves, so the photo's versions must not show through.
  for (const pts of faces) {
    const oval = pick(pts, LM.faceOval);
    const c = centroid(oval);
    const inner = polygonMask(scaleAbout(oval, c.x, c.y, 0.86, 0.86), w, h);
    for (let i = 0; i < labels.length; i++) if (inner[i]) labels[i] = SEG.FACE;
  }

  // 3. The background, flattened.
  onStatus('Simplifying the background…');
  await nextFrame();
  const bg = await cartoonBackground(photo, w, h, night);

  // 4. Character colors, taken from the photo.
  onStatus('Coloring…');
  await nextFrame();
  // Skin tone must match the person. The hue and saturation come from the
  // ORIGINAL photo; the brightness is the original's, moved only 35% of the
  // way toward the brightened copy (so faces in very dark photos still read
  // as lit, without washing out someone's real skin tone). Hair and clothes
  // come from the original too, so black hair stays black.
  // (toneSample ignores flash glare and shine; see its comment.)
  const skinClasses = [SEG.FACE, SEG.BODY];
  const skinOrig = toneSample(data, labels, skinClasses, 0.25, 0.55, 0.12);
  const skinLit = toneSample(photo, labels, skinClasses, 0.25, 0.55, 0.12);
  let skinSample = [200, 150, 120];
  if (skinOrig && skinLit) {
    const [sh, ss, sl] = rgbToHsl(...skinOrig);
    const litL = rgbToHsl(...skinLit)[2];
    // Dark (night/indoor) photos underexpose skin, so they lean more on the brightened copy.
    skinSample = hsl(sh, ss, sl + (litL - sl) * (night ? 0.6 : 0.35));
  }
  const hairSample = toneSample(data, labels, [SEG.HAIR], 0.15, 0.45) || [35, 28, 25];
  const skin = styleSkin(skinSample), hair = styleHair(hairSample);

  // Shading uses brightness from a heavily blurred photo (big, soft light and shadow areas only).
  const lumSharp = toGrayscale(photo, w, h);
  const lum = blurTimes(lumSharp, w, h, 6);
  const lumMedian = (label, source = lum) => {
    const vals = [];
    for (let i = 0; i < labels.length; i += 3) if (labels[i] === label) vals.push(source[i]);
    vals.sort((a, b) => a - b);
    return vals.length ? vals[vals.length >> 1] : 128;
  };
  const faceLum = lumMedian(SEG.FACE);
  const lumHair = blurTimes(lumSharp, w, h, 2);
  const hairLumFine = lumMedian(SEG.HAIR, lumHair);

  // Where hair is close, it casts a shadow onto the forehead (a classic cartoon touch).
  const hairNear = blurTimes(Float32Array.from(labels, (l) => (l === SEG.HAIR ? 1 : 0)), w, h, 5);
  // The face casts a shadow onto the neck just below the jaw. "Just below the
  // jaw" = body-skin pixels close to the face (a blurred face mask) and
  // lower than the face's center, which follows the jawline even when the
  // head is tilted (a straight horizontal cut looked like a band).
  const faceNear = blurTimes(Float32Array.from(labels, (l) => (l === SEG.FACE ? 1 : 0)), w, h, 8);
  let faceCenterY = Infinity;
  for (const pts of faces) faceCenterY = Math.min(faceCenterY, centroid(pick(pts, LM.faceOval)).y);

  // Tone maps (0 = shadow, 1 = base, 2 = highlight), smoothed into blobby cel shapes.
  const shadowRaw = new Float32Array(w * h), highlightRaw = new Float32Array(w * h);
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i], y = (i / w) | 0;
    if (l === SEG.FACE) {
      shadowRaw[i] = hairNear[i] > 0.25 || lum[i] < faceLum * 0.7 ? 1 : 0;
    } else if (l === SEG.BODY) {
      const underJaw = faces.length && y > faceCenterY && faceNear[i] > 0.12;
      shadowRaw[i] = underJaw || lum[i] < faceLum * 0.68 ? 1 : 0;
    } else if (l === SEG.HAIR) {
      // Hair uses a less-blurred brightness so curls and clumps show as
      // separate shapes: darker clumps, the base, and lighter shine.
      shadowRaw[i] = lumHair[i] < hairLumFine * 0.85 ? 1 : 0;
      highlightRaw[i] = lumHair[i] > hairLumFine * 1.6 + 10 ? 1 : 0; // only the brightest shine (flash glare made big blobs)
    } else if (l === SEG.CLOTHES) {
      shadowRaw[i] = 0; // set below, relative to the clothes' own brightness
    }
  }

  // Clothes: ONE base color (median of the original photo) with one soft
  // shadow tone for folds, plus small high-contrast details (logos, zips,
  // piping) as their own light color. Many k-means colors looked like camouflage.
  const clothesLum = lumMedian(SEG.CLOTHES), clothesSharpLum = lumMedian(SEG.CLOTHES, lumSharp);
  const clothesSample = medianColor(data, labels, SEG.CLOTHES);
  const clothes = clothesSample ? {
    base: styleFlat(clothesSample, 0.1),
    shadow: styleFlat(clothesSample.map((v) => v * 0.62), 0.06),
  } : null;
  const detailRaw = new Float32Array(w * h);
  const lumLight = gaussianBlur(lumSharp, w, h);
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== SEG.CLOTHES) continue;
    shadowRaw[i] = lum[i] < clothesLum * 0.8 ? 1 : 0;
    detailRaw[i] = lumLight[i] > clothesSharpLum + 75 ? 1 : 0;
  }
  // Blurring 3 times before the 0.5 threshold makes tiny specks (polka dots,
  // sparkles) vanish while real details like a logo or a zip survive.
  const detailMask = blurTimes(detailRaw, w, h, 3);
  const detailColor = (() => {
    const tmp = new Uint8Array(w * h);
    for (let i = 0; i < tmp.length; i++) tmp[i] = detailMask[i] > 0.5 ? 1 : 0;
    const c = medianColor(photo, tmp, 1);
    return c ? styleFlat(c, 0.1) : [236, 236, 236];
  })();

  const shadow = blurTimes(shadowRaw, w, h, 3), highlight = blurTimes(highlightRaw, w, h, 3);

  // 5. Paint the character over the background (soft 1–2 px edge from the segmenter's confidence).
  const painted = new Uint8ClampedArray(bg.rgba);
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i];
    let c = null;
    if (l === SEG.FACE || l === SEG.BODY) c = shadow[i] > 0.5 ? skin.shadow : skin.base;
    else if (l === SEG.HAIR) c = highlight[i] > 0.5 ? hair.highlight : shadow[i] > 0.5 ? hair.shadow : hair.base;
    else if (l === SEG.CLOTHES && clothes) c = detailMask[i] > 0.5 ? detailColor : shadow[i] > 0.5 ? clothes.shadow : clothes.base;
    const alpha = clamp((0.62 - soft[SEG.BG][i]) / 0.24, 0, 1); // 1 inside the person, fading out at the edge
    if (!c || alpha <= 0) continue;
    const j = i * 4;
    painted[j] += (c[0] - painted[j]) * alpha;
    painted[j + 1] += (c[1] - painted[j + 1]) * alpha;
    painted[j + 2] += (c[2] - painted[j + 2]) * alpha;
  }

  // 6. Draw the faces.
  if (faces.length) {
    onStatus(faces.length > 1 ? `Drawing ${faces.length} faces…` : 'Drawing the face…');
    await nextFrame();
    const fc = makeCanvas(w, h);
    const fctx = fc.getContext('2d', { willReadFrequently: true });
    fctx.putImageData(new ImageData(painted, w, h), 0, 0);
    for (const pts of faces) {
      // Iris color: sampled from the photo at the iris, then deepened; brown if unclear.
      const ic = pts[LM.rightIris];
      const ii = (Math.round(ic.y) * w + Math.round(ic.x)) * 4;
      const [ih, is, il] = rgbToHsl(photo[ii], photo[ii + 1], photo[ii + 2]);
      const iris = is > 0.15 ? hsl(ih, Math.min(0.7, is * 1.3), clamp(il, 0.22, 0.4)) : [92, 58, 38];
      // Which side of the face is darker in the photo? That side gets the cel shadow.
      const noseX = pts[LM.noseBottom].x;
      let leftSum = 0, leftN = 0, rightSum = 0, rightN = 0;
      const oval = pick(pts, LM.faceOval);
      const minY = Math.max(0, Math.round(Math.min(...oval.map((p) => p.y))));
      const maxY = Math.min(h - 1, Math.round(Math.max(...oval.map((p) => p.y))));
      for (let y = minY; y <= maxY; y += 2) {
        for (let x = 0; x < w; x += 2) {
          const i = y * w + x;
          if (labels[i] !== SEG.FACE) continue;
          if (x < noseX) { leftSum += lum[i]; leftN++; } else { rightSum += lum[i]; rightN++; }
        }
      }
      const shadeSide = leftN && rightN && leftSum / leftN < rightSum / rightN ? -1 : 1;
      drawCartoonFace(fctx, pts, skin, hair, iris, shadeSide);
    }
    painted.set(fctx.getImageData(0, 0, w, h).data);
  }

  // 7. Outlines. Bold lines along every border between character regions
  //    (and between clothes colors), traced from a "label map" image where
  //    each region is a different flat gray, so edges are exactly the borders.
  onStatus('Inking the outlines…');
  await nextFrame();
  const scale = outputScale(w, h);
  const grayOf = { [SEG.BG]: 0, [SEG.HAIR]: 40, [SEG.BODY]: 150, [SEG.FACE]: 215 };
  const labelImg = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i];
    const g = l === SEG.CLOTHES ? (detailMask[i] > 0.5 ? 255 : 90) : grayOf[l];
    labelImg[i * 4] = labelImg[i * 4 + 1] = labelImg[i * 4 + 2] = g;
    labelImg[i * 4 + 3] = 255;
  }
  const personStrokes = extractRawStrokes(labelImg, w, h, { detail: 85, strictness: 1, minPoints: 12 });
  for (const s of personStrokes) {
    const p = strokeProminence(s);
    s.outlineWidth = (2.8 + 1.8 * p) / scale;   // 2.8–4.6 px: bold, even cartoon ink
    s.outlineColor = '#22160f';
    s.outlineAlpha = 1;
  }

  // Clothing detail lines: seams, zips, pocket edges and big folds, traced
  // from the PHOTO's real edges, but kept only well inside the clothes (so
  // they don't double the silhouette). Thinner than the outline, like an
  // illustrator's inner lines.
  //   All inner lines are traced from a TEXTURE-FREE copy of the photo
  //   (Kuwahara-smoothed), and texture scribbles are filtered out, so polka
  //   dots, knit patterns, carpet and skin pores don't become lines.
  const lineSrc = textureFree(photo, w, h, 4);
  const insideOf = (mask, s, need) => {
    let inside = 0;
    for (const pt of s.points) if (mask[Math.round(pt.y) * w + Math.round(pt.x)] > 0.95) inside++;
    return inside / s.points.length > need;
  };
  const innerLines = suppressTexture(extractRawStrokes(lineSrc, w, h, { detail: Math.min(100, detail + 10), strictness: 1.3, minPoints: 30 }), w, h);
  const clothesInner = blurTimes(Float32Array.from(labels, (l) => (l === SEG.CLOTHES ? 1 : 0)), w, h, 4);
  const clothesLines = innerLines.filter((s) => insideOf(clothesInner, s, 0.85));
  for (const s of clothesLines) {
    s.outlineWidth = (1.6 + 0.9 * strokeProminence(s)) / scale; // 1.6–2.5 px
    s.outlineColor = '#1a1210';
    s.outlineAlpha = 0.85;
  }
  personStrokes.push(...clothesLines);

  // Skin lines: where an arm crosses the body, fingers, elbows and knees.
  // Arms, hands and chest are all "body skin" to the segmenter, so without
  // these they'd merge into one flat blob.
  const skinInner = blurTimes(Float32Array.from(labels, (l) => (l === SEG.BODY ? 1 : 0)), w, h, 3);
  const skinLines = innerLines.filter((s) => insideOf(skinInner, s, 0.8));
  for (const s of skinLines) {
    s.outlineWidth = (1.7 + 1.0 * strokeProminence(s)) / scale; // 1.7–2.7 px
    s.outlineColor = css(mixColor(skin.line, [0, 0, 0], 0.35));
    s.outlineAlpha = 0.9;
  }
  personStrokes.push(...skinLines);

  // Hair strands: the photo's own edges inside the hair (curls, parting,
  // clumps), drawn as fine lines. Strands along bright, shiny hair are drawn
  // in the highlight color; the rest in a darker hair tone. This is what makes
  // hair read as hair instead of a flat helmet.
  const hairInner = blurTimes(Float32Array.from(labels, (l) => (l === SEG.HAIR ? 1 : 0)), w, h, 2);
  // (Hair uses a lighter smoothing, radius 2, so strands survive but frizz and noise don't.)
  const hairLines = suppressTexture(
    extractRawStrokes(textureFree(photo, w, h, 2), w, h, { detail: Math.min(100, detail + 25), strictness: 1.15, minPoints: 20 }),
    w, h, { maxDensity: 0.12 })
    .filter((s) => {
      let inside = 0;
      for (const pt of s.points) if (hairInner[Math.round(pt.y) * w + Math.round(pt.x)] > 0.9) inside++;
      return inside / s.points.length > 0.85;
    });
  const strandDark = css(mixColor(hair.shadow, [0, 0, 0], 0.45));
  const strandLight = css(mixColor(hair.highlight, [255, 255, 255], 0.2));
  for (const s of hairLines) {
    let sum = 0;
    for (const pt of s.points) sum += lumHair[Math.round(pt.y) * w + Math.round(pt.x)];
    const shiny = sum / s.points.length > hairLumFine * 1.3;
    s.outlineWidth = (shiny ? 1.6 : 1.3 + 0.6 * strokeProminence(s)) / scale;
    s.outlineColor = shiny ? strandLight : strandDark;
    s.outlineAlpha = shiny ? 0.8 : 0.9;
  }
  personStrokes.push(...hairLines);

  // Background ink: thin lines traced from the PHOTO's real edges (building
  // outlines, windows, railings, the horizon), so the scene keeps its
  // structure. The Edge detail slider controls how many.
  const personNear = blurTimes(Float32Array.from(labels, (l) => (l === SEG.BG ? 0 : 1)), w, h, 3);
  const bgRaw = suppressTexture(extractRawStrokes(lineSrc, w, h, { detail, strictness: 1.4, minPoints: 36 }), w, h, { maxDensity: 0.07 });
  const bgStrokes = bgRaw.filter((s) => {
    let inside = 0;
    for (const pt of s.points) {
      const i = Math.round(pt.y) * w + Math.round(pt.x);
      if (personNear[i] > 0.1 || (bg.sky && bg.sky[i] > 0.3)) inside++;
    }
    return inside / s.points.length < 0.2; // not over (or hugging) the character, and not in the painted sky
  });
  for (const s of bgStrokes) {
    s.outlineWidth = (1.2 + 1.0 * strokeProminence(s)) / scale; // 1.2–2.2 px
    s.outlineColor = night ? '#171630' : '#352c48';
    s.outlineAlpha = 0.75;
  }
  const strokes = finalizeStrokes(personStrokes.concat(bgStrokes));

  // The live animation's "paint wash" plan (which regions appear first).
  const { labels: washLabels, centers } = await runPaintStage(painted, w, h, { radii: [], k: 12, maxIter: 10 });
  const art = makeStorybookArt({
    w, h, painted, grain: makePaperGrain(w, h), labels: washLabels, centers,
    customStrokes: strokes, outlineOpacity: 1, faces: faces.length,
  });
  art.mode = 'cartoon';
  if (personCount < w * h * 0.01) art.note = 'No person found, so the scene itself was cartoonized.';
  else if (!faces.length) art.note = "Couldn't find a clear face, so facial features weren't drawn. A closer, front-facing photo works best.";
  return art;
}

/**
 * Find the sky, conservatively. Returns a soft mask (0–1 per pixel) or null.
 * Sky in a photo is (a) smooth, with almost no edges, (b) fairly bright,
 * and (c) connected to the top edge of the picture. So we mark smooth, bright
 * pixels as candidates and "flood fill" down from the top row through
 * connected candidates. The flood stops at edges like rooftops, hills and
 * heads. We only accept the result if it really looks like sky: it touches
 * most of the top edge, sits in the upper part of the image, has a sensible
 * size, and is blue (or it's a night photo). Otherwise, for example with an
 * indoor wall, we leave the picture alone.
 */
function detectSkyMask(rgba, w, h, night) {
  const gray = gaussianBlur(toGrayscale(rgba, w, h), w, h);
  const { magnitude } = sobelEdges(gray, w, h);

  // Typical sky brightness = median lightness of the top 5% of rows. Sky
  // pixels must stay close to it, so hazy hills or dark trees that are just as
  // smooth (but darker) don't get swallowed.
  const topRows = Math.max(2, Math.round(h * 0.05));
  const topL = [];
  for (let i = 0; i < topRows * w; i++) topL.push(gray[i] / 255);
  topL.sort((a, b) => a - b);
  const skyL = topL[topL.length >> 1];

  const cand = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    const [hue, s, l] = rgbToHsl(rgba[j], rgba[j + 1], rgba[j + 2]);
    const lamp = s > 0.4 && (hue < 70 || hue > 330) && l > 0.5; // warm lights aren't sky
    // Skies get BRIGHTER toward the horizon (haze, city glow), while hills and
    // trees are DARKER, so we allow much more room upward than downward.
    const diff = gray[i] / 255 - skyL;
    const nearSkyBrightness = diff > (night ? -0.16 : -0.25) && diff < 0.45;
    cand[i] = magnitude[i] < 18 && l > (night ? 0.12 : 0.35) && nearSkyBrightness && !lamp ? 1 : 0;
  }
  // Rows 0 and h-1 have no Sobel value; treat the top row like row 1.
  for (let x = 0; x < w; x++) cand[x] = cand[w + x];

  const mask = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let top = 0;
  for (let x = 0; x < w; x++) if (cand[x]) { mask[x] = 1; stack[top++] = x; }
  while (top > 0) {
    const i = stack[--top];
    const x = i % w, y = (i / w) | 0;
    const neighbors = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
    for (const j of neighbors) if (j >= 0 && cand[j] && !mask[j]) { mask[j] = 1; stack[top++] = j; }
  }

  // Is it really sky?
  let area = 0, sumY = 0, r = 0, g = 0, b = 0, topTouch = 0;
  for (let x = 0; x < w; x++) topTouch += mask[w + x];
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) continue;
    area++; sumY += (i / w) | 0;
    r += rgba[i * 4]; g += rgba[i * 4 + 1]; b += rgba[i * 4 + 2];
  }
  if (!area) return null;
  const [meanHue, meanSat] = rgbToHsl(r / area, g / area, b / area);
  const blueish = meanSat > 0.12 && meanHue > 180 && meanHue < 260;
  const areaFrac = area / (w * h);
  const ok = topTouch / w > 0.5 && areaFrac > 0.06 && areaFrac < 0.7 && sumY / area < 0.4 * h && (blueish || night);
  if (!ok) return null;

  // Close small holes (birds, wires, stars), then soften the edge by a couple of pixels.
  let soft = new Float32Array(w * h);
  for (let i = 0; i < soft.length; i++) soft[i] = mask[i];
  soft = gaussianBlur(gaussianBlur(soft, w, h), w, h);
  for (let i = 0; i < soft.length; i++) soft[i] = soft[i] > 0.6 ? 1 : 0; // > 0.5 also shrinks it a hair, so no fringe on edges
  return gaussianBlur(soft, w, h);
}

/**
 * Paint an anime sky into the masked area (in place).
 * - A vertical gradient: deep at the top, lighter toward the horizon.
 * - Night: a warm glow above the horizon (city light on the clouds) and
 *   small twinkling stars in the upper sky.
 * - Clouds from layered value noise, stretched sideways into long streaks.
 *   A sharp threshold gives them the crisp, cel-shaded edge of anime clouds,
 *   with a second, brighter band as the sunlit (or moonlit) highlight.
 * `maskSmall` is at size (mw × mh) and is scaled up to the painting's size.
 */
function paintAnimeSky(painted, w, h, maskSmall, mw, mh, night, crisp = false) {
  const mask = new Float32Array(w * h);
  let skyBottom = 1;
  for (let y = 0; y < h; y++) {
    const my = Math.min(mh - 1, Math.floor((y * mh) / h));
    for (let x = 0; x < w; x++) {
      const m = maskSmall[my * mw + Math.min(mw - 1, Math.floor((x * mw) / w))];
      mask[y * w + x] = m;
      if (m > 0.5 && y > skyBottom) skyBottom = y;
    }
  }

  const colors = night
    ? { top: [14, 24, 62], mid: [38, 56, 118], low: [96, 96, 150], glow: [255, 176, 120], cloud: [58, 70, 122], lit: [120, 126, 178] }
    : { top: [52, 120, 214], mid: [96, 164, 236], low: [178, 216, 246], glow: [255, 236, 214], cloud: [236, 241, 250], lit: [255, 255, 255] };
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

  // Cloud noise, stretched ~2.5x horizontally (we sample it with a squashed y).
  const big = makeValueNoise(w, h, 110), med = makeValueNoise(w, h, 45), fine = makeValueNoise(w, h, 14);
  const cloudAt = (x, y) => {
    const i = Math.min(h - 1, Math.floor(y * 2.5)) * w + x;
    // Cartoon clouds skip the fine noise layer, so their outlines are smooth, simple curves.
    return crisp ? 0.62 * big[i] + 0.38 * med[i] : 0.55 * big[i] + 0.3 * med[i] + 0.15 * fine[i];
  };

  for (let y = 0; y < h; y++) {
    const t = clamp(y / skyBottom, 0, 1); // 0 at the top of the picture, 1 at the horizon
    let c = t < 0.6 ? mix(colors.top, colors.mid, t / 0.6) : mix(colors.mid, colors.low, (t - 0.6) / 0.4);
    c = mix(c, colors.glow, Math.pow(t, 3) * (night ? 0.45 : 0.25));
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const m = mask[i];
      if (m <= 0) continue;
      const n = cloudAt(x, y);
      let px = c;
      // Cloud edges: soft for the painted look; for cartoons, fewer clouds with
      // hard, flat edges (a fade of just ~1 pixel keeps them anti-aliased).
      const cloud = crisp
        ? clamp((n - 0.6) / 0.008, 0, 1)
        : clamp((n - 0.56) / 0.05, 0, 1) * (0.35 + 0.65 * t);
      if (cloud > 0) {
        const lit = crisp ? clamp((n - 0.67) / 0.008, 0, 1) : clamp((n - 0.66) / 0.04, 0, 1); // brighter band inside the cloud
        px = mix(px, mix(colors.cloud, colors.lit, lit), cloud * (crisp ? 1 : night ? 0.75 : 0.9));
      }
      const j = i * 4;
      painted[j] += (px[0] - painted[j]) * m;
      painted[j + 1] += (px[1] - painted[j + 1]) * m;
      painted[j + 2] += (px[2] - painted[j + 2]) * m;
    }
  }

  if (night) {
    // Stars: a few hundred small soft dots in the upper, cloud-free sky.
    const count = Math.round(w * h * 0.0005);
    for (let s = 0; s < count; s++) {
      const x = Math.floor(Math.random() * w), y = Math.floor(Math.random() * skyBottom * 0.75);
      if (mask[y * w + x] < 0.95 || cloudAt(x, y) > 0.54) continue;
      const bright = 150 + Math.random() * 105, rad = Math.random() < 0.15 ? 1.6 : 0.9;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const X = x + dx, Y = y + dy;
          if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
          const a = clamp(1 - Math.hypot(dx, dy) / (rad + 0.6), 0, 1) * mask[Y * w + X];
          if (a <= 0) continue;
          const j = (Y * w + X) * 4;
          painted[j] += (bright - painted[j]) * a;
          painted[j + 1] += (bright - painted[j + 1]) * a;
          painted[j + 2] += (Math.min(255, bright + 20) - painted[j + 2]) * a;
        }
      }
    }
  }
}

/** Average brightness (0–1) of RGBA pixels. */
function averageLightness(rgba) {
  let sum = 0;
  for (let j = 0; j < rgba.length; j += 4) sum += 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return sum / (rgba.length / 4) / 255;
}

/**
 * Soft glow ("bloom") around bright lights, especially street lights and
 * windows at night.
 * 1. Keep only the brightest pixels (a "bright pass"), tinted warm.
 * 2. Blur them a lot. A cheap blur trick: shrink the image to 1/8 and 1/24
 *    size, then stretch it back up with smoothing.
 * 3. Add the glow back with a "screen" blend: result = 1 − (1 − a)(1 − b),
 *    which only ever brightens and never blows past white.
 * Bright daytime scenes (lots of bright sky) get less glow so they don't turn hazy.
 */
function addLightGlow(rgba, w, h) {
  const bright = new Uint8ClampedArray(rgba.length);
  let brightCount = 0;
  for (let j = 0; j < rgba.length; j += 4) {
    const lum = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
    const t = clamp((lum - 175) / 60, 0, 1);
    if (t > 0) brightCount++;
    bright[j] = rgba[j] * t;
    bright[j + 1] = rgba[j + 1] * t * 0.9;
    bright[j + 2] = rgba[j + 2] * t * 0.7;
    bright[j + 3] = 255;
  }
  const strength = 0.75 * clamp(1 - (brightCount / (w * h)) * 3, 0.25, 1);

  const src = makeCanvas(w, h);
  src.getContext('2d').putImageData(new ImageData(bright, w, h), 0, 0);
  const glow = makeCanvas(w, h);
  const gctx = glow.getContext('2d', { willReadFrequently: true });
  gctx.imageSmoothingQuality = 'high';
  for (const [div, alpha] of [[8, 0.7], [24, 0.8]]) {
    const small = makeCanvas(Math.max(1, Math.round(w / div)), Math.max(1, Math.round(h / div)));
    const sctx = small.getContext('2d');
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(src, 0, 0, small.width, small.height);
    gctx.globalAlpha = alpha;
    gctx.globalCompositeOperation = 'lighter';
    gctx.drawImage(small, 0, 0, w, h);
  }
  const g = gctx.getImageData(0, 0, w, h).data;
  for (let j = 0; j < rgba.length; j += 4) {
    for (let c = 0; c < 3; c++) {
      const a = rgba[j + c] / 255, b = (g[j + c] / 255) * strength;
      rgba[j + c] = 255 * (1 - (1 - a) * (1 - b));
    }
  }
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
  ui.resultTitle.textContent = mode === 'lineart' ? 'Line art' : 'Cartoon';
  setResultButtons({});
  ui.saveVideoBtn.hidden = !live;
  ui.videoNote.hidden = true;
  ui.emptyMsg.hidden = true;
  ui.canvas.hidden = true;
  ui.progress.hidden = true;
  ui.progressBar.style.width = '0%';
  ui.status.textContent = '';
  ui.loadingText.textContent = mode === 'cartoon' ? 'Drawing…' : 'Sketching…';
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
      : await buildCartoonOrFallback(state.image, detail, onStatus);
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
