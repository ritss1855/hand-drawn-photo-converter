# Photo → Drawing

Turn any photo into **glowing line art**, a **colored-pencil drawing**, or a **2D animation style illustration**, and watch it get sketched and colored in live. You can save the result as an image or a video, or share it straight from your phone.

Built with plain **HTML, CSS and vanilla JavaScript**: no frameworks and no build step. The one runtime dependency is Google's MediaPipe Tasks Vision library, which is loaded from a CDN only when it's needed.

> 🔒 **Privacy:** everything runs client-side, inside your browser. Photos are **never uploaded, sent or stored** anywhere. The site is just static files, and all the image processing happens on your own device. The only downloads are the AI model files, which travel *to* your browser. Your photo never goes the other way.

---

## How the three styles work

**Works on all kinds of photos.** Every style first smooths the photo with an *edge-aware* filter, which flattens fabric patterns, carpet, skin pores and noise but keeps real edges. It then finds lines that follow the shapes' flow, and drops "texture scribbles": short lines crammed into busy areas. Long real contours are always kept. Skin is evened out toward its real mid-tone color, so camera flash, glare and shine don't wash out skin tones or turn black hair gray.

**Illustration** turns the photo into a **2D animation** frame, like a still from an animated series: the person is repainted in flat colors with crisp cel shading and clean outlines, over a soft painted background. The shapes all come from the photo itself, so the person still looks like themselves:
1. **Flat base colors:** edge-aware smoothing (the *domain transform*, guided by MediaPipe's body segmentation) flattens each area, skin, hair and each piece of clothing, into one clean color, while the edges between areas stay sharp. Skin gets one even tone: the person's own mid-tone skin color, so flash glare and shadows don't count.
2. **Cel shading:** the photo's light-and-shadow shapes (lightly smoothed) are compared with each area's flat color. Where they're clearly darker, the area gets a second, darker tone with a crisp edge, a little richer in color and warmer on skin, like painted cel shadows. Hair and skin also get a lighter tone where they catch the light, and hair keeps sharper shapes, so it shows clumps and a shine.
3. **Clean outlines:** flow-based XDoG finds the edges between the flat areas and the strongest edges of the form (jaw, ear, the side of the nose), plus the person's outline. They're drawn in a deep shade of the color underneath (dark brown on skin, deep navy on a blue jacket), never on the background.
4. **An animated face** from the 478 face landmarks: clean white eyes with a flat iris in the person's own eye color, a dark pupil and two sparkles, solid brows, a simple nose, flat lips (teeth kept) and a soft blush.
5. **A soft painted background:** the scene behind the person is smoothed into big, gently blended shapes, with no lines, the way animation paints backgrounds separately from characters.

Being an on-device filter, it won't redraw a photo the way large generative AI image models can, but it works offline, costs nothing and never uploads your photo.

**Line art** draws glowing, detailed lines with the *flow-based XDoG* method (after Kang et al., "Coherent Line Drawing", and Winnemöller et al.):
1. Edge-aware smoothing (domain transform) removes noise but keeps fine edges.
2. An *edge tangent flow* field measures which way lines run at every pixel.
3. The Difference-of-Gaussians ink signal is smoothed *along* that flow, so faint, broken detections (hair strands, fabric folds) join into long, flowing lines.
4. This runs at two sizes: one for the main contours and a finer one that adds small details (strands of hair, lashes, creases, textures), merged into one drawing.
5. A third pass looks only inside dark areas (black clothes, dark hair). Their detail sits near the bottom of the lightness scale, where differences are too small to notice, so the darks are stretched (a *gamma* curve, like lifting the shadows in a photo editor) and searched with a more sensitive threshold. That's what brings out the folds and texture of a black dress.

The finished picture is rendered straight from that ink, with natural thick-to-thin line weight, vivid colors from the photo and a neon glow. For the live drawing, the ink is thinned to center lines (Zhang–Suen) and traced into pen strokes, which then cross-fade into the full-detail ink. Face landmarks add exact outlines of the eyes, irises, brows, nose and lips. Dark photos are brightened for line finding only.

**Cartoon** is a **colored-pencil drawing of your photo**: sketched in graphite the way it really is, then colored in with pencil strokes, the way an artist draws from a reference photo. It runs on the same engine as Illustration with a different recipe (a "preset"):

1. **Pencil lines that follow the photo as it is:** flow-based XDoG at two sizes finds the main contours and the fine detail (hair strands, folds, fingers, facial features), and tiny specks are removed. The lines are drawn in graphite gray with a real *pencil texture*: random noise smeared along the edge flow (line integral convolution), so every line looks like it's made of small parallel pencil strokes.
2. **Clean colors:** edge-aware smoothing flattens each area into a clean fill with a few clear shading tones. The colors come from the photo: the hair, clothes and background with a gentle vibrance boost, and the person's **real skin tone**, kept close to the photo (flash glare evened out, but no lightening or orange tint).
3. **Texture vs. structure,** the way an artist sees it. The *structure tensor* measures, around every pixel, how strong the detail is and whether it all runs one way (*coherence*):
   - **Busy texture** (carpet, lace, gravel; strong but pointing every which way) is flattened into a clean fill instead of scribbles.
   - **Directional texture** (wall streaks, wood grain) is smoothed *along* its own grain, so it becomes clean streaks of color instead of smudges.
   - **Bold patterns** (a printed pillow) count as design and are kept.
4. **Detail drawn back in:** the detail the smoothing took away (the shape of the face, folds and lace in the dress, grain in the wall) is added back as shading, so nothing looks airbrushed.
5. **Drawn in pencil, not painted:**
   - The colors get a fine *colored-pencil grain*: evenly spaced diagonal strokes a little lighter or stronger than the color (on average exactly the color), more visible in light areas and subtle on skin.
   - *Graphite hatching* shades whatever is darker than its surroundings (a fold, the side of the nose, under the chin), with diagonal strokes that get denser in deeper shadow and cross-hatched in the deepest. The hatch lines are drawn procedurally (evenly spaced lines, broken into strokes with random pressure that fade in and out), so they look hand-drawn. Skin gets very little hatching, so faces stay smooth.
   - Everything sits on a subtle paper grain.
6. **Outline of the person** from MediaPipe's body segmentation, so the figure stands out from the background.
7. **Face** from the 478 face landmarks. The eyes and the inside of the mouth are kept true to the photo, so the **eye color** (sampled from the iris) and a **smile's teeth** survive. On top go the details an artist draws: individual **brow hairs**, the **eyelid crease**, natural **lashes** (a clean lid line and a few short lashes, on everyone, so no one gets a heavy mascara look), fine lines in the **iris**, a catchlight, the lips, and the nostril wings of the nose. Stray lines across the cheeks are softened.

The *Edge detail* slider controls how many pencil lines appear and how flat the colors get. It works on any photo. With people in it, the face and body tools make skin, hair and clothes look their best, and with no person it sketches and colors the scene.

### Live drawing

Turn on **Show it drawn out live** to watch the picture being made (and save it as a video):

- **Line art:** a glowing pen traces every line, then the full-detail ink fades in.
- **Cartoon and Illustration:** first a pencil sketches the whole drawing on blank paper, including the face's features. Then it's **colored in by hand**, the way you'd color a sketch with colored pencils: skin first, then hair, clothes, and the background last, each area inside its own lines. Every area is colored in layers:
  1. a first, light layer of thin diagonal hatching, with paper still showing between the strokes;
  2. a second layer of strokes crossing it (cross-hatching);
  3. filling in with wider, overlapping strokes until the color is solid.

  The strokes go back and forth across the area patch by patch, with a grainy colored-pencil texture, so the color builds up gradually instead of popping in. Each stroke reveals the finished picture underneath, so the last frame is exactly the final image.
- **Drawing speed** goes from **0.1x** (10 times slower) to **2x**. At 1x, Line art takes about 8 s, and Cartoon and Illustration about 16 s (they're sketched and then colored in). At 0.1x that becomes about 80 s and 160 s.

| Library / model | Size | Source | License |
| --- | --- | --- | --- |
| MediaPipe Tasks Vision 1.0.1 (JS + WebAssembly) | ~10 MB | [jsDelivr](https://www.jsdelivr.com/package/npm/@mediapipe/tasks-vision) | Apache 2.0 |
| Selfie multiclass segmenter (256×256) | 16 MB | [Google MediaPipe models](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) | Apache 2.0 |
| Face landmarker | 3.6 MB | [Google MediaPipe models](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | Apache 2.0 |

The files download the first time you use a style that needs them, then they're kept in the browser's cache. After that it starts quickly, even offline.

**Speed:** Cartoon and Illustration take roughly 5–20 s depending on the device, and a progress message shows each step. Line art is a little quicker. The first run also downloads the models.

### Music while it draws

When **Show it drawn out live** is on, a **Music** menu appears. The song starts when the drawing starts, loops if it's shorter than the drawing, fades out over the final frame, and is **recorded into "Save as video"** (as the video's audio track).

- **Built-in songs** (Lo-fi chill, Dreamy piano, Upbeat chiptune, Night drive synth) aren't audio files. They're *composed in code* with the Web Audio API: a small sequencer schedules chords, bass, melody and drums bar by bar, using oscillators, filters, noise and an echo effect. There's nothing to download and no copyright issues.
- **Your own song:** pick any audio file your browser can play (MP3, M4A, WAV…). It's decoded and played on your device and is never uploaded. If you share videos publicly, only use music you have the rights to.
- **Preview** plays the selected song for 15 seconds.

---

## Files

| File | What it is |
| --- | --- |
| `index.html` | Page structure: the upload/options view and the result view, plus Open Graph / Twitter meta tags and an inline SVG favicon |
| `style.css` | Mobile-first, dark, minimal styling |
| `app.js` | All the logic, split into labeled sections (see below) |
| `README.md` | This file |

The sections in `app.js` are:

0. Settings & helpers
1. **Shared pipeline**: downscale, grayscale, contrast stretch, Gaussian blur, Sobel, non-max suppression, hysteresis threshold, contour tracing, smoothing, RDP simplification, stroke ordering, smooth curves (every style's lines go through the tracing and curve steps to become pen strokes)
2. **Mode 1: Line art**: two-size flow-based XDoG plus a dark-area pass, glowing ink rendering, the live pen
3. **Painting helpers**: paper grain, the "sketch first, then color it in" live drawing (coloring strokes planned patch by patch, revealed through per-area patterns), the result object
4b. **Face & body finder**: loading MediaPipe, segmentation, face landmarks, geometry helpers
4c. **Mode 2: Cartoon** (`buildSketchCartoon`) and the tools it shares with the Illustration: Lab color, domain-transform smoothing, texture vs. structure detection (`textureMap`), flow-aligned smoothing, face feature masks and iris color, skin-tone evening, vibrance, soft quantization, flow-based XDoG lines, detail restoration, colored-pencil grain and graphite hatching (`hatchLines`, `hatchCoverage`), speck removal, Zhang–Suen thinning (for the live pen), face refinement (`refineFace`), and the shared first and last steps (`preparePhoto`, `finishPaintedArt`)
4d. **Mode 3: Illustration** (`buildIllustration`): the 2D animation look: flat base colors, crisp two-tone cel shading, clean outlines, the animated face, a soft painted background
5. Live drawing (animation)
6. Export & sharing (video recording with the music's audio track)
6b. **Music**: Web Audio sequencer, the four built-in songs, user song playback
7. UI wiring

Sections 1–4d don't touch the page's DOM, so they can later be moved into their own files (for example `shared-pipeline.js`, `line-art.js`, `cartoon.js`, `illustration.js`, loaded with extra `<script defer>` tags before `app.js`).

> After changing `app.js` or `style.css`, bump the `?v=` number on their links in `index.html` so visitors' browsers load the new version instead of an old cached copy.

---

## Run it locally (VS Code + Live Server)

1. Install [VS Code](https://code.visualstudio.com/).
2. In VS Code, open the **Extensions** panel (`Ctrl+Shift+X` / `Cmd+Shift+X`), search for **Live Server** (by Ritwick Dey) and click **Install**.
3. **File → Open Folder…** and choose the folder that contains `index.html`.
4. Right-click `index.html` in the file list and choose **Open with Live Server**. You can also click **Go Live** in the blue status bar.
5. Your browser opens at something like `http://127.0.0.1:5500/index.html`. Upload a photo and press **Generate**.

> Tip: to test on your phone, make sure it's on the same Wi-Fi and open `http://<your-computer's-IP>:5500` on the phone.
> Note: some features (Share with files, clipboard) need HTTPS, so they work fully only once the site is deployed. Localhost is treated as secure too.

---

## Deploy to GitHub Pages (free public URL)

### 1. Create a GitHub repository
1. Sign in at [github.com](https://github.com) (create a free account if needed).
2. Click **+** (top-right) → **New repository**.
3. Name it, for example `photo-to-drawing`. Leave it **Public**, and don't add a README (you already have one).
4. Click **Create repository**. Keep the page open, since it shows your repo URL.

### 2. Push the code
Install [Git](https://git-scm.com/downloads) if you don't have it, then open a terminal in the project folder (in VS Code: **Terminal → New Terminal**) and run:

```bash
git init
git add .
git commit -m "Photo to Drawing: first version"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/photo-to-drawing.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. The first push may ask you to sign in to GitHub.

*No terminal?* On the new repo page you can also click **uploading an existing file** and drag in `index.html`, `style.css`, `app.js` and `README.md`, then click **Commit changes**.

### 3. Turn on GitHub Pages
1. In your repo, go to **Settings → Pages** (left sidebar).
2. Under **Build and deployment → Source**, choose **Deploy from a branch**.
3. Under **Branch**, pick **main** and **/ (root)**, then click **Save**.
4. Wait about 1 minute and refresh. GitHub shows your live URL:
   `https://YOUR-USERNAME.github.io/photo-to-drawing/`

Every later `git push` updates the site automatically within a minute or so.

All paths in the project are relative (`style.css`, `app.js`), so it works fine in the `/photo-to-drawing/` sub-folder that GitHub Pages uses.

---

## Alternative: Netlify drag-and-drop deploy

1. Go to [app.netlify.com/drop](https://app.netlify.com/drop) and sign up or log in (free).
2. Drag the **whole project folder** (the one containing `index.html`) onto the page.
3. Netlify uploads it and gives you a URL like `https://random-name-123.netlify.app` within seconds.
4. Optional: **Site configuration → Change site name** to get a nicer URL.

To update, drag the folder onto your site's **Deploys** tab again.

**Vercel** also works with no changes: import the GitHub repo at [vercel.com/new](https://vercel.com/new), keep the framework preset **Other**, and deploy.

---

## Link previews

`index.html` includes Open Graph and Twitter meta tags (title and description), so shared links show a nice card. If you want a preview **image** too, add a screenshot (e.g. `preview.png`) to the repo and add:

```html
<meta property="og:image" content="https://YOUR-USERNAME.github.io/photo-to-drawing/preview.png">
<meta name="twitter:image" content="https://YOUR-USERNAME.github.io/photo-to-drawing/preview.png">
```

(Preview images must be **absolute** URLs, which is why this one isn't included by default.)

---

## Browser support notes

- **Video export** uses `canvas.captureStream()` + `MediaRecorder`. Chrome, Edge and Firefox record **WebM** (VP9 or VP8). Safari only records **MP4**, which the app uses automatically. If neither works, the button is disabled with an explanation.
- **Share** uses the Web Share API with files, mostly on phones. Where that isn't available, the file is downloaded and the site link is copied to your clipboard instead.
- HEIC photos can't be decoded by every browser. iPhones usually convert them to JPEG automatically when you pick them in the browser.
