# Photo → Drawing

Turn any photo into **glowing line art** or a **flat, bold-outlined cartoon of you**, and watch it get drawn live. You can save the result as an image or a video, or share it straight from your phone.

Built with plain **HTML, CSS and vanilla JavaScript**: no frameworks and no build step. The one runtime dependency is Google's MediaPipe Tasks Vision library, which is loaded from a CDN only when it's needed.

> 🔒 **Privacy:** everything runs client-side, inside your browser. Photos are **never uploaded, sent or stored** anywhere. The site is just static files, and all the image processing happens on your own device. The only downloads are the AI model files, which travel *to* your browser. Your photo never goes the other way.

---

## How the two styles work

**Line art** uses classic computer vision, written from scratch: Sobel edges → non-maximum suppression → Canny-style hysteresis → contour tracing → smoothing → RDP simplification → smooth curves. It works at 1400px for fine lines and lots of detail. A **face landmark model** finds faces: each one (with its hair) is re-traced from a zoomed-in crop, and the exact outlines of the eyes, irises, eyebrows, nose, lips and jaw are added from the landmarks.

**Cartoon** doesn't filter the photo. It **draws** a cartoon of it, the way an illustrator works from a reference photo:

1. **Body parts:** MediaPipe's *multiclass selfie segmenter*, a small neural network, labels every pixel as background, hair, face skin, body skin, clothes or accessories. The labels are smoothed into clean, drawable shapes.
2. **Face landmarks:** MediaPipe's *face landmarker* finds 478 points on each face: the outlines of the eyes, irises, eyebrows, lips, nose and jaw.
3. **Flat colors:** each region gets one flat color taken from the photo (the real skin tone, hair and clothing colors), plus simple cel shading. The shading includes a side shadow on the face, a shadow under the hairline and chin, and soft fabric folds. Hair gets three tones (shadow clumps, base, shine) plus fine strand lines traced from the photo's curls. Small high-contrast details like logos and zips keep their own color.
4. **Drawn face:** everything is drawn as vector shapes exactly where the landmarks are:
   - cartoon eyes (whites, iris with a ring, pupil, sparkles, lid shadow, bold upper lid and an eyelid crease), and solid eyebrows;
   - the nose: its bridge shadow, nostrils and tip;
   - two-tone lips with a shine (an open smile gets teeth), smile lines, and blush.

   Squinting eyes become happy closed-eye curves.
5. **Ink:** bold outlines are traced along every border between regions, using the same contour pipeline as Line art. Thinner lines follow real seams, zips and pockets in the clothes.
6. **Background:** flattened into simple color shapes (Kuwahara + k-means + a majority filter), with thin ink lines traced from the photo's real edges (buildings, windows, railings). The *Edge detail* slider sets how many. Everything stays crisp and hard-edged (no blur or glow, which would make it look painted). Night photos get a cartoon starry sky with flat, simple clouds and a blue night tint.
7. **Fallback:** if the cartoon tools can't load (offline, very old browser), it uses a simpler classic filter painting and says so.

It works best on photos with **1–3 people, faces visible and roughly facing the camera**. Closer photos give bigger, more detailed faces. With no person in the photo, it cartoonizes the scene only.

| Library / model | Size | Source | License |
| --- | --- | --- | --- |
| MediaPipe Tasks Vision 1.0.1 (JS + WebAssembly) | ~10 MB | [jsDelivr](https://www.jsdelivr.com/package/npm/@mediapipe/tasks-vision) | Apache 2.0 |
| Selfie multiclass segmenter (256×256) | 16 MB | [Google MediaPipe models](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) | Apache 2.0 |
| Face landmarker | 3.6 MB | [Google MediaPipe models](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | Apache 2.0 |

The files download the first time you use a style that needs them, then they're kept in the browser's cache. After that it starts quickly, even offline.

**Speed:** Cartoon takes roughly 5–20 s depending on the device, and a progress message shows each step. Line art takes a few seconds.

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
1. **Shared pipeline**: downscale, grayscale, contrast stretch, Gaussian blur, Sobel, non-max suppression, hysteresis threshold, contour tracing, smoothing, RDP simplification, stroke ordering, smooth curves
2. **Mode 1: Line art**
3. **Painted fallback + shared painting helpers**: Kuwahara, k-means++, color grading, paper grain, reveal animation, sky detection, painted skies, light glow
4. Web Worker helper
4b. **Mode 2: Cartoon**: loading MediaPipe, segmentation, face landmarks, flat coloring and cel shading, drawing the face, inking the outlines, cartoon background
5. Live drawing (animation)
6. Export & sharing (video recording with the music's audio track)
6b. **Music**: Web Audio sequencer, the four built-in songs, user song playback
7. UI wiring

Sections 1–4b don't touch the page's DOM, so they can later be moved into their own files (for example `shared-pipeline.js`, `line-art.js`, `cartoon.js`, loaded with extra `<script defer>` tags before `app.js`).

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
