/*
 * Visage — app.js
 * ---------------
 * WHAT THIS FILE OWNS
 *   The page: reading buttons, inputs and files; deciding what the stage shows;
 *   painting boxes; the people list, the log, the "Under the hood" panel, and
 *   every message the visitor reads. Nothing here knows how a face becomes a
 *   descriptor.
 *
 * WHAT THE ENGINE OWNS (engine/face-engine.js, global `FaceEngine`)
 *   Loading the models, the detect → align → embed → match pipeline, the
 *   enrolled people and their storage, the camera stream, and the timing of
 *   the last analysis. This file only calls it.
 *
 * READING ORDER (matches the numbered section comments below)
 *   1. boot()                 load models, unlock controls, restore people
 *   2. camera mode            startCamera() / stopCamera() and the 8 fps loop
 *   3. photo mode             showPhoto() / recogniseFile(): the no-webcam path
 *   4. enrolment              from a captured still, or from an uploaded image
 *   5. people, log, threshold
 *   6. under the hood         the viva panel, fed by FaceEngine.lastRun()
 *   7. drawing                boxes and labels, scaled to the source
 *   8. listeners              every user action wired in one place
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------ DOM refs */
  const $ = (id) => document.getElementById(id);
  const el = {
    stage: $('stage'), video: $('video'), photo: $('photo'), overlay: $('overlay'),
    stageEmptyText: $('stage-empty-text'), stageStatus: $('stage-status'), stageMsg: $('stage-msg'),
    btnCamera: $('btn-camera'), btnStop: $('btn-stop'), fileRecognise: $('file-recognise'),
    chkLandmarks: $('chk-landmarks'),
    enrolForm: $('enrol-form'), enrolName: $('enrol-name'), btnEnrolCamera: $('btn-enrol-camera'),
    fileEnrol: $('file-enrol'), enrolMsg: $('enrol-msg'),
    threshold: $('threshold'), thresholdVal: $('threshold-val'),
    peopleList: $('people-list'), peopleCount: $('people-count'),
    peopleMsg: $('people-msg'),
    logList: $('log-list'),
    hoodMs: $('hood-ms'), hoodFaces: $('hood-faces'), hoodSize: $('hood-size'), hoodList: $('hood-list'),
  };

  /* --------------------------------------------------------------- state */
  const state = {
    mode: 'idle',        // 'idle' | 'camera' | 'photo'; mirrored on the stage's data-mode for CSS
    stopLoop: null,      // stop() from FaceEngine.startLoop while the camera loop runs
    photoImg: null,      // the <img> from FaceEngine.fileToImage() currently on the stage
    photoFaces: [],      // last analyse() result for that photo, so the threshold can
                         // re-match without re-running the models
    frozenUntil: 0,      // while Date.now() < this, the loop leaves the "captured" still on screen
    gallery: [],         // [{ name, descriptors: [Float32Array] }] mirror of the enrolled people,
                         // used to find the *nearest* name even when the match says Unknown
    lastLogged: {},      // name -> time of its last log line (live-mode throttle)
    log: [],             // newest first, capped at LOG_CAP
  };
  const LOG_CAP = 50;
  const LOG_COOLDOWN_MS = 3000;   // one log line per person per 3 s in live mode
  const LOOP_FPS = 8;             // the brief's target; the models are the real limit
  const CAPTURE_HOLD_MS = 800;    // how long the captured still stays on screen

  /* ---------------------------------------------------------- utilities */
  // One place that changes the stage's mode so CSS and state never disagree.
  function setMode(mode) { state.mode = mode; el.stage.dataset.mode = mode; }

  // Writes a message into an aria-live region; `kind` ('ok' | 'err') only sets colour.
  function say(target, text, kind) {
    target.textContent = text || '';
    target.classList.toggle('is-ok', kind === 'ok');
    target.classList.toggle('is-err', kind === 'err');
  }

  // Controls stay disabled until the models are ready, so a click can never hit an unloaded engine.
  function setEnabled(enabled) {
    [el.btnCamera, el.fileRecognise, el.btnEnrolCamera, el.fileEnrol]
      .forEach((c) => { c.disabled = !enabled; });
  }

  // Log timestamps in the "14:32:05" shape the brief asks for.
  function timeNow() { return new Date().toLocaleTimeString('en-GB', { hour12: false }); }

  // Clears the overlay without touching the video or photo underneath.
  function clearOverlay() {
    el.overlay.getContext('2d').clearRect(0, 0, el.overlay.width, el.overlay.height);
  }

  /* ------------------------------------------------------------ 1. boot */
  // Loads the models, shows elapsed seconds while waiting, and explains the two ways loading fails.
  async function boot() {
    setEnabled(false);
    el.thresholdVal.textContent = Number(el.threshold.value).toFixed(2);

    // file:// blocks both the camera and fetch() of the weights. Say so before anything is clicked.
    if (location.protocol === 'file:') {
      el.stageEmptyText.textContent =
        'Open this folder through a local server (Live Server or python3 -m http.server), not as a file:// URL. ' +
        'Browsers block the camera and the model files on file://.';
      say(el.stageStatus, 'Needs a local server');
      return;
    }

    const t0 = Date.now();
    const ticker = setInterval(() => {
      el.stageEmptyText.textContent = 'Loading the recognition models… ' + Math.round((Date.now() - t0) / 1000) + ' s';
    }, 500);
    try {
      await FaceEngine.load({ modelsPath: './models' });
    } catch (e) {
      clearInterval(ticker);
      console.error(e);
      el.stageEmptyText.textContent =
        'The models could not be loaded. Open this folder through a local server (Live Server or ' +
        'python3 -m http.server), not as a file:// URL, and check that models/ sits next to index.html. (' + e.message + ')';
      say(el.stageStatus, 'Models failed to load');
      return;
    }
    clearInterval(ticker);

    setEnabled(true);
    el.stageEmptyText.textContent = 'Start the camera, or pick a photo to recognise faces in.';
    say(el.stageStatus, 'Ready · models loaded in ' + ((Date.now() - t0) / 1000).toFixed(1) + ' s');
    renderPeople(FaceEngine.list());
    FaceEngine.on('enrolmentsChanged', renderPeople);
  }

  /* ----------------------------------------------------- 2. camera mode */
  // Asks for the webcam, then runs the loop; every failure gets a message that says what to do next.
  async function startCamera() {
    say(el.stageMsg, '');
    el.btnCamera.disabled = true;
    try {
      await FaceEngine.startCamera(el.video);           // resolves once frames flow
    } catch (e) {
      el.btnCamera.disabled = false;
      say(el.stageMsg, cameraErrorText(e), 'err');
      return;
    }
    FaceEngine.fitCanvas(el.overlay, el.video);         // overlay = the video's pixel grid
    clearOverlay();
    setMode('camera');
    el.btnCamera.hidden = true;
    el.btnCamera.disabled = false;
    el.btnStop.hidden = false;
    el.btnStop.focus();
    runLoop();
  }

  // Maps the browser's DOMException names to instructions the teacher can act on.
  function cameraErrorText(e) {
    switch (e && e.name) {
      case 'NotAllowedError':
        return 'Camera permission was denied. Click the camera icon in the address bar to allow it, then try again. ' +
               'Recognise in a photo works without a camera.';
      case 'NotFoundError':
      case 'OverconstrainedError':
        return 'No camera was found on this device. Use "Recognise in a photo" and "Enrol from a photo" instead.';
      case 'NotReadableError':
      case 'AbortError':
        return 'The camera is busy. Close the other app or tab that is using it, then try again.';
      default:
        return (e && e.message ? e.message : 'The camera could not be started.') + ' You can still use a photo.';
    }
  }

  // The 8 fps loop: analyse a frame, paint it, feed the log and the hood panel.
  function runLoop() {
    if (state.stopLoop) return;
    state.stopLoop = FaceEngine.startLoop(el.video, (faces) => {
      // While a captured still is being shown, leave it alone so the teacher sees the sample.
      if (Date.now() < state.frozenUntil) return;
      drawFaces(el.overlay, faces, true);              // video is CSS-mirrored, so mirror x
      say(el.stageStatus, faces.length === 0 ? 'Camera live · nobody in frame'
        : 'Camera live · ' + faces.length + (faces.length === 1 ? ' face' : ' faces'));
      faces.forEach((f) => {
        // Only recognised people are logged live; "Unknown" every frame would bury the list.
        if (f.match.label !== 'unknown') logRecognition(f.match.label, f.match.confidence, true);
      });
      renderHood(faces);
    }, { fps: LOOP_FPS });
  }

  // Pauses the models (not the stream) while the tab is hidden; nothing to show, nothing to compute.
  function pauseLoop() {
    if (state.stopLoop) { state.stopLoop(); state.stopLoop = null; }
  }

  // Stops the stream and returns the stage to idle; also called on pagehide.
  function stopCamera() {
    pauseLoop();
    FaceEngine.stopCamera(el.video);
    clearOverlay();
    setMode('idle');
    el.btnStop.hidden = true;
    el.btnCamera.hidden = false;
    say(el.stageStatus, 'Ready · camera stopped');
    if (document.activeElement === el.btnStop) el.btnCamera.focus();
  }

  /* ------------------------------------------------------ 3. photo mode */
  // Puts an uploaded image on the stage and draws whatever the engine finds in it.
  async function showPhoto(file) {
    const img = await FaceEngine.fileToImage(file);     // rejects non-images with a readable error
    if (state.mode === 'camera') stopCamera();
    if (state.photoImg) FaceEngine.releaseImage(state.photoImg);   // free the previous object URL
    state.photoImg = img;
    el.photo.src = img.src;                             // v1.1.0 keeps the URL alive, so reuse it
    await new Promise((res) => { if (el.photo.complete && el.photo.naturalWidth) res(); else el.photo.onload = res; });
    FaceEngine.fitCanvas(el.overlay, el.photo);
    clearOverlay();
    setMode('photo');
    say(el.stageStatus, 'Analysing photo…');
    state.photoFaces = await FaceEngine.analyse(el.photo);
    paintPhoto();
    return state.photoFaces;
  }

  // Redraws the photo's boxes after a threshold or enrolment change using the cached descriptors:
  // FaceEngine.match() is step 4 alone, so no network has to run again.
  function paintPhoto() {
    if (state.mode !== 'photo') return;
    state.photoFaces = state.photoFaces.map((f) => Object.assign({}, f, { match: FaceEngine.match(f.descriptor) }));
    drawFaces(el.overlay, state.photoFaces, false);
    const n = state.photoFaces.length;
    say(el.stageStatus, 'Photo · ' + (n === 0 ? 'no faces found' : n + (n === 1 ? ' face' : ' faces')));
    renderHood(state.photoFaces);
  }

  // "Recognise in a photo": the whole pipeline on one file, every face logged.
  async function recogniseFile(file) {
    if (!file) return;
    say(el.stageMsg, '');
    try {
      const faces = await showPhoto(file);
      if (faces.length === 0) {
        say(el.stageMsg, 'No face found in that photo. Try one where the face is larger and well lit.', 'err');
        return;
      }
      faces.forEach((f) => logRecognition(f.match.label === 'unknown' ? 'Unknown' : f.match.label, f.match.confidence, false));
      const known = faces.filter((f) => f.match.label !== 'unknown').length;
      say(el.stageMsg, faces.length + (faces.length === 1 ? ' face' : ' faces') + ' found, ' + known + ' recognised.', 'ok');
    } catch (e) {
      say(el.stageMsg, e.message, 'err');
    }
  }

  /* -------------------------------------------------------- 4. enrolment */
  // Enter or "Enrol from camera": samples the live camera, or the photo on stage, or asks for a photo.
  async function enrolFromStage(event) {
    event.preventDefault();
    const name = requireName();
    if (!name) return;
    if (state.mode === 'camera') {
      // Freeze one frame first so a blink or head turn between click and analysis
      // cannot change the sample, and show that frame so the teacher sees what was kept.
      const still = FaceEngine.captureFrame(el.video);
      showCaptured(still);
      await enrol(name, still);
    } else if (state.mode === 'photo') {
      await enrol(name, el.photo);
      paintPhoto();                                     // the box should now carry the name
    } else {
      say(el.enrolMsg, 'No camera running. Choose a photo of ' + name + ' to enrol from.');
      el.fileEnrol.click();                             // opens the picker; allowed because Enter/click is a user gesture
    }
  }

  // Paints the captured still over the live feed for a moment, with a short flash.
  function showCaptured(still) {
    const ctx = el.overlay.getContext('2d');
    ctx.save();
    ctx.translate(el.overlay.width, 0); ctx.scale(-1, 1);   // match the mirrored video
    ctx.drawImage(still, 0, 0, el.overlay.width, el.overlay.height);
    ctx.restore();
    state.frozenUntil = Date.now() + CAPTURE_HOLD_MS;
    el.stage.classList.remove('is-captured');
    void el.stage.offsetWidth;                          // restart the CSS flash animation
    el.stage.classList.add('is-captured');
    say(el.stageStatus, 'Captured');
  }

  // "Enrol from a photo": show the file, then enrol from it.
  async function enrolFromFile(file) {
    if (!file) return;
    const name = requireName();
    if (!name) { say(el.enrolMsg, 'Type a name first, then choose the photo.', 'err'); return; }
    try {
      await showPhoto(file);
      await enrol(name, el.photo);
      paintPhoto();
    } catch (e) {
      say(el.enrolMsg, e.message, 'err');
    }
  }

  // Reads the name field; an empty name gets a message and focus instead of an engine error.
  function requireName() {
    const name = el.enrolName.value.trim();
    if (!name) { say(el.enrolMsg, 'Type a name to enrol.', 'err'); el.enrolName.focus(); }
    return name;
  }

  // The one call to FaceEngine.enrol(); the engine's own errors (0 or >1 faces) are shown as-is.
  async function enrol(name, source) {
    say(el.enrolMsg, 'Capturing…');
    try {
      const result = await FaceEngine.enrol(name, source);   // { name, samples }
      say(el.enrolMsg, result.name + ' enrolled · ' + result.samples +
        (result.samples === 1 ? ' sample' : ' samples') + '.', 'ok');
    } catch (e) {
      say(el.enrolMsg, e.message, 'err');
    }
  }

  /* ---------------------------------------------------- 5. bookkeeping */
  // Rebuilds the people list and the local gallery whenever the engine says enrolments changed.
  function renderPeople(list) {
    state.gallery = JSON.parse(FaceEngine.exportJSON())
      .map((p) => ({ name: p.name, descriptors: p.descriptors.map((d) => new Float32Array(d)) }));
    el.peopleList.textContent = '';
    el.peopleCount.textContent = String(list.length);
    list.forEach((p) => {
      const li = document.createElement('li');
      const name = document.createElement('span'); name.className = 'name'; name.textContent = p.name;
      const samples = document.createElement('span'); samples.className = 'samples';
      samples.textContent = p.samples + (p.samples === 1 ? ' sample' : ' samples');
      const btn = document.createElement('button'); btn.type = 'button'; btn.textContent = 'Remove';
      btn.setAttribute('aria-label', 'Remove ' + p.name);
      btn.addEventListener('click', () => { FaceEngine.remove(p.name); say(el.peopleMsg, p.name + ' removed.'); paintPhoto(); });
      li.append(name, samples, btn);
      el.peopleList.appendChild(li);
    });
  }

  // Adds a log line; in live mode the same person is written at most once per cooldown.
  function logRecognition(name, confidence, throttle) {
    const now = Date.now();
    if (throttle) {
      if (now - (state.lastLogged[name] || 0) < LOG_COOLDOWN_MS) return;
      state.lastLogged[name] = now;
    }
    state.log.unshift({ name, confidence, time: timeNow() });
    if (state.log.length > LOG_CAP) state.log.length = LOG_CAP;
    renderLog();
  }

  // Paints the log, newest first: name, confidence, time.
  function renderLog() {
    el.logList.textContent = '';
    state.log.forEach((entry) => {
      const li = document.createElement('li');
      const who = document.createElement('span');
      who.className = 'who ' + (entry.name === 'Unknown' ? 'unknown' : 'known');
      who.textContent = entry.name;
      const conf = document.createElement('span');
      conf.textContent = entry.name === 'Unknown' ? '—' : Math.round(entry.confidence * 100) + '%';
      const when = document.createElement('span'); when.className = 'when'; when.textContent = entry.time;
      li.append(who, conf, when);
      el.logList.appendChild(li);
    });
  }

  // Applies the slider: the engine rebuilds its matcher, the photo re-matches from cached descriptors.
  function onThreshold() {
    const v = Number(el.threshold.value);
    el.thresholdVal.textContent = v.toFixed(2);
    FaceEngine.threshold = v;
    paintPhoto();                  // live mode picks the new value up on its next frame
  }

  /* ------------------------------------------------- 6. under the hood */
  // Finds the closest enrolled name for a descriptor even when the match says Unknown,
  // so the panel can show *why*: the distance sits above the threshold.
  function nearestEnrolled(descriptor) {
    let best = null;
    state.gallery.forEach((p) => p.descriptors.forEach((d) => {
      const dist = FaceEngine.distance(descriptor, d);
      if (!best || dist < best.distance) best = { name: p.name, distance: dist };
    }));
    return best;
  }

  // Fills the viva panel from FaceEngine.lastRun() plus the per-face nearest distances.
  function renderHood(faces) {
    const run = FaceEngine.lastRun();
    el.hoodMs.textContent = run ? String(run.ms) : '–';
    el.hoodFaces.textContent = run ? String(run.faces) : '–';
    el.hoodSize.textContent = run && run.inputSize ? String(run.inputSize).replace('+', ' + ') + ' px' : '–';   // which detector size(s) the last pass used
    el.hoodList.textContent = '';
    const threshold = FaceEngine.threshold;
    faces.forEach((f, i) => {
      const li = document.createElement('li');
      const n = document.createElement('span'); n.className = 'n'; n.textContent = 'Face ' + (i + 1);
      const known = f.match.label !== 'unknown';
      const verdict = document.createElement('span');
      verdict.className = 'verdict ' + (known ? 'known' : 'unknown');
      verdict.textContent = known ? f.match.label : 'Unknown';
      const why = document.createElement('span'); why.className = 'why';
      const near = nearestEnrolled(f.descriptor);
      if (!near) {
        why.textContent = 'nobody enrolled to compare with';
      } else {
        const d = near.distance.toFixed(2);
        why.innerHTML = 'nearest <b></b> at distance <b></b> ' + (near.distance < threshold ? 'below' : 'above') +
                        ' the threshold <b></b>';
        const b = why.querySelectorAll('b');
        b[0].textContent = near.name; b[1].textContent = d; b[2].textContent = threshold.toFixed(2);
      }
      li.append(n, verdict, why);
      el.hoodList.appendChild(li);
    });
  }

  /* ----------------------------------------------------------- 7. drawing */
  // Boxes are painted here rather than with FaceEngine.draw() so the label scales with the
  // source: a 1280 px webcam frame on a projector needs a bigger label than a 640 px photo.
  // Semantics are the engine's: blue = recognised (name + confidence), amber = Unknown.
  const COLOUR_KNOWN = '#7C8BFF';
  const COLOUR_UNKNOWN = '#F2A93B';

  function drawFaces(canvas, faces, mirror) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = Math.max(1, canvas.width / 640);      // 1 at 640 px wide, 2 at 1280
    const fontPx = Math.round(18 * scale);
    const pad = Math.round(7 * scale);
    const labelH = Math.round(28 * scale);
    ctx.font = '600 ' + fontPx + 'px "Bricolage Grotesque", "Avenir Next", "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = Math.round(3 * scale);

    faces.forEach((f) => {
      const known = f.match && f.match.label !== 'unknown';
      const colour = known ? COLOUR_KNOWN : COLOUR_UNKNOWN;
      let { x, y, width, height } = f.box;
      if (mirror) x = canvas.width - x - width;        // keep the box on the mirrored face

      ctx.strokeStyle = colour;
      ctx.strokeRect(x, y, width, height);

      if (el.chkLandmarks.checked && f.landmarks) {
        ctx.fillStyle = colour;
        const r = Math.max(1.5, 1.5 * scale);
        f.landmarks.forEach((p) => {
          const px = mirror ? canvas.width - p.x : p.x;
          ctx.beginPath(); ctx.arc(px, p.y, r, 0, Math.PI * 2); ctx.fill();
        });
      }

      // Label: "Name · 91% · happy" or "Unknown · neutral", the same shape as the log.
      const parts = [known ? f.match.label : 'Unknown'];
      if (known) parts.push(Math.round(f.match.confidence * 100) + '%');
      if (f.expression && f.expression.label) parts.push(f.expression.label);
      const text = parts.join(' · ');
      const labelW = ctx.measureText(text).width + pad * 2;
      const labelY = y - labelH - 4 >= 0 ? y - labelH - 4 : y + height + 4;   // above, else below
      ctx.fillStyle = colour;
      ctx.fillRect(x, labelY, labelW, labelH);
      ctx.fillStyle = '#101820';
      ctx.fillText(text, x + pad, labelY + labelH / 2);
    });
  }

  /* --------------------------------------------------------- 8. listeners */
  el.btnCamera.addEventListener('click', startCamera);
  el.btnStop.addEventListener('click', stopCamera);
  el.fileRecognise.addEventListener('change', (e) => { recogniseFile(e.target.files[0]); e.target.value = ''; });
  el.enrolForm.addEventListener('submit', enrolFromStage);          // covers Enter in the name field
  el.fileEnrol.addEventListener('change', (e) => { enrolFromFile(e.target.files[0]); e.target.value = ''; });
  el.threshold.addEventListener('input', onThreshold);
  el.chkLandmarks.addEventListener('change', paintPhoto);          // camera mode reads the box each frame
  document.addEventListener('visibilitychange', () => {
    if (state.mode !== 'camera') return;
    if (document.hidden) pauseLoop(); else runLoop();
  });
  window.addEventListener('pagehide', () => { if (state.mode === 'camera') stopCamera(); });

  boot();
})();
