/*
 * FaceEngine — a small, readable wrapper around face-api.js
 * ---------------------------------------------------------
 * Everything the website needs for facial recognition lives here, in one file,
 * so the UI code only has to call a handful of functions:
 *
 *   await FaceEngine.load({ modelsPath: './models' });
 *   const faces = await FaceEngine.analyse(videoOrImage);   // detect + identify
 *   await FaceEngine.enrol('Aarav', imageElement);           // remember a person
 *   FaceEngine.draw(canvas, faces);                          // paint boxes + names
 *
 * HOW RECOGNITION WORKS (the pipeline, in plain words)
 *   1. DETECT     Tiny Face Detector (a small CNN) finds every face box in the frame.
 *   2. ALIGN      68 facial landmarks (eyes, nose, mouth outline) are located so the
 *                 face can be normalised regardless of tilt or position.
 *   3. EMBED      A ResNet-34 style network turns each aligned face into a
 *                 128-number vector (the "face descriptor"). Faces of the same person
 *                 produce vectors that are close together in this 128-D space.
 *   4. MATCH      We measure Euclidean distance between the live descriptor and
 *                 every enrolled descriptor. Below the threshold (0.55) = same person.
 *
 * Nothing leaves the browser. Descriptors (not photos) are stored in localStorage.
 */
(function (global) {
  'use strict';

  const DEFAULTS = {
    modelsPath: './models',   // folder containing the *.weights_manifest.json files
    threshold: 0.55,          // max Euclidean distance to call two faces "the same"
    inputSize: 416,           // detector input size for LIVE VIDEO; must be a multiple of 32. Bigger = slower, finds smaller faces.
                              // 416 (not 320) because a face filling the frame scores low at 320 and can drop out.
    photoInputSizes: [608, 320], // STILL IMAGES run at BOTH sizes and the boxes are merged: 608 catches a small
                              // far-away face (full-body shot), 320 catches a huge close-up face. Duplicates are
                              // removed by box overlap. If both find nothing, one last try at photoFallbackSize.
    photoFallbackSize: 1024,
    videoBoostSize: 608,      // when live video finds nothing for a while, retry occasionally at this size
    scoreThreshold: 0.5,      // min detector confidence to keep a box
    storageKey: 'faceengine.enrolments.v1',
    withExpressions: true,    // also classify neutral/happy/sad/... (cheap, fun for demos)
  };

  const state = {
    loaded: false,
    loading: null,
    opts: Object.assign({}, DEFAULTS),
    enrolments: [],           // [{ name: 'Aarav', descriptors: [Float32Array, ...] }]
    matcher: null,            // faceapi.FaceMatcher, rebuilt whenever enrolments change
    listeners: {},
  };

  /* ------------------------------------------------------------------ events */
  function on(event, fn) {
    (state.listeners[event] = state.listeners[event] || []).push(fn);
    return () => off(event, fn);
  }
  function off(event, fn) {
    state.listeners[event] = (state.listeners[event] || []).filter((f) => f !== fn);
  }
  function emit(event, payload) {
    (state.listeners[event] || []).forEach((fn) => {
      try { fn(payload); } catch (e) { console.error('[FaceEngine] listener error', e); }
    });
  }

  /* ---------------------------------------------------------------- loading */
  async function load(opts) {
    if (typeof faceapi === 'undefined') {
      throw new Error('face-api.js is not loaded. Include vendor/face-api.min.js before face-engine.js.');
    }
    state.opts = Object.assign({}, DEFAULTS, opts || {});
    if (state.loading) return state.loading;

    const p = state.opts.modelsPath;
    const jobs = [
      faceapi.nets.tinyFaceDetector.loadFromUri(p),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(p),
      faceapi.nets.faceRecognitionNet.loadFromUri(p),
    ];
    if (state.opts.withExpressions) jobs.push(faceapi.nets.faceExpressionNet.loadFromUri(p));

    state.loading = Promise.all(jobs).then(() => {
      restore();
      state.loaded = true;
      emit('loaded', { enrolments: list() });
      return true;
    });
    return state.loading;
  }

  function isLoaded() { return state.loaded; }

  function detectorOptions(inputSize, scoreThreshold) {
    return new faceapi.TinyFaceDetectorOptions({
      inputSize: inputSize || state.opts.inputSize,
      scoreThreshold: scoreThreshold == null ? state.opts.scoreThreshold : scoreThreshold,
    });
  }

  function assertLoaded() {
    if (!state.loaded) throw new Error('FaceEngine.load() has not finished yet.');
  }

  /* -------------------------------------------------------------- analysing */
  /**
   * Detect every face in an <img>, <video> or <canvas>, compute its descriptor,
   * and (if anyone is enrolled) identify it.
   *
   * Returns an array of plain objects, one per face:
   * {
   *   box:        { x, y, width, height }   in the input's natural pixel space
   *   score:      0..1 detector confidence
   *   landmarks:  [{x,y} x 68]
   *   descriptor: Float32Array(128)
   *   expression: { label: 'happy', probability: 0.93 } | null
   *   match:      { label: 'Aarav' | 'unknown', distance: 0.41, confidence: 0.59 }
   * }
   */
  async function analyse(input, options) {
    assertLoaded();
    const o = options || {};
    const t0 = performance.now();

    // Resolution ladder. Live video uses one small size (speed). Still images try a
    // bigger size first and step up once if nothing is found, because a face that is
    // small in the frame disappears when the whole picture is shrunk to 320 px.
    const isVideo = typeof HTMLVideoElement !== 'undefined' && input instanceof HTMLVideoElement;
    let sizes;
    if (o.inputSize) sizes = [o.inputSize];
    else if (isVideo) sizes = [state.opts.inputSize];
    else sizes = state.opts.photoInputSizes.slice();

    async function runAt(size, score) {
      let chain = faceapi.detectAllFaces(input, detectorOptions(size, score)).withFaceLandmarks(true);
      if (state.opts.withExpressions) chain = chain.withFaceExpressions();
      return chain.withFaceDescriptors();
    }

    let raw = [];
    let usedSize = sizes[0];
    if (isVideo || o.inputSize) {
      raw = await runAt(usedSize);
    } else {
      // Photos: run every size in the list and merge, so one pass is tuned for small faces
      // and another for large ones. mergeByOverlap() drops the duplicate box for the same face.
      const results = [];
      for (const size of sizes) results.push(await runAt(size));
      raw = mergeByOverlap(results.flat());
      usedSize = sizes.join('+');
      if (!raw.length && state.opts.photoFallbackSize) {
        usedSize = state.opts.photoFallbackSize;
        raw = await runAt(usedSize, Math.min(state.opts.scoreThreshold, 0.4));
      }
    }

    const faces = raw.map(toPlain);
    // Keep a small record of the last pass so the UI can show "what just happened":
    // how long the models took, at what resolution, and how close the nearest enrolled person was.
    state.lastRun = {
      ms: Math.round(performance.now() - t0),
      faces: faces.length,
      inputSize: usedSize,
      at: Date.now(),
      nearest: faces.length ? faces.map((f) => ({ label: f.match.label, distance: +f.match.distance.toFixed(3) })) : [],
    };
    return faces;
  }

  /** Intersection-over-union of two boxes: 1 = identical, 0 = no overlap. */
  function iou(a, b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width), y2 = Math.min(a.y + a.height, b.y + b.height);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / (a.width * a.height + b.width * b.height - inter);
  }

  /** True when two boxes are almost certainly the same face: strong overlap, or one box's centre inside the other. */
  function sameFace(a, b) {
    if (iou(a, b) > 0.3) return true;
    const inside = (p, box) => p.x > box.x && p.x < box.x + box.width && p.y > box.y && p.y < box.y + box.height;
    const ca = { x: a.x + a.width / 2, y: a.y + a.height / 2 };
    const cb = { x: b.x + b.width / 2, y: b.y + b.height / 2 };
    return inside(ca, b) && inside(cb, a);
  }

  /** Keep the highest-scoring detection of each face when several passes found the same one. */
  function mergeByOverlap(results) {
    const sorted = results.slice().sort((p, q) => q.detection.score - p.detection.score);
    const kept = [];
    sorted.forEach((r) => {
      if (!kept.some((k) => sameFace(k.detection.box, r.detection.box))) kept.push(r);
    });
    return kept;
  }

  /** Details of the most recent analyse() call: { ms, faces, inputSize, at, nearest[] }. */
  function lastRun() { return state.lastRun ? Object.assign({}, state.lastRun) : null; }

  /** Same as analyse() but expects exactly one face. Throws a friendly error otherwise. */
  async function analyseOne(input) {
    const faces = await analyse(input);
    if (faces.length === 0) throw new Error('No face found. Face the camera with good lighting.');
    if (faces.length > 1) throw new Error('More than one face in frame. Only one person can enrol at a time.');
    return faces[0];
  }

  function toPlain(r) {
    const box = r.detection.box;
    const expr = r.expressions ? topExpression(r.expressions) : null;
    return {
      box: { x: box.x, y: box.y, width: box.width, height: box.height },
      score: r.detection.score,
      landmarks: r.landmarks ? r.landmarks.positions.map((p) => ({ x: p.x, y: p.y })) : [],
      descriptor: r.descriptor,
      expression: expr,
      match: match(r.descriptor),
    };
  }

  function topExpression(expressions) {
    let best = { label: null, probability: 0 };
    Object.keys(expressions).forEach((k) => {
      if (expressions[k] > best.probability) best = { label: k, probability: expressions[k] };
    });
    return best;
  }

  /* --------------------------------------------------------------- matching */
  /** Compare one descriptor against everyone enrolled. */
  function match(descriptor) {
    if (!state.matcher || !descriptor) {
      return { label: 'unknown', distance: 1, confidence: 0 };
    }
    const m = state.matcher.findBestMatch(descriptor);
    const distance = m.distance;
    // Map distance to a 0..1 "confidence" for display. 0 distance = identical.
    const confidence = Math.max(0, Math.min(1, 1 - distance));
    return { label: m.label, distance, confidence: Math.round(confidence * 100) / 100 };
  }

  /** Euclidean distance between two 128-D descriptors (exposed for teaching/demos). */
  function distance(a, b) {
    return faceapi.euclideanDistance(a, b);
  }

  function rebuildMatcher() {
    if (state.enrolments.length === 0) { state.matcher = null; return; }
    const labeled = state.enrolments.map(
      (e) => new faceapi.LabeledFaceDescriptors(e.name, e.descriptors)
    );
    state.matcher = new faceapi.FaceMatcher(labeled, state.opts.threshold);
  }

  /* -------------------------------------------------------------- enrolling */
  /**
   * Remember a person. `input` is an <img>/<video>/<canvas> containing exactly one face.
   * Enrolling the same name again adds another sample (more samples = more robust).
   */
  async function enrol(name, input) {
    const face = await analyseOne(input);
    return enrolDescriptor(name, face.descriptor);
  }

  function enrolDescriptor(name, descriptor) {
    assertLoaded();
    name = String(name || '').trim();
    if (!name) throw new Error('A name is required to enrol a face.');
    let entry = state.enrolments.find((e) => e.name.toLowerCase() === name.toLowerCase());
    if (!entry) { entry = { name, descriptors: [] }; state.enrolments.push(entry); }
    entry.descriptors.push(descriptor);
    rebuildMatcher();
    persist();
    const summary = { name: entry.name, samples: entry.descriptors.length };
    emit('enrolmentsChanged', list());
    return summary;
  }

  function list() {
    return state.enrolments.map((e) => ({ name: e.name, samples: e.descriptors.length }));
  }

  function remove(name) {
    const before = state.enrolments.length;
    state.enrolments = state.enrolments.filter((e) => e.name.toLowerCase() !== String(name).toLowerCase());
    if (state.enrolments.length !== before) {
      rebuildMatcher(); persist(); emit('enrolmentsChanged', list());
      return true;
    }
    return false;
  }

  function clear() {
    state.enrolments = [];
    rebuildMatcher(); persist(); emit('enrolmentsChanged', list());
  }

  /* ------------------------------------------------------------ persistence */
  function persist() {
    try {
      const data = state.enrolments.map((e) => ({
        name: e.name,
        descriptors: e.descriptors.map((d) => Array.from(d)),
      }));
      localStorage.setItem(state.opts.storageKey, JSON.stringify(data));
    } catch (e) { /* private mode etc. — enrolments simply won't survive a reload */ }
  }

  function restore() {
    try {
      const raw = localStorage.getItem(state.opts.storageKey);
      if (!raw) return;
      importJSON(raw, { silent: true });
    } catch (e) { state.enrolments = []; }
    rebuildMatcher();
  }

  function exportJSON() {
    return JSON.stringify(
      state.enrolments.map((e) => ({ name: e.name, descriptors: e.descriptors.map((d) => Array.from(d)) })),
      null, 2
    );
  }

  function importJSON(json, options) {
    const data = typeof json === 'string' ? JSON.parse(json) : json;
    if (!Array.isArray(data)) throw new Error('Enrolment file must be a JSON array.');
    state.enrolments = data
      .filter((e) => e && e.name && Array.isArray(e.descriptors))
      .map((e) => ({ name: String(e.name), descriptors: e.descriptors.map((d) => new Float32Array(d)) }));
    rebuildMatcher();
    if (!(options && options.silent)) { persist(); emit('enrolmentsChanged', list()); }
    return list();
  }

  /* ------------------------------------------------------------- utilities */
  /** Start the webcam into a <video>. Resolves once frames are flowing. */
  async function startCamera(video, constraints) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API unavailable. Open the site via http://localhost (Live Server), not file://.');
    }
    const stream = await navigator.mediaDevices.getUserMedia(
      constraints || { video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }
    );
    video.srcObject = stream;
    video.setAttribute('playsinline', 'true');
    video.muted = true;
    await new Promise((res) => {
      if (video.readyState >= 2) return res();
      video.onloadedmetadata = () => res();
    });
    await video.play();
    return stream;
  }

  function stopCamera(video) {
    const s = video && video.srcObject;
    if (s && s.getTracks) s.getTracks().forEach((t) => t.stop());
    if (video) video.srcObject = null;
  }

  /**
   * Turn a File (from <input type=file>) into a loaded <img>.
   * The object URL stays valid so the same `img.src` can be reused (for example
   * to show the photo on the stage). Call FaceEngine.releaseImage(img) when the
   * photo is no longer needed to free the memory behind the URL.
   */
  function fileToImage(file) {
    return new Promise((resolve, reject) => {
      if (!file || !/^image\//.test(file.type || '')) {
        return reject(new Error('Please choose an image file (JPG, PNG or WebP).'));
      }
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not read that image.')); };
      img.src = url;
    });
  }

  /** Free the object URL behind an <img> created by fileToImage(). */
  function releaseImage(img) {
    if (img && img.src && img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
  }

  /**
   * Freeze the current <video> frame into a <canvas>. Useful for enrolling from a
   * still (so the person can check the frame) and for "capture" buttons.
   */
  function captureFrame(video) {
    const c = document.createElement('canvas');
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
    return c;
  }

  /** Natural pixel size of an <img>/<video>/<canvas>. */
  function naturalSize(input) {
    if (input instanceof HTMLVideoElement) return { width: input.videoWidth, height: input.videoHeight };
    if (input instanceof HTMLImageElement) return { width: input.naturalWidth, height: input.naturalHeight };
    return { width: input.width, height: input.height };
  }

  /**
   * Paint boxes + labels onto an overlay canvas.
   * The canvas is resized to the input's natural size, so keep the canvas and the
   * video/image stacked with identical CSS size and object-fit and the boxes line up.
   */
  function draw(canvas, faces, style) {
    const s = Object.assign({
      known: '#22c55e',        // colour for a recognised face
      unknown: '#f59e0b',      // colour for an unrecognised face
      lineWidth: 2,
      font: '600 14px system-ui, sans-serif',
      labelBg: true,
      showConfidence: true,
      showExpression: true,
      landmarks: false,
      mirror: false,           // set true when the video is CSS-mirrored (selfie view)
    }, style || {});
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    faces.forEach((f) => {
      const isKnown = f.match && f.match.label !== 'unknown';
      const color = isKnown ? s.known : s.unknown;
      let { x, y, width, height } = f.box;
      if (s.mirror) x = canvas.width - x - width;

      ctx.lineWidth = s.lineWidth;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, width, height);

      if (s.landmarks && f.landmarks) {
        ctx.fillStyle = color;
        f.landmarks.forEach((p) => {
          const px = s.mirror ? canvas.width - p.x : p.x;
          ctx.fillRect(px - 1, p.y - 1, 2, 2);
        });
      }

      const parts = [isKnown ? f.match.label : 'Unknown'];
      if (s.showConfidence && isKnown) parts.push(Math.round(f.match.confidence * 100) + '%');
      if (s.showExpression && f.expression && f.expression.label) parts.push(f.expression.label);
      const text = parts.join(' · ');

      ctx.font = s.font;
      const pad = 6;
      const tw = ctx.measureText(text).width + pad * 2;
      const th = 22;
      const ty = y - th - 4 >= 0 ? y - th - 4 : y + height + 4;
      if (s.labelBg) { ctx.fillStyle = color; ctx.fillRect(x, ty, tw, th); ctx.fillStyle = '#000'; }
      else { ctx.fillStyle = color; }
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x + pad, ty + th / 2);
    });
  }

  /** Size an overlay canvas to match its input's natural pixels. Call once per source change. */
  function fitCanvas(canvas, input) {
    const { width, height } = naturalSize(input);
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    return { width, height };
  }

  /**
   * Continuously analyse a <video> and call onResults(faces) each pass.
   * Returns a stop() function. `fps` caps how often we run the (expensive) models.
   */
  function startLoop(video, onResults, options) {
    const o = Object.assign({ fps: 8, adaptive: true }, options || {});
    let running = true;
    let busy = false;
    let last = 0;
    let emptyStreak = 0;   // consecutive passes with no face
    let passes = 0;
    const interval = 1000 / o.fps;

    async function tick(now) {
      if (!running) return;
      if (!busy && now - last >= interval && video.readyState >= 2 && !video.paused) {
        busy = true; last = now;
        try {
          // Adaptive boost: if the small fast pass keeps finding nothing (person far
          // from the camera), every third pass retries at a larger input size.
          const boost = o.adaptive && emptyStreak >= 8 && passes % 3 === 0;
          const faces = await analyse(video, boost ? { inputSize: state.opts.videoBoostSize } : undefined);
          emptyStreak = faces.length ? 0 : emptyStreak + 1;
          passes++;
          onResults(faces);
        }
        catch (e) { console.error('[FaceEngine] loop error', e); }
        busy = false;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    return function stop() { running = false; };
  }

  /* ------------------------------------------------------------------ export */
  global.FaceEngine = {
    load, isLoaded, analyse, analyseOne, match, distance, lastRun,
    enrol, enrolDescriptor, list, remove, clear, exportJSON, importJSON,
    startCamera, stopCamera, captureFrame, fileToImage, releaseImage,
    naturalSize, fitCanvas, draw, startLoop,
    on, off,
    get options() { return Object.assign({}, state.opts); },
    get threshold() { return state.opts.threshold; },
    set threshold(v) { state.opts.threshold = Number(v); rebuildMatcher(); },
    version: '1.3.0',
  };
})(window);
