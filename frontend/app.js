/* ══════════════════════════════════════════════════════════════════════════
   SignBridge — Frontend App Logic
   - Webcam capture via getUserMedia
   - Sends frames to Flask /predict endpoint every ~120ms
   - Draws MediaPipe hand skeleton on canvas overlay
   - Animates letter prediction, confidence ring, top-5 bars, word builder
   ══════════════════════════════════════════════════════════════════════════ */

const API = window.location.origin.includes("http") ? window.location.origin : "http://localhost:5050";

// ─── State ─────────────────────────────────────────────────────────────────
let cameraRunning   = false;
let stream          = null;
let loopTimer       = null;
let lastLetter      = null;
let sameLetterCount = 0;
let totalDetections = 0;
let confSum         = 0;
let lettersTyped    = 0;
let currentWord     = "";
let sentence        = "";
let modelStatusText = "Model active";
let islAvailable    = false;

// FPS tracking
let frameCount = 0;
let fpsTimer   = Date.now();

// ─── DOM refs ───────────────────────────────────────────────────────────────
const video        = document.getElementById("video");
const canvas       = document.getElementById("overlay-canvas");
const ctx          = canvas.getContext("2d");
const statusDot    = document.getElementById("status-dot");
const statusText   = document.getElementById("status-text");
const statusPill   = document.getElementById("status-pill");
const letterGlow   = document.getElementById("letter-glow");
const ringFill     = document.getElementById("ring-fill");
const ringPct      = document.getElementById("ring-pct");
const detBadge     = document.getElementById("detection-badge");
const fpsBadge     = document.getElementById("fps-badge");
const barList      = document.getElementById("bar-list");
const wordText     = document.getElementById("word-text");
const sentenceText = document.getElementById("sentence-text");
const histList     = document.getElementById("history-list");
const statTotal    = document.getElementById("stat-total");
const statAvg      = document.getElementById("stat-avg");
const statLetters  = document.getElementById("stat-letters");
const toggleCam    = document.getElementById("toggle-camera");
const showSkel     = document.getElementById("show-skeleton");
const mirrorMode   = document.getElementById("mirror-mode");
const clearBtn     = document.getElementById("clear-history");
const spaceBtn     = document.getElementById("add-space");
const backspBtn    = document.getElementById("backspace-btn");
const toast        = document.getElementById("toast");

// ─── Inject SVG gradient defs ───────────────────────────────────────────────
document.querySelector(".ring-svg").insertAdjacentHTML("afterbegin", `
  <defs>
    <linearGradient id="ring-grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#63b3ed" />
      <stop offset="100%" stop-color="#9f7aea" />
    </linearGradient>
  </defs>
`);

// ─── Toast ──────────────────────────────────────────────────────────────────
function showToast(msg, duration = 2800) {
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), duration);
}

// ─── Backend health check ───────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API}/health`);
    const data = await res.json();
    const ensN = data.ensemble?.num_classes ?? 23;
    islAvailable = data.engines?.dynamic_bilstm ?? true;
    modelStatusText = `Ensemble Engine (3 Models: MLP + BiLSTM + CNN) — ${ensN} ISL signs`;
    setStatus("ok", modelStatusText);
    showToast(`⚡ SignBridge Ensemble Engine Active — ${ensN} ISL Signs`);
    updateIslBadge(islAvailable, 0, false);
    startCamera();
  } catch {
    setStatus("err", "Backend offline — run: python backend/app.py");
    showToast("⚠️ Backend not reachable. Start the Flask server.", 5000);
  }
}

function setStatus(state, msg) {
  statusDot.className  = `status-dot ${state}`;
  statusText.textContent = msg;
}

// ─── Camera ─────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cameraRunning = true;
    toggleCam.textContent = "Stop Camera";
    detBadge.textContent  = "Live";
    detBadge.classList.add("active");
    scheduleFrame();
  } catch (err) {
    setStatus("err", "Camera access denied");
    showToast(`Camera error: ${err.message}`, 4000);
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  clearTimeout(loopTimer);
  cameraRunning = false;
  toggleCam.textContent = "Start Camera";
  detBadge.textContent  = "Stopped";
  detBadge.classList.remove("active");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

toggleCam.addEventListener("click", () => cameraRunning ? stopCamera() : startCamera());

// ─── Capture & Predict loop ─────────────────────────────────────────────────
function scheduleFrame() {
  if (!cameraRunning) return;
  loopTimer = setTimeout(captureAndPredict, 120);   // ~8 fps
}

async function captureAndPredict() {
  if (!cameraRunning || video.readyState < 2) { scheduleFrame(); return; }

  // Sync canvas size to video
  const { videoWidth: vw, videoHeight: vh } = video;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }

  // Draw video frame to hidden tmp canvas → Base64 JPEG
  const tmp = document.createElement("canvas");
  tmp.width  = vw;
  tmp.height = vh;
  const tc   = tmp.getContext("2d");
  if (mirrorMode.checked) {
    tc.translate(vw, 0);
    tc.scale(-1, 1);
  }
  tc.drawImage(video, 0, 0, vw, vh);
  const dataUrl = tmp.toDataURL("image/jpeg", 0.75);

  try {
    const res  = await fetch(`${API}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ frame: dataUrl }),
    });
    const data = await res.json();
    handlePrediction(data, vw, vh);
    setStatus("ok", modelStatusText);
  } catch {
    setStatus("err", "Prediction failed — reconnecting…");
  }

  // FPS counter
  frameCount++;
  const now = Date.now();
  if (now - fpsTimer >= 1000) {
    fpsBadge.textContent = `${frameCount} fps`;
    frameCount = 0;
    fpsTimer   = now;
  }

  scheduleFrame();
}

// ─── Handle prediction response ──────────────────────────────────────────────
function handlePrediction(data, vw, vh) {
  if (!data.detected) {
    setLetter("–", 0, false);
    updateRing(0);
    updateBars([]);
    drawSkeleton([], vw, vh);
    updateIslBadge(islAvailable, 0, false);
    return;
  }

  const { letter, confidence, all_probs, landmarks, isl } = data;

  setLetter(letter, confidence, true);
  updateRing(confidence);
  updateBars(all_probs || []);
  drawSkeleton(landmarks || [], vw, vh);

  // ISL dynamic prediction badge
  if (isl && isl.available) {
    updateIslBadge(true, isl.buffer_frames, isl.buffer_ready, isl.letter, isl.confidence);
  }

  // Word builder: commit letter after 2 stable frames of the same prediction
  if (letter === lastLetter && letter !== "?" && letter !== "–") {
    sameLetterCount++;
    if (sameLetterCount === 2) {
      appendLetter(letter, confidence);
    }
  } else {
    sameLetterCount = 0;
    lastLetter = letter;
  }

  // Stats
  if (letter !== "?" && letter !== "–") {
    totalDetections++;
    confSum += confidence;
    statTotal.textContent = totalDetections;
    statAvg.textContent   = `${(confSum / totalDetections).toFixed(0)}%`;
  }
}

// ─── Letter Display ──────────────────────────────────────────────────────────
function setLetter(letter, conf, detected) {
  if (letterGlow.textContent === letter) return;
  letterGlow.textContent = letter;
  letterGlow.classList.remove("pop");
  if (detected && letter !== "?" && letter !== "–") {
    void letterGlow.offsetWidth;  // force reflow
    letterGlow.classList.add("pop");
    setTimeout(() => letterGlow.classList.remove("pop"), 350);
  }

  const color = !detected || letter === "–" ? "rgba(100,100,120,0.5)"
    : conf >= 70  ? "#63b3ed"
    : conf >= 40  ? "#f6ad55"
    : "#fc8181";

  letterGlow.style.filter = `drop-shadow(0 0 30px ${color})`;
}

// ─── Confidence Ring ─────────────────────────────────────────────────────────
const CIRCUMFERENCE = 2 * Math.PI * 50;   // r = 50

function updateRing(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  const offset  = CIRCUMFERENCE * (1 - clamped / 100);
  ringFill.style.strokeDashoffset = offset;
  ringPct.textContent = `${Math.round(clamped)}%`;
}

// ─── Probability Bars ────────────────────────────────────────────────────────
function updateBars(probs) {
  barList.innerHTML = "";
  probs.forEach(({ label, prob }) => {
    const disp = label === "{" ? "SP" : label.toUpperCase();
    const item = document.createElement("div");
    item.className = "prob-bar-item";
    item.innerHTML = `
      <span class="prob-bar-lbl">${disp}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:${prob}%"></div>
      </div>
      <span class="prob-bar-pct">${prob}%</span>
    `;
    barList.appendChild(item);
  });
}

// ─── Hand Skeleton Drawing ───────────────────────────────────────────────────
// MediaPipe hand connection pairs (21 landmarks)
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],           // thumb
  [0,5],[5,6],[6,7],[7,8],           // index
  [5,9],[9,10],[10,11],[11,12],      // middle
  [9,13],[13,14],[14,15],[15,16],    // ring
  [13,17],[17,18],[18,19],[19,20],   // pinky
  [0,17],[0,5],                      // palm
];

function drawSkeleton(landmarks, vw, vh) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!showSkel.checked || !landmarks.length) return;

  landmarks.forEach((hand) => {
    const pts = hand.map(([x, y]) => {
      const px = mirrorMode.checked ? (1 - x) * vw : x * vw;
      const py = y * vh;
      return { x: px, y: py };
    });

    // Draw connections
    ctx.lineWidth   = 2.5;
    ctx.strokeStyle = "rgba(99, 179, 237, 0.75)";
    ctx.shadowColor = "rgba(99,179,237,0.6)";
    ctx.shadowBlur  = 8;
    CONNECTIONS.forEach(([a, b]) => {
      if (!pts[a] || !pts[b]) return;
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    });

    // Draw keypoints
    ctx.shadowColor = "rgba(159,122,234,0.8)";
    pts.forEach((pt, i) => {
      const r = i === 0 ? 6 : 4;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? "#63b3ed" : "#9f7aea";
      ctx.fill();
    });

    ctx.shadowBlur = 0;
  });
}

// ─── Word Builder ─────────────────────────────────────────────────────────────
function appendLetter(letter, conf) {
  if (letter === "SPACE") {
    if (currentWord.length > 0) {
      sentence += currentWord + " ";
      sentenceText.textContent = sentence || "Start signing…";
      currentWord = "";
      wordText.textContent = "_";
    }
  } else {
    currentWord += letter;
    wordText.textContent = currentWord;
    lettersTyped++;
    statLetters.textContent = lettersTyped;
  }

  addHistoryItem(letter, conf);
}

function addHistoryItem(letter, conf) {
  const now  = new Date();
  const time = now.toLocaleTimeString("en", { hour12: false, hour:"2-digit", minute:"2-digit", second:"2-digit" });
  const li   = document.createElement("li");
  li.className = "history-item";
  li.innerHTML = `
    <span class="history-letter">${letter === "SPACE" ? "SP" : letter}</span>
    <span class="history-conf">${conf.toFixed(1)}%</span>
    <span class="history-time">${time}</span>
  `;
  histList.prepend(li);
  // Keep max 60 items
  if (histList.children.length > 60) histList.lastElementChild.remove();
}

// ─── Controls ─────────────────────────────────────────────────────────────────
spaceBtn.addEventListener("click", () => {
  if (currentWord.length > 0) {
    sentence += currentWord + " ";
    sentenceText.textContent = sentence;
    currentWord = "";
    wordText.textContent = "_";
    showToast(`Word added: "${sentence.trim().split(" ").pop()}" 🔤`);
  }
});

backspBtn.addEventListener("click", () => {
  if (currentWord.length > 0) {
    currentWord = currentWord.slice(0, -1);
    wordText.textContent = currentWord || "_";
    lettersTyped = Math.max(0, lettersTyped - 1);
    statLetters.textContent = lettersTyped;
  }
});

clearBtn.addEventListener("click", () => {
  histList.innerHTML   = "";
  currentWord          = "";
  sentence             = "";
  wordText.textContent = "_";
  sentenceText.textContent = "Start signing…";
  totalDetections = confSum = lettersTyped = 0;
  statTotal.textContent = statAvg.textContent = statLetters.textContent = "0";
  showToast("Cleared ✨");
});

// ─── ISL Dynamic Badge ────────────────────────────────────────────────────────
function updateIslBadge(available, bufFrames, ready, letter, conf) {
  let badge = document.getElementById("isl-badge");
  if (!badge) {
    // Inject ISL badge into the DOM once
    badge = document.createElement("div");
    badge.id = "isl-badge";
    badge.style.cssText = [
      "position:fixed", "bottom:24px", "left:50%", "transform:translateX(-50%)",
      "background:rgba(16,18,30,0.92)", "border:1px solid rgba(99,179,237,0.3)",
      "border-radius:14px", "padding:10px 20px", "display:flex",
      "align-items:center", "gap:12px", "z-index:999",
      "font-family:'Inter',sans-serif", "font-size:13px", "color:#e2e8f0",
      "backdrop-filter:blur(10px)", "box-shadow:0 4px 24px rgba(0,0,0,0.4)",
    ].join(";");
    document.body.appendChild(badge);
  }

  if (!available) {
    badge.style.display = "none";
    return;
  }
  badge.style.display = "flex";

  const TOTAL = 30;
  const filled = Math.min(bufFrames, TOTAL);
  const pct    = Math.round((filled / TOTAL) * 100);
  const barColor = ready ? "#63b3ed" : "#4a5568";
  const letterHtml = (ready && letter)
    ? `<span style="font-size:22px;font-weight:700;color:#63b3ed;letter-spacing:1px">${letter}</span>
       <span style="color:#90cdf4;font-size:11px">${conf?.toFixed(0)}%</span>`
    : `<span style="color:#718096;font-size:12px">gathering…</span>`;

  badge.innerHTML = `
    <span style="color:#9f7aea;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:1px">ISL BiLSTM</span>
    <div style="display:flex;align-items:center;gap:6px">
      <div style="width:80px;height:5px;background:#2d3748;border-radius:3px;overflow:hidden">
        <div style="width:${pct}%;height:100%;background:${barColor};border-radius:3px;transition:width 0.15s"></div>
      </div>
      <span style="color:#718096;font-size:11px">${filled}/${TOTAL}</span>
    </div>
    ${letterHtml}
  `;
}

// ─── Boot ────────────────────────────────────────────────────────────────────
checkHealth();
