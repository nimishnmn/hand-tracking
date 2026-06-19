// DOM Elements
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const startCameraBtn = document.getElementById('start-camera-btn');
const loadingContainer = document.getElementById('loading-container');
const loadingText = document.getElementById('loading-text');
const progressBarFill = document.getElementById('progress-bar-fill');

// Mobile Control Bar Elements
const controlBar = document.getElementById('control-bar');
const btnPose = document.getElementById('btn-pose');
const btnHands = document.getElementById('btn-hands');
const btnFlipH = document.getElementById('btn-fliph');
const btnFlipV = document.getElementById('btn-flipv');
const btnQuit = document.getElementById('btn-quit');

// Preset Selector Elements
const presetSelector = document.getElementById('preset-selector');
const btnPreset1 = document.getElementById('preset-1');
const btnPreset2 = document.getElementById('preset-2');

// Global States
let activeStream = null;
let camera = null;
let progressVal = 0;
let progressInterval = null;
let isLoaded = false;
let lastFrameTime = 0;
let fps = 0;

// Default toggle states (All OFF by default)
const options = {
  showPose: false,
  showHands: false,
  showCornerpin: true, // Always active
  flipH: false,
  flipV: false,
  activePreset: 1
};

// Update CSS classes for active buttons
function updateButtonHighlights() {
  btnPose.classList.toggle('active', options.showPose);
  btnHands.classList.toggle('active', options.showHands);
  btnFlipH.classList.toggle('active', options.flipH);
  btnFlipV.classList.toggle('active', options.flipV);
}

// Update preset highlights
function updatePresetHighlights() {
  btnPreset1.classList.toggle('active', options.activePreset === 1);
  btnPreset2.classList.toggle('active', options.activePreset === 2);
}

// Preset buttons click listeners
btnPreset1.addEventListener('click', () => {
  options.activePreset = 1;
  updatePresetHighlights();
  console.log('Preset 1 activated');
});

btnPreset2.addEventListener('click', () => {
  options.activePreset = 2;
  updatePresetHighlights();
  console.log('Preset 2 activated');
});

// Attach tap/click events to on-screen control buttons

btnPose.addEventListener('click', () => {
  options.showPose = !options.showPose;
  updateButtonHighlights();
  console.log(`Pose: ${options.showPose ? 'ON' : 'OFF'}`);
});

btnHands.addEventListener('click', () => {
  options.showHands = !options.showHands;
  updateButtonHighlights();
  console.log(`Hands: ${options.showHands ? 'ON' : 'OFF'}`);
});

btnFlipH.addEventListener('click', () => {
  options.flipH = !options.flipH;
  updateButtonHighlights();
  console.log(`Flip H: ${options.flipH ? 'ON' : 'OFF'}`);
});

btnFlipV.addEventListener('click', () => {
  options.flipV = !options.flipV;
  updateButtonHighlights();
  console.log(`Flip V: ${options.flipV ? 'ON' : 'OFF'}`);
});

btnQuit.addEventListener('click', () => {
  stopTracking();
});

// Progress bar simulator
function startLoadingProgress() {
  clearInterval(progressInterval);
  progressVal = 0;
  updateProgress(0);
  
  // Progress up to 20% quickly on camera start request
  progressVal = 20;
  updateProgress(progressVal);

  progressInterval = setInterval(() => {
    if (progressVal < 90) {
      // Simulate steady load
      progressVal += Math.floor(Math.random() * 8) + 2;
      if (progressVal > 90) progressVal = 90;
      updateProgress(progressVal);
    }
  }, 100);
}

function updateProgress(value) {
  progressBarFill.style.width = `${value}%`;
  loadingText.textContent = `Loading: ${value}%`;
}

function finishLoadingProgress() {
  clearInterval(progressInterval);
  updateProgress(100);
  setTimeout(() => {
    loadingContainer.style.display = 'none';
    canvasElement.style.display = 'block';
    // Show flat on-screen buttons bar & highlight defaults
    controlBar.style.display = 'flex';
    presetSelector.style.display = 'flex';
    updateButtonHighlights();
    updatePresetHighlights();
  }, 350);
}

// MediaPipe Holistic Setup
const holistic = new Holistic({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// Holistic OnResults
holistic.onResults((results) => {
  if (!isLoaded) {
    isLoaded = true;
    finishLoadingProgress();
  }

  // Calculate FPS
  const now = performance.now();
  if (lastFrameTime > 0) {
    const delta = (now - lastFrameTime) / 1000;
    fps = Math.round(1 / delta);
  }
  lastFrameTime = now;

  // Clear and prepare canvas
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // 1. Mirror / Flip transformation if active
  if (options.flipH || options.flipV) {
    const tX = options.flipH ? canvasElement.width : 0;
    const tY = options.flipV ? canvasElement.height : 0;
    const sX = options.flipH ? -1 : 1;
    const sY = options.flipV ? -1 : 1;
    
    canvasCtx.translate(tX, tY);
    canvasCtx.scale(sX, sY);
  }

  // Draw base video frame
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  // Preset 2: Reduce exposure of background video by 75%
  if (options.activePreset === 2) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
  }



  // Draw Pose Skeleton if enabled
  if (options.showPose && results.poseLandmarks) {
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: '#00e676',
      lineWidth: 2
    });
    drawLandmarks(canvasCtx, results.poseLandmarks, {
      color: '#ff3d00',
      lineWidth: 1,
      radius: 3
    });
  }

  // Draw Hand Skeletons if enabled
  if (options.showHands) {
    if (results.leftHandLandmarks) {
      drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
        color: '#00e676',
        lineWidth: 1.5
      });
      drawLandmarks(canvasCtx, results.leftHandLandmarks, {
        color: '#ffffff',
        lineWidth: 1,
        radius: 2
      });
    }
    if (results.rightHandLandmarks) {
      drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
        color: '#00e676',
        lineWidth: 1.5
      });
      drawLandmarks(canvasCtx, results.rightHandLandmarks, {
        color: '#ffffff',
        lineWidth: 1,
        radius: 2
      });
    }
  }

  // Draw Corner-pinned White Box connecting thumb/index tips of both hands (Preset 1 only)
  if (options.activePreset === 1 && options.showCornerpin && results.leftHandLandmarks && results.rightHandLandmarks) {
    const w = canvasElement.width;
    const h = canvasElement.height;

    // Landmark 4 = THUMB_TIP, Landmark 8 = INDEX_FINGER_TIP
    const lhThumb = results.leftHandLandmarks[4];
    const lhIndex = results.leftHandLandmarks[8];
    const rhThumb = results.rightHandLandmarks[4];
    const rhIndex = results.rightHandLandmarks[8];

    const pt1 = { x: lhThumb.x * w, y: lhThumb.y * h };
    const pt2 = { x: lhIndex.x * w, y: lhIndex.y * h };
    const pt3 = { x: rhIndex.x * w, y: rhIndex.y * h };
    const pt4 = { x: rhThumb.x * w, y: rhThumb.y * h };

    // Fill transparent white box
    canvasCtx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    canvasCtx.beginPath();
    canvasCtx.moveTo(pt1.x, pt1.y);
    canvasCtx.lineTo(pt2.x, pt2.y);
    canvasCtx.lineTo(pt3.x, pt3.y);
    canvasCtx.lineTo(pt4.x, pt4.y);
    canvasCtx.closePath();
    canvasCtx.fill();

    // Outline
    canvasCtx.strokeStyle = '#ffffff';
    canvasCtx.lineWidth = 3;
    canvasCtx.stroke();
  }

  // Draw Preset 2 Glow Mesh (Connecting all H1 points to H2 points, length color reactive & glow)
  if (options.activePreset === 2 && results.leftHandLandmarks && results.rightHandLandmarks) {
    const w = canvasElement.width;
    const h = canvasElement.height;
    const lh = results.leftHandLandmarks;
    const rh = results.rightHandLandmarks;

    canvasCtx.save();
    canvasCtx.globalCompositeOperation = 'lighter';
    canvasCtx.lineWidth = 1;

    const maxDiag = Math.sqrt(w * w + h * h);

    for (let i = 0; i < 21; i++) {
      const pt1 = { x: lh[i].x * w, y: lh[i].y * h };
      for (let j = 0; j < 21; j++) {
        const pt2 = { x: rh[j].x * w, y: rh[j].y * h };

        const dx = pt1.x - pt2.x;
        const dy = pt1.y - pt2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Length reactive color spectrum (Cyan to Pink/Magenta)
        const ratio = Math.min(dist / (maxDiag * 0.65), 1.0);
        const hue = 180 + ratio * 140;

        // Drawing glow line (highly transparent, slightly wider)
        canvasCtx.strokeStyle = `hsla(${hue}, 100%, 55%, 0.15)`;
        
        canvasCtx.beginPath();
        canvasCtx.moveTo(pt1.x, pt1.y);
        canvasCtx.lineTo(pt2.x, pt2.y);
        canvasCtx.stroke();
      }
    }
    canvasCtx.restore();
  }

  canvasCtx.restore(); // Restore context to default (unmirrored) for HUD text

  // 2. Draw HUD Skeletons Status (Exactly replicating CV2 putText colors & positions)
  canvasCtx.font = 'bold 20px monospace';
  
  // Draw ESC = Quit status
  canvasCtx.fillStyle = 'rgb(0, 255, 255)';
  canvasCtx.fillText('ESC = Quit', 20, 40);

  // Draw FPS Status
  canvasCtx.fillText(`FPS: ${fps}`, 20, 75);
});

// Setup Keyboard Listeners (Replicating desktop tracking control shortcuts)
document.addEventListener('keydown', (event) => {
  if (!activeStream) return; // Only listen if tracking is running

  const key = event.key.toLowerCase();

  if (event.key === 'Escape') {
    // ESC -> Exit tracking
    stopTracking();
  } else if (key === 'h') {
    options.showHands = !options.showHands;
    updateButtonHighlights();
    console.log(`Hands: ${options.showHands ? 'ON' : 'OFF'}`);
  } else if (key === 'p') {
    options.showPose = !options.showPose;
    updateButtonHighlights();
    console.log(`Pose: ${options.showPose ? 'ON' : 'OFF'}`);
  } else if (key === 'x') {
    options.flipH = !options.flipH;
    updateButtonHighlights();
    console.log(`Flip H: ${options.flipH ? 'ON' : 'OFF'}`);
  } else if (key === 'y') {
    options.flipV = !options.flipV;
    updateButtonHighlights();
    console.log(`Flip V: ${options.flipV ? 'ON' : 'OFF'}`);
  } else if (key === '1') {
    options.activePreset = 1;
    updatePresetHighlights();
    console.log('Preset 1 activated');
  } else if (key === '2') {
    options.activePreset = 2;
    updatePresetHighlights();
    console.log('Preset 2 activated');
  }
});

// Start camera stream & initialization
async function startTracking() {
  try {
    startCameraBtn.style.display = 'none';
    loadingContainer.style.display = 'block';
    startLoadingProgress();

    activeStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 1280,
        height: 720,
        facingMode: 'user'
      },
      audio: false
    });

    videoElement.srcObject = activeStream;

    camera = new Camera(videoElement, {
      onFrame: async () => {
        await holistic.send({ image: videoElement });
      },
      width: 1280,
      height: 720
    });

    await camera.start();
  } catch (error) {
    console.error("Camera access or init error:", error);
    clearInterval(progressInterval);
    alert(`Failed to start tracking: ${error.message}`);
    stopTracking();
  }
}

// Stop tracking & cleanup
function stopTracking() {
  isLoaded = false;
  lastFrameTime = 0;
  fps = 0;
  clearInterval(progressInterval);

  if (camera) {
    camera.stop();
    camera = null;
  }

  if (activeStream) {
    activeStream.getTracks().forEach(track => track.stop());
    activeStream = null;
  }

  videoElement.srcObject = null;

  // Clear Canvas
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Hide Canvas, Loader, and Controls, Show Start Button
  canvasElement.style.display = 'none';
  loadingContainer.style.display = 'none';
  controlBar.style.display = 'none';
  presetSelector.style.display = 'none';
  startCameraBtn.style.display = 'block';
}

// Event Listeners
startCameraBtn.addEventListener('click', startTracking);
