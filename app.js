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
const btnFace = document.getElementById('btn-face');
const btnPose = document.getElementById('btn-pose');
const btnHands = document.getElementById('btn-hands');
const btnQuit = document.getElementById('btn-quit');

// Global States
let activeStream = null;
let camera = null;
let progressVal = 0;
let progressInterval = null;
let isLoaded = false;

// Default toggle states (All OFF by default)
const options = {
  showFace: false,
  showPose: false,
  showHands: false,
  showCornerpin: true, // Always active
  mirror: true
};

// Update CSS classes for active buttons
function updateButtonHighlights() {
  btnFace.classList.toggle('active', options.showFace);
  btnPose.classList.toggle('active', options.showPose);
  btnHands.classList.toggle('active', options.showHands);
}

// Attach tap/click events to on-screen control buttons
btnFace.addEventListener('click', () => {
  options.showFace = !options.showFace;
  updateButtonHighlights();
  console.log(`Face: ${options.showFace ? 'ON' : 'OFF'}`);
});

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
    updateButtonHighlights();
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
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// Holistic OnResults
holistic.onResults((results) => {
  if (!isLoaded) {
    isLoaded = true;
    finishLoadingProgress();
  }

  // Clear and prepare canvas
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // 1. Mirror transformation if active
  if (options.mirror) {
    canvasCtx.translate(canvasElement.width, 0);
    canvasCtx.scale(-1, 1);
  }

  // Draw base video frame
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  // Draw Face Mesh if enabled
  if (options.showFace && results.faceLandmarks) {
    drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_CONTOURS, {
      color: 'rgba(255, 255, 255, 0.4)',
      lineWidth: 0.8
    });
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

  // Draw Corner-pinned White Box connecting thumb/index tips of both hands
  if (options.showCornerpin && results.leftHandLandmarks && results.rightHandLandmarks) {
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

  canvasCtx.restore(); // Restore context to default (unmirrored) for HUD text

  // 2. Draw HUD Skeletons Status (Exactly replicating CV2 putText colors & positions)
  canvasCtx.font = 'bold 20px monospace';
  
  // Draw Face Status
  canvasCtx.fillStyle = 'rgb(0, 255, 0)';
  canvasCtx.fillText(`Face [F]: ${options.showFace ? 'ON' : 'OFF'}`, 20, 40);

  // Draw Pose Status
  canvasCtx.fillText(`Pose [P]: ${options.showPose ? 'ON' : 'OFF'}`, 20, 75);

  // Draw Hands Status
  canvasCtx.fillText(`Hands [H]: ${options.showHands ? 'ON' : 'OFF'}`, 20, 110);

  // Draw ESC = Quit status
  canvasCtx.fillStyle = 'rgb(0, 255, 255)';
  canvasCtx.fillText('ESC = Quit', 20, 145);
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
  } else if (key === 'f') {
    options.showFace = !options.showFace;
    updateButtonHighlights();
    console.log(`Face: ${options.showFace ? 'ON' : 'OFF'}`);
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
  startCameraBtn.style.display = 'block';
}

// Event Listeners
startCameraBtn.addEventListener('click', startTracking);
