// Global variables for MediaPipe Tasks Vision API
let FilesetResolver = window.FilesetResolver;
let HandLandmarker = window.HandLandmarker;
let PoseLandmarker = window.PoseLandmarker;

// MediaPipe Constants
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30],
  [29, 31], [30, 32], [27, 31], [28, 32]
];

// Drawing Utilities
function drawLandmarks(ctx, landmarks, style) {
  if (!landmarks) return;
  const color = style?.color || 'white';
  const radius = style?.radius || 4;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const checkVisibility = style?.checkVisibility ?? false;
  
  ctx.save();
  ctx.fillStyle = color;
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (!lm || (checkVisibility && lm.visibility !== undefined && lm.visibility < 0.5)) continue;
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, radius, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
}

function drawConnectors(ctx, landmarks, connections, style) {
  if (!landmarks || !connections) return;
  const color = style?.color || 'white';
  const lineWidth = style?.lineWidth || 1;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const checkVisibility = style?.checkVisibility ?? false;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < connections.length; i++) {
    const [startIdx, endIdx] = connections[i];
    const startLm = landmarks[startIdx];
    const endLm = landmarks[endIdx];
    if (!startLm || (checkVisibility && startLm.visibility !== undefined && startLm.visibility < 0.5)) continue;
    if (!endLm || (checkVisibility && endLm.visibility !== undefined && endLm.visibility < 0.5)) continue;
    ctx.moveTo(startLm.x * w, startLm.y * h);
    ctx.lineTo(endLm.x * w, endLm.y * h);
  }
  ctx.stroke();
  ctx.restore();
}

// DOM Elements
let videoElement;
let canvasElement;
let canvasCtx;
let startCameraBtn;
let welcomeScreen;

let loadingContainer;
let loadingText;
let progressBarFill;

// Mobile Control Bar Elements
let controlBar;
let btnPose;
let btnHands;
let btnFlipH;
let btnFlipV;
let btnQuit;

// Preset Selector Elements
let presetSelector;
let btnPreset1;
let btnPreset2;
let btnPreset3;
let btnPreset4;
let btnPreset5;
let btnPreset6;
let btnOutline;

let preset6Panel;
let preset6PreviewContainer;
let preset6FileInput;
let preset6ChooseBtn;
let preset6FlipHBtn;
let preset6FlipVBtn;
let preset6OverlayContainer;
let overlayFlipH = false;
let overlayFlipV = false;
let universalPanel;
let btnUnlimitedFps;
let btnGlasses;
const glassesImg = new Image();
glassesImg.src = 'glasses.png';
let btnPreset6ModeHand;
let btnPreset6ModeFingers;
let btnPreset6ModePinch;
let btnPreset6Mode3d;
let preset6TrackingMode = 'hand';
let dotCoords = [
  { x: 0.25, y: 0.25 },
  { x: 0.75, y: 0.25 },
  { x: 0.75, y: 0.75 },
  { x: 0.25, y: 0.75 }
];
let dotLoading = [0, 0, 0, 0];
let selectedDotIndex = -1;
let selectedHand = null;

// Offscreen canvas for pixel manipulation (Preset 3)
let offscreenCanvas;
let offCtx;

// Trail canvas for persistent frame buffer (Preset 4)
let trailCanvas;
let trailCtx;

// Global States
let activeStream = null;
let handLandmarker = null;
let poseLandmarker = null;
let reqFrameId = null;
let lastLeftHandLandmarks = null;
let lastRightHandLandmarks = null;
let lastPoseLandmarks = null;
let leftHandAge = 0;
let rightHandAge = 0;
let poseAge = 0;
let loadingAnimation = null;
let lastHandTimestamp = -1;
let lastPoseTimestamp = -1;
let lastFrameTime = 0;
let lastRenderTime = 0;
let fps = 0;

// Gaussian Elimination for Homography mapping
function solveGaussian(A, b) {
  const n = b.length;
  for (let i = 0; i < n; i++) {
    let maxEl = Math.abs(A[i][i]);
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > maxEl) {
        maxEl = Math.abs(A[k][i]);
        maxRow = k;
      }
    }
    for (let k = i; k < n; k++) {
      const tmp = A[maxRow][k];
      A[maxRow][k] = A[i][k];
      A[i][k] = tmp;
    }
    const tmp = b[maxRow];
    b[maxRow] = b[i];
    b[i] = tmp;

    for (let k = i + 1; k < n; k++) {
      const c = -A[k][i] / A[i][i];
      for (let j = i; j < n; j++) {
        if (i === j) {
          A[k][j] = 0;
        } else {
          A[k][j] += c * A[i][j];
        }
      }
      b[k] += c * b[i];
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = b[i] / A[i][i];
    for (let k = i - 1; k >= 0; k--) {
      b[k] -= A[k][i] * x[i];
    }
  }
  return x;
}

function getHomographyMatrix(src, dst) {
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const u = src[i][0];
    const v = src[i][1];
    const x = dst[i][0];
    const y = dst[i][1];
    A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]);
    b.push(x);
    A.push([0, 0, 0, u, v, 1, -u * y, -v * y]);
    b.push(y);
  }
  const c = solveGaussian(A, b);
  return [
    c[0], c[3], 0, c[6],
    c[1], c[4], 0, c[7],
    0,    0,    1, 0,
    c[2], c[5], 0, 1
  ];
}

function getScreenCoords(lm, rect) {
  let x = lm.x;
  let y = lm.y;
  if (options.flipH) {
    x = 1.0 - x;
  }
  if (options.flipV) {
    y = 1.0 - y;
  }
  return {
    x: x * rect.width,
    y: y * rect.height
  };
}

function isConvex(pts) {
  const n = pts.length;
  let sign = 0;
  for (let i = 0; i < n; i++) {
    const p1 = pts[i];
    const p2 = pts[(i + 1) % n];
    const p3 = pts[(i + 2) % n];
    const dx1 = p2[0] - p1[0];
    const dy1 = p2[1] - p1[1];
    const dx2 = p3[0] - p2[0];
    const dy2 = p3[1] - p2[1];
    const cross = dx1 * dy2 - dy1 * dx2;
    if (Math.abs(cross) > 10.0) {
      const currentSign = cross > 0 ? 1 : -1;
      if (sign === 0) {
        sign = currentSign;
      } else if (sign !== currentSign) {
        return false;
      }
    }
  }
  return true;
}

function loadPreset6Media(source, isFile = false) {
  const mediaUrl = isFile ? URL.createObjectURL(source) : source;
  const isImage = isFile ? source.type.startsWith('image/') : false;

  if (isImage) {
    preset6PreviewContainer.innerHTML = `
      <img id="preset6-preview-image" src="${mediaUrl}" style="width: 100%; height: 100%; object-fit: contain;">
    `;
    preset6OverlayContainer.innerHTML = `
      <img id="preset6-overlay-image" src="${mediaUrl}" style="position: absolute; left: 0; top: 0; width: 640px; height: 360px; transform-origin: 0 0; pointer-events: none;">
    `;
  } else {
    preset6PreviewContainer.innerHTML = `
      <video id="preset6-preview-video" controls autoplay loop style="width: 100%; height: 100%; object-fit: contain;">
      </video>
    `;
    preset6OverlayContainer.innerHTML = `
      <video id="preset6-overlay-video" muted autoplay loop playsinline style="position: absolute; left: 0; top: 0; width: 640px; height: 360px; transform-origin: 0 0; pointer-events: none;">
      </video>
    `;

    const previewVideo = document.getElementById('preset6-preview-video');
    const overlayVideo = document.getElementById('preset6-overlay-video');

    if (previewVideo && overlayVideo) {
      if (source === 'camera') {
        previewVideo.srcObject = activeStream;
        overlayVideo.srcObject = activeStream;
        previewVideo.removeAttribute('controls');
        previewVideo.muted = true;
        previewVideo.play().catch(e => console.error("Error playing previewVideo:", e));
        overlayVideo.play().catch(e => console.error("Error playing overlayVideo:", e));
      } else {
        previewVideo.src = mediaUrl;
        overlayVideo.src = mediaUrl;
        previewVideo.load();
        overlayVideo.load();
      }

      previewVideo.addEventListener('play', () => overlayVideo.play());
      previewVideo.addEventListener('pause', () => overlayVideo.pause());
      previewVideo.addEventListener('seeking', () => {
        overlayVideo.currentTime = previewVideo.currentTime;
      });
      previewVideo.addEventListener('seeked', () => {
        overlayVideo.currentTime = previewVideo.currentTime;
      });
      previewVideo.addEventListener('timeupdate', () => {
        const diff = Math.abs(previewVideo.currentTime - overlayVideo.currentTime);
        if (diff > 0.3) {
          overlayVideo.currentTime = previewVideo.currentTime;
        }
      });
    }
  }
}

function pausePlayers() {
  const previewVideo = document.getElementById('preset6-preview-video');
  const overlayVideo = document.getElementById('preset6-overlay-video');
  if (previewVideo && previewVideo.pause) {
    previewVideo.pause();
  }
  if (overlayVideo && overlayVideo.pause) {
    overlayVideo.pause();
  }
}

// Default toggle states (showOutline is ON by default for Preset 3, showGlasses is OFF)
const options = {
  showPose: false,
  showHands: false,
  showCornerpin: true, // Always active
  showOutline: true,
  flipH: false,
  flipV: false,
  activePreset: 1,
  unlimitedFps: false,
  showGlasses: false
};

// Update CSS classes for active buttons
function updateButtonHighlights() {
  btnPose.classList.toggle('active', options.showPose);
  btnHands.classList.toggle('active', options.showHands);
  btnFlipH.classList.toggle('active', options.flipH);
  btnFlipV.classList.toggle('active', options.flipV);
  if (btnUnlimitedFps) btnUnlimitedFps.classList.toggle('active', options.unlimitedFps);
  if (btnGlasses) btnGlasses.classList.toggle('active', options.showGlasses);
  if (btnPreset6ModeHand) btnPreset6ModeHand.classList.toggle('active', preset6TrackingMode === 'hand');
  if (btnPreset6ModeFingers) btnPreset6ModeFingers.classList.toggle('active', preset6TrackingMode === 'fingers');
  if (btnPreset6ModePinch) btnPreset6ModePinch.classList.toggle('active', preset6TrackingMode === 'pinch');
  if (btnPreset6Mode3d) btnPreset6Mode3d.classList.toggle('active', preset6TrackingMode === '3d');
}

// Update preset highlights
function updatePresetHighlights() {
  btnPreset1.classList.toggle('active', options.activePreset === 1);
  btnPreset2.classList.toggle('active', options.activePreset === 2);
  btnPreset3.classList.toggle('active', options.activePreset === 3);
  btnPreset4.classList.toggle('active', options.activePreset === 4);
  btnPreset5.classList.toggle('active', options.activePreset === 5);
  btnPreset6.classList.toggle('active', options.activePreset === 6);
  btnOutline.classList.toggle('active', options.showOutline);
  if (preset6FlipHBtn) preset6FlipHBtn.classList.toggle('active', overlayFlipH);
  if (preset6FlipVBtn) preset6FlipVBtn.classList.toggle('active', overlayFlipV);

  // Show outline toggle only when Preset 3 is active and tracking is active
  if (activeStream && options.activePreset === 3) {
    btnOutline.style.display = 'block';
  } else {
    btnOutline.style.display = 'none';
  }

  // Show Preset 6 control panel and overlay only when Preset 6 is active and tracking is active
  if (activeStream && options.activePreset === 6) {
    preset6Panel.style.display = 'flex';
    preset6OverlayContainer.style.display = 'block';
    
    if (!document.getElementById('preset6-preview-video') && !document.getElementById('preset6-preview-image')) {
      loadPreset6Media("camera", false);
    }
  } else {
    preset6Panel.style.display = 'none';
    preset6OverlayContainer.style.display = 'none';
    pausePlayers();
  }
}

// Initialize DOM and Event Listeners after document loads
document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  videoElement = document.getElementById('webcam');
  canvasElement = document.getElementById('output-canvas');
  canvasCtx = canvasElement.getContext('2d');
  startCameraBtn = document.getElementById('start-camera-btn');
  welcomeScreen = document.getElementById('welcome-screen');

  loadingContainer = document.getElementById('loading-container');
  loadingText = document.getElementById('loading-text');
  progressBarFill = document.getElementById('progress-bar-fill');

  // Mobile Control Bar Elements
  controlBar = document.getElementById('control-bar');
  btnPose = document.getElementById('btn-pose');
  btnHands = document.getElementById('btn-hands');
  btnFlipH = document.getElementById('btn-fliph');
  btnFlipV = document.getElementById('btn-flipv');
  btnQuit = document.getElementById('btn-quit');

  // Preset Selector Elements
  presetSelector = document.getElementById('preset-selector');
  btnPreset1 = document.getElementById('preset-1');
  btnPreset2 = document.getElementById('preset-2');
  btnPreset3 = document.getElementById('preset-3');
  btnPreset4 = document.getElementById('preset-4');
  btnPreset5 = document.getElementById('preset-5');
  btnPreset6 = document.getElementById('preset-6');
  btnOutline = document.getElementById('btn-outline');

  preset6Panel = document.getElementById('preset6-panel');
  preset6PreviewContainer = document.getElementById('preset6-preview-container');
  preset6FileInput = document.getElementById('preset6-file-input');
  preset6ChooseBtn = document.getElementById('preset6-choose-btn');
  preset6FlipHBtn = document.getElementById('preset6-fliph-btn');
  preset6FlipVBtn = document.getElementById('preset6-flipv-btn');
  preset6OverlayContainer = document.getElementById('preset6-overlay-container');
  universalPanel = document.getElementById('universal-panel');
  btnUnlimitedFps = document.getElementById('btn-unlimited-fps');
  btnGlasses = document.getElementById('btn-glasses');
  btnPreset6ModeHand = document.getElementById('preset6-mode-hand');
  btnPreset6ModeFingers = document.getElementById('preset6-mode-fingers');
  btnPreset6ModePinch = document.getElementById('preset6-mode-pinch');
  btnPreset6Mode3d = document.getElementById('preset6-mode-3d');

  // Offscreen canvas for pixel manipulation (Preset 3)
  offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = 1280;
  offscreenCanvas.height = 720;
  offCtx = offscreenCanvas.getContext('2d');

  // Trail canvas for persistent frame buffer (Preset 4)
  trailCanvas = document.createElement('canvas');
  trailCanvas.width = 1280;
  trailCanvas.height = 720;
  trailCtx = trailCanvas.getContext('2d');
  trailCtx.fillStyle = '#000000';
  trailCtx.fillRect(0, 0, 1280, 720);

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

  btnPreset3.addEventListener('click', () => {
    options.activePreset = 3;
    updatePresetHighlights();
    console.log('Preset 3 activated');
  });

  btnPreset4.addEventListener('click', () => {
    options.activePreset = 4;
    updatePresetHighlights();
    console.log('Preset 4 activated');
  });

  btnPreset5.addEventListener('click', () => {
    options.activePreset = 5;
    updatePresetHighlights();
    console.log('Preset 5 activated');
  });

  btnPreset6.addEventListener('click', () => {
    options.activePreset = 6;
    updatePresetHighlights();
    console.log('Preset 6 activated');
  });

  preset6ChooseBtn.addEventListener('click', () => {
    preset6FileInput.click();
  });

  preset6FileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
      loadPreset6Media(file, true);
    }
  });

  preset6FlipHBtn.addEventListener('click', () => {
    overlayFlipH = !overlayFlipH;
    preset6FlipHBtn.classList.toggle('active', overlayFlipH);
  });

  preset6FlipVBtn.addEventListener('click', () => {
    overlayFlipV = !overlayFlipV;
    preset6FlipVBtn.classList.toggle('active', overlayFlipV);
  });

  btnUnlimitedFps.addEventListener('click', () => {
    options.unlimitedFps = !options.unlimitedFps;
    updateButtonHighlights();
    console.log(`Unlimited FPS: ${options.unlimitedFps ? 'ON' : 'OFF'}`);
  });

  btnGlasses.addEventListener('click', () => {
    options.showGlasses = !options.showGlasses;
    updateButtonHighlights();
    console.log(`Glasses: ${options.showGlasses ? 'ON' : 'OFF'}`);
  });

  btnPreset6ModeHand.addEventListener('click', () => {
    preset6TrackingMode = 'hand';
    updateButtonHighlights();
    console.log('Preset 6 Tracking Mode: Hand');
  });

  btnPreset6ModeFingers.addEventListener('click', () => {
    preset6TrackingMode = 'fingers';
    updateButtonHighlights();
    console.log('Preset 6 Tracking Mode: Fingers');
  });

  btnPreset6ModePinch.addEventListener('click', () => {
    preset6TrackingMode = 'pinch';
    updateButtonHighlights();
    console.log('Preset 6 Tracking Mode: Pinch');
  });

  btnPreset6Mode3d.addEventListener('click', () => {
    preset6TrackingMode = '3d';
    updateButtonHighlights();
    console.log('Preset 6 Tracking Mode: 3D');
  });

  btnOutline.addEventListener('click', () => {
    options.showOutline = !options.showOutline;
    updatePresetHighlights();
    console.log(`Outline: ${options.showOutline ? 'ON' : 'OFF'}`);
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

  startCameraBtn.addEventListener('click', startTracking);
});

async function renderLoop(nowMs) {
  if (!handLandmarker || !poseLandmarker) {
    reqFrameId = requestAnimationFrame(renderLoop);
    return;
  }

  // Lock to 25 FPS (40ms interval) unless unlimitedFps is active
  if (!options.unlimitedFps) {
    const fpsInterval = 1000 / 25;
    const elapsed = nowMs - lastRenderTime;
    if (elapsed < fpsInterval) {
      reqFrameId = requestAnimationFrame(renderLoop);
      return;
    }
    lastRenderTime = nowMs - (elapsed % fpsInterval);
  } else {
    lastRenderTime = nowMs;
  }

  console.log('frame, video ready state:', videoElement.readyState);

  const w = canvasElement.width;
  const h = canvasElement.height;

  // Preset 4: Fade trail instead of full clear
  if (options.activePreset === 4) {
    trailCtx.fillStyle = 'rgba(0, 0, 0, 0.15)';
    trailCtx.fillRect(0, 0, w, h);
  }

  // 1. Reset transform to identity and clear screen
  canvasCtx.setTransform(1, 0, 0, 1, 0, 0);
  canvasCtx.clearRect(0, 0, w, h);

  // 2. Save clean state and apply flip transformations for video and landmarks
  canvasCtx.save();

  if (options.flipH) {
    canvasCtx.translate(w, 0);
    canvasCtx.scale(-1, 1);
  }
  if (options.flipV) {
    canvasCtx.translate(0, h);
    canvasCtx.scale(1, -1);
  }

  // Calculate FPS
  if (lastFrameTime > 0) {
    const delta = (nowMs - lastFrameTime) / 1000;
    fps = Math.round(1 / delta);
  }
  lastFrameTime = nowMs;

  let handResult = null;
  let poseResult = null;
  
  if (videoElement.currentTime > 0) {
    let ts = performance.now();
    
    if (handLandmarker) {
      if (ts <= lastHandTimestamp) ts = lastHandTimestamp + 1;
      lastHandTimestamp = ts;
      handResult = handLandmarker.detectForVideo(videoElement, ts);
    }
    
    if (options.showPose || options.activePreset === 5 || options.showGlasses) {
      let pts = performance.now();
      if (pts <= lastPoseTimestamp) pts = lastPoseTimestamp + 1;
      lastPoseTimestamp = pts;
      poseResult = poseLandmarker.detectForVideo(videoElement, pts);
    }
  }

  let currentLeftHand = null;
  let currentRightHand = null;
  
  if (handResult && handResult.landmarks && handResult.landmarks.length > 0) {
    for (let i = 0; i < handResult.landmarks.length; i++) {
      const handedness = handResult.handednesses[i][0].categoryName; 
      if (handedness === 'Left') {
        currentLeftHand = handResult.landmarks[i];
      } else {
        currentRightHand = handResult.landmarks[i];
      }
    }
  }

  const alpha = 0.85; // Smoothing factor (higher = more responsive/less lag)
  const MAX_AGE = 5;
  
  // Smooth & Persist Left Hand
  if (currentLeftHand) {
    leftHandAge = 0;
    if (!lastLeftHandLandmarks) {
      lastLeftHandLandmarks = currentLeftHand.map(pt => ({ ...pt }));
    } else {
      for (let i = 0; i < 21; i++) {
        lastLeftHandLandmarks[i].x += (currentLeftHand[i].x - lastLeftHandLandmarks[i].x) * alpha;
        lastLeftHandLandmarks[i].y += (currentLeftHand[i].y - lastLeftHandLandmarks[i].y) * alpha;
        lastLeftHandLandmarks[i].z += (currentLeftHand[i].z - lastLeftHandLandmarks[i].z) * alpha;
        if (currentLeftHand[i].visibility !== undefined) {
          lastLeftHandLandmarks[i].visibility = currentLeftHand[i].visibility;
        }
      }
    }
  } else {
    leftHandAge++;
    if (leftHandAge > MAX_AGE) {
      lastLeftHandLandmarks = null;
    }
  }

  // Smooth & Persist Right Hand
  if (currentRightHand) {
    rightHandAge = 0;
    if (!lastRightHandLandmarks) {
      lastRightHandLandmarks = currentRightHand.map(pt => ({ ...pt }));
    } else {
      for (let i = 0; i < 21; i++) {
        lastRightHandLandmarks[i].x += (currentRightHand[i].x - lastRightHandLandmarks[i].x) * alpha;
        lastRightHandLandmarks[i].y += (currentRightHand[i].y - lastRightHandLandmarks[i].y) * alpha;
        lastRightHandLandmarks[i].z += (currentRightHand[i].z - lastRightHandLandmarks[i].z) * alpha;
        if (currentRightHand[i].visibility !== undefined) {
          lastRightHandLandmarks[i].visibility = currentRightHand[i].visibility;
        }
      }
    }
  } else {
    rightHandAge++;
    if (rightHandAge > MAX_AGE) {
      lastRightHandLandmarks = null;
    }
  }

  let currentPose = null;
  if (poseResult && poseResult.landmarks && poseResult.landmarks.length > 0) {
    currentPose = poseResult.landmarks[0];
  }

  // Smooth & Persist Pose
  if (currentPose) {
    poseAge = 0;
    if (!lastPoseLandmarks) {
      lastPoseLandmarks = currentPose.map(pt => ({ ...pt }));
    } else {
      for (let i = 0; i < lastPoseLandmarks.length; i++) {
        lastPoseLandmarks[i].x += (currentPose[i].x - lastPoseLandmarks[i].x) * alpha;
        lastPoseLandmarks[i].y += (currentPose[i].y - lastPoseLandmarks[i].y) * alpha;
        lastPoseLandmarks[i].z += (currentPose[i].z - lastPoseLandmarks[i].z) * alpha;
        if (currentPose[i].visibility !== undefined) {
          lastPoseLandmarks[i].visibility = currentPose[i].visibility;
        }
      }
    }
  } else {
    poseAge++;
    if (poseAge > MAX_AGE) {
      lastPoseLandmarks = null;
    }
  }

  const results = {
    image: videoElement,
    poseLandmarks: lastPoseLandmarks,
    leftHandLandmarks: lastLeftHandLandmarks,
    rightHandLandmarks: lastRightHandLandmarks
  };

  // Draw base video frame
  canvasCtx.drawImage(results.image, 0, 0, w, h);

  // Preset 2: Reduce exposure of background video by 75%
  if (options.activePreset === 2) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    canvasCtx.fillRect(0, 0, w, h);
  }



  // Preset 5: Full black background
  if (options.activePreset === 5) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 1.0)';
    canvasCtx.fillRect(0, 0, w, h);
  }

  // Draw Pose Skeleton if enabled (except in Preset 5 which has custom neon styling)
  if (options.showPose && results.poseLandmarks && options.activePreset !== 5) {
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: '#00e676',
      lineWidth: 2,
      checkVisibility: true
    });
    drawLandmarks(canvasCtx, results.poseLandmarks, {
      color: '#ff3d00',
      lineWidth: 1,
      radius: 3,
      checkVisibility: true
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
    const lh = results.leftHandLandmarks;
    const rh = results.rightHandLandmarks;

    canvasCtx.save();
    canvasCtx.globalCompositeOperation = 'lighter';

    const maxDiag = Math.sqrt(w * w + h * h);
    const tips = [4, 8, 12, 16, 20];

    for (const i of tips) {
      const pt1 = { x: lh[i].x * w, y: lh[i].y * h };
      for (const j of tips) {
        const pt2 = { x: rh[j].x * w, y: rh[j].y * h };

        const dx = pt1.x - pt2.x;
        const dy = pt1.y - pt2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Length reactive thickness and color (red when short, thinner/whiter when far)
        const ratio = Math.min(dist / (maxDiag * 0.65), 1.0);
        const glowWidth = Math.max(1.0, 8.0 - 6.0 * ratio);
        const coreWidth = Math.max(0.5, 2.0 - 1.2 * ratio);

        // 1. Thicker outer glow line (transitioning from red to white)
        canvasCtx.lineWidth = glowWidth;
        canvasCtx.strokeStyle = `hsla(0, ${100 * (1 - ratio)}%, ${50 + 50 * ratio}%, 0.45)`;
        canvasCtx.beginPath();
        canvasCtx.moveTo(pt1.x, pt1.y);
        canvasCtx.lineTo(pt2.x, pt2.y);
        canvasCtx.stroke();

        // 2. Thinner inner core line (transitioning from light red to white)
        canvasCtx.lineWidth = coreWidth;
        canvasCtx.strokeStyle = `hsla(0, ${100 * (1 - ratio)}%, ${75 + 25 * ratio}%, 0.85)`;
        canvasCtx.beginPath();
        canvasCtx.moveTo(pt1.x, pt1.y);
        canvasCtx.lineTo(pt2.x, pt2.y);
        canvasCtx.stroke();
      }
    }
    canvasCtx.restore();
  }

  // ===== PRESET 3: Finger Portal Filters =====
  if (options.activePreset === 3 && results.leftHandLandmarks && results.rightHandLandmarks) {
    const lh = results.leftHandLandmarks;
    const rh = results.rightHandLandmarks;

    // Define finger pairs: [leftIdx1, leftIdx2, rightIdx1, rightIdx2, filterName]
    const fingerPairs = [
      [4, 8, 4, 8, 'invert'],
      [8, 12, 8, 12, 'halftone'],
      [12, 16, 12, 16, 'duotone'],
      [16, 20, 16, 20, 'pixelate'],
    ];

    for (const [li1, li2, ri1, ri2, filterName] of fingerPairs) {
      // Copy current video frame to offscreen canvas for pixel sampling
      offCtx.drawImage(results.image, 0, 0, w, h);

      const pts = [
        { x: lh[li1].x * w, y: lh[li1].y * h },
        { x: lh[li2].x * w, y: lh[li2].y * h },
        { x: rh[ri2].x * w, y: rh[ri2].y * h },
        { x: rh[ri1].x * w, y: rh[ri1].y * h },
      ];

      const minX = Math.max(0, Math.floor(Math.min(pts[0].x, pts[1].x, pts[2].x, pts[3].x)));
      const minY = Math.max(0, Math.floor(Math.min(pts[0].y, pts[1].y, pts[2].y, pts[3].y)));
      const maxX = Math.min(w, Math.ceil(Math.max(pts[0].x, pts[1].x, pts[2].x, pts[3].x)));
      const maxY = Math.min(h, Math.ceil(Math.max(pts[0].y, pts[1].y, pts[2].y, pts[3].y)));
      const bw = maxX - minX;
      const bh = maxY - minY;

      if (bw < 2 || bh < 2) continue;

      const imgData = offCtx.getImageData(minX, minY, bw, bh);
      const data = imgData.data;

      if (filterName === 'invert') {
        for (let i = 0; i < data.length; i += 4) {
          data[i] = 255 - data[i];
          data[i + 1] = 255 - data[i + 1];
          data[i + 2] = 255 - data[i + 2];
        }
      } else if (filterName === 'halftone') {
        // Fill the bounding box region with solid black
        offCtx.fillStyle = '#000000';
        offCtx.fillRect(minX, minY, bw, bh);
        
        offCtx.fillStyle = '#ffffff';
        const grid = 6;
        const maxRadius = (grid / 2) * 1.2;
        
        for (let gy = Math.floor(minY + grid / 2); gy < minY + bh; gy += grid) {
          for (let gx = Math.floor(minX + grid / 2); gx < minX + bw; gx += grid) {
            const lx = Math.floor(gx - minX);
            const ly = Math.floor(gy - minY);
            if (lx >= 0 && lx < bw && ly >= 0 && ly < bh) {
              const idx = (ly * bw + lx) * 4;
              const r = data[idx];
              const g = data[idx + 1];
              const b = data[idx + 2];
              const lum = r * 0.299 + g * 0.587 + b * 0.114;
              
              const radius = (lum / 255) * maxRadius;
              if (radius > 0.5) {
                offCtx.beginPath();
                offCtx.arc(gx, gy, radius, 0, Math.PI * 2);
                offCtx.fill();
              }
            }
          }
        }
      } else if (filterName === 'duotone') {
        for (let i = 0; i < data.length; i += 4) {
          const lum = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          const t = lum / 255;
          data[i] = Math.round(255 * t);
          data[i + 1] = Math.round(255 * (1 - t));
          data[i + 2] = 255;
        }
      } else if (filterName === 'pixelate') {
        const blockSize = 8;
        for (let by = 0; by < bh; by += blockSize) {
          for (let bx = 0; bx < bw; bx += blockSize) {
            let rSum = 0, gSum = 0, bSum = 0, count = 0;
            for (let dy = 0; dy < blockSize && by + dy < bh; dy++) {
              for (let dx = 0; dx < blockSize && bx + dx < bw; dx++) {
                const idx = ((by + dy) * bw + (bx + dx)) * 4;
                rSum += data[idx]; gSum += data[idx + 1]; bSum += data[idx + 2]; count++;
              }
            }
            const rAvg = rSum / count, gAvg = gSum / count, bAvg = bSum / count;
            for (let dy = 0; dy < blockSize && by + dy < bh; dy++) {
              for (let dx = 0; dx < blockSize && bx + dx < bw; dx++) {
                const idx = ((by + dy) * bw + (bx + dx)) * 4;
                data[idx] = rAvg; data[idx + 1] = gAvg; data[idx + 2] = bAvg;
              }
            }
          }
        }
      }

      if (filterName !== 'halftone') {
        offCtx.putImageData(imgData, minX, minY);
      }

      canvasCtx.save();
      canvasCtx.beginPath();
      canvasCtx.moveTo(pts[0].x, pts[0].y);
      canvasCtx.lineTo(pts[1].x, pts[1].y);
      canvasCtx.lineTo(pts[2].x, pts[2].y);
      canvasCtx.lineTo(pts[3].x, pts[3].y);
      canvasCtx.closePath();
      canvasCtx.clip();
      canvasCtx.drawImage(offscreenCanvas, 0, 0, w, h);
      canvasCtx.restore();

      if (options.showOutline) {
        canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        canvasCtx.lineWidth = 2;
        canvasCtx.beginPath();
        canvasCtx.moveTo(pts[0].x, pts[0].y);
        canvasCtx.lineTo(pts[1].x, pts[1].y);
        canvasCtx.lineTo(pts[2].x, pts[2].y);
        canvasCtx.lineTo(pts[3].x, pts[3].y);
        canvasCtx.closePath();
        canvasCtx.stroke();
      }
    }
  }

  // ===== PRESET 4: Ghost Trails =====
  if (options.activePreset === 4 && results.leftHandLandmarks) {
    canvasCtx.save();
    canvasCtx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 21; i++) {
      const lm = results.leftHandLandmarks[i];
      const px = lm.x * w, py = lm.y * h;
      trailCtx.fillStyle = 'rgba(0, 255, 255, 0.6)';
      trailCtx.beginPath(); trailCtx.arc(px, py, 5, 0, Math.PI * 2); trailCtx.fill();
      canvasCtx.fillStyle = 'rgba(0, 255, 255, 0.9)';
      canvasCtx.beginPath(); canvasCtx.arc(px, py, 6, 0, Math.PI * 2); canvasCtx.fill();
    }
    canvasCtx.restore();
  }
  if (options.activePreset === 4 && results.rightHandLandmarks) {
    canvasCtx.save();
    canvasCtx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 21; i++) {
      const lm = results.rightHandLandmarks[i];
      const px = lm.x * w, py = lm.y * h;
      trailCtx.fillStyle = 'rgba(255, 0, 128, 0.6)';
      trailCtx.beginPath(); trailCtx.arc(px, py, 5, 0, Math.PI * 2); trailCtx.fill();
      canvasCtx.fillStyle = 'rgba(255, 0, 128, 0.9)';
      canvasCtx.beginPath(); canvasCtx.arc(px, py, 6, 0, Math.PI * 2); canvasCtx.fill();
    }
    canvasCtx.restore();
  }
  if (options.activePreset === 4) {
    canvasCtx.save();
    canvasCtx.globalCompositeOperation = 'lighter';
    canvasCtx.globalAlpha = 0.7;
    canvasCtx.drawImage(trailCanvas, 0, 0, w, h);
    canvasCtx.restore();
  }

  // ===== PRESET 5: Void Skeleton =====
  if (options.activePreset === 5) {
    const elapsed = performance.now() / 1000;
    const pulseRadius = 4 + 3 * Math.sin(elapsed * 4);
    canvasCtx.save();
    canvasCtx.globalCompositeOperation = 'lighter';

    // Draw custom Neon-Green Pose Skeleton if pose landmarks are detected
    if (results.poseLandmarks) {
      const pl = results.poseLandmarks;
      // Outer glow (neon green)
      canvasCtx.strokeStyle = 'rgba(57, 255, 20, 0.25)'; canvasCtx.lineWidth = 8;
      for (const [a, b] of POSE_CONNECTIONS) {
        if (!pl[a] || !pl[b]) continue;
        if (pl[a].visibility !== undefined && pl[a].visibility < 0.5) continue;
        if (pl[b].visibility !== undefined && pl[b].visibility < 0.5) continue;
        canvasCtx.beginPath();
        canvasCtx.moveTo(pl[a].x * w, pl[a].y * h);
        canvasCtx.lineTo(pl[b].x * w, pl[b].y * h);
        canvasCtx.stroke();
      }
      // Inner line (neon green)
      canvasCtx.strokeStyle = 'rgba(57, 255, 20, 0.9)'; canvasCtx.lineWidth = 3;
      for (const [a, b] of POSE_CONNECTIONS) {
        if (!pl[a] || !pl[b]) continue;
        if (pl[a].visibility !== undefined && pl[a].visibility < 0.5) continue;
        if (pl[b].visibility !== undefined && pl[b].visibility < 0.5) continue;
        canvasCtx.beginPath();
        canvasCtx.moveTo(pl[a].x * w, pl[a].y * h);
        canvasCtx.lineTo(pl[b].x * w, pl[b].y * h);
        canvasCtx.stroke();
      }
      // Joints (pulsing key points, smaller normal points)
      const poseJoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
      for (let i = 0; i < pl.length; i++) {
        if (!pl[i] || (pl[i].visibility !== undefined && pl[i].visibility < 0.5)) continue;
        const r = poseJoints.includes(i) ? pulseRadius : 3;
        const alpha = poseJoints.includes(i) ? 1.0 : 0.7;
        canvasCtx.fillStyle = `rgba(57, 255, 20, ${alpha})`;
        canvasCtx.beginPath(); canvasCtx.arc(pl[i].x * w, pl[i].y * h, r, 0, Math.PI * 2); canvasCtx.fill();
      }
    }

    const handConns = [
      [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
    ];
    const fingertips = [4, 8, 12, 16, 20];

    if (results.leftHandLandmarks) {
      const lh = results.leftHandLandmarks;
      canvasCtx.strokeStyle = 'rgba(0, 255, 255, 0.25)'; canvasCtx.lineWidth = 8;
      for (const [a, b] of handConns) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(lh[a].x * w, lh[a].y * h);
        canvasCtx.lineTo(lh[b].x * w, lh[b].y * h);
        canvasCtx.stroke();
      }
      canvasCtx.strokeStyle = 'rgba(0, 255, 255, 0.9)'; canvasCtx.lineWidth = 3;
      for (const [a, b] of handConns) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(lh[a].x * w, lh[a].y * h);
        canvasCtx.lineTo(lh[b].x * w, lh[b].y * h);
        canvasCtx.stroke();
      }
      for (let i = 0; i < 21; i++) {
        const r = fingertips.includes(i) ? pulseRadius : 3;
        const alpha = fingertips.includes(i) ? 1.0 : 0.7;
        canvasCtx.fillStyle = `rgba(0, 255, 255, ${alpha})`;
        canvasCtx.beginPath(); canvasCtx.arc(lh[i].x * w, lh[i].y * h, r, 0, Math.PI * 2); canvasCtx.fill();
      }
    }

    if (results.rightHandLandmarks) {
      const rh = results.rightHandLandmarks;
      canvasCtx.strokeStyle = 'rgba(255, 0, 200, 0.25)'; canvasCtx.lineWidth = 8;
      for (const [a, b] of handConns) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(rh[a].x * w, rh[a].y * h);
        canvasCtx.lineTo(rh[b].x * w, rh[b].y * h);
        canvasCtx.stroke();
      }
      canvasCtx.strokeStyle = 'rgba(255, 0, 200, 0.9)'; canvasCtx.lineWidth = 3;
      for (const [a, b] of handConns) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(rh[a].x * w, rh[a].y * h);
        canvasCtx.lineTo(rh[b].x * w, rh[b].y * h);
        canvasCtx.stroke();
      }
      for (let i = 0; i < 21; i++) {
        const r = fingertips.includes(i) ? pulseRadius : 3;
        const alpha = fingertips.includes(i) ? 1.0 : 0.7;
        canvasCtx.fillStyle = `rgba(255, 0, 200, ${alpha})`;
        canvasCtx.beginPath(); canvasCtx.arc(rh[i].x * w, rh[i].y * h, r, 0, Math.PI * 2); canvasCtx.fill();
      }
    }

    canvasCtx.restore();
  }

  // ===== PRESET 6: YouTube/Video Corner-pinned Overlay =====
  if (options.activePreset === 6) {
    const lh = results.leftHandLandmarks;
    const rh = results.rightHandLandmarks;
    
    const rect = canvasElement.getBoundingClientRect();
    preset6OverlayContainer.style.left = rect.left + 'px';
    preset6OverlayContainer.style.top = rect.top + 'px';
    preset6OverlayContainer.style.width = rect.width + 'px';
    preset6OverlayContainer.style.height = rect.height + 'px';
    
    const playerEl = document.getElementById('preset6-overlay-video') || document.getElementById('preset6-overlay-image');
    if (playerEl) {
      let w_o = 640;
      let h_o = 360;
      if (playerEl.tagName === 'VIDEO') {
        if (playerEl.videoWidth) {
          w_o = playerEl.videoWidth;
          h_o = playerEl.videoHeight;
        }
      } else if (playerEl.tagName === 'IMG') {
        if (playerEl.naturalWidth) {
          w_o = playerEl.naturalWidth;
          h_o = playerEl.naturalHeight;
        }
      }
      playerEl.style.width = w_o + 'px';
      playerEl.style.height = h_o + 'px';

      const dt = lastFrameTime ? (nowMs - lastFrameTime) / 1000 : 0.04;
      lastFrameTime = nowMs;

      // Extract active hands pinch data
      const activeHands = [];
      if (lh) {
        const p = { x: (lh[8].x + lh[4].x)/2, y: (lh[8].y + lh[4].y)/2 };
        if (options.flipH) p.x = 1.0 - p.x;
        if (options.flipV) p.y = 1.0 - p.y;
        const dx = (lh[8].x - lh[4].x) * canvasElement.width;
        const dy = (lh[8].y - lh[4].y) * canvasElement.height;
        const dist = Math.sqrt(dx*dx + dy*dy);
        activeHands.push({ side: 'left', pinch: p, isPinching: dist < 45 });
      }
      if (rh) {
        const p = { x: (rh[8].x + rh[4].x)/2, y: (rh[8].y + rh[4].y)/2 };
        if (options.flipH) p.x = 1.0 - p.x;
        if (options.flipV) p.y = 1.0 - p.y;
        const dx = (rh[8].x - rh[4].x) * canvasElement.width;
        const dy = (rh[8].y - rh[4].y) * canvasElement.height;
        const dist = Math.sqrt(dx*dx + dy*dy);
        activeHands.push({ side: 'right', pinch: p, isPinching: dist < 45 });
      }

      let pt1, pt2, pt3, pt4;
      let hasValidWarpCoords = false;

      if (preset6TrackingMode === '3d') {
        // Dragging & selector updates
        if (selectedDotIndex === -1) {
          for (let i = 0; i < 4; i++) {
            let closeHand = null;
            for (const h of activeHands) {
              const dx = (h.pinch.x - dotCoords[i].x) * canvasElement.width;
              const dy = (h.pinch.y - dotCoords[i].y) * canvasElement.height;
              if (Math.sqrt(dx*dx + dy*dy) < 35) {
                closeHand = h;
                break;
              }
            }
            if (closeHand) {
              dotLoading[i] = Math.min(1.0, dotLoading[i] + dt / 0.5);
              if (dotLoading[i] >= 1.0) {
                selectedDotIndex = i;
                selectedHand = closeHand.side;
                dotLoading[i] = 0;
              }
            } else {
              dotLoading[i] = Math.max(0, dotLoading[i] - dt / 0.5);
            }
          }
        } else {
          // Drag selected dot
          const h = activeHands.find(hand => hand.side === selectedHand);
          if (h && h.isPinching) {
            dotCoords[selectedDotIndex] = { x: h.pinch.x, y: h.pinch.y };
          } else {
            selectedDotIndex = -1;
            selectedHand = null;
          }
          // Clear other loading states
          for (let i = 0; i < 4; i++) {
            if (i !== selectedDotIndex) dotLoading[i] = 0;
          }
        }

        pt2 = getScreenCoords(dotCoords[0], rect); // TL
        pt3 = getScreenCoords(dotCoords[1], rect); // TR
        pt4 = getScreenCoords(dotCoords[2], rect); // BR
        pt1 = getScreenCoords(dotCoords[3], rect); // BL
        hasValidWarpCoords = true;
      } else if (lh && rh) {
        const pt1_screen = getScreenCoords(lh[0], rect);
        const pt2_screen = getScreenCoords(rh[0], rect);
        
        let vL, vR;
        if (pt1_screen.x < pt2_screen.x) {
          vL = lh;
          vR = rh;
        } else {
          vL = rh;
          vR = lh;
        }

        if (preset6TrackingMode === 'hand') {
          let lowestL = vL[0];
          let lowestR = vR[0];
          for (let i = 1; i < 21; i++) {
            if (vL[i].y > lowestL.y) lowestL = vL[i];
            if (vR[i].y > lowestR.y) lowestR = vR[i];
          }
          pt1 = getScreenCoords(lowestL, rect);
          pt2 = getScreenCoords(vL[12], rect);
          pt3 = getScreenCoords(vR[12], rect);
          pt4 = getScreenCoords(lowestR, rect);
          hasValidWarpCoords = true;
        } else if (preset6TrackingMode === 'fingers') {
          pt1 = getScreenCoords(vL[4], rect);
          pt2 = getScreenCoords(vL[8], rect);
          pt3 = getScreenCoords(vR[8], rect);
          pt4 = getScreenCoords(vR[4], rect);
          hasValidWarpCoords = true;
        } else { // 'pinch'
          const ptL = getScreenCoords({ x: (vL[8].x + vL[4].x)/2, y: (vL[8].y + vL[4].y)/2 }, rect);
          const ptR = getScreenCoords({ x: (vR[8].x + vR[4].x)/2, y: (vR[8].y + vR[4].y)/2 }, rect);
          const dx = ptR.x - ptL.x;
          const dy = ptR.y - ptL.y;
          const D = Math.sqrt(dx*dx + dy*dy) || 1;
          const ux = dx / D;
          const uy = dy / D;
          const vx = -uy;
          const vy = ux;
          const H = D / (w_o / h_o);

          pt2 = { x: ptL.x - (H/2) * vx, y: ptL.y - (H/2) * vy };
          pt3 = { x: ptR.x - (H/2) * vx, y: ptR.y - (H/2) * vy };
          pt4 = { x: ptR.x + (H/2) * vx, y: ptR.y + (H/2) * vy };
          pt1 = { x: ptL.x + (H/2) * vx, y: ptL.y + (H/2) * vy };
          hasValidWarpCoords = true;
        }
      }

      if (hasValidWarpCoords) {
        const x0 = overlayFlipH ? w_o : 0;
        const x1 = overlayFlipH ? 0 : w_o;
        const y0 = overlayFlipV ? h_o : 0;
        const y1 = overlayFlipV ? 0 : h_o;
        
        const src = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        const dst = [[pt2.x, pt2.y], [pt3.x, pt3.y], [pt4.x, pt4.y], [pt1.x, pt1.y]];
        
        if (isConvex(dst)) {
          try {
            const matrix = getHomographyMatrix(src, dst);
            playerEl.style.transform = `matrix3d(${matrix.join(',')})`;
          } catch (err) {
            console.error('Error computing homography matrix:', err);
          }
        }
      }

      // Render dots and loading indicators on canvas (visible when hand is near)
      if (preset6TrackingMode === '3d') {
        for (let i = 0; i < 4; i++) {
          let isNear = (selectedDotIndex === i);
          for (const h of activeHands) {
            const dx = (h.pinch.x - dotCoords[i].x) * canvasElement.width;
            const dy = (h.pinch.y - dotCoords[i].y) * canvasElement.height;
            if (Math.sqrt(dx*dx + dy*dy) < 100) {
              isNear = true;
              break;
            }
          }
          if (isNear) {
            const cx = dotCoords[i].x * canvasElement.width;
            const cy = dotCoords[i].y * canvasElement.height;
            const r = (selectedDotIndex === i) ? 12 : 6;
            
            // Draw outer green glow circle
            canvasCtx.fillStyle = 'rgba(0, 230, 118, 0.3)';
            canvasCtx.beginPath();
            canvasCtx.arc(cx, cy, r + 4, 0, 2 * Math.PI);
            canvasCtx.fill();
            
            // Draw solid inner dot
            canvasCtx.fillStyle = (selectedDotIndex === i) ? '#00e676' : '#ffffff';
            canvasCtx.beginPath();
            canvasCtx.arc(cx, cy, r, 0, 2 * Math.PI);
            canvasCtx.fill();
            
            // Draw circular loading indicator
            if (dotLoading[i] > 0) {
              canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
              canvasCtx.lineWidth = 3;
              canvasCtx.beginPath();
              canvasCtx.arc(cx, cy, r + 8, 0, 2 * Math.PI);
              canvasCtx.stroke();
              
              canvasCtx.strokeStyle = '#00e676';
              canvasCtx.lineWidth = 3;
              canvasCtx.beginPath();
              canvasCtx.arc(cx, cy, r + 8, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * dotLoading[i]);
              canvasCtx.stroke();
            }
          }
        }
      }
    }
  }

  // Draw face-tracked glasses if enabled
  if (options.showGlasses && results.poseLandmarks) {
    const pl = results.poseLandmarks;
    if (pl[2] && pl[5]) {
      const leX = pl[2].x * w;
      const leY = pl[2].y * h;
      const reX = pl[5].x * w;
      const reY = pl[5].y * h;

      const midX = (leX + reX) / 2;
      const midY = (leY + reY) / 2;

      const dx = reX - leX;
      const dy = reY - leY;
      const eyeDist = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx);

      if (glassesImg.complete && glassesImg.naturalWidth > 0) {
        canvasCtx.save();
        canvasCtx.translate(midX, midY);
        canvasCtx.rotate(angle);
        const glassesWidth = eyeDist * 2.2;
        const glassesHeight = glassesWidth * (glassesImg.naturalHeight / glassesImg.naturalWidth);
        canvasCtx.drawImage(glassesImg, -glassesWidth / 2, -glassesHeight / 2, glassesWidth, glassesHeight);
        canvasCtx.restore();
      }
    }
  }

  canvasCtx.restore(); // Restore flip scale/translate, returning context to clean screen-space

  // 3. Draw HUD Skeletons Status (Exactly replicating CV2 putText colors & positions)
  canvasCtx.font = 'bold 20px monospace';
  
  // Draw FPS Status
  canvasCtx.fillStyle = 'rgb(0, 255, 255)';
  canvasCtx.fillText(`FPS: ${fps}`, 20, 40);

  reqFrameId = requestAnimationFrame(renderLoop);
}

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
  } else if (key === '3') {
    options.activePreset = 3;
    updatePresetHighlights();
    console.log('Preset 3 activated');
  } else if (key === '4') {
    options.activePreset = 4;
    updatePresetHighlights();
    console.log('Preset 4 activated');
  } else if (key === '5') {
    options.activePreset = 5;
    updatePresetHighlights();
    console.log('Preset 5 activated');
  } else if (key === '6') {
    options.activePreset = 6;
    updatePresetHighlights();
    console.log('Preset 6 activated');
  } else if (key === 'o' && options.activePreset === 3) {
    options.showOutline = !options.showOutline;
    updatePresetHighlights();
    console.log(`Outline: ${options.showOutline ? 'ON' : 'OFF'}`);
  } else if (key === 'g') {
    options.showGlasses = !options.showGlasses;
    updateButtonHighlights();
    console.log(`Glasses: ${options.showGlasses ? 'ON' : 'OFF'}`);
  } else if (options.activePreset === 6 && (event.key === ' ' || event.key === 'ArrowRight' || event.key === 'ArrowLeft')) {
    event.preventDefault(); // Prevent page scrolling
    const previewVideo = document.getElementById('preset6-preview-video');
    const overlayVideo = document.getElementById('preset6-overlay-video');
    if (previewVideo && overlayVideo) {
      if (event.key === ' ') {
        if (previewVideo.paused) {
          previewVideo.play();
          overlayVideo.play();
          console.log('Video play');
        } else {
          previewVideo.pause();
          overlayVideo.pause();
          console.log('Video pause');
        }
      } else if (event.key === 'ArrowRight') {
        previewVideo.currentTime = Math.min(previewVideo.duration || 0, previewVideo.currentTime + 5);
        overlayVideo.currentTime = previewVideo.currentTime;
        console.log('Video skip forward 5s');
      } else if (event.key === 'ArrowLeft') {
        previewVideo.currentTime = Math.max(0, previewVideo.currentTime - 5);
        overlayVideo.currentTime = previewVideo.currentTime;
        console.log('Video skip backward 5s');
      }
    }
  }
});

// Start camera stream & initialization
async function startTracking() {
  console.log('start clicked');

  if (typeof FilesetResolver === 'undefined') {
    console.log('FilesetResolver is undefined, attempting dynamic import...');
    try {
      const module = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs");
      FilesetResolver = module.FilesetResolver;
      HandLandmarker = module.HandLandmarker;
      PoseLandmarker = module.PoseLandmarker;
      console.log('Successfully loaded FilesetResolver dynamically.');
    } catch (e) {
      console.error('MediaPipe tasks-vision bundle did not load.', e);
      return;
    }
  }

  try {
    welcomeScreen.style.display = 'none';

    loadingContainer.style.display = 'block';
    loadingText.textContent = 'Loading model...';
    loadingAnimation = progressBarFill.animate(
      [{ width: '30%' }, { width: '70%' }],
      { duration: 800, direction: 'alternate', iterations: Infinity, easing: 'ease-in-out' }
    );

    console.log('1. FilesetResolver loading...');
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    console.log('2. FilesetResolver ready');

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    });
    console.log('3. HandLandmarker ready');

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1
    });
    console.log('4. PoseLandmarker ready');

    // Hide loading container and clear pulse animation
    if (loadingAnimation) {
      loadingAnimation.cancel();
    }
    loadingContainer.style.display = 'none';
    console.log('5. Loading hidden');

    activeStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 1280,
        height: 720,
        facingMode: 'user'
      },
      audio: false
    });
    console.log('6. Camera stream granted');

    videoElement.srcObject = activeStream;

    await new Promise(r => videoElement.addEventListener('loadedmetadata', r, { once: true }));
    console.log('7. Video metadata loaded, dimensions:', videoElement.videoWidth, videoElement.videoHeight);

    await videoElement.play();
    console.log('8. Video playing');

    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    offscreenCanvas.width = videoElement.videoWidth;
    offscreenCanvas.height = videoElement.videoHeight;
    trailCanvas.width = videoElement.videoWidth;
    trailCanvas.height = videoElement.videoHeight;

    canvasElement.style.display = 'block';
    controlBar.style.display = 'flex';
    presetSelector.style.display = 'flex';
    universalPanel.style.display = 'flex';
    updateButtonHighlights();
    updatePresetHighlights();
    console.log('9. Canvas sized:', canvasElement.width, canvasElement.height);

    lastHandTimestamp = -1;
    lastPoseTimestamp = -1;

    reqFrameId = requestAnimationFrame(renderLoop);
    console.log('10. Loop started');

  } catch (error) {
    console.error("Camera access or init error:", error);
    if (loadingAnimation) {
      loadingAnimation.cancel();
    }
    loadingContainer.style.display = 'none';
    welcomeScreen.style.display = 'block';

    alert(`Failed to start tracking: ${error.message}`);
  }
}

// Stop tracking & cleanup
function stopTracking() {
  lastLeftHandLandmarks = null;
  lastRightHandLandmarks = null;
  lastPoseLandmarks = null;
  lastRenderTime = 0;

  if (reqFrameId) {
    cancelAnimationFrame(reqFrameId);
    reqFrameId = null;
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
  universalPanel.style.display = 'none';
  btnOutline.style.display = 'none';
  preset6Panel.style.display = 'none';
  preset6OverlayContainer.style.display = 'none';
  pausePlayers();
  if (preset6PreviewContainer) {
    preset6PreviewContainer.innerHTML = `
      <div style="color: #666; font-size: 12px; height: 100%; display: flex; align-items: center; justify-content: center;">No file selected</div>
    `;
  }
  if (preset6OverlayContainer) {
    preset6OverlayContainer.innerHTML = '';
  }
  welcomeScreen.style.display = 'block';


  if (handLandmarker) {
    handLandmarker.close();
    handLandmarker = null;
  }
  if (poseLandmarker) {
    poseLandmarker.close();
    poseLandmarker = null;
  }

  lastFrameTime = 0;
  fps = 0;
}


