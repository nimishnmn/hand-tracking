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
  
  ctx.save();
  ctx.fillStyle = color;
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (!lm || (lm.visibility !== undefined && lm.visibility < 0.5)) continue;
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

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < connections.length; i++) {
    const [startIdx, endIdx] = connections[i];
    const startLm = landmarks[startIdx];
    const endLm = landmarks[endIdx];
    if (!startLm || (startLm.visibility !== undefined && startLm.visibility < 0.5)) continue;
    if (!endLm || (endLm.visibility !== undefined && endLm.visibility < 0.5)) continue;
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
let loadingAnimation = null;
let lastHandTimestamp = -1;
let lastPoseTimestamp = -1;
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
  btnPreset3.classList.toggle('active', options.activePreset === 3);
  btnPreset4.classList.toggle('active', options.activePreset === 4);
  btnPreset5.classList.toggle('active', options.activePreset === 5);
}

// Initialize DOM and Event Listeners after document loads
document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  videoElement = document.getElementById('webcam');
  canvasElement = document.getElementById('output-canvas');
  canvasCtx = canvasElement.getContext('2d');
  startCameraBtn = document.getElementById('start-camera-btn');
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
    
    if (options.showPose) {
      let pts = performance.now();
      if (pts <= lastPoseTimestamp) pts = lastPoseTimestamp + 1;
      lastPoseTimestamp = pts;
      poseResult = poseLandmarker.detectForVideo(videoElement, pts);
    }
  }

  const results = {
    image: videoElement,
    poseLandmarks: null,
    leftHandLandmarks: null,
    rightHandLandmarks: null
  };
  
  if (poseResult && poseResult.landmarks && poseResult.landmarks.length > 0) {
    results.poseLandmarks = poseResult.landmarks[0];
  }
  
  if (handResult && handResult.landmarks && handResult.landmarks.length > 0) {
    for (let i = 0; i < handResult.landmarks.length; i++) {
      const handedness = handResult.handednesses[i][0].categoryName; 
      if (handedness === 'Left') {
        results.leftHandLandmarks = handResult.landmarks[i];
      } else {
        results.rightHandLandmarks = handResult.landmarks[i];
      }
    }
  }

  // Draw base video frame
  canvasCtx.drawImage(results.image, 0, 0, w, h);

  // Preset 2: Reduce exposure of background video by 75%
  if (options.activePreset === 2) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    canvasCtx.fillRect(0, 0, w, h);
  }

  // Preset 3: Dim background by 40%
  if (options.activePreset === 3) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.40)';
    canvasCtx.fillRect(0, 0, w, h);
  }

  // Preset 4: Dim background by 85%
  if (options.activePreset === 4) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    canvasCtx.fillRect(0, 0, w, h);
  }

  // Preset 5: Full black background
  if (options.activePreset === 5) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 1.0)';
    canvasCtx.fillRect(0, 0, w, h);
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

  // ===== PRESET 3: Finger Portal Filters =====
  if (options.activePreset === 3 && results.leftHandLandmarks && results.rightHandLandmarks) {
    const lh = results.leftHandLandmarks;
    const rh = results.rightHandLandmarks;

    // Define finger pairs: [leftIdx1, leftIdx2, rightIdx1, rightIdx2, filterName]
    const fingerPairs = [
      [4, 8, 4, 8, 'invert'],
      [8, 12, 8, 12, 'grayscale'],
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
      } else if (filterName === 'grayscale') {
        for (let i = 0; i < data.length; i += 4) {
          const avg = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          data[i] = data[i + 1] = data[i + 2] = avg;
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

      offCtx.putImageData(imgData, minX, minY);

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

  canvasCtx.restore(); // Restore flip scale/translate, returning context to clean screen-space

  // 3. Draw HUD Skeletons Status (Exactly replicating CV2 putText colors & positions)
  canvasCtx.font = 'bold 20px monospace';
  
  // Draw ESC = Quit status
  canvasCtx.fillStyle = 'rgb(0, 255, 255)';
  canvasCtx.fillText('ESC = Quit', 20, 40);

  // Draw FPS Status
  canvasCtx.fillText(`FPS: ${fps}`, 20, 75);

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
  }
});

// Start camera stream & initialization
async function startTracking() {
  console.log('start clicked');

  if (typeof FilesetResolver === 'undefined') {
    console.log('FilesetResolver is undefined, attempting dynamic import...');
    try {
      const module = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js");
      FilesetResolver = module.FilesetResolver;
      HandLandmarker = module.HandLandmarker;
      PoseLandmarker = module.PoseLandmarker;
      console.log('Successfully loaded FilesetResolver dynamically.');
    } catch (e) {
      console.error('MediaPipe tasks-vision bundle did not load. Check the script tag URL and order in index.html.', e);
      return;
    }
  }

  try {
    startCameraBtn.style.display = 'none';
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
    startCameraBtn.style.display = 'block';
    alert(`Failed to start tracking: ${error.message}`);
  }
}

// Stop tracking & cleanup
function stopTracking() {
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
  startCameraBtn.style.display = 'block';

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
