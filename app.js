import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { alpha: true, desynchronized: true });

const cameraSelect = document.getElementById("cameraSelect");
const settingsCameraSelect = document.getElementById("settingsCameraSelect");
const startButton = document.getElementById("startButton");
const fullscreenButton = document.getElementById("fullscreenButton");
const displayMode = document.getElementById("displayMode");
const backgroundColor = document.getElementById("backgroundColor");
const colorValue = document.getElementById("colorValue");
const applyButton = document.getElementById("applyButton");
const settingsPanel = document.getElementById("settingsPanel");
const startStatus = document.getElementById("startStatus");
const errorOverlay = document.getElementById("errorOverlay");
const errorMessage = document.getElementById("errorMessage");
const errorCloseButton = document.getElementById("errorCloseButton");

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

let handLandmarker = null;
let currentStream = null;
let currentDeviceId = "";
let tracking = false;
let animationFrameId = null;
let lastVideoTime = -1;
let handsResult = null;
let mpReady = false;
let mpLoading = false;
let currentDisplayMode = "background";
let currentBackgroundColor = "#000000";
let longPressTimer = null;
let touchStartY = 0;
let touchStartX = 0;
let touchActive = false;

function setStatus(message) {
  startStatus.textContent = message || "";
}

function showError(message) {
  errorMessage.textContent = message;
  errorOverlay.classList.remove("hidden");
}

function hideError() {
  errorOverlay.classList.add("hidden");
}

function getCameraLabel(device, index) {
  const label = device.label?.trim() || `Camera ${index + 1}`;
  const id = device.deviceId || "no-device-id";
  const lower = label.toLowerCase();
  let direction = "";

  if (lower.includes("front") || lower.includes("user") || lower.includes("selfie")) {
    direction = " · FRONT";
  } else if (lower.includes("back") || lower.includes("rear") || lower.includes("environment")) {
    direction = " · BACK";
  } else if (lower.includes("external") || lower.includes("usb")) {
    direction = " · EXTERNAL";
  }

  return `${label}${direction} · ID: ${id}`;
}

async function requestInitialPermission() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera API unavailable. Use recent Android Chrome over HTTPS.");
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    stream.getTracks().forEach(track => track.stop());
  } catch (error) {
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      throw new Error("Camera permission was denied. Allow camera access for this site in Chrome settings, then reload.");
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      throw new Error("No camera was found on this device.");
    }
    throw new Error(`Unable to access the camera: ${error.message || error.name}`);
  }
}

async function enumerateCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    throw new Error("Camera enumeration is not supported by this browser.");
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter(d => d.kind === "videoinput");
}

function populateCameraSelectors(cameras, preferredId = "") {
  for (const select of [cameraSelect, settingsCameraSelect]) {
    select.innerHTML = "";

    if (!cameras.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No camera available";
      select.appendChild(option);
      continue;
    }

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId || "";
      option.textContent = getCameraLabel(camera, index);
      select.appendChild(option);
    });

    if (cameras.some(c => c.deviceId === preferredId)) {
      select.value = preferredId;
    }
  }
}

async function refreshCameraList(preferredId = "") {
  const cameras = await enumerateCameras();

  if (!cameras.length) {
    populateCameraSelectors([]);
    throw new Error("No camera was detected. Connect or enable a camera and try again.");
  }

  populateCameraSelectors(cameras, preferredId || currentDeviceId);
  return cameras;
}

async function loadHandLandmarker() {
  if (mpReady && handLandmarker) return;

  if (mpLoading) {
    while (mpLoading) await new Promise(resolve => setTimeout(resolve, 50));
    if (handLandmarker) return;
  }

  mpLoading = true;

  try {
    setStatus("Loading hand tracking...");
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    let loaded = false;

    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      loaded = true;
    } catch (gpuError) {
      console.warn("GPU delegate failed. Falling back to CPU.", gpuError);
      handLandmarker = null;
    }

    if (!loaded) {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      loaded = true;
    }

    mpReady = loaded;
    setStatus("");
  } catch (error) {
    mpReady = false;
    handLandmarker = null;
    throw new Error(
      "MediaPipe Hand Landmarker could not load. Check your internet connection and reload."
    );
  } finally {
    mpLoading = false;
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
  video.pause();
  video.srcObject = null;
}

async function startCamera(deviceId = "") {
  stopCamera();

  const baseVideo = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30, max: 30 }
  };

  let constraints = {
    audio: false,
    video: deviceId
      ? { ...baseVideo, deviceId: { exact: deviceId } }
      : baseVideo
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    if (deviceId && (error.name === "OverconstrainedError" || error.name === "NotFoundError")) {
      currentStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: baseVideo
      });
    } else {
      throw error;
    }
  }

  const track = currentStream.getVideoTracks()[0];
  if (!track) {
    stopCamera();
    throw new Error("The selected camera did not provide a video track.");
  }

  const settings = track.getSettings();
  currentDeviceId = settings.deviceId || deviceId || "";
  video.srcObject = currentStream;

  await new Promise((resolve, reject) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };

    video.onloadedmetadata = finish;
    setTimeout(finish, 3000);
    setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error("Camera video did not become ready."));
      }
    }, 10000);
  });

  await video.play();
  await refreshCameraList(currentDeviceId);
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(window.innerWidth * dpr));
  const height = Math.max(1, Math.floor(window.innerHeight * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getVideoCoverTransform() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const cw = canvas.width;
  const ch = canvas.height;

  if (!vw || !vh) return { scale: 1, offsetX: 0, offsetY: 0 };

  const scale = Math.max(cw / vw, ch / vh);
  return {
    scale,
    offsetX: (cw - vw * scale) / 2,
    offsetY: (ch - vh * scale) / 2
  };
}

function isFrontCamera() {
  const track = currentStream?.getVideoTracks()?.[0];
  const settings = track?.getSettings?.() || {};
  const label = track?.label?.toLowerCase() || "";

  return settings.facingMode === "user" ||
    label.includes("front") ||
    label.includes("selfie");
}

function drawCamera() {
  if (currentDisplayMode !== "camera" || video.readyState < 2) return;

  const transform = getVideoCoverTransform();
  const front = isFrontCamera();

  ctx.save();

  if (front) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }

  const x = front ? -transform.offsetX : transform.offsetX;

  ctx.drawImage(
    video,
    x,
    transform.offsetY,
    video.videoWidth * transform.scale,
    video.videoHeight * transform.scale
  );

  ctx.restore();
}

function drawHand(landmarks) {
  if (!landmarks || landmarks.length !== 21) return;

  const transform = getVideoCoverTransform();
  const front = isFrontCamera();

  const points = landmarks.map(point => {
    let x = point.x * video.videoWidth;
    const y = point.y * video.videoHeight;

    if (front && currentDisplayMode === "camera") {
      x = video.videoWidth - x;
    }

    return {
      x: x * transform.scale + transform.offsetX,
      y: y * transform.scale + transform.offsetY
    };
  });

  ctx.save();
  ctx.lineWidth = Math.max(2, canvas.width / 700);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#ffffff";

  for (const [a, b] of CONNECTIONS) {
    const p1 = points[a];
    const p2 = points[b];
    if (!p1 || !p2) continue;

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  ctx.fillStyle = "#ffffff";
  const radius = Math.max(4, canvas.width / 250);

  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function renderFrame() {
  if (!tracking) {
    animationFrameId = null;
    return;
  }

  resizeCanvas();
  clearCanvas();

  if (currentDisplayMode === "background") {
    ctx.fillStyle = currentBackgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawCamera();
  }

  if (
    handLandmarker &&
    video.readyState >= 2 &&
    video.currentTime !== lastVideoTime
  ) {
    lastVideoTime = video.currentTime;

    try {
      handsResult = handLandmarker.detectForVideo(video, performance.now());
    } catch (error) {
      console.warn("Hand detection frame failed:", error);
    }
  }

  if (handsResult?.landmarks) {
    for (const hand of handsResult.landmarks) {
      drawHand(hand);
    }
  }

  animationFrameId = requestAnimationFrame(renderFrame);
}

function startRenderLoop() {
  if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
  lastVideoTime = -1;
  handsResult = null;
  animationFrameId = requestAnimationFrame(renderFrame);
}

async function requestFullscreen() {
  try {
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      await document.documentElement.requestFullscreen();
    }

    try {
      if (screen.orientation?.lock) {
        await screen.orientation.lock("landscape");
      }
    } catch (_) {}
  } catch (error) {
    console.warn("Fullscreen unavailable:", error);
  }
}

function enterTrackingMode() {
  tracking = true;
  document.body.classList.add("tracking");

  if (currentDisplayMode === "camera") {
    document.body.classList.add("camera-mode");
    document.body.classList.remove("background-mode");
  } else {
    document.body.classList.add("background-mode");
    document.body.classList.remove("camera-mode");
  }

  startRenderLoop();
}

function leaveTrackingMode() {
  tracking = false;

  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  document.body.classList.remove("tracking", "camera-mode", "background-mode");
}

function openSettings() {
  if (!tracking) return;

  settingsCameraSelect.value = currentDeviceId;
  displayMode.value = currentDisplayMode;
  backgroundColor.value = currentBackgroundColor;
  colorValue.textContent = currentBackgroundColor;

  document.body.classList.add("settings-open");
  settingsPanel.setAttribute("aria-hidden", "false");
}

function closeSettings() {
  document.body.classList.remove("settings-open");
  settingsPanel.setAttribute("aria-hidden", "true");
}

async function applySettings() {
  const selectedMode = displayMode.value;
  const selectedColor = backgroundColor.value;
  const selectedCamera = settingsCameraSelect.value;

  closeSettings();

  currentDisplayMode = selectedMode;
  currentBackgroundColor = selectedColor || "#000000";

  document.documentElement.style.setProperty("--background", currentBackgroundColor);

  if (selectedCamera && selectedCamera !== currentDeviceId) {
    try {
      setStatus("Changing camera...");
      await startCamera(selectedCamera);
      setStatus("");
    } catch (error) {
      showError(`Unable to switch camera: ${error.message || error}`);
      try { await refreshCameraList(currentDeviceId); } catch (_) {}
    }
  }

  document.body.classList.toggle("camera-mode", currentDisplayMode === "camera");
  document.body.classList.toggle("background-mode", currentDisplayMode === "background");

  await requestFullscreen();
}

function beginLongPress(event) {
  if (!tracking) return;

  touchActive = true;

  if (event.touches?.length) {
    touchStartX = event.touches[0].clientX;
    touchStartY = event.touches[0].clientY;
  }

  clearTimeout(longPressTimer);

  longPressTimer = setTimeout(() => {
    if (touchActive) openSettings();
  }, 700);
}

function cancelLongPress() {
  touchActive = false;
  clearTimeout(longPressTimer);
  longPressTimer = null;
}

function handleTouchEnd(event) {
  if (!tracking) return;

  clearTimeout(longPressTimer);

  if (!event.changedTouches.length) {
    touchActive = false;
    return;
  }

  const touch = event.changedTouches[0];
  const deltaY = touch.clientY - touchStartY;
  const deltaX = touch.clientX - touchStartX;

  if (deltaY < -100 && Math.abs(deltaY) > Math.abs(deltaX)) {
    openSettings();
  }

  touchActive = false;
}

async function handleStart() {
  startButton.disabled = true;
  fullscreenButton.disabled = true;

  try {
    hideError();
    setStatus("Requesting camera permission...");

    await requestInitialPermission();
    await refreshCameraList();

    const selectedCamera = cameraSelect.value;

    await loadHandLandmarker();

    setStatus("Starting camera...");
    await startCamera(selectedCamera);

    currentDisplayMode = displayMode.value || "background";
    currentBackgroundColor = backgroundColor.value || "#000000";

    document.documentElement.style.setProperty("--background", currentBackgroundColor);

    enterTrackingMode();
    await requestFullscreen();
  } catch (error) {
    console.error(error);
    leaveTrackingMode();

    const message = error?.message || "An unexpected error occurred.";
    showError(message);
    setStatus(message);
  } finally {
    startButton.disabled = false;
    fullscreenButton.disabled = false;
  }
}

async function initialize() {
  resizeCanvas();

  colorValue.textContent = backgroundColor.value;
  document.documentElement.style.setProperty("--background", backgroundColor.value);

  if (!navigator.mediaDevices?.getUserMedia) {
    showError("This browser does not support the Camera API. Use Android Chrome over HTTPS.");
    return;
  }

  if (!navigator.mediaDevices.enumerateDevices) {
    showError("This browser does not support camera enumeration.");
    return;
  }

  try {
    setStatus("Camera permission required...");
    await requestInitialPermission();
    await refreshCameraList();
    setStatus("Ready.");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Camera initialization failed.");
    showError(error.message || "Unable to initialize the camera.");
  }
}

startButton.addEventListener("click", handleStart);
fullscreenButton.addEventListener("click", requestFullscreen);

applyButton.addEventListener("click", async () => {
  try {
    await applySettings();
  } catch (error) {
    showError(error.message || "Unable to apply settings.");
  }
});

backgroundColor.addEventListener("input", () => {
  colorValue.textContent = backgroundColor.value;

  if (currentDisplayMode === "background") {
    currentBackgroundColor = backgroundColor.value;
    document.documentElement.style.setProperty("--background", currentBackgroundColor);
  }
});

displayMode.addEventListener("change", () => {
  if (!tracking) return;

  document.body.classList.toggle("camera-mode", displayMode.value === "camera");
  document.body.classList.toggle("background-mode", displayMode.value === "background");
});

cameraSelect.addEventListener("change", () => {
  if (!tracking) currentDeviceId = cameraSelect.value;
});

errorCloseButton.addEventListener("click", hideError);

canvas.addEventListener("touchstart", beginLongPress, { passive: true });
canvas.addEventListener("touchmove", cancelLongPress, { passive: true });
canvas.addEventListener("touchend", handleTouchEnd, { passive: true });
canvas.addEventListener("touchcancel", cancelLongPress, { passive: true });

navigator.mediaDevices?.addEventListener("devicechange", async () => {
  try {
    await refreshCameraList(currentDeviceId);
  } catch (error) {
    console.warn("Camera list update failed:", error);
  }
});

window.addEventListener("resize", resizeCanvas);
window.addEventListener("orientationchange", () => setTimeout(resizeCanvas, 150));

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && tracking) {
    resizeCanvas();
    if (video.paused && currentStream) video.play().catch(() => {});
  }
});

canvas.addEventListener("contextmenu", event => {
  if (tracking) event.preventDefault();
});

initialize();
