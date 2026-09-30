let dashboardData = null;

const elements = {};

function cacheElements() {
  ["time", "stamp", "source", "detectionBadge", "confidence", "eyeStatus", "eyeLatitude", "eyeLongitude", "eyeBBox", "eyeDiameter", "eyeBadge", "eyeConfidence", "direction", "arrow", "speed", "bearing", "forecastRows", "landfallProbability", "errorBody", "processing", "liveStatus", "notice", "satelliteImage", "eyeImage", "authScreen", "dashboardApp", "loginForm", "loginMessage", "userGreeting", "logoutButton",
  "liveSatImage", "liveSatName", "liveSatSensor", "liveSatTimestamp", "liveSatStatus", "liveSatRefreshBtn", "liveSatDataset", "liveSatSatellite", "liveSatChannel", "liveSatFrame", "livePulse"].forEach((id) => { elements[id] = document.getElementById(id); });
}

function numericValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatCoordinate(value, positive, negative) {
  const number = numericValue(value);
  return number === null ? "--" : `${Math.abs(number).toFixed(2)}°${number >= 0 ? positive : negative}`;
}

function formatNumber(value, decimals = 1) {
  const number = numericValue(value);
  return number === null ? "--" : number.toFixed(decimals);
}

function displayCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = value;
  row.appendChild(cell);
}
function formatTimestamp(date = new Date()) { return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")} ${date.toUTCString().slice(17, 22)} UTC`; }

function updateTimestamp(timestamp) {
  const value = timestamp || "No processed frame";
  elements.time.textContent = timestamp ? timestamp.slice(11) : "--";
  elements.stamp.textContent = value;
}

function updateDetection(data, eye, payload, ready) {
  const detection = data || {};
  const detected = typeof detection.detected === "boolean" ? detection.detected : payload.cyclone_detected;
  const confidence = numericValue(detection.confidence) ?? numericValue(payload.confidence);
  const eyeDetected = typeof detection.eyeDetected === "boolean"
    ? detection.eyeDetected
    : (typeof eye.detected === "boolean" ? eye.detected : detected);
  const latitude = detection.latitude ?? payload.eye_latitude;
  const longitude = detection.longitude ?? payload.eye_longitude;
  const bbox = detection.eye_bbox ?? eye.bbox ?? payload.eye_bbox;
  const diameter = detection.eye_diameter_km ?? eye.diameter_km ?? payload.eye_diameter_km;

  if (!ready || typeof detected !== "boolean") {
    elements.detectionBadge.textContent = "MODEL UNAVAILABLE";
    elements.detectionBadge.classList.add("warning");
    elements.confidence.textContent = "--";
    elements.eyeStatus.textContent = "--";
    elements.eyeLatitude.textContent = "--";
    elements.eyeLongitude.textContent = "--";
    elements.eyeBBox.textContent = "--";
    elements.eyeDiameter.textContent = "--";
    return;
  }
  elements.detectionBadge.textContent = detected ? "Cyclone detected" : "No cyclone detected";
  elements.detectionBadge.classList.toggle("warning", !detected);
  const confidencePercent = confidence !== null && confidence <= 1 ? confidence * 100 : confidence;
  elements.confidence.textContent = confidencePercent === null ? "--" : `${confidencePercent.toFixed(1)}%`;
  elements.eyeStatus.textContent = eyeDetected ? "Detected" : "Not detected";
  elements.eyeLatitude.textContent = formatCoordinate(latitude, "N", "S");
  elements.eyeLongitude.textContent = formatCoordinate(longitude, "E", "W");
  elements.eyeBBox.textContent = Array.isArray(bbox) && bbox.length
    ? bbox.map((value) => formatNumber(value, 3)).join(", ")
    : "--";
  const diameterValue = numericValue(diameter);
  elements.eyeDiameter.textContent = diameterValue === null ? "--" : `${diameterValue.toFixed(1)} km`;
}

function updateMovement(data) {
  const movement = data || {};
  elements.direction.textContent = movement.direction || "--";
  elements.speed.textContent = Number.isFinite(movement.speed) ? `${movement.speed.toFixed(1)} km/h` : "--";
  elements.bearing.textContent = Number.isFinite(movement.bearing) ? `${movement.bearing}°` : "--";
  if (Number.isFinite(movement.bearing)) {
    elements.arrow.style.transform = `rotate(${movement.bearing - 315}deg)`;
    elements.arrow.setAttribute("aria-label", `Movement bearing ${movement.bearing} degrees`);
  }
}

function updateSatellite(data) {
  const overlayUrl = data.overlay_url || data.satellite?.overlayUrl;
  const hasOverlay = data.status === "ready" && overlayUrl;
  elements.source.textContent = hasOverlay ? "Model overlay" : "Overlay unavailable";
  elements.satelliteImage.closest(".image").classList.toggle("empty", !hasOverlay);
  if (hasOverlay) {
    elements.satelliteImage.src = `${API_BASE}${overlayUrl}?t=${Date.now()}`;
    elements.satelliteImage.classList.remove("hidden");
  } else {
    elements.satelliteImage.removeAttribute("src");
    elements.satelliteImage.classList.add("hidden");
  }
}

function updateEyeImage(data, ready) {
  const eye = data || {};
  const available = ready && eye.detected && eye.imageUrl;
  elements.eyeBadge.textContent = !ready ? "Inference unavailable" : (eye.detected ? "Eye crop available" : "Eye not detected");
  elements.eyeBadge.classList.toggle("warning", !available);
  if (available) {
    elements.eyeImage.src = `${API_BASE}${eye.imageUrl}?t=${Date.now()}`;
    elements.eyeImage.classList.remove("hidden");
  } else {
    elements.eyeImage.removeAttribute("src");
    elements.eyeImage.classList.add("hidden");
  }
  const confidence = numericValue(eye.confidence);
  elements.eyeConfidence.textContent = available && confidence !== null ? `${confidence.toFixed(1)}%` : "--";
}

function formatUncertaintyRadii(radii) {
  if (Array.isArray(radii) && radii.length) {
    const values = radii.map((radius) => numericValue(radius));
    if (values.some((value) => value !== null)) {
      return `${values.map((value) => value === null ? "--" : value.toFixed(1)).join(" x ")} km`;
    }
    return "--";
  }
  if (radii && typeof radii === "object") {
    const values = [radii.x, radii.y].map((value) => numericValue(value));
    return values.some((value) => value !== null)
      ? `${values.map((value) => value === null ? "--" : value.toFixed(1)).join(" x ")} km`
      : "--";
  }
  const value = numericValue(radii);
  return value === null ? "--" : `${value.toFixed(1)} km`;
}

function renderForecastTrack(forecast, uncertainty, landfallProbability) {
  const track = Array.isArray(forecast) ? forecast : [];
  const radii = Array.isArray(uncertainty?.radii_km) ? uncertainty.radii_km : [];
  const rows = track.map((point, index) => {
    const row = document.createElement("tr");
    const hour = numericValue(point.hour);
    const coneRadius = numericValue(point.cone_radius_km);
    displayCell(row, hour === null ? "--" : `+${hour}h`);
    displayCell(row, formatCoordinate(point.latitude, "N", "S"));
    displayCell(row, formatCoordinate(point.longitude, "E", "W"));
    displayCell(row, coneRadius === null ? "--" : `${coneRadius.toFixed(1)} km`);
    displayCell(row, formatUncertaintyRadii(radii[index]));
    return row;
  });
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No forecast points available";
    row.appendChild(cell);
    rows.push(row);
  }
  elements.forecastRows.replaceChildren(...rows);

  const probability = numericValue(landfallProbability);
  const probabilityPercent = probability !== null && probability <= 1 ? probability * 100 : probability;
  elements.landfallProbability.textContent = probabilityPercent === null ? "--" : `${probabilityPercent.toFixed(1)}%`;
}

function renderErrorHistory(history) {
  const rows = history.map((item) => {
    const row = document.createElement("tr");
    const error = numericValue(item.error);
    displayCell(row, item.forecast || "--");
    displayCell(row, error === null ? "--" : error.toFixed(2));
    return row;
  });
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 2;
    cell.textContent = "No verified forecast errors";
    row.appendChild(cell);
    rows.push(row);
  }
  elements.errorBody.replaceChildren(...rows);
}

function renderDashboard(data) {
  dashboardData = data;
  const ready = data.status === "ready" || (!data.status && typeof data.cyclone_detected === "boolean");
  const detection = data.detection || {};
  const eye = data.eye || {};
  const forecast = Array.isArray(data.forecast_track) ? data.forecast_track : (Array.isArray(data.forecast) ? data.forecast : []);
  updateDetection(detection, eye, data, ready);
  updateMovement(data.movement);
  updateSatellite(data);
  updateEyeImage(data.eye, ready);
  renderForecastTrack(forecast, data.uncertainty, data.uncertainty?.landfall_probability);
  renderErrorHistory(Array.isArray(data.errorHistory) ? data.errorHistory : []);
  elements.processing.textContent = data.processingTime || "--";
  updateTimestamp(data.satellite?.timestamp);
}

function setLiveState(status) {
  const labels = { ready: "LIVE", in_progress: "PROCESSING", available: "PROCESSING", stale: "STALE", error: "UNAVAILABLE", unavailable: "WAITING" };
  const label = labels[status] || "UNAVAILABLE";
  const available = status === "ready";
  elements.liveStatus.classList.toggle("unavailable", !available);
  elements.liveStatus.innerHTML = `<span class="dot"></span>${label}`;
}
function showNotice(message) { elements.notice.textContent = message; elements.notice.classList.remove("hidden"); setTimeout(() => elements.notice.classList.add("hidden"), 4500); }

/* ===== Live Satellite Card Integration (Requirement 11) ===== */
async function updateLiveSatellite(data = null) {
  try {
    data = data || await fetchSatelliteData();
    if (!data) return;

    elements.liveSatName.textContent = data.satellite || "INSAT satellite";
    elements.liveSatSensor.textContent = data.sensor || "--";
    elements.liveSatTimestamp.textContent = data.timestamp || "--";
    elements.liveSatSatellite.textContent = data.satellite || "--";
    elements.liveSatChannel.textContent = data.sensor || "--";
    elements.liveSatDataset.textContent = data.datasetId || "--";
    elements.liveSatFrame.textContent = data.filename || "--";

    const imageAvailable = data.status === "ready" && data.preview_url;
    elements.liveSatImage.closest(".live-sat-image-wrap").classList.toggle("empty", !imageAvailable);
    if (imageAvailable) {
      elements.liveSatImage.src = `${API_BASE}${data.preview_url}?t=${Date.now()}`;
      elements.liveSatImage.classList.remove("hidden");
    } else {
      elements.liveSatImage.removeAttribute("src");
      elements.liveSatImage.classList.add("hidden");
    }

    const statusText = data.status === "ready" ? "Processed frame ready" :
      data.status === "available" ? "Frame found; processing" :
      data.status === "in_progress" ? "MOSDAC check/download in progress" :
      data.last_error || `Pipeline ${data.status || "unavailable"}`;
    elements.liveSatStatus.textContent = `${statusText}${data.last_checked ? ` · ${data.last_checked}` : ""}`;
    elements.livePulse.style.display = data.status === "ready" ? "inline-flex" : "none";
  } catch (err) {
    console.warn("Live satellite update failed:", err);
    elements.liveSatStatus.textContent = "Backend unavailable";
    setLiveState("error");
  }
}

async function loadDashboardData() {
  try {
    const [satellite, cyclone, health] = await Promise.all([fetchSatelliteData(), fetchCycloneData(), fetchHealthData()]);
    if (!satellite || !cyclone || !health) throw new Error("Dashboard API data is unavailable.");
    cyclone.status = cyclone.status || "unavailable";
    renderDashboard(cyclone);
    await updateLiveSatellite(satellite);
    if (health.scheduler?.pipeline_in_progress) {
      setLiveState("in_progress");
      elements.liveSatStatus.textContent = "MOSDAC download / model processing in progress…";
    } else {
      setLiveState(cyclone.status === "ready" && satellite.status === "ready" ? "ready" : satellite.status);
    }
  } catch (err) {
    console.error("Dashboard load failed:", err);
    setLiveState("error");
    elements.liveSatStatus.textContent = "Backend unavailable";
  }
}

async function waitForPipeline(timeoutMs = 180000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const health = await fetchHealthData();
    if (!health.scheduler?.pipeline_in_progress) return health;
    elements.liveSatStatus.textContent = "MOSDAC download / model processing in progress…";
    await new Promise((resolve) => setTimeout(resolve, 2500));
  }
  throw new Error("The pipeline is still working. Check the dashboard again shortly.");
}

async function runManualRefresh(button) {
  if (!button) return;
  const originalText = button.textContent;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = "Checking…";
  try {
    const result = await triggerPipelineRefresh();
    if (result.status === "in_progress") await waitForPipeline();
    await loadDashboardData();
    if (result.status === "updated") showNotice("New satellite frame processed and model results updated.");
    else if (result.status === "skipped") showNotice("No newer frame was available. Showing the latest processed frame.");
    else if (result.status === "error") showNotice(result.message || "Pipeline failed. Check MOSDAC access and server logs.");
    else if (result.status === "in_progress") showNotice("Pipeline finished. Dashboard data has been refreshed.");
  } catch (error) {
    showNotice(error.message || "Refresh failed. The backend may be unavailable.");
    setLiveState("error");
    console.error(error);
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.textContent = originalText;
  }
}

function refreshLiveSatellite() { return runManualRefresh(elements.liveSatRefreshBtn); }

function initializeFoldTitle() {
  document.querySelectorAll(".fold-animated").forEach((title) => {
    const text = title.dataset.foldText || title.textContent;
    title.setAttribute("aria-label", text);
    title.setAttribute("role", "text");
    title.textContent = "";
    Array.from(text).forEach((character, index) => {
      const piece = document.createElement("span");
      piece.textContent = character === " " ? "\u00a0" : character;
      piece.className = character === " " ? "space" : "";
      piece.style.setProperty("--fold-index", index);
      piece.setAttribute("aria-hidden", "true");
      title.appendChild(piece);
    });
  });
}

function showDashboard(userName) {
  elements.userGreeting.textContent = userName;
  elements.authScreen.classList.add("hidden");
  elements.dashboardApp.classList.remove("hidden");
}

function showLogin() {
  elements.dashboardApp.classList.add("hidden");
  elements.authScreen.classList.remove("hidden");
  const password = document.getElementById("userPassword");
  password.type = "password";
  document.getElementById("passwordToggle").textContent = "Show";
  document.getElementById("passwordToggle").setAttribute("aria-label", "Show password");
  document.getElementById("userName").focus();
}

function initializeAuthentication() {
  const password = document.getElementById("userPassword");
  const passwordToggle = document.getElementById("passwordToggle");
  passwordToggle.addEventListener("click", () => {
    const isHidden = password.type === "password";
    password.type = isHidden ? "text" : "password";
    passwordToggle.textContent = isHidden ? "Hide" : "Show";
    passwordToggle.setAttribute("aria-label", `${isHidden ? "Hide" : "Show"} password`);
  });
  elements.loginForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(elements.loginForm);
    const userName = String(formData.get("userName")).trim();
    const userEmail = String(formData.get("userEmail")).trim();
    const userPassword = String(formData.get("userPassword"));
    if (!userName || !userEmail || userPassword.length < 4) {
      elements.loginMessage.textContent = "Please complete all fields. Your password needs at least 4 characters.";
      return;
    }
    elements.loginMessage.textContent = "";
    showDashboard(userName.split(" ")[0]);
  });
  elements.logoutButton.addEventListener("click", showLogin);
}

async function refreshDashboard() {
  return runManualRefresh(document.getElementById("refreshButton"));
}

async function updateSettingsStatus() {
  try {
    const health = await fetchHealthData();
    document.getElementById("settingBackend").textContent = health.status === "healthy" ? "Connected" : health.status;
    document.getElementById("settingMosdac").textContent = health.mosdac_configured ? "Configured" : "Credentials missing";
    document.getElementById("settingDataset").textContent = `${health.datasetId} / ${health.channel}`;
    document.getElementById("settingInterval").textContent = `${Math.round(health.scheduler.poll_interval_seconds / 60)} minutes`;
    document.getElementById("settingLastCheck").textContent = health.scheduler.last_poll_time || "Not checked yet";
  } catch (error) {
    document.getElementById("settingBackend").textContent = "Unavailable";
    document.getElementById("settingMosdac").textContent = "Unknown";
    document.getElementById("settingDataset").textContent = "Unknown";
    document.getElementById("settingInterval").textContent = "Unknown";
    document.getElementById("settingLastCheck").textContent = "Unknown";
    console.error("Could not load pipeline status:", error);
  }
}

function bindNavButtons() {
  const navButtons = document.querySelectorAll(".nav button");
  const sections = [
    { name: "Dashboard", target: document.querySelector(".content") },
    { name: "Live Feed", target: document.getElementById("liveSatelliteCard") },
    { name: "Track", target: document.querySelector(".card.track") },
    { name: "Forecast", target: document.querySelector(".card.errors") },
    { name: "Settings", target: null }
  ];

  navButtons.forEach((button, index) => {
    button.addEventListener("click", () => {
      navButtons.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const section = sections[index];
      if (!section) return;
      if (section.name === "Settings") {
        const dialog = document.getElementById("settingsDialog");
        dialog.showModal();
        updateSettingsStatus();
        return;
      }
      if (section.target) {
        section.target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

function initializeDashboard() {
  cacheElements(); initializeFoldTitle(); initializeAuthentication(); document.getElementById("refreshButton").addEventListener("click", refreshDashboard);
  bindNavButtons();
  document.getElementById("settingsRefreshButton").addEventListener("click", updateSettingsStatus);
  document.querySelector(".settings-close").addEventListener("click", () => document.getElementById("settingsDialog").close());

  // ===== Live Satellite Integration (Requirement 11) =====
  if (elements.liveSatRefreshBtn) elements.liveSatRefreshBtn.addEventListener("click", refreshLiveSatellite);
  loadDashboardData();
  setInterval(loadDashboardData, 60000);
  setInterval(async () => {
    try {
      const health = await fetchHealthData();
      if (health.scheduler?.pipeline_in_progress) {
        setLiveState("in_progress");
        elements.liveSatStatus.textContent = "MOSDAC download / model processing in progress…";
      } else if (dashboardData?.status === "ready") {
        setLiveState("ready");
        elements.liveSatStatus.textContent = `Processed frame ready · ${dashboardData.satellite?.timestamp || "latest frame"}`;
      }
    } catch (error) {
      console.warn("Pipeline status check failed:", error);
    }
  }, 5000);
}

document.addEventListener("DOMContentLoaded", initializeDashboard);