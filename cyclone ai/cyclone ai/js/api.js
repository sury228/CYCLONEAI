/**
 * CycloneVision AI - API Client
 * Connects Frontend Dashboard directly to FastAPI Backend.
 * (Frontend NEVER communicates with MOSDAC directly, credentials remain strictly backend-only)
 */

const API_BASE = window.location.origin.includes(":8000")
  ? ""
  : "http://127.0.0.1:8000";

async function fetchSatelliteData() {
  try {
    const response = await fetch(`${API_BASE}/api/satellite/latest`, {
      method: "GET",
      headers: { "Accept": "application/json" }
    });
    if (!response.ok) {
      throw new Error(`Satellite API request failed: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.warn("Could not fetch live satellite endpoint:", err);
    return null;
  }
}

async function fetchCycloneData() {
  const response = await fetch(`${API_BASE}/api/cyclone/latest`, {
    method: "GET",
    headers: { "Accept": "application/json" }
  });

  if (!response.ok) {
    throw new Error(`Cyclone API request failed: ${response.status}`);
  }

  return await response.json();
}

async function fetchHealthData() {
  const response = await fetch(`${API_BASE}/api/health`, {
    method: "GET",
    headers: { "Accept": "application/json" }
  });

  if (!response.ok) {
    throw new Error(`Health API request failed: ${response.status}`);
  }

  return await response.json();
}

async function triggerPipelineRefresh() {
  const response = await fetch(`${API_BASE}/api/satellite/refresh`, {
    method: "POST",
    headers: { "Accept": "application/json" }
  });

  if (!response.ok) {
    throw new Error(`Refresh trigger failed: ${response.status}`);
  }

  return await response.json();
}