const mockCycloneData = {
  detection: { detected: true, confidence: 96.8, latitude: 16.42, longitude: 88.21, eyeDetected: true },
  eye: { imageUrl: "", detected: true, confidence: 94.2 },
  movement: { direction: "WNW", speed: 12.4, bearing: 292 },
  satellite: { source: "INSAT-3D", imageUrl: "", timestamp: "" },
  forecast: [
    { hour: 6, latitude: 16.9, longitude: 87.8 },
    { hour: 12, latitude: 17.4, longitude: 87.2 },
    { hour: 24, latitude: 18.42, longitude: 86.50 }
  ],
  errorHistory: [
    { forecast: "+06h", error: 14.56 }, { forecast: "+12h", error: 28.60 },
    { forecast: "+18h", error: 44.97 }, { forecast: "+24h", error: 62.18 },
    { forecast: "+36h", error: 80.85 }, { forecast: "+48h", error: 100.26 }
  ],
  processingTime: "2.4s"
};