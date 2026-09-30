# Cyclone AI Dashboard

Open `index.html` directly in a browser. No build step or external frontend framework is required.

The existing dashboard layout is kept intact, while the UI now renders from structured data in `js/mockData.js`. `js/api.js` provides the async integration boundary for a future backend, and `js/app.js` owns the individual dashboard updates and refresh flow.

Project structure:

```text
index.html
css/style.css
js/app.js
js/mockData.js
js/api.js
```

The satellite image, cropped eye, and map are intentionally local placeholders. Integration TODOs identify where to connect the satellite feed, detection and prediction models, backend API, history database, and a real map provider.
