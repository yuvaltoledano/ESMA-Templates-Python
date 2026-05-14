# ESMA-MILAN GUI

A minimum-viable local web frontend for the ESMA-MILAN pipeline API. One
screen, one form: upload a loans CSV and a collaterals CSV (plus an optional
taxonomy XLSX), enter a deal name, click **Process Pool**, and the composed
workbook downloads to your browser. A footer indicator polls the API's health
check so a server that died is visible at a glance.

This is intentionally minimal — there is no dry-run preview, no history panel,
no auth, and no deployment story. It exists so the API can be exercised through
real use rather than `curl`.

## Stack

React 19 + Vite 8 + Tailwind CSS v3, in plain JavaScript (no TypeScript).

## Running it

The API must be running on `localhost:8000` at the same time. In one terminal,
from the repo root:

```sh
uv run esma-milan-server
```

In another terminal:

```sh
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>.

Requests to `/api/*` are proxied to `http://localhost:8000` by the Vite dev
server (see `vite.config.js`), so the browser only ever makes same-origin
requests — no CORS configuration is needed on the API.

## Scripts

- `npm run dev` — start the dev server with HMR
- `npm run build` — production build into `dist/`
- `npm run lint` — ESLint
- `npm run preview` — serve the production build locally
