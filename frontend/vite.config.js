import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The frontend (localhost:5173) and the API (localhost:8000) are
// different origins. Rather than configuring CORS on the API, the dev
// server proxies every /api request to the backend server-side, so the
// browser only ever sees same-origin requests. App code uses relative
// URLs (fetch('/api/...')), which also keeps it deployment-portable.
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
