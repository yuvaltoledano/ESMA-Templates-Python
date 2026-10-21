// Fonts are bundled from @fontsource (latin subsets only: ASCII, Latin-1
// accents such as "Müller", "€" and "—"), so the built page never requests
// a font from a third-party host. Vite copies the woff2 files into
// dist/assets and rewrites the @font-face URLs.
import '@fontsource/ibm-plex-sans/latin-400.css'
import '@fontsource/ibm-plex-sans/latin-500.css'
import '@fontsource/ibm-plex-sans/latin-600.css'
import '@fontsource/ibm-plex-mono/latin-500.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
