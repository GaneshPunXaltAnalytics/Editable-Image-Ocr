import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

// Suppress findDOMNode deprecation from react-quill
if (import.meta.env.DEV && typeof console.error === 'function') {
  const orig = console.error
  console.error = (...args) => {
    const msg = args[0]
    if (typeof msg === 'string' && msg.includes('findDOMNode') && msg.includes('deprecated')) return
    orig.apply(console, args)
  }
}

const root = createRoot(document.getElementById('root'))
root.render(<App />)

