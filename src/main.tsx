import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// import App from './App.tsx'
import GeometryComparisonTool from './geometry_comparison.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <GeometryComparisonTool />
    {/* <App /> */}
  </StrictMode>,
)
