import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// import App from './App.tsx'
import GeometryComparisonTool from './geometry_comparison.tsx'
// @ts-ignore: No declaration file for module.
import COMSOLAnalysis from './geometry_comparison_v2.jsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* <GeometryComparisonTool /> */}
    <COMSOLAnalysis />
    {/* <App /> */}
  </StrictMode>,
)
