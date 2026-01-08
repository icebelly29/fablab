import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import AppESP32 from "./AppESP32.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <AppESP32 />
  </StrictMode>,
);
