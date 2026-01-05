import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist-esp32',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index_esp32.html'),
      },
    },
  },
});
