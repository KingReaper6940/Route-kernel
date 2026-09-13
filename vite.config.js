import { defineConfig } from 'vite';
export default defineConfig({
  root: 'site',
  build: { outDir: '../site-dist', emptyOutDir: true },
  server: { port: 5173, strictPort: true },
  preview: { port: 4173, strictPort: true },
});
