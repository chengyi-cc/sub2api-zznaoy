import { createRequire } from 'node:module'
import { createHash } from 'node:crypto'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { readFile, writeFile } from 'node:fs/promises'

export async function buildSelectedHomePreview() {
  const frontend = new URL('../../frontend/', import.meta.url)
  const output = new URL('public/homepage-concepts/', frontend)
  const require = createRequire(new URL('package.json', frontend))
  const vitePackage = pathToFileURL(require.resolve('vite/package.json'))
  const viteEntry = JSON.parse(await readFile(vitePackage, 'utf8')).exports['.'].import.default
  const { build } = await import(new URL(viteEntry, vitePackage).href)
  const { default: vue } = await import(pathToFileURL(require.resolve('@vitejs/plugin-vue')).href)
  await build({
    configFile: false,
    root: fileURLToPath(frontend),
    publicDir: false,
    plugins: [vue()],
    resolve: { alias: { '@': fileURLToPath(new URL('src/', frontend)) }, dedupe: ['vue', 'vue-i18n'] },
    define: { 'process.env.NODE_ENV': JSON.stringify('production') },
    css: { postcss: { plugins: [] } },
    build: {
      outDir: fileURLToPath(new URL('selected/', output)),
      emptyOutDir: false,
      lib: { entry: fileURLToPath(new URL('selected-preview.ts', import.meta.url)), name: 'SelectedHomePreview', formats: ['iife'], fileName: () => 'selected-home.js' },
      rollupOptions: { output: { assetFileNames: 'selected-home[extname]' } },
    },
  })
  const revision = createHash('sha256').update(await readFile(new URL('selected/selected-home.js', output))).update(await readFile(new URL('selected/selected-home.css', output))).digest('hex').slice(0, 12)
  await writeFile(new URL('selected-home.html', output), `<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>定稿 · 深蓝与雾银双主题</title><link rel="stylesheet" href="selected/selected-home.css?v=${revision}"></head><body><div id="app"></div><script src="selected/selected-home.js?v=${revision}"></script></body></html>\n`)
}
