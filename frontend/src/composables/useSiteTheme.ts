import { onBeforeUnmount, onMounted, ref } from 'vue'

export function useSiteTheme() {
  const isDark = ref(document.documentElement.classList.contains('dark'))
  const preference = window.matchMedia('(prefers-color-scheme: dark)')
  const savedTheme = () => {
    try { return localStorage.getItem('theme') } catch { return null }
  }
  const applyTheme = (dark: boolean) => {
    isDark.value = dark
    document.documentElement.classList.toggle('dark', dark)
  }
  const synchronize = () => {
    const saved = savedTheme()
    applyTheme(saved === 'dark' || (saved !== 'light' && preference.matches))
  }
  const toggleTheme = () => {
    applyTheme(!isDark.value)
    try { localStorage.setItem('theme', isDark.value ? 'dark' : 'light') } catch { return }
  }
  const systemChanged = () => { if (!savedTheme()) synchronize() }
  const storageChanged = (event: StorageEvent) => { if (event.key === 'theme' || event.key === null) synchronize() }
  let observer: MutationObserver | undefined

  onMounted(() => {
    synchronize()
    observer = new MutationObserver(() => { isDark.value = document.documentElement.classList.contains('dark') })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    preference.addEventListener?.('change', systemChanged)
    window.addEventListener('storage', storageChanged)
  })
  onBeforeUnmount(() => {
    observer?.disconnect()
    preference.removeEventListener?.('change', systemChanged)
    window.removeEventListener('storage', storageChanged)
  })
  return { isDark, toggleTheme }
}
