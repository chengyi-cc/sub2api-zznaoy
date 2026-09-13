export interface AccountTestStreamEvent {
  type: string
  text?: string
  model?: string
  success?: boolean
  error?: string
  image_url?: string
  audio_url?: string
  video_url?: string
  mime_type?: string
  data?: unknown
}

export async function readAccountTestStream(reader: ReadableStreamDefaultReader<Uint8Array>, onEvent: (event: AccountTestStreamEvent) => void): Promise<void> {
  const decoder = new TextDecoder()
  let buffer = ''
  const consume = (line: string) => {
    if (!line.startsWith('data:')) return
    const text = line.slice(5).trim()
    if (!text || text === '[DONE]') return
    let event: AccountTestStreamEvent
    try {
      event = JSON.parse(text) as AccountTestStreamEvent
    } catch {
      return
    }
    if (event && typeof event.type === 'string') onEvent(event)
  }
  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      lines.forEach(consume)
      if (done) {
        consume(buffer)
        break
      }
    }
  } finally {
    reader.releaseLock?.()
  }
}
