import { describe, expect, it, vi } from 'vitest'
import { readAccountTestStream, type AccountTestStreamEvent } from '../accountTestStream'

describe('readAccountTestStream', () => {
  it('handles split UTF-8, CRLF, no-space data and a final unterminated line', async () => {
    const bytes = new TextEncoder().encode('event: content\r\ndata:{"type":"content","text":"糖果🍬"}\r\n\r\ndata: malformed\ndata: null\ndata: {"missing":"type"}\ndata: [DONE]\ndata: {"type":"test_complete","success":true}')
    const chunks = Array.from(bytes, byte => new Uint8Array([byte]))
    const reader = {
      read: vi.fn(async () => chunks.length ? { done: false, value: chunks.shift() } : { done: true }),
      releaseLock: vi.fn()
    }
    const events: AccountTestStreamEvent[] = []
    await readAccountTestStream(reader as unknown as ReadableStreamDefaultReader<Uint8Array>, event => events.push(event))
    expect(events).toEqual([{ type: 'content', text: '糖果🍬' }, { type: 'test_complete', success: true }])
    expect(reader.releaseLock).toHaveBeenCalledOnce()
  })

  it('propagates reader failure and releases the lock', async () => {
    const reader = { read: vi.fn().mockRejectedValue(new Error('disconnected')), releaseLock: vi.fn() }
    await expect(readAccountTestStream(reader as unknown as ReadableStreamDefaultReader<Uint8Array>, vi.fn())).rejects.toThrow('disconnected')
    expect(reader.releaseLock).toHaveBeenCalledOnce()
  })
})
