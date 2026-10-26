import { beforeEach, describe, expect, it, vi } from 'vitest'
import { generate } from '../admin/redeem'
const { post } = vi.hoisted(() => ({ post: vi.fn() }))
vi.mock('../client', () => ({ apiClient: { post } }))

describe('admin redeem code generation', () => {
  beforeEach(() => { vi.clearAllMocks(); post.mockResolvedValue({ data: [] }) })
  it('sends only the gift opt-in flag with a stable operation key', async () => {
    await generate(2, 'balance', 10, undefined, undefined, 7, true, 'same-operation')
    expect(post).toHaveBeenCalledWith('/admin/redeem-codes/generate', {
      count: 2, type: 'balance', value: 10, expires_in_days: 7, include_gift: true
    }, { headers: { 'Idempotency-Key': 'same-operation' } })
  })
  it('does not include gifts for other code types', async () => {
    await generate(1, 'concurrency', 10, undefined, undefined, undefined, true)
    expect(post.mock.calls[0][1]).not.toHaveProperty('include_gift')
    expect(post.mock.calls[0][2].headers['Idempotency-Key']).toMatch(/^redeem-generate-/)
  })
})
