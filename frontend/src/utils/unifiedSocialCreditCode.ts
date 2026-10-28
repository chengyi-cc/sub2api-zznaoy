/**
 * 统一社会信用代码（GB 32100-2015）校验，与后端 NormalizeUnifiedSocialCreditCode 保持一致。
 * 开票资料绑定后即锁定，前端先拦一次手误，避免用户提交后才发现填错。
 */

const USCC_CHARSET = '0123456789ABCDEFGHJKLMNPQRTUWXY'
const USCC_WEIGHTS = [1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28]

export function normalizeUnifiedSocialCreditCode(raw: string): string {
  return raw.trim().toUpperCase()
}

export function isValidUnifiedSocialCreditCode(raw: string): boolean {
  const code = normalizeUnifiedSocialCreditCode(raw)
  if (code.length !== 18) return false
  let sum = 0
  for (let i = 0; i < 17; i++) {
    const v = USCC_CHARSET.indexOf(code[i])
    if (v < 0) return false
    sum += v * USCC_WEIGHTS[i]
  }
  let check = 31 - (sum % 31)
  if (check === 31) check = 0
  return code[17] === USCC_CHARSET[check]
}
