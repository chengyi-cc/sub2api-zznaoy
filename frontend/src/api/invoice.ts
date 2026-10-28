/**
 * Invoice request API endpoints (user-side)
 */

import { apiClient } from './client'

export type InvoiceStatus = 'pending' | 'approved' | 'rejected' | 'issued'
export type InvoiceType = 'personal' | 'company'

export interface EligibleOrder {
  order_id: number
  out_trade_no: string
  amount: number
  order_type: string
  payment_type: string
  completed_at: string
}

export interface EligibleRedeemCode {
  redeem_code_id: number
  code: string
  value: number
  used_at: string
}

export interface EligibleSources {
  orders: EligibleOrder[]
  redeem_codes: EligibleRedeemCode[]
}

export interface InvoiceRequest {
  id: number
  user_id: number
  payment_order_ids: number[]
  redeem_code_ids: number[]
  amount: number
  invoice_type: InvoiceType
  title: string
  tax_no?: string
  recipient_email?: string
  remark?: string
  status: InvoiceStatus
  reject_reason?: string
  invoice_no?: string
  has_file: boolean
  issued_at?: string
  processed_by?: number
  created_at: string
  updated_at: string
  // 申请人注册邮箱（管理端列表补充，用户端不返回）
  user_email?: string
}

// 抬头/税号由后端取自已绑定的开票资料（仅企业），不再随申请提交
export interface CreateInvoiceRequestPayload {
  payment_order_ids?: number[]
  redeem_code_ids?: number[]
  recipient_email?: string
  remark?: string
}

// 用户固定开票资料：绑定一次后锁定，仅管理员可修改或重置
export interface InvoiceProfile {
  title: string
  tax_no: string
  locked: boolean
  admin_modified: boolean
  created_at: string
  updated_at: string
}

export interface PaginatedInvoiceRequests {
  items: InvoiceRequest[]
  total: number
  page: number
  page_size: number
  pages: number
}

const BASE = '/invoice'

export async function listEligibleOrders(): Promise<EligibleOrder[]> {
  const { data } = await apiClient.get<EligibleOrder[]>(`${BASE}/eligible-orders`)
  return data
}

export async function listEligibleSources(): Promise<EligibleSources> {
  const { data } = await apiClient.get<EligibleSources>(`${BASE}/eligible-sources`)
  return data
}

export async function getInvoiceProfile(): Promise<InvoiceProfile | null> {
  const { data } = await apiClient.get<{ profile: InvoiceProfile | null }>(`${BASE}/profile`)
  return data.profile ?? null
}

export async function bindInvoiceProfile(payload: {
  title: string
  tax_no: string
}): Promise<InvoiceProfile> {
  const { data } = await apiClient.post<{ profile: InvoiceProfile }>(`${BASE}/profile`, payload)
  return data.profile
}

export async function createInvoiceRequest(
  payload: CreateInvoiceRequestPayload
): Promise<InvoiceRequest> {
  const { data } = await apiClient.post<InvoiceRequest>(`${BASE}/requests`, payload)
  return data
}

export async function listInvoiceRequests(params: {
  status?: InvoiceStatus | ''
  page?: number
  page_size?: number
}): Promise<PaginatedInvoiceRequests> {
  const { data } = await apiClient.get<PaginatedInvoiceRequests>(`${BASE}/requests`, {
    params
  })
  return data
}

export async function getInvoiceRequest(id: number): Promise<InvoiceRequest> {
  const { data } = await apiClient.get<InvoiceRequest>(`${BASE}/requests/${id}`)
  return data
}

export function getInvoiceDownloadUrl(id: number): string {
  const base = import.meta.env.VITE_API_BASE_URL || '/api/v1'
  return `${base}${BASE}/requests/${id}/download`
}

/** Trigger PDF download via authenticated XHR (so we can pass Authorization header) */
export async function downloadInvoice(id: number, filename = 'invoice.pdf'): Promise<void> {
  const resp = await apiClient.get(`${BASE}/requests/${id}/download`, {
    responseType: 'blob'
  })
  // After response interceptor unwrap, resp.data is the blob (interceptor only unwraps JSON)
  const blob = resp.data as unknown as Blob
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

export const invoiceAPI = {
  listEligibleOrders,
  listEligibleSources,
  getInvoiceProfile,
  bindInvoiceProfile,
  createInvoiceRequest,
  listInvoiceRequests,
  getInvoiceRequest,
  getInvoiceDownloadUrl,
  downloadInvoice
}

export default invoiceAPI