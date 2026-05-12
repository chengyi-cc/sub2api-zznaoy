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

export interface InvoiceRequest {
  id: number
  user_id: number
  payment_order_ids: number[]
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
}

export interface CreateInvoiceRequestPayload {
  payment_order_ids: number[]
  invoice_type: InvoiceType
  title: string
  tax_no?: string
  recipient_email?: string
  remark?: string
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
  createInvoiceRequest,
  listInvoiceRequests,
  getInvoiceRequest,
  getInvoiceDownloadUrl,
  downloadInvoice
}

export default invoiceAPI