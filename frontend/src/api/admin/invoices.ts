/**
 * Invoice request API endpoints (admin-side)
 */

import { apiClient } from '../client'
import type { InvoiceRequest, InvoiceStatus, PaginatedInvoiceRequests } from '../invoice'

const BASE = '/admin/invoice'

export async function adminListInvoiceRequests(params: {
  status?: InvoiceStatus | ''
  user_id?: number
  q?: string
  page?: number
  page_size?: number
}): Promise<PaginatedInvoiceRequests> {
  const { data } = await apiClient.get<PaginatedInvoiceRequests>(`${BASE}/requests`, {
    params
  })
  return data
}

export async function adminGetInvoiceRequest(id: number): Promise<InvoiceRequest> {
  const { data } = await apiClient.get<InvoiceRequest>(`${BASE}/requests/${id}`)
  return data
}

export interface InvoiceDetailOrder {
  order_id: number
  out_trade_no: string
  amount: number
  status: string
  order_type: string
  payment_type: string
  completed_at: string
  eligible: boolean
}

export interface InvoiceDetailRedeem {
  redeem_code_id: number
  code: string
  value: number
  status: string
  type: string
  used_by: number
  used_at: string
  eligible: boolean
}

export interface InvoiceRequestDetail {
  request: InvoiceRequest
  user_email: string
  user_name: string
  orders: InvoiceDetailOrder[]
  redeem_codes: InvoiceDetailRedeem[]
  computed_sum: number
  amount_match: boolean
  all_eligible: boolean
}

export async function adminGetInvoiceDetail(id: number): Promise<InvoiceRequestDetail> {
  const { data } = await apiClient.get<InvoiceRequestDetail>(`${BASE}/requests/${id}/detail`)
  return data
}

export async function adminApproveInvoiceRequest(id: number): Promise<InvoiceRequest> {
  const { data } = await apiClient.post<InvoiceRequest>(`${BASE}/requests/${id}/approve`)
  return data
}

export async function adminRejectInvoiceRequest(
  id: number,
  reason: string
): Promise<InvoiceRequest> {
  const { data } = await apiClient.post<InvoiceRequest>(`${BASE}/requests/${id}/reject`, {
    reason
  })
  return data
}

export async function adminIssueInvoiceRequest(
  id: number,
  invoiceNo: string,
  file: File
): Promise<InvoiceRequest> {
  const form = new FormData()
  form.append('invoice_no', invoiceNo)
  form.append('file', file)
  const { data } = await apiClient.post<InvoiceRequest>(`${BASE}/requests/${id}/issue`, form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return data
}

export async function adminDownloadInvoice(id: number, filename = 'invoice.pdf'): Promise<void> {
  const resp = await apiClient.get(`${BASE}/requests/${id}/download`, {
    responseType: 'blob'
  })
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

export const adminInvoiceAPI = {
  list: adminListInvoiceRequests,
  get: adminGetInvoiceRequest,
  detail: adminGetInvoiceDetail,
  approve: adminApproveInvoiceRequest,
  reject: adminRejectInvoiceRequest,
  issue: adminIssueInvoiceRequest,
  download: adminDownloadInvoice
}

export default adminInvoiceAPI