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
  approve: adminApproveInvoiceRequest,
  reject: adminRejectInvoiceRequest,
  issue: adminIssueInvoiceRequest,
  download: adminDownloadInvoice
}

export default adminInvoiceAPI