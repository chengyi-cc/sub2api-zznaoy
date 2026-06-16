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

// 导出「已通过待开票」用的扁平行（对应后端 service.InvoiceExportRow）
export interface InvoiceExportRow {
  id: number
  user_id: number
  title: string
  tax_no: string
  amount: number
  user_email: string
  recipient_email: string
}

// 拉取所有 status=approved（已通过待开票）的发票申请，供前端生成 Excel
export async function adminExportApprovedInvoices(): Promise<InvoiceExportRow[]> {
  const { data } = await apiClient.get<{ items: InvoiceExportRow[] }>(`${BASE}/requests/export`)
  return data.items || []
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

export interface InvoiceUserOverview {
  total_recharged: number
  balance: number
  issued_total: number
  in_flight_total: number
  issued_plus_in_flight: number
  exceeds_total_recharge: boolean
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
  user_overview: InvoiceUserOverview
}

export async function adminGetInvoiceDetail(id: number): Promise<InvoiceRequestDetail> {
  const { data } = await apiClient.get<InvoiceRequestDetail>(`${BASE}/requests/${id}/detail`)
  return data
}

export async function adminApproveInvoiceRequest(id: number): Promise<InvoiceRequest> {
  const { data } = await apiClient.post<InvoiceRequest>(`${BASE}/requests/${id}/approve`)
  return data
}

export interface BatchApproveResult {
  succeeded_ids: number[]
  failed: { id: number; reason: string }[]
}

// 批量审核通过；单条失败不影响其它条目
export async function adminBatchApproveInvoices(ids: number[]): Promise<BatchApproveResult> {
  const { data } = await apiClient.post<BatchApproveResult>(`${BASE}/requests/batch-approve`, { ids })
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
  exportApproved: adminExportApprovedInvoices,
  detail: adminGetInvoiceDetail,
  approve: adminApproveInvoiceRequest,
  batchApprove: adminBatchApproveInvoices,
  reject: adminRejectInvoiceRequest,
  issue: adminIssueInvoiceRequest,
  download: adminDownloadInvoice
}

export default adminInvoiceAPI