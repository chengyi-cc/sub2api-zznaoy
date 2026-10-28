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
  // 申请税号还被多少个其他账号绑定（同一公司多账号提示）
  tax_no_shared_user_count: number
  // 申请人当前绑定的开票资料；null = 未绑定或已被重置
  current_profile: AdminInvoiceProfile | null
}

export interface AdminInvoiceProfile {
  user_id: number
  user_email?: string
  title: string
  tax_no: string
  updated_by?: number
  shared_user_count: number
  created_at: string
  updated_at: string
}

export interface PaginatedInvoiceProfiles {
  items: AdminInvoiceProfile[]
  total: number
  page: number
  page_size: number
  pages: number
}

export async function adminListInvoiceProfiles(params: {
  q?: string
  page?: number
  page_size?: number
}): Promise<PaginatedInvoiceProfiles> {
  const { data } = await apiClient.get<PaginatedInvoiceProfiles>(`${BASE}/profiles`, { params })
  return data
}

// 管理员直接设置/修改用户开票资料（不存在则创建）
export async function adminUpsertInvoiceProfile(
  userId: number,
  payload: { title: string; tax_no: string }
): Promise<AdminInvoiceProfile> {
  const { data } = await apiClient.put<AdminInvoiceProfile>(`${BASE}/profiles/${userId}`, payload)
  return data
}

// 一键重置：删除用户开票资料，用户可重新绑定一次
export async function adminResetInvoiceProfile(userId: number): Promise<void> {
  await apiClient.delete(`${BASE}/profiles/${userId}`)
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
  download: adminDownloadInvoice,
  listProfiles: adminListInvoiceProfiles,
  upsertProfile: adminUpsertInvoiceProfile,
  resetProfile: adminResetInvoiceProfile
}

export default adminInvoiceAPI