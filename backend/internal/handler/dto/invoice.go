package dto

import (
	"time"

	dbent "github.com/Wei-Shaw/sub2api/ent"
)

// InvoiceRequest 发票申请 DTO（用户/管理员通用响应）
type InvoiceRequest struct {
	ID              int64     `json:"id"`
	UserID          int64     `json:"user_id"`
	PaymentOrderIDs []int64   `json:"payment_order_ids"`
	Amount          float64   `json:"amount"`
	InvoiceType     string    `json:"invoice_type"`
	Title           string    `json:"title"`
	TaxNo           string    `json:"tax_no,omitempty"`
	RecipientEmail  string    `json:"recipient_email,omitempty"`
	Remark          string    `json:"remark,omitempty"`
	Status          string    `json:"status"`
	RejectReason    string    `json:"reject_reason,omitempty"`
	InvoiceNo       string    `json:"invoice_no,omitempty"`
	HasFile         bool      `json:"has_file"`
	IssuedAt        *time.Time `json:"issued_at,omitempty"`
	ProcessedBy     int64      `json:"processed_by,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at"`
}

// InvoiceRequestFromEnt 将 ent.InvoiceRequest 转为 DTO
func InvoiceRequestFromEnt(r *dbent.InvoiceRequest) *InvoiceRequest {
	if r == nil {
		return nil
	}
	d := &InvoiceRequest{
		ID:              r.ID,
		UserID:          r.UserID,
		PaymentOrderIDs: append([]int64(nil), r.PaymentOrderIds...),
		Amount:          r.Amount,
		InvoiceType:     r.InvoiceType,
		Title:           r.Title,
		Status:          r.Status,
		CreatedAt:       r.CreatedAt,
		UpdatedAt:       r.UpdatedAt,
	}
	if r.TaxNo != nil {
		d.TaxNo = *r.TaxNo
	}
	if r.RecipientEmail != nil {
		d.RecipientEmail = *r.RecipientEmail
	}
	if r.Remark != nil {
		d.Remark = *r.Remark
	}
	if r.RejectReason != nil {
		d.RejectReason = *r.RejectReason
	}
	if r.InvoiceNo != nil {
		d.InvoiceNo = *r.InvoiceNo
	}
	if r.InvoiceFilePath != nil && *r.InvoiceFilePath != "" {
		d.HasFile = true
	}
	if r.IssuedAt != nil {
		d.IssuedAt = r.IssuedAt
	}
	if r.ProcessedBy != nil {
		d.ProcessedBy = *r.ProcessedBy
	}
	return d
}

// InvoiceRequestListFromEnt 批量转换
func InvoiceRequestListFromEnt(rs []*dbent.InvoiceRequest) []*InvoiceRequest {
	out := make([]*InvoiceRequest, 0, len(rs))
	for _, r := range rs {
		out = append(out, InvoiceRequestFromEnt(r))
	}
	return out
}