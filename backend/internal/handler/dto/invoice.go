package dto

import (
	"time"

	dbent "github.com/Wei-Shaw/sub2api/ent"
)

// InvoiceRequest 发票申请 DTO（用户/管理员通用响应）
type InvoiceRequest struct {
	ID              int64      `json:"id"`
	UserID          int64      `json:"user_id"`
	PaymentOrderIDs []int64    `json:"payment_order_ids"`
	RedeemCodeIDs   []int64    `json:"redeem_code_ids"`
	Amount          float64    `json:"amount"`
	InvoiceType     string     `json:"invoice_type"`
	Title           string     `json:"title"`
	TaxNo           string     `json:"tax_no,omitempty"`
	RecipientEmail  string     `json:"recipient_email,omitempty"`
	Remark          string     `json:"remark,omitempty"`
	Status          string     `json:"status"`
	RejectReason    string     `json:"reject_reason,omitempty"`
	InvoiceNo       string     `json:"invoice_no,omitempty"`
	HasFile         bool       `json:"has_file"`
	IssuedAt        *time.Time `json:"issued_at,omitempty"`
	ProcessedBy     int64      `json:"processed_by,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at"`

	// UserEmail 申请人注册邮箱（仅列表/管理端补充，方便审核判断；ent 记录本身不含）
	UserEmail string `json:"user_email,omitempty"`
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
		RedeemCodeIDs:   append([]int64(nil), r.RedeemCodeIds...),
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

// InvoiceProfile 用户端开票资料视图。绑定后即锁定，用户只读。
type InvoiceProfile struct {
	Title     string    `json:"title"`
	TaxNo     string    `json:"tax_no"`
	Locked    bool      `json:"locked"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	// AdminModified 资料是否由管理员设置/修改过（不暴露管理员 ID）
	AdminModified bool `json:"admin_modified"`
}

// InvoiceProfileFromEnt 转换为用户端视图；nil 表示未绑定。
func InvoiceProfileFromEnt(p *dbent.InvoiceProfile) *InvoiceProfile {
	if p == nil {
		return nil
	}
	return &InvoiceProfile{
		Title:         p.Title,
		TaxNo:         p.TaxNo,
		Locked:        true,
		CreatedAt:     p.CreatedAt,
		UpdatedAt:     p.UpdatedAt,
		AdminModified: p.UpdatedBy != nil,
	}
}

// AdminInvoiceProfile 管理端开票资料视图。
type AdminInvoiceProfile struct {
	UserID          int64     `json:"user_id"`
	UserEmail       string    `json:"user_email,omitempty"`
	Title           string    `json:"title"`
	TaxNo           string    `json:"tax_no"`
	UpdatedBy       int64     `json:"updated_by,omitempty"`
	SharedUserCount int       `json:"shared_user_count"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// AdminInvoiceProfileFromEnt 转换为管理端视图；nil 表示未绑定。
func AdminInvoiceProfileFromEnt(p *dbent.InvoiceProfile) *AdminInvoiceProfile {
	if p == nil {
		return nil
	}
	d := &AdminInvoiceProfile{
		UserID:    p.UserID,
		Title:     p.Title,
		TaxNo:     p.TaxNo,
		CreatedAt: p.CreatedAt,
		UpdatedAt: p.UpdatedAt,
	}
	if p.UpdatedBy != nil {
		d.UpdatedBy = *p.UpdatedBy
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
