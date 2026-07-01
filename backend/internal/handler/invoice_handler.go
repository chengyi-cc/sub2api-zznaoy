package handler

import (
	"io"
	"net/http"
	"os"
	"strconv"

	"github.com/Wei-Shaw/sub2api/internal/handler/dto"
	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"

	"github.com/gin-gonic/gin"
)

// InvoiceHandler 用户端发票申请 handler
type InvoiceHandler struct {
	invoiceService *service.InvoiceService
}

func NewInvoiceHandler(invoiceService *service.InvoiceService) *InvoiceHandler {
	return &InvoiceHandler{invoiceService: invoiceService}
}

// createInvoiceRequestPayload 用户提交发票申请的请求体
type createInvoiceRequestPayload struct {
	PaymentOrderIDs []int64 `json:"payment_order_ids"`
	RedeemCodeIDs   []int64 `json:"redeem_code_ids"`
	InvoiceType     string  `json:"invoice_type" binding:"required"`
	Title           string  `json:"title" binding:"required"`
	TaxNo           string  `json:"tax_no"`
	RecipientEmail  string  `json:"recipient_email"`
	Remark          string  `json:"remark"`
}

// ListEligibleOrders GET /api/v1/invoice/eligible-orders
//
// Deprecated: 仅返回订单的旧接口，新前端应使用 ListEligibleSources。
func (h *InvoiceHandler) ListEligibleOrders(c *gin.Context) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "User not authenticated")
		return
	}
	orders, err := h.invoiceService.ListEligibleOrders(c.Request.Context(), subject.UserID)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, orders)
}

// ListEligibleSources GET /api/v1/invoice/eligible-sources
// 同时返回可开票订单 + 余额兑换码。
func (h *InvoiceHandler) ListEligibleSources(c *gin.Context) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "User not authenticated")
		return
	}
	sources, err := h.invoiceService.ListEligibleSources(c.Request.Context(), subject.UserID)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, sources)
}

// Create POST /api/v1/invoice/requests
func (h *InvoiceHandler) Create(c *gin.Context) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "User not authenticated")
		return
	}
	var body createInvoiceRequestPayload
	if err := c.ShouldBindJSON(&body); err != nil {
		response.BadRequest(c, "Invalid request: "+err.Error())
		return
	}
	created, err := h.invoiceService.CreateRequest(c.Request.Context(), service.CreateInvoiceRequestInput{
		UserID:          subject.UserID,
		PaymentOrderIDs: body.PaymentOrderIDs,
		RedeemCodeIDs:   body.RedeemCodeIDs,
		InvoiceType:     body.InvoiceType,
		Title:           body.Title,
		TaxNo:           body.TaxNo,
		RecipientEmail:  body.RecipientEmail,
		Remark:          body.Remark,
	})
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Created(c, dto.InvoiceRequestFromEnt(created))
}

// List GET /api/v1/invoice/requests
func (h *InvoiceHandler) List(c *gin.Context) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "User not authenticated")
		return
	}
	page, pageSize := response.ParsePagination(c)
	status := c.Query("status")
	items, total, err := h.invoiceService.ListUserRequests(c.Request.Context(), subject.UserID, service.UserListInvoiceParams{
		Status:   status,
		Page:     page,
		PageSize: pageSize,
	})
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Paginated(c, dto.InvoiceRequestListFromEnt(items), int64(total), page, pageSize)
}

// Get GET /api/v1/invoice/requests/:id
func (h *InvoiceHandler) Get(c *gin.Context) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "User not authenticated")
		return
	}
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	r, err := h.invoiceService.GetUserRequest(c.Request.Context(), subject.UserID, id)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, dto.InvoiceRequestFromEnt(r))
}

// Download GET /api/v1/invoice/requests/:id/download
func (h *InvoiceHandler) Download(c *gin.Context) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "User not authenticated")
		return
	}
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	absPath, filename, err := h.invoiceService.OpenInvoiceFileForUser(c.Request.Context(), subject.UserID, id)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	streamInvoiceFile(c, absPath, filename)
}

// streamInvoiceFile 流式下发 PDF，强制 attachment 头
func streamInvoiceFile(c *gin.Context, absPath, filename string) {
	f, err := os.Open(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			response.NotFound(c, "invoice file not found")
			return
		}
		response.InternalError(c, "open invoice file: "+err.Error())
		return
	}
	defer func() { _ = f.Close() }()
	stat, err := f.Stat()
	if err != nil {
		response.InternalError(c, "stat invoice file: "+err.Error())
		return
	}
	c.Header("Content-Type", "application/pdf")
	c.Header("Content-Disposition", `attachment; filename="`+filename+`"`)
	c.Header("Content-Length", strconv.FormatInt(stat.Size(), 10))
	c.Header("Cache-Control", "private, no-store")
	c.Header("X-Content-Type-Options", "nosniff")
	c.Status(http.StatusOK)
	_, _ = io.Copy(c.Writer, f)
}
