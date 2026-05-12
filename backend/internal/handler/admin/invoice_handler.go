package admin

import (
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/handler/dto"
	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"

	"github.com/gin-gonic/gin"
)

// InvoiceHandler 管理员端发票申请 handler
type InvoiceHandler struct {
	invoiceService *service.InvoiceService
}

func NewInvoiceHandler(invoiceService *service.InvoiceService) *InvoiceHandler {
	return &InvoiceHandler{invoiceService: invoiceService}
}

// List GET /api/v1/admin/invoice/requests
func (h *InvoiceHandler) List(c *gin.Context) {
	page, pageSize := response.ParsePagination(c)
	params := service.AdminListInvoiceParams{
		Status:   c.Query("status"),
		Keyword:  c.Query("q"),
		Page:     page,
		PageSize: pageSize,
	}
	if v := c.Query("user_id"); v != "" {
		if uid, err := strconv.ParseInt(v, 10, 64); err == nil && uid > 0 {
			params.UserID = uid
		}
	}
	items, total, err := h.invoiceService.AdminList(c.Request.Context(), params)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Paginated(c, dto.InvoiceRequestListFromEnt(items), int64(total), page, pageSize)
}

// Get GET /api/v1/admin/invoice/requests/:id
func (h *InvoiceHandler) Get(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	r, err := h.invoiceService.AdminGet(c.Request.Context(), id)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, dto.InvoiceRequestFromEnt(r))
}

// Detail GET /api/v1/admin/invoice/requests/:id/detail
// 审核详情：含关联订单 / 兑换码当前状态、申请人邮箱、金额一致性校验。
func (h *InvoiceHandler) Detail(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	d, err := h.invoiceService.AdminGetDetail(c.Request.Context(), id)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	// 直接返回 service.InvoiceRequestDetail（已有 JSON 标签），
	// 但 Request 字段是 ent.InvoiceRequest 含敏感字段（如内部文件路径），
	// 替换成与列表一致的 DTO 视图。
	response.Success(c, gin.H{
		"request":       dto.InvoiceRequestFromEnt(d.Request),
		"user_email":    d.UserEmail,
		"user_name":     d.UserName,
		"orders":        d.Orders,
		"redeem_codes":  d.RedeemCodes,
		"computed_sum":  d.ComputedSum,
		"amount_match":  d.AmountMatch,
		"all_eligible":  d.AllEligible,
		"user_overview": d.UserOverview,
	})
}

// Approve POST /api/v1/admin/invoice/requests/:id/approve
func (h *InvoiceHandler) Approve(c *gin.Context) {
	adminID, ok := h.getAdminID(c)
	if !ok {
		return
	}
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	updated, err := h.invoiceService.AdminApprove(c.Request.Context(), id, adminID)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, dto.InvoiceRequestFromEnt(updated))
}

type rejectInvoicePayload struct {
	Reason string `json:"reason" binding:"required"`
}

// Reject POST /api/v1/admin/invoice/requests/:id/reject
func (h *InvoiceHandler) Reject(c *gin.Context) {
	adminID, ok := h.getAdminID(c)
	if !ok {
		return
	}
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	var body rejectInvoicePayload
	if err := c.ShouldBindJSON(&body); err != nil {
		response.BadRequest(c, "Invalid request: "+err.Error())
		return
	}
	updated, err := h.invoiceService.AdminReject(c.Request.Context(), id, adminID, body.Reason)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, dto.InvoiceRequestFromEnt(updated))
}

// Issue POST /api/v1/admin/invoice/requests/:id/issue
// multipart/form-data: invoice_no (string), file (PDF)
func (h *InvoiceHandler) Issue(c *gin.Context) {
	adminID, ok := h.getAdminID(c)
	if !ok {
		return
	}
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	invoiceNo := strings.TrimSpace(c.PostForm("invoice_no"))
	if invoiceNo == "" {
		response.BadRequest(c, "invoice_no is required")
		return
	}
	fileHeader, err := c.FormFile("file")
	if err != nil {
		response.BadRequest(c, "file is required: "+err.Error())
		return
	}
	// 校验文件类型（粗校验：扩展名 + 上报的 MIME）
	if !isLikelyPDF(fileHeader.Filename, fileHeader.Header.Get("Content-Type")) {
		response.BadRequest(c, "only PDF file is allowed")
		return
	}
	src, err := fileHeader.Open()
	if err != nil {
		response.InternalError(c, "open uploaded file: "+err.Error())
		return
	}
	defer src.Close()

	updated, err := h.invoiceService.AdminIssue(c.Request.Context(), id, adminID, invoiceNo, src, fileHeader.Size)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, dto.InvoiceRequestFromEnt(updated))
}

// Download GET /api/v1/admin/invoice/requests/:id/download
func (h *InvoiceHandler) Download(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	absPath, filename, err := h.invoiceService.OpenInvoiceFileForAdmin(c.Request.Context(), id)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	f, err := os.Open(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			response.NotFound(c, "invoice file not found")
			return
		}
		response.InternalError(c, "open invoice file: "+err.Error())
		return
	}
	defer f.Close()
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

func (h *InvoiceHandler) getAdminID(c *gin.Context) (int64, bool) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "Not authenticated")
		return 0, false
	}
	return subject.UserID, true
}

func isLikelyPDF(filename, mime string) bool {
	if strings.EqualFold(strings.TrimSpace(mime), "application/pdf") {
		return true
	}
	return strings.HasSuffix(strings.ToLower(strings.TrimSpace(filename)), ".pdf")
}