// Package service: invoice_service handles user invoice requests for completed payment orders.
//
// 一期范围：
//   - 用户基于已完成（COMPLETED）订单提交开票申请，支持多单合并
//   - 管理员审核：通过/驳回
//   - 通过后管理员开票：上传 PDF + 填发票号
//   - 邮件通知用户：审核结果 + 开具完成
//   - PDF 仅通过鉴权下载接口分发，不直接暴露公网
package service

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/mail"
	"os"
	"path/filepath"
	"strings"
	"time"

	"entgo.io/ent/dialect"

	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/ent/invoicerequest"
	"github.com/Wei-Shaw/sub2api/ent/paymentorder"
	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/domain"
	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
)

// pdfMagic 是 PDF 1.x 文件头：%PDF-（5 字节）。
var pdfMagic = []byte{0x25, 0x50, 0x44, 0x46, 0x2D}

// detectClientDialect 通过 ent 公开的 Driver() 拿到方言名（例如 "postgres" / "sqlite3"）。
// 用于决定 lockPaymentOrdersByIDs / lockInvoiceRequestByID 是否启用 SELECT ... FOR UPDATE。
func detectClientDialect(c *dbent.Client) string {
	if c == nil || c.Driver() == nil {
		return ""
	}
	return c.Driver().Dialect()
}

// lockPaymentOrdersByIDs 在事务内按 ID 列表查询 payment_orders 并加行锁。
// 仅当方言为 PostgreSQL 时使用 SELECT ... FOR UPDATE；其他方言（如 SQLite 测试
// 环境）会跳过 FOR UPDATE 子句，但仍然在事务内读取，保证应用层 claimed 重检
// 仍能正确发挥作用。
func (s *InvoiceService) lockPaymentOrdersByIDs(ctx context.Context, tx *dbent.Tx, ids []int64) ([]*dbent.PaymentOrder, error) {
	q := tx.PaymentOrder.Query().Where(paymentorder.IDIn(ids...))
	if s.dialect == dialect.Postgres {
		q = q.ForUpdate()
	}
	return q.All(ctx)
}

// lockInvoiceRequestByID 在事务内按 ID 查询发票申请并加行锁（仅 Postgres）。
func (s *InvoiceService) lockInvoiceRequestByID(ctx context.Context, tx *dbent.Tx, id int64) (*dbent.InvoiceRequest, error) {
	q := tx.InvoiceRequest.Query().Where(invoicerequest.IDEQ(id))
	if s.dialect == dialect.Postgres {
		q = q.ForUpdate()
	}
	return q.Only(ctx)
}

var (
	ErrInvoiceRequestNotFound    = infraerrors.NotFound("INVOICE_REQUEST_NOT_FOUND", "invoice request not found")
	ErrInvoiceForbidden          = infraerrors.Forbidden("INVOICE_FORBIDDEN", "no permission for this invoice request")
	ErrInvoiceOrdersEmpty        = infraerrors.BadRequest("INVOICE_ORDERS_EMPTY", "at least one payment order is required")
	ErrInvoiceOrdersInvalid      = infraerrors.BadRequest("INVOICE_ORDERS_INVALID", "one or more payment orders are invalid or not yours")
	ErrInvoiceOrderNotEligible   = infraerrors.BadRequest("INVOICE_ORDER_NOT_ELIGIBLE", "payment order is not eligible for invoice")
	ErrInvoiceOrderAlreadyClaimed = infraerrors.Conflict("INVOICE_ORDER_ALREADY_CLAIMED", "payment order already has an active invoice request")
	ErrInvoiceTaxNoRequired      = infraerrors.BadRequest("INVOICE_TAX_NO_REQUIRED", "tax_no is required for company invoice")
	ErrInvoiceTitleRequired      = infraerrors.BadRequest("INVOICE_TITLE_REQUIRED", "title is required")
	ErrInvoiceTypeInvalid        = infraerrors.BadRequest("INVOICE_TYPE_INVALID", "invoice_type must be personal or company")
	ErrInvoiceEmailInvalid       = infraerrors.BadRequest("INVOICE_EMAIL_INVALID", "recipient_email is invalid")
	ErrInvoiceInvalidStatus      = infraerrors.Conflict("INVOICE_INVALID_STATUS", "invoice request status does not allow this operation")
	ErrInvoiceFileNotFound       = infraerrors.NotFound("INVOICE_FILE_NOT_FOUND", "invoice file not found")
	ErrInvoiceFileInvalid        = infraerrors.BadRequest("INVOICE_FILE_INVALID", "invoice file is invalid (only PDF, max 10MB)")
	ErrInvoiceNoRequired         = infraerrors.BadRequest("INVOICE_NO_REQUIRED", "invoice_no is required")
	ErrInvoiceRejectReason       = infraerrors.BadRequest("INVOICE_REJECT_REASON_REQUIRED", "reject reason is required")
)

const (
	invoiceMaxTitleLen     = 200
	invoiceMaxTaxNoLen     = 50
	invoiceMaxRemarkLen    = 1000
	invoiceMaxFileBytes    = 10 * 1024 * 1024 // 10 MB
	invoiceMaxOrdersPerReq = 50               // 单次最多合并开票订单数
	invoicePDFContentType  = "application/pdf"
)

// CreateInvoiceRequestInput 用户提交开票申请的入参
type CreateInvoiceRequestInput struct {
	UserID          int64
	PaymentOrderIDs []int64
	InvoiceType     string  // personal / company
	Title           string  // 抬头
	TaxNo           string  // 税号（企业必填）
	RecipientEmail  string  // 接收邮箱（可选）
	Remark          string  // 备注
}

// AdminListInvoiceParams 管理员列表查询参数
type AdminListInvoiceParams struct {
	Status   string // 可选过滤
	UserID   int64  // 可选过滤（0 = 不过滤）
	Keyword  string // 抬头/税号/邮箱模糊匹配
	Page     int
	PageSize int
}

// UserListInvoiceParams 用户列表查询参数
type UserListInvoiceParams struct {
	Status   string
	Page     int
	PageSize int
}

// EligibleOrder 可开票订单视图（仅传给前端的最小字段集）
type EligibleOrder struct {
	OrderID     int64     `json:"order_id"`
	OutTradeNo  string    `json:"out_trade_no"`
	Amount      float64   `json:"amount"`
	OrderType   string    `json:"order_type"`
	PaymentType string    `json:"payment_type"`
	CompletedAt time.Time `json:"completed_at"`
}

// InvoiceService 发票申请服务
type InvoiceService struct {
	entClient      *dbent.Client
	userRepo       UserRepository
	emailService   *EmailService
	settingService *SettingService
	dataDir        string // 发票 PDF 存储根目录（绝对路径或相对项目根的路径）
	frontendURL    string // 用于生成邮件中的下载链接
	dialect        string // 数据库方言（用于决定是否启用 SELECT ... FOR UPDATE）
}

// NewInvoiceService 构造函数
func NewInvoiceService(
	entClient *dbent.Client,
	userRepo UserRepository,
	emailService *EmailService,
	settingService *SettingService,
	cfg *config.Config,
) *InvoiceService {
	dataDir := "./data"
	frontendURL := ""
	if cfg != nil {
		if v := strings.TrimSpace(cfg.Pricing.DataDir); v != "" {
			dataDir = v
		}
		frontendURL = strings.TrimSpace(cfg.Server.FrontendURL)
	}
	return &InvoiceService{
		entClient:      entClient,
		userRepo:       userRepo,
		emailService:   emailService,
		settingService: settingService,
		dataDir:        dataDir,
		frontendURL:    frontendURL,
		dialect:        detectClientDialect(entClient),
	}
}

// invoiceRoot 返回发票文件的根目录（绝对化以避免路径穿越）
func (s *InvoiceService) invoiceRoot() (string, error) {
	abs, err := filepath.Abs(filepath.Join(s.dataDir, "invoices"))
	if err != nil {
		return "", fmt.Errorf("resolve invoice root: %w", err)
	}
	return abs, nil
}

// resolveInvoiceFile 把数据库中存储的相对路径还原为绝对路径，并校验它落在 invoiceRoot 之下，防路径穿越
func (s *InvoiceService) resolveInvoiceFile(rel string) (string, error) {
	if strings.TrimSpace(rel) == "" {
		return "", ErrInvoiceFileNotFound
	}
	root, err := s.invoiceRoot()
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(filepath.Join(root, rel))
	if err != nil {
		return "", fmt.Errorf("resolve invoice file: %w", err)
	}
	// 防穿越：abs 必须以 root 为前缀
	if !strings.HasPrefix(abs+string(filepath.Separator), root+string(filepath.Separator)) && abs != root {
		return "", ErrInvoiceFileNotFound
	}
	if _, err := os.Stat(abs); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", ErrInvoiceFileNotFound
		}
		return "", fmt.Errorf("stat invoice file: %w", err)
	}
	return abs, nil
}

// ---- User-side ----

// ListEligibleOrders 列出当前用户可开票的订单。规则：
//   - status == COMPLETED
//   - 未被任何 pending/approved/issued 的发票申请占用
func (s *InvoiceService) ListEligibleOrders(ctx context.Context, userID int64) ([]EligibleOrder, error) {
	orders, err := s.entClient.PaymentOrder.Query().
		Where(
			paymentorder.UserIDEQ(userID),
			paymentorder.StatusEQ(OrderStatusCompleted),
		).
		Order(dbent.Desc(paymentorder.FieldCompletedAt)).
		All(ctx)
	if err != nil {
		return nil, fmt.Errorf("query user completed orders: %w", err)
	}

	claimed, err := s.claimedOrderIDsForUser(ctx, userID)
	if err != nil {
		return nil, err
	}

	out := make([]EligibleOrder, 0, len(orders))
	for _, o := range orders {
		if _, taken := claimed[o.ID]; taken {
			continue
		}
		var completedAt time.Time
		if o.CompletedAt != nil {
			completedAt = *o.CompletedAt
		}
		out = append(out, EligibleOrder{
			OrderID:     o.ID,
			OutTradeNo:  o.OutTradeNo,
			Amount:      o.Amount,
			OrderType:   o.OrderType,
			PaymentType: o.PaymentType,
			CompletedAt: completedAt,
		})
	}
	return out, nil
}

// claimedOrderIDsForUser 返回该用户名下所有处于 pending/approved/issued 状态的发票申请所引用的订单 ID 集合
func (s *InvoiceService) claimedOrderIDsForUser(ctx context.Context, userID int64) (map[int64]struct{}, error) {
	reqs, err := s.entClient.InvoiceRequest.Query().
		Where(
			invoicerequest.UserIDEQ(userID),
			invoicerequest.StatusIn(
				domain.InvoiceStatusPending,
				domain.InvoiceStatusApproved,
				domain.InvoiceStatusIssued,
			),
		).
		All(ctx)
	if err != nil {
		return nil, fmt.Errorf("query active invoice requests: %w", err)
	}
	claimed := make(map[int64]struct{})
	for _, r := range reqs {
		for _, oid := range r.PaymentOrderIds {
			claimed[oid] = struct{}{}
		}
	}
	return claimed, nil
}

// CreateRequest 用户提交发票申请
//
// 并发安全：整个流程在事务内执行，并对所有关联的 payment_orders 加 FOR UPDATE
// 行锁。这样两个并发请求若引用同一订单，后到者会阻塞至前者提交后再读到最新
// claimed 集合并被拒绝，避免重复占用同一订单创建多条 pending 申请。
func (s *InvoiceService) CreateRequest(ctx context.Context, input CreateInvoiceRequestInput) (*dbent.InvoiceRequest, error) {
	if err := s.validateCreateInput(&input); err != nil {
		return nil, err
	}

	tx, err := s.entClient.Tx(ctx)
	if err != nil {
		return nil, fmt.Errorf("begin invoice tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// 锁定所有引用的订单行：先 FOR UPDATE 拿订单，再做归属/状态校验，
	// 锁会一直持有到事务提交，期间其他并发请求会在同一订单上阻塞。
	orders, err := s.lockPaymentOrdersByIDs(ctx, tx, input.PaymentOrderIDs)
	if err != nil {
		return nil, fmt.Errorf("lock payment orders: %w", err)
	}
	if len(orders) != len(input.PaymentOrderIDs) {
		return nil, ErrInvoiceOrdersInvalid
	}

	var totalAmount float64
	for _, o := range orders {
		if o.UserID != input.UserID {
			return nil, ErrInvoiceOrdersInvalid
		}
		if o.Status != OrderStatusCompleted {
			return nil, ErrInvoiceOrderNotEligible
		}
		totalAmount += o.Amount
	}

	// 在事务内重新计算 claimed 集合：因为我们已对订单行加锁，其他并发请求
	// 此时尚未提交，但若先到者已提交了一条占用相同订单的申请，这里会读到。
	claimed, err := s.claimedOrderIDsForUserInTx(ctx, tx, input.UserID)
	if err != nil {
		return nil, err
	}
	for _, oid := range input.PaymentOrderIDs {
		if _, taken := claimed[oid]; taken {
			return nil, ErrInvoiceOrderAlreadyClaimed
		}
	}

	builder := tx.InvoiceRequest.Create().
		SetUserID(input.UserID).
		SetPaymentOrderIds(input.PaymentOrderIDs).
		SetAmount(totalAmount).
		SetInvoiceType(input.InvoiceType).
		SetTitle(input.Title).
		SetStatus(domain.InvoiceStatusPending)

	if input.InvoiceType == domain.InvoiceTypeCompany {
		builder = builder.SetTaxNo(input.TaxNo)
	}
	if v := strings.TrimSpace(input.RecipientEmail); v != "" {
		builder = builder.SetRecipientEmail(v)
	}
	if v := strings.TrimSpace(input.Remark); v != "" {
		builder = builder.SetRemark(v)
	}

	created, err := builder.Save(ctx)
	if err != nil {
		return nil, fmt.Errorf("create invoice request: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit invoice tx: %w", err)
	}
	return created, nil
}

// claimedOrderIDsForUserInTx 与 claimedOrderIDsForUser 同语义，但使用调用者提供的事务，
// 让锁视图保持一致。
func (s *InvoiceService) claimedOrderIDsForUserInTx(ctx context.Context, tx *dbent.Tx, userID int64) (map[int64]struct{}, error) {
	reqs, err := tx.InvoiceRequest.Query().
		Where(
			invoicerequest.UserIDEQ(userID),
			invoicerequest.StatusIn(
				domain.InvoiceStatusPending,
				domain.InvoiceStatusApproved,
				domain.InvoiceStatusIssued,
			),
		).
		All(ctx)
	if err != nil {
		return nil, fmt.Errorf("query active invoice requests in tx: %w", err)
	}
	claimed := make(map[int64]struct{})
	for _, r := range reqs {
		for _, oid := range r.PaymentOrderIds {
			claimed[oid] = struct{}{}
		}
	}
	return claimed, nil
}

func (s *InvoiceService) validateCreateInput(input *CreateInvoiceRequestInput) error {
	input.Title = strings.TrimSpace(input.Title)
	input.TaxNo = strings.TrimSpace(input.TaxNo)
	input.RecipientEmail = strings.TrimSpace(input.RecipientEmail)
	input.Remark = strings.TrimSpace(input.Remark)
	input.InvoiceType = strings.TrimSpace(input.InvoiceType)

	if input.InvoiceType != domain.InvoiceTypePersonal && input.InvoiceType != domain.InvoiceTypeCompany {
		return ErrInvoiceTypeInvalid
	}
	if input.Title == "" {
		return ErrInvoiceTitleRequired
	}
	if len(input.Title) > invoiceMaxTitleLen {
		return ErrInvoiceTitleRequired
	}
	if input.InvoiceType == domain.InvoiceTypeCompany {
		if input.TaxNo == "" {
			return ErrInvoiceTaxNoRequired
		}
		if len(input.TaxNo) > invoiceMaxTaxNoLen {
			return ErrInvoiceTaxNoRequired
		}
	}
	if len(input.Remark) > invoiceMaxRemarkLen {
		return infraerrors.BadRequest("INVOICE_REMARK_TOO_LONG", "remark too long")
	}
	if input.RecipientEmail != "" {
		if _, err := mail.ParseAddress(input.RecipientEmail); err != nil {
			return ErrInvoiceEmailInvalid
		}
	}
	if len(input.PaymentOrderIDs) == 0 {
		return ErrInvoiceOrdersEmpty
	}
	if len(input.PaymentOrderIDs) > invoiceMaxOrdersPerReq {
		return infraerrors.BadRequest("INVOICE_TOO_MANY_ORDERS", fmt.Sprintf("at most %d orders per request", invoiceMaxOrdersPerReq))
	}
	// 去重 + 校验正数
	seen := make(map[int64]struct{}, len(input.PaymentOrderIDs))
	deduped := make([]int64, 0, len(input.PaymentOrderIDs))
	for _, id := range input.PaymentOrderIDs {
		if id <= 0 {
			return ErrInvoiceOrdersInvalid
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		deduped = append(deduped, id)
	}
	input.PaymentOrderIDs = deduped
	return nil
}

// ListUserRequests 用户分页查询自己的申请
func (s *InvoiceService) ListUserRequests(ctx context.Context, userID int64, p UserListInvoiceParams) ([]*dbent.InvoiceRequest, int, error) {
	q := s.entClient.InvoiceRequest.Query().Where(invoicerequest.UserIDEQ(userID))
	if p.Status != "" {
		q = q.Where(invoicerequest.StatusEQ(p.Status))
	}
	total, err := q.Clone().Count(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("count user invoice requests: %w", err)
	}
	ps, pg := applyPagination(p.PageSize, p.Page)
	items, err := q.Order(dbent.Desc(invoicerequest.FieldCreatedAt)).
		Limit(ps).Offset((pg - 1) * ps).All(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("query user invoice requests: %w", err)
	}
	return items, total, nil
}

// GetUserRequest 用户读取单条（带权限校验）
func (s *InvoiceService) GetUserRequest(ctx context.Context, userID, requestID int64) (*dbent.InvoiceRequest, error) {
	r, err := s.entClient.InvoiceRequest.Get(ctx, requestID)
	if err != nil {
		if dbent.IsNotFound(err) {
			return nil, ErrInvoiceRequestNotFound
		}
		return nil, fmt.Errorf("get invoice request: %w", err)
	}
	if r.UserID != userID {
		return nil, ErrInvoiceForbidden
	}
	return r, nil
}

// OpenInvoiceFileForUser 用户下载（仅 issued 状态可下载）。返回 (绝对路径, 下载文件名)
func (s *InvoiceService) OpenInvoiceFileForUser(ctx context.Context, userID, requestID int64) (string, string, error) {
	r, err := s.GetUserRequest(ctx, userID, requestID)
	if err != nil {
		return "", "", err
	}
	return s.openInvoiceFile(r)
}

// OpenInvoiceFileForAdmin 管理员下载（任何 issued 申请均可）
func (s *InvoiceService) OpenInvoiceFileForAdmin(ctx context.Context, requestID int64) (string, string, error) {
	r, err := s.AdminGet(ctx, requestID)
	if err != nil {
		return "", "", err
	}
	return s.openInvoiceFile(r)
}

func (s *InvoiceService) openInvoiceFile(r *dbent.InvoiceRequest) (string, string, error) {
	if r.Status != domain.InvoiceStatusIssued {
		return "", "", ErrInvoiceFileNotFound
	}
	if r.InvoiceFilePath == nil || strings.TrimSpace(*r.InvoiceFilePath) == "" {
		return "", "", ErrInvoiceFileNotFound
	}
	abs, err := s.resolveInvoiceFile(*r.InvoiceFilePath)
	if err != nil {
		return "", "", err
	}
	name := "invoice.pdf"
	if r.InvoiceNo != nil && strings.TrimSpace(*r.InvoiceNo) != "" {
		name = fmt.Sprintf("invoice-%s.pdf", sanitizeFilename(*r.InvoiceNo))
	}
	return abs, name, nil
}

// ---- Admin-side ----

// AdminList 管理员分页查询全部申请
func (s *InvoiceService) AdminList(ctx context.Context, p AdminListInvoiceParams) ([]*dbent.InvoiceRequest, int, error) {
	q := s.entClient.InvoiceRequest.Query()
	if p.Status != "" {
		q = q.Where(invoicerequest.StatusEQ(p.Status))
	}
	if p.UserID > 0 {
		q = q.Where(invoicerequest.UserIDEQ(p.UserID))
	}
	if kw := strings.TrimSpace(p.Keyword); kw != "" {
		q = q.Where(invoicerequest.Or(
			invoicerequest.TitleContainsFold(kw),
			invoicerequest.TaxNoContainsFold(kw),
			invoicerequest.RecipientEmailContainsFold(kw),
			invoicerequest.InvoiceNoContainsFold(kw),
		))
	}
	total, err := q.Clone().Count(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("count invoice requests: %w", err)
	}
	ps, pg := applyPagination(p.PageSize, p.Page)
	items, err := q.Order(dbent.Desc(invoicerequest.FieldCreatedAt)).
		Limit(ps).Offset((pg - 1) * ps).All(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("query invoice requests: %w", err)
	}
	return items, total, nil
}

// AdminGet 管理员读取单条
func (s *InvoiceService) AdminGet(ctx context.Context, requestID int64) (*dbent.InvoiceRequest, error) {
	r, err := s.entClient.InvoiceRequest.Get(ctx, requestID)
	if err != nil {
		if dbent.IsNotFound(err) {
			return nil, ErrInvoiceRequestNotFound
		}
		return nil, fmt.Errorf("get invoice request: %w", err)
	}
	return r, nil
}

// AdminApprove 审核通过：pending -> approved
func (s *InvoiceService) AdminApprove(ctx context.Context, requestID, adminID int64) (*dbent.InvoiceRequest, error) {
	r, err := s.AdminGet(ctx, requestID)
	if err != nil {
		return nil, err
	}
	if r.Status != domain.InvoiceStatusPending {
		return nil, ErrInvoiceInvalidStatus
	}
	updated, err := s.entClient.InvoiceRequest.UpdateOneID(r.ID).
		Where(invoicerequest.StatusEQ(domain.InvoiceStatusPending)).
		SetStatus(domain.InvoiceStatusApproved).
		SetProcessedBy(adminID).
		Save(ctx)
	if err != nil {
		if dbent.IsNotFound(err) {
			return nil, ErrInvoiceInvalidStatus
		}
		return nil, fmt.Errorf("approve invoice request: %w", err)
	}
	s.sendStatusEmailAsync(updated, domain.InvoiceStatusApproved, "")
	return updated, nil
}

// AdminReject 审核驳回：pending -> rejected
func (s *InvoiceService) AdminReject(ctx context.Context, requestID, adminID int64, reason string) (*dbent.InvoiceRequest, error) {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return nil, ErrInvoiceRejectReason
	}
	r, err := s.AdminGet(ctx, requestID)
	if err != nil {
		return nil, err
	}
	if r.Status != domain.InvoiceStatusPending {
		return nil, ErrInvoiceInvalidStatus
	}
	updated, err := s.entClient.InvoiceRequest.UpdateOneID(r.ID).
		Where(invoicerequest.StatusEQ(domain.InvoiceStatusPending)).
		SetStatus(domain.InvoiceStatusRejected).
		SetRejectReason(reason).
		SetProcessedBy(adminID).
		Save(ctx)
	if err != nil {
		if dbent.IsNotFound(err) {
			return nil, ErrInvoiceInvalidStatus
		}
		return nil, fmt.Errorf("reject invoice request: %w", err)
	}
	s.sendStatusEmailAsync(updated, domain.InvoiceStatusRejected, reason)
	return updated, nil
}

// AdminIssue 开具发票：approved -> issued。落盘 PDF + 记录路径 + 发票号。
//
// 并发安全：整个流程在事务内执行。先 FOR UPDATE 锁定关联订单行并校验状态，
// 再锁定发票申请行并校验状态为 approved；然后落盘 PDF；最后在事务内更新
// 发票申请记录。事务回滚时清理已落盘的 PDF，避免出现孤儿文件。
func (s *InvoiceService) AdminIssue(
	ctx context.Context,
	requestID, adminID int64,
	invoiceNo string,
	file io.Reader,
	fileSize int64,
) (*dbent.InvoiceRequest, error) {
	invoiceNo = strings.TrimSpace(invoiceNo)
	if invoiceNo == "" {
		return nil, ErrInvoiceNoRequired
	}
	if fileSize <= 0 || fileSize > invoiceMaxFileBytes {
		return nil, ErrInvoiceFileInvalid
	}

	tx, err := s.entClient.Tx(ctx)
	if err != nil {
		return nil, fmt.Errorf("begin issue tx: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback()
		}
	}()

	// 锁定发票申请行 + 校验状态。
	r, err := s.lockInvoiceRequestByID(ctx, tx, requestID)
	if err != nil {
		if dbent.IsNotFound(err) {
			return nil, ErrInvoiceRequestNotFound
		}
		return nil, fmt.Errorf("lock invoice request: %w", err)
	}
	if r.Status != domain.InvoiceStatusApproved {
		return nil, ErrInvoiceInvalidStatus
	}

	// 锁定关联订单行 + 复核状态。事务持锁直到 commit，确保 issue 期间订单不会
	// 因退款 / 状态变更而被同步修改。
	//
	// 注意：PostgreSQL 不允许在聚合查询上加 FOR UPDATE（"SELECT count(*) ... FOR UPDATE"
	// 在 PG 中会报错）。所以这里取出完整订单列表后再在内存中校验，而不是用 Count()。
	if len(r.PaymentOrderIds) > 0 {
		orders, err := s.lockPaymentOrdersByIDs(ctx, tx, r.PaymentOrderIds)
		if err != nil {
			return nil, fmt.Errorf("lock payment orders for issue: %w", err)
		}
		if len(orders) != len(r.PaymentOrderIds) {
			return nil, infraerrors.Conflict(
				"INVOICE_ORDERS_STATE_CHANGED",
				"one or more referenced orders are no longer in COMPLETED state",
			)
		}
		for _, o := range orders {
			if o.Status != OrderStatusCompleted {
				return nil, infraerrors.Conflict(
					"INVOICE_ORDERS_STATE_CHANGED",
					"one or more referenced orders are no longer in COMPLETED state",
				)
			}
		}
	}

	// 落盘 PDF。事务回滚（或更新失败）时清理文件。
	relPath, absPath, err := s.savePDF(file, fileSize)
	if err != nil {
		return nil, err
	}
	cleanupFile := func() { _ = os.Remove(absPath) }

	now := time.Now()
	updated, err := tx.InvoiceRequest.UpdateOneID(r.ID).
		Where(invoicerequest.StatusEQ(domain.InvoiceStatusApproved)).
		SetStatus(domain.InvoiceStatusIssued).
		SetInvoiceNo(invoiceNo).
		SetInvoiceFilePath(relPath).
		SetIssuedAt(now).
		SetProcessedBy(adminID).
		Save(ctx)
	if err != nil {
		cleanupFile()
		if dbent.IsNotFound(err) {
			return nil, ErrInvoiceInvalidStatus
		}
		return nil, fmt.Errorf("issue invoice request: %w", err)
	}

	if err := tx.Commit(); err != nil {
		cleanupFile()
		return nil, fmt.Errorf("commit issue tx: %w", err)
	}
	committed = true

	s.sendStatusEmailAsync(updated, domain.InvoiceStatusIssued, "")
	return updated, nil
}

// savePDF 流式写入 PDF，返回 (相对 invoiceRoot 的存储路径, 绝对路径)。
//
// 内容级校验：读取前 5 字节并校验 PDF 文件头 (%PDF-)，避免仅依赖客户端上报的
// Content-Type / 文件名而落盘任意文件。
func (s *InvoiceService) savePDF(src io.Reader, declaredSize int64) (string, string, error) {
	// 用 bufio 读 PDF 头但保留剩余内容供后续 io.Copy 使用。
	br := bufio.NewReader(src)
	head, err := br.Peek(len(pdfMagic))
	if err != nil {
		return "", "", ErrInvoiceFileInvalid
	}
	if !bytes.Equal(head, pdfMagic) {
		return "", "", ErrInvoiceFileInvalid
	}

	root, err := s.invoiceRoot()
	if err != nil {
		return "", "", err
	}
	now := time.Now()
	subdir := filepath.Join(fmt.Sprintf("%04d", now.Year()), fmt.Sprintf("%02d", int(now.Month())))
	dir := filepath.Join(root, subdir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", "", fmt.Errorf("mkdir invoice dir: %w", err)
	}
	id := randomHex(16)
	filename := id + ".pdf"
	absPath := filepath.Join(dir, filename)
	rel := filepath.ToSlash(filepath.Join(subdir, filename))

	f, err := os.OpenFile(absPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return "", "", fmt.Errorf("create invoice file: %w", err)
	}
	defer f.Close()

	// 限制读取上限，避免客户端撒谎的 Content-Length
	limit := declaredSize
	if limit <= 0 || limit > invoiceMaxFileBytes {
		limit = invoiceMaxFileBytes
	}
	written, err := io.Copy(f, io.LimitReader(br, limit+1))
	if err != nil {
		_ = os.Remove(absPath)
		return "", "", fmt.Errorf("write invoice file: %w", err)
	}
	if written > invoiceMaxFileBytes {
		_ = os.Remove(absPath)
		return "", "", ErrInvoiceFileInvalid
	}
	if written == 0 {
		_ = os.Remove(absPath)
		return "", "", ErrInvoiceFileInvalid
	}
	return rel, absPath, nil
}

// ---- Notifications ----

// sendStatusEmailAsync 异步发送状态变更邮件（best-effort，失败仅记录日志）
func (s *InvoiceService) sendStatusEmailAsync(r *dbent.InvoiceRequest, newStatus, rejectReason string) {
	if s.emailService == nil {
		return
	}
	// 复制 invoice 上必要的字段，避免并发竞争
	userID := r.UserID
	requestID := r.ID
	amount := r.Amount
	title := r.Title
	invoiceNo := ""
	if r.InvoiceNo != nil {
		invoiceNo = *r.InvoiceNo
	}
	var recipient string
	if r.RecipientEmail != nil {
		recipient = strings.TrimSpace(*r.RecipientEmail)
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		// 解析收件地址
		to := recipient
		if to == "" {
			user, err := s.userRepo.GetByID(ctx, userID)
			if err != nil || user == nil || user.Email == "" {
				logger.LegacyPrintf("service.invoice", "lookup user email failed for invoice %d: %v", requestID, err)
				return
			}
			to = user.Email
		}

		siteName := "Sub2API"
		if s.settingService != nil {
			if n := s.settingService.GetSiteName(ctx); n != "" {
				siteName = n
			}
		}

		subject, body := s.buildStatusEmail(siteName, newStatus, rejectReason, requestID, amount, title, invoiceNo)
		if err := s.emailService.SendEmail(ctx, to, subject, body); err != nil {
			logger.LegacyPrintf("service.invoice", "send invoice status email failed (request %d, status %s): %v", requestID, newStatus, err)
		}
	}()
}

func (s *InvoiceService) buildStatusEmail(siteName, status, rejectReason string, requestID int64, amount float64, title, invoiceNo string) (string, string) {
	var subject, headline string
	switch status {
	case domain.InvoiceStatusApproved:
		subject = fmt.Sprintf("[%s] 您的发票申请已通过审核", siteName)
		headline = "您的发票申请已通过审核，我们会尽快为您开具发票。"
	case domain.InvoiceStatusRejected:
		subject = fmt.Sprintf("[%s] 您的发票申请被驳回", siteName)
		headline = "很抱歉，您的发票申请被驳回。"
	case domain.InvoiceStatusIssued:
		subject = fmt.Sprintf("[%s] 您的发票已开具", siteName)
		headline = "您的发票已开具完成，可登录后在「发票管理」中下载。"
	default:
		subject = fmt.Sprintf("[%s] 发票申请状态更新", siteName)
		headline = "您的发票申请状态有更新。"
	}

	var b strings.Builder
	b.WriteString("<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 560px; margin: 0 auto; padding: 24px; color: #1f2937;\">")
	fmt.Fprintf(&b, "<h2 style=\"margin: 0 0 16px;\">%s</h2>", escapeHTML(siteName))
	fmt.Fprintf(&b, "<p style=\"margin: 0 0 12px; font-size: 14px;\">%s</p>", escapeHTML(headline))

	b.WriteString("<table style=\"width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 16px;\">")
	fmt.Fprintf(&b, "<tr><td style=\"padding: 6px 0; color: #6b7280; width: 120px;\">申请编号</td><td>%d</td></tr>", requestID)
	fmt.Fprintf(&b, "<tr><td style=\"padding: 6px 0; color: #6b7280;\">抬头</td><td>%s</td></tr>", escapeHTML(title))
	fmt.Fprintf(&b, "<tr><td style=\"padding: 6px 0; color: #6b7280;\">金额</td><td>%.2f</td></tr>", amount)
	if invoiceNo != "" {
		fmt.Fprintf(&b, "<tr><td style=\"padding: 6px 0; color: #6b7280;\">发票号</td><td>%s</td></tr>", escapeHTML(invoiceNo))
	}
	if status == domain.InvoiceStatusRejected && rejectReason != "" {
		fmt.Fprintf(&b, "<tr><td style=\"padding: 6px 0; color: #6b7280;\">驳回原因</td><td>%s</td></tr>", escapeHTML(rejectReason))
	}
	b.WriteString("</table>")

	if status == domain.InvoiceStatusIssued && s.frontendURL != "" {
		url := strings.TrimRight(s.frontendURL, "/") + "/invoices"
		fmt.Fprintf(&b, "<p style=\"margin-top: 24px;\"><a href=\"%s\" style=\"display: inline-block; padding: 10px 18px; background: #2563eb; color: white; text-decoration: none; border-radius: 6px; font-size: 14px;\">查看并下载发票</a></p>", escapeHTML(url))
	}

	b.WriteString("<p style=\"margin-top: 32px; color: #9ca3af; font-size: 12px;\">此邮件由系统自动发送，请勿直接回复。</p>")
	b.WriteString("</div>")
	return subject, b.String()
}

// ---- helpers ----

func sanitizeFilename(s string) string {
	s = strings.TrimSpace(s)
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z',
			r >= 'A' && r <= 'Z',
			r >= '0' && r <= '9',
			r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := b.String()
	if out == "" {
		return "invoice"
	}
	return out
}

func escapeHTML(s string) string {
	r := strings.NewReplacer(
		"&", "&amp;",
		"<", "&lt;",
		">", "&gt;",
		"\"", "&quot;",
		"'", "&#39;",
	)
	return r.Replace(s)
}