//go:build unit

package service

import (
	"bytes"
	"context"
	"database/sql"
	"strings"
	"testing"
	"time"

	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/ent/enttest"
	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/domain"

	"entgo.io/ent/dialect"
	entsql "entgo.io/ent/dialect/sql"
	_ "modernc.org/sqlite"
	"github.com/stretchr/testify/require"
)

func newInvoiceServiceTestClient(t *testing.T) (*InvoiceService, *dbent.Client) {
	t.Helper()

	// 每个测试独立的 in-memory DB，避免互相污染
	dsn := "file:" + t.Name() + "?mode=memory&cache=shared"
	db, err := sql.Open("sqlite", dsn)
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })

	_, err = db.Exec("PRAGMA foreign_keys = ON")
	require.NoError(t, err)

	drv := entsql.OpenDB(dialect.SQLite, db)
	client := enttest.NewClient(t, enttest.WithOptions(dbent.Driver(drv)))
	t.Cleanup(func() { _ = client.Close() })

	cfg := &config.Config{}
	cfg.Pricing.DataDir = t.TempDir()

	svc := NewInvoiceService(client, nil, nil, nil, cfg)
	return svc, client
}

// seedUserAndCompletedOrders 创建用户 + N 个 COMPLETED 订单，返回 user 和订单 ID 列表。
func seedUserAndCompletedOrders(t *testing.T, client *dbent.Client, count int, amountEach float64) (*dbent.User, []int64) {
	t.Helper()
	ctx := context.Background()

	user, err := client.User.Create().
		SetEmail("invoice-test@example.com").
		SetPasswordHash("hash").
		SetRole(RoleUser).
		SetStatus(StatusActive).
		Save(ctx)
	require.NoError(t, err)

	ids := make([]int64, 0, count)
	for i := 0; i < count; i++ {
		o, err := client.PaymentOrder.Create().
			SetUserID(user.ID).
			SetUserEmail(user.Email).
			SetUserName("test").
			SetAmount(amountEach).
			SetPayAmount(amountEach).
			SetRechargeCode("").
			SetPaymentType("alipay").
			SetPaymentTradeNo("").
			SetStatus(OrderStatusCompleted).
			SetExpiresAt(time.Now().Add(1 * time.Hour)).
			SetClientIP("127.0.0.1").
			SetSrcHost("localhost").
			Save(ctx)
		require.NoError(t, err)
		ids = append(ids, o.ID)
	}
	return user, ids
}

// ---- CreateRequest validation tests ----

func TestInvoiceService_CreateRequest_RejectsCompanyWithoutTaxNo(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	_, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypeCompany,
		Title:           "Acme Inc.",
		// TaxNo intentionally empty
	})
	require.ErrorIs(t, err, ErrInvoiceTaxNoRequired)
}

func TestInvoiceService_CreateRequest_RejectsInvalidEmail(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	_, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John Doe",
		RecipientEmail:  "not-an-email",
	})
	require.ErrorIs(t, err, ErrInvoiceEmailInvalid)
}

func TestInvoiceService_CreateRequest_RejectsForeignOrder(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 1, 100)

	// 另一个用户的订单
	otherUser, err := client.User.Create().
		SetEmail("other@example.com").
		SetPasswordHash("hash").
		SetRole(RoleUser).
		SetStatus(StatusActive).
		Save(context.Background())
	require.NoError(t, err)

	otherOrder, err := client.PaymentOrder.Create().
		SetUserID(otherUser.ID).
		SetUserEmail(otherUser.Email).
		SetUserName("other").
		SetAmount(50).
		SetPayAmount(50).
		SetRechargeCode("").
		SetPaymentType("alipay").
		SetPaymentTradeNo("").
		SetStatus(OrderStatusCompleted).
		SetExpiresAt(time.Now().Add(1 * time.Hour)).
		SetClientIP("127.0.0.1").
		SetSrcHost("localhost").
		Save(context.Background())
	require.NoError(t, err)

	_, err = svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID, // 用 user.ID 试图占用 otherOrder.ID
		PaymentOrderIDs: []int64{otherOrder.ID},
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.ErrorIs(t, err, ErrInvoiceOrdersInvalid)
}

func TestInvoiceService_CreateRequest_RejectsNonCompletedOrder(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)

	pendingOrder, err := client.PaymentOrder.Create().
		SetUserID(user.ID).
		SetUserEmail(user.Email).
		SetUserName("u").
		SetAmount(100).
		SetPayAmount(100).
		SetRechargeCode("").
		SetPaymentType("alipay").
		SetPaymentTradeNo("").
		SetStatus(OrderStatusPending).
		SetExpiresAt(time.Now().Add(1 * time.Hour)).
		SetClientIP("127.0.0.1").
		SetSrcHost("localhost").
		Save(context.Background())
	require.NoError(t, err)

	_, err = svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: []int64{pendingOrder.ID},
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.ErrorIs(t, err, ErrInvoiceOrderNotEligible)
}

// 验证 "同一订单已被一条 pending 申请占用时，第二次申请被拒"。
// 这覆盖了关键的 claimed-set 校验逻辑（事务里的"先到先得"，因为 SQLite
// 的 FOR UPDATE 是 no-op，所以这里测的是应用层重检语义，而不是 PG 的行锁）。
func TestInvoiceService_CreateRequest_RejectsAlreadyClaimedOrder(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	_, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "First Request",
	})
	require.NoError(t, err)

	_, err = svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs, // 同一订单
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "Second Request",
	})
	require.ErrorIs(t, err, ErrInvoiceOrderAlreadyClaimed)
}

func TestInvoiceService_CreateRequest_AllowsMultiOrderMerge(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 3, 50)

	req, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypeCompany,
		Title:           "Acme Inc.",
		TaxNo:           "91110000000000000X",
		RecipientEmail:  "billing@acme.example",
		Remark:          "Q1 invoices",
	})
	require.NoError(t, err)
	require.Equal(t, 150.0, req.Amount)
	require.Equal(t, domain.InvoiceStatusPending, req.Status)
	require.Len(t, req.PaymentOrderIds, 3)
}

// ---- State machine ----

func TestInvoiceService_AdminReject_RequiresPendingStatus(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	created, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.NoError(t, err)

	// First approval moves it out of pending
	_, err = svc.AdminApprove(context.Background(), created.ID, 999)
	require.NoError(t, err)

	// Now reject should be rejected (state machine guard)
	_, err = svc.AdminReject(context.Background(), created.ID, 999, "too late")
	require.ErrorIs(t, err, ErrInvoiceInvalidStatus)
}

func TestInvoiceService_AdminIssue_RequiresApprovedStatus(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	created, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.NoError(t, err)

	// Try to issue a pending (not approved) request — should fail state-machine check
	_, err = svc.AdminIssue(
		context.Background(), created.ID, 999, "INV-001",
		bytes.NewReader([]byte("%PDF-1.4 stub content")), int64(len("%PDF-1.4 stub content")),
	)
	require.ErrorIs(t, err, ErrInvoiceInvalidStatus)
}

// 验证 issue 时若关联订单已退款（不再是 COMPLETED），整个事务回滚，
// 既不更新 invoice 状态，也不留下孤儿 PDF 文件。
func TestInvoiceService_AdminIssue_BlocksOnRefundedOrder(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	created, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.NoError(t, err)

	_, err = svc.AdminApprove(context.Background(), created.ID, 999)
	require.NoError(t, err)

	// Simulate refund happening between approve and issue.
	_, err = client.PaymentOrder.UpdateOneID(orderIDs[0]).
		SetStatus(OrderStatusRefunded).
		Save(context.Background())
	require.NoError(t, err)

	_, err = svc.AdminIssue(
		context.Background(), created.ID, 999, "INV-002",
		bytes.NewReader([]byte("%PDF-1.4 dummy")), int64(len("%PDF-1.4 dummy")),
	)
	require.Error(t, err)
	require.Contains(t, err.Error(), "INVOICE_ORDERS_STATE_CHANGED")

	// Invoice request should remain in approved state, not issued.
	reloaded, err := svc.AdminGet(context.Background(), created.ID)
	require.NoError(t, err)
	require.Equal(t, domain.InvoiceStatusApproved, reloaded.Status)
}

// ---- PDF magic number ----

func TestInvoiceService_AdminIssue_RejectsNonPDFContent(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	created, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.NoError(t, err)

	_, err = svc.AdminApprove(context.Background(), created.ID, 999)
	require.NoError(t, err)

	// Wrong magic — even though the filename/MIME would say PDF, the bytes don't.
	bogus := []byte("GIF89a fake-content")
	_, err = svc.AdminIssue(
		context.Background(), created.ID, 999, "INV-003",
		bytes.NewReader(bogus), int64(len(bogus)),
	)
	require.ErrorIs(t, err, ErrInvoiceFileInvalid)

	// Invoice should remain approved (not advanced to issued).
	reloaded, err := svc.AdminGet(context.Background(), created.ID)
	require.NoError(t, err)
	require.Equal(t, domain.InvoiceStatusApproved, reloaded.Status)
}

func TestInvoiceService_AdminIssue_AcceptsValidPDFHeader(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)

	created, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.NoError(t, err)

	_, err = svc.AdminApprove(context.Background(), created.ID, 999)
	require.NoError(t, err)

	// Minimal but valid PDF header
	content := []byte("%PDF-1.7\n%dummy trailer\n")
	issued, err := svc.AdminIssue(
		context.Background(), created.ID, 999, "INV-100",
		bytes.NewReader(content), int64(len(content)),
	)
	require.NoError(t, err)
	require.Equal(t, domain.InvoiceStatusIssued, issued.Status)
	require.NotNil(t, issued.InvoiceNo)
	require.Equal(t, "INV-100", *issued.InvoiceNo)
	require.NotNil(t, issued.InvoiceFilePath)
	require.True(t, strings.HasSuffix(*issued.InvoiceFilePath, ".pdf"))
}