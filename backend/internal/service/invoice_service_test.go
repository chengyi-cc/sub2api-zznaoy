//go:build unit

package service

import (
	"bytes"
	"context"
	"database/sql"
	"strconv"
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

// ---- Redeem-code source paths ----

// seedUsedBalanceRedeemCode 给 user 注入一条已使用的余额兑换码，返回它的 ID。
func seedUsedBalanceRedeemCode(t *testing.T, client *dbent.Client, userID int64, code string, value float64) int64 {
	t.Helper()
	now := time.Now()
	rc, err := client.RedeemCode.Create().
		SetCode(code).
		SetType(RedeemTypeBalance).
		SetValue(value).
		SetStatus(StatusUsed).
		SetUsedBy(userID).
		SetUsedAt(now).
		Save(context.Background())
	require.NoError(t, err)
	return rc.ID
}

func TestInvoiceService_ListEligibleSources_ReturnsBalanceCodesAndExcludesOthers(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	ctx := context.Background()

	// Eligible: balance code, used, owned by user, value > 0
	balanceID := seedUsedBalanceRedeemCode(t, client, user.ID, "BAL-1234567890", 50)

	// Ineligible: concurrency type
	_, err := client.RedeemCode.Create().
		SetCode("CONC-1111111111").
		SetType(RedeemTypeConcurrency).
		SetValue(5).
		SetStatus(StatusUsed).
		SetUsedBy(user.ID).
		SetUsedAt(time.Now()).
		Save(ctx)
	require.NoError(t, err)

	// Ineligible: balance but unused
	_, err = client.RedeemCode.Create().
		SetCode("BAL-UNUSED-22222").
		SetType(RedeemTypeBalance).
		SetValue(30).
		SetStatus(StatusUnused).
		Save(ctx)
	require.NoError(t, err)

	// Ineligible: balance used but value=0 (defensive guard)
	_, err = client.RedeemCode.Create().
		SetCode("BAL-ZERO-33333").
		SetType(RedeemTypeBalance).
		SetValue(0).
		SetStatus(StatusUsed).
		SetUsedBy(user.ID).
		SetUsedAt(time.Now()).
		Save(ctx)
	require.NoError(t, err)

	sources, err := svc.ListEligibleSources(ctx, user.ID)
	require.NoError(t, err)
	require.Len(t, sources.RedeemCodes, 1)
	require.Equal(t, balanceID, sources.RedeemCodes[0].RedeemCodeID)
	require.Equal(t, 50.0, sources.RedeemCodes[0].Value)
}

func TestInvoiceService_CreateRequest_AcceptsBalanceRedeemCode(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	codeID := seedUsedBalanceRedeemCode(t, client, user.ID, "BAL-CREATE-AAAA", 80)

	req, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:        user.ID,
		RedeemCodeIDs: []int64{codeID},
		InvoiceType:   domain.InvoiceTypePersonal,
		Title:         "John",
	})
	require.NoError(t, err)
	require.Equal(t, 80.0, req.Amount)
	require.Empty(t, req.PaymentOrderIds)
	require.Equal(t, []int64{codeID}, req.RedeemCodeIds)
}

func TestInvoiceService_CreateRequest_MixedOrderAndRedeemCode_MergesAmounts(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)
	codeID := seedUsedBalanceRedeemCode(t, client, user.ID, "BAL-MIX-AAAAA", 50)

	req, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		RedeemCodeIDs:   []int64{codeID},
		InvoiceType:     domain.InvoiceTypeCompany,
		Title:           "Acme Inc.",
		TaxNo:           "91110000000000000X",
	})
	require.NoError(t, err)
	require.Equal(t, 150.0, req.Amount)
	require.Len(t, req.PaymentOrderIds, 1)
	require.Len(t, req.RedeemCodeIds, 1)
}

func TestInvoiceService_CreateRequest_RejectsNonBalanceRedeemCode(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	rc, err := client.RedeemCode.Create().
		SetCode("CONC-TEST-12345").
		SetType(RedeemTypeConcurrency).
		SetValue(5).
		SetStatus(StatusUsed).
		SetUsedBy(user.ID).
		SetUsedAt(time.Now()).
		Save(context.Background())
	require.NoError(t, err)

	_, err = svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:        user.ID,
		RedeemCodeIDs: []int64{rc.ID},
		InvoiceType:   domain.InvoiceTypePersonal,
		Title:         "John",
	})
	require.ErrorIs(t, err, ErrInvoiceRedeemNotEligible)
}

func TestInvoiceService_CreateRequest_RejectsForeignRedeemCode(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)

	other, err := client.User.Create().
		SetEmail("other-redeem@example.com").
		SetPasswordHash("hash").
		SetRole(RoleUser).
		SetStatus(StatusActive).
		Save(context.Background())
	require.NoError(t, err)

	otherCodeID := seedUsedBalanceRedeemCode(t, client, other.ID, "BAL-OTHER-FFFF", 30)

	_, err = svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:        user.ID,
		RedeemCodeIDs: []int64{otherCodeID},
		InvoiceType:   domain.InvoiceTypePersonal,
		Title:         "John",
	})
	require.ErrorIs(t, err, ErrInvoiceRedeemInvalid)
}

func TestInvoiceService_CreateRequest_RejectsAlreadyClaimedRedeemCode(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	codeID := seedUsedBalanceRedeemCode(t, client, user.ID, "BAL-CLAIM-12345", 60)

	_, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:        user.ID,
		RedeemCodeIDs: []int64{codeID},
		InvoiceType:   domain.InvoiceTypePersonal,
		Title:         "First",
	})
	require.NoError(t, err)

	_, err = svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:        user.ID,
		RedeemCodeIDs: []int64{codeID},
		InvoiceType:   domain.InvoiceTypePersonal,
		Title:         "Second",
	})
	require.ErrorIs(t, err, ErrInvoiceRedeemAlreadyClaimed)
}

func TestInvoiceService_CreateRequest_RejectsEmptySources(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)

	_, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:      user.ID,
		InvoiceType: domain.InvoiceTypePersonal,
		Title:       "No sources",
	})
	require.ErrorIs(t, err, ErrInvoiceSourcesEmpty)
}

// AdminIssue 复核期间，若兑换码 used_by 在 approved → issue 之间被改成别人，
// 必须中止开票（防止把发票开给当前申请人但凭证已不在他名下）。
func TestInvoiceService_AdminIssue_BlocksOnRedeemCodeOwnershipChange(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	codeID := seedUsedBalanceRedeemCode(t, client, user.ID, "BAL-OWN-CHANGE-AAA", 80)

	created, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:        user.ID,
		RedeemCodeIDs: []int64{codeID},
		InvoiceType:   domain.InvoiceTypePersonal,
		Title:         "John",
	})
	require.NoError(t, err)

	_, err = svc.AdminApprove(context.Background(), created.ID, 999)
	require.NoError(t, err)

	// Simulate ownership rewrite (data fix / manual intervention) between approve and issue.
	other, err := client.User.Create().
		SetEmail("rewriter@example.com").
		SetPasswordHash("hash").
		SetRole(RoleUser).
		SetStatus(StatusActive).
		Save(context.Background())
	require.NoError(t, err)
	_, err = client.RedeemCode.UpdateOneID(codeID).
		SetUsedBy(other.ID).
		Save(context.Background())
	require.NoError(t, err)

	_, err = svc.AdminIssue(
		context.Background(), created.ID, 999, "INV-OWN-001",
		bytes.NewReader([]byte("%PDF-1.4 dummy")), int64(len("%PDF-1.4 dummy")),
	)
	require.Error(t, err)
	require.Contains(t, err.Error(), "INVOICE_REDEEM_STATE_CHANGED")

	// Invoice request should remain in approved state, not issued.
	reloaded, err := svc.AdminGet(context.Background(), created.ID)
	require.NoError(t, err)
	require.Equal(t, domain.InvoiceStatusApproved, reloaded.Status)
}

// 与 ownership 变更对称的场景：value 在 approved 期间被人工改成 0（或负数）。
func TestInvoiceService_AdminIssue_BlocksOnRedeemCodeValueZeroed(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	codeID := seedUsedBalanceRedeemCode(t, client, user.ID, "BAL-ZERO-CHANGE-A", 80)

	created, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:        user.ID,
		RedeemCodeIDs: []int64{codeID},
		InvoiceType:   domain.InvoiceTypePersonal,
		Title:         "John",
	})
	require.NoError(t, err)

	_, err = svc.AdminApprove(context.Background(), created.ID, 999)
	require.NoError(t, err)

	_, err = client.RedeemCode.UpdateOneID(codeID).SetValue(0).Save(context.Background())
	require.NoError(t, err)

	_, err = svc.AdminIssue(
		context.Background(), created.ID, 999, "INV-ZERO-001",
		bytes.NewReader([]byte("%PDF-1.4 dummy")), int64(len("%PDF-1.4 dummy")),
	)
	require.Error(t, err)
	require.Contains(t, err.Error(), "INVOICE_REDEEM_STATE_CHANGED")
}

// ---- Min-amount guard ----

// invoiceTestSettingRepo 是最小化的 SettingRepository mock，仅实现测试需要的方法。
type invoiceTestSettingRepo struct {
	values map[string]string
}

func newInvoiceTestSettingRepo(initial map[string]string) *invoiceTestSettingRepo {
	values := make(map[string]string, len(initial))
	for k, v := range initial {
		values[k] = v
	}
	return &invoiceTestSettingRepo{values: values}
}

func (r *invoiceTestSettingRepo) Get(ctx context.Context, key string) (*Setting, error) {
	v, ok := r.values[key]
	if !ok {
		return nil, ErrSettingNotFound
	}
	return &Setting{Key: key, Value: v}, nil
}
func (r *invoiceTestSettingRepo) GetValue(ctx context.Context, key string) (string, error) {
	v, ok := r.values[key]
	if !ok {
		return "", nil
	}
	return v, nil
}
func (r *invoiceTestSettingRepo) Set(ctx context.Context, key, value string) error {
	r.values[key] = value
	return nil
}
func (r *invoiceTestSettingRepo) GetMultiple(ctx context.Context, keys []string) (map[string]string, error) {
	out := make(map[string]string, len(keys))
	for _, k := range keys {
		if v, ok := r.values[k]; ok {
			out[k] = v
		}
	}
	return out, nil
}
func (r *invoiceTestSettingRepo) SetMultiple(ctx context.Context, settings map[string]string) error {
	for k, v := range settings {
		r.values[k] = v
	}
	return nil
}
func (r *invoiceTestSettingRepo) GetAll(ctx context.Context) (map[string]string, error) {
	out := make(map[string]string, len(r.values))
	for k, v := range r.values {
		out[k] = v
	}
	return out, nil
}
func (r *invoiceTestSettingRepo) Delete(ctx context.Context, key string) error {
	delete(r.values, key)
	return nil
}

// withInvoiceMinAmount 给 service 挂一个返回固定 min 值的 SettingService。
func withInvoiceMinAmount(t *testing.T, svc *InvoiceService, minAmount float64) {
	t.Helper()
	repo := newInvoiceTestSettingRepo(map[string]string{
		SettingKeyInvoiceMinAmount: strconv.FormatFloat(minAmount, 'f', -1, 64),
	})
	svc.settingService = NewSettingService(repo, nil)
}

func TestInvoiceService_CreateRequest_BlocksWhenBelowMinAmount(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	withInvoiceMinAmount(t, svc, 200)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100) // 100 < 200

	_, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "INVOICE_AMOUNT_BELOW_MIN")
}

func TestInvoiceService_CreateRequest_AllowsAtOrAboveMinAmount(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	withInvoiceMinAmount(t, svc, 200)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 2, 100) // 100*2 = 200

	req, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.NoError(t, err)
	require.Equal(t, 200.0, req.Amount)
}

func TestInvoiceService_CreateRequest_ZeroMinAmount_NotEnforced(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	withInvoiceMinAmount(t, svc, 0) // 0 = 不限制
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 1)

	req, err := svc.CreateRequest(context.Background(), CreateInvoiceRequestInput{
		UserID:          user.ID,
		PaymentOrderIDs: orderIDs,
		InvoiceType:     domain.InvoiceTypePersonal,
		Title:           "John",
	})
	require.NoError(t, err)
	require.Equal(t, 1.0, req.Amount)
}