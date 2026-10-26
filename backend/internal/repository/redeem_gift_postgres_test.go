//go:build unit

package repository

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"entgo.io/ent/dialect"
	entsql "entgo.io/ent/dialect/sql"
	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/ent/redeemcode"
	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/domain"
	"github.com/Wei-Shaw/sub2api/internal/handler"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/lib/pq"
	"github.com/stretchr/testify/require"
)

type failingGiftCreditRepo struct {
	service.UserRepository
	gift service.GiftBalanceRepository
}

func (r failingGiftCreditRepo) CreditGiftBalance(ctx context.Context, id int64, amount float64) error {
	if err := r.gift.CreditGiftBalance(ctx, id, amount); err != nil {
		return err
	}
	return errors.New("forced failure after balance update")
}

// Requires an explicitly supplied disposable server. It never reads application
// configuration and creates/drops its own database for every run.
func TestRedeemGiftPostgres(t *testing.T) {
	dsn := os.Getenv("SUB2API_GIFT_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("set SUB2API_GIFT_TEST_POSTGRES_DSN to test on a disposable PostgreSQL server")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	adminDB, err := sql.Open("postgres", dsn)
	require.NoError(t, err)
	defer adminDB.Close()
	name := fmt.Sprintf("gift_test_%d", time.Now().UnixNano())
	_, err = adminDB.ExecContext(ctx, "CREATE DATABASE "+pq.QuoteIdentifier(name))
	require.NoError(t, err)
	defer func() {
		_, err := adminDB.ExecContext(context.Background(), "DROP DATABASE "+pq.QuoteIdentifier(name)+" WITH (FORCE)")
		require.NoError(t, err)
	}()
	db, err := sql.Open("postgres", dsn+" dbname="+name)
	require.NoError(t, err)
	client := dbent.NewClient(dbent.Driver(entsql.OpenDB(dialect.Postgres, db)))
	defer client.Close()
	require.NoError(t, client.Schema.Create(ctx))
	for _, migrationName := range []string{"247_redeem_code_gifts.sql", "248_redeem_gift_consumption_guard.sql"} {
		migration, err := os.ReadFile("../../migrations/" + migrationName)
		require.NoError(t, err)
		_, err = db.ExecContext(ctx, string(migration))
		require.NoError(t, err)
	}
	repo := NewRedeemCodeRepository(client)
	users := NewUserRepository(client, db)
	adminSvc := service.NewAdminService(nil, users, nil, nil, nil, nil, repo, nil, nil, nil, nil, nil, nil, client, nil, nil, nil, nil, nil, nil, nil, nil, nil)
	expiry := time.Now().UTC().Add(time.Hour)
	codes, err := adminSvc.GenerateRedeemCodes(ctx, &service.GenerateRedeemCodesInput{
		Count: 2, Type: service.RedeemTypeBalance, Value: 10, IncludeGift: true, ExpiresAt: &expiry, OperationKey: "gift-postgres-generation",
	})
	require.NoError(t, err)
	require.Len(t, codes, 4)
	paid, gift := codes[0], codes[1]
	// Recover the exact original secrets after a response/cache failure.
	replay, err := adminSvc.GenerateRedeemCodes(ctx, &service.GenerateRedeemCodesInput{
		Count: 2, Type: service.RedeemTypeBalance, Value: 10, IncludeGift: true, OperationKey: "gift-postgres-generation",
	})
	require.NoError(t, err)
	require.Len(t, replay, len(codes))
	for i := range codes {
		require.Equal(t, codes[i].Code, replay[i].Code)
		require.Equal(t, codes[i].ID, replay[i].ID)
	}
	_, err = adminSvc.GenerateRedeemCodes(ctx, &service.GenerateRedeemCodesInput{
		Count: 2, Type: service.RedeemTypeBalance, Value: 20, IncludeGift: true, OperationKey: "gift-postgres-generation",
	})
	require.Error(t, err)
	require.Equal(t, 4, client.RedeemCode.Query().CountX(ctx))
	// Concurrent generation retries with a missing response cache cannot mint
	// multiple batches: all workers recover the single unique committed pair.
	var genWG sync.WaitGroup
	genStart := make(chan struct{})
	genResults := make(chan []service.RedeemCode, 8)
	genErrors := make(chan error, 8)
	for i := 0; i < 8; i++ {
		genWG.Add(1)
		go func() {
			defer genWG.Done()
			<-genStart
			result, err := adminSvc.GenerateRedeemCodes(ctx, &service.GenerateRedeemCodesInput{
				Count: 1, Type: service.RedeemTypeBalance, Value: 10, IncludeGift: true, OperationKey: "gift-parallel-generation",
			})
			genResults <- result
			genErrors <- err
		}()
	}
	close(genStart)
	genWG.Wait()
	close(genResults)
	close(genErrors)
	for err := range genErrors {
		require.NoError(t, err)
	}
	var generated []service.RedeemCode
	for result := range genResults {
		if generated == nil {
			generated = result
		} else {
			require.Len(t, result, len(generated))
			for i := range result {
				require.Equal(t, generated[i].Code, result[i].Code)
				require.Equal(t, generated[i].ID, result[i].ID)
			}
		}
	}
	require.Equal(t, 6, client.RedeemCode.Query().CountX(ctx))

	for _, code := range codes {
		require.Positive(t, code.ID)
		require.False(t, code.CreatedAt.IsZero())
	}
	stored, err := repo.GetByCode(ctx, gift.Code)
	require.NoError(t, err)
	require.True(t, stored.IsGift)
	require.Equal(t, paid.GiftPairID, stored.GiftPairID)
	require.Equal(t, 0.1, stored.Value)
	// Database guards survive application bugs and ordinary edits.
	for _, query := range []string{
		"UPDATE redeem_codes SET is_gift = false WHERE id = $1",
		"UPDATE redeem_codes SET gift_pair_id = NULL WHERE id = $1",
		"UPDATE redeem_codes SET value = 100 WHERE id = $1",
		"UPDATE redeem_codes SET type = 'concurrency' WHERE id = $1",
	} {
		_, err := db.ExecContext(ctx, query, gift.ID)
		require.Error(t, err)
	}
	// One conflict must roll back the entire batch, including the paid code.
	before := client.RedeemCode.Query().CountX(ctx)
	err = repo.CreateBatch(ctx, []service.RedeemCode{
		{Code: "ROLLBACK-NEW", Type: service.RedeemTypeBalance, Value: 10, Status: service.StatusUnused},
		{Code: gift.Code, Type: service.RedeemTypeBalance, Value: 0.1, Status: service.StatusUnused},
	})
	require.Error(t, err)
	require.Equal(t, before, client.RedeemCode.Query().CountX(ctx))
	// Bulk creation joins an existing transaction, whose rollback removes both codes.
	tx, err := client.Tx(ctx)
	require.NoError(t, err)
	err = repo.CreateBatch(dbent.NewTxContext(ctx, tx), []service.RedeemCode{
		{Code: "OUTER-TX-1", Type: service.RedeemTypeBalance, Value: 10, Status: service.StatusUnused},
		{Code: "OUTER-TX-2", Type: service.RedeemTypeBalance, Value: 0.1, Status: service.StatusUnused},
	})
	require.NoError(t, err)
	require.NoError(t, tx.Rollback())
	require.Equal(t, before, client.RedeemCode.Query().CountX(ctx))
	user := client.User.Create().SetEmail("gift@example.com").SetPasswordHash("test").SetBalance(7).SaveX(ctx)
	redeemSvc := service.NewRedeemService(repo, users, nil, nil, nil, client, nil, nil)
	// Even a failure after the real credit must roll back balance and code use.
	failingSvc := service.NewRedeemService(repo, failingGiftCreditRepo{UserRepository: users, gift: users.(service.GiftBalanceRepository)}, nil, nil, nil, client, nil, nil)
	_, err = failingSvc.Redeem(ctx, user.ID, gift.Code)
	require.Error(t, err)
	require.Equal(t, 7.0, client.User.GetX(ctx, user.ID).Balance)
	require.Equal(t, service.StatusUnused, client.RedeemCode.GetX(ctx, gift.ID).Status)
	// No Redis lock: the database alone must allow exactly one credit.
	var successes atomic.Int32
	var wg sync.WaitGroup
	start := make(chan struct{})
	results := make(chan error, 20)
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := redeemSvc.Redeem(ctx, user.ID, gift.Code)
			if err == nil {
				successes.Add(1)
			}
			results <- err
		}()
	}
	close(start)
	wg.Wait()
	close(results)
	require.EqualValues(t, 1, successes.Load())
	for err := range results {
		if err != nil {
			require.ErrorIs(t, err, service.ErrRedeemCodeUsed)
		}
	}
	updated := client.User.GetX(ctx, user.ID)
	require.InDelta(t, 7.1, updated.Balance, 1e-8)
	require.Zero(t, updated.TotalRecharged)
	_, err = redeemSvc.Redeem(ctx, user.ID, paid.Code)
	require.NoError(t, err)
	updated = client.User.GetX(ctx, user.ID)
	require.InDelta(t, 17.1, updated.Balance, 1e-8)
	require.Equal(t, 10.0, updated.TotalRecharged)
	recharged, err := repo.SumPositiveBalanceByUser(ctx, user.ID)
	require.NoError(t, err)
	require.Equal(t, 10.0, recharged)
	// Consumed paired codes cannot be reactivated, even by a direct database
	// update or by expiring then re-enabling through the admin service.
	for _, query := range []string{
		"UPDATE redeem_codes SET status = 'unused' WHERE id = $1",
		"UPDATE redeem_codes SET status = 'expired' WHERE id = $1",
		"UPDATE redeem_codes SET used_at = NULL, used_by = NULL WHERE id = $1",
	} {
		_, err := db.ExecContext(ctx, query, gift.ID)
		require.Error(t, err)
	}
	_, err = adminSvc.ExpireRedeemCode(ctx, gift.ID)
	require.ErrorIs(t, err, service.ErrRedeemCodeUsed)
	unused := service.StatusUnused
	_, err = repo.BatchUpdate(ctx, []int64{gift.ID}, service.RedeemCodeBatchUpdateFields{Status: &unused})
	require.ErrorIs(t, err, service.ErrRedeemCodeUsed)
	require.Equal(t, service.StatusUsed, client.RedeemCode.GetX(ctx, gift.ID).Status)
	_, err = redeemSvc.Redeem(ctx, user.ID, gift.Code)
	require.ErrorIs(t, err, service.ErrRedeemCodeUsed)
	require.InDelta(t, 17.1, client.User.GetX(ctx, user.ID).Balance, 1e-8)

	// Editing notes never alters eligibility or exposes gift identity for editing.
	notes := "ordinary paid recharge"
	_, err = repo.BatchUpdate(ctx, []int64{gift.ID}, service.RedeemCodeBatchUpdateFields{Notes: &notes})
	require.NoError(t, err)
	cfg := &config.Config{}
	cfg.Pricing.DataDir = t.TempDir()
	invoices := service.NewInvoiceService(client, users, nil, nil, cfg)
	sources, err := invoices.ListEligibleSources(ctx, user.ID)
	require.NoError(t, err)
	require.Len(t, sources.RedeemCodes, 1)
	require.Equal(t, paid.ID, sources.RedeemCodes[0].RedeemCodeID)
	_, err = invoices.CreateRequest(ctx, service.CreateInvoiceRequestInput{UserID: user.ID, RedeemCodeIDs: []int64{paid.ID, gift.ID}, InvoiceType: domain.InvoiceTypePersonal, Title: "Forged gift request"})
	require.ErrorIs(t, err, service.ErrInvoiceRedeemNotEligible)
	request, err := invoices.CreateRequest(ctx, service.CreateInvoiceRequestInput{UserID: user.ID, RedeemCodeIDs: []int64{paid.ID}, InvoiceType: domain.InvoiceTypePersonal, Title: "Paid code"})
	require.NoError(t, err)
	require.Equal(t, 10.0, request.Amount)
	_, err = invoices.AdminApprove(ctx, request.ID, 999)
	require.NoError(t, err)
	pdf := []byte("%PDF-1.7\npaid invoice\n")
	_, err = invoices.AdminIssue(ctx, request.ID, 999, "PAID-OK", bytes.NewReader(pdf), int64(len(pdf)))
	require.NoError(t, err)
	// Expiry is also checked in the atomic database update, not only in the earlier read.
	past := time.Now().Add(-time.Hour)
	client.RedeemCode.UpdateOneID(codes[3].ID).SetExpiresAt(past).SaveX(ctx)
	require.ErrorIs(t, repo.Use(ctx, codes[3].ID, user.ID), service.ErrRedeemCodeUsed)
	require.Equal(t, service.StatusUnused, client.RedeemCode.GetX(ctx, codes[3].ID).Status)
	// Each pair permits exactly one paid and one gift record.
	_, err = client.RedeemCode.Create().SetCode("DUPLICATE-GIFT").SetType(service.RedeemTypeBalance).SetValue(0.1).
		SetIsGift(true).SetGiftPairID(*gift.GiftPairID).Save(ctx)
	require.Error(t, err)
	require.Equal(t, 1, client.RedeemCode.Query().Where(redeemcode.GiftPairIDEQ(*gift.GiftPairID), redeemcode.IsGiftEQ(true)).CountX(ctx))
	require.NoError(t, repo.Delete(ctx, codes[3].ID))
	countAfterDelete := client.RedeemCode.Query().CountX(ctx)
	_, err = adminSvc.GenerateRedeemCodes(ctx, &service.GenerateRedeemCodesInput{
		Count: 2, Type: service.RedeemTypeBalance, Value: 10, IncludeGift: true, OperationKey: "gift-postgres-generation",
	})
	require.Error(t, err)
	require.Equal(t, countAfterDelete, client.RedeemCode.Query().CountX(ctx))
	// The public limit is 500 paid codes plus 500 gifts in one transaction.
	large, err := adminSvc.GenerateRedeemCodes(ctx, &service.GenerateRedeemCodesInput{
		Count: 500, Type: service.RedeemTypeBalance, Value: 0.01, IncludeGift: true, OperationKey: "gift-maximum-batch",
	})
	require.NoError(t, err)
	require.Len(t, large, 1000)
	require.Equal(t, 0.0001, large[1].Value)
	require.Equal(t, countAfterDelete+1000, client.RedeemCode.Query().CountX(ctx))
	t.Run("public input cannot change gift amount owner or invoice status", func(t *testing.T) {
		testGiftPublicRequestCannotOverrideStoredBalance(t, ctx, client, db, large[1])
	})
	t.Run("batch edit cannot undo a concurrent redemption", func(t *testing.T) {
		testRedeemBatchUpdateCannotResurrectCode(t, ctx, client, db)
	})
	t.Run("expiry cannot reset a redeemed code", func(t *testing.T) {
		fresh := client.RedeemCode.Create().SetCode("EXPIRE-ONCE").SetType(service.RedeemTypeBalance).
			SetValue(1).SetStatus(service.StatusUnused).SaveX(ctx)
		_, err := adminSvc.ExpireRedeemCode(ctx, fresh.ID)
		require.NoError(t, err)
		_, err = redeemSvc.Redeem(ctx, user.ID, fresh.Code)
		require.ErrorIs(t, err, service.ErrRedeemCodeExpired)
		_, err = repo.BatchUpdate(ctx, []int64{fresh.ID}, service.RedeemCodeBatchUpdateFields{Status: &unused})
		require.NoError(t, err)
		_, err = redeemSvc.Redeem(ctx, user.ID, fresh.Code)
		require.NoError(t, err)
		_, err = adminSvc.ExpireRedeemCode(ctx, fresh.ID)
		require.ErrorIs(t, err, service.ErrRedeemCodeUsed)
	})

}

// The hook pauses the admin update after validation, then a separate connection
// commits a real redemption before the admin write resumes. Test ordinary codes
// too, so this proves the application predicate rather than only the DB trigger.
func testRedeemBatchUpdateCannotResurrectCode(t *testing.T, ctx context.Context, client *dbent.Client, db *sql.DB) {
	u := client.User.Create().SetEmail("gift-race@example.com").SetPasswordHash("test").SaveX(ctx)
	original := client.RedeemCode.Create().SetCode("RACE-REDEEM").SetType(service.RedeemTypeBalance).
		SetValue(10).SetStatus(service.StatusUnused).SaveX(ctx)
	other := client.RedeemCode.Create().SetCode("RACE-OTHER").SetType(service.RedeemTypeBalance).
		SetValue(10).SetStatus(service.StatusUnused).SaveX(ctx)
	entered, resume := make(chan struct{}), make(chan struct{})
	var resumeOnce sync.Once
	release := func() { resumeOnce.Do(func() { close(resume) }) }
	defer release()
	// Sharing sql.DB is intentional; do not Close this client independently.
	editClient := dbent.NewClient(dbent.Driver(entsql.OpenDB(dialect.Postgres, db)))
	editClient.RedeemCode.Use(func(next dbent.Mutator) dbent.Mutator {
		return dbent.MutateFunc(func(ctx context.Context, m dbent.Mutation) (dbent.Value, error) {
			if m.Op().Is(dbent.OpUpdate) {
				close(entered)
				select {
				case <-resume:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}
			return next.Mutate(ctx, m)
		})
	})
	editRepo := NewRedeemCodeRepository(editClient)
	unused, notes := service.StatusUnused, "must roll back with batch"
	done := make(chan error, 1)
	go func() {
		_, err := editRepo.BatchUpdate(ctx, []int64{original.ID, other.ID}, service.RedeemCodeBatchUpdateFields{Status: &unused, Notes: &notes})
		done <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("admin update did not reach barrier")
	}
	repo := NewRedeemCodeRepository(client)
	svc := service.NewRedeemService(repo, NewUserRepository(client, db), nil, nil, nil, client, nil, nil)
	_, err := svc.Redeem(ctx, u.ID, original.Code)
	require.NoError(t, err)
	release()
	select {
	case err := <-done:
		require.ErrorIs(t, err, service.ErrRedeemCodeUsed)
	case <-time.After(5 * time.Second):
		t.Fatal("admin update did not complete")
	}
	after := client.RedeemCode.GetX(ctx, original.ID)
	require.Equal(t, service.StatusUsed, after.Status)
	require.Equal(t, u.ID, *after.UsedBy)
	require.NotNil(t, after.UsedAt)
	require.Nil(t, client.RedeemCode.GetX(ctx, other.ID).Notes)
	_, err = svc.Redeem(ctx, u.ID, original.Code)
	require.ErrorIs(t, err, service.ErrRedeemCodeUsed)
	require.Equal(t, 10.0, client.User.GetX(ctx, u.ID).Balance)
}

func testGiftPublicRequestCannotOverrideStoredBalance(t *testing.T, ctx context.Context, client *dbent.Client, db *sql.DB, gift service.RedeemCode) {
	buyer := client.User.Create().SetEmail("gift-public-buyer@example.com").SetPasswordHash("test").SaveX(ctx)
	victim := client.User.Create().SetEmail("gift-public-other@example.com").SetPasswordHash("test").SaveX(ctx)
	repo := NewRedeemCodeRepository(client)
	svc := service.NewRedeemService(repo, NewUserRepository(client, db), nil, nil, nil, client, nil, nil)
	h := handler.NewRedeemHandler(svc)
	requestPayload, err := json.Marshal(map[string]any{
		"code": gift.Code, "value": 999999, "amount": 999999, "is_gift": false,
		"user_id": victim.ID, "type": service.RedeemTypeBalance, "gift_pair_id": "attacker-value",
	})
	require.NoError(t, err)
	call := func(authenticated bool) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/redeem", strings.NewReader(string(requestPayload)))
		c.Request.Header.Set("Content-Type", "application/json")
		if authenticated {
			c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: buyer.ID})
		}
		h.Redeem(c)
		return rec
	}
	require.Equal(t, http.StatusUnauthorized, call(false).Code)
	result := call(true)
	require.Equal(t, http.StatusOK, result.Code, result.Body.String())
	require.NotContains(t, result.Body.String(), "gift_pair_id")
	require.Contains(t, result.Body.String(), `"is_gift":true`)
	require.Equal(t, gift.Value, client.User.GetX(ctx, buyer.ID).Balance)
	require.Zero(t, client.User.GetX(ctx, buyer.ID).TotalRecharged)
	require.Zero(t, client.User.GetX(ctx, victim.ID).Balance)
	require.True(t, client.RedeemCode.GetX(ctx, gift.ID).IsGift)
	require.Equal(t, buyer.ID, *client.RedeemCode.GetX(ctx, gift.ID).UsedBy)
	require.Equal(t, http.StatusConflict, call(true).Code)
	require.Equal(t, gift.Value, client.User.GetX(ctx, buyer.ID).Balance)
}
