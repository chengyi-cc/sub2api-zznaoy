//go:build unit

package service

import (
	"context"
	"fmt"
	"testing"

	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/internal/domain"
	"github.com/stretchr/testify/require"
)

func TestNormalizeUnifiedSocialCreditCode(t *testing.T) {
	for _, valid := range []string{"91350100M000100Y43", "91110000600001760P", " 91110108551385082q "} {
		code, err := NormalizeUnifiedSocialCreditCode(valid)
		require.NoError(t, err, valid)
		require.Len(t, code, 18)
	}
	code, err := NormalizeUnifiedSocialCreditCode("91110108551385082q")
	require.NoError(t, err)
	require.Equal(t, "91110108551385082Q", code)

	for _, invalid := range []string{
		"",
		"91350100M000100Y44",  // 校验位错误
		"91350100M000100Y4",   // 17 位
		"91350100M000100Y430", // 19 位
		"91350100I000100Y43",  // I 不在字符集
		"91110000000000000X",  // 格式像但校验位不对
	} {
		_, err := NormalizeUnifiedSocialCreditCode(invalid)
		require.ErrorIs(t, err, ErrInvoiceTaxNoInvalid, invalid)
	}
}

func TestInvoiceProfile_BindOnceThenLocked(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	ctx := context.Background()
	_, err := client.InvoiceProfile.Delete().Exec(ctx)
	require.NoError(t, err)

	p, err := svc.GetInvoiceProfile(ctx, user.ID)
	require.NoError(t, err)
	require.Nil(t, p)

	_, err = svc.BindInvoiceProfile(ctx, user.ID, "Bad Tax Co.", "91350100M000100Y44")
	require.ErrorIs(t, err, ErrInvoiceTaxNoInvalid)
	_, err = svc.BindInvoiceProfile(ctx, user.ID, "   ", "91350100M000100Y43")
	require.ErrorIs(t, err, ErrInvoiceTitleRequired)

	p, err = svc.BindInvoiceProfile(ctx, user.ID, "  某某科技有限公司 ", "91110000600001760p")
	require.NoError(t, err)
	require.Equal(t, "某某科技有限公司", p.Title)
	require.Equal(t, "91110000600001760P", p.TaxNo)
	require.Nil(t, p.UpdatedBy)

	_, err = svc.BindInvoiceProfile(ctx, user.ID, "Another Co.", "91350100M000100Y43")
	require.ErrorIs(t, err, ErrInvoiceProfileLocked)

	p, err = svc.GetInvoiceProfile(ctx, user.ID)
	require.NoError(t, err)
	require.Equal(t, "某某科技有限公司", p.Title)
}

// 申请保存的是提交时的快照：管理员事后修改资料只影响之后的新申请。
func TestInvoiceProfile_RequestSnapshotsProfileAndAdminEditAppliesToNewRequests(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 2, 100)
	ctx := context.Background()

	first, err := svc.CreateRequest(ctx, CreateInvoiceRequestInput{UserID: user.ID, PaymentOrderIDs: orderIDs[:1]})
	require.NoError(t, err)
	require.Equal(t, testInvoiceProfileTitle, first.Title)

	updated, err := svc.AdminUpsertInvoiceProfile(ctx, user.ID, 999, "Renamed Co.", "91110108551385082Q")
	require.NoError(t, err)
	require.Equal(t, "Renamed Co.", updated.Title)
	require.NotNil(t, updated.UpdatedBy)
	require.Equal(t, int64(999), *updated.UpdatedBy)

	second, err := svc.CreateRequest(ctx, CreateInvoiceRequestInput{UserID: user.ID, PaymentOrderIDs: orderIDs[1:]})
	require.NoError(t, err)
	require.Equal(t, "Renamed Co.", second.Title)
	require.Equal(t, "91110108551385082Q", *second.TaxNo)
	require.Equal(t, domain.InvoiceTypeCompany, second.InvoiceType)

	reloaded, err := svc.AdminGet(ctx, first.ID)
	require.NoError(t, err)
	require.Equal(t, testInvoiceProfileTitle, reloaded.Title)
	require.Equal(t, testInvoiceProfileTaxNo, *reloaded.TaxNo)
}

func TestInvoiceProfile_AdminResetAllowsRebind(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)
	ctx := context.Background()

	require.NoError(t, svc.AdminResetInvoiceProfile(ctx, user.ID))
	require.ErrorIs(t, svc.AdminResetInvoiceProfile(ctx, user.ID), ErrInvoiceProfileNotFound)

	_, err := svc.CreateRequest(ctx, CreateInvoiceRequestInput{UserID: user.ID, PaymentOrderIDs: orderIDs})
	require.ErrorIs(t, err, ErrInvoiceProfileRequired)

	_, err = svc.BindInvoiceProfile(ctx, user.ID, "Rebound Co.", "91110000600001760P")
	require.NoError(t, err)
	req, err := svc.CreateRequest(ctx, CreateInvoiceRequestInput{UserID: user.ID, PaymentOrderIDs: orderIDs})
	require.NoError(t, err)
	require.Equal(t, "Rebound Co.", req.Title)
}

func TestInvoiceProfile_AdminUpsertCreatesForUnboundUserAndRejectsUnknownUser(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, _ := seedUserAndCompletedOrders(t, client, 0, 0)
	ctx := context.Background()
	require.NoError(t, svc.AdminResetInvoiceProfile(ctx, user.ID))

	p, err := svc.AdminUpsertInvoiceProfile(ctx, user.ID, 7, "Admin Set Co.", "91350100M000100Y43")
	require.NoError(t, err)
	require.Equal(t, "Admin Set Co.", p.Title)
	require.Equal(t, int64(7), *p.UpdatedBy)

	// 管理员设置后用户仍不能自行改
	_, err = svc.BindInvoiceProfile(ctx, user.ID, "User Co.", "91110000600001760P")
	require.ErrorIs(t, err, ErrInvoiceProfileLocked)

	_, err = svc.AdminUpsertInvoiceProfile(ctx, user.ID+1000, 7, "Ghost Co.", "91350100M000100Y43")
	require.ErrorIs(t, err, ErrInvoiceUserNotFound)
	_, err = svc.AdminUpsertInvoiceProfile(ctx, user.ID, 7, "Admin Set Co.", "123")
	require.ErrorIs(t, err, ErrInvoiceTaxNoInvalid)
}

func TestInvoiceProfile_SharedTaxNoIsFlaggedNotBlocked(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	user, orderIDs := seedUserAndCompletedOrders(t, client, 1, 100)
	ctx := context.Background()

	other, err := client.User.Create().
		SetEmail("same-company@example.com").
		SetPasswordHash("hash").
		SetRole(RoleUser).
		SetStatus(StatusActive).
		Save(ctx)
	require.NoError(t, err)
	_, err = svc.BindInvoiceProfile(ctx, other.ID, "Acme Inc. (branch)", testInvoiceProfileTaxNo)
	require.NoError(t, err)

	rows, total, err := svc.AdminListInvoiceProfiles(ctx, AdminListInvoiceProfileParams{Keyword: "same-company"})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.Equal(t, other.ID, rows[0].Profile.UserID)
	require.Equal(t, "same-company@example.com", rows[0].UserEmail)
	require.Equal(t, 1, rows[0].SharedUserCount)

	rows, total, err = svc.AdminListInvoiceProfiles(ctx, AdminListInvoiceProfileParams{Keyword: testInvoiceProfileTaxNo})
	require.NoError(t, err)
	require.Equal(t, 2, total)
	require.Len(t, rows, 2)

	req, err := svc.CreateRequest(ctx, CreateInvoiceRequestInput{UserID: user.ID, PaymentOrderIDs: orderIDs})
	require.NoError(t, err)
	svc.userRepo = &mockUserRepo{getByIDUser: &User{ID: user.ID}}
	detail, err := svc.AdminGetDetail(ctx, req.ID)
	require.NoError(t, err)
	require.Equal(t, 1, detail.TaxNoSharedUserCount)
	require.NotNil(t, detail.CurrentProfile)
	require.Equal(t, testInvoiceProfileTitle, detail.CurrentProfile.Title)
}

// 邮箱搜索在数据库里关联 users 筛选后再分页：命中邮箱的未绑定用户再多，
// 也不能把真正绑定了资料的用户挤出结果（回归：曾先取最多 500 个匹配用户 ID）。
func TestInvoiceProfile_AdminListEmailSearchIsNotCappedByMatchingUsers(t *testing.T) {
	svc, client := newInvoiceServiceTestClient(t)
	ctx := context.Background()

	builders := make([]*dbent.UserCreate, 0, 600)
	for i := 0; i < 600; i++ {
		builders = append(builders, client.User.Create().
			SetEmail(fmt.Sprintf("unbound-%03d@bulk.example", i)).
			SetPasswordHash("hash").
			SetRole(RoleUser).
			SetStatus(StatusActive))
	}
	_, err := client.User.CreateBulk(builders...).Save(ctx)
	require.NoError(t, err)

	bound := make([]int64, 0, 3)
	for i := 0; i < 3; i++ {
		u, err := client.User.Create().
			SetEmail(fmt.Sprintf("bound-%d@bulk.example", i)).
			SetPasswordHash("hash").
			SetRole(RoleUser).
			SetStatus(StatusActive).
			Save(ctx)
		require.NoError(t, err)
		_, err = svc.BindInvoiceProfile(ctx, u.ID, fmt.Sprintf("Bulk Co. %d", i), testInvoiceProfileTaxNo)
		require.NoError(t, err)
		bound = append(bound, u.ID)
	}

	page1, total, err := svc.AdminListInvoiceProfiles(ctx, AdminListInvoiceProfileParams{Keyword: "BULK.example", Page: 1, PageSize: 2})
	require.NoError(t, err)
	require.Equal(t, 3, total)
	require.Len(t, page1, 2)
	page2, total, err := svc.AdminListInvoiceProfiles(ctx, AdminListInvoiceProfileParams{Keyword: "BULK.example", Page: 2, PageSize: 2})
	require.NoError(t, err)
	require.Equal(t, 3, total)
	require.Len(t, page2, 1)

	got := []int64{page1[0].Profile.UserID, page1[1].Profile.UserID, page2[0].Profile.UserID}
	require.ElementsMatch(t, bound, got)
}
