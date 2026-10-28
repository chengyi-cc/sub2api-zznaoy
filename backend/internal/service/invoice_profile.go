package service

// invoice_profile: 用户固定开票资料（仅企业抬头）。
//
// 用户在个人资料页绑定一次「单位名称 + 统一社会信用代码」后即锁定，
// 之后所有发票申请都由后端从这里取抬头/税号快照，前端传入一律忽略。
// 只有管理员可以修改或重置（删除）；重置后用户可以重新绑定一次。

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	entsql "entgo.io/ent/dialect/sql"

	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/ent/invoiceprofile"
	"github.com/Wei-Shaw/sub2api/ent/predicate"
	"github.com/Wei-Shaw/sub2api/ent/user"
	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
)

var (
	ErrInvoiceProfileRequired = infraerrors.BadRequest("INVOICE_PROFILE_REQUIRED", "invoice profile is not bound; bind the company name and tax number in profile settings first")
	ErrInvoiceProfileLocked   = infraerrors.Conflict("INVOICE_PROFILE_LOCKED", "invoice profile is already bound and can only be changed by an administrator")
	ErrInvoiceProfileNotFound = infraerrors.NotFound("INVOICE_PROFILE_NOT_FOUND", "invoice profile not found")
	ErrInvoiceTaxNoInvalid    = infraerrors.BadRequest("INVOICE_TAX_NO_INVALID", "tax_no must be a valid 18-character unified social credit code")
	ErrInvoiceUserNotFound    = infraerrors.NotFound("INVOICE_USER_NOT_FOUND", "user not found")
)

// invoiceMaxTitleRunes 单位名称最大字符数（中文按 1 个字符计）。
// 数据库列是 VARCHAR(200)，ent 的 MaxLen(200) 按字节计，中文 3 字节/字，
// 这里取 66 保证任何输入都不会撞上 ent 校验抛出不友好的错误。
const invoiceMaxTitleRunes = 66

// uscc 统一社会信用代码（GB 32100-2015）字符集与加权因子。
const usccCharset = "0123456789ABCDEFGHJKLMNPQRTUWXY"

var usccWeights = [17]int{1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28}

// NormalizeUnifiedSocialCreditCode 去首尾空白并转大写；若不是合法的 18 位
// 统一社会信用代码（字符集 + 第 18 位校验码）则返回 ErrInvoiceTaxNoInvalid。
func NormalizeUnifiedSocialCreditCode(raw string) (string, error) {
	code := strings.ToUpper(strings.TrimSpace(raw))
	if len(code) != 18 {
		return "", ErrInvoiceTaxNoInvalid
	}
	sum := 0
	for i := 0; i < 17; i++ {
		v := strings.IndexByte(usccCharset, code[i])
		if v < 0 {
			return "", ErrInvoiceTaxNoInvalid
		}
		sum += v * usccWeights[i]
	}
	check := 31 - sum%31
	if check == 31 {
		check = 0
	}
	if code[17] != usccCharset[check] {
		return "", ErrInvoiceTaxNoInvalid
	}
	return code, nil
}

func normalizeInvoiceProfileInput(title, taxNo string) (string, string, error) {
	title = strings.TrimSpace(title)
	if title == "" {
		return "", "", ErrInvoiceTitleRequired
	}
	if utf8.RuneCountInString(title) > invoiceMaxTitleRunes || len(title) > invoiceMaxTitleLen {
		return "", "", infraerrors.BadRequest("INVOICE_TITLE_TOO_LONG", "title is too long").
			WithMetadata(map[string]string{"max": strconv.Itoa(invoiceMaxTitleRunes)})
	}
	if strings.TrimSpace(taxNo) == "" {
		return "", "", ErrInvoiceTaxNoRequired
	}
	code, err := NormalizeUnifiedSocialCreditCode(taxNo)
	if err != nil {
		return "", "", err
	}
	return title, code, nil
}

// GetInvoiceProfile 返回用户已绑定的开票资料；未绑定时返回 (nil, nil)。
func (s *InvoiceService) GetInvoiceProfile(ctx context.Context, userID int64) (*dbent.InvoiceProfile, error) {
	p, err := s.entClient.InvoiceProfile.Query().
		Where(invoiceprofile.UserIDEQ(userID)).
		Only(ctx)
	if err != nil {
		if dbent.IsNotFound(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("get invoice profile: %w", err)
	}
	return p, nil
}

// BindInvoiceProfile 用户首次绑定开票资料。已绑定（含并发重复提交）一律返回
// ErrInvoiceProfileLocked；锁定后只有管理员可以修改或重置。
func (s *InvoiceService) BindInvoiceProfile(ctx context.Context, userID int64, title, taxNo string) (*dbent.InvoiceProfile, error) {
	title, taxNo, err := normalizeInvoiceProfileInput(title, taxNo)
	if err != nil {
		return nil, err
	}
	exists, err := s.entClient.InvoiceProfile.Query().
		Where(invoiceprofile.UserIDEQ(userID)).
		Exist(ctx)
	if err != nil {
		return nil, fmt.Errorf("check invoice profile: %w", err)
	}
	if exists {
		return nil, ErrInvoiceProfileLocked
	}
	created, err := s.entClient.InvoiceProfile.Create().
		SetUserID(userID).
		SetTitle(title).
		SetTaxNo(taxNo).
		Save(ctx)
	if err != nil {
		// user_id 唯一索引兜底并发的重复绑定
		if dbent.IsConstraintError(err) {
			return nil, ErrInvoiceProfileLocked
		}
		return nil, fmt.Errorf("create invoice profile: %w", err)
	}
	return created, nil
}

// AdminUpsertInvoiceProfile 管理员直接设置用户的开票资料（不存在则创建）。
func (s *InvoiceService) AdminUpsertInvoiceProfile(ctx context.Context, userID, adminID int64, title, taxNo string) (*dbent.InvoiceProfile, error) {
	title, taxNo, err := normalizeInvoiceProfileInput(title, taxNo)
	if err != nil {
		return nil, err
	}
	userExists, err := s.entClient.User.Query().Where(user.IDEQ(userID)).Exist(ctx)
	if err != nil {
		return nil, fmt.Errorf("check invoice profile user: %w", err)
	}
	if !userExists {
		return nil, ErrInvoiceUserNotFound
	}

	update := func() (int, error) {
		return s.entClient.InvoiceProfile.Update().
			Where(invoiceprofile.UserIDEQ(userID)).
			SetTitle(title).
			SetTaxNo(taxNo).
			SetUpdatedBy(adminID).
			Save(ctx)
	}
	updated, err := update()
	if err != nil {
		return nil, fmt.Errorf("update invoice profile: %w", err)
	}
	if updated == 0 {
		_, err = s.entClient.InvoiceProfile.Create().
			SetUserID(userID).
			SetTitle(title).
			SetTaxNo(taxNo).
			SetUpdatedBy(adminID).
			Save(ctx)
		if dbent.IsConstraintError(err) {
			// 与用户首次绑定并发：用户那条已落库，以管理员的值覆盖
			_, err = update()
		}
		if err != nil {
			return nil, fmt.Errorf("save invoice profile: %w", err)
		}
	}
	p, err := s.GetInvoiceProfile(ctx, userID)
	if err != nil {
		return nil, err
	}
	if p == nil {
		return nil, ErrInvoiceProfileNotFound
	}
	return p, nil
}

// AdminResetInvoiceProfile 管理员重置（删除）用户的开票资料，用户之后可重新绑定一次。
// 已提交的发票申请保存的是提交时的快照，不受影响。
func (s *InvoiceService) AdminResetInvoiceProfile(ctx context.Context, userID int64) error {
	n, err := s.entClient.InvoiceProfile.Delete().
		Where(invoiceprofile.UserIDEQ(userID)).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("reset invoice profile: %w", err)
	}
	if n == 0 {
		return ErrInvoiceProfileNotFound
	}
	return nil
}

// AdminListInvoiceProfileParams 管理员查询开票资料的参数。
type AdminListInvoiceProfileParams struct {
	Keyword  string // 单位名称 / 税号 / 用户邮箱模糊匹配；纯数字时同时匹配用户 ID
	Page     int
	PageSize int
}

// InvoiceProfileAdminRow 管理端开票资料列表行。
type InvoiceProfileAdminRow struct {
	Profile   *dbent.InvoiceProfile
	UserEmail string
	// SharedUserCount 同一税号还被多少个「其他」账号绑定（0 = 独占）
	SharedUserCount int
}

// AdminListInvoiceProfiles 分页列出已绑定的开票资料，附带用户邮箱与税号共用账号数。
func (s *InvoiceService) AdminListInvoiceProfiles(ctx context.Context, p AdminListInvoiceProfileParams) ([]InvoiceProfileAdminRow, int, error) {
	q := s.entClient.InvoiceProfile.Query()
	if kw := strings.TrimSpace(p.Keyword); kw != "" {
		preds := []predicate.InvoiceProfile{
			invoiceprofile.TitleContainsFold(kw),
			invoiceprofile.TaxNoContainsFold(kw),
		}
		if uid, err := strconv.ParseInt(kw, 10, 64); err == nil && uid > 0 {
			preds = append(preds, invoiceprofile.UserIDEQ(uid))
		}
		preds = append(preds, invoiceProfileUserEmailContainsFold(kw))
		q = q.Where(invoiceprofile.Or(preds...))
	}

	total, err := q.Clone().Count(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("count invoice profiles: %w", err)
	}
	ps, pg := applyPagination(p.PageSize, p.Page)
	profiles, err := q.Order(dbent.Desc(invoiceprofile.FieldUpdatedAt)).
		Limit(ps).Offset((pg - 1) * ps).All(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("query invoice profiles: %w", err)
	}

	userIDs := make([]int64, 0, len(profiles))
	taxNos := make([]string, 0, len(profiles))
	for _, pr := range profiles {
		userIDs = append(userIDs, pr.UserID)
		taxNos = append(taxNos, pr.TaxNo)
	}
	emails, err := s.BatchUserEmails(ctx, userIDs)
	if err != nil {
		return nil, 0, err
	}
	bindings, err := s.countInvoiceProfilesByTaxNo(ctx, taxNos)
	if err != nil {
		return nil, 0, err
	}

	rows := make([]InvoiceProfileAdminRow, 0, len(profiles))
	for _, pr := range profiles {
		shared := bindings[pr.TaxNo] - 1
		if shared < 0 {
			shared = 0
		}
		rows = append(rows, InvoiceProfileAdminRow{
			Profile:         pr,
			UserEmail:       emails[pr.UserID],
			SharedUserCount: shared,
		})
	}
	return rows, total, nil
}

// invoiceProfileUserEmailContainsFold 按绑定用户的邮箱模糊匹配：
// user_id IN (SELECT id FROM users WHERE email ILIKE %kw%)。
// 在数据库里完成筛选后再分页，命中用户再多也不会漏（InvoiceProfile 刻意没有 User edge，
// 所以手写子查询而不是 HasUserWith）。
func invoiceProfileUserEmailContainsFold(kw string) predicate.InvoiceProfile {
	return predicate.InvoiceProfile(func(s *entsql.Selector) {
		users := entsql.Table(user.Table)
		s.Where(entsql.In(
			s.C(invoiceprofile.FieldUserID),
			entsql.Select(users.C(user.FieldID)).
				From(users).
				Where(entsql.ContainsFold(users.C(user.FieldEmail), kw)),
		))
	})
}

// countInvoiceProfilesByTaxNo 返回每个税号当前被多少个账号绑定。
func (s *InvoiceService) countInvoiceProfilesByTaxNo(ctx context.Context, taxNos []string) (map[string]int, error) {
	out := make(map[string]int, len(taxNos))
	if len(taxNos) == 0 {
		return out, nil
	}
	var counts []struct {
		TaxNo string `json:"tax_no"`
		Count int    `json:"count"`
	}
	err := s.entClient.InvoiceProfile.Query().
		Where(invoiceprofile.TaxNoIn(taxNos...)).
		GroupBy(invoiceprofile.FieldTaxNo).
		Aggregate(dbent.Count()).
		Scan(ctx, &counts)
	if err != nil {
		return nil, fmt.Errorf("count invoice profiles by tax_no: %w", err)
	}
	for _, c := range counts {
		out[c.TaxNo] = c.Count
	}
	return out, nil
}

// countOtherUsersWithTaxNo 返回除 excludeUserID 外绑定了同一税号的账号数。
func (s *InvoiceService) countOtherUsersWithTaxNo(ctx context.Context, taxNo string, excludeUserID int64) (int, error) {
	taxNo = strings.TrimSpace(taxNo)
	if taxNo == "" {
		return 0, nil
	}
	n, err := s.entClient.InvoiceProfile.Query().
		Where(
			invoiceprofile.TaxNoEQ(taxNo),
			invoiceprofile.UserIDNEQ(excludeUserID),
		).
		Count(ctx)
	if err != nil {
		return 0, fmt.Errorf("count shared tax_no bindings: %w", err)
	}
	return n, nil
}
