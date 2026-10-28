package schema

import (
	"github.com/Wei-Shaw/sub2api/ent/schema/mixins"

	"entgo.io/ent"
	"entgo.io/ent/dialect/entsql"
	"entgo.io/ent/schema"
	"entgo.io/ent/schema/field"
	"entgo.io/ent/schema/index"
)

// InvoiceProfile 用户绑定的固定开票资料（仅企业抬头）。
//
// 规则：
//   - 每个用户最多一条（user_id 唯一）；用户只能绑定一次，绑定后即锁定
//   - 仅管理员可修改或删除（删除 = 重置，用户可重新绑定一次）
//   - 发票申请提交时由后端从此表取抬头/税号快照，前端传入的抬头/税号一律忽略
//
// 刻意不建立与 User 的 edge：避免改动 ent 生成的 user*.go（上游同步冲突重灾区）。
// 外键与级联删除由 SQL 迁移保证。
type InvoiceProfile struct {
	ent.Schema
}

func (InvoiceProfile) Annotations() []schema.Annotation {
	return []schema.Annotation{
		entsql.Annotation{Table: "user_invoice_profiles"},
	}
}

func (InvoiceProfile) Mixin() []ent.Mixin {
	return []ent.Mixin{
		mixins.TimeMixin{},
	}
}

func (InvoiceProfile) Fields() []ent.Field {
	return []ent.Field{
		// 绑定用户（外键 → users.id，迁移里 ON DELETE CASCADE）
		field.Int64("user_id").
			Unique().
			Immutable(),

		// 单位名称（发票抬头）
		field.String("title").
			MaxLen(200).
			NotEmpty(),

		// 统一社会信用代码（18 位，已大写规范化并通过校验位校验）
		field.String("tax_no").
			MaxLen(50).
			NotEmpty(),

		// 最后修改该资料的管理员 user_id；nil 表示由用户本人绑定
		field.Int64("updated_by").
			Optional().
			Nillable(),
	}
}

func (InvoiceProfile) Indexes() []ent.Index {
	return []ent.Index{
		index.Fields("tax_no"),
	}
}
