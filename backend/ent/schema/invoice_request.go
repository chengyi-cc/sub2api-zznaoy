package schema

import (
	"github.com/Wei-Shaw/sub2api/ent/schema/mixins"
	"github.com/Wei-Shaw/sub2api/internal/domain"

	"entgo.io/ent"
	"entgo.io/ent/dialect"
	"entgo.io/ent/dialect/entsql"
	"entgo.io/ent/schema"
	"entgo.io/ent/schema/edge"
	"entgo.io/ent/schema/field"
	"entgo.io/ent/schema/index"
)

// InvoiceRequest 用户发票申请。
//
// 删除策略：硬删除
//   - 申请记录通过 status 字段追踪生命周期（pending/approved/rejected/issued）
//   - 关联 payment_order_ids 以 JSONB 存储，支持多单合并开票
//   - PDF 文件以 invoice_file_path 引用服务器本地相对路径，绝不直接暴露公网
type InvoiceRequest struct {
	ent.Schema
}

func (InvoiceRequest) Annotations() []schema.Annotation {
	return []schema.Annotation{
		entsql.Annotation{Table: "invoice_requests"},
	}
}

func (InvoiceRequest) Mixin() []ent.Mixin {
	return []ent.Mixin{
		mixins.TimeMixin{},
	}
}

func (InvoiceRequest) Fields() []ent.Field {
	return []ent.Field{
		// 申请人（外键 → users.id）
		field.Int64("user_id"),

		// 关联的支付订单 IDs（JSONB 数组，支持多单合并开票）
		field.JSON("payment_order_ids", []int64{}).
			SchemaType(map[string]string{dialect.Postgres: "jsonb"}),

		// 关联的兑换码 IDs（JSONB 数组；仅 type=balance 的已使用兑换码可以开票，
		// 与 payment_order_ids 可混合同一张发票）
		field.JSON("redeem_code_ids", []int64{}).
			SchemaType(map[string]string{dialect.Postgres: "jsonb"}).
			Default([]int64{}),

		// 开票总金额（= 所有关联订单 amount + 兑换码 value 之和，冗余存储用于审核与一致性校验）
		field.Float("amount").
			SchemaType(map[string]string{dialect.Postgres: "decimal(20,2)"}),

		// 发票类型：personal / company
		field.String("invoice_type").
			MaxLen(20).
			Default(domain.InvoiceTypePersonal),

		// 抬头
		field.String("title").
			MaxLen(200).
			NotEmpty(),

		// 税号（统一社会信用代码，企业必填）
		field.String("tax_no").
			MaxLen(50).
			Optional().
			Nillable(),

		// 接收发票邮箱（可选；未填时使用 user.email）
		field.String("recipient_email").
			MaxLen(255).
			Optional().
			Nillable(),

		// 备注
		field.String("remark").
			SchemaType(map[string]string{dialect.Postgres: "text"}).
			Optional().
			Nillable(),

		// 状态：pending / approved / rejected / issued
		field.String("status").
			MaxLen(20).
			Default(domain.InvoiceStatusPending),

		// 驳回原因（仅 status=rejected 时有值）
		field.String("reject_reason").
			SchemaType(map[string]string{dialect.Postgres: "text"}).
			Optional().
			Nillable(),

		// 发票号（仅 status=issued 时有值）
		field.String("invoice_no").
			MaxLen(100).
			Optional().
			Nillable(),

		// 发票 PDF 服务器内部相对路径（如 invoices/2026/05/<uuid>.pdf）
		// 重要：此路径不可直接拼接 URL 公网暴露，仅通过鉴权下载接口返回
		field.String("invoice_file_path").
			MaxLen(500).
			Optional().
			Nillable(),

		// 开票时间
		field.Time("issued_at").
			Optional().
			Nillable().
			SchemaType(map[string]string{dialect.Postgres: "timestamptz"}),

		// 处理人（管理员 user_id）
		field.Int64("processed_by").
			Optional().
			Nillable(),
	}
}

func (InvoiceRequest) Edges() []ent.Edge {
	return []ent.Edge{
		edge.From("user", User.Type).
			Ref("invoice_requests").
			Field("user_id").
			Unique().
			Required(),
	}
}

func (InvoiceRequest) Indexes() []ent.Index {
	return []ent.Index{
		index.Fields("user_id"),
		index.Fields("status"),
		index.Fields("created_at"),
		index.Fields("user_id", "status"),
	}
}
