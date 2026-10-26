//go:build unit

package repository

import (
	"context"
	"database/sql"
	"fmt"
	"io/fs"
	"os"
	"testing"
	"testing/fstest"
	"time"

	"github.com/Wei-Shaw/sub2api/migrations"
	"github.com/lib/pq"
	"github.com/stretchr/testify/require"
)

func TestMergedUpstreamAndGiftMigrationsPostgres(t *testing.T) {
	dsn := os.Getenv("SUB2API_GIFT_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("requires an explicitly configured disposable PostgreSQL server")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	admin, err := sql.Open("postgres", dsn)
	require.NoError(t, err)
	defer admin.Close()
	name := fmt.Sprintf("upstream_merge_%d", time.Now().UnixNano())
	_, err = admin.ExecContext(ctx, "CREATE DATABASE "+pq.QuoteIdentifier(name))
	require.NoError(t, err)
	defer func() {
		_, err := admin.ExecContext(context.Background(), "DROP DATABASE "+pq.QuoteIdentifier(name)+" WITH (FORCE)")
		require.NoError(t, err)
	}()
	db, err := sql.Open("postgres", dsn+" dbname="+name)
	require.NoError(t, err)
	defer db.Close()
	// Model an existing custom installation, with both gift migrations already
	// applied, then add the three official migrations with overlapping numbers.
	oldFS := fstest.MapFS{}
	names, err := fs.Glob(migrations.FS, "*.sql")
	require.NoError(t, err)
	for _, filename := range names {
		switch filename {
		case "246_enable_gemini_image_generation_groups.sql", "247_scoped_admin_api_keys.sql", "248_group_image_count_billing.sql":
			continue
		}
		data, err := fs.ReadFile(migrations.FS, filename)
		require.NoError(t, err)
		oldFS[filename] = &fstest.MapFile{Data: data}
	}
	require.NoError(t, applyMigrationsFS(ctx, db, oldFS))
	var userID int64
	require.NoError(t, db.QueryRowContext(ctx, `INSERT INTO users(email,password_hash,balance,total_recharged)
        VALUES('merge-test@example.com','test-only',10.1,10) RETURNING id`).Scan(&userID))
	_, err = db.ExecContext(ctx, `INSERT INTO redeem_codes(code,type,value,status,is_gift,gift_pair_id,used_by,used_at)
        VALUES('MERGE-PAID','balance',10,'used',false,'0123456789abcdef0123456789abcdef',$1,NOW()),
              ('MERGE-GIFT','balance',0.1,'used',true,'0123456789abcdef0123456789abcdef',$1,NOW())`, userID)
	require.NoError(t, err)
	_, err = db.ExecContext(ctx, `INSERT INTO groups(name,platform,allow_image_generation,allow_batch_image_generation)
        VALUES('merge-gemini','gemini',false,false)`)
	require.NoError(t, err)
	require.NoError(t, ApplyMigrations(ctx, db))
	var balance, paid float64
	require.NoError(t, db.QueryRowContext(ctx, "SELECT balance,total_recharged FROM users WHERE id=$1", userID).Scan(&balance, &paid))
	require.Equal(t, 10.1, balance)
	require.Equal(t, 10.0, paid)
	var isGift bool
	var status string
	require.NoError(t, db.QueryRowContext(ctx, "SELECT is_gift,status FROM redeem_codes WHERE code='MERGE-GIFT'").Scan(&isGift, &status))
	require.True(t, isGift)
	require.Equal(t, "used", status)
	var syncImages, batchImages, imageCountBilling bool
	require.NoError(t, db.QueryRowContext(ctx, `SELECT allow_image_generation,allow_batch_image_generation,image_count_billing FROM groups WHERE name='merge-gemini'`).Scan(&syncImages, &batchImages, &imageCountBilling))
	require.True(t, syncImages, "upstream migration preserves pre-gate synchronous image access")
	require.False(t, batchImages, "our separate batch-image permission must not be enabled")
	require.False(t, imageCountBilling, "new billing mode must remain opt-in")
	// Re-running must preserve checksums and skip already applied files.
	require.NoError(t, ApplyMigrations(ctx, db))
	for _, filename := range []string{
		"247_redeem_code_gifts.sql", "247_scoped_admin_api_keys.sql",
		"248_redeem_gift_consumption_guard.sql", "248_group_image_count_billing.sql",
	} {
		var count int
		require.NoError(t, db.QueryRowContext(ctx, "SELECT count(*) FROM schema_migrations WHERE filename = $1", filename).Scan(&count))
		require.Equal(t, 1, count, filename)
	}
	var columns int
	require.NoError(t, db.QueryRowContext(ctx, `SELECT count(*) FROM information_schema.columns WHERE
        (table_name = 'redeem_codes' AND column_name IN ('is_gift', 'gift_pair_id')) OR
        (table_name = 'groups' AND column_name IN ('allow_batch_image_generation', 'image_count_billing'))`).Scan(&columns))
	require.Equal(t, 4, columns)
	var triggers int
	require.NoError(t, db.QueryRowContext(ctx, `SELECT count(*) FROM pg_trigger WHERE tgname IN
        ('redeem_code_gift_identity_immutable','redeem_code_gift_consumption_immutable')`).Scan(&triggers))
	require.Equal(t, 2, triggers)
}
