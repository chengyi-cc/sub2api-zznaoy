package repository

import (
	"context"
	"database/sql/driver"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/require"
)

func TestOpsStatusConsistency_ErrorDetail(t *testing.T) {
	for _, upstream := range []driver.Value{int64(403), nil} {
		db, mock := newSQLMock(t)
		repo := &opsRepository{db: db}
		effective := driver.Value(int64(502))
		if upstream != nil {
			effective = upstream
		}
		columns := strings.Fields("id created_at phase type owner source severity effective client platform model resolved resolved_at resolved_by client_request_id request_id message body upstream upstream_message upstream_detail upstream_errors business user email key account account_name group group_name ip path stream inbound outbound requested_model upstream_model request_type ua auth routing upstream_latency response_latency ttft key_prefix key_name key_deleted")
		values := []driver.Value{
			int64(1), time.Now(), "request", "upstream_error", "provider", "gateway", "P1", effective, int64(502),
			"openai", "gpt-5", false, nil, nil, "client-1", "req-1", "denied", "{}", upstream, "denied", "{}", "[]", false,
			nil, "", nil, nil, "", nil, "", nil, "/v1/responses", false, "", "", "", "", nil, "", nil, nil, nil, nil, nil, "", "", nil,
		}
		require.Len(t, values, len(columns))
		mock.ExpectQuery(`COALESCE\(e.upstream_status_code, e.status_code, 0\),\s+e.status_code,`).WithArgs(int64(1)).WillReturnRows(sqlmock.NewRows(columns).AddRow(values...))
		detail, err := repo.GetErrorLogByID(context.Background(), 1)
		require.NoError(t, err)
		require.Equal(t, int(effective.(int64)), detail.StatusCode)
		require.NotNil(t, detail.ClientStatusCode)
		require.Equal(t, 502, *detail.ClientStatusCode)
		if upstream == nil {
			require.Nil(t, detail.UpstreamStatusCode)
		} else {
			require.Equal(t, 403, *detail.UpstreamStatusCode)
		}
		require.NoError(t, mock.ExpectationsWereMet())
	}
}

func TestOpsStatusConsistency_RequestList(t *testing.T) {
	db, mock := newSQLMock(t)
	repo := &opsRepository{db: db}
	mock.ExpectQuery(`o.status_code AS status_code,\s+o.upstream_status_code AS upstream_status_code,[\s\S]*SELECT COUNT\(1\)`).WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(3))
	columns := strings.Fields("kind created_at request_id platform model duration status upstream_status error_id phase severity message user_id api_key_id account_id group_id stream")
	rows := sqlmock.NewRows(columns).
		AddRow("error", time.Now(), "req-1", "openai", "gpt-5", 10, 502, 403, 1, "request", "P1", "denied", nil, nil, nil, nil, false).
		AddRow("error", time.Now(), "req-2", "openai", "gpt-5", 10, 400, nil, 2, "request", "P3", "invalid", nil, nil, nil, nil, false).
		AddRow("success", time.Now(), "req-3", "openai", "gpt-5", 10, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, false)
	mock.ExpectQuery(`SELECT\s+kind,[\s\S]*status_code,\s+upstream_status_code,\s+error_id`).WillReturnRows(rows)
	items, total, err := repo.ListRequestDetails(context.Background(), nil)
	require.NoError(t, err)
	require.EqualValues(t, 3, total)
	require.Len(t, items, 3)
	require.Equal(t, 502, *items[0].StatusCode)
	require.Equal(t, 403, *items[0].UpstreamStatusCode)
	require.Nil(t, items[1].UpstreamStatusCode)
	require.Nil(t, items[2].StatusCode)
	require.Nil(t, items[2].UpstreamStatusCode)
	require.NoError(t, mock.ExpectationsWereMet())
}
