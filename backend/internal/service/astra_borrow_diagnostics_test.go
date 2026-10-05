package service

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAstraBorrowFailureEvidenceIsNotDuplicatedOrSecret(t *testing.T) {
	s, _, _, tr, _ := astraFixture(t)
	base := tr.fn
	calls := 0
	tr.fn = func(req *http.Request, proxy string, id int64) (*http.Response, error) {
		if id == 1 {
			return base(req, proxy, id)
		}
		calls++
		state := strings.Repeat("A", 780)
		if calls == 2 {
			require.Equal(t, state, req.Header.Get("x-codex-turn-state"))
			state = strings.Repeat("B", 780)
		}
		return astraTestResponse("", state, astraComplete), nil
	}
	require.ErrorContains(t, s.Verify(context.Background(), 2), "astra_ticket_changed")
	rows, err := s.History(context.Background(), 0)
	require.NoError(t, err)
	require.Len(t, rows, 2, "one source acquisition and one target result")
	require.Equal(t, int64(1), rows[1].SourceAccountID)
	require.Equal(t, int64(2), rows[1].TargetAccountID)
	snapshot, err := s.Snapshot(context.Background())
	require.NoError(t, err)
	evidence := snapshot.Statuses[1].Probe
	require.NotNil(t, evidence)
	require.Equal(t, &AstraBorrowProbeDetails{MintStatus: 200, ContinueStatus: 200, TicketLength: 780, ContinueTicketLength: 780, NewTicket: true}, evidence)
	data, err := json.Marshal(snapshot)
	require.NoError(t, err)
	for _, secret := range []string{strings.Repeat("A", 780), strings.Repeat("B", 780), "source-secret", "target-secret", "route-secret"} {
		require.NotContains(t, string(data), secret)
	}
}

func TestAstraBorrowProbeReportsPartialAttemptStatus(t *testing.T) {
	for _, failSecond := range []bool{false, true} {
		t.Run(map[bool]string{false: "mint", true: "continue"}[failSecond], func(t *testing.T) {
			s, _, _, tr, _ := astraFixture(t)
			base := tr.fn
			calls := 0
			tr.fn = func(req *http.Request, proxy string, id int64) (*http.Response, error) {
				if id == 1 {
					return base(req, proxy, id)
				}
				calls++
				if failSecond && calls == 1 {
					return astraTestResponse("", strings.Repeat("A", 332), astraComplete), nil
				}
				return &http.Response{StatusCode: 429, Header: http.Header{}, Body: http.NoBody}, nil
			}
			require.ErrorContains(t, s.Verify(context.Background(), 2), "astra_upstream_429")
			snapshot, err := s.Snapshot(context.Background())
			require.NoError(t, err)
			evidence := snapshot.Statuses[1].Probe
			require.NotNil(t, evidence)
			if failSecond {
				require.Equal(t, 200, evidence.MintStatus)
				require.Equal(t, 332, evidence.TicketLength)
				require.Equal(t, 429, evidence.ContinueStatus)
			} else {
				require.Equal(t, 429, evidence.MintStatus)
				require.Zero(t, evidence.ContinueStatus)
				require.Zero(t, evidence.TicketLength)
			}
		})
	}
}

func TestAstraBorrowProbeIdentityMatchesReferenceVersionFloor(t *testing.T) {
	defer SetCodexCanonicalUserAgentResolver(nil)
	for _, tc := range []struct{ configured, expected, originator string }{{"0.145.0", "0.153.4", "codex-tui"}, {"0.200.1", "0.200.1", "codex_cli_rs"}} {
		t.Run(tc.configured, func(t *testing.T) {
			SetCodexCanonicalUserAgentResolver(func() string { return "codex_cli_rs/" + tc.configured + codexCLIUserAgentSuffix })
			s, _, _, _, a := astraFixture(t)
			headers, err := s.headers(context.Background(), a)
			require.NoError(t, err)
			require.Equal(t, tc.expected, headers.Get("version"))
			require.Contains(t, headers.Get("User-Agent"), "/"+tc.expected+" ")
			require.Equal(t, tc.originator, headers.Get("originator"))
			require.Empty(t, headers.Get("X-Codex-Window-ID"))
			require.Equal(t, "Bearer target-secret", headers.Get("Authorization"))
		})
	}
}
