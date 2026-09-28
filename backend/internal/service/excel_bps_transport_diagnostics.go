package service

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/Wei-Shaw/sub2api/internal/util/transportdiag"
	"github.com/gin-gonic/gin"
)

// Never persist raw network errors: they may contain proxy credentials.
func recordExcelBPSTransportFailure(c *gin.Context, account *Account, req *http.Request, err error) []byte {
	kind := transportdiag.Classify(err)
	raw := excelBPSTransportDiagnostic(req, err)
	message := fmt.Sprintf("Excel BPS upstream transport failed (%s); request was not replayed", kind)
	// No HTTP rejection was received. Do not invent a provider 502 or inherit
	// a previous attempt's status; the gateway still returns 502 to its client.
	if c != nil {
		c.Set(OpsUpstreamStatusCodeKey, 0)
	}
	SetOpsUpstreamError(c, 0, message, string(raw))
	appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
		Platform: account.Platform, AccountID: account.ID, AccountName: account.Name,
		ProxyID: opsUpstreamProxyID(account), ProxyName: opsUpstreamProxyName(account),
		UpstreamURL: basispoints.ResponsesURL, Kind: "request_error",
		Stage: "inference", Scope: "excel_bps", Reason: kind, Message: message, Detail: string(raw),
	})
	return raw
}

func excelBPSTransportDiagnostic(req *http.Request, err error) []byte {
	kind := transportdiag.Classify(err)
	diagnostic := map[string]any{"error_kind": kind}
	if req != nil {
		if trace := transportdiag.FromContext(req.Context()); trace != nil {
			diagnostic["transport"] = trace.Snapshot()
		}
	}
	raw, _ := json.Marshal(diagnostic)
	return raw
}
