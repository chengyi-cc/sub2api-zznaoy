//go:build unit

package service

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNormalizeInvoiceNotice_TrimsAndCaps(t *testing.T) {
	require.Equal(t, "", normalizeInvoiceNotice("   \n "))
	require.Equal(t, "发票每周四开具。\n暂不支持加急。", normalizeInvoiceNotice("  发票每周四开具。\n暂不支持加急。\n"))

	long := strings.Repeat("票", invoiceNoticeMaxRunes+50)
	got := normalizeInvoiceNotice(long)
	require.Equal(t, invoiceNoticeMaxRunes, len([]rune(got)))
}
