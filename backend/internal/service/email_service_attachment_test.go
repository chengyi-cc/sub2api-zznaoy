package service

import (
	"context"
	"encoding/base64"
	"io"
	"mime"
	"mime/multipart"
	"net/mail"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEmailServiceSendEmailWithAttachment(t *testing.T) {
	ctx := context.Background()
	repo := newNotificationEmailMemorySettingRepo()
	smtpServer := startNotificationEmailTestSMTPServer(t)
	require.NoError(t, repo.SetMultiple(ctx, smtpServer.settings()))

	svc := NewEmailService(repo, nil)
	pdf := []byte("%PDF-1.7\nattachment content\n")
	err := svc.SendEmailWithAttachment(ctx, "billing@example.com", "Invoice ready", "<p>Your invoice is attached.</p>", EmailAttachment{
		Filename:    "invoice-INV-100.pdf",
		ContentType: invoicePDFContentType,
		Data:        pdf,
	})
	require.NoError(t, err)
	require.Equal(t, int64(1), smtpServer.messageCount())

	htmlBody, filename, attachmentData := parseAttachmentEmail(t, smtpServer.lastMessage())
	require.Contains(t, htmlBody, "Your invoice is attached.")
	require.Equal(t, "invoice-INV-100.pdf", filename)
	require.Equal(t, pdf, attachmentData)
}

func parseAttachmentEmail(t *testing.T, raw string) (htmlBody, filename string, attachmentData []byte) {
	t.Helper()

	message, err := mail.ReadMessage(strings.NewReader(raw))
	require.NoError(t, err)
	mediaType, params, err := mime.ParseMediaType(message.Header.Get("Content-Type"))
	require.NoError(t, err)
	require.Equal(t, "multipart/mixed", mediaType)

	reader := multipart.NewReader(message.Body, params["boundary"])
	for {
		part, partErr := reader.NextPart()
		if partErr == io.EOF {
			break
		}
		require.NoError(t, partErr)

		partType, _, parseTypeErr := mime.ParseMediaType(part.Header.Get("Content-Type"))
		require.NoError(t, parseTypeErr)
		switch partType {
		case "text/html":
			body, readErr := io.ReadAll(part)
			require.NoError(t, readErr)
			htmlBody = string(body)
		case invoicePDFContentType:
			_, dispositionParams, dispositionErr := mime.ParseMediaType(part.Header.Get("Content-Disposition"))
			require.NoError(t, dispositionErr)
			filename = dispositionParams["filename"]
			decoded, readErr := io.ReadAll(base64.NewDecoder(base64.StdEncoding, part))
			require.NoError(t, readErr)
			attachmentData = decoded
		}
	}

	require.NotEmpty(t, htmlBody)
	require.NotEmpty(t, filename)
	require.NotEmpty(t, attachmentData)
	return htmlBody, filename, attachmentData
}
