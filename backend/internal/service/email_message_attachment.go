package service

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"mime/quotedprintable"
	"net/textproto"
	"strings"
)

// buildSMTPMessageWithAttachment 构造带单个附件的 multipart/mixed 邮件。
// 信封头复用 writeCommonEmailHeaders，与纯正文路径保持同样的合规性
// （Date/Message-ID/RFC 2047 主题编码）。
func buildSMTPMessageWithAttachment(config *SMTPConfig, to, subject, body string, attachment EmailAttachment) (smtpMessage, error) {
	var message bytes.Buffer
	fromAddress, recipientAddress, err := writeCommonEmailHeaders(&message, config, to, subject)
	if err != nil {
		return smtpMessage{}, err
	}

	filename := strings.TrimSpace(sanitizeEmailHeader(attachment.Filename))
	if filename == "" {
		filename = "attachment"
	}
	contentType := strings.TrimSpace(sanitizeEmailHeader(attachment.ContentType))
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return smtpMessage{}, fmt.Errorf("invalid attachment content type: %w", err)
	}

	multipartWriter := multipart.NewWriter(&message)
	fmt.Fprintf(&message, "MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=%q\r\n\r\n", multipartWriter.Boundary())

	htmlHeader := make(textproto.MIMEHeader)
	htmlHeader.Set("Content-Type", "text/html; charset=UTF-8")
	htmlHeader.Set("Content-Transfer-Encoding", "quoted-printable")
	htmlPart, err := multipartWriter.CreatePart(htmlHeader)
	if err != nil {
		return smtpMessage{}, fmt.Errorf("create email HTML part: %w", err)
	}
	bodyWriter := quotedprintable.NewWriter(htmlPart)
	if _, err := bodyWriter.Write([]byte(body)); err != nil {
		return smtpMessage{}, fmt.Errorf("encode email body: %w", err)
	}
	if err := bodyWriter.Close(); err != nil {
		return smtpMessage{}, fmt.Errorf("close email body encoder: %w", err)
	}
	attachmentHeader := make(textproto.MIMEHeader)
	attachmentHeader.Set("Content-Type", mime.FormatMediaType(mediaType, map[string]string{"name": filename}))
	attachmentHeader.Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": filename}))
	attachmentHeader.Set("Content-Transfer-Encoding", "base64")
	attachmentPart, err := multipartWriter.CreatePart(attachmentHeader)
	if err != nil {
		return smtpMessage{}, fmt.Errorf("create email attachment part: %w", err)
	}
	if err := writeMIMEBase64(attachmentPart, attachment.Data); err != nil {
		return smtpMessage{}, fmt.Errorf("write email attachment: %w", err)
	}
	if err := multipartWriter.Close(); err != nil {
		return smtpMessage{}, fmt.Errorf("close email multipart body: %w", err)
	}

	return smtpMessage{
		envelopeFrom: fromAddress.Address,
		envelopeTo:   recipientAddress.Address,
		data:         message.Bytes(),
	}, nil
}

// writeMIMEBase64 按 RFC 2045 的 76 列限制写出 base64 编码内容。
func writeMIMEBase64(dst io.Writer, data []byte) error {
	encoded := base64.StdEncoding.EncodeToString(data)
	for len(encoded) > 76 {
		if _, err := io.WriteString(dst, encoded[:76]+"\r\n"); err != nil {
			return err
		}
		encoded = encoded[76:]
	}
	if len(encoded) > 0 {
		_, err := io.WriteString(dst, encoded+"\r\n")
		return err
	}
	return nil
}
