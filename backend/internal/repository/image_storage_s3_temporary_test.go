package repository

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func temporaryS3ForTest(t *testing.T, handler http.HandlerFunc) *S3ImageStorage {
	t.Helper()
	server := httptest.NewTLSServer(handler)
	t.Cleanup(server.Close)
	client := s3.NewFromConfig(aws.Config{Region: "us-east-1", Credentials: credentials.NewStaticCredentialsProvider("test-key", "test-secret", ""), HTTPClient: server.Client()}, func(o *s3.Options) {
		o.BaseEndpoint = aws.String(server.URL)
		o.UsePathStyle = true
		o.RequestChecksumCalculation = aws.RequestChecksumCalculationWhenRequired
	})
	return &S3ImageStorage{client: client, bucket: "images", publicBaseURL: "https://public.example", presignExpiry: 24 * time.Hour}
}

func TestTemporaryImageS3AlwaysSignsForFiveMinutes(t *testing.T) {
	puts := 0
	s := temporaryS3ForTest(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPut, r.Method)
		puts++
		require.Equal(t, "image/png", r.Header.Get("Content-Type"))
		require.Equal(t, "private, no-store, max-age=0", r.Header.Get("Cache-Control"))
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Equal(t, []byte("test-image"), body)
		w.WriteHeader(200)
	})
	key := service.ExcelBPSImageObjectPrefix + uuid.NewString()
	link, err := s.SaveTemporary(context.Background(), key, "image/png", []byte("test-image"), 5*time.Minute)
	require.NoError(t, err)
	parsed, err := url.Parse(link)
	require.NoError(t, err)
	require.Equal(t, "https", parsed.Scheme)
	require.NotEqual(t, "public.example", parsed.Host)
	require.Equal(t, "300", parsed.Query().Get("X-Amz-Expires"))
	require.NotEmpty(t, parsed.Query().Get("X-Amz-Signature"))
	require.Equal(t, "private, no-store, max-age=0", parsed.Query().Get("response-cache-control"))
	require.Equal(t, "/images/"+key, parsed.Path)
	_, err = s.SaveTemporary(context.Background(), key, "image/png", nil, 24*time.Hour)
	require.Error(t, err)
	_, err = s.SaveTemporary(context.Background(), "backups/other", "image/png", nil, 5*time.Minute)
	require.Error(t, err)
	require.Equal(t, 1, puts)
}

func TestTemporaryImageS3CleanupOnlyExpiredOwnedObjects(t *testing.T) {
	oldKey := service.ExcelBPSImageObjectPrefix + uuid.NewString()
	newKey := service.ExcelBPSImageObjectPrefix + uuid.NewString()
	cutoff := time.Now().UTC().Truncate(time.Second).Add(-6 * time.Minute)
	var deleted []string
	s := temporaryS3ForTest(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		switch r.Method {
		case http.MethodGet:
			require.Equal(t, service.ExcelBPSImageObjectPrefix, r.URL.Query().Get("prefix"))
			fmt.Fprintf(w, `<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated><Contents><Key>%s</Key><LastModified>%s</LastModified></Contents><Contents><Key>%s</Key><LastModified>%s</LastModified></Contents><Contents><Key>backups/never-delete</Key><LastModified>%s</LastModified></Contents><Contents><Key>%snot-owned</Key><LastModified>%s</LastModified></Contents></ListBucketResult>`, oldKey, cutoff.Add(-time.Second).Format(time.RFC3339), newKey, cutoff.Format(time.RFC3339), cutoff.Add(-time.Hour).Format(time.RFC3339), service.ExcelBPSImageObjectPrefix, cutoff.Add(-time.Hour).Format(time.RFC3339))
		case http.MethodPost:
			var body struct {
				Objects []struct {
					Key string `xml:"Key"`
				} `xml:"Object"`
			}
			require.NoError(t, xml.NewDecoder(r.Body).Decode(&body))
			for _, obj := range body.Objects {
				deleted = append(deleted, obj.Key)
			}
			_, _ = io.WriteString(w, `<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"/>`)
		default:
			t.Fatalf("unexpected method: %s", r.Method)
		}
	})
	require.NoError(t, s.DeleteExpiredTemporary(context.Background(), cutoff))
	require.Equal(t, []string{oldKey}, deleted)
	require.Error(t, s.DeleteTemporary(context.Background(), "images/generated.png"))
}

func TestTemporaryImageS3CleanupDetectsPartialDeletionFailures(t *testing.T) {
	s := temporaryS3ForTest(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.Method == http.MethodGet {
			_, _ = fmt.Fprintf(w, `<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated><Contents><Key>%s</Key><LastModified>2020-01-01T00:00:00Z</LastModified></Contents></ListBucketResult>`, service.ExcelBPSImageObjectPrefix+uuid.NewString())
		} else {
			_, _ = io.WriteString(w, `<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Error><Code>AccessDenied</Code><Message>private-key-detail</Message></Error></DeleteResult>`)
		}
	})
	err := s.DeleteExpiredTemporary(context.Background(), time.Now())
	require.Error(t, err)
	require.False(t, strings.Contains(err.Error(), "private-key-detail"))
}
