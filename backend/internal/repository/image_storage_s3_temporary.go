package repository

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/google/uuid"
)

var _ service.TemporaryImageStorage = (*S3ImageStorage)(nil)

func isTemporaryImageKey(key string) bool {
	if !strings.HasPrefix(key, service.ExcelBPSImageObjectPrefix) {
		return false
	}
	_, err := uuid.Parse(strings.TrimPrefix(key, service.ExcelBPSImageObjectPrefix))
	return err == nil
}

// Never return publicBaseURL for customer images, even when generated images
// are configured to use a CDN. The bucket/prefix must remain private.
func (s *S3ImageStorage) SaveTemporary(ctx context.Context, key, contentType string, data []byte, ttl time.Duration) (string, error) {
	if !isTemporaryImageKey(key) || ttl != service.ExcelBPSImageTTL {
		return "", fmt.Errorf("invalid temporary image key or expiry")
	}
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: &s.bucket, Key: &key, Body: bytes.NewReader(data), ContentType: &contentType,
		CacheControl: aws.String("private, no-store, max-age=0"),
	})
	if err != nil {
		return "", fmt.Errorf("temporary image upload failed")
	}
	result, err := s3.NewPresignClient(s.client).PresignGetObject(ctx, &s3.GetObjectInput{
		Bucket: &s.bucket, Key: &key, ResponseCacheControl: aws.String("private, no-store, max-age=0"),
	}, s3.WithPresignExpires(ttl))
	if err != nil {
		return "", fmt.Errorf("temporary image signing failed")
	}
	return result.URL, nil
}

func (s *S3ImageStorage) DeleteTemporary(ctx context.Context, key string) error {
	if !isTemporaryImageKey(key) {
		return fmt.Errorf("refusing to delete a non-temporary image")
	}
	_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{Bucket: &s.bucket, Key: &key})
	if err != nil {
		return fmt.Errorf("temporary image deletion failed")
	}
	return nil
}

// Listing the fixed owned prefix allows cleanup to resume after process
// restarts. Never list or remove generated images or backup objects.
func (s *S3ImageStorage) DeleteExpiredTemporary(ctx context.Context, before time.Time) error {
	pages := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: &s.bucket, Prefix: aws.String(service.ExcelBPSImageObjectPrefix), MaxKeys: aws.Int32(1000),
	})
	for page := 0; page < 20 && pages.HasMorePages(); page++ {
		result, err := pages.NextPage(ctx)
		if err != nil {
			return fmt.Errorf("temporary image listing failed")
		}
		var expired []types.ObjectIdentifier
		for _, object := range result.Contents {
			if object.Key != nil && isTemporaryImageKey(*object.Key) && object.LastModified != nil && object.LastModified.Before(before) {
				expired = append(expired, types.ObjectIdentifier{Key: object.Key})
			}
		}
		if len(expired) == 0 {
			continue
		}
		deleted, err := s.client.DeleteObjects(ctx, &s3.DeleteObjectsInput{
			Bucket: &s.bucket, Delete: &types.Delete{Objects: expired, Quiet: aws.Bool(true)},
		})
		if err != nil || len(deleted.Errors) != 0 {
			return fmt.Errorf("temporary image cleanup failed")
		}
	}
	return nil
}
