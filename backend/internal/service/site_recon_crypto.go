package service

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
)

// 站点对账自己的持久密钥：首次使用时随机生成 32 字节，存进 security_secrets 表（只新增一行），
// 之后所有实例、每次重启都读同一把。不依赖 TOTP_ENCRYPTION_KEY（未配置时每次启动随机生成），
// 也不依赖 OpenAI 重新授权功能的凭据密钥（初始化有额外前置条件）。
//
// 密文格式：sr1: + base64(nonce | AES-256-GCM 密文)。

const (
	siteReconSecretKeyName = "site_recon_secret_key_v1"
	siteReconCipherPrefix  = "sr1:"
)

type siteReconKeyring struct {
	store *siteReconStore
	mu    sync.Mutex
	aead  cipher.AEAD
}

func (k *siteReconKeyring) load(ctx context.Context) (cipher.AEAD, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.aead != nil {
		return k.aead, nil
	}
	read := func() (string, error) {
		var v string
		err := k.store.db.QueryRowContext(ctx, k.store.q(`SELECT value FROM security_secrets WHERE key = ?`), siteReconSecretKeyName).Scan(&v)
		return v, err
	}
	value, err := read()
	if errors.Is(err, sql.ErrNoRows) {
		raw := make([]byte, 32)
		if _, err := rand.Read(raw); err != nil {
			return nil, err
		}
		// 多实例并发初始化：DO NOTHING 后再读一次，大家拿到的是同一把。
		if _, err := k.store.db.ExecContext(ctx, k.store.q(`INSERT INTO security_secrets (key, value, created_at, updated_at)
VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) ON CONFLICT (key) DO NOTHING`), siteReconSecretKeyName, hex.EncodeToString(raw)); err != nil {
			return nil, fmt.Errorf("保存站点对账密钥失败: %w", err)
		}
		value, err = read()
	}
	if err != nil {
		return nil, fmt.Errorf("读取站点对账密钥失败: %w", err)
	}
	key, err := hex.DecodeString(strings.TrimSpace(value))
	if err != nil || len(key) != 32 {
		return nil, errors.New("站点对账密钥格式不对")
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	k.aead = aead
	return aead, nil
}

func (k *siteReconKeyring) encrypt(ctx context.Context, plain string) (string, error) {
	aead, err := k.load(ctx)
	if err != nil {
		return "", err
	}
	nonce := make([]byte, aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return "", err
	}
	return siteReconCipherPrefix + base64.StdEncoding.EncodeToString(aead.Seal(nonce, nonce, []byte(plain), nil)), nil
}

func (k *siteReconKeyring) decrypt(ctx context.Context, ciphertext string) (string, error) {
	if !strings.HasPrefix(ciphertext, siteReconCipherPrefix) {
		return "", errors.New("not a site recon ciphertext")
	}
	aead, err := k.load(ctx)
	if err != nil {
		return "", err
	}
	raw, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(ciphertext, siteReconCipherPrefix))
	if err != nil || len(raw) < aead.NonceSize() {
		return "", errors.New("invalid ciphertext")
	}
	plain, err := aead.Open(nil, raw[:aead.NonceSize()], raw[aead.NonceSize():], nil)
	if err != nil {
		return "", err
	}
	return string(plain), nil
}

// encryptSecret 用站点对账自有密钥加密。
func (s *SiteReconService) encryptSecret(ctx context.Context, plain string) (string, error) {
	enc, err := s.keyring.encrypt(ctx, plain)
	if err != nil {
		return "", infraerrors.ServiceUnavailable("SITE_RECON_KEY_UNAVAILABLE", "site recon encryption key unavailable: "+err.Error())
	}
	return enc, nil
}

// decryptSecret 新格式用自有密钥；旧版本（sr1: 之前）写入的密文退回原加密器尝试一次。
func (s *SiteReconService) decryptSecret(ctx context.Context, ciphertext string) (string, error) {
	if strings.HasPrefix(ciphertext, siteReconCipherPrefix) {
		return s.keyring.decrypt(ctx, ciphertext)
	}
	if s.encryptor == nil {
		return "", errors.New("legacy ciphertext without encryptor")
	}
	return s.encryptor.Decrypt(ciphertext)
}
