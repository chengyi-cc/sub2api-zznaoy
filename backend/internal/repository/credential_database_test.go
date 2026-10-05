//go:build unit

package repository

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

func TestCredentialEncryptionDatabaseSurvivesEmptyDataDirectory(t *testing.T) {
	t.Setenv("DATA_DIR", t.TempDir())
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer func() { _ = db.Close() }()
	for i := 0; i < 3; i++ {
		mock.ExpectQuery("SELECT value FROM security_secrets").WithArgs(credentialDatabaseKey).WillReturnRows(sqlmock.NewRows([]string{"value"}).AddRow(strings.Repeat("ab", 32)))
	}
	manager := NewOpenAICredentialEncryptor(&config.Config{}, aesEncryptor(t), db)
	status, err := manager.EncryptionStatus()
	require.NoError(t, err)
	require.True(t, status.Configured)
	require.Equal(t, "database", status.Source)
	data, err := json.Marshal(status)
	require.NoError(t, err)
	require.JSONEq(t, `{"configured":true,"source":"database"}`, string(data))
	ciphertext, err := manager.Encrypt("fixture-password")
	require.NoError(t, err)
	t.Setenv("DATA_DIR", t.TempDir())
	restarted := NewOpenAICredentialEncryptor(&config.Config{}, aesEncryptor(t), db)
	plaintext, err := restarted.Decrypt(ciphertext)
	require.NoError(t, err)
	require.Equal(t, "fixture-password", plaintext)
	_, err = os.Stat(filepath.Join(os.Getenv("DATA_DIR"), "secrets"))
	require.True(t, os.IsNotExist(err))
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestCredentialEncryptionDatabaseRefusesInvalidKey(t *testing.T) {
	for _, key := range []string{"", "invalid", strings.Repeat("ab", 16)} {
		t.Run(key, func(t *testing.T) {
			t.Setenv("DATA_DIR", t.TempDir())
			db, mock, err := sqlmock.New()
			require.NoError(t, err)
			defer func() { _ = db.Close() }()
			mock.ExpectQuery("SELECT value FROM security_secrets").WithArgs(credentialDatabaseKey).WillReturnRows(sqlmock.NewRows([]string{"value"}).AddRow(key))
			manager := NewOpenAICredentialEncryptor(&config.Config{}, aesEncryptor(t), db)
			_, err = manager.InitializeEncryption()
			require.Error(t, err)
			require.NoError(t, mock.ExpectationsWereMet())
		})
	}
}
