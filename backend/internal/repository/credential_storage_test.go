package repository

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCredentialDirectoryPersistentMount(t *testing.T) {
	root := "1 0 0:1 / / rw - overlay overlay rw\n"
	for _, tc := range []struct {
		name, dir, info string
		want            bool
	}{
		{"container layer", "/app/data/secrets", root, false},
		{"persistent bind", "/app/data/secrets", root + "2 1 8:1 /srv/data /app/data rw - ext4 /dev/sda rw", true},
		{"named volume", "/app/data/secrets", root + "2 1 8:1 /var/lib/docker/volumes/data/_data /app/data rw - xfs /dev/sda rw", true},
		{"wrong path prefix", "/app/database/secrets", root + "2 1 8:1 /srv/data /app/data rw - ext4 /dev/sda rw", false},
		{"temporary mount", "/app/data/secrets", root + "2 1 0:2 / /app/data rw - tmpfs tmpfs rw", false},
		{"nested temporary mount", "/app/data/secrets", root + "2 1 8:1 /srv/data /app/data rw - ext4 /dev/sda rw\n3 2 0:3 / /app/data/secrets rw - tmpfs tmpfs rw", false},
		{"read only", "/app/data/secrets", root + "2 1 8:1 /srv/data /app/data ro - ext4 /dev/sda rw", false},
		{"escaped directory", "/srv/my data/secrets", root + `2 1 8:1 /srv/data /srv/my\040data rw - ext4 /dev/sda rw`, true},
		{"unreadable metadata", "/app/data/secrets", "invalid", false},
	} {
		t.Run(tc.name, func(t *testing.T) { require.Equal(t, tc.want, credentialDirectoryHasPersistentMount(tc.dir, tc.info)) })
	}
}
