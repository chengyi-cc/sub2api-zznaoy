package repository

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/service"
)

func checkCredentialContainerStorage(dir string) error {
	if runtime.GOOS != "linux" {
		return nil
	}
	container := false
	for _, marker := range []string{"/.dockerenv", "/run/.containerenv"} {
		if _, err := os.Stat(marker); err == nil {
			container = true
		}
	}
	if !container {
		return nil
	}
	target, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	// Resolve existing ancestors even when the secrets directory is not created yet.
	suffix := ""
	for {
		resolved, resolveErr := filepath.EvalSymlinks(target)
		if resolveErr == nil {
			target = filepath.Join(resolved, suffix)
			break
		}
		if !errors.Is(resolveErr, os.ErrNotExist) {
			return resolveErr
		}
		parent := filepath.Dir(target)
		if parent == target {
			return resolveErr
		}
		suffix = filepath.Join(filepath.Base(target), suffix)
		target = parent
	}
	data, err := os.ReadFile("/proc/self/mountinfo")
	if err != nil {
		return err
	}
	if !credentialDirectoryHasPersistentMount(target, string(data)) {
		return service.ErrCredentialEncryptionNotPersistent
	}
	return nil
}

// Inspect mount metadata only; no Docker socket or key contents are needed.
func credentialDirectoryHasPersistentMount(directory, mountinfo string) bool {
	directory = strings.TrimRight(directory, "/")
	if directory == "" {
		directory = "/"
	}
	best, persistent := -1, false
	decode := strings.NewReplacer(`\040`, " ", `\011`, "\t", `\012`, "\n", `\134`, `\`)
	for _, line := range strings.Split(mountinfo, "\n") {
		left, right, ok := strings.Cut(line, " - ")
		fields, filesystem := strings.Fields(left), strings.Fields(right)
		if !ok || len(fields) < 6 || len(filesystem) < 1 {
			continue
		}
		mount := decode.Replace(fields[4])
		if directory != mount && !(mount == "/" || strings.HasPrefix(directory, strings.TrimRight(mount, "/")+"/")) {
			continue
		}
		if len(mount) < best {
			continue
		}
		best = len(mount)
		writable := false
		for _, option := range strings.Split(fields[5], ",") {
			if option == "rw" {
				writable = true
			}
		}
		persistent = mount != "/" && writable && filesystem[0] != "tmpfs" && filesystem[0] != "ramfs"
	}
	return best >= 0 && persistent
}
