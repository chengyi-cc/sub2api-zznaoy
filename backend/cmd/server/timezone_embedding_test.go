package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// Exercise the real server package with both configured and Go-installed zone
// databases absent. This file deliberately does not import time/tzdata.
func TestServerEmbeddedTimezoneWithoutExternalDatabase(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows regression: Unix hosts may still supply a system timezone database")
	}
	if os.Getenv("SUB2API_TZDATA_TEST_CHILD") == "1" {
		for _, tc := range []struct {
			zone   string
			month  time.Month
			offset int
		}{
			{"Asia/Shanghai", time.January, 8 * 3600},
			{"America/Los_Angeles", time.January, -8 * 3600},
			{"America/Los_Angeles", time.July, -7 * 3600},
		} {
			loc, err := time.LoadLocation(tc.zone)
			if err != nil {
				t.Fatal(err)
			}
			_, offset := time.Date(2026, tc.month, 15, 12, 0, 0, 0, loc).Zone()
			if offset != tc.offset {
				t.Fatalf("%s/%s offset=%d, want %d", tc.zone, tc.month, offset, tc.offset)
			}
		}
		return
	}
	empty := t.TempDir()
	t.Setenv("ZONEINFO", filepath.Join(empty, "absent-zoneinfo"))
	t.Setenv("GOROOT", empty)
	t.Setenv("SUB2API_TZDATA_TEST_CHILD", "1")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(executable, "-test.run=^TestServerEmbeddedTimezoneWithoutExternalDatabase$")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("embedded timezone lookup: %v\n%s", err, out)
	}
}
