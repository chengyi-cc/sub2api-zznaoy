package service

import (
	"context"
	"log/slog"
	"sort"
	"time"
)

// ImageJobCleanerConfig 控制清理 ticker 的行为。
type ImageJobCleanerConfig struct {
	Store         *ImageJobStore
	TTL           time.Duration
	MaxTotalBytes int64
	Interval      time.Duration
	// ZombieThreshold: 处于 queued/processing 状态超过该时长才被视为"僵尸任务"清理。
	// 应至少为 RunTimeout + 一定 grace。<=0 时按 2*TTL 兜底。
	ZombieThreshold time.Duration
}

// StartImageJobCleaner 启动后台清理循环：
//  1. 启动时立即把残留的 queued/processing 任务标记为 failed（server_restarted）；
//  2. 每隔 Interval 跑一次 runOnce：先按 TTL 删过期 *终态* 任务，再按磁盘配额强删。
//
// ctx 取消后 goroutine 自动退出。
func StartImageJobCleaner(ctx context.Context, cfg ImageJobCleanerConfig) {
	if cfg.Store == nil || cfg.Interval <= 0 {
		return
	}
	if cfg.ZombieThreshold <= 0 {
		cfg.ZombieThreshold = 2 * cfg.TTL
	}
	cfg.markStaleAtStartup()
	cfg.runOnce()
	go func() {
		ticker := time.NewTicker(cfg.Interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				cfg.runOnce()
			}
		}
	}()
}

func isTerminalImageJobStatus(s ImageJobStatus) bool {
	return s == ImageJobStatusSucceeded || s == ImageJobStatusFailed
}

func (c ImageJobCleanerConfig) runOnce() {
	metas, err := c.Store.ListAll()
	if err != nil {
		slog.Warn("image_job_cleaner.list_failed", "error", err)
		return
	}

	now := time.Now().Unix()

	// TTL 阶段：
	//   - 终态任务（succeeded/failed）一旦超过 TTL 即删；
	//   - 活任务（queued/processing）只有超过 ZombieThreshold 才删（防止跑到一半被腰斩）。
	if c.TTL > 0 {
		ttlSec := int64(c.TTL.Seconds())
		zombieSec := int64(c.ZombieThreshold.Seconds())
		for _, m := range metas {
			age := now - m.CreatedAt
			var deadline int64
			if isTerminalImageJobStatus(m.Status) {
				deadline = ttlSec
			} else {
				deadline = zombieSec
			}
			if age < deadline {
				continue
			}
			if err := c.Store.Delete(m.JobID); err != nil {
				slog.Warn("image_job_cleaner.delete_failed", "job_id", m.JobID, "status", string(m.Status), "error", err)
			}
		}
	}

	// 磁盘硬上限阶段：超出时按 created_at 由旧到新强删，
	// 但**优先删终态** —— 所有终态删完仍超限时，只删已经超过 ZombieThreshold 的活任务。
	// 未达阈值的活任务即便会暂时让总容量超过 MaxTotalBytes，也绝不动它们：
	// runner 可能已经计费、正在写 result.json，把目录删了会让用户既扣了钱又拿不到图。
	if c.MaxTotalBytes > 0 {
		size, err := c.Store.TotalSizeBytes()
		if err != nil {
			slog.Warn("image_job_cleaner.size_failed", "error", err)
			return
		}
		if size <= c.MaxTotalBytes {
			return
		}
		metas, err = c.Store.ListAll()
		if err != nil {
			slog.Warn("image_job_cleaner.list_failed", "error", err)
			return
		}
		zombieSec := int64(c.ZombieThreshold.Seconds())
		now2 := time.Now().Unix()
		// 终态优先 → 旧的优先
		sort.SliceStable(metas, func(i, j int) bool {
			ti := isTerminalImageJobStatus(metas[i].Status)
			tj := isTerminalImageJobStatus(metas[j].Status)
			if ti != tj {
				return ti
			}
			return metas[i].CreatedAt < metas[j].CreatedAt
		})
		for _, m := range metas {
			if size <= c.MaxTotalBytes {
				break
			}
			// 活任务必须超过 ZombieThreshold 才动；否则即便超限也跳过。
			if !isTerminalImageJobStatus(m.Status) && now2-m.CreatedAt < zombieSec {
				continue
			}
			if err := c.Store.Delete(m.JobID); err != nil {
				slog.Warn("image_job_cleaner.delete_failed", "job_id", m.JobID, "status", string(m.Status), "error", err)
				continue
			}
			if newSize, err := c.Store.TotalSizeBytes(); err == nil {
				size = newSize
			}
		}
		if size > c.MaxTotalBytes {
			slog.Warn("image_job_cleaner.disk_cap_still_exceeded",
				"size_bytes", size,
				"cap_bytes", c.MaxTotalBytes,
				"note", "remaining live jobs are within zombie grace; will retry next tick",
			)
		}
	}
}

// markStaleAtStartup 启动时把所有 queued/processing 状态强制标 failed。
// 这是因为我们不持久化 goroutine 的状态：上一次运行未完成的任务 goroutine 已经永远消失了。
func (c ImageJobCleanerConfig) markStaleAtStartup() {
	metas, err := c.Store.ListAll()
	if err != nil {
		slog.Warn("image_job_cleaner.startup_list_failed", "error", err)
		return
	}
	now := time.Now().Unix()
	for _, m := range metas {
		if isTerminalImageJobStatus(m.Status) {
			continue
		}
		_, err := c.Store.UpdateMeta(m.JobID, func(meta *ImageJobMeta) error {
			meta.Status = ImageJobStatusFailed
			meta.ErrorType = "server_restarted"
			meta.ErrorMessage = "Server restarted before this job completed."
			meta.HTTPStatus = 500
			meta.CompletedAt = now
			return nil
		})
		if err != nil {
			slog.Warn("image_job_cleaner.startup_mark_failed", "job_id", m.JobID, "error", err)
		}
	}
}
