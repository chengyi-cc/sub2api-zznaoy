#!/usr/bin/env bash
#
# 同步上游后的一键验证。用法：bash scripts/verify-sync.sh [--quick]
#
#   --quick  只跑 build/vet/typecheck，跳过耗时的单测与集成测试（约 30s）
#
# 设计要点：不用 -skip 把已知失败藏起来，而是**跑全量再与下面的清单比对**。
# 这样三种情况都能区分开：
#   1. 清单里的失败又失败了  → 静默，属预期
#   2. 清单里的失败不再失败了 → 提示「可以摘掉了」（上游修好了）
#   3. 出现清单外的新失败    → 报错退出，这才是真需要看的
#
# 每次同步上游后跑一遍，只看最后的 SUMMARY 即可。
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# ============================================================================
# 已知失败清单 —— 全部经实证与本仓库的自有改动无关。
# 加/减条目前请先确认根因，不要拿它当"跑不过就加进来"的垃圾桶。
# ============================================================================

# 后端（Go 测试名，正则；用 | 分隔）
declare -a BACKEND_KNOWN=(
  # 上游测试自身的数据竞争：fakeCNQuotaProber.probed 被 4 个 goroutine 无锁
  # append（cnQuotaProbeConcurrency=4），丢更新导致 ElementsMatch 断言失败。
  # go test -race 在纯上游代码树上同样报 DATA RACE 同样失败。
  'TestCNProviderBalanceCheckRunOnceProbesCodingPlanQuota'

  # 上游测试用 time.Nanosecond TTL + 后台 goroutine 刷新 + Eventually(1s)。
  # 单独跑能过，全包并发跑时抢不到调度。Linux CI 通常过，Windows 稳定失败。
  'TestContentModerationRuntimeSnapshotRefreshFailureKeepsStaleConfig'

  # 需要访问外网 https://tls.peet.ws/api/all，本机被中间人证书拦截
  # （x509: certificate signed by unknown authority）。纯环境问题。
  'TestJA3Fingerprint'
  'TestAllProfiles'
)

# 前端（vitest fullName 全串精确匹配）
declare -a FRONTEND_KNOWN=(
  # 上游改了 CreateAccountModal.vue 的模板但没同步自己 spec 里的字符串断言
  # （断言找 "? 'xai-...'"）。spec 与 .vue 两个文件都与 upstream/main 字节一致。
  'CreateAccountModal Grok account types offers API-key setup alongside OAuth with the official xAI default'
)

# ============================================================================

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYA=$'\033[36m'; RST=$'\033[0m'
FAILURES=()
NOTES=()

section() { printf '\n%s==> %s%s\n' "$CYA" "$1" "$RST"; }
ok()      { printf '%s  OK%s   %s\n' "$GRN" "$RST" "$1"; }
bad()     { printf '%s  FAIL%s %s\n' "$RED" "$RST" "$1"; FAILURES+=("$1"); }
note()    { printf '%s  NOTE%s %s\n' "$YEL" "$RST" "$1"; NOTES+=("$1"); }

# 用 | 拼成 Go 的 -run 正则
backend_known_regex() {
  local IFS='|'
  echo "${BACKEND_KNOWN[*]}"
}

# 判断某个 Go 测试名是否在已知清单里
backend_is_known() {
  local name="$1"
  for known in "${BACKEND_KNOWN[@]}"; do
    [ "$name" = "$known" ] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# 后端：build / vet
# ---------------------------------------------------------------------------
section "后端 build"
if (cd backend && go build ./... 2>&1 | head -30); then
  ok "go build ./..."
else
  bad "go build ./... 失败（见上方输出）"
fi

section "后端 vet"
vet_out="$(cd backend && go vet ./... 2>&1 | grep -v '^#' | head -30)"
if [ -z "$vet_out" ]; then
  ok "go vet ./..."
else
  printf '%s\n' "$vet_out"
  bad "go vet ./... 有告警"
fi

# ---------------------------------------------------------------------------
# 后端测试：跑全量 → 解析失败的测试名 → 与清单比对
# Windows 上 AV 会瞬时拦截 fork/exec（表现为包在 ~1s 内失败且无测试级报错），
# 所以对失败的包整体重跑一次，只有两次都失败才算真失败。
# ---------------------------------------------------------------------------
run_go_suite() {
  local tag="$1" label="$2"
  section "后端测试 ($label)"

  local out
  out="$(cd backend && go test -tags="$tag" ./... 2>/dev/null)"

  # 失败的包（可能含 AV 瞬时拦截）
  local failed_pkgs
  failed_pkgs="$(printf '%s\n' "$out" | awk '/^FAIL[ \t]+github/ {print $2}' | sort -u)"

  if [ -n "$failed_pkgs" ]; then
    printf '  重跑失败的包（排除 AV 瞬时拦截）：\n'
    local retry_pkgs=()
    while IFS= read -r pkg; do
      [ -n "$pkg" ] && retry_pkgs+=("./${pkg#github.com/Wei-Shaw/sub2api/}")
    done <<< "$failed_pkgs"
    local retry_out
    retry_out="$(cd backend && go test -tags="$tag" "${retry_pkgs[@]}" 2>/dev/null)"
    out="$retry_out"
  fi

  # 解析测试级失败
  local failed_tests
  failed_tests="$(printf '%s\n' "$out" \
    | sed -n 's/^[[:space:]]*--- FAIL: \([A-Za-z0-9_]*\).*/\1/p' | sort -u)"

  local new_failures=0
  if [ -n "$failed_tests" ]; then
    while IFS= read -r t; do
      [ -z "$t" ] && continue
      if backend_is_known "$t"; then
        printf '  %s(已知)%s %s\n' "$YEL" "$RST" "$t"
      else
        bad "[$label] 新失败测试：$t"
        new_failures=1
      fi
    done <<< "$failed_tests"
  fi

  # 包级失败但没有测试级失败 → 编译错误或 AV 二次拦截，必须报出来
  local still_failed_pkgs
  still_failed_pkgs="$(printf '%s\n' "$out" | awk '/^FAIL[ \t]+github/ {print $2}' | sort -u)"
  if [ -n "$still_failed_pkgs" ] && [ -z "$failed_tests" ]; then
    printf '%s\n' "$still_failed_pkgs" | while IFS= read -r p; do
      [ -n "$p" ] && printf '  包级失败（无测试级报错，疑编译错误）：%s\n' "$p"
    done
    bad "[$label] 有包级失败但无测试级失败，需人工看"
  elif [ "$new_failures" -eq 0 ]; then
    ok "[$label] 无清单外失败"
  fi

  # 清单里没再失败的 → 提示可以摘掉
  for known in "${BACKEND_KNOWN[@]}"; do
    case "$known" in TestJA3Fingerprint|TestAllProfiles)
      [ "$tag" = "unit" ] && continue ;;  # 这两个只有 integration tag 才跑
    esac
    if ! printf '%s\n' "$failed_tests" | grep -qx "$known"; then
      note "[$label] 已知失败 $known 本次没失败 —— 若持续如此，可从清单摘掉"
    fi
  done
}

if [ "$QUICK" -eq 0 ]; then
  run_go_suite unit "unit"
  run_go_suite integration "integration"
else
  note "--quick：跳过后端 unit/integration"
fi

# ---------------------------------------------------------------------------
# 前端：typecheck + 单测（JSON reporter 精确比对）
# 注意必须用 pnpm exec 而不是 pnpm run test:run --，后者不透传参数。
# ---------------------------------------------------------------------------
section "前端 typecheck"
if (cd frontend && pnpm run typecheck >/dev/null 2>&1); then
  ok "pnpm run typecheck (vue-tsc)"
else
  (cd frontend && pnpm run typecheck 2>&1 | tail -20)
  bad "前端 typecheck 失败"
fi

if [ "$QUICK" -eq 0 ]; then
  section "前端单测"
  report="frontend/vitest-report.json"
  rm -f "$report"
  (cd frontend && pnpm exec vitest run --reporter=json --outputFile=vitest-report.json >/dev/null 2>&1)

  if [ ! -f "$report" ]; then
    bad "前端单测没有产出 JSON 报告（vitest 未运行？）"
  else
    py_out="$(FRONTEND_KNOWN_JOINED="$(printf '%s\n' "${FRONTEND_KNOWN[@]}")" \
      python - "$report" <<'PY'
import io, json, os, sys

report_path = sys.argv[1]
known = {l for l in os.environ.get('FRONTEND_KNOWN_JOINED', '').split('\n') if l.strip()}
data = json.load(io.open(report_path, encoding='utf-8'))

failed = set()
for tr in data.get('testResults', []):
    for a in tr.get('assertionResults', []):
        if a.get('status') == 'failed':
            failed.add(a.get('fullName', '').strip())

total = data.get('numTotalTests', 0)
print(f"TOTAL {total}")
for f in sorted(failed - known):
    print(f"NEW {f}")
for f in sorted(failed & known):
    print(f"KNOWN {f}")
for f in sorted(known - failed):
    print(f"GONE {f}")
PY
)"
    total_line="$(printf '%s\n' "$py_out" | sed -n 's/^TOTAL \(.*\)/\1/p')"
    new_cnt=0
    while IFS= read -r line; do
      case "$line" in
        NEW\ *)   bad "前端新失败：${line#NEW }"; new_cnt=$((new_cnt+1)) ;;
        KNOWN\ *) printf '  %s(已知)%s %s\n' "$YEL" "$RST" "${line#KNOWN }" ;;
        GONE\ *)  note "已知前端失败不再失败 —— 可从清单摘掉：${line#GONE }" ;;
      esac
    done <<< "$py_out"
    [ "$new_cnt" -eq 0 ] && ok "前端单测无清单外失败（共 ${total_line:-?} 个用例）"
    rm -f "$report"
  fi
else
  note "--quick：跳过前端单测"
fi

# ---------------------------------------------------------------------------
# 同步后的例行核对：Dockerfile 的 Go 版本要跟 go.mod 一致
# 上游升 go.mod 时常忘了改根 Dockerfile，会炸「服务器上从源码构建」这条链。
# ---------------------------------------------------------------------------
section "Dockerfile Go 版本对齐"
gomod_go="$(grep -m1 '^go ' backend/go.mod | awk '{print $2}')"
docker_go="$(grep -m1 'ARG GOLANG_IMAGE' Dockerfile | sed 's/.*golang:\([0-9.]*\)-alpine.*/\1/')"
if [ "$gomod_go" = "$docker_go" ]; then
  ok "go.mod ($gomod_go) == Dockerfile GOLANG_IMAGE ($docker_go)"
else
  bad "go.mod 是 $gomod_go 但 Dockerfile GOLANG_IMAGE 是 $docker_go —— 需跟着 go.mod 改"
fi

# ---------------------------------------------------------------------------
section "SUMMARY"
if [ "${#NOTES[@]}" -gt 0 ]; then
  printf '%s提示 %d 条：%s\n' "$YEL" "${#NOTES[@]}" "$RST"
  for n in "${NOTES[@]}"; do printf '  - %s\n' "$n"; done
fi
if [ "${#FAILURES[@]}" -eq 0 ]; then
  printf '%s全部通过（已知失败已按清单忽略）%s\n' "$GRN" "$RST"
  exit 0
fi
printf '%s%d 处需要处理：%s\n' "$RED" "${#FAILURES[@]}" "$RST"
for f in "${FAILURES[@]}"; do printf '  - %s\n' "$f"; done
exit 1
