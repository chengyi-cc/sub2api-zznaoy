#!/usr/bin/env bash
# Run in the existing deployment directory. No host Python or local build needed.
set -euo pipefail
umask 077
image='ghcr.io/chengyi-cc/sub2api:latest'
container=''
pull_image=true
while (($#)); do
  case "$1" in
    --image) image="${2:?--image requires a published image}"; shift 2 ;;
    --local-image) image="${2:?--local-image requires an image}"; pull_image=false; shift 2 ;;
    --container) container="${2:?--container requires a name}"; shift 2 ;;
    -h|--help) echo 'Usage: bash upgrade-prism.sh [--image IMAGE | --local-image IMAGE] [--container NAME]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
fail() { echo "Upgrade stopped: $*" >&2; exit 1; }
command -v docker >/dev/null || fail 'Docker is required'
docker compose version >/dev/null || fail 'Docker Compose v2 is required'
label() { docker inspect --format "{{index .Config.Labels \"$2\"}}" "$1"; }
if [[ -z "$container" ]]; then
  matches=()
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    directory=$(label "$candidate" com.docker.compose.project.working_dir)
    if [[ "$directory" == "$(pwd -P)" ]]; then matches+=("$candidate"); fi
  done < <(docker ps -a --filter label=com.docker.compose.service=sub2api --format '{{.ID}}')
  ((${#matches[@]} == 1)) || fail 'Run from the original deployment directory, or select --container NAME'
  container="${matches[0]}"
fi
[[ $(label "$container" com.docker.compose.service) == sub2api ]] || fail 'Container is not a Compose sub2api service'
project=$(label "$container" com.docker.compose.project)
directory=$(label "$container" com.docker.compose.project.working_dir)
source=$(label "$container" com.docker.compose.project.config_files)
[[ -n "$project" && -d "$directory" && -f "$source" && "$source" != *,* ]] || fail 'A single original Compose file is required; multiple-file deployments need an explicit migration'
cd "$directory"
[[ $(cd -- "$(dirname -- "$source")" && pwd -P) == "$(pwd -P)" ]] || fail 'Compose file must be in the original project directory'
file=$(basename -- "$source")
# Preserve the environment file(s) recorded by Compose. Never print their contents.
environment_files=$(label "$container" com.docker.compose.project.environment_file)
compose=(docker compose --project-directory "$directory" -p "$project")
if [[ -n "$environment_files" && "$environment_files" != '<no value>' ]]; then
  IFS=',' read -r -a env_files <<< "$environment_files"
  for env_file in "${env_files[@]}"; do
    [[ -f "$env_file" ]] || fail 'An original environment file is missing'
    compose+=(--env-file "$env_file")
  done
elif [[ -f .env ]]; then
  compose+=(--env-file "$directory/.env")
fi
[[ -z "${COMPOSE_FILE:-}" ]] || fail 'Unset COMPOSE_FILE before running this single-file upgrade'
base=("${compose[@]}" -f "$source")
"${base[@]}" config --quiet
original_id=$(docker inspect --format '{{.Id}}' "$container")
resolved_id=$("${base[@]}" ps -a -q sub2api)
[[ -n "$resolved_id" && $(docker inspect --format '{{.Id}}' "$resolved_id") == "$original_id" ]] || fail 'Compose configuration does not select the original application container'
echo "Preparing bundled browser image for project $project ..."
if [[ "$pull_image" == true ]]; then docker pull "$image"; fi
[[ $(docker image inspect --format '{{index .Config.Labels "io.sub2api.prism.managed"}}' "$image") == 1 ]] || fail 'Selected image has no bundled browser. Publish the updated release first, then retry'
image_id=$(docker image inspect --format '{{.Id}}' "$image")
helper=(docker run --rm --network none --user "$(id -u):$(id -g)" --mount "type=bind,src=$directory,dst=/deployment" --entrypoint python3 "$image_id" /opt/sub2api/deploy/prism/upgrade_config.py)
backup=$("${helper[@]}" prepare --file "$file" --image "$image")
[[ "$backup" == .prism-upgrade-* && "$backup" != */* && -d "$backup" ]] || fail 'Could not prepare configuration backup'
"${base[@]}" config --format json > "$backup/before.json"
"${compose[@]}" -f "$directory/$backup/candidate.yml" config --format json > "$backup/after.json"
docker inspect "$container" > "$backup/container.json"
"${helper[@]}" validate --backup "$backup"
rm -f -- "$backup/before.json" "$backup/after.json" "$backup/container.json"
docker run --rm --network none --entrypoint cat "$image_id" /opt/sub2api/deploy/prism/seccomp_profile.json > "$backup/browser-seccomp.json"
echo 'Checking the bundled browser before replacing the application ...'
docker run --rm --network none --init --user pwuser --shm-size=256m \
  --security-opt "seccomp=$directory/$backup/browser-seccomp.json" \
  --entrypoint python3 "$image_id" /opt/sub2api/deploy/prism/smoke_browser.py
[[ $(docker image inspect --format '{{.Id}}' "$image") == "$image_id" ]] || fail 'Image tag changed during preparation; retry'
"${helper[@]}" install --file "$file" --backup "$backup"
echo "Original configuration saved in $directory/$backup/original.yml"
# Never run down, remove volumes, or rebuild/restart dependent services.
if ! "${base[@]}" up -d --no-deps --no-build --pull never --wait --wait-timeout 120 sub2api; then
  echo "Application did not become healthy. Configuration backup: $directory/$backup/original.yml" >&2
  echo 'Inspect the application logs before retrying. Database and data volumes were retained.' >&2
  exit 1
fi
echo 'Upgrade complete. Open the account Prism panel and click Start, then enable and save the account.'
echo 'Future updates use your existing image pull and Compose up commands.'
