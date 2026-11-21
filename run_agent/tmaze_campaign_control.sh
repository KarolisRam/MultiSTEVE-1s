#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CAMPAIGN_ROOT status|stop|tail" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CAMPAIGN_ROOT="$1"
COMMAND="$2"
shift 2

if [[ "${CAMPAIGN_ROOT}" != /* ]]; then
  ROOT_ABS="${REPO_ROOT}/${CAMPAIGN_ROOT}"
else
  ROOT_ABS="${CAMPAIGN_ROOT}"
fi

find_queue_pids() {
  local pid_file="${ROOT_ABS}/queue_controller.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}" || true)"
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      echo "${pid}"
    fi
  fi
  ps -eo pid,args | awk -v root_abs="${ROOT_ABS}" -v root_arg="${CAMPAIGN_ROOT}" '
    /python -m steve1\.run_agent\.teleport_campaign_queue/ && $0 !~ /conda run/ {
      if (index($0, root_abs) || index($0, root_arg)) print $1
    }
  '
}

case "${COMMAND}" in
  status)
    echo "campaign_root=${ROOT_ABS}"
    echo "queue_pids=$(find_queue_pids | sort -u | xargs echo)"
    echo
    echo "status counts:"
    awk -F, 'NR>1 {c[$4]++} END {for (k in c) print "  " k, c[k]}' "${ROOT_ABS}/rollout_status.csv" | sort
    echo
    echo "running rows:"
    awk -F, 'NR==1 || $4=="running" {print}' "${ROOT_ABS}/rollout_status.csv" | column -s, -t | head -10
    echo
    echo "active processes:"
    ps -eo pid,ppid,stat,etime,cmd | rg -i 'teleport_campaign_queue|run_agent_with_logging_teleport|mcprec|minecraft|xvfb' | rg -v 'rg -i|bash -lc' || true
    ;;
  stop)
    pids="$(find_queue_pids | sort -u | xargs echo)"
    if [[ -z "${pids}" ]]; then
      echo "no queue controller found for ${ROOT_ABS}"
      exit 0
    fi
    echo "sending SIGTERM to queue controller pid(s): ${pids}"
    # This asks the queue to stop launching new rollouts and wait for active rollouts to finish.
    kill -TERM ${pids}
    ;;
  tail)
    tail -n "${1:-120}" -f "${ROOT_ABS}"/logs/queue*.log "${ROOT_ABS}"/logs/worker_*.log
    ;;
  *)
    echo "unknown command: ${COMMAND}" >&2
    exit 2
    ;;
esac
