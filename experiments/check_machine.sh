#!/usr/bin/env bash
#
# Run before each session on worker1:
#
#     ./experiments/check_machine.sh
#     ./experiments/check_machine.sh /mnt/prof-XXX/garvit/rpucb   # also test a checkpoint root
#
# Shared machine, so the question is not "is a GPU free right now" but
# "has it been free for a while, and is anyone about to want it back".
# A single nvidia-smi snapshot can catch a busy job between batches and
# call its GPU idle.

CKPT_ROOT="${1:-}"

echo "=== GPUs (index matches CUDA_DEVICE_ORDER=PCI_BUS_ID) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu \
           --format=csv,noheader

echo
echo "=== utilisation over 10 s (a busy GPU between batches can read 0% once) ==="
timeout 11 nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
                      --format=csv,noheader -l 2 2>/dev/null \
  | awk -F', ' '{u[$1]+=$2; n[$1]++; m[$1]=$3}
                END {for (g in u) printf "  GPU %s  mean util %3.0f%%  mem %s\n", g, u[g]/n[g], m[g]}' \
  | sort

echo
echo "=== who is using them ==="
procs=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader)
if [[ -z "$procs" ]]; then
  echo "  no compute processes"
else
  while IFS=', ' read -r bus pid mem unit; do
    idx=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader \
          | awk -F', ' -v b="$bus" '$2==b {print $1}')
    info=$(ps -o user=,etime= -p "$pid" 2>/dev/null | awk '{print "user=" $1 "  running=" $2}')
    echo "  GPU ${idx:-?}  pid=$pid  ${mem} ${unit}  ${info:-'(process not visible)'}"
  done <<< "$procs"
fi

echo
echo "=== disk ==="
df -h ~ /mnt/prof-* 2>/dev/null
for d in checkpoints checkpoints_dryrun checkpoints_tuning checkpoints_calib \
         results results_dryrun results_tuning; do
  [[ -d "$d" ]] && du -sh "$d" 2>/dev/null
done

if [[ -n "$CKPT_ROOT" ]]; then
  echo
  echo "=== write test: $CKPT_ROOT ==="
  if mkdir -p "$CKPT_ROOT" 2>/dev/null && touch "$CKPT_ROOT/.w" 2>/dev/null; then
    rm -f "$CKPT_ROOT/.w"
    echo "  writable"
    df -h "$CKPT_ROOT" | tail -1
  else
    echo "  NOT writable"
  fi
fi

echo
echo "=== CPU and RAM ==="
echo "  cores: $(nproc)   load (1/5/15 min):$(uptime | awk -F'load average:' '{print $2}')"
free -g | awk 'NR==1 || /Mem/'
echo
echo "  Load near the core count means the CPUs are busy: num_workers=8 will"
echo "  slow down there, and so will the timing you are trying to measure."