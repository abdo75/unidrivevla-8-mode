#!/bin/bash
#
# Scorer ablations for the thesis (sec:ablation) + EPDMS sub-score breakdown.
# All reuse the navhard scorer dataset already on disk -- NO model retraining, no GPU
# fine-tune. Each cv_select run is a fresh 5-fold cross-validation (minutes). One batch,
# tmux-friendly; paste the log back.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
DS="${DS:-$NAVSIM_WS/exp/uni_scorer/dataset_navhard_two_stage.pt}"
EPOCHS="${EPOCHS:-200}"
LOG="./ablations_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== scorer ablations on $DS  (epochs=$EPOCHS) — $(date -Is) ==="

run () {  # label  env-overrides...
    local label="$1"; shift
    echo; echo "##### $label #####"
    ( cd "$HERE" && env "$@" "$MM" run -n "${UNI_ENV:-unidrivevla}" \
        python cv_select.py "$DS" "/tmp/sel_abl.json" 5 "$EPOCHS" 0.0 ) \
      | grep -E "baselines:|pure-argmax|tau=0.000|tau sweep" || true
}

echo; echo "===== EPDMS sub-score breakdown (single / best-fixed / oracle) ====="
( cd "$HERE" && "$MM" run -n "${UNI_ENV:-unidrivevla}" python aggregate_subscores.py "$DS" ) || true

echo; echo "===== Ablation 1: training objective (ranking weight lambda) ====="
run "lambda=0  (BCE only, no ranking)"      LAM=0
run "lambda=1  (BCE + ranking, default)"    LAM=1

echo; echo "===== Ablation 2: number of candidates N ====="
for n in 2 3 4 6; do run "N=$n candidates" N_CAND=$n; done

echo; echo "===== Ablation 3: how the candidate reads the scene ====="
run "attn  (cross-attention over scene tokens, default)" POOL=attn
run "mean  (mean-pooled scene tokens)"                   POOL=mean
run "trajonly  (candidate geometry only, no scene)"      POOL=trajonly

echo; echo "===== Ablation 4: scorer width d ====="
for dd in 64 128 256; do run "d=$dd" SCORER_D=$dd; done

echo; echo "=== DONE — log: $LOG ==="
