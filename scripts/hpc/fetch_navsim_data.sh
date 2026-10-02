#!/bin/bash
#
# Fetch the NavSim / OpenScene data needed to run the DrivoR pipeline, reproducibly.
# -----------------------------------------------------------------------------------
# WHY: setup_navsim_env.sh builds the env but ASSUMES the data already exists (on ubix
# it was pre-provided). To reproduce on a fresh machine (e.g. the GCP VM) we must
# actually download it. This wraps the NavSim devkit's OWN official download scripts
# (download/download_*.sh, public HuggingFace tarballs) so there are no hand-copied
# URLs to rot. Idempotent: a split whose dir already exists + is non-empty is SKIPPED
# (no re-downloading 33 GB, no wasted egress).
#
# WHAT it fetches by default (a COMPLETE navhard setup for the oracle + scorer refresh):
#   maps (~2 GB) + warmup_two_stage (~5 GB, dev/CV-floor) + navhard_two_stage (~33 GB)
#   + test-split LOGS (metadata only, ~1 GB) + test-split CAMERA sensors for the navhard
#     logs (32 tarballs ~128 GB downloaded, only the navhard subset ~100 GB kept).
# The last item is REQUIRED: navhard's stage-1 scenes are the standard test split, so their
# original cameras live in sensor_blobs/test, not in navhard's self-contained bundle.
# The 445 GB navtrain split is NOT fetched (the scorer uses 5-fold CV on navhard). Set
# WITH_NAVTRAIN=1 only if you truly need it.
#
# Usage:  bash scripts/hpc/fetch_navsim_data.sh          (no exports needed)
#   Paths come from scripts/navsim/navsim_paths.sh (the ONE place to edit to relocate):
#     OPENSCENE_DATA_ROOT  where the data lands   (default $HOME/navsim_dataset)
#     NAVSIM_WS            devkit-clone workspace  (default $HOME/navsim_ws)
#   Other knobs (edit inline or set once): WITH_NAVTRAIN=1 also fetch navtrain (+445 GB,
#   off by default); REQUIRED_GB free-space floor before downloading (default 160).
# After it finishes, run setup_navsim_env.sh (env) then the metric-cache / Stage-A steps.
set -eo pipefail

# central path config (edit scripts/navsim/navsim_paths.sh to relocate) + fallback
source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$HOME/navsim_dataset}"
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
NAVSIM_REPO="${NAVSIM_REPO:-https://github.com/autonomousvision/navsim.git}"
DEVKIT="$NAVSIM_WS/navsim"
REQUIRED_GB="${REQUIRED_GB:-160}"   # maps+warmup+navhard (~40) + navhard test-sensor subset (~100) + headroom

LOG="./fetch_navsim_data_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== NavSim data fetch — $(date -Is) on $(hostname -s) ==="
echo "OPENSCENE_DATA_ROOT=$OPENSCENE_DATA_ROOT"
echo "devkit=$DEVKIT   log=$LOG"

mkdir -p "$OPENSCENE_DATA_ROOT"
[ -w "$OPENSCENE_DATA_ROOT" ] || { echo "!! not writable: $OPENSCENE_DATA_ROOT"; exit 1; }

# --- required tools (the devkit scripts wget + unzip + tar; fail fast, not mid-extract) ---
MISSING=""
for t in wget unzip tar; do command -v "$t" >/dev/null 2>&1 || MISSING="$MISSING $t"; done
if [ -n "$MISSING" ]; then
    echo "!! missing required tool(s):$MISSING"
    echo "   install them first, e.g.:  sudo apt-get update && sudo apt-get install -y$MISSING"
    exit 1
fi

# --- space preflight (the 'check we have enough space' guard) ----------------
AVAIL_GB="$(df -PBG "$OPENSCENE_DATA_ROOT" | awk 'NR==2{gsub("G","",$4); print $4}')"
echo "=== free space on target fs: ${AVAIL_GB} GB  (need >= ${REQUIRED_GB} GB) ==="
if [ "${AVAIL_GB:-0}" -lt "$REQUIRED_GB" ]; then
    echo "!! only ${AVAIL_GB} GB free, need ${REQUIRED_GB} GB. Free space or set OPENSCENE_DATA_ROOT"
    echo "   to a bigger disk, then re-run. (navtrain adds ~445 GB — leave WITH_NAVTRAIN off.)"
    exit 1
fi

# --- devkit clone (holds the official download scripts) ----------------------
if [ -d "$DEVKIT/download" ]; then
    echo "=== devkit present ($DEVKIT) ==="
else
    echo "=== cloning devkit -> $DEVKIT ==="
    mkdir -p "$NAVSIM_WS"
    git clone "$NAVSIM_REPO" "$DEVKIT"
fi
[ -d "$DEVKIT/download" ] || { echo "!! $DEVKIT/download not found after clone"; exit 1; }

# --- fetch one split via the devkit's own script, idempotently ---------------
# The devkit scripts download+extract into the CURRENT dir, so run them from the data
# root. $1 = expected result dir (skip if it already exists non-empty); $2 = script name.
fetch_split() {
    local result_dir="$1" script="$2"
    local target="$OPENSCENE_DATA_ROOT/$result_dir"
    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "--- $result_dir already present ($(du -sh "$target" 2>/dev/null | cut -f1)) — skipping"
        return 0
    fi
    [ -f "$DEVKIT/download/$script" ] || { echo "!! devkit script missing: download/$script (devkit version drift?)"; exit 1; }
    echo "--- fetching $result_dir via download/$script ..."
    ( cd "$OPENSCENE_DATA_ROOT" && bash "$DEVKIT/download/$script" )
    echo "--- done $result_dir  ($(du -sh "$target" 2>/dev/null | cut -f1 || echo '?'))"
}

# Fetch a base split's LOGS ONLY (metadata tarball; sensors skipped -- caching/scoring
# use no sensors, and Stage A/B use the two-stage split's own bundled sensor_blobs). The
# two-stage splits filter a standard base split for their stage-1 scenes: navhard uses
# data_split=test, so it needs $DATA/navsim_logs/test. $1=base name, $2=metadata tarball.
fetch_base_logs() {
    local base="$1" tarball="$2"
    local target="$OPENSCENE_DATA_ROOT/navsim_logs/$base"
    # already correct? log-named .pkl files sit DIRECTLY in navsim_logs/<base>/.
    if ls "$target"/*.pkl >/dev/null 2>&1; then
        echo "--- navsim_logs/$base already present ($(du -sh "$target" 2>/dev/null | cut -f1)) — skipping"
        return 0
    fi
    # present but double-nested (navsim_logs/<base>/<base>/*.pkl from an older fetch)? self-heal.
    if ls "$target/$base"/*.pkl >/dev/null 2>&1; then
        echo "--- navsim_logs/$base is double-nested — flattening"
        mv "$target/$base" "$target.__flat" && rm -rf "$target" && mv "$target.__flat" "$target"
        echo "--- flattened ($(du -sh "$target" 2>/dev/null | cut -f1))"
        return 0
    fi
    echo "--- fetching navsim_logs/$base (metadata only; sensors skipped) ..."
    ( cd "$OPENSCENE_DATA_ROOT"
      wget -q --show-progress "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/$tarball"
      tar -xzf "$tarball"
      rm -f "$tarball"
      mkdir -p navsim_logs
      # the tarball unpacks to openscene-v1.1/meta_datas/<base>/*.pkl (split dir nested
      # INSIDE meta_datas); we want navsim_logs/<base>/*.pkl, so move the inner split dir,
      # not meta_datas itself (which would double-nest to navsim_logs/<base>/<base>/).
      src="openscene-v1.1/meta_datas/$base"; [ -d "$src" ] || src="openscene-v1.1/meta_datas"
      mv "$src" "navsim_logs/$base"
      rm -rf openscene-v1.1 )
    echo "--- done navsim_logs/$base ($(du -sh "$target" 2>/dev/null | cut -f1 || echo '?'))"
}

# navhard's STAGE-1 scenes are the standard test split, so their ORIGINAL cameras live in
# sensor_blobs/test — NOT in navhard's self-contained bundle (which re-encodes frames under
# its own filenames the main devkit can't address). Fetch the test CAMERA sensors (32
# tarballs, ~128 GB download; lidar skipped) but KEEP ONLY the navhard logs' dirs, so the
# on-disk footprint stays ~navhard-subset instead of the full ~180 GB test split.
fetch_test_sensors() {
    target="$OPENSCENE_DATA_ROOT/sensor_blobs/test"
    navhard_sb="$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs"
    [ -d "$navhard_sb" ] || { echo "!! $navhard_sb missing — fetch navhard first"; return 1; }
    mapfile -t LOGS < <(ls "$navhard_sb")
    [ "${#LOGS[@]}" -gt 0 ] || { echo "!! no navhard logs found under $navhard_sb"; return 1; }
    missing=0; for log in "${LOGS[@]}"; do [ -d "$target/$log" ] || missing=1; done
    if [ "$missing" = 0 ]; then
        echo "--- sensor_blobs/test already has all ${#LOGS[@]} navhard logs — skipping"
        return 0
    fi
    mkdir -p "$target"
    echo "--- fetching test camera sensors (32 tarballs ~128 GB; keeping only ${#LOGS[@]} navhard logs) ..."
    base="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera"
    # tar include-patterns: extract ONLY the navhard logs directly (fast — writes just those
    # files instead of the whole tarball). Layout is openscene-v1.1/sensor_blobs/test/<log>/...
    # so --strip-components=3 drops that prefix, landing files at sensor_blobs/test/<log>/.
    patterns=(); for log in "${LOGS[@]}"; do patterns+=("openscene-v1.1/sensor_blobs/test/$log/*"); done
    for i in $(seq 0 31); do
        marker="$OPENSCENE_DATA_ROOT/sensor_blobs/.test_camera_${i}.done"
        [ -f "$marker" ] && { echo "  [$((i+1))/32] tarball $i already processed — skipping"; continue; }
        tb="$OPENSCENE_DATA_ROOT/openscene_sensor_test_camera_${i}.tgz"
        # reuse a probe tarball for idx 0 if present (avoids re-downloading it)
        if [ "$i" = 0 ] && [ -f "$OPENSCENE_DATA_ROOT/_probe.tgz" ]; then
            mv "$OPENSCENE_DATA_ROOT/_probe.tgz" "$tb"
            echo "  [1/32] using existing _probe.tgz"
        elif [ ! -s "$tb" ]; then
            echo "  [$((i+1))/32] downloading openscene_sensor_test_camera_${i}.tgz"
            wget -q --show-progress --timeout=120 --tries=3 -O "$tb" \
                "$base/openscene_sensor_test_camera_${i}.tgz" \
                || { echo "  !! download failed idx $i — skipping (re-run to retry)"; rm -f "$tb"; continue; }
        fi
        gzip -t "$tb" 2>/dev/null || { echo "  !! tarball $i corrupt — deleting, re-run to retry"; rm -f "$tb"; continue; }
        before=$(ls "$target" 2>/dev/null | wc -l)
        # extract only the navhard logs; unmatched patterns make GNU tar exit non-zero -> ignore.
        tar -xzf "$tb" -C "$target" --strip-components=3 --wildcards "${patterns[@]}" 2>/dev/null || true
        rm -f "$tb"
        after=$(ls "$target" 2>/dev/null | wc -l)
        echo "  [$((i+1))/32] +$((after - before)) navhard logs (total now: $after)"
        touch "$marker"   # resume marker: only set after a real (integrity-checked) extraction
    done
    echo "--- done: $(ls "$target" 2>/dev/null | wc -l) logs in sensor_blobs/test ($(du -sh "$target" 2>/dev/null | cut -f1))"
}

# navtrain (the training split, ~445 GB). The devkit's download_navtrain_hf.sh extracts to
# PREFIXED dirs -- trainval_navsim_logs/ and trainval_sensor_blobs/trainval/ -- NOT the
# standard navsim_logs/trainval + sensor_blobs/trainval our scripts read. So run the official
# downloader (it fetches 64 sensor tarballs + metadata over HF, no AWS creds) then reorganize.
# NOTE: the devkit downloader is not resumable -- run under tmux; if it dies mid-way, re-run
# (already-reorganized data is skipped, but a partial devkit run re-downloads from scratch).
fetch_navtrain() {
    local logs="$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
    local sens="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
    if ls "$logs"/*.pkl >/dev/null 2>&1 && [ -d "$sens" ] && [ -n "$(ls -A "$sens" 2>/dev/null)" ]; then
        echo "--- navtrain already present (navsim_logs/trainval + sensor_blobs/trainval) — skipping"
        return 0
    fi
    [ -f "$DEVKIT/download/download_navtrain_hf.sh" ] || { echo "!! devkit download_navtrain_hf.sh missing"; exit 1; }
    echo "--- fetching navtrain via download/download_navtrain_hf.sh (~445 GB, run under tmux) ..."
    ( cd "$OPENSCENE_DATA_ROOT" && bash "$DEVKIT/download/download_navtrain_hf.sh" )
    echo "--- reorganizing devkit prefixed dirs -> standard layout ..."
    mkdir -p "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs"
    [ -d "$OPENSCENE_DATA_ROOT/trainval_navsim_logs" ]        && mv "$OPENSCENE_DATA_ROOT/trainval_navsim_logs"        "$logs"
    [ -d "$OPENSCENE_DATA_ROOT/trainval_sensor_blobs/trainval" ] && mv "$OPENSCENE_DATA_ROOT/trainval_sensor_blobs/trainval" "$sens"
    rmdir "$OPENSCENE_DATA_ROOT/trainval_sensor_blobs" 2>/dev/null || true
    echo "--- navtrain done: $(ls "$logs" 2>/dev/null | wc -l) log-pkls, sensors $(du -sh "$sens" 2>/dev/null | cut -f1)"
}

fetch_split "maps"              "download_maps.sh"
fetch_split "warmup_two_stage"  "download_warmup_two_stage.sh"
fetch_split "navhard_two_stage" "download_navhard_two_stage.sh"
fetch_base_logs "test" "openscene_metadata_test.tgz"   # navhard's stage-1 base-split logs
fetch_test_sensors                                     # navhard's stage-1 original cameras
if [ "${WITH_NAVTRAIN:-0}" = "1" ]; then
    echo "=== WITH_NAVTRAIN=1: fetching navtrain (+~445 GB) ==="
    fetch_navtrain   # official HF downloader + reorganize to navsim_logs/trainval + sensor_blobs/trainval
fi

# --- report resulting layout (verify against our script expectations) --------
echo "=== resulting layout under $OPENSCENE_DATA_ROOT ==="
du -sh "$OPENSCENE_DATA_ROOT"/* 2>/dev/null
echo
echo "=== DONE — log: $LOG ==="
echo "NOTE: our metric-cache / Stage-A scripts expect the two-stage split NESTED as"
echo "  \$OPENSCENE_DATA_ROOT/navhard_two_stage/<split>/{sensor_blobs,synthetic_scene_pickles}."
echo "  If the devkit (v2.2) extracts different subdir names (curr_sensors/scene_pickles/...),"
echo "  the split-path overrides in the run scripts may need adjusting — check the tree above."
echo "Next: bash scripts/hpc/setup_navsim_env.sh   # env (data now present)"