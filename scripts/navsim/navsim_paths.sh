#!/bin/bash
# ============================================================================
#  NavSim paths — THE SINGLE PLACE TO EDIT to relocate data / workspace.
# ----------------------------------------------------------------------------
#  Sourced by every navsim script (fetch, setup, stage_a/b/c, oracle, caches).
#  The data needs ~160 GB (maps + warmup + navhard + navhard test sensors) and
#  many small files: put it on a disk with enough space and inodes. To move the
#  data or the workspace, change the two lines below only.
#
#  `:=` sets the var only if unset, so an explicit env override still wins if
#  you ever want one, but you never NEED one.
# ============================================================================

: "${OPENSCENE_DATA_ROOT:=$HOME/navsim_dataset}"  # maps + warmup + navhard land / are read here
: "${NAVSIM_WS:=$HOME/navsim_ws}"                 # devkit clone + exp/ (Stage A dumps) + navsim_env.sh

export OPENSCENE_DATA_ROOT NAVSIM_WS
