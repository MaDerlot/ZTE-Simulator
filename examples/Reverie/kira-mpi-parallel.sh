#!/bin/bash
# kira-mpi-parallel.sh
#
# MPI-based spatial-partition parallel simulation launcher.
#
# Strategy:
#   - Each MPI rank owns one Pod (a group of ToRs + their servers).
#   - Spine switches are ghost nodes replicated on all ranks.
#   - Cross-pod (DP) flows cross the MPI rank boundary via QbbRemoteChannel.
#   - NullMessageSimulatorImpl provides conservative PDES with lookahead=2us
#     (Spine link delay), guaranteeing causal correctness with no rollback.
#
# Usage:
#   bash kira-mpi-parallel.sh [NUM_RANKS] [TASKINDEX]
#
#   NUM_RANKS   number of MPI processes (default: number of ToR pods in topo)
#   TASKINDEX   task index to simulate  (default: 1)

set -euo pipefail

NUM_RANKS=${1:-4}
TASKINDEX=${2:-1}

# ---- navigate to repo root ------------------------------------------------
cd "$(dirname "$0")/../.."
NS3_ROOT=$(pwd)

# ---- build with MPI enabled -----------------------------------------------
echo "[kira-mpi] Building reverie-evaluation-sigcomm2023 with MPI support..."
./ns3 build reverie-evaluation-sigcomm2023

# ---- locate simulation binary ---------------------------------------------
SIM_BIN=$(find build/ -name "reverie-evaluation-sigcomm2023*" -type f \
          ! -name "*.cc" ! -name "CMake*" 2>/dev/null | head -1)
if [ -z "$SIM_BIN" ]; then
    echo "[kira-mpi] ERROR: simulation binary not found after build"
    exit 1
fi
echo "[kira-mpi] Using binary: $SIM_BIN"

# ---- paths ----------------------------------------------------------------
REVERIE_DIR="$NS3_ROOT/examples/Reverie"
TASK_DIR="$REVERIE_DIR/task${TASKINDEX}"
DUMP_DIR="$REVERIE_DIR/dump"
RESULTS_DIR="$REVERIE_DIR/results"
PARTITION_FILE="$TASK_DIR/node_partition.txt"
TOPO_FILE="$REVERIE_DIR/leaf-spine.txt"
ALPHA_FILE="$REVERIE_DIR/alphas"

mkdir -p "$DUMP_DIR" "$RESULTS_DIR"

# ---- source config (N_CORES, etc.) ----------------------------------------
source "$REVERIE_DIR/config.sh"

# ---- generate spatial partition -------------------------------------------
echo "[kira-mpi] Generating spatial partition (${NUM_RANKS} ranks)..."
python3 "$REVERIE_DIR/partition.py" \
    --topo  "$TOPO_FILE" \
    --ranks "$NUM_RANKS" \
    --output "$PARTITION_FILE"

# ---- simulation parameters (mirrors kira-parallel.sh) --------------------
REVERIE=111
INTCC=3

START_TIME=0
END_TIME=10000
FLOW_LAUNCH_END_TIME=3
BUFFER_PER_PORT_PER_GBPS=5.12   # KB per port per Gbps
BUFFERSIZE=$(python3 -c "print(20*25*1000*${BUFFER_PER_PORT_PER_GBPS})")  # Bytes

alg=$REVERIE
egresslossyFrac=0.8
gamma=0.999
RDMACC=$INTCC
BUFFERMODEL="reverie"

FCTFILE="$DUMP_DIR/evaluation_mpi.fct"
TORFILE="$DUMP_DIR/evaluation_mpi.tor"
DUMPFILE="$DUMP_DIR/evaluation_mpi.out"
PFCFILE="$DUMP_DIR/evaluation_mpi.pfc"

echo "[kira-mpi] Starting MPI simulation with ${NUM_RANKS} ranks..."
echo "[kira-mpi] Output -> $DUMPFILE"

# ---- launch mpirun --------------------------------------------------------
time mpirun -np "$NUM_RANKS" \
    "$SIM_BIN" \
        --TASKINDEX="$TASKINDEX" \
        --partitionFile="$PARTITION_FILE" \
        --START_TIME="$START_TIME" \
        --END_TIME="$END_TIME" \
        --FLOW_LAUNCH_END_TIME="$FLOW_LAUNCH_END_TIME" \
        --buffersize="$BUFFERSIZE" \
        --bufferalgIngress="$alg" \
        --bufferalgEgress="$alg" \
        --egressLossyShare="$egresslossyFrac" \
        --bufferModel="$BUFFERMODEL" \
        --gamma="$gamma" \
        --rdmacc="$RDMACC" \
        --alphasFile="$ALPHA_FILE" \
        --fctOutFile="$FCTFILE" \
        --torOutFile="$TORFILE" \
        --pfcOutFile="$PFCFILE" \
    > "$DUMPFILE" 2>&1

echo "[kira-mpi] Simulation complete. Results in $DUMP_DIR"
