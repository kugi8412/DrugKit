#!/bin/bash -l
#SBATCH --job-name=drugkit
#SBATCH --qos=jgiezgala_common
#SBATCH --partition=common
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --output="logs/drugkit_%j.out"
#SBATCH --error="logs/drugkit_%j.err"

mkdir -p logs

CONFIG_FILE=$1
STAGE=${2:-full}

echo "============================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURM_NODELIST"
echo "Config file:      $CONFIG_FILE"
echo "Stage:            $STAGE"
echo "Start time:       $(date)"
echo "============================================"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found!"
    exit 1
fi

time srun python3 drugkit_pipeline.py --config "$CONFIG_FILE" --stage "$STAGE" --n-cpu "$SLURM_CPUS_PER_TASK"

echo "============================================"
echo "End time: $(date)"
echo "============================================"
