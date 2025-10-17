conda init && source activate
conda activate simpler_env

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (two levels up from scripts/env_sh/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Set default configuration file path
config_path="$PROJECT_ROOT/evaluation/configs/simpler/example_simpler.yaml"

# Check if configuration file parameter is passed
if [[ $# -gt 0 ]]; then
    config_path="$1"
fi

# Verify if configuration file exists
if [[ ! -f "$config_path" ]]; then
    echo "[ERROR] Configuration file does not exist: $config_path"
    exit 1
fi

echo "[INFO] Using configuration file: $config_path"

# Extract output_dir from config file
output_dir=$(python -c "
import yaml
import sys
try:
    with open('$config_path', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('output_dir', './results'))
except Exception as e:
    print('./results')
")

echo "[INFO] Output directory from config: $output_dir"

# Execute evaluation
for init_rng in 0 2 4; do
    python $PROJECT_ROOT/evaluation/run_simpler_evaluation.py --config ${config_path} \
    --set octo-init-rng ${init_rng} --set additional-env-save-tags octo_init_rng_${init_rng} \
    --set env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --set scene-name bridge_table_1_v1 \
    --set rgb-overlay-path simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
    --set robot widowx --set robot-init-x-range "0.147,0.147,1" --set robot-init-y-range "0.028,0.028,1" \
    --set output-dir ${output_dir}/env_StackGreenCubeOnYellowCubeBakedTexInScene_octo_init_rng_${init_rng};

    python $PROJECT_ROOT/evaluation/run_simpler_evaluation.py --config ${config_path} \
    --set octo-init-rng ${init_rng} --set additional-env-save-tags octo_init_rng_${init_rng} \
    --set env-name PutCarrotOnPlateInScene-v0 --set scene-name bridge_table_1_v1 \
    --set rgb-overlay-path simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
    --set robot widowx --set robot-init-x-range "0.147,0.147,1" --set robot-init-y-range "0.028,0.028,1" \
    --set output-dir ${output_dir}/env_PutCarrotOnPlateInScene_octo_init_rng_${init_rng};

    python $PROJECT_ROOT/evaluation/run_simpler_evaluation.py --config ${config_path} \
    --set octo-init-rng ${init_rng} --set additional-env-save-tags octo_init_rng_${init_rng} \
    --set env-name PutSpoonOnTableClothInScene-v0 --set scene-name bridge_table_1_v1 \
    --set rgb-overlay-path simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
    --set robot widowx --set robot-init-x-range "0.147,0.147,1" --set robot-init-y-range "0.028,0.028,1" \
    --set output-dir ${output_dir}/env_PutSpoonOnTableClothInScene_octo_init_rng_${init_rng};

    python $PROJECT_ROOT/evaluation/run_simpler_evaluation.py --config ${config_path} \
    --set octo-init-rng ${init_rng} --set additional-env-save-tags octo_init_rng_${init_rng} \
    --set env-name PutEggplantInBasketScene-v0 --set scene-name bridge_table_1_v2 \
    --set rgb-overlay-path simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
    --set robot widowx_sink_camera_setup --set robot-init-x-range "0.127,0.127,1" --set robot-init-y-range "0.06,0.06,1" \
    --set output-dir ${output_dir}/env_PutEggplantInBasketScene_octo_init_rng_${init_rng};
done

python $PROJECT_ROOT/evaluation/utils/aggregate_simpler_results.py --base-dir ${output_dir} --output-file ${output_dir}/aggregated_simpler_results.json
