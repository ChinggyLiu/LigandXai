# attribution_pipeline/Snakefile

# 1) Ensure all paths are resolved from the project root
workdir: "."

# 2) Load this pipeline’s minimal config
configfile: "config.yaml"

import os

# 3) Attribution methods to run
METHODS = ["integrated", "inputxgrad", "guided", "shap"]

# 4) Derive dataset name from the dataset YAML basename
#    e.g. "config/dti/kiba.yaml" → "kiba"
dataset_name = os.path.splitext(os.path.basename(config["data_config"]))[0]

# 5) Top-level rule: expect one .pkl per method in the output_dir
rule all:
    input:
        expand(
            "{output_dir}/{method}.pkl",
            output_dir=config["output_dir"],
            method=METHODS
        )

# 6) Attribution rule: compute all methods in one shot
rule attribution:
    input:
        # dataset YAML path (relative to project root)
        data_cfg   = config["data_config"],
        # model checkpoint path
        checkpoint = config["checkpoint"]
    output:
        expand(
            "{output_dir}/{method}.pkl",
            output_dir=config["output_dir"],
            method=METHODS
        )
    params:
        methods    = " ".join(METHODS),
        output_dir = config["output_dir"],
        batch_size = config.get("batch_size", 1)
    threads: 1
    shell:
        """
        mkdir -p {params.output_dir} && \
        python run_attribution.py \
          --config     {input.data_cfg} \
          --checkpoint {input.checkpoint} \
          --methods    {params.methods} \
          --split      test \
          --batch_size {params.batch_size} \
          --device     cuda \
          --output_dir {params.output_dir}
        """
