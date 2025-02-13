# Safety-Focused Large Language Model (LLM) Fine-Tuning and Dataset Generation

## Overview
This project focuses on enhancing the safety and robustness of Large Language Models (LLMs) through supervised fine-tuning (SFT) and strategic dataset generation. The primary objectives are to reduce harmful outputs and improve adversarial resilience.

## Key Achievements
- **Supervised Fine-Tuning (SFT):** Utilized the Alignment Handbook's scripts and configurations to fine-tune Llama 3.1 8B with the HH-RLHF dataset, achieving a 35% reduction in harmful outputs as assessed by the OpenAI Moderation API.
- **Adversarial Robustness:** Generated five diverse responses per prompt to bolster the model's resilience against adversarial inputs.
- **Safety-Optimized Dataset Creation:** Ranked 10,000 responses using Llama 3.1 70B to develop a safety-focused dataset, resulting in a 60% reduction in annotation costs compared to traditional human labeling methods for downstream preference training.

## Repository Structure
- **SFT/**
  - `config_full.yaml`: Contains model parameters, dataset configurations, and training settings for SFT.
  - `deepspeed_zero3.yaml`: Configuration file for DeepSpeed Zero-3, facilitating efficient distributed training.
  - `run_sft.py`: Main script to execute the supervised fine-tuning process.

- **Notebooks/**
  - `Five_response_sft_model.ipynb`: Generates five diverse responses for 10K randomly sampled prompts from the HH-RLHF dataset.
  - `Ranker_Llama70B.ipynb`: Uses Llama 3.1 70B to rank the five responses from 1-5 based on safety and helpfulness.
  - `Preprocs.ipynb`: Extracts ranks per prompt from the model responses and appends them to the prompt, creating the final dataset.

## Supervised Fine-Tuning (SFT) Details
The fine-tuning process was conducted using the Alignment Handbook's methodologies, with the following configurations:

### Model Configuration
- **Model:** `meta-llama/Llama-3.1-8B`
- **Torch Data Type:** `bfloat16`
- **Attention Implementation:** `flash_attention_2`

### Dataset
- **Primary Dataset:** `nkulka29/HH-RLHF-Chosen` with a weighting of 1.0
- **Dataset Splits:** `train_sft` and `test_sft`

### Training Parameters
- **Batch Size:** 16 for training, 8 for evaluation
- **Learning Rate:** 2e-5
- **Scheduler:** Cosine Annealing
- **Gradient Checkpointing:** Enabled
- **Logging:** Configured for TensorBoard with logging steps set to 5
- **Evaluation Strategy:** Conducted per epoch
- **Checkpoint Saving:** Every 100 steps, with a limit of 1 checkpoint
- **Training Epochs:** 2
- **Maximum Sequence Length:** 2048 tokens
- **Output Directory:** `data/llama-3.1-8B-sft`
- **Random Seed:** 42 for reproducibility

### DeepSpeed Zero-3 Configuration
- **Zero-3 Stage:** Enabled with `zero3_init_flag` set to true
- **Mixed Precision:** `bf16`
- **Distributed Training:** Utilized DeepSpeed with 1 machine and 4 processes
- **Offloading:** Not utilized

## Running the Fine-Tuning Process
To initiate the fine-tuning, execute:
```bash
python SFT/run_sft.py --config SFT/config_full.yaml --deepspeed SFT/deepspeed_zero3.yaml
```

## Model Saving and Deployment
Post-training, the model is saved in the `data/llama-3.1-8B-sft` directory. If the `push_to_hub` parameter is enabled, the model will be automatically uploaded to the Hugging Face Hub.

## Evaluation Metrics
The model's performance was evaluated based on:
- **Perplexity:** Assessed on the test dataset
- **Safety Benchmarks:** Performance evaluated against human-aligned safety standards

## Future Directions
- **Dataset Expansion:** Incorporate a broader range of safety scenarios
- **Reinforcement Learning Integration:** Implement reinforcement learning from human feedback (RLHF) for preference optimization
- **Deployment Optimization:** Enhance inference efficiency and scalability

---
For inquiries or contributions, please submit an issue or pull request.

