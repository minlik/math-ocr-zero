set -x
ENGINE=${1:-vllm}
TRAIN_PATH=${2:-"/root/data/code/math-ocr-zero/data/deepmath-ocr-10000/train.parquet"}
VALID_PATH=${3:-"/root/data/code/math-ocr-zero/data/deepmath-ocr-10000/test.parquet"}
N_GPUS=${4:-4}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
TEMPLATE="{% for message in messages %}{% for content in message['content'] %}{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endfor %}"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VALID_PATH \
    data.train_batch_size=128 \
    data.max_prompt_length=256 \
    data.max_response_length=840 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.image_min_pixels=784 \
    data.image_max_pixels=100352 \
    'data.custom_chat_template="'"$TEMPLATE"'"' \
    actor_rollout_ref.model.path=/root/data/models/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_deepmath' \
    trainer.experiment_name='qwen2_5_vl_3b_deepmath-10000' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=210 \
    trainer.test_freq=210 \
    trainer.total_epochs=10 $@