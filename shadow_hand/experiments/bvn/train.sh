envs=('HandReach-v0' \
      'HandManipulateBlockRotateZ-v0' \
      'HandManipulateBlockRotateParallel-v0' \
      'HandManipulateBlockRotateXYZ-v0' \
      'HandManipulateBlock-v0' \
      'HandManipulateEggRotate-v0' \
      'HandManipulateEgg-v0' \
      'HandManipulatePenRotate-v0'\
      'HandManipulatePen-v0')
seeds=(100 200 300 400 500)


for env in "${envs[@]}"
do
  for seed in "${seeds[@]}"
  do
    mpirun -np 8 python3 ddpg_her/main.py \
      --Args.gamma=0.99 \
      --Args.agent_type='ddpg' \
      --Args.n_workers=20 \
      --Args.n_epochs=200 \
      --Args.n_cycles=50 \
      --Args.critic_type='fullrank-dot' \
      --MetricArgs.metric_embed_dim=16 \
      --Args.env_name=$env \      
      --Args.seed=$seed
  done
done