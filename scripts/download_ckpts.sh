mkdir -p checkpoints

echo "Downloading 52M DiT trained with AvgFlow objective"
ngc registry resource download-version "nvidia/clara/avgflow_52m_ckpt:avgflow_52m_ckpt" \
    --dest ./checkpoints
# Simplify the checkpoint path
mv ./checkpoints/avgflow_52m_ckpt_vavgflow_52m_ckpt/avgflow_52m.pkl ./checkpoints/avgflow_52m_ckpt.pkl
rm -rf ./checkpoints/avgflow_52m_ckpt_vavgflow_52m_ckpt

echo "Downloading 64M DiT trained with AvgFlow objective"
ngc registry resource download-version "nvidia/clara/avgflow_64m_ckpt:avgflow_64m_ckpt" \
    --dest ./checkpoints
# Simplify the checkpoint path
mv ./checkpoints/avgflow_64m_ckpt_vavgflow_64m_ckpt/avgflow_64m.pkl ./checkpoints/avgflow_64m_ckpt.pkl
rm -rf ./checkpoints/avgflow_64m_ckpt_vavgflow_64m_ckpt

echo "Downloading 52M DiT finetuned with reflow for few step generation"
ngc registry resource download-version "nvidia/clara/avgflow_52m_reflow_ckpt:avgflow_52m_reflow_ckpt" \
    --dest ./checkpoints
# Simplify the checkpoint path
mv ./checkpoints/avgflow_52m_reflow_ckpt_vavgflow_52m_reflow_ckpt/avgflow_52m_reflow.pkl ./checkpoints/avgflow_52m_reflow_ckpt.pkl
rm -rf ./checkpoints/avgflow_52m_reflow_ckpt_vavgflow_52m_reflow_ckpt

echo "Downloading 52M DiT finetuned with reflow+distillation for 1-step generation"
ngc registry resource download-version "nvidia/clara/avgflow_52m_distill_ckpt:avgflow_52m_distill_ckpt" \
    --dest ./checkpoints
# Simplify the checkpoint path
mv ./checkpoints/avgflow_52m_distill_ckpt_vavgflow_52m_distill_ckpt/avgflow_52m_distill.pkl ./checkpoints/avgflow_52m_distill_ckpt.pkl
rm -rf ./checkpoints/avgflow_52m_distill_ckpt_vavgflow_52m_distill_ckpt