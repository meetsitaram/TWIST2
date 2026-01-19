SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# Set safe standing pose in Redis to prevent robot from falling
echo "🤖 Setting safe standing pose in Redis..."
SAFE_BODY='[0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
SAFE_HANDS='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
SAFE_NECK='[0.0, 0.0]'
redis-cli SET action_body_unitree_g1_with_hands "$SAFE_BODY" > /dev/null
redis-cli SET action_hand_left_unitree_g1_with_hands "$SAFE_HANDS" > /dev/null
redis-cli SET action_hand_right_unitree_g1_with_hands "$SAFE_HANDS" > /dev/null
redis-cli SET action_neck_unitree_g1_with_hands "$SAFE_NECK" > /dev/null
echo "✅ Safe pose set! Robot will start in stable standing position."
echo ""

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
