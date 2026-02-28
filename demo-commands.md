# Prerequisites
conda activate gmr
cd TWIST2

# 1. Setup cameras (detect + assign left/center/right)
python deploy_real/utils/setup_cameras.py

# 2. Capture test views to verify
python deploy_real/utils/capture_camera_views.py

# 3. Run calibration (recommended: sync mode with 70 frames)
python deploy_real/calibrate_cameras.py --recapture --target-frames 70

# 4. Test triangulation quality
python deploy_real/test_triangulation.py

### teleop publisher
cd /home/stickbot/projects/g1-pick-n-place/TWIST2
conda activate gmr
python deploy_real/isaac_lab_teleop_publisher.py --display --track upper_body

### g1 teleop
cd /home/stickbot/projects/g1-pick-n-place/TWIST2
conda activate env_isaaclab
python deploy_sim2real/play_kitchen_teleop.py \
    --teleop redis --auto_loop \
    --loop_interval 30


### auto loop every 10 seconds
cd ~/projects/g1-pick-n-place/TWIST2
python deploy_sim2real/play_kitchen_teleop.py \
    --teleop pkl \
    --motion_file deploy_sim2real/sample_motions/open-doors.pkl \
    --motion_loop \
    --auto_loop \
    --loop_interval 10

