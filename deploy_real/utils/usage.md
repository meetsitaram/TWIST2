conda activate gmr
cd /home/stickbot/projects/g1-pick-n-place/TWIST2

# Capture camera views
python deploy_real/utils/capture_camera_views.py

# Setup cameras interactively
python deploy_real/utils/setup_cameras.py

# Generate Charuco board
python deploy_real/utils/generate_charuco_board.py

# Annotate board with measurement guides
python deploy_real/utils/annotate_charuco_board.py