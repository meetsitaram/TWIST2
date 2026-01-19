#!/usr/bin/env python3
"""
Generate a LARGE Charuco board split into 4 A4 pages with COMPLETE squares only.

Each page contains only whole squares - no overlapping or broken squares.
Board dimensions are chosen to divide evenly into 2x2 pages.

Board options (all divide evenly into 4 pages):
  - 6x8 at 55mm = 330x440mm (3x4 squares per page)
  - 4x6 at 70mm = 280x420mm (2x3 squares per page) - BIGGEST squares
  - 6x8 at 60mm = 360x480mm (3x4 squares per page)

Layout:
    +-------+-------+
    | Page1 | Page2 |
    | (TL)  | (TR)  |
    +-------+-------+
    | Page3 | Page4 |
    | (BL)  | (BR)  |
    +-------+-------+

Usage:
    conda activate gmr
    python deploy_real/utils/generate_4page_large_charuco.py
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os
import yaml


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    deploy_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(deploy_dir)
    calibration_dir = os.path.join(project_dir, "calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    
    print("=" * 60)
    print("  4-Page LARGE Charuco Board Generator")
    print("  (Complete squares only - no broken edges)")
    print("=" * 60)
    
    # Board options that divide evenly into 2x2 pages
    # Format: (total_x, total_y, per_page_x, per_page_y, square_size_mm)
    print("\nBoard size options (all have complete squares per page):")
    print("  1. 6x8 at 55mm = 330x440mm (3x4 per page, 35 corners)")
    print("  2. 4x6 at 70mm = 280x420mm (2x3 per page, 15 corners) - BIGGEST")
    print("  3. 6x8 at 60mm = 360x480mm (3x4 per page, 35 corners)")
    print("  4. 4x8 at 70mm = 280x560mm (2x4 per page, 21 corners)")
    
    choice = input("Select [1/2/3/4, default=2]: ").strip() or "2"
    
    if choice == "1":
        squares_per_page_x, squares_per_page_y = 3, 4
        square_size_mm = 55
    elif choice == "2":
        squares_per_page_x, squares_per_page_y = 2, 3
        square_size_mm = 70
    elif choice == "3":
        squares_per_page_x, squares_per_page_y = 3, 4
        square_size_mm = 60
    else:
        squares_per_page_x, squares_per_page_y = 2, 4
        square_size_mm = 70
    
    # Total board is 2x2 pages
    total_squares_x = squares_per_page_x * 2
    total_squares_y = squares_per_page_y * 2
    
    marker_size_mm = int(square_size_mm * 0.75)
    
    board_width_mm = total_squares_x * square_size_mm
    board_height_mm = total_squares_y * square_size_mm
    internal_corners = (total_squares_x - 1) * (total_squares_y - 1)
    
    print(f"\nBoard configuration:")
    print(f"  Total: {total_squares_x}x{total_squares_y} squares at {square_size_mm}mm")
    print(f"  Per page: {squares_per_page_x}x{squares_per_page_y} complete squares")
    print(f"  Board size: {board_width_mm}mm x {board_height_mm}mm")
    print(f"  Internal corners: {internal_corners}")
    
    # Ink saving
    print("\nInk saving:")
    print("  1. Maximum (~75% less ink)")
    print("  2. None (pure black)")
    ink_choice = input("Select [1/2, default=1]: ").strip() or "1"
    ink_save = (ink_choice == "1")
    black_intensity = 60 if ink_save else 0
    
    # Create ArUco dictionary and board
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    board = aruco.CharucoBoard(
        (total_squares_x, total_squares_y),
        square_size_mm / 1000.0,
        marker_size_mm / 1000.0,
        dictionary
    )
    
    # Generate board at print resolution (300 DPI)
    dpi = 300
    mm_to_inch = 1 / 25.4
    
    # Calculate pixel sizes
    square_size_px = int(square_size_mm * mm_to_inch * dpi)
    page_squares_width_px = squares_per_page_x * square_size_px
    page_squares_height_px = squares_per_page_y * square_size_px
    board_width_px = total_squares_x * square_size_px
    board_height_px = total_squares_y * square_size_px
    
    print(f"\nGenerating {board_width_px}x{board_height_px} pixel board...")
    
    # Generate full board
    full_board = board.generateImage((board_width_px, board_height_px), marginSize=0)
    
    # Apply ink saving
    if ink_save:
        full_board = np.where(full_board < 128, black_intensity, full_board).astype(np.uint8)
    
    # Convert to color
    full_board_color = cv2.cvtColor(full_board, cv2.COLOR_GRAY2BGR)
    
    # A4 page dimensions at 300 DPI
    a4_width_mm = 210
    a4_height_mm = 297
    page_width_px = int(a4_width_mm * mm_to_inch * dpi)
    page_height_px = int(a4_height_mm * mm_to_inch * dpi)
    
    # Extract 4 tiles with complete squares only
    # Each tile is exactly squares_per_page_x * squares_per_page_y squares
    tiles = {}
    
    for row in range(2):
        for col in range(2):
            # Calculate pixel coordinates for this tile
            start_x = col * page_squares_width_px
            start_y = row * page_squares_height_px
            end_x = start_x + page_squares_width_px
            end_y = start_y + page_squares_height_px
            
            # Extract tile (complete squares only)
            tile = full_board_color[start_y:end_y, start_x:end_x].copy()
            
            page_num = row * 2 + col + 1
            tiles[page_num] = tile
    
    print("\nGenerating 4 pages with complete squares...")
    
    # Position names for labeling
    positions = {
        1: "TOP LEFT",
        2: "TOP RIGHT", 
        3: "BOTTOM LEFT",
        4: "BOTTOM RIGHT"
    }
    
    for page_num, tile in tiles.items():
        # Create white A4 page
        page = np.ones((page_height_px, page_width_px, 3), dtype=np.uint8) * 255
        
        # Center tile on page
        tile_h, tile_w = tile.shape[:2]
        start_x = (page_width_px - tile_w) // 2
        start_y = (page_height_px - tile_h) // 2
        
        # Paste tile
        page[start_y:start_y+tile_h, start_x:start_x+tile_w] = tile
        
        # Draw cut lines (thin gray lines showing exact tile boundary)
        cut_color = (180, 180, 180)
        cv2.rectangle(page, (start_x, start_y), (start_x + tile_w, start_y + tile_h), cut_color, 2)
        
        # Add corner marks for alignment (red L-shapes at corners)
        mark_len = 60
        mark_color = (0, 0, 255)  # Red
        thickness = 4
        
        # Corner marks outside the tile area
        corners = [
            (start_x, start_y),  # Top-left
            (start_x + tile_w, start_y),  # Top-right
            (start_x, start_y + tile_h),  # Bottom-left
            (start_x + tile_w, start_y + tile_h),  # Bottom-right
        ]
        
        for cx, cy in corners:
            # Draw L-shape marks
            cv2.line(page, (cx - 20, cy), (cx - 20 - mark_len, cy), mark_color, thickness)
            cv2.line(page, (cx, cy - 20), (cx, cy - 20 - mark_len), mark_color, thickness)
            cv2.line(page, (cx + 20, cy), (cx + 20 + mark_len, cy), mark_color, thickness)
            cv2.line(page, (cx, cy + 20), (cx, cy + 20 + mark_len), mark_color, thickness)
        
        # Add page label
        label = f"Page {page_num}: {positions[page_num]}"
        cv2.putText(page, label, (50, page_height_px - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add dimensions
        dim_text = f"{squares_per_page_x}x{squares_per_page_y} squares @ {square_size_mm}mm"
        cv2.putText(page, dim_text, (50, page_height_px - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Save page
        path = os.path.join(calibration_dir, f"charuco_large_4page_{page_num}.png")
        cv2.imwrite(path, page)
        print(f"  Page {page_num} ({positions[page_num]}): {path}")
    
    # Create assembly guide showing how pages fit together
    guide_size = 500
    guide = np.ones((guide_size, guide_size, 3), dtype=np.uint8) * 255
    
    # Draw the 2x2 grid with page numbers
    margin = 50
    cell_w = (guide_size - 3 * margin) // 2
    cell_h = (guide_size - 3 * margin) // 2
    
    for row in range(2):
        for col in range(2):
            x1 = margin + col * (cell_w + margin)
            y1 = margin + row * (cell_h + margin)
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            
            # Alternate colors for visibility
            color = (230, 230, 230) if (row + col) % 2 == 0 else (210, 210, 210)
            cv2.rectangle(guide, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(guide, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
            page_num = row * 2 + col + 1
            cv2.putText(guide, str(page_num), (x1 + cell_w//2 - 15, y1 + cell_h//2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Add title
    cv2.putText(guide, "Assembly Guide", (guide_size//2 - 100, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(guide, "Align edges - no overlap needed!", (guide_size//2 - 140, guide_size - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    guide_path = os.path.join(calibration_dir, "charuco_large_4page_assembly.png")
    cv2.imwrite(guide_path, guide)
    print(f"  Assembly guide: {guide_path}")
    
    # Save preview
    preview_height = 400
    preview_width = int(preview_height * board_width_px / board_height_px)
    preview = cv2.resize(full_board_color, (preview_width, preview_height))
    
    # Draw grid lines showing page boundaries
    half_w = preview_width // 2
    half_h = preview_height // 2
    cv2.line(preview, (half_w, 0), (half_w, preview_height), (0, 0, 255), 2)
    cv2.line(preview, (0, half_h), (preview_width, half_h), (0, 0, 255), 2)
    
    preview_path = os.path.join(calibration_dir, "charuco_large_4page_preview.png")
    cv2.imwrite(preview_path, preview)
    print(f"  Preview: {preview_path}")
    
    # Update config
    config_path = os.path.join(calibration_dir, "camera_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['calibration']['board']['squares_x'] = total_squares_x
        config['calibration']['board']['squares_y'] = total_squares_y
        config['calibration']['board']['square_size'] = square_size_mm
        config['calibration']['board']['marker_size'] = marker_size_mm
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\n  Config updated: {config_path}")
    
    print("\n" + "=" * 60)
    print("  PRINTING & ASSEMBLY INSTRUCTIONS")
    print("=" * 60)
    print(f"""
1. Print all 4 pages at 100% scale (no fit-to-page!)

2. Cut along the GRAY rectangle on each page
   (This is the exact tile boundary)

3. Arrange pages in 2x2 grid - edges should meet exactly:

   +-------+-------+
   |   1   |   2   |
   +-------+-------+
   |   3   |   4   |
   +-------+-------+

4. Tape together on the BACK (no overlap on front!)

5. Mount on flat, rigid surface (cardboard, foam board)

6. Verify: measure a square - should be {square_size_mm}mm

Each page has {squares_per_page_x}x{squares_per_page_y} COMPLETE squares.
Total board: {total_squares_x}x{total_squares_y} squares = {board_width_mm}x{board_height_mm}mm
""")


if __name__ == "__main__":
    main()
