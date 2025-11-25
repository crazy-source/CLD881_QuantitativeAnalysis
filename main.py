import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")

# List of folders to analyze
FOLDERS = ["20", "30", "30_2d", "30_2rho", "45"]

# Time step settings
TIME_STEP_REAL_WORLD = 0.001

# CALIBRATION (Update this if you know the real width!)
PIXEL_TO_METER = 1.0 

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_output_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for sub in ["csv_data", "individual_plots", "comparative_plots", "debug_crops"]:
        path = os.path.join(OUTPUT_DIR, sub)
        if not os.path.exists(path):
            os.makedirs(path)

def extract_step_number(filename):
    match = re.search(r'_(\d+)\.(png|jpg|jpeg)$', filename)
    if match:
        return int(match.group(1))
    return 0

def crop_image_to_content(img):
    """
    Auto-crops the image to the simulation domain.
    It looks for the largest rectangular colored area, ignoring white/text.
    """
    # Convert to HSV to easily detect colored regions (simulation) vs white/gray (background)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define a mask for colored pixels (S > 20, V > 20) - ignore white/black/gray
    # This works for both your "rainbow" plots
    mask = cv2.inRange(hsv, (0, 20, 20), (180, 255, 255))
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the colored region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img # Return original if crop fails

    # Assume the simulation domain is the largest colored bounding box
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Add a small sanity check: if the box is too small, it might be a logo
    if w < 50 or h < 50:
        return img

    return img[y:y+h, x:x+w]

def analyze_single_image(image_path, folder_name):
    img = cv2.imread(image_path)
    if img is None: return None

    # 1. AUTO-CROP to the simulation domain
    cropped = crop_image_to_content(img)
    
    # Optional: Save debug crop for the first image to verify
    if "0010." in image_path: # Save just a few for checking
        debug_path = os.path.join(OUTPUT_DIR, "debug_crops", f"{folder_name}_crop_check.png")
        cv2.imwrite(debug_path, cropped)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # 3. Adaptive Thresholding for Gradient Boundaries
    # Your 20m/s case has a sharp gradient, 45m/s has a smoother one.
    # Otsu's binarization automatically finds the best threshold value.
    # We invert because in your plots, GAS (Blue) is dark in grayscale, Liquid (Red) is light.
    # Wait... Blue (0,0,255) -> Gray ~29. Red (0,0,255) -> Gray ~76.
    # So Gas (Blue) is DARKER than Liquid (Red).
    # We want the Bubble (Gas) to be White for finding contours.
    # So we need to threshold the DARK pixels.
    
    # Simple Threshold: Pixels DARKER than X become WHITE
    # (Inverted Thresholding)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # 4. Find Contours (Bubble)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return None

    # Filter contours: keep largest ones (ignore tiny noise specs)
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not contours: return None
    
    # The bubble might be split into multiple blobs if gradient is weird
    # Combine all significant contours or just take the largest
    c = max(contours, key=cv2.contourArea)
    
    # Metrics Calculation
    area_px = cv2.contourArea(c)
    if area_px == 0: return None
    
    area_m2 = area_px * (PIXEL_TO_METER ** 2)
    diameter_px = np.sqrt(4 * area_px / np.pi)
    diameter_m = diameter_px * PIXEL_TO_METER
    
    x, y, w, h = cv2.boundingRect(c)
    width_m = w * PIXEL_TO_METER
    height_m = h * PIXEL_TO_METER
    
    shape_factor = float(h) / float(w) if w != 0 else 0
    
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0
    
    M = cv2.moments(c)
    cx = int(M["m10"] / M["m00"]) * PIXEL_TO_METER if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) * PIXEL_TO_METER if M["m00"] != 0 else 0

    # Adjust Y centroid to be relative to the BOTTOM of the crop (Height - cy)
    # because images start 0 at top-left.
    cy_from_bottom = (cropped.shape[0] * PIXEL_TO_METER) - cy

    return {
        "area": area_m2,
        "diameter": diameter_m,
        "width": width_m,
        "height": height_m,
        "shape_factor": shape_factor,
        "circularity": circularity,
        "cx": cx,
        "cy": cy_from_bottom
    }

def process_folder(folder_name):
    print(f"--- Processing folder: {folder_name} ---")
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    # Match png, jpg, jpeg
    image_files = glob.glob(os.path.join(folder_path, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {folder_name}")
        return None
        
    image_files.sort(key=extract_step_number)

    data_records = []

    for img_path in image_files:
        step_num = extract_step_number(img_path)
        real_time = step_num * TIME_STEP_REAL_WORLD
        
        metrics = analyze_single_image(img_path, folder_name)
        
        if metrics:
            record = {
                "time_step": step_num,
                "real_time": real_time,
                **metrics
            }
            data_records.append(record)

    df = pd.DataFrame(data_records)
    
    # Calculate Vertical Velocity
    if not df.empty and "cy" in df.columns:
        df["velocity_y"] = df["cy"].diff() / TIME_STEP_REAL_WORLD
        df["velocity_y"] = df["velocity_y"].fillna(0)
        
        # Calculate Horizontal Drift Velocity
        df["velocity_x"] = df["cx"].diff() / TIME_STEP_REAL_WORLD
        df["velocity_x"] = df["velocity_x"].fillna(0)

    if not df.empty:
        csv_name = f"data_{folder_name}.csv"
        csv_path = os.path.join(OUTPUT_DIR, "csv_data", csv_name)
        df.to_csv(csv_path, index=False)
        return df
    return None

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_comparative(all_data, col, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for name, df in all_data.items():
        # Smooth noisy data (common in image analysis)
        smooth_data = df[col].rolling(window=3, center=True).mean()
        plt.plot(df["real_time"], smooth_data, label=f"Case: {name}", linewidth=2)
    
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_plots", filename))
    plt.close()

def plot_mixed_interesting(all_data):
    # 1. Trajectory Map (X vs Y position)
    plt.figure(figsize=(6, 10))
    for name, df in all_data.items():
        plt.plot(df["cx"], df["cy"], label=name)
    plt.title("Bubble Centroid Trajectory")
    plt.xlabel("Horizontal Position (m)")
    plt.ylabel("Vertical Height (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_plots", "mixed_trajectory.png"))
    plt.close()

    # 2. Aspect Ratio vs Rise Velocity (Do faster bubbles flatten more?)
    plt.figure(figsize=(10, 6))
    for name, df in all_data.items():
        plt.scatter(df["velocity_y"], df["shape_factor"], label=name, alpha=0.5, s=10)
    plt.title("Shape Factor vs Rise Velocity")
    plt.xlabel("Rise Velocity (m/s)")
    plt.ylabel("Shape Factor (H/W)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_plots", "mixed_shape_vs_vel.png"))
    plt.close()

# ==========================================
# MAIN
# ==========================================

def main():
    create_output_dirs()
    all_dfs = {}

    for folder in FOLDERS:
        if os.path.exists(os.path.join(BASE_DIR, folder)):
            df = process_folder(folder)
            if df is not None:
                all_dfs[folder] = df
    
    if not all_dfs:
        print("No data found.")
        return

    print("--- Generating Plots ---")
    
    # Standard Plots
    plot_comparative(all_dfs, "diameter", "Equivalent Bubble Diameter", "Diameter (m)", "compare_diameter.png")
    plot_comparative(all_dfs, "shape_factor", "Bubble Shape Factor (H/W)", "Shape Factor (-)", "compare_shape_factor.png")
    plot_comparative(all_dfs, "circularity", "Bubble Circularity", "Circularity (-)", "compare_circularity.png")
    plot_comparative(all_dfs, "velocity_y", "Bubble Rise Velocity", "Velocity Y (m/s)", "compare_velocity.png")

    # Interesting Plots
    plot_mixed_interesting(all_dfs)

    print("Done. Check 'analysis_results' folder.")

if __name__ == "__main__":
    main()