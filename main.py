import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
from scipy.signal import savgol_filter

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")

# List of folders to analyze
FOLDERS = ["20", "30", "30_2d", "30_2rho", "45"]

# Time step settings
TIME_STEP_REAL_WORLD = 0.001

# CALIBRATION
# UPDATE THIS: Pixels to Meters conversion factor.
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

def crop_image_to_content(img, folder_name, filename):
    """
    Auto-crops to the simulation domain (the colored box), removing white/text.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 20, 20), (180, 255, 255))
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    if w < 50 or h < 50:
        return img

    cropped = img[y:y+h, x:x+w]
    
    if "0010." in filename: 
        debug_path = os.path.join(OUTPUT_DIR, "debug_crops", f"crop_{folder_name}.png")
        cv2.imwrite(debug_path, cropped)
        
    return cropped

def analyze_single_image(image_path, folder_name):
    img = cv2.imread(image_path)
    if img is None: return None

    filename = os.path.basename(image_path)
    img = crop_image_to_content(img, folder_name, filename)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask for Blue/Cyan bubble
    blue_mask = cv2.inRange(hsv, (80, 40, 40), (150, 255, 255))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    bubble_contour = None
    
    if len(contours) == 1:
        return None
    elif len(contours) >= 2:
        c1 = contours[0]
        c2 = contours[1]
        M1 = cv2.moments(c1)
        cy1 = int(M1["m01"] / M1["m00"]) if M1["m00"] != 0 else 0
        M2 = cv2.moments(c2)
        cy2 = int(M2["m01"] / M2["m00"]) if M2["m00"] != 0 else 0
        
        if cy1 > cy2:
            bubble_contour = c1 
        else:
            bubble_contour = c2
            
    if bubble_contour is None:
        return None

    c = bubble_contour
    area_px = cv2.contourArea(c)
    area_m2 = area_px * (PIXEL_TO_METER ** 2)
    diameter_px = np.sqrt(4 * area_px / np.pi)
    diameter_m = diameter_px * PIXEL_TO_METER
    
    x, y, w, h = cv2.boundingRect(c)
    width_m = w * PIXEL_TO_METER
    height_m = h * PIXEL_TO_METER
    shape_factor = float(h) / float(w) if w != 0 else 0
    
    M = cv2.moments(c)
    cx = int(M["m10"] / M["m00"]) * PIXEL_TO_METER if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) * PIXEL_TO_METER if M["m00"] != 0 else 0
    
    img_height_px = img.shape[0]
    cy_from_bottom = (img_height_px * PIXEL_TO_METER) - cy
    
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0

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

# ==========================================
# DATA CLEANING
# ==========================================

def remove_outliers(series, window=7, sigma=2.0):
    """
    Robust outlier removal using rolling median and standard deviation.
    Points > sigma * std_dev away from the local median are replaced.
    """
    # Calculate local statistics
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
    
    # Identify outliers
    # Add small epsilon to std to prevent division by zero/tight constraints on flat data
    diff = np.abs(series - rolling_median)
    outlier_mask = diff > (sigma * (rolling_std + 1e-6))
    
    # Create cleaned series
    clean_series = series.copy()
    clean_series[outlier_mask] = np.nan
    
    # Interpolate to fill the gaps
    clean_series = clean_series.interpolate(method='linear', limit_direction='both')
    
    # Fill edges if necessary
    clean_series = clean_series.bfill().ffill()
    
    return clean_series

def process_folder(folder_name):
    print(f"--- Processing {folder_name} ---")
    folder_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_files.sort(key=extract_step_number)
    
    if not image_files:
        print(f"No images in {folder_name}")
        return None

    data = []
    for img_path in image_files:
        step = extract_step_number(img_path)
        time = step * TIME_STEP_REAL_WORLD
        
        metrics = analyze_single_image(img_path, folder_name)
        if metrics:
            metrics["time_step"] = step
            metrics["real_time"] = time
            data.append(metrics)
            
    df = pd.DataFrame(data)
    if not df.empty:
        # 1. Calculate Raw Velocity
        df["velocity_y_raw"] = np.gradient(df["cy"], df["real_time"])
        
        # 2. CLEAN DATA (Remove Outliers)
        # We create new "clean" columns for the CSV and plotting
        df["diameter_clean"] = remove_outliers(df["diameter"])
        df["shape_factor_clean"] = remove_outliers(df["shape_factor"])
        df["velocity_y_clean"] = remove_outliers(df["velocity_y_raw"])
        df["circularity_clean"] = remove_outliers(df["circularity"])
        df["area_clean"] = remove_outliers(df["area"])
        
        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, "csv_data", f"data_{folder_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV for {folder_name} (with cleaned data)")
        return df
    return None

# ==========================================
# PLOTTING
# ==========================================

def smooth_data(y_values, window=11, poly=3):
    """Applies Savitzky-Golay filter for smooth trendlines"""
    if len(y_values) < window:
        return y_values 
    return savgol_filter(y_values, window, poly)

def plot_individual(df, folder_name):
    """Plot Diameter, Shape, Velocity as SEPARATE images using cleaned data"""
    t = df["real_time"]
    
    # Use the cleaned columns we created in process_folder
    dia = df["diameter_clean"]
    shape = df["shape_factor_clean"]
    vel = df["velocity_y_clean"]

    # 1. Diameter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(t, dia, alpha=0.4, color='blue', label='Data (Cleaned)', s=15)
    plt.plot(t, smooth_data(dia), color='darkblue', linewidth=2, label='Trend')
    plt.title(f"{folder_name}: Diameter vs Time")
    plt.ylabel("Diameter (m)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "individual_plots", f"{folder_name}_diameter.png"))
    plt.close()

    # 2. Shape Factor Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(t, shape, alpha=0.4, color='red', label='Data (Cleaned)', s=15)
    plt.plot(t, smooth_data(shape), color='darkred', linewidth=2, label='Trend')
    plt.title(f"{folder_name}: Shape Factor vs Time")
    plt.ylabel("Shape Factor (H/W)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "individual_plots", f"{folder_name}_shape.png"))
    plt.close()

    # 3. Velocity Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(t, vel, alpha=0.4, color='green', label='Data (Cleaned)', s=15)
    plt.plot(t, smooth_data(vel), color='darkgreen', linewidth=2, label='Trend')
    plt.title(f"{folder_name}: Rise Velocity vs Time")
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "individual_plots", f"{folder_name}_velocity.png"))
    plt.close()

def plot_comparative(all_dfs, column, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_dfs)))
    
    # Map the requested column to its cleaned version
    clean_col_map = {
        "diameter": "diameter_clean",
        "shape_factor": "shape_factor_clean",
        "velocity_y": "velocity_y_clean",
        "circularity": "circularity_clean",
        "area": "area_clean"
    }
    
    target_col = clean_col_map.get(column, column)
    
    for (name, df), color in zip(all_dfs.items(), colors):
        t = df["real_time"]
        y = df[target_col]
        
        # Plot Raw Points (faint)
        plt.scatter(t, y, color=color, alpha=0.15, s=10)
        
        # Plot Smooth Trendline (solid)
        y_smooth = smooth_data(y)
        plt.plot(t, y_smooth, color=color, linewidth=2, label=name)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend(title="Case")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_plots", filename))
    plt.close()

def plot_interesting(all_dfs):
    # 1. Trajectory
    plt.figure(figsize=(8, 10))
    for name, df in all_dfs.items():
        # Use raw coords for trajectory as outliers are less jumpy here
        plt.plot(df["cx"], df["cy"], label=name, linewidth=2)
    plt.title("Bubble Trajectory (Centroid Path)")
    plt.xlabel("Horizontal Position (m)")
    plt.ylabel("Vertical Height (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_plots", "mixed_trajectory.png"))
    plt.close()
    
    # 2. Shape vs Velocity (Cleaned)
    plt.figure(figsize=(10, 6))
    for name, df in all_dfs.items():
        vel = df["velocity_y_clean"]
        shape = df["shape_factor_clean"]
        plt.scatter(vel, shape, label=name, alpha=0.5, s=15)
    plt.title("Correlation: Shape Factor vs Rise Velocity")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Shape Factor")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_plots", "mixed_shape_vs_velocity.png"))
    plt.close()

# ==========================================
# MAIN
# ==========================================

def main():
    create_output_dirs()
    all_dfs = {}
    
    for folder in FOLDERS:
        df = process_folder(folder)
        if df is not None:
            all_dfs[folder] = df
            plot_individual(df, folder)
            
    if not all_dfs:
        print("No data found!")
        return

    print("Generating Comparative Plots...")
    plot_comparative(all_dfs, "diameter", "Equivalent Diameter Comparison", "Diameter (m)", "compare_diameter.png")
    plot_comparative(all_dfs, "shape_factor", "Shape Factor Comparison", "Shape Factor (-)", "compare_shape.png")
    plot_comparative(all_dfs, "velocity_y", "Rise Velocity Comparison", "Velocity (m/s)", "compare_velocity.png")
    plot_comparative(all_dfs, "circularity", "Circularity Comparison", "Circularity (-)", "compare_circularity.png")
    
    print("Generating Mixed Plots...")
    plot_interesting(all_dfs)
    
    print(f"Done! Analysis saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()