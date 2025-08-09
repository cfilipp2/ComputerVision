##### CONSTANTS #####

MAX_AREA_POINTDEFECTS_IN_NM = 40*40 # [nm*nm]
MIN_AREA_POINTDEFECTS_IN_NM = 20*20 # [nm*nm]
NOISE_FLOOR = 130e-12 # [pA]

import cv2
import numpy as np
import random
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Config:
    """Configuration parameters for the analysis"""
    SHOW_PLOT: bool = True
    MAX_AREA_POINTDEFECTS_NM2: float = 40 * 40  # nm²
    MIN_AREA_POINTDEFECTS_NM2: float = 20 * 20  # nm²
    NOISE_FLOOR: float = 130e-12  # pA
    SCALE_BAR_NM: int = 500  # nm for scale bar

config = Config()
@dataclass
class ImageData:
    """Container for image data and metadata"""
    image_size: Tuple[int, int]  # (width, height) in pixels
    sample_size: Tuple[float, float]  # (width, height) in meters
    gray_image: np.ndarray
    float_image: np.ndarray
    current_matrix: np.ndarray
    extreme_values: Tuple[float, float]  # (min, max) current values
    nm_per_pixel: float
    
    @property
    def pixel_to_nm(self) -> float:
        """Conversion factor from pixels to nanometers"""
        return self.sample_size[0] / self.image_size[0] * 1e9

def normalize_value(value: float, matrix: np.ndarray) -> float:
    """Normalize a value to 0-1 range based on matrix min/max"""
    return (value - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

def detect_extended_defects(image_data: ImageData) -> Tuple[np.ndarray, float]:
    """
    Detect extended defects using Canny edge detection
    Returns: (edge_overlay_image, coverage_percentage)
    """
    # Calculate statistics
    mean_val = np.mean(image_data.current_matrix)
    std_val = np.std(image_data.current_matrix)
    
    # Set Canny thresholds
    lower_threshold = max(0, normalize_value(mean_val - std_val, image_data.current_matrix))
    upper_threshold = normalize_value(mean_val + std_val, image_data.current_matrix)
    
    # Apply bilateral filter for noise reduction
    filtered = cv2.bilateralFilter(image_data.gray_image, 9, 75, 75)
    
    # Detect edges
    edges = cv2.Canny(filtered, 
                      int(lower_threshold * 255), 
                      int(upper_threshold * 255))
    
    # Create colored overlay
    overlay = create_edge_overlay(image_data.gray_image, edges, color=(203, 112, 51))
    
    # Calculate coverage
    coverage = np.count_nonzero(edges) / edges.size
    
    return overlay, coverage * 100

def detect_point_defects(image_data: ImageData) -> Tuple[np.ndarray, List[int], int]:
    """
    Detect point defects using multi-level Otsu thresholding
    Returns: (colored_image, blob_counts_per_band, total_blobs)
    """
    # Determine number of bands using KDE
    n_bands = min(4, len(find_kde_peaks(image_data.current_matrix.flatten())) + 1)
    
    # Apply multi-Otsu thresholding
    thresholds = threshold_multiotsu(image_data.float_image, classes=n_bands)
    
    # Calculate area limits in pixels
    max_area_px = int(config.MAX_AREA_POINTDEFECTS_NM2 / (image_data.nm_per_pixel ** 2))
    min_area_px = int(config.MIN_AREA_POINTDEFECTS_NM2 / (image_data.nm_per_pixel ** 2))
    
    # Process each band
    result_image = cv2.cvtColor(image_data.gray_image, cv2.COLOR_GRAY2BGR)
    result_image = cv2.GaussianBlur(result_image, (9, 9), 0)
    
    blob_counts = []
    total_blobs = 0
    
    # Create band images
    band_images = create_band_images(image_data.gray_image, 
                                     [int(t * 255) for t in thresholds])
    
    # Detect blobs in each band
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Red, Blue, Green
    
    for i, band_img in enumerate(band_images[:n_bands]):
        blobs = detect_blobs_in_band(band_img, min_area_px, max_area_px)
        
        # Draw blobs
        color = colors[i] if i < len(colors) else (128, 128, 128)
        for blob in blobs:
            cv2.drawContours(result_image, [blob], -1, color, 1)
        
        blob_counts.append(len(blobs))
        total_blobs += len(blobs)
    
    return result_image, blob_counts, total_blobs

def detect_blobs_in_band(band_image: np.ndarray, 
                         min_area: int, max_area: int) -> List[np.ndarray]:
    """Detect blob contours in a single band"""
    # Convert to grayscale if needed
    if len(band_image.shape) == 3:
        gray = cv2.cvtColor(band_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = band_image
    
    # Threshold and blur
    _, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by size and shape
    valid_blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            if is_blob_valid(contour):
                valid_blobs.append(contour)
    
    return valid_blobs

def is_blob_valid(contour: np.ndarray, solidity_threshold: float = 0.6) -> bool:
    """Check if a contour represents a valid blob (rounded and solid)"""
    # Check aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    is_rounded = 0.5 <= aspect_ratio <= 1.5
    
    # Check solidity
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area > 0:
        solidity = float(area) / hull_area
        return is_rounded and solidity > solidity_threshold
    
    return False

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def find_kde_peaks(data: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """Find peaks in data using Kernel Density Estimation"""
    kde = gaussian_kde(data)
    x_range = np.linspace(np.min(data), np.max(data), 1000)
    kde_values = kde(x_range)
    
    # Find peaks
    peaks, _ = find_peaks(kde_values)
    peak_values = kde_values[peaks]
    
    # Filter significant peaks
    threshold = threshold_ratio * np.max(kde_values)
    significant_peaks = x_range[peaks[peak_values > threshold]]
    
    return significant_peaks

def create_band_images(gray_image: np.ndarray, thresholds: List[int]) -> List[np.ndarray]:
    """Create separate images for each intensity band"""
    normalized = gray_image.astype(np.float32) / 255.0
    colors = [(0, 0, 255), (245, 0, 0), (0, 255, 0), (0, 255, 255), (128, 0, 128)]
    
    band_images = []
    lower_bound = 0
    
    for i, upper_bound in enumerate(thresholds + [255]):
        # Create band mask
        band_mask = cv2.inRange(normalized, lower_bound/255.0, upper_bound/255.0)
        
        # Create colored image
        color = colors[i % len(colors)]
        color_layer = np.full((band_mask.shape[0], band_mask.shape[1], 3), 
                              color, dtype=np.uint8)
        colored_band = cv2.bitwise_and(color_layer, color_layer, mask=band_mask)
        
        band_images.append(colored_band)
        lower_bound = upper_bound + 1
    
    return band_images

def create_edge_overlay(gray_image: np.ndarray, edges: np.ndarray, 
                        color: Tuple[int, int, int]) -> np.ndarray:
    """Create colored edge overlay on grayscale image"""
    # Convert to RGB
    gray_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    
    # Create edge color layer
    edge_color = np.zeros((*edges.shape, 3), dtype=np.uint8)
    edge_color[..., :] = color
    
    # Blend images
    edge_alpha = edges.astype(np.float32) / 255.0
    result = gray_rgb.astype(np.float32) / 255.0
    overlay = edge_color.astype(np.float32) / 255.0
    
    blended = result * (1 - edge_alpha[..., None]) + overlay * edge_alpha[..., None]
    
    return (blended * 255).astype(np.uint8)


def load_xyz_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, z data from file"""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2]

def process_xyz_to_image(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> ImageData:
    """Convert XYZ data to image matrices"""
    
    # Determine image dimensions
    width = len(np.unique(x))
    height = len(np.unique(y))
    
    # Get sample dimensions
    sample_width = x[-1] - x[0]
    sample_height = y[-1] - y[0]
    
    # Create current matrix
    current_matrix = z.reshape(height, width).astype(np.float32)
    
    # Apply noise floor
    current_matrix[current_matrix <= config.NOISE_FLOOR] = config.NOISE_FLOOR - 0.1e-12
    
    # Store extreme values
    extreme_values = (np.min(z), np.max(z))
    
    # Normalize to 0-255 for image processing
    normalized = ((current_matrix - np.min(current_matrix)) / 
                  (np.max(current_matrix) - np.min(current_matrix)) * 255)
    gray_image = normalized.astype(np.uint8)
    
    # Create float image (0-1 range)
    float_image = gray_image.astype(np.float32) / 255.0
    
    # Calculate resolution
    nm_per_pixel = (sample_width / width) * 1e9
    
    return ImageData(
        image_size=(width, height),
        sample_size=(sample_width, sample_height),
        gray_image=gray_image,
        float_image=float_image,
        current_matrix=current_matrix,
        extreme_values=extreme_values,
        nm_per_pixel=nm_per_pixel
    )

def analyze_sample(filepath: str, sample_name: str = None) -> dict:

    if sample_name is None:
        sample_name = filepath.split("/")[-1].split(".")[0]
    
    
    print(f"Analyzing: {sample_name}\n\n")
    
    
    # Load and process data
    x, y, z = load_xyz_data(filepath)
    image_data = process_xyz_to_image(x, y, z)
    
    print(f"Image Size: {image_data.image_size[0]} × {image_data.image_size[1]} px")
    print(f"Sample Size: {image_data.sample_size[0]*1e9:.1f} × {image_data.sample_size[1]*1e9:.1f} nm")
    print(f"Resolution: {image_data.nm_per_pixel:.2f} nm/pixel")
    print(f"Current Range: {image_data.extreme_values[0]:.2e} to {image_data.extreme_values[1]:.2e} A")
    
    # Detect extended defects
    print("\nDetecting extended defects...")
    edge_image, edge_coverage = detect_extended_defects(image_data)
    print(f"Extended defects coverage: {edge_coverage:.2f}%")
    
    # Detect point defects
    print("\nDetecting point defects...")
    defect_image, blob_counts, total_blobs = detect_point_defects(image_data)
    print(f"Point defects found: {total_blobs}")
    print(f"Distribution by band: {blob_counts}")
    
    # Create visualization report
    print("\nGenerating analysis report...")
    create_analysis_report(sample_name, image_data, edge_image, 
                          edge_coverage, defect_image, blob_counts)
    
    # Return results dictionary
    results = {
        'sample_name': sample_name,
        'image_size': image_data.image_size,
        'sample_size_nm': (image_data.sample_size[0]*1e9, image_data.sample_size[1]*1e9),
        'resolution_nm_per_px': image_data.nm_per_pixel,
        'current_range': image_data.extreme_values,
        'current_mean': np.mean(image_data.current_matrix),
        'extended_defects_coverage': edge_coverage,
        'point_defects_total': total_blobs,
        'point_defects_distribution': blob_counts
    }
    
    return results

def process_multiple_samples(filepaths: List[str]) -> List[dict]:
    """Process multiple samples and return results"""
    all_results = []
    
    for filepath in filepaths:
        try:
            results = analyze_sample(filepath)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"\n{result['sample_name']}:")
        print(f"  - Extended defects: {result['extended_defects_coverage']:.1f}%")
        print(f"  - Point defects: {result['point_defects_total']}")
    
    return all_results

def add_scale_bar(ax, nm_per_pixel: float, length_nm: int = 500) -> None:
    """Add scale bar to image"""
    length_px = int(length_nm / nm_per_pixel)
    scalebar = AnchoredSizeBar(ax.transData, 
                               size=length_px,
                               label=f"{length_nm} nm", 
                               loc='lower right', 
                               pad=0.1, 
                               color='white', 
                               frameon=False, 
                               size_vertical=2,
                               fontproperties=fm.FontProperties(size=10))
    ax.add_artist(scalebar)


def create_combined_overlay(gray_image: np.ndarray, 
                           edge_image: np.ndarray, 
                           defect_image: np.ndarray) -> np.ndarray:
    """
    Combine original image with edge and defect overlays
    - Extended defects (edges) shown from edge_image
    - Point defects shown from defect_image
    """
    # Start with the grayscale image as RGB base
    if len(gray_image.shape) == 2:
        base = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    else:
        base = gray_image.copy()
    
    # The edge_image already contains the gray background with orange edges overlaid
    # The defect_image already contains the gray background with colored point defects
    
    # We need to extract just the colored pixels (defects/edges) from each image
    
    # For edge detection: edge_image has orange color (203, 112, 51) where edges are
    # Create a mask for orange pixels
    if len(edge_image.shape) == 3:
        # Find pixels that are NOT grayscale (indicating they are colored edges)
        edge_mask = np.logical_or(
            edge_image[:,:,0] != edge_image[:,:,1],
            edge_image[:,:,1] != edge_image[:,:,2]
        )
    else:
        edge_mask = edge_image > 0
    
    # For point defects: defect_image has colors (red, blue, green) where defects are
    if len(defect_image.shape) == 3:
        # Find pixels that are NOT grayscale (indicating they are colored defects)
        defect_mask = np.logical_or(
            defect_image[:,:,0] != defect_image[:,:,1],
            defect_image[:,:,1] != defect_image[:,:,2]
        )
    else:
        defect_mask = defect_image > 0
    
    # Apply overlays
    result = base.copy()
    
    # First apply edge overlay (extended defects)
    if len(edge_image.shape) == 3 and np.any(edge_mask):
        result[edge_mask] = edge_image[edge_mask]
    
    # Then apply defect overlay (point defects) - this overwrites edges if they overlap
    if len(defect_image.shape) == 3 and np.any(defect_mask):
        result[defect_mask] = defect_image[defect_mask]
    
    return result


def create_analysis_report(sample_name: str, image_data: ImageData,
                          edge_image: np.ndarray, edge_coverage: float,
                          defect_image: np.ndarray, blob_counts: List[int]) -> None:
    """Create visualization with defects overlay on original image"""
    
    # Create the combined overlay
    overlay = create_combined_overlay(image_data.gray_image, edge_image, defect_image)
    
    # Create figure with two subplots for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(image_data.gray_image, cmap='gray')
    ax1.set_title(f"{sample_name} - Original", fontsize=12, fontweight='bold')
    ax1.axis('off')
    add_scale_bar(ax1, image_data.nm_per_pixel)
    
    # Overlay with defects
    ax2.imshow(overlay)
    total_defects = sum(blob_counts)
    title = (f"{sample_name} - Defect Analysis\n"
             f"Point Defects: {total_defects} (R:{blob_counts[0] if len(blob_counts)>0 else 0}, "
             f"B:{blob_counts[1] if len(blob_counts)>1 else 0}, "
             f"G:{blob_counts[2] if len(blob_counts)>2 else 0}) | "
             f"Extended Defects: {edge_coverage:.1f}% coverage")
    ax2.set_title(title, fontsize=12, fontweight='bold')
    ax2.axis('off')
    add_scale_bar(ax2, image_data.nm_per_pixel)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(203/255, 112/255, 51/255), label='Extended Defects (Edges)'),
        Patch(facecolor='red', label='Point Defects (Band 1)'),
        Patch(facecolor='blue', label='Point Defects (Band 2)'),
        Patch(facecolor='green', label='Point Defects (Band 3)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()


# Alternative: If the above doesn't work, here's a more direct approach
def create_combined_overlay_v2(image_data: ImageData, edges: np.ndarray, 
                              point_defect_contours: List[List[np.ndarray]], 
                              defect_colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Direct method to create overlay from raw detection results
    
    Args:
        image_data: Original image data
        edges: Binary edge mask from Canny detection
        point_defect_contours: List of contour lists for each band
        defect_colors: Colors for each band of point defects
    """
    if defect_colors is None:
        defect_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Red, Blue, Green
    
    # Start with grayscale as RGB
    result = cv2.cvtColor(image_data.gray_image, cv2.COLOR_GRAY2RGB)
    
    # Draw extended defects (edges) in orange
    edge_color = (203, 112, 51)  # Orange
    result[edges > 0] = edge_color
    
    # Draw point defects
    for band_idx, contours in enumerate(point_defect_contours):
        color = defect_colors[band_idx % len(defect_colors)]
        for contour in contours:
            cv2.drawContours(result, [contour], -1, color, 2)  # Thickness of 2 for visibility
    
    return result

if __name__ == "__main__":
    # Define sample paths
    sample_paths = [
        "../MoS2_1ML.xyz","../MoS2_3ML.xyz"
    ]
    
    # Process all samples
    results = process_multiple_samples(sample_paths)
    
    print("\n✓ Analysis complete!")
