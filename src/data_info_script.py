import cv2
import numpy as np
from config import (CHAOS_REGION_ALIAS_TO_FILE_MAP,
                    CHAOS_REGION_ALIAS_TO_LABEL_MAP,
                    CHAOS_REGION_ALIAS_TO_REGION_MAP,
                    CHAOS_REGION_RESOLUTION_MAP,
                    INFO_OUTPUT_PATH
                    )
from utils.file_utils import create_output_csv, append_input_to_file


def calculate_metrics(region_alias):
    """
    Calculate ice block size metrics (in km) for a given chaos region.

    Args:
        region_alias (str): Alias of the region.

    Returns:
        str: Computed metrics for the region.
    """
    # Get resolution for region
    res = CHAOS_REGION_RESOLUTION_MAP[region_alias] / 1000

    # Load Chaos Region mask
    chaos_region_mask = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])[:, :, 0]

    # Calculate Chaos Region Area
    chaos_region_area = np.sum(np.where(chaos_region_mask > 0, 1, 0)) * res * res

    # Object Count
    chaos_region_lbls = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])
    rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
    struc_data = chaos_region_lbls.view(rgb_type)
    unique_colors = np.unique(struc_data)

    # Calculate Ice Block Areas and Counts
    block_count = 0
    block_areas = []

    for void_color in unique_colors:
        mask_color = void_color.tolist()
        # Skip black mask
        if mask_color == (0, 0, 0):
            continue
        single_mask = np.all(chaos_region_lbls == mask_color, axis=2)
        crop_area = np.sum(np.where(single_mask > 0, 1, 0))

        # Only count ice blocks with area > 5 pixels
        if crop_area > 5:
            crop_area = crop_area * res * res
            block_areas.append(crop_area)
            block_count += 1

    # Calculate Ice Block Area Metrics
    block_areas = np.asarray(block_areas)
    total_block_area = np.sum(block_areas)
    mean_block_area = np.mean(block_areas)
    std_block_area = np.std(block_areas)

    # Calculate Ice Block Coverage
    block_coverage = total_block_area / chaos_region_area * 100
    obs = f"{region_alias} & {block_count} & {chaos_region_area:.3f} & {total_block_area:.3f} & {mean_block_area:.3f} & {std_block_area:.3f} & {block_coverage:.2f}%"

    return obs


def main():
    """Main function to calculate and write plate metrics for different regions."""
    data_path = f"{INFO_OUTPUT_PATH}/pixel_metrics.txt"
    header = f"Region & Plate Count & Chaos Area & Plate Area & Plate Area Mean & Plate Area Std & Plate Coverage\n"
    create_output_csv(data_path, header)

    print("Starting...")

    region_aliases = ["A", "aa", "B", "bb", "C", "Co",
                      "D", "dd", "E", "ee", "F", "ff",
                      "G", "gg", "H", "hh", "I", "ii",
                      "jj", "kk"]

    for region_alias in region_aliases:
        obs = calculate_metrics(region_alias)
        append_input_to_file(data_path, obs)


if __name__ == "__main__":
    main()
