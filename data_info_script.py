import cv2
import numpy as np
from src.info.nasa_data import CHAOS_REGION_ALIAS_TO_FILE_MAP, CHAOS_REGION_ALIAS_TO_LABEL_MAP, CHAOS_REGION_ALIAS_TO_REGION_MAP
from src.info.resolutions import CHAOS_REGION_RESOLUTION_MAP
from src.info.file_structure import INFO_OUTPUT_PATH
import src.utility.file_text_processing as ftp

# Avg plate size km
# std plate size km
# Total Plate Area km
# Total Chaos Area km

data_path = f"{INFO_OUTPUT_PATH}/pixel_metrics.txt"
header = f"Region & Plate Count & Chaos Area & Plate Area & Plate Area Mean & Plate Area Std & Plate Coverage\n"
ftp.create_output_csv(data_path, header)

if __name__ == "__main__":
    print("Starting")
    region_aliases = ["A", "aa", "B", "bb", "C", "Co", "D", "dd", "E", "ee", "F", "ff", "G", "gg", "H", "hh", "I", "ii", "jj", "kk"]
    # region_aliases = ["H", "hh", "I", "ii", "jj", "kk"]
    for region_alias in region_aliases:
        reso = CHAOS_REGION_RESOLUTION_MAP[region_alias]/1000
        # Chaos Region Area
        chaos_region_mask = cv2.imread(CHAOS_REGION_ALIAS_TO_REGION_MAP[region_alias])[:, :, 0]
        chaos_region_area = np.sum(np.where(chaos_region_mask > 0, 1, 0))*reso*reso

        # Object Count
        chaos_region_lbls = cv2.imread(CHAOS_REGION_ALIAS_TO_LABEL_MAP[region_alias])
        rgb_type = np.dtype([('R', np.uint8), ('G', np.uint8), ('B', np.uint8)])
        struc_data = chaos_region_lbls.view(rgb_type)
        unique_colors = np.unique(struc_data)

        # Plate Area Stats
        plate_count = 0
        plate_areas = []
        for void_color in unique_colors:
            mask_color = void_color.tolist()
            if mask_color == (0,0,0):
                continue
            single_mask = np.all(chaos_region_lbls == mask_color, axis=2)
            crop_area = np.sum(np.where(single_mask > 0, 1, 0))
            if crop_area > 5:
                crop_area = crop_area*reso*reso
                plate_areas.append(crop_area)
                plate_count += 1

        plate_areas = np.asarray(plate_areas)
        total_plate_area = np.sum(plate_areas)
        mean_plate_area = np.mean(plate_areas)
        std_plate_area = np.std(plate_areas)

        plate_coverage = total_plate_area/chaos_region_area*100
        obs = f"{region_alias} & {plate_count} & {chaos_region_area:.3f} & {total_plate_area:.3f} & {mean_plate_area:.3f} & {std_plate_area:.3f} & {plate_coverage:.2f}%"
        ftp.append_input_to_file(data_path, obs)