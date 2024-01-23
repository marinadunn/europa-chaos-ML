from PIL import Image
import os

def get_seq_crop(img_dir_in, lbl_dir_out, img_dir_out, lbl_dir_out, crop_height, crop_width):
    """
    
    Input: files, directory containing raw images, directory containing raw labels, image & label output directories, desired crop height & width with no overlapping image tiles
    
    Output: sequentially cropped images only containing ice blocks
    
    """
    for fn in os.listdir(img_dir_in):
        if fn.endswith(".png"): 
            cnt=0
            img = np.array(Image.open(os.path.join(img_dir_in, fn)))
            lbl = np.array(Image.open(os.path.join(lbl_dir_in, fn)))
            name, ext = os.path.splitext(fn)
            max_x = img.shape[0] - crop_width
            max_y = img.shape[1] - crop_height
            cnt = 0
            for x in range(0,max_x,crop_width):
                for y in range(0,max_y,crop_height):
                    img_crop = img[x: x + int(crop_height), y: y + int(crop_width)]
                    lbl_crop = lbl[x: x + int(crop_height), y: y + int(crop_width)]
                    obj_ids = np.unique(lbl_crop)
                    if len(obj_ids)>1: 
                        Image.fromarray(img_crop).save(f'{img_dir_out}{fn}_{x}_{y}_{cnt}.png',"PNG")
                        Image.fromarray(lbl_crop).save(f'{lbl_dir_out}{fn}_{x}_{y}_{cnt}.png',"PNG")
                        cnt+=1

if __name__ == "__main__":
    get_seq_crop("./LabelMe_files/PNG RegMaps/data_annotated/val/Europa_Chaos_hh-kk.png", 
                 "./LabelMe_files/PNG RegMaps/data_dataset_voc/SegmentationObjectPNG/val/Europa_Chaos_hh-kk.png",
                 "./InstanceSeg/ImageSeg/",
                 "./InstanceSeg/MaskSeg/", 
                 100, 
                 100
                )