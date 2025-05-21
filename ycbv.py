import os
import time
import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def find_image_groups(root_dir, categories):
    image_groups = []
    for category in categories:
        category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(category_dir):
            continue

        for seq_id in os.listdir(category_dir):
            rgb_dir = os.path.join(category_dir, seq_id, "rgb")
            if not os.path.isdir(rgb_dir):
                continue
            img_path = os.path.join(rgb_dir, "rgb_0013.png") 
            if os.path.exists(img_path):
                image_groups.append((category, seq_id, img_path))
    return image_groups

def process_single_image_groups(image_groups, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-dit-v2-0',
        variant='fp16'
    )

    rembg = BackgroundRemover()

    for category, seq_id, image_path in image_groups:
        print(f"Processing {category} / {seq_id}")
        try:
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB":
                image = rembg(image)

            start_time = time.time()
            mesh = pipeline(
                image=image,
                num_inference_steps=50,
                octree_resolution=380,
                num_chunks=20000,
                generator=torch.manual_seed(12345),
                output_type='trimesh'
            )[0]
            elapsed = time.time() - start_time
            print(f"--> Finished in {elapsed:.2f} seconds")

            save_path = os.path.join(output_dir, f"obj_{seq_id}.glb")
            mesh.export(save_path)
            print(f"--> Saved to {save_path}")
        except Exception as e:
            print(f"Failed to process {category} / {seq_id}: {e}")

# Categories to process
categories = [
    "master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "tuna_fish_can", "pudding_box",
    "gelatin_box", "potted_meat_can", "banana", "pitcher_base", "bleach_cleanser", "bowl", "mug", "power_drill",
    "wood_block", "scissors", "large_marker", "large_clamp", "extra_large_clamp", "foam_brick"
]

# Find valid image groups
root_dir = "//mnt/homes/junxiao-ldap/6D-Pose-Anything/tmp/YCBVideo_render"
image_groups = find_image_groups(root_dir, categories)

# Output folder
output_dir = "/mnt/homes/junxiao-ldap/6D-Pose-Anything/tmp/YCBV_Cousin_Reference/hunyuan_single"
process_single_image_groups(image_groups, output_dir)
