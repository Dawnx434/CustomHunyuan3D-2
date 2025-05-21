import os
import torch
from PIL import Image
import trimesh

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline

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
            img_paths = {
                "front": os.path.join(rgb_dir, "rgb_0013.png"),
                "left":  os.path.join(rgb_dir, "rgb_0011.png"),
                "back":  os.path.join(rgb_dir, "rgb_0012.png")
            }
            if all(os.path.exists(p) for p in img_paths.values()):
                image_groups.append((category, seq_id, img_paths))
    return image_groups

def process_texturing(glb_input_dir, image_groups, textured_output_dir):
    os.makedirs(textured_output_dir, exist_ok=True)

    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-paint-v2-0-turbo'
    )
    rembg = BackgroundRemover()

    for category, seq_id, images_dict in image_groups:
        model_path = os.path.join(glb_input_dir, f"obj_{seq_id}.glb")
        if not os.path.exists(model_path):
            print(f"Skipped {category}/{seq_id}: model not found.")
            continue

        try:
            # Load and preprocess images
            images = []
            for key in ["front", "left", "back"]:
                image = Image.open(images_dict[key])
                if image.mode == "RGB":
                    image = rembg(image)
                images.append(image)

            print(f"Texturing {category}/{seq_id} ...")
            mesh = trimesh.load(model_path)
            mesh = pipeline(mesh, image=images)

            output_path = os.path.join(textured_output_dir, f"obj_{seq_id}_textured.glb")
            mesh.export(output_path)
            print(f"--> Saved textured model to {output_path}")
        except Exception as e:
            print(f"Failed to texture {category}/{seq_id}: {e}")

# 配置路径和类别
categories = [
    "master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "tuna_fish_can", "pudding_box",
    "gelatin_box", "potted_meat_can", "banana", "pitcher_base", "bleach_cleanser", "bowl", "mug", "power_drill",
    "wood_block", "scissors", "large_marker", "large_clamp", "extra_large_clamp", "foam_brick"
]

image_root_dir = "/mnt/homes/junxiao-ldap/Hunyuan3D-2/YCBVideo_render"
glb_input_dir = "/mnt/homes/junxiao-ldap/6D-Pose-Anything/tmp/YCBV_Cousin_Reference/hunyuan_multi"
textured_output_dir = "/mnt/homes/junxiao-ldap/6D-Pose-Anything/tmp/YCBV_Cousin_Reference/hunyuan_multi_textured"

# 查找图像组
image_groups = find_image_groups(image_root_dir, categories)

# 批量纹理生成
process_texturing(glb_input_dir, image_groups, textured_output_dir)
