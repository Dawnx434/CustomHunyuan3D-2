import os
from PIL import Image
import trimesh

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# 模型路径和图片路径设置
model_path = 'tencent/Hunyuan3D-2'
root_dir = "/mnt/homes/junxiao-ldap/6D-Pose-Anything/tmp/YCBVideo_render"
input_glb_dir = "/mnt/homes/junxiao-ldap/6D-Pose-Anything/tmp/YCBV_Cousin_Reference/hunyuan_single"
output_glb_dir = "/mnt/homes/junxiao-ldap/6D-Pose-Anything/tmp/YCBV_Cousin_Reference/hunyuan_single_tex"
os.makedirs(output_glb_dir, exist_ok=True)

# 加载流水线
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
rembg = BackgroundRemover()

# 分类
categories = [
    "master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "tuna_fish_can", "pudding_box",
    "gelatin_box", "potted_meat_can", "banana", "pitcher_base", "bleach_cleanser", "bowl", "mug", "power_drill",
    "wood_block", "scissors", "large_marker", "large_clamp", "extra_large_clamp", "foam_brick"
]

# 遍历模型
for category in categories:
    category_dir = os.path.join(root_dir, category)
    if not os.path.isdir(category_dir):
        continue

    for seq_id in os.listdir(category_dir):
        # 构造输入输出路径
        glb_path = os.path.join(input_glb_dir, f"obj_{seq_id}.glb")
        out_path = os.path.join(output_glb_dir, f"obj_{seq_id}.glb")
        image_path = os.path.join(category_dir, seq_id, "rgb", "rgb_0013.png")

        if not os.path.exists(glb_path) or not os.path.exists(image_path):
            continue
        if os.path.exists(out_path):
            print(f"Already textured: {seq_id}")
            continue

        try:
            print(f"Texturing {category} / {seq_id}")
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB":
                image = rembg(image)

            # 加载 mesh 并纹理化
            mesh = trimesh.load(glb_path, force='mesh')
            mesh = pipeline_texgen(mesh, image=image)

            mesh.export(out_path)
            print(f"--> Saved textured model to {out_path}")
        except Exception as e:
            print(f"Failed to texture {category} / {seq_id}: {e}")
