import time

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

images = {
    "front": "/mnt/homes/junxiao-ldap/Hunyuan3D-2/YCBVideo_render/bleach_cleanser/000012/rgb/rgb_0013.png",
    "left": "/mnt/homes/junxiao-ldap/Hunyuan3D-2/YCBVideo_render/bleach_cleanser/000012/rgb/rgb_0011.png",
    "back": "/mnt/homes/junxiao-ldap/Hunyuan3D-2/YCBVideo_render/bleach_cleanser/000012/rgb/rgb_0012.png"
}

for key in images:
    image = Image.open(images[key]).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    images[key] = image

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv-turbo',
    variant='fp16'
)
pipeline.enable_flashvdm()
start_time = time.time()
mesh = pipeline(
    image=images,
    num_inference_steps=5,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
print("--- %s seconds ---" % (time.time() - start_time))
mesh.export(f'demo_mv3.glb')
