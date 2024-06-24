'''
To my understanding, the consistency decoder is a conditional generater 
whose condition is the latent feature of Lantet Diffusion VAE.
'''
import torch
from diffusers import StableDiffusionPipeline
from consistencydecoder import ConsistencyDecoder, save_image, load_image

# encode with stable diffusion vae
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, device="cuda:0"
)
pipe.vae.cuda()
decoder_consistency = ConsistencyDecoder(device="cuda:0") # Model size: 2.49 GB

image = load_image("assets/gt1.png", size=(256, 256), center_crop=True)
latent = pipe.vae.encode(image.half().cuda()).latent_dist.mean

import time
start = time.time()
# decode with gan
sample_gan = pipe.vae.decode(latent).sample.detach()
save_image(sample_gan, "gan.png")
print("Time taken for GAN: ", time.time()-start)

start = time.time()
# decode with vae
sample_consistency = decoder_consistency(latent)
save_image(sample_consistency, "con.png")
print("Time taken for Consistency: ", time.time()-start)