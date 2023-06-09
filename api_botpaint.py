from diffusers import DiffusionPipeline
import torch

pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
pipe_prior.to("cuda")

t2i_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
t2i_pipe.to("cuda")

prompt = "A McDonals cheeseburger creature eating itself, claymation, cinematic, moody lighting"

negative_prompt = "low quality, bad quality"

generator = torch.Generator(device="cuda").manual_seed(12)
image_embeds, negative_image_embeds = pipe_prior(prompt, negative_prompt, guidance_scale=1.0, generator=generator).to_tuple()

image = t2i_pipe(prompt, negative_prompt=negative_prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds).images[0]
image.save("cheeseburger_monster.png")
