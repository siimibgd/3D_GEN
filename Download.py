from diffusers import DiffusionPipeline

model_id = "LiheYoung/depth-anything-small-hf"  # Repo de pe Hugging Face
pipeline = DiffusionPipeline.from_pretrained(model_id)
pipeline.to("cuda")
