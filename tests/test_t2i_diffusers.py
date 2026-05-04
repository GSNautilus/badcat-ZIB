"""Phase 0 validation: confirm Z-Image-Turbo runs T2I via diffusers on RTX 3060 12GB."""

import torch
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

def main():
    model_id = "Tongyi-MAI/Z-Image-Turbo"

    print("Loading pipeline (bf16)...")
    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.transformer.enable_gradient_checkpointing()

    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    prompt = "a photo of a golden retriever sitting in a field of sunflowers"
    print(f"Generating: {prompt!r}")

    image = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    out_path = "outputs/phase0_t2i_test.png"
    import os
    os.makedirs("outputs", exist_ok=True)
    image.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()
