#!/usr/bin/env python3
"""Export MobileCLIP/MobileCLIP2 models to ONNX format.

Usage:
    python scripts/export_mobileclip_onnx.py \
        --model_name MobileCLIP2-S0 \
        --pretrained models_cache/mobileclip2/MOBILECLIP2-S0/mobileclip2_s0.pt \
        --save_dir models \
        --image_res 224 \
        --opset 17
"""

import argparse
import torch
import open_clip
from pathlib import Path
from mobileclip.modules.common.mobileone import reparameterize_model

def main():
    ap = argparse.ArgumentParser(description="Export MobileCLIP/MobileCLIP2 to ONNX")
    ap.add_argument("--model_name", default="MobileCLIP2-S0", help="Model name (e.g., MobileCLIP2-S0)")
    ap.add_argument("--pretrained", required=True, help="Path to model checkpoint (.pt file)")
    ap.add_argument("--save_dir", default="models", help="Output directory for ONNX files")
    ap.add_argument("--image_res", type=int, default=224, help="Image resolution (224 or 336)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--fp16", action="store_true", help="Export weights as FP16")
    args = ap.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)

    # S0/S2/B use no normalization; S3/S4/L-14 use CLIP normalization
    model_kwargs = {}
    if not (args.model_name.endswith("S3") or args.model_name.endswith("S4") or args.model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    # Load model + eval + reparameterize
    print(f"Loading model: {args.model_name} from {args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, **model_kwargs, device="cpu"
    )
    model.eval()
    model = reparameterize_model(model)  # Must reparameterize before export

    # Wrapper to call encode_image / encode_text directly
    class ImageWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        
        def forward(self, pixel_values: torch.FloatTensor):
            return self.m.encode_image(pixel_values)  # [B, D]

    class TextWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        
        def forward(self, input_ids: torch.LongTensor):
            return self.m.encode_text(input_ids)  # [B, D]

    img_wrap = ImageWrap(model)
    text_wrap = TextWrap(model)

    # Create dummy inputs and export
    dummy_img = torch.randn(
        1, 3, args.image_res, args.image_res,
        dtype=torch.float16 if args.fp16 else torch.float32
    )
    dummy_ids = open_clip.tokenize(["a traffic light"], context_length=77)  # int64 [1,77]

    img_path = save / "image_encoder.onnx"
    txt_path = save / "text_encoder.onnx"

    print(f"Exporting image encoder to {img_path}")
    torch.onnx.export(
        img_wrap, dummy_img, str(img_path),
        input_names=["pixel_values"],
        output_names=["image_emb"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "h", 3: "w"},
            "image_emb": {0: "batch"}
        },
        do_constant_folding=True,
        opset_version=args.opset
    )

    print(f"Exporting text encoder to {txt_path}")
    torch.onnx.export(
        text_wrap, dummy_ids, str(txt_path),
        input_names=["input_ids"],
        output_names=["text_emb"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "text_emb": {0: "batch"}
        },
        do_constant_folding=True,
        opset_version=args.opset
    )

    # Save preprocessing info for C++ deployment
    preprocess_file = save / "preprocess.txt"
    with open(preprocess_file, "w") as f:
        f.write(f"model={args.model_name}\n")
        f.write(f"image_res={args.image_res}\n")
        if model_kwargs:
            f.write("mean=0,0,0\n")
            f.write("std=1,1,1\n")
        else:
            f.write("mean=0.481,0.457,0.406\n")
            f.write("std=0.268,0.261,0.275\n")

    print(f"\nONNX export completed:")
    print(f"  - Image encoder: {img_path}")
    print(f"  - Text encoder: {txt_path}")
    print(f"  - Preprocessing info: {preprocess_file}")

if __name__ == "__main__":
    main()
