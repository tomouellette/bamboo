#!/usr/bin/env python3
import ssl
import torch
import datetime
from transformers import AutoModel

MODEL_ID = "google/medsiglip-448"


def echo(message: str):
    print(f"{datetime.datetime.now().isoformat()} | {message}")


echo("Setting context.")
ssl._create_default_https_context = ssl._create_unverified_context
device = "cpu"

echo("Downloading and loading model.")
model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()


echo("Tracing model.")


class MedSigLIPVisionTower(torch.nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        return self.vision_model(pixel_values).pooler_output


example_input = torch.randn(1, 3, 448, 448).to(device)
traced_model = torch.jit.trace(MedSigLIPVisionTower(model.vision_model), example_input)

echo("Saving model.")
traced_filename = "medsiglip_vision_traced.pt"
traced_model.save(traced_filename)

echo("Complete.")
