#!/usr/bin/env python3
import io
import ssl
import torch
import requests
import datetime
import segmentation_models_pytorch as smp

MODEL_LINK = "https://zenodo.org/records/14041538/files/GrandQC_MPP15.pth?download=1"


def echo(message: str):
    print(f"{datetime.datetime.now().isoformat()} | {message}")


echo("Setting context.")
ssl._create_default_https_context = ssl._create_unverified_context

echo("Downloading and loading model.")
model = torch.load(
    io.BytesIO(requests.get(MODEL_LINK, stream=True).content),
    map_location="cpu",
    weights_only=False,
)
model.eval()

echo("Tracing model.")
example_input = torch.randn(1, 3, 512, 512)
traced_prim = torch.jit.trace(model, example_input)

echo("Saving model.")
traced_prim.save("grandqc.traced.pt")

echo("Complete.")
