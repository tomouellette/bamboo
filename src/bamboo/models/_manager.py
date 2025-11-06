# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import urllib.request
import urllib.parse
import http.cookiejar
import re

from pathlib import Path
from tqdm import tqdm
from urllib.error import URLError, HTTPError


MODELS: dict = {
    "grandqc": {"source": "gdrive", "identifier": "1B_uRQW-LWUUaIxowibaOCFFibFDwp_u6"}
}


def download_gdrive(identifier: str, destination: str | Path) -> None:
    """Download a file from google drive using its file identifier.

    Parameters
    ----------
    identifier : str
        Google drive file identifier specifying a publicly available file.
    destination : str or Path
        Path to save the downloaded file.

    Raises
    ------
    RuntimeError
        If the download fails or cannot be retrieved from Google Drive.
    URLError
        If the network request fails due to connectivity or URL issues.
    HTTPError
        If the HTTP request returns a non-successful status code.
    """
    URL = "https://docs.google.com/uc?export=download"
    destination = Path(destination)

    session = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )

    try:
        response = session.open(f"{URL}&id={identifier}")
        html = response.read()
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to initiate download for {identifier}: {e}")

    try:
        html_text = html.decode("utf-8")
        token_match = re.search(r"confirm=([0-9A-Za-z_]+)", html_text)
        token = token_match.group(1) if token_match else None

        if token:
            response = session.open(f"{URL}&confirm={token}&id={identifier}")
    except UnicodeDecodeError:
        # Already binary content, no token step required
        response = session.open(f"{URL}&id={identifier}")

    total_size = int(response.headers.get("Content-Length", 0))

    with (
        open(destination, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {destination.name}",
            ncols=80,
        ) as pbar,
    ):
        while True:
            chunk = response.read(32768)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))


def download_model(name: str, dest: str | Path) -> Path:
    """Download a model from a predefined registry of available models.

    Parameters
    ----------
    name : str
        The model name as defined `MODELS` dictionary.
    dest : str or Path
        Path where the model file should be saved.

    Returns
    -------
    pathlib.Path
        The resolved path to the downloaded model file.

    Raises
    ------
    ValueError
        If the model name is not defined in ``MODELS``.
    NotImplementedError
        If the model source type is unsupported.
    RuntimeError
        If the download process encounters an unrecoverable error.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'")

    model = MODELS[name]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if model["source"] == "gdrive":
        download_gdrive(model["identifier"], dest)
    else:
        raise NotImplementedError(f"Unsupported source: {model['source']}")

    return dest
