# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import urllib.request
import urllib.parse
import http.cookiejar
import re


def download_gdrive(identifier, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )

    response = session.open(f"{URL}&id={identifier}")
    content = response.read()

    try:
        html = content.decode("utf-8")
        token_match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
        token = token_match.group(1) if token_match else None

        if token:
            download_url = f"{URL}&confirm={token}&id={identifier}"
            response = session.open(download_url)
            content = response.read()
    except UnicodeDecodeError:
        pass

    with open(destination, "wb") as f:
        f.write(content)
