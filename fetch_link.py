import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import time

BASE_URL = "https://alphaleo.tech/"
DOMAIN = urlparse(BASE_URL).netloc

headers = {
    "User-Agent": "Mozilla/5.0"
}

visited = set()


def crawl(url):

    if url in visited:
        return

    visited.add(url)

    print("Visiting:", url)

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return

    except Exception as e:
        print(e)
        return

    soup = BeautifulSoup(response.text, "html.parser")

    for a in soup.find_all("a", href=True):

        href = a["href"]

        full_url = urljoin(BASE_URL, href)

        parsed = urlparse(full_url)

        # Remove query string and fragment
        clean_url = parsed.scheme + "://" + parsed.netloc + parsed.path

        # Only internal links
        if parsed.netloc == DOMAIN:

            if clean_url not in visited:

                time.sleep(0.5)
                crawl(clean_url)


crawl(BASE_URL)

df = pd.DataFrame(sorted(visited), columns=["URL"])

df.to_csv("website_links.csv", index=False)

print("Total Links:", len(df))