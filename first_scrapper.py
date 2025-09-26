import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
from collections import deque
from bs4 import XMLParsedAsHTMLWarning
import warnings
import time
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


BASE_URL = "https://www.mosdac.gov.in"
visited = set()
queue = deque([BASE_URL])
data = []

def is_valid_url(url):
    """Check if URL belongs to mosdac.gov.in"""
    return urlparse(url).netloc.endswith("mosdac.gov.in")

while queue:
    url = queue.popleft()
    if url in visited:
        continue

    try:
        response = requests.get(url, timeout=5)
        if "text/html" not in response.headers.get("Content-Type", ""):
            print("Skipping non-HTML:", url)
            continue
    except Exception as e:
        print("Failed to fetch:", url, e)
        continue
    time.sleep(0.5)


    visited.add(url)
    try:
    # First, try parsing as HTML
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
    # If that fails, fall back to XML
        soup = BeautifulSoup(response.text, "xml")



    # Collect page data
    page = {
        "url": url,
        "title": soup.title.string if soup.title else "",
        "headings": [h.get_text(strip=True) for h in soup.find_all(['h1','h2','h3'])],
        "paragraphs": [p.get_text(strip=True) for p in soup.find_all('p')],
        "tables": []
    }

    # Extract table data
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
            rows.append(cells)
        page["tables"].append(rows)

    data.append(page)

    # Find new links
    for a in soup.find_all("a", href=True):
        new_url = urljoin(url, a["href"])
        if is_valid_url(new_url) and new_url not in visited:
            queue.append(new_url)

# Save to JSON file
with open("mosdac_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Finished! Data saved to mosdac_data.json")
