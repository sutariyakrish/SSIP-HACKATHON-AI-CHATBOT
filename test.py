import re
import time
import logging
import threading
from typing import Optional, List
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class FastWebsiteQABot:
    """
    Fast website QA bot with:
    - Limited crawl with strict timeouts
    - Clean main-content extraction
    - Small chunking + fast embeddings (all-MiniLM-L6-v2)
    - FAISS cosine search
    - Local LLM generation via GPT4All with tiny output for speed

    Target: <5s response after models are loaded (network dependent).
    """

    def __init__(
        self,
        model_path: str,
        max_links: int = 6,          # keep low for speed
        chunk_size: int = 500,       # small chunks -> faster embed + better recall
        per_request_timeout: tuple = (2, 2),  # (connect, read) seconds
        read_limit_bytes: int = 120_000,      # read only first 120KB per page
        crawl_budget_sec: float = 2.0,        # time budget for crawling
    ):
        self.model_path = model_path
        self.model: Optional[GPT4All] = None
        self.embedder: Optional[SentenceTransformer] = None

        self.max_links = max_links
        self.chunk_size = chunk_size
        self.per_request_timeout = per_request_timeout
        self.read_limit_bytes = read_limit_bytes
        self.crawl_budget_sec = crawl_budget_sec

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # Corpus state
        self.visited = set()
        self.chunks: List[str] = []
        self.chunk_map = {}   # chunk_idx -> url
        self.index = None

        # Locks for thread safety
        self._visited_lock = threading.Lock()
        self._chunks_lock = threading.Lock()

    # -------- Models --------

    def load_model(self) -> bool:
        """Load GPT4All model with broad version compatibility."""
        try:
            print("Loading GPT4All...")
            try:
                # Newer gpt4all versions
                self.model = GPT4All(
                    model_name="E:\hackathon\models\gpt4all-falcon.gguf"
,
                    allow_download=False
                )
            except TypeError:
                # Older versions: only positional model_name/path
                self.model = GPT4All(self.model_path)
            print("✅ Model ready!")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

    def load_embedder(self) -> bool:
        """Load sentence-transformer embedder."""
        try:
            print("Loading embedder...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            self.embedder.eval()
            print("✅ Embedder ready!")
            return True
        except Exception as e:
            print(f"❌ Embedder loading failed: {e}")
            return False

    def load_models(self) -> bool:
        """Load both LLM and embedder."""
        ok_m = self.load_model()
        ok_e = self.load_embedder()
        return ok_m and ok_e

    # -------- Crawl + Extract --------

    @staticmethod
    def _normalize_url(url: str) -> str:
        if not url.startswith(('http://', 'https://')):
            return 'https://' + url
        return url

    def _fetch_page_fast(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch only the first read_limit_bytes from a page; require text/html; close promptly."""
        try:
            with self.session.get(url, timeout=self.per_request_timeout, stream=True) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get('Content-Type', '')
                if 'text/html' not in ctype.lower():
                    return None

                # Read only up to read_limit_bytes
                buf = []
                total = 0
                for chunk in resp.iter_content(chunk_size=16384):
                    if not chunk:
                        break
                    buf.append(chunk)
                    total += len(chunk)
                    if total >= self.read_limit_bytes:
                        break
                content = b''.join(buf)
                try:
                    return BeautifulSoup(content, 'lxml')
                except Exception:
                    return BeautifulSoup(content, 'html.parser')
        except Exception:
            return None

    @staticmethod
    def _extract_text_fast(soup: BeautifulSoup) -> str:
        """Try main/article; fallback to largest text block; strip noise."""
        if not soup:
            return ""

        # Remove noise
        for tag in soup(["script", "style", "noscript", "svg", "img", "video", "iframe",
                         "nav", "aside", "footer", "header", "form"]):
            tag.decompose()

        main = soup.find('main') or soup.find('article')
        node = main or soup.body or soup

        # Fallback to largest text block if little content
        text = node.get_text(" ", strip=True) if node else ""
        text = re.sub(r'\s+', ' ', text)
        if len(text) < 200:
            best = None
            best_len = 0
            for el in soup.find_all(['section', 'div', 'p']):
                t = el.get_text(" ", strip=True)
                l = len(t)
                if l > best_len:
                    best_len = l
                    best = t
            text = best or text

        text = re.sub(r'\s+', ' ', text or "")
        return text[:4000]  # cap for speed

    def _add_chunks(self, url: str, text: str):
        """Chunk with small overlap and keep only meaningful chunks."""
        if not text:
            return
        overlap = 80
        step = max(50, self.chunk_size - overlap)
        new_chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i + self.chunk_size].strip()
            if len(chunk) > 100:
                new_chunks.append(chunk)
            i += step

        with self._chunks_lock:
            base_idx = len(self.chunks)
            self.chunks.extend(new_chunks)
            for j, _ in enumerate(new_chunks):
                self.chunk_map[base_idx + j] = url

    # -------- Crawl Orchestration --------

    def _collect_links(self, start_url: str, soup: BeautifulSoup, base_domain: str) -> List[str]:
        links = []
        for a in soup.find_all("a", href=True, limit=60):
            href = a.get("href")
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
            full = urljoin(start_url, href)
            if (urlparse(full).netloc == base_domain and
                not re.search(r'\.(pdf|jpg|jpeg|png|gif|css|js|ico|xml|zip|mp4|webm)$', full.lower())):
                links.append(full)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for l in links:
            if l not in seen:
                deduped.append(l)
                seen.add(l)
        return deduped

    def _process_link(self, url: str):
        with self._visited_lock:
            if url in self.visited or len(self.visited) >= self.max_links:
                return
            self.visited.add(url)

        soup = self._fetch_page_fast(url)
        if not soup:
            return
        text = self._extract_text_fast(soup)
        if text:
            self._add_chunks(url, text)

    def crawl_parallel(self, start_url: str) -> None:
        """Crawl main page + a few internal links within a strict time budget."""
        start_url = self._normalize_url(start_url)
        base_domain = urlparse(start_url).netloc

        start_time = time.time()
        self.visited = set()
        self.chunks = []
        self.chunk_map = {}
        self.index = None

        # Fetch main page
        main_soup = self._fetch_page_fast(start_url)
        if not main_soup:
            print("❌ Could not fetch initial page")
            return

        with self._visited_lock:
            self.visited.add(start_url)
        main_text = self._extract_text_fast(main_soup)
        if main_text:
            self._add_chunks(start_url, main_text)

        # Collect a small set of internal links
        links = self._collect_links(start_url, main_soup, base_domain)
        links = links[:max(0, self.max_links - 1)]

        # If no time left, skip parallel crawl
        elapsed = time.time() - start_time
        remaining_budget = max(0.0, self.crawl_budget_sec - elapsed)
        if remaining_budget < 0.2 or not links:
            return

        # Process in parallel with a short global timeout
        futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for link in links:
                futures.append(executor.submit(self._process_link, link))

            # Wait only within remaining budget
            end_by = time.time() + remaining_budget
            try:
                for f in as_completed(futures, timeout=remaining_budget):
                    try:
                        _ = f.result(timeout=max(0.0, end_by - time.time()))
                    except Exception:
                        pass
            except Exception:
                # Timeout or other issues; proceed with whatever we have
                pass

    # -------- Index + Search --------

    def build_index_fast(self) -> bool:
        """Fast FAISS index build for current chunks."""
        if not self.chunks or not self.embedder:
            return False
        try:
            # Small corpora -> single batch is fine
            embeddings = self.embedder.encode(
                self.chunks,
                batch_size=32,
                show_progress_bar=False
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)

            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            return True
        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            return False

    def search_fast(self, question: str, top_k: int = 3) -> str:
        if not self.index or not self.embedder or not self.chunks:
            return ""
        try:
            q_emb = self.embedder.encode([question])
            q_emb = np.asarray(q_emb, dtype=np.float32)
            faiss.normalize_L2(q_emb)
            _, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
            parts = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    parts.append(self.chunks[idx])
            return "\n\n".join(parts)
        except Exception:
            return ""

    # -------- Answer --------

    def answer_fast(self, question: str) -> str:
        """Answer using GPT4All with small output for speed."""
        if not self.index or not self.model:
            return "❌ System not ready"

        context = self.search_fast(question, top_k=3)
        if not context:
            return "❌ No relevant content found"

        prompt = f"Use the context to answer concisely.\n\nContext:\n{context[:2000]}\n\nQ: {question}\nA:"

        try:
            # Compatible generate call (fallback if params unsupported)
            try:
                answer = self.model.generate(
                    prompt,
                    max_tokens=64,     # small output for speed
                    temp=0.1,
                    top_p=0.9
                )
            except TypeError:
                # Older API with fewer kwargs
                answer = self.model.generate(prompt)
            return answer.strip()[:500]
        except Exception as e:
            return f"Error: {e}"

    # -------- One-shot URL QA (fits in ~5s) --------

    def answer_from_url(self, url: str, question: str) -> str:
        """
        End-to-end: crawl a few pages from `url`, build index, answer `question`.
        After models are loaded, this aims for ~<5s per query (subject to network).
        """
        t0 = time.time()
        self.crawl_parallel(url)
        self.build_index_fast()
        ans = self.answer_fast(question)
        elapsed = time.time() - t0
        return f"{ans}\n\n⏱️ {elapsed:.2f}s"


if __name__ == "__main__":
    # Example CLI usage:
    # 1) Load models once (cold start can take longer)
    # 2) Reuse the bot to answer multiple questions quickly
    bot = FastWebsiteQABot(
        model_path="gpt4all-falcon-q4_0.gguf",  # change to your local model file
        max_links=6,
        chunk_size=500,
        crawl_budget_sec=2.0
    )

    if not bot.load_models():
        raise SystemExit(1)

    while True:
        url = input("🌐 Enter URL (or 'quit'): ").strip()
        if url.lower() in ("quit", "exit"):
            break
        q = input("❓ Question: ").strip()
        print("💨 Answering...")
        print(bot.answer_from_url(url, q))