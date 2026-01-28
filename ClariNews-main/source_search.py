import logging
import re
import time
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from ddgs import DDGS
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

logger = logging.getLogger(__name__)

# Entity extraction patterns
PERSON_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',  # Full names
]

ORG_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+(?:Inc|Corp|Ltd|LLC|Company|Corporation|Co|Group)\.?))\b',
]

LOCATION_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+(?:City|State|Country|Province)))\b',
]


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text"""
    entities = {
        'persons': [],
        'organizations': [],
        'locations': [],
        'numbers': [],
        'dates': []
    }
    
    # Extract persons
    for pattern in PERSON_PATTERNS:
        matches = re.findall(pattern, text)
        entities['persons'].extend(matches[:5])
    
    # Extract organizations
    for pattern in ORG_PATTERNS:
        matches = re.findall(pattern, text)
        entities['organizations'].extend(matches[:5])
    
    # Extract locations
    for pattern in LOCATION_PATTERNS:
        matches = re.findall(pattern, text)
        entities['locations'].extend(matches[:3])
    
    # Extract numbers with units
    numbers = re.findall(r'\$?\d+(?:\.\d+)?\s*(?:trillion|billion|million|thousand|crore|lakh|%)', text.lower())
    entities['numbers'] = numbers[:5]
    
    # Extract years
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    entities['dates'] = years[:3]
    
    return entities


def calculate_source_credibility(url: str, content: str, title: str) -> float:
    """Calculate credibility score for a source"""
    score = 0.5  # Base score
    
    domain = _extract_domain(url).lower()
    
    # Tier 1 sources
    tier1 = ['reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk']
    if any(d in domain for d in tier1):
        score += 0.45
    
    # Tier 2 sources
    tier2 = ['nytimes.com', 'theguardian.com', 'washingtonpost.com', 'thehindu.com', 'indianexpress.com']
    if any(d in domain for d in tier2):
        score += 0.40
    
    # Tier 3 sources
    tier3 = ['cnn.com', 'bloomberg.com', 'forbes.com', 'economist.com', 'ndtv.com', 'hindustantimes.com']
    if any(d in domain for d in tier3):
        score += 0.35
    
    # Fact-checkers
    factcheckers = ['factcheck.org', 'snopes.com', 'politifact.com', 'boomlive.in', 'altnews.in']
    if any(d in domain for d in factcheckers):
        score += 0.40
    
    # Content quality bonus
    if len(content) > 2000:
        score += 0.05
    if len(content) > 3000:
        score += 0.05
    
    # Title quality
    if title and len(title) > 20:
        score += 0.02
    
    return min(1.0, score)


def search_and_extract(query: str, category: Optional[str] = None, max_results: int = 15) -> List[Dict[str, Any]]:
    """Enhanced search with better content extraction"""
    logger.info(f"Searching for: '{query}'")
    
    # Build search queries
    search_queries = [
        query,
        f"{query} news",
        f"{query} verified",
    ]
    
    if category and category.lower() != 'all':
        search_queries.insert(1, f"{query} {category}")
    
    all_sources = []
    seen_urls = set()
    
    # Try each query
    for search_query in search_queries[:3]:
        try:
            logger.info(f"Searching: '{search_query}'")
            results = list(DDGS().text(search_query, max_results=20))
            
            for result in results:
                url = result.get('link', result.get('href', ''))
                if not url or url in seen_urls:
                    continue
                
                seen_urls.add(url)
                
                title = result.get('title', 'No Title')
                snippet = result.get('body', result.get('snippet', ''))
                
                # Try to extract full content
                full_content = extract_full_content(url)
                
                if full_content and len(full_content) > 300:
                    credibility = calculate_source_credibility(url, full_content, title)
                    
                    all_sources.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'full_content': full_content,
                        'credibility': credibility,
                        'domain': _extract_domain(url)
                    })
                    
                    logger.info(f"✓ Extracted from {_extract_domain(url)}: {len(full_content)} chars (credibility: {credibility:.2f})")
                
                if len(all_sources) >= max_results:
                    break
            
            if len(all_sources) >= 5:
                break
                
            time.sleep(1)
        
        except Exception as e:
            logger.warning(f"Search failed for '{search_query}': {e}")
            continue
    
    # Sort by credibility
    all_sources.sort(key=lambda x: x.get('credibility', 0), reverse=True)
    
    logger.info(f"Found {len(all_sources)} sources")
    return all_sources[:max_results]


def extract_full_content(url: str) -> Optional[str]:
    """Extract full content with multiple methods"""
    
    # Method 1: Try trafilatura (best for news)
    content = _extract_with_trafilatura(url)
    if content and len(content) > 500:
        return content
    
    # Method 2: Try newspaper3k
    content = _extract_with_newspaper(url)
    if content and len(content) > 500:
        return content
    
    # Method 3: Try Selenium for JavaScript-rendered content
    content = _extract_with_selenium(url)
    if content and len(content) > 500:
        return content
    
    # Method 4: Fallback to BeautifulSoup
    content = _extract_with_beautifulsoup(url)
    return content


def _extract_with_trafilatura(url: str) -> Optional[str]:
    """Extract using trafilatura"""
    try:
        import trafilatura
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        downloaded = trafilatura.fetch_url(url, headers=headers)
        if not downloaded:
            return None
        
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
        
        if text and len(text) > 300:
            logger.debug(f"Trafilatura: {len(text)} chars from {url}")
            return text[:8000]
        
        return None
    
    except Exception as e:
        logger.debug(f"Trafilatura failed: {e}")
        return None


def _extract_with_newspaper(url: str) -> Optional[str]:
    """Extract using newspaper3k"""
    try:
        from newspaper import Article
        
        article = Article(url)
        article.download()
        article.parse()
        
        if article.text and len(article.text) > 300:
            logger.debug(f"Newspaper3k: {len(article.text)} chars from {url}")
            return article.text[:8000]
        
        return None
    
    except Exception as e:
        logger.debug(f"Newspaper3k failed: {e}")
        return None


def _extract_with_selenium(url: str) -> Optional[str]:
    """Extract using Selenium for JavaScript-rendered pages"""
    driver = None
    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Use chromium if available
        chrome_options.binary_location = '/usr/bin/chromium-browser'  # Adjust path as needed
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(20)
        
        # Load page
        driver.get(url)
        
        # Wait for content to load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except TimeoutException:
            pass  # Continue anyway
        
        # Extract text from multiple possible containers
        text = ""
        
        # Try article tag
        try:
            article = driver.find_element(By.TAG_NAME, "article")
            text = article.text
        except:
            pass
        
        # Try main tag
        if not text or len(text) < 300:
            try:
                main = driver.find_element(By.TAG_NAME, "main")
                text = main.text
            except:
                pass
        
        # Try common content classes
        if not text or len(text) < 300:
            content_selectors = [
                ".article-body", ".article-content", ".story-body",
                ".post-content", ".entry-content", ".content-body"
            ]
            for selector in content_selectors:
                try:
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    text = element.text
                    if len(text) > 300:
                        break
                except:
                    continue
        
        # Fallback: get all paragraphs
        if not text or len(text) < 300:
            try:
                paragraphs = driver.find_elements(By.TAG_NAME, "p")
                text = " ".join([p.text for p in paragraphs if len(p.text) > 50])
            except:
                pass
        
        if text and len(text) > 300:
            logger.debug(f"Selenium: {len(text)} chars from {url}")
            return text[:8000]
        
        return None
    
    except WebDriverException as e:
        logger.debug(f"Selenium WebDriver error: {e}")
        return None
    except Exception as e:
        logger.debug(f"Selenium failed: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def _extract_with_beautifulsoup(url: str) -> Optional[str]:
    """Extract using BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()
        
        text = ""
        
        # Strategy 1: article tag
        article = soup.find('article')
        if article:
            text = article.get_text(strip=True, separator=' ')
        
        # Strategy 2: main tag
        if not text or len(text) < 300:
            main = soup.find('main')
            if main:
                text = main.get_text(strip=True, separator=' ')
        
        # Strategy 3: content divs
        if not text or len(text) < 300:
            content_classes = [
                'article-body', 'article-content', 'story-body', 'post-content',
                'entry-content', 'content-body', 'article_body'
            ]
            for class_name in content_classes:
                content_div = soup.find(['div', 'section'], class_=lambda x: x and class_name in str(x).lower())
                if content_div:
                    text = content_div.get_text(strip=True, separator=' ')
                    break
        
        # Strategy 4: all paragraphs
        if not text or len(text) < 300:
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        
        # Clean text
        text = ' '.join(text.split())
        
        if len(text) > 300:
            logger.debug(f"BeautifulSoup: {len(text)} chars from {url}")
            return text[:8000]
        
        return None
    
    except Exception as e:
        logger.debug(f"BeautifulSoup failed for {url}: {e}")
        return None


def _extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        domain = urlparse(url).netloc
        return domain.replace('www.', '') if domain else 'Unknown'
    except:
        return 'Unknown'