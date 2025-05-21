from typing import Any, Optional, Dict
from smolagents.tools import Tool
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse

class WebScrapingTool(Tool):
    name = "web_scraping"
    description = "Scrape content from web pages including text, links, and specific HTML elements"
    inputs = {
        'url': {
            'type': 'string',
            'description': 'The URL of the webpage to scrape',
            'nullable': True
        },
        'action': {
            'type': 'string',
            'description': 'The scraping action to perform: "text" (get all text), "links" (get all links), "element" (get specific elements)',
            'default': 'text',
            'nullable': True
        },
        'selector': {
            'type': 'string',
            'description': 'CSS selector for specific elements (used with "element" action)',
            'nullable': True
        },
        'attributes': {
            'type': 'array',
            'description': 'List of attributes to extract from elements',
            'items': {'type': 'string'},
            'nullable': True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _get_soup(self, url: str) -> BeautifulSoup:
        """Get BeautifulSoup object from URL."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            raise Exception(f"Error fetching URL: {str(e)}")

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract all text from webpage."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract all links from webpage."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            absolute_url = urljoin(base_url, href)
            links.append({
                'text': text,
                'url': absolute_url
            })
        return links

    def _extract_elements(self, soup: BeautifulSoup, selector: str, attributes: Optional[list] = None) -> list:
        """Extract specific elements using CSS selector."""
        elements = []
        for element in soup.select(selector):
            if not attributes:
                elements.append(element.get_text(strip=True))
            else:
                elem_data = {'text': element.get_text(strip=True)}
                for attr in attributes:
                    elem_data[attr] = element.get(attr, '')
                elements.append(elem_data)
        return elements

    def forward(self, url: Optional[str] = None, action: str = 'text', selector: Optional[str] = None, attributes: Optional[list] = None) -> str:
        """
        Execute the web scraping operation.
        
        Args:
            url: The URL to scrape. Required for all operations.
            action: The type of scraping to perform ('text', 'links', or 'element'). Defaults to 'text'.
            selector: CSS selector for finding specific elements. Required for 'element' action.
            attributes: List of attributes to extract from elements. Optional.
            
        Returns:
            str: JSON string containing the scraping results
        """
        if not url:
            return json.dumps({
                'error': 'URL is required',
                'action': action
            }, indent=2)

        try:
            soup = self._get_soup(url)
            
            if action == 'text':
                result = self._extract_text(soup)
            elif action == 'links':
                result = self._extract_links(soup, url)
            elif action == 'element' and selector:
                result = self._extract_elements(soup, selector, attributes)
            else:
                raise ValueError("Invalid action or missing selector for 'element' action")
            
            return json.dumps({
                'url': url,
                'action': action,
                'result': result
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                'error': str(e),
                'url': url,
                'action': action
            }, indent=2)
