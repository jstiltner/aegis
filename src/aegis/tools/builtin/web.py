"""
Web tools for the agent runtime.

Provides tools for:
- HTTP requests
- Web scraping
- URL handling
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urljoin

from aegis.tools.models import ToolParameter, ParameterType, Tool, ToolDefinition
from aegis.tools.registry import ToolRegistry


async def http_request(args: dict[str, Any]) -> dict[str, Any]:
    """Make an HTTP request."""
    import httpx
    
    url = args["url"]
    method = args.get("method", "GET").upper()
    headers = args.get("headers", {})
    params = args.get("params", {})
    body = args.get("body")
    timeout = args.get("timeout", 30.0)
    follow_redirects = args.get("follow_redirects", True)
    
    # Validate URL
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
    ) as client:
        # Prepare request kwargs
        kwargs: dict[str, Any] = {
            "headers": headers,
            "params": params,
        }
        
        if body is not None:
            if isinstance(body, dict):
                kwargs["json"] = body
            else:
                kwargs["content"] = body
        
        response = await client.request(method, url, **kwargs)
        
        # Try to parse JSON response
        try:
            response_body = response.json()
        except Exception:
            response_body = response.text
        
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response_body,
            "url": str(response.url),
            "elapsed_ms": response.elapsed.total_seconds() * 1000,
        }


async def http_get(args: dict[str, Any]) -> dict[str, Any]:
    """Make an HTTP GET request."""
    args["method"] = "GET"
    return await http_request(args)


async def http_post(args: dict[str, Any]) -> dict[str, Any]:
    """Make an HTTP POST request."""
    args["method"] = "POST"
    return await http_request(args)


async def fetch_webpage(args: dict[str, Any]) -> dict[str, Any]:
    """Fetch a webpage and extract text content."""
    import httpx
    
    url = args["url"]
    timeout = args.get("timeout", 30.0)
    include_links = args.get("include_links", False)
    max_length = args.get("max_length", 50000)
    
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        html = response.text
    
    # Extract text content
    try:
        from html.parser import HTMLParser
        
        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts: list[str] = []
                self.links: list[dict[str, str]] = []
                self._in_script = False
                self._in_style = False
                self._current_href: str | None = None
            
            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag == "script":
                    self._in_script = True
                elif tag == "style":
                    self._in_style = True
                elif tag == "a":
                    for name, value in attrs:
                        if name == "href" and value:
                            self._current_href = value
            
            def handle_endtag(self, tag: str) -> None:
                if tag == "script":
                    self._in_script = False
                elif tag == "style":
                    self._in_style = False
                elif tag == "a":
                    self._current_href = None
            
            def handle_data(self, data: str) -> None:
                if not self._in_script and not self._in_style:
                    text = data.strip()
                    if text:
                        self.text_parts.append(text)
                        if self._current_href and include_links:
                            self.links.append({
                                "text": text,
                                "href": urljoin(url, self._current_href),
                            })
        
        extractor = TextExtractor()
        extractor.feed(html)
        
        text = " ".join(extractor.text_parts)
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        result: dict[str, Any] = {
            "url": str(response.url),
            "title": "",  # Would need more parsing
            "text": text,
            "length": len(text),
        }
        
        if include_links:
            result["links"] = extractor.links[:100]  # Limit links
        
        return result
        
    except Exception as e:
        # Fallback: return raw HTML truncated
        text = html[:max_length]
        return {
            "url": str(response.url),
            "text": text,
            "length": len(text),
            "raw_html": True,
            "parse_error": str(e),
        }


async def download_file(args: dict[str, Any]) -> dict[str, Any]:
    """Download a file from a URL."""
    import httpx
    from pathlib import Path
    
    url = args["url"]
    destination = Path(args["destination"])
    timeout = args.get("timeout", 60.0)
    overwrite = args.get("overwrite", False)
    max_size = args.get("max_size", 100 * 1024 * 1024)  # 100MB default
    
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    
    # Create parent directories
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_size:
                raise ValueError(f"File too large: {content_length} > {max_size}")
            
            # Download in chunks
            total_size = 0
            with open(destination, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    total_size += len(chunk)
                    if total_size > max_size:
                        raise ValueError(f"File too large: {total_size} > {max_size}")
                    f.write(chunk)
    
    return {
        "url": url,
        "destination": str(destination),
        "size": total_size,
        "content_type": response.headers.get("content-type"),
    }


async def parse_url(args: dict[str, Any]) -> dict[str, Any]:
    """Parse a URL into its components."""
    url = args["url"]
    parsed = urlparse(url)
    
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": "***" if parsed.password else None,
    }


async def build_url(args: dict[str, Any]) -> str:
    """Build a URL from components."""
    from urllib.parse import urlencode, urlunparse
    
    scheme = args.get("scheme", "https")
    host = args["host"]
    port = args.get("port")
    path = args.get("path", "")
    params = args.get("params", {})
    fragment = args.get("fragment", "")
    
    netloc = host
    if port:
        netloc = f"{host}:{port}"
    
    query = urlencode(params) if params else ""
    
    return urlunparse((scheme, netloc, path, "", query, fragment))


def register_web_tools(registry: ToolRegistry) -> None:
    """Register web tools with the registry."""
    
    # HTTP request
    registry.register_tool(Tool.create(
        name="http_request",
        description="Make an HTTP request to a URL",
        handler=http_request,
        parameters=[
            ToolParameter(
                name="url",
                type=ParameterType.STRING,
                description="URL to request",
                required=True,
            ),
            ToolParameter(
                name="method",
                type=ParameterType.STRING,
                description="HTTP method (GET, POST, PUT, DELETE, etc.)",
                required=False,
                default="GET",
                enum=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
            ToolParameter(
                name="headers",
                type=ParameterType.OBJECT,
                description="Request headers",
                required=False,
            ),
            ToolParameter(
                name="params",
                type=ParameterType.OBJECT,
                description="Query parameters",
                required=False,
            ),
            ToolParameter(
                name="body",
                type=ParameterType.OBJECT,
                description="Request body (will be JSON encoded if object)",
                required=False,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Request timeout in seconds",
                required=False,
                default=30.0,
            ),
            ToolParameter(
                name="follow_redirects",
                type=ParameterType.BOOLEAN,
                description="Whether to follow redirects",
                required=False,
                default=True,
            ),
        ],
        category="web",
        tags=["http", "request", "api"],
    ))
    
    # HTTP GET (convenience)
    registry.register_tool(Tool.create(
        name="http_get",
        description="Make an HTTP GET request",
        handler=http_get,
        parameters=[
            ToolParameter(
                name="url",
                type=ParameterType.STRING,
                description="URL to request",
                required=True,
            ),
            ToolParameter(
                name="headers",
                type=ParameterType.OBJECT,
                description="Request headers",
                required=False,
            ),
            ToolParameter(
                name="params",
                type=ParameterType.OBJECT,
                description="Query parameters",
                required=False,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Request timeout in seconds",
                required=False,
                default=30.0,
            ),
        ],
        category="web",
        tags=["http", "get", "api"],
    ))
    
    # HTTP POST (convenience)
    registry.register_tool(Tool.create(
        name="http_post",
        description="Make an HTTP POST request",
        handler=http_post,
        parameters=[
            ToolParameter(
                name="url",
                type=ParameterType.STRING,
                description="URL to request",
                required=True,
            ),
            ToolParameter(
                name="headers",
                type=ParameterType.OBJECT,
                description="Request headers",
                required=False,
            ),
            ToolParameter(
                name="body",
                type=ParameterType.OBJECT,
                description="Request body",
                required=False,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Request timeout in seconds",
                required=False,
                default=30.0,
            ),
        ],
        category="web",
        tags=["http", "post", "api"],
    ))
    
    # Fetch webpage
    registry.register_tool(Tool.create(
        name="fetch_webpage",
        description="Fetch a webpage and extract text content",
        handler=fetch_webpage,
        parameters=[
            ToolParameter(
                name="url",
                type=ParameterType.STRING,
                description="URL of the webpage",
                required=True,
            ),
            ToolParameter(
                name="include_links",
                type=ParameterType.BOOLEAN,
                description="Whether to extract links",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="max_length",
                type=ParameterType.INTEGER,
                description="Maximum text length to return",
                required=False,
                default=50000,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Request timeout in seconds",
                required=False,
                default=30.0,
            ),
        ],
        category="web",
        tags=["http", "scrape", "webpage"],
    ))
    
    # Download file
    registry.register_tool(Tool.create(
        name="download_file",
        description="Download a file from a URL",
        handler=download_file,
        parameters=[
            ToolParameter(
                name="url",
                type=ParameterType.STRING,
                description="URL of the file to download",
                required=True,
            ),
            ToolParameter(
                name="destination",
                type=ParameterType.STRING,
                description="Local path to save the file",
                required=True,
            ),
            ToolParameter(
                name="overwrite",
                type=ParameterType.BOOLEAN,
                description="Whether to overwrite existing file",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="max_size",
                type=ParameterType.INTEGER,
                description="Maximum file size in bytes",
                required=False,
                default=100 * 1024 * 1024,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Request timeout in seconds",
                required=False,
                default=60.0,
            ),
        ],
        category="web",
        tags=["http", "download", "file"],
        is_dangerous=True,
    ))
    
    # Parse URL
    registry.register_tool(Tool.create(
        name="parse_url",
        description="Parse a URL into its components",
        handler=parse_url,
        parameters=[
            ToolParameter(
                name="url",
                type=ParameterType.STRING,
                description="URL to parse",
                required=True,
            ),
        ],
        category="web",
        tags=["url", "parse"],
    ))
    
    # Build URL
    registry.register_tool(Tool.create(
        name="build_url",
        description="Build a URL from components",
        handler=build_url,
        parameters=[
            ToolParameter(
                name="host",
                type=ParameterType.STRING,
                description="Hostname",
                required=True,
            ),
            ToolParameter(
                name="scheme",
                type=ParameterType.STRING,
                description="URL scheme (http or https)",
                required=False,
                default="https",
                enum=["http", "https"],
            ),
            ToolParameter(
                name="port",
                type=ParameterType.INTEGER,
                description="Port number",
                required=False,
            ),
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="URL path",
                required=False,
                default="",
            ),
            ToolParameter(
                name="params",
                type=ParameterType.OBJECT,
                description="Query parameters",
                required=False,
            ),
            ToolParameter(
                name="fragment",
                type=ParameterType.STRING,
                description="URL fragment",
                required=False,
                default="",
            ),
        ],
        category="web",
        tags=["url", "build"],
    ))
