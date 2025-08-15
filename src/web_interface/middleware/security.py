"""
Security Middleware for T-Bot web interface.

This middleware provides additional security features including CSP,
input validation, and protection against common web vulnerabilities.
"""

import re
from typing import Dict, List, Optional, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware for web application protection.
    
    This middleware provides:
    - Content Security Policy (CSP) headers
    - XSS protection headers
    - Input validation and sanitization
    - Request size limits
    - Suspicious pattern detection
    - Security headers for trading system protection
    """

    def __init__(
        self,
        app,
        enable_csp: bool = True,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        enable_input_validation: bool = True,
    ):
        """
        Initialize security middleware.
        
        Args:
            app: FastAPI application
            enable_csp: Enable Content Security Policy
            max_request_size: Maximum request size in bytes
            enable_input_validation: Enable input validation and sanitization
        """
        super().__init__(app)
        self.enable_csp = enable_csp
        self.max_request_size = max_request_size
        self.enable_input_validation = enable_input_validation
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), "
                "fullscreen=(self), payment=()"
            ),
        }
        
        # Content Security Policy
        if self.enable_csp:
            self.security_headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
                "https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss: https://api.coingecko.com; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        
        # Suspicious patterns for injection detection
        self.sql_injection_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bSELECT\b.*\bFROM\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bUPDATE\b.*\bSET\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bALTER\b.*\bTABLE\b)",
            r"(\';|\";|--|\/\*|\*\/)",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"eval\s*\(",
            r"document\.",
            r"window\.",
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$]",
            r"\.\./",
            r"/etc/passwd",
            r"/bin/",
            r"cmd\.exe",
            r"powershell",
        ]
        
        # Compile patterns for performance
        self.compiled_sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.xss_patterns]
        self.compiled_cmd_patterns = [re.compile(p, re.IGNORECASE) for p in self.command_injection_patterns]
        
        # Rate limit tracking for suspicious activity
        self.suspicious_ips: Dict[str, int] = {}
        self.blocked_ips: Set[str] = set()

    async def dispatch(self, request: Request, call_next):
        """
        Process request through security middleware.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response with security headers
        """
        # Check if IP is blocked
        client_ip = self._get_client_ip(request)
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked request from {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )
        
        # Check request size
        if not await self._validate_request_size(request):
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large"}
            )
        
        # Validate input if enabled
        if self.enable_input_validation:
            validation_result = await self._validate_input(request)
            if not validation_result["valid"]:
                await self._handle_suspicious_activity(client_ip, validation_result["threat_type"])
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Invalid input detected",
                        "details": validation_result["message"]
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    async def _validate_request_size(self, request: Request) -> bool:
        """
        Validate request size against limits.
        
        Args:
            request: HTTP request
            
        Returns:
            True if request size is valid
        """
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    logger.warning(f"Request too large: {size} bytes (max: {self.max_request_size})")
                    return False
            except ValueError:
                logger.warning("Invalid Content-Length header")
                return False
        
        return True

    async def _validate_input(self, request: Request) -> Dict[str, any]:
        """
        Validate input for malicious patterns.
        
        Args:
            request: HTTP request
            
        Returns:
            Validation result with threat information
        """
        # Check URL parameters
        for param, value in request.query_params.items():
            threat = self._detect_threat_in_value(str(value))
            if threat:
                return {
                    "valid": False,
                    "threat_type": threat,
                    "message": f"Suspicious pattern detected in parameter '{param}'"
                }
        
        # Check request body for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    body = await request.body()
                    if body:
                        body_str = body.decode("utf-8", errors="ignore")
                        threat = self._detect_threat_in_value(body_str)
                        if threat:
                            return {
                                "valid": False,
                                "threat_type": threat,
                                "message": "Suspicious pattern detected in request body"
                            }
            except Exception as e:
                logger.warning(f"Error validating request body: {e}")
        
        return {"valid": True}

    def _detect_threat_in_value(self, value: str) -> Optional[str]:
        """
        Detect threats in input value.
        
        Args:
            value: Input value to check
            
        Returns:
            Threat type if detected, None otherwise
        """
        # Check for SQL injection
        for pattern in self.compiled_sql_patterns:
            if pattern.search(value):
                return "sql_injection"
        
        # Check for XSS
        for pattern in self.compiled_xss_patterns:
            if pattern.search(value):
                return "xss"
        
        # Check for command injection
        for pattern in self.compiled_cmd_patterns:
            if pattern.search(value):
                return "command_injection"
        
        # Check for path traversal
        if "../" in value or "..\\" in value:
            return "path_traversal"
        
        # Check for potential trading-specific attacks
        if self._detect_trading_specific_threats(value):
            return "trading_manipulation"
        
        return None

    def _detect_trading_specific_threats(self, value: str) -> bool:
        """
        Detect trading-specific threats.
        
        Args:
            value: Input value to check
            
        Returns:
            True if trading-specific threat detected
        """
        # Check for extremely large or suspicious financial values
        financial_patterns = [
            r"\b\d{20,}\b",  # Very large numbers
            r"-\d+\.\d+e\+\d+",  # Scientific notation (might indicate manipulation)
            r"999999999",  # Repeated 9s
            r"\.0{10,}1",  # Precision manipulation attempts
        ]
        
        for pattern in financial_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        # Check for potential order manipulation strings
        manipulation_strings = [
            "infinity", "inf", "nan", "null", "undefined",
            "max_int", "min_int", "overflow", "underflow"
        ]
        
        value_lower = value.lower()
        for string in manipulation_strings:
            if string in value_lower:
                return True
        
        return False

    async def _handle_suspicious_activity(self, client_ip: str, threat_type: str):
        """
        Handle suspicious activity from client.
        
        Args:
            client_ip: Client IP address
            threat_type: Type of threat detected
        """
        # Track suspicious activity
        self.suspicious_ips[client_ip] = self.suspicious_ips.get(client_ip, 0) + 1
        
        # Block IP after multiple violations
        if self.suspicious_ips[client_ip] >= 5:
            self.blocked_ips.add(client_ip)
            logger.error(f"Blocked IP {client_ip} after {self.suspicious_ips[client_ip]} violations")
        
        # Log the incident
        logger.warning(
            "Suspicious activity detected",
            client_ip=client_ip,
            threat_type=threat_type,
            violation_count=self.suspicious_ips[client_ip]
        )

    def _add_security_headers(self, response: Response):
        """
        Add security headers to response.
        
        Args:
            response: HTTP response
        """
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add HSTS for HTTPS
        if hasattr(response, 'headers'):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

    def get_security_stats(self) -> Dict[str, any]:
        """
        Get security middleware statistics.
        
        Returns:
            Security statistics
        """
        return {
            "enabled_features": {
                "csp": self.enable_csp,
                "input_validation": self.enable_input_validation,
                "request_size_limits": True,
                "suspicious_pattern_detection": True,
                "ip_blocking": True,
            },
            "security_headers_count": len(self.security_headers),
            "pattern_counts": {
                "sql_injection": len(self.sql_injection_patterns),
                "xss": len(self.xss_patterns),
                "command_injection": len(self.command_injection_patterns),
            },
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "max_request_size_mb": self.max_request_size / (1024 * 1024),
        }

    def unblock_ip(self, ip_address: str) -> bool:
        """
        Unblock an IP address.
        
        Args:
            ip_address: IP address to unblock
            
        Returns:
            True if IP was unblocked
        """
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            self.suspicious_ips.pop(ip_address, None)
            logger.info(f"Unblocked IP address: {ip_address}")
            return True
        return False

    def clear_blocked_ips(self):
        """Clear all blocked IP addresses."""
        count = len(self.blocked_ips)
        self.blocked_ips.clear()
        self.suspicious_ips.clear()
        logger.info(f"Cleared {count} blocked IP addresses")


class InputSanitizer:
    """
    Input sanitization utilities for trading data.
    
    This class provides methods to sanitize and validate input data
    specifically for trading system requirements.
    """
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """
        Sanitize trading symbol input.
        
        Args:
            symbol: Trading symbol to sanitize
            
        Returns:
            Sanitized symbol
        """
        if not symbol:
            return ""
        
        # Remove non-alphanumeric characters except common separators
        sanitized = re.sub(r'[^A-Za-z0-9\-_/]', '', symbol.upper())
        
        # Limit length
        return sanitized[:20]

    @staticmethod
    def sanitize_decimal_string(value: str) -> str:
        """
        Sanitize decimal string for financial calculations.
        
        Args:
            value: Decimal string to sanitize
            
        Returns:
            Sanitized decimal string
        """
        if not value:
            return "0"
        
        # Remove all characters except digits, decimal point, and minus sign
        sanitized = re.sub(r'[^0-9.\-]', '', str(value))
        
        # Ensure only one decimal point
        parts = sanitized.split('.')
        if len(parts) > 2:
            sanitized = parts[0] + '.' + ''.join(parts[1:])
        
        # Ensure minus sign is only at the beginning
        if '-' in sanitized:
            sanitized = sanitized.replace('-', '')
            if value.strip().startswith('-'):
                sanitized = '-' + sanitized
        
        # Limit decimal places to prevent precision attacks
        if '.' in sanitized:
            integer_part, decimal_part = sanitized.split('.')
            decimal_part = decimal_part[:18]  # Max 18 decimal places
            sanitized = f"{integer_part}.{decimal_part}"
        
        # Limit total length
        return sanitized[:30]

    @staticmethod
    def validate_exchange_name(exchange: str) -> bool:
        """
        Validate exchange name.
        
        Args:
            exchange: Exchange name to validate
            
        Returns:
            True if valid exchange name
        """
        if not exchange:
            return False
        
        # Check length
        if len(exchange) > 50:
            return False
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r'^[a-zA-Z0-9_\-]+$', exchange):
            return False
        
        return True

    @staticmethod
    def validate_order_side(side: str) -> bool:
        """
        Validate order side.
        
        Args:
            side: Order side to validate
            
        Returns:
            True if valid order side
        """
        valid_sides = {'buy', 'sell', 'BUY', 'SELL'}
        return side in valid_sides