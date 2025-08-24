"""
Versioning middleware for T-Bot Trading System.

This middleware handles API version resolution, compatibility checking,
and adds appropriate headers to responses.
"""

import re
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.base import BaseComponent

from .version_manager import get_version_manager


class VersioningMiddleware(BaseHTTPMiddleware, BaseComponent):
    """Middleware to handle API versioning."""

    def __init__(self, app, version_header: str = "X-API-Version"):
        super(BaseHTTPMiddleware, self).__init__(app)
        BaseComponent.__init__(self)
        self.version_header = version_header
        self.version_manager = get_version_manager()

        # Regex to extract version from path
        self.version_path_regex = re.compile(r"^/api/(v\d+(?:\.\d+)?(?:\.\d+)?)/.*$")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and add versioning information."""

        # Extract version from various sources
        requested_version = self._extract_version(request)

        try:
            # Resolve the version to use
            resolved_version = self.version_manager.resolve_version(requested_version)

            # Store resolved version in request state
            request.state.api_version = resolved_version
            request.state.requested_version = requested_version

            # Check if the version is deprecated
            deprecation_info = self.version_manager.get_deprecation_info(resolved_version.version)
            if deprecation_info:
                request.state.deprecation_info = deprecation_info

        except ValueError as e:
            # Return error for invalid versions
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid API Version",
                    "message": str(e),
                    "supported_versions": [
                        v.version
                        for v in self.version_manager.list_versions(include_deprecated=False)
                    ],
                },
                headers={
                    "X-API-Error": "version-invalid",
                    "X-Supported-Versions": ",".join(
                        v.version
                        for v in self.version_manager.list_versions(include_deprecated=False)
                    ),
                },
            )

        # Process the request
        response = await call_next(request)

        # Add version information to response headers
        self._add_version_headers(response, request)

        return response

    def _extract_version(self, request: Request) -> str:
        """Extract API version from request."""
        # 1. Check X-API-Version header
        version = request.headers.get(self.version_header)
        if version:
            return version.lower() if not version.lower().startswith("v") else version.lower()

        # 2. Check path parameter
        version = request.path_params.get("version")
        if version:
            return version.lower()

        # 3. Check URL path for version pattern
        path_match = self.version_path_regex.match(request.url.path)
        if path_match:
            return path_match.group(1).lower()

        # 4. Check query parameter
        version = request.query_params.get("api_version") or request.query_params.get("version")
        if version:
            return version.lower() if not version.lower().startswith("v") else version.lower()

        # 5. Check Accept header for version
        accept_header = request.headers.get("Accept", "")
        if "application/vnd.tbot" in accept_header:
            # Extract version from Accept header like "application/vnd.tbot.v2+json"
            version_match = re.search(r"application/vnd\.tbot\.(v\d+(?:\.\d+)?)", accept_header)
            if version_match:
                return version_match.group(1).lower()

        # No version specified, will use default
        return None

    def _add_version_headers(self, response: Response, request: Request) -> None:
        """Add version-related headers to the response."""
        # Add API version header
        if hasattr(request.state, "api_version"):
            api_version = request.state.api_version
            response.headers["X-API-Version"] = api_version.version
            response.headers["X-API-Version-Status"] = api_version.status.value

            # Add deprecation headers if applicable
            if hasattr(request.state, "deprecation_info"):
                deprecation_info = request.state.deprecation_info
                response.headers["X-API-Deprecated"] = "true"
                response.headers["X-API-Deprecation-Version"] = deprecation_info["version"]
                if deprecation_info.get("sunset_date"):
                    response.headers["X-API-Sunset-Date"] = deprecation_info["sunset_date"]
                if deprecation_info.get("recommended_version"):
                    response.headers["X-API-Recommended-Version"] = deprecation_info[
                        "recommended_version"
                    ]

        # Add supported versions header
        supported_versions = [
            v.version for v in self.version_manager.list_versions(include_deprecated=False)
        ]
        response.headers["X-Supported-API-Versions"] = ",".join(supported_versions)

        # Add latest version header
        latest_version = self.version_manager.get_latest_version()
        if latest_version:
            response.headers["X-Latest-API-Version"] = latest_version.version

        # Add rate limit information based on version
        if hasattr(request.state, "api_version"):
            api_version = request.state.api_version
            # Different rate limits for different versions
            if api_version.major >= 2:
                response.headers["X-RateLimit-Version"] = "enhanced"
            else:
                response.headers["X-RateLimit-Version"] = "standard"

        # Add feature support information
        if hasattr(request.state, "api_version"):
            api_version = request.state.api_version
            response.headers["X-API-Features"] = ",".join(sorted(api_version.features))

        # CORS headers for version information
        response.headers["Access-Control-Expose-Headers"] = ",".join(
            [
                "X-API-Version",
                "X-API-Version-Status",
                "X-API-Deprecated",
                "X-API-Deprecation-Version",
                "X-API-Sunset-Date",
                "X-API-Recommended-Version",
                "X-Supported-API-Versions",
                "X-Latest-API-Version",
                "X-API-Features",
            ]
        )


class VersionRoutingMiddleware(BaseHTTPMiddleware, BaseComponent):
    """Middleware to handle version-specific routing."""

    def __init__(self, app):
        super(BaseHTTPMiddleware, self).__init__(app)
        BaseComponent.__init__(self)
        self.version_manager = get_version_manager()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Route requests to version-specific handlers."""

        # Check if this is a version-specific path
        original_path = request.url.path

        # Pattern: /api/v1/... -> /api/bots/... with version context
        version_match = re.match(r"^/api/(v\d+(?:\.\d+)?(?:\.\d+)?)/(.+)$", original_path)

        if version_match:
            version = version_match.group(1)
            remaining_path = version_match.group(2)

            # Validate version exists
            if not self.version_manager.get_version(version):
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": "API Version Not Found",
                        "message": f"API version {version} does not exist",
                        "supported_versions": [
                            v.version for v in self.version_manager.list_versions()
                        ],
                    },
                )

            # Rewrite the path to remove version prefix
            # This allows the same endpoints to handle multiple versions
            request.scope["path"] = f"/api/{remaining_path}"
            request.url = request.url.replace(path=f"/api/{remaining_path}")

            # Set version in path parameters
            if "path_params" not in request.scope:
                request.scope["path_params"] = {}
            request.scope["path_params"]["version"] = version

        return await call_next(request)
