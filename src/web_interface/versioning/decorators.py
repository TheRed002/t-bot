"""
Decorators for API versioning in T-Bot Trading System.

This module provides decorators to handle API versioning,
deprecation warnings, and feature flags for endpoints.
"""

import warnings
from collections.abc import Callable
from functools import wraps

from fastapi import HTTPException, Request, status

from .version_manager import get_version_manager


def versioned_endpoint(
    min_version: str,
    max_version: str | None = None,
    features_required: set[str] | None = None,
    deprecated_in: str | None = None,
):
    """
    Decorator to version-control API endpoints.

    Args:
        min_version: Minimum version required for this endpoint
        max_version: Maximum version supported (None for latest)
        features_required: Set of features required for this endpoint
        deprecated_in: Version in which this endpoint was deprecated
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # Try to find request in kwargs
                request = kwargs.get("request")

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found for version checking",
                )

            version_manager = get_version_manager()

            # Get requested version from headers or path
            requested_version = (
                request.headers.get("X-API-Version") or request.path_params.get("version") or None
            )

            try:
                # Resolve the version to use
                resolved_version = version_manager.resolve_version(requested_version)
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

            # Check minimum version requirement
            min_api_version = version_manager.get_version(min_version)
            if not min_api_version or resolved_version < min_api_version:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Minimum API version {min_version} required",
                )

            # Check maximum version requirement
            if max_version:
                max_api_version = version_manager.get_version(max_version)
                if max_api_version and resolved_version > max_api_version:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Maximum API version {max_version} supported",
                    )

            # Check required features
            if features_required:
                for feature in features_required:
                    if not version_manager.check_feature_availability(
                        resolved_version.version, feature
                    ):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Feature '{feature}' not available in version {resolved_version.version}",
                        )

            # Add version info to request state
            request.state.api_version = resolved_version

            # Check for deprecation
            deprecation_info = None
            if deprecated_in:
                deprecated_version = version_manager.get_version(deprecated_in)
                if deprecated_version and resolved_version >= deprecated_version:
                    deprecation_info = {
                        "deprecated": True,
                        "deprecated_in": deprecated_in,
                        "message": f"This endpoint is deprecated as of version {deprecated_in}",
                    }

            # Call the original function
            result = await func(*args, **kwargs)

            # Add version headers to response
            if hasattr(result, "headers"):
                result.headers["X-API-Version"] = resolved_version.version
                if deprecation_info:
                    result.headers["X-API-Deprecated"] = "true"
                    result.headers["X-API-Deprecation-Info"] = deprecation_info["message"]

            # If result is a dict, add version info
            if isinstance(result, dict):
                result["_api_version"] = resolved_version.version
                if deprecation_info:
                    result["_deprecation"] = deprecation_info

            return result

        # Add metadata to the function
        wrapper._versioned = True
        wrapper._min_version = min_version
        wrapper._max_version = max_version
        wrapper._features_required = features_required or set()
        wrapper._deprecated_in = deprecated_in

        return wrapper

    return decorator


def deprecated(version: str, removal_version: str | None = None, message: str | None = None):
    """
    Decorator to mark endpoints as deprecated.

    Args:
        version: Version in which the endpoint was deprecated
        removal_version: Version in which the endpoint will be removed
        message: Custom deprecation message
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Issue deprecation warning
            warning_msg = (
                message or f"Endpoint {func.__name__} is deprecated as of version {version}"
            )
            if removal_version:
                warning_msg += f" and will be removed in version {removal_version}"

            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)

            # Extract request from args/kwargs
            for arg in args:
                if isinstance(arg, Request):
                    break

            # Call the original function
            result = await func(*args, **kwargs)

            # Add deprecation headers if we have a response object
            if hasattr(result, "headers"):
                result.headers["X-API-Deprecated"] = "true"
                result.headers["X-API-Deprecation-Version"] = version
                result.headers["X-API-Deprecation-Message"] = warning_msg
                if removal_version:
                    result.headers["X-API-Removal-Version"] = removal_version

            # If result is a dict, add deprecation info
            if isinstance(result, dict):
                result["_deprecation"] = {
                    "deprecated": True,
                    "version": version,
                    "message": warning_msg,
                    "removal_version": removal_version,
                }

            return result

        # Add metadata to the function
        wrapper._deprecated = True
        wrapper._deprecation_version = version
        wrapper._removal_version = removal_version
        wrapper._deprecation_message = message

        return wrapper

    return decorator


def feature_flag(feature_name: str, default: bool = True):
    """
    Decorator to control endpoint availability based on feature flags.

    Args:
        feature_name: Name of the feature
        default: Default availability if feature flag not found
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            # Check feature availability
            version_manager = get_version_manager()

            # Get the API version being used
            api_version = getattr(request.state, "api_version", None)
            if api_version:
                feature_available = version_manager.check_feature_availability(
                    api_version.version, feature_name
                )

                if not feature_available:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Feature '{feature_name}' is not available in API version {api_version.version}",
                    )

            return await func(*args, **kwargs)

        # Add metadata to the function
        wrapper._feature_flag = feature_name
        wrapper._feature_default = default

        return wrapper

    return decorator


def version_specific(versions: list[str]):
    """
    Decorator to make an endpoint available only in specific versions.

    Args:
        versions: List of API versions where this endpoint is available
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found for version checking",
                )

            # Get the API version being used
            api_version = getattr(request.state, "api_version", None)
            if not api_version or api_version.version not in versions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Endpoint not available in API version {api_version.version if api_version else 'unknown'}",
                )

            return await func(*args, **kwargs)

        # Add metadata to the function
        wrapper._version_specific = versions

        return wrapper

    return decorator
