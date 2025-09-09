"""Tests for error categorization utilities module."""


from src.utils.error_categorization import (
    AUTH_KEYWORDS,
    AUTH_MESSAGE_KEYWORDS,
    AUTH_TOKEN_KEYWORDS,
    CRITICAL_KEYWORDS,
    DATA_VALIDATION_KEYWORDS,
    DATABASE_KEYWORDS,
    DATABASE_MESSAGE_KEYWORDS,
    ERROR_KEYWORDS,
    EXCHANGE_KEYWORDS,
    EXCHANGE_MESSAGE_KEYWORDS,
    FINANCIAL_COMPONENTS,
    NETWORK_KEYWORDS,
    RATE_LIMIT_DETECTION_KEYWORDS,
    RATE_LIMIT_KEYWORDS,
    SENSITIVE_KEY_PATTERNS,
    SYSTEM_KEYWORDS,
    VALIDATION_KEYWORDS,
    VALIDATION_MESSAGE_KEYWORDS,
    WARNING_KEYWORDS,
    categorize_by_keywords,
    categorize_error_from_message,
    categorize_error_from_type_and_message,
    contains_keywords,
    detect_auth_token_error,
    detect_data_validation_error,
    detect_rate_limiting,
    determine_alert_severity_from_message,
    get_error_category_keywords,
    get_fallback_type_keywords,
    is_financial_component,
    is_retryable_error,
    is_sensitive_key,
)


class TestContainsKeywords:
    """Test contains_keywords function."""

    def test_contains_keywords_match(self):
        """Test contains_keywords with matching keywords."""
        assert contains_keywords("This is an authentication error", ["auth", "permission"])
        assert contains_keywords("DATABASE CONNECTION FAILED", ["database", "connection"])
        assert contains_keywords("Rate limit exceeded", ["rate limit", "throttle"])

    def test_contains_keywords_no_match(self):
        """Test contains_keywords with no matching keywords."""
        assert not contains_keywords("Simple message", ["auth", "database"])
        assert not contains_keywords("Nothing special here", ["critical", "fatal"])

    def test_contains_keywords_case_insensitive(self):
        """Test contains_keywords is case insensitive."""
        assert contains_keywords("AUTH ERROR", ["auth"])
        assert contains_keywords("Database Problem", ["database"])
        assert contains_keywords("CRITICAL FAILURE", ["critical"])

    def test_contains_keywords_empty_inputs(self):
        """Test contains_keywords with empty inputs."""
        assert not contains_keywords("", ["auth"])
        assert not contains_keywords("text", [])
        assert not contains_keywords("", [])
        assert not contains_keywords(None, ["auth"])
        assert not contains_keywords("text", None)

    def test_contains_keywords_partial_match(self):
        """Test contains_keywords with partial keyword matches."""
        assert contains_keywords("authentication failed", ["auth"])
        assert contains_keywords("network timeout", ["timeout"])
        assert contains_keywords("validation error occurred", ["validation"])


class TestCategorizeByKeywords:
    """Test categorize_by_keywords function."""

    def test_categorize_by_keywords_match(self):
        """Test categorize_by_keywords with matching categories."""
        keyword_map = {
            "auth": ["auth", "permission"],
            "network": ["connection", "timeout"],
            "database": ["sql", "database"],
        }

        assert categorize_by_keywords("Authentication failed", keyword_map) == "auth"
        assert categorize_by_keywords("Connection timeout", keyword_map) == "network"
        assert categorize_by_keywords("SQL error", keyword_map) == "database"

    def test_categorize_by_keywords_no_match(self):
        """Test categorize_by_keywords with no matches."""
        keyword_map = {"auth": ["auth", "permission"], "network": ["connection", "timeout"]}

        assert categorize_by_keywords("Simple message", keyword_map) is None
        assert categorize_by_keywords("Unknown error", keyword_map) is None

    def test_categorize_by_keywords_empty_text(self):
        """Test categorize_by_keywords with empty text."""
        keyword_map = {"auth": ["auth"]}
        assert categorize_by_keywords("", keyword_map) is None
        assert categorize_by_keywords(None, keyword_map) is None

    def test_categorize_by_keywords_first_match_priority(self):
        """Test categorize_by_keywords returns first matching category."""
        keyword_map = {
            "first": ["error"],
            "second": ["error", "failure"],  # Also contains "error"
        }

        # Should return "first" as it appears first in iteration order
        result = categorize_by_keywords("Error occurred", keyword_map)
        # Note: dict iteration order in Python 3.7+ is insertion order
        assert result == "first"


class TestGetErrorCategoryKeywords:
    """Test get_error_category_keywords function."""

    def test_get_error_category_keywords_structure(self):
        """Test get_error_category_keywords returns correct structure."""
        categories = get_error_category_keywords()

        expected_categories = [
            "authentication",
            "validation",
            "network",
            "database",
            "exchange",
            "system",
        ]

        assert isinstance(categories, dict)
        assert set(categories.keys()) == set(expected_categories)

        # Each category should map to a list of strings
        for category, keywords in categories.items():
            assert isinstance(keywords, list)
            assert all(isinstance(keyword, str) for keyword in keywords)

    def test_get_error_category_keywords_content(self):
        """Test get_error_category_keywords returns expected keywords."""
        categories = get_error_category_keywords()

        assert categories["authentication"] == AUTH_KEYWORDS
        assert categories["validation"] == VALIDATION_KEYWORDS
        assert categories["network"] == NETWORK_KEYWORDS
        assert categories["database"] == DATABASE_KEYWORDS
        assert categories["exchange"] == EXCHANGE_KEYWORDS
        assert categories["system"] == SYSTEM_KEYWORDS


class TestGetFallbackTypeKeywords:
    """Test get_fallback_type_keywords function."""

    def test_get_fallback_type_keywords_structure(self):
        """Test get_fallback_type_keywords returns correct structure."""
        types = get_fallback_type_keywords()

        expected_types = ["list", "dict", "set", "tuple", "str", "int", "bool"]

        assert isinstance(types, dict)
        assert set(types.keys()) == set(expected_types)

        # Each type should map to a list of strings
        for type_name, keywords in types.items():
            assert isinstance(keywords, list)
            assert all(isinstance(keyword, str) for keyword in keywords)

    def test_get_fallback_type_keywords_non_empty(self):
        """Test get_fallback_type_keywords returns non-empty lists."""
        types = get_fallback_type_keywords()

        for type_name, keywords in types.items():
            assert len(keywords) > 0, f"Type {type_name} has empty keyword list"


class TestIsFinancialComponent:
    """Test is_financial_component function."""

    def test_is_financial_component_positive(self):
        """Test is_financial_component with financial components."""
        assert is_financial_component("trading_service")
        assert is_financial_component("ExchangeHandler")
        assert is_financial_component("wallet_manager")
        assert is_financial_component("payment_processor")
        assert is_financial_component("order_book")
        assert is_financial_component("position_tracker")

    def test_is_financial_component_negative(self):
        """Test is_financial_component with non-financial components."""
        assert not is_financial_component("logger_service")
        assert not is_financial_component("config_manager")
        assert not is_financial_component("user_interface")
        assert not is_financial_component("database_connection")
        assert not is_financial_component("simple_component")

    def test_is_financial_component_case_insensitive(self):
        """Test is_financial_component is case insensitive."""
        assert is_financial_component("TRADING_SERVICE")
        assert is_financial_component("Exchange_Handler")
        assert is_financial_component("WALLET")

    def test_is_financial_component_empty_input(self):
        """Test is_financial_component with empty input."""
        assert not is_financial_component("")
        assert not is_financial_component(None)


class TestIsSensitiveKey:
    """Test is_sensitive_key function."""

    def test_is_sensitive_key_positive(self):
        """Test is_sensitive_key with sensitive keys."""
        sensitive_keys = [
            "password",
            "api_key",
            "secret_token",
            "private_key",
            "wallet_seed",
            "auth_credential",
            "jwt_token",
            "api_secret",
        ]

        for key in sensitive_keys:
            assert is_sensitive_key(key), f"Key '{key}' should be detected as sensitive"

    def test_is_sensitive_key_negative(self):
        """Test is_sensitive_key with non-sensitive keys."""
        non_sensitive_keys = [
            "username",
            "email",
            "user_id",
            "timestamp",
            "symbol",
            "price",
            "quantity",
            "order_id",
        ]

        for key in non_sensitive_keys:
            assert not is_sensitive_key(key), f"Key '{key}' should not be detected as sensitive"

    def test_is_sensitive_key_case_insensitive(self):
        """Test is_sensitive_key is case insensitive."""
        assert is_sensitive_key("PASSWORD")
        assert is_sensitive_key("Api_Key")
        assert is_sensitive_key("SECRET")

    def test_is_sensitive_key_partial_match(self):
        """Test is_sensitive_key with partial matches."""
        assert is_sensitive_key("user_password_hash")
        assert is_sensitive_key("exchange_api_key")
        assert is_sensitive_key("wallet_private_data")


class TestCategorizeErrorFromTypeAndMessage:
    """Test categorize_error_from_type_and_message function."""

    def test_categorize_authentication_errors(self):
        """Test categorization of authentication errors."""
        assert (
            categorize_error_from_type_and_message("AuthError", "Login failed") == "authentication"
        )
        # PermissionError contains "permission" which maps to authorization in AUTH_MESSAGE_KEYWORDS
        assert (
            categorize_error_from_type_and_message("PermissionError", "Access denied")
            == "authorization"
        )
        # "unauthorized" is in AUTH_MESSAGE_KEYWORDS, should return authorization
        assert (
            categorize_error_from_type_and_message("Error", "Unauthorized access")
            == "authorization"
        )

    def test_categorize_validation_errors(self):
        """Test categorization of validation errors."""
        assert (
            categorize_error_from_type_and_message("ValidationError", "Invalid data")
            == "validation"
        )
        # ValueError contains "value" which is in VALIDATION_KEYWORDS
        assert categorize_error_from_type_and_message("ValueError", "Bad format") == "validation"
        # "required" is in VALIDATION_MESSAGE_KEYWORDS
        assert (
            categorize_error_from_type_and_message("Error", "Required field missing")
            == "validation"
        )

    def test_categorize_network_errors(self):
        """Test categorization of network errors."""
        assert (
            categorize_error_from_type_and_message("ConnectionError", "Failed to connect")
            == "network"
        )
        assert (
            categorize_error_from_type_and_message("TimeoutError", "Request timed out") == "network"
        )
        assert categorize_error_from_type_and_message("Error", "Network unavailable") == "network"

    def test_categorize_database_errors(self):
        """Test categorization of database errors."""
        assert categorize_error_from_type_and_message("DatabaseError", "Query failed") == "database"
        assert (
            categorize_error_from_type_and_message("OperationalError", "Connection lost")
            == "database"
        )
        assert categorize_error_from_type_and_message("Error", "SQL syntax error") == "database"

    def test_categorize_exchange_errors(self):
        """Test categorization of exchange errors."""
        assert categorize_error_from_type_and_message("ExchangeError", "Order failed") == "exchange"
        assert categorize_error_from_type_and_message("TradingError", "Market closed") == "exchange"
        assert categorize_error_from_type_and_message("Error", "Exchange maintenance") == "exchange"

    def test_categorize_rate_limit_errors(self):
        """Test categorization of rate limit errors."""
        assert (
            categorize_error_from_type_and_message("Error", "Rate limit exceeded") == "rate_limit"
        )
        assert (
            categorize_error_from_type_and_message("HTTPError", "Too many requests") == "rate_limit"
        )
        assert categorize_error_from_type_and_message("Error", "API throttled") == "rate_limit"

    def test_categorize_maintenance_errors(self):
        """Test categorization of maintenance errors."""
        assert (
            categorize_error_from_type_and_message("Error", "Service unavailable") == "maintenance"
        )
        assert (
            categorize_error_from_type_and_message("HTTPError", "Maintenance mode") == "maintenance"
        )

    def test_categorize_system_errors(self):
        """Test categorization of system errors."""
        assert (
            categorize_error_from_type_and_message("SystemError", "Internal failure") == "internal"
        )
        assert categorize_error_from_type_and_message("MemoryError", "Out of memory") == "internal"

    def test_categorize_unknown_errors(self):
        """Test categorization of unknown errors."""
        assert (
            categorize_error_from_type_and_message("UnknownError", "Strange failure") == "unknown"
        )
        assert categorize_error_from_type_and_message("Error", "Undefined problem") == "unknown"


class TestCategorizeErrorFromMessage:
    """Test categorize_error_from_message function."""

    def test_categorize_message_authentication(self):
        """Test categorizing authentication messages."""
        assert categorize_error_from_message("Authentication failed") == "authentication"
        assert categorize_error_from_message("Auth token expired") == "authentication"

    def test_categorize_message_validation(self):
        """Test categorizing validation messages."""
        assert categorize_error_from_message("Validation error occurred") == "validation"
        assert categorize_error_from_message("Invalid input format") == "validation"

    def test_categorize_message_network(self):
        """Test categorizing network messages."""
        assert categorize_error_from_message("Connection timeout") == "network"
        assert categorize_error_from_message("Network unreachable") == "network"

    def test_categorize_message_database(self):
        """Test categorizing database messages."""
        assert categorize_error_from_message("Database connection failed") == "database"
        assert categorize_error_from_message("SQL syntax error") == "database"

    def test_categorize_message_exchange(self):
        """Test categorizing exchange messages."""
        assert categorize_error_from_message("Exchange API error") == "exchange"
        assert categorize_error_from_message("Trading session closed") == "exchange"

    def test_categorize_message_unknown(self):
        """Test categorizing unknown messages."""
        assert categorize_error_from_message("Random error occurred") == "unknown"
        assert categorize_error_from_message("Undefined problem") == "unknown"


class TestDetermineAlertSeverityFromMessage:
    """Test determine_alert_severity_from_message function."""

    def test_determine_severity_critical(self):
        """Test determining critical severity."""
        assert determine_alert_severity_from_message("Critical system failure") == "critical"
        assert determine_alert_severity_from_message("Fatal error occurred") == "critical"
        assert determine_alert_severity_from_message("Security breach detected") == "critical"

    def test_determine_severity_high(self):
        """Test determining high severity."""
        assert determine_alert_severity_from_message("Error processing request") == "high"
        assert determine_alert_severity_from_message("Operation failed") == "high"
        assert determine_alert_severity_from_message("Exception thrown") == "high"

    def test_determine_severity_medium(self):
        """Test determining medium severity."""
        assert determine_alert_severity_from_message("Warning: low memory") == "medium"
        assert determine_alert_severity_from_message("Warn user about issue") == "medium"

    def test_determine_severity_info(self):
        """Test determining info severity."""
        assert determine_alert_severity_from_message("Information logged") == "info"
        assert determine_alert_severity_from_message("Status update") == "info"
        assert determine_alert_severity_from_message("Normal operation") == "info"


class TestIsRetryableError:
    """Test is_retryable_error function."""

    def test_is_retryable_error_retryable(self):
        """Test is_retryable_error with retryable errors."""
        retryable_messages = [
            "Connection timeout",
            "Temporary failure",
            "Service unavailable",
            "Network error",
        ]

        for message in retryable_messages:
            assert is_retryable_error(message), f"Message '{message}' should be retryable"

    def test_is_retryable_error_non_retryable(self):
        """Test is_retryable_error with non-retryable errors."""
        non_retryable_messages = [
            "Authentication failed",
            "Invalid input format",
            "Permission denied",
            "Malformed request",
            "Validation error",
            "Syntax error",
        ]

        for message in non_retryable_messages:
            assert not is_retryable_error(message), f"Message '{message}' should not be retryable"


class TestDetectRateLimiting:
    """Test detect_rate_limiting function."""

    def test_detect_rate_limiting_positive(self):
        """Test detect_rate_limiting with rate limiting messages."""
        rate_limit_messages = [
            "Rate limit exceeded",
            "Too many requests",
            "API throttled",
            "429 error occurred",
            "Rate-limit reached",
            "Request throttled",
        ]

        for message in rate_limit_messages:
            assert detect_rate_limiting(message), (
                f"Message '{message}' should be detected as rate limiting"
            )

    def test_detect_rate_limiting_negative(self):
        """Test detect_rate_limiting with non-rate-limiting messages."""
        normal_messages = [
            "Connection failed",
            "Invalid request",
            "Server error",
            "Authentication required",
        ]

        for message in normal_messages:
            assert not detect_rate_limiting(message), (
                f"Message '{message}' should not be detected as rate limiting"
            )


class TestDetectAuthTokenError:
    """Test detect_auth_token_error function."""

    def test_detect_auth_token_error_positive(self):
        """Test detect_auth_token_error with token-related messages."""
        token_messages = [
            "Token expired",
            "Bearer token invalid",
            "JWT verification failed",
            "Authorization header missing",
        ]

        for message in token_messages:
            assert detect_auth_token_error(message), (
                f"Message '{message}' should be detected as auth token error"
            )

    def test_detect_auth_token_error_negative(self):
        """Test detect_auth_token_error with non-token messages."""
        non_token_messages = [
            "Password incorrect",
            "User not found",
            "Database connection failed",
            "Network timeout",
        ]

        for message in non_token_messages:
            assert not detect_auth_token_error(message), (
                f"Message '{message}' should not be detected as auth token error"
            )


class TestDetectDataValidationError:
    """Test detect_data_validation_error function."""

    def test_detect_data_validation_error_positive(self):
        """Test detect_data_validation_error with validation messages."""
        validation_messages = [
            "Data format invalid",
            "Schema validation failed",
            "Model structure incorrect",
            "Data integrity check failed",
        ]

        for message in validation_messages:
            assert detect_data_validation_error(message), (
                f"Message '{message}' should be detected as data validation error"
            )

    def test_detect_data_validation_error_negative(self):
        """Test detect_data_validation_error with non-validation messages."""
        non_validation_messages = [
            "Network connection lost",
            "Authentication failed",
            "Rate limit exceeded",
            "Server maintenance",
        ]

        for message in non_validation_messages:
            assert not detect_data_validation_error(message), (
                f"Message '{message}' should not be detected as data validation error"
            )


class TestKeywordConstants:
    """Test that keyword constants are properly defined."""

    def test_keyword_constants_not_empty(self):
        """Test that all keyword constant lists are not empty."""
        keyword_lists = [
            CRITICAL_KEYWORDS,
            ERROR_KEYWORDS,
            WARNING_KEYWORDS,
            AUTH_KEYWORDS,
            AUTH_MESSAGE_KEYWORDS,
            VALIDATION_KEYWORDS,
            VALIDATION_MESSAGE_KEYWORDS,
            NETWORK_KEYWORDS,
            DATABASE_KEYWORDS,
            DATABASE_MESSAGE_KEYWORDS,
            EXCHANGE_KEYWORDS,
            EXCHANGE_MESSAGE_KEYWORDS,
            RATE_LIMIT_KEYWORDS,
            SYSTEM_KEYWORDS,
            FINANCIAL_COMPONENTS,
            SENSITIVE_KEY_PATTERNS,
            AUTH_TOKEN_KEYWORDS,
            RATE_LIMIT_DETECTION_KEYWORDS,
            DATA_VALIDATION_KEYWORDS,
        ]

        for keyword_list in keyword_lists:
            assert isinstance(keyword_list, list)
            assert len(keyword_list) > 0
            assert all(isinstance(keyword, str) for keyword in keyword_list)

    def test_keyword_constants_lowercase(self):
        """Test that keywords are in lowercase for consistency."""
        keyword_lists_to_check = [
            CRITICAL_KEYWORDS,
            ERROR_KEYWORDS,
            WARNING_KEYWORDS,
            AUTH_KEYWORDS,
            VALIDATION_KEYWORDS,
            NETWORK_KEYWORDS,
        ]

        for keyword_list in keyword_lists_to_check:
            for keyword in keyword_list:
                assert keyword == keyword.lower(), f"Keyword '{keyword}' should be lowercase"
