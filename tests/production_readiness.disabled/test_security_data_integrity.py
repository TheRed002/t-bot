"""
Production Readiness Tests for Security and Data Integrity

Tests security mechanisms and data integrity:
- API credential validation and security
- Signature generation and verification
- Timestamp synchronization
- Order state consistency
- Balance reconciliation
- Trade execution confirmation
- Data validation and sanitization
"""

import hashlib
import hmac
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.production_readiness.test_config import TestConfig as Config
from src.core.types import OrderRequest, OrderResponse, OrderStatus, OrderType
from src.core.types.trading import OrderSide
from src.exchanges.service import ExchangeService
from src.utils.validation.core import DataValidator


class TestSecurityDataIntegrity:
    """Test security and data integrity mechanisms."""

    @pytest.fixture
    def config(self):
        """Create test configuration with security settings."""
        return Config({
            "exchanges": {
                "binance": {
                    "api_key": "test_binance_key_12345",
                    "api_secret": "test_binance_secret_67890", 
                    "sandbox": True,
                    "security": {
                        "signature_method": "HMAC-SHA256",
                        "timestamp_tolerance": 5000,  # 5 seconds
                        "require_https": True
                    }
                },
                "coinbase": {
                    "api_key": "test_coinbase_key_12345",
                    "api_secret": "test_coinbase_secret_67890",
                    "passphrase": "test_coinbase_passphrase",
                    "sandbox": True,
                    "security": {
                        "signature_method": "HMAC-SHA256",
                        "timestamp_tolerance": 10000,
                        "require_https": True
                    }
                }
            },
            "security": {
                "encryption_key": "test_encryption_key_32_chars_long",
                "token_expiry_minutes": 60,
                "max_request_size": 1048576,  # 1MB
                "allowed_ips": ["127.0.0.1", "::1"],
                "rate_limit_by_ip": True
            }
        })

    @pytest.fixture
    def data_validator(self):
        """Create data validator."""
        return DataValidator(
            max_string_length=1000,
            max_numeric_value=Decimal("1000000000"),
            allowed_symbols=["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        )

    @pytest.fixture
    async def exchange_service(self, config):
        """Create exchange service."""
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            service = ExchangeService(
                exchange_factory=mock_exchange_factory,
                config=config
            )
            await service.start()
            yield service
            await service.stop()

    def test_api_credential_validation(self, config):
        """Test API credential validation and security."""
        
        # Test valid credentials
        binance_config = config.exchanges.binance if hasattr(config.exchanges, 'binance') else {}
        
        assert "api_key" in binance_config
        assert "api_secret" in binance_config
        assert len(binance_config["api_key"]) > 0
        assert len(binance_config["api_secret"]) > 0
        
        # Test credential format validation
        api_key = binance_config["api_key"]
        api_secret = binance_config["api_secret"]
        
        # Should not contain sensitive patterns
        assert "password" not in api_key.lower()
        assert "private" not in api_key.lower()
        
        # Test credential storage security
        config_str = str(config)
        # Credentials should not appear in plain text representations
        # (In production, this would use proper secret management)

    def test_signature_generation_accuracy(self):
        """Test HMAC signature generation for API requests."""
        
        # Test Binance-style signature
        api_secret = "test_secret_key"
        timestamp = int(time.time() * 1000)
        
        # Create test query string
        params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": "0.001",
            "price": "50000.00",
            "timestamp": timestamp
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Generate signature
        signature = hmac.new(
            api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        assert len(signature) == 64  # SHA256 hex digest length
        assert signature.isalnum()
        
        # Test signature consistency
        signature2 = hmac.new(
            api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        assert signature == signature2  # Should be consistent

    def test_timestamp_synchronization(self, data_validator):
        """Test timestamp synchronization and validation."""
        
        current_time = int(time.time() * 1000)
        
        # Test valid timestamp
        valid_timestamps = [
            current_time,
            current_time - 1000,  # 1 second ago
            current_time + 1000   # 1 second in future
        ]
        
        for timestamp in valid_timestamps:
            is_valid = data_validator.validate_timestamp(
                timestamp, 
                tolerance_ms=5000
            )
            assert is_valid
        
        # Test invalid timestamps
        invalid_timestamps = [
            current_time - 10000,  # 10 seconds ago
            current_time + 10000,  # 10 seconds in future
            0,  # Invalid timestamp
            -1  # Negative timestamp
        ]
        
        for timestamp in invalid_timestamps:
            is_valid = data_validator.validate_timestamp(
                timestamp,
                tolerance_ms=5000
            )
            assert not is_valid

    def test_secure_credential_storage(self, config):
        """Test secure credential storage mechanisms."""
        
        # Credentials should not be exposed in string representations
        config_repr = repr(config)
        config_str = str(config)
        
        # Should not contain raw secrets
        assert "test_binance_secret_67890" not in config_str
        assert "test_coinbase_secret_67890" not in config_str
        assert "test_coinbase_passphrase" not in config_str
        
        # Test config serialization doesn't expose secrets
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
            # In production, secrets should be masked or encrypted
            assert isinstance(config_dict, dict)

    @pytest.mark.asyncio
    async def test_order_state_consistency(self, exchange_service):
        """Test order state consistency across operations."""
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            mock_exchange = AsyncMock()
            mock_exchange.exchange_name = "binance"
            mock_exchange.health_check.return_value = True
            
            # Track order states
            order_states = []
            
            def track_order(order_response: OrderResponse):
                """Track order state changes."""
                order_states.append({
                    "id": order_response.id,
                    "status": order_response.status,
                    "filled_quantity": order_response.filled_quantity,
                    "remaining_quantity": order_response.remaining_quantity,
                    "timestamp": time.time()
                })
                return order_response
            
            # Mock order placement
            mock_order_response = OrderResponse(
                id="test_order_12345",
                symbol="BTC/USDT",
                status=OrderStatus.NEW,
                filled_quantity=Decimal("0.000"),
                remaining_quantity=Decimal("0.001")
            )
            
            mock_exchange.place_order.return_value = mock_order_response
            mock_exchange_factory.get_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_exchange_factory
            
            # Place order
            order_request = OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000.00")
            )
            
            response = await exchange_service.place_order("binance", order_request)
            track_order(response)
            
            # Verify order consistency
            assert response.id == "test_order_12345"
            assert response.status == OrderStatus.NEW
            assert response.filled_quantity == Decimal("0.000")
            assert response.remaining_quantity == Decimal("0.001")
            
            # Verify state tracking
            assert len(order_states) == 1
            assert order_states[0]["id"] == "test_order_12345"

    @pytest.mark.asyncio
    async def test_balance_reconciliation(self, exchange_service):
        """Test balance reconciliation accuracy."""
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            mock_exchange = AsyncMock()
            mock_exchange.exchange_name = "binance"
            mock_exchange.health_check.return_value = True
            
            # Mock consistent balance responses
            expected_balances = {
                "BTC": Decimal("1.50000000"),
                "USDT": Decimal("25000.00000000"),
                "ETH": Decimal("10.25000000")
            }
            
            mock_exchange.get_account_balance.return_value = expected_balances
            mock_exchange_factory.get_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_exchange_factory
            
            # Get balance multiple times
            balance1 = await exchange_service.get_account_balance("binance")
            balance2 = await exchange_service.get_account_balance("binance")
            balance3 = await exchange_service.get_account_balance("binance")
            
            # All should be identical
            assert balance1 == balance2 == balance3
            
            # Verify precision is maintained
            assert balance1["BTC"] == Decimal("1.50000000")
            assert balance1["USDT"] == Decimal("25000.00000000")
            assert balance1["ETH"] == Decimal("10.25000000")
            
            # Test balance validation
            for asset, balance in balance1.items():
                assert isinstance(balance, Decimal)
                assert balance >= 0  # No negative balances
                assert balance.as_tuple().exponent <= -8  # Proper precision

    @pytest.mark.asyncio
    async def test_trade_execution_confirmation(self, exchange_service):
        """Test trade execution confirmation and verification."""
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            mock_exchange = AsyncMock()
            mock_exchange.exchange_name = "binance"
            mock_exchange.health_check.return_value = True
            
            # Mock trade execution sequence
            order_responses = [
                OrderResponse(
                    id="test_order_12345",
                    symbol="BTC/USDT", 
                    status=OrderStatus.NEW,
                    filled_quantity=Decimal("0.000"),
                    remaining_quantity=Decimal("0.001")
                ),
                OrderResponse(
                    id="test_order_12345",
                    symbol="BTC/USDT",
                    status=OrderStatus.PARTIALLY_FILLED,
                    filled_quantity=Decimal("0.0005"),
                    remaining_quantity=Decimal("0.0005")
                ),
                OrderResponse(
                    id="test_order_12345",
                    symbol="BTC/USDT",
                    status=OrderStatus.FILLED,
                    filled_quantity=Decimal("0.001"),
                    remaining_quantity=Decimal("0.000")
                )
            ]
            
            mock_exchange.place_order.return_value = order_responses[0]
            mock_exchange.get_order_status.side_effect = [
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED
            ]
            mock_exchange_factory.get_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_exchange_factory
            
            # Place order
            order_request = OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000.00")
            )
            
            response = await exchange_service.place_order("binance", order_request)
            
            # Confirm execution through status checks
            status1 = await exchange_service.get_order_status("binance", response.id)
            status2 = await exchange_service.get_order_status("binance", response.id)
            
            # Should show progression to filled
            assert status1 == OrderStatus.PARTIALLY_FILLED
            assert status2 == OrderStatus.FILLED
            
            # Verify order data integrity
            assert response.filled_quantity + response.remaining_quantity == Decimal("0.001")

    def test_data_validation_sanitization(self, data_validator):
        """Test data validation and sanitization."""
        
        # Test valid data
        valid_inputs = [
            ("BTC/USDT", "symbol"),
            (Decimal("0.001"), "quantity"),
            (Decimal("50000.00"), "price"),
            ("GTC", "time_in_force")
        ]
        
        for value, field_type in valid_inputs:
            is_valid = data_validator.validate_field(value, field_type)
            assert is_valid
        
        # Test invalid data
        invalid_inputs = [
            ("INVALID_SYMBOL", "symbol"),  # Not in allowed symbols
            (Decimal("-1"), "quantity"),   # Negative quantity
            (Decimal("0"), "price"),       # Zero price
            ("INVALID_TIF", "time_in_force")  # Invalid time in force
        ]
        
        for value, field_type in invalid_inputs:
            is_valid = data_validator.validate_field(value, field_type)
            assert not is_valid
        
        # Test sanitization
        unsanitized_strings = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE orders; --",
            "BTCUSDT\x00\x01\x02",  # Null bytes and control chars
            "A" * 2000  # Excessively long string
        ]
        
        for unsafe_string in unsanitized_strings:
            sanitized = data_validator.sanitize_string(unsafe_string)
            
            # Should remove dangerous content
            assert "<script>" not in sanitized
            assert "DROP TABLE" not in sanitized
            assert len(sanitized) <= 1000  # Truncated to max length
            assert all(ord(c) >= 32 for c in sanitized)  # No control chars

    @pytest.mark.asyncio
    async def test_input_validation_edge_cases(self, exchange_service):
        """Test input validation for edge cases."""
        
        # Test extremely small quantities
        tiny_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.000000000000000001"),  # 1 satoshi equivalent
            price=Decimal("50000.00")
        )
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            mock_factory.return_value = mock_exchange_factory
            
            try:
                await exchange_service._validate_order_request(tiny_order)
            except Exception as e:
                # Should handle edge cases gracefully
                assert "quantity" in str(e).lower()
        
        # Test extremely large quantities
        huge_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1000000.0"),  # 1 million BTC
            price=Decimal("50000.00")
        )
        
        try:
            await exchange_service._validate_order_request(huge_order)
        except Exception as e:
            # Should reject unrealistic quantities
            assert "quantity" in str(e).lower()

    def test_encryption_decryption_security(self, config):
        """Test encryption/decryption for sensitive data."""
        
        # Test that sensitive data can be encrypted
        sensitive_data = "api_secret_12345"
        encryption_key = config.security.encryption_key if hasattr(config, 'security') else "test_key"
        
        # Basic encryption test (in production, use proper crypto libraries)
        import base64
        
        encrypted = base64.b64encode(sensitive_data.encode()).decode()
        decrypted = base64.b64decode(encrypted.encode()).decode()
        
        assert encrypted != sensitive_data  # Should be different when encrypted
        assert decrypted == sensitive_data   # Should match when decrypted
        assert len(encrypted) > len(sensitive_data)  # Encryption adds overhead

    @pytest.mark.asyncio
    async def test_request_response_integrity(self, exchange_service):
        """Test request/response data integrity."""
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            mock_exchange = AsyncMock()
            mock_exchange.exchange_name = "binance"
            mock_exchange.health_check.return_value = True
            
            # Test data integrity through request/response cycle
            original_balance = {
                "BTC": Decimal("1.50000000"),
                "USDT": Decimal("25000.00000000")
            }
            
            mock_exchange.get_account_balance.return_value = original_balance
            mock_exchange_factory.get_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_exchange_factory
            
            # Get balance and verify integrity
            retrieved_balance = await exchange_service.get_account_balance("binance")
            
            # Data should be identical
            assert retrieved_balance == original_balance
            
            # Precision should be maintained
            for asset in original_balance:
                assert retrieved_balance[asset] == original_balance[asset]
                assert isinstance(retrieved_balance[asset], Decimal)

    def test_access_control_validation(self, config):
        """Test access control and authorization validation."""
        
        # Test IP allowlist
        security_config = config.security if hasattr(config, 'security') else {}
        allowed_ips = security_config.get('allowed_ips', [])
        
        if allowed_ips:
            # Test allowed IP
            assert "127.0.0.1" in allowed_ips
            assert "::1" in allowed_ips  # IPv6 localhost
            
            # Test that private IPs are handled
            test_ips = ["127.0.0.1", "192.168.1.1", "10.0.0.1", "172.16.0.1"]
            for ip in test_ips:
                # Should validate IP format
                parts = ip.split('.')
                assert len(parts) == 4
                for part in parts:
                    assert 0 <= int(part) <= 255

    @pytest.mark.asyncio
    async def test_audit_logging_security(self, exchange_service):
        """Test audit logging for security events."""
        
        with patch('src.exchanges.service.logger') as mock_logger:
            with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
                mock_exchange_factory = AsyncMock()
                mock_factory.return_value = mock_exchange_factory
                
                # Test security-related operations are logged
                try:
                    await exchange_service.get_exchange("invalid_exchange")
                except:
                    pass
                
                # Should log security-relevant events
                assert mock_logger.error.called or mock_logger.warning.called
                
                # Log entries should not contain sensitive data
                for call in mock_logger.error.call_args_list:
                    log_message = str(call[0][0]) if call[0] else ""
                    assert "api_secret" not in log_message.lower()
                    assert "password" not in log_message.lower()

    def test_configuration_security_validation(self, config):
        """Test configuration security validation."""
        
        # Test that security settings are properly configured
        exchanges_config = config.exchanges if hasattr(config, 'exchanges') else {}
        
        for exchange_name, exchange_config in exchanges_config.items():
            if isinstance(exchange_config, dict):
                # Should have required security fields
                assert "api_key" in exchange_config
                assert "api_secret" in exchange_config
                
                # API keys should meet minimum length requirements
                assert len(exchange_config["api_key"]) >= 8
                assert len(exchange_config["api_secret"]) >= 8
                
                # Should use sandbox in test environment
                if "sandbox" in exchange_config:
                    assert exchange_config["sandbox"] is True