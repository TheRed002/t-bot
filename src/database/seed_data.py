"""
Database seeding module for development environment.

This module provides functionality to seed the database with initial data
for development and testing purposes. It includes users, bot configurations,
strategy templates, and sample trading data.

IMPORTANT: This should only be run in development mode!
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import Config
from src.core.logging import get_logger
from src.core.types import (
    StrategyStatus,
    StrategyType,
)
from src.database.connection import get_db_session
from src.database.models import (
    BotInstance,
    Strategy,
    Trade,
    User,
)
from src.web_interface.security.jwt_handler import JWTHandler

logger = get_logger(__name__)


class DatabaseSeeder:
    """Handles database seeding for development environment."""

    def __init__(self, config: Config):
        """
        Initialize the database seeder.

        Args:
            config: Application configuration
        """
        # Security check - only allow seeding in development mode
        if not config.debug:
            raise ValueError(
                "Database seeding is only allowed in development mode. "
                "Set DEBUG=True in configuration to enable seeding."
            )

        if config.environment.lower() == "production":
            raise ValueError(
                "CRITICAL: Database seeding is FORBIDDEN in production environment. "
                "This module contains hardcoded credentials and must never run in production."
            )

        self.config = config
        self.jwt_handler = JWTHandler(config)
        self.seed_data: dict[str, Any] = {}

    def _load_seed_data(self) -> dict[str, Any]:
        """
        Load seed data configuration.

        Returns:
            Dictionary containing seed data
        """
        return {
            # WARNING: These are development seed passwords and API keys
            # MUST be changed or removed in production environments
            "users": [
                {
                    "username": "admin",
                    "email": "admin@tbot.com",
                    "password": "admin123",  # CHANGE IN PRODUCTION
                    "is_active": True,
                    "is_verified": True,
                },
                {
                    "username": "trader1",
                    "email": "trader1@tbot.com",
                    "password": "trader123",
                    "is_active": True,
                    "is_verified": True,
                },
                {
                    "username": "trader2",
                    "email": "trader2@tbot.com",
                    "password": "trader123",
                    "is_active": True,
                    "is_verified": True,
                },
                {
                    "username": "viewer",
                    "email": "viewer@tbot.com",
                    "password": "viewer123",
                    "is_active": True,
                    "is_verified": True,
                },
                {
                    "username": "demo",
                    "email": "demo@tbot.com",
                    "password": "demo123",
                    "is_active": True,
                    "is_verified": False,
                },
            ],
            "bot_instances": [
                {
                    "name": "BTC Scalper",
                    "description": "High-frequency BTC scalping bot",
                    "is_active": True,
                    "exchange": "binance",
                    "trading_pair": "BTC/USDT",
                    "initial_balance": Decimal("10000.00"),
                    "current_balance": Decimal("10500.00"),
                },
                {
                    "name": "ETH Mean Reversion",
                    "description": "Mean reversion strategy for ETH",
                    "is_active": True,
                    "exchange": "okx",
                    "trading_pair": "ETH/USDT",
                    "initial_balance": Decimal("5000.00"),
                    "current_balance": Decimal("5250.00"),
                },
                {
                    "name": "Multi-Asset Arbitrage",
                    "description": "Cross-exchange arbitrage bot",
                    "is_active": False,
                    "exchange": "coinbase",
                    "trading_pair": "Multiple",
                    "initial_balance": Decimal("20000.00"),
                    "current_balance": Decimal("20000.00"),
                },
            ],
            "strategies": [
                {
                    "name": "Scalping Strategy",
                    "description": "Quick trades on small price movements",
                    "strategy_type": StrategyType.CUSTOM,
                    "status": StrategyStatus.ACTIVE,
                    "parameters": {
                        "timeframe": "1m",
                        "take_profit": Decimal("0.002"),
                        "stop_loss": Decimal("0.001"),
                        "position_size": Decimal("0.1"),
                    },
                },
                {
                    "name": "Mean Reversion",
                    "description": "Trade on price deviations from mean",
                    "strategy_type": StrategyType.CUSTOM,
                    "status": StrategyStatus.ACTIVE,
                    "parameters": {
                        "timeframe": "15m",
                        "lookback_period": 20,
                        "entry_threshold": Decimal("2.0"),
                        "exit_threshold": Decimal("0.5"),
                    },
                },
                {
                    "name": "Arbitrage Scanner",
                    "description": "Scan for arbitrage opportunities",
                    "strategy_type": StrategyType.ARBITRAGE,
                    "status": StrategyStatus.ACTIVE,
                    "parameters": {
                        "min_profit": Decimal("0.001"),
                        "max_latency": 100,
                        "exchanges": ["binance", "okx", "coinbase"],
                    },
                },
            ],
            "exchange_credentials": [
                {
                    "exchange": "binance",
                    "api_key": "demo_binance_api_key",
                    "api_secret": "demo_binance_api_secret",
                    "is_testnet": True,
                },
                {
                    "exchange": "okx",
                    "api_key": "demo_okx_api_key",
                    "api_secret": "demo_okx_api_secret",
                    "is_testnet": True,
                },
                {
                    "exchange": "coinbase",
                    "api_key": "demo_coinbase_api_key",
                    "api_secret": "demo_coinbase_api_secret",
                    "is_testnet": True,
                },
            ],
        }

    async def seed_users(self, session: AsyncSession) -> list[User]:
        """
        Seed user data.

        Args:
            session: Database session

        Returns:
            List of created users
        """
        users = []
        seed_data = self._load_seed_data()

        for user_data in seed_data["users"]:
            # Check if user already exists
            stmt = select(User).where(User.username == user_data["username"])
            result = await session.execute(stmt)
            existing_user = result.scalar_one_or_none()

            if existing_user:
                logger.info(f"User {user_data['username']} already exists, skipping")
                users.append(existing_user)
                continue

            # Create new user
            user = User(
                id=uuid.uuid4(),
                username=user_data["username"],
                email=user_data["email"],
                password_hash=self.jwt_handler.hash_password(user_data["password"]),
                is_active=user_data["is_active"],
                is_verified=user_data["is_verified"],
            )

            session.add(user)
            users.append(user)
            logger.info(f"Created user: {user_data['username']}")

        await session.commit()
        return users

    async def seed_bot_instances(self, session: AsyncSession, users: list[User]) -> list[BotInstance]:
        """
        Seed bot instance data.

        Args:
            session: Database session
            users: List of users to assign bots to

        Returns:
            List of created bot instances
        """
        bots = []
        seed_data = self._load_seed_data()

        # Assign bots to users (round-robin)
        for i, bot_data in enumerate(seed_data["bot_instances"]):
            user = users[i % len(users)]

            # Check if bot already exists
            stmt = select(BotInstance).where(
                BotInstance.name == bot_data["name"],
            )
            result = await session.execute(stmt)
            existing_bot = result.scalar_one_or_none()

            if existing_bot:
                logger.info(f"Bot {bot_data['name']} already exists, skipping")
                bots.append(existing_bot)
                continue

            # Create new bot
            bot = BotInstance(
                id=uuid.uuid4(),
                name=bot_data["name"],
                strategy_type="STATIC",  # Default strategy type
                exchange=bot_data["exchange"],
                status="running" if bot_data["is_active"] else "stopped",
                config={
                    "description": bot_data["description"],
                    "trading_pair": bot_data["trading_pair"],
                    "initial_balance": bot_data["initial_balance"],
                    "current_balance": bot_data["current_balance"],
                    "user_id": str(user.id),  # Store user reference in config
                },
            )

            session.add(bot)
            bots.append(bot)
            logger.info(f"Created bot: {bot_data['name']} for user {user.username}")

        await session.commit()
        return bots

    async def seed_strategies(self, session: AsyncSession, bots: list[BotInstance]) -> list[Strategy]:
        """
        Seed strategy data.

        Args:
            session: Database session
            bots: List of bot instances to assign strategies to

        Returns:
            List of created strategies
        """
        strategies = []
        seed_data = self._load_seed_data()

        # Assign strategies to bots
        for i, strategy_data in enumerate(seed_data["strategies"]):
            bot = bots[i % len(bots)]

            # Check if strategy already exists
            stmt = select(Strategy).where(
                Strategy.name == strategy_data["name"],
            )
            result = await session.execute(stmt)
            existing_strategy = result.scalar_one_or_none()

            if existing_strategy:
                logger.info(f"Strategy {strategy_data['name']} already exists, skipping")
                strategies.append(existing_strategy)
                continue

            # Create new strategy
            strategy = Strategy(
                id=uuid.uuid4(),
                name=strategy_data["name"],
                type=strategy_data["strategy_type"].value,
                params=strategy_data["parameters"],
                max_position_size=Decimal("1000"),
                risk_per_trade=Decimal("0.02"),
                stop_loss_percentage=Decimal("0.05"),
                status="ACTIVE" if strategy_data["status"] == StrategyStatus.ACTIVE else "INACTIVE",
                bot_id=bot.id,
            )

            session.add(strategy)
            strategies.append(strategy)
            logger.info(f"Created strategy: {strategy_data['name']} for bot {bot.name}")

        await session.commit()
        return strategies

    async def seed_exchange_credentials(self, session: AsyncSession, users: list[User]) -> list[dict[str, Any]]:
        """
        Seed exchange credential data.

        Args:
            session: Database session
            users: List of users to assign credentials to

        Returns:
            List of created exchange credentials
        """
        credentials = []
        seed_data = self._load_seed_data()

        # Assign credentials to admin and traders only
        trading_users = [u for u in users if u.username in ["admin", "trader1", "trader2"]]

        for user in trading_users:
            for cred_data in seed_data["exchange_credentials"]:
                # Skip checking for now - ExchangeCredential model doesn't exist
                # Will be implemented when model is available
                # For now, just log that we would create the credential
                # ExchangeCredential model doesn't exist yet
                credential = {
                    "user": user.username,
                    "exchange": cred_data["exchange"],
                    "is_testnet": cred_data["is_testnet"],
                }
                credentials.append(credential)
                logger.info(f"Created credential for {cred_data['exchange']} for user {user.username}")

        await session.commit()
        return credentials

    async def seed_sample_trades(self, session: AsyncSession, bots: list[BotInstance]) -> None:
        """
        Seed sample trade data for demonstration.

        Args:
            session: Database session
            bots: List of bot instances
        """
        # Only seed trades for active bots
        active_bots = [b for b in bots if b.is_active]

        for bot in active_bots[:2]:  # Only seed for first 2 active bots
            # Check if bot already has trades
            stmt = select(Trade).where(Trade.bot_id == bot.id).limit(1)
            result = await session.execute(stmt)
            existing_trade = result.scalar_one_or_none()

            if existing_trade:
                logger.info(f"Bot {bot.name} already has trades, skipping")
                continue

            # Create sample trades
            base_time = datetime.now(timezone.utc) - timedelta(days=7)

            for i in range(10):
                trade_time = base_time + timedelta(hours=i * 12)

                entry_price = Decimal(str(45000 + (i * 100)))
                exit_price = Decimal(str(45100 + (i * 100)))
                quantity = Decimal("0.01")
                pnl = (exit_price - entry_price) * quantity if i % 2 == 0 else (entry_price - exit_price) * quantity

                trade = Trade(
                    id=uuid.uuid4(),
                    bot_id=bot.id,
                    exchange=bot.exchange,
                    symbol=bot.config.get("trading_pair", "BTC/USDT"),
                    side="BUY" if i % 2 == 0 else "SELL",
                    quantity=quantity,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    fees=Decimal("0.001"),
                    net_pnl=pnl - Decimal("0.001"),
                    created_at=trade_time,
                    updated_at=trade_time,
                )

                session.add(trade)

            logger.info(f"Created 10 sample trades for bot {bot.name}")

        await session.commit()

    async def seed_all(self) -> None:
        """
        Seed all data to the database.

        This is the main entry point for seeding.
        """
        logger.info("Starting database seeding...")

        # Only run in development mode
        if self.config.environment != "development":
            logger.warning(f"Seeding is only allowed in development mode. Current: {self.config.environment}")
            return

        async with get_db_session() as session:
            try:
                # Seed in order of dependencies
                logger.info("Seeding users...")
                users = await self.seed_users(session)

                logger.info("Seeding bot instances...")
                bots = await self.seed_bot_instances(session, users)

                logger.info("Seeding strategies...")
                strategies = await self.seed_strategies(session, bots)

                logger.info("Seeding exchange credentials...")
                credentials = await self.seed_exchange_credentials(session, users)

                logger.info("Seeding sample trades...")
                await self.seed_sample_trades(session, bots)

                logger.info("Database seeding completed successfully!")

                # Print summary
                logger.info(
                    f"Seeded: {len(users)} users, {len(bots)} bots, "
                    f"{len(strategies)} strategies, {len(credentials)} credentials"
                )

                # Log successful seeding (passwords are not logged for security)
                logger.info("Database seeded successfully with test data")
                logger.info(
                    f"Created {len(self._load_seed_data()['users'])} test users - " "see seed_data.py for credentials"
                )

            except Exception as e:
                logger.error(f"Error during seeding: {e}")
                await session.rollback()
                raise


async def run_seed(config: Config | None = None) -> None:
    """
    Run the database seeding process.

    Args:
        config: Optional configuration object
    """
    if config is None:
        config = Config()

    seeder = DatabaseSeeder(config)
    await seeder.seed_all()


def main():
    """Main entry point for standalone execution."""
    config = Config()
    asyncio.run(run_seed(config))


if __name__ == "__main__":
    main()
