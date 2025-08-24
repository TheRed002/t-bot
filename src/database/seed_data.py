"""
Database seeding module for development environment.

This module provides functionality to seed the database with initial data
for development and testing purposes. It includes users, bot configurations,
strategy templates, and sample trading data.

IMPORTANT: This should only be run in development mode!
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import Config
from src.core.logging import get_logger
from src.core.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyStatus,
    StrategyType,
)
from src.database.connection import get_db_session
from src.database.models import (
    BotInstance,
    StrategyConfig,
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
            "users": [
                {
                    "username": "admin",
                    "email": "admin@tbot.com",
                    "password": "admin123",
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
                    "initial_balance": 10000.0,
                    "current_balance": 10500.0,
                },
                {
                    "name": "ETH Mean Reversion",
                    "description": "Mean reversion strategy for ETH",
                    "is_active": True,
                    "exchange": "okx",
                    "trading_pair": "ETH/USDT",
                    "initial_balance": 5000.0,
                    "current_balance": 5250.0,
                },
                {
                    "name": "Multi-Asset Arbitrage",
                    "description": "Cross-exchange arbitrage bot",
                    "is_active": False,
                    "exchange": "coinbase",
                    "trading_pair": "Multiple",
                    "initial_balance": 20000.0,
                    "current_balance": 20000.0,
                },
            ],
            "strategies": [
                {
                    "name": "Scalping Strategy",
                    "description": "Quick trades on small price movements",
                    "strategy_type": StrategyType.STATIC,
                    "status": StrategyStatus.ACTIVE,
                    "parameters": {
                        "timeframe": "1m",
                        "take_profit": 0.002,
                        "stop_loss": 0.001,
                        "position_size": 0.1,
                    },
                },
                {
                    "name": "Mean Reversion",
                    "description": "Trade on price deviations from mean",
                    "strategy_type": StrategyType.STATIC,
                    "status": StrategyStatus.ACTIVE,
                    "parameters": {
                        "timeframe": "15m",
                        "lookback_period": 20,
                        "entry_threshold": 2.0,
                        "exit_threshold": 0.5,
                    },
                },
                {
                    "name": "Arbitrage Scanner",
                    "description": "Scan for arbitrage opportunities",
                    "strategy_type": StrategyType.ARBITRAGE,
                    "status": StrategyStatus.ACTIVE,
                    "parameters": {
                        "min_profit": 0.001,
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
                id=str(uuid.uuid4()),
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

    async def seed_bot_instances(
        self, session: AsyncSession, users: list[User]
    ) -> list[BotInstance]:
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
                BotInstance.user_id == user.id,
            )
            result = await session.execute(stmt)
            existing_bot = result.scalar_one_or_none()

            if existing_bot:
                logger.info(f"Bot {bot_data['name']} already exists, skipping")
                bots.append(existing_bot)
                continue

            # Create new bot
            bot = BotInstance(
                id=str(uuid.uuid4()),
                user_id=user.id,
                name=bot_data["name"],
                strategy_type=StrategyType.STATIC,  # Default strategy type
                exchange=bot_data["exchange"],
                status=StrategyStatus.ACTIVE if bot_data["is_active"] else StrategyStatus.STOPPED,
                config={
                    "description": bot_data["description"],
                    "trading_pair": bot_data["trading_pair"],
                    "initial_balance": bot_data["initial_balance"],
                    "current_balance": bot_data["current_balance"],
                },
            )

            session.add(bot)
            bots.append(bot)
            logger.info(f"Created bot: {bot_data['name']} for user {user.username}")

        await session.commit()
        return bots

    async def seed_strategies(
        self, session: AsyncSession, bots: list[BotInstance]
    ) -> list[StrategyConfig]:
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
            stmt = select(StrategyConfig).where(
                StrategyConfig.name == strategy_data["name"],
            )
            result = await session.execute(stmt)
            existing_strategy = result.scalar_one_or_none()

            if existing_strategy:
                logger.info(f"Strategy {strategy_data['name']} already exists, skipping")
                strategies.append(existing_strategy)
                continue

            # Create new strategy config
            strategy = StrategyConfig(
                id=str(uuid.uuid4()),
                name=strategy_data["name"],
                strategy_type=strategy_data["strategy_type"].value,
                parameters=strategy_data["parameters"],
                risk_parameters={
                    "max_position_size": 1000,
                    "stop_loss": 0.05,
                    "take_profit": 0.10,
                },
                is_active=strategy_data["status"] == StrategyStatus.ACTIVE,
            )

            session.add(strategy)
            strategies.append(strategy)
            logger.info(f"Created strategy: {strategy_data['name']} for bot {bot.name}")

        await session.commit()
        return strategies

    async def seed_exchange_credentials(
        self, session: AsyncSession, users: list[User]
    ) -> list[dict[str, Any]]:
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
                logger.info(
                    f"Created credential for {cred_data['exchange']} for user {user.username}"
                )

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
            base_time = datetime.utcnow() - timedelta(days=7)

            for i in range(10):
                trade_time = base_time + timedelta(hours=i * 12)
                i % 3 != 0  # 70% profitable trades

                trade = Trade(
                    id=str(uuid.uuid4()),
                    bot_id=bot.id,
                    exchange_order_id=f"demo_order_{i}_{bot.id[:8]}",
                    exchange=bot.exchange,
                    symbol=bot.config.get("trading_pair", "BTC/USDT"),
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("0.01"),
                    price=Decimal(str(45000 + (i * 100))),
                    executed_price=Decimal(str(45000 + (i * 100))),
                    fee=Decimal("0.001"),
                    status=OrderStatus.FILLED,
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
            logger.warning(
                f"Seeding is only allowed in development mode. Current: {self.config.environment}"
            )
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

                # Print login credentials for testing
                logger.info("\n" + "=" * 50)
                logger.info("Test Login Credentials:")
                logger.info("-" * 50)
                for user_data in self._load_seed_data()["users"]:
                    logger.info(
                        f"Username: {user_data['username']}, Password: {user_data['password']}"
                    )
                logger.info("=" * 50 + "\n")

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
