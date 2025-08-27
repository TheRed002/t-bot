#!/usr/bin/env python3
"""
Secure Database Seeding Script for T-Bot Trading System

This script creates secure default users with proper password hashing
and JWT secret management. It should only be run during initial setup
or when resetting the system.

Security Features:
- Generates cryptographically secure passwords
- Uses bcrypt for password hashing with proper salt rounds
- Generates and stores secure JWT secrets
- Creates audit logs for all user creation
- Validates environment configuration before seeding
"""

import os
import sys
import secrets
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import uuid

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import bcrypt
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install with: pip install bcrypt sqlalchemy asyncpg")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            '/mnt/e/Work/P-41 Trading/code/t-bot/.claude_experiments/seed_database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SecureDatabaseSeeder:
    """Secure database seeding with proper credential management."""

    def __init__(self):
        self.engine = None
        self.session = None
        self.jwt_secret = None
        self.created_users: List[Dict[str, Any]] = []

    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a cryptographically secure password."""
        # Use a mix of characters excluding similar-looking ones
        alphabet = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789!@#$%&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with proper salt rounds."""
        # Use 12 rounds for security vs performance balance
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def generate_jwt_secret(self) -> str:
        """Generate a cryptographically secure JWT secret."""
        # Generate 64 bytes (512 bits) for maximum security
        return secrets.token_urlsafe(64)

    def validate_environment(self) -> bool:
        """Validate that required environment variables are set."""
        required_vars = [
            'DATABASE_URL',
            'ENVIRONMENT'
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        # Warn about production environment
        if os.getenv('ENVIRONMENT') == 'production':
            logger.warning("Running in PRODUCTION mode. Are you sure? (y/N)")
            response = input().strip().lower()
            return response == 'y'

        return True

    def setup_database_connection(self) -> bool:
        """Setup database connection."""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                logger.error("DATABASE_URL environment variable not set")
                return False

            self.engine = create_engine(database_url)
            SessionLocal = sessionmaker(bind=self.engine)
            self.session = SessionLocal()

            # Test connection
            self.session.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to setup database connection: {e}")
            return False

    def create_users_table(self):
        """Create users table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(255) PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name VARCHAR(255),
            is_active BOOLEAN DEFAULT true,
            scopes JSON DEFAULT '["read"]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            failed_login_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP,
            password_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR(255) DEFAULT 'system-seed'
        );
        
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
        """

        try:
            self.session.execute(text(create_table_sql))
            self.session.commit()
            logger.info("Users table created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create users table: {e}")
            raise

    def create_audit_table(self):
        """Create audit table for tracking user operations."""
        create_audit_sql = """
        CREATE TABLE IF NOT EXISTS user_audit (
            audit_id SERIAL PRIMARY KEY,
            user_id VARCHAR(255),
            action VARCHAR(100) NOT NULL,
            details JSON,
            ip_address VARCHAR(45),
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR(255)
        );
        
        CREATE INDEX IF NOT EXISTS idx_user_audit_user_id ON user_audit(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_audit_action ON user_audit(action);
        CREATE INDEX IF NOT EXISTS idx_user_audit_created_at ON user_audit(created_at);
        """

        try:
            self.session.execute(text(create_audit_sql))
            self.session.commit()
            logger.info("User audit table created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create audit table: {e}")
            raise

    def user_exists(self, username: str, email: str) -> bool:
        """Check if user already exists."""
        result = self.session.execute(
            text("SELECT COUNT(*) FROM users WHERE username = :username OR email = :email"),
            {"username": username, "email": email}
        )
        count = result.scalar()
        return count > 0

    def create_user(self, username: str, email: str, password: str,
                    full_name: str, scopes: List[str]) -> Dict[str, Any]:
        """Create a new user with secure password hashing."""
        try:
            # Check if user exists
            if self.user_exists(username, email):
                logger.warning(f"User {username} or email {email} already exists, skipping")
                return None

            # Generate user ID
            user_id = str(uuid.uuid4())

            # Hash password
            password_hash = self.hash_password(password)

            # Create user record
            insert_sql = """
            INSERT INTO users (
                user_id, username, email, password_hash, full_name, 
                scopes, created_at, password_changed_at, created_by
            ) VALUES (
                :user_id, :username, :email, :password_hash, :full_name,
                :scopes, :created_at, :password_changed_at, :created_by
            )
            """

            now = datetime.now(timezone.utc)
            scopes_json = str(scopes).replace("'", '"')  # Convert to JSON format

            self.session.execute(text(insert_sql), {
                "user_id": user_id,
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "full_name": full_name,
                "scopes": scopes_json,
                "created_at": now,
                "password_changed_at": now,
                "created_by": "system-seed-script"
            })

            # Log audit entry
            audit_sql = """
            INSERT INTO user_audit (user_id, action, details, created_by)
            VALUES (:user_id, :action, :details, :created_by)
            """

            audit_details = {
                "username": username,
                "email": email,
                "scopes": scopes,
                "created_by_script": True
            }

            self.session.execute(text(audit_sql), {
                "user_id": user_id,
                "action": "user_created",
                "details": str(audit_details).replace("'", '"'),
                "created_by": "system-seed-script"
            })

            self.session.commit()

            user_info = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "password": password,  # Only for display during setup
                "full_name": full_name,
                "scopes": scopes
            }

            self.created_users.append(user_info)
            logger.info(f"Created user: {username} with scopes: {scopes}")

            return user_info

        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to create user {username}: {e}")
            raise

    def store_jwt_secret(self):
        """Generate and store JWT secret securely."""
        try:
            self.jwt_secret = self.generate_jwt_secret()

            # Create config table if it doesn't exist
            create_config_sql = """
            CREATE TABLE IF NOT EXISTS system_config (
                config_key VARCHAR(255) PRIMARY KEY,
                config_value TEXT NOT NULL,
                is_encrypted BOOLEAN DEFAULT false,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(255)
            );
            """

            self.session.execute(text(create_config_sql))

            # Store JWT secret
            insert_secret_sql = """
            INSERT INTO system_config (config_key, config_value, created_by)
            VALUES ('jwt_secret_key', :secret, 'system-seed-script')
            ON CONFLICT (config_key) DO UPDATE SET
                config_value = :secret,
                updated_at = CURRENT_TIMESTAMP
            """

            self.session.execute(text(insert_secret_sql), {"secret": self.jwt_secret})
            self.session.commit()

            logger.info("JWT secret generated and stored in database")

            # Also write to .env if it doesn't exist
            env_file = Path(__file__).parent.parent / ".env"
            if not env_file.exists():
                with open(env_file, 'w') as f:
                    f.write(f"# Generated JWT secret - DO NOT COMMIT TO VERSION CONTROL\n")
                    f.write(f"JWT_SECRET_KEY={self.jwt_secret}\n")
                    f.write(f"# Generated on {datetime.now().isoformat()}\n")
                logger.info("JWT secret also written to .env")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to store JWT secret: {e}")
            raise

    def create_default_users(self):
        """Create default system users with secure passwords."""
        default_users = [
            {
                "username": "admin",
                "email": "admin@tbot-system.local",
                "full_name": "System Administrator",
                "scopes": ["admin", "read", "write", "trade", "manage"]
            },
            {
                "username": "trader_primary",
                "email": "trader1@tbot-system.local",
                "full_name": "Primary Trader",
                "scopes": ["read", "write", "trade"]
            },
            {
                "username": "trader_secondary",
                "email": "trader2@tbot-system.local",
                "full_name": "Secondary Trader",
                "scopes": ["read", "write", "trade"]
            },
            {
                "username": "analyst",
                "email": "analyst@tbot-system.local",
                "full_name": "Market Analyst",
                "scopes": ["read"]
            },
            {
                "username": "monitor",
                "email": "monitor@tbot-system.local",
                "full_name": "System Monitor",
                "scopes": ["read", "monitor"]
            }
        ]

        logger.info("Creating default users...")

        for user_data in default_users:
            # Generate secure password
            password = self.generate_secure_password(20)

            user_info = self.create_user(
                username=user_data["username"],
                email=user_data["email"],
                password=password,
                full_name=user_data["full_name"],
                scopes=user_data["scopes"]
            )

            if user_info:
                logger.info(f"✓ Created user: {user_info['username']}")

    def print_credentials_summary(self):
        """Print summary of created users and credentials."""
        print("\n" + "="*80)
        print("SECURE CREDENTIALS CREATED - SAVE THESE SECURELY!")
        print("="*80)

        for user in self.created_users:
            print(f"\nUsername: {user['username']}")
            print(f"Email: {user['email']}")
            print(f"Password: {user['password']}")
            print(f"Scopes: {', '.join(user['scopes'])}")
            print("-" * 50)

        print(f"\nJWT Secret Key (first 20 chars): {self.jwt_secret[:20]}...")
        print(f"Full JWT secret stored in database and .env")

        print("\n⚠️  SECURITY REMINDERS:")
        print("1. Change all passwords after first login")
        print("2. Store credentials in a secure password manager")
        print("3. Never commit .env to version control")
        print("4. Enable 2FA for production deployments")
        print("5. Regularly rotate JWT secrets in production")
        print("="*80)

        # Save credentials to secure file
        creds_file = Path(__file__).parent.parent / ".claude_experiments/initial_credentials.txt"
        creds_file.parent.mkdir(exist_ok=True)

        with open(creds_file, 'w') as f:
            f.write("T-Bot Initial Credentials\n")
            f.write("========================\n\n")
            for user in self.created_users:
                f.write(f"Username: {user['username']}\n")
                f.write(f"Email: {user['email']}\n")
                f.write(f"Password: {user['password']}\n")
                f.write(f"Scopes: {', '.join(user['scopes'])}\n")
                f.write("-" * 30 + "\n")
            f.write(f"\nJWT Secret: {self.jwt_secret}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")

        print(f"\n📁 Credentials also saved to: {creds_file}")
        print("⚠️  Delete this file after securely storing credentials!")

    async def seed_database(self):
        """Main seeding process."""
        try:
            logger.info("Starting secure database seeding...")

            # Validate environment
            if not self.validate_environment():
                logger.error("Environment validation failed")
                return False

            # Setup database connection
            if not self.setup_database_connection():
                logger.error("Database setup failed")
                return False

            # Create tables
            self.create_users_table()
            self.create_audit_table()

            # Generate and store JWT secret
            self.store_jwt_secret()

            # Create default users
            self.create_default_users()

            # Print summary
            self.print_credentials_summary()

            logger.info("Database seeding completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Database seeding failed: {e}")
            if self.session:
                self.session.rollback()
            return False
        finally:
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()


def main():
    """Main entry point."""
    print("T-Bot Secure Database Seeding Script")
    print("====================================")

    # Check for force flag
    if "--force" not in sys.argv:
        print("\nThis script will create default users and generate secure credentials.")
        print("Run with --force to skip this confirmation.")
        print("\nContinue? (y/N): ", end="")

        if input().strip().lower() != 'y':
            print("Seeding cancelled.")
            return

    seeder = SecureDatabaseSeeder()
    success = asyncio.run(seeder.seed_database())

    if success:
        print("\n✅ Database seeding completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Database seeding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
