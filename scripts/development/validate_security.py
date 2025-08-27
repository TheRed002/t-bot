#!/usr/bin/env python3
"""
Security Validation Script for T-Bot Trading System

This script validates the security configuration of the T-Bot system,
checking for common security issues and misconfigurations.

Usage:
    python scripts/validate_security.py
    python scripts/validate_security.py --fix-permissions
    python scripts/validate_security.py --generate-secrets
"""

import os
import sys
import secrets
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
import argparse

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validates security configuration and settings."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []

    def add_issue(self, severity: str, category: str, description: str, fix: str = None):
        """Add a security issue."""
        issue = {
            "severity": severity,
            "category": category,
            "description": description,
            "fix": fix
        }

        if severity in ["CRITICAL", "HIGH"]:
            self.issues.append(issue)
        else:
            self.warnings.append(issue)

    def check_environment_files(self):
        """Check environment files for security issues."""
        logger.info("Checking environment files...")

        # Check for .env files in git
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()

            if ".env" not in gitignore_content:
                self.add_issue(
                    "HIGH",
                    "Environment Security",
                    ".env not in .gitignore - secrets may be committed",
                    "Add '.env' to .gitignore file"
                )

        # Check for existing .env files with secrets
        env_files = [
            ".env",
            ".env.development",
            ".env.production"
        ]

        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                self._check_env_file_contents(env_path)

    def _check_env_file_contents(self, env_path: Path):
        """Check contents of an environment file."""
        try:
            with open(env_path, 'r') as f:
                content = f.read()

            # Check for default/weak values
            weak_patterns = [
                (r'JWT_SECRET_KEY=.*your.*secret.*', "Weak JWT secret key"),
                (r'JWT_SECRET_KEY=.{1,31}$', "JWT secret key too short (< 32 chars)"),
                (r'password=.*password.*', "Default password detected"),
                (r'password=.*123.*', "Weak password detected"),
                (r'API_KEY=.*your.*api.*', "Default API key placeholder"),
                (r'SECRET=.*change.*this.*', "Default secret placeholder"),
            ]

            for pattern, description in weak_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    self.add_issue(
                        "HIGH",
                        "Weak Credentials",
                        f"{description} in {env_path.name}",
                        "Update with secure, random values"
                    )

            # Check for empty critical values
            critical_vars = ["JWT_SECRET_KEY", "DATABASE_URL"]
            for var in critical_vars:
                if f"{var}=" in content:
                    # Extract value after =
                    match = re.search(f'{var}=(.*)$', content, re.MULTILINE)
                    if match and not match.group(1).strip():
                        self.add_issue(
                            "CRITICAL",
                            "Missing Configuration",
                            f"{var} is empty in {env_path.name}",
                            f"Set a secure value for {var}"
                        )

        except Exception as e:
            logger.error(f"Error checking {env_path}: {e}")

    def check_file_permissions(self):
        """Check file permissions for security issues."""
        logger.info("Checking file permissions...")

        # Check sensitive files
        sensitive_files = [
            ".env",
            ".env.production",
            "scripts/seed_database.py",
            ".claude_experiments/initial_credentials.txt"
        ]

        for file_path in sensitive_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                stat = full_path.stat()
                mode = oct(stat.st_mode)[-3:]

                # Check if file is readable by others
                if int(mode[2]) > 0:
                    self.add_issue(
                        "MEDIUM",
                        "File Permissions",
                        f"{file_path} is readable by others (mode: {mode})",
                        f"chmod 600 {file_path}"
                    )

    def check_hardcoded_secrets(self):
        """Check for hardcoded secrets in code."""
        logger.info("Checking for hardcoded secrets...")

        # Patterns to search for
        secret_patterns = [
            (r'password\s*=\s*["\'].*123.*["\']', "Weak hardcoded password"),
            (r'secret\s*=\s*["\'].*secret.*["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'].*token.*["\']', "Hardcoded token"),
            (r'api_key\s*=\s*["\'].*key.*["\']', "Hardcoded API key"),
            (r'admin123|trader123|viewer123', "Hardcoded test credentials"),
            (r'mock-token', "Mock token in code"),
        ]

        # Files to check
        code_files = list(self.project_root.rglob("*.py"))
        code_files.extend(self.project_root.rglob("*.ts"))
        code_files.extend(self.project_root.rglob("*.tsx"))
        code_files.extend(self.project_root.rglob("*.js"))

        for file_path in code_files:
            # Skip test files and examples
            if any(x in str(file_path) for x in ["test", "spec", "__pycache__", "node_modules", ".git"]):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for pattern, description in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.add_issue(
                            "HIGH",
                            "Hardcoded Secrets",
                            f"{description} in {file_path.relative_to(self.project_root)}",
                            "Replace with environment variables or secure configuration"
                        )
                        break  # Only report one issue per file

            except Exception as e:
                logger.debug(f"Error checking {file_path}: {e}")

    def check_database_security(self):
        """Check database security configuration."""
        logger.info("Checking database security...")

        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Check for passwords in URL
            if '@' in database_url and ':' in database_url:
                # Extract password part
                try:
                    if '://username:password@' in database_url.lower():
                        self.add_issue(
                            "HIGH",
                            "Database Security",
                            "Database URL contains default credentials",
                            "Use strong, unique database credentials"
                        )
                except Exception:
                    pass

            # Check for localhost in production
            environment = os.getenv('ENVIRONMENT', 'development')
            if environment == 'production' and 'localhost' in database_url:
                self.add_issue(
                    "MEDIUM",
                    "Database Security",
                    "Using localhost database in production",
                    "Use a dedicated database server in production"
                )

    def check_jwt_security(self):
        """Check JWT configuration security."""
        logger.info("Checking JWT security...")

        jwt_secret = os.getenv('JWT_SECRET_KEY') or os.getenv('JWT_SECRET')

        if not jwt_secret:
            self.add_issue(
                "CRITICAL",
                "JWT Security",
                "No JWT secret key configured",
                "Set JWT_SECRET_KEY environment variable"
            )
        elif len(jwt_secret) < 32:
            self.add_issue(
                "HIGH",
                "JWT Security",
                f"JWT secret key too short ({len(jwt_secret)} chars, minimum 32)",
                "Generate a longer JWT secret key (64+ characters recommended)"
            )
        elif jwt_secret.lower() in ["your-secret-key", "secret", "jwt-secret"]:
            self.add_issue(
                "CRITICAL",
                "JWT Security",
                "JWT secret key using default/weak value",
                "Generate a strong, random JWT secret key"
            )

    def generate_secure_secrets(self):
        """Generate secure secrets for the application."""
        logger.info("Generating secure secrets...")

        secrets_file = self.project_root / ".claude_experiments/generated_secrets.txt"
        secrets_file.parent.mkdir(exist_ok=True)

        # Generate various secrets
        secrets_data = {
            "JWT_SECRET_KEY": secrets.token_urlsafe(64),
            "DATABASE_PASSWORD": self._generate_password(20),
            "REDIS_PASSWORD": self._generate_password(16),
            "API_SECRET": secrets.token_urlsafe(32),
            "SESSION_SECRET": secrets.token_urlsafe(32),
            "ENCRYPTION_KEY": secrets.token_urlsafe(32)
        }

        with open(secrets_file, 'w') as f:
            f.write("Generated Secure Secrets\n")
            f.write("=" * 50 + "\n\n")
            f.write("IMPORTANT: Use these secrets in your .env file\n")
            f.write("Then delete this file after copying the values!\n\n")

            for key, value in secrets_data.items():
                f.write(f"{key}={value}\n")

            f.write(f"\nGenerated: {datetime.now().isoformat()}\n")

        logger.info(f"Secure secrets generated at: {secrets_file}")
        logger.warning("Remember to delete the secrets file after copying values!")

    def _generate_password(self, length: int = 16) -> str:
        """Generate a secure password."""
        alphabet = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789!@#$%&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def fix_file_permissions(self):
        """Fix file permissions for sensitive files."""
        logger.info("Fixing file permissions...")

        sensitive_files = [
            ".env",
            ".env.production",
            ".claude_experiments/initial_credentials.txt"
        ]

        fixed_count = 0
        for file_path in sensitive_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    os.chmod(full_path, 0o600)  # Owner read/write only
                    logger.info(f"Fixed permissions for {file_path}")
                    fixed_count += 1
                except Exception as e:
                    logger.error(f"Failed to fix permissions for {file_path}: {e}")

        logger.info(f"Fixed permissions for {fixed_count} files")

    def run_validation(self):
        """Run all security validations."""
        logger.info("Starting security validation...")

        self.check_environment_files()
        self.check_file_permissions()
        self.check_hardcoded_secrets()
        self.check_database_security()
        self.check_jwt_security()

        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        """Generate security validation report."""
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)

        # Count by severity
        critical = len([i for i in self.issues if i["severity"] == "CRITICAL"])
        high = len([i for i in self.issues if i["severity"] == "HIGH"])
        medium = len([i for i in self.issues + self.warnings if i["severity"] == "MEDIUM"])

        security_score = max(0, 100 - (critical * 25 + high * 10 + medium * 5))

        report = {
            "security_score": security_score,
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "breakdown": {
                "critical": critical,
                "high": high,
                "medium": medium
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self._get_recommendations()
        }

        return report

    def _get_recommendations(self) -> List[str]:
        """Get security recommendations."""
        recommendations = [
            "Run database seeding script to replace hardcoded users",
            "Generate strong JWT secret keys (64+ characters)",
            "Use environment variables for all sensitive configuration",
            "Enable HTTPS in production with valid SSL certificates",
            "Implement rate limiting and DDoS protection",
            "Set up monitoring and alerting for security events",
            "Regularly rotate API keys and secrets",
            "Use database connection pooling with proper limits",
            "Implement proper backup and disaster recovery procedures",
            "Conduct regular security audits and penetration testing"
        ]

        return recommendations

    def print_report(self, report: Dict[str, Any]):
        """Print security validation report."""
        print("\n" + "="*80)
        print("T-BOT SECURITY VALIDATION REPORT")
        print("="*80)

        score = report["security_score"]
        if score >= 90:
            score_color = "🟢 EXCELLENT"
        elif score >= 75:
            score_color = "🟡 GOOD"
        elif score >= 50:
            score_color = "🟠 FAIR"
        else:
            score_color = "🔴 POOR"

        print(f"\nSecurity Score: {score}/100 {score_color}")
        print(f"Total Issues: {report['total_issues']}")
        print(f"Total Warnings: {report['total_warnings']}")

        breakdown = report["breakdown"]
        if breakdown["critical"] > 0:
            print(f"🚨 Critical Issues: {breakdown['critical']}")
        if breakdown["high"] > 0:
            print(f"⚠️  High Issues: {breakdown['high']}")
        if breakdown["medium"] > 0:
            print(f"ℹ️  Medium Issues: {breakdown['medium']}")

        # Print detailed issues
        if report["issues"]:
            print("\n" + "="*50)
            print("CRITICAL & HIGH SECURITY ISSUES")
            print("="*50)

            for i, issue in enumerate(report["issues"], 1):
                print(f"\n{i}. {issue['severity']} - {issue['category']}")
                print(f"   Issue: {issue['description']}")
                if issue.get('fix'):
                    print(f"   Fix: {issue['fix']}")

        # Print warnings
        if report["warnings"]:
            print("\n" + "="*50)
            print("WARNINGS")
            print("="*50)

            for i, warning in enumerate(report["warnings"], 1):
                print(f"\n{i}. {warning['category']}: {warning['description']}")
                if warning.get('fix'):
                    print(f"   Fix: {warning['fix']}")

        # Print recommendations
        print("\n" + "="*50)
        print("SECURITY RECOMMENDATIONS")
        print("="*50)

        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")

        print("\n" + "="*80)

        if report["total_issues"] == 0:
            print("✅ No critical security issues found!")
        else:
            print(f"❌ Please address {report['total_issues']} security issues before deployment")

        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate T-Bot security configuration")
    parser.add_argument("--fix-permissions", action="store_true",
                        help="Fix file permissions for sensitive files")
    parser.add_argument("--generate-secrets", action="store_true",
                        help="Generate secure secrets for configuration")

    args = parser.parse_args()

    validator = SecurityValidator()

    if args.generate_secrets:
        validator.generate_secure_secrets()
        return

    if args.fix_permissions:
        validator.fix_file_permissions()

    # Run validation and print report
    report = validator.run_validation()
    validator.print_report(report)

    # Exit with non-zero code if critical issues found
    if report["breakdown"]["critical"] > 0:
        sys.exit(1)
    elif report["total_issues"] > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    from datetime import datetime
    main()
