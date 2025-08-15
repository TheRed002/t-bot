#!/bin/bash
# Database backup script for T-Bot Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}💾 T-Bot Database Backup Script${NC}"

# Configuration
BACKUP_DIR="./backups/database"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="tbot_backup_${TIMESTAMP}.sql"

# Check if backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo -e "${YELLOW}📁 Creating backup directory...${NC}"
    mkdir -p "$BACKUP_DIR"
fi

# Determine which compose file to use
COMPOSE_FILE="docker-compose.yml"
if [ "$1" = "--prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    echo -e "${BLUE}🏭 Using production configuration${NC}"
else
    echo -e "${BLUE}🔧 Using development configuration${NC}"
fi

# Check if PostgreSQL container is running
if ! docker-compose -f "$COMPOSE_FILE" ps postgresql | grep -q "Up"; then
    echo -e "${RED}❌ PostgreSQL container is not running${NC}"
    exit 1
fi

echo -e "${YELLOW}💾 Creating database backup...${NC}"

# Create PostgreSQL backup
if docker-compose -f "$COMPOSE_FILE" exec -T postgresql pg_dumpall -c -U tbot > "${BACKUP_DIR}/${BACKUP_FILE}"; then
    echo -e "${GREEN}✅ Database backup created: ${BACKUP_DIR}/${BACKUP_FILE}${NC}"
else
    echo -e "${RED}❌ Database backup failed${NC}"
    exit 1
fi

# Compress the backup
echo -e "${YELLOW}🗜️  Compressing backup...${NC}"
gzip "${BACKUP_DIR}/${BACKUP_FILE}"
COMPRESSED_FILE="${BACKUP_FILE}.gz"

echo -e "${GREEN}✅ Compressed backup created: ${BACKUP_DIR}/${COMPRESSED_FILE}${NC}"

# Show backup size
BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${COMPRESSED_FILE}" | cut -f1)
echo -e "${BLUE}📊 Backup size: ${BACKUP_SIZE}${NC}"

# Clean up old backups (keep last 7 days)
echo -e "${YELLOW}🧹 Cleaning up old backups...${NC}"
find "$BACKUP_DIR" -name "tbot_backup_*.sql.gz" -mtime +7 -delete
REMAINING_BACKUPS=$(ls -1 "${BACKUP_DIR}"/tbot_backup_*.sql.gz | wc -l)
echo -e "${GREEN}✅ Cleanup complete. ${REMAINING_BACKUPS} backup(s) remaining${NC}"

# Optional: Upload to cloud storage (uncomment and configure as needed)
# if [ -n "$AWS_S3_BUCKET" ]; then
#     echo -e "${YELLOW}☁️  Uploading to S3...${NC}"
#     aws s3 cp "${BACKUP_DIR}/${COMPRESSED_FILE}" "s3://${AWS_S3_BUCKET}/tbot-backups/"
#     echo -e "${GREEN}✅ Backup uploaded to S3${NC}"
# fi

echo -e "${GREEN}🎉 Database backup completed successfully!${NC}"