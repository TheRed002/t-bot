-- Initialize T-Bot Trading System Database
-- This script runs when PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set default search path
ALTER DATABASE tbot_dev SET search_path TO trading, public;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA trading TO tbot;
GRANT ALL PRIVILEGES ON SCHEMA audit TO tbot;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit.logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id VARCHAR(255),
    old_value JSONB,
    new_value JSONB,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB
);

-- Create index for audit logs
CREATE INDEX idx_audit_logs_timestamp ON audit.logs(timestamp DESC);
CREATE INDEX idx_audit_logs_user_id ON audit.logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit.logs(action);
CREATE INDEX idx_audit_logs_entity ON audit.logs(entity_type, entity_id);

-- Create performance indexes
CREATE INDEX idx_audit_logs_metadata ON audit.logs USING gin(metadata);

-- Add comments
COMMENT ON SCHEMA trading IS 'Main schema for trading operations';
COMMENT ON SCHEMA audit IS 'Audit logging schema for compliance';
COMMENT ON TABLE audit.logs IS 'Comprehensive audit trail for all system actions';

-- Initialize with system message
INSERT INTO audit.logs (action, entity_type, metadata) 
VALUES ('SYSTEM_INIT', 'DATABASE', '{"message": "T-Bot database initialized successfully"}');