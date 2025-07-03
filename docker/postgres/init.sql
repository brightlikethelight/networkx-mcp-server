-- PostgreSQL initialization script for NetworkX MCP Server
-- Creates necessary schemas, tables, and indexes

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS networkx_mcp;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO networkx_mcp, public;

-- Graph metadata table
CREATE TABLE IF NOT EXISTS graph_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    graph_type VARCHAR(50) DEFAULT 'undirected',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    tags TEXT[],
    properties JSONB DEFAULT '{}',
    CONSTRAINT graph_name_unique UNIQUE(name, created_by)
);

-- Graph data storage (for metadata only, actual graph data in Redis/cache)
CREATE TABLE IF NOT EXISTS graph_data (
    graph_id UUID REFERENCES graph_metadata(id) ON DELETE CASCADE,
    data_type VARCHAR(50) NOT NULL, -- 'nodes', 'edges', 'attributes'
    data_key VARCHAR(255) NOT NULL,
    data_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (graph_id, data_type, data_key)
);

-- Algorithm execution history
CREATE TABLE IF NOT EXISTS algorithm_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id UUID REFERENCES graph_metadata(id) ON DELETE CASCADE,
    algorithm_name VARCHAR(255) NOT NULL,
    parameters JSONB DEFAULT '{}',
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    result_summary JSONB,
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    executed_by VARCHAR(255)
);

-- User sessions and authentication
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    active BOOLEAN DEFAULT true
);

-- Rate limiting tracking
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL,
    limit_type VARCHAR(50) NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    INDEX (identifier, limit_type, window_start, window_end)
);

-- Feature flags storage
CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    enabled BOOLEAN DEFAULT false,
    flag_type VARCHAR(50) DEFAULT 'boolean',
    default_value JSONB DEFAULT 'false',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Feature flag rules
CREATE TABLE IF NOT EXISTS feature_flag_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    flag_id UUID REFERENCES feature_flags(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,
    value JSONB NOT NULL,
    priority INTEGER DEFAULT 0,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Circuit breaker state
CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    name VARCHAR(255) PRIMARY KEY,
    state VARCHAR(20) DEFAULT 'closed', -- closed, open, half_open
    failure_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_failure_time TIMESTAMP WITH TIME ZONE,
    last_success_time TIMESTAMP WITH TIME ZONE,
    last_state_change TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    configuration JSONB DEFAULT '{}'
);

-- Audit schema tables
SET search_path TO audit, public;

-- Audit log for all operations
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(255) NOT NULL,
    operation VARCHAR(10) NOT NULL, -- INSERT, UPDATE, DELETE
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(255),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id UUID,
    ip_address INET,
    user_agent TEXT
);

-- Security events log
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'info', -- info, warning, error, critical
    description TEXT NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    labels JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX (metric_name, recorded_at)
);

-- Reset search path
SET search_path TO networkx_mcp, public;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_graph_metadata_created_by ON graph_metadata(created_by);
CREATE INDEX IF NOT EXISTS idx_graph_metadata_created_at ON graph_metadata(created_at);
CREATE INDEX IF NOT EXISTS idx_graph_metadata_tags ON graph_metadata USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_graph_metadata_properties ON graph_metadata USING GIN(properties);

CREATE INDEX IF NOT EXISTS idx_graph_data_graph_id ON graph_data(graph_id);
CREATE INDEX IF NOT EXISTS idx_graph_data_type ON graph_data(data_type);

CREATE INDEX IF NOT EXISTS idx_algorithm_executions_graph_id ON algorithm_executions(graph_id);
CREATE INDEX IF NOT EXISTS idx_algorithm_executions_algorithm_name ON algorithm_executions(algorithm_name);
CREATE INDEX IF NOT EXISTS idx_algorithm_executions_started_at ON algorithm_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_algorithm_executions_status ON algorithm_executions(status);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier_type ON rate_limits(identifier, limit_type);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_start, window_end);

CREATE INDEX IF NOT EXISTS idx_feature_flags_name ON feature_flags(name);
CREATE INDEX IF NOT EXISTS idx_feature_flag_rules_flag_id ON feature_flag_rules(flag_id);
CREATE INDEX IF NOT EXISTS idx_feature_flag_rules_priority ON feature_flag_rules(priority);

-- Audit schema indexes
SET search_path TO audit, public;

CREATE INDEX IF NOT EXISTS idx_audit_log_table_name ON audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation);
CREATE INDEX IF NOT EXISTS idx_audit_log_changed_at ON audit_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_changed_by ON audit_log(changed_by);

CREATE INDEX IF NOT EXISTS idx_security_events_event_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_created_at ON security_events(created_at);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_time ON performance_metrics(metric_name, recorded_at);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, record_id, new_values, changed_at
        ) VALUES (
            TG_TABLE_NAME, TG_OP, NEW.id, to_jsonb(NEW), CURRENT_TIMESTAMP
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, record_id, old_values, new_values, changed_at
        ) VALUES (
            TG_TABLE_NAME, TG_OP, NEW.id, to_jsonb(OLD), to_jsonb(NEW), CURRENT_TIMESTAMP
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, record_id, old_values, changed_at
        ) VALUES (
            TG_TABLE_NAME, TG_OP, OLD.id, to_jsonb(OLD), CURRENT_TIMESTAMP
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers
SET search_path TO networkx_mcp, public;

CREATE TRIGGER audit_graph_metadata_trigger
    AFTER INSERT OR UPDATE OR DELETE ON graph_metadata
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_algorithm_executions_trigger
    AFTER INSERT OR UPDATE OR DELETE ON algorithm_executions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create updated_at triggers
CREATE TRIGGER update_graph_metadata_updated_at
    BEFORE UPDATE ON graph_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feature_flags_updated_at
    BEFORE UPDATE ON feature_flags
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default feature flags
INSERT INTO feature_flags (name, enabled, flag_type, default_value, description) VALUES
    ('api_v2_enabled', true, 'boolean', 'true', 'Enable API v2 endpoints'),
    ('advanced_algorithms', true, 'boolean', 'true', 'Enable advanced graph algorithms'),
    ('caching_enabled', true, 'boolean', 'true', 'Enable result caching'),
    ('max_graph_nodes', true, 'number', '10000', 'Maximum number of nodes in a graph'),
    ('visualization_engine', true, 'string', '"plotly"', 'Default visualization engine'),
    ('rate_limit_config', true, 'json', '{"requests": 100, "window": 60}', 'Rate limiting configuration')
ON CONFLICT (name) DO NOTHING;

-- Create database roles and permissions
DO $$
BEGIN
    -- Create read-only role
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'networkx_readonly') THEN
        CREATE ROLE networkx_readonly;
    END IF;
    
    -- Create application role
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'networkx_app') THEN
        CREATE ROLE networkx_app;
    END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA networkx_mcp TO networkx_readonly, networkx_app;
GRANT USAGE ON SCHEMA audit TO networkx_readonly, networkx_app;

GRANT SELECT ON ALL TABLES IN SCHEMA networkx_mcp TO networkx_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO networkx_readonly;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA networkx_mcp TO networkx_app;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO networkx_app;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA networkx_mcp TO networkx_app;

-- Create maintenance procedures
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM audit.audit_log 
    WHERE changed_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Set up row-level security (example)
ALTER TABLE graph_metadata ENABLE ROW LEVEL SECURITY;

CREATE POLICY graph_metadata_user_policy ON graph_metadata
    FOR ALL TO networkx_app
    USING (created_by = current_user OR created_by IS NULL);

-- Comment tables for documentation
COMMENT ON TABLE graph_metadata IS 'Stores metadata and properties for graphs';
COMMENT ON TABLE graph_data IS 'Stores additional graph data and attributes';
COMMENT ON TABLE algorithm_executions IS 'Tracks algorithm execution history and performance';
COMMENT ON TABLE user_sessions IS 'Manages user authentication sessions';
COMMENT ON TABLE feature_flags IS 'Runtime feature flag configuration';
COMMENT ON TABLE audit.audit_log IS 'Comprehensive audit trail for all operations';
COMMENT ON TABLE audit.security_events IS 'Security-related events and incidents';

-- Performance optimization settings
SET shared_preload_libraries = 'pg_stat_statements';
SET log_statement = 'mod';
SET log_min_duration_statement = 1000; -- Log slow queries