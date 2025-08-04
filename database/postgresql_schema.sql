-- =====================================================
-- VISION EDGE HEALTHCARE - POSTGRESQL SCHEMA
-- Multi-module system: Vision Edge + AI Analyst + AI Consumer
-- =====================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    avatar_url TEXT,
    role VARCHAR(50) DEFAULT 'patient',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Devices table  
CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    device_name VARCHAR(255) NOT NULL,
    device_type VARCHAR(100) NOT NULL,
    mac_address VARCHAR(17) UNIQUE,
    ip_address INET,
    is_active BOOLEAN DEFAULT true,
    last_seen TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- DETECTION & AI PROCESSING
-- =====================================================

-- Detection events (from AI Analyst module)
CREATE TABLE detection_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(255) NOT NULL,
    image_url TEXT NOT NULL,
    status VARCHAR(50) NOT NULL, -- Normal, Warning, Danger
    action VARCHAR(255),
    location JSONB, -- {x: number, y: number, room: string}
    confidence_score DECIMAL(5,4),
    ai_metadata JSONB, -- AI processing details
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Daily activity summary (aggregated by AI Consumer)
CREATE TABLE daily_activity_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_detections INTEGER DEFAULT 0,
    normal_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    danger_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'Normal', -- Overall day status
    activity_score DECIMAL(5,2),
    summary_data JSONB, -- Detailed metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, date)
);

-- AI analysis and insights (from AI Consumer module)
CREATE TABLE ai_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    analysis_period VARCHAR(50) NOT NULL, -- daily, weekly, monthly
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_detection_sessions INTEGER DEFAULT 0,
    progress_compared_to_last_week DECIMAL(5,2),
    ai_summary TEXT,
    recommendations JSONB,
    health_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- REAL-TIME & MONITORING
-- =====================================================

-- Active sessions (for real-time tracking)
CREATE TABLE active_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    device_id UUID REFERENCES devices(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'active',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Real-time alerts
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    message TEXT NOT NULL,
    metadata JSONB,
    is_read BOOLEAN DEFAULT false,
    is_resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System logs for debugging
CREATE TABLE system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module VARCHAR(100) NOT NULL, -- vision_edge, ai_analyst, ai_consumer
    level VARCHAR(20) NOT NULL, -- info, warning, error
    message TEXT NOT NULL,
    context JSONB,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Detection events indexes
CREATE INDEX idx_detection_events_user_created ON detection_events(user_id, created_at DESC);
CREATE INDEX idx_detection_events_session ON detection_events(session_id);
CREATE INDEX idx_detection_events_status ON detection_events(status);
CREATE INDEX idx_detection_events_processed_at ON detection_events(processed_at DESC);

-- Daily activity indexes
CREATE INDEX idx_daily_activity_user_date ON daily_activity_summary(user_id, date DESC);
CREATE INDEX idx_daily_activity_date ON daily_activity_summary(date DESC);

-- AI analysis indexes
CREATE INDEX idx_ai_analysis_user_period ON ai_analysis(user_id, analysis_period, created_at DESC);

-- Active sessions indexes
CREATE INDEX idx_active_sessions_user ON active_sessions(user_id);
CREATE INDEX idx_active_sessions_session_id ON active_sessions(session_id);
CREATE INDEX idx_active_sessions_last_activity ON active_sessions(last_activity DESC);

-- Alerts indexes
CREATE INDEX idx_alerts_user_created ON alerts(user_id, created_at DESC);
CREATE INDEX idx_alerts_unread ON alerts(user_id, is_read) WHERE is_read = false;

-- System logs indexes
CREATE INDEX idx_system_logs_module_created ON system_logs(module, created_at DESC);
CREATE INDEX idx_system_logs_level ON system_logs(level, created_at DESC);

-- =====================================================
-- FUNCTIONS FOR API
-- =====================================================

-- Get real-time detection events
CREATE OR REPLACE FUNCTION get_detection_events(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 50,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    id UUID,
    session_id VARCHAR,
    image_url TEXT,
    status VARCHAR,
    action VARCHAR,
    location JSONB,
    confidence_score DECIMAL,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        de.id,
        de.session_id,
        de.image_url,
        de.status,
        de.action,
        de.location,
        de.confidence_score,
        de.created_at
    FROM detection_events de
    WHERE de.user_id = p_user_id
    ORDER BY de.created_at DESC
    LIMIT p_limit OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

-- Get daily activity log
CREATE OR REPLACE FUNCTION get_daily_activity(
    p_user_id UUID,
    p_start_date DATE,
    p_end_date DATE
) RETURNS TABLE (
    date DATE,
    status VARCHAR,
    total_detections INTEGER,
    activity_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        das.date,
        das.status,
        das.total_detections,
        das.activity_score
    FROM daily_activity_summary das
    WHERE das.user_id = p_user_id
    AND das.date BETWEEN p_start_date AND p_end_date
    ORDER BY das.date DESC;
END;
$$ LANGUAGE plpgsql;

-- Get AI summary
CREATE OR REPLACE FUNCTION get_ai_summary(
    p_user_id UUID,
    p_period_days INTEGER DEFAULT 7
) RETURNS TABLE (
    total_detection_sessions INTEGER,
    progress_compared_to_last_week DECIMAL,
    ai_summary TEXT,
    daily_activity JSONB
) AS $$
DECLARE
    v_end_date DATE := CURRENT_DATE;
    v_start_date DATE := CURRENT_DATE - INTERVAL '1 day' * p_period_days;
    v_analysis ai_analysis%ROWTYPE;
    v_daily_data JSONB;
BEGIN
    -- Get latest AI analysis
    SELECT * INTO v_analysis
    FROM ai_analysis aa
    WHERE aa.user_id = p_user_id
    AND aa.analysis_period = 'weekly'
    ORDER BY aa.created_at DESC
    LIMIT 1;
    
    -- Get daily activity data
    SELECT jsonb_agg(
        jsonb_build_object(
            'date', das.date,
            'status', das.status
        ) ORDER BY das.date
    ) INTO v_daily_data
    FROM daily_activity_summary das
    WHERE das.user_id = p_user_id
    AND das.date BETWEEN v_start_date AND v_end_date;
    
    RETURN QUERY
    SELECT 
        COALESCE(v_analysis.total_detection_sessions, 0),
        COALESCE(v_analysis.progress_compared_to_last_week, 0.0),
        COALESCE(v_analysis.ai_summary, 'No analysis available'),
        COALESCE(v_daily_data, '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Insert detection event (from AI Analyst)
CREATE OR REPLACE FUNCTION insert_detection_event(
    p_user_id UUID,
    p_session_id VARCHAR,
    p_image_url TEXT,
    p_status VARCHAR,
    p_action VARCHAR,
    p_location JSONB,
    p_confidence_score DECIMAL DEFAULT NULL,
    p_ai_metadata JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_event_id UUID;
BEGIN
    INSERT INTO detection_events (
        user_id, session_id, image_url, status, action, 
        location, confidence_score, ai_metadata
    ) VALUES (
        p_user_id, p_session_id, p_image_url, p_status, p_action,
        p_location, p_confidence_score, p_ai_metadata
    ) RETURNING id INTO v_event_id;
    
    -- Update active session
    UPDATE active_sessions 
    SET last_activity = NOW()
    WHERE session_id = p_session_id;
    
    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS FOR AUTO-UPDATES
-- =====================================================

-- Update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =====================================================
-- SAMPLE DATA FOR TESTING
-- =====================================================

-- Insert test user
INSERT INTO users (id, email, name, role) VALUES 
('550e8400-e29b-41d4-a716-446655440000', 'patient@visionedge.com', 'Test Patient', 'patient'),
('550e8400-e29b-41d4-a716-446655440001', 'doctor@visionedge.com', 'Dr. Smith', 'doctor');

-- Insert test device
INSERT INTO devices (user_id, device_name, device_type) VALUES 
('550e8400-e29b-41d4-a716-446655440000', 'Vision Camera 1', 'camera');

-- Insert sample detection events
INSERT INTO detection_events (user_id, session_id, image_url, status, action, location) VALUES 
('550e8400-e29b-41d4-a716-446655440000', 'session_001', '/images/detection_001.jpg', 'Normal', 'walking', '{"x": 100, "y": 200, "room": "living_room"}'),
('550e8400-e29b-41d4-a716-446655440000', 'session_001', '/images/detection_002.jpg', 'Warning', 'sitting_long', '{"x": 150, "y": 220, "room": "living_room"}');

COMMIT;
