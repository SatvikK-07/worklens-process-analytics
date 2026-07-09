PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

DROP VIEW IF EXISTS automation_candidate_view;
DROP VIEW IF EXISTS sla_breach_view;
DROP VIEW IF EXISTS team_performance_view;
DROP VIEW IF EXISTS activity_performance_view;
DROP VIEW IF EXISTS case_summary_view;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    team TEXT NOT NULL,
    role TEXT NOT NULL,
    region TEXT NOT NULL,
    experience_level TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS providers (
    provider_id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL,
    region TEXT NOT NULL,
    historical_delay_rate REAL NOT NULL,
    document_error_rate REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    claim_type TEXT NOT NULL,
    priority TEXT NOT NULL,
    region TEXT NOT NULL,
    provider_id TEXT NOT NULL REFERENCES providers(provider_id),
    member_id TEXT NOT NULL,
    diagnosis_group TEXT NOT NULL,
    procedure_group TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP,
    outcome TEXT NOT NULL,
    sla_threshold_hours INTEGER NOT NULL,
    total_duration_hours REAL NOT NULL,
    sla_breached INTEGER NOT NULL CHECK (sla_breached IN (0, 1)),
    total_cost REAL NOT NULL,
    rework_count INTEGER NOT NULL DEFAULT 0,
    anomaly_label INTEGER NOT NULL DEFAULT 0 CHECK (anomaly_label IN (0, 1))
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    activity TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    duration_minutes REAL NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    team TEXT NOT NULL,
    application_used TEXT NOT NULL,
    screen_name TEXT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS automation_candidates (
    activity TEXT PRIMARY KEY,
    frequency INTEGER NOT NULL,
    avg_duration_minutes REAL NOT NULL,
    monthly_manual_hours REAL NOT NULL,
    repetitiveness_score REAL NOT NULL,
    rule_based_score REAL NOT NULL,
    error_rate REAL NOT NULL,
    automation_feasibility REAL NOT NULL,
    estimated_monthly_savings REAL NOT NULL,
    automation_priority_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    case_id TEXT PRIMARY KEY REFERENCES cases(case_id),
    sla_breach_probability REAL,
    predicted_completion_hours REAL,
    anomaly_score REAL,
    risk_level TEXT,
    top_risk_factors TEXT,
    recommended_action TEXT
);

CREATE INDEX IF NOT EXISTS idx_cases_claim_type ON cases(claim_type);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
CREATE INDEX IF NOT EXISTS idx_cases_sla_breached ON cases(sla_breached);
CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at);
CREATE INDEX IF NOT EXISTS idx_events_case_id ON events(case_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_activity ON events(activity);
CREATE INDEX IF NOT EXISTS idx_events_team ON events(team);

CREATE VIEW case_summary_view AS
SELECT
    c.*,
    p.provider_type,
    p.historical_delay_rate,
    p.document_error_rate,
    COUNT(e.event_id) AS event_count,
    COUNT(DISTINCT e.team) AS unique_team_count,
    COUNT(DISTINCT e.user_id) AS handoff_count,
    COALESCE(SUM(e.duration_minutes), 0) AS total_manual_minutes
FROM cases c
JOIN providers p ON p.provider_id = c.provider_id
LEFT JOIN events e ON e.case_id = c.case_id
GROUP BY c.case_id;

CREATE VIEW activity_performance_view AS
WITH sequenced AS (
    SELECT
        e.*,
        LAG(julianday(e.timestamp) + e.duration_minutes / 1440.0)
            OVER (PARTITION BY e.case_id ORDER BY e.timestamp) AS previous_end
    FROM events e
)
SELECT
    activity,
    COUNT(*) AS event_count,
    COUNT(DISTINCT s.case_id) AS total_cases,
    ROUND(AVG(duration_minutes), 2) AS avg_duration_minutes,
    ROUND(AVG(
        MAX(0, (julianday(timestamp) - previous_end) * 24)
    ), 2) AS avg_wait_hours,
    ROUND(SUM(duration_minutes) / 60.0 * 35, 2) AS labor_cost,
    COUNT(DISTINCT CASE WHEN sla_breached = 1 THEN s.case_id END) AS breached_cases
FROM sequenced s
JOIN cases c ON c.case_id = s.case_id
GROUP BY activity;

CREATE VIEW team_performance_view AS
SELECT
    e.team,
    COUNT(DISTINCT e.case_id) AS cases_touched,
    COUNT(e.event_id) AS events_completed,
    ROUND(AVG(e.duration_minutes), 2) AS avg_activity_minutes,
    ROUND(AVG(c.total_duration_hours), 2) AS avg_case_duration_hours,
    ROUND(100.0 * AVG(c.sla_breached), 2) AS sla_breach_rate_pct,
    ROUND(100.0 * AVG(c.rework_count > 0), 2) AS rework_rate_pct
FROM events e
JOIN cases c ON c.case_id = e.case_id
GROUP BY e.team;

CREATE VIEW sla_breach_view AS
SELECT
    claim_type,
    priority,
    region,
    COUNT(*) AS total_cases,
    SUM(sla_breached) AS breached_cases,
    ROUND(100.0 * AVG(sla_breached), 2) AS breach_rate_pct,
    ROUND(AVG(total_duration_hours), 2) AS avg_duration_hours,
    ROUND(SUM(MAX(0, total_duration_hours - sla_threshold_hours)), 2)
        AS total_breach_hours
FROM cases
GROUP BY claim_type, priority, region;

CREATE VIEW automation_candidate_view AS
SELECT
    e.activity,
    COUNT(*) AS six_month_frequency,
    ROUND(COUNT(*) / 6.0, 0) AS monthly_volume,
    ROUND(AVG(e.duration_minutes), 2) AS avg_duration_minutes,
    ROUND(COUNT(*) / 6.0 * AVG(e.duration_minutes) / 60.0, 2)
        AS monthly_manual_hours,
    ROUND(COUNT(*) / 6.0 * AVG(e.duration_minutes) / 60.0 * 35, 2)
        AS gross_monthly_labor_cost
FROM events e
WHERE e.activity NOT IN ('Claim Received', 'Case Closed')
GROUP BY e.activity;
