-- Executive KPIs
SELECT
    COUNT(*) AS total_cases,
    SUM(closed_at IS NOT NULL) AS completed_cases,
    SUM(closed_at IS NULL) AS open_cases,
    AVG(total_duration_hours) AS avg_handling_hours,
    AVG(sla_breached) AS sla_breach_rate,
    AVG(rework_count > 0) AS rework_rate,
    SUM(total_cost) AS total_operating_cost
FROM cases;

-- Monthly operational trend
SELECT
    strftime('%Y-%m', created_at) AS month,
    COUNT(*) AS cases_created,
    AVG(total_duration_hours) AS avg_duration_hours,
    AVG(sla_breached) AS sla_breach_rate,
    AVG(rework_count > 0) AS rework_rate,
    SUM(total_cost) AS operating_cost
FROM cases
GROUP BY month
ORDER BY month;

-- High-risk operational cases
SELECT *
FROM predictions
WHERE sla_breach_probability >= 0.5
ORDER BY sla_breach_probability DESC;

