---
date: '2026-02-05'
description: and first assignment.
id: srs
modified: 2026-02-05 06:16:55 GMT-05:00
tags:
  - sfwr3a04
title: srs
---

## 1 Introduction

This document presents the Software Requirements Specification (SRS) for the Smart City Environmental Monitoring & Alert System (SCEMAS). It describes the functional and non-functional requirements for a cloud-native IoT software platform that collects, processes, and acts upon environmental sensor data to support city operations and public awareness.

### 1.1 Purpose

The purpose is to define the software requirements for SCEMAS in sufficient detail to guide architectural design, implementation, and verification. This document serves as a contractual reference between the development team and the course instructor for evaluating project deliverables. The intended audience includes SCEMAS stakeholders, including project managers, developers, domain experts, and SCEMAS team members or investors. No prior readings are required.

### 1.2 Scope

The Smart City Environmental Monitoring & Alert System (SCEMAS) is a cloud-native IoT software platform that continuously collects environmental sensor telemetry, processes and validates incoming data, and produces actionable intelligence for city operators and the general public. The platform ingests data streams from a distributed network of simulated sensors reporting air quality, noise levels, temperature, and humidity measurements across predefined geographical zones within a city.

The system includes four core functionalities:

1. Telemetry Ingestion and Processing: Sensors publish environmental readings via the MQTT protocol. The system validates each incoming message for schema conformance and plausible value ranges, persists validated data in a time-series database, and computes real-time aggregations (e.g., 5-minute averages, hourly maximums) per geographical zone.
2. Rule-Based Alerting: System administrators define threshold and anomaly-based alert rules. The alerting engine evaluates incoming telemetry against active rules in near real-time, triggers alerts upon detecting violations, logs each event with full context, and notifies subscribed external systems through a dedicated API endpoint. A complete alert history with status tracking (active, acknowledged, resolved) is maintained and queryable.
3. Data Access and Visualization: City operators access a secure, role-authenticated dashboard displaying geographical maps of sensor locations, charts of environmental metrics, gauge indicators for current readings, system health status, and active alerts. For public and third-party consumption, the system exposes a read-only REST API serving aggregated, non-sensitive environmental data such as the current Air Quality Index for a given zone. A functional client simulating a public digital signage display demonstrates consumption of this API.
4. Security and Administration: All entities interacting with the system (IoT devices and human users) must complete strong authentication before any data exchange or access is permitted. A role-based access control model distinguishes between standard operators and system administrators. An audit log records all significant events, including device lifecycle changes, alert triggers, and user management actions.

The primary objective of SCEMAS is to replace sparse, manual, or siloed environmental monitoring with an integrated, real-time platform that enables faster detection of and response to pollution events, heatwaves, noise disturbances, and other environmental anomalies. The system aims to reduce operator response latency by surfacing threshold violations as structured alerts rather than requiring manual inspection of raw data streams. A secondary objective is public transparency, where aggregated environmental data is made available through the public API so that citizens and third-party developers can build on top of the platform's data.

One innovative feature includes personalized alert subscriptions for city operators. This feature allows individual operators to subscribe to specific environmental metrics, geographical zones, or alert severity levels, filtering dashboard notifications to show only the alerts relevant to their responsibilities. The goal is to reduce alert fatigue and improve per-operator response efficiency in multi-zone deployments.

Physical sensor hardware is simulated for development and testing purposes. The project focuses on architectural design and core implementation of backend services, data pipelines, and APIs. Production-grade cloud deployment, integration with real municipal data systems or third-party weather services, and mobile application development are outside the scope of this project.

### 1.3 Definitions, Acronyms, and Abbreviations

| Term                     | Definition                                                                                                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _Alert Fatigue_          | A condition in which operators become desensitized to frequent alerts, leading to slower response times or missed critical events.                                                                    |
| _Anomalous Trend_        | A pattern in sensor data that deviates significantly from expected behavior over time, potentially indicating a developing environmental event.                                                       |
| _AQI_                    | Air Quality Index, a standardized numerical scale for reporting air quality, calculated from pollutant concentrations.                                                                                |
| _At-Least-Once Delivery_ | A messaging guarantee in which every message is delivered to the consumer at least one time, though duplicates may occur.                                                                             |
| _Audit Log_              | A chronological record of significant system events (e.g., authentication attempts, alert triggers, configuration changes) used for security monitoring and compliance.                               |
| _Cloud-Native_           | Software designed to run on cloud infrastructure, leveraging containerization, microservices, and scalable resource management.                                                                       |
| _Digital Signage_        | A public-facing electronic display used to present information; in this project, simulated as a functional client application.                                                                        |
| _Geographical Zone_      | A predefined administrative region within the city used for aggregating and reporting environmental data. Zones do not resolve to precise locations, in accordance with privacy-by-design principles. |
| _Horizontal Scaling_     | A scaling strategy in which additional instances of a service are deployed to distribute load, as opposed to increasing the resources of a single instance (vertical scaling).                        |
| _IoT_                    | Internet of Things — a network of interconnected sensors and devices that collect and exchange data.                                                                                                  |
| _JSON_                   | JavaScript Object Notation, a lightweight data interchange format used for sensor telemetry messages.                                                                                                 |
| _Kafka_                  | A distributed event streaming platform suitable for high-volume telemetry ingestion pipelines.                                                                                                        |
| _MQTT_                   | Message Queuing Telemetry Transport, a lightweight publish-subscribe messaging protocol commonly used in IoT systems.                                                                                 |
| _MQTTs_                  | MQTT over TLS, an encrypted variant of the MQTT protocol for secure data transmission.                                                                                                                |
| _p95_                    | The 95th percentile of a latency distribution; 95% of requests complete within the stated time.                                                                                                       |
| _PII_                    | Personally Identifiable Information, any data that could be used to identify a specific individual. SCEMAS shall not collect, process, or store PII.                                                  |
| _PM₂.₅_                  | Particulate Matter with diameter less than 2.5 micrometers, a key air quality indicator measured in $\mu g / m^{3}$                                                                                   |
| _Privacy-by-Design_      | A systems engineering approach in which privacy protections are embedded into the architecture from the outset, rather than added retroactively.                                                      |
| _Rate Limiting_          | A mechanism that restricts the number of API requests a client can make within a given time window, used to prevent abuse and ensure fair access.                                                     |
| _RBAC_                   | Role-Based Access Control, an authorization model that assigns permissions based on user roles (e.g., operator, administrator).                                                                       |
| _REST API_               | Representational State Transfer Application Programming Interface, a standardized architectural style for web service communication.                                                                  |
| _Telemetry_              | Automated measurement data transmitted from remote sensors to the system for processing and storage.                                                                                                  |
| _Threshold Violation_    | An event in which a sensor reading exceeds or falls below a predefined acceptable range, triggering an alert.                                                                                         |
| _Time-Series Database_   | A database optimized for storing and querying timestamped data points, such as InfluxDB or TimescaleDB.                                                                                               |
| _TLS_                    | Transport Layer Security, a cryptographic protocol for encrypting data in transit. Version 1.3 (RFC 8446) is the current standard.                                                                    |
| _WCAG_                   | Web Content Accessibility Guidelines, a set of recommendations for making web content more accessible, published by the W3C.                                                                          |

### 1.4 References

[1] N. Jiang et al., "On thresholds for controlling negative particle (PM2.5) readings in Air Quality Reporting — environmental monitoring and assessment," SpringerLink, https://link.springer.com/article/10.1007/s10661-023-11750-4 (accessed Jan. 29, 2026).

[2] IEEE Std 830-1998, "IEEE Recommended Practice for Software Requirements Specifications," IEEE, 1998.

[3] McMaster University, "SE 3A04 — Project Outline: Smart City Environmental Monitoring & Alert System (SCEMAS)," Winter 2026, Last Updated January 9, 2026.

[4] OASIS Standard, "MQTT Version 5.0," March 2019. Available: https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html

[5] U.S. Environmental Protection Agency, "Technical Assistance Document for the Reporting of Daily Air Quality — the Air Quality Index (AQI)," EPA-454/B-18-007, 2018.

[6] IETF, "The Transport Layer Security (TLS) Protocol Version 1.3," RFC 8446, August 2018. Available: https://www.rfc-editor.org/rfc/rfc8446

[7] W3C, "Web Content Accessibility Guidelines (WCAG) 2.1," June 2018. Available: https://www.w3.org/TR/WCAG21/

[8] J. Nielsen, "Response Times: The 3 Important Limits," Nielsen Norman Group, 1993. Available: https://www.nngroup.com/articles/response-times-3-important-limits/

### 1.5 Overview

The remainder of this document is organized as follows. Section 2 provides an overall product description, including the product's context, major functions, user characteristics, constraints, assumptions, and deferred requirements. Section 3 presents the use case diagram for the system. Section 4 specifies the functional requirements as use cases organized by business event and viewpoint, culminating in global scenarios. Section 5 enumerates the non-functional requirements across look and feel, usability, performance, operational, maintainability, security, cultural, and legal categories. Section 6 describes the innovative feature selected for implementation.

---

## 5.1 Look and Feel Requirements

### 5.1.1 Appearance Requirements

**LF-A1.** The operator dashboard shall present environmental data through geographical maps displaying sensor locations, line/bar charts for environmental metrics, and gauge indicators for current values. All visualizations shall be legible at a viewing distance of 0.5 meters on a standard 1920x1080 display.

Rationale: A legibility standard is included because operators in a control room setting must read dashboard information without physical strain [3].

**LF-A2.** The public-facing functional client (simulating a digital signage display) shall present aggregated environmental data (e.g., AQI by zone) using high-contrast text and visual indicators sized for readability at a distance of 3 meters.

Rationale: Digital signage is typically viewed at standing distance, so 3-meter readability is a standard baseline for public displays.

**LF-A3.** The system shall use a consistent color scheme across all interfaces, with distinct color-coding for alert severity levels (e.g., green for normal, yellow for warning, red for critical).

Rationale: Color-coded severity is standard practice in monitoring dashboards and aligns with ISO 22727 guidelines for hazard communication. Consistent color conventions reduce cognitive load during time-sensitive alert response.

**LF-A4.** The operator dashboard shall achieve a minimum color contrast ratio of 4.5:1 for normal text and 3:1 for large text, in compliance with WCAG 2.1 Level AA [7].

Rationale: Municipal workplaces must accommodate employees with visual impairments. WCAG 2.1 Level AA is the accepted standard for public-sector web applications in Canada.

**LF-A5.** The system shall not include animations or UI elements that flash more than 3 times per second.

Rationale: Prevents triggering photosensitive epilepsy and complies with WCAG 2.1 guideline 2.3.1 [7].

### 5.1.2 Style Requirements

**LF-S1.** The operator dashboard shall follow a professional, utilitarian visual style appropriate for a municipal control room environment. Decorative elements that do not serve an informational purpose shall be avoided.

Rationale: We imagined stakeholders (city officials and operators) would expect an interface that conveys reliability. A utilitarian style reduces visual clutter and supports sustained attention during long monitoring shifts.

**LF-S2.** All typography across the system shall use a sans-serif font family, with a minimum body text size of 14px for the operator dashboard and 24px for the public display client.

Rationale: Sans-serif fonts are established as more readable on screens. Minimum font sizes are set based on the intended viewing distance for each interface (0.5 m for operator dashboard, 3 m for public display).

**LF-S3.** The operator dashboard shall maintain a consistent and distortion-free layout across all supported viewport sizes (minimum 1280×720). UI elements shall scale proportionately and text shall remain legible without horizontal scrolling.

Rationale: Operators may use different monitor configurations. The system should prevent visual distortion that could cause misinterpretation of environmental data during critical events.

---

## 5.2 Usability and Humanity Requirements

### 5.2.1 Ease of Use Requirements

**UH-EOU1.** A trained city operator shall be able to log into the dashboard and successfully acknowledge a critical environmental alert within 30 seconds of the alert appearing on the dashboard.

Rationale: This is an explicit requirement from the project specification [3]. It establishes a measurable upper bound on the combined authentication and alert response workflow.

**UH-EOU2.** The operator dashboard shall require no more than 3 clicks (or equivalent interactions) to navigate from the main dashboard view to the detail view of any active alert.

Rationale: The 30-second acknowledgment target (UH-EOU1) implies that critical workflows must have minimal navigation depth, particularly during multi-alert scenarios.

**UH-EOU3.** Alert subscription preferences (the innovative feature) shall be configurable from a single interface without requiring the operator to navigate away from the dashboard.

Rationale: Personalized alert subscriptions reduce alert fatigue. If the configuration interface itself introduces navigation friction, the usability benefit is undermined.

**UH-EOU4.** The system shall display a progress indicator when any asynchronous operation is in progress (e.g., loading dashboard data, processing an API request).

Rationale: Progress indicators prevent user frustration by providing feedback on system status, ensuring operators know the system has not crashed or stalled [8].

### 5.2.2 Personalization and Internationalization Requirements

**UH-PI1.** The system shall support personalized alert subscriptions, allowing each city operator to subscribe to specific environmental metrics, geographical zones, or alert severity levels. The system shall filter dashboard notifications to display only alerts matching the operator's active subscriptions.

Rationale: This is the selected innovative feature. Operators responsible for different districts or concerns receive all alerts by default, leading to alert fatigue. Personalized subscriptions ensure each operator sees only relevant alerts, improving response efficiency.

**UH-PI2.** The operator dashboard shall display timestamps in Eastern Time (ET) and measurements in metric units (°C, µg/m³, dB).

Rationale: We imagined stakeholders in a Canadian municipality would expect local time conventions and metric units. Unfamiliar units introduce a translation step that slows interpretation during time-sensitive scenarios.

### 5.2.3 Learning Requirements

**UH-L1.** A competent third-party developer with no prior exposure to SCEMAS shall be able to successfully request and interpret environmental data from the public REST API within a two-hour timeframe, using only the provided API documentation.

Rationale: This is an explicit requirement from the project specification [3].

**UH-L2.** A new city operator shall be able to perform all core dashboard operations (view real-time data, acknowledge alerts, configure alert subscriptions) after a training session of no more than 1 hour.

Rationale: Municipal staff turnover and shift rotations require new personnel to become productive quickly. One hour is a reasonable upper bound given that the interface supports a limited set of well-defined tasks.

### 5.2.4 Understandability and Politeness Requirements

**UH-UP1.** All error messages displayed to users shall include: (a) a description of what went wrong, (b) the likely cause, and (c) a suggested corrective action. The system shall avoid blaming language (e.g., "Unable to update alert status. The database may be temporarily unavailable. Please retry in a few moments." instead of "Error: database write failed").

Rationale: Generic or blaming error messages do not support efficient problem resolution and can erode operator trust during high-stress situations.

**UH-UP2.** The system shall provide confirmation feedback for all state-changing operations (e.g., alert acknowledgment, rule creation, subscription update) within 2 seconds of the action being submitted.

Rationale: Users require feedback to confirm their actions have been registered. A 2-second window aligns with Nielsen's response time thresholds for system acknowledgment of user input [8].

**UH-UP3.** All system messages shall be written in plain language without unnecessary technical jargon, accessible to operators who are not software engineers.

Rationale: City operators are domain experts in environmental management, not software. Technical language (e.g., "query returned no matches in the database") creates unnecessary confusion.

### 5.2.5 Accessibility Requirements

**UH-A1.** All interactive elements on the operator dashboard shall be navigable using keyboard input alone, without requiring a mouse.

Rationale: Keyboard accessibility is a core requirement of WCAG 2.1 Level AA [7] and supports operators who rely on assistive technologies. In control room settings, keyboard-only workflows can also be faster for experienced operators.

**UH-A2.** All non-text elements (maps, charts, gauges, icons) shall include descriptive alternative text or ARIA labels for screen reader compatibility.

Rationale: Ensures the dashboard is usable by visually impaired operators relying on screen readers, in compliance with WCAG 2.1 [7].

---

## 5.3 Performance Requirements

### 5.3.1 Speed and Latency Requirements

**PR-SL1.** The alerting engine shall evaluate incoming sensor telemetry against all active alert rules and, upon detecting a threshold violation, generate and log the corresponding alert within 5 seconds of the telemetry message being received by the ingestion module.

Rationale: Environmental conditions change on the order of minutes, so a 5-second end-to-end latency from ingestion to alert generation is sufficient for timely operator response while accounting for the validation, storage, and rule evaluation pipeline.

**PR-SL2.** The operator dashboard shall render the initial view (including map, current metrics, and active alert list) within 3 seconds of successful authentication, measured on a standard broadband connection (≥10 Mbps).

Rationale: The 30-second acknowledgment target (UH-EOU1) includes login and navigation time. If the dashboard takes longer than 3 seconds to render, the remaining time budget for alert acknowledgment becomes impractically constrained.

**PR-SL3.** The public REST API shall respond to valid requests for aggregated environmental data within 500 milliseconds (p95) under normal operating conditions.

Rationale: A 500 ms p95 response time is consistent with industry expectations for public-facing REST APIs. Digital signage clients and third-party integrations require timely updates to display current conditions.

**PR-SL4.** The telemetry ingestion module shall process and persist a validated sensor message within 2 seconds of receipt via the MQTT broker.

Rationale: The alerting engine (PR-SL1) and dashboard depend on telemetry being available in the database promptly. A 2-second ingestion-to-storage target ensures the 5-second alerting budget remains achievable.

### 5.3.2 Safety-Critical Requirements

**PR-SC1.** The system shall not suppress, discard, or silently fail to process any telemetry message that has been validated and accepted by the ingestion module. All validated telemetry shall be durably persisted before acknowledgment to the MQTT broker.

Rationale: Environmental monitoring data informs public health decisions. Silently losing validated data could result in missed threshold violations, with potential consequences for public health. Durable persistence before acknowledgment ensures at-least-once delivery semantics.

**PR-SC2.** The alerting engine shall not fail to evaluate any validated telemetry message against active alert rules. In the event of an evaluation failure, the system shall log the failure and flag the affected data for manual review.

Rationale: A missed alert evaluation is functionally equivalent to a missed alert. For hazardous conditions (e.g., PM₂.₅ exceeding 100 µg/m³), failing to trigger an alert could delay protective actions [1].

**PR-SC3.** The system shall validate all incoming MQTT messages and API inputs to prevent injection attacks or malformed payloads from compromising system integrity.

Rationale: IoT systems are exposed to potentially untrusted data sources. Input validation at every entry point prevents malicious payloads from corrupting the database or triggering unintended system behavior.

### 5.3.3 Precision or Accuracy Requirements

**PR-PA1.** The telemetry validation module shall reject any incoming sensor message that: (a) does not conform to the defined JSON schema, (b) contains values outside the plausible range for the corresponding sensor type, or (c) contains a timestamp deviating by more than 5 minutes from the server's current time.

Rationale: The 5-minute timestamp tolerance accounts for IoT clock drift while preventing replay attacks. Plausible value ranges (e.g., temperature between −50°C and 60°C, PM₂.₅ between 0 and 1000 µg/m³) prevent sensor malfunctions from corrupting aggregated metrics.

**PR-PA2.** Real-time aggregation calculations (5-minute averages, hourly maximums) shall be computed with floating-point precision of at least 2 decimal places and shall include only validated telemetry data points.

Rationale: Two decimal places are sufficient for AQI reporting standards [5]. Including only validated data prevents spurious outliers from distorting zone-level metrics.

**PR-PA3.** The public API shall report AQI values consistent with the EPA AQI calculation methodology [5], using the most recent complete aggregation window for the requested zone.

Rationale: AQI is a standardized metric with an established calculation methodology. Deviating from the EPA standard would produce values incomparable with other reporting systems, confusing public users and third-party developers.

### 5.3.4 Reliability and Availability Requirements

**PR-RA1.** The system shall target an availability of 99.5% measured on a monthly basis, equivalent to no more than approximately 3.6 hours of unplanned downtime per month.

Rationale: Environmental monitoring requires continuous operation. A 99.5% target is achievable on cloud free tiers while reflecting the expectation that the system should be available for the vast majority of each month. This is appropriate for a non-life-safety monitoring system.

**PR-RA2.** Scheduled maintenance windows shall not exceed 30 minutes and shall be announced to authenticated operators at least 24 hours in advance via the dashboard.

Rationale: Predictable downtime allows operators to arrange alternative monitoring procedures. A 30-minute window limits the period during which environmental data is not being processed.

### 5.3.5 Robustness or Fault-Tolerance Requirements

**PR-RFT1.** If the connection to the time-series database is temporarily unavailable, the telemetry ingestion module shall buffer incoming validated messages in memory (up to a configurable limit) and persist them upon reconnection without data loss.

Rationale: Database connectivity interruptions are a common failure mode. Buffering prevents data loss during brief outages and supports PR-SC1 (no silent data loss). The configurable limit prevents unbounded memory consumption during extended outages.

**PR-RFT2.** Failure of any single module (telemetry ingestion, alerting engine, data access, or security) shall not cause cascading failure of the remaining modules. Each module shall degrade independently and report its health status to the operator dashboard.

Rationale: The project specification identifies four distinct modules. A failure in the alerting engine should not prevent the ingestion module from continuing to accept and store telemetry. Independent degradation preserves partial functionality during component failures.

**PR-RFT3.** The system shall handle malformed MQTT messages by rejecting them with an appropriate error response and logging the event, without impacting the processing of subsequent valid messages.

Rationale: Individual sensors may malfunction or transmit corrupted data. The system must be resilient to invalid input and continue normal operation for all other data sources.

### 5.3.6 Capacity Requirements

**PR-C1.** The telemetry ingestion module shall support concurrent data streams from at least 500 simulated sensors, each transmitting at an interval of once per minute (approximately 8.3 messages per second sustained throughput).

Rationale: A city district might deploy 50–100 sensors across zones for air quality, noise, temperature, and humidity. 500 sensors is a reasonable simulation target and produces throughput within the capacity of a single MQTT broker instance on free-tier cloud infrastructure.

**PR-C2.** The operator dashboard shall support at least 20 concurrent authenticated sessions without degradation of the rendering or refresh performance specified in PR-SL2.

Rationale: A mid-sized operations center might have 5–15 operators per shift, plus supervisors and administrators. 20 concurrent sessions provides sufficient headroom for peak usage.

**PR-C3.** The public REST API shall support at least 100 requests per second while meeting the response time target specified in PR-SL3.

Rationale: The rate-limiting controls required by the project specification [3] will enforce this capacity ceiling and protect against denial-of-service scenarios. 100 req/s accommodates a moderate number of concurrent third-party consumers and digital signage clients.

### 5.3.7 Scalability or Extensibility Requirements

**PR-SE1.** The system architecture shall support horizontal scaling of the telemetry ingestion module and alerting engine (i.e., deploying additional instances) to handle increased sensor volume without changes to the module's internal code.

Rationale: The project specification requires a "scalable software platform" [3]. Horizontal scalability is the standard approach for IoT ingestion pipelines and avoids architectural rework if capacity requirements increase.

**PR-SE2.** The system shall support the addition of new environmental sensor types (e.g., UV index, wind speed) by extending the telemetry schema and validation rules, without modifying the ingestion pipeline's core processing logic.

Rationale: The project specification defines four environmental indicators. A scalable system should accommodate new sensor types as monitoring needs expand. Schema-driven validation ensures adding a sensor type is a configuration change, not a code change.

### 5.3.8 Longevity Requirements

**PR-L1.** The system shall be built using open-source technologies and standard protocols (MQTT, REST, TLS) that are actively maintained and widely adopted, to minimize the risk of dependency obsolescence over a 5-year operational horizon.

Rationale: We imagined stakeholders (a municipal government) would expect the system to remain operable for several years. A 5-year horizon is typical for municipal technology procurement cycles.

**PR-L2.** The time-series database schema shall support data retention of at least 12 months of raw telemetry and 5 years of aggregated data without requiring schema migration.

Rationale: Historical data is required for trend analysis and regulatory reporting. 12 months of raw data supports operational analysis, while 5 years of aggregated data aligns with typical municipal record-keeping requirements.
