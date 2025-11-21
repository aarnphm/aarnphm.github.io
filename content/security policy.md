---
date: "2025-02-21"
description: How I handle and communicate security issues for aarnphm.xyz and related services
id: security policy
modified: 2025-10-29 02:15:14 GMT-04:00
noindex: true
layout: technical
permalinks:
  - /security
tags:
  - evergreen
title: Security Policy
---

**Effective date**: February 21, 2025

_Last updated_: February 21, 2025

### Scope

This policy applies to `https://aarnphm.xyz` and subdomains, static content, and any code/services deployed from this repository (including Workers/Edge functions). It complements the [[privacy policy|Privacy Policy]] and [[terms of service|Terms of Service]].

Out of scope (report to the vendor directly): Cloudflare platform issues, Plausible Analytics, Stripe, GitHub, browser/vendor vulnerabilities.

### Reporting security issues

Please report issues privately and avoid public disclosure until a fix is available.

- Email: [security\[at\]aarnphm.xyz](mailto:security@aarnphm.xyz) with the subject `SECURITY`
- Optional: Private GitHub [advisory](https://github.com/aarnphm/aarnphm.github.io/security/advisories/new)

Include clear reproduction steps, affected URL/component, potential impact, and any mitigations. Please do not include sensitive personal data in reports.

Acknowledgement target: within 72 hours. Status update target: within 7 days.

### Issue triage

I will validate the report, determine severity and scope, identify affected versions/deployments, prepare a fix/mitigation, and coordinate a timeline for release and disclosure. If a third‑party vendor is involved (e.g., Cloudflare), I will coordinate with them.

### Threat model

Assumptions: this is a personal site with minimal data collection (see [[privacy policy|Privacy Policy]]). Primary risks include supply‑chain vulnerabilities, XSS from user‑generated or third‑party embeds, misconfigured headers/CSP, token/secret exposure in Workers, SSRF to cloud metadata, and denial of service. There are no authenticated user areas.

### Issue severity

Severity reflects likelihood and impact (inspired by CVSS). Ranges are indicative.

#### CRITICAL (CVSS ≥ 9.0)

Remote code execution, secret/key exfiltration, domain/account takeover, or full compromise of confidentiality, integrity, or availability without special conditions.

#### HIGH (7.0–8.9)

Serious impact requiring certain conditions or privileges, e.g., XSS enabling persistent content injection, SSRF reaching sensitive metadata, privilege escalation in operational tooling, or significant data loss.

#### MODERATE (4.0–6.9)

Denial of service, reflected XSS with limited impact, open redirect enabling phishing, or vulnerabilities with meaningful but contained blast radius.

#### LOW (< 4.0)

Informational issues or low‑risk weaknesses such as overly permissive headers with no practical exploit, minor CSP gaps, or verbose error messages.

### Prenotification policy

For CRITICAL, HIGH, or select MODERATE issues, I may privately notify affected vendors or partners (e.g., Cloudflare, Plausible, Stripe) shortly before release to enable coordinated fixes.

- Notification is via private email and/or adding vendor contacts to a private advisory a few days before release.
- Requests to join a notification list can be emailed to [security\[at\]aarnphm.xyz](mailto:security@aarnphm.xyz) and will be considered case‑by‑case.
- Organizations may be removed for premature disclosure before fixes are public.

### Responsible disclosure and safe harbor

If you follow this policy and make a good‑faith effort to avoid privacy violations, service disruption, or data destruction, I will not initiate legal action. Do not run automated DDoS, spam, or social engineering. Coordinate timelines; public disclosure is preferred after a fix is deployed or within a mutually agreed window.

### Operational practices

- HTTPS, HSTS, and standard transport security are enforced by default
- Least‑privilege secrets management and periodic key rotation
- Regular dependency updates and supply‑chain monitoring
- Minimal analytics and logging; see [[privacy policy|Privacy Policy]]
- CSP and security headers reviewed periodically

### Changes

This policy may change over time. Updates will be posted here with a new date.

### Contact

Questions or reports: [security\[at\]aarnphm.xyz](mailto:security@aarnphm.xyz)
