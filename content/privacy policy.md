---
date: '2025-02-21'
description: what this site collects, where it is stored, and how to request access, correction, or deletion
id: privacy policy
layout: technical
modified: 2026-08-13 00:00:00 GMT-04:00
noindex: true
permalinks:
  - /privacy
tags:
  - evergreen
title: Privacy Policy
transclude:
  dynalist: false
  title: false
---

**Effective date**: February 21, 2025

_Last updated_: August 13, 2026

This policy applies to `aarnphm.xyz` and its subdomains. The site is a personal website operated by Aaron.

## information collected

### ordinary visits

The site uses Plausible Analytics and Cloudflare analytics to measure visits, popular pages, referrers, approximate geography, performance, and errors. I use these measurements to understand whether the site works and which pages people read.

Cloudflare processes requests before they reach the site. Its service can process IP addresses, routing data, request metadata, system information, and security signals. Cloudflare describes this processing in its [Privacy Policy](https://www.cloudflare.com/policies/privacy/). Cloudflare Web Analytics says it does not use cookies, local storage, or fingerprinting to identify visitors for analytics.

Plausible provides aggregate website analytics under its own [data policy](https://plausible.io/data-policy).

### comments

The comment system stores the information needed to place and display a comment:

- the author name you choose
- the comment text
- the page and text anchor attached to the comment
- creation, update, resolution, and deletion times
- recovery data used when the surrounding page changes

If you sign in with GitHub for comment identity, the site stores your GitHub login, display name, avatar URL, first-seen time, last-seen time, and the comment name associated with that login. GitHub also processes the OAuth request under its own policies.

The site limits comment-author renames. To enforce that limit, the Worker stores keys derived from the old author name, new author name, and connecting IP address in Cloudflare KV for 90 days. The IP address can therefore appear in a KV key during that period. WebSocket comment sessions also use the connecting IP in memory for the active session.

Comments and their history are stored in Cloudflare D1 and Durable Objects. Deleted and resolved comments can remain in storage so synchronization and moderation continue to work.

### browser storage

The site uses local storage or session storage for features you choose to use. Examples include theme, navigation state, search mode, comment identity, pending comment operations, notebook source, editor preferences, flashcard login state, and triathlon display preferences. This data normally stays in your browser. A feature sends the relevant value to the server when the feature requires it, such as submitting a comment or using GitHub identity.

### direct contact

If you email me, I receive your email address and everything you include in the message. I use it to read and answer the message.

## how the information is used

I use this information to:

- serve and secure the site
- operate comments, authentication, and interactive features
- prevent abuse and enforce rate limits
- diagnose errors and improve performance
- measure aggregate readership
- answer direct messages

I do not sell personal information or use it for targeted advertising.

## service providers

Cloudflare hosts the site and its Worker, D1, Durable Object, KV, R2, analytics, security, and network services. Plausible provides analytics. GitHub provides optional OAuth identity. These providers process data under their own terms and privacy policies. Information can be processed in Canada, the United States, and other places where these providers operate.

## retention and deletion

The comment rename limit expires after 90 days. Browser storage remains until you clear it or the site removes it. Analytics retention follows the settings and policies of Cloudflare and Plausible. Comment records and GitHub identity mappings remain until they are removed from the site's storage.

You can ask for access, correction, or deletion of information associated with you by emailing [contact@aarnphm.xyz](mailto:contact@aarnphm.xyz). Include enough detail to locate the relevant comment, login, or message. I may need to verify that the request concerns your information.

## children

This site is intended for a general audience and is not directed at children. Do not submit a comment or other personal information if you are not old enough to consent where you live.

## changes

Changes appear on this page with a new last-updated date.

## contact

Privacy requests can be sent to [contact@aarnphm.xyz](mailto:contact@aarnphm.xyz).
