---
date: '2024-09-04'
description: database management systems, queries and search, transaction guarantees, recovery, and logical and physical data independence.
id: DBMS
modified: 2026-10-08 09:06:17 GMT-04:00
tags:
  - sfwr3db3
  - university
title: DBMS
---

Book: Database Management System [ISBN-13:978-0072465631](https://www.amazon.ca/Database-Management-Systems-Raghu-Ramakrishnan/dp/0072465638), or [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/Ramakrishnan - Database Management Systems 3rd Edition.pdf|pdf]]

> [!important] Midterm
> Thurs Oct.24 2024 (during lecture time)

Due at 2200, late penalty of 20% per 24h, max 5 days.

```bash
ssh se3db3
```

Relational Model, E-R Model, Views, Indexes, Constraints, Relational Algebra

A database holds data and its relationships. A DBMS supplies the software for defining, querying, and updating it. For a course-enrolment database, that includes finding a student's courses, enforcing unique student IDs, coordinating simultaneous updates, and recovering after a crash. The textbook introduces these responsibilities in §§1.3–1.7.

## [[thoughts/Search|search]] vs. query

A query states what to retrieve. A relational query might ask for the students enrolled in 3DB3, using a course ID to filter enrolment records and a join to retrieve their names. A keyword search for “database courses” retrieves documents under a chosen text-matching rule and may rank them by relevance. [[thoughts/PageRank]] supplies a [link-based ranking signal](https://developers.google.com/search/docs/appearance/ranking-systems-guide).

Search is therefore one use of a query system. PostgreSQL, for example, supports [full-text matching and ranking inside SQL](https://www.postgresql.org/docs/current/textsearch-intro.html). Both relational queries and text searches can use indexes to avoid examining every stored record; an index is an access method.

## transactions and recovery

A bank transfer needs a debit and a credit. Putting both updates in one [transaction](https://www.postgresql.org/docs/current/tutorial-transactions.html) lets the DBMS commit or roll back the pair together. This is atomicity. The application still has to encode the right transfer and enforce its rules, such as sufficient funds.

Concurrency control determines how overlapping transactions can interact. At [serializable isolation](https://www.postgresql.org/docs/current/transaction-iso.html), committed transactions have an effect consistent with some one-at-a-time order. Weaker isolation levels permit particular anomalies, so “uses transactions” alone tells us too little about concurrent behaviour. Applications may also need to retry aborted transactions.

[Write-ahead logging](https://www.postgresql.org/docs/current/wal-intro.html) records changes durably before their corresponding data pages reach storage. Recovery can then replay logged changes after a crash. Durability depends on the failure and configuration: PostgreSQL's [asynchronous commit](https://www.postgresql.org/docs/current/wal-async-commit.html) can acknowledge a transaction before its log records reach disk, allowing recent commits to be lost in a crash. Loss of the stored database itself calls for a separate [backup and restore](https://www.postgresql.org/docs/current/backup.html) plan.

## independence

Data independence lets an application keep using a stable view of the data while the DBMS changes how the data is organised. The textbook distinguishes two boundaries in §1.5.3:

- **Physical independence**: changing storage or access paths preserves the logical schema. Adding an index on student IDs can change the cost of a lookup while leaving the query's meaning unchanged.
- **Logical independence**: changing the logical schema preserves the views used by applications. Suppose a faculty table is split into public details and salary details, each keyed by faculty ID. A view can join the two tables to expose the original columns, provided the split preserves the required rows and relationships. Existing reads through that view can keep working; updates through the view need separate handling.

Logical independence depends on maintaining that interface. Renaming a column used directly by an application still requires a migration or a compatibility view.
