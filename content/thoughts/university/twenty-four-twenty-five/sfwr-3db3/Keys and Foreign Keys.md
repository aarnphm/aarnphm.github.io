---
date: '2024-09-09'
description: relational database concepts covering superkeys, candidate keys, primary keys, foreign keys, and referential integrity constraints.
id: Keys and Foreign Keys
modified: 2025-10-29 02:16:02 GMT-04:00
tags:
  - sfwr3db3
title: Foreign Keys and Relational Models
---

See also [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/relationalModel_Sept5.pdf|slides]]

> A relation is a table

Relations are **unordered** => ==relations are sets==

## tuple and domain constraints

- tuple: expresses conditions on the values of each tuple
- domain constraint: tuple constrain that involves a single attributes

```sql
(GPA <= 4.0) AND (GPA >= 0.0)
```

## unique identifier

> A _superkey_ is a set of attributes for a relation $r$ if $r$ cannot contain two distinct tuples $t_1$ and $t_2$ such that $t_1{[K]} = t_2{[K]}$

> A _(candidate) key_ for $r$ if $K$ is a minimal superkey

ex: superkey of `RegNum`

## primary value

handles `null` value

> Presence of nulls in keys

> [!important] definition
>
> Each relation must have a **primary key** on which nulls are not allowed.
>
> notation: the attributes of the primary keys are _underlined_

=> references between relations are realised through primary keys

> [!note] Remark
>
> A set of fields is a _key_ for a relation if:
>
> 1. No two distinct tuples can have same values in all key fields
> 2. This is not true for any subset of the key (minimal)
>
> If #2 is false, then a _superkey_
>
> If there's > 1 key for a relation, one of the keys is chosen to be _primary key_

Example:

requirements:

- For a given student and course, there is a single grade.

```sql
CREATE TABLE Enrolled (
  sid INTEGER,
  cid INTEGER,
  grade INTEGER,
  PRIMARY KEY (sid, cid),
  UNIQUE (cid, grade)
);
```

- Students can take only one course, and received a single grade for that courses; further, no two students in a course receive the grade

```sql
CREATE TABLE Enrolled (
  sid INTEGER,
  cid INTEGER,
  grade INTEGER,
  PRIMARY KEY sid,
  KEY (cid, grade)
);
```

> IC are validated when data is updated

## interpolation constraints (foreign keys)

Referential integrity constraints _are imposed in order to guarantee **values** refer to existing tuples_

> [!note] Definition
>
> A _foreign key_ requires that the values on a set $X$ of attributes of a relation $R_1$ **must appear as values** for the _primary key_ of another relation $R_2$

Ex: _sid_ is a _foreign key_ referring to _Students_

> If al foreign key constraints are enforced => referential integrity is enforced

## enforcing referential integrity

See also [source](https://www.ibm.com/docs/en/informix-servers/14.10?topic=integrity-referential)
