---
date: '2024-09-09'
description: relational database concepts covering superkeys, candidate keys, primary keys, foreign keys, and referential integrity constraints.
id: Keys and Foreign Keys
modified: 2026-10-07 09:15:28 GMT-04:00
tags:
  - sfwr3db3
title: Foreign Keys and Relational Models
---

See also [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/relationalModel.pdf|slides]]. The SQL examples use Db2, as in the course exercises.

A relation is a **set of tuples**: duplicate tuples and row order have no meaning in the relational model. An SQL table can contain duplicate rows unless its constraints prevent them. Query results need an `ORDER BY` clause when their order matters.

## tuple and domain constraints

A **domain constraint** restricts the values of one attribute, such as the range of a GPA. A **tuple constraint** checks a row and may involve several attributes, such as requiring an end date to follow a start date.

For this example, assume GPA is measured on a scale from zero to four:

```sql
CREATE TABLE Students (
  sid INTEGER NOT NULL,
  gpa DECIMAL(3, 2),
  PRIMARY KEY (sid),
  CHECK (gpa >= 0.0 AND gpa <= 4.0)
);
```

The check rejects an out-of-range GPA. It permits `NULL`, because the comparison then evaluates to unknown. Add `NOT NULL` to `gpa` when every row must contain a known GPA. A range check alone does not supply that requirement. See Db2's [`CHECK` semantics](https://www.ibm.com/docs/en/db2/11.1.0?topic=statements-create-table).

## unique identifier

A **superkey** is a set of attributes $K$ whose values identify at most one tuple. For every legal instance $r$ of the relation:

$$
\forall t_1,t_2\in r,\qquad
 t_1[K]=t_2[K] \Longrightarrow t_1=t_2
$$

A **candidate key** is a minimal superkey: no proper subset of $K$ is still a superkey. Minimal means that every included attribute is needed; candidate keys can have different numbers of attributes.

If a student registration number `RegNum` uniquely identifies a student, then $\{\mathrm{RegNum}\}$ is a candidate key. The set $\{\mathrm{RegNum},\mathrm{Surname}\}$ is also a superkey, with an unnecessary attribute. Two students may share a surname.

Keys describe the permitted data, including future rows. A column that happens to have distinct values in today's table has not thereby become a candidate key.

## primary key

The **primary key** is the candidate key chosen as the table's main identifier. Its columns cannot contain `NULL`. The lecture notation underlines those attributes.

> [!important] definition
>
> Choose one candidate key as the primary key and enforce the others as alternate keys. SQL permits tables without a declared primary key, so declaring one is a schema-design decision rather than a condition for `CREATE TABLE` to succeed.

> [!note] Remark
>
> A candidate key has two requirements:
>
> 1. Its values distinguish every pair of distinct tuples in every legal instance.
> 2. Removing any attribute loses that guarantee.
>
> Every candidate key is a superkey. A superkey with redundant attributes fails the second requirement.

Db2 requires columns in a `PRIMARY KEY` or `UNIQUE` constraint to be declared `NOT NULL`. It permits several unique constraints and at most one primary key. Null handling for unique constraints differs across database systems. See [Db2 unique constraints](https://www.ibm.com/docs/en/db2/11.1?topic=constraints-unique).

The two enrollment rules in the slides produce different schemas. Read the allowed rows before choosing the key.

**One grade per student-course pair.** A student may take several courses, and several students may receive the same grade:

```sql
CREATE TABLE Enrolled (
  sid INTEGER NOT NULL,
  cid INTEGER NOT NULL,
  grade INTEGER NOT NULL,
  PRIMARY KEY (sid, cid)
);
```

Once a row for student 1 and course 10 exists, a second row for that pair is rejected. Student 1 can still take course 11. Student 2 can take course 10 and earn the same grade as student 1. Adding `UNIQUE (cid, grade)` would reject that last case, introducing a rule the first requirement never asked for.

**At most one course per student, with distinct grades within each course.** This is the slides' second, deliberately restrictive rule:

```sql
CREATE TABLE EnrolledOneCourse (
  sid INTEGER NOT NULL,
  cid INTEGER NOT NULL,
  grade INTEGER NOT NULL,
  PRIMARY KEY (sid),
  UNIQUE (cid, grade)
);
```

Here, `sid` prevents a student from appearing twice. The unique pair prevents two students in the same course from sharing a grade. Equal grades in different courses remain valid. Neither schema requires every student to enroll, and neither includes a term or attempt number for repeated courses.

## referential integrity constraints (foreign keys)

A foreign key requires a child row's identifying values to match a referenced key in a parent table. Db2 allows that parent key to be a primary key or a declared unique constraint. The parent and child can also be the same table, as with an employee's manager. See [Db2's foreign-key declaration](https://www.ibm.com/docs/en/db2/11.1.0?topic=statements-create-table).

For a non-null foreign key $X$ in relation $R_1$ referencing key $K$ in $R_2$, the condition is:

$$
\forall t\in R_1,\quad \exists u\in R_2:\quad t[X]=u[K]
$$

A single-column nullable foreign key can use `NULL` to represent an absent reference. The following enrollment schema makes both references mandatory with `NOT NULL`. It uses the `Students` table defined above:

```sql
CREATE TABLE Courses (
  cid INTEGER NOT NULL,
  PRIMARY KEY (cid)
);

CREATE TABLE EnrolledWithReferences (
  sid INTEGER NOT NULL,
  cid INTEGER NOT NULL,
  grade INTEGER NOT NULL,
  PRIMARY KEY (sid, cid),
  FOREIGN KEY (sid) REFERENCES Students (sid) ON DELETE RESTRICT,
  FOREIGN KEY (cid) REFERENCES Courses (cid) ON DELETE RESTRICT
);
```

Each enrollment must refer to an existing student and course. A student or course may exist without an enrollment. That is the same distinction between cardinality and participation made in the [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/Entity-Relationship Models|entity-relationship model]].

## enforcing referential integrity

An insert with a missing student or course is rejected. Under the explicit `ON DELETE RESTRICT` rules, a parent row with an enrollment is also protected from deletion. Remove its dependent enrollments first when deletion is intended.

Other deletion policies include `CASCADE`, which deletes dependent rows, and `SET NULL`, which clears nullable foreign-key columns. Choose the policy from what deleting the parent means for the data. The non-null enrollment keys above cannot use `SET NULL`.

The original [Informix source](https://www.ibm.com/docs/en/informix-servers/14.10?topic=integrity-referential) gives a customer-orders example of these dependencies. It is a separate database product; use the Db2 references above for the course's SQL syntax.
