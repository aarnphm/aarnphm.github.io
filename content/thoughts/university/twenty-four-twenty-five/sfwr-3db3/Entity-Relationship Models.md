---
date: '2024-09-11'
description: database schema design using entity sets, attributes, relationships, and many-to-many or many-to-one relationship constraints.
id: Entity-Relationship Models
modified: 2026-10-07 09:14:20 GMT-04:00
tags:
  - sfwr3db3
title: Entity-Relationship Models
---

## E/R model

An entity-relationship model describes the objects a database records, their attributes, and the relationships between them. Start with the rules the data must obey, then choose a schema that can enforce them.

In the notation used in the [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/ER_Model.pdf|course slides]]:

- An **entity set** is a rectangle. `Students` is a set; one particular student is an entity in that set.
- An **attribute** is an oval connected to its entity set. A student number is an attribute of `Students`.
- A **relationship** is a diamond connected to the entity sets that take part in it.

## relationship

A relationship connects entities in two or more roles. For a binary relationship between entity sets $A$ and $B$, its current **relationship set** is a set of pairs:

$$
R \subseteq A \times B
$$

For example, `Enrolled` contains a pair for each student-course enrollment. The schema describes which pairs are allowed; the relationship set records which pairs currently exist. A relationship can also have attributes: a grade belongs to a particular enrollment.

### many-to-many relationship

A student may enroll in several courses, and a course may contain several students. This is many-to-many: neither endpoint is restricted to one partner.

“Many” permits several; it does not require several. A course with no students and a student taking one course both satisfy this cardinality constraint. Requiring an entity to appear in the relationship is a separate **participation constraint**.

### many-to-one relationship

Suppose each student can declare at most one department. The relationship from `Students` to `Departments` is many-to-one:

$$
(s,d_1)\in R \;\land\; (s,d_2)\in R
\quad\Longrightarrow\quad d_1=d_2
$$

Several students can declare the same department. A department can also have no students, and a student can have no declared department. Those cases satisfy “at most one.”

The participation rules supply the minimum:

| Additional rule                         | Consequence                                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Every student must declare a department | Total participation of `Students`; together with many-to-one, each student has exactly one department. |
| Every department must have a student    | Total participation of `Departments`; a department with no students is invalid.                        |
| Neither rule is imposed                 | Participation may be partial on both sides.                                                            |

These rules are independent. Making every student choose a department still allows all students to choose the same one, leaving another department empty.

When this becomes a relational schema, a nullable department foreign key in `Students` can represent the optional many-to-one relationship. Making that column `NOT NULL` requires each student to choose a department. Requiring every department to have a student needs a further constraint; the student-side foreign key alone cannot enforce it. See [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/Keys and Foreign Keys|keys and foreign keys]].
