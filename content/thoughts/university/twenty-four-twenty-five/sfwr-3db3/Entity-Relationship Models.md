---
date: "2024-09-11"
description: database schema design using entity sets, attributes, relationships, and many-to-many or many-to-one relationship constraints.
id: Entity-Relationship Models
modified: 2025-10-29 02:16:02 GMT-04:00
tags:
  - sfwr3db3
title: Entity-Relationship Models
---

## E/R model

> sketch databse schemas including constraints.

- Entity set = rectangle
- Attribute = oval, with a line to the rectangle (representing its entity set)

## relationship

- connects two or more entity sets.
- represented by a _diamonds_

value of a relationship is a **relationship set**

### many-to-many relationship

> an entity of either set can be connected to many entities of the other set.

### many-to-one relationship

> each entity of the first set can be connected to at most one entity of the second set.
> and each entity of the second set can be connected to at least one entity of the first set.
