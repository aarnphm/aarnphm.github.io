---
date: "2025-03-03"
description: and designs.
id: midterm
modified: 2025-10-29 02:16:02 GMT-04:00
tags:
  - sfwr3a04
  - midterm
title: software architectures
---

- Design: the process of ==embedding a function== in a form that is dependable,
  sustainable, and in harmony with its environment.

- Architecture: both the process and the product of planning,
  designing, and constructing buildings or other structures.

- Form: material manifestation of the design object in its environment

- Function: fundamental (form is derived from function)
  - Program, designed

reentrant objects/components (executed concurrently)

### code structure

- direction: A -> B
- synchronization: async/sync ops
- sequence: connectors must be used on given seqs

### runtime structure

- multiplicity: connected to other elements
- distance and connection media: connected

- universal invocable: ANY external to invoke
- self-descriptive: allow external to invoke w/o pre-installation

> [!note] 4+1 model
>
> different views with different concerns:
>
> - Logical view: identifies software modules and their boundaries, interfaces, external environment
> - Process view: addresses non-functional requirements (module communication styles)
> - Development view: organizes the software units in a well-defined ways according to file/directory structure
> - Physical view: specifies physical software, hardware, topology view
> - UI view: look and feel view

### relations and UML

- inheritance (is a)
- aggregation (has a)
- association (uses a)
- messaging (method invocation)

Static structural Diagram

- control flow (time-independant) between elements and system
- class diagram, component diagram
- independent from time

Dynamic software architecture

- behaviours of objects
- object collaboration, interaction, activity
- sequences diagram, collaboration diagram
