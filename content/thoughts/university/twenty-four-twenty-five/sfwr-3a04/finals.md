---
id: finals
tags:
  - sfwr3a04
date: 2025-04-17
description: And everything in between
modified: 2025-04-17 16:08:20 GMT-04:00
title: Software Design 2
---

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-3a04/midterm|midterm]]

> [!question] 1
>
> The Credit Swiss group Information Technology (IT) partners closely with the business to deliver innovative and cost-efficient results. In today’s competitive environment, IT drives performanceand revenue growth. By directly aligning Credit Swiss IT initiatives with the bank’s overall business objectives, Information Technology helps provide Credit Suisse with a distinct competitive advantage.Credit Swiss IT manages more than 1,000 services. All its services are made available for 66,400supported users in 550 locations using different software and hardware technologies. Obviously, CreditSwiss IT organize their services within an architectural style.
>
> 1. What should be this suitable style that provide thin service interfaces and contracts, allows loose coupling, service abstraction and reusability, ease of service discovery, and data sharing (interop-erability)?
> 2. An architecture that has the above strengths exhibits at the same time several weaknesses. Discuss two weaknesses related to this architectural style

1. SOA or Broker
2. Governance and Complexity Overhead

---

## Interaction-Oriented Architecture



## Hierarchy Architecture

| Architecture Type   | Key Characteristics                                                                                                                                                              | Benefits                                                                                                                     | Limitations                                                                                                          | Use Cases                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Main/Subroutine** | - Main program controls sequence<br>- Hierarchical decomposition<br>- Focus on behavior/decision hiding<br>- In procedure orientation: shared data<br>- In OO: encapsulated data | - Easy decomposition<br>- Top-down refinement<br>- Usable in OO subsystems                                                   | - Vulnerable shared data<br>- Tight coupling<br>- Potential ripple effects                                           | - Traditional software systems<br>- Systems with clear functional hierarchy                                  |
| **Master/Slaves**   | - Master controls multiple slaves<br>- Slaves provide replicated services<br>- Master selects results by strategy<br>- Slaves use different algorithms                           | - Fault tolerance<br>- Improved reliability<br>- Parallel execution                                                          | - Communication overhead<br>- Complexity in coordination                                                             | - Mission-critical systems<br>- Performance-critical systems<br>- Systems requiring high accuracy            |
| **Layered**         | - Higher/lower layer division<br>- Each layer has sole responsibility<br>- Up/low interfaces<br>- Higher layers: abstract services<br>- Lower layers: utility services           | - Incremental development<br>- Enhanced independence<br>- Improved reusability<br>- Portability<br>- Component compatibility | - Lower runtime performance<br>- Data marshaling overhead<br>- Not suitable for all apps<br>- Complex error handling | - OS architecture<br>- Network protocols<br>- Enterprise applications<br>- Cross-platform systems            |
| **Virtual Machine** | - Built on existing system<br>- Separates language/hardware from execution<br>- Emulation software<br>- Reproduces external behavior                                             | - Platform independence<br>- Simplified development<br>- Non-native simulation                                               | - Slower execution<br>- Additional overhead                                                                          | - Cross-platform applications<br>- Language interpreters<br>- Simulation environments<br>- Java VM, .NET CLR |

## Heterogeneous Architecture

![[thoughts/university/twenty-four-twenty-five/sfwr-3a04/heterogeneous-decision.webp]]

![[thoughts/university/twenty-four-twenty-five/sfwr-3a04/decision-matrix-architecture-saam.webp]]

Idea of SAAM:

- define collection of scenarios that cover functional and non-functional requirements
- evaluation on candidate architecture
- perform analysis on interaction

> [!important]
>
> A design scenario represents an ==important usage of a system== and reflects the viewpoints of stakeholders

| Concept                                  | Description                                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Definition**                           | Multiple architecture styles used together in the same project to combine benefits and ensure quality                                                                                      |
| **Purpose**                              | To achieve optimal system design by selecting appropriate architectures for different parts of a system                                                                                    |
| **Core Process**                         | Closely related to requirements analysis, considers system requirements, priorities, and constraints                                                                                       |
| **Architecture Decision Methodology**    | Systematic approach to select optimal architecture based on requirements, constraints, and quality attributes                                                                              |
| **Quality Attributes Evaluation**        | Quantitative method using weighted scores for each quality attribute (e.g., performance 50%, security 10%)                                                                                 |
| **Architecture Style Selection Factors** | - Project requirements<br>- Quality attributes<br>- Application domain<br>- Project constraints (budget, deadline)                                                                         |
| **SAAM Method**                          | Software Architecture Analysis Method for evaluating candidate architectures using scenarios                                                                                               |
| **SAAM Process Stages**                  | 1. Define scenarios covering functional and non-functional requirements<br>2. Evaluate all candidate architectures using scenarios<br>3. Analyze interaction relationships among scenarios |
| **Example Quality Attributes**           | - Expandability<br>- Performance<br>- Modifiability<br>- Reliability<br>- Maintainability                                                                                                  |
| **Benefits**                             | - Combines strengths of multiple architectures<br>- Allows optimization for specific system aspects<br>- More flexible approach to complex systems                                         |
| **Challenges**                           | - Increased complexity<br>- Integration issues between different architecture styles<br>- Requires broader expertise from architects                                                       |

## Distributed Architecture

| Architecture Style                      | Pros                                                                                                                                                                                                                            | Cons                                                                                                                                                                                                                                                |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Client/Server**                       | - Separation of responsibilities (UI presentation and business logic)<br>- Reusability of server components                                                                                                                     | - Lack of heterogeneous infrastructure for requirement changes<br>- Security complications<br>- Server availability and reliability concerns<br>- Testability and scalability challenges<br>- Fat-client/thin-client issues (application dependent) |
| **Multi-tier**                          | - Enhanced reusability<br>- Scalability through middle tier<br>- Multi-threading support<br>- Reduced network traffic                                                                                                           | - Complex testability<br>- Additional overhead through multiple layers                                                                                                                                                                              |
| **Broker**                              | - Server implementation and location transparency<br>- Changeability and extensibility<br>- Simple client access to servers<br>- Interoperability via broker bridges<br>- Reusability<br>- Runtime changes to server components | - Inefficiency due to proxy overhead<br>- Low fault-tolerance<br>- Difficulty in testing                                                                                                                                                            |
| **Service-Oriented Architecture (SOA)** | - Loosely-coupled connections<br>- Independent service components (stateless)<br>- Interoperability across platforms/technologies<br>- High reusability of services<br>- Scalability                                            | - Complexity in service orchestration<br>- Service management overhead<br>- Potential performance impacts from service discovery                                                                                                                    |
| **Enterprise Service Bus (ESB)**        | - Unified architecture for high reusability<br>- Addresses reliability and scalability issues<br>- Supports asynchronous queuing<br>- Enables event-driven messaging<br>- Centralizes policy and rules management               | - Additional architectural complexity<br>- Performance overhead from intermediate processing<br>- Configuration and management challenges                                                                                                           |

- Think about Broker as a middleware style architecture

  - RPC

  ```mermaid
  graph LR
    Client1[Client A] --> ClientProxy1[Client Proxy A]
    Client2[Client B] --> ClientProxy2[Client Proxy B]

    ClientProxy1 --> Broker1[Broker 1]
    ClientProxy2 --> Broker1

    Broker1 <--> Bridge[Bridge]
    Bridge <--> Broker2[Broker 2]

    Broker1 --> ServerProxy1[Server Proxy X]
    Broker1 --> ServerProxy2[Server Proxy Y]

    Broker2 --> ServerProxy3[Server Proxy Z]

    ServerProxy1 --> Server1[Server X]
    ServerProxy2 --> Server2[Server Y]
    ServerProxy3 --> Server3[Server Z]

    classDef client fill:#d4f1f9,stroke:#333,stroke-width:1px
    classDef proxy fill:#ffdebd,stroke:#333,stroke-width:1px
    classDef broker fill:#ffe6e6,stroke:#333,stroke-width:1px
    classDef server fill:#dff9d4,stroke:#333,stroke-width:1px
    classDef bridge fill:#f9e4f9,stroke:#333,stroke-width:1px

    class Client1,Client2 client
    class ClientProxy1,ClientProxy2,ServerProxy1,ServerProxy2,ServerProxy3 proxy
    class Broker1,Broker2 broker
    class Server1,Server2,Server3 server
    class Bridge bridge
  ```

## Data-Centered Architecture

| Concept             | Description                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Definition**      | Architecture where a central data structure is accessed by independent components that do not interact directly with each other |
| **Core Principle**  | System components communicate through a shared data store rather than directly with each other                                  |
| **Primary Styles**  | Repository and Blackboard                                                                                                       |
| **Main Components** | Central data store and independent components/knowledge sources                                                                 |

| Feature                    | Repository Style                                                                           | Blackboard Style                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **Central Data Structure** | Passive data store with well-defined schema                                                | Dynamic shared knowledge base                                                                  |
| **Control Flow**           | Components directly access the repository; no centralized control                          | Control component mediates access and determines processing sequence                           |
| **Component Interaction**  | Components operate independently, unaware of other components                              | Components contribute incrementally and can build on others' results                           |
| **Data Access Pattern**    | Data queries and transactions (usually structured)                                         | Opportunistic problem-solving; components trigger when relevant                                |
| **Use Cases**              | Structured data-intensive systems (databases, CASE tools)                                  | AI systems, pattern recognition, complex problem-solving                                       |
| **Problem Domain**         | Well-understood problems with clear data models                                            | Uncertain or open-ended problems requiring multiple solution strategies                        |
| **Examples**               | Database management systems, version control systems                                       | Speech recognition, image understanding, diagnostic systems                                    |
| **Processing Model**       | Transaction-based, data-driven                                                             | Event-driven, opportunity-driven                                                               |
| **Advantages**             | - Data integrity<br>- Data independence<br>- Efficient data sharing                        | - Flexible problem-solving<br>- Supports partial solutions<br>- Adaptable to uncertain domains |
| **Disadvantages**          | - Single point of failure<br>- Performance bottlenecks<br>- Limited to structured problems | - Complex control mechanisms<br>- Difficult to debug<br>- Higher overhead for coordination     |
| **Component Knowledge**    | Components know repository schema but not other components                                 | Components know blackboard structure but not other components                                  |
| **Evolution**              | Easier to modify data schema, harder to change processing                                  | Easier to add new knowledge sources, modify problem-solving approach                           |
| **Suitable for**           | Business systems, data warehousing, content management                                     | Prototype systems, complex event processing, intelligent systems                               |
