---
date: '2024-09-26'
description: some notes about entity-relationship models and foreign keys
id: a1
modified: 2025-11-09 01:40:57 GMT-05:00
tags:
  - sfwr3db3
title: E/R models and keys
---

**Problem 1**: Consider the relations `PLAYERS` and `PLAYS` given by the schemas below.

- `PLAYERS (playerID, firstName, lastName, gender, DOB, height, weight, drafted)`
- `PLAYS (playerID, teamID, teamName, number, position, startYear)`

PLAYERS provides information on all basketball players in the league, giving the playerID, first name and last name of the player,
the gender, the date of birth (DOB), the player’s height and weight, and the year they were drafted into the league.

PLAYS provides information about which players play on which teams. A player with playerID plays on a team with a teamID and team name.
The player has a number, the position they play on the team, and the year they started playing with this team.

For example, playerID 5 plays with teamID 1, the Toronto Raptors, with the number 4, in the point guard position, since 2021. Given these schemas, answer the following questions:

> [!question] 1.a (9 marks)
> Identify three candidate keys. For each candidate key, describe the key, and briefly state the assumptions or conditions under which each candidate key would be valid

Candidate keys:

1. $\text{playerID}$ in `PLAYERS` relation:

- description: playerID contains a sole attribute, so it is minimal superkey. Given that each player will have unique `playerID`
- assumption: each players has unique playerID

2. $\{\text{playerID}, \text{teamID}, \text{number}\}$ in `PLAYS` relation:

- description: $\{\text{playerID}, \text{teamID}, \text{number}\}$ is minimal superkey given assumption.
- assumption: A player uses the same number for their duration at a given team.

3. $\{\text{playerID}, \text{teamID}, \text{startYear}\}$ in `PLAYS` relation:

- description: $\{\text{playerID}, \text{teamID}, \text{startYear}\}$ identifies the assumption, making it a minimal superkey.
- assumption: A player can only be associated with a team at a given period in time.

> [!question] 1.b (6 marks)
> List three integrity constraints that should hold over these relations. For each constraint, describe in one sentence why your constraint is necessary.

1. `playerID` in `PLAYS` references `playerID` in `PLAYERS`:

- reason: foreign key constraint is necessary to ensure referential integrity, in other word, every player in `PLAY` must exist in `PLAYERS`

2. `drafted` in `PLAYERS` must be less than or equal to `startYear` in `PLAYS`:

- reason: temporal integrity constraint, i.e., a player cannot start playing for a team before they were drafted into the league

3. $\{\text{teamID}, \text{number}\}$ in `PLAYS` table must be unique per `playerID`

- reason: uniqueness constraint, i.e., no two players on the same team have the same number at any point in time

---

**Problem 2**: You will prepare an E-R diagram describing the schema of airline operations storing information in an airline database.
MacAir Aviation manages flight operations, passenger services, fleet maintenance, and staff. The company, henceforth referred to as “MacAir”, has hired you to design their database.

MacAir wants to store information about people, where a person is represented with a person ID, name, age, and phone number.

There are four types of persons: passenger, pilot, cabin crew, and ground staff:

- A passenger has a dietary preference (e.g., ‘Vegan’, ‘Gluten-Free’, ‘Lactose- Free’, etc.).
- A pilot, and a cabin crew both have a position (e.g., ‘Captain’, ‘First Officer’, etc.) and a salary.
- Ground staff have attributes for salary and department (e.g. Billing and invoicing, Information Technology, etc.).

An airline ticket has a 13-digit numeric ticket number, a seat number (e.g., 38A, 2E, etc.), and a class (‘E’, ‘B’, or ‘F’, representing economy, business, and first-class, respectively).
Passengers book one or more tickets through a travel website (e.g., ‘Expedia’, ‘SkyScanner’, etc.) with an associated price.

A ticket is bought by exactly one passenger.

MacAir records an airline with an identifying alias, which is a 2-letter alphabetic code (‘AC’ for Air Canada), and the airline name (e.g., ‘Air Canada’).

Airplanes have a serial number, a manufacturer, and a model (e.g. 737MAX).

A pilot flies many airplanes, however, an airplane must be flown by at least one pilot.
A cabin crew member works for at most one airline, and an airline has to have at least one cabin crew member working for it.

An airline must own at least one airplane, but an airplane is owned by exactly one airline.

A country has a code (a 3-letter alphabetic code, e.g., ‘CAN’ for Canada), a name, and a continent.

An airport has an IATA code (International Air Transport Association, 3-letter alphabetic code, e.g., ‘YYZ’ for Toronto Pearson Airport), a name, and a city.

A country has zero or more airports, however, an airport must be in exactly one country.

An airline belongs to exactly one country, but a country can have many airlines.

Ground staff work for at most one airport but an airport must have at least one ground staff.

A (flight) route is represented with a numeric ID, the number of stops (e.g., 0 for nonstop), and the duration (in hours).
A route contains exactly one source airport and exactly one destination airport (e.g., source airport: ’YYZ’, destination airport: ’MCO’).

However, airports serve as the source or destination on many routes.

An airline has many routes around the world, and a route is used by many airlines.

The entity ‘Scheduled Flights’ contains all flights that serve a route.
Scheduled flights are defined via an alpha-numeric flight number, departure date, arrival date, scheduled departure time, scheduled arrival time, actual departure time, and actual arrival time.

A scheduled flight contains exactly one route, but a route participates in many (scheduled) flights.
For example, the ‘YYZ’ to ‘MCO’ route appears in the scheduled flights for (AC1670, Sept. 13, Sept 13, 17:45, 20:35, 18:00, 20:50)
Airlines use at least one scheduled flight to conduct operations, but a scheduled flight is associated to exactly one airline.

A ticket is bought for exactly one (scheduled) flight, and there must be at least one ticket purchased for a (scheduled) flight.

Baggage is associated to exactly one ticket.
We record the type of bags (i.e., carry-on, checked, oversized, strollers),
total quantity of bags for each type (e.g., 2 carry-on bags, 2 checked bags, 1 stroller,
total weight of all bags for a type (e.g., 30kg for carry-on bags, 60kg for checked bags, 5kg for stroller),
and whether the bags (per type) are fragile.

A ticket is associated to many (types of) bags.

> [!question] 2.a
> Draw the ER diagram capturing the described requirements. You may use any drawing tool of your choice, but please ensure your ER diagram is clearly readable, and the notation you use is clear and consistent (i.e., notation from the lecture slides or textbook).

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/a1/ER.pdf]]

> [!question] 2.b
> Give a brief (one sentence) description of each of your entities and relationships, and any constraints that exist. For example, $X$ is a weak entity with attributes $(a, b, c)$, and has a many-one relationship with $Y$

_Person_: denotes the meta definition of a person with attributes $(\text{id [PK], name, age, phone\_number})$

_Baggage_: is an entity with attributes $(\text{type}, \text{quantity}, \text{weight}, \text{is\_fragile})$, has a many-to-one relationship with _Ticket_

_Passenger_: is a subclass of _Person_, with attributes $(\text{dietary\_preference})$, has a one-many relationship with _Ticket_

_Ticket_: is a strong entity with atributes $(\text{ticket\_number [PK]}, \text{seat\_number, class, price, travel\_website})$, having one-to-many relationship with _Baggage_

_Pilot_: is a subclass of _Person_, with attributes $(\text{position},\text{salary})$, has a "fly" one-to-many relationship with _airplane_

_Cabin Crew_: is a subclass of _Person_, with attributes $(\text{position},\text{salary})$, has a "work" many-to-one relationship with _airline_

_Ground Staff_: is a subclass of _Person_, with attributes $(\text{department},\text{salary})$, has a "work" many-to-one relationship with _airport_

_airport_: is a strong entity with attributes $(\text{iata\_code [PK, FK]}, \text{name [PK]}, \text{city})$, has "has" one-to-many relationship with _Ground Staff_ and many-to-one with _country_

_country_: is a strong entity with attributes $(\text{code [PK]}, \text{name}, \text{continent})$, has one-to-many relationship with _airline_

_airline_: is a strong entity with attributes $(\text{name}, \text{alias [PK]})$, has one-to-many relationship with _scheduled_flight_, and one-to-many with _airplane_

_airplane_: is a strong entity with attributes $(\text{serial\_number [PK]}, \text{manufacturer}, \text{model})$, has many-to-one relationship with _pilot_

_flight_route_: is a strong entity with attributes $(\text{id [PK]}, \text{stop, duration})$, has one-to-many relationship with _scheduled_flight_ and one-to-one with _airport_ through relationship `source` and `dest`

_scheduled_flight_: is a strong entity with attributes:

$$
\begin{aligned}
(\text{flight\_number [PK]}, \text{departure\_date}, \text{arrival\_date} & \\
\text{scheduled\_departure\_time}, & \text{scheduled\_arrival\_time}, \\
\text{actual\_departure\_time}, & \text{actual\_arrival\_time})
\end{aligned}
$$

has one-to-many relationship with _flight_route_ and one-to-many with _airport_ through relationship `source`

Constraints:

- All person id are unique.
- An airline must own at least one airplane and have at least one cabin crew member.
- An airplane must be flown by at least one pilot.
- An airport must have at least one ground staff.
- A scheduled flight must have at least one ticket purchased for it.
- A country can have zero or more airports, but an airport must be in exactly one country.
- An airline belongs to exactly one country.
- A route contains exactly one source airport and one destination airport.
- A scheduled flight contains exactly one route and is associated with exactly one airline.
- A ticket is bought for exactly one scheduled flight and by exactly one passenger.

> [!question] 2.c
> Provide the corresponding DB2 `CREATE TABLE`` statements describing the relational schema.
Please include all your statements in an executable script `airline.ddl` that can be run on the DB2 command line, in a single command.
> Ensure that your script runs on the CAS DB2 server.

See also: [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/a1/airline.ddl]]
