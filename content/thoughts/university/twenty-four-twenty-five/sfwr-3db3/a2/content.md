---
date: "2024-11-11"
description: assignment covering sql queries, relational algebra expressions, and database index design for airline reservation system.
id: a2
modified: 2025-11-09 01:41:11 GMT-05:00
tags:
  - sfwr3db3
title: SQL and Relational Algebra
---

## 1. SQL

> [!question] Q1
>
> Find all passengers, between the ages of 20 and 30 (inclusive), who have a "Vegan" or "Vegetarian" dietary preference. Return their ID, name, and age.

```sql
SELECT
    p.personid AS id,
    p.name,
    p.age
FROM person p
JOIN passenger pass ON p.personid = pass.personid
WHERE
    p.age BETWEEN 20 AND 30
    AND pass.dietarypref IN ('Vegan', 'Vegetarian')
ORDER BY p.personid;
```

> [!question] Q2
>
> a. Find the number of airplanes that exist for each model. Return the model and the count for each model.
> b. Extend your query from (a) to find the number of airplanes in each model for any of the following airlines: ‘Air Canada’, ‘Etihad Airways’, or ‘United Airlines’. Return the name of the airline, the model, and the number of airplanes.

```sql
-- Q2a
SELECT
    model,
    COUNT(*) AS numairplanes
FROM airplane
GROUP BY model
ORDER BY model;

-- Q2b
SELECT
    a.name AS airlinename,
    p.model,
    COUNT(*) AS numairplanes
FROM airplane p
JOIN airline a ON p.airlinealias = a.alias
WHERE a.name IN ('Air Canada', 'Etihad Airways', 'United Airlines')
GROUP BY a.name, p.model
ORDER BY a.name, p.model;
```

> [!question] Q3
>
> a. For each "Air Canada" ticket, find the average of the total weight, for all baggage associated to the ticket. Return the ticket number, and the average total (baggage) weight.
> b. Find all tickets with "Oversized", non-fragile baggage with a total weight (strictly) greater than 90 lbs, during the holiday season from Dec. 10, 2023 to Jan. 3, 2024 (inclusive). Return all qualifying ticket numbers, and the total `(Oversized)` baggage weight.

```sql
-- Q3a
SELECT
    t.ticketno,
    AVG(b.totalweight) AS AverageBaggageWeight
FROM ticket t
JOIN scheduledflight sf
    ON
        t.flightno = sf.flightno
        AND t.flightdepdate = sf.depdate
JOIN airline a ON sf.airlinealias = a.alias
LEFT JOIN baggage b ON t.ticketno = b.ticketno
WHERE a.name = 'Air Canada'
GROUP BY t.ticketno
ORDER BY t.ticketno;

-- Q3b
SELECT
    b.ticketno,
    b.totalweight AS OversizedBaggageWeight
FROM baggage b
JOIN ticket t ON b.ticketno = t.ticketno
JOIN scheduledflight sf
    ON
        t.flightno = sf.flightno
        AND t.flightdepdate = sf.depdate
WHERE
    b.bagtype = 'Oversized'
    AND b.fragile = FALSE
    AND b.totalweight > 90
    AND sf.depdate BETWEEN '2023-12-10' AND '2024-01-03'
ORDER BY b.ticketno;
```

> [!question] Q4
>
> Where and when are the cheapest tickets for flights from Toronto "YYZ" to Orlando "MCO"?
> Return the ticket number, the date of departure, the minimum price (rename to min-Price), and the website where the ticket(s) were purchased.

```sql
WITH MinPriceFlights AS (
    -- First find the minimum price for this route
    SELECT MIN(b.Price) as min_price
    FROM Route r
    JOIN ScheduledFlight sf ON r.RouteID = sf.RouteID
    JOIN Ticket t ON sf.FlightNo = t.FlightNo
        AND sf.DepDate = t.FlightDepDate
    JOIN Book b ON t.TicketNo = b.TicketNo
    WHERE r.srcAirport = 'YYZ'
        AND r.dstAirport = 'MCO'
)
SELECT
t.TicketNo,
    sf.DepDate as DepartureDate,
    b.Price as minPrice,
    b.Website
FROM Route r
JOIN ScheduledFlight sf ON r.RouteID = sf.RouteID
JOIN Ticket t ON sf.FlightNo = t.FlightNo
    AND sf.DepDate = t.FlightDepDate
JOIN Book b ON t.TicketNo = b.TicketNo
CROSS JOIN MinPriceFlights mpf
WHERE r.srcAirport = 'YYZ'
    AND r.dstAirport = 'MCO'
    AND b.Price = mpf.min_price
ORDER BY sf.DepDate;
```

> [!question] Q5
>
> a. Which routes are served by at least three airlines? Return the routeID, and display your results in descending order by the number of airlines.
> b. Which routes are not served by any airline? Return the routeID, the source and destination airports

```sql
-- Q5a

SELECT
    u.RouteID,
    COUNT(DISTINCT u.AirlineAlias) as NumAirlines
FROM Use u
GROUP BY u.RouteID
HAVING COUNT(DISTINCT u.AirlineAlias) >= 3
ORDER BY NumAirlines DESC;

-- Q5b

SELECT
    r.RouteID,
    r.srcAirport as SourceAirport,
    r.dstAirport as DestinationAirport
FROM Route r
LEFT JOIN Use u ON r.RouteID = u.RouteID
WHERE u.AirlineAlias IS NULL
ORDER BY r.RouteID;
```

> [!question] Q6
>
> a. Find the number of distinct passengers who also work as either a pilot, cabin crew, or ground staff. Rename this result as NumStaffPassengers.
> b. For each airline, how many pilots or cabin crew are also passengers? Return the airline (alias), and the corresponding count

```sql
-- Q6a

SELECT
    COUNT(DISTINCT p.PersonID) as NumStaffPassengers
FROM Passenger p
WHERE p.PersonID IN (
    SELECT PersonID FROM Pilot
    UNION
    SELECT PersonID FROM CabinCrew
    UNION
    SELECT PersonID FROM GroundStaff
);

-- Q6b

SELECT
    a.Alias as AirlineAlias,
    COUNT(DISTINCT p.PersonID) as StaffPassengerCount
FROM Airline a
LEFT JOIN (
    -- Get all pilots and cabin crew
    SELECT PersonID, AirlineAlias
    FROM CabinCrew
    UNION
    -- For pilots, we need to get their airline through the planes they fly
    SELECT DISTINCT pi.PersonID, ap.AirlineAlias
    FROM Pilot pi
    JOIN Flies f ON pi.PersonID = f.PilotID
    JOIN Airplane ap ON f.AirplaneSNo = ap.SerialNo
) AS staff ON a.Alias = staff.AirlineAlias
-- Join with Passenger to check which staff are also passengers
JOIN Passenger pass ON staff.PersonID = pass.PersonID
GROUP BY a.Alias
ORDER BY a.Alias;
```

> [!question] Q7
>
> a. Find all the one-way routes operated by airline "ACA", i.e., airline alias = ‘ACA’. In this context, a one-way route is where the airline serves from a source airport to a destination airport, but not in the reverse direction. Return the route ID, and the corresponding source and destination airports, respectively.
> b. Find the most popular route where the departure date lies between "2023-12-01" to "2023-12-31" (inclusive). Popularity is defined as the maximum number of tickets purchased during this time duration. Return the route ID, the corresponding source and destination air- ports, and number of tickets sold along this route.

```sql
-- Q7a
SELECT
    r1.RouteID,
    r1.srcAirport as SourceAirport,
    r1.dstAirport as DestinationAirport
FROM Route r1
JOIN Use u1 ON r1.RouteID = u1.RouteID
WHERE u1.AirlineAlias = 'ACA'
AND NOT EXISTS (
    -- Check if reverse route exists
    SELECT 1
    FROM Route r2
    JOIN Use u2 ON r2.RouteID = u2.RouteID
    WHERE u2.AirlineAlias = 'ACA'
    AND r2.srcAirport = r1.dstAirport
    AND r2.dstAirport = r1.srcAirport
)
ORDER BY r1.RouteID;

-- Q7b
WITH RouteTickets AS (
    -- Count tickets per route in December 2023
    SELECT
        r.RouteID,
        r.srcAirport,
        r.dstAirport,
        COUNT(*) as TicketCount
    FROM Route r
    JOIN ScheduledFlight sf ON r.RouteID = sf.RouteID
    JOIN Ticket t ON sf.FlightNo = t.FlightNo
        AND sf.DepDate = t.FlightDepDate
    WHERE sf.DepDate BETWEEN '2023-12-01' AND '2023-12-31'
    GROUP BY r.RouteID, r.srcAirport, r.dstAirport
),
MaxTickets AS (
    -- Find the maximum ticket count
    SELECT MAX(TicketCount) as MaxCount
    FROM RouteTickets
)
SELECT
    rt.RouteID,
    rt.srcAirport as SourceAirport,
    rt.dstAirport as DestinationAirport,
    rt.TicketCount as NumberOfTickets
FROM RouteTickets rt, MaxTickets mt
WHERE rt.TicketCount = mt.MaxCount
ORDER BY rt.RouteID;
```

> [!question] Q8
>
> a. Which Air Canada (alias "ACA") flights from source airport "YYZ" to destination airport "MCO" have "First" class tickets? Return all satisfying flight numbers.
> b. Find all airlines that are unique to their country (i.e., they are the only airline for their country). Return the airline alias, airline name, and the country name

```sql
-- Q8a
WITH AirlinesPerCountry AS (
    -- Count airlines per country
    SELECT
        c.Code as CountryCode,
        c.Name as CountryName,
        COUNT(*) as AirlineCount
    FROM Country c
    JOIN Airline a ON c.Code = a.CountryCode
    GROUP BY c.Code, c.Name
    HAVING COUNT(*) = 1
)
SELECT
    a.Alias as AirlineAlias,
    a.Name as AirlineName,
    apc.CountryName
FROM Airline a
JOIN AirlinesPerCountry apc ON a.CountryCode = apc.CountryCode
ORDER BY apc.CountryName, a.Name;

-- Q8b
SELECT
    a1.Alias as AirlineAlias,
    a1.Name as AirlineName,
    c.Name as CountryName
FROM Airline a1
JOIN Country c ON a1.CountryCode = c.Code
WHERE NOT EXISTS (
    SELECT 1
    FROM Airline a2
    WHERE a2.CountryCode = a1.CountryCode
    AND a2.Alias != a1.Alias
)
ORDER BY c.Name, a1.Name;
```

## 2. Relational Algebra

> [!question]
>
> For queries Q1 - Q6, give the corresponding relational algebra expression

### Q1

$$
\begin{align}
& R_1 = \text{Person} \bowtie_{\text{Person.PersonID} = \text{Passenger.PersonID}} \text{Passenger} \\[6pt]
& R_2 = \sigma_{\substack{
    \text{Age} \geq 20 \\
    \wedge \, \text{Age} \leq 30 \\
    \wedge \, \big(\text{DietaryPref} = \text{'Vegan'} \\
    \phantom{\wedge \,} \vee \, \text{DietaryPref} = \text{'Vegetarian'}\big)
}} (R_1) \\[6pt]
& \text{Result} = \pi_{\text{PersonID}, \, \text{Name}, \, \text{Age}} (R_2)
\end{align}
$$

### Q2

a.

$$
\gamma_{\text{Model}, \text{count}(*) \rightarrow \text{NumAirplanes}}(\text{Airplane})
$$

b.

$$
\begin{align}
& R_1 = \text{Airplane} \bowtie_{\text{AirlineAlias = Alias}} \text{Airline} \\[6pt]
& R_2 = \sigma_{\substack{
    \text{Name} = \text{'Air Canada'} \\
    \vee \, \text{Name} = \text{'Etihad Airways'} \\
    \vee \, \text{Name} = \text{'United Airlines'}
}} (R_1) \\[6pt]
& \text{Result} = \gamma_{\substack{
    \text{Name}, \text{Model}, \\
    \text{count}(*) \rightarrow \text{NumAirplanes}
}} (R_2)
\end{align}
$$

### Q3

a.

$$
\begin{align}
& R_1 = \text{Ticket} \bowtie_{
    \substack{
        \text{FlightNo = FlightNo} \\
        \wedge \, \text{FlightDepDate = DepDate}
    }} \text{ScheduledFlight} \\[6pt]
& R_2 = R_1 \bowtie_{\text{AirlineAlias = Alias}} \text{Airline} \\[6pt]
& R_3 = R_2 \Join_{\text{Ticket.TicketNo = Baggage.TicketNo}} \text{Baggage} \\[6pt]
& R_4 = \sigma_{\text{Name} = \text{'Air Canada'}} (R_3) \\[6pt]
& R_5 = \pi_{\text{TicketNo}, \text{TotalWeight}} (R_4) \\[6pt]
& \text{Result} = \\
& \quad \gamma_{\text{TicketNo}, \, \text{avg}(\text{TotalWeight}) \rightarrow \text{AverageBaggageWeight}} (R_5)
\end{align}
$$

_NOTE_: R2 should "\leftouterjoin" instead (but current limitation of LaTeX renderer)

b.

$$
\begin{align}
& R_1 = \text{Ticket} \bowtie_{
    \substack{
        \text{FlightNo = FlightNo} \\
        \wedge \, \text{FlightDepDate = DepDate}
    }
} \text{ScheduledFlight} \\[6pt]
& R_2 = \text{Baggage} \bowtie_{\text{TicketNo = TicketNo}} R_1 \\[6pt]
& R_3 = \sigma_{\substack{
    \text{BagType} = \text{'Oversized'} \\
    \wedge \, \text{Fragile} = \text{False} \\
    \wedge \, \text{TotalWeight} > 90 \\
    \wedge \, \text{DepDate} \geq \text{'2023-12-10'} \\
    \wedge \, \text{DepDate} \leq \text{'2024-01-03'}
}} (R_2) \\[6pt]
& \text{Result} = \pi_{\text{TicketNo, TotalWeight}} (R_3)
\end{align}
$$

### Q4

$$
\begin{align}
& R_1 = \sigma_{\substack{\text{srcAirport} = \text{'YYZ'} \\ \land \, \text{dstAirport} = \text{'MCO'}}} (\text{Route}) \\[6pt]
& R_2 = R_1 \bowtie_{\text{Route.RouteID} = \text{ScheduledFlight.RouteID}} \text{ScheduledFlight} \\[6pt]
& R_3 = R_2 \bowtie_{
    \substack{
        \text{ScheduledFlight.FlightNo} = \text{Ticket.FlightNo} \\
        \land \, \text{ScheduledFlight.DepDate} = \text{Ticket.FlightDepDate}
    }} \text{Ticket} \\[6pt]
& R_4 = R_3 \bowtie_{\text{Ticket.TicketNo} = \text{Book.TicketNo}} \text{Book} \\[6pt]
& \text{MinPrice} = \mathcal{G}_{\emptyset, \, \text{min\_price} \leftarrow \text{MIN(Price)}} \Big(
    \Pi_{\text{Price}} (R_4)
\Big) \\[6pt]
& \text{Result} = \\
& \quad \Pi_{
    \substack{
        \text{TicketNo}, \, \text{DepDate} \rightarrow \text{DepartureDate}, \\
        \text{Price} \rightarrow \text{minPrice}, \, \text{Website}
    }} \Big(
    \sigma_{\text{Price} = \text{min\_price}} (R_4 \times \text{MinPrice})
\Big)
\end{align}
$$

### Q5

a.

$$
\begin{align}
R_1 &= \Pi_{\text{RouteID}, \text{AirlineAlias}} (\text{Use}) \\[8pt]
R_2 &= \mathcal{G}_{\text{RouteID}, \text{NumAirlines} \leftarrow \text{COUNT}(\text{AirlineAlias})} (R_1) \\
\text{Result} &= \Pi_{\text{RouteID}} (\sigma_{\text{NumAirlines} \geq 3} (R_2))
\end{align}
$$

b.

$$
\begin{align}
R_1 &= \text{Route} \: \Join_{\text{Route.RouteID = Use.RouteID}} \: \text{Use} \\[6pt]
R_2 &= \sigma_{\text{AirlineAlias} \: \text{IS} \: \text{NULL}} (R_1) \\[6pt]
\text{Result} &= \\
& \quad \Pi_{\text{RouteID}, \,
    \substack{
        \text{srcAirport} \rightarrow \text{SourceAirport}, \\
        \text{dstAirport} \rightarrow \text{DestinationAirport}
    }} (R_2)
\end{align}
$$

_NOTE_: Route should "\leftouterjoin" instead (but current limitation of LaTeX renderer)

### Q6

a.

$$
\begin{align}
& \text{Staff} = \\
& \quad \Pi_{\text{PersonID}} (\text{Pilot}) \space \cup \\
& \quad \Pi_{\text{PersonID}} (\text{CabinCrew}) \space \cup \\
& \quad \Pi_{\text{PersonID}} (\text{GroundStaff}) \\[6pt]
& \text{StaffPassengers} = \\
& \quad \Pi_{\text{PersonID}} (\text{Passenger}) \cap \text{Staff} \\[6pt]
& \text{Result} = \\
& \quad \mathcal{G}_{\emptyset, \, \text{NumStaffPassengers} \leftarrow \text{COUNT(PersonID)}} (\text{StaffPassengers})
\end{align}
$$

b.

$$
\begin{align}
& \text{CabinCrewWithAirline} = \\
& \quad \Pi_{\text{PersonID}, \, \text{AirlineAlias}} (\text{CabinCrew}) \\[6pt]
& \text{PilotsWithPlanes} = \\
& \quad \Pi_{\text{PersonID}, \, \text{AirlineAlias}} (\\
& \qquad \text{Pilot} \bowtie_{\text{Pilot.PersonID} = \text{Flies.PilotID}} \text{Flies} \\
& \qquad \bowtie_{\text{Flies.AirplaneSNo} = \text{Airplane.SerialNo}} \text{Airplane}\\
& \quad ) \\[6pt]
& \text{AllStaffWithAirline} = \\
& \quad \text{CabinCrewWithAirline} \cup \text{PilotsWithPlanes} \\[6pt]
& \text{StaffPassengers} = \\
& \quad \text{AllStaffWithAirline} \bowtie_{\text{PersonID}} \Pi_{\text{PersonID}} (\text{Passenger}) \\[6pt]
& \text{Result} = \\
& \quad \mathcal{G}_{\text{AirlineAlias}, \, \text{StaffPassengerCount} \leftarrow \text{COUNT(PersonID)}} (\text{StaffPassengers})
\end{align}
$$

## 3. Indexes

The following includes two possible indexes:

### $\text{(FlightNo, DeptDate)}$ on `ScheduledFlight` table

- Attributes: (FlightNo, DeptDate) on `ScheduledFlight` table
- Properties: composite index on both attributes , clustered index respectively
- Benefits
  - Q3, Q4, Q7b given these queries heavily join with ScheduledFlight and filter on depature dates
  - composite nature supports queries that use both FlightNo and DepDate in joins (frequently due to the foreign key relationship with Ticket table)
  - Since these fields are part of the primary key of ScheduledFlight and are frequently used in joins with Ticket
  - help with range scan on DepDate

### $\text{(RouteID, AirlineAlias)}$ on `Use` table

- Attributes: (RouteID, AirlineAlias) on `Use` table
- Properties: composite index., unclustered index respectively
- Benefits:
  - Q5a, Q5b, Q7a and indirect Q4
  - given these rely on route-airline relationship
  - Q5a needs to count distinct airlines per route, so this index eliminate this scan
  - Q7a looks for ACA airline routes, so this will provide direct access
  - Being unclustered is appropriate as `Use` is frequently accessed for lookups but doesn't require physical ordering
