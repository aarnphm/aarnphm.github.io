---
id: midterm
tags:
  - sfwr3db3
date: "2024-10-23"
modified: "2024-10-23"
title: databases internals
---

## Practice

Q1.
a. F
b. F (wrong: Must be T)
  - A relation R(A,B,C) **may** have at most three minimal keys (not superkey)
c. T
d. T
e. T (any ops involving a null is a null)
f. F (DML: data manipulation, not management)
g. F (a weak entity set has one or more many-many relationship)
h. F

Q3.

```prolog
Product(maker, model, price)
PC(model, speed)
Printer(model, type)
```
- model is PK for all relations
- `type` are "laser" and "ink-jet"
- every PC model and every printer model is a Product model (every PC/printer must be referenced in relation to Product)
- price of a product should not be more than 10% higher than the average price of all product (average price of all product is given value avgPrice)
- model and price are int, all other attributes of type char(20)


```sql {title="create schema"}
create table Product(
  model INTEGER PRIMARY KEY NOT NULL;
  maker CHAR(20),
  price INTEGER (CHECK price <= (SELECT AVG(price)*1.10 FROM Product))
);

create table PC(
  model INTEGER PRIMARY KEY NOT NULL;
  speed CHAR(20),
  FOREIGN KEY(model) REFERENCES Product(model)
);

create table Printer(
  model INTEGER PRIMARY KEY NOT NULL;
  type CHAR(20) (CHECK (type IN ('laser', 'ink-jet')))
  FOREIGN KEY(model) REFERENCES Product(model)
);
```

```sql {title="find makers from whom a combination (PC and Printer) can be bought for less than 2000"}
SELECT DISTINCT p1.maker FROM Product p
WHERE EXISTS (
  SELECT * FROM PC pc, Printer pr, Product p1, Product p2
  WHERE p1.model = pc.model and p2.model = pr.model and
        p1.price + p2.price < 2000 and p1.maker = p.maker and p2.maker = p.maker
  )
```

```sql {title="For each maker, find the min and max price of a (PC, ink-jet printer) combination"}
SELECT p1.maker, min(p1.price+p2.price), max(p1.price+p2.price)
FROM Product p1, Product p2, PC pc, Printer pr
WHERE pr.type = 'ink-jet' AND p1.model = pc.model AND p2.model = pr.model and p1.maker = p2.maker
ORDER BY p1.maker;
```

Q4.

a. (1,3)
b. cartesian products


![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/Keys and Foreign Keys|Keys and Foreign Keys]]

## [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/Entity-Relationship Models|ER Model]]

> A weak entity doesn't have enough information to have its own PK and relies on supporting entity for unique identification

> [!important] Weak Entity
>
> ==weak== identity we need one (or more) many-to-one (==supporting==) relationship(s) to other (supporting) entity sets

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/weak-entity.png]]

Role

- entity set may appear more than once in a relationship (label the edge between relationship)

## sql.

```sql
create table Beers (
  name CHAR(20) PRIMARY KEY,  -- fixed-length of $n$ character
  manf VARCHAR(20),  -- variable length of $n$ character
)

create table Sells (
  bar CHAR(20),
  beer CHAR(20) REFERENCES Beers(name),
  price REAL NOT NULL,
  PRIMARY KEY (bar, beer)
)

-- or
create table Sells (
  bar CHAR(20),
  beer CHAR(20),
  price REAL NOT NULL,
  PRIMARY KEY (bar, beer),
  FOREIGN KEY(beer)
    REFERENCES Beers(name)
)
```

> [!important] values
>
> any values can be `NULL`, unless specified otherwise

> [!important] PRIMARY KEYS vs. UNIQUE.
>
> - 1 PK for a relation, but several UNIQUE
> - No attributes of PK can be NULL
> - Attributes declared UNIQUE may have NULL

### DATE and TIME

```sql
DATE("yyyy-mm-dd")
TIME("hh:mm:ss")
```

### constraints.

- keys
- foreign keys
- domain
- tuple-based
- assertions

- `REFERENCES` attribute ==**must be**== `PRIMARY KEY` or `UNIQUE`
  ```prolog
  FOREIGN KEYS <list-attributes>
    REFERENCES <relation>(attributes)
  ```

**enforcing** constraints from relation $R$ to relation $S$, the following violation are possible:

1. insert/update $R$ introduces values not found in $S$
2. deletion/update to $S$ causes tuple of $R$  to "dangle"

ex: suppose $R=\text{Sell} \cap S=\text{Beer}$

_delete or update to $S$ that removes a beer value found in some tuples of $R$_

actions:
1. _Default_: reject modification
2. `CASCADE`: make the same changes in Sells
   - Delete beer: delete Sells tuple
   - Update beer: change value in Sells
3. `SET NULL`: change beer to `NULL`

> Can choose either `CASCADE` or `SET NULL` as policy, otherwise reject as default

```sql
create table Sells (
  bar CHAR(20),
  beer CHAR(20) CHECK (beer IN
      (SELECT name FROM Beers)),
  price REAL CHECK (price <= 5.00),
  FOREIGN KEY(beer)
    REFERENCES Beers(name)
    ON DELETE SET NULL
    ON UPDATE CASCADE
)
```

> [!important] attributed-based check
>
> `CHECK(<cond>)`: cond may use name of attribute, but **any other relation/attribute name MUST BE IN subquery**
>
> `CHECK` only runs when a value for that attribute is inserted or updated.

> [!note] Tuple-based checks
>
> added as a relation-schema element
>
> check on insert or update only

```sql
create table Sells (
  bar CHAR(20),
  beer CHAR(20),
  price REAL,
  CHECK (bar = 'Joe''s Bar' OR price <= 5.00),
)
```

### queries

```sql
SELECT name FROM Beers WHERE manf = 'Anheuser-Busch';

SELECT t.name FROM Beers t WHERE t.manf = 'Anheuser-Busch';

SELECT * FROM Beers WHERE manf = 'Anheuser-Busch';

SELECT name AS beer, manf FROM Beers WHERE manf = 'Anheuser-Busch';

SELECT bar, beer,
       price*95 AS priceInYen
FROM Sells;

-- constants as expr (using Likes(drinker,beer))
SELECT drinker,
       'likes Bud' as whoLikesBud
FROM Likes
WHERE beer = 'Bud';
```

> [!note] patterns
>
> `%` is any string, and `_` is any character
> ```sql
> SELECT name FROM Drinkers
> WHERE phone LIKE '%555-_ _ _ _';
> ```

> In sql, logics are 3-valued: TRUE, FALSE, UNKNOWN
>
> - comparing any value with `NULL` yields `UNKNOWN`
> - A tuple in a query answer iff the `WHERE` is `TRUE`

`ANY(<queries>)` and `ALL(<queries>)` ensures anyof or allof relations.

> [!important] `IN` operator
>
> `IN` is concise
> ```sql
> SELECT * FROM Cartoons WHERE LastName IN ('Simpsons', 'Smurfs', 'Flintstones')
> ```

IN is a predicate about `R` tuples

```sql
-- (1,2) satisfies the condition, 1 is output once
SELECT a FROM R -- loop once
where b in (SELECT b FROM S);

-- (1,2) with (2,5) and (1,2) with (2,6) both satisfies the condition, 1 is output twice
SELECT a FROM R, S  -- double loop
WHERE R.b = S.b;
```

> NOT EQUAL operator in SQL is `<>`

> [!note] Difference between `ANY` and `NOT IN`
>
> - `ANY` means not = a, ==or== not = b, ==or== not = c
> - `NOT IN` means not = a, ==and== not = b, ==and== not = c. (analogous to `ALL`)

> [!note] `EXISTS` operator
>
> `EXISTS(<subquery>)` is true iff subquery result is not empty.

> [!note] `UNION`, `INTERSECT`, `EXCEPT`
>
> structure: `(<subquery>)<predicate>(<subquery>)`

### bag

> or a multiset, is like a set, but an element may appear more than once.

- Force results to be a set with `SELECT DISTINCT`
- Force results to be a bag with `UNION ALL`

`ORDER BY` ops followed with `desc`


### insert, update, delete

```sql
INSERT INTO Likes VALUES('Sally', 'Bud');

-- or
INSERT INTO Likes(beer, drinker) VALUES('Bud', 'Sally');
```

add `DEFAULT` value during `CREATE TABLE` (`DEFAULT` value will be used if inserted tuple has no value for given attributes)

```sql
create table Drinkers (
  name CHAR(30) PRIMARY KEY,
  addr CHAR(50)
    DEFAULT '123 Sesame Street',
  phone CHAR(16)
);

-- in this case, this will use DEFAULT value for addr
-- | name  | address           | phone |
-- | Sally | 123 Sesame Street | NULL  |
INSERT INTO Drinkers(name) VALUES('Sally');
```

_Those drinkers who frequent at least one bar that Sally also frequents_

```sql
INSERT INTO Buddies
  (SELECT d2.drinker
   FROM Frequents d1, Frequents d2
   WHERE d1.drinker = 'Sally' AND
     d2.drinker <> 'Sally' AND d1.bar = d2.bar)
```

`DELETE FROM`:

```sql
-- remove a relation
DELETE FROM Beers WHERE name = 'Bud';

-- remove all relation
DELETE FROM Likes;

-- Delete from Beer(name, manf) all beers for which there is another beer by the same manufacturer
DELETE FROM Beers b
  WHERE EXISTS (
      SELECT name FROM Beers
      WHERE manf = b.manf AND name <> b.name
    )
```

`UPDATE` schema:

```prolog
UPDATE <relation>
SET <list-of-attribute-assigments>
WHERE <condition-on-tuple>
```

### aggregations

`SUM`, `AVG`, `COUNT`, `MIN`, `MAX` can be applied toa column in `SELECT` clause

`COUNT(*)` counts the number of tuples

```sql
-- find average price of Bud
SELECT AVG(price) FROM Sells WHERE beer = 'Bud';

-- to get distinct value, then use DISTINCE
SELECT COUNT(DISTINCT price) FROM Sells WHERE beer = 'Bud';
```

> `NULL` ==never== contributes to a sum, average, or count
>
> however, if all values in a column are `NULL`, then aggregation is `NULL`
>
> exception: `COUNT` of an empty set is 0

`GROUP BY`: according to the values of all those attributes, and any aggregation is applied only within each group:

```sql
-- find the youngest employees per rating
SELECT rating, MIN(age)
FROM Employees
GROUP BY rating

-- find for each drinker the average price of Bud at the bars they frequent
SELECT drinker, AVG(price)
FROM Frequents, Sells
WHERE beer = 'Bud' AND Frequents.bar = Sells.bar
GROUP BY drinker;
```

> [!important] restriction on `SELECT` with aggregation
>
> each element of `SELECT` must be either:
> 1. Aggregated
> 2. An attribute on `GROUP BY` list
>
> > [!attention]- illegal example
> > ```sql
> > SELECT bar,beer,AVG(price) FROM Sells GROUP BY bar
> > -- only one tuple out for each bar, no unique way to select which beer to output
> > ```

`HAVING(<condition>)` _may_ followed by `GROUP_BY`

> If so, the condition applies to each group, and groups not satisfying the condition are eliminated.

```sql
-- Get average price of beer given all beer groups exists with at
-- least three bars or manufactured by Pete's
SELECT beer, AVG(price)
FROM Sells
GROUP BY beer
HAVING COUNT(bar) >= 3 OR
  beer in (SELECT name FROM Beers WHERE manf = 'Pete''s');
```

requirements on `HAVING`:
- Anything goes in a subquery
- Outside subqueries they may refer to attributes only if they are either:
  - A grouping attribute
  - aggregated

### cross product (cartesian product)

```sql
-- Frequents            x Sells
-- (Bar) | Beer | Price | Drinker | (Bar)
-- Joe   | Bud  | 3.00  | Aaron   | Joe
-- Joe   | Bud  | 3.00  | Mary    | Jane
SELECT drinker
FROM Frequents, Sells
WHERE beer = 'Bud' AND Frequents.bar = Sells.bar;
```

Or known as **join operations** => all join operations are considered cartesian products.

Outer join preserves dangling tuples by padding with `NULL`

> A tuple of $R$ that has no tuple of $S$ which it joins is said to be `dangling`

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/left-outer-join.png]]
_Left outer join_


![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/right-outer-join.png]]
_Right outer join_


![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/full-outer-join.png]]
_full outer join_


![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/inner-join.png]]
_inner join_

```sql
R [NATURAL] [LEFT|RIGHT|FULL] OUTERJOIN [ON<condition>] S

-- example
R NATURAL FULL OUTERJOIN S
```
- natural: check equality on all common attributes && no two attributes with same name
- left: padding dangling tuples of R only
- right: padding dangling tuples of S only
- full: padding both (default)


## views

- many views (how users see data), single _logical schema (logical structure)_ and _physical schema (files and indexes used)_

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/view-abstraction.png]]

==virtual== views _does not stored in database_ (think of query for constructing relations)

==materialized== views are constructed and stored in DB.

```sql {title="view default to virtual"}
CREATE [MATERIALIZED] VIEW <name> as <query>;

-- example: CanDrink(drinker, beer)
create view CanDrink AS
  SELECT drinker, beer
  FROM Frequents f, Sells s
  WHERE f.bar = s.bar;
```

> Usually one shouldn't update view, as it simply doesn't exists.

## index

idea: think of DS to speed access tuple of relations, organize records via tree or hashing

DS: B+ Tree Index or Hash-based Index

### B+ Tree

note: each node are at least 50% full

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/b-plus-tree.png]]

> [!important] cost
>
> tree is "height-balanced"
>
> insert/delete at $\log_{F}N$ cost
>
> min 50% occupancy, each node contains $d \leq m \leq 2d$ entries, where $d$ is the _order or the tree_


#### insert a data entry

- find correct leaf $L$
- put data entry onto $L$
  - if $L$ has enough space => done!
  - `split` $L$
    - redistribute entries evenly, `copy up` middle key
    - insert index entry point to $L_{2}$ in parent of $L$

> split grow trees, root split increase heights

#### delete a data entry

- find leaf $L$ where entry belongs
- remove entry
  - if L is at least half-full => done!
  - if not
    - redistribute, borrowing from ==sibling== (adjacent node with same parent of $L$)
    - if fails, ==merge== and sibling
  - merge occured then delete entry (point to $L$ or sibling) from parent of $L$

> merge propagate root, decrease heights

### Hash-based Index

- index is a collection of _buckets_

Insert: if bucket is full => `split`

### Alternatives for data entries

|  | How |
| --------------- | --------------- |
| By Value | record contents are stored in index file (no need to follow pointers) |
| By Reference | <k, rid of matching data record> |
| By List of References | <k, [rid of matching data record, ...]> |

