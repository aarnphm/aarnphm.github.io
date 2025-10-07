---
id: obsidian-bases-functions
tags:
  - documentation
  - bases
  - reference
date: "2025-10-07"
modified: 2025-10-07 10:44:20 GMT-04:00
draft: true
title: obsidian bases functions reference
---

comprehensive documentation of all available functions in obsidian bases, organized by category and type.

## function categories

### global functions

standalone functions called directly without a receiver object.

#### date()

converts a value to a date object.

```
date("2025-01-01")
date(property_name)
```

**parameters:**

- `value: string | number` - date string (ISO 8601) or timestamp

**returns:** `Date`

**examples:**

```
filters: "PubDate >= date('2025-01-01')"
formula: "date(created)"
```

#### if()

conditional expression evaluator.

```
if(condition, valueIfTrue, valueIfFalse)
```

**parameters:**

- `condition: boolean` - condition to evaluate
- `valueIfTrue: any` - value returned when condition is true
- `valueIfFalse: any` - (optional) value returned when condition is false

**returns:** `any`

**examples:**

```
if(price, price.toFixed(2) + " dollars")
if(status == "done", "✓", "✗")
if(tags.contains("urgent"), "HIGH", "NORMAL")
```

#### image()

creates an image reference.

```
image(url)
```

**parameters:**

- `url: string` - image URL or attachment path

**returns:** `ImageReference`

#### icon()

creates an icon reference.

```
icon(name)
```

**parameters:**

- `name: string` - icon name

**returns:** `IconReference`

#### max()

returns the maximum value from a set of numbers.

```
max(value1, value2, ...)
```

**parameters:**

- `values: number[]` - variable number of numeric values

**returns:** `number`

**examples:**

```
max(price, min_price)
max(rating1, rating2, rating3)
```

#### min()

returns the minimum value from a set of numbers.

```
min(value1, value2, ...)
```

**parameters:**

- `values: number[]` - variable number of numeric values

**returns:** `number`

**examples:**

```
min(price, max_budget)
```

#### link()

creates a link reference.

```
link(target)
```

**parameters:**

- `target: string` - link target path

**returns:** `LinkReference`

#### list()

creates a list from values.

```
list(value1, value2, ...)
```

**parameters:**

- `values: any[]` - variable number of values

**returns:** `Array<any>`

#### now()

returns the current date and time.

```
now()
```

**returns:** `Date`

**examples:**

```
formula: "(now() - created) / 86400000"  # days since creation
```

#### today()

returns today's date at midnight.

```
today()
```

**returns:** `Date`

**examples:**

```
filters: "due_date <= today()"
```

#### number()

converts a value to a number.

```
number(value)
```

**parameters:**

- `value: string | number` - value to convert

**returns:** `number`

#### duration()

creates a duration value from milliseconds.

```
duration(milliseconds)
```

**parameters:**

- `milliseconds: number` - duration in milliseconds

**returns:** `Duration`

---

## type-specific methods

methods called on property values using dot notation: `property.method(args)`

### any type

methods available on all values.

#### toString()

converts value to string representation.

```
value.toString()
```

**returns:** `string`

**examples:**

```
status.toString()
count.toString()
```

#### isTruthy()

checks if value is truthy (not null, undefined, false, 0, "").

```
value.isTruthy()
```

**returns:** `boolean`

**examples:**

```
filters: "description.isTruthy()"
```

---

### string methods

methods for string properties.

#### contains()

checks if string contains a substring.

```
string.contains(substring)
```

**parameters:**

- `substring: string` - substring to search for

**returns:** `boolean`

**examples:**

```
filters: "title.contains('obsidian')"
tags.contains("urgent")
```

#### containsAny()

checks if string contains any of the provided substrings.

```
string.containsAny(substring1, substring2, ...)
```

**parameters:**

- `substrings: string[]` - variable number of substrings to search for

**returns:** `boolean`

**examples:**

```
filters: "tags.containsAny('book', 'article', 'paper')"
title.containsAny("urgent", "important")
```

**implementation notes:**

- for arrays: returns true if any element in the array matches any of the provided values
- for strings: returns true if the string contains any of the provided substrings
- uses OR semantics (ANY match)

#### containsAll()

checks if string contains all of the provided substrings.

```
string.containsAll(substring1, substring2, ...)
```

**parameters:**

- `substrings: string[]` - variable number of substrings that must all be present

**returns:** `boolean`

**examples:**

```
filters: "tags.containsAll('book', 'read')"
description.containsAll("machine", "learning")
```

**implementation notes:**

- for arrays: returns true if all provided values exist in the array
- for strings: returns true if the string contains all provided substrings
- uses AND semantics (ALL must match)

#### endsWith()

checks if string ends with a suffix.

```
string.endsWith(suffix)
```

**parameters:**

- `suffix: string` - suffix to check for

**returns:** `boolean`

**examples:**

```
filters: "file.name.endsWith('.md')"
title.endsWith("2025")
```

#### startsWith()

checks if string starts with a prefix.

```
string.startsWith(prefix)
```

**parameters:**

- `prefix: string` - prefix to check for

**returns:** `boolean`

**examples:**

```
filters: "file.name.startsWith('draft-')"
category.startsWith("tech")
```

#### isEmpty()

checks if string is empty or null/undefined.

```
string.isEmpty()
```

**returns:** `boolean`

**examples:**

```
filters: "!description.isEmpty()"
```

**implementation notes:**

- returns true for: `undefined`, `null`, `""`, `[]` (empty array)
- returns false otherwise

#### replace()

replaces all occurrences of a substring.

```
string.replace(search, replacement)
```

**parameters:**

- `search: string` - substring to find
- `replacement: string` - substring to replace with

**returns:** `string`

**examples:**

```
formula: "title.replace('draft-', '')"
```

#### lower()

converts string to lowercase.

```
string.lower()
```

**returns:** `string`

**examples:**

```
formula: "title.lower()"
```

#### reverse()

reverses the string.

```
string.reverse()
```

**returns:** `string`

#### slice()

extracts a substring.

```
string.slice(start, end)
```

**parameters:**

- `start: number` - starting index (inclusive)
- `end: number` - (optional) ending index (exclusive)

**returns:** `string`

**examples:**

```
formula: "title.slice(0, 50)"  # first 50 characters
```

#### split()

splits string into array by delimiter.

```
string.split(delimiter)
```

**parameters:**

- `delimiter: string` - delimiter to split on

**returns:** `Array<string>`

**examples:**

```
formula: "tags.split(',')"
```

#### title()

converts string to title case.

```
string.title()
```

**returns:** `string`

**examples:**

```
formula: "name.title()"
```

#### trim()

removes whitespace from both ends.

```
string.trim()
```

**returns:** `string`

---

### number methods

methods for numeric properties.

#### abs()

returns absolute value.

```
number.abs()
```

**returns:** `number`

**examples:**

```
formula: "difference.abs()"
```

#### ceil()

rounds up to nearest integer.

```
number.ceil()
```

**returns:** `number`

**examples:**

```
formula: "price.ceil()"
```

#### floor()

rounds down to nearest integer.

```
number.floor()
```

**returns:** `number`

**examples:**

```
formula: "rating.floor()"
```

#### round()

rounds to nearest integer.

```
number.round()
```

**returns:** `number`

#### toFixed()

formats number to fixed decimal places.

```
number.toFixed(decimals)
```

**parameters:**

- `decimals: number` - number of decimal places

**returns:** `string`

**examples:**

```
formula: "price.toFixed(2)"  # "19.99"
formula: "price.toFixed(2) + ' dollars'"
```

#### isEmpty()

checks if number is null or undefined.

```
number.isEmpty()
```

**returns:** `boolean`

---

### date methods

methods for date properties.

#### date()

extracts date component (without time).

```
date.date()
```

**returns:** `Date`

#### format()

formats date according to format string.

```
date.format(formatString)
```

**parameters:**

- `formatString: string` - format pattern

**returns:** `string`

**examples:**

```
formula: "created.format('YYYY-MM-DD')"
```

#### time()

extracts time component.

```
date.time()
```

**returns:** `Time`

#### relative()

returns relative time string (e.g., "2 days ago").

```
date.relative()
```

**returns:** `string`

**examples:**

```
formula: "modified.relative()"  # "3 hours ago"
```

#### isEmpty()

checks if date is null or undefined.

```
date.isEmpty()
```

**returns:** `boolean`

---

### list/array methods

methods for array properties.

#### contains()

checks if array contains a value.

```
array.contains(value)
```

**parameters:**

- `value: any` - value to search for

**returns:** `boolean`

**examples:**

```
filters: "tags.contains('important')"
```

#### containsAny()

checks if array contains any of the provided values.

```
array.containsAny(value1, value2, ...)
```

**parameters:**

- `values: any[]` - variable number of values to search for

**returns:** `boolean`

**examples:**

```
filters: "tags.containsAny('urgent', 'high-priority')"
```

#### containsAll()

checks if array contains all provided values.

```
array.containsAll(value1, value2, ...)
```

**parameters:**

- `values: any[]` - variable number of values that must all exist

**returns:** `boolean`

**examples:**

```
filters: "tags.containsAll('book', 'read', '2025')"
```

#### isEmpty()

checks if array is empty or null/undefined.

```
array.isEmpty()
```

**returns:** `boolean`

#### join()

joins array elements into string.

```
array.join(separator)
```

**parameters:**

- `separator: string` - separator between elements (default: ",")

**returns:** `string`

**examples:**

```
formula: "tags.join(', ')"
```

#### reverse()

reverses array order.

```
array.reverse()
```

**returns:** `Array<any>`

#### sort()

sorts array elements.

```
array.sort()
array.sort(direction)
```

**parameters:**

- `direction: string` - (optional) "asc" or "desc"

**returns:** `Array<any>`

**examples:**

```
formula: "scores.sort('desc')"
```

#### flat()

flattens nested arrays.

```
array.flat()
array.flat(depth)
```

**parameters:**

- `depth: number` - (optional) depth to flatten (default: 1)

**returns:** `Array<any>`

#### unique()

returns array with duplicate values removed.

```
array.unique()
```

**returns:** `Array<any>`

**examples:**

```
formula: "all_tags.unique()"
```

#### slice()

extracts portion of array.

```
array.slice(start, end)
```

**parameters:**

- `start: number` - starting index (inclusive)
- `end: number` - (optional) ending index (exclusive)

**returns:** `Array<any>`

**examples:**

```
formula: "recent_items.slice(0, 5)"  # first 5 items
```

#### map()

transforms each element using lambda expression.

```
array.map(element => expression)
```

**parameters:**

- `lambda: Function` - transformation function

**returns:** `Array<any>`

**examples:**

```
formula: "prices.map(p => p * 1.13)"
formula: "items.map(item => item.name)"
```

**lambda syntax:**

- single parameter: `value => expression`
- multiple parameters: `(a, b) => expression`

#### filter()

filters elements using lambda expression.

```
array.filter(element => condition)
```

**parameters:**

- `lambda: Function` - predicate function

**returns:** `Array<any>`

**examples:**

```
filters: "tags.filter(t => t.startsWith('project-')).length > 0"
formula: "items.filter(item => item.price > 10)"
formula: "values.filter(value => value != null)"
```

**lambda syntax:**

- parameter represents each array element
- expression should return boolean
- supports method chaining

---

### link methods

methods for link properties.

#### linksTo()

checks if link points to a target.

```
link.linksTo(target)
```

**parameters:**

- `target: string` - target path or page

**returns:** `boolean`

**examples:**

```
filters: "parent.linksTo('index')"
```

---

### file methods

special methods available on `file` implicit object.

#### asLink()

returns file as a link.

```
file.asLink()
```

**returns:** `Link`

#### hasLink()

checks if file contains link to target.

```
file.hasLink(target)
```

**parameters:**

- `target: string` - link target path

**returns:** `boolean`

**examples:**

```
filters: "file.hasLink('[[MOC]]')"
filters: "file.hasLink(this)"  # backlinks
```

**special values:**

- `this` - refers to the current file (for finding backlinks)

#### hasTag()

checks if file has any of the specified tags (in frontmatter or inline).

```
file.hasTag(tag1, tag2, ...)
```

**parameters:**

- `tags: string[]` - variable number of tags to search for

**returns:** `boolean`

**examples:**

```
filters: "file.hasTag('book')"
filters: "file.hasTag('urgent', 'high-priority')"  # OR semantics
```

**implementation notes:**

- checks both frontmatter tags property AND inline tags in content
- multiple tags use OR semantics (ANY match)
- tag matching is exact (case-sensitive)

#### inFolder()

checks if file is in specified folder.

```
file.inFolder(path)
```

**parameters:**

- `path: string` - folder path

**returns:** `boolean`

**examples:**

```
filters: "file.inFolder('library')"
filters: "file.inFolder('projects/active')"
```

**implementation notes:**

- folder path matching is prefix-based
- automatically normalizes paths (adds trailing slash)
- supports nested folders

#### hasProperty()

checks if file has a frontmatter property.

```
file.hasProperty(propertyName)
```

**parameters:**

- `propertyName: string` - property name to check

**returns:** `boolean`

**examples:**

```
filters: "file.hasProperty('rating')"
```

---

### object methods

methods for object/record properties.

#### isEmpty()

checks if object has no properties.

```
object.isEmpty()
```

**returns:** `boolean`

#### keys()

returns array of object keys.

```
object.keys()
```

**returns:** `Array<string>`

**examples:**

```
formula: "metadata.keys().join(', ')"
```

#### values()

returns array of object values.

```
object.values()
```

**returns:** `Array<any>`

**examples:**

```
formula: "scores.values().sort('desc')"
```

---

### regular expression methods

methods for regex pattern matching.

#### matches()

checks if value matches regex pattern.

```
value.matches(pattern)
```

**parameters:**

- `pattern: string` - regular expression pattern

**returns:** `boolean`

**examples:**

```
filters: "email.matches('^[a-z]+@[a-z]+\\.com$')"
```

---

## special syntax patterns

### method chaining

methods can be chained for complex transformations.

```
property.method1().method2().method3()
```

**examples:**

```
formula: "tags.filter(t => t.startsWith('2025')).sort().join(', ')"
formula: "title.lower().replace(' ', '-').slice(0, 50)"
formula: "prices.map(p => p * 1.13).filter(p => p > 100).sort('desc')"
```

### lambda expressions

anonymous functions for `map()` and `filter()` operations.

**single parameter syntax:**

```
array.filter(value => condition)
array.map(item => expression)
```

**multiple parameters (for indexed operations):**

```
array.map((item, index) => expression)
```

**examples:**

```
# filtering with lambda
tags.filter(tag => tag.startsWith("project-"))
values.filter(v => v > 0 && v < 100)

# mapping with lambda
prices.map(price => price * 1.13)
items.map(item => item.name.lower())

# chaining
tags.filter(t => !t.isEmpty()).map(t => t.lower()).sort().unique()
```

### negation

methods can be negated with `!` operator.

```
!property.method()
```

**examples:**

```
filters: "!description.isEmpty()"
filters: "!tags.containsAny('draft', 'archived')"
```

### type checking with isType()

check property types dynamically.

```
property.isType(typeName)
```

**supported types:**

- `"null"` - null or undefined
- `"string"` - string value
- `"number"` - numeric value
- `"boolean"` - boolean value
- `"array"` - array/list
- `"object"` - object/record (not array)

**examples:**

```
filters: "rating.isType('number')"
filters: "tags.isType('array')"
filters: "!description.isType('null')"
```

### arithmetic expressions

numeric properties support arithmetic operations in comparisons.

```
(property1 + property2) > value
property * constant >= threshold
```

**supported operators:**

- `+` addition
- `-` subtraction
- `*` multiplication
- `/` division
- `%` modulo
- `()` parentheses for grouping

**examples:**

```
filters: "price * 1.13 > 50"  # price with tax
filters: "(end - start) / 86400000 > 7"  # duration in days
formula: "(price / quantity).toFixed(2)"
```

### property access

access nested properties with dot notation.

```
object.property.subproperty
```

**special properties:**

- `file.name` - filename
- `file.path` - full path
- `file.ext` - file extension
- `file.ctime` - creation time
- `file.mtime` - modification time
- `file.size` - file size in bytes

**examples:**

```
filters: "file.mtime >= date('2025-01-01')"
formula: "file.name.replace('.md', '')"
```

### length property

arrays and strings have implicit `length` property.

```
property.length
```

**examples:**

```
filters: "tags.length > 3"
filters: "title.length < 50"
formula: "items.filter(i => i.price > 10).length"
```

---

## filter composition patterns

### combining filters

**AND logic:**

```yaml
filters:
  and:
    - file.hasTag("book")
    - rating >= 4
    - !finished.isEmpty()
```

**OR logic:**

```yaml
filters:
  or:
    - file.hasTag("urgent")
    - priority == "high"
    - due_date <= today()
```

**NOT logic:**

```yaml
filters:
  not:
    - file.hasTag("archived")
    - status == "deleted"
```

**nested composition:**

```yaml
filters:
  and:
    - file.inFolder("library")
    - or:
        - file.hasTag("book")
        - file.hasTag("article")
    - not:
        - status == "archived"
```

### advanced filter examples

**filtering by tag presence:**

```yaml
# has ANY of these tags
filters: "file.hasTag('book', 'article', 'paper')"

# has tag only in properties (not inline)
filters: "tags.contains('book')"

# has inline tag but not in properties
filters:
  and:
    - file.hasTag("book")
    - if(tags.contains("book"), false, true)
```

**filtering by links:**

```yaml
# notes linking to current file (backlinks)
filters: "file.hasLink(this)"

# property contains link to specific file
filters: "parent.linksTo('index')"
```

**filtering by date:**

```yaml
# dates must be coerced
filters: "created >= date('2025-01-01')"

# relative date filtering
filters: "(now() - modified) / 86400000 < 7"  # modified in last 7 days
```

**complex formula filtering:**

```yaml
filters:
  and:
    - "price.isType('number')"
    - "price * 1.13 > 50"
    - "tags.filter(t => t.startsWith('sale-')).length > 0"
```

---

## formula usage

formulas create computed properties displayed in views.

```yaml
formulas:
  property_name: "expression"
```

**examples:**

```yaml
formulas:
  # formatted price
  display_price: 'if(price, price.toFixed(2) + " dollars")'

  # price per unit
  ppu: "(price / quantity).toFixed(2)"

  # days since creation
  age_days: "((now() - created) / 86400000).floor()"

  # tag summary
  tag_list: "tags.join(', ')"

  # conditional formatting
  status_icon: 'if(status == "done", "✓", if(status == "todo", "○", "●"))'

  # complex transformations
  categories: "tags.filter(t => t.startsWith('cat-')).map(t => t.replace('cat-', '')).join(', ')"
```

**accessing formulas:**

```yaml
# in views
order:
  - file.name
  - formula.display_price
  - formula.age_days

# in filters
filters: "formula.age_days > 30"
```

---

## implementation notes for quartz bases

### current implementation status

based on code in `/Users/aarnphm/workspace/garden/quartz/util/base/`:

**✅ fully implemented:**

**comparison operators:**

- `==`, `!=` - equality/inequality with date normalization
- `>`, `<`, `>=`, `<=` - numeric and string comparison
- `contains`, `!contains` - array and string containment
- arithmetic expressions in comparisons with proper precedence

**boolean operators:**

- `&&` (AND) - inline syntax with tokenization
- `||` (OR) - inline syntax with tokenization
- negation with `!` prefix
- yaml-based `and`, `or`, `not` logical operators

**global functions:**

- `file.hasTag()` - checks frontmatter tags (OR semantics)
- `file.inFolder()` - folder path matching with normalization
- `file.hasProperty()` - property existence check
- `file.hasLink()` - link target matching
- `date()` - converts value to Date object
- `now()` - returns current Date object
- `today()` - returns today at midnight as Date object
- `if()` - conditional evaluation (basic implementation)
- `max()` - maximum of numeric values
- `min()` - minimum of numeric values
- `number()` - type conversion
- `duration()` - parses duration strings (7 days, 3h, etc.)
- `link()` - link reference checking
- `list()` - list creation

**type checking methods:**

- `isType()` - supports null, string, number, boolean, array, object
- `isEmpty()` - checks undefined, null, empty string, empty array

**string predicate methods:**

- `containsAny()` - OR semantics for arrays/strings
- `containsAll()` - AND semantics for arrays/strings
- `startsWith()` - string prefix matching
- `endsWith()` - string suffix matching

**property resolution:**

- `file.name`, `file.title`, `file.path`, `file.link`, `file.folder`
- `file.ext`, `file.tags`, `file.outlinks`, `file.inlinks`, `file.backlinks`
- `file.aliases`, `file.ctime`, `file.mtime`
- nested property access with dot notation
- `note.` prefix support

**expression parsing:**

- recursive descent parser for arithmetic
- operator precedence: parentheses > multiply/divide/modulo > add/subtract
- tokenization with quote and parenthesis tracking
- date string parsing (ISO 8601)

**⚠️ partially implemented (filter context only, not formula):**

these methods work for filtering (checking type/existence) but don't transform values for use in formulas:

**string methods:**

- `replace()` - checks if string type
- `lower()` - checks if string type
- `upper()` - checks if string type
- `slice()` - checks if string/array type
- `split()` - checks if string type
- `trim()` - checks if string type

**number methods:**

- `abs()` - checks if number type
- `ceil()` - checks if number type
- `floor()` - checks if number type
- `round()` - checks if number type
- `toFixed()` - checks if number type

**array methods:**

- `join()` - checks if array type
- `reverse()` - checks if array type
- `sort()` - checks if array type
- `unique()` - checks if array type
- `length` - available as property

**❌ not implemented:**

**transformation methods (need formula evaluator):**

- string: `reverse()`, `title()`
- array: `flat()`, `map()`, `filter()`, `slice()`
- object: `keys()`, `values()`, `isEmpty()` for objects
- date: `date()` method, `format()`, `time()`, `relative()`
- file: `asLink()`

**advanced features:**

- lambda expressions for `map()` and `filter()`
- regex pattern matching with `matches()`
- method chaining in formulas
- formula evaluation context (separate from filtering)
- case-insensitive operators
- `in` operator
- range operators

**reference resolution:**

- `image()` - image reference creation
- `icon()` - icon reference creation

### what's the gap?

**critical insight:** the current implementation supports **filtering predicates** (boolean checks) but not **formula evaluation** (value transformations).

methods like `replace()`, `lower()`, `toFixed()` are implemented as type checkers for filtering:

```yaml
# this works (filtering)
filters: "title.lower()" # checks if title is a string

# this doesn't work (formula)
formulas:
  lower_title: "title.lower()" # would need to return transformed value
```

to achieve full obsidian compatibility, need:

1. **formula evaluation engine** - separate from filter predicates
   - evaluates expressions to values (not just boolean)
   - supports method chaining: `title.lower().replace(' ', '-').slice(0, 50)`
   - returns typed values for display

2. **lambda expression parser** - for `map()` and `filter()`
   - parse `element => expression` syntax
   - support parameter binding and scope
   - recursive expression evaluation

3. **value transformation methods** - implement actual transformations
   - string: `replace()`, `lower()`, `upper()`, `slice()`, `split()`, `trim()`, `reverse()`, `title()`
   - number: `abs()`, `ceil()`, `floor()`, `round()`, `toFixed()`
   - array: `join()`, `reverse()`, `sort()`, `flat()`, `unique()`, `slice()`, `map()`, `filter()`
   - date: `format()`, `time()`, `relative()`
   - object: `keys()`, `values()`

4. **regex support** - pattern matching
   - `matches(pattern)` method
   - support js regex syntax
   - escape handling

### extension recommendations

**priority 1 (critical for formula support):**

1. create separate formula evaluator in `quartz/util/base/formula.ts`
2. implement value transformation for string/number/array methods
3. add method chaining support in formula context

**priority 2 (high value features):**

1. lambda expression parser for `map()` and `filter()`
2. date formatting methods (`format()`, `relative()`)
3. object methods (`keys()`, `values()`)

**priority 3 (nice to have):**

1. regex support with `matches()`
2. case-insensitive operators
3. `image()` and `icon()` references
4. enhanced `if()` with recursive parsing

---

## references

- [obsidian bases documentation](https://help.obsidian.md/bases/functions)
- [obsidian bases syntax](https://help.obsidian.md/bases/syntax)
- [bases migration guide](https://forum.obsidian.md/t/bases-migration-quick-start-guide/101571)

---

last updated: 2025-10-07
