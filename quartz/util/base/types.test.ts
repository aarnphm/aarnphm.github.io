import assert from "node:assert"
import test, { describe } from "node:test"
import { parseFilter, parseViews, compileExpression } from "./types"

describe("base types parser", () => {
  describe("parseFilter", () => {
    describe("comparison expressions", () => {
      test("== operator", () => {
        const result = parseFilter('status == "published"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "status")
          assert.strictEqual(result.operator, "==")
          assert.strictEqual(result.value, "published")
        }
      })

      test("!= operator", () => {
        const result = parseFilter("score != 0")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "score")
          assert.strictEqual(result.operator, "!=")
          assert.strictEqual(result.value, 0)
        }
      })

      test("> operator", () => {
        const result = parseFilter("count > 100")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "count")
          assert.strictEqual(result.operator, ">")
          assert.strictEqual(result.value, 100)
        }
      })

      test("< operator", () => {
        const result = parseFilter("age < 30")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "age")
          assert.strictEqual(result.operator, "<")
          assert.strictEqual(result.value, 30)
        }
      })

      test(">= operator", () => {
        const result = parseFilter("rating >= 4.5")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "rating")
          assert.strictEqual(result.operator, ">=")
          assert.strictEqual(result.value, 4.5)
        }
      })

      test("<= operator", () => {
        const result = parseFilter("price <= 99.99")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "price")
          assert.strictEqual(result.operator, "<=")
          assert.strictEqual(result.value, 99.99)
        }
      })

      test("contains operator", () => {
        const result = parseFilter('description contains "test"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "description")
          assert.strictEqual(result.operator, "contains")
          assert.strictEqual(result.value, "test")
        }
      })

      test("!contains operator", () => {
        const result = parseFilter('tags !contains "deprecated"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "tags")
          assert.strictEqual(result.operator, "!contains")
          assert.strictEqual(result.value, "deprecated")
        }
      })

      test("quoted string value", () => {
        const result = parseFilter("title == 'hello world'")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, "hello world")
        }
      })

      test("unquoted string value", () => {
        const result = parseFilter("category == philosophy")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, "philosophy")
        }
      })

      test("boolean true value", () => {
        const result = parseFilter("published == true")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, true)
        }
      })

      test("boolean false value", () => {
        const result = parseFilter("archived == false")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, false)
        }
      })

      test("numeric value", () => {
        const result = parseFilter("score == 42")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, 42)
        }
      })

      test("negative numeric value", () => {
        const result = parseFilter("temperature == -5")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, -5)
        }
      })

      test("decimal value", () => {
        const result = parseFilter("price == 19.99")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, 19.99)
        }
      })

      test("date string value - parsed as timestamp", () => {
        const result = parseFilter('date >= "2023-01-01"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(typeof result.value, "number")
          assert.strictEqual(result.value, new Date("2023-01-01").getTime())
        }
      })

      test("ISO datetime string value - parsed as timestamp", () => {
        const result = parseFilter('created >= "2024-01-01T14:30:00"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(typeof result.value, "number")
          assert.strictEqual(result.value, new Date("2024-01-01T14:30:00").getTime())
        }
      })

      test("ISO datetime with milliseconds - parsed as timestamp", () => {
        const result = parseFilter('created >= "2024-01-01T14:30:00.000"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(typeof result.value, "number")
          assert.strictEqual(result.value, new Date("2024-01-01T14:30:00.000").getTime())
        }
      })

      test("ISO datetime with Z timezone - parsed as timestamp", () => {
        const result = parseFilter('created >= "2024-01-01T14:30:00Z"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(typeof result.value, "number")
          assert.strictEqual(result.value, new Date("2024-01-01T14:30:00Z").getTime())
        }
      })

      test("now() function value returns Date object", () => {
        const beforeParse = Date.now()
        const result = parseFilter("due < now()")
        const afterParse = Date.now()
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(typeof result.value, "object")
          assert.ok((result.value as any) instanceof Date)
          const timestamp = (result.value as Date).getTime()
          assert.ok(timestamp >= beforeParse && timestamp <= afterParse)
        }
      })

      test("today() function value returns Date object", () => {
        const expectedToday = new Date()
        expectedToday.setHours(0, 0, 0, 0)
        const expectedTimestamp = expectedToday.getTime()

        const result = parseFilter("created >= today()")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(typeof result.value, "object")
          assert.ok((result.value as any) instanceof Date)
          assert.strictEqual((result.value as Date).getTime(), expectedTimestamp)
        }
      })

      test("date() function value returns Date object", () => {
        const result = parseFilter('created >= date("2025-01-01")')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(typeof result.value, "object")
          assert.ok((result.value as any) instanceof Date)
          assert.strictEqual((result.value as Date).getTime(), new Date("2025-01-01").getTime())
        }
      })
    })

    describe("method calls", () => {
      test("method with single arg", () => {
        const result = parseFilter('tags.containsAny("typescript")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.property, "tags")
          assert.strictEqual(result.method, "containsAny")
          assert.deepStrictEqual(result.args, ["typescript"])
          assert.strictEqual(result.negated, false)
        }
      })

      test("method with multiple args", () => {
        const result = parseFilter('tags.containsAny("typescript", "javascript")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.property, "tags")
          assert.strictEqual(result.method, "containsAny")
          assert.deepStrictEqual(result.args, ["typescript", "javascript"])
        }
      })

      test("method with no args", () => {
        const result = parseFilter("description.isEmpty()")
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.property, "description")
          assert.strictEqual(result.method, "isEmpty")
          assert.deepStrictEqual(result.args, [])
        }
      })

      test("negated method call", () => {
        const result = parseFilter('!tags.containsAny("deprecated")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.property, "tags")
          assert.strictEqual(result.method, "containsAny")
          assert.strictEqual(result.negated, true)
        }
      })

      test("containsAll method", () => {
        const result = parseFilter('tags.containsAll("typescript", "react")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.method, "containsAll")
          assert.deepStrictEqual(result.args, ["typescript", "react"])
        }
      })

      test("startsWith method", () => {
        const result = parseFilter('title.startsWith("intro")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.method, "startsWith")
          assert.deepStrictEqual(result.args, ["intro"])
        }
      })

      test("endsWith method", () => {
        const result = parseFilter('filename.endsWith(".ts")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.method, "endsWith")
          assert.deepStrictEqual(result.args, [".ts"])
        }
      })

      test("isType method", () => {
        const result = parseFilter('value.isType("string")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.strictEqual(result.method, "isType")
          assert.deepStrictEqual(result.args, ["string"])
        }
      })
    })

    describe("function calls", () => {
      test("file.hasTag with single tag", () => {
        const result = parseFilter('file.hasTag("book")')
        assert.strictEqual(result.type, "function")
        if (result.type === "function") {
          assert.strictEqual(result.name, "file.hasTag")
          assert.deepStrictEqual(result.args, ["book"])
        }
      })

      test("file.hasTag with multiple tags", () => {
        const result = parseFilter('file.hasTag("book", "article")')
        assert.strictEqual(result.type, "function")
        if (result.type === "function") {
          assert.strictEqual(result.name, "file.hasTag")
          assert.deepStrictEqual(result.args, ["book", "article"])
        }
      })

      test("file.inFolder", () => {
        const result = parseFilter('file.inFolder("library")')
        assert.strictEqual(result.type, "function")
        if (result.type === "function") {
          assert.strictEqual(result.name, "file.inFolder")
          assert.deepStrictEqual(result.args, ["library"])
        }
      })

      test("file.hasProperty", () => {
        const result = parseFilter('file.hasProperty("author")')
        assert.strictEqual(result.type, "function")
        if (result.type === "function") {
          assert.strictEqual(result.name, "file.hasProperty")
          assert.deepStrictEqual(result.args, ["author"])
        }
      })

      test("file.hasLink", () => {
        const result = parseFilter('file.hasLink("target-page")')
        assert.strictEqual(result.type, "function")
        if (result.type === "function") {
          assert.strictEqual(result.name, "file.hasLink")
          assert.deepStrictEqual(result.args, ["target-page"])
        }
      })

      test("negated function call", () => {
        const result = parseFilter('!file.hasTag("deprecated")')
        assert.strictEqual(result.type, "not")
        if (result.type === "not") {
          const inner = result.conditions[0]
          assert.strictEqual(inner.type, "function")
          if (inner.type === "function") {
            assert.strictEqual(inner.name, "file.hasTag")
            assert.deepStrictEqual(inner.args, ["deprecated"])
          }
        }
      })
    })

    describe("arithmetic expressions", () => {
      test("multiplication expression", () => {
        const result = parseFilter("price * 1.13 > 50")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "price * 1.13")
          assert.strictEqual(result.operator, ">")
          assert.strictEqual(result.value, 50)
          assert.strictEqual(result.isExpression, true)
        }
      })

      test("addition expression", () => {
        const result = parseFilter("base + bonus >= 100")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "base + bonus")
          assert.strictEqual(result.isExpression, true)
        }
      })

      test("subtraction expression", () => {
        const result = parseFilter("total - discount < 50")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "total - discount")
          assert.strictEqual(result.isExpression, true)
        }
      })

      test("division expression", () => {
        const result = parseFilter("amount / count == 10")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "amount / count")
          assert.strictEqual(result.isExpression, true)
        }
      })

      test("parenthesized expression", () => {
        const result = parseFilter("(end - start) / 86400000 > 7")
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "(end - start) / 86400000")
          assert.strictEqual(result.isExpression, true)
        }
      })
    })

    describe("logical operators", () => {
      test("AND object syntax", () => {
        const result = parseFilter({
          and: ['status == "published"', 'tags.containsAny("featured")'],
        })
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "comparison")
          assert.strictEqual(result.conditions[1].type, "method")
        }
      })

      test("OR object syntax", () => {
        const result = parseFilter({
          or: ['status == "published"', 'status == "featured"'],
        })
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 2)
        }
      })

      test("NOT object syntax", () => {
        const result = parseFilter({
          not: ['status == "draft"'],
        })
        assert.strictEqual(result.type, "not")
        if (result.type === "not") {
          assert.strictEqual(result.conditions.length, 1)
          assert.strictEqual(result.conditions[0].type, "comparison")
        }
      })

      test("nested AND in OR", () => {
        const result = parseFilter({
          or: [
            {
              and: ['status == "published"', 'file.hasTag("featured")'],
            },
            'priority == "high"',
          ],
        })
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "and")
        }
      })

      test("complex nested structure", () => {
        const result = parseFilter({
          and: [
            {
              or: ['file.hasTag("book")', 'file.inFolder("library")'],
            },
            {
              not: ['file.hasTag("deprecated")'],
            },
          ],
        })
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "or")
          assert.strictEqual(result.conditions[1].type, "not")
        }
      })
    })

    describe("inline boolean operators", () => {
      test("simple && operator", () => {
        const result = parseFilter('status == "published" && rating > 4')
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "comparison")
          assert.strictEqual(result.conditions[1].type, "comparison")
        }
      })

      test("simple || operator", () => {
        const result = parseFilter('priority == "high" || urgent == true')
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "comparison")
          assert.strictEqual(result.conditions[1].type, "comparison")
        }
      })

      test("multiple && operators", () => {
        const result = parseFilter("a == 1 && b == 2 && c == 3")
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 3)
        }
      })

      test("multiple || operators", () => {
        const result = parseFilter("a == 1 || b == 2 || c == 3")
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 3)
        }
      })

      test("mixed && and || with precedence", () => {
        const result = parseFilter("a == 1 && b == 2 || c == 3")
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "and")
          assert.strictEqual(result.conditions[1].type, "comparison")
        }
      })

      test("parentheses override precedence", () => {
        const result = parseFilter('(status == "done" || status == "archived") && !deleted')
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "or")
          if (result.conditions[0].type === "or") {
            assert.strictEqual(result.conditions[0].conditions.length, 2)
          }
        }
      })

      test("negation with && operator", () => {
        const result = parseFilter('status == "current" && !archived')
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "comparison")
          const secondCond = result.conditions[1]
          assert.ok(secondCond !== undefined)
        }
      })

      test("function calls with &&", () => {
        const result = parseFilter('file.hasTag("book") && rating > 4')
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "function")
          assert.strictEqual(result.conditions[1].type, "comparison")
        }
      })

      test("method calls with ||", () => {
        const result = parseFilter('tags.isEmpty() || status == "draft"')
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "method")
          assert.strictEqual(result.conditions[1].type, "comparison")
        }
      })

      test("complex nested expression", () => {
        const result = parseFilter("(a == 1 || b == 2) && (c == 3 || d == 4)")
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          assert.strictEqual(result.conditions[0].type, "or")
          assert.strictEqual(result.conditions[1].type, "or")
        }
      })

      test("whitespace around operators", () => {
        const result = parseFilter("  a == 1   &&   b == 2  ")
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
        }
      })

      test("quoted strings with special chars", () => {
        const result = parseFilter('title == "Hello && World" || status == "published"')
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 2)
          const firstCond = result.conditions[0]
          if (firstCond.type === "comparison") {
            assert.strictEqual(firstCond.value, "Hello && World")
          }
        }
      })

      test("negation with implicit boolean and &&", () => {
        const result = parseFilter('status == "current" && !archived')
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          const secondCond = result.conditions[1]
          assert.strictEqual(secondCond.type, "comparison")
          if (secondCond.type === "comparison") {
            assert.strictEqual(secondCond.property, "archived")
            assert.strictEqual(secondCond.operator, "==")
            assert.strictEqual(secondCond.value, false)
          }
        }
      })

      test("negation with method call and ||", () => {
        const result = parseFilter('!description.isEmpty() || status == "draft"')
        assert.strictEqual(result.type, "or")
        if (result.type === "or") {
          assert.strictEqual(result.conditions.length, 2)
          const firstCond = result.conditions[0]
          assert.strictEqual(firstCond.type, "method")
          if (firstCond.type === "method") {
            assert.strictEqual(firstCond.method, "isEmpty")
            assert.strictEqual(firstCond.negated, true)
          }
        }
      })

      test("complex negation with parentheses", () => {
        const result = parseFilter('!(status == "done" || status == "archived") && !deleted')
        assert.strictEqual(result.type, "and")
        if (result.type === "and") {
          assert.strictEqual(result.conditions.length, 2)
          const firstCond = result.conditions[0]
          assert.strictEqual(firstCond.type, "not")
          if (firstCond.type === "not") {
            assert.strictEqual(firstCond.conditions.length, 1)
            assert.strictEqual(firstCond.conditions[0].type, "or")
          }
          const secondCond = result.conditions[1]
          assert.strictEqual(secondCond.type, "comparison")
          if (secondCond.type === "comparison") {
            assert.strictEqual(secondCond.property, "deleted")
            assert.strictEqual(secondCond.value, false)
          }
        }
      })
    })

    describe("edge cases", () => {
      test("property with dots", () => {
        const result = parseFilter('file.name == "test"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "file.name")
        }
      })

      test("property with underscores", () => {
        const result = parseFilter('internal_notes == "value"')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "internal_notes")
        }
      })

      test("whitespace handling", () => {
        const result = parseFilter('  status   ==   "published"  ')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.property, "status")
          assert.strictEqual(result.value, "published")
        }
      })

      test("empty quotes", () => {
        const result = parseFilter('value == ""')
        assert.strictEqual(result.type, "comparison")
        if (result.type === "comparison") {
          assert.strictEqual(result.value, "")
        }
      })

      test("method with multiple simple args", () => {
        const result = parseFilter('tags.containsAny("hello", "world", "test")')
        assert.strictEqual(result.type, "method")
        if (result.type === "method") {
          assert.deepStrictEqual(result.args, ["hello", "world", "test"])
        }
      })

      test("invalid filter string throws error", () => {
        assert.throws(() => {
          parseFilter("invalid syntax without operator")
        })
      })

      test("invalid filter object throws error", () => {
        assert.throws(() => {
          parseFilter({ unknown: ["test"] })
        })
      })

      test("null filter throws error", () => {
        assert.throws(() => {
          parseFilter(null)
        })
      })
    })
  })

  describe("parseViews", () => {
    test("single view", () => {
      const result = parseViews([
        {
          type: "table",
          name: "all",
          order: ["title", "date", "status"],
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].type, "table")
      assert.strictEqual(result[0].name, "all")
      assert.deepStrictEqual(result[0].order, ["title", "date", "status"])
    })

    test("multiple views", () => {
      const result = parseViews([
        {
          type: "table",
          name: "view1",
        },
        {
          type: "list",
          name: "view2",
        },
      ])
      assert.strictEqual(result.length, 2)
      assert.strictEqual(result[0].type, "table")
      assert.strictEqual(result[1].type, "list")
    })

    test("view with filters", () => {
      const result = parseViews([
        {
          type: "table",
          name: "filtered",
          filters: 'status == "published"',
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.ok(result[0].filters)
      assert.strictEqual(result[0].filters?.type, "comparison")
    })

    test("view with sort config", () => {
      const result = parseViews([
        {
          type: "table",
          name: "sorted",
          sort: [
            { property: "date", direction: "DESC" },
            { property: "title", direction: "ASC" },
          ],
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.deepStrictEqual(result[0].sort, [
        { property: "date", direction: "DESC" },
        { property: "title", direction: "ASC" },
      ])
    })

    test("view with groupBy string", () => {
      const result = parseViews([
        {
          type: "table",
          name: "grouped",
          groupBy: "category",
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].groupBy, "category")
    })

    test("view with groupBy object", () => {
      const result = parseViews([
        {
          type: "table",
          name: "grouped",
          groupBy: {
            property: "status",
            direction: "DESC",
          },
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.ok(typeof result[0].groupBy === "object")
      if (typeof result[0].groupBy === "object") {
        assert.strictEqual(result[0].groupBy.property, "status")
        assert.strictEqual(result[0].groupBy.direction, "DESC")
      }
    })

    test("view with limit", () => {
      const result = parseViews([
        {
          type: "table",
          name: "limited",
          limit: 10,
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].limit, 10)
    })

    test("view with columnSize", () => {
      const result = parseViews([
        {
          type: "table",
          name: "sized",
          columnSize: {
            "note.title": 401,
            "note.date": 147,
          },
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.deepStrictEqual(result[0].columnSize, {
        "note.title": 401,
        "note.date": 147,
      })
    })

    test("list view type", () => {
      const result = parseViews([
        {
          type: "list",
          name: "items",
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].type, "list")
    })

    test("gallery view type", () => {
      const result = parseViews([
        {
          type: "gallery",
          name: "images",
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].type, "gallery")
    })

    test("board view type", () => {
      const result = parseViews([
        {
          type: "board",
          name: "kanban",
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].type, "board")
    })

    test("calendar view type", () => {
      const result = parseViews([
        {
          type: "calendar",
          name: "events",
        },
      ])
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].type, "calendar")
    })

    test("throws error for non-array input", () => {
      assert.throws(() => {
        parseViews({ type: "table", name: "test" } as any)
      })
    })

    test("throws error for view without type", () => {
      assert.throws(() => {
        parseViews([{ name: "test" } as any])
      })
    })

    test("throws error for view without name", () => {
      assert.throws(() => {
        parseViews([{ type: "table" } as any])
      })
    })

    test("throws error for invalid view object", () => {
      assert.throws(() => {
        parseViews([null as any])
      })
    })
  })

  describe("compileExpression", () => {
    test("simple number", () => {
      const evaluator = compileExpression("42")
      const result = evaluator({})
      assert.strictEqual(result, 42)
    })

    test("property reference", () => {
      const evaluator = compileExpression("price")
      const result = evaluator({ price: 100 })
      assert.strictEqual(result, 100)
    })

    test("addition", () => {
      const evaluator = compileExpression("a + b")
      const result = evaluator({ a: 10, b: 20 })
      assert.strictEqual(result, 30)
    })

    test("subtraction", () => {
      const evaluator = compileExpression("a - b")
      const result = evaluator({ a: 50, b: 20 })
      assert.strictEqual(result, 30)
    })

    test("multiplication", () => {
      const evaluator = compileExpression("a * b")
      const result = evaluator({ a: 5, b: 7 })
      assert.strictEqual(result, 35)
    })

    test("division", () => {
      const evaluator = compileExpression("a / b")
      const result = evaluator({ a: 20, b: 4 })
      assert.strictEqual(result, 5)
    })

    test("modulo", () => {
      const evaluator = compileExpression("a % b")
      const result = evaluator({ a: 17, b: 5 })
      assert.strictEqual(result, 2)
    })

    test("mixed operations", () => {
      const evaluator = compileExpression("a + b * c")
      const result = evaluator({ a: 2, b: 3, c: 4 })
      assert.strictEqual(result, 14)
    })

    test("parentheses", () => {
      const evaluator = compileExpression("(a + b) * c")
      const result = evaluator({ a: 2, b: 3, c: 4 })
      assert.strictEqual(result, 20)
    })

    test("nested parentheses", () => {
      const evaluator = compileExpression("((a + b) * c) - d")
      const result = evaluator({ a: 2, b: 3, c: 4, d: 5 })
      assert.strictEqual(result, 15)
    })

    test("decimal numbers", () => {
      const evaluator = compileExpression("price * 1.13")
      const result = evaluator({ price: 100 })
      assert.ok(Math.abs(result - 113) < 0.001)
    })

    test("whitespace handling", () => {
      const evaluator = compileExpression("  a   +   b  ")
      const result = evaluator({ a: 10, b: 20 })
      assert.strictEqual(result, 30)
    })

    test("property with dots", () => {
      const evaluator = compileExpression("obj.value")
      const result = evaluator({ "obj.value": 42 })
      assert.strictEqual(result, 42)
    })

    test("property with underscores", () => {
      const evaluator = compileExpression("my_value")
      const result = evaluator({ my_value: 123 })
      assert.strictEqual(result, 123)
    })

    test("throws error for non-number property", () => {
      const evaluator = compileExpression("value")
      assert.throws(() => {
        evaluator({ value: "not a number" })
      })
    })

    test("throws error for missing property", () => {
      const evaluator = compileExpression("missing")
      assert.throws(() => {
        evaluator({})
      })
    })

    test("throws error for unexpected character", () => {
      assert.throws(() => {
        compileExpression("a $ b")
      })
    })

    test("throws error for unclosed parenthesis", () => {
      assert.throws(() => {
        compileExpression("(a + b")
      })
    })

    test("operator precedence", () => {
      const evaluator = compileExpression("2 + 3 * 4")
      const result = evaluator({})
      assert.strictEqual(result, 14)
    })

    test("left associativity", () => {
      const evaluator = compileExpression("10 - 5 - 2")
      const result = evaluator({})
      assert.strictEqual(result, 3)
    })

    test("complex expression", () => {
      const evaluator = compileExpression("(end - start) / 86400000")
      const result = evaluator({ start: 0, end: 86400000 * 7 })
      assert.strictEqual(result, 7)
    })
  })
})
