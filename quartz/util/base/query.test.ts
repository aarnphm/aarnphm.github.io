import test, { describe } from "node:test"
import assert from "node:assert"
import { evaluateFilter, BaseFilter, resolvePropertyValue } from "./query"
import { QuartzPluginData } from "../../plugins/vfile"
import { FullSlug } from "../path"

// helper to create mock file data
function createMockFile(
  frontmatter: Record<string, any>,
  slug?: string,
  links?: string[],
  filePath?: string,
): QuartzPluginData {
  return {
    slug: (slug || "test") as FullSlug,
    frontmatter,
    links: links || [],
    tags: frontmatter.tags || [],
    filePath,
  } as unknown as QuartzPluginData
}

describe("resolvePropertyValue", () => {
  test("returns basename for file.name", () => {
    const file = createMockFile({}, "library/note.md", undefined, "library/Note Title.md")
    const result = resolvePropertyValue(file, "file.name", [])
    assert.strictEqual(result, "Note Title")
  })

  test("returns backlinks for file.backlinks", () => {
    const target = createMockFile({}, "library/note.md")
    const source = createMockFile({}, "library/source.md", ["library/note.md"])
    const result = resolvePropertyValue(target, "file.backlinks", [target, source])
    assert.ok(Array.isArray(result))
    assert.strictEqual(result.length, 1)
    assert.strictEqual(result[0], "library/source.md")
  })
})

describe("base query engine", () => {
  describe("comparison operators", () => {
    test("== operator with strings", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "status",
        operator: "==",
        value: "published",
      }
      const files = [
        createMockFile({ status: "published" }),
        createMockFile({ status: "draft" }),
        createMockFile({ status: "published" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("!= operator with strings", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "status",
        operator: "!=",
        value: "draft",
      }
      const files = [
        createMockFile({ status: "published" }),
        createMockFile({ status: "draft" }),
        createMockFile({ status: "archived" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
      assert.strictEqual(result[0].frontmatter?.status, "published")
      assert.strictEqual(result[1].frontmatter?.status, "archived")
    })

    test("file.name equality", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.name",
        operator: "==",
        value: "note",
      }
      const files = [createMockFile({}, "library/note.md"), createMockFile({}, "library/other.md")]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].slug, "library/note.md")
    })

    test("> operator with numbers", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "score",
        operator: ">",
        value: 50,
      }
      const files = [
        createMockFile({ score: 30 }),
        createMockFile({ score: 60 }),
        createMockFile({ score: 100 }),
        createMockFile({ score: 50 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
      assert.strictEqual(result[0].frontmatter?.score, 60)
      assert.strictEqual(result[1].frontmatter?.score, 100)
    })

    test("< operator with numbers", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "score",
        operator: "<",
        value: 50,
      }
      const files = [
        createMockFile({ score: 30 }),
        createMockFile({ score: 60 }),
        createMockFile({ score: 20 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test(">= operator with numbers", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "score",
        operator: ">=",
        value: 50,
      }
      const files = [
        createMockFile({ score: 30 }),
        createMockFile({ score: 50 }),
        createMockFile({ score: 60 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("<= operator with numbers", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "score",
        operator: "<=",
        value: 50,
      }
      const files = [
        createMockFile({ score: 30 }),
        createMockFile({ score: 50 }),
        createMockFile({ score: 60 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("> operator with strings", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "title",
        operator: ">",
        value: "m",
      }
      const files = [
        createMockFile({ title: "apple" }),
        createMockFile({ title: "banana" }),
        createMockFile({ title: "zebra" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].frontmatter?.title, "zebra")
    })

    test("contains operator with arrays", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "tags",
        operator: "contains",
        value: "typescript",
      }
      const files = [
        createMockFile({ tags: ["javascript", "typescript"] }),
        createMockFile({ tags: ["python", "rust"] }),
        createMockFile({ tags: ["typescript", "react"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("contains operator with strings", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "description",
        operator: "contains",
        value: "test",
      }
      const files = [
        createMockFile({ description: "this is a test" }),
        createMockFile({ description: "no matches here" }),
        createMockFile({ description: "another test case" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("!contains operator with arrays", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "tags",
        operator: "!contains",
        value: "deprecated",
      }
      const files = [
        createMockFile({ tags: ["active", "featured"] }),
        createMockFile({ tags: ["deprecated", "old"] }),
        createMockFile({ tags: ["new", "beta"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("null/undefined handling", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "status",
        operator: "==",
        value: "published",
      }
      const files = [
        createMockFile({ status: "published" }),
        createMockFile({}),
        createMockFile({ status: undefined }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("type mismatch handling", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "score",
        operator: ">",
        value: 50,
      }
      const files = [
        createMockFile({ score: 60 }),
        createMockFile({ score: "high" }),
        createMockFile({ score: 40 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].frontmatter?.score, 60)
    })
  })

  describe("method calls", () => {
    test("containsAny with arrays", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "tags",
        method: "containsAny",
        args: ["typescript", "javascript"],
        negated: false,
      }
      const files = [
        createMockFile({ tags: ["typescript", "react"] }),
        createMockFile({ tags: ["python", "rust"] }),
        createMockFile({ tags: ["javascript", "node"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("containsAny with strings", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "description",
        method: "containsAny",
        args: ["test", "demo"],
        negated: false,
      }
      const files = [
        createMockFile({ description: "this is a test" }),
        createMockFile({ description: "production code" }),
        createMockFile({ description: "demo project" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("containsAll with arrays", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "tags",
        method: "containsAll",
        args: ["typescript", "react"],
        negated: false,
      }
      const files = [
        createMockFile({ tags: ["typescript", "react", "frontend"] }),
        createMockFile({ tags: ["typescript", "node"] }),
        createMockFile({ tags: ["react", "javascript"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("containsAll with strings", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "description",
        method: "containsAll",
        args: ["test", "unit"],
        negated: false,
      }
      const files = [
        createMockFile({ description: "unit test suite" }),
        createMockFile({ description: "integration test" }),
        createMockFile({ description: "test coverage" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("startsWith", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "title",
        method: "startsWith",
        args: ["intro"],
        negated: false,
      }
      const files = [
        createMockFile({ title: "introduction to typescript" }),
        createMockFile({ title: "advanced typescript" }),
        createMockFile({ title: "intro to react" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("endsWith", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "filename",
        method: "endsWith",
        args: [".test.ts"],
        negated: false,
      }
      const files = [
        createMockFile({ filename: "query.test.ts" }),
        createMockFile({ filename: "query.ts" }),
        createMockFile({ filename: "types.test.ts" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isEmpty with null", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "description",
        method: "isEmpty",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ description: "has content" }),
        createMockFile({ description: null }),
        createMockFile({}),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isEmpty with empty string", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "notes",
        method: "isEmpty",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ notes: "" }),
        createMockFile({ notes: "some notes" }),
        createMockFile({ notes: null }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isEmpty with empty array", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "tags",
        method: "isEmpty",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ tags: [] }),
        createMockFile({ tags: ["one"] }),
        createMockFile({}),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isType for null", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "value",
        method: "isType",
        args: ["null"],
        negated: false,
      }
      const files = [
        createMockFile({ value: null }),
        createMockFile({}),
        createMockFile({ value: "something" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isType for string", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "value",
        method: "isType",
        args: ["string"],
        negated: false,
      }
      const files = [
        createMockFile({ value: "text" }),
        createMockFile({ value: 123 }),
        createMockFile({ value: "another" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isType for number", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "value",
        method: "isType",
        args: ["number"],
        negated: false,
      }
      const files = [
        createMockFile({ value: 123 }),
        createMockFile({ value: "text" }),
        createMockFile({ value: 456 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isType for boolean", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "published",
        method: "isType",
        args: ["boolean"],
        negated: false,
      }
      const files = [
        createMockFile({ published: true }),
        createMockFile({ published: "yes" }),
        createMockFile({ published: false }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isType for array", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "value",
        method: "isType",
        args: ["array"],
        negated: false,
      }
      const files = [
        createMockFile({ value: [1, 2, 3] }),
        createMockFile({ value: "text" }),
        createMockFile({ value: [] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("isType for object", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "metadata",
        method: "isType",
        args: ["object"],
        negated: false,
      }
      const files = [
        createMockFile({ metadata: { key: "value" } }),
        createMockFile({ metadata: "text" }),
        createMockFile({ metadata: {} }),
        createMockFile({ metadata: [1, 2] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("negation prefix (!)", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "tags",
        method: "containsAny",
        args: ["deprecated"],
        negated: true,
      }
      const files = [
        createMockFile({ tags: ["active", "new"] }),
        createMockFile({ tags: ["deprecated"] }),
        createMockFile({ tags: ["featured"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })

  describe("global functions", () => {
    test("file.hasTag with single tag", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "file.hasTag",
        args: ["typescript"],
      }
      const files = [
        createMockFile({ tags: ["typescript", "react"] }),
        createMockFile({ tags: ["python"] }),
        createMockFile({ tags: ["typescript"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.hasTag with multiple tags (OR semantics)", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "file.hasTag",
        args: ["typescript", "javascript"],
      }
      const files = [
        createMockFile({ tags: ["typescript"] }),
        createMockFile({ tags: ["python"] }),
        createMockFile({ tags: ["javascript"] }),
        createMockFile({ tags: ["rust"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.inFolder", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "file.inFolder",
        args: ["library"],
      }
      const files = [
        createMockFile({}, "library/book1"),
        createMockFile({}, "posts/article1"),
        createMockFile({}, "library/subfolder/book2"),
        createMockFile({}, "thoughts/note"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.inFolder with trailing slash", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "file.inFolder",
        args: ["library/"],
      }
      const files = [
        createMockFile({}, "library/book1"),
        createMockFile({}, "posts/article1"),
        createMockFile({}, "library/subfolder/book2"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.hasProperty", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "file.hasProperty",
        args: ["author"],
      }
      const files = [
        createMockFile({ author: "john doe" }),
        createMockFile({ title: "test" }),
        createMockFile({ author: "jane smith" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.hasLink", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "file.hasLink",
        args: ["target-page"],
      }
      const files = [
        createMockFile({}, "page1", ["target-page", "other-page"]),
        createMockFile({}, "page2", ["some-page"]),
        createMockFile({}, "page3", ["target-page"]),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })

  describe("logical operators", () => {
    test("AND with multiple conditions", () => {
      const filter: BaseFilter = {
        type: "and",
        conditions: [
          {
            type: "comparison",
            property: "status",
            operator: "==",
            value: "published",
          },
          {
            type: "function",
            name: "file.hasTag",
            args: ["featured"],
          },
        ],
      }
      const files = [
        createMockFile({ status: "published", tags: ["featured"] }),
        createMockFile({ status: "published", tags: ["normal"] }),
        createMockFile({ status: "draft", tags: ["featured"] }),
        createMockFile({ status: "published", tags: ["featured", "trending"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("OR with multiple conditions", () => {
      const filter: BaseFilter = {
        type: "or",
        conditions: [
          {
            type: "comparison",
            property: "status",
            operator: "==",
            value: "published",
          },
          {
            type: "comparison",
            property: "status",
            operator: "==",
            value: "featured",
          },
        ],
      }
      const files = [
        createMockFile({ status: "published" }),
        createMockFile({ status: "draft" }),
        createMockFile({ status: "featured" }),
        createMockFile({ status: "archived" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("NOT negation", () => {
      const filter: BaseFilter = {
        type: "not",
        conditions: [
          {
            type: "comparison",
            property: "status",
            operator: "==",
            value: "draft",
          },
        ],
      }
      const files = [
        createMockFile({ status: "published" }),
        createMockFile({ status: "draft" }),
        createMockFile({ status: "archived" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("nested AND in OR", () => {
      const filter: BaseFilter = {
        type: "or",
        conditions: [
          {
            type: "and",
            conditions: [
              {
                type: "comparison",
                property: "status",
                operator: "==",
                value: "published",
              },
              {
                type: "function",
                name: "file.hasTag",
                args: ["featured"],
              },
            ],
          },
          {
            type: "comparison",
            property: "priority",
            operator: "==",
            value: "high",
          },
        ],
      }
      const files = [
        createMockFile({ status: "published", tags: ["featured"] }),
        createMockFile({ status: "draft", priority: "high" }),
        createMockFile({ status: "published", tags: ["normal"] }),
        createMockFile({ priority: "high" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 3)
    })

    test("complex nested structure", () => {
      const filter: BaseFilter = {
        type: "and",
        conditions: [
          {
            type: "or",
            conditions: [
              {
                type: "function",
                name: "file.hasTag",
                args: ["book"],
              },
              {
                type: "function",
                name: "file.inFolder",
                args: ["library"],
              },
            ],
          },
          {
            type: "not",
            conditions: [
              {
                type: "function",
                name: "file.hasTag",
                args: ["deprecated"],
              },
            ],
          },
        ],
      }
      const files = [
        createMockFile({ tags: ["book"] }, "posts/book1"),
        createMockFile({ tags: ["article"] }, "library/item"),
        createMockFile({ tags: ["book", "deprecated"] }, "posts/book2"),
        createMockFile({ tags: ["normal"] }, "library/book3"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 3)
    })
  })

  describe("date comparisons", () => {
    test("date string parsing and comparison", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "publishDate",
        operator: ">",
        value: "2023-01-01",
      }
      const files = [
        createMockFile({ publishDate: "2023-06-15" }),
        createMockFile({ publishDate: "2022-12-31" }),
        createMockFile({ publishDate: "2024-01-01" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("date comparison with Date objects", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "created",
        operator: ">=",
        value: "2023-01-01",
      }
      const files = [
        createMockFile({ created: new Date("2023-01-01") }),
        createMockFile({ created: new Date("2022-12-31") }),
        createMockFile({ created: new Date("2023-06-15") }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("invalid date strings", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "date",
        operator: ">",
        value: "2023-01-01",
      }
      const files = [
        createMockFile({ date: "invalid-date" }),
        createMockFile({ date: "2023-06-15" }),
        createMockFile({ date: "not-a-date" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })
  })

  describe("arithmetic expressions", () => {
    test("simple arithmetic expression", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "price * 1.13",
        operator: ">",
        value: 50,
        isExpression: true,
      }
      const files = [
        createMockFile({ price: 40 }),
        createMockFile({ price: 50 }),
        createMockFile({ price: 20 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].frontmatter?.price, 50)
    })

    test("arithmetic with addition", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "base + bonus",
        operator: ">=",
        value: 100,
        isExpression: true,
      }
      const files = [
        createMockFile({ base: 80, bonus: 20 }),
        createMockFile({ base: 70, bonus: 15 }),
        createMockFile({ base: 90, bonus: 10 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("arithmetic with subtraction", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "total - discount",
        operator: "<",
        value: 50,
        isExpression: true,
      }
      const files = [
        createMockFile({ total: 60, discount: 15 }),
        createMockFile({ total: 80, discount: 20 }),
        createMockFile({ total: 50, discount: 5 }),
      ]
      const result = evaluateFilter(filter, files)
      // 60-15=45 < 50 ✓, 80-20=60 < 50 ✗, 50-5=45 < 50 ✓
      assert.strictEqual(result.length, 2)
    })

    test("expression evaluation failure", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "price * 1.13",
        operator: ">",
        value: 50,
        isExpression: true,
      }
      const files = [createMockFile({ price: "not-a-number" }), createMockFile({ price: 60 })]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].frontmatter?.price, 60)
    })
  })

  describe("edge cases", () => {
    test("empty filter conditions", () => {
      const filter: BaseFilter = {
        type: "and",
        conditions: [],
      }
      const files = [createMockFile({ title: "test" }), createMockFile({ title: "test2" })]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("missing properties", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "nonexistent",
        operator: "==",
        value: "test",
      }
      const files = [createMockFile({ title: "test" }), createMockFile({ other: "value" })]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 0)
    })

    test("empty file list", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "status",
        operator: "==",
        value: "published",
      }
      const result = evaluateFilter(filter, [])
      assert.strictEqual(result.length, 0)
    })

    test("files without frontmatter", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "status",
        operator: "==",
        value: "published",
      }
      const files = [
        createMockFile({ status: "published" }),
        { slug: "test" as FullSlug, links: [], tags: [] } as QuartzPluginData,
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("array with single element", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "tags",
        operator: "contains",
        value: "test",
      }
      const files = [createMockFile({ tags: ["test"] })]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("method with no args", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "description",
        method: "isEmpty",
        args: [],
        negated: false,
      }
      const files = [createMockFile({ description: "" }), createMockFile({ description: "text" })]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })
  })

  describe("file system date properties", () => {
    test("file.ctime comparison", () => {
      const baseTime = new Date("2024-01-01").getTime()
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.ctime",
        operator: ">",
        value: baseTime,
      }
      const files = [
        {
          slug: "test1" as FullSlug,
          frontmatter: {},
          dates: {
            created: new Date("2024-01-15"),
            modified: new Date("2024-01-15"),
            published: new Date("2024-01-15"),
          },
        } as unknown as QuartzPluginData,
        {
          slug: "test2" as FullSlug,
          frontmatter: {},
          dates: {
            created: new Date("2023-12-15"),
            modified: new Date("2024-01-15"),
            published: new Date("2024-01-15"),
          },
        } as unknown as QuartzPluginData,
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].slug, "test1")
    })

    test("file.mtime comparison", () => {
      const baseTime = new Date("2024-01-10").getTime()
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.mtime",
        operator: ">=",
        value: baseTime,
      }
      const files = [
        {
          slug: "test1" as FullSlug,
          frontmatter: {},
          dates: {
            created: new Date("2024-01-01"),
            modified: new Date("2024-01-15"),
            published: new Date("2024-01-15"),
          },
        } as unknown as QuartzPluginData,
        {
          slug: "test2" as FullSlug,
          frontmatter: {},
          dates: {
            created: new Date("2024-01-01"),
            modified: new Date("2024-01-05"),
            published: new Date("2024-01-05"),
          },
        } as unknown as QuartzPluginData,
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].slug, "test1")
    })

    test("file.ctime with missing dates property", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.ctime",
        operator: ">",
        value: 0,
      }
      const files = [
        {
          slug: "test" as FullSlug,
          frontmatter: {},
        } as unknown as QuartzPluginData,
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 0)
    })

    test("file.mtime with missing dates property", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.mtime",
        operator: "<",
        value: Date.now(),
      }
      const files = [
        {
          slug: "test" as FullSlug,
          frontmatter: {},
        } as unknown as QuartzPluginData,
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 0)
    })
  })

  describe("file.* metadata properties", () => {
    test("file.name extracts filename without extension", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.name",
        operator: "==",
        value: "myfile",
      }
      const files = [
        createMockFile({}, "folder/myfile"),
        createMockFile({}, "folder/other"),
        createMockFile({}, "myfile"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.path returns full slug", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.path",
        operator: "==",
        value: "folder/subdir/file",
      }
      const files = [
        createMockFile({}, "folder/subdir/file"),
        createMockFile({}, "folder/file"),
        createMockFile({}, "other/subdir/file"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("file.folder extracts parent folder", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.folder",
        operator: "==",
        value: "library",
      }
      const files = [
        createMockFile({}, "library/book1"),
        createMockFile({}, "posts/article"),
        createMockFile({}, "library/book2"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.ext returns file extension", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.ext",
        operator: "==",
        value: "md",
      }
      const files = [
        createMockFile({}, "file.md"),
        createMockFile({}, "other.txt"),
        createMockFile({}, "doc"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.link returns slug", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.link",
        operator: "contains",
        value: "library",
      }
      const files = [
        createMockFile({}, "library/book"),
        createMockFile({}, "posts/article"),
        createMockFile({}, "library/notes"),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.outlinks returns array of outgoing links", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.outlinks",
        operator: "contains",
        value: "target-page",
      }
      const files = [
        createMockFile({}, "page1", ["target-page", "other"]),
        createMockFile({}, "page2", ["different"]),
        createMockFile({}, "page3", ["target-page"]),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("file.inlinks returns array of incoming backlinks", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.inlinks",
        operator: "contains",
        value: "page1",
      }
      const files = [
        createMockFile({}, "page1", ["page2"]),
        createMockFile({}, "page2", ["page1", "page3"]),
        createMockFile({}, "page3", ["page2"]),
        createMockFile({}, "page4", ["page1"]),
      ]
      const result = evaluateFilter(filter, files)
      // page2 links to page1, page4 links to page1
      // so files that have page1 in their inlinks should be the files that are pointed to by page1
      // but the filter checks if the file's inlinks contains "page1"
      // that means we want files where page1 links to them
      // page1 links to page2, so page2 should match
      assert.strictEqual(result.length, 1)
      assert.strictEqual(result[0].slug, "page2")
    })

    test("file.aliases returns frontmatter aliases", () => {
      const filter: BaseFilter = {
        type: "comparison",
        property: "file.aliases",
        operator: "contains",
        value: "alternative-name",
      }
      const files = [
        createMockFile({ aliases: ["alternative-name", "other"] }),
        createMockFile({ aliases: ["something"] }),
        createMockFile({ aliases: ["alternative-name"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })

  describe("toString() method", () => {
    test("toString() on existing property", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "title",
        method: "toString",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ title: "test" }),
        createMockFile({}),
        createMockFile({ title: "another" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("toString() on file.name", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "file.name",
        method: "toString",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({}, "folder/file1"),
        createMockFile({}, "folder/file2"),
        createMockFile({}, ""),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 3)
    })

    test("negated toString()", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "description",
        method: "toString",
        args: [],
        negated: true,
      }
      const files = [
        createMockFile({ description: "text" }),
        createMockFile({}),
        createMockFile({ description: null }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })

  describe("inline boolean operators", () => {
    test("simple AND with &&", () => {
      const filter: BaseFilter = {
        type: "and",
        conditions: [
          {
            type: "comparison",
            property: "status",
            operator: "==",
            value: "published",
          },
          {
            type: "comparison",
            property: "rating",
            operator: ">",
            value: 4,
          },
        ],
      }
      const files = [
        createMockFile({ status: "published", rating: 5 }),
        createMockFile({ status: "published", rating: 3 }),
        createMockFile({ status: "draft", rating: 5 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("simple OR with ||", () => {
      const filter: BaseFilter = {
        type: "or",
        conditions: [
          {
            type: "comparison",
            property: "priority",
            operator: "==",
            value: "high",
          },
          {
            type: "comparison",
            property: "urgent",
            operator: "==",
            value: true,
          },
        ],
      }
      const files = [
        createMockFile({ priority: "high", urgent: false }),
        createMockFile({ priority: "low", urgent: true }),
        createMockFile({ priority: "low", urgent: false }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("complex expression with && and ||", () => {
      const filter: BaseFilter = {
        type: "or",
        conditions: [
          {
            type: "and",
            conditions: [
              {
                type: "comparison",
                property: "status",
                operator: "==",
                value: "done",
              },
              {
                type: "comparison",
                property: "archived",
                operator: "==",
                value: false,
              },
            ],
          },
          {
            type: "comparison",
            property: "status",
            operator: "==",
            value: "archived",
          },
        ],
      }
      const files = [
        createMockFile({ status: "done", archived: false }),
        createMockFile({ status: "done", archived: true }),
        createMockFile({ status: "archived", archived: true }),
        createMockFile({ status: "in-progress", archived: false }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("parentheses grouping", () => {
      const filter: BaseFilter = {
        type: "and",
        conditions: [
          {
            type: "or",
            conditions: [
              {
                type: "comparison",
                property: "status",
                operator: "==",
                value: "done",
              },
              {
                type: "comparison",
                property: "status",
                operator: "==",
                value: "archived",
              },
            ],
          },
          {
            type: "comparison",
            property: "deleted",
            operator: "==",
            value: false,
          },
        ],
      }
      const files = [
        createMockFile({ status: "done", deleted: false }),
        createMockFile({ status: "archived", deleted: false }),
        createMockFile({ status: "done", deleted: true }),
        createMockFile({ status: "in-progress", deleted: false }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })

  describe("string methods", () => {
    test("lower() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "title",
        method: "lower",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ title: "HELLO" }),
        createMockFile({ title: 123 }),
        createMockFile({ title: "world" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("upper() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "name",
        method: "upper",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ name: "test" }),
        createMockFile({ name: null }),
        createMockFile({ name: "another" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("slice() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "text",
        method: "slice",
        args: ["0", "5"],
        negated: false,
      }
      const files = [
        createMockFile({ text: "hello world" }),
        createMockFile({ text: 123 }),
        createMockFile({ text: "test" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("split() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "csv",
        method: "split",
        args: [","],
        negated: false,
      }
      const files = [
        createMockFile({ csv: "a,b,c" }),
        createMockFile({ csv: 123 }),
        createMockFile({ csv: "x,y,z" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("trim() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "text",
        method: "trim",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ text: "  hello  " }),
        createMockFile({ text: null }),
        createMockFile({ text: "world" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("replace() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "text",
        method: "replace",
        args: ["old", "new"],
        negated: false,
      }
      const files = [
        createMockFile({ text: "old text" }),
        createMockFile({ text: 123 }),
        createMockFile({ text: "more old stuff" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })

  describe("number methods", () => {
    test("abs() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "value",
        method: "abs",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ value: -5 }),
        createMockFile({ value: "not a number" }),
        createMockFile({ value: 10 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("ceil() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "price",
        method: "ceil",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ price: 9.99 }),
        createMockFile({ price: "text" }),
        createMockFile({ price: 15.5 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("floor() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "score",
        method: "floor",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ score: 7.8 }),
        createMockFile({ score: null }),
        createMockFile({ score: 3.2 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("round() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "rating",
        method: "round",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ rating: 4.6 }),
        createMockFile({ rating: "high" }),
        createMockFile({ rating: 3.2 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("toFixed() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "percentage",
        method: "toFixed",
        args: ["2"],
        negated: false,
      }
      const files = [
        createMockFile({ percentage: 0.123456 }),
        createMockFile({ percentage: "text" }),
        createMockFile({ percentage: 0.987654 }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })

  describe("duration function", () => {
    test("duration with milliseconds number", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "duration",
        args: ["86400000"],
      }
      const files = [createMockFile({})]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("duration with string '7 days'", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "duration",
        args: ["7 days"],
      }
      const files = [createMockFile({})]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("duration with string '3 hours'", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "duration",
        args: ["3 hours"],
      }
      const files = [createMockFile({})]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("duration with combined string '1 day 12 hours'", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "duration",
        args: ["1 day 12 hours"],
      }
      const files = [createMockFile({})]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })

    test("duration with shorthand '2h 30m'", () => {
      const filter: BaseFilter = {
        type: "function",
        name: "duration",
        args: ["2h 30m"],
      }
      const files = [createMockFile({})]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 1)
    })
  })

  describe("array methods", () => {
    test("join() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "items",
        method: "join",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ items: ["a", "b", "c"] }),
        createMockFile({ items: "not an array" }),
        createMockFile({ items: [1, 2, 3] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("reverse() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "list",
        method: "reverse",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ list: [1, 2, 3] }),
        createMockFile({ list: null }),
        createMockFile({ list: ["x", "y", "z"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("sort() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "numbers",
        method: "sort",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ numbers: [3, 1, 2] }),
        createMockFile({ numbers: "not array" }),
        createMockFile({ numbers: [9, 5, 7] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("unique() method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "tags",
        method: "unique",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ tags: ["a", "b", "a"] }),
        createMockFile({ tags: {} }),
        createMockFile({ tags: ["x", "y", "x", "z"] }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })

    test("length method", () => {
      const filter: BaseFilter = {
        type: "method",
        property: "items",
        method: "length",
        args: [],
        negated: false,
      }
      const files = [
        createMockFile({ items: [1, 2, 3] }),
        createMockFile({ items: null }),
        createMockFile({ items: "text" }),
      ]
      const result = evaluateFilter(filter, files)
      assert.strictEqual(result.length, 2)
    })
  })
})
