import test, { describe } from "node:test"
import assert from "node:assert"
import { FilePath, slugifyFilePath, joinSegments, FullSlug } from "../../util/path"

describe("base emitter slug generation", () => {
  describe("view slug generation", () => {
    test("first view uses baseSlug", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewIndex = 0
      const viewName = "philosophy"

      const viewSlug = viewIndex === 0 ? baseSlug : joinSegments(baseSlug, viewName)

      assert.strictEqual(viewSlug, "antilibrary")
    })

    test("second view uses baseSlug/slugified-name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "philosophy"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/philosophy")
    })

    test("view name with spaces is slugified", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "top of mind"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/top-of-mind")
    })

    test("view name with multiple spaces", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "work in progress"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/work-in-progress")
    })

    test("view name with special characters", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "books & articles"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      // & becomes -and- after spaces become -, resulting in --and--
      assert.strictEqual(viewSlug, "antilibrary/books--and--articles")
    })

    test("view name with percent sign", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "100% complete"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/100-percent-complete")
    })

    test("view name with question mark", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "what's next?"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      // question marks are removed
      assert.strictEqual(viewSlug, "antilibrary/what's-next")
    })

    test("view name with hash", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "c# books"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      // hashes are removed
      assert.strictEqual(viewSlug, "antilibrary/c-books")
    })

    test("simple single-word view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "philosophy"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/philosophy")
    })

    test("view name: wip", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "wip"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/wip")
    })

    test("view name: dynalist", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "dynalist"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/dynalist")
    })

    test("view name: all", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "all"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/all")
    })

    test("nested base path", () => {
      const baseSlug = "projects/reading" as FullSlug
      const viewName = "in progress"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "projects/reading/in-progress")
    })

    test("multiple views with different names", () => {
      const baseSlug = "antilibrary" as FullSlug
      const views = [
        { name: "philosophy", index: 0 },
        { name: "top of mind", index: 1 },
        { name: "wip", index: 2 },
        { name: "all", index: 3 },
        { name: "dynalist", index: 4 },
      ]

      const slugs = views.map((view) => {
        if (view.index === 0) {
          return baseSlug
        }
        const slugifiedName = slugifyFilePath((view.name + ".tmp") as FilePath, true)
        return joinSegments(baseSlug, slugifiedName) as FullSlug
      })

      assert.strictEqual(slugs[0], "antilibrary")
      assert.strictEqual(slugs[1], "antilibrary/top-of-mind")
      assert.strictEqual(slugs[2], "antilibrary/wip")
      assert.strictEqual(slugs[3], "antilibrary/all")
      assert.strictEqual(slugs[4], "antilibrary/dynalist")
    })
  })

  describe("view metadata generation", () => {
    test("first view metadata", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "philosophy"
      const viewType = "table"
      const viewIndex = 0

      const metadata = {
        name: viewName,
        type: viewType,
        slug: viewIndex === 0 ? baseSlug : (joinSegments(baseSlug, viewName) as FullSlug),
      }

      assert.strictEqual(metadata.name, "philosophy")
      assert.strictEqual(metadata.type, "table")
      assert.strictEqual(metadata.slug, "antilibrary")
    })

    test("subsequent view metadata with space in name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "top of mind"
      const viewType = "table"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const metadata = {
        name: viewName,
        type: viewType,
        slug: joinSegments(baseSlug, slugifiedName) as FullSlug,
      }

      assert.strictEqual(metadata.name, "top of mind")
      assert.strictEqual(metadata.type, "table")
      assert.strictEqual(metadata.slug, "antilibrary/top-of-mind")
    })

    test("list view type metadata", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "dynalist"
      const viewType = "list"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const metadata = {
        name: viewName,
        type: viewType,
        slug: joinSegments(baseSlug, slugifiedName) as FullSlug,
      }

      assert.strictEqual(metadata.type, "list")
      assert.strictEqual(metadata.slug, "antilibrary/dynalist")
    })

    test("all views metadata array", () => {
      const baseSlug = "antilibrary" as FullSlug
      const views = [
        { name: "philosophy", type: "table" as const },
        { name: "top of mind", type: "table" as const },
        { name: "wip", type: "table" as const },
        { name: "all", type: "table" as const },
        { name: "dynalist", type: "list" as const },
      ]

      const allViewsMetadata = views.map((v, idx) => {
        if (idx === 0) {
          return {
            name: v.name,
            type: v.type,
            slug: baseSlug,
          }
        }
        const slugifiedName = slugifyFilePath((v.name + ".tmp") as FilePath, true)
        return {
          name: v.name,
          type: v.type,
          slug: joinSegments(baseSlug, slugifiedName) as FullSlug,
        }
      })

      assert.strictEqual(allViewsMetadata.length, 5)
      assert.strictEqual(allViewsMetadata[0].slug, "antilibrary")
      assert.strictEqual(allViewsMetadata[1].slug, "antilibrary/top-of-mind")
      assert.strictEqual(allViewsMetadata[2].slug, "antilibrary/wip")
      assert.strictEqual(allViewsMetadata[3].slug, "antilibrary/all")
      assert.strictEqual(allViewsMetadata[4].slug, "antilibrary/dynalist")
    })
  })

  describe("edge cases", () => {
    test("empty view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = ""

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      // empty string should remain empty after slugification
      assert.ok(viewSlug.startsWith("antilibrary"))
    })

    test("view name with only spaces", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "   "

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      // spaces should be replaced with dashes
      assert.ok(viewSlug.includes("antilibrary"))
    })

    test("view name with trailing spaces", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "philosophy  "

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/philosophy--")
    })

    test("view name with leading spaces", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "  philosophy"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/--philosophy")
    })

    test("unicode characters in view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "café ☕"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      // unicode should be preserved
      assert.ok(viewSlug.includes("antilibrary"))
    })

    test("numbers in view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "top 10 books"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/top-10-books")
    })

    test("hyphen in view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "sci-fi"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/sci-fi")
    })

    test("underscore in view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "work_in_progress"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.strictEqual(viewSlug, "antilibrary/work_in_progress")
    })

    test("mixed case view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName = "ToP oF MiNd"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      // case should be preserved
      assert.strictEqual(viewSlug, "antilibrary/ToP-oF-MiNd")
    })

    test("very long view name", () => {
      const baseSlug = "antilibrary" as FullSlug
      const viewName =
        "this is a very long view name that should still be slugified correctly without any issues"

      const slugifiedName = slugifyFilePath((viewName + ".tmp") as FilePath, true)
      const viewSlug = joinSegments(baseSlug, slugifiedName) as FullSlug

      assert.ok(viewSlug.startsWith("antilibrary/this-is-a-very-long"))
      assert.ok(viewSlug.includes("slugified"))
    })
  })
})
