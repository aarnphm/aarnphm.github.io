import { sqliteTable, text, integer, index } from "drizzle-orm/sqlite-core"

export const comments = sqliteTable(
  "comments",
  {
    id: text("id").primaryKey(),
    pageId: text("pageId").notNull(),
    parentId: text("parentId"),
    anchorHash: text("anchorHash").notNull(),
    anchorStart: integer("anchorStart").notNull(),
    anchorEnd: integer("anchorEnd").notNull(),
    anchorText: text("anchorText").notNull(),
    content: text("content").notNull(),
    author: text("author").notNull(),
    createdAt: integer("createdAt").notNull(),
    updatedAt: integer("updatedAt"),
    deletedAt: integer("deletedAt"),
    anchor: text("anchor"),
    orphaned: integer("orphaned", { mode: "boolean" }),
    lastRecoveredAt: integer("lastRecoveredAt"),
  },
  (table) => [
    index("idx_comments_page").on(table.pageId, table.createdAt),
    index("idx_comments_parent").on(table.parentId),
    index("idx_comments_hash").on(table.anchorHash),
  ],
)
