CREATE TABLE `arena_read_links` (
  `subject` text NOT NULL,
  `article_id` text NOT NULL,
  `read_at` integer,
  `updated_at` integer NOT NULL,
  `revision` integer NOT NULL,
  PRIMARY KEY (`subject`, `article_id`),
  CONSTRAINT `arena_read_links_revision` CHECK (`revision` > 0)
);
--> statement-breakpoint
CREATE TABLE `arena_notes` (
  `subject` text NOT NULL,
  `note_id` text NOT NULL,
  `article_id` text NOT NULL,
  `source_url` text NOT NULL,
  `body` text NOT NULL,
  `snapshot_id` text,
  `quote` text,
  `occurrence` text,
  `created_at` integer NOT NULL,
  `updated_at` integer NOT NULL,
  `revision` integer NOT NULL,
  `ready_revision` integer,
  `exported_revision` integer,
  `export_receipt` text,
  `deleted_at` integer,
  PRIMARY KEY (`subject`, `note_id`),
  CONSTRAINT `arena_notes_revision` CHECK (`revision` > 0),
  CONSTRAINT `arena_notes_ready_revision` CHECK (`ready_revision` IS NULL OR `ready_revision` = `revision`),
  CONSTRAINT `arena_notes_exported_revision` CHECK (`exported_revision` IS NULL OR (`exported_revision` > 0 AND `exported_revision` <= `revision`))
);
--> statement-breakpoint
CREATE INDEX `idx_arena_notes_article` ON `arena_notes` (`subject`, `article_id`, `created_at`);
--> statement-breakpoint
CREATE INDEX `idx_arena_notes_updated` ON `arena_notes` (`subject`, `updated_at`);
