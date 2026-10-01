CREATE TABLE `pdf_marks` (
  `mark_id` text PRIMARY KEY NOT NULL,
  `owner` text NOT NULL,
  `doc` text NOT NULL,
  `src` text NOT NULL,
  `kind` text NOT NULL,
  `page` integer NOT NULL,
  `target` text NOT NULL,
  `body` text DEFAULT '' NOT NULL,
  `visibility` text NOT NULL,
  `revision` integer NOT NULL,
  `created_at` integer NOT NULL,
  `updated_at` integer NOT NULL,
  `deleted_at` integer,
  CONSTRAINT `pdf_marks_kind` CHECK (`kind` IN ('mark', 'question', 'contra')),
  CONSTRAINT `pdf_marks_visibility` CHECK (`visibility` IN ('public', 'private')),
  CONSTRAINT `pdf_marks_revision` CHECK (`revision` > 0)
);
--> statement-breakpoint
CREATE INDEX `idx_pdf_marks_doc` ON `pdf_marks` (`doc`, `deleted_at`, `visibility`, `page`);
--> statement-breakpoint
CREATE INDEX `idx_pdf_marks_src` ON `pdf_marks` (`owner`, `src`, `deleted_at`);
