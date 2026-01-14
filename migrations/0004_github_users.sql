CREATE TABLE `github_users` (
  `login` text PRIMARY KEY NOT NULL,
  `display_name` text,
  `avatar_url` text,
  `last_seen_at` integer NOT NULL,
  `first_seen_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_github_users_last_seen` ON `github_users` (`last_seen_at`);
