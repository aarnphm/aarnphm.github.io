CREATE TABLE `flashcard_review_log` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`login` text NOT NULL,
	`card_id` text NOT NULL,
	`deck_slug` text NOT NULL,
	`grade` integer NOT NULL,
	`reviewed_at` integer NOT NULL,
	`prior_state` integer,
	`prior_stability` real,
	`prior_difficulty` real,
	`prior_due` integer,
	`prior_reps` integer,
	`prior_lapses` integer,
	`prior_learning_steps` integer,
	`prior_reviewed_at` integer
);
--> statement-breakpoint
CREATE INDEX `idx_fc_log_card` ON `flashcard_review_log` (`login`,`card_id`,`reviewed_at`);--> statement-breakpoint
CREATE INDEX `idx_fc_deck` ON `flashcard_reviews` (`login`,`deck_slug`);