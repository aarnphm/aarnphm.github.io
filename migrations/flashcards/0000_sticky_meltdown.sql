CREATE TABLE `flashcard_reviews` (
	`login` text NOT NULL,
	`card_id` text NOT NULL,
	`deck_slug` text NOT NULL,
	`stability` real NOT NULL,
	`difficulty` real NOT NULL,
	`due` integer NOT NULL,
	`state` integer NOT NULL,
	`reps` integer NOT NULL,
	`lapses` integer NOT NULL,
	`learning_steps` integer NOT NULL,
	`last_reviewed_at` integer NOT NULL,
	PRIMARY KEY(`login`, `card_id`)
);
--> statement-breakpoint
CREATE INDEX `idx_fc_due` ON `flashcard_reviews` (`login`,`due`);