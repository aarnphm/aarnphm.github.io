ALTER TABLE `comments` ADD `anchor` text;--> statement-breakpoint
ALTER TABLE `comments` ADD `orphaned` integer;--> statement-breakpoint
ALTER TABLE `comments` ADD `lastRecoveredAt` integer;