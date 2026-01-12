CREATE INDEX `idx_comments_page` ON `comments` (`pageId`,`createdAt`);--> statement-breakpoint
CREATE INDEX `idx_comments_parent` ON `comments` (`parentId`);--> statement-breakpoint
CREATE INDEX `idx_comments_hash` ON `comments` (`anchorHash`);