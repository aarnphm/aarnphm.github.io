CREATE TABLE `comments` (
	`id` text PRIMARY KEY NOT NULL,
	`pageId` text NOT NULL,
	`parentId` text,
	`anchorHash` text NOT NULL,
	`anchorStart` integer NOT NULL,
	`anchorEnd` integer NOT NULL,
	`anchorText` text NOT NULL,
	`content` text NOT NULL,
	`author` text NOT NULL,
	`createdAt` integer NOT NULL,
	`updatedAt` integer,
	`deletedAt` integer
);
