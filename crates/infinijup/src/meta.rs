use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum SchemaVersion {
  V1,
}

pub struct Db {
  conn: Connection,
}

impl Db {
  pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
    let mut conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", &"WAL")?;
    conn.pragma_update(None, "synchronous", &"NORMAL")?;
    Ok(Self { conn })
  }

  pub fn init_schema(&mut self) -> Result<()> {
    self.conn.execute_batch(SCHEMA_V1).context("init schema")?;
    Ok(())
  }
}

// Minimal, opinionated schema for single-node devbox.
const SCHEMA_V1: &str = r#"
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS snapshots (
  id            TEXT PRIMARY KEY,          -- uuid
  parent_id     TEXT REFERENCES snapshots(id) ON DELETE SET NULL,
  created_at    TEXT NOT NULL,
  kernel_id     TEXT,
  note          TEXT,
  tags_json     TEXT,
  driver_ver    TEXT,
  cuda_ver      TEXT,
  gpu_cc        TEXT,
  env_hash      TEXT,
  manifest_id   TEXT
);

CREATE TABLE IF NOT EXISTS manifests (
  id            TEXT PRIMARY KEY,          -- content-hash of manifest
  snapshot_id   TEXT NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
  size_bytes    INTEGER NOT NULL,
  chunks_count  INTEGER NOT NULL,
  meta_json     TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  digest        TEXT PRIMARY KEY,          -- BLAKE3 hex
  size_bytes    INTEGER NOT NULL,
  storage_class TEXT NOT NULL,             -- local|remote|both
  local_path    TEXT,
  remote_uri    TEXT,
  refcount      INTEGER NOT NULL DEFAULT 0,
  last_access   TEXT
);

CREATE TABLE IF NOT EXISTS manifest_chunks (
  manifest_id   TEXT NOT NULL REFERENCES manifests(id) ON DELETE CASCADE,
  ord           INTEGER NOT NULL,
  digest        TEXT NOT NULL REFERENCES chunks(digest) ON DELETE RESTRICT,
  offset        INTEGER NOT NULL,
  length        INTEGER NOT NULL,
  PRIMARY KEY (manifest_id, ord)
);

CREATE INDEX IF NOT EXISTS idx_chunks_refcount ON chunks(refcount);
CREATE INDEX IF NOT EXISTS idx_snapshots_parent ON snapshots(parent_id);

-- simple kv for settings
CREATE TABLE IF NOT EXISTS kv (
  k TEXT PRIMARY KEY,
  v TEXT
);
"#;

