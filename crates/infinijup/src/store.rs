use anyhow::Result;
use blake3::Hasher;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub type SnapshotId = String; // uuid
pub type ChunkDigest = String; // blake3 hex

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
  pub id: String,
  pub size_bytes: u64,
  pub chunks: Vec<ManifestChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestChunk {
  pub ord: u32,
  pub digest: ChunkDigest,
  pub offset: u64,
  pub length: u64,
}

pub struct ChunkStore {
  root: PathBuf,
}

impl ChunkStore {
  pub fn new<P: AsRef<Path>>(root: P) -> Self { Self { root: root.as_ref().to_path_buf() } }

  pub fn put_chunk(&self, data: &[u8]) -> Result<ChunkDigest> {
    let mut h = Hasher::new();
    h.update(data);
    let digest = h.finalize().to_hex().to_string();
    let path = self.chunk_path(&digest);
    if !path.exists() { fs::create_dir_all(path.parent().unwrap())?; fs::write(&path, data)?; }
    Ok(digest)
  }

  pub fn get_chunk(&self, digest: &str) -> Result<Bytes> {
    let path = self.chunk_path(digest);
    let data = fs::read(path)?;
    Ok(Bytes::from(data))
  }

  fn chunk_path(&self, digest: &str) -> PathBuf {
    let (a, b) = digest.split_at(2);
    self.root.join("chunks").join(a).join(b)
  }
}

