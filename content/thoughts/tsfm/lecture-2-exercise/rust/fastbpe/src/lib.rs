use pyo3::prelude::*;

mod bpe;
use bpe::compile_with_fallback;
use hashbrown::HashMap as HbMap;
use rayon::prelude::*;

#[pyclass]
pub struct Tokenizer {
    inner: bpe::PreTrainedBPE,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    #[pyo3(signature = (paths, num_merges, num_threads=None, pattern=None))]
    fn train_from_files(
        paths: Vec<String>,
        num_merges: usize,
        num_threads: Option<usize>,
        pattern: Option<String>,
    ) -> Self {
        let trained = bpe::train_from_files(paths, num_merges, pattern, num_threads);
        Self { inner: trained }
    }

    #[staticmethod]
    #[pyo3(signature = (texts, num_merges, num_threads=None, pattern=None))]
    fn train_from_texts(
        texts: Vec<String>,
        num_merges: usize,
        num_threads: Option<usize>,
        pattern: Option<String>,
    ) -> Self {
        if let Some(n) = num_threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
        let regex = compile_with_fallback(pattern.as_deref());
        let counts = {
            texts
                .into_par_iter()
                .map(|t| {
                    let mut local: HbMap<String, u32> = HbMap::new();
                    for m in regex.find_iter(&t) {
                        *local.entry(m.as_str().to_string()).or_insert(0) += 1;
                    }
                    local
                })
                .reduce(
                    HbMap::new,
                    |mut a, b| {
                        for (k, v) in b {
                            *a.entry(k).or_insert(0) += v;
                        }
                        a
                    },
                )
        };
        let (merges_seq, id_to_bytes) = bpe::train_bpe_internal_private(counts, num_merges);
        let mut merges: hashbrown::HashMap<(u32, u32), u32> = hashbrown::HashMap::new();
        for (pair, nid) in merges_seq.iter() {
            merges.insert(*pair, *nid);
        }
        let ranks = bpe::build_ranks_private(&merges_seq);
        Self {
            inner: bpe::PreTrainedBPE {
                merges,
                ranks,
                id_to_bytes,
                merges_ordered: merges_seq,
                pattern: regex,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (dir, pattern=None))]
    fn from_pretrained(dir: String, pattern: Option<String>) -> PyResult<Self> {
        let trained = bpe::load_from_dir(dir, pattern)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: trained })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        bpe::encode_str(&self.inner, text)
    }
    fn decode(&self, ids: Vec<u32>) -> String {
        bpe::decode_ids(&self.inner, &ids)
    }

    fn merges_list(&self) -> Vec<(u32, u32, u32)> {
        self.inner
            .merges_ordered
            .iter()
            .map(|(p, n)| (p.0, p.1, *n))
            .collect()
    }
    fn vocab_pairs(&self) -> Vec<(Vec<u32>, u32)> {
        let mut out = Vec::with_capacity(256 + self.inner.merges_ordered.len());
        for i in 0u32..256 {
            out.push((vec![i], i));
        }
        for (pair, nid) in self.inner.merges_ordered.iter() {
            out.push((vec![pair.0, pair.1], *nid));
        }
        out
    }
}

#[pymodule]
fn _core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
