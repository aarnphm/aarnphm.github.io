use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod bpe;

#[pyclass]
pub struct Tokenizer {
    inner: bpe::PreTrainedBPE,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    #[pyo3(signature = (dir))]
    fn from_pretrained(dir: String) -> PyResult<Self> {
        let trained = bpe::load_from_dir(dir)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: trained })
    }

    fn encode(&self, text: &str) -> Vec<u32> { bpe::encode_str(&self.inner, text) }
    fn encode_bytes<'py>(&self, py: Python<'py>, data: &Bound<'py, PyBytes>) -> Vec<u32> {
        // SAFETY: Holding GIL only to access bytes; merge is pure Rust
        let slice = data.as_bytes();
        // Release GIL during compute-heavy merge
        py.allow_threads(|| bpe::encode_bytes(&self.inner, slice))
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
