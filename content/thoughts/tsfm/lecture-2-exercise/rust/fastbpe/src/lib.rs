use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod bpe;

fn value_error(error: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

#[pyclass]
pub struct Tokenizer {
    inner: bpe::PreTrainedBPE,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn from_pretrained(dir: String) -> PyResult<Self> {
        Ok(Self {
            inner: bpe::load_from_dir(dir).map_err(value_error)?,
        })
    }

    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.detach(|| bpe::encode_str(&self.inner, text))
            .map_err(value_error)
    }

    fn encode_bytes(&self, py: Python<'_>, data: &Bound<'_, PyBytes>) -> Vec<u32> {
        let slice = data.as_bytes();
        py.detach(|| bpe::encode_bytes(&self.inner, slice))
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        bpe::decode_ids(&self.inner, &ids).map_err(value_error)
    }

    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bpe::decode_bytes(&self.inner, &ids).map_err(value_error)?;
        Ok(PyBytes::new(py, &bytes))
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
        for (pair, nid) in &self.inner.merges_ordered {
            out.push((vec![pair.0, pair.1], *nid));
        }
        out
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
