use crate::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

/// Deterministic bag-of-words embedder for use in tests and CI only.
///
/// Each token is hashed to a vector dimension. NOT semantic — "read the file"
/// and "check the document" score zero similarity. Use this when you need a
/// fast, no-download fake to test the sync/store pipeline. Do NOT use in
/// production: `similar()` and the vector leg of `search()` become meaningless,
/// and the RRF hybrid degrades to keyword+keyword.
pub struct HashEmbedder {
    dimensions: usize,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    fn vectorize(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.dimensions];

        for token in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|token| !token.is_empty())
        {
            let mut hasher = DefaultHasher::new();
            token.to_lowercase().hash(&mut hasher);
            let index = (hasher.finish() as usize) % self.dimensions;
            vector[index] += 1.0;
        }

        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }

        vector
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.vectorize(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| self.vectorize(text)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(feature = "local-embed")]
mod local {
    use super::{Embedder, Result};
    use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
    use std::sync::Mutex;

    pub struct FastEmbedder {
        model: Mutex<TextEmbedding>,
        dimensions: usize,
    }

    impl FastEmbedder {
        pub fn new() -> Result<Self> {
            let model = TextEmbedding::try_new(
                TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_show_download_progress(true),
            )
            .map_err(|err| crate::AgentSessionsError::Embedder(err.to_string()))?;

            Ok(Self {
                model: Mutex::new(model),
                dimensions: 384,
            })
        }
    }

    impl Embedder for FastEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let mut model = self
                .model
                .lock()
                .map_err(|err| crate::AgentSessionsError::Embedder(err.to_string()))?;
            let outputs = model
                .embed(vec![text], None)
                .map_err(|err| crate::AgentSessionsError::Embedder(err.to_string()))?;
            outputs.into_iter().next().ok_or_else(|| {
                crate::AgentSessionsError::Embedder("fastembed returned no vectors".into())
            })
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut model = self
                .model
                .lock()
                .map_err(|err| crate::AgentSessionsError::Embedder(err.to_string()))?;
            model
                .embed(texts.to_vec(), None)
                .map_err(|err| crate::AgentSessionsError::Embedder(err.to_string()))
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }
    }

    pub fn load() -> crate::Result<Box<dyn Embedder>> {
        Ok(Box::new(FastEmbedder::new()?))
    }
}

#[cfg(feature = "local-embed")]
pub use local::FastEmbedder;

/// Load the default production embedder. Fails loudly if the model can't initialise
/// rather than silently degrading to a non-semantic fallback.
pub fn default_embedder() -> crate::Result<Box<dyn Embedder>> {
    #[cfg(feature = "local-embed")]
    return local::load();

    #[cfg(not(feature = "local-embed"))]
    Err(crate::AgentSessionsError::Embedder(
        "no embedder available: build with feature = \"local-embed\" or supply one via SessionIndex::from_parts".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedder_is_deterministic() {
        let embedder = HashEmbedder::new(16);
        let a = embedder.embed("hello world").unwrap();
        let b = embedder.embed("hello world").unwrap();
        assert_eq!(a, b);
    }
}
