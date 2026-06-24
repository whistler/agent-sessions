use crate::Result;

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

/// Stub embedder that returns zero vectors. Replace with a real model.
pub struct DefaultEmbedder {
    dimensions: usize,
}

impl DefaultEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Default for DefaultEmbedder {
    fn default() -> Self {
        Self::new(384)
    }
}

impl Embedder for DefaultEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; self.dimensions])
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0f32; self.dimensions]).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
