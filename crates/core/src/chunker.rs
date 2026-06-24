use text_splitter::TextSplitter;

pub trait Chunker: Send + Sync {
    /// Split text into chunks. Deterministic - same input always produces the same splits.
    /// Empty or whitespace-only text returns an empty list.
    fn chunk(&self, text: &str) -> Vec<String>;
}

/// Character-based splitter, 512-char max.
pub struct DefaultChunker {
    max_chars: usize,
}

impl Default for DefaultChunker {
    fn default() -> Self {
        Self { max_chars: 512 }
    }
}

impl Chunker for DefaultChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let text = text.trim();
        if text.is_empty() {
            return vec![];
        }

        TextSplitter::new(self.max_chars)
            .chunks(text)
            .map(str::to_string)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        let chunker = DefaultChunker::default();
        assert_eq!(chunker.chunk("hello world"), vec!["hello world"]);
    }

    #[test]
    fn empty_text_returns_empty() {
        let chunker = DefaultChunker::default();
        assert!(chunker.chunk("  ").is_empty());
    }

    #[test]
    fn long_text_splits_into_multiple_chunks() {
        let chunker = DefaultChunker { max_chars: 20 };
        let text = "word ".repeat(20);
        let chunks = chunker.chunk(&text);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.len() <= 25, "chunk too long: {}", chunk.len());
        }
    }
}
