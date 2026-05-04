//! perf - performance-critical primitives.
//!
//! Provides parallel cosine similarity, vector L2 normalization, and a
//! CRC32 checksum suitable for memory shard validation.  Designed to be
//! invoked from the Python AI core via the HTTP server in `main.rs`.

use rayon::prelude::*;

/// Compute the cosine similarity between two equal-length vectors.
/// Returns 0.0 if either vector is the zero vector.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine: vector length mismatch");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Compute cosine similarity between `query` and every row in `corpus` in
/// parallel using rayon.  Returns a Vec<f32> aligned with corpus rows.
pub fn cosine_batch(query: &[f32], corpus: &[Vec<f32>]) -> Vec<f32> {
    corpus.par_iter().map(|row| cosine(query, row)).collect()
}

/// L2-normalize a single vector in place. Returns the original norm.
pub fn normalize(v: &mut [f32]) -> f32 {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// CRC32 checksum of a byte slice (used for memory shard validation).
pub fn checksum(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identity() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine(&v, &v);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn batch_matches_single() {
        let q = vec![1.0, 0.5, -0.25];
        let corpus = vec![
            vec![1.0, 0.5, -0.25],
            vec![0.0, 1.0, 0.0],
            vec![-1.0, -0.5, 0.25],
        ];
        let scores = cosine_batch(&q, &corpus);
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_unit() {
        let mut v = vec![3.0f32, 4.0];
        let n = normalize(&mut v);
        assert!((n - 5.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn checksum_stable() {
        assert_eq!(checksum(b"hello"), checksum(b"hello"));
        assert_ne!(checksum(b"hello"), checksum(b"hellp"));
    }
}
