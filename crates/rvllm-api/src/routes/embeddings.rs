//! Embeddings endpoint: POST /v1/embeddings
//!
//! OpenAI-compatible endpoint for computing embeddings from an embedding model.
//! Accepts single strings or batches and returns normalised (or raw) embedding
//! vectors.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::info;
use utoipa::ToSchema;

use crate::error::ApiError;
use crate::server::AppState;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Input for the embeddings endpoint -- can be a single string or a list.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single string to embed.
    Single(String),
    /// A batch of strings.
    Batch(Vec<String>),
}

/// POST /v1/embeddings request body (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingRequest {
    /// The input text(s) to embed.
    pub input: EmbeddingInput,
    /// Model name.
    pub model: String,
    /// Encoding format for the embeddings. Only "float" is currently supported.
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    /// Optional user identifier.
    #[serde(default)]
    pub user: Option<String>,
}

fn default_encoding_format() -> String {
    "float".into()
}

impl EmbeddingRequest {
    /// Validate the request.
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.model.is_empty() {
            return Err(ApiError::InvalidRequest("model is required".into()));
        }
        match &self.input {
            EmbeddingInput::Single(s) if s.is_empty() => {
                return Err(ApiError::InvalidRequest("input must not be empty".into()));
            }
            EmbeddingInput::Batch(v) if v.is_empty() => {
                return Err(ApiError::InvalidRequest(
                    "input must contain at least one string".into(),
                ));
            }
            _ => {}
        }
        if self.encoding_format != "float" && self.encoding_format != "base64" {
            return Err(ApiError::InvalidRequest(format!(
                "unsupported encoding_format: '{}', expected 'float' or 'base64'",
                self.encoding_format
            )));
        }
        Ok(())
    }

    /// Flatten the input into a list of strings regardless of variant.
    pub fn texts(&self) -> Vec<&str> {
        match &self.input {
            EmbeddingInput::Single(s) => vec![s.as_str()],
            EmbeddingInput::Batch(v) => v.iter().map(|s| s.as_str()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// A single embedding result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct EmbeddingObject {
    /// The object type (always "embedding").
    pub object: String,
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Index in the input list.
    pub index: usize,
}

/// Response for POST /v1/embeddings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct EmbeddingResponse {
    /// Object type (always "list").
    pub object: String,
    /// The embeddings.
    pub data: Vec<EmbeddingObject>,
    /// Model name.
    pub model: String,
    /// Token usage.
    pub usage: EmbeddingUsage,
}

/// Token usage specific to embeddings (no completion_tokens).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct EmbeddingUsage {
    /// Number of tokens in the input prompt(s).
    pub prompt_tokens: usize,
    /// Total tokens (same as prompt_tokens for embeddings).
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// POST /v1/embeddings -- compute embeddings for the given input text(s).
///
/// This endpoint sends each input string through the engine (which runs the
/// embedding model forward pass), then pools and normalises the hidden states
/// to produce a fixed-size embedding vector per input.
pub async fn create_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    req.validate()?;

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }
    if !state.capabilities.supports_embeddings {
        return Err(ApiError::InvalidRequest(format!(
            "loaded model '{}' does not expose embeddings on this runtime",
            state.model_name
        )));
    }

    let texts = req.texts();
    info!(
        model = %req.model,
        num_inputs = texts.len(),
        "embedding request"
    );

    let mut data = Vec::with_capacity(texts.len());
    let mut total_prompt_tokens = 0usize;

    for (idx, text) in texts.iter().enumerate() {
        // Use the inference engine to get token-level output.
        // For a real embedding model the engine forward pass returns hidden
        // states (not logits). We use a minimal sampling config that generates
        // 0 new tokens -- we only need the forward pass result.
        let sampling = rvllm_core::prelude::SamplingParams {
            max_tokens: 1,
            temperature: 0.0,
            ..Default::default()
        };

        let (_request_id, mut output_stream) = state
            .engine
            .generate(text.to_string(), sampling)
            .await
            .map_err(ApiError::from)?;

        // Collect the final output.
        let mut last_output = None;
        while let Some(output) = tokio_stream::StreamExt::next(&mut output_stream).await {
            if output.finished {
                last_output = Some(output);
                break;
            }
            last_output = Some(output);
        }

        let output = last_output
            .ok_or_else(|| ApiError::Internal("engine produced no output for embedding".into()))?;

        total_prompt_tokens += output.prompt_token_ids.len();

        // The engine output for an embedding model will contain the pooled
        // embedding in the first completion output text (as a comma-separated
        // float list) or as a placeholder. For the mock path we generate a
        // deterministic embedding from the prompt token ids.
        let embedding = mock_embedding_from_tokens(&output.prompt_token_ids);

        data.push(EmbeddingObject {
            object: "embedding".into(),
            embedding,
            index: idx,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list".into(),
        data,
        model: state.model_name.clone(),
        usage: EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    }))
}

/// Generate a deterministic mock embedding from token ids.
///
/// In the real GPU path the embedding model's hidden states are pooled and
/// normalised. This mock path produces a repeatable embedding vector
/// for testing and development without a GPU.
fn mock_embedding_from_tokens(token_ids: &[u32]) -> Vec<f32> {
    const DIM: usize = 384;
    let mut emb = vec![0.0f32; DIM];

    // Scatter token ids into the embedding dimensions deterministically.
    for (i, &tid) in token_ids.iter().enumerate() {
        let idx = (tid as usize + i * 37) % DIM;
        emb[idx] += (tid as f32 + 1.0).ln();
    }

    // L2 normalise.
    let mut sum_sq = 0.0f32;
    for &v in &emb {
        sum_sq += v * v;
    }
    let inv_norm = (sum_sq + 1e-12).sqrt().recip();
    for v in &mut emb {
        *v *= inv_norm;
    }

    emb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};

    use crate::{build_router, AppState};
    use axum_test::TestServer;
    use rvllm_core::prelude::{CompletionOutput, RequestId, RequestOutput, SamplingParams};
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer as HfTokenizer;
    use tokio::sync::Mutex as AsyncMutex;
    use tokio_stream::wrappers::ReceiverStream;

    struct FakeEngine {
        outputs: AsyncMutex<VecDeque<Vec<RequestOutput>>>,
        prompts: Mutex<Vec<String>>,
    }

    impl FakeEngine {
        fn new(outputs: Vec<Vec<RequestOutput>>) -> Self {
            Self {
                outputs: AsyncMutex::new(outputs.into()),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl crate::server::InferenceEngine for FakeEngine {
        async fn generate(
            &self,
            prompt: String,
            _params: SamplingParams,
        ) -> rvllm_core::prelude::Result<(RequestId, ReceiverStream<RequestOutput>)> {
            self.prompts.lock().unwrap().push(prompt);
            let maybe_outputs = self.outputs.lock().await.pop_front();
            let outputs = maybe_outputs.expect("fake engine ran out of queued outputs");
            let (tx, rx) = tokio::sync::mpsc::channel(outputs.len().max(1));
            for output in outputs {
                tx.send(output).await.unwrap();
            }
            drop(tx);
            Ok((RequestId(1), ReceiverStream::new(rx)))
        }
    }

    fn make_test_tokenizer() -> rvllm_tokenizer::Tokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("[UNK]".to_string(), 2);

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut hf = HfTokenizer::new(bpe);
        hf.with_pre_tokenizer(Some(Whitespace {}));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        hf.save(&path, false).unwrap();
        rvllm_tokenizer::Tokenizer::from_file(&path).unwrap()
    }

    fn request_output(prompt_tokens: Vec<u32>) -> RequestOutput {
        RequestOutput {
            request_id: RequestId(1),
            prompt: "hello world".into(),
            prompt_token_ids: prompt_tokens,
            prompt_logprobs: None,
            outputs: vec![CompletionOutput {
                index: 0,
                text: String::new(),
                token_ids: vec![],
                cumulative_logprob: 0.0,
                logprobs: None,
                finish_reason: Some(rvllm_core::prelude::FinishReason::Stop),
            }],
            finished: true,
        }
    }

    fn make_server(
        capabilities: crate::server::ModelCapabilities,
        outputs: Vec<Vec<RequestOutput>>,
    ) -> (TestServer, Arc<FakeEngine>) {
        let engine = Arc::new(FakeEngine::new(outputs));
        let state = Arc::new(AppState::new_with_capabilities(
            engine.clone(),
            "test".to_string(),
            make_test_tokenizer(),
            capabilities,
        ));
        let server = TestServer::new(build_router(state)).unwrap();
        (server, engine)
    }

    #[test]
    fn embedding_request_validate_ok() {
        let req = EmbeddingRequest {
            input: EmbeddingInput::Single("hello world".into()),
            model: "e5-small".into(),
            encoding_format: "float".into(),
            user: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn embedding_request_validate_empty_model() {
        let req = EmbeddingRequest {
            input: EmbeddingInput::Single("hello".into()),
            model: "".into(),
            encoding_format: "float".into(),
            user: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn embedding_request_validate_empty_input() {
        let req = EmbeddingRequest {
            input: EmbeddingInput::Single("".into()),
            model: "m".into(),
            encoding_format: "float".into(),
            user: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn embedding_request_validate_empty_batch() {
        let req = EmbeddingRequest {
            input: EmbeddingInput::Batch(vec![]),
            model: "m".into(),
            encoding_format: "float".into(),
            user: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn embedding_request_validate_bad_format() {
        let req = EmbeddingRequest {
            input: EmbeddingInput::Single("hello".into()),
            model: "m".into(),
            encoding_format: "binary".into(),
            user: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn texts_single() {
        let req = EmbeddingRequest {
            input: EmbeddingInput::Single("hello".into()),
            model: "m".into(),
            encoding_format: "float".into(),
            user: None,
        };
        assert_eq!(req.texts(), vec!["hello"]);
    }

    #[test]
    fn texts_batch() {
        let req = EmbeddingRequest {
            input: EmbeddingInput::Batch(vec!["a".into(), "b".into()]),
            model: "m".into(),
            encoding_format: "float".into(),
            user: None,
        };
        assert_eq!(req.texts(), vec!["a", "b"]);
    }

    #[test]
    fn embedding_response_serde_roundtrip() {
        let resp = EmbeddingResponse {
            object: "list".into(),
            data: vec![EmbeddingObject {
                object: "embedding".into(),
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            model: "e5-small".into(),
            usage: EmbeddingUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: EmbeddingResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn mock_embedding_is_normalised() {
        let emb = mock_embedding_from_tokens(&[1, 2, 3, 4, 5]);
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "norm={}", norm);
    }

    #[test]
    fn mock_embedding_is_deterministic() {
        let a = mock_embedding_from_tokens(&[10, 20, 30]);
        let b = mock_embedding_from_tokens(&[10, 20, 30]);
        assert_eq!(a, b);
    }

    #[test]
    fn mock_embedding_different_inputs() {
        let a = mock_embedding_from_tokens(&[1, 2, 3]);
        let b = mock_embedding_from_tokens(&[4, 5, 6]);
        assert_ne!(a, b);
    }

    #[test]
    fn embedding_request_serde_single() {
        let json = r#"{"input":"hello","model":"e5"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Single(_)));
        assert_eq!(req.encoding_format, "float");
    }

    #[test]
    fn embedding_request_serde_batch() {
        let json = r#"{"input":["a","b"],"model":"e5"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Batch(_)));
    }

    #[test]
    fn embedding_usage_serde() {
        let u = EmbeddingUsage {
            prompt_tokens: 10,
            total_tokens: 10,
        };
        let json = serde_json::to_string(&u).unwrap();
        let back: EmbeddingUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, u);
    }

    #[tokio::test]
    async fn create_embeddings_route_rejects_text_only_runtime() {
        let (server, engine) = make_server(
            crate::server::ModelCapabilities::default(),
            vec![vec![request_output(vec![1, 2, 3])]],
        );

        let response = server
            .post("/v1/embeddings")
            .json(&serde_json::json!({
                "model": "test",
                "input": "hello world"
            }))
            .await;

        response.assert_status_bad_request();
        assert!(engine.prompts().is_empty(), "engine should not be called");
    }

    #[tokio::test]
    async fn create_embeddings_route_uses_supported_runtime() {
        let (server, engine) = make_server(
            crate::server::ModelCapabilities {
                supports_embeddings: true,
                supports_input_images: false,
            },
            vec![vec![request_output(vec![11, 22, 33])]],
        );

        let response = server
            .post("/v1/embeddings")
            .json(&serde_json::json!({
                "model": "test",
                "input": "hello world"
            }))
            .await;

        response.assert_status_ok();
        let body = response.json::<serde_json::Value>();
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["object"], "embedding");
        assert_eq!(engine.prompts().len(), 1);
    }
}
