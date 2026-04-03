//! HTTP server setup, AppState, router construction, and graceful shutdown.
//!
//! The server requires a CUDA-capable GPU and uses AsyncGpuLLMEngine for real
//! inference. It fails fast instead of falling back to a mock executor.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use rvllm_config::EngineConfig;
use rvllm_core::prelude::RequestId;
use rvllm_tokenizer::Tokenizer;

use crate::routes;

// ------------------------------------------------------------------
// Engine trait object for unified API
// ------------------------------------------------------------------

/// Trait abstracting over AsyncLLMEngine and AsyncGpuLLMEngine so the
/// AppState can hold either one.
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn generate(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )>;

    async fn generate_with_mode(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
        emit_intermediate: bool,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        let _ = emit_intermediate;
        self.generate(prompt, params).await
    }
}

#[async_trait::async_trait]
#[cfg(feature = "cuda")]
#[async_trait::async_trait]
impl InferenceEngine for rvllm_engine::AsyncGpuLLMEngine {
    async fn generate(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        self.generate(prompt, params).await
    }

    async fn generate_with_mode(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
        emit_intermediate: bool,
    ) -> rvllm_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        self.generate_with_mode(prompt, params, emit_intermediate).await
    }
}

/// Shared application state available to all route handlers.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub supports_embeddings: bool,
    pub supports_input_images: bool,
}

pub struct AppState {
    pub engine: Arc<dyn InferenceEngine>,
    pub model_name: String,
    pub capabilities: ModelCapabilities,
    pub tokenizer: Arc<RwLock<Tokenizer>>,
    /// Batch job store (None if batch API is not enabled).
    pub batch_store: Option<crate::routes::batch::SharedBatchStore>,
    /// Stored response objects for Responses API follow-up turns and retrieval.
    pub response_store: crate::routes::responses::SharedResponseStore,
    /// Stored conversation state for Responses API conversation-based turns.
    pub conversation_store: crate::routes::responses::SharedConversationStore,
    next_id: AtomicU64,
}

impl AppState {
    pub fn new(engine: Arc<dyn InferenceEngine>, model_name: String, tokenizer: Tokenizer) -> Self {
        Self::new_with_capabilities(
            engine,
            model_name,
            tokenizer,
            ModelCapabilities::default(),
        )
    }

    pub fn new_with_capabilities(
        engine: Arc<dyn InferenceEngine>,
        model_name: String,
        tokenizer: Tokenizer,
        capabilities: ModelCapabilities,
    ) -> Self {
        Self {
            engine,
            model_name,
            capabilities,
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            batch_store: Some(crate::routes::batch::create_batch_store(None)),
            response_store: Arc::new(RwLock::new(HashMap::new())),
            conversation_store: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/v1/completions",
            post(routes::completions::create_completion),
        )
        .route(
            "/v1/chat/completions",
            post(routes::chat::create_chat_completion),
        )
        .route("/v1/responses", post(routes::responses::create_response))
        .route(
            "/v1/responses/:response_id",
            get(routes::responses::get_response),
        )
        .route(
            "/v1/responses/:response_id/input_items",
            get(routes::responses::list_response_input_items),
        )
        .route(
            "/v1/embeddings",
            post(routes::embeddings::create_embeddings),
        )
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/batches", post(routes::batch::create_batch))
        .route("/v1/batches/:batch_id", get(routes::batch::get_batch))
        .route(
            "/v1/batches/:batch_id/output",
            get(routes::batch::get_batch_output),
        )
        .route(
            "/v1/batches/:batch_id/cancel",
            post(routes::batch::cancel_batch),
        )
        .route(
            "/v1/chat/completions/tools",
            post(routes::tools::create_chat_completion_with_tools),
        )
        .route("/health", get(routes::health::health_check))
        .route("/metrics", get(metrics_placeholder))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn metrics_placeholder() -> &'static str {
    "# vllm-rs metrics endpoint\n"
}

fn architecture_supports_embeddings(architecture: &str) -> bool {
    matches!(
        architecture,
        "BertModel"
            | "RobertaModel"
            | "XLMRobertaModel"
            | "E5Model"
            | "GTEModel"
            | "BGEModel"
            | "EmbeddingModel"
            | "SentenceTransformer"
    )
}

fn read_model_architecture(snapshot_dir: &Path) -> rvllm_core::prelude::Result<Option<String>> {
    let config_path = snapshot_dir.join("config.json");
    let content = std::fs::read_to_string(&config_path).map_err(|e| {
        rvllm_core::prelude::LLMError::ModelError(format!(
            "failed to read {}: {e}",
            config_path.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        rvllm_core::prelude::LLMError::ModelError(format!("invalid config.json: {e}"))
    })?;
    Ok(json
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .map(str::to_owned))
}

fn detect_model_capabilities(model_name: &str) -> ModelCapabilities {
    let architecture = match rvllm_engine::hf_snapshot::ensure_snapshot(model_name)
        .and_then(|snapshot| read_model_architecture(&snapshot))
    {
        Ok(arch) => arch,
        Err(e) => {
            warn!(
                model = model_name,
                error = %e,
                "failed to detect model capabilities, defaulting to text-only runtime"
            );
            None
        }
    };

    let capabilities = ModelCapabilities {
        supports_embeddings: architecture
            .as_deref()
            .map(architecture_supports_embeddings)
            .unwrap_or(false),
        // The runtime does not yet execute native image inputs.
        supports_input_images: false,
    };

    info!(
        model = model_name,
        architecture = architecture.as_deref().unwrap_or("unknown"),
        supports_embeddings = capabilities.supports_embeddings,
        supports_input_images = capabilities.supports_input_images,
        "detected model capabilities"
    );

    capabilities
}

fn cuda_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        let devices = rvllm_gpu::prelude::list_devices();
        if devices.is_empty() {
            info!("cuda feature enabled but no CUDA devices found");
            return false;
        }
        for dev in &devices {
            info!(
                id = dev.id, name = %dev.name,
                memory_gb = dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                "CUDA device available"
            );
        }
        true
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub async fn serve(config: EngineConfig) -> rvllm_core::prelude::Result<()> {
    let model_name = config.model.model_path.clone();
    let tokenizer_path = config
        .model
        .tokenizer_path
        .clone()
        .unwrap_or_else(|| config.model.model_path.clone());

    info!(model = %model_name, "initializing engine");

    let tokenizer = Tokenizer::from_pretrained(&tokenizer_path)?;
    let capabilities = detect_model_capabilities(&model_name);

    if !cuda_gpu_available() {
        return Err(rvllm_core::prelude::LLMError::GpuError(
            "no CUDA GPU detected; refusing to start mock/non-GPU server backend".into(),
        ));
    }

    info!("GPU detected, creating AsyncGpuLLMEngine for real inference");
    let engine: Arc<dyn InferenceEngine> = create_gpu_engine(config).await?;

    let state = Arc::new(AppState::new_with_capabilities(
        engine,
        model_name,
        tokenizer,
        capabilities,
    ));
    let app = build_router(state);

    let host = std::env::var("VLLM_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("VLLM_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8000);
    let addr = format!("{host}:{port}");
    info!(addr = %addr, "starting API server");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(rvllm_core::prelude::LLMError::IoError)?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(rvllm_core::prelude::LLMError::IoError)?;

    info!("server shut down gracefully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_architectures_are_detected() {
        assert!(architecture_supports_embeddings("BertModel"));
        assert!(architecture_supports_embeddings("SentenceTransformer"));
        assert!(!architecture_supports_embeddings("LlamaForCausalLM"));
    }
}

/// Create the real GPU engine using AsyncGpuLLMEngine.
#[cfg(feature = "cuda")]
async fn create_gpu_engine(
    config: EngineConfig,
) -> rvllm_core::prelude::Result<Arc<dyn InferenceEngine>> {
    let engine = rvllm_engine::AsyncGpuLLMEngine::new(config).await?;
    Ok(Arc::new(engine))
}

#[cfg(not(feature = "cuda"))]
async fn create_gpu_engine(
    _config: EngineConfig,
) -> rvllm_core::prelude::Result<Arc<dyn InferenceEngine>> {
    Err(rvllm_core::prelude::LLMError::GpuError(
        "CUDA not available".into(),
    ))
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { info!("received Ctrl+C, shutting down"); }
        _ = terminate => { info!("received SIGTERM, shutting down"); }
    }
}
