use crate::{
    client::{
        CompletionClient, EmbeddingsClient, ProviderClient, VerifyClient, VerifyError,
    },
    extractor::ExtractorBuilder,
    http_client::{self, HttpClientExt},
    impl_conversion_traits,
    providers::llama_cpp::CompletionModel,
};
use std::fmt::Debug;

// ================================================================
// Main LlamaCpp Client
// ================================================================
const LLAMA_CPP_API_BASE_URL: &str = "http://localhost:8402/v1";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: Option<&'a str>,
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    pub fn new() -> Self {
        Self {
            api_key: None,
            base_url: LLAMA_CPP_API_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn api_key(mut self, api_key: &'a str) -> Self {
        self.api_key = Some(api_key);
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.map(|k| k.to_string()),
            http_client: self.http_client,
        }
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    api_key: Option<String>,
    http_client: T,
}

impl<T> Debug for Client<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &self.api_key.as_ref().map(|_| "<REDACTED>"))
            .finish()
    }
}

impl<T> Client<T>
where
    T: Default,
{
    /// Create a new LlamaCpp client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::llama_cpp::{ClientBuilder, self};
    ///
    /// // Initialize the LlamaCpp client
    /// let client = Client::builder()
    ///    .base_url("http://localhost:8402/v1")
    ///    .build()
    /// ```
    pub fn builder<'a>() -> ClientBuilder<'a, T> {
        ClientBuilder::new()
    }

    /// Create a new LlamaCpp client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new() -> Self {
        Self::builder().build()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    pub(crate) fn post(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        match &self.api_key {
            Some(api_key) => http_client::with_bearer_auth(http_client::Request::post(url), api_key),
            None => Ok(http_client::Request::post(url)),
        }
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        match &self.api_key {
            Some(api_key) => http_client::with_bearer_auth(http_client::Request::get(url), api_key),
            None => Ok(http_client::Request::get(url)),
        }
    }

    pub(crate) async fn send<U, R>(
        &self,
        req: http_client::Request<U>,
    ) -> http_client::Result<http_client::Response<http_client::LazyBody<R>>>
    where
        U: Into<bytes::Bytes> + Send,
        R: From<bytes::Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }
}

impl Client<reqwest::Client> {
    pub(crate) fn post_reqwest(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        match &self.api_key {
            Some(api_key) => self.http_client.post(url).bearer_auth(api_key),
            None => self.http_client.post(url),
        }
    }

    /// Create an extractor builder with the given completion model.
    /// Intended for use exclusively with the Chat Completions API.
    /// Useful for using extractors with Chat Completion compliant APIs.
    pub fn extractor_completions_api<U>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<CompletionModel<reqwest::Client>, U>
    where
        U: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + serde::Serialize + Send + Sync,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

impl ProviderClient for Client<reqwest::Client> {
    /// Create a new LlamaCpp client from the `LLAMA_CPP_API_BASE_URL` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let base_url: Option<String> = std::env::var("LLAMA_CPP_API_BASE_URL").ok();
        let api_key: Option<String> = std::env::var("LLAMA_CPP_API_KEY").ok();

        let mut builder = Self::builder();
        
        if let Some(url) = &base_url {
            builder = builder.base_url(url);
        }
        
        if let Some(key) = &api_key {
            builder = builder.api_key(key);
        }
        
        builder.build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(_) = input else {
            panic!("Incorrect provider value type")
        };
        Self::new()
    }
}

impl CompletionClient for Client<reqwest::Client> {
    type CompletionModel = CompletionModel<reqwest::Client>;
    
    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::llama_cpp::{Client, self};
    ///
    /// // Initialize the LlamaCpp client
    /// let client = Client::new();
    ///
    /// let model = client.completion_model("qwen2.5");
    /// ```
    fn completion_model(&self, model: &str) -> CompletionModel<reqwest::Client> {
        CompletionModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client<reqwest::Client> {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/models")?
            .body(http_client::NoBody)
            .map_err(|e| VerifyError::HttpError(e.into()))?;

        let response = self.send(req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                Ok(())
            }
        }
    }
}

// We need to implement EmbeddingsClient for the Client to satisfy the ProviderClient trait
impl EmbeddingsClient for Client<reqwest::Client> {
    type EmbeddingModel = DummyEmbeddingModel;
    
    fn embedding_model(&self, _model: &str) -> Self::EmbeddingModel {
        DummyEmbeddingModel::new()
    }
    
    fn embedding_model_with_ndims(&self, _model: &str, _ndims: usize) -> Self::EmbeddingModel {
        DummyEmbeddingModel::new()
    }
    
    fn embeddings<D: crate::Embed>(&self, _model: &str) -> crate::embeddings::EmbeddingsBuilder<Self::EmbeddingModel, D> {
        crate::embeddings::EmbeddingsBuilder::new(self.embedding_model(""))
    }
}

// Dummy embedding model that always returns an error since LlamaCpp doesn't support embeddings
#[derive(Debug, Clone)]
pub struct DummyEmbeddingModel;

impl DummyEmbeddingModel {
    pub fn new() -> Self {
        Self
    }
}

impl crate::embeddings::EmbeddingModel for DummyEmbeddingModel {
    const MAX_DOCUMENTS: usize = 0;
    
    fn ndims(&self) -> usize {
        0
    }
    
    async fn embed_texts(
        &self,
        _documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<crate::embeddings::Embedding>, crate::embeddings::EmbeddingError> {
        Err(crate::embeddings::EmbeddingError::ProviderError(
            "LlamaCpp does not support embeddings".to_string(),
        ))
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);

#[derive(Debug, serde::Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}