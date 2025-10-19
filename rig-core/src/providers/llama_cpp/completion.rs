use super::{ApiErrorResponse, ApiResponse, Client};
use crate::completion::{
    CompletionError, CompletionRequest as CoreCompletionRequest, GetTokenUsage,
};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::{self, merge};
use crate::message::{AudioMediaType, DocumentSourceKind, ImageDetail, MimeType};
use crate::one_or_many::string_or_one_or_many;
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use crate::telemetry::{ProviderResponseExt, SpanCombinator};
use crate::{OneOrMany, completion, message};
use async_stream::stream;
use futures::StreamExt;
use reqwest::RequestBuilder;
use reqwest_eventsource::Event;
use reqwest_eventsource::RequestBuilderExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;
use tracing::{debug, info_span};
use tracing_futures::Instrument;

/// `qwen2.5` completion model
pub const QWEN2_5: &str = "qwen2.5";
/// `qwen3-8b` completion model
pub const QWEN3_8B: &str = "qwen3-8b";

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, deserialize_with = "json_utils::string_or_vec")]
        content: Vec<AssistantContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: OneOrMany<ToolResultContent>,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::one(content.to_owned().into()),
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    pub r#type: SystemContentType,
    pub text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
    #[default]
    Text,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
    Refusal { refusal: String },
    Reasoning { reasoning: String },
}

impl From<AssistantContent> for completion::AssistantContent {
    fn from(value: AssistantContent) -> Self {
        match value {
            AssistantContent::Text { text } => completion::AssistantContent::text(text),
            AssistantContent::Refusal { refusal } => completion::AssistantContent::text(refusal),
            AssistantContent::Reasoning { reasoning } => completion::AssistantContent::Reasoning(
                completion::message::Reasoning::new(&reasoning),
            ),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    Image {
        image_url: ImageUrl,
    },
    Audio {
        input_audio: InputAudio,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: ImageDetail,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: AudioMediaType,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolResultContent {
    #[serde(default)]
    pub r#type: ToolResultContentType,
    pub text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultContentType {
    #[default]
    Text,
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent {
            r#type: ToolResultContentType::default(),
            text: s,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;
    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Specific { .. } => {
                return Err(CompletionError::ProviderError(
                    "Provider doesn't support only using specific tools".to_string(),
                ));
            }
            message::ToolChoice::Auto => Self::Auto,
            message::ToolChoice::None => Self::None,
            message::ToolChoice::Required => Self::Required,
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let (tool_results, other_content): (Vec<_>, Vec<_>) = content
                    .into_iter()
                    .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

                if !tool_results.is_empty() {
                    tool_results
                        .into_iter()
                        .map(|content| match content {
                            message::UserContent::ToolResult(message::ToolResult {
                                id,
                                content,
                                ..
                            }) => Ok::<_, message::MessageError>(Message::ToolResult {
                                tool_call_id: id,
                                content: content.try_map(|content| match content {
                                    message::ToolResultContent::Text(message::Text { text }) => {
                                        Ok(text.into())
                                    }
                                    _ => Err(message::MessageError::ConversionError(
                                        "Tool result content does not support non-text".into(),
                                    )),
                                })?,
                            }),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    let other_content: Vec<UserContent> = other_content
                        .into_iter()
                        .map(|content| match content {
                            message::UserContent::Text(message::Text { text }) => {
                                Ok(UserContent::Text { text })
                            }
                            message::UserContent::Image(message::Image {
                                data,
                                detail,
                                media_type,
                                ..
                            }) => match data {
                                DocumentSourceKind::Url(url) => Ok(UserContent::Image {
                                    image_url: ImageUrl {
                                        url,
                                        detail: detail.unwrap_or_default(),
                                    },
                                }),
                                DocumentSourceKind::Base64(data) => {
                                    let url = format!(
                                        "data:{};base64,{}",
                                        media_type.map(|i| i.to_mime_type()).ok_or(
                                            message::MessageError::ConversionError(
                                                "LlamaCpp Image URI must have media type".into()
                                            )
                                        )?,
                                        data
                                    );

                                    let detail =
                                        detail.ok_or(message::MessageError::ConversionError(
                                            "LlamaCpp image URI must have image detail".into(),
                                        ))?;

                                    Ok(UserContent::Image {
                                        image_url: ImageUrl { url, detail },
                                    })
                                }
                                DocumentSourceKind::Raw(_) => {
                                    Err(message::MessageError::ConversionError(
                                        "Raw files not supported, encode as base64 first".into(),
                                    ))
                                }
                                DocumentSourceKind::Unknown => {
                                    Err(message::MessageError::ConversionError(
                                        "Document has no body".into(),
                                    ))
                                }
                                doc => Err(message::MessageError::ConversionError(format!(
                                    "Unsupported document type: {doc:?}"
                                ))),
                            },
                            message::UserContent::Document(message::Document { data, .. }) => {
                                if let DocumentSourceKind::Base64(text) = data {
                                    Ok(UserContent::Text { text })
                                } else {
                                    Err(message::MessageError::ConversionError(
                                        "Documents must be base64".into(),
                                    ))
                                }
                            }
                            message::UserContent::Audio(message::Audio {
                                data: DocumentSourceKind::Base64(data),
                                media_type,
                                ..
                            }) => Ok(UserContent::Audio {
                                input_audio: InputAudio {
                                    data,
                                    format: match media_type {
                                        Some(media_type) => media_type,
                                        None => AudioMediaType::MP3,
                                    },
                                },
                            }),
                            _ => Err(message::MessageError::ConversionError(
                                "Tool result is in unsupported format".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let other_content = OneOrMany::many(other_content).expect(
                        "There must be other content here if there were no tool result content",
                    );

                    Ok(vec![Message::User {
                        content: other_content,
                        name: None,
                    }])
                }
            }
            message::Message::Assistant { content, .. } => {
                let (text_content, reasoning_content, tool_calls) = content.into_iter().fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut texts, mut reasoning, mut tools), content| {
                        match content {
                            message::AssistantContent::Text(text) => texts.push(text),
                            message::AssistantContent::ToolCall(tool_call) => tools.push(tool_call),
                            message::AssistantContent::Reasoning(reasoning_data) => {
                                {
                                    println!("kankerdennis2");
                                }
                                reasoning.push(reasoning_data)
                            }
                        }
                        (texts, reasoning, tools)
                    },
                );

                let mut assistant_contents = Vec::new();

                // Add reasoning content if present
                if !reasoning_content.is_empty() {
                    let reasoning_text = reasoning_content
                        .into_iter()
                        .map(|r| r.reasoning.join("\n"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    assistant_contents.push(AssistantContent::Reasoning {
                        reasoning: reasoning_text,
                    });
                }

                // Add text content
                assistant_contents.extend(
                    text_content
                        .into_iter()
                        .map(|content| content.text.into())
                        .collect::<Vec<_>>(),
                );

                Ok(vec![Message::Assistant {
                    content: assistant_contents,
                    refusal: None,
                    reasoning: None, // Reasoning is now in content
                    name: None,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tool_call| tool_call.into())
                        .collect::<Vec<_>>(),
                }])
            }
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            id: tool_call.id,
            r#type: ToolType::default(),
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl From<ToolCall> for message::ToolCall {
    fn from(tool_call: ToolCall) -> Self {
        Self {
            id: tool_call.id,
            call_id: None,
            function: message::ToolFunction {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message {
            Message::User { content, .. } => message::Message::User {
                content: content.map(|content| content.into()),
            },
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
                ..
            } => {
                let mut content = content
                    .into_iter()
                    .map(|content| match content {
                        AssistantContent::Text { text } => message::AssistantContent::text(text),
                        AssistantContent::Refusal { refusal } => {
                            message::AssistantContent::text(refusal)
                        }
                        AssistantContent::Reasoning { reasoning } => {
                            message::AssistantContent::Reasoning(message::Reasoning::new(
                                &reasoning,
                            ))
                        }
                    })
                    .collect::<Vec<_>>();

                // Add reasoning from the reasoning field if present
                if let Some(reasoning_text) = reasoning {
                    content.push(message::AssistantContent::Reasoning(
                        message::Reasoning::new(&reasoning_text),
                    ));
                }

                content.extend(
                    tool_calls
                        .into_iter()
                        .map(|tool_call| Ok(message::AssistantContent::ToolCall(tool_call.into())))
                        .collect::<Result<Vec<_>, _>>()?,
                );

                message::Message::Assistant {
                    id: None,
                    content: OneOrMany::many(content).map_err(|_| {
                        message::MessageError::ConversionError(
                            "Neither `content` nor `tool_calls` was provided to the Message"
                                .to_owned(),
                        )
                    })?,
                }
            }

            Message::ToolResult {
                tool_call_id,
                content,
            } => message::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    tool_call_id,
                    content.map(|content| message::ToolResultContent::text(content.text)),
                )),
            },

            Message::System { content, .. } => message::Message::User {
                content: content.map(|content| message::UserContent::text(content.text)),
            },
        })
    }
}

impl From<UserContent> for message::UserContent {
    fn from(content: UserContent) -> Self {
        match content {
            UserContent::Text { text } => message::UserContent::text(text),
            UserContent::Image { image_url } => {
                message::UserContent::image_url(image_url.url, None, Some(image_url.detail))
            }
            UserContent::Audio { input_audio } => {
                message::UserContent::audio(input_audio.data, Some(input_audio.format))
            }
        }
    }
}

impl From<String> for UserContent {
    fn from(s: String) -> Self {
        UserContent::Text { text: s }
    }
}

impl FromStr for UserContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text {
            text: s.to_string(),
        })
    }
}

impl From<String> for AssistantContent {
    fn from(s: String) -> Self {
        AssistantContent::Text { text: s }
    }
}

impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent::Text {
            text: s.to_string(),
        })
    }
}

impl From<String> for SystemContent {
    fn from(s: String) -> Self {
        SystemContent {
            r#type: SystemContentType::default(),
            text: s,
        }
    }
}

impl FromStr for SystemContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent {
            r#type: SystemContentType::default(),
            text: s.to_string(),
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
                ..
            } => {
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            AssistantContent::Text { text } => Some(text),
                            AssistantContent::Refusal { refusal } => Some(refusal),
                            AssistantContent::Reasoning { reasoning: _ } => {
                                // Reasoning is handled separately
                                None
                            }
                        };
                        if let Some(text) = s {
                            if text.is_empty() {
                                None
                            } else {
                                Some(completion::AssistantContent::text(text))
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                // Add reasoning content if present
                if let Some(reasoning_text) = reasoning {
                    if !reasoning_text.is_empty() {
                        content.push(completion::AssistantContent::Reasoning(
                            message::Reasoning::new(&reasoning_text),
                        ));
                    }
                }

                // Add reasoning from content items (this is already handled above)

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

impl ProviderResponseExt for CompletionResponse {
    type OutputMessage = Choice;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.to_owned())
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.to_owned())
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.choices.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        let Message::User { ref content, .. } = self.choices.last()?.message.clone() else {
            return None;
        };

        let UserContent::Text { text } = content.first() else {
            return None;
        };

        Some(text)
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            total_tokens: 0,
        }
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Usage {
            prompt_tokens,
            total_tokens,
        } = self;
        write!(
            f,
            "Prompt tokens: {prompt_tokens} Total tokens: {total_tokens}"
        )
    }
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = (self.total_tokens - self.prompt_tokens) as u64;
        usage.total_tokens = self.total_tokens as u64;

        Some(usage)
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    /// Name of the model (e.g.: qwen2.5)
    pub model: String,
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Default + std::fmt::Debug + Clone + 'static,
{
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    tool_choice: Option<ToolChoice>,
    temperature: Option<f64>,
    #[serde(flatten)]
    additional_params: Option<serde_json::Value>,
}

impl TryFrom<(String, CoreCompletionRequest)> for CompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (String, CoreCompletionRequest)) -> Result<Self, Self::Error> {
        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        let CoreCompletionRequest {
            preamble,
            chat_history,
            tools,
            temperature,
            additional_params,
            tool_choice,
            ..
        } = req;

        partial_history.extend(chat_history);

        let mut full_history: Vec<Message> =
            preamble.map_or_else(Vec::new, |preamble| vec![Message::system(&preamble)]);

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let tool_choice = tool_choice.map(ToolChoice::try_from).transpose()?;

        let res = Self {
            model,
            messages: full_history,
            tools: tools
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            temperature,
            additional_params,
        };

        Ok(res)
    }
}

impl crate::telemetry::ProviderRequestExt for CompletionRequest {
    type InputMessage = Message;

    fn get_input_messages(&self) -> Vec<Self::InputMessage> {
        self.messages.clone()
    }

    fn get_system_prompt(&self) -> Option<String> {
        let first_message = self.messages.first()?;

        let Message::System { ref content, .. } = first_message.clone() else {
            return None;
        };

        let SystemContent { text, .. } = content.first();

        Some(text)
    }

    fn get_prompt(&self) -> Option<String> {
        let last_message = self.messages.last()?;

        let Message::User { ref content, .. } = last_message.clone() else {
            return None;
        };

        let UserContent::Text { text } = content.first() else {
            return None;
        };

        Some(text)
    }

    fn get_model_name(&self) -> String {
        self.model.clone()
    }
}

impl CompletionModel<reqwest::Client> {
    pub fn into_agent_builder(self) -> crate::agent::AgentBuilder<Self> {
        crate::agent::AgentBuilder::new(self)
    }
}

// ================================================================
// LlamaCpp Completion Streaming API
// ================================================================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingToolCall {
    pub index: usize,
    pub id: Option<String>,
    pub function: StreamingFunction,
}

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.total_tokens as u64 - self.usage.prompt_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;
        Some(usage)
    }
}

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "llama_cpp",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = CompletionRequest::try_from((self.model.to_owned(), completion_request))?;

        span.record_model_input(&request.messages);

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send(req).await?;

            if response.status().is_success() {
                let text = http_client::text(response).await?;

                match serde_json::from_str::<ApiResponse<CompletionResponse>>(&text)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_model_output(&response.choices);
                        span.record_response_metadata(&response);
                        span.record_token_usage(&response.usage);
                        tracing::debug!("LlamaCpp response: {response:?}");
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                let text = http_client::text(response).await?;
                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let request = CompletionRequest::try_from((self.model.to_owned(), completion_request))?;
        let request_messages = serde_json::to_string(&request.messages)
            .expect("Converting to JSON from a Rust struct shouldn't fail");
        let mut request_as_json = serde_json::to_value(request).expect("this should never fail");

        request_as_json = merge(
            request_as_json,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self
            .client
            .post_reqwest("/chat/completions")
            .json(&request_as_json);

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "llama_cpp",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = request_messages,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::Instrument::instrument(send_compatible_streaming_request(builder), span).await
    }
}

pub async fn send_compatible_streaming_request(
    request_builder: RequestBuilder,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError> {
    let span = tracing::Span::current();
    // Build the request with proper headers for SSE
    let mut event_source = request_builder
        .eventsource()
        .expect("Cloning request must always succeed");

    let stream = stream! {
        let span = tracing::Span::current();
        let mut final_usage = Usage::new();

        // Track in-progress tool calls
        let mut tool_calls: HashMap<usize, (String, String, String)> = HashMap::new();

        let mut text_content = String::new();

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }
                Ok(Event::Message(message)) => {
                    if message.data.trim().is_empty() || message.data == "[DONE]" {
                        continue;
                    }

                    println!("kankerdennis3: {}", message.data);
                    // kankerdennis3: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":" I"}}],"created":1760873067,"id":"chatcmpl-Ms92NrkOH8jnDHU4gcXtQw7GEtr5umnD","model":"qwen3-8b","system_fingerprint":"b6785-9ad4f193","object":"chat.completion.chunk"}
                    let data = serde_json::from_str::<StreamingCompletionChunk>(&message.data);
                    let Ok(data) = data else {
                        let err = data.unwrap_err();
                        debug!("Couldn't serialize data as StreamingCompletionChunk: {:?}", err);
                        continue;
                    };

                    if let Some(choice) = data.choices.first() {
                        let delta = &choice.delta;

                        // Tool calls
                        if !delta.tool_calls.is_empty() {
                            for tool_call in &delta.tool_calls {
                                let function = tool_call.function.clone();

                                // Start of tool call
                                if function.name.is_some() && function.arguments.is_empty() {
                                    let id = tool_call.id.clone().unwrap_or_default();
                                    tool_calls.insert(
                                        tool_call.index,
                                        (id, function.name.clone().unwrap(), "".to_string()),
                                    );
                                }
                                // tool call partial (ie, a continuation of a previously received tool call)
                                // name: None or Empty String
                                // arguments: Some(String)
                                else if function.name.clone().is_none_or(|s| s.is_empty())
                                    && !function.arguments.is_empty()
                                {
                                    if let Some((id, name, arguments)) =
                                        tool_calls.get(&tool_call.index)
                                    {
                                        let new_arguments = &tool_call.function.arguments;
                                        let arguments = format!("{arguments}{new_arguments}");
                                        tool_calls.insert(
                                            tool_call.index,
                                            (id.clone(), name.clone(), arguments),
                                        );
                                    } else {
                                        debug!("Partial tool call received but tool call was never started.");
                                    }
                                }
                                // Complete tool call
                                else {
                                    let id = tool_call.id.clone().unwrap_or_default();
                                    let name = function.name.expect("tool call should have a name");
                                    let arguments = function.arguments;
                                    let Ok(arguments) = serde_json::from_str(&arguments) else {
                                        debug!("Couldn't serialize '{arguments}' as JSON");
                                        continue;
                                    };

                                    yield Ok(streaming::RawStreamingChoice::ToolCall {
                                        id,
                                        name,
                                        arguments,
                                        call_id: None,
                                    });
                                }
                            }
                        }

                        // Message content
                        println!("kankerden: {:#?}", choice);
                        if let Some(content) = &choice.delta.content {
                            text_content += content;
                            yield Ok(streaming::RawStreamingChoice::Message(content.clone()))
                        }
                    }

                    // Usage updates
                    if let Some(usage) = data.usage {
                        final_usage = usage.clone();
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => {
                    break;
                }
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::ResponseError(error.to_string()));
                    break;
                }
            }
        }

        // Ensure event source is closed when stream ends
        event_source.close();

        let mut vec_toolcalls = vec![];

        // Flush any tool calls that weren't fully yielded
        for (_, (id, name, arguments)) in tool_calls {
            let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&arguments) else {
                continue;
            };

            vec_toolcalls.push(ToolCall {
                r#type: ToolType::Function,
                id: id.clone(),
                function: Function {
                    name: name.clone(), arguments: arguments.clone()
                },
            });

            yield Ok(RawStreamingChoice::ToolCall {
                id,
                name,
                arguments,
                call_id: None,
            });
        }

        let message_output = Message::Assistant {
            content: vec![AssistantContent::Text { text: text_content }],
            refusal: None,
            reasoning: None,
            name: None,
            tool_calls: vec_toolcalls
        };

        span.record("gen_ai.usage.input_tokens", final_usage.prompt_tokens);
        span.record("gen_ai.usage.output_tokens", final_usage.total_tokens - final_usage.prompt_tokens);
        span.record("gen_ai.output.messages", serde_json::to_string(&vec![message_output]).expect("Converting from a Rust struct should always convert to JSON without failing"));

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage.clone()
        }));
    }.instrument(span);

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}
