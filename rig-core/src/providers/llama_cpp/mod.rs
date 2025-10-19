//! LlamaCpp API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::llama_cpp;
//!
//! // Create a new LlamaCpp client (defaults to http://localhost:8402/v1)
//! let client = llama_cpp::Client::new();
//!
//! // Create a completion model interface using, for example, the "qwen2.5" model
//! let comp_model = client.completion_model("qwen2.5");
//!
//! let agent = client.agent("qwen2.5")
//!     .preamble("You are a helpful assistant.")
//!     .build();
//! ```

pub mod client;
pub mod completion;

#[cfg(test)]
mod tests;

pub use client::*;
pub use completion::*;