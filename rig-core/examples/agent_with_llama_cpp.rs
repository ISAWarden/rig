use rig::providers::llama_cpp;
use rig::client::CompletionClient;
use rig::completion::Prompt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create LlamaCpp client
    // This assumes you have a LlamaCpp server running on localhost:8402
    let llama_cpp_client = llama_cpp::Client::builder()
        .base_url("http://localhost:8402/v1")
        .build();

    // Create an agent with the Qwen3-8B-Q8_0.gguf model
    let agent = llama_cpp_client
        .agent("Qwen3-8B-Q8_0.gguf")
        .preamble("You are a helpful assistant that provides concise and accurate answers.")
        .build();

    // Prompt the agent and print its response
    let response = agent
        .prompt("What is the capital of France?")
        .await?;

    println!("Agent: {}", response);

    Ok(())
}