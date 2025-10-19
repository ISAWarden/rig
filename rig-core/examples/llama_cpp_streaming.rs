use rig::agent::stream_to_stdout;
use rig::providers::llama_cpp;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::agent::Agent;
use rig::streaming::StreamingPrompt;

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
        .preamble("You are a helpful assistant that provides detailed and thoughtful answers.")
        .build();

    // Create a streaming request
    let prompt = "Explain the concept of quantum computing in simple terms.";
    
    println!("Sending prompt: {}", prompt);
    println!("--- Streaming Response ---");
    
    let mut stream = agent.stream_prompt(prompt).await;
    
    // Stream the response to stdout
    let result = stream_to_stdout(&mut stream).await?;
    
    println!("\n--- End of Streaming ---");
    println!("Final result: {:?}", result);
    
    Ok(())
}