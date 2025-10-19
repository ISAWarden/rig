use anyhow::Result;
use rig::agent::stream_to_stdout;
use rig::providers::llama_cpp;
use rig::client::CompletionClient;
use rig_derive::rig_tool;

#[rig_tool(
    description = "Perform basic arithmetic operations",
    params(
        x = "Left side of operation",
        y = "Right side of operation",
        operation = "Can either be 'add', 'subtract', 'multiply' or 'divide'"
    ),
    required(x, y, operation)
)]
pub fn calculator(x: i32, y: i32, operation: String) -> Result<i32, rig::tool::ToolError> {
    println!("Ran calculator! {} {} {}", x, y, operation);
    match operation.as_str() {
        "add" => Ok(x + y),
        "subtract" => Ok(x - y),
        "multiply" => Ok(x * y),
        "divide" => {
            if y == 0 {
                Err(rig::tool::ToolError::ToolCallError(
                    "Division by zero".into(),
                ))
            } else {
                Ok(x / y)
            }
        }
        _ => Err(rig::tool::ToolError::ToolCallError(
            format!("Unknown operation: {operation}").into(),
        )),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create LlamaCpp client
    // This assumes you have a LlamaCpp server running on localhost:8402
    let llama_cpp_client = llama_cpp::Client::builder()
        .base_url("http://localhost:8402/v1")
        .build();

    // Create an agent with the Qwen3-8B-Q8_0.gguf model and calculator tool
    let agent = llama_cpp_client
        .agent("Qwen3-8B-Q8_0.gguf")
        .preamble("You are a helpful assistant that can perform calculations using the calculator tool when needed.")
        .tool(calculator)
        .build();

    // Test a calculation with streaming
    let prompt = "What is 123 + 456? Please use the calculator tool to find the answer and explain your steps.";
    
    println!("Sending prompt: {}", prompt);
    println!("--- Streaming Response with Tools ---");
    
    let mut stream = agent.stream_prompt(prompt).await?;
    
    // Stream the response to stdout
    let result = stream_to_stdout(&mut stream).await?;
    
    println!("\n--- End of Streaming ---");
    println!("Final result: {:?}", result);
    
    // Test another calculation
    let prompt2 = "Calculate 25 * 4 and then subtract 17 from the result.";
    
    println!("\nSending second prompt: {}", prompt2);
    println!("--- Streaming Response with Tools ---");
    
    let mut stream2 = agent.stream_prompt(prompt2).await?;
    
    // Stream the response to stdout
    let result2 = stream_to_stdout(&mut stream2).await?;
    
    println!("\n--- End of Streaming ---");
    println!("Final result: {:?}", result2);
    
    Ok(())
}