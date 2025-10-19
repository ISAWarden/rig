#[cfg(test)]
mod tests {
    use crate::providers::llama_cpp::Client;
    use crate::client::completion::CompletionClient;
    use crate::completion::{Prompt, CompletionRequest};
    use crate::completion::request::CompletionModel as CompletionModelTrait;
    use crate::agent::{AgentBuilder, Agent};
    use crate::tool::Tool;
    use rig::rig_tool;
    use std::process::{Command, Child};
    use std::time::Duration;
    use tokio::time::sleep;
    use std::sync::Once;

    static INIT: Once = Once::new();

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

    // Struct to hold the calculator tool
    pub struct CalculatorTool;
    
    // Create an instance of the calculator tool
    pub fn calculator_tool() -> CalculatorTool {
        CalculatorTool
    }
    
    // Implement the Tool trait for the calculator struct
    impl Tool for CalculatorTool {
        const NAME: &'static str = "calculator";
        
        type Error = rig::tool::ToolError;
        type Args = CalculatorArgs;
        type Output = i32;
        
        async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
            crate::completion::ToolDefinition {
                name: "calculator".to_string(),
                description: "Perform basic arithmetic operations".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "integer",
                            "description": "Left side of operation"
                        },
                        "y": {
                            "type": "integer",
                            "description": "Right side of operation"
                        },
                        "operation": {
                            "type": "string",
                            "description": "Can either be 'add', 'subtract', 'multiply' or 'divide'"
                        }
                    },
                    "required": ["x", "y", "operation"]
                })
            }
        }
        
        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            calculator(args.x, args.y, args.operation)
        }
    }
    
    #[derive(serde::Deserialize)]
    pub struct CalculatorArgs {
        pub x: i32,
        pub y: i32,
        pub operation: String,
    }

    // Helper function to check if the LlamaCpp server is running
    async fn is_server_running() -> bool {
        match reqwest::get("http://localhost:8402/v1/models").await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    // Helper function to check if the model is ready
    async fn is_model_ready() -> bool {
        if let Ok(response) = reqwest::get("http://localhost:8402/v1/models").await {
            if response.status().is_success() {
                if let Ok(text) = response.text().await {
                    return text.contains("Qwen3") && !text.contains("Loading model");
                }
            }
        }
        false
    }

    // Helper function to start the LlamaCpp server
    async fn start_server() -> Result<Child, Box<dyn std::error::Error>> {
        println!("Starting LlamaCpp server...");
        
        let server_path = "/Users/peter/Library/Application Support/com.llama_cpp_toolchain.llama_cpp/llama_cpp_b6785_Metal/bin/llama-server";
        let model_path = "/Users/peter/Library/Application Support/com.isawarden.chat.staging/model/Qwen3-8B-Q8_0.gguf";
        
        let child = Command::new(server_path)
            .arg("-m")
            .arg(model_path)
            .arg("--port")
            .arg("8402")
            .arg("-v")
            .arg("--log-prefix")
            .arg("--ctx-size")
            .arg("4096")
            .arg("--jinja")
            .spawn()?;
            
        // Give the server some time to start up
        println!("Waiting for server to start...");
        for _ in 0..30 {
            sleep(Duration::from_secs(2)).await;
            if is_server_running().await {
                println!("Server is now running!");
                break;
            }
        }
        
        // Wait additional time for the model to fully load
        println!("Waiting for model to load...");
        for _ in 0..60 {
            sleep(Duration::from_secs(2)).await;
            if is_model_ready().await {
                println!("Model is ready!");
                return Ok(child);
            }
        }
        
        Err("Server failed to start within timeout".into())
    }

    #[tokio::test]
    async fn test_llama_cpp_calculator_tool() {
        // Initialize logging
        INIT.call_once(|| {
            // Skip env_logger for now to avoid dependency issues
        });

        // Check if server is running, start it if needed
        let _server_child = if !is_server_running().await {
            Some(start_server().await.expect("Failed to start server"))
        } else {
            println!("Server is already running");
            // Wait for model to be ready if server is already running
            println!("Waiting for model to be ready...");
            for _ in 0..60 {
                sleep(Duration::from_secs(2)).await;
                if is_model_ready().await {
                    println!("Model is ready!");
                    break;
                }
            }
            None
        };

        // Create the LlamaCpp client
        let client = Client::new();
        
        // Create the completion model
        let model = client.completion_model("qwen3-8b");
        
        // Create an agent with the calculator tool
        let agent = AgentBuilder::new(model)
            .tool(calculator_tool())
            .preamble("You are a helpful assistant that can perform calculations using the calculator tool when needed.")
            .build();

        // Test a simple calculation
        let prompt = "What is 15 + 27? Please use the calculator tool to find the answer.";
        
        let response = agent.prompt(prompt).await.expect("Failed to get response");
        
        println!("Agent response: {}", response);
        
        // Verify the response contains the correct answer
        assert!(response.contains("42") || response.contains("forty-two"), 
                "Response should contain the correct answer (42). Got: {}", response);
        
        // Test another simple calculation
        let simple_prompt2 = "What is 10 * 5? Use the calculator tool.";
        
        let response2 = agent.prompt(simple_prompt2).await.expect("Failed to get second response");
        
        println!("Second agent response: {}", response2);
        
        // Verify the response contains the correct answer (50)
        assert!(response2.contains("50"),
                "Response should contain the correct answer (50). Got: {}", response2);
        
        println!("Test completed successfully!");
    }

    #[tokio::test]
    #[ignore] // Use `cargo test -- --ignored` to run this test
    async fn test_llama_cpp_basic_completion() {
        // Initialize logging
        INIT.call_once(|| {
            // Skip env_logger for now to avoid dependency issues
        });

        // Check if server is running, start it if needed
        let _server_child = if !is_server_running().await {
            Some(start_server().await.expect("Failed to start server"))
        } else {
            println!("Server is already running");
            None
        };

        // Create the LlamaCpp client
        let client = Client::new();
        
        // Create the completion model
        let model = client.completion_model("qwen3-8b");
        
        // Create a simple completion request
        let request = CompletionRequest {
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: crate::OneOrMany::many(vec![]).unwrap_or_else(|_| crate::OneOrMany::one(crate::completion::Message::user(""))),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };
        
        let response = model.completion(request).await.expect("Failed to get completion response");
        
        println!("Completion response: {:?}", response);
        
        // Verify the response contains information about Paris
        let response_text = format!("{:?}", response.choice);
        assert!(response_text.to_lowercase().contains("paris"), 
                "Response should mention Paris. Got: {}", response_text);
        
        println!("Basic completion test completed successfully!");
    }

    #[tokio::test]
    #[ignore] // Use `cargo test -- --ignored` to run this test
    async fn test_llama_cpp_calculator_tool_streaming() {
        // Initialize logging
        INIT.call_once(|| {
            // Skip env_logger for now to avoid dependency issues
        });

        // Check if server is running, start it if needed
        let _server_child = if !is_server_running().await {
            Some(start_server().await.expect("Failed to start server"))
        } else {
            println!("Server is already running");
            // Wait for model to be ready if server is already running
            println!("Waiting for model to be ready...");
            for _ in 0..60 {
                sleep(Duration::from_secs(2)).await;
                if is_model_ready().await {
                    println!("Model is ready!");
                    break;
                }
            }
            None
        };

        // Create the LlamaCpp client
        let client = Client::new();
        
        // Create the completion model
        let model = client.completion_model("qwen3-8b");
        
        // Create an agent with the calculator tool
        let agent = AgentBuilder::new(model)
            .tool(calculator_tool())
            .preamble("You are a helpful assistant that can perform calculations using the calculator tool when needed.")
            .build();

        // Test a simple calculation with streaming
        let prompt = "What is 15 + 27? Please use the calculator tool to find the answer.";
        
        println!("Starting streaming test with prompt: {}", prompt);
        
        // Create a streaming request
        let mut stream = agent.stream_prompt(prompt).await.expect("Failed to create stream");
        
        // Stream to stdout
        use rig::agent::stream_to_stdout;
        let result = stream_to_stdout(&mut stream).await.expect("Failed to stream to stdout");
        
        println!("\nStreaming test completed successfully!");
        println!("Final result: {:?}", result);
        
        // Verify the result contains the correct answer
        let result_text = format!("{:?}", result);
        assert!(result_text.contains("42") || result_text.contains("forty-two"),
                "Result should contain the correct answer (42). Got: {}", result_text);
    }
}
