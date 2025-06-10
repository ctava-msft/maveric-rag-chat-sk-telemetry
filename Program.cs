using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.TextGeneration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel.ChatCompletion;
using System;
using System.Threading.Tasks;
using DotNetEnv;
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using SharpToken; // Add this using directive
using Azure.Monitor.OpenTelemetry.Exporter; // Add this for Application Insights
using System.IO; // Add this for file operations

class Program
{
    // Create static meter for metrics
    private static readonly Meter s_meter = new("AzureSK.Streaming");
    private static readonly Counter<int> s_totalTokensCounter = s_meter.CreateCounter<int>("sk.total_tokens");
    private static readonly Counter<int> s_inputTokensCounter = s_meter.CreateCounter<int>("sk.input_tokens");
    private static readonly Counter<int> s_completionTokensCounter = s_meter.CreateCounter<int>("sk.completion_tokens");
    private static readonly Histogram<double> s_latencyHistogram = s_meter.CreateHistogram<double>("sk.latency_ms");
    
    // Create ActivitySource for tracing
    private static readonly ActivitySource s_activitySource = new("AzureSK.Streaming");
    
    // Create tokenizer for accurate token counting
    private static readonly GptEncoding s_tokenizer = GptEncoding.GetEncoding("cl100k_base"); // GPT-4 encoding

    // Helper method for accurate token counting
    private static int CountTokens(string text)
    {
        if (string.IsNullOrEmpty(text)) return 0;
        return s_tokenizer.Encode(text).Count;
    }

    static async Task Main(string[] args)
    {
        // Load environment variables from .env file
        Env.Load();
        
        // Get Application Insights connection string
        var connectionString = Environment.GetEnvironmentVariable("APPLICATIONINSIGHTS_CONNECTION_STRING");
        if (string.IsNullOrEmpty(connectionString))
        {
            throw new InvalidOperationException("APPLICATIONINSIGHTS_CONNECTION_STRING environment variable is not set");
        }
        
        Console.WriteLine($"Application Insights Connection String configured: {connectionString[..20]}...");
        
        // Create a kernel builder
        var builder = Kernel.CreateBuilder();
        
        // Add logging
        builder.Services.AddLogging(logging => 
        {
            logging.AddConsole();
            logging.SetMinimumLevel(LogLevel.Information);
        });

        Console.WriteLine("Configuring OpenTelemetry with Application Insights...");

        // Configure OpenTelemetry with Application Insights
        builder.Services.AddOpenTelemetry()
            .ConfigureResource(resource => resource
                .AddService("AzureSK.Streaming", "1.0.0")
                .AddAttributes(new Dictionary<string, object>
                {
                    ["service.instance.id"] = Environment.MachineName,
                    ["deployment.environment"] = "development"
                }))
            .WithTracing(tracing => tracing
                .AddSource("AzureSK.Streaming")
                .AddSource("Microsoft.SemanticKernel*")
                .AddAzureMonitorTraceExporter(options =>
                {
                    options.ConnectionString = connectionString;
                    Console.WriteLine("Azure Monitor Trace Exporter configured successfully");
                })
                .AddConsoleExporter()) // Keep console for local debugging
            .WithMetrics(metrics => metrics
                .AddMeter("AzureSK.Streaming")
                .AddMeter("Microsoft.SemanticKernel*")
                .AddAzureMonitorMetricExporter(options =>
                {
                    options.ConnectionString = connectionString;
                    Console.WriteLine("Azure Monitor Metric Exporter configured successfully");
                })
                .AddConsoleExporter()); // Keep console for local debugging

        Console.WriteLine("Adding Azure OpenAI Chat Completion service...");
        builder.AddAzureOpenAIChatCompletion(
            deploymentName: "gpt-4o-mini",
            endpoint: Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? "https://jpggz.openai.azure.com",
            apiKey: Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY") ?? "e9ee56f0b75648d6a9f10ebd358db260");

        var kernel = builder.Build();
        var logger = kernel.GetRequiredService<ILogger<Program>>();
        
        logger.LogInformation("Application started with Application Insights integration");
        logger.LogInformation("OpenTelemetry configured with connection string: {ConnectionStringPrefix}...", connectionString[..20]);
        logger.LogInformation("Service name: AzureSK.Streaming, Version: 1.0.0");
        logger.LogInformation("Machine name: {MachineName}, Environment: development", Environment.MachineName);

        // Import plugins from the prompts directory
        var pluginDirectoryPath = "./Prompts";
        logger.LogInformation("Importing plugins from {PluginDirectory}", pluginDirectoryPath);
        KernelPlugin plugins = kernel.ImportPluginFromPromptDirectory(pluginDirectoryPath);
        
        // create user prompt and mock context.
        string userPrompt = "What information can you provide about Tricare coverage? Please include details about different plans, eligibility requirements, costs, and benefits. Format your response using numbered bullet points (1., 2., 3., etc.) to organize the information clearly.";
        string mockContext = @"Tricare is a health care program for uniformed service members and their families. 
        It offers several plans including Tricare Prime, Tricare Select, Tricare for Life, and others. 
        Coverage varies by plan and includes medical, dental, and pharmacy benefits.";
        
        // Get the TricareManual plugin function
        var tricareManualFunction = plugins["TricareManual"];
        logger.LogInformation("Retrieved TricareManual function from plugins");
        
        // // Print the prompt template to help diagnose issues
        // Console.WriteLine("Plugin prompt template:");
        // // Access the prompt template using the correct property
        // var promptTemplate = tricareManualFunction.Metadata.PromptTemplate;
        // Console.WriteLine(promptTemplate ?? "Prompt template not available");
        // Console.WriteLine("\n");
        
        // Execute the plugin function with streaming
        Console.WriteLine("Streaming response from plugin:");
        
        try
        {
            // Create activity for tracing
            using var activity = s_activitySource.StartActivity("InvokePluginFunction", ActivityKind.Client);
            activity?.SetTag("function.name", "TricareManual");
            activity?.SetTag("user.prompt.length", userPrompt.Length);
            activity?.SetTag("service.name", "AzureSK.Streaming");
            activity?.SetTag("ai.model.name", "gpt-4o-mini");
            
            logger.LogInformation("Started activity: {ActivityId} for plugin function invocation", activity?.Id);
            
            // Calculate input tokens
            var pluginInputText = $"{userPrompt}\n{mockContext}";
            int inputTokens = CountTokens(pluginInputText);
            
            // Record input tokens with additional dimensions
            s_inputTokensCounter.Add(inputTokens, new KeyValuePair<string, object?>("operation", "plugin_function"));
            activity?.SetTag("tokens.input", inputTokens);
            logger.LogInformation("Plugin input tokens: {InputTokens}", inputTokens);
            logger.LogInformation("Recording input tokens metric: {InputTokens} for operation: plugin_function", inputTokens);
            
            var stopwatch = Stopwatch.StartNew();
            
            logger.LogInformation("Invoking TricareManual function with streaming");
            
            // Generate random suffix for output file
            var random = new Random();
            var randomSuffix = random.Next(100, 999).ToString();
            var outputFileName = $"output-{randomSuffix}.md";
            
            Console.WriteLine($"Writing chunks to file: {outputFileName}");
            logger.LogInformation("Created output file: {FileName}", outputFileName);
            
            // Use the correct streaming API
            var streamingResult = kernel.InvokeStreamingAsync(tricareManualFunction, 
                new() { 
                    {"userPrompt", userPrompt}, 
                    {"context", mockContext} 
                });
                
            // Option to collect the full response for verification
            var fullResponse = new System.Text.StringBuilder();
            int chunkCount = 0;
            
            // Create and write to markdown file - keep it open for both responses
            using (var fileWriter = new StreamWriter(outputFileName))
            {
                // Write markdown header
                await fileWriter.WriteLineAsync("# Tricare Coverage Information");
                await fileWriter.WriteLineAsync();
                await fileWriter.WriteLineAsync($"**Generated on:** {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                await fileWriter.WriteLineAsync($"**User Question:** {userPrompt}");
                await fileWriter.WriteLineAsync();
                await fileWriter.WriteLineAsync("## Plugin Function Response");
                await fileWriter.WriteLineAsync();
                
                // Process the streaming chunks
                var responseStream = new MemoryStream();
                var streamWriter = new StreamWriter(responseStream);
                
                await foreach (var chunk in streamingResult)
                {
                    chunkCount++;
                    string chunkText = chunk.ToString();
                    int chunkTokens = CountTokens(chunkText);
                    s_completionTokensCounter.Add(chunkTokens);
                    
                    activity?.AddEvent(new ActivityEvent("ChunkReceived", 
                        tags: new ActivityTagsCollection { 
                            { "chunk.size", chunkText.Length },
                            { "chunk.tokens", chunkTokens }
                        }));
                    
                    Console.Write(chunk);
                    Console.Out.Flush();
                    fullResponse.Append(chunkText);
                    
                    // Write chunk to stream for line-by-line processing
                    await streamWriter.WriteAsync(chunkText);
                    await streamWriter.FlushAsync();
                }
                
                await streamWriter.FlushAsync();
                responseStream.Position = 0;
                
                // Stream Response line by line to file
                using (var reader = new StreamReader(responseStream))
                {
                    while (!reader.EndOfStream)
                    {
                        string line = await reader.ReadLineAsync();
                        if (line != null)
                        {
                            await fileWriter.WriteLineAsync(line);
                            await fileWriter.FlushAsync();
                        }
                    }
                }
                
                // Write plugin section footer
                await fileWriter.WriteLineAsync();
                await fileWriter.WriteLineAsync();
                await fileWriter.WriteLineAsync("---");
                await fileWriter.WriteLineAsync($"**Plugin Response - Total chunks:** {chunkCount}");
                await fileWriter.WriteLineAsync($"**Plugin Response - Length:** {fullResponse.Length} characters");
                await fileWriter.WriteLineAsync();
                
                // Now try direct chat completion for comparison
                Console.WriteLine("\n\nTrying direct chat completion:");
                await fileWriter.WriteLineAsync("## Direct Chat Completion Response");
                await fileWriter.WriteLineAsync();
                
                // Get the chat completion service
                var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
                
                // Create a direct prompt with the same information
                string directPrompt = $@"
You are an expert on Tricare health insurance.
Context information: {mockContext}

User question: {userPrompt}

Please provide a detailed and comprehensive answer. Format your response using numbered bullet points (1., 2., 3., etc.) to organize the information clearly. Use sub-bullets with letters (a., b., c., etc.) for additional details under each main point.
";
            
                // Execute with streaming using a new activity
                using var directActivity = s_activitySource.StartActivity("DirectChatCompletion", ActivityKind.Client);
                directActivity?.SetTag("prompt.length", directPrompt.Length);
                directActivity?.SetTag("service.name", "AzureSK.Streaming");
                directActivity?.SetTag("ai.model.name", "gpt-4o-mini");
            
                logger.LogInformation("Started direct chat activity: {ActivityId}", directActivity?.Id);
            
                // Calculate direct chat input tokens
                int directInputTokens = CountTokens(directPrompt);
                s_inputTokensCounter.Add(directInputTokens, new KeyValuePair<string, object?>("operation", "direct_chat"));
                directActivity?.SetTag("tokens.input", directInputTokens);
                logger.LogInformation("Direct chat input tokens: {InputTokens}", directInputTokens);
            
                var directStopwatch = Stopwatch.StartNew();
                logger.LogInformation("Starting direct chat completion streaming");
            
                // Using the correct method for ITextGenerationService
                var chatHistory = new ChatHistory();
                chatHistory.AddUserMessage(directPrompt);
            
                var directStreamingResult = chatCompletionService.GetStreamingChatMessageContentsAsync(
                    chatHistory,
                    new OpenAIPromptExecutionSettings { 
                        MaxTokens = 2000,
                        Temperature = 0.7f
                    }
                );
                
                var directFullResponse = new System.Text.StringBuilder();
                int directChunkCount = 0;
                
                // Fixed: Use await foreach for the IAsyncEnumerable
                var directResponseStream = new MemoryStream();
                var directStreamWriter = new StreamWriter(directResponseStream);
                
                await foreach (var chunk in directStreamingResult)
                {
                    directChunkCount++;
                    string chunkText = chunk.Content ?? "";
                    int chunkTokens = CountTokens(chunkText);
                    s_completionTokensCounter.Add(chunkTokens);
                    
                    directActivity?.AddEvent(new ActivityEvent("ChunkReceived", 
                        tags: new ActivityTagsCollection { 
                            { "chunk.size", chunkText.Length },
                            { "chunk.tokens", chunkTokens }
                        }));
                    
                    Console.Write(chunk.Content);
                    Console.Out.Flush();
                    directFullResponse.Append(chunk.Content);
                    
                    // Write chunk to stream for line-by-line processing
                    await directStreamWriter.WriteAsync(chunkText);
                    await directStreamWriter.FlushAsync();
                }
                
                await directStreamWriter.FlushAsync();
                directResponseStream.Position = 0;
                
                // Stream Response line by line to file
                using (var directReader = new StreamReader(directResponseStream))
                {
                    while (!directReader.EndOfStream)
                    {
                        string line = await directReader.ReadLineAsync();
                        if (line != null)
                        {
                            await fileWriter.WriteLineAsync(line);
                            await fileWriter.FlushAsync();
                        }
                    }
                }
                
                // Write final footer with both responses
                await fileWriter.WriteLineAsync();
                await fileWriter.WriteLineAsync();
                await fileWriter.WriteLineAsync("---");
                await fileWriter.WriteLineAsync($"**Direct Chat - Total chunks:** {directChunkCount}");
                await fileWriter.WriteLineAsync($"**Direct Chat - Length:** {directFullResponse.Length} characters");
                await fileWriter.WriteLineAsync();
                await fileWriter.WriteLineAsync("## Summary");
                await fileWriter.WriteLineAsync($"- **Total plugin chunks:** {chunkCount}");
                await fileWriter.WriteLineAsync($"- **Total direct chat chunks:** {directChunkCount}");
                await fileWriter.WriteLineAsync($"- **Plugin response length:** {fullResponse.Length} characters");
                await fileWriter.WriteLineAsync($"- **Direct chat response length:** {directFullResponse.Length} characters");
                
                // Store values for later logging
                var finalDirectChunkCount = directChunkCount;
                var finalDirectFullResponse = directFullResponse.ToString();
                var finalDirectElapsedMs = directStopwatch.Elapsed.TotalMilliseconds;
                var finalDirectCompletionTokens = CountTokens(finalDirectFullResponse);
                
                // Record metrics for direct chat
                s_latencyHistogram.Record(finalDirectElapsedMs, new KeyValuePair<string, object?>("operation", "direct_chat"));
                s_totalTokensCounter.Add(directInputTokens + finalDirectCompletionTokens, new KeyValuePair<string, object?>("operation", "direct_chat"));
                
                logger.LogInformation("Recording latency metric: {ElapsedMs}ms for operation: direct_chat", finalDirectElapsedMs);
                logger.LogInformation("Recording total tokens metric: {TotalTokens} for operation: direct_chat", directInputTokens + finalDirectCompletionTokens);
                
                directActivity?.SetTag("response.total_chunks", finalDirectChunkCount);
                directActivity?.SetTag("response.length", finalDirectFullResponse.Length);
                directActivity?.SetTag("tokens.completion", finalDirectCompletionTokens);
                directActivity?.SetTag("tokens.total", directInputTokens + finalDirectCompletionTokens);
                directActivity?.SetTag("response.duration_ms", finalDirectElapsedMs);
                
                logger.LogInformation("Direct streaming completed. Received {ChunkCount} chunks with {CompletionTokens} completion tokens in {ElapsedMs:F2}ms", 
                    finalDirectChunkCount, finalDirectCompletionTokens, finalDirectElapsedMs);
                logger.LogInformation("Direct chat activity completed: {ActivityId} with {TagCount} tags", directActivity?.Id, directActivity?.TagObjects.Count());
            } // File is closed here after both responses are written
            
            stopwatch.Stop();
            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
            
            // Calculate final completion tokens
            int completionTokens = CountTokens(fullResponse.ToString());
            
            // Record metrics
            s_latencyHistogram.Record(elapsedMs, new KeyValuePair<string, object?>("operation", "plugin_function"));
            s_totalTokensCounter.Add(inputTokens + completionTokens, new KeyValuePair<string, object?>("operation", "plugin_function"));
            
            logger.LogInformation("Recording latency metric: {ElapsedMs}ms for operation: plugin_function", elapsedMs);
            logger.LogInformation("Recording total tokens metric: {TotalTokens} for operation: plugin_function", inputTokens + completionTokens);
            
            activity?.SetTag("response.total_chunks", chunkCount);
            activity?.SetTag("response.length", fullResponse.Length);
            activity?.SetTag("tokens.completion", completionTokens);
            activity?.SetTag("tokens.total", inputTokens + completionTokens);
            activity?.SetTag("response.duration_ms", elapsedMs);
            activity?.SetTag("output.file", outputFileName);
            
            logger.LogInformation("Plugin streaming completed. Received {ChunkCount} chunks with {CompletionTokens} completion tokens in {ElapsedMs:F2}ms", 
                chunkCount, completionTokens, elapsedMs);
            logger.LogInformation("Output written to file: {FileName}", outputFileName);
            logger.LogInformation("Activity completed: {ActivityId} with {TagCount} tags", activity?.Id, activity?.TagObjects.Count());
            
            Console.WriteLine("\n\n--- Plugin Streaming Complete ---");
            Console.WriteLine($"Full response length: {fullResponse.Length} characters");
            Console.WriteLine("\n\n--- Direct Streaming Complete ---");
            Console.WriteLine($"Both responses saved to: {outputFileName}");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error during streaming operation");
            Console.WriteLine($"\nError during streaming: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
            }
        }
        
        logger.LogInformation("Application completed - preparing to flush telemetry");
        logger.LogInformation("Total metrics recorded: Input tokens, Completion tokens, Total tokens, Latency");
        logger.LogInformation("Total activities created: 2 (Plugin function, Direct chat)");
        
        // Ensure all telemetry is flushed before application exits
        Console.WriteLine("Flushing telemetry data to Application Insights...");
        logger.LogInformation("Waiting 5 seconds for telemetry flush to Application Insights");
        await Task.Delay(5000); // Give time for telemetry to be sent
        
        Console.WriteLine("Telemetry flush completed. Check Application Insights for data.");
        logger.LogInformation("Application shutdown complete");
    }
}
