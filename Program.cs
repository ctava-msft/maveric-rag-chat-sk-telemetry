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

class Program
{
    // Create static meter for metrics
    private static readonly Meter s_meter = new("AzureSK.Streaming");
    private static readonly Counter<int> s_totalTokensCounter = s_meter.CreateCounter<int>("sk.total_tokens");
    private static readonly Histogram<double> s_latencyHistogram = s_meter.CreateHistogram<double>("sk.latency_ms");
    
    // Create ActivitySource for tracing
    private static readonly ActivitySource s_activitySource = new("AzureSK.Streaming");

    static async Task Main(string[] args)
    {
        // Load environment variables from .env file
        Env.Load();
        
        // Create a kernel builder
        var builder = Kernel.CreateBuilder();
        
        // Add logging
        builder.Services.AddLogging(logging => 
        {
            logging.AddConsole();
            logging.SetMinimumLevel(LogLevel.Information);
        });

        // Configure OpenTelemetry
        builder.Services.AddOpenTelemetry()
            .ConfigureResource(resource => resource
                .AddService("AzureSK.Streaming"))
            .WithTracing(tracing => tracing
                .AddSource("AzureSK.Streaming")
                .AddSource("Microsoft.SemanticKernel*")
                .AddConsoleExporter())
            .WithMetrics(metrics => metrics
                .AddMeter("AzureSK.Streaming")
                .AddMeter("Microsoft.SemanticKernel*")
                .AddConsoleExporter());

        builder.AddAzureOpenAIChatCompletion(
            deploymentName: "gpt-4o-mini",
            endpoint: "https://jpggz.openai.azure.com",
            apiKey: "e9ee56f0b75648d6a9f10ebd358db260");

        var kernel = builder.Build();
        var logger = kernel.GetRequiredService<ILogger<Program>>();
        
        logger.LogInformation("Application started");

        // Import plugins from the prompts directory
        var pluginDirectoryPath = "./Prompts";
        logger.LogInformation("Importing plugins from {PluginDirectory}", pluginDirectoryPath);
        KernelPlugin plugins = kernel.ImportPluginFromPromptDirectory(pluginDirectoryPath);
        
        // create user prompt and mock context.
        string userPrompt = "What information can you provide about Tricare coverage? Please include details about different plans, eligibility requirements, costs, and benefits.";
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
            
            var stopwatch = Stopwatch.StartNew();
            
            logger.LogInformation("Invoking TricareManual function with streaming");
            
            // Use the correct streaming API
            var streamingResult = kernel.InvokeStreamingAsync(tricareManualFunction, 
                new() { 
                    {"userPrompt", userPrompt}, 
                    {"context", mockContext} 
                });
                
            // Option to collect the full response for verification
            var fullResponse = new System.Text.StringBuilder();
            int chunkCount = 0;
            
            // Process the streaming chunks
            await foreach (var chunk in streamingResult)
            {
                chunkCount++;
                activity?.AddEvent(new ActivityEvent("ChunkReceived", 
                    tags: new ActivityTagsCollection { { "chunk.size", chunk.ToString().Length } }));
                
                Console.Write(chunk);
                Console.Out.Flush();
                fullResponse.Append(chunk);
            }
            
            stopwatch.Stop();
            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
            
            // Record metrics
            s_latencyHistogram.Record(elapsedMs);
            s_totalTokensCounter.Add(fullResponse.Length / 4); // Rough token estimate
            
            activity?.SetTag("response.total_chunks", chunkCount);
            activity?.SetTag("response.length", fullResponse.Length);
            activity?.SetTag("response.duration_ms", elapsedMs);
            
            logger.LogInformation("Plugin streaming completed. Received {ChunkCount} chunks with total length {TotalLength} in {ElapsedMs:F2}ms", 
                chunkCount, fullResponse.Length, elapsedMs);
            
            Console.WriteLine("\n\n--- Plugin Streaming Complete ---");
            Console.WriteLine($"Full response length: {fullResponse.Length} characters");
            
            // Now try direct chat completion for comparison
            Console.WriteLine("\n\nTrying direct chat completion:");
            
            // Get the chat completion service
            var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
            
            // Create a direct prompt with the same information
            string directPrompt = $@"
You are an expert on Tricare health insurance.
Context information: {mockContext}

User question: {userPrompt}

Please provide a detailed and comprehensive answer.
";
            
            // Execute with streaming using a new activity
            using var directActivity = s_activitySource.StartActivity("DirectChatCompletion", ActivityKind.Client);
            directActivity?.SetTag("prompt.length", directPrompt.Length);
            
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
            await foreach (var chunk in directStreamingResult)
            {
                directChunkCount++;
                directActivity?.AddEvent(new ActivityEvent("ChunkReceived", 
                    tags: new ActivityTagsCollection { { "chunk.size", chunk.Content?.Length ?? 0 } }));
                
                Console.Write(chunk.Content);
                Console.Out.Flush();
                directFullResponse.Append(chunk.Content);
            }
            
            directStopwatch.Stop();
            double directElapsedMs = directStopwatch.Elapsed.TotalMilliseconds;
            
            // Record metrics
            s_latencyHistogram.Record(directElapsedMs);
            s_totalTokensCounter.Add(directFullResponse.Length / 4); // Rough token estimate
            
            directActivity?.SetTag("response.total_chunks", directChunkCount);
            directActivity?.SetTag("response.length", directFullResponse.Length);
            directActivity?.SetTag("response.duration_ms", directElapsedMs);
            
            logger.LogInformation("Direct streaming completed. Received {ChunkCount} chunks with total length {TotalLength} in {ElapsedMs:F2}ms", 
                directChunkCount, directFullResponse.Length, directElapsedMs);
            
            Console.WriteLine("\n\n--- Direct Streaming Complete ---");
            Console.WriteLine($"Direct response length: {directFullResponse.Length} characters");
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
        
        logger.LogInformation("Application completed");
    }
}
