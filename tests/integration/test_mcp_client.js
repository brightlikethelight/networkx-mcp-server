/**
 * Integration tests for NetworkX MCP server with TypeScript/JavaScript SDK
 * 
 * Tests the server with the official @modelcontextprotocol/sdk client
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { spawn } from "child_process";
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import assert from 'assert';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..', '..');

/**
 * Create and connect MCP client to NetworkX server
 */
async function createMCPClient() {
    const client = new Client({
        name: "test-client",
        version: "1.0.0"
    }, { 
        capabilities: {} 
    });

    const transport = new StdioClientTransport({
        command: "python",
        args: ["-m", "networkx_mcp", "--jsonrpc"],
        env: {
            ...process.env,
            PYTHONPATH: join(projectRoot, "src")
        }
    });

    await client.connect(transport);
    return { client, transport };
}

/**
 * Test basic tool discovery
 */
async function testToolDiscovery() {
    console.log("Testing tool discovery...");
    
    const { client, transport } = await createMCPClient();
    
    try {
        const tools = await client.listTools();
        
        assert(tools.tools.length >= 15, "Should have at least 15 tools");
        
        // Check for specific tools
        const toolNames = tools.tools.map(t => t.name);
        assert(toolNames.includes("create_graph"), "Should have create_graph tool");
        assert(toolNames.includes("add_nodes"), "Should have add_nodes tool");
        assert(toolNames.includes("add_edges"), "Should have add_edges tool");
        assert(toolNames.includes("graph_info"), "Should have graph_info tool");
        assert(toolNames.includes("shortest_path"), "Should have shortest_path tool");
        
        console.log("✅ Tool discovery test passed");
        console.log(`   Found ${tools.tools.length} tools`);
        
    } finally {
        await transport.close();
    }
}

/**
 * Test graph creation
 */
async function testGraphCreation() {
    console.log("\nTesting graph creation...");
    
    const { client, transport } = await createMCPClient();
    
    try {
        const result = await client.callTool({
            name: "create_graph",
            arguments: {
                name: "test_graph",
                graph_type: "undirected"
            }
        });
        
        // Parse result
        const content = JSON.parse(result.content[0].text);
        assert(content.success === true, "Graph creation should succeed");
        assert(content.name === "test_graph", "Graph name should match");
        assert(content.type === "undirected", "Graph type should match");
        
        console.log("✅ Graph creation test passed");
        
    } finally {
        await transport.close();
    }
}

/**
 * Test complete workflow
 */
async function testCompleteWorkflow() {
    console.log("\nTesting complete workflow...");
    
    const { client, transport } = await createMCPClient();
    
    try {
        // 1. Create graph
        let result = await client.callTool({
            name: "create_graph",
            arguments: {
                name: "workflow_graph",
                graph_type: "directed"
            }
        });
        
        let content = JSON.parse(result.content[0].text);
        assert(content.success === true, "Graph creation should succeed");
        
        // 2. Add nodes
        result = await client.callTool({
            name: "add_nodes",
            arguments: {
                graph_name: "workflow_graph",
                nodes: ["A", "B", "C", "D", "E"]
            }
        });
        
        content = JSON.parse(result.content[0].text);
        assert(content.nodes_added === 5, "Should add 5 nodes");
        
        // 3. Add edges
        result = await client.callTool({
            name: "add_edges",
            arguments: {
                graph_name: "workflow_graph",
                edges: [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["A", "E"]]
            }
        });
        
        content = JSON.parse(result.content[0].text);
        assert(content.edges_added === 5, "Should add 5 edges");
        
        // 4. Get graph info
        result = await client.callTool({
            name: "graph_info",
            arguments: {
                graph_name: "workflow_graph"
            }
        });
        
        content = JSON.parse(result.content[0].text);
        assert(content.nodes === 5, "Should have 5 nodes");
        assert(content.edges === 5, "Should have 5 edges");
        
        // 5. Find shortest path
        result = await client.callTool({
            name: "shortest_path",
            arguments: {
                graph_name: "workflow_graph",
                source: "A",
                target: "C"
            }
        });
        
        content = JSON.parse(result.content[0].text);
        assert(Array.isArray(content.path), "Path should be an array");
        assert(content.path.join(",") === "A,B,C", "Path should be A->B->C");
        assert(content.length === 2, "Path length should be 2");
        
        console.log("✅ Complete workflow test passed");
        
    } finally {
        await transport.close();
    }
}

/**
 * Test error handling
 */
async function testErrorHandling() {
    console.log("\nTesting error handling...");
    
    const { client, transport } = await createMCPClient();
    
    try {
        // Try to get info for non-existent graph
        const result = await client.callTool({
            name: "graph_info",
            arguments: {
                graph_name: "non_existent_graph"
            }
        });
        
        // Should return error
        assert(result.isError === true, "Should return error for non-existent graph");
        assert(result.content[0].text.toLowerCase().includes("not found"), 
               "Error message should indicate graph not found");
        
        console.log("✅ Error handling test passed");
        
    } finally {
        await transport.close();
    }
}

/**
 * Test concurrent connections
 */
async function testConcurrentConnections() {
    console.log("\nTesting concurrent connections...");
    
    const numClients = 5;
    const clients = [];
    
    try {
        // Create multiple clients
        for (let i = 0; i < numClients; i++) {
            const clientData = await createMCPClient();
            clients.push(clientData);
        }
        
        // Perform operations concurrently
        const promises = clients.map(async (clientData, index) => {
            const { client } = clientData;
            
            // Create a graph for each client
            const result = await client.callTool({
                name: "create_graph",
                arguments: {
                    name: `concurrent_graph_${index}`,
                    graph_type: "undirected"
                }
            });
            
            const content = JSON.parse(result.content[0].text);
            return content.success;
        });
        
        const results = await Promise.all(promises);
        
        // All should succeed
        assert(results.every(r => r === true), "All concurrent operations should succeed");
        
        console.log("✅ Concurrent connections test passed");
        console.log(`   Successfully handled ${numClients} concurrent clients`);
        
    } finally {
        // Clean up all clients
        for (const { transport } of clients) {
            await transport.close();
        }
    }
}

/**
 * Generate Claude Desktop configuration
 */
function generateClaudeDesktopConfig() {
    console.log("\nGenerating Claude Desktop configuration...");
    
    const config = {
        "networkx-mcp": {
            "command": "python",
            "args": ["-m", "networkx_mcp", "--jsonrpc"],
            "env": {
                "PYTHONPATH": join(projectRoot, "src")
            }
        }
    };
    
    console.log("Claude Desktop configuration:");
    console.log(JSON.stringify(config, null, 2));
    
    console.log("\n✅ Configuration generated");
    console.log("   Add this to your Claude Desktop settings");
}

/**
 * Main test runner
 */
async function runAllTests() {
    console.log("🧪 NetworkX MCP Server - JavaScript SDK Integration Tests\n");
    
    try {
        await testToolDiscovery();
        await testGraphCreation();
        await testCompleteWorkflow();
        await testErrorHandling();
        await testConcurrentConnections();
        generateClaudeDesktopConfig();
        
        console.log("\n✨ All tests passed!");
        process.exit(0);
        
    } catch (error) {
        console.error("\n❌ Test failed:", error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run tests if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
    runAllTests();
}

// Export for use in other test suites
export {
    createMCPClient,
    testToolDiscovery,
    testGraphCreation,
    testCompleteWorkflow,
    testErrorHandling,
    testConcurrentConnections
};