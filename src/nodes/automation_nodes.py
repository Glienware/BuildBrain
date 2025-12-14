"""Complete n8n-style node catalog with all node types, ports, and execution logic."""

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field


class DataType(Enum):
    """Supported data types for node ports."""
    JSON = "json"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATASET = "dataset"
    MODEL = "model"
    METRICS = "metrics"
    ANY = "any"


@dataclass
class Port:
    """Input/output port with type information."""
    name: str
    data_type: DataType = DataType.ANY
    description: str = ""
    required: bool = True
    default_value: Any = None


@dataclass
class NodeConfig:
    """Node configuration and settings."""
    node_type: str
    display_name: str
    description: str
    category: str
    input_ports: Dict[str, Port] = field(default_factory=dict)
    output_ports: Dict[str, Port] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TRIGGER NODES
# ============================================================================

class ManualTrigger(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="manual_trigger",
            display_name="Manual Trigger",
            description="Start workflow manually",
            category="TRIGGERS",
            settings={
                "execution_name": "Untitled Workflow",
                "input_variables": {
                    "param_1": "default_value"
                },
                "raw_json_input": "{}",
                "mode": "test"
            },
            output_ports={
                "output": Port("output", DataType.JSON, "Execution context")
            }
        )


class CronTrigger(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="cron_trigger",
            display_name="Cron Trigger",
            description="Trigger on schedule",
            category="TRIGGERS",
            settings={"cron_expression": "0 0 * * *"},
            output_ports={
                "output": Port("output", DataType.JSON, "Execution context")
            }
        )


class WebhookTrigger(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="webhook_trigger",
            display_name="Webhook Trigger",
            description="Trigger on HTTP webhook",
            category="TRIGGERS",
            settings={"webhook_url": "", "method": "POST"},
            output_ports={
                "body": Port("body", DataType.JSON, "Request body"),
                "headers": Port("headers", DataType.OBJECT, "Request headers"),
                "query": Port("query", DataType.OBJECT, "Query parameters")
            }
        )


# ============================================================================
# CONTROL FLOW NODES
# ============================================================================

class IfElseNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="if_else",
            display_name="If / Else",
            description="Conditional branch",
            category="LOGIC",
            input_ports={
                "condition": Port("condition", DataType.BOOLEAN, "Condition to evaluate", required=True),
                "input": Port("input", DataType.ANY, "Data to pass through")
            },
            output_ports={
                "true": Port("true", DataType.ANY, "Output if true"),
                "false": Port("false", DataType.ANY, "Output if false")
            }
        )


class LoopNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="loop",
            display_name="Loop / ForEach",
            description="Iterate over array items",
            category="LOGIC",
            input_ports={
                "array": Port("array", DataType.ARRAY, "Array to iterate"),
                "item": Port("item", DataType.ANY, "Item to process")
            },
            output_ports={
                "item": Port("item", DataType.ANY, "Current item"),
                "index": Port("index", DataType.NUMBER, "Current index"),
                "result": Port("result", DataType.ARRAY, "Collected results")
            }
        )


class DelayNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="delay",
            display_name="Delay / Wait",
            description="Wait for a specified time",
            category="LOGIC",
            settings={"delay_ms": 1000},
            input_ports={
                "input": Port("input", DataType.ANY, "Data to pass")
            },
            output_ports={
                "output": Port("output", DataType.ANY, "Data after delay")
            }
        )


# ============================================================================
# AI / LLM NODES
# ============================================================================

class LLMNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="llm",
            display_name="LLM / Agent",
            description="Call language model",
            category="AI",
            settings={
                "model": "claude",
                "prompt": "Answer the following question:\n{{input}}",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            input_ports={
                "input": Port("input", DataType.STRING, "Input prompt or data"),
                "context": Port("context", DataType.JSON, "Additional context")
            },
            output_ports={
                "response": Port("response", DataType.STRING, "LLM response"),
                "parsed": Port("parsed", DataType.JSON, "Parsed JSON response"),
                "metadata": Port("metadata", DataType.JSON, "Response metadata")
            }
        )


class DatasetNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="dataset",
            display_name="Dataset",
            description="Load and summarize dataset",
            category="AI",
            settings={
                "dataset_path": "",
                "file_type": "csv"
            },
            output_ports={
                "summary": Port("summary", DataType.JSON, "Dataset info"),
                "data": Port("data", DataType.ARRAY, "Dataset rows"),
                "columns": Port("columns", DataType.ARRAY, "Column names")
            }
        )


class VectorStoreNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="vector_store",
            display_name="Vector Store",
            description="Embeddings and vector search",
            category="AI",
            settings={"model": "embedding_v1"},
            input_ports={
                "texts": Port("texts", DataType.ARRAY, "Texts to embed")
            },
            output_ports={
                "embeddings": Port("embeddings", DataType.ARRAY, "Embedding vectors"),
                "metadata": Port("metadata", DataType.JSON, "Embedding metadata")
            }
        )


# ============================================================================
# ACTION / SYSTEM NODES
# ============================================================================

class RunScriptNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="run_script",
            display_name="Run Script",
            description="Execute Python, JS, or Bash script",
            category="ACTIONS",
            settings={
                "language": "python",
                "script": "# Write your script here\nreturn input"
            },
            input_ports={
                "input": Port("input", DataType.ANY, "Input data")
            },
            output_ports={
                "output": Port("output", DataType.ANY, "Script output"),
                "error": Port("error", DataType.STRING, "Error message if any")
            }
        )


class HttpRequestNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="http_request",
            display_name="HTTP Request",
            description="Make HTTP request to API",
            category="ACTIONS",
            settings={
                "url": "",
                "method": "GET",
                "headers": {},
                "body": ""
            },
            input_ports={
                "input": Port("input", DataType.ANY, "Request body or params")
            },
            output_ports={
                "body": Port("body", DataType.JSON, "Response body"),
                "status": Port("status", DataType.NUMBER, "HTTP status code"),
                "headers": Port("headers", DataType.OBJECT, "Response headers")
            }
        )


class DatabaseNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="database",
            display_name="Database Query",
            description="Query database (SQL)",
            category="ACTIONS",
            settings={
                "db_type": "mysql",
                "host": "localhost",
                "port": "3306",
                "database": "",
                "username": "",
                "password": "",
                "connection_status": "not_connected",
                "operation": "SELECT",
                "table": "",
                "columns": "id,name",
                "filters": {},
                "order_by": "id ASC",
                "limit": "10",
                "block_raw_sql": True,
                "sanitize_inputs": True,
                "only_select": True,
                "query_preview": "",
                "last_test_result": ""
            },
            input_ports={
                "input": Port("input", DataType.ANY, "Query params")
            },
            output_ports={
                "rows": Port("rows", DataType.ARRAY, "Query results"),
                "count": Port("count", DataType.NUMBER, "Number of rows"),
                "error": Port("error", DataType.STRING, "Error if any")
            }
        )
        self.supported_databases = ["mysql", "postgres", "sqlite", "mariadb", "sqlserver"]


# ============================================================================
# TRANSFORMATION NODES
# ============================================================================

class MapTransformNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="map_transform",
            display_name="Map / Transform",
            description="Transform data structure",
            category="DATA",
            settings={
                "mapping": {}
            },
            input_ports={
                "input": Port("input", DataType.ANY, "Input data")
            },
            output_ports={
                "output": Port("output", DataType.ANY, "Transformed data")
            }
        )


class FilterNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="filter",
            display_name="Filter",
            description="Filter array items",
            category="DATA",
            settings={
                "condition": "item.value > 0"
            },
            input_ports={
                "input": Port("input", DataType.ARRAY, "Array to filter")
            },
            output_ports={
                "output": Port("output", DataType.ARRAY, "Filtered array")
            }
        )


class AggregateNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="aggregate",
            display_name="Aggregate",
            description="Aggregate array data",
            category="DATA",
            settings={
                "operation": "sum",
                "field": ""
            },
            input_ports={
                "input": Port("input", DataType.ARRAY, "Array to aggregate")
            },
            output_ports={
                "result": Port("result", DataType.NUMBER, "Aggregation result")
            }
        )


# ============================================================================
# OUTPUT NODES
# ============================================================================

class LogNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="log",
            display_name="Log",
            description="Log output to console",
            category="ACTIONS",
            input_ports={
                "message": Port("message", DataType.ANY, "Message to log"),
                "level": Port("level", DataType.STRING, "Log level (info/warn/error)")
            },
            output_ports={
                "success": Port("success", DataType.BOOLEAN, "Logged successfully")
            }
        )


class NotificationNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="notification",
            display_name="Send Notification",
            description="Send email, Slack, WhatsApp",
            category="ACTIONS",
            settings={
                "channel": "email",
                "recipient": "",
                "subject": ""
            },
            input_ports={
                "message": Port("message", DataType.STRING, "Message content")
            },
            output_ports={
                "success": Port("success", DataType.BOOLEAN, "Sent successfully"),
                "response": Port("response", DataType.JSON, "Service response")
            }
        )


class ExportNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="export",
            display_name="Export File",
            description="Export data to file",
            category="OUTPUT",
            settings={
                "format": "json",
                "filename": "export.json"
            },
            input_ports={
                "data": Port("data", DataType.ANY, "Data to export")
            },
            output_ports={
                "file_path": Port("file_path", DataType.STRING, "Exported file path"),
                "success": Port("success", DataType.BOOLEAN, "Export successful")
            }
        )


# Additional Trigger Nodes
class FileWatcherNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="file_watcher",
            display_name="File Watcher",
            description="Trigger on file changes",
            category="TRIGGERS",
            settings={"watch_path": "", "event_type": "change"},
            output_ports={
                "file_path": Port("file_path", DataType.STRING, "Changed file path"),
                "event": Port("event", DataType.STRING, "Event type")
            }
        )


class DatabaseTrigger(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="database_trigger",
            display_name="Database Trigger",
            description="Trigger on database changes",
            category="TRIGGERS",
            settings={"table": "", "operation": "insert"},
            output_ports={
                "record": Port("record", DataType.JSON, "Changed record"),
                "operation": Port("operation", DataType.STRING, "Operation type")
            }
        )


class APITrigger(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="api_trigger",
            display_name="API Trigger",
            description="Expose workflow as API endpoint",
            category="TRIGGERS",
            settings={"method": "POST", "path": "/api/webhook"},
            output_ports={
                "body": Port("body", DataType.JSON, "Request body"),
                "params": Port("params", DataType.OBJECT, "URL parameters")
            }
        )


# Additional Control Flow Nodes
class SwitchNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="switch",
            display_name="Switch",
            description="Multi-branch condition",
            category="LOGIC",
            settings={"default_branch": "default"},
            input_ports={
                "input": Port("input", DataType.ANY, "Value to switch on")
            },
            output_ports={
                "case1": Port("case1", DataType.ANY, "Case 1 output"),
                "case2": Port("case2", DataType.ANY, "Case 2 output"),
                "default": Port("default", DataType.ANY, "Default output")
            }
        )


class RetryNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="retry",
            display_name="Retry",
            description="Retry failed operations",
            category="LOGIC",
            settings={"max_attempts": 3, "delay_ms": 1000},
            input_ports={
                "input": Port("input", DataType.ANY, "Operation to retry")
            },
            output_ports={
                "output": Port("output", DataType.ANY, "Result"),
                "attempts": Port("attempts", DataType.NUMBER, "Attempts used")
            }
        )


class MergeNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="merge",
            display_name="Merge",
            description="Merge multiple flows",
            category="LOGIC",
            input_ports={
                "input1": Port("input1", DataType.ANY, "Input 1"),
                "input2": Port("input2", DataType.ANY, "Input 2")
            },
            output_ports={
                "output": Port("output", DataType.ARRAY, "Merged output")
            }
        )


class SplitNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="split",
            display_name="Split",
            description="Split one flow into many",
            category="LOGIC",
            input_ports={
                "input": Port("input", DataType.ANY, "Data to split")
            },
            output_ports={
                "output": Port("output", DataType.ARRAY, "Split outputs")
            }
        )


# AI/LLM Extended Nodes
class OpenRouterNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="open_router",
            display_name="OpenRouter LLM",
            description="Call models via OpenRouter API",
            category="AI",
            settings={
                "model": "mistralai/devstral-2512:free",
                "api_key": "",
                "prompt": "You are helpful assistant.",
                "temperature": 0.7,
                "max_tokens": 2000,
                "last_response": "",
                "last_response_status": "pending"
            },
            input_ports={
                "prompt": Port("prompt", DataType.STRING, "User prompt"),
                "context": Port("context", DataType.JSON, "Context/system info")
            },
            output_ports={
                "response": Port("response", DataType.STRING, "Model response"),
                "tokens_used": Port("tokens_used", DataType.NUMBER, "Tokens used")
            }
        )
        
        # Modelos disponibles en OpenRouter
        self.available_models = [
            "mistralai/devstral-2512:free",
            "nex-agi/deepseek-v3.1-nex-n1:free",
            "amazon/nova-2-lite-v1:free",
            "nvidia/nemotron-nano-12b-v2-vl:free"
        ]


class AgentNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="agent",
            display_name="Agent / Tool Caller",
            description="AI agent that can call tools",
            category="AI",
            settings={
                "model": "gpt-4",
                "system_prompt": "You are an AI assistant.",
                "tools": []
            },
            input_ports={
                "goal": Port("goal", DataType.STRING, "Task to accomplish"),
                "tools": Port("tools", DataType.ARRAY, "Available tools")
            },
            output_ports={
                "result": Port("result", DataType.JSON, "Agent result"),
                "reasoning": Port("reasoning", DataType.STRING, "Agent reasoning")
            }
        )


class MemoryNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="memory",
            display_name="Memory / Context",
            description="Store and retrieve conversation memory",
            category="AI",
            settings={"memory_type": "short_term", "max_size": 10},
            input_ports={
                "content": Port("content", DataType.STRING, "Content to remember"),
                "action": Port("action", DataType.STRING, "store/retrieve/clear")
            },
            output_ports={
                "memory": Port("memory", DataType.ARRAY, "Stored memory"),
                "retrieved": Port("retrieved", DataType.JSON, "Retrieved content")
            }
        )


class VectorStoreExtendedNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="vector_store_extended",
            display_name="Vector Database",
            description="Advanced embeddings and vector search",
            category="AI",
            settings={"db": "pinecone", "index": "default"},
            input_ports={
                "texts": Port("texts", DataType.ARRAY, "Texts to embed"),
                "query": Port("query", DataType.STRING, "Search query")
            },
            output_ports={
                "results": Port("results", DataType.ARRAY, "Search results"),
                "scores": Port("scores", DataType.ARRAY, "Relevance scores")
            }
        )


class RAGRetrieverNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="rag_retriever",
            display_name="RAG Retriever",
            description="Retrieve from knowledge base for LLM context",
            category="AI",
            settings={"top_k": 5, "similarity_threshold": 0.7},
            input_ports={
                "query": Port("query", DataType.STRING, "Search query")
            },
            output_ports={
                "documents": Port("documents", DataType.ARRAY, "Retrieved docs"),
                "context": Port("context", DataType.STRING, "Formatted context")
            }
        )


# System/Execution Nodes
class RunScriptExtendedNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="run_script_extended",
            display_name="Execute Script",
            description="Run Python, JS, Bash safely",
            category="SYSTEM",
            settings={
                "language": "python",
                "script": "# Write code here",
                "timeout": 30
            },
            input_ports={
                "input": Port("input", DataType.ANY, "Script input")
            },
            output_ports={
                "output": Port("output", DataType.ANY, "Script output"),
                "error": Port("error", DataType.STRING, "Error if any"),
                "status_code": Port("status_code", DataType.NUMBER, "Exit code")
            }
        )


class CommandExecutor(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="command_executor",
            display_name="Execute Command",
            description="Run system commands (with whitelist)",
            category="SYSTEM",
            settings={"command": "", "timeout": 30, "whitelist": []},
            input_ports={
                "args": Port("args", DataType.ARRAY, "Command arguments")
            },
            output_ports={
                "stdout": Port("stdout", DataType.STRING, "Command output"),
                "stderr": Port("stderr", DataType.STRING, "Error output")
            }
        )


class FileSystemNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="file_system",
            display_name="File System",
            description="Read/write/manage files",
            category="SYSTEM",
            settings={"operation": "read", "path": ""},
            input_ports={
                "path": Port("path", DataType.STRING, "File path"),
                "content": Port("content", DataType.STRING, "File content for write")
            },
            output_ports={
                "content": Port("content", DataType.STRING, "File content"),
                "success": Port("success", DataType.BOOLEAN, "Operation success")
            }
        )


class EnvironmentVariables(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="env_vars",
            display_name="Environment Variables",
            description="Read/set environment variables",
            category="SYSTEM",
            settings={"operation": "get", "var_name": ""},
            input_ports={
                "var_name": Port("var_name", DataType.STRING, "Variable name"),
                "value": Port("value", DataType.STRING, "Variable value to set")
            },
            output_ports={
                "value": Port("value", DataType.STRING, "Variable value")
            }
        )


# Database Nodes
class PostgreSQLNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="postgresql",
            display_name="PostgreSQL",
            description="Query PostgreSQL database",
            category="DATABASE",
            settings={
                "host": "localhost",
                "port": 5432,
                "database": "",
                "query": "SELECT * FROM table"
            },
            input_ports={
                "params": Port("params", DataType.OBJECT, "Query parameters")
            },
            output_ports={
                "rows": Port("rows", DataType.ARRAY, "Query results"),
                "count": Port("count", DataType.NUMBER, "Row count")
            }
        )


class MySQLNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="mysql",
            display_name="MySQL",
            description="Query MySQL database",
            category="DATABASE",
            settings={
                "host": "localhost",
                "port": 3306,
                "database": "",
                "query": "SELECT * FROM table"
            },
            input_ports={
                "params": Port("params", DataType.OBJECT, "Query parameters")
            },
            output_ports={
                "rows": Port("rows", DataType.ARRAY, "Query results"),
                "count": Port("count", DataType.NUMBER, "Row count")
            }
        )


class MongoDBNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="mongodb",
            display_name="MongoDB",
            description="Query MongoDB database",
            category="DATABASE",
            settings={
                "connection_string": "",
                "database": "",
                "collection": "",
                "operation": "find"
            },
            input_ports={
                "query": Port("query", DataType.JSON, "MongoDB query"),
                "document": Port("document", DataType.JSON, "Document to insert")
            },
            output_ports={
                "results": Port("results", DataType.ARRAY, "Query results"),
                "inserted_id": Port("inserted_id", DataType.STRING, "Inserted ID")
            }
        )


class RedisNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="redis",
            display_name="Redis",
            description="Redis cache and data store",
            category="DATABASE",
            settings={"host": "localhost", "port": 6379, "operation": "get"},
            input_ports={
                "key": Port("key", DataType.STRING, "Redis key"),
                "value": Port("value", DataType.ANY, "Value to store")
            },
            output_ports={
                "value": Port("value", DataType.ANY, "Retrieved value"),
                "success": Port("success", DataType.BOOLEAN, "Operation success")
            }
        )


class SQLiteNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="sqlite",
            display_name="SQLite",
            description="Query SQLite database",
            category="DATABASE",
            settings={
                "db_path": "data.db",
                "query": "SELECT * FROM table"
            },
            input_ports={
                "params": Port("params", DataType.OBJECT, "Query parameters")
            },
            output_ports={
                "rows": Port("rows", DataType.ARRAY, "Query results"),
                "count": Port("count", DataType.NUMBER, "Row count")
            }
        )


# REST/Web Nodes
class GraphQLNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="graphql",
            display_name="GraphQL Query",
            description="Execute GraphQL queries",
            category="WEB",
            settings={
                "endpoint": "",
                "query": "query { }"
            },
            input_ports={
                "variables": Port("variables", DataType.JSON, "Query variables")
            },
            output_ports={
                "data": Port("data", DataType.JSON, "Response data"),
                "errors": Port("errors", DataType.ARRAY, "Errors if any")
            }
        )


class WebhookResponseNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="webhook_response",
            display_name="Webhook Response",
            description="Send response to webhook caller",
            category="WEB",
            settings={"status_code": 200},
            input_ports={
                "body": Port("body", DataType.JSON, "Response body")
            },
            output_ports={
                "success": Port("success", DataType.BOOLEAN, "Response sent")
            }
        )


# Data Transformation Nodes
class SetMapNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="set_map",
            display_name="Set / Map",
            description="Set values and transform data",
            category="DATA",
            settings={"mapping": {}},
            input_ports={
                "data": Port("data", DataType.JSON, "Input data")
            },
            output_ports={
                "result": Port("result", DataType.JSON, "Mapped data")
            }
        )


class JoinNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="join",
            display_name="Join",
            description="Join data from multiple sources",
            category="DATA",
            settings={"join_type": "inner"},
            input_ports={
                "left": Port("left", DataType.ARRAY, "Left array"),
                "right": Port("right", DataType.ARRAY, "Right array"),
                "key": Port("key", DataType.STRING, "Join key")
            },
            output_ports={
                "result": Port("result", DataType.ARRAY, "Joined result")
            }
        )


class ValidateSchemaNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="validate_schema",
            display_name="Validate Schema",
            description="Validate data against schema",
            category="DATA",
            settings={"schema": {}},
            input_ports={
                "data": Port("data", DataType.JSON, "Data to validate"),
                "schema": Port("schema", DataType.JSON, "Validation schema")
            },
            output_ports={
                "valid": Port("valid", DataType.BOOLEAN, "Is valid"),
                "errors": Port("errors", DataType.ARRAY, "Validation errors")
            }
        )


class NormalizeNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="normalize",
            display_name="Normalize",
            description="Normalize data format",
            category="DATA",
            settings={"format": "json"},
            input_ports={
                "data": Port("data", DataType.ANY, "Data to normalize")
            },
            output_ports={
                "result": Port("result", DataType.JSON, "Normalized data")
            }
        )


class FormatNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="format",
            display_name="Format",
            description="Format data as JSON/CSV/Text",
            category="DATA",
            settings={"format": "json", "template": ""},
            input_ports={
                "data": Port("data", DataType.ANY, "Data to format")
            },
            output_ports={
                "output": Port("output", DataType.STRING, "Formatted output")
            }
        )


# Output/Notification Nodes
class EmailNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="email",
            display_name="Send Email",
            description="Send emails",
            category="OUTPUT",
            settings={
                "from": "",
                "to": "",
                "subject": "Message",
                "smtp_host": "smtp.gmail.com"
            },
            input_ports={
                "body": Port("body", DataType.STRING, "Email body"),
                "html": Port("html", DataType.STRING, "HTML content")
            },
            output_ports={
                "success": Port("success", DataType.BOOLEAN, "Email sent")
            }
        )


class SlackNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="slack",
            display_name="Send Slack Message",
            description="Send message to Slack",
            category="OUTPUT",
            settings={
                "webhook_url": "",
                "channel": "#general"
            },
            input_ports={
                "message": Port("message", DataType.STRING, "Message text")
            },
            output_ports={
                "success": Port("success", DataType.BOOLEAN, "Message sent")
            }
        )


class WhatsAppNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="whatsapp",
            display_name="Send WhatsApp",
            description="Send WhatsApp message",
            category="OUTPUT",
            settings={
                "api_key": "",
                "from": "",
                "to": ""
            },
            input_ports={
                "message": Port("message", DataType.STRING, "Message text")
            },
            output_ports={
                "success": Port("success", DataType.BOOLEAN, "Message sent"),
                "message_id": Port("message_id", DataType.STRING, "Message ID")
            }
        )


class DashboardOutputNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="dashboard_output",
            display_name="Dashboard Output",
            description="Display on dashboard/UI",
            category="OUTPUT",
            settings={"widget_type": "text", "widget_id": ""},
            input_ports={
                "data": Port("data", DataType.ANY, "Data to display")
            },
            output_ports={
                "rendered": Port("rendered", DataType.BOOLEAN, "Rendered successfully")
            }
        )


class ConsoleLogNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="console_log",
            display_name="Console Log",
            description="Log to console/terminal",
            category="OUTPUT",
            settings={"level": "info"},
            input_ports={
                "message": Port("message", DataType.ANY, "Message to log")
            },
            output_ports={
                "logged": Port("logged", DataType.BOOLEAN, "Logged successfully")
            }
        )


# Training/Model Nodes
class TrainModelNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="train_model",
            display_name="Train Model",
            description="Train ML model",
            category="AI",
            settings={
                "model_type": "neural_net",
                "epochs": 10,
                "batch_size": 32
            },
            input_ports={
                "train_data": Port("train_data", DataType.DATASET, "Training data"),
                "hyperparams": Port("hyperparams", DataType.JSON, "Hyperparameters")
            },
            output_ports={
                "model": Port("model", DataType.MODEL, "Trained model"),
                "metrics": Port("metrics", DataType.METRICS, "Training metrics")
            }
        )


class PredictNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="predict",
            display_name="Predict",
            description="Make predictions with model",
            category="AI",
            input_ports={
                "model": Port("model", DataType.MODEL, "Trained model"),
                "input": Port("input", DataType.JSON, "Input data")
            },
            output_ports={
                "prediction": Port("prediction", DataType.ANY, "Prediction result"),
                "confidence": Port("confidence", DataType.NUMBER, "Confidence score")
            }
        )


class EvaluateModelNode(NodeConfig):
    def __init__(self):
        super().__init__(
            node_type="evaluate_model",
            display_name="Evaluate Model",
            description="Evaluate model performance",
            category="AI",
            input_ports={
                "model": Port("model", DataType.MODEL, "Model to evaluate"),
                "test_data": Port("test_data", DataType.DATASET, "Test data")
            },
            output_ports={
                "metrics": Port("metrics", DataType.JSON, "Evaluation metrics"),
                "accuracy": Port("accuracy", DataType.NUMBER, "Accuracy score")
            }
        )


# ============================================================================
# NODE FACTORY
# ============================================================================

NODE_REGISTRY = {
    # Triggers
    "manual_trigger": ManualTrigger,
    "cron_trigger": CronTrigger,
    "webhook_trigger": WebhookTrigger,
    "file_watcher": FileWatcherNode,
    "database_trigger": DatabaseTrigger,
    "api_trigger": APITrigger,
    # Control Flow
    "if_else": IfElseNode,
    "switch": SwitchNode,
    "loop": LoopNode,
    "retry": RetryNode,
    "delay": DelayNode,
    "merge": MergeNode,
    "split": SplitNode,
    # AI/LLM
    "llm": LLMNode,
    "open_router": OpenRouterNode,
    "agent": AgentNode,
    "memory": MemoryNode,
    "vector_store": VectorStoreNode,
    "vector_store_extended": VectorStoreExtendedNode,
    "rag_retriever": RAGRetrieverNode,
    "dataset": DatasetNode,
    "train_model": TrainModelNode,
    "predict": PredictNode,
    "evaluate_model": EvaluateModelNode,
    # System
    "run_script": RunScriptNode,
    "run_script_extended": RunScriptExtendedNode,
    "command_executor": CommandExecutor,
    "file_system": FileSystemNode,
    "env_vars": EnvironmentVariables,
    # Database
    "database": DatabaseNode,
    "postgresql": PostgreSQLNode,
    "mysql": MySQLNode,
    "mongodb": MongoDBNode,
    "redis": RedisNode,
    "sqlite": SQLiteNode,
    # Web/API
    "http_request": HttpRequestNode,
    "graphql": GraphQLNode,
    "webhook_response": WebhookResponseNode,
    # Data Transformation
    "map_transform": MapTransformNode,
    "set_map": SetMapNode,
    "filter": FilterNode,
    "aggregate": AggregateNode,
    "join": JoinNode,
    "validate_schema": ValidateSchemaNode,
    "normalize": NormalizeNode,
    "format": FormatNode,
    # Output/Notifications
    "log": LogNode,
    "console_log": ConsoleLogNode,
    "notification": NotificationNode,
    "email": EmailNode,
    "slack": SlackNode,
    "whatsapp": WhatsAppNode,
    "dashboard_output": DashboardOutputNode,
    "export": ExportNode,
}


def create_node_config(node_type: str) -> Optional[NodeConfig]:
    """Create a node config from registry."""
    node_class = NODE_REGISTRY.get(node_type)
    if node_class:
        return node_class()
    return None


def get_all_nodes_by_category() -> Dict[str, List[NodeConfig]]:
    """Get all nodes organized by category."""
    categories = {}
    for node_type, node_class in NODE_REGISTRY.items():
        config = node_class()
        cat = config.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(config)
    return categories
