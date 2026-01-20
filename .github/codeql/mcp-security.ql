/**
 * @name MCP Protocol Security Analysis
 * @description Identifies potential security vulnerabilities specific to MCP protocol implementations
 * @kind problem
 * @problem.severity warning
 * @security-severity 7.0
 * @id python/mcp-security
 * @tags security mcp protocol
 */

import python
import semmle.python.security.dataflow.CodeInjectionQuery
import semmle.python.security.dataflow.CommandInjectionQuery

// Detect unsafe JSON deserialization in MCP handlers
class McpJsonDeserialization extends TaintTracking::Configuration {
  McpJsonDeserialization() { this = "McpJsonDeserialization" }

  override predicate isSource(DataFlow::Node source) {
    exists(Call call |
      call.getFunc().(Name).getId() = "json.loads" or
      call.getFunc().(Attribute).getName() = "loads"
    |
      source.asExpr() = call.getArg(0)
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(Call call |
      call.getFunc().(Name).getId() in ["eval", "exec", "compile"] or
      call.getFunc().(Attribute).getName() in ["execute", "run"]
    |
      sink.asExpr() = call.getArg(0)
    )
  }
}

// Detect potential prompt injection in MCP tool arguments
class McpPromptInjection extends TaintTracking::Configuration {
  McpPromptInjection() { this = "McpPromptInjection" }

  override predicate isSource(DataFlow::Node source) {
    exists(FunctionDef func, Parameter param |
      func.getName().matches("%tool%") and
      param = func.getArg(_) and
      param.getName() in ["args", "arguments", "params", "parameters"] and
      source.asExpr() = param
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(Call call |
      call.getFunc().(Name).getId() in ["subprocess.run", "os.system", "eval", "exec"]
    |
      sink.asExpr() = call.getArg(0)
    )
  }
}

// Detect unsafe file path construction in MCP tools
class McpPathTraversal extends TaintTracking::Configuration {
  McpPathTraversal() { this = "McpPathTraversal" }

  override predicate isSource(DataFlow::Node source) {
    exists(Subscript subscript |
      subscript.getObject().(Name).getId() in ["args", "arguments", "params"] and
      subscript.getIndex().(StrConst).getText() in ["path", "file", "filename", "directory"]
    |
      source.asExpr() = subscript
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(Call call |
      call.getFunc().(Name).getId() in ["open", "os.path.join"] or
      call.getFunc().(Attribute).getName() in ["read", "write", "mkdir", "rmdir"]
    |
      sink.asExpr() = call.getArg(0)
    )
  }
}

// Detect missing input validation in MCP handlers
predicate lacksInputValidation(FunctionDef func) {
  func.getName().matches("%_tool") and
  not exists(Call call |
    call.getFunc().(Name).getId() in ["isinstance", "validate", "check"] or
    call.getFunc().(Attribute).getName() in ["validate", "check", "verify"]
  |
    call.getScope() = func
  )
}

// Detect hardcoded secrets in MCP configuration
predicate containsHardcodedSecret(StrConst str) {
  exists(string value | value = str.getText() |
    value.regexpMatch("(?i).*(password|secret|key|token|api[_-]?key).*=.*['\"][^'\"]{8,}['\"].*") or
    value.regexpMatch("(?i).*['\"][a-zA-Z0-9]{20,}['\"].*") // Potential API keys
  )
}

// Main query predicates
from DataFlow::PathNode source, DataFlow::PathNode sink, string message
where
  (
    exists(McpJsonDeserialization config |
      config.hasFlowPath(source, sink) and
      message = "Potential unsafe JSON deserialization in MCP handler"
    )
    or
    exists(McpPromptInjection config |
      config.hasFlowPath(source, sink) and
      message = "Potential prompt injection vulnerability in MCP tool"
    )
    or
    exists(McpPathTraversal config |
      config.hasFlowPath(source, sink) and
      message = "Potential path traversal vulnerability in MCP tool"
    )
  )
select sink.getNode(), source, sink, message

// Additional queries for specific MCP security issues
from FunctionDef func
where lacksInputValidation(func)
select func, "MCP tool handler lacks proper input validation"

from StrConst str
where containsHardcodedSecret(str)
select str, "Potential hardcoded secret detected"
