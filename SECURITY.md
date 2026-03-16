# Security Policy

## Overview

This document describes the actual security posture of NetworkX MCP Server.
It distinguishes between what is implemented and what is not.

## Implemented

### Input Validation

Graph IDs are validated by `validate_graph_id()` in `src/networkx_mcp/errors.py`:

- Pattern: alphanumeric, underscore, and hyphen only
- Max length: 100 characters
- Path traversal blocked (`..`, `/`, `\` rejected)
- Node IDs and edge specs are also validated

Error messages are sanitized -- malicious input is not echoed back in responses,
and stack traces are not exposed to callers.

### API Key Authentication

Opt-in authentication system, disabled by default for MCP stdio compatibility.

- Enable: `export NETWORKX_MCP_AUTH=true`
- Generate keys: `python -m networkx_mcp.auth generate <name>`
- Keys support read/write permission scoping
- Production mode requires auth to be enabled, or explicit opt-out
  via `export NETWORKX_MCP_INSECURE_CONFIRM=true`

### Code Safety

- No `eval()` or `exec()` usage
- No `pickle` deserialization of untrusted data
- No dynamic code generation from user input

### CI/CD Input Sanitization

- Key name validation in CI workflows
- Flag injection prevention
- Maximum value length enforcement

## NOT Implemented

These are known gaps, listed here so users can make informed decisions.

### Resource Limits

There are no limits on graph size (nodes, edges), operation duration,
or memory consumption. A single request can create an arbitrarily large graph
or trigger an expensive algorithm. This is tracked as future work.

### Rate Limiting

No rate limiting on incoming requests. Every request is processed immediately.

### Network Security / TLS

The server uses MCP stdio transport (stdin/stdout). There is no HTTP listener,
so TLS is not applicable in the default configuration. If you expose the server
over a network, you are responsible for transport security.

### Audit Logging

Basic Python logging only. There is no structured security audit trail,
no tamper-evident log storage, and no alerting on suspicious activity.

## Transport Model

NetworkX MCP Server communicates over stdio, not HTTP. This means:

- No network ports are opened by default
- No cookies, CORS, or HTTP headers to configure
- Attack surface is limited to the MCP client sending requests over stdin
- Authentication, when enabled, validates API keys in the MCP request metadata

## Data Persistence

All graphs are in-memory only. Data is lost on server restart.
There is no persistent storage, no database, and no filesystem writes
beyond standard log output.

## Vulnerability Reporting

**Do not** open public GitHub issues for security vulnerabilities.

Email: brightliu@college.harvard.edu

Include:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

Response target: acknowledgment within 7 days. Fix timeline depends on severity.

## Security Summary

| Category | Status | Detail |
|---|---|---|
| Input Validation | Implemented | Graph/node/edge ID validation, path traversal prevention |
| Authentication | Implemented (opt-in) | API key auth, disabled by default |
| Code Injection | Mitigated | No eval/exec/pickle, safe error messages |
| Resource Limits | Not implemented | No max nodes, edges, timeout, or memory cap |
| Rate Limiting | Not implemented | All requests processed without throttling |
| Network Security | N/A | stdio transport, no HTTP listener |
| Audit Logging | Minimal | Basic Python logging only |
| Data Encryption | N/A | In-memory only, no persistent storage |

## Compliance

This server has not been penetration tested, formally audited, or certified
for any compliance standard (SOC 2, HIPAA, PCI-DSS, GDPR, etc.).

---

Last updated: 2026-03-15
Contact: brightliu@college.harvard.edu
