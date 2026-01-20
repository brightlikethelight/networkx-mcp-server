# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported |
| ------- | --------- |
| 3.0.x   | Yes       |
| 2.0.x   | Yes       |
| 1.0.x   | Limited   |
| < 1.0   | No        |

## Security Features

NetworkX MCP Server includes several built-in security features:

### Input Validation

- All graph IDs validated against safe patterns
- File paths restricted to safe directories
- Format whitelisting for imports
- Parameter validation for all MCP tools

### Memory Protection

- 1GB memory limit to prevent DoS attacks
- Automatic cleanup of large operations
- Resource monitoring and limits

### File Security

- Sandboxed file operations
- No directory traversal allowed
- Disabled dangerous formats (pickle)
- Safe temporary file handling

### Network Security

- Rate limiting support
- Secure error messages (no stack traces)
- No code execution (eval/exec disabled)

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

### Contact

Send security reports to: **<brightliu@college.harvard.edu>**

Please include the following information:

1. **Type of issue** (e.g., buffer overflow, injection, cross-site scripting)
2. **Full paths** of source file(s) related to the issue
3. **Location** of the affected source code (tag/branch/commit or direct URL)
4. **Special configuration** required to reproduce the issue
5. **Step-by-step instructions** to reproduce the issue
6. **Proof-of-concept or exploit code** (if possible)
7. **Impact** of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Investigation**: Within 7 days
- **Fix Development**: Depends on severity
- **Public Disclosure**: After fix is available

## Security Assessment Process

When you report a vulnerability:

1. **Acknowledgment**: We'll acknowledge receipt within 48 hours
2. **Initial Assessment**: We'll assess severity and impact
3. **Investigation**: Our team will investigate the issue
4. **Fix Development**: We'll develop and test a fix
5. **Coordinated Disclosure**: We'll work with you on disclosure timing
6. **Public Disclosure**: After fixes are deployed

## Security Recognition

Contributors who responsibly disclose security issues will be:

- Listed in our security acknowledgments (with permission)
- Credited in release notes
- Thanked publicly (if desired)

## Security Best Practices for Users

### For Developers

- Always validate input from untrusted sources
- Use the latest version of NetworkX MCP Server
- Monitor server logs for suspicious activity
- Run the server with minimal required privileges

### For Deployment

- Use Docker containers for isolation
- Set up proper network security (firewalls, VPNs)
- Enable logging and monitoring
- Apply regular security updates
- Implement backup and recovery procedures

### For Data

- Don't process sensitive data without encryption
- Be careful with graph data containing personal information
- Use secure channels for data transmission
- Implement access controls for multi-user environments

## Security Tools

Security scanning is included in our CI/CD pipeline:

- Use `bandit` for Python security analysis
- Run `safety` to check for vulnerable dependencies
- CodeQL analysis for code security issues

## Security Changelog

### Version 3.0.0

- Added protocol-based dependency injection for better isolation
- Enhanced exception handling with specific error types
- Custom CodeQL security queries for MCP-specific vulnerabilities

### Version 2.0.0

- Added rate limiting and input validation middleware
- Security audit logging improvements
- Enhanced file security sandboxing

### Version 1.0.0

- Initial security hardening implementation
- Input validation and sanitization
- Memory limits and resource protection
- File access restrictions
- Disabled dangerous operations

---

**Contact**: <brightliu@college.harvard.edu>

*This security policy is based on security best practices and will be updated as needed.*
