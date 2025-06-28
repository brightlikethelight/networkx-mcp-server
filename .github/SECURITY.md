# Security Policy

## 🔒 Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## 🛡️ Security Features

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

## 🚨 Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security issues responsibly:

### 📧 Email
Send security reports to: **security@networkx-mcp.org**

### 🔐 Encrypted Communication
For sensitive reports, use our PGP key:
```
Key ID: [TO BE ADDED]
Fingerprint: [TO BE ADDED]
```

### 📋 What to Include

Please include the following information in your report:

1. **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting)
2. **Full paths** of source file(s) related to the issue
3. **Location** of the affected source code (tag/branch/commit or direct URL)
4. **Special configuration** required to reproduce the issue
5. **Step-by-step instructions** to reproduce the issue
6. **Proof-of-concept or exploit code** (if possible)
7. **Impact** of the issue, including how an attacker might exploit it

### ⏱️ Response Timeline

- **Initial Response**: Within 48 hours
- **Investigation**: Within 7 days  
- **Fix Development**: Depends on severity
- **Public Disclosure**: After fix is available

## 🔍 Security Assessment Process

When you report a vulnerability:

1. **Acknowledgment**: We'll acknowledge receipt within 48 hours
2. **Initial Assessment**: We'll assess severity and impact
3. **Investigation**: Our team will investigate the issue
4. **Fix Development**: We'll develop and test a fix
5. **Coordinated Disclosure**: We'll work with you on disclosure timing
6. **Public Disclosure**: After fixes are deployed

## 🏆 Security Recognition

We believe in recognizing security researchers who help keep our users safe:

### Hall of Fame
Contributors who responsibly disclose security issues will be:
- Listed in our security hall of fame (with permission)
- Credited in release notes
- Thanked publicly (if desired)

### Bug Bounty
While we don't currently offer monetary rewards, we're considering a bug bounty program for the future.

## 🛠️ Security Best Practices for Users

### For Developers
- Always validate input from untrusted sources
- Use the latest version of NetworkX MCP Server
- Enable Redis authentication if using persistence
- Monitor server logs for suspicious activity
- Run the server with minimal required privileges

### For Deployment
- Use Docker containers for isolation
- Set up proper network security (firewalls, VPNs)
- Enable logging and monitoring
- Regular security updates
- Backup and recovery procedures

### For Data
- Don't process sensitive data without encryption
- Be careful with graph data containing personal information
- Use secure channels for data transmission
- Implement access controls for multi-user environments

## 📚 Security Resources

### Documentation
- [Security Best Practices](docs/security-best-practices.md)
- [Deployment Security Guide](docs/deployment-security.md)
- [API Security Reference](docs/api-security.md)

### Tools
- Security scanning is included in our CI/CD pipeline
- Use `bandit` for additional security analysis
- Run `safety` to check for vulnerable dependencies

### Community
- Join our [Security Discussions](https://github.com/brightliu/networkx-mcp-server/discussions/categories/security)
- Follow [@NetworkXMCP](https://twitter.com/NetworkXMCP) for security announcements

## 📝 Security Changelog

### Version 1.0.0
- Initial security hardening implementation
- Input validation and sanitization
- Memory limits and resource protection
- File access restrictions
- Disabled dangerous operations

---

## 🤝 Thank You

We appreciate the security research community's efforts to responsibly disclose vulnerabilities. Your work helps keep NetworkX MCP Server secure for everyone.

**Contact**: security@networkx-mcp.org  
**PGP Key**: [TO BE ADDED]

---

*This security policy is based on security best practices and will be updated as needed.*