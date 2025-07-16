"""
Zero-Trust Input/Output Validation System

Comprehensive validation system that treats all inputs as potentially malicious
and applies strict validation, sanitization, and data loss prevention measures.

Key Features:
- Schema-based input validation
- Content sanitization and filtering
- Data loss prevention (DLP)
- Output sanitization
- Encoding attack prevention
"""

import re
import json
import html
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging


class ValidationLevel(Enum):
    """Validation severity levels."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    status: ValidationStatus
    sanitized_input: Optional[Dict[str, Any]] = None
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    processing_time: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "sanitized_input": self.sanitized_input,
            "violations": self.violations,
            "warnings": self.warnings,
            "risk_score": self.risk_score,
            "processing_time": self.processing_time,
            "validation_level": self.validation_level.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ZeroTrustValidator:
    """
    Zero-trust input/output validation system.
    
    Implements multiple validation layers:
    1. Schema validation - Structure and type checking
    2. Content validation - Malicious content detection
    3. Encoding validation - Encoding attack prevention
    4. Size validation - Resource exhaustion prevention
    5. Output sanitization - Data loss prevention
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.logger = logging.getLogger(__name__)
        self.validation_level = validation_level
        self.content_sanitizer = ContentSanitizer()
        self.dlp_engine = DataLossPreventionEngine()
        self.input_schemas = self._load_input_schemas()
        
        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "passed": 0,
            "failed": 0,
            "sanitized": 0,
            "blocked": 0,
            "last_reset": datetime.utcnow()
        }
    
    def _load_input_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for tool input validation."""
        return {
            "create_graph": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{1,50}$",
                        "description": "Graph name (alphanumeric, underscore, dash only)"
                    },
                    "directed": {
                        "type": "boolean",
                        "default": False
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            },
            "add_nodes": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{1,50}$"
                    },
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[a-zA-Z0-9_-]{1,100}$"
                        },
                        "maxItems": 10000,
                        "minItems": 1
                    }
                },
                "required": ["graph", "nodes"],
                "additionalProperties": False
            },
            "add_edges": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{1,50}$"
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": "^[a-zA-Z0-9_-]{1,100}$"
                            },
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "maxItems": 10000,
                        "minItems": 1
                    }
                },
                "required": ["graph", "edges"],
                "additionalProperties": False
            },
            "visualize_graph": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{1,50}$"
                    },
                    "layout": {
                        "type": "string",
                        "enum": ["spring", "circular", "random", "shell", "spectral"]
                    },
                    "node_color": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9#]{1,20}$"
                    },
                    "node_size": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 1000
                    }
                },
                "required": ["graph"],
                "additionalProperties": False
            },
            "import_csv": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{1,50}$"
                    },
                    "csv_data": {
                        "type": "string",
                        "maxLength": 10485760  # 10MB limit
                    }
                },
                "required": ["graph", "csv_data"],
                "additionalProperties": False
            }
        }
    
    def validate_input(self, tool_name: str, args: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive input validation.
        
        Args:
            tool_name: Name of the tool being called
            args: Tool arguments to validate
            
        Returns:
            ValidationResult: Detailed validation result
        """
        start_time = datetime.utcnow()
        self.validation_stats["total_validations"] += 1
        
        violations = []
        warnings = []
        sanitized_input = None
        risk_score = 0.0
        
        try:
            # 1. Schema validation
            schema_result = self._validate_schema(tool_name, args)
            if schema_result["violations"]:
                violations.extend(schema_result["violations"])
                risk_score += 30
            
            # 2. Content validation
            content_result = self._validate_content(args)
            if content_result["violations"]:
                violations.extend(content_result["violations"])
                risk_score += 40
            
            # 3. Size validation
            size_result = self._validate_size(args)
            if size_result["violations"]:
                violations.extend(size_result["violations"])
                risk_score += 20
            
            # 4. Encoding validation
            encoding_result = self._validate_encoding(args)
            if encoding_result["violations"]:
                violations.extend(encoding_result["violations"])
                risk_score += 25
            
            # 5. Content sanitization
            sanitized_input = self.content_sanitizer.sanitize_input(args)
            if sanitized_input != args:
                warnings.append("Input was sanitized to remove potentially harmful content")
                risk_score += 10
            
            # Determine validation status
            if risk_score >= 70:
                status = ValidationStatus.BLOCKED
                self.validation_stats["blocked"] += 1
            elif violations:
                status = ValidationStatus.FAILED
                self.validation_stats["failed"] += 1
            elif warnings:
                status = ValidationStatus.SANITIZED
                self.validation_stats["sanitized"] += 1
            else:
                status = ValidationStatus.PASSED
                self.validation_stats["passed"] += 1
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ValidationResult(
                status=status,
                sanitized_input=sanitized_input,
                violations=violations,
                warnings=warnings,
                risk_score=risk_score,
                processing_time=processing_time,
                validation_level=self.validation_level,
                metadata={
                    "tool_name": tool_name,
                    "original_args": args,
                    "schema_result": schema_result,
                    "content_result": content_result,
                    "size_result": size_result,
                    "encoding_result": encoding_result
                }
            )
            
            self.logger.info(f"Input validation complete: {status.value} "
                           f"(risk: {risk_score:.1f}, time: {processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                violations=[f"Validation error: {str(e)}"],
                risk_score=100.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _validate_schema(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input against JSON schema."""
        violations = []
        
        if tool_name not in self.input_schemas:
            violations.append(f"Unknown tool: {tool_name}")
            return {"violations": violations}
        
        schema = self.input_schemas[tool_name]
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in args:
                violations.append(f"Missing required field: {field}")
        
        # Check field types and patterns
        properties = schema.get("properties", {})
        for field, value in args.items():
            if field not in properties:
                if not schema.get("additionalProperties", True):
                    violations.append(f"Unknown field: {field}")
                continue
            
            field_schema = properties[field]
            field_violations = self._validate_field(field, value, field_schema)
            violations.extend(field_violations)
        
        return {"violations": violations}
    
    def _validate_field(self, field_name: str, value: Any, field_schema: Dict[str, Any]) -> List[str]:
        """Validate individual field against schema."""
        violations = []
        
        # Type validation
        expected_type = field_schema.get("type")
        if expected_type == "string" and not isinstance(value, str):
            violations.append(f"Field {field_name} must be a string")
        elif expected_type == "integer" and not isinstance(value, int):
            violations.append(f"Field {field_name} must be an integer")
        elif expected_type == "boolean" and not isinstance(value, bool):
            violations.append(f"Field {field_name} must be a boolean")
        elif expected_type == "array" and not isinstance(value, list):
            violations.append(f"Field {field_name} must be an array")
        
        # Pattern validation for strings
        if isinstance(value, str) and "pattern" in field_schema:
            pattern = field_schema["pattern"]
            if not re.match(pattern, value):
                violations.append(f"Field {field_name} does not match required pattern")
        
        # Enum validation
        if "enum" in field_schema:
            if value not in field_schema["enum"]:
                violations.append(f"Field {field_name} must be one of: {field_schema['enum']}")
        
        # Length validation
        if isinstance(value, str):
            if "maxLength" in field_schema and len(value) > field_schema["maxLength"]:
                violations.append(f"Field {field_name} exceeds maximum length")
            if "minLength" in field_schema and len(value) < field_schema["minLength"]:
                violations.append(f"Field {field_name} below minimum length")
        
        # Numeric range validation
        if isinstance(value, (int, float)):
            if "minimum" in field_schema and value < field_schema["minimum"]:
                violations.append(f"Field {field_name} below minimum value")
            if "maximum" in field_schema and value > field_schema["maximum"]:
                violations.append(f"Field {field_name} above maximum value")
        
        # Array validation
        if isinstance(value, list):
            if "maxItems" in field_schema and len(value) > field_schema["maxItems"]:
                violations.append(f"Field {field_name} has too many items")
            if "minItems" in field_schema and len(value) < field_schema["minItems"]:
                violations.append(f"Field {field_name} has too few items")
            
            # Validate array items
            if "items" in field_schema:
                items_schema = field_schema["items"]
                for i, item in enumerate(value):
                    item_violations = self._validate_field(f"{field_name}[{i}]", item, items_schema)
                    violations.extend(item_violations)
        
        return violations
    
    def _validate_content(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content for malicious patterns."""
        violations = []
        
        # Check for suspicious patterns in all string values
        def check_string_content(value: str, path: str = "") -> None:
            # Command injection patterns
            cmd_patterns = [
                r";\s*(?:rm|del|delete|drop|truncate)",
                r"(?:&&|\|\|)\s*(?:rm|del|delete)",
                r"(?:exec|eval|system|subprocess|shell)",
                r"`.*`",  # Command substitution
                r"\$\(.*\)",  # Command substitution
                r"(?:\/etc\/|\/var\/|\/usr\/|\/root\/)",  # Path traversal
                r"(?:\.\.\/|\.\.\\)",  # Directory traversal
            ]
            
            for pattern in cmd_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    violations.append(f"Suspicious pattern detected in {path}: {pattern}")
            
            # Script injection patterns
            script_patterns = [
                r"<script.*?>.*?</script>",
                r"javascript:",
                r"(?:onerror|onload|onclick)\s*=",
                r"(?:eval|setTimeout|setInterval)\s*\(",
            ]
            
            for pattern in script_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    violations.append(f"Script injection pattern detected in {path}: {pattern}")
        
        # Recursively check all string values
        def traverse_args(obj: Any, path: str = "") -> None:
            if isinstance(obj, str):
                check_string_content(obj, path)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    traverse_args(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    traverse_args(item, f"{path}[{i}]")
        
        traverse_args(args)
        
        return {"violations": violations}
    
    def _validate_size(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input size limits."""
        violations = []
        
        # Calculate total size
        total_size = len(json.dumps(args))
        if total_size > 10 * 1024 * 1024:  # 10MB limit
            violations.append(f"Input size ({total_size} bytes) exceeds 10MB limit")
        
        # Check specific field sizes
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 1024 * 1024:  # 1MB per string
                violations.append(f"Field {key} size exceeds 1MB limit")
            elif isinstance(value, list) and len(value) > 100000:  # 100k items per array
                violations.append(f"Field {key} has too many items (>{100000})")
        
        return {"violations": violations}
    
    def _validate_encoding(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate encoding and prevent encoding attacks."""
        violations = []
        
        def check_encoding(value: str, path: str = "") -> None:
            # Check for multiple encodings
            try:
                # URL encoding
                if "%" in value and re.search(r"%[0-9a-fA-F]{2}", value):
                    violations.append(f"URL encoding detected in {path}")
                
                # HTML encoding
                if "&" in value and re.search(r"&#?\w+;", value):
                    violations.append(f"HTML encoding detected in {path}")
                
                # Unicode encoding
                if "\\u" in value and re.search(r"\\u[0-9a-fA-F]{4}", value):
                    violations.append(f"Unicode encoding detected in {path}")
                
                # Base64 encoding (suspicious patterns)
                if len(value) > 20 and re.match(r"^[A-Za-z0-9+/]*={0,2}$", value):
                    violations.append(f"Possible Base64 encoding detected in {path}")
                
            except Exception as e:
                violations.append(f"Encoding validation error in {path}: {e}")
        
        # Check all string values for encoding
        def traverse_for_encoding(obj: Any, path: str = "") -> None:
            if isinstance(obj, str):
                check_encoding(obj, path)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    traverse_for_encoding(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    traverse_for_encoding(item, f"{path}[{i}]")
        
        traverse_for_encoding(args)
        
        return {"violations": violations}
    
    def validate_output(self, output: Any) -> Dict[str, Any]:
        """Validate and sanitize output data."""
        start_time = datetime.utcnow()
        
        # Apply DLP to output
        dlp_result = self.dlp_engine.scan_output(output)
        
        # Sanitize output
        sanitized_output = self.content_sanitizer.sanitize_output(output)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "sanitized_output": sanitized_output,
            "dlp_violations": dlp_result["violations"],
            "warnings": dlp_result["warnings"],
            "processing_time": processing_time
        }
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        return {
            **self.validation_stats,
            "success_rate": self.validation_stats["passed"] / max(total, 1),
            "failure_rate": self.validation_stats["failed"] / max(total, 1),
            "sanitization_rate": self.validation_stats["sanitized"] / max(total, 1),
            "block_rate": self.validation_stats["blocked"] / max(total, 1)
        }


class ContentSanitizer:
    """Content sanitization for inputs and outputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def sanitize_input(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input arguments."""
        sanitized = {}
        
        for key, value in args.items():
            if isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def sanitize_output(self, output: Any) -> Any:
        """Sanitize output data."""
        if isinstance(output, str):
            return self._sanitize_string(output)
        elif isinstance(output, dict):
            return {k: self.sanitize_output(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self.sanitize_output(item) for item in output]
        else:
            return output
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize individual string value."""
        # Remove potentially harmful characters
        sanitized = value
        
        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")
        
        # Remove control characters (except newline, tab, carriage return)
        sanitized = re.sub(r"[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", sanitized)
        
        # HTML escape
        sanitized = html.escape(sanitized)
        
        # Remove suspicious patterns
        suspicious_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"(?:onerror|onload|onclick)\s*=",
        ]
        
        for pattern in suspicious_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized


class DataLossPreventionEngine:
    """Data Loss Prevention (DLP) engine for sensitive data detection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sensitive_patterns = self._load_sensitive_patterns()
    
    def _load_sensitive_patterns(self) -> List[Dict[str, Any]]:
        """Load sensitive data patterns."""
        return [
            {
                "name": "API Keys",
                "pattern": r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
                "severity": "high"
            },
            {
                "name": "Passwords",
                "pattern": r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\\s'\"]{8,})['\"]?",
                "severity": "high"
            },
            {
                "name": "Email Addresses",
                "pattern": r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
                "severity": "medium"
            },
            {
                "name": "Social Security Numbers",
                "pattern": r"\\b\\d{3}-\\d{2}-\\d{4}\\b",
                "severity": "high"
            },
            {
                "name": "Credit Card Numbers",
                "pattern": r"\\b(?:\\d{4}[\\s-]?){3}\\d{4}\\b",
                "severity": "high"
            },
            {
                "name": "IP Addresses",
                "pattern": r"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b",
                "severity": "low"
            }
        ]
    
    def scan_output(self, output: Any) -> Dict[str, Any]:
        """Scan output for sensitive data."""
        violations = []
        warnings = []
        
        if isinstance(output, str):
            self._scan_string(output, violations, warnings)
        elif isinstance(output, dict):
            self._scan_dict(output, violations, warnings)
        elif isinstance(output, list):
            self._scan_list(output, violations, warnings)
        
        return {
            "violations": violations,
            "warnings": warnings
        }
    
    def _scan_string(self, text: str, violations: List[str], warnings: List[str]) -> None:
        """Scan string for sensitive patterns."""
        for pattern_info in self.sensitive_patterns:
            matches = re.findall(pattern_info["pattern"], text)
            if matches:
                if pattern_info["severity"] == "high":
                    violations.append(f"Sensitive data detected: {pattern_info['name']}")
                else:
                    warnings.append(f"Potentially sensitive data: {pattern_info['name']}")
    
    def _scan_dict(self, data: Dict[str, Any], violations: List[str], warnings: List[str]) -> None:
        """Scan dictionary for sensitive data."""
        for key, value in data.items():
            if isinstance(value, str):
                self._scan_string(value, violations, warnings)
            elif isinstance(value, (dict, list)):
                self.scan_output(value)
    
    def _scan_list(self, data: List[Any], violations: List[str], warnings: List[str]) -> None:
        """Scan list for sensitive data."""
        for item in data:
            if isinstance(item, str):
                self._scan_string(item, violations, warnings)
            elif isinstance(item, (dict, list)):
                self.scan_output(item)