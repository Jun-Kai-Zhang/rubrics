# Workflow Module Migration Guide

## Overview

The workflow module has been refactored from a monolithic structure into a modular, layered architecture following SOLID principles and clean architecture patterns.

## New Structure

```
workflow/
├── core/                    # Domain logic and business rules
│   ├── interfaces.py       # Protocol definitions
│   └── services.py         # Business services
├── infrastructure/         # External dependencies and I/O
│   ├── file_handler.py    # File operations
│   └── model_adapter.py   # External model integration
├── orchestration/          # Workflow coordination
│   ├── iteration_manager.py    # Single iteration logic
│   └── workflow_coordinator.py # Overall workflow management
├── data_structures.py      # Data models (unchanged)
├── workflow_engine.py      # Backward-compatible wrapper
└── workflow.py            # Entry point (unchanged)
```

## Key Changes

### 1. Separation of Concerns

- **Core**: Pure business logic without external dependencies
- **Infrastructure**: All I/O operations and external integrations
- **Orchestration**: Workflow coordination and iteration management

### 2. Dependency Injection

Instead of creating dependencies internally, components now receive them via constructor:

```python
# Old
class WorkflowEngine:
    def __init__(self):
        self.file_handler = FileHandler()
        self.rubrics_generator = RubricsGenerator()

# New
class WorkflowCoordinator:
    def __init__(self, file_handler: FileHandler, model_adapter: ModelAdapter):
        self.file_handler = file_handler
        self.model_adapter = model_adapter
```

### 3. Protocol-Based Design

External dependencies are abstracted behind protocols:

```python
class RubricGenerator(Protocol):
    def generate_initial_rubrics(self, responses: List[Dict], model: str) -> Dict[str, Dict]:
        ...
```

### 4. Simplified File Operations

The FileHandler now focuses solely on I/O operations without business logic:

```python
# Old: Mixed concerns
def analyze_scores_for_ties(self, scored_file: str) -> TieAnalysis:
    # File I/O + business logic

# New: Separated
# FileHandler: Just I/O
def load_json(self, filepath: str) -> Dict[str, Any]:
    # Pure file reading

# TieAnalysisService: Business logic
def analyze_scored_responses(self, scored_data: Dict, prompt_ids: Set[str]) -> TieAnalysis:
    # Pure business logic
```

## Migration Steps

### For Users

1. **No changes required** - The public API remains the same through `WorkflowEngine`
2. Configuration files work as before
3. Command-line usage is unchanged

### For Developers

1. **File Removals and Mappings**:
   
   The following files have been removed and replaced:
   - `file_operations.py` → `infrastructure/file_handler.py` + `core/services.py`
   - `response_processing.py` → `core/services.py` (ScoringService, TieAnalysisService)
   - `rubric_operations.py` → `infrastructure/model_adapter.py`

2. **Import Updates**:
   ```python
   # Old (no longer exists)
   from workflow.file_operations import FileHandler
   from workflow.response_processing import ResponseProcessor
   from workflow.rubric_operations import RubricOperations
   
   # New
   from workflow.infrastructure import FileHandler, ModelAdapter
   from workflow.core import RubricService, ScoringService, TieAnalysisService
   ```

2. **Service Usage**:
   ```python
   # Old: Direct method calls
   tie_analysis = file_handler.analyze_scores_for_ties(scored_file)
   
   # New: Use appropriate service
   tie_analysis = tie_analysis_service.analyze_scored_responses(scored_data, prompt_ids)
   ```

3. **Testing**:
   - Mock protocols instead of concrete classes
   - Test services in isolation
   - Use dependency injection for test doubles

## Benefits

1. **Maintainability**: Smaller, focused modules are easier to understand and modify
2. **Testability**: Components can be tested in isolation with mocked dependencies
3. **Flexibility**: Easy to swap implementations (e.g., different model backends)
4. **Scalability**: Clear boundaries make it easier to add new features
5. **Error Handling**: Consistent patterns across layers

## Backward Compatibility

The `WorkflowEngine` class maintains full backward compatibility by delegating to the new modular system. Existing code will continue to work without modifications.

## Future Enhancements

1. **Plugin System**: The protocol-based design enables easy plugin development
2. **Async Support**: Infrastructure layer can be made async without affecting business logic
3. **Alternative Backends**: Easy to add new model providers or storage backends
4. **Monitoring**: Clean insertion points for metrics and logging 