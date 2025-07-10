# Multi-Part Report System with Model Allocation

## 🎯 Overview

The multi-part report system now includes intelligent model allocation to optimize costs while maintaining quality. You can easily switch between cost-optimized and premium modes.

## 🚀 Quick Start

### Basic Usage

```python
from src.llm.rag_system.memory import quick_switch_to_optimized, quick_switch_to_o3

# For cost-effective analysis (recommended for development)
quick_switch_to_optimized()

# For maximum quality (recommended for final production)
quick_switch_to_o3()
```

### Current Model Configuration

**Optimized Mode (Default):**
- Simple tasks: `gpt-4.1-mini` ($0.15/1M tokens)
- Complex synthesis: `o3` ($15/1M tokens)
- **Estimated savings: 70-80% vs premium everywhere**

**Premium Mode:**
- All tasks: `o3` ($15/1M tokens)
- Maximum quality, highest cost

## 📊 Model Allocation Strategy

### Task Categories

| Task Type | Complexity | Optimized Model | Premium Model |
|-----------|------------|-----------------|---------------|
| Report classification | Simple | gpt-4.1-mini | o3 |
| Executive summary | Medium | gpt-4.1-mini | o3 |
| Report parts | Medium | gpt-4.1-mini | o3 |
| Final synthesis | Complex | o3 | o3 |
| Biological interpretation | Complex | o3 | o3 |

### When to Use Each Mode

**Optimized Mode** - Use for:
- Development and testing
- Cost-sensitive analysis
- Large-scale batch processing
- Initial exploration

**Premium Mode** - Use for:
- Final production reports
- Critical analysis requiring maximum accuracy
- Complex biological interpretations
- Publication-quality results

## 🔧 Configuration Functions

### Simple Switching

```python
from src.llm.rag_system.memory import (
    quick_switch_to_optimized,
    quick_switch_to_o3,
    quick_switch_to_testing,
    print_model_status
)

# Switch modes
quick_switch_to_optimized()  # Cost-effective
quick_switch_to_o3()         # Maximum quality
quick_switch_to_testing()    # Development mode

# Check current status
print_model_status()
```

### Advanced Configuration

```python
from src.llm.rag_system.memory import (
    get_model_allocator,
    set_optimized_mode,
    set_premium_mode,
    get_current_mode
)

# Get current mode
current_mode = get_current_mode()

# Get detailed allocation info
allocator = get_model_allocator()
summary = allocator.get_allocation_summary()

# Estimate costs
cost = allocator.get_cost_estimate("final_synthesis", 10000)
```

## 💡 Development Workflow

### Recommended Development Process

1. **Development Phase**
   ```python
   quick_switch_to_testing()
   # Fast iteration with cheap models
   ```

2. **Quality Testing**
   ```python
   quick_switch_to_optimized()
   # Better quality with reasonable costs
   ```

3. **Final Production**
   ```python
   quick_switch_to_o3()
   # Maximum quality for final results
   ```

## 💰 Cost Optimization Examples

### Example Cost Comparison

For a typical multi-part report with 50,000 tokens:

**Optimized Mode:**
- Simple tasks (5,000 tokens): $0.0008
- Complex synthesis (10,000 tokens): $0.15
- **Total: ~$0.16**

**Premium Mode:**
- All tasks (50,000 tokens): $0.75
- **Total: ~$0.75**

**Savings: ~79%**

### Cost Monitoring

```python
# Check cost estimates before running
allocator = get_model_allocator()
tasks = ["executive_summary", "report_part_generation", "final_synthesis"]

for task in tasks:
    cost = allocator.get_cost_estimate(task, 5000)
    print(f"{task}: ${cost:.4f}")
```

## 🧪 Testing the System

### Run the Test Suite

```bash
# Test model allocation
python -m src.llm.rag_system.memory.test_model_allocation

# Test model switching
python -m src.llm.rag_system.memory.model_config
```

### Test Prompts for Multi-Part Reports

1. **Large Dataset Analysis:**
   ```
   "Provide a comprehensive analysis of all transport proteins across all genomes"
   ```

2. **Comparative Analysis:**
   ```
   "Compare the metabolic capabilities across all genomes in detail"
   ```

3. **System-Specific:**
   ```
   "Generate a detailed report on all protein families and their functions"
   ```

## 🔍 Features

### Automatic Detection

The system automatically:
- Detects when multi-part reports are needed (>50 data points)
- Selects appropriate models based on task complexity
- Provides fallback logic when models fail
- Estimates costs and provides savings information

### Easy Switching

```python
# One-line switches
quick_switch_to_optimized()  # 70-80% cost savings
quick_switch_to_o3()         # Maximum quality

# Status checking
print_model_status()         # Current configuration
```

### Fallback Logic

- If a cheaper model fails, automatically falls back to premium model
- Maintains system reliability while optimizing costs
- Logs all fallback events for monitoring

## 🚨 Important Notes

1. **Default Mode**: System starts in optimized mode by default
2. **Easy Switching**: Can switch between modes at any time
3. **No API Waste**: Use optimized mode for development and testing
4. **Premium When Needed**: Switch to o3 for final production results
5. **Fallback Safety**: System falls back to premium models if cheaper ones fail

## 📈 Performance Impact

**Optimized Mode:**
- 70-80% cost reduction
- Slightly longer processing time for complex tasks
- Maintains high quality for final synthesis

**Premium Mode:**
- Maximum accuracy and biological insight
- Fastest processing for complex reasoning
- Higher cost per token

## 🎉 Ready to Use!

The system is now ready for testing with significant cost optimizations. Start with optimized mode for development, then switch to premium mode when you need maximum quality results.

```python
# Start here
from src.llm.rag_system.memory import quick_switch_to_optimized
quick_switch_to_optimized()

# Test with one of the sample prompts
# Switch to premium when ready for production
```