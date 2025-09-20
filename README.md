# 🧠 Language Emergence Lab

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/yourusername/langlab/actions)

A comprehensive research framework for studying **emergent language** in multi-agent referential games. This project explores how artificial agents develop communication protocols through interaction, investigating phenomena like compositional language, cultural transmission, and pragmatic inference.

## 🌟 Key Features

### 🤖 **Multi-Agent Communication**
- **Speaker-Listener Architecture**: Neural agents that learn to communicate about objects
- **Discrete Communication**: Gumbel-Softmax for differentiable discrete message generation
- **Multimodal Support**: Parallel gesture and token communication channels
- **Sequence Models**: Autoregressive GRU-based message generation

### 🧬 **Population Dynamics**
- **Cultural Transmission**: Study language evolution across agent populations
- **Agent Lifespans**: Realistic population turnover and replacement dynamics
- **Cross-Population Contact**: Analyze language intelligibility between populations
- **Replacement Strategies**: Configurable noise injection for new agents

### 🔬 **Advanced Experiments**
- **Compositional Generalization**: Test systematicity in emergent languages
- **Pragmatic Inference**: Distractor-based pragmatic reasoning
- **Ablation Studies**: Systematic parameter sweeps across vocabulary, noise, and length costs
- **Grounded Navigation**: Spatial referential games in grid worlds

### 📊 **Analysis & Visualization**
- **Zipf Analysis**: Measure language efficiency and structure
- **Interactive Dashboard**: Streamlit-based experiment visualization
- **Comprehensive Metrics**: Accuracy, compositionality, entropy, and more
- **Export Capabilities**: CSV reports and publication-ready figures

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bangyen/emergent.git
cd emergent

# Install with development dependencies
make init
```

### Basic Usage

```bash
# Run a simple referential game experiment
langlab train --steps 1000 --k 5 --v 10

# Launch the interactive dashboard
langlab dash

# Generate a sample scene
langlab sample --k 3

# Evaluate a trained model
langlab eval --ckpt outputs/checkpoints/checkpoint.pt

# Generate a dataset
langlab dataset --n-scenes 100 --k 5

# View system information
langlab info
```

### Python API

```python
from langlab import Speaker, Listener, sample_scene, CommunicationConfig

# Create a communication configuration
config = CommunicationConfig(
    vocabulary_size=10,
    message_length=1,
    hidden_size=64
)

# Initialize agents
speaker = Speaker(config)
listener = Listener(config)

# Sample a scene
scene_objects, target_idx = sample_scene(k=3, seed=42)
print(f"Target object: {scene_objects[target_idx]}")
```

## 📚 Research Applications

### **Language Emergence Studies**
- Investigate how communication protocols emerge from interaction
- Study the role of environmental constraints on language structure
- Analyze the emergence of compositional vs. holistic languages

### **Cultural Transmission**
- Model language evolution across generations
- Study the effects of population dynamics on language stability
- Investigate contact-induced language change

### **Pragmatic Communication**
- Explore how context affects communication strategies
- Study the emergence of pragmatic inference
- Analyze efficiency vs. robustness trade-offs

## 🧪 Experiment Types

### Available CLI Commands

| Command | Description |
|---------|-------------|
| `langlab train` | Train Speaker and Listener agents for emergent language |
| `langlab eval` | Evaluate model performance on specified data split |
| `langlab pop-train` | Train a population of agent pairs for cultural transmission |
| `langlab contact` | Cross-population contact experiments |
| `langlab train-grid` | Grounded navigation in grid worlds |
| `langlab ablate` | Run ablation studies across parameter configurations |
| `langlab sample` | Generate and display sample scenes |
| `langlab dataset` | Generate and analyze referential game datasets |
| `langlab dash` | Launch interactive Streamlit dashboard |
| `langlab report` | Generate ablation study reports |
| `langlab info` | Display system information |

### 1. **Basic Referential Games**
```bash
langlab train --steps 5000 --k 5 --v 10 --message-length 1
```

### 2. **Population Dynamics**
```bash
langlab pop-train --pairs 5 --lifespan 1000 --steps 10000
```

### 3. **Cross-Population Contact**
```bash
langlab contact --pairs 4 --steps-a 4000 --steps-b 4000 --contact-steps 2000
```

### 4. **Grounded Navigation**
```bash
langlab train-grid --episodes 500 --grid 5 --message-length 3
```

### 5. **Ablation Studies**
```bash
langlab ablate --vocab-sizes "6,12,24" --noise-levels "0,0.05,0.1"
```

## 📊 Results & Analysis

The framework includes comprehensive analysis tools:

- **Zipf Analysis**: Measure language efficiency and structure
- **Compositional Accuracy**: Test systematicity in language use
- **Token Distribution**: Analyze vocabulary usage patterns
- **Training Dynamics**: Monitor learning progress and convergence

### Example Results

Our experiments show that:
- **Vocabulary size** significantly affects language efficiency
- **Channel noise** promotes more robust communication strategies
- **Population dynamics** lead to cultural drift and language evolution
- **Compositional languages** emerge under specific environmental constraints

## 🏗️ Architecture

### Core Components

```
src/langlab/
├── agents.py          # Speaker/Listener neural architectures
├── world.py           # Object generation and scene sampling
├── channel.py         # Communication channel implementation
├── data.py            # Dataset and data loading utilities
├── train.py           # Training loops and optimization
├── population.py      # Population management and dynamics
├── contact.py         # Cross-population interaction experiments
├── analysis.py        # Language analysis and metrics
├── app.py            # Streamlit dashboard
└── cli.py            # Command-line interface
```

### Key Design Principles

- **Modular Architecture**: Clean separation of concerns
- **Type Safety**: Full type hints with MyPy validation
- **Reproducibility**: Comprehensive seeding and logging
- **Extensibility**: Easy to add new agent architectures or experiments

## 🔧 Development

### Code Quality
```bash
make fmt    # Format with Black
make lint   # Lint with Ruff
make type   # Type check with MyPy
make test   # Run tests with pytest
make all    # Run all quality checks
```

### Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/test_agents.py
python -m pytest tests/test_population.py
```

## 📈 Performance

The framework is optimized for research-scale experiments:

- **GPU Support**: Automatic CUDA detection and utilization
- **Efficient Batching**: Optimized data loading and training loops
- **Memory Management**: Careful tensor management for large experiments
- **Parallel Processing**: Support for multi-GPU training

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/bangyen/emergent.git
cd emergent

# Install in development mode
make init

# Run tests to ensure everything works
make test
```
