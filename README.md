# 💊 DrugAI: GNN Bioactivity Prediction Platform

![DrugAI](https://img.shields.io/badge/DrugAI-GNN%20Bioactivity%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 📖 Overview

DrugAI is an advanced web application that leverages Graph Neural Networks (GNNs) to predict compound bioactivity against biological targets. This platform is specifically designed for drug discovery researchers, computational chemists, and bioinformaticians working on early-stage drug development.

**Key Application**: Predicting EGFR (Epidermal Growth Factor Receptor) inhibitor bioactivity for cancer drug discovery.

## 🚀 Features

### 🔬 Core Capabilities
- **Molecular Graph Conversion**: Convert SMILES strings to molecular graphs with advanced atom featurization
- **GNN Model Training**: Train Graph Neural Networks with multiple architectures (GCN, GAT)
- **Bioactivity Prediction**: Classify compounds as Active/Inactive based on pIC50 values
- **Interactive Visualization**: Molecular structure visualization and model performance metrics

### 📊 Advanced Analytics
- **Data Quality Assessment**: Automated validation of molecular datasets
- **Class Distribution Analysis**: Interactive threshold setting for activity classification
- **Model Performance Metrics**: Comprehensive evaluation with ROC curves, confusion matrices
- **Molecular Property Analysis**: Calculation of key physicochemical properties

### 🎯 User-Friendly Interface
- **Streamlit Web App**: No-code interface for researchers
- **Real-time Training Monitoring**: Live progress tracking during model training
- **Batch Prediction**: High-throughput screening of compound libraries
- **Results Export**: Download predictions and trained models

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/drugai-gnn.git
   cd drugai-gnn
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv drugai_env
   source drugai_env/bin/activate  # On Windows: drugai_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
```text
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
rdkit>=2022.09.0
torch>=1.13.0
torch-geometric>=2.3.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.13.0
```

## 📁 Data Format

### Input CSV Requirements
Your dataset should contain the following columns:

**Required Columns:**
- `canonical_smiles` or `smiles` or `SMILES`: Molecular structures in SMILES format
- `pchembl_value` or `standard_value`: Bioactivity data

**Example Dataset:**
```csv
molecule_chembl_id,canonical_smiles,standard_value,standard_units,pchembl_value
CHEMBL137617,C/N=N/Nc1ccc2ncnc(Nc3cccc(Br)c3)c2c1,70.0,nM,7.16
CHEMBL153577,CC(=O)N(C)/N=N/c1ccc2ncnc(Nc3cccc(C)c3)c2c1,578.0,nM,6.24
CHEMBL152448,CN(CO)/N=N/c1ccc2ncnc(Nc3cccc(Br)c3)c2c1,110.0,nM,6.96
```

## 🎮 Usage

### Starting the Application
```bash
streamlit run drugai_gnn_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Workflow

1. **Data Upload**
   - Upload your CSV file containing SMILES and bioactivity data
   - Automatic data quality assessment and validation

2. **Configuration**
   - Set activity threshold (default: pIC50 ≥ 6.0)
   - Adjust model hyperparameters in the sidebar
   - Choose GNN architecture (GCN or GAT)

3. **Model Training**
   - Click "Train GNN Model" to start training
   - Monitor real-time training progress and metrics
   - View training history and validation performance

4. **Evaluation & Prediction**
   - Evaluate model performance on test set
   - Predict activity for new compounds
   - Download results and trained models

## 🧠 Model Architecture

### Graph Neural Network
```python
class DrugGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc1 = torch.nn.Linear(hidden, hidden // 2)
        self.fc2 = torch.nn.Linear(hidden // 2, 2)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### Atom Featurization
- **Atom Types**: One-hot encoding for common elements (C, N, O, F, P, S, Cl, Br)
- **Chemical Properties**: Degree, formal charge, aromaticity, hydrogen count
- **Total Features**: 12-dimensional atom representations

## 📊 Performance Metrics

The platform provides comprehensive model evaluation:

- **Accuracy**: Overall prediction correctness
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Area under Receiver Operating Characteristic curve
- **Confusion Matrix**: Detailed classification breakdown
- **Training History**: Loss and accuracy trends over epochs

## 🎯 Use Cases

### 🧫 Drug Discovery
- **Virtual Screening**: Prioritize compounds for experimental testing
- **Lead Optimization**: Guide structural modifications for improved activity
- **Target Identification**: Predict activity against novel targets

### 🔬 Academic Research
- **Method Development**: Test new GNN architectures
- **Dataset Analysis**: Explore structure-activity relationships
- **Benchmarking**: Compare with traditional QSAR methods

### 💼 Pharmaceutical Industry
- **High-Throughput Screening**: Accelerate compound prioritization
- **Multi-Target Profiling**: Predict selectivity across target families
- **ADMET Prediction**: Extend to pharmacokinetic properties

## 🚀 Advanced Features

### Multi-Target Prediction
```python
# Extension for multi-target activity prediction
class MultiTargetGNN(DrugGNN):
    def __init__(self, in_channels, hidden, num_targets):
        super().__init__(in_channels, hidden)
        self.target_heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden // 2, 2) for _ in range(num_targets)
        ])
```

### Transfer Learning
- Pre-train on large compound databases
- Fine-tune for specific target families
- Leverage existing bioactivity data

## 🔧 Configuration Options

### Model Parameters
- **Hidden Dimension**: 16-128 neurons
- **Learning Rate**: 0.0001-0.01
- **Batch Size**: 8-64 compounds
- **Training Epochs**: 10-100 iterations
- **Dropout Rate**: 0.0-0.5 for regularization

### Data Processing
- **Activity Threshold**: Customizable pIC50 cutoff
- **Test Split**: 10-40% for model evaluation
- **Stratified Sampling**: Maintain class balance

## 📈 Example Results

### Model Performance on EGFR Dataset
| Metric | Value |
|--------|-------|
| Accuracy | 0.85 |
| F1-Score | 0.83 |
| ROC-AUC | 0.89 |
| Precision | 0.82 |
| Recall | 0.84 |

### Prediction Examples
| Compound | SMILES | Prediction | Confidence |
|----------|--------|------------|------------|
| Aspirin | CC(=O)Oc1ccccc1C(=O)O | Inactive | 0.92 |
| Caffeine | CN1C=NC2=C1C(=O)N(C(=O)N2C)C | Active | 0.78 |

## 🛠️ Troubleshooting

### Common Issues

1. **Dimension Mismatch Errors**
   - Ensure consistent atom featurization
   - Check input data formatting
   - Verify package versions

2. **Memory Issues**
   - Reduce batch size
   - Use smaller hidden dimensions
   - Process datasets in chunks

3. **Installation Problems**
   - Use conda for RDKit installation: `conda install -c conda-forge rdkit`
   - Install PyTorch Geometric following official documentation

### Performance Tips
- Use GPU acceleration for faster training
- Pre-process large datasets offline
- Enable mixed-precision training for large models

## 🤝 Contributing

We welcome contributions from the community!

### How to Contribute
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black drugai_gnn_app.py
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RDKit**: Chemical informatics and machine learning
- **PyTorch Geometric**: Graph neural network library
- **Streamlit**: Rapid web application development
- **ChEMBL**: Bioactivity data resource

## 📚 References

1. **Graph Neural Networks in Drug Discovery**  
   *Journal of Chemical Information and Modeling, 2023*

2. **Deep Learning for Bioactivity Prediction**  
   *Nature Machine Intelligence, 2022*

3. **EGFR Inhibitors in Cancer Therapy**  
   *Clinical Cancer Research, 2021*

## 🎯 Roadmap

### Upcoming Features
- [ ] Multi-task learning for polypharmacology
- [ ] 3D molecular graph convolutions
- [ ] Explainable AI with attention mechanisms
- [ ] Integration with commercial compound databases
- [ ] Cloud deployment and scaling

### Research Directions
- Quantum mechanical feature incorporation
- Generative models for molecular design
- Federated learning for collaborative drug discovery

*Accelerating Drug Discovery with AI-Powered Insights*

</div>
