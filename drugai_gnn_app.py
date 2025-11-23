import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="DrugAI GNN", layout="wide")
st.title("💊 DrugAI: GNN Bioactivity Prediction")

# -------------------- Sidebar Configuration --------------------
st.sidebar.header("⚙️ Model Configuration")

hidden_dim = st.sidebar.slider("Hidden Dimension", 16, 128, 64, 16)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
epochs = st.sidebar.slider("Epochs", 10, 100, 30, 5)
batch_size = st.sidebar.slider("Batch Size", 8, 64, 16, 8)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

# -------------------- Simple Atom Features --------------------
def atom_feature(atom):
    """Simple atom features to avoid dimension issues"""
    features = []
    
    # Atom type (one-hot for common atoms)
    atom_types = [6, 7, 8, 9, 15, 16, 17, 35]  # C, N, O, F, P, S, Cl, Br
    for at_type in atom_types:
        features.append(1 if atom.GetAtomicNum() == at_type else 0)
    
    # Basic properties
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(atom.GetTotalNumHs())
    
    return np.array(features, dtype=float)

def mol_to_graph(smiles, label):
    """Convert SMILES to graph without complex descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_feature(atom))
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Undirected graph
    
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    y = torch.tensor([label], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y, smiles=smiles)

# -------------------- Simple GNN Model --------------------
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# -------------------- Training Function --------------------
def train_simple_model(model, train_loader, val_loader, epochs, device, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses, train_accs, val_accs = [], [], []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y.view(-1)).sum().item()
                val_total += batch.y.size(0)
        
        val_acc = val_correct / val_total
        
        # Store metrics
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update UI
        progress_bar.progress(epoch / epochs)
        status_text.text(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    return train_losses, train_accs, val_accs

# -------------------- Prediction Function --------------------
def predict_smiles_simple(model, smiles_list, device):
    model.eval()
    predictions = []
    probabilities = []
    valid_smiles = []
    
    for smiles in smiles_list:
        graph = mol_to_graph(smiles, 0)
        if graph is not None:
            graph = graph.to(device)
            with torch.no_grad():
                out = model(graph.x, graph.edge_index, torch.tensor([0], device=device))
                prob = F.softmax(out, dim=1)[0, 1].detach().cpu().numpy()
                pred = out.argmax(dim=1).detach().cpu().numpy()[0]
                
            predictions.append(pred)
            probabilities.append(prob)
            valid_smiles.append(smiles)
        else:
            st.warning(f"Invalid SMILES skipped: {smiles}")
    
    return valid_smiles, predictions, probabilities

# -------------------- Main Application --------------------
def main():
    uploaded_file = st.file_uploader("📁 Upload CSV file with molecular data", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Dataset Overview")
        
        # Handle column names
        if 'pchembl_value' in df.columns:
            df = df.rename(columns={'pchembl_value': 'pIC50'})
        elif 'standard_value' in df.columns:
            df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
        
        # Find SMILES column
        smiles_col = None
        for col in ['canonical_smiles', 'smiles', 'SMILES']:
            if col in df.columns:
                smiles_col = col
                break
        
        if not smiles_col:
            st.error("No SMILES column found!")
            return
        
        if smiles_col != 'canonical_smiles':
            df = df.rename(columns={smiles_col: 'canonical_smiles'})
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Compounds", len(df))
        with col2:
            st.metric("Unique SMILES", df['canonical_smiles'].nunique())
        with col3:
            if 'pIC50' in df.columns:
                st.metric("pIC50 Range", f"{df['pIC50'].min():.2f}-{df['pIC50'].max():.2f}")
        
        # Activity threshold
        st.subheader("🎯 Activity Classification")
        
        if 'pIC50' in df.columns:
            threshold = st.slider("pIC50 Threshold for Active/Inactive", 
                                float(df['pIC50'].min()), 
                                float(df['pIC50'].max()), 6.0, 0.1)
            
            df['label'] = (df['pIC50'] >= threshold).astype(int)
            
            # Display class distribution
            active_count = df['label'].sum()
            inactive_count = len(df) - active_count
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=[inactive_count, active_count], 
                           names=['Inactive', 'Active'],
                           title='Class Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Active Compounds", active_count)
                st.metric("Inactive Compounds", inactive_count)
                st.metric("Active Ratio", f"{active_count/len(df)*100:.1f}%")
        
        # Model Training
        st.subheader("🚀 Model Training")
        
        if st.button("Train GNN Model", type="primary"):
            with st.spinner("Converting molecules to graphs..."):
                graphs = []
                invalid_count = 0
                
                for i, (smi, lbl) in enumerate(zip(df['canonical_smiles'], df['label'])):
                    graph = mol_to_graph(smi, lbl)
                    if graph is not None:
                        graphs.append(graph)
                    else:
                        invalid_count += 1
                
                if invalid_count > 0:
                    st.warning(f"Skipped {invalid_count} invalid SMILES strings")
                
                if len(graphs) < 10:
                    st.error("Not enough valid molecules for training (need at least 10)")
                    return
                
                st.success(f"✅ Converted {len(graphs)} molecules to graph data")
                
                # Check input dimension
                input_dim = graphs[0].x.shape[1]
                st.info(f"Input dimension: {input_dim}")
                
                # Split data
                train_graphs, test_graphs = train_test_split(
                    graphs, test_size=test_size, random_state=42, 
                    stratify=[g.y.item() for g in graphs])
                
                train_graphs, val_graphs = train_test_split(
                    train_graphs, test_size=0.2, random_state=42,
                    stratify=[g.y.item() for g in train_graphs])
                
                train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
                
                st.write(f"**Dataset Split:**")
                st.write(f"- Training: {len(train_graphs)} graphs")
                st.write(f"- Validation: {len(val_graphs)} graphs")
                st.write(f"- Test: {len(test_graphs)} graphs")
                
                # Initialize model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                st.info(f"Using device: {device}")
                
                model = SimpleGNN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
                
                # Display model info
                total_params = sum(p.numel() for p in model.parameters())
                st.info(f"Model parameters: {total_params:,}")
                
                # Train model
                with st.spinner("Training in progress..."):
                    train_losses, train_accs, val_accs = train_simple_model(
                        model, train_loader, val_loader, epochs, device, learning_rate)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.test_loader = test_loader
                    st.session_state.device = device
                    
                    # Plot training history
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=train_losses, name='Training Loss', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(y=train_accs, name='Training Accuracy', line=dict(color='green'), yaxis='y2'))
                    fig.add_trace(go.Scatter(y=val_accs, name='Validation Accuracy', line=dict(color='orange'), yaxis='y2'))
                    
                    fig.update_layout(
                        title='Training History',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Training completed successfully!")
        
        # Model Evaluation
        if "model" in st.session_state:
            model = st.session_state.model
            test_loader = st.session_state.test_loader
            device = st.session_state.device
            
            st.subheader("📈 Model Evaluation")
            
            if st.button("Evaluate Model"):
                with st.spinner("Evaluating..."):
                    model.eval()
                    all_y = []
                    all_pred = []
                    all_probs = []
                    
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = batch.to(device)
                            out = model(batch.x, batch.edge_index, batch.batch)
                            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                            preds = out.argmax(dim=1).cpu().numpy()
                            
                            all_y.extend(batch.y.cpu().numpy())
                            all_pred.extend(preds)
                            all_probs.extend(probs)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(all_y, all_pred)
                    f1 = f1_score(all_y, all_pred)
                    auc = roc_auc_score(all_y, all_probs)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("F1 Score", f"{f1:.4f}")
                    with col3:
                        st.metric("ROC-AUC", f"{auc:.4f}")
                    
                    # Confusion matrix
                    cm = confusion_matrix(all_y, all_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                              xticklabels=['Inactive', 'Active'],
                              yticklabels=['Inactive', 'Active'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
            
            # Prediction Interface
            st.subheader("🔮 Predict New Molecules")
            
            tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
            
            with tab1:
                smiles_input = st.text_input("Enter SMILES:", value="CCO")
                if st.button("Predict") and smiles_input:
                    valid_smiles, preds, probs = predict_smiles_simple(model, [smiles_input], device)
                    if valid_smiles:
                        pred_text = "Active" if preds[0] == 1 else "Inactive"
                        confidence = probs[0] if preds[0] == 1 else 1 - probs[0]
                        
                        st.success(f"**Prediction:** {pred_text}")
                        st.info(f"**Confidence:** {confidence:.3f}")
                        
                        # Show molecule
                        mol = Chem.MolFromSmiles(smiles_input)
                        if mol:
                            img = Draw.MolToImage(mol, size=(300, 300))
                            st.image(img, caption="Molecule Structure")
            
            with tab2:
                batch_smiles = st.text_area("Enter SMILES (one per line):", 
                                          "CCO\nCC(=O)O\nC1=CC=CC=C1")
                if st.button("Predict Batch") and batch_smiles:
                    smiles_list = [s.strip() for s in batch_smiles.split('\n') if s.strip()]
                    valid_smiles, preds, probs = predict_smiles_simple(model, smiles_list, device)
                    
                    if valid_smiles:
                        results = pd.DataFrame({
                            'SMILES': valid_smiles,
                            'Prediction': ['Active' if p == 1 else 'Inactive' for p in preds],
                            'Probability_Active': probs,
                            'Confidence': [p if pred == 1 else 1-p for p, pred in zip(probs, preds)]
                        })
                        
                        st.dataframe(results)
                        
                        # Download
                        csv = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )

if __name__ == "__main__":
    main()