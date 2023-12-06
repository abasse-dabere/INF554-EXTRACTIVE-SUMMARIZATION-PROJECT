# **PyTorch Geometric**
from tqdm import trange, tqdm
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import f1_score

# Assuming sentences, relations, and labels are lists and have been defined
sentences = X_training  # list of sentences
relations = graph  # list of relations corresponding to sentences
labels = y_training  # list of labels corresponding to sentences

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

model = model.to('cuda')

# Put the model in "evaluation" mode
model.eval()

# Create a NetworkX graph
G = nx.Graph()

features = []

# Batch size
batch_size = 32

# Add nodes with BERT embeddings
for i in trange(0, len(sentences), batch_size):
    # Tokenize input
    inputs = tokenizer(sentences[i:i+batch_size], return_tensors='pt', padding=True, truncation=True, max_length=512)

    # If a GPU is available, move the inputs to GPU
    if torch.cuda.is_available():
        inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}

    # Predict hidden states features for each layer
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings of the last layer
    last_hidden_states = outputs.last_hidden_state

    # Use the mean of the embeddings as the sentence representation
    sentence_embeddings = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()

    # Add features
    features.extend(sentence_embeddings)

    # Add the sentences to the graph
    for j, sentence_embedding in enumerate(sentence_embeddings):
        G.add_node(i+j, embedding=sentence_embedding, label=labels[i+j])

# Add edges with relation types
for rel in relations:
    G.add_edge(rel[0], rel[1], relation= categories[rel[2]])

# Now, G is a graph with sentence embeddings as node features and relations as edge features

features = np.array(features)

# !pip install torch_geometric --quiet



# Get a list of all relation types
relation_types = [G.edges[edge]['relation'] for edge in G.edges]

# One-hot encode the relation types
encoder = OneHotEncoder(sparse=False)
edge_features = encoder.fit_transform(np.array(relation_types).reshape(-1, 1))

# Convert edge features to tensor
edge_features = torch.tensor(edge_features, dtype=torch.float)

# Get the indices of the source and target nodes for each edge
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

# Get the node features and labels
node_features = torch.stack([torch.tensor(G.nodes[node]['embedding']) for node in G.nodes])
node_labels = torch.tensor([1 if G.nodes[node]['label'] == 1 else 0 for node in G.nodes], dtype=torch.long)

# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=node_labels)


data.train_mask = idx_train
data.val_mask = idx_val
data.test_mask = idx_test


class GCN(torch.nn.Module):
    def __init__(self, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 512)
        self.conv2 = GCNConv(512, 128)
        self.classifier = Linear(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

epochs = 150
# Initialize the model and optimizer
model = GCN(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
weights = [1., 2.5]  # Assuming class 1 is the minority class
class_weights = torch.FloatTensor(weights).to(device)
criterion = torch.nn.NLLLoss(weight=class_weights)
# Use a GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# History
h = {
    'loss_train': [],
    'loss_val': [],
    'f1_train': [],
    'f1_val': []
}

# Training loop
model.train()
for epoch in trange(epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Calculate F1-score
    model.eval()
    with torch.no_grad():
        predictions = model(data).max(dim=1)[1]
        train_f1 = f1_score(data.y[data.train_mask].cpu(), predictions[data.train_mask].cpu(), average='weighted')
    
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    val_f1 = f1_score(data.y[data.val_mask].cpu(), predictions[data.val_mask].cpu(), average='weighted')
    
    h['loss_train'].append(loss.item())
    h['f1_train'].append(loss.item())

    h['loss_val'].append(val_loss.item())
    h['f1_val'].append(val_f1)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Training F1-score: {train_f1}, Validation F1-score: {val_f1}')



# Calculate loss and F1-score for the test set after training
model.eval()
with torch.no_grad():
    out = model(data)
    predictions = out.max(dim=1)[1]
    test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
    print(sum(predictions[data.test_mask].cpu() - data.y[data.test_mask].cpu()))
    test_f1 = f1_score(data.y[data.test_mask].cpu(), predictions[data.test_mask].cpu(), average='weighted')
    print(f1_score(data.y.cpu(), predictions.cpu(), average='weighted'))
print(f'Test Loss: {test_loss.item()}, Test F1-score: {test_f1}')


plt.plot(h['loss_train'][20:])

plt.plot(h['f1_val'])

cm = confusion_matrix(data.y[data.test_mask].cpu(), predictions[data.test_mask].cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot()
plt.show()
