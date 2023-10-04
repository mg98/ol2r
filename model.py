import torch
import torch.nn as nn
import numpy as np

def make_input(query_vector, sup_doc_vector, inf_doc_vector):
  """
  Make (query, document-pair) input for model.
  """
  return np.array([query_vector, sup_doc_vector, inf_doc_vector], dtype=np.float32).flatten()

# Model definition
model = nn.Sequential(
    nn.Linear(3*768, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(pos_train_data, neg_train_data, num_epochs):
  for epoch in range(num_epochs):
      losses = []
      # train positive pairs
      for data in pos_train_data:
          output = model(data)
          loss = criterion(output, torch.tensor([1.0]))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          losses.append(loss.item())
      # train negative pairs too
      for data in neg_train_data:
          output = model(data)
          loss = criterion(output, torch.tensor([0.0]))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          losses.append(loss.item())

      if (epoch + 1) == num_epochs:
          print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {(sum(losses) / len(losses)):.4f}')