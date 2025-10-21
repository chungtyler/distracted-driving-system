import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


'''
Load the data set to inputs X and output y

X:
1) Number of times pregnant
2) Plasma glucose concentration at 2h in oral glucose tolerance test
3) Diastolic blood pressure (mm Hg)
4) Triceps skin fold thickness (mm)
5) 2-h serum insulin (MuU/ml)
6) Body mass index (kg/m2)
7) Diabetes pedigree function
8) Age (years)

y:
1) Class label (0 or 1), diabetic
'''

dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',') # Load csv
X = dataset[:,0:8] # Grab columns 0 to 7
y = dataset[:,8] # Grab columns 8

# Convert Numpy 64 bit data to pytorch compatible 32 bit
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y ,dtype=torch.float32).reshape(-1,1) # Transpose

'''
Define the model
Use 4 layers
ReLU for hidden layers for improved performance (reduces vanishing gradient problem)
Sigmoid for output (to make probability or threshold between 1 and 0)
'''

model = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 75),
    nn.ReLU(),
    nn.Linear(75, 1),
    nn.Sigmoid()
)

'''
Prepare Training
'''

loss_fn = nn.BCELoss() # Binary cross entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate @ 0.001, optimizer method of gradient descent

'''
Training the Model
'''

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size] # Batch of data inputs
        y_pred = model(Xbatch) # Predicted y output
        ybatch = y[i:i+batch_size] # Actual y output
        loss = loss_fn(y_pred, ybatch) # Loss
        optimizer.zero_grad() # Gradient descent
        loss.backward() # Backwards propagation
        optimizer.step() # Update parameters
    print(f'Finished epoch {epoch}, latest loss {loss}')

'''
Evaluate the Model
'''

with torch.no_grad(): # Prevents model parameter changes
    y_pred = model(X) # Get's predicted output from all inputs

accuracy = (y_pred.round() == y).float().mean() # From probability between 0 to 1 converts to binary 0 and 1 and finds mean
print(f"Accuracy: {accuracy}")

'''
Perform Inference
'''
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
