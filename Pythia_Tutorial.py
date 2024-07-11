# Add the necessary pythia build paths to the python paths
import sys
with open("Software/Software_Path.txt") as f:
    path = f.readline().strip()
cfg = open(path+"/pythia8312/Makefile.inc")  # Read necessary paths from this file
lib = "../lib"
for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break  # Find build paths
sys.path.insert(0, lib)   # Add build paths to system path

# Import Pythia8
import pythia8

# Import ML Libs
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# Initialize Pythia with Signal and Background Processes
pythia_sig = pythia8.Pythia()                  # Define a pythia8.Pythia() object
pythia_sig.readString("Beams:eCM = 14000.")    # Beam energy is 14TeV
pythia_sig.readString("Beams:idA = 2212")      # Incoming particle 1 is proton
pythia_sig.readString("Beams:idB = 2212")      # Incoming particle 2 is proton
pythia_sig.readString("Top:all = on")          # Turn on all top processes
pythia_sig.init()                              # Initialize object with user defined settings

pythia_bkg = pythia8.Pythia()                  # Define a pythia8.Pythia() object
pythia_bkg.readString("Beams:eCM = 14000.")    # Beam enegery is 14TeV
pythia_bkg.readString("Beams:idA = 2212")      # Incoming particle 1 is proton
pythia_bkg.readString("Beams:idB = 2212")      # Incoming particle 2 is proton
pythia_bkg.readString("HardQCD:all = on")      # Turn on all HardQCD processes
pythia_bkg.init()                              # Initialize object with user defined settings


# Generate Dataset
# Generate events and append stable particles to a list. Then histogram pT, eta, and phi for signal vs background.
num_events = 100

# Begin event loop. Generate event. Skip if error.
stbl_sig = []
for iEvent in range(num_events):           # Loop through events
    if not pythia_sig.next(): continue     # Standard pythia syntax to trigger next event generation
    for prt in pythia_sig.event:           # Loop through particles in each event
        if prt.isFinal():                  # Check if particle is final state particle and store pT, eta, phi
            stbl_sig.append([prt.pT(),prt.eta(),prt.phi()])
            
# Begin event loop. Generate event. Skip if error.
stbl_bkg = []
for iEvent in range(num_events):          # Loop through events
    if not pythia_bkg.next(): continue    # Standard pythia syntax to trigger enxt event generation
    for prt in pythia_bkg.event:          # Loop through particles in each event
        if prt.isFinal():                 # Chekc if particle is final stat particle and store pT, eta, phi
            stbl_bkg.append([prt.pT(),prt.eta(),prt.phi()])

num_particles = min([len(stbl_sig),len(stbl_bkg)])   # Determine which list is shorter
stbl_sig = np.array(stbl_sig[0:num_particles])       # Balance the dataset
stbl_bkg = np.array(stbl_bkg[0:num_particles])       # Balance the dataset

# Plot pT for signal and background
plt.figure()
plt.title("Particle p   $\mathregular{_{\tT}}$")
plt.hist(stbl_sig[:,0],bins=40,range=(0,10),histtype='step',label='signal',color='r',density=True)
plt.hist(stbl_bkg[:,0],bins=40,range=(0,10),histtype='step',label='background',color='b',density=True)
plt.yscale('log')
plt.legend()
plt.savefig("trk_pt.pdf")
plt.show()

# Plot eta for signal and background
plt.figure()
plt.title("Particle \u03B7")
plt.hist(stbl_sig[:,1],bins=30,range=(-10,10),histtype='step',label='signal',color='r',density=True)
plt.hist(stbl_bkg[:,1],bins=30,range=(-10,10),histtype='step',label='background',color='b',density=True)
plt.legend()
plt.savefig("trk_eta.pdf")
plt.show()

# Plot phi for signal and background
plt.figure()
plt.title("Particle \u03D5")
plt.hist(stbl_sig[:,2],bins=16,range=(-4,4),histtype='step',label='signal',color='r',density=True)
plt.hist(stbl_bkg[:,2],bins=16,range=(-4,4),histtype='step',label='background',color='b',density=True)
plt.legend()
plt.savefig("trk_phi.pdf")
plt.show()


# Split Dataset Into Training and Testing Samples
# It's best practice to use a training and testing dataset to verify that your model is not overfitting. Its also best practice to shuffle the dataset before training.

# Convert python lists to numpy arrays
X_sig = np.array(stbl_sig)
X_bkg = np.array(stbl_bkg)

# Generate labels sig=1 bkg=0
Y_sig = np.ones(num_particles)
Y_bkg = np.zeros(num_particles)

# Combine sig and bkg arrays
X = np.concatenate((X_sig, X_bkg),axis=0)
Y = np.concatenate((Y_sig, Y_bkg),axis=0)

# Generate random permuation of the indices
p = np.random.permutation(len(X))

# Shuffle X and Y according to randomized indices
X = X[p]
Y = Y[p]

# Split dataset into training and testing samples
# 70% train, 30% test
num_particles = len(Y)
X_train = X[0:int(0.7*num_particles)]
Y_train = Y[0:int(0.7*num_particles)].reshape((-1,1))
X_test = X[int(0.7*num_particles):]
Y_test = Y[int(0.7*num_particles):].reshape((-1,1))

# Convert numpy arrays to float32 torch tensors
X_train = torch.from_numpy(np.float32(X_train))
Y_train = torch.from_numpy(np.float32(Y_train))
X_test = torch.from_numpy(np.float32(X_test))
Y_test = torch.from_numpy(np.float32(Y_test))

# Print shapes of samples
print("X_train shape:\t", X_train.shape)
print("Y_train shape:\t", Y_train.shape)
print("X_test shape:\t", X_test.shape)
print("X_test shape:\t", Y_test.shape)


# Define the Model
# Define a class that inherits from torch.nn.Module
class NeuralNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):  # Called when object is initialized
        super(NeuralNet, self).__init__()                         # Call init function from parent class
        self.lin1 = nn.Linear(in_dim,hidden_dim)        # Define linear transformation 1
        self.lin2 = nn.Linear(hidden_dim,out_dim)       # Define linear transformation 2
    def forward(self, data):                         # Define a forward pass
        hidden_layer = F.relu(self.lin1(data))       # Transform using lin1 and use relu activation 
        output = F.sigmoid(self.lin2(hidden_layer))  # Transform using lin2 and use sigmoid activation
        return output                                # Return the models prediction

# Define the training loop
def train(model, data, epochs=20):
    X_train, y_train, X_test, y_test = data        # Unpack data

    history = {'train_loss':[],'test_loss':[]}     # Define history dictionary

    # Loop through epoches
    for e in range(epochs):
        # Train Model
        model.train()                        # Switch model to training mode
        optimizer.zero_grad()                # Reset the optimizers gradients
        outputs = model(X_train)             # Get the model prediction
        loss = loss_fn(outputs, y_train)     # Evaluate loss function
        loss.backward()                      # Backward propogation
        optimizer.step()                     # Gradient Descent

        # Validate Model
        model.eval()
        y_pred = model(X_test.to(device))    # Get model output on test data
        test_loss = loss_fn(y_pred,y_test)   # Evaluate loss on test preditions
        
        history['train_loss'].append(loss.detach().cpu().numpy())        # Append train loss to history (detach and convert to numpy array)
        history['test_loss'].append(test_loss.detach().cpu().numpy())    # Append test loss to history (detach and convert to numpy array)
        if e%1==0:
            print('Epoch:',e,'\tTrain Loss:',round(float(loss),4),'\tTest Loss:',round(float(test_loss),4))

    return history


# Check if GPU is available, if not use cpu
print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Train Model and Plot Training History

# Initialize model
model = NeuralNet(in_dim=3,hidden_dim=64,out_dim=1)    # Declare model using NeuralNet Class
model.to(device)                                       # Put model on device (cpu or gpu)
print(model)                                           # Print layers in model

# Calculate and print trainable parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters: ", pytorch_total_params,"\n")

# Declare optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam = Adaptive Moment Estimation, lr = learning rate
loss_fn = nn.BCELoss()                                 # BCE = Binary Cross Entropy, used for binary classification

#Train Model
data = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)   # Send data to device (cpu or gpu)
history = train(model, data, epochs=100)                                             # Train the model!

# Plot Training History
plt.figure()
plt.plot(history['train_loss'],label='train')
plt.plot(history['test_loss'],label='test')
plt.title("Loss")
plt.legend()
plt.savefig("History.pdf")
plt.show()

# Define traditional ROC curve
def roc(y_pred,y_true):    
    sig_eff = []
    bkg_eff = []
    
    # Iterate over thresholds and calculate sig and bkg efficiency
    for threshold in np.linspace(0,1,50):
        sig_eff.append(((y_pred[sig] > threshold).sum() / y_true[sig].shape[0]))   # Sum over sig predictions > threshold and divide by total number of true sig instances 
        bkg_eff.append(((y_pred[bkg] < threshold).sum()  / y_true[bkg].shape[0]))  # Sum over bkg predictions < threshold and divide by total number of true bkg instances 
        
    return sig_eff, bkg_eff

# Define ATLAS Style ROC curve
def ATLAS_roc(y_pred,y_true):
    sig_eff = []
    bkg_eff = []
    
    for threshold in np.linspace(0,0.55,50):
        sig_eff.append(((y_pred[sig] > threshold).sum() / y_true[sig].shape[0]))
        bkg_eff.append(1-((y_pred[bkg] < threshold).sum()  / y_true[bkg].shape[0]))
        
    bkg_rej = [1/x for x in bkg_eff]  # ATLAS inverts bkg eff and uses bkg rejection instead
    return sig_eff, bkg_rej

# Get Models predictions
y_pred = model(X_test.to(device)).detach().cpu().numpy()

# Find indices of sig and bkg labels
sig = np.where(Y_test==1)[0]
bkg = np.where(Y_test==0)[0]

# Plot Model Predictions split by sig and bkg
plt.figure()
plt.title("Predicted Output by Truth Label")
plt.hist(y_pred[sig],histtype='step',label='ttbar',color='r')
plt.axvline(x = np.mean(y_pred[sig]), color="r", linestyle="--", label="Sig Mean")
plt.hist(y_pred[bkg],histtype='step',label='QCD',color='b')
plt.axvline(x = np.mean(y_pred[bkg]), color="b", linestyle="--", label="Bkg Mean")
plt.legend()
plt.savefig("Predicted_Output.pdf")
plt.show()

# Plot Tradiation ROC Curve
plt.figure()
eff_sig, eff_bkg = roc(y_pred,Y_test.detach().cpu().numpy())
plt.title("ROC Curve")
plt.plot(eff_sig,eff_bkg,color='b',label="Trained Model")
plt.plot([1,0],'--',color='k',label="Random Model")
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Efficiency")
plt.legend()
plt.savefig("ROC.pdf")
plt.show()

# Plot ATLAS Style ROC Curve
plt.figure()
eff_sig, eff_bkg = ATLAS_roc(y_pred,Y_test.detach().cpu().numpy())
plt.title("ATLAS ROC Curve")
plt.plot(eff_sig,eff_bkg,color='b',label="Trained Model")
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Rejection")
plt.yscale('log')
plt.grid(True,which='both')
plt.xlim([0.6, 1])
plt.savefig("ATLAS_ROC.pdf")
plt.show()
