# Pythia8 + Python3 + PyTorch2 Tutorial
This tutorial will go step-by-step on how to setup Pythia8 in Python3. Pythia will be used to generate a ttbar and HardQCD dataset. A simple PyTorch model is then trained to distinguish particles coming from ttbar vs QCD. <br>
<br>
To automate the setup, run the code:
```
git clone https://github.com/LukeV37/Pythia_PyTorch_Tutorial.git
cd Pythia_PyTorch_Tutorial
source setup.sh
```
<br>
Alternatively, the setup process can be performed manually using:
<br>

```
./download.sh
./build.sh
source virt_env.sh
# Optional jupyter notebook
./notebook.sh
```

<br>
The setup script will ask you a few questions: <br>
<ol>
<li>Would you like to download pythia source code? (Required for first time install)</li>
<li>Would you like to compile pythia? (Required for first time install)</li>
<li>Would you like to setup python virtual env? (Required for first time install)</li>
<li>Would you like to run a jupyter notebook? (Optional)</li>
</ol>
<br>
After the setup is completed, you should be able to run the code in juypter notebook if you requested so, or you can run directly in the terminal using: <br>
<br>

```
python3 Pythia_Tutorial.py
```
<br>
Make sure you forward your graphics over ssh if you are connected to a remote server. Additionally, you must have python3 and g++ installed for the setup to work - but most computers already do.
