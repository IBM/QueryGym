## AAAI Demo 

Before running the demo, make sure you have completed the installation steps mentioned in the [main README](../../README.md#installation).

1. Activate Your Conda Environment

Ensure you are in the correct environment created during installation:

```
conda activate qgym_env
```

Replace qgym_env with the name of your environment if you used a different one.


2. Register the Kernel for Jupyter

Add your environment as a Jupyter kernel so notebooks can use it:

```
python -m ipykernel install --user --name=aaai_demo --display-name "Python (aaai_demo)"
```

--name=aaai_demo is the internal kernel name.
--display-name "Python (aaai_demo)" is what you’ll see in Jupyter Notebook or Voila.


3. Clear Notebook Metadata

Remove unnecessary metadata from the demo notebook to avoid conflicts:

```
python clear_nb_metadata.py
```

This script ensures the notebook runs cleanly without environment-specific metadata.


4. Launch the Demo with Voila

Start the interactive demo using Voila, which turns Jupyter notebooks into web apps:

```
voila QueryGym_Demo.ipynb --kernel=aaai_demo
```

QueryGym_Demo.ipynb is the demo notebook.
--kernel=aaai_demo ensures it uses the correct environment.
