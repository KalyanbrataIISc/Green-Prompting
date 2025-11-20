# Green-Prompting

A repository for exploring green prompting techniques with Jupyter Notebook support.

## Getting Started with Codespaces

This repository is configured to work seamlessly with GitHub Codespaces and includes Jupyter Notebook support out of the box.

### Using GitHub Codespaces

1. Click the "Code" button on the repository page
2. Select "Codespaces" tab
3. Click "Create codespace on main" (or your preferred branch)
4. Wait for the container to build and install dependencies

The codespace will automatically:
- Set up a Python 3.11 environment
- Install Jupyter Notebook, JupyterLab, and essential data science packages
- Configure VS Code extensions for Python and Jupyter
- Forward port 8888 for Jupyter Notebook server

### Running Jupyter Notebook

Once the codespace is ready, you can start Jupyter in several ways:

#### Option 1: Using JupyterLab (Recommended)
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### Option 2: Using Jupyter Notebook Classic
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### Option 3: Using VS Code Jupyter Extension
Simply create or open a `.ipynb` file in VS Code, and the Jupyter extension will handle the kernel automatically.

### Installed Packages

The following packages are pre-installed:
- **Jupyter**: `jupyter`, `jupyterlab`, `notebook`, `ipykernel`, `ipywidgets`
- **Data Science**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- **Utilities**: `tqdm`

### Local Development with Dev Container

If you have Docker and VS Code with the Dev Containers extension installed, you can also run this environment locally:

1. Clone the repository
2. Open the repository in VS Code
3. When prompted, click "Reopen in Container"
4. VS Code will build the container and set up the environment

## Contributing

Feel free to add more notebooks and explore green prompting techniques!