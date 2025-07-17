## 📦 Installation Instructions

This project is built using Python and Jupyter Notebook. To run it successfully, you'll need to install several libraries used for data analysis, machine learning, and interactive visualization.

### ✅ Recommended: Set up a virtual environment

```         
python -m venv venv source venv/bin/activate \# or venv\Scripts\activate on Windows

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 1: Install Jupyter Notebook

If you haven't already, install Jupyter Notebook:

```         
pip install notebook
```

### Step 2: Install Required Libraries

Run the following command in your terminal or command prompt to install all required packages:

```         
pip install numpy pandas scikit-learn ipywidgets seaborn matplotlib
```

### Step 3: Enable Jupyter Widgets Extension (for UI controls)

To ensure ipywidgets work properly in the notebook, run:

```         
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

### Step 4: Launch the notebook

You can start the notebook server from the command line by runing:

```         
jupyter notebook
```

It will then open your default web browser to this URL. When the notebook opens in your browser, you will see the Notebook Dashboard, which will show a list of the notebooks, files, and subdirectories in the directory where the notebook server was started.

Alternatively, you can run:

```         
jupyter execute KNN.ipynb
```
