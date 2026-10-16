import nbformat  # Library for reading and writing Jupyter Notebook files

# Path to the notebook that needs metadata cleanup
nb_path = "QueryGym_Demo.ipynb"

# Read the notebook content as version 4 (current Jupyter format)
nb = nbformat.read(open(nb_path), as_version=4)

# Remove kernel and language metadata to make the notebook environment-agnostic
# This prevents issues when sharing or running the notebook in different environments
nb['metadata'].pop('kernelspec', None)       # Remove kernel specification
nb['metadata'].pop('language_info', None)    # Remove language info (e.g., Python version)

# Write the cleaned notebook back to the same file
nbformat.write(nb, open(nb_path, "w"))

# Print confirmation message
print(f"Cleaned metadata for {nb_path}")
