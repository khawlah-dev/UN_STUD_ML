import nbformat
import re

nb_path = "Model_Development.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if "class_weight=\"balanced\"," in cell.source:
        cell.source = cell.source.replace("class_weight=\"balanced\",\n", "")
        cell.source = cell.source.replace("class_weight=\"balanced\",", "")
    
    if "class_weight=\"balanced\"" in cell.source:
        cell.source = cell.source.replace("class_weight=\"balanced\"", "")
        
    if "_balanced" in cell.source:
        cell.source = cell.source.replace("_balanced", "_original")

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Removed class_weight='balanced' and updated names to '_original'")
