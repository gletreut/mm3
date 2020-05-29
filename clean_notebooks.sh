#!/bin/bash
# this is based on the jupyter documentation page: https://mg.readthedocs.io/git-jupyter.html
# the following commands removes outputs from versioned IPython notebooks.


# remove outputs in notebook in current branch
git filter-branch --tree-filter "python3 -m nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb **/*.ipynb || true"

# remove dead references
#git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
#git reflog expire --expire=now --all
#git gc --prune=now --aggressive
