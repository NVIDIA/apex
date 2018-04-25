#!/bin/bash
python setup.py clean
yes | pip uninstall apex
python setup.py install
# pip install ../apex
