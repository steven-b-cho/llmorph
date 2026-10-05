#Tests if both the virtual environment and the project are properly loaded
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data

print("--------------")
print("Python interpreter used: ", sys.executable)
print("--------------")
print("Data module loaded: ", data)
print("--------------")
print("PyTorch version: ", torch.__version__)
print("--------------")
print("Your virtual environment and LLMorph are properly set up!")