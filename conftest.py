import sys
import os

# Add agent-python directory to sys.path so tests can import modules directly
root_dir = os.path.dirname(os.path.abspath(__file__))
agent_python_dir = os.path.join(root_dir, "agent-python")
if agent_python_dir not in sys.path:
    sys.path.insert(0, agent_python_dir)
