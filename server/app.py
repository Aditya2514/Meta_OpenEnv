import os
import uvicorn
import sys

# Ensure our local src directory is visible to resolve models
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src", "envs")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from assignment_planner.server.app import app

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))

if __name__ == "__main__":
    main()
