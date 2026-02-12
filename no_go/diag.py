import sys
try:
    import pytest
    print("pytest is installed")
except ImportError:
    print("pytest is NOT installed")

import os
print(f"Current CWD: {os.getcwd()}")
with open("test_diag.log", "w") as f:
    f.write("Diagnostics ran successfully.")
print("File write test complete.")
