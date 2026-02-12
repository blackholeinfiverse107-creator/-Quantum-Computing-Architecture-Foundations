import pytest
import sys

if __name__ == "__main__":
    with open("test_results.log", "w") as f:
        f.write("Running tests...\n")
        class Tie:
            def write(self, data):
                f.write(data)
                sys.__stdout__.write(data)
            def flush(self):
                f.flush()
                sys.__stdout__.flush()
        
        # Redirect stdout/stderr
        sys.stdout = Tie()
        sys.stderr = Tie()
        
        retcode = pytest.main(["tests/test_adversarial.py", "-v"])
        
        if retcode == 0:
            print("\nTESTS PASSED")
        else:
            print(f"\nTESTS FAILED with code {retcode}")

    sys.exit(retcode)
