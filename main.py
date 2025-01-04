import subprocess
import sys

def run_script(script_name):
    """
    Runs a Python script and handles any errors.
    """
    try:
        print(f"Running {script_name}...")
        subprocess.run(["python", f"src/{script_name}"], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}\n")
        sys.exit(1)

def main():
    """
    Main function to run data preprocessing and pipline building scripts.
    """
    print("Starting the workflow...\n")

    # Step 1: Run data preprocessing
    run_script("data.py")

    # Step 2: Build the pipeline
    run_script("build.py")

    print("All steps completed successfully!")

if __name__ == "__main__":
    main()