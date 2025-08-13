#!/usr/bin/env python3
import subprocess
import sys
import signal
import os

def run_with_log_monitoring(command, log_file="log.txt"):
    """Run a command, log output to file, and stop on 'with error: Broken pipe'"""

    # Open log file for writing
    with open(log_file, 'w') as log:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )

        try:
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                # Write to log file
                log.write(line)
                log.flush()  # Ensure immediate write

                # Also print to console (optional)
                print(line, end='')

                # Check for the error condition
                if "with error: Broken pipe" in line:
                    print(f"\n🛑 Detected 'with error: Broken pipe' - stopping logging")
                    # Terminate the process
                    process.terminate()
                    break

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            process.terminate()

        # Wait for process to complete
        process.wait()
        return process.returncode

if __name__ == "__main__":
    # Example usage - replace with your actual command
    command = ["python", "smollm/text/data/smoltalk/magpie_ultra_v1/pipeline_pt_v2_test.py"]
    exit_code = run_with_log_monitoring(command)
    print(f"Process exited with code: {exit_code}")