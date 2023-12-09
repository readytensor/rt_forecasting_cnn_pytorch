#!/bin/bash
# fix_line_endings.sh - Script to convert line endings of Python scripts from Windows 
# to Unix format

# This script uses the 'dos2unix' utility to convert line endings in Python script
# files (.py) from Windows-style (CRLF) to Unix-style (LF).
# It is intended to fix line ending issues that can occur when scripts created on
# Windows machines are used in Unix-like environments.

# Instructions:
# 1. Place this script in the root directory of your repository (directory which#
#    contains the `src` directory).
# 2. Ensure that the 'dos2unix' utility is installed in the Docker container (it is 
#    used to perform the line ending conversion). You can install it by adding 
#    'RUN apt-get update && apt-get install -y dos2unix' to your Dockerfile.
# 3. Make this script executable by running 'chmod +x fix_line_endings.sh'.
# 4. Run this script with either a file or a directory as an argument, it will 
#    automatically convert the line endings of the passed Python script file or all
#    Python script files within the passed directory.

# Check if the argument is passed
if [ "$#" -ne 1 ]; then
    echo "You need to pass a directory or a file as an argument."
    exit 1
fi

# Get the passed argument
TARGET_PATH="$1"

# Check if the passed argument is a directory or a file and run the dos2unix command
# accordingly
if [ -d "$TARGET_PATH" ]; then
    # If a directory is passed
    find "$TARGET_PATH" -type f -name '*.py' -exec dos2unix {} \;
elif [ -f "$TARGET_PATH" ]; then
    # If a file is passed
    dos2unix "$TARGET_PATH"
else
    echo "The passed argument is not a valid directory or file."
    exit 1
fi

# Note: This script assumes that the 'dos2unix' utility is available in the
#       system path.
# If you encounter any issues, ensure that 'dos2unix' is properly installed
# and accessible.

# End of script
