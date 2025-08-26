#!/bin/bash

# Script to download artifacts with checksum verification
# Usage: ./download_artifacts.sh <artifact_url> <checksum> <artifact_path>
# 
# Arguments:
#   artifact_url: URL to download the file from
#   checksum: Checksum in format "algorithm::hash" (e.g., "md5::8d4fb7e6c68d591d4c3dfef9ec88bf0d")
#   artifact_path: Local path where the downloaded file should be saved
#
# Example:
#   ./download_artifacts.sh "https://example.com/file.tar.gz" "md5::8d4fb7e6c68d591d4c3dfef9ec88bf0d" "./downloads/file.tar.gz"

set -euo pipefail

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <artifact_url> <checksum> <artifact_path>"
    echo ""
    echo "Arguments:"
    echo "  artifact_url: URL to download the file from"
    echo "  checksum: Checksum in format 'algorithm::hash' (e.g., 'md5::8d4fb7e6c68d591d4c3dfef9ec88bf0d')"
    echo "  artifact_path: Local path where the downloaded file should be saved"
    echo ""
    echo "Example:"
    echo "  $0 'https://example.com/file.tar.gz' 'md5::8d4fb7e6c68d591d4c3dfef9ec88bf0d' './downloads/file.tar.gz'"
    exit 1
fi

# Parse arguments
ARTIFACT_URL="$1"
CHECKSUM="$2"
ARTIFACT_PATH="$3"

# Parse checksum format (algorithm::hash)
if [[ ! "$CHECKSUM" =~ ^([a-zA-Z0-9]+)::([a-fA-F0-9]+)$ ]]; then
    echo "Error: Invalid checksum format. Expected format: 'algorithm::hash'"
    echo "Examples: 'md5::8d4fb7e6c68d591d4c3dfef9ec88bf0d', 'sha256::abc123...', 'sha1::def456...'"
    exit 1
fi

CHECKSUM_ALGORITHM="${BASH_REMATCH[1]}"
EXPECTED_HASH="${BASH_REMATCH[2]}"

# Convert algorithm name to lowercase for consistency
CHECKSUM_ALGORITHM=$(echo "$CHECKSUM_ALGORITHM" | tr '[:upper:]' '[:lower:]')

# Validate supported checksum algorithms
case "$CHECKSUM_ALGORITHM" in
    md5|sha1|sha256|sha512)
        ;;
    *)
        echo "Error: Unsupported checksum algorithm '$CHECKSUM_ALGORITHM'"
        echo "Supported algorithms: md5, sha1, sha256, sha512"
        exit 1
        ;;
esac

# Check if required tools are available
for tool in curl "${CHECKSUM_ALGORITHM}sum"; do
    if ! command -v "$tool" &> /dev/null; then
        echo "Error: Required tool '$tool' is not installed or not in PATH"
        exit 1
    fi
done

echo "Downloading artifact from: $ARTIFACT_URL"
echo "Expected $CHECKSUM_ALGORITHM checksum: $EXPECTED_HASH"
echo "Saving to: $ARTIFACT_PATH"

# Create directory for artifact if it doesn't exist
ARTIFACT_DIR=$(dirname "$ARTIFACT_PATH")
if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "Creating directory: $ARTIFACT_DIR"
    mkdir -p "$ARTIFACT_DIR"
fi

# Download the file
echo "Starting download..."
if ! curl -L -o "$ARTIFACT_PATH" "$ARTIFACT_URL"; then
    echo "Error: Failed to download file from $ARTIFACT_URL"
    exit 1
fi

echo "Download completed successfully."

# Verify checksum
echo "Verifying checksum..."
ACTUAL_HASH=$(${CHECKSUM_ALGORITHM}sum "$ARTIFACT_PATH" | cut -d' ' -f1)

# Convert both hashes to lowercase for comparison
EXPECTED_HASH_LOWER=$(echo "$EXPECTED_HASH" | tr '[:upper:]' '[:lower:]')
ACTUAL_HASH_LOWER=$(echo "$ACTUAL_HASH" | tr '[:upper:]' '[:lower:]')

if [ "$EXPECTED_HASH_LOWER" = "$ACTUAL_HASH_LOWER" ]; then
    echo "Checksum verification successful!"
    echo "File saved to: $ARTIFACT_PATH"
    echo "Verified $CHECKSUM_ALGORITHM checksum: $ACTUAL_HASH"
    
    # Check if file is a .gz file and extract it
    if [[ "$ARTIFACT_PATH" == *.gz ]]; then
        echo "Detected .gz file, extracting..."
        
        # Check if gunzip is available
        if ! command -v gunzip &> /dev/null; then
            echo "Warning: gunzip is not available. File remains compressed."
        else
            # Extract the file
            if gunzip -dk "$ARTIFACT_PATH"; then
                # Get the extracted filename (remove .gz extension)
                EXTRACTED_FILE="${ARTIFACT_PATH%.gz}"
                echo "Successfully extracted to: $EXTRACTED_FILE"
            else
                echo "Warning: Failed to extract $ARTIFACT_PATH. File remains compressed."
            fi
        fi
    fi
else
    echo "Error: Checksum verification failed!"
    echo "Expected: $EXPECTED_HASH_LOWER"
    echo "Actual:   $ACTUAL_HASH_LOWER"
    echo "Removing downloaded file..."
    rm -f "$ARTIFACT_PATH"
    exit 1
fi

echo "Artifact download and verification completed successfully."
