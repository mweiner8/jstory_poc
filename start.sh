#!/bin/bash

DB_VERSION_FILE="db_version.txt"
CURRENT_VERSION_FILE="chroma_db/.db_version"

# Read the desired DB version
if [ -f "$DB_VERSION_FILE" ]; then
    DESIRED_VERSION=$(cat $DB_VERSION_FILE)
else
    DESIRED_VERSION="0"
fi

# Read the current DB version
if [ -f "$CURRENT_VERSION_FILE" ]; then
    CURRENT_VERSION=$(cat $CURRENT_VERSION_FILE)
else
    CURRENT_VERSION="0"
fi

# Check if we need to rebuild
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ] || [ "$DESIRED_VERSION" != "$CURRENT_VERSION" ]; then
    echo "Creating/rebuilding vector database (version $DESIRED_VERSION)..."
    rm -rf chroma_db
    python create_vector_db.py

    # Save the version
    mkdir -p chroma_db
    echo "$DESIRED_VERSION" > "$CURRENT_VERSION_FILE"
else
    echo "Vector database up to date (version $CURRENT_VERSION). Skipping creation."
fi

# Start the Streamlit app
streamlit run app.py --server.port="$PORT" --server.address=0.0.0.0 --server.headless=true