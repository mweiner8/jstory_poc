#!/bin/bash

# Just start Streamlit - database is already committed
streamlit run app.py --server.port="$PORT" --server.address=0.0.0.0 --server.headless=true