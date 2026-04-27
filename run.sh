#!/bin/bash

pushd frontend
npm run build:web
popd

uv run run_server.py