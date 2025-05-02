#!/usr/bin/env bash

uvicorn --host 0.0.0.0 --port 6380 --workers 1 src.server:APP --root-path /