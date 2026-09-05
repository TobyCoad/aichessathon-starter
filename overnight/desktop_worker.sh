#!/bin/bash
# Compatibility wrapper: the desktop keeps running this name.
exec bash "$(dirname "$0")/worker.sh" desktop
