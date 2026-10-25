#!/usr/bin/env bash
# Provider API keys are read from the environment. Nothing is written to disk
# or passed on a command line, so `ps` never shows a credential.
set -euo pipefail
cd "$(dirname "$0")"
exec ./offpeak-router
