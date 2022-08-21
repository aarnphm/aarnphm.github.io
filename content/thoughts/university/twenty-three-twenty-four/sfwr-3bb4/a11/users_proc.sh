#!/usr/bin/env bash
ps aux | cut -d ' ' -f1 | sort | uniq
