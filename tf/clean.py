#!/usr/bin/env python
import sys


def clean(inp):
    for line in inp:
        line = line.strip()
        if len(line) > 5:
            sys.stdout.write(line + '\n')


if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        clean(f)
else:
    clean(sys.stdin)
