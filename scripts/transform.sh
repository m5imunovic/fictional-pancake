#!/bin/bash
PROJECT_ROOT=./ python src/transform.py graph=multidigraph
PROJECT_ROOT=./ python src/transform.py graph=digraph
