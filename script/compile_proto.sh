#!/usr/bin/env bash
    
# move to project root directory
cd "$(dirname $0)/.." || { echo "Failed to change dicrectory to project root"; exit 1; }

if [ ! -d .venv ]; then
    echo "You need to create a virtual environment first."
    return 1
fi

source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
    
python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=./src/specedge_grpc \
    --grpc_python_out=./src/specedge_grpc \
    --pyi_out=./src/specedge_grpc \
    specedge.proto

# grpc_tools emits a bare `import specedge_pb2`, which only resolves if
# src/specedge_grpc is itself on sys.path. Rewrite it to a package-relative
# import so `from specedge_grpc import specedge_pb2_grpc` works from anywhere.
sed -i.bak 's/^import specedge_pb2 as specedge__pb2$/from . import specedge_pb2 as specedge__pb2/' \
    ./src/specedge_grpc/specedge_pb2_grpc.py
rm -f ./src/specedge_grpc/specedge_pb2_grpc.py.bak
