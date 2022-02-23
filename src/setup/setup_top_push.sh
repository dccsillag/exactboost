#!/bin/sh

# Download and extract julia.

cd setup/
wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
gzip -d julia-1.5.3-linux-x86_64.tar.gz
tar -xvf julia-1.5.3-linux-x86_64.tar
rm julia-1.5.3-linux-x86_64.tar
cd julia-1.5.3

# Install python's julia package and create a sysimage with PyCall compiled in.

pip install julia
python3 -m julia.sysimage sys.so

# Install PackageCompiler and ClassificationOnTop.

cat > install_packages.jl <<- EOF
using Pkg
Pkg.add("PackageCompiler")
Pkg.add(url="https://github.com/VaclavMacha/ClassificationOnTop.jl")
EOF

./bin/julia install_packages.jl
rm install_packages.jl

# Explictly set data[2] as a BitArray in ClassificationOnTop.
# This prevents a bug when calling `solver` from python.

sed -i 's/Batch(data)/Batch((data[1], BitArray(data[2])))/g' ~/.julia/packages/ClassificationOnTop/tK2xU/src/utilities/solver.jl

# Create custom sysimage with ClassificationOnTop compiled in.

cat > create_custom_sysimage.jl <<- EOF
using PackageCompiler
using ClassificationOnTop
create_sysimage(:ClassificationOnTop, sysimage_path="custom_sysimage.so")
EOF

julia-py -J./sys.so create_custom_sysimage.jl
rm create_custom_sysimage.jl

cd ../../

