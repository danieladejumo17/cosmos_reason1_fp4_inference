#!/bin/bash
# =============================================================================
# Build OpenMPI with internal PMIx for Docker/vast.ai containers
# =============================================================================
#
# Problem: Ubuntu 24.04 ships OpenMPI 4.1.6 with ext3x PMIx module that
#          depends on system PMIx 5.0.1. This causes MPI_Init to hang in
#          Docker containers (orted can't initialize PMIx server).
#
# Fix:     Build OpenMPI 4.1.6 from source with --with-pmix=internal
#          which bundles a compatible PMIx version.
#
# Usage:   bash build_openmpi_docker.sh
#
# Note: setup_fp4_vast.sh calls this logic internally. This standalone script
#       exists for cases where you only need to rebuild OpenMPI without the
#       full FP4 setup (e.g. after a container image update).
#
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

OMPI_VERSION="4.1.6"
INSTALL_PREFIX="/usr/local/openmpi-${OMPI_VERSION}"
BUILD_DIR="/tmp/openmpi-build"

if [ -f "$INSTALL_PREFIX/lib/libmpi.so" ]; then
    echo -e "${GREEN}[MPI]${NC} OpenMPI ${OMPI_VERSION} already installed at ${INSTALL_PREFIX}"
    echo "       To rebuild, remove it first: rm -rf ${INSTALL_PREFIX}"
    exit 0
fi

echo -e "${GREEN}[MPI]${NC} Building OpenMPI ${OMPI_VERSION} with internal PMIx..."
echo "       This takes ~5 minutes on a multi-core machine."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download
if [ ! -f "openmpi-${OMPI_VERSION}.tar.gz" ]; then
    echo -e "${GREEN}[MPI]${NC} Downloading OpenMPI ${OMPI_VERSION}..."
    wget -q "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz"
fi

# Extract
echo -e "${GREEN}[MPI]${NC} Extracting..."
tar xf "openmpi-${OMPI_VERSION}.tar.gz"
cd "openmpi-${OMPI_VERSION}"

# Configure with internal PMIx to avoid system PMIx mismatch
echo -e "${GREEN}[MPI]${NC} Configuring (internal PMIx, libevent, hwloc)..."
./configure \
    --prefix="$INSTALL_PREFIX" \
    --with-pmix=internal \
    --with-libevent=internal \
    --with-hwloc=internal \
    --disable-mpi-fortran \
    --enable-mpi-cxx \
    --without-verbs \
    --disable-dlopen \
    --enable-static=no \
    --enable-shared=yes \
    > /tmp/openmpi-configure.log 2>&1

# Build
NCORES=$(nproc)
echo -e "${GREEN}[MPI]${NC} Building with ${NCORES} cores..."
make -j"$NCORES" > /tmp/openmpi-build.log 2>&1

# Install
echo -e "${GREEN}[MPI]${NC} Installing to ${INSTALL_PREFIX}..."
make install > /tmp/openmpi-install.log 2>&1

# Test
echo -e "${GREEN}[MPI]${NC} Testing MPI_Init..."
cat > /tmp/test_mpi_init.c << 'EOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("MPI OK: rank=%d size=%d\n", rank, size);
    MPI_Finalize();
    return 0;
}
EOF

"$INSTALL_PREFIX/bin/mpicc" /tmp/test_mpi_init.c -o /tmp/test_mpi_init 2>/dev/null
timeout 10 /tmp/test_mpi_init 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[MPI]${NC} Success! OpenMPI ${OMPI_VERSION} installed and working."
else
    echo -e "${YELLOW}[MPI]${NC} Warning: MPI test did not pass. Check logs in /tmp/openmpi-*.log"
fi

# Cleanup
rm -rf "$BUILD_DIR"
echo -e "${GREEN}[MPI]${NC} Build directory cleaned up."
echo ""
echo "To use: source activate_fp4.sh"
