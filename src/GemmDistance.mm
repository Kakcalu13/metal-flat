// metal-flat/src/GemmDistance.mm — MPS-backed GEMM for flat search.

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "GemmDistance.h"

namespace mflat {

GemmDistance::GemmDistance(id<MTLDevice> device)
    : mDevice(device) {
    mReady = (device != nil) && MPSSupportsMTLDevice(device);
}

void GemmDistance::encode(id<MTLCommandBuffer> cb,
                          id<MTLBuffer> queries, int m,
                          id<MTLBuffer> db, int dbRowOffset, int n, int d,
                          id<MTLBuffer> out) {
    if (!mReady || !cb || !queries || !db || !out) return;
    if (m <= 0 || n <= 0 || d <= 0) return;

    const NSUInteger fd = sizeof(float);
    // Q is m×d, D is n×d (both row-major). transposeRight=YES makes MPS
    // treat D as Dᵀ (d×n), so result = Q·Dᵀ = m×n. The database block is
    // addressed in place via a byte offset (dbRowOffset rows in).
    const NSUInteger dbByteOffset =
        (NSUInteger)dbRowOffset * (NSUInteger)d * fd;
    MPSMatrixDescriptor* qDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:d
                                             rowBytes:(NSUInteger)d * fd
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:d
                                             rowBytes:(NSUInteger)d * fd
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* rDesc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
                                             rowBytes:(NSUInteger)n * fd
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix* qMat = [[MPSMatrix alloc] initWithBuffer:queries descriptor:qDesc];
    MPSMatrix* dMat = [[MPSMatrix alloc] initWithBuffer:db
                                                 offset:dbByteOffset
                                             descriptor:dDesc];
    MPSMatrix* rMat = [[MPSMatrix alloc] initWithBuffer:out     descriptor:rDesc];

    MPSMatrixMultiplication* mm =
        [[MPSMatrixMultiplication alloc] initWithDevice:mDevice
                                          transposeLeft:NO
                                         transposeRight:YES
                                             resultRows:m
                                          resultColumns:n
                                        interiorColumns:d
                                                  alpha:1.0
                                                   beta:0.0];
    [mm encodeToCommandBuffer:cb
                   leftMatrix:qMat
                  rightMatrix:dMat
                 resultMatrix:rMat];
}

}  // namespace mflat
