/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

/*! \file thrust/system/cuda/execution_policy.h
 *  \brief Execution policies for Thrust's CUDA system.
 */

#include <thrust/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/cuda/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/cuda/detail/par.h>

// now get all the algorithm defintitions

// the order of the following #includes seems to matter, unfortunately

// primitives come first, in order of increasing sophistication
#include <thrust/system/cuda/detail/get_value.h>
#include <thrust/system/cuda/detail/assign_value.h>
#include <thrust/system/cuda/detail/iter_swap.h>

#include <thrust/system/cuda/detail/for_each.h>
#include <thrust/system/cuda/detail/copy.h>
#include <thrust/system/cuda/detail/reduce.h>
#include <thrust/system/cuda/detail/scan.h>
#include <thrust/system/cuda/detail/sort.h>

// these are alphabetical
#include <thrust/system/cuda/detail/adjacent_difference.h>
#include <thrust/system/cuda/detail/assign_value.h>
#include <thrust/system/cuda/detail/binary_search.h>
#include <thrust/system/cuda/detail/copy_if.h>
#include <thrust/system/cuda/detail/count.h>
#include <thrust/system/cuda/detail/equal.h>
#include <thrust/system/cuda/detail/extrema.h>
#include <thrust/system/cuda/detail/fill.h>
#include <thrust/system/cuda/detail/find.h>
#include <thrust/system/cuda/detail/gather.h>
#include <thrust/system/cuda/detail/generate.h>
#include <thrust/system/cuda/detail/inner_product.h>
#include <thrust/system/cuda/detail/iter_swap.h>
#include <thrust/system/cuda/detail/logical.h>
#include <thrust/system/cuda/detail/malloc_and_free.h>
#include <thrust/system/cuda/detail/merge.h>
#include <thrust/system/cuda/detail/mismatch.h>
#include <thrust/system/cuda/detail/partition.h>
#include <thrust/system/cuda/detail/reduce_by_key.h>
#include <thrust/system/cuda/detail/remove.h>
#include <thrust/system/cuda/detail/replace.h>
#include <thrust/system/cuda/detail/reverse.h>
#include <thrust/system/cuda/detail/scan_by_key.h>
#include <thrust/system/cuda/detail/scatter.h>
#include <thrust/system/cuda/detail/sequence.h>
#include <thrust/system/cuda/detail/set_operations.h>
#include <thrust/system/cuda/detail/sort.h>
#include <thrust/system/cuda/detail/swap_ranges.h>
#include <thrust/system/cuda/detail/tabulate.h>
#include <thrust/system/cuda/detail/transform.h>
#include <thrust/system/cuda/detail/transform_reduce.h>
#include <thrust/system/cuda/detail/transform_scan.h>
#include <thrust/system/cuda/detail/uninitialized_copy.h>
#include <thrust/system/cuda/detail/uninitialized_fill.h>
#include <thrust/system/cuda/detail/unique.h>
#include <thrust/system/cuda/detail/unique_by_key.h>


// define these entities here for the purpose of Doxygenating them
// they are actually defined elsewhere
#if 0
namespace thrust
{
namespace system
{
namespace cuda
{


/*! \addtogroup execution_policies
 *  \{
 */


/*! \p thrust::cuda::execution_policy is the base class for all Thrust parallel execution
 *  policies which are derived from Thrust's CUDA backend system.
 */
template<typename DerivedPolicy>
struct execution_policy : thrust::execution_policy<DerivedPolicy>
{};


/*! \p cuda::tag is a type representing Thrust's CUDA backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p cuda::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p cuda system.
 */
struct tag : thrust::system::cuda::execution_policy<tag> { unspecified };


/*! \p thrust::cuda::par is the parallel execution policy associated with Thrust's CUDA
 *  backend system.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags, users may
 *  directly target Thrust's CUDA backend system by providing \p thrust::cuda::par as an algorithm
 *  parameter.
 *
 *  Explicit dispatch can be useful in avoiding the introduction of data copies into containers such
 *  as \p thrust::cuda::vector.
 *
 *  The type of \p thrust::cuda::par is implementation-defined.
 *
 *  The following code snippet demonstrates how to use \p thrust::cuda::par to explicitly dispatch an
 *  invocation of \p thrust::for_each to the CUDA backend system:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/cuda/execution_policy.h>
 *  #include <cstdio>
 *
 *  struct printf_functor
 *  {
 *    __host__ __device__
 *    void operator()(int x)
 *    {
 *      printf("%d\n", x);
 *    }
 *  };
 *  ...
 *  int vec[3];
 *  vec[0] = 0; vec[1] = 1; vec[2] = 2;
 *
 *  thrust::for_each(thrust::cuda::par, vec.begin(), vec.end(), printf_functor());
 *
 *  // 0 1 2 is printed to standard output in some unspecified order
 *  \endcode
 *
 *  Explicit dispatch may also be used to direct Thrust's CUDA backend to launch CUDA kernels implementing
 *  an algorithm invocation on a particular CUDA stream. In some cases, this may achieve concurrency with the
 *  caller and other algorithms and CUDA kernels executing on a separate CUDA stream. The following code
 *  snippet demonstrates how to use the \p thrust::cuda::par execution policy to explicitly dispatch invocations
 *  of \p thrust::for_each on separate CUDA streams:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/cuda/execution_policy.h>
 *
 *  struct printf_functor
 *  {
 *    cudaStream_t s;
 *
 *    printf_functor(cudaStream_t s) : s(s) {}
 *
 *    __host__ __device__
 *    void operator()(int)
 *    {
 *      printf("Hello, world from stream %p\n", static_cast<void*>(s));
 *    }
 *  };
 *
 *  int main()
 *  {
 *    // create two CUDA streams
 *    cudaStream_t s1, s2;
 *    cudaStreamCreate(&s1);
 *    cudaStreamCreate(&s2);
 *  
 *    thrust::counting_iterator<int> iter(0);
 *  
 *    // execute for_each on two different streams
 *    thrust::for_each(thrust::cuda::par.on(s1), iter, iter + 1, printf_functor(s1));
 *    thrust::for_each(thrust::cuda::par.on(s2), iter, iter + 1, printf_functor(s2));
 *  
 *    // synchronize with both streams
 *    cudaStreamSynchronize(s1);
 *    cudaStreamSynchronize(s2);
 *  
 *    // destroy streams
 *    cudaStreamDestroy(s1);
 *    cudaStreamDestroy(s2);
 *  
 *    return 0;
 *  }
 *  \endcode
 *
 *  Even when using CUDA streams with \p thrust::cuda::par.on(), there is no guarantee of concurrency. Algorithms
 *  which return a data-dependent result or whose implementations require temporary memory allocation may
 *  cause blocking synchronization events. Moreover, it may be necessary to explicitly synchronize through
 *  \p cudaStreamSynchronize or similar before any effects induced through algorithm execution are visible to
 *  the rest of the system. Finally, it is the responsibility of the caller to own the lifetime of any CUDA
 *  streams involved.
 */
static const unspecified par;


/*! \}
 */


} // end cuda
} // end system
} // end thrust
#endif


