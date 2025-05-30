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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/alignment.hpp>
#include <cstddef>
#include <new>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<typename T>
  class uninitialized
{
  private:
    typename bulk::detail::aligned_storage<
      sizeof(T),
      bulk::detail::alignment_of<T>::value
    >::type storage;

    __host__ __device__ __thrust_forceinline__
    const T* ptr() const
    {
      const void *result = storage.data;
      return reinterpret_cast<const T*>(result);
    }

    __host__ __device__ __thrust_forceinline__
    T* ptr()
    {
      void *result = storage.data;
      return reinterpret_cast<T*>(result);
    }

  public:
    // copy assignment
    __host__ __device__ __thrust_forceinline__
    uninitialized<T> &operator=(const T &other)
    {
      T& self = *this;
      self = other;
      return *this;
    }

    __host__ __device__ __thrust_forceinline__
    T& get()
    {
      return *ptr();
    }

    __host__ __device__ __thrust_forceinline__
    const T& get() const
    {
      return *ptr();
    }

    __host__ __device__ __thrust_forceinline__
    operator T& ()
    {
      return get();
    }

    __host__ __device__ __thrust_forceinline__
    operator const T&() const
    {
      return get();
    }

    __bulk_exec_check_disable__
    __host__ __device__ __thrust_forceinline__
    void construct()
    {
      ::new(ptr()) T();
    }

    __bulk_exec_check_disable__
    template<typename Arg>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg &a)
    {
      ::new(ptr()) T(a);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2)
    {
      ::new(ptr()) T(a1,a2);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3)
    {
      ::new(ptr()) T(a1,a2,a3);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4)
    {
      ::new(ptr()) T(a1,a2,a3,a4);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7,a8);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7,a8,a9);
    }

    __bulk_exec_check_disable__
    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
    __host__ __device__ __thrust_forceinline__
    void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9, const Arg10 &a10)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10);
    }

    __bulk_exec_check_disable__
    __host__ __device__ __thrust_forceinline__
    void destroy()
    {
      T& self = *this;
      self.~T();
    }
};


template<typename T, std::size_t N>
  class uninitialized_array
{
  public:
    typedef T             value_type; 
    typedef T&            reference;
    typedef const T&      const_reference;
    typedef T*            pointer;
    typedef const T*      const_pointer;
    typedef pointer       iterator;
    typedef const_pointer const_iterator;
    typedef std::size_t   size_type;

    __thrust_forceinline__ __host__ __device__
    iterator begin()
    {
      return data();
    }

    __thrust_forceinline__ __host__ __device__
    const_iterator begin() const
    {
      return data();
    }

    __thrust_forceinline__ __host__ __device__
    iterator end()
    {
      return begin() + size();
    }

    __thrust_forceinline__ __host__ __device__
    const_iterator end() const
    {
      return begin() + size();
    }

    __thrust_forceinline__ __host__ __device__
    const_iterator cbegin() const
    {
      return begin();
    }

    __thrust_forceinline__ __host__ __device__
    const_iterator cend() const
    {
      return end();
    }

    __thrust_forceinline__ __host__ __device__
    size_type size() const
    {
      return N;
    }

    __thrust_forceinline__ __host__ __device__
    bool empty() const
    {
      return false;
    }

    __thrust_forceinline__ __host__ __device__
    T* data()
    {
      return impl.get();
    }

    __thrust_forceinline__ __host__ __device__
    const T* data() const
    {
      return impl.get();
    }

    // element access
    __thrust_forceinline__ __host__ __device__
    reference operator[](size_type n)
    {
      return data()[n];
    }

    __thrust_forceinline__ __host__ __device__
    const_reference operator[](size_type n) const
    {
      return data()[n];
    }

    __thrust_forceinline__ __host__ __device__
    reference front()
    {
      return *data();
    }

    __thrust_forceinline__ __host__ __device__
    const_reference front() const
    {
      return *data();
    }

    __thrust_forceinline__ __host__ __device__
    reference back()
    {
      return data()[size() - size_type(1)];
    }

    __thrust_forceinline__ __host__ __device__
    const_reference back() const
    {
      return data()[size() - size_type(1)];
    }

  private:
    uninitialized<T[N]> impl;
};


} // end bulk
BULK_NAMESPACE_SUFFIX

