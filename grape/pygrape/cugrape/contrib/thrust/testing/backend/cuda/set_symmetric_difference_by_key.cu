#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename Iterator7>
__global__
void set_symmetric_difference_by_key_kernel(ExecutionPolicy exec,
                                            Iterator1 keys_first1, Iterator1 keys_last1,
                                            Iterator2 keys_first2, Iterator2 keys_last2,
                                            Iterator3 values_first1,
                                            Iterator4 values_first2,
                                            Iterator5 keys_result,
                                            Iterator6 values_result,
                                            Iterator7 result)
{
  *result = thrust::set_symmetric_difference_by_key(exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
}


template<typename ExecutionPolicy>
void TestSetSymmetricDifferenceByKeyDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::iterator Iterator;

  Vector a_key(4), b_key(5);
  Vector a_val(4), b_val(5);

  a_key[0] = 0; a_key[1] = 2; a_key[2] = 4; a_key[3] = 6;
  a_val[0] = 0; a_val[1] = 0; a_val[2] = 0; a_val[3] = 0;

  b_key[0] = 0; b_key[1] = 3; b_key[2] = 3; b_key[3] = 4; b_key[4] = 7;
  b_val[0] = 1; b_val[1] = 1; b_val[2] = 1; b_val[3] = 1; b_val[4] = 1;

  Vector ref_key(5), ref_val(5);
  ref_key[0] = 2; ref_key[1] = 3; ref_key[2] = 3; ref_key[3] = 6; ref_key[4] = 7;
  ref_val[0] = 0; ref_val[1] = 1; ref_val[2] = 1; ref_val[3] = 0; ref_val[4] = 1;

  Vector result_key(5), result_val(5);

  typedef thrust::pair<Iterator,Iterator> iter_pair;
  thrust::device_vector<iter_pair> end_vec(1);

  set_symmetric_difference_by_key_kernel<<<1,1>>>(exec,
                                                  a_key.begin(), a_key.end(),
                                                  b_key.begin(), b_key.end(),
                                                  a_val.begin(),
                                                  b_val.begin(),
                                                  result_key.begin(),
                                                  result_val.begin(),
                                                  end_vec.begin());
  iter_pair end = end_vec[0];

  ASSERT_EQUAL_QUIET(result_key.end(), end.first);
  ASSERT_EQUAL_QUIET(result_val.end(), end.second);
  ASSERT_EQUAL(ref_key, result_key);
  ASSERT_EQUAL(ref_val, result_val);
}


void TestSetSymmetricDifferenceByKeyDeviceSeq()
{
  TestSetSymmetricDifferenceByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetSymmetricDifferenceByKeyDeviceSeq);


void TestSetSymmetricDifferenceByKeyDeviceDevice()
{
  TestSetSymmetricDifferenceByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetSymmetricDifferenceByKeyDeviceDevice);


void TestSetSymmetricDifferenceByKeyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a_key(4), b_key(5);
  Vector a_val(4), b_val(5);

  a_key[0] = 0; a_key[1] = 2; a_key[2] = 4; a_key[3] = 6;
  a_val[0] = 0; a_val[1] = 0; a_val[2] = 0; a_val[3] = 0;

  b_key[0] = 0; b_key[1] = 3; b_key[2] = 3; b_key[3] = 4; b_key[4] = 7;
  b_val[0] = 1; b_val[1] = 1; b_val[2] = 1; b_val[3] = 1; b_val[4] = 1;

  Vector ref_key(5), ref_val(5);
  ref_key[0] = 2; ref_key[1] = 3; ref_key[2] = 3; ref_key[3] = 6; ref_key[4] = 7;
  ref_val[0] = 0; ref_val[1] = 1; ref_val[2] = 1; ref_val[3] = 0; ref_val[4] = 1;

  Vector result_key(5), result_val(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::pair<Iterator,Iterator> end =
    thrust::set_symmetric_difference_by_key(thrust::cuda::par.on(s),
                                            a_key.begin(), a_key.end(),
                                            b_key.begin(), b_key.end(),
                                            a_val.begin(),
                                            b_val.begin(),
                                            result_key.begin(),
                                            result_val.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(result_key.end(), end.first);
  ASSERT_EQUAL_QUIET(result_val.end(), end.second);
  ASSERT_EQUAL(ref_key, result_key);
  ASSERT_EQUAL(ref_val, result_val);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSetSymmetricDifferenceByKeyCudaStreams);

