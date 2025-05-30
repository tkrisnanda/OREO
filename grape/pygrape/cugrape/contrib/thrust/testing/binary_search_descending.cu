#include <unittest/unittest.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>

//////////////////////
// Scalar Functions //
//////////////////////

template <class Vector>
void TestScalarLowerBoundDescendingSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), 0, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), 1, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), 2, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), 3, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), 4, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), 5, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), 6, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::lower_bound(vec.begin(), vec.end(), 7, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), 8, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), 9, thrust::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarLowerBoundDescendingSimple);


template <class Vector>
void TestScalarUpperBoundDescendingSimple(void)
{
    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL_QUIET(vec.begin() + 5, thrust::upper_bound(vec.begin(), vec.end(), 0, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), 1, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), 2, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), 3, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), 4, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), 5, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), 6, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), 7, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::upper_bound(vec.begin(), vec.end(), 8, thrust::greater<int>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::upper_bound(vec.begin(), vec.end(), 9, thrust::greater<int>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarUpperBoundDescendingSimple);


template <class Vector>
void TestScalarBinarySearchDescendingSimple(void)
{
    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), 0, thrust::greater<int>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), 1, thrust::greater<int>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), 2, thrust::greater<int>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), 3, thrust::greater<int>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), 4, thrust::greater<int>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), 5, thrust::greater<int>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), 6, thrust::greater<int>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), 7, thrust::greater<int>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), 8, thrust::greater<int>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), 9, thrust::greater<int>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarBinarySearchDescendingSimple);


template <class Vector>
void TestScalarEqualRangeDescendingSimple(void)
{
    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), 0, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), 1, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), 2, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), 3, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), 4, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), 5, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), 6, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::equal_range(vec.begin(), vec.end(), 7, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), 8, thrust::greater<int>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), 9, thrust::greater<int>()).first);
    
    ASSERT_EQUAL_QUIET(vec.begin() + 5, thrust::equal_range(vec.begin(), vec.end(), 0, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), 1, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), 2, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), 3, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), 4, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), 5, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), 6, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), 7, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::equal_range(vec.begin(), vec.end(), 8, thrust::greater<int>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), 9, thrust::greater<int>()).second);
}
DECLARE_VECTOR_UNITTEST(TestScalarEqualRangeDescendingSimple);

