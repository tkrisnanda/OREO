#include <unittest/unittest.h>
#include <thrust/copy.h>

#include <list>
#include <iterator>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

void TestCopyFromConstIterator(void)
{
    typedef int T;

    std::vector<T> v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    std::vector<int>::const_iterator begin = v.begin();
    std::vector<int>::const_iterator end = v.end();

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    thrust::host_vector<T>::iterator h_result = thrust::copy(begin, end, h.begin());
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    thrust::device_vector<T>::iterator d_result = thrust::copy(begin, end, d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_UNITTEST(TestCopyFromConstIterator);

void TestCopyToDiscardIterator(void)
{
    typedef int T;

    thrust::host_vector<T> h_input(5,1);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> reference(5);

    // copy from host_vector
    thrust::discard_iterator<> h_result =
      thrust::copy(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

    // copy from device_vector
    thrust::discard_iterator<> d_result =
      thrust::copy(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_UNITTEST(TestCopyToDiscardIterator);

void TestCopyToDiscardIteratorZipped(void)
{
    typedef int T;

    thrust::host_vector<T> h_input(5,1);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>     h_output(5);
    thrust::device_vector<T>   d_output(5);
    thrust::discard_iterator<> reference(5);

    typedef thrust::tuple<thrust::discard_iterator<>,thrust::host_vector<T>::iterator>   Tuple1;
    typedef thrust::tuple<thrust::discard_iterator<>,thrust::device_vector<T>::iterator> Tuple2;

    typedef thrust::zip_iterator<Tuple1> ZipIterator1;
    typedef thrust::zip_iterator<Tuple2> ZipIterator2;

    // copy from host_vector
    ZipIterator1 h_result =
      thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(h_input.begin(),                 h_input.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(h_input.end(),                   h_input.end())),
                   thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), h_output.begin())));

    // copy from device_vector
    ZipIterator2 d_result =
      thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(d_input.begin(),                 d_input.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(d_input.end(),                   d_input.end())),
                   thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), d_output.begin())));

    ASSERT_EQUAL(h_output, h_input);
    ASSERT_EQUAL(d_output, d_input);
    ASSERT_EQUAL_QUIET(reference, thrust::get<0>(h_result.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(reference, thrust::get<0>(d_result.get_iterator_tuple()));
}
DECLARE_UNITTEST(TestCopyToDiscardIteratorZipped);

template <class Vector>
void TestCopyMatchingTypes(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    typename thrust::host_vector<T>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    typename thrust::device_vector<T>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyMatchingTypes);

template <class Vector>
void TestCopyMixedTypes(void)
{
    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    // copy to host_vector with different type
    thrust::host_vector<float> h(5, (float) 10);
    typename thrust::host_vector<float>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());

    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector with different type
    thrust::device_vector<float> d(5, (float) 10);
    typename thrust::device_vector<float>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyMixedTypes);


void TestCopyVectorBool(void)
{
    std::vector<bool> v(3);
    v[0] = true; v[1] = false; v[2] = true;

    thrust::host_vector<bool> h(3);
    thrust::device_vector<bool> d(3);
    
    thrust::copy(v.begin(), v.end(), h.begin());
    thrust::copy(v.begin(), v.end(), d.begin());

    ASSERT_EQUAL(h[0], true);
    ASSERT_EQUAL(h[1], false);
    ASSERT_EQUAL(h[2], true);

    ASSERT_EQUAL(d[0], true);
    ASSERT_EQUAL(d[1], false);
    ASSERT_EQUAL(d[2], true);
}
DECLARE_UNITTEST(TestCopyVectorBool);


template <class Vector>
void TestCopyListTo(void)
{
    typedef typename Vector::value_type T;

    // copy from list to Vector
    std::list<T> l;
    l.push_back(0);
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    l.push_back(4);
   
    Vector v(l.size());

    typename Vector::iterator v_result = thrust::copy(l.begin(), l.end(), v.begin());

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);
    ASSERT_EQUAL(v[4], 4);
    ASSERT_EQUAL_QUIET(v_result, v.end());

    l.clear();

    thrust::copy(v.begin(), v.end(), std::back_insert_iterator< std::list<T> >(l));

    ASSERT_EQUAL(l.size(), 5);

    typename std::list<T>::const_iterator iter = l.begin();
    ASSERT_EQUAL(*iter, 0);  iter++;
    ASSERT_EQUAL(*iter, 1);  iter++;
    ASSERT_EQUAL(*iter, 2);  iter++;
    ASSERT_EQUAL(*iter, 3);  iter++;
    ASSERT_EQUAL(*iter, 4);  iter++;
}
DECLARE_VECTOR_UNITTEST(TestCopyListTo);


template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x) { return (static_cast<unsigned int>(x) & 1) == 0; }
};

template<typename T>
struct is_true
{
    __host__ __device__
    bool operator()(T x) { return x ? true : false; }
};

template<typename T>
struct mod_3
{
    __host__ __device__
    unsigned int operator()(T x) { return static_cast<unsigned int>(x) % 3; }
};
    
    

template <class Vector>
void TestCopyIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    Vector dest(3);

    typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), dest.begin(), is_even<T>());

    ASSERT_EQUAL(0, dest[0]);
    ASSERT_EQUAL(2, dest[1]);
    ASSERT_EQUAL(4, dest[2]);
    ASSERT_EQUAL_QUIET(dest.end(), dest_end);
}
DECLARE_VECTOR_UNITTEST(TestCopyIfSimple);


template <typename T>
void TestCopyIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_new_end;
    typename thrust::device_vector<T>::iterator d_new_end;

    // test with Predicate that returns a bool
    {
        thrust::host_vector<T>   h_result(n);
        thrust::device_vector<T> d_result(n);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQUAL(h_result, d_result);
    }
    
    // test with Predicate that returns a non-bool
    {
        thrust::host_vector<T>   h_result(n);
        thrust::device_vector<T> d_result(n);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQUAL(h_result, d_result);
    }
}
DECLARE_VARIABLE_UNITTEST(TestCopyIf);


template <class Vector>
void TestCopyIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    Vector s(5);
    s[0] = 1; s[1] = 1; s[2] = 0; s[3] = 1; s[4] = 0;

    Vector dest(3);

    typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), s.begin(), dest.begin(), is_true<T>());

    ASSERT_EQUAL(0, dest[0]);
    ASSERT_EQUAL(1, dest[1]);
    ASSERT_EQUAL(3, dest[2]);
    ASSERT_EQUAL_QUIET(dest.end(), dest_end);
}
DECLARE_VECTOR_UNITTEST(TestCopyIfStencilSimple);


template <typename T>
void TestCopyIfStencil(const size_t n)
{
    thrust::host_vector<T>   h_data(n); thrust::sequence(h_data.begin(), h_data.end());
    thrust::device_vector<T> d_data(n); thrust::sequence(d_data.begin(), d_data.end()); 

    thrust::host_vector<T>   h_stencil = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_stencil = unittest::random_integers<T>(n);

    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);

    typename thrust::host_vector<T>::iterator   h_new_end;
    typename thrust::device_vector<T>::iterator d_new_end;

    // test with Predicate that returns a bool
    {
        thrust::host_vector<T>   h_result(n);
        thrust::device_vector<T> d_result(n);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQUAL(h_result, d_result);
    }
    
    // test with Predicate that returns a non-bool
    {
        thrust::host_vector<T>   h_result(n);
        thrust::device_vector<T> d_result(n);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQUAL(h_result, d_result);
    }
}
DECLARE_VARIABLE_UNITTEST(TestCopyIfStencil);

template <typename Vector>
void TestCopyCountingIterator(void)
{
    typedef typename Vector::value_type T;

    thrust::counting_iterator<T> iter(1);

    Vector vec(4);

    thrust::copy(iter, iter + 4, vec.begin());

    ASSERT_EQUAL(vec[0], 1);
    ASSERT_EQUAL(vec[1], 2);
    ASSERT_EQUAL(vec[2], 3);
    ASSERT_EQUAL(vec[3], 4);
}
DECLARE_VECTOR_UNITTEST(TestCopyCountingIterator);

template <typename Vector>
void TestCopyZipIterator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3); v1[0] = 1; v1[1] = 2; v1[2] = 3;
    Vector v2(3); v2[0] = 4; v2[1] = 5; v2[2] = 6; 
    Vector v3(3, T(0));
    Vector v4(3, T(0));

    thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(),v2.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(v1.end(),v2.end())),
                 thrust::make_zip_iterator(thrust::make_tuple(v3.begin(),v4.begin())));

    ASSERT_EQUAL(v1, v3);
    ASSERT_EQUAL(v2, v4);
};
DECLARE_VECTOR_UNITTEST(TestCopyZipIterator);

template <typename Vector>
void TestCopyConstantIteratorToZipIterator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3,T(0));
    Vector v2(3,T(0));

    thrust::copy(thrust::make_constant_iterator(thrust::tuple<T,T>(4,7)),
                 thrust::make_constant_iterator(thrust::tuple<T,T>(4,7)) + v1.size(),
                 thrust::make_zip_iterator(thrust::make_tuple(v1.begin(),v2.begin())));

    ASSERT_EQUAL(v1[0], 4);
    ASSERT_EQUAL(v1[1], 4);
    ASSERT_EQUAL(v1[2], 4);
    ASSERT_EQUAL(v2[0], 7);
    ASSERT_EQUAL(v2[1], 7);
    ASSERT_EQUAL(v2[2], 7);
};
DECLARE_VECTOR_UNITTEST(TestCopyConstantIteratorToZipIterator);

template<typename InputIterator, typename OutputIterator>
OutputIterator copy(my_system &system, InputIterator, InputIterator, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestCopyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy(sys,
                 vec.begin(),
                 vec.end(),
                 vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyDispatchExplicit);


template<typename InputIterator, typename OutputIterator>
OutputIterator copy(my_tag, InputIterator, InputIterator, OutputIterator result)
{
    *result = 13;
    return result;
}

void TestCopyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::copy(thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.end()),
                 thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyDispatchImplicit);


template<typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_system &system, InputIterator, InputIterator, OutputIterator result, Predicate pred)
{
    system.validate_dispatch();
    return result;
}

void TestCopyIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy_if(sys,
                    vec.begin(),
                    vec.end(),
                    vec.begin(),
                    0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyIfDispatchExplicit);


template<typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_tag, InputIterator, InputIterator, OutputIterator result, Predicate pred)
{
    *result = 13;
    return result;
}

void TestCopyIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::copy_if(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()),
                    thrust::retag<my_tag>(vec.begin()),
                    0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyIfDispatchImplicit);


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_system &system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate pred)
{
    system.validate_dispatch();
    return result;
}

void TestCopyIfStencilDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy_if(sys,
                    vec.begin(),
                    vec.end(),
                    vec.begin(),
                    vec.begin(),
                    0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyIfStencilDispatchExplicit);


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate pred)
{
    *result = 13;
    return result;
}

void TestCopyIfStencilDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::copy_if(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyIfStencilDispatchImplicit);

