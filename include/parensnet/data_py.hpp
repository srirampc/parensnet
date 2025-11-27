//
// Copyright [2024]
//
#ifndef DATA_PY_HPP
#define DATA_PY_HPP

// #include <Eigen/Core>
#include <parensnet/py_types.hpp>

template <typename IndexType>
static inline IndexType py_config_size(pydict in_dict, const char* uniq_key) {
    pyiarray_t py_unq = in_dict[uniq_key].cast<pyiarray_t>();
    return IndexType(py_unq.shape(0));
}


#endif  //! PY_LOADER_HPP
