
#include "mpi.h"
#include <cassert>
#include <functional>
#include <numeric>

#include <mxx/comm.hpp>
#include <mxx/collective.hpp>
#include <prettyprint.hpp>

#include <parensnet/py_types.hpp>
#include <parensnet/data_py.hpp>
#include <parensnet/utils.hpp>
#ifdef USE_PARALLEL_HDF5
#include <parensnet/mpi_hdf.hpp>
#endif

void print_dict(pydict in_dict) {
    const char* dct_keys[]{"data", "index", "shape",  "npairs", "nobs", "nvars"};
    const char* float_array_keys[]{"data"};
    const char* int_array_keys[]{"index"};
    for (const char* pkey : float_array_keys) {
        pyfarray_t px = in_dict[pkey].cast<pyfarray_t>();
        pybind11::print(pkey, px.ndim(), px.shape(0),
                        px.ndim() > 1 ? px.shape(1) : 0);
    }
    for (const char* pkey : int_array_keys) {
        pyiarray_t px = in_dict[pkey].cast<pyiarray_t>();
        pybind11::print(pkey, px.ndim(), px.size(), px.shape(0),
                        px.ndim() > 1 ? px.shape(1) : 0);
    }
}


int save_puc(pydict data_dict, const char *file_name) {
    timer run_timer;
    //print_dict(data_dict);
    pyfarray_t px_data = data_dict["data"].cast<pyfarray_t>();
    pyiarray_t px_index = data_dict["index"].cast<pyiarray_t>();

    hsize_t ld_size = px_data.shape(0);
    hsize_t lr_size = px_index.shape(0);
    hsize_t n_data = mxx::allreduce(ld_size, std::plus<hsize_t>());
    hsize_t n_index = mxx::allreduce(lr_size, std::plus<hsize_t>());
    hsize_t d_offset = mxx::exscan(ld_size, std::plus<hsize_t>());
    hsize_t i_offset = mxx::exscan(lr_size, std::plus<hsize_t>());
    assert(n_data == n_index);

    mxx::comm cworld;
    if (cworld.rank() == 1) {
        std::cout << "ndata :: " << n_data << std::endl;
        std::cout << "nindex :: " << n_index << std::endl;
        std::cout << "ndoffset :: " << d_offset << std::endl;
        std::cout << "nioffset :: " << i_offset << std::endl;
    }

    // write
    hid_t file_id = H5Utils::create_file(file_name, MPI_COMM_WORLD,
                                         MPI_INFO_NULL);

    //
    hid_t group_id = H5Gcreate(file_id, "data",
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5VecData<float, 2, pyfarray_t> h5vecdata(
        H5T_NATIVE_FLOAT, sizeof(float),
        {n_data}, {ld_size}, {d_offset}, px_data
    );
    h5vecdata.write(group_id, "puc");

    //
    H5VecData<int, 2, pyiarray_t> h5vecindex(
        H5T_NATIVE_INT, sizeof(int),
        {n_index, 2}, {lr_size, 2}, {i_offset, 0}, px_index
    );
    h5vecindex.write(group_id, "index");

    H5Gclose(group_id); 
    H5Fclose(file_id);
    return 0;
}

PYBIND11_MODULE(PARENSNET_MODULE_NAME, m) {
    pyoptions options;
    // options.disable_function_signatures();

    m.doc() = "PARENSNET";
    m.def("save_puc", &save_puc, R"(
    Save 
    Args:
        in_dict dict with the following keys:
            index      (n_elem, 2)     np.ndarray[int32]
            data       (n_elem, 1)     np.ndarray[float32] 
            ---
            npairs     1               uint32
            nobs       1               uint32
            nvars      1               uint32
          print_debug flag to print  debug statements default:False
          print_timings flag to print  debug statements default:False
    Returns:
        Batch Corrected Matrix )",
          pyarg("data_dict"), pyarg("file_name"));

    //m.def("arma_available", &arma_available, "Is Armadillo available ?");
#ifdef PARENSNET_VERSION
    m.attr("__version__") = PARENSNET_VERSION;
#else
    m.attr("__version__") = "dev";
#endif
}
