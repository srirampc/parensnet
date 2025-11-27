#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Tpublic.h"
#include "H5public.h"
#include "H5version.h"
#include "hdf5.h"
#include "stdlib.h"
#include <cstdlib>
#include <functional>
#include <numeric>
#include <vector>
#include <mpi.h>

#include "parensnet/utils.hpp"

struct H5DataTD {
private:
    H5DataTD(H5DataTD&){};
public:
    hid_t type_id;
    hsize_t type_size;
    std::vector<hsize_t> dims;
    bool should_close;

    H5DataTD(hid_t file_id, const char* dset_name) {
        //
        hid_t dataset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
        hid_t dataspace_id = H5Dget_space(dataset_id);
        const int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        //
        dims.resize(ndims);
        auto hstatus = H5Sget_simple_extent_dims(dataspace_id,
                                                 dims.data(),
                                                 NULL);
        hid_t dttype = H5Dget_type(dataset_id);
        type_id = H5Tget_native_type(dttype, H5T_DIR_DEFAULT);
        should_close = true;
        type_size = H5Tget_size(dttype);

        //
        H5Tclose(dttype);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
    }

    H5DataTD(hid_t rtid, hsize_t rsize, std::vector<hsize_t> rdims):
        type_id(rtid), type_size(rsize), dims(rdims), should_close(false) { }

    ~H5DataTD() {
        if (should_close) H5Tclose(type_id);
    }

};

template<typename DT, const int RANK>
struct H5DataInterface {
    int rank() { return RANK; }
    virtual const DT* data() = 0;
    virtual hsize_t* offsets() = 0;
    virtual hsize_t* counts() = 0;
    virtual const H5DataTD& td() = 0;

    herr_t write_dataset(hid_t file_id, const char* name) {
        hid_t   plist_id; // property list identifier
        hid_t   dset_id;    // dataset identifiers 
        hid_t   dimspace; // file and memory dataspace identifiers
        herr_t  status;
        auto& hdt = td();
        dimspace = H5Screate_simple(RANK, hdt.dims.data() , NULL);
    
        // Create the dataset with default properties and close filespace.
        dset_id =
            H5Dcreate(file_id, name, hdt.type_id, dimspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
        // Create property list for collective dataset write.
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        status = H5Dwrite(dset_id, hdt.type_id, H5S_ALL,
                          H5S_ALL, H5P_DEFAULT, data());
    
        H5Sclose(dimspace);
        return status;
    }

    herr_t write_dataset_mpi(hid_t file_id, const char* name){
        hid_t   plist_id; // property list identifier
        hid_t   dset_id;    // dataset identifiers 
        hid_t   dimspace, memspace; // file and memory dataspace identifiers 
        herr_t  status;
        int      mpi_size, mpi_rank;
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Info info = MPI_INFO_NULL;
    
        MPI_Comm_size(comm, &mpi_size);
        MPI_Comm_rank(comm, &mpi_rank);
        auto& hdt = td();
        //  Create the dataspace for the dataset.
        dimspace = H5Screate_simple(RANK, hdt.dims.data(), NULL);
    
        // Create the dataset with default properties and close filespace.
        dset_id =
            H5Dcreate(file_id, name, hdt.type_id, dimspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(dimspace);
    
        // Each process defines dataset in memory and writes it to the hyperslab
        // in the file.
        memspace  = H5Screate_simple(RANK, counts(), NULL);
    
        // Select hyperslab in the file.
        dimspace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(dimspace, H5S_SELECT_SET, offsets(),
                            NULL, counts(), NULL);
    
        // Create property list for collective dataset write.
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    
        status = H5Dwrite(dset_id, hdt.type_id, memspace,
                          dimspace, plist_id, data());
    
        // Close/release resources.
        H5Dclose(dset_id);
        H5Sclose(dimspace);
        H5Sclose(memspace);
        H5Pclose(plist_id);
        return status;
    }
}; 


template<typename T, int const RANK>
herr_t read_dataset_mpi(
    hid_t file_id,
    const char* dataset_name,
    H5DataInterface<T, RANK>& data_if
){

    hid_t   plist_id; // property list identifier 
    hid_t   dset_id;  // dataset identifiers 
    hid_t   dimspace, memspace; // file and memory dataspace identifiers
    herr_t  status;
    int      mpi_size, mpi_rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    //  Create the dataspace for the dataset.
    //dimspace = H5Screate_simple(RANK, data_dim, NULL);
    //H5Sclose(dimspace);

    // Create the dataset with default properties and close filespace.
    dset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);

    // Each process defines dataset in memory and writes it to the hyperslab
    // in the file.
    memspace  = H5Screate_simple(RANK, data_if.counts(), NULL);

    // Select hyperslab in the file.
    dimspace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(dimspace, H5S_SELECT_SET, data_if.offsets(),
                        NULL, data_if.counts(), NULL);

    //  Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    auto& hdt = data_if.td();
    status = H5Dread(dset_id, hdt.type_id, memspace, dimspace,
                     plist_id, data_if.data());

    // Close/release resources.
    H5Dclose(dset_id);
    H5Sclose(dimspace);
    H5Sclose(memspace);
    H5Pclose(plist_id);
    return status;
}

template<typename DT, const int RANK, typename VECT>
struct H5VecData: H5DataInterface<DT, RANK>{
    H5DataTD hdt;
    std::vector<hsize_t> _counts, _offsets;
    VECT _data;

    //
    H5VecData(hid_t file_id, const char* dset_name): hdt(file_id, dset_name) {}

    H5VecData(hid_t rtid, hsize_t rsize, std::vector<hsize_t> rdims,
              std::vector<hsize_t> cts, std::vector<hsize_t> offs, VECT in_data):
        hdt(rtid, rsize, rdims), _counts(cts), _offsets(offs), _data(in_data){}

    const DT* data(){
        return (const DT*) _data.data();
    }

    const H5DataTD& td() { return hdt; }
    hsize_t* offsets() { return _offsets.data(); }
    hsize_t* counts() { return _counts.data(); }

    void print(const char* pfmt){
        printf("Size: [%d] ; Data: [", hdt.type_size);
        for(int i=0; i < 20; i++){
            printf(pfmt, _data[i]);
        }
        printf("..]\n");
    }

    herr_t write(hid_t parent_id, const char* dataset_name){
        auto& hdt = td();
        return this->write_dataset_mpi(
            parent_id, dataset_name 
        );
    }

    static H5VecData<DT, 2, std::vector<DT>> read_dataset_2d(
        hid_t file_id, const char* dset_name,
        int mpi_rank, int mpi_size
    ){
        H5VecData<DT, 2, std::vector<DT>> hdata(file_id, dset_name);
        assert(hdata.hdt.size() == 2);
        hsize_t nrows = hdata.hdt.dims[0];
        hsize_t ncols = hdata.hdt.dims[1];
    
        //
        hdata._counts = {block_size(mpi_rank, mpi_size, nrows), ncols};
        hdata._offsets = {block_low(mpi_rank, mpi_size, nrows), 0 } ;
    
        //
        hsize_t dfsize = std::accumulate(hdata._counts.begin(),
                                         hdata._counts.end(), 1,
                                         std::multiplies<hsize_t>());
        hdata._data.resize(dfsize);
    
        read_dataset_mpi<RANK, DT>(file_id, dset_name, hdata);
        return hdata;
    }

};

struct H5Utils {
    static hid_t create_file(const char* fname, MPI_Comm comm, MPI_Info info) {
        // Set up file access property list with parallel I/O access
        auto plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, info);

        // set collective metadata reads on FAPL to perform metadata reads
        // collectively, which usually allows datasets to perform better at scale
        H5Pset_all_coll_metadata_ops(plist_id, true);
        H5Pset_coll_metadata_write(plist_id, true);

        // Create a new file collectively and release property list identifier.
        hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        H5Pclose(plist_id);
        return file_id;
    }

    static H5VecData<float, 1, std::vector<float>> test_data_1d(hsize_t dimx, int mpi_rank, int mpi_size){
        hsize_t bsize = block_size(mpi_rank, mpi_size, dimx);
        hsize_t boffset = block_low(mpi_rank, mpi_size, dimx);
        std::vector<float> rdata(bsize, float(mpi_rank));
        std::transform(rdata.cbegin(), rdata.cend(), rdata.begin(),
                       [](float in){ return in + 10.0; } );
        return H5VecData< float, 1, std::vector<float>>(
            H5T_NATIVE_FLOAT, sizeof(float), {dimx},
            { bsize }, { boffset }, rdata
        );
    }

    static H5VecData< int, 2, std::vector<int>> test_data_2d(
        hsize_t dimx, hsize_t dimy,
        int mpi_rank, int mpi_size
    ){
        hsize_t bsize = block_size(mpi_rank, mpi_size, dimx);
        hsize_t boffset = block_low(mpi_rank, mpi_size, dimx);
        //Initialize data buffer
        std::vector<int> rdata(bsize * dimy, int(mpi_rank));
        std::transform(rdata.cbegin(), rdata.cend(), rdata.begin(),
                       [](int in){ return in + 10; } );

        return H5VecData<int, 2, std::vector<int>>(
            H5T_NATIVE_INT, sizeof(int), {dimx, dimy},
            { bsize, dimy }, {boffset, 2}, rdata
        );
    }

};
