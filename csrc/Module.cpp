#define PY_SSIZE_T_CLEAN
#define ARG_OFFSET 5

#include <Python.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <iostream>

// #define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

//Meta-data format we will use
#include <THCTensorInfo.cuh>

//Cuda kernels
#include <kernel.h>

#define ERROR_MSG cout << "Error at " << __FILE__ << ":" << __LINE__ << "\n";

using namespace std;

TensorInfo<void, idxType> PyOb_2_tinfo(PyObject* tensor, float_types data_type)
{
  PyObject* PyStrides = PyObject_CallMethod(tensor, "stride", NULL);
  if(PyStrides == NULL)
  {
    ERROR_MSG;
    cout << "PyStrides = NULL" << endl;
  }

  PyObject* PySizes = PyObject_CallMethod(tensor, "size", NULL);
  if(PySizes == NULL)
  {
    ERROR_MSG;
    cout << "PySizes = NULL" << endl;
  }

  PyObject* PyDataPtr = PyObject_CallMethod(tensor, "data_ptr", NULL);
  if(PyDataPtr == NULL)
  {
    ERROR_MSG;
    cout << "PyDataPtr = NULL" << endl;
  }

  void* data_ptr = (void*) PyLong_AsLong(PyDataPtr);
  Py_ssize_t ndims = PyList_GET_SIZE(PySizes);

  // TODO put proper checking on ndims < MAX_CUTORCH_DIMS
  idxType strides[MAX_CUTORCH_DIMS], sizes[MAX_CUTORCH_DIMS];

  for(int i = 0; i < ndims; i++)
  {
    strides[i] = PyLong_AsLong(PyTuple_GetItem(PyStrides, i));
    sizes[i] = PyLong_AsLong(PyTuple_GetItem(PySizes, i));
  }

  Py_DECREF(PyStrides);
  Py_DECREF(PySizes);
  Py_DECREF(PyDataPtr);

  return TensorInfo<void, idxType>(data_ptr, ndims, sizes, strides, data_type);
}

vector<TensorInfo<void, idxType> > get_TInfos(PyObject* args)
{
  vector<TensorInfo<void, idxType> > info_vec;
#ifdef DEBUG_ANY 
  cout << "Processing " << PyTuple_GET_SIZE(args) << " arguments" << endl;
#endif

#ifdef CHECK_MEMLEAK
  for(int iter = 0; iter < 1e7; iter++ )
#endif
    for(Py_ssize_t i = 0; i<PyTuple_GET_SIZE(args) - 1; i++)
    {
      PyObject* pyTensor = PyTuple_GetItem(args, i);

      // check type, only take if Tensor, Variable, or Parameter
      string objType(pyTensor->ob_type->tp_name);

      PyObject* pyObjTypeCall = PyObject_CallMethod(pyTensor, "type", NULL);
      if(pyObjTypeCall == NULL)
      {
	ERROR_MSG;
	cout << "For args item " << i << ", pyObjTypeCall = NULL" << endl;
      }

      PyObject* pyObjASCII = PyUnicode_AsASCIIString(pyObjTypeCall);
      if(pyObjASCII == NULL)
      {
	ERROR_MSG;
	cout << "For args item " << i << ", pyObjASCII = NULL " << endl;
      }

      Py_DECREF(pyObjTypeCall);

      string objTypeCall(PyBytes_AsString(pyObjASCII));

      Py_DECREF(pyObjASCII);

#ifdef DEBUG_ANY
      cout << "arg " << i << endl;
      cout << "objType = " << objType << endl;
      cout << "objTypeCall = " << objTypeCall << endl;
#endif

      if(objTypeCall == "torch.cuda.FloatTensor")
#ifdef CHECK_MEMLEAK
	if(iter == 0 )
#endif
	  info_vec.push_back(PyOb_2_tinfo(pyTensor, FLOAT));
#ifdef CHECK_MEMLEAK
	else
	  info_vec[i] = PyOb_2_tinfo(pyTensor, FLOAT);
#endif
      else if(objTypeCall == "torch.cuda.HalfTensor")
	info_vec.push_back(PyOb_2_tinfo(pyTensor, HALF));
      // TODO add double
      else
      {
	ERROR_MSG;
	cout << "For args item " << i << ", unsupported .type() found: "
	     << objTypeCall << "\n"
		"Supported types:\n"
		"torch.cuda.FloatTensor\n"
		"torch.cuda.HalfTensor\n"
		"torch.autograd.variable.Variable containing FloatTensor\n"
		"torch.autograd.variable.Variable containing HalfTensor\n"
		"torch.nn.parameter.Parameter containing FloatTensor\n" 
		"torch.nn.parameter.Parameter containing HalfTensor\n" 
	     << endl;
      }
    }

  return info_vec;
}

int getLastArg_AsInt(PyObject* args)
{
  // None of these should return new references so I don't think this leaks memory.
  int dims = PyLong_AsLong(PyTuple_GetItem(args, PyTuple_GET_SIZE(args) - 1));
  return dims;
}

// Stepping stone, can evolve to be more general (argument forwarding?)
template<typename wrapper>
void dispatch
(
  float_types rtti, 
  vector<TensorInfo<void, idxType>>& tensors, 
  int dim
)
{
  switch(rtti)
  {
    case FLOAT:
      wrapper::template call<float, float, idxType>(tensors, dim);
      break;
    case HALF:
      wrapper::template call<half, float, idxType>(tensors, dim);
      break;
    default:
      std::cout << "Unsupported rtti in Module.cpp:dispatch()" << std::endl;
      PyErr_SetString(PyExc_RuntimeError, "Unsupported data type in Module.cpp:dispatch, "
                                          "supported data types are half and float");
      exit(-1);
  }
}

// Will extract all tensors in order. Assumes flat structure, tensors can not be wrapped in lists
// tuples or any other iterator structure.
static PyObject* weight_norm_fwd(PyObject* self, PyObject* args)
{
#ifdef USE_NVTX
nvtxRangePushA("weight_norm_fwd C backend");
#endif

  vector<TensorInfo<void, idxType> > tensors = get_TInfos(args);
  int dim = getLastArg_AsInt(args);

  if(dim != 0 && dim != tensors[2].dims - 1)
    PyErr_SetString(PyExc_RuntimeError, "weight_norm_fwd currently only "
                                        "supports first or last dimension.");
  else
  {
#ifdef DEBUG_ANY
    cout << "tensors.size() = " << tensors.size() << ", dim = " << dim << endl;
#endif
 
    dispatch<send_to_fwd_wrapper>(tensors[0].type, tensors, dim);

#ifdef USE_NVTX
    nvtxRangePop();
#endif
  }

  Py_RETURN_NONE;
}

static PyObject* weight_norm_bwd(PyObject* self, PyObject* args)
{
#ifdef USE_NVTX
  nvtxRangePushA("weight_norm_bwd C backend");
#endif

  vector<TensorInfo<void, idxType> >tensors = get_TInfos(args);
  int dim = getLastArg_AsInt(args);

  if(dim != 0 && dim != tensors[3].dims - 1)
    PyErr_SetString(PyExc_RuntimeError, "weight_norm_bwd currently only "
                                        "supports first or last dimension.");
  else
  {
#ifdef DEBUG_ANY
    cout << "tensors.size() = " << tensors.size() << ", dim = " << dim << endl;
#endif

    dispatch<send_to_bwd_wrapper>(tensors[0].type, tensors, dim);

#ifdef USE_NVTX
    nvtxRangePop();
#endif
  }

  Py_RETURN_NONE;
}

//*******************PYTHON BOILER PLATE*******************
static PyMethodDef apex_methods[] = {
  {"weight_norm_fwd", (PyCFunction) weight_norm_fwd, METH_VARARGS, "Slowest-dim norm, forward pass."},
  {"weight_norm_bwd", (PyCFunction) weight_norm_bwd, METH_VARARGS, "Slowest-dim norm, backward pass."},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

//Module Definitions
static struct PyModuleDef apex = {
  PyModuleDef_HEAD_INIT, "apex._C", "Module to add CUDA extensions to Pytorch.", -1, apex_methods
};
//Initialization Function
PyMODINIT_FUNC PyInit__C(void){

  //Let's throw an error if we can't find pytorch.
  PyImport_ImportModule("torch");
  Py_Initialize();
  return PyModule_Create(&apex);
}
#else
PyMODINIT_FUNC initMODULE(void){
  //Let's throw an error if we can't find pytorch.
  PyImport_ImportModule("torch");
  (void) Py_InitModule3("apex._C", apex, "A PyTorch Extension.");
}

#endif
//*********************************************************

