import numpy as np
import ctypes as ct

# must be a double array, with single dimension that is contiguous
array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# Load the library as _libspfc.
_libspfc = np.ctypeslib.load_library('../src/libspatialfuncs', '.')

_libspfc.get_pi_typed.argtypes = [array_1d_int,\
                                    array_1d_double,\
                                    array_1d_double,\
                                    ct.c_void_p,\
                                    ct.c_void_p,\
                                    ct.c_void_p,\
                                    array_1d_double,\
                                    array_1d_double,\
                                    ct.c_void_p,\
                                    array_1d_int,\
                                    array_1d_double,\
                                    ]

_libspfc.get_pi_typed.restype  = None

_libspfc.get_theta_typed.argtypes = [array_1d_int,\
                                    array_1d_double,\
                                    array_1d_double,\
                                    ct.c_void_p,\
                                    ct.c_void_p,\
                                    ct.c_void_p,\
                                    array_1d_double,\
                                    array_1d_double,\
                                    ct.c_void_p,\
                                    array_1d_int,\
                                    array_1d_double,\
                                    ]

_libspfc.get_theta_typed.restype  = None

_libspfc.get_tau_typed.argtypes = [array_1d_int,\
                                    array_1d_double,\
                                    array_1d_double,\
                                    ct.c_void_p,\
                                    ct.c_void_p,\
                                    ct.c_void_p,\
                                    array_1d_double,\
                                    array_1d_double,\
                                    ct.c_void_p,\
                                    array_1d_int,\
                                    array_1d_double,\
                                    ct.c_void_p,\
                                    ]

_libspfc.get_tau_typed.restype  = None

def getPiTyped(posmat,typeA=-1,typeB=-1,r=np.array([1.]),rLow=None):
	"""
	Python equivalent to get_pi_typed.

	posmat:	 a matrix with columns type, x and y
	typeA:	 the "from" type that we are interested in, -1 is wildcard
	typeB:	 the "to" type that we are interested i, -1 is wildcard
	r:		 the series of spatial distances wer are interested in
	rLow:	 the low end of each range....0  by default
	"""

	#prepare inputs

	# argument 1 to 3: make a copy, so the matrix is C contiguous (already included in astype)
	ty=posmat[:,0].astype(np.int32)
	x=posmat[:,1].astype(np.double)
	y=posmat[:,2].astype(np.double)

	n = ct.c_int(posmat.shape[0])
	typeA = ct.c_int(typeA)
	typeB = ct.c_int(typeB)

	if not isinstance(r, np.ndarray): #if it is not a 1D numpy array (for ex a scalar or a list), bring it into that shape
		r=np.array(r)
		r=r.reshape((-1))

	if rLow is None:
		rLow = np.zeros_like(r)

	if not isinstance(rLow, np.ndarray): #if it is not a 1D numpy array (for ex a scalar or a list), bring it into that shape
		rLow=np.array(rLow)
		rLow=rLow.reshape((-1))

	rLen=ct.c_int(r.shape[0])
	ind=np.arange(1,r.shape[0]+1,dtype=np.int32)

	#prepare output array
	rc = np.empty_like(r, dtype=np.double)
	
	_libspfc.get_pi_typed(ty,\
							x,\
							y,\
							ct.byref(n),\
							ct.byref(typeA),\
							ct.byref(typeB),\
							rLow,\
							r,\
							ct.byref(rLen),\
							ind,\
							rc)
	return rc



def getThetaTyped(posmat,typeA=-1,typeB=-1,r=np.array([1.]),rLow=None):
	"""
	Python equivalent to get_theta_typed.

	posmat:	 a matrix with columns type, x and y
	typeA:	 the "from" type that we are interested in, -1 is wildcard
	typeB:	 the "to" type that we are interested i, -1 is wildcard
	r:		 the series of spatial distances wer are interested in
	rLow:	 the low end of each range....0  by default
	"""

	#prepare inputs

	# argument 1 to 3: make a copy, so the matrix is C contiguous (already included in astype)
	ty=posmat[:,0].astype(np.int32)
	x=posmat[:,1].astype(np.double)
	y=posmat[:,2].astype(np.double)

	n = ct.c_int(posmat.shape[0])
	typeA = ct.c_int(typeA)
	typeB = ct.c_int(typeB)

	if not isinstance(r, np.ndarray): #if it is not a 1D numpy array (for ex a scalar or a list), bring it into that shape
		r=np.array(r)
		r=r.reshape((-1))

	if rLow is None:
		rLow = np.zeros_like(r)

	if not isinstance(rLow, np.ndarray): #if it is not a 1D numpy array (for ex a scalar or a list), bring it into that shape
		rLow=np.array(rLow)
		rLow=rLow.reshape((-1))

	rLen=ct.c_int(r.shape[0])
	ind=np.arange(1,r.shape[0]+1,dtype=np.int32)

	#prepare output array
	rc = np.empty_like(r, dtype=np.double)
	
	_libspfc.get_theta_typed(ty,\
							x,\
							y,\
							ct.byref(n),\
							ct.byref(typeA),\
							ct.byref(typeB),\
							rLow,\
							r,\
							ct.byref(rLen),\
							ind,\
							rc)
	return rc


def getTauTyped(posmat,typeA=-1,typeB=-1,r=np.array([1.]),rLow=None,compType="representative"):
	"""
	Python equivalent to get_tau_typed.

	posmat: 	a matrix with columns type, x and y
	typeA: 		the "from" type that we are interested in, -1 is wildcard
	typeB: 		the "to" type that we are interested i, -1 is wildcard
	r: 			the series of spatial distances wer are interested in
	r.low: 		the low end of each range....0  by default
	compType: 	what type of points are included in the comparison set.
					"representative" if comparison set is representative of the underlying population
					"independent" if comparison set is cases/events coming from an indepedent process
	
	return: tau values for all the distances we looked at
	"""

	if compType == "representative":
		compTypeInt = 0
	elif compType == "independent":
		compTypeInt = 1
	else:
		raise ValueError("unkown comparison type specified")


	#prepare inputs

	# argument 1 to 3: make a copy, so the matrix is C contiguous (already included in astype)
	ty=posmat[:,0].astype(np.int32)
	x=posmat[:,1].astype(np.double)
	y=posmat[:,2].astype(np.double)

	n = ct.c_int(posmat.shape[0])
	typeA = ct.c_int(typeA)
	typeB = ct.c_int(typeB)

	if not isinstance(r, np.ndarray): #if it is not a 1D numpy array (for ex a scalar or a list), bring it into that shape
		r=np.array(r)
		r=r.reshape((-1))

	if rLow is None:
		rLow = np.zeros_like(r)

	if not isinstance(rLow, np.ndarray): #if it is not a 1D numpy array (for ex a scalar or a list), bring it into that shape
		rLow=np.array(rLow)
		rLow=rLow.reshape((-1))

	rLen=ct.c_int(r.shape[0])
	ind=np.arange(1,r.shape[0]+1,dtype=np.int32)

	#prepare output array
	rc = np.empty_like(r, dtype=np.double)
	compTypeInt=ct.c_int(compTypeInt)

	#prepare output array
	rc = np.empty_like(r, dtype=np.double)    

	_libspfc.get_tau_typed(ty,\
							x,\
							y,\
							ct.byref(n),\
							ct.byref(typeA),\
							ct.byref(typeB),\
							rLow,\
							r,\
							ct.byref(rLen),\
							ind,\
							rc)
	return rc