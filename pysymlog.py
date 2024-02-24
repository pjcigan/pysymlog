### pysymlog
### v1.0
### written by Phil Cigan
__author__ = "Phil Cigan"
__version__ = "1.0.1"

"""
Utilities for plotting in matplotlib, plotly, etc. using a symmetric or signed log transform.
This allows the plotting of positive and negative data with a pseudo-log axis scale or stretch. 

"""


import numpy as np

def symmetric_logarithm(arg, base=10, shift=1):
    """
    Transform a value into a symmetric log space that allows for smooth continuous
    stretch display in plotting utilites.  The following describes the transform:
    # positive: log10( (arg+shift)/shift )  | negative: -log( shift/(arg+shift) )
    #         = log(arg+shift) - log(shift) |         = -log(-arg+shift) + log(shift)
    
    Parameters
    ----------
    arg : float, or array-like
        The value(s) to be transformed
    base : int, or float
        The logarithm base
    shift : int, or float
        The amount to shift values in the transform.  Values smaller in scale than
        this value will appear more stretched -- decrease to stretch small values
        more, or increase to minimize the stretching effect.  Similar to the 
        parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
        shift here should be ~ 1/10 of linthresh for close results.
    
    Returns
    -------
    symmetric_logarithm : float, or numpy.array
    """
    
    if np.isscalar(arg):
        if arg >= 0:
            #return log(arg+shift,base)-log(shift,base)
            return np.emath.logn(base, arg+shift) - np.emath.logn(base,shift)
        else:
            #return -log(-arg+shift,base)+log(shift,base)
            return -np.emath.logn(base, -arg+shift) + np.emath.logn(base,shift)
    else:
        sl = np.zeros_like(arg)
        negmask = ( np.array(arg)<=0 )
        sl[~negmask] = np.emath.logn(base, np.array(arg)[~negmask]+shift) - np.emath.logn(base,shift)
        sl[negmask] = -np.emath.logn(base, -np.array(arg)[negmask]+shift) + np.emath.logn(base,shift)
        return sl

def inverse_symmetric_logarithm(arginv, base=10, shift=1):
    """
    Inverse symmetric logarithm transform, for converting transformed values back
    to linear space.  The following describes this transform:
    # positive: (arg+shift)/shift = 10**arginv  |  negative: -shift/(arg+shift) = 10**argvinv
    #           arg = (10**arginv)*shift-shift  |      arg = -shift/(10**arginv)-shift
    
    Parameters
    ----------
    arginv : float, or array-like
        The value(s) to be transformed
    base : int, or float
        The logarithm base
    shift : int, or float
        The amount to shift values in the transform.  Values smaller in scale than
        this value will appear more stretched -- decrease to stretch small values
        more, or increase to minimize the stretching effect.  Similar to the 
        parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
        shift here should be ~ 1/10 of linthresh for close results.
    
    Returns
    -------
    inverse_symmetric_logarithm : float, or numpy.array
    """
    
    if np.isscalar(arginv):
        if arginv >=0:
            #return shift*(base**(arginv)) - shift
            return shift*(np.power(base,arginv)) - shift
        else:
            #return -( shift/(base**(arginv)) - shift )
            return -( shift/(np.power(base,arginv)) - shift )
    else:
        isl = np.zeros_like(arginv)
        negmask = ( np.array(arginv)<=0 )
        isl[~negmask] = shift*np.power(base,np.array(arginv)[~negmask]) - shift
        isl[negmask]  = -( shift/np.power(base,np.array(arginv)[negmask]) - shift )
        return isl

def make_symmetric_logarithm(base=10, shift=1):
    """
    Convenience function to generate a new symmetriclog function with specified
    logarithm base and shift values.
    """
    return lambda arg: symmetric_logarithm(arg, base=base, shift=shift)

def make_inverse_symmetric_logarithm(base=10, shift=1):
    """
    Convenience function to generate a new inverse symmetriclog function with 
    specified logarithm base and shift values.
    """
    return lambda arg: inverse_symmetric_logarithm(arg, base=base, shift=shift)

#symmetric_logspace: transform to symlog, then np.linspace on those Values

def symmetric_logspace(lo,hi,N, input_format='linear', shift=1, base=10):
    """
    Function to generate a sequence of values in log space based on low and high
    values, similar to the functionality of e.g. np.logspace but instead using
    a symmetric log transform.
    
    Parameters
    ----------
    lo : float
        The low limit
    hi : 
        The high limit
    N : int
        The number of elements the output array should have
    input_format : str
        Specifies whether the input lo,hi values are given in 'linear', 'log',
        or 'symlog'.  For example, if input_format='linear' then lo,hi are given 
        as their linear values (e.g. 1e-2 and 1000), if input_format='log' then 
        they are taken as log values (e.g. -2 and 3 instead of 1e-2 and 1000).  
        If input_format='symlog' then they are taken as log values with negative
        values meaning less than zero (not as a fraction less than one). 
        To avoid confusion, it's best to input lo and hi values in linear space.
    shift : float
        The amount to shift values in the transform. See symmetric_logarithm() 
        for more detail.
    base : int, or float
        The logarithm base
    
    Returns
    -------
    inverse_symmetriclog_array : numpy.array
        The linear scale values of 
    
    Examples
    --------
    psl.symmetric_logspace(-10, 100, N=5)
    #--> np.array([-10., -0.90530352, 2.03015151, 16.49415053, 100.])
    psl.symmetric_logspace(-np.log10(10), np.log10(100), N=5, input_format='symlog')
    #--> np.array([-10., -0.90530352, 2.03015151, 16.49415053, 100.])
    psl.symmetric_logspace(-2, 3, N=3, input_format='log')
    #--> np.array([1.00000000e-02, 3.07963834e+01, 1.00000000e+03])
    """
    #if input_format='linear', then the input values are the actual linear values
    #   to use, e.g. "1000" like in np.linspace or np.geomspace instead
    #   of "3" as would be used for np.logspace
    if input_format.lower()=='log':
        lo_xform = symmetric_logarithm(10**lo, shift=shift, base=base)
        hi_xform = symmetric_logarithm(10**hi, shift=shift, base=base)
    else: 
        if 'sym' in input_format.lower():
            lo = np.sign(lo) * 10**np.abs(lo)
            hi = np.sign(hi) * 10**np.abs(hi)
        lo_xform = symmetric_logarithm(lo, shift=shift, base=base)
        hi_xform = symmetric_logarithm(hi, shift=shift, base=base)
    
    #np.geomspace(...)
    linspace_xform = np.linspace(lo_xform,hi_xform,N+1)
    return inverse_symmetric_logarithm(linspace_xform, shift=shift, base=base)

def symmetric_logspace_from_array(array, N, input_format='linear', shift=1, base=10):
    """
    Function to generate a sequence of values in log space based on the low and 
    high values from an input array. Similar to the functionality of np.logspace 
    but instead using a symmetric log transform. Useful in particular for 
    generating histogram bins in symmetric log space from a particular array.
    
    Parameters
    ----------
    array : array-like (list, tuple, np.array...)
        The input array of values from which to determine low and high limits.
    N : int
        The number of elements the output array should have
    input_format : str
        Specifies whether the input lo,hi values are given in 'linear', 'log',
        or 'symlog'.  For example, if input_format='linear' then lo,hi are given 
        as their linear values (e.g. 1e-2 and 1000), if input_format='log' then 
        they are taken as log values (e.g. -2 and 3 instead of 1e-2 and 1000).  
        If input_format='symlog' then they are taken as log values with negative
        values meaning less than zero (not as a fraction less than one). 
        To avoid confusion, it's best to input lo and hi values in linear space.
    shift : float
        The amount to shift values in the transform. See symmetric_logarithm() 
        for more detail.
    base : int, or float
        The logarithm base
    
    Returns
    -------
    inverse_symmetriclog_array : numpy.array
    
    Example
    -------
    testdat = np.random.randn(1000)
    print([testdat.min(),testdat.max()])
    #--> [-3.0152414613412186, 3.336199242183869]
    psl.symmetric_logspace_from_array(testdat, N=4, shift=1e-2)
    #--> np.array([-3.01524146, -0.05491167,  0.06179836,  3.33619924])
    """
    return symmetric_logspace(np.nanmin(array), np.nanmax(array), N, input_format=input_format, shift=shift, base=base)

## Functions to generate the decades to use for e.g. ticks in logspace
def symmetric_log_decades(lo, hi, thresh, include_zero=True):
    """
    Generate the decades from lo to hi, using the threshold, stopping at the 
    specified threshold if the range crosses zero.  This is used in particular
    for generating lists of tick locations and labels.
    
    Parameters
    ----------
    lo : float
        The low value to consider, given in linear space
    hi : float
        The high value to consider, given in linear space
    thresh : float
        The lowest (absolute value) numerical scale to use in the generation of
        the list of decades -- if the lo-hi range crosses zero (lo is negative
        and hi is positive).
    include_zero : bool
        If True, include zero in the resulting array (if the array crosses zero)
    
    Returns
    -------
    decades : numpy.array
    
    Examples
    --------
    symmetric_log_decades(-100, 100, 1e-1)
    #--> array([-100. ,  -10. ,   -1. ,   -0.1,    0. ,   0.1,    1. ,   10. ,  100. ])
    symmetric_log_decades(-1.2345, 0.987, 1e-2)
    #--> array([-1.   , -0.1  , -0.01,   0. ,   0.01 ,  0.1  ,  1.   ])
    symmetric_log_decades(-1e6, 1e3, 1000)
    #--> array([-1000000.,  -100000.,   -10000.,    -1000.,     0.,    1000.])
    symmetric_log_decades(-1e6, 1e3, 1000, include_zero=False)
    #--> array([-1000000.,  -100000.,   -10000.,    -1000.,     1000.])
    """
    if lo>hi: 
        raise Exception('symmetric_log_decades: lo value %s is higher than hi value %s'%(lo,hi))
    decade_floor_base = np.around( np.log10(np.sign(lo)*lo) )
    decade_ceil_base = np.around( np.log10(np.sign(hi)*hi) )
    decades = [ np.sign(lo)*10**decade_floor_base, ]
    if lo<0:
        if hi<0:
            #Case where all values are negative
            for i in np.arange(decade_floor_base, decade_ceil_base+.1, -1)[1:] :
                decades.append(-10**i)
        else:
            #Case where values span negative and positive
            if -10**(decade_floor_base-1) > -thresh: 
                #Do not append any more negative decades if it's already past the threshold
                pass
            else: 
                #Append negative decades until the threshold is reached
                for i in np.arange(decade_floor_base,np.log10(thresh)-.1, -1)[1:] :
                    decades.append(-10**i)
            #Now append positive decades until the ceiling is reached
            for i in np.arange(np.log10(thresh), decade_ceil_base+.1, 1):
                decades.append(10**i)
    else:
        #Case where all values are positive
        for i in np.arange(decade_floor_base, decade_ceil_base+.1, 1):
                decades.append(10**i)
    decades = np.array(decades)
    if include_zero==True:
        if True in (decades<0) and True in (decades>0):
            #Spans across zero, add zero in to the list.
            decades = np.array(list(decades[decades<0])+[0,]+list(decades[decades>0]))
    return decades

def symmetric_log_decades_from_array(data, thresh='auto', auto_percentile=10., include_zero=True):
    """
    Convenience function to calculate an array of symmetric log decades from an
    input data set based on the range in its values.  If a scale threshold is not
    specified, or set to 'auto', a reasonable attempt is made to estimate a
    threshold heuristically based on a percentile of its absolute values.  
    (This hopefully captures a meaningful lower scale.)  
    
    Parameters
    ---------- 
    data : array-like
        The data set from which a range will be determined to calculate the array
        of symmetric log decades. 
    thresh : float, or 'auto'
        The threshold lowest absolute scale to consider for the decades, which is
        used when the data set range crosses the zero mark (includes positive and
        negative values).
    auto_percentile : float
        The percentile to use in automatic threshold determination.
    include_zero : bool
        If True, include zero in the resulting array (if the array crosses zero)
    
    Returns
    -------
    decades : numpy.array
    
    Example
    -------
    testdat = np.tan( np.linspace(-5,10, 1000) )
    print(np.nanpercentile(testdat, [0.01,99.99]))
    #--> [-404.29964885  840.36255943]
    psl.symmetric_log_decades_from_array(testdat)#, thresh='auto', auto_percentile=10., include_zero=True)
    #--> np.array([-1.e+03, -1.e+02, -1.e+01, -1.e+00, -1.e-01,  0.e+00,  1.e-01,
    #               1.e+00,  1.e+01,  1.e+02,  1.e+03])
    """
    ##--> nanmin/nanmax doesn't catch inf...  Better to use np.ma.masked_invalid
    data_cleaned = np.ma.masked_invalid(data).compressed()
    
    lo = np.nanmin(data_cleaned)
    hi = np.nanmax(data_cleaned)
    
    ## if thresh is left to 'auto', calculate a reasonable threshold based on the data
    if thresh=='auto':
        #Use the order of magnitude of the Nth percentile of the absval of the data
        percscale = np.nanpercentile(np.abs(data_cleaned),auto_percentile) 
        thresh = 10**np.floor(np.log10( percscale ))
    else: thresh = np.abs(thresh)
    return symmetric_log_decades(lo, hi, thresh, include_zero=include_zero)

def symlogbin_histogram(data, Nbins, limits=['auto','auto'], shift=1, base=10, density=False, weights=None):
    """
    Convenience function to call numpy.histogram() using symmetric log scaled
    bins, which are calculated from the data array and specified number of bins.
    
    Parameters
    ----------
    data : array_like
        Input data. The histogram is computed over the flattened array.
    Nbins : int
        Desired number of bins that will be equal-width in symmetric log space.
    limits : [float,float] or ['auto','auto']
        Limits to impose on the output histogram bin minimum/maximum, specified
        in order [lo,hi].  'auto' determines the value automatically as the min
        or max value in the input data array. Supercedes the usage of the range
        parameter in np.histogram
    base : int, or float
        The logarithm base for the transform
    shift : int, or float
        The amount to shift values in the transform.  Values smaller in scale than
        this value will appear more stretched -- decrease to stretch small values
        more, or increase to minimize the stretching effect.  Similar to the 
        parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
        shift here should be ~ 1/10 of linthresh for close results.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.
    
    Returns
    -------
    hist : array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    
    Examples
    --------
    testdat = np.tan( np.linspace(-5,10, 1000) )
    counts,symlogbins = symlogbin_histogram(testdat, 101, limits=['auto','auto'], shift=1e-4)
    count_densities,symlogbins2 = symlogbin_histogram(testdat, 101, limits=[-1e-5,1e3], shift=1e-3, density=True)
    """
    lo, hi = limits
    if limits[0] == 'auto':
        #lo = np.nanmin(data)
        lo = np.ma.masked_invalid(data).compressed().min() #This also handles np.inf...
    if limits[1] == 'auto':
        #hi = np.nanmax(data)
        hi = np.ma.masked_invalid(data).compressed().max() #This also handles np.inf...
    symlogbins = symmetric_logspace(lo, hi, N=Nbins, shift=shift, base=base)
    npcounts, npbins = np.histogram(data, bins=symlogbins, density=density, weights=weights) #, range=None
    return npcounts, npbins 

def symlogbin_histogram2d(xdata, ydata, Nbins, limits=[['auto','auto'], ['auto','auto']], shift=1, base=10, density=False, weights=None):
    """
    Convenience function to call numpy.histogram() using symmetric log scaled
    bins, which are calculated from the data array and specified number of bins.
    
    Parameters
    ----------
    xdata : array_like
        Array containing the x-coordinate values of the data to be histogrammed.
    ydata : array_like
        Array containing the y-coordinate values of the data to be histogrammed.
    Nbins : int or [int,int]
        Desired number of bins that will be equal-width in symmetric log space.
        If int, the number of bins for each of the two dimensions (nx=ny=Nbins).
        If [int, int], the number of bins in each dimension (nx, ny = Nbins).
    limits : [[float,float], [float,float]] or [['auto','auto'], ['auto','auto']]
        Limits to impose on the output histogram bin minima/maxima, specified
        in order [[x_lo,x_hi], [y_lo,y_hi]].  'auto' determines the specific 
        values automatically as the min or max value in the input data array.
        Supercedes the usage of the range parameter in np.histogram2d
    base : int, or float
        The logarithm base for the transform
    shift : float or [float, float]
        The amount to shift values in the transform.  Values smaller in scale than
        this value will appear more stretched -- decrease to stretch small values
        more, or increase to minimize the stretching effect.  Similar to the 
        parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
        shift here should be ~ 1/10 of linthresh for close results.
        Either input as a single value to apply to both x and y components, or 
        as a list/tuple/array_like of values in order [x_shift, y_shift] 
    weights : array_like, optional
        An array of values w_i weighing each sample (x_i, y_i). Weights are 
        normalized to 1 if density is True. If density is False, the values 
        of the returned histogram are equal to the sum of the weights belonging 
        to the samples falling into each bin.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.
        If ``False`` (the default), returns the number of samples in each bin. 
        If ``True``, returns the probability *density* function at the bin, 
        normalized as ( bin_count / sample_count / bin_area ).
    
    Returns
    -------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram of samples x and y. Values in x are 
        histogrammed along the first dimension and values in y are histogrammed 
        along the second dimension.
    xedges : ndarray, shape(nx+1,)
        The bin edges along the first (x-coordinate) dimension.
    yedges : ndarray, shape(ny+1,)
        The bin edges along the second (y-coordinate) dimension.
    
    Examples
    --------
    dx = np.random.exponential(scale=1, size=5000)*np.random.randn(5000)
    dy = np.random.exponential(scale=3, size=5000)*np.random.randn(5000)
    # 101 auto-calculated bins in each of x,y directions
    counts, bins_x, bins_y = psl.symlogbin_histogram2d(dx,dy, 101, limits=['auto','auto'], shift=1e-1)
    # 31 bins in x, 101 bins in y, and returning count density
    count_densities,bins_x2, bins_y2 = psl.symlogbin_histogram2d(dx,dy, [31,101], limits=[[-5,5],[-50,50]], shift=1e-1, density=True)
    """
    if len(np.shape(limits))==1:
        xlo = ylo = limits[0]
        xhi = yhi = limits[1]
    elif len(np.shape(limits))==2:
        xlo, xhi = limits[0]
        ylo, yhi = limits[1]
    else:
        raise Exception('psl.histogram2d: limits parameter %s invalid. Must be [lo,hi] or [[xlo,xhi], [ylo,yhi]]'%limits)
    if xlo == 'auto':
        xlo = np.ma.masked_invalid(xdata).compressed().min() #This also handles np.inf...
    if xhi == 'auto':
        xhi = np.ma.masked_invalid(xdata).compressed().max() #This also handles np.inf...
    if ylo == 'auto':
        ylo = np.ma.masked_invalid(ydata).compressed().min() #This also handles np.inf...
    if yhi == 'auto':
        yhi = np.ma.masked_invalid(ydata).compressed().max() #This also handles np.inf...
    
    if np.isscalar(Nbins):
        Nbins_x = Nbins_y = Nbins
    else: 
        Nbins_x, Nbins_y = Nbins
    
    if np.isscalar(shift):
        shift_x = shift_y = shift
    else: 
        shift_x, shift_y = shift
    
    if np.isscalar(base):
        base_x = base_y = base
    else: 
        base_x, base_y = base
    
    symlogbins_x = symmetric_logspace(xlo, xhi, N=Nbins_x, shift=shift_x, base=base_x)
    symlogbins_y = symmetric_logspace(ylo, yhi, N=Nbins_y, shift=shift_y, base=base_y)
    
    npcounts, npbins_x, npbins_y = np.histogram2d(xdata, ydata, bins=[symlogbins_x,symlogbins_y], density=density, weights=weights) #, range=None
    return npcounts, npbins_x, npbins_y 



###-----------------------------------------------------------------------------

### Functions to register or use these transforms in various plotting packages:
#   matplotlib, plotly...

def register_mpl():
    """
    Calling this function imports matplotlib and registers the symmetric log
    scale for use, and also adds these classes and functions to the pysymlog 
    namespace:
    - SymmetricLogarithmLocator
    - SymmetricLogarithmTransform
    - InvertedSymmetricLogarithmTransform
    - MinorSymmetricLogLocator
    - SymmetricLogarithmScale
    - MinorSymLogLocator
    - set_symmetriclog_minorticks()
    - set_symlog_minorticks()
    - symlogbin_hist_mpl()
    - SymmetricLogarithmNorm [colorbar normalization]
    """
    
    import matplotlib.pyplot as plt
    #import matplotlib as mpl
    import matplotlib.scale as mscale
    import matplotlib.transforms as mtransforms
    import matplotlib.ticker as ticker
    #from matplotlib.ticker import FixedLocator, FuncFormatter
    
    global SymmetricLogarithmLocator #Register the name in the global namespace
    class SymmetricLogarithmLocator(ticker.Locator):
        """
        Determine the tick locations for symmetric logarithm axes. (Crudely 
        modified from  matplotlib.ticker.SymmetricalLogLocator )
        """
        
        def __init__(self, transform=None, subs=None, shift=None, base=None):
            """
            Parameters
            ----------
            transform : `~.scale.SymmetricLogarithmTransform`, optional
                If set, defines the *base* and *shift* of the symmetriclog transform.
            base, shift : float, optional
                The *base* and *shift* of the symmetriclog transform, as documented
                for `.SymmetricLogarithmScale`.  These parameters are only used if
                *transform* is not set.
            subs : sequence of float, default: [1]
                The multiples of integer powers of the base where ticks are placed,
                i.e., ticks are placed at
                ``[sub * base**i for i in ... for sub in subs]``.
            Notes
            -----
            Either *transform*, or both *base* and *shift*, must be given.
            """
            if transform is not None:
                self._base = transform.base
                self._shift = transform.shift
            elif shift is not None and base is not None:
                self._base = base
                self._shift = shift
            else:
                raise ValueError("Either transform, or both shift "
                                 "and base, must be provided.")
            if subs is None:
                self._subs = [1.0]
            else:
                self._subs = subs
            self.numticks = 15
        
        def set_params(self, subs=None, numticks=None):
            """Set parameters within this locator."""
            if numticks is not None:
                self.numticks = numticks
            if subs is not None:
                self._subs = subs
        
        def __call__(self):
            """Return the locations of the ticks."""
            # Note, these are untransformed coordinates
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)
        
        def tick_values(self, vmin, vmax):
            base = self._base
            shift = self._shift
            
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            
            # The domain is divided into three sections, only some of
            # which may actually be present.
            #
            # <======== -t ==0== t ========>
            # aaaaaaaaa    bbbbb   ccccccccc
            #
            # a) and c) will have ticks at integral log positions.  The
            # number of ticks needs to be reduced if there are more
            # than self.numticks of them.
            #
            # b) has a tick at 0 and only 0 (we assume t is a small
            # number, and the linear segment is just an implementation
            # detail and not interesting.)
            #
            # We could also add ticks at t, but that seems to usually be
            # uninteresting.
            #
            # "simple" mode is when the range falls entirely within (-t,
            # t) -- it should just display (vmin, 0, vmax)
            if -shift < vmin < vmax < shift:
                # only the linear range is present
                return [vmin, vmax]
            
            # Lower log range is present
            has_a = (vmin < -shift)
            # Upper log range is present
            has_c = (vmax > shift)
            
            # Check if linear range is present
            has_b = (has_a and vmax > -shift) or (has_c and vmin < shift)
            
            def get_log_range(lo, hi):
                lo = np.floor(np.log(lo) / np.log(base))
                hi = np.ceil(np.log(hi) / np.log(base))
                return lo, hi
            
            # Calculate all the ranges, so we can determine striding
            a_lo, a_hi = (0, 0)
            if has_a:
                a_upper_lim = min(-shift, vmax)
                a_lo, a_hi = get_log_range(abs(a_upper_lim), abs(vmin) + 1)
            
            c_lo, c_hi = (0, 0)
            if has_c:
                c_lower_lim = max(shift, vmin)
                c_lo, c_hi = get_log_range(c_lower_lim, vmax + 1)
            
            # Calculate the total number of integer exponents in a and c ranges
            total_ticks = (a_hi - a_lo) + (c_hi - c_lo)
            if has_b:
                total_ticks += 1
            stride = max(total_ticks // (self.numticks - 1), 1)
            
            decades = []
            if has_a:
                decades.extend(-1 * (base ** (np.arange(a_lo, a_hi, stride)[::-1])))
                #decades.extend(-np.arange(a_lo, a_hi, stride)[::-1])
            
            if has_b:
                #decades.extend(-1 * (base ** (np.arange(a_hi, 0,
                #                                        stride)[::-1])))
                decades.append(0.0)
                #decades.extend(base ** (np.arange(0, c_lo, stride)))
            
            if has_c:
                decades.extend(base ** (np.arange(c_lo, c_hi, stride)))
                #decades.extend(np.arange(c_lo, c_hi, stride))
            
            # Add the subticks if requested
            if self._subs is None:
                subs = np.arange(2.0, base)
            else:
                subs = np.asarray(self._subs)
            
            if len(subs) > 1 or subs[0] != 1.0:
                ticklocs = []
                for decade in decades:
                    if decade == 0:
                        ticklocs.append(decade)
                    else:
                        ticklocs.extend(subs * decade)
            else:
                ticklocs = decades
            
            return self.raise_if_exceeds(np.array(ticklocs))
        
        def view_limits(self, vmin, vmax):
            """Try to choose the view limits intelligently."""
            b = self._base
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            
            #if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            if plt.rcParams['axes.autolimit_mode'] == 'round_numbers':
                vmin = _decade_less_equal(vmin, b)
                vmax = _decade_greater_equal(vmax, b)
                if vmin == vmax:
                    vmin = _decade_less(vmin, b)
                    vmax = _decade_greater(vmax, b)
            
            result = mtransforms.nonsingular(vmin, vmax)
            return result
    
    global SymmetricLogarithmTransform #Register the name in the global namespace
    class SymmetricLogarithmTransform(mtransforms.Transform):
            input_dims = 1
            output_dims = 1
            is_separable = True
           
            def __init__(self, base, shift):
                mtransforms.Transform.__init__(self)
                self.base = base
                self.shift = shift
           
            def transform_non_affine(self, a):
                    if np.isscalar(a):
                        if a >= 0:
                            return log(a+self.shift,self.base)-log(self.shift,self.base)
                        else:
                            return -log(-a+self.shift,self.base)+log(self.shift,self.base)
                    else:
                        sl = np.zeros_like(a)
                        negmask = np.array( (a<=0) )
                        # positive: log10( (arg+shift)/shift )  | negative: -log( shift/(arg+shift) )
                        #         = log(arg+shift) - log(shift) |         = -log(-arg+shift) + log(shift)
                        #sl[~negmask] = np.emath.logn(self.base,a[~negmask]+self.shift)-log(self.shift,self.base)
                        #sl[negmask] = -np.emath.logn(self.base,-a[negmask]+self.shift)+log(self.shift,self.base)
                        sl[~negmask] = np.emath.logn(self.base,a[~negmask]+self.shift)-np.emath.logn(self.base,self.shift)
                        sl[negmask] = -np.emath.logn(self.base,-a[negmask]+self.shift)+np.emath.logn(self.base,self.shift)
                        return sl
     
            def inverted(self):
                return InvertedSymmetricLogarithmTransform(self.base,self.shift)
    
    global InvertedSymmetricLogarithmTransform #Register the name in the global namespace 
    class InvertedSymmetricLogarithmTransform(mtransforms.Transform):
            input_dims = 1
            output_dims = 1
            is_separable = True
            
            def __init__(self, base, shift):
                mtransforms.Transform.__init__(self)
                self.base = base
                self.shift = shift
            
            def transform_non_affine(self, a):
                if np.isscalar(a):
                    if a >=0:
                        return self.base**(a/self.shift) - self.shift
                    else:
                        return -self.base**(-a*self.shift) + self.shift
                else:
                    sl = np.zeros_like(a)
                    negmask = np.array( (a<=0) )
                    # positive: (arg+shift)/shift = 10**arginv  |  negative: -shift/(arg+shift) = 10**argvinv
                    #           arg = (10**arginv)*shift-shift  |      arg = -shift/(10**arginv)-shift
                    sl[~negmask] = self.shift*np.power(self.base,a[~negmask])-self.shift
                    sl[negmask] = -( self.shift/np.power(self.base,a[negmask])-self.shift )
                    return sl
            
            def inverted(self):
                return SymmetricLogarithmTransform(self.base,self.shift)
    
    global MinorSymmetricLogLocator #Register the name in the global namespace
    class MinorSymmetricLogLocator(ticker.Locator):
        """
        Dynamically find minor tick positions based on the positions of major ticks for a symlog scaling.
        From https://stackoverflow.com/a/20495928
        """
        def __init__(self, shift, nints=10):
            """
            Ticks will be placed between the major ticks.
            The placement is logarithmic, with adjusted numbers in the shifted
            region around zero. nints gives the number of
            intervals that will be bounded by the minor ticks.
            """
            self.shift = shift
            self.nintervals = nints
        
        def __call__(self):
            # Return the locations of the ticks
            majorlocs = self.axis.get_majorticklocs()
            
            if len(majorlocs) == 1:
                return self.raise_if_exceeds(np.array([]))
            
            # add temporary major tick locs at either end of the current range
            # to fill in minor tick gaps
            dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
            dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end
            
            # add temporary major tick location at the lower end
            if majorlocs[0] != 0. and ((majorlocs[0] != self.shift and dmlower > self.shift) or (dmlower == self.shift and majorlocs[0] < 0)):
                majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
            else:
                majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.shift)
            
            # add temporary major tick location at the upper end
            if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.shift and dmupper > self.shift) or (dmupper == self.shift and majorlocs[-1] > 0)):
                majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
            else:
                majorlocs = np.append(majorlocs, majorlocs[-1]+self.shift)
            
            # iterate through minor locs
            minorlocs = []
            
            # handle the lowest part
            for i in range(1, len(majorlocs)):
                majorstep = majorlocs[i] - majorlocs[i-1]
                if abs(majorlocs[i-1] + majorstep/2) < self.shift:
                    ndivs = self.nintervals
                else:
                    ndivs = self.nintervals - 1.
                
                minorstep = majorstep / ndivs
                locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
                minorlocs.extend(locs)
            
            return self.raise_if_exceeds(np.array(minorlocs))
        
        def tick_values(self, vmin, vmax):
            raise NotImplementedError('Cannot get tick locations for a '
                              '%s type.' % type(self))
    
    #axin.yaxis.set_minor_locator(MinorSymmetricLogLocator(1e-1)) #1e-1 here is shift used in symmetriclog
    
    global set_symmetriclog_minorticks #Register the name in the global namespace
    #def set_symlog_minorticks(axin,xy='y',base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12):
    def set_symmetriclog_minorticks(axin,xy='y',thresh=1e-1, formatter=ticker.NullFormatter() ):
        """
        Sets minor ticks for a symmetric log axis. Best for matplotlib v2.0.2 and above.  
        
        Parameters
        ----------
        axin : matplotlib.axis object
            The matplotlib axis object to plot to, created with plt.subplot(), plt.axis() etc.
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        thresh : float
            The lowest (absolute value) numerical scale to use in the generation of
            the list of decades -- if the lo-hi range crosses zero (lo is negative
            and hi is positive).
        
        See https://stackoverflow.com/a/44079725 for examples with LogLocator
        numticks=12 : Set this number to something larger than the number of *major* ticks
        """
        #from matplotlib import ticker
        if xy.lower() in ['x','xy','both']:
            axin.xaxis.set_minor_locator(MinorSymmetricLogLocator(thresh)) #1e-1 here is shift used in symlog
            axin.xaxis.set_minor_formatter(formatter)
        if xy.lower() in ['y','xy','both']:
            axin.yaxis.set_minor_locator(MinorSymmetricLogLocator(thresh)) #1e-1 here is shift used in symlog
            axin.yaxis.set_minor_formatter(formatter)
        if xy.lower() not in ['x', 'y','xy','both']: 
            raise Exception('set_symmetriclog_minorticks(): option xy="%s" not valid.  Please use "x","y","xy", or "both"'%(xy))
    
    global SymmetricLogarithmScale #Register the name in the global namespace
    class SymmetricLogarithmScale(mscale.ScaleBase):
        """
        ScaleBase class for generating Symmetric Logarithm scale.
        """
        
        name = 'symmetriclog'
        
        def __init__(self, axis, base=10, shift=1., **kwargs):
            # note in older versions of matplotlib (<3.1), this worked fine.
            # mscale.ScaleBase.__init__(self)
            # In newer versions (>=3.1), you also need to pass in `axis` as an arg
            mscale.ScaleBase.__init__(self, axis)
            self.base=base
            self.shift=shift
            self._transform = SymmetricLogarithmTransform(base, shift)
        
        def set_default_locators_and_formatters(self, axis):
            #axis.set_major_locator(ticker.SymmetricalLogLocator(linthresh=self.shift*10, base=self.base))
            #axis.set_major_formatter(ticker.LogFormatterSciNotation(linthresh=self.shift*10, base=self.base))
            axis.set_major_locator(SymmetricLogarithmLocator(self.get_transform())) #<-- need to get these working to fix zoom issues?
            axis.set_major_formatter(ticker.LogFormatterSciNotation(self.base))
            #axis.set_major_formatter(ticker.ScalarFormatter()) #cuts off decimals...
            axis.set_minor_locator(MinorSymmetricLogLocator(self.shift*10))
            axis.set_minor_formatter(ticker.NullFormatter())
        
        #... No need to limit range for symlog!
        def limit_range_for_scale(self, vmin, vmax, minpos):
            return  vmin, vmax
        
        def get_transform(self):
            return SymmetricLogarithmTransform(self.base,self.shift)
    
    mscale.register_scale(SymmetricLogarithmScale)
    
    
    
    ### Some equivalent utilities for matplotlib's symlog (with linear scale interior)
    global MinorSymLogLocator #Register the name in the global namespace
    class MinorSymLogLocator(ticker.Locator):
        """
        Dynamically find minor tick positions based on the positions of major ticks for a symlog scaling.
        From https://stackoverflow.com/a/20495928
        """
        def __init__(self, linthresh, nints=10):
            """
            Ticks will be placed between the major ticks.
            The placement is linear for x between -linthresh and linthresh,
            otherwise its logarithmically. nints gives the number of
            intervals that will be bounded by the minor ticks.
            """
            self.linthresh = linthresh
            self.nintervals = nints
        
        def __call__(self):
            # Return the locations of the ticks
            majorlocs = self.axis.get_majorticklocs()
            
            if len(majorlocs) == 1:
                return self.raise_if_exceeds(np.array([]))
            
            # add temporary major tick locs at either end of the current range
            # to fill in minor tick gaps
            dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
            dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end
            
            # add temporary major tick location at the lower end
            if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
                majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
            else:
                majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)
            
            # add temporary major tick location at the upper end
            if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
                majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
            else:
                majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)
            
            # iterate through minor locs
            minorlocs = []
            
            # handle the lowest part
            for i in range(1, len(majorlocs)):
                majorstep = majorlocs[i] - majorlocs[i-1]
                if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                    ndivs = self.nintervals
                else:
                    ndivs = self.nintervals - 1.
                
                minorstep = majorstep / ndivs
                locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
                minorlocs.extend(locs)
            
            return self.raise_if_exceeds(np.array(minorlocs))
        
        def tick_values(self, vmin, vmax):
            raise NotImplementedError('Cannot get tick locations for a '
                              '%s type.' % type(self))
    
    #axin.yaxis.set_minor_locator(MinorSymLogLocator(1e-1)) #1e-1 here is linthresh used in symlog
    
    global set_symlog_minorticks #Register the name in the global namespace
    #def set_symlog_minorticks(axin,xy='y',base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12):
    def set_symlog_minorticks(axin,xy='y',thresh=1e-1, formatter=ticker.NullFormatter()):
        """
        Sets minor ticks for a symlog (linear transition through zero) axis. 
        Best for matplotlib v2.0.2 and above.  
        
        Parameters
        ----------
        axin : matplotlib.axis object
            The matplotlib axis object to plot to, created with plt.subplot(), plt.axis() etc.
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        thresh : float
            The lowest (absolute value) numerical scale to use in the generation of
            the list of decades -- if the lo-hi range crosses zero (lo is negative
            and hi is positive).
        
        See https://stackoverflow.com/a/44079725 for examples with LogLocator
        numticks=12 : Set this number to something larger than the number of *major* ticks
        """
        from matplotlib import ticker
        if xy.lower() in ['x','xy','both']:
            axin.xaxis.set_minor_locator(MinorSymLogLocator(thresh)) #1e-1 here is linthresh used in symlog
            axin.xaxis.set_minor_formatter(formatter)
        if xy.lower() in ['y','xy','both']:
            axin.yaxis.set_minor_locator(MinorSymLogLocator(thresh)) #1e-1 here is linthresh used in symlog
            axin.yaxis.set_minor_formatter(formatter)
        if xy.lower() not in ['x', 'y','xy','both']: 
            raise Exception('set_symlog_minorticks(): option xy="%s" not valid.  Please use "x","y","xy", or "both"'%(xy))


    global reformat_major_ticklabels #Register the name in the global namespace
    def reformat_major_ticklabels(ax, xy='y', tickvals='auto', fmt='{:g}', lineobjind=0, thresh='auto', auto_percentile=10.):
        """
        Reformat the major tick labels, using the specified string format.
        Default option is to use 'g' formatting, for linear numbers on small
        scales and scientific notation on large scales.
        
        Inputs
        ------
        ax : matplotlib.axs object
            The matplotlib axis object to plot to, created with plt.subplot(), plt.axis() etc.
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        tickvals : 'auto' or list 
            The tick values to use -- 'auto' will attempt to generate sensible 
            values from the axis array, otherwise an explicit list of values 
            can be specified.
        fmt : str
            The string format to use for the python .format() method.  
            Default is {:g}.
        lineobjind : int
            The line object index to use for ax.axes.get_lines()[lineobj] when 
            attempting to generate tickvals in 'auto' mode.
        thresh : 'auto' or float
            The thresh value to use in automatic tick value calculation,
            input to symmetric_log_decades_from_array()
        auto_percentile : float
            The automatic percentile to use in automatic tick value calculation,
            input to symmetric_log_decades_from_array()
        
        Examples
        --------
        ## manual tick value specification
        ax1=plt.subplot(111)
        ax1.plot(np.arange(0,10,.01), np.arange(-100,100,.2))
        ax1.set_yscale('symmetriclog')
        psl.reformat_major_ticklabels(ax1, xy='y', tickvals=[-100,-10,-1,0,1,10,100])
        plt.show(); plt.clf()
        ## auto tick value calculation
        ax1=plt.subplot(111)
        ax1.plot(np.arange(0,10,.01), np.arange(-100,100,.2))
        ax1.set_yscale('symmetriclog')
        psl.reformat_major_ticklabels(ax1, xy='y', tickvals='auto', thresh=1.)
        plt.show(); plt.clf()
        """
        if xy.lower() in ['x','xy','both']:
            if tickvals == 'auto':
                try: 
                    tickvals_x = symmetric_log_decades_from_array(ax.axes.get_lines()[lineobjind].get_data()[0], thresh=thresh, auto_percentile=auto_percentile)
                except:
                    tickvals_x = symmetric_log_decades_from_array(ax.axes.get_xlim(), thresh=thresh, auto_percentile=auto_percentile)
            else:
                tickvals_x = tickvals
            ax.xaxis.set_major_locator(ticker.FixedLocator(tickvals_x))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt.format(x)))
        if xy.lower() in ['y','xy','both']:
            if tickvals == 'auto':
                try:
                    tickvals_y = symmetric_log_decades_from_array(ax.axes.get_lines()[lineobjind].get_data()[1], thresh=thresh, auto_percentile=auto_percentile)
                except:
                    tickvals_y = symmetric_log_decades_from_array(ax.axes.get_ylim(), thresh=thresh, auto_percentile=auto_percentile)
            else:
                tickvals_y = tickvals
            ax.yaxis.set_major_locator(ticker.FixedLocator(tickvals_y))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: fmt.format(y)))
        if xy.lower() not in ['x', 'y','xy','both']: 
            raise Exception('reformat_major_ticklabels(): option xy="%s" not valid.  Please use "x","y","xy", or "both"'%(xy))

    global symlogbin_hist_mpl #Register the name in the global namespace
    def symlogbin_hist_mpl(ax, data, Nbins, limits=['auto','auto'], shift=1, base=10, orientation='vertical', density=False, **hist_kwargs):
        """
        Plot a histogram with symmetric log scale bins using matplotlib.hist().
        NOTE -- parameter "orientation" follows the pyplot.hist() definition, which 
        is the axis of the bar lengths, NOT the direction of the bins.
        That is, orientation='vertical' (the default) produces a histogram with 
        bins in the x-axis and the bar heights rising in the y-axis. Likewise, 
        orientation='horizontal' will produce a histogram on its side, with bins
        spanning the y-axis and bars increasing along the x-axis. 
        
        Parameters
        ----------
        ax : matplotlib.axis object
            The matplotlib axis object to plot to, created with plt.subplot(), plt.axis() etc.
        data : array_like
            Input data. The histogram is computed over the flattened array.
        Nbins : int
            Desired number of bins that will be equal-width in symmetric log space.
        base : int, or float
            The logarithm base
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            If 'horizontal', `~.Axes.barh` will be used for bar-type histograms
            and the *bottom* kwarg will be the left edges.
            NOTE -- this is the same defininition as used in pyplot.hist()
        density : bool, default: False
            If ``True``, draw and return a probability density: each bin
            will display the bin's raw count divided by the total number of
            counts *and the bin width*
            (``density = counts / (sum(counts) * np.diff(bins))``),
            so that the area under the histogram integrates to 1
            (``np.sum(density * np.diff(bins)) == 1``).
            If *stacked* is also ``True``, the sum of the histograms is
            normalized to 1.
        hist_kwargs : various
            Any other keyword args to be passed to pyplot.hist(). These include
            range, weights, cumulative, bottom, histtype, align, orientation, 
            rwidth, log, color, label, stacked
            For descriptions, see  help(pyplot.hist)
            
        Returns
        -------
        n : array or list of arrays
            The values of the histogram bins. See *density* and *weights* for a
            description of the possible semantics.  If input *x* is an array,
            then this is an array of length *nbins*. If input is a sequence of
            arrays ``[data1, data2, ...]``, then this is a list of arrays with
            the values of the histograms for each of the arrays in the same
            order.  The dtype of the array *n* (or of its element arrays) will
            always be float even if no weighting or normalization is used.
        bins : array
            The edges of the bins. Length nbins + 1 (nbins left edges and right
            edge of last bin).  Always a single array even when multiple data
            sets are passed in.
        patches : `.BarContainer` or list of a single `.Polygon` or list of such objects
            Container of individual artists used to create the histogram
            or list of such containers if there are multiple input datasets.

        Notes
        -----
        For large numbers of bins (>1000), plotting can be significantly
        accelerated by using `~.Axes.stairs` to plot a pre-computed histogram
        (``plt.stairs(*np.histogram(data))``), or by setting *histtype* to
        'step' or 'stepfilled' rather than 'bar' or 'barstacked'.
        
        Examples
        --------
        testdat = np.tan( np.linspace(-5,10, 1000) )
        ax1 = plt.subplot(111)
        counts,symlogbins,patches = symlogbin_hist_mpl(ax1, testdat, 101, shift=1e-4)
        plt.show()
        #
        ax1 = plt.subplot(111)
        count_densities,symlogbins2,patches2 = symlogbin_hist_mpl(ax1, testdat, 101, limits=[-1e-5,1e3], shift=1e-3, density=True)
        plt.show()
        """
        #symlogbins = symmetric_logspace(np.nanmin(data), histdat2.max(), Nbins, shift=shift, base=base )
        #--> Just call symlogbin_histogram since it already handles limits etc 
        counts, symlogbins = symlogbin_histogram(data, Nbins, limits=limits, shift=shift, base=base, density=density)
        
        bars, bin_edges, patches = ax.hist(data, bins=symlogbins, density=density, orientation=orientation, **hist_kwargs); 
        if orientation=='vertical':
            ax.set_xscale('symmetriclog', shift=shift, base=base)
        else:
            ax.set_yscale('symmetriclog', shift=shift, base=base)
        return bars, bin_edges, patches
    
    
    ### Colorbar normalization
    from matplotlib.colors import make_norm_from_scale, Normalize
    global SymmetricLogarithmNorm #Register the name in the global namespace
    @make_norm_from_scale(
    SymmetricLogarithmScale,
    init=lambda shift, base=10, vmin=None, vmax=None, *,clip=False: None)
    class SymmetricLogarithmNorm(Normalize):
        """
        The symmetrical logarithmic scale is logarithmic in both the
        positive and negative directions from the origin.

        Since the values close to zero tend toward infinity, there is a
        need to have a range around zero that is linear.  The parameter
        *linthresh* allows the user to specify the size of this range
        (-*linthresh*, *linthresh*).

        Parameters
        ----------
        shift : float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : float, default: 10
        """
        
        @property
        def shift(self):
            return self._scale.shift
        
        @shift.setter
        def shift(self, value):
            self._scale.shift = value





###------- Plotly-specific functions -------

def register_plotly():
    """
    Calling this function imports plotly graph_objects and defines functions for
    using symmetric log transforms in plotly Figures.  The following functions
    are defined and added to the pysymlog namespace:
    - set_plotly_scale_symmetriclog()
    - go_scatter_symlog()
    - go_line_symlog()
    - go_histogram_symlog()
    - px_scatter_symlog()
    - px_line_symlog()
    - px_histogram_symlog()
    
    Example
    -------
    xdata = np.arange(-2, 5.0, 0.01)
    ydata = np.tan(xdata)
    fig = go.Figure()
    psl.go_scatter_symlog(fig, xdata, ydata, xy='y')
    fig.show()
    """
    
    import plotly.graph_objects as go
    import plotly.express as px
    
    global set_plotly_scale_symmetriclog #Register the name in the global namespace
    def set_plotly_scale_symmetriclog(fig, plot_obj_index=0, xy='both', tickvals_x='auto', tickvals_y='auto', auto_percentile=10., shift=1., base=10):
        """
        Sets a plotly figure scale to symmetric logarithm.
        tickvals_x and tickvals_y in linear scale, not log.  
        e.g., tickvals_y=[-100,-10,10,100], NOT [-2,-1,1,2]
        
        Parameters
        ----------
        fig : plotly.graph_objects.Figure or plotly.express scatter etc object
            The figure object to apply scaling to.
        plot_obj_index : int
            The index number of the plot object to use for 'auto' mode scale 
            calculations.  e.g., 0 for the first scatter/line/etc object
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        tickvals_x : array_like or 'auto'
            The major tick values to use for the x-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        tickvals_y : array_like or 'auto'
            The major tick values to use for the y-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        auto_percentile : float
            The percentile to use in automatic threshold determination.
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : int, or float
            The logarithm base
        """
        if xy.lower() in ['x','xy','both']: 
            if tickvals_x=='auto':
                tickvals_x = symmetric_log_decades_from_array( inverse_symmetric_logarithm(fig.data[plot_obj_index]['x'], shift=shift, base=base ), auto_percentile=auto_percentile )
            ### Be aware that when specifying manual ticks, at least one should be a float...
            #symmetric_logarithm([-1000,-100,-10,0,10,100,1000], shift=1) #array([-3, -2, -1,  0,  1,  2,  3])
            fig.update_xaxes(tickvals=symmetric_logarithm(np.array(tickvals_x).astype(float), shift=shift, base=base), ticktext=tickvals_x)
        if xy.lower() in ['y','xy','both']: 
            if tickvals_y=='auto':
                tickvals_y = symmetric_log_decades_from_array( inverse_symmetric_logarithm(fig.data[plot_obj_index]['y'], shift=shift, base=base ), auto_percentile=auto_percentile)
            fig.update_yaxes(tickvals=symmetric_logarithm(np.array(tickvals_y).astype(float), shift=shift, base=base), ticktext=tickvals_y)
    
    
    ### Variations using plotly graph_objects
    
    global go_scatter_symlog #Register the name in the global namespace
    def go_scatter_symlog(fig, xvals, yvals, tickvals_x='auto', tickvals_y='auto', xy='both', shift=1., base=10, **scatter_kwargs):
        """
        Add a plotly.graph_objects.Scatter trace to a figure using symmetric log scaling.
        
        Parameters
        ----------
        fig : plotly.graph_objects.Figure object
            The figure object to apply scaling to.
        xvals : array_like
            The array of x-axis data for plotting.
        yvals : array_like
            The array of y-axis data for plotting.
        tickvals_x : array_like or 'auto'
            The major tick values to use for the x-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        tickvals_y : array_like or 'auto'
            The major tick values to use for the y-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : int, or float
            The logarithm base
        **scatter_kwargs
            Any other keyword args to pass to go.Scatter()
        
        Examples
        --------
        xdata = np.arange(-2, 5.0, 0.01)
        ydata = np.tan(xdata)
        fig = go.Figure()
        psl.go_scatter_symlog(fig, xdata, ydata, xy='y')
        fig.show()
        #
        yerrs = np.random.randn(len(xdata))/2
        sizes = np.abs(np.int32(10*yerrs))
        fig = go.Figure()
        psl.go_scatter_symlog(fig, xdata, ydata, xy='y', error_y=dict(array=yerrs, color='#555555', thickness=0.7, width=2), mode='markers', marker=dict(size=sizes, color=yerrs, colorscale='Plasma_r'))
        fig.show()
        """
        if xy.lower()=='x':
            fig.add_trace(go.Scatter(x=symmetric_logarithm(xvals, shift=shift, base=base), y=yvals, hovertemplate='( %{customdata:G}, %{y} )<extra></extra>', customdata=xvals, **scatter_kwargs)) 
        elif xy.lower()=='y':
            fig.add_trace(go.Scatter(x=xvals, y=symmetric_logarithm(yvals, shift=shift, base=base), hovertemplate='( %{x}, %{customdata:G} )<extra></extra>', customdata=yvals, **scatter_kwargs)) 
        elif xy.lower() in ['xy','both']: 
            fig.add_trace(go.Scatter(x=symmetric_logarithm(xvals, shift=shift, base=base), y=symmetric_logarithm(yvals, shift=shift, base=base), hovertemplate='( %{customdata[0]:G}, %{customdata[1]:G} )<extra></extra>', customdata=np.dstack([xvals,yvals]).squeeze(), **scatter_kwargs)) 
        else: raise Exception("go_scatter_symlog(): 'xy' %s is invalid, must be one of ['x','y','xy','both']"%(xy))
        set_plotly_scale_symmetriclog(fig, xy=xy, tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift, base=base)

    global go_line_symlog
    def go_line_symlog(fig, xvals, yvals, tickvals_x='auto', tickvals_y='auto', xy='both', shift=1., base=10, **line_kwargs):
        """
        Add a plotly.graph_objects.Line trace to a figure using symmetric log scaling.
        
        Parameters
        ----------
        fig : plotly.graph_objects.Figure object
            The figure object to apply scaling to.
        xvals : array_like
            The array of x-axis data for plotting.
        yvals : array_like
            The array of y-axis data for plotting.
        tickvals_x : array_like or 'auto'
            The major tick values to use for the x-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        tickvals_y : array_like or 'auto'
            The major tick values to use for the y-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : int, or float
            The logarithm base
        **line_kwargs
            Any other keyword args to pass to go.Line()
        
        Examples
        --------
        """
        if xy.lower()=='x':
            fig.add_trace(go.Line(x=symmetric_logarithm(xvals, shift=shift, base=base), y=yvals, hovertemplate='( %{customdata:G}, %{y} )<extra></extra>', customdata=xvals, **line_kwargs)) 
        elif xy.lower()=='y':
            fig.add_trace(go.Line(x=xvals, y=symmetric_logarithm(yvals, shift=shift, base=base), hovertemplate='( %{x}, %{customdata:G} )<extra></extra>', customdata=yvals, **line_kwargs)) 
        elif xy.lower() in ['xy','both']: 
            fig.add_trace(go.Line(x=symmetric_logarithm(xvals, shift=shift, base=base), y=symmetric_logarithm(yvals, shift=shift, base=base), hovertemplate='( %{customdata[0]:G}, %{customdata[1]:G} )<extra></extra>', customdata=np.dstack([xvals,yvals]).squeeze(), **line_kwargs)) 
        else: raise Exception("go_line_symlog(): 'xy' %s is invalid, must be one of ['x','y','xy','both']"%(xy))
        set_plotly_scale_symmetriclog(fig, xy=xy, tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift, base=base)

    global go_histogram_symlog #Register the name in the global namespace
    def go_histogram_symlog(histdata, bins, density=False, binwidth_frac=1., shift=1, base=10, orientation='vertical', tickvals_x='auto', tickvals_y='auto', **traces_kwargs):
        """
        Make a histogram in plotly using graph_objects.Bar and with symmetric log scaling.
        
        Parameters
        ----------
        histdata : array-like
            The data to bin into a histogram.
        bins : int or array-like
            The bins to use for creating the histogram. Supplied as either an 
            integer denoting the number of bins to calculate, or as an array-like
            of explicit bin edges to use, which are expected to already be 
            symmetric log scaled.
        density : bool
            If False, computes the histogram values as counts. If True, computes
            them as count densities.
        binwidth_frac : float
            Fractional width of bins, in symmetric log space.  Default is 1, 
            meaning bins will touch each other.  0.5 would mean bins are each
            half of their full width, and so on.
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : int, or float
            The logarithm base
        orientation : str
            'horizontal' or 'vertical', the direction of the histogram. NOTE, 
            this is consistent with the pyplot definition: Default here
            is 'vertical', meaning bins span horizontally along the x-axis and
            bin height increases along the y-axis.
            'vertical' here means bins spanning the x-axis with bin lengths
            increasing along the x-axis.
        tickvals_x : array_like or 'auto'
            The major tick values to use for the x-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        tickvals_y : array_like or 'auto'
            The major tick values to use for the y-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        **traces_kwargs
            Keyword args to pass to go.Figure.update_traces().  Some examples: 
            marker_color='rgb(158,202,225)', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        
        Returns
        -------
        plotly.graph_objects.Figure object
        
        Example
        -------
        histdata = np.random.lognormal(size=1000)
        fig = psl.go_histogram_symlog(histdata, 101, shift=1e-2)
        fig.show()
        #
        histdata2 = np.random.randn(1000)
        fig = psl.go_histogram_symlog(histdata2, 101, binwidth_frac=0.9, marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
        fig.show()
        # Histogram with bins spanning the y-axis ('horizontal' orientation)
        fig = psl.go_histogram_symlog(histdata2, 101, orientation='horizontal')
        fig.show()
        """
        if np.isscalar(bins)==True:
            #If bins is given as a number, calculate the bin edges
            symlogbins = symmetric_logspace(np.nanmin(histdata), np.nanmax(histdata), bins, input_format='linear', shift=shift, base=base)
        else: 
            symlogbins = np.copy(bins)
        
        npcounts, npbins = np.histogram(histdata, bins=symlogbins, density=density)
        #if density==True: 
        #    #As in numpy documentation, density = counts / (sum(counts) * np.diff(bins))
        #    npcounts / (np.nansum(npcounts) * np.diff(npbins))
        
        center_bins = 0.5 * (symlogbins[:-1] + symlogbins[1:])
        center_bins_symlog = symmetric_logarithm(center_bins, shift=shift, base=base)
        widths_symlog = np.diff(center_bins_symlog)
        widths_symlog = np.array( list(widths_symlog) + [widths_symlog[-1],] ) #To make it the same shape
        
        if 'hor' in orientation.lower():
            fig = go.Figure()
            fig.add_trace( go.Bar( y=center_bins_symlog, x=npcounts, orientation='h' ) )
            fig.update_traces(hovertemplate= "<b>Counts:%{x:.3f}</b>")
            fig.update_traces(width=binwidth_frac*widths_symlog, **traces_kwargs) 
            set_plotly_scale_symmetriclog(fig, xy='y', tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift, base=base)
            fig.update_yaxes(showgrid=True) #These get turned off otherwise by default
        else:
            fig = go.Figure()
            fig.add_trace( go.Bar( x=center_bins_symlog, y=npcounts ) )
            fig.update_traces(hovertemplate= "<b>Counts:%{y:.3f}</b>")
            fig.update_traces(width=binwidth_frac*widths_symlog, **traces_kwargs) 
            set_plotly_scale_symmetriclog(fig, xy='x', tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift, base=base)
            fig.update_xaxes(showgrid=True) #These get turned off otherwise by default
        #fig.show()
        return fig


    
    ### Variations using plotly express
    
    global px_scatter_symlog #Register the name in the global namespace
    def px_scatter_symlog(xvals, yvals, tickvals_x='auto', tickvals_y='auto', xy='both', shift=1., base=10, **scatter_kwargs):
        """
        Make a plotly.express.scatter trace using symmetric log scaling.
        
        Parameters
        ----------
        xvals : array_like
            The array of x-axis data for plotting.
        yvals : array_like
            The array of y-axis data for plotting.
        tickvals_x : array_like or 'auto'
            The major tick values to use for the x-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        tickvals_y : array_like or 'auto'
            The major tick values to use for the y-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : int, or float
            The logarithm base
        **scatter_kwargs
            Any other keyword args to pass to px.scatter()
        
        Returns
        -------
        plotly.express.scatter object
        
        Examples
        --------
        xdata = np.arange(-2, 5.0, 0.01)
        ydata = np.tan(xdata)
        fig = psl.px_scatter_symlog(xdata, ydata, xy='y')
        fig.show()
        #
        yerrs = np.random.randn(len(xdata))/2
        sizes = np.abs(np.int32(5*yerrs))
        fig = psl.px_scatter_symlog(xdata, ydata, xy='y', error_y=yerrs, size=sizes, color=yerrs, color_discrete_sequence= px.colors.sequential.Plasma_r, labels={'x':'xdata', 'y':'symmetric log y'})
        fig.show()
        """
        ### When specifying x and y data without a dataframe, can't use hover_data and custom_data...
        #   Instead, just use labels={'x':'<xlabel>', 'y':'<ylabel>'} in the scatter_kwargs
        if xy.lower()=='x':
            fig = px.scatter(x=symmetric_logarithm(xvals, shift=shift, base=base), y=yvals, **scatter_kwargs)#hover_data='( %{custom_data:G}, %{y} )<extra></extra>', custom_data=xvals, **scatter_kwargs)
        elif xy.lower()=='y':
            fig = px.scatter(x=xvals, y=symmetric_logarithm(yvals, shift=shift, base=base), **scatter_kwargs)#hover_data='( %{x}, %{custom_data:G} )<extra></extra>', custom_data=yvals, **scatter_kwargs)
        elif xy.lower() in ['xy','both']: 
            fig = px.scatter(xsymmetric_logarithm(xvals, shift=shift, base=base), y=symmetric_logarithm(yvals, shift=shift, base=base), **scatter_kwarg)#, hover_data='( %{custom_data[0]:G}, %{customdata[1]:G} )<extra></extra>', custom_data=np.dstack([xvals,yvals]).squeeze(), **scatter_kwargs)
        else: raise Exception("px_scatter_symlog(): 'xy' %s is invalid, must be one of ['x','y','xy','both']"%(xy))
        set_plotly_scale_symmetriclog(fig, xy=xy, tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift)
        return fig
    
    global px_line_symlog #Register the name in the global namespace
    def px_line_symlog(xvals, yvals, tickvals_x='auto', tickvals_y='auto', xy='both', shift=1., base=10, **line_kwargs):
        """
        Make a plotly.express.line trace using symmetric log scaling.
        
        Parameters
        ----------
        xvals : array_like
            The array of x-axis data for plotting.
        yvals : array_like
            The array of y-axis data for plotting.
        tickvals_x : array_like or 'auto'
            The major tick values to use for the x-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        tickvals_y : array_like or 'auto'
            The major tick values to use for the y-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        xy : str
            The axis to modify 'x', 'y', or 'xy' or 'both' for both axes.
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : int, or float
            The logarithm base
        **line_kwargs
            Any other keyword args to pass to px.line()
        
        Returns
        -------
        plotly.express.line object
        
        Example
        -------
        xdata = np.arange(-2, 5.0, 0.01)
        ydata = np.tan(xdata)
        fig = psl.px_line_symlog(xdata, ydata, xy='y', labels={'x':'xdata', 'y':'symmetric log y'})
        fig.show()
        """
        ### When specifying x and y data without a dataframe, can't use hover_data and custom_data...
        #   Instead, just use labels={'x':'<xlabel>', 'y':'<ylabel>'} in the line_kwargs
        if xy.lower()=='x':
            fig = px.line(x=symmetric_logarithm(xvals, shift=shift, base=base), y=yvals, **line_kwargs)
        elif xy.lower()=='y':
            fig = px.line(x=xvals, y=symmetric_logarithm(yvals, shift=shift, base=base), **line_kwargs)
        elif xy.lower() in ['xy','both']: 
            fig = px.line(xsymmetric_logarithm(xvals, shift=shift, base=base), y=symmetric_logarithm(yvals, shift=shift, base=base), **line_kwarg)#, hover_data='( %{custom_data[0]:G}, %{customdata[1]:G} )<extra></extra>', custom_data=np.dstack([xvals,yvals]).squeeze(), **line_kwargs)
        else: raise Exception("px_line_symlog(): 'xy' %s is invalid, must be one of ['x','y','xy','both']"%(xy))
        set_plotly_scale_symmetriclog(fig, xy=xy, tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift, base=base)
        return fig
    
    global px_histogram_symlog #Register the name in the global namespace
    def px_histogram_symlog(histdata, bins, density=False, binwidth_frac=1., shift=1, base=10, orientation='vertical', labels={'x':'Bin values','y':'Counts'}, tickvals_x='auto', tickvals_y='auto', **traces_kwargs):
        """
        Make a histogram in plotly using plot.express.bar and with symmetric log scaling.
        
        Parameters
        ----------
        histdata : array-like
            The data to bin into a histogram.
        bins : int or array-like
            The bins to use for creating the histogram. Supplied as either an 
            integer denoting the number of bins to calculate, or as an array-like
            of explicit bin edges to use, which are expected to already be 
            symmetric log scaled.
        density : bool
            If False, computes the histogram values as counts. If True, computes
            them as count densities.
        binwidth_frac : float
            Fractional width of bins, in symmetric log space.  Default is 1, 
            meaning bins will touch each other.  0.5 would mean bins are each
            half of their full width, and so on.
        shift : int, or float
            The amount to shift values in the transform.  Values smaller in scale than
            this value will appear more stretched -- decrease to stretch small values
            more, or increase to minimize the stretching effect.  Similar to the 
            parameter 'linthresh' in matplotlib's 'symlog' scale, though values for 
            shift here should be ~ 1/10 of linthresh for close results.
        base : int, or float
            The logarithm base
        orientation : str
            'horizontal' or 'vertical', the direction of the histogram. NOTE, 
            this is consistent with the pyplot definition: Default here
            is 'vertical', meaning bins span horizontally along the x-axis and
            bin height increases along the y-axis.
            'vertical' here means bins spanning the x-axis with bin lengths
            increasing along the x-axis.
        labels : dict of strings
            The dictionary of axis labels to apply.  
            For example, {'x':'Bin values','y':'Counts'}
        tickvals_x : array_like or 'auto'
            The major tick values to use for the x-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        tickvals_y : array_like or 'auto'
            The major tick values to use for the y-axis.  Either supplied as an
            explicit array of values, or use 'auto' to automatically calculate
            tick values from an object plotted in the figure.
        **traces_kwargs
            Keyword args to pass to go.Figure.update_traces().  Some examples: 
            marker_color='rgb(158,202,225)', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        
        Returns
        -------
        plotly.express.bar object
        
        Example
        -------
        histdata = np.random.lognormal(size=1000)
        fig = psl.px_histogram_symlog(histdata, 101, shift=1e-2)
        fig.show()
        #
        histdata2 = np.random.randn(1000)
        fig = psl.go_histogram_symlog(histdata2, 101, binwidth_frac=0.9, marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
        fig.show()
        # Histogram with bins spanning the y-axis ('horizontal' orientation)
        fig = psl.px_histogram_symlog(histdata2, 101, orientation='horizontal', labels={'y':'Bin values','x':'Counts'})
        fig.show()
        """
        
        if np.isscalar(bins)==True:
            #If bins is given as a number, calculate the bin edges
            symlogbins = symmetric_logspace(np.nanmin(histdata), np.nanmax(histdata), bins, input_format='linear', shift=shift, base=base)
        else: 
            symlogbins = np.copy(bins)
        
        npcounts, npbins = np.histogram(histdata, bins=symlogbins, density=density)
        #if density==True: 
        #    #As in numpy documentation, density = counts / (sum(counts) * np.diff(bins))
        #    npcounts / (np.nansum(npcounts) * np.diff(npbins))
        
        center_bins = 0.5 * (symlogbins[:-1] + symlogbins[1:])
        center_bins_symlog = symmetric_logarithm(center_bins, shift=shift, base=base)
        widths_symlog = np.diff(center_bins_symlog)
        widths_symlog = np.array( list(widths_symlog) + [widths_symlog[-1],] ) #To make it the same shape
        
        if 'hor' in orientation.lower():
            fig = px.bar( y=center_bins_symlog, x=npcounts, labels=labels , orientation='h')#,labels={'x':'Binned values', 'y':'Counts'})
            #==> NOTE: for px bar and histogram, width means figure width, not bar width.  
            #    Instead, set bar width by updating traces
            fig.update_traces(hovertemplate= "<b>Counts:%{x:.3f}</b>")
            fig.update_traces(width=binwidth_frac*widths_symlog, **traces_kwargs) 
            set_plotly_scale_symmetriclog(fig, xy='y', tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift, base=base)
            fig.update_yaxes(showgrid=True) #These get turned off otherwise by default
        else:
            fig = px.bar( x=center_bins_symlog, y=npcounts, labels=labels )#,labels={'x':'Binned values', 'y':'Counts'})
            #==> NOTE: for px bar and histogram, width means figure width, not bar width.  
            #    Instead, set bar width by updating traces
            fig.update_traces(hovertemplate= "<b>Counts:%{y:.3f}</b>")
            fig.update_traces(width=binwidth_frac*widths_symlog, **traces_kwargs) 
            set_plotly_scale_symmetriclog(fig, xy='x', tickvals_x=tickvals_x, tickvals_y=tickvals_y, shift=shift, base=base)
            fig.update_xaxes(showgrid=True) #These get turned off otherwise by default
        #fig.show()
        return fig




