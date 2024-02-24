#####################
pysymlog 
#####################



=====================
Installation
=====================

* With pip:

.. code-block:: shell

    pip install pysymlog


* Standalone:

Alternatively, can download just the pysymlog.py to run in standalone mode -- this may be a better option if pip installation is not possible on your machine.  It's possible to simply run/execfile pysymlog.py in your python session to load its functions.  You could also add the file location to your local path:

(for .bashrc)

.. code-block:: shell

    export PYTHONPATH=$PYTHONPATH:/path/to/dir/with/obs/

(for .cshrc)

.. code-block:: shell

    setenv PYTHONPATH ${PYTHONPATH}:/path/to/dir/with/obs/


=====================
Basic Usage
=====================

.. code-block:: python

    import pysymlog as psl
    
    #to use the matplotlip functionality:
    psl.register_mpl()
    
    #to use the plotly functionality:
    psl.register_plotly()
    

Or, as a standalone script, download and cd to that directory, and type: 

.. code-block:: shell
    
    python pysymlog.py


=====================
Examples
=====================

See the main examples tutorial jupyter notebook.
View on nbviewer here:

 `https://github.com/pjcigan/pysymlog/blob/master/examples/tutorial_mpl.ipynb <https://github.com/pjcigan/obsplanning/blob/master/examples/tutorial_mpl.ipynb>`_



