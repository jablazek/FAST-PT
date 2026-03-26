.. _changelog:

Changelog / Migration Guide
===========================

This document details the changes between the original FAST-PT implementation and the current version.

For previous versions' changes, see the :download:`complete changelog history </usr_manual.pdf>`.

NOTE: FAST-PT 4.0 is completely backwards compatible with FAST-PT 3.0. However, adjustments to your code may be required to take advantage of the new features and performance improvements.

Major Changes
-------------

* Caching: FAST-PT now caches individual terms and intermediate calculations to speed up computation. 
* FPTHandler: The handler class has been introduced to improve the the user's ability to manage FAST-PT and provide many new convenience features that compliment the FAST-PT class. NOTE: the FPTHandler class is not a replacement for the FAST-PT class, but rather a wrapper that provides additional functionality. It is not necessary for computation.
* To_do list: The to_do list is no longer needed to initialize FAST-PT. The terms will now be calculated as needed and stored as a property of the FAST-PT class.


Minor Changes
-------------

* Simple flag: A new "simple" kwarg has been added to FAST-PT which will instead initialize an instance of FAST-PT simple.
* Private k: The input k is now "private" after initialization via Python's name mangling. This means that the user cannot change the value of k after initialization but can still access the value of k.
* Gamma functions cache: A separate (and simpler) caching system has been implemented to cache gamma functions and save time on the calculation of the X terms.
* Parameter validation: The parameters P, P_window, and C_window are now validated at every function call to ensure that they have the proper traits needed for the calculation. This is done to prevent errors from propagating through the code and causing issues later on.
* N_pad default: If no n_pad is provided during initialization, the default value is now set to 0.5 * len(k).
* Nu deprecation: The nu parameter is now deprecated as it is no longer needed for initialization. It will default to -2 unless a different nu value is needed in which case it will be calculated internally.
* One_loop_dd return: One_loop_dd will only return P_1loop, Ps. Previously returned the dd_bias terms as well however this was contingent on the to_do list which is being deprecated. 
* Cleft_QR: The Cleft_QR function has been removed due to missing internal functions.


Performance Improvements
------------------------

The improvement in performance of FAST-PT is going to varry largely with your use case. However, about half of the calculation done for most terms was redundant granting a two times speedup do to the new caching system.
FAST-PT also now calculates terms in a modular format. This means that the user can now choose to calculate only the terms they need, rather than all of the terms grouped into one FAST-PT function. 
This is done by using the FPTHandler class and the get method, or by calling compute_term with the necessary parameters for each term. 
This will greatly improve the performance of each FAST-PT function if your use case only requires a select few terms.


Description of Caching System
-----------------------------

Caching in Fast-PT is done via a CacheManager_ object that is initialized with the FAST-PT class. This cache tracks various different "layers" of the calculation of Fast-PT terms. These layers include:

* Individual Power Spectra: Fast-PT functions return a tuple of multiple power spectra, each of which is cached individually.
* Jk Scalar and Tensor Calculations: Most Fast-PT terms require the calculation of the Jk scalar or tensor functions. Some terms have identical parameters that are passed to these functions, so they are cached individually as well.
* Fourier Coefficients: Fourier coefficients of the user provided power spectra are cached and then used in the Jk functions.
* Convolutions: This is the convolution of the power spectrum fourier coefficients and the relevant k grid quantities (X terms). The convolution function is called with fourier coefficients and k grid quantities, which are dependent on the user provided k and P and cached individually.

There is also a separate form of caching used for the k grid quantities (X terms) and the gamma functions used to calculate them. The gamma functions are cached outside the Fast-PT object as they will not be frequently changed, while the k grid quantities are cached as properties of the Fast-PT object and can be accessed via `fastpt_instance.X_IA_A` for example.
In total, there are three forms of caching used in Fast-PT 4.0, two of which are stored within the object. The gamma functions cache will persist as long as the program runs.

This multi-tiered caching system allows Fast-PT to avoid redundant calculations both on individual power spectra terms and the intermediate calculations that are needed to compute them. 
To avoid the cache from growing too large, a "dump_cache" flag is provided in initialization that, when True, will clear the cache when a new power spectra is inputted by the user. The user is also able to specify (during Fast-PT initialization) a maximum cache size in mb. This will evict cached items randomly (in linear time) to avoid slowing down the total computation time. However, the cache size limit is meant as a safeguard and should not be treated as a form of memory management for the program. 

.. _CacheManager: https://github.com/jablazek/FAST-PT/tree/master/fastpt/core/CacheManager.py


Additional Notes
------------------

* The 1loop term (one_loop_dd[0]) currently does not pass np.allclose due to numerical noise when compared with FAST-PT 3 when a C_window of 0.63 or less is provided. This only occurs at the very low k end and is negligible. This issue is currently being investigated and will be fixed in a future release.