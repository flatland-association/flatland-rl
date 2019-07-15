========================================
Frequently Asked Questions (FAQs)
========================================

-   I get a runtime error with `Click` complaining about the encoding

    .. code-block:: python

        RuntimeError('Click will abort further execution because Python 3 \
        was configured to use ASCII as encoding for ...sk_SK.UTF-8, \
        sl_SI.UTF-8, sr_YU.UTF-8, sv_SE.UTF-8, tr_TR.UTF-8, \
        uk_UA.UTF-8, zh_CN.UTF-8, zh_HK.UTF-8, zh_TW.UTF-8')

    This can be solved by :

    .. code-block:: bash

        export LC_ALL=en_US.utf-8
        export LANG=en_US.utf-8

-   We use `importlib-resources`_ to read from local files.
    Sample usages:

    .. code-block:: python

        from importlib_resources import path

        with path(package, resource) as file_in:
            new_grid = np.load(file_in)
            
    And:

    .. code-block:: python

        from importlib_resources import read_binary
        
        load_data = read_binary(package, resource)
        self.set_full_state_msg(load_data)
        

    .. _importlib-resources: https://importlib-resources.readthedocs.io/en/latest/

    Renders the scene into a image (screenshot)
    .. code-block:: python

    renderer.gl.save_image("filename.bmp")

