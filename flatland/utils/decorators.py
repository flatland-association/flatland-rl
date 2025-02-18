from functools import lru_cache

infrastructure_lru_cache_functions = []


# TODO https://github.com/flatland-association/flatland-rl/issues/104 1. revise which caches need to be scoped at all - some seem not to require cache clearing at all. 2. refactor with need to explicitly reset cache in calls dispersed in the whole code base. Use classes to group the cache scope using methodtools for instance method lru caching.
def enable_infrastructure_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        infrastructure_lru_cache_functions.append(func)
        return func

    return decorator


# send_infrastructure_data_change_signal_to_reset_lru_cache() has a problem with instance methods - the methods are not properly cleared.
# Therefore, make sure to use methodtools for instance methods and to instantiantiate new objects to match instance and cache lifecycle.
# See https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-instance-methods-release-object
def send_infrastructure_data_change_signal_to_reset_lru_cache():
    for func in infrastructure_lru_cache_functions:
        func.cache_clear()
