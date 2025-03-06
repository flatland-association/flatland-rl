from functools import lru_cache

infrastructure_lru_cache_functions = []


# TODO https://github.com/flatland-association/flatland-rl/issues/104 1. revise which caches need to be scoped at all - some seem not to require cache clearing at all. 2. refactor with need to explicitly reset cache in calls dispersed in the whole code base. Use classes to group the cache scope by overriding eq/hash for instance method lru caching (see https://docs.python.org/3/faq/programming.html#how-do-i-cache-method-calls)
def enable_infrastructure_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        infrastructure_lru_cache_functions.append(func)
        return func

    return decorator


# send_infrastructure_data_change_signal_to_reset_lru_cache() has a problem with instance methods - the methods are not properly cleared by it.
# Therefore, make sure to override eq/hash to control cache lifecycle for instance method lru caching (see https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-instance-methods-release-object and https://docs.python.org/3/faq/programming.html#how-do-i-cache-method-calls)
def send_infrastructure_data_change_signal_to_reset_lru_cache():
    for func in infrastructure_lru_cache_functions:
        func.cache_clear()
