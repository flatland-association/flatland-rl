from functools import lru_cache

infrastructure_lru_cache_functions = []


def enable_infrastructure_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        infrastructure_lru_cache_functions.append(func)
        return func

    return decorator


def send_infrastructure_data_change_signal_to_reset_lru_cache():
    for func in infrastructure_lru_cache_functions:
        func.cache_clear()
