
def get_number_of_steps_to_break(malfunction_generator, np_random):
    if hasattr(malfunction_generator, "generate"):
        malfunction = malfunction_generator.generate(np_random)
    else:
        malfunction = malfunction_generator(np_random)

    return malfunction.num_broken_steps

class MalfunctionHandler:
    def __init__(self):
        self._malfunction_down_counter = 0
    
    @property
    def in_malfunction(self):
        return self._malfunction_down_counter > 0
    
    @property
    def malfunction_counter_complete(self):
        return self._malfunction_down_counter == 0

    @property
    def malfunction_down_counter(self):
        return self._malfunction_down_counter

    @malfunction_down_counter.setter
    def malfunction_down_counter(self, val):
        self._set_malfunction_down_counter(val)

    def _set_malfunction_down_counter(self, val):
        if val < 0:
            raise ValueError("Cannot set a negative value to malfunction down counter")
        self._malfunction_down_counter = val

    def generate_malfunction(self, malfunction_generator, np_random):
        num_broken_steps = get_number_of_steps_to_break(malfunction_generator, np_random)
        self._set_malfunction_down_counter(num_broken_steps)

    def update_counter(self):
        if self._malfunction_down_counter > 0:
            self._malfunction_down_counter -= 1

    def to_dict(self):
        return {"malfunction_down_counter": self._malfunction_down_counter}
    
    def from_dict(self, load_dict):
        self._malfunction_down_counter = load_dict['malfunction_down_counter']


    

    

