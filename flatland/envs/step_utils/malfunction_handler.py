
def get_number_of_steps_to_break(malfunction_generator, np_random):
    if hasattr(malfunction_generator, "generate"):
        malfunction = malfunction_generator.generate(np_random)
    else:
        malfunction = malfunction_generator(np_random)

    return malfunction.num_broken_steps

class MalfunctionHandler:
    def __init__(self):
        self._malfunction_down_counter = 0
        self.num_malfunctions = 0

    def reset(self):
        self._malfunction_down_counter = 0
        self.num_malfunctions = 0
    
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
        # Only set new malfunction value if old malfunction is completed
        if self._malfunction_down_counter == 0:
            self._malfunction_down_counter = val
            self.num_malfunctions += 1

    def generate_malfunction(self, malfunction_generator, np_random):
        num_broken_steps = get_number_of_steps_to_break(malfunction_generator, np_random)
        self._set_malfunction_down_counter(num_broken_steps)

    def update_counter(self):
        if self._malfunction_down_counter > 0:
            self._malfunction_down_counter -= 1

    def __repr__(self):
        return f"malfunction_down_counter: {self._malfunction_down_counter} \
                in_malfunction: {self.in_malfunction} \
                num_malfunctions: {self.num_malfunctions}"

    def to_dict(self):
        return {"malfunction_down_counter": self._malfunction_down_counter,
                "num_malfunctions": self.num_malfunctions}
    
    def from_dict(self, load_dict):
        self._malfunction_down_counter = load_dict['malfunction_down_counter']
        self.num_malfunctions = load_dict['num_malfunctions']

    def __eq__(self, other):
        return self._malfunction_down_counter == other._malfunction_down_counter and \
               self.num_malfunctions == other.num_malfunctions

    

    

