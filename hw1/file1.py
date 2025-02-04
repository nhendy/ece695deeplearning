#!/usr/bin/env python3
import sys
import random
from string import ascii_lowercase

# Fixed seed to init the RNG
SEED = 0
NUM_NAMES = 10
NAME_LENGTH = 5
WEALTH_RANGE = (0, 1000)

random.seed(SEED)


def _make_random_string(length):
    return ''.join([random.choice(ascii_lowercase) for i in range(length)])


def _make_random_strings(num_strings, length):
    return [_make_random_string(length) for i in range(num_strings)]


def _oneof(container):
    result = False
    for x in container:
        result ^= bool(x)
    return result


class People:
    def __init__(self):
        self.first_names = []
        self.middle_names = []
        self.last_names = []
        # Default behavior is to print in first name first
        # order.
        self.first_name_first = True
        self.last_name_first = False
        self.last_name_with_comma_first = False

    def __iter__(self):
        self._validate_names()
        self._validate_order()
        for i in range(len(self.first_names)):
            yield self._build_name(self.first_names[i], self.middle_names[i],
                                   self.last_names[i])

    def __call__(self):
        for name in sorted(self.last_names):
            print(name)

    def _validate_names(self):
        if not (len(self.first_names) == len(self.middle_names) == len(
                self.last_names)):
            raise ValueError(
                'first names, middle names and last names should be of the same length'
            )

    def _validate_order(self):
        if not _oneof([
                self.first_name_first, self.last_name_first,
                self.last_name_with_comma_first
        ]):
            raise ValueError(
                'One of first_name_first, first_name_first, last_name_with_comma_first must be set'
            )

    def _build_name(self, first_name, middle_name, last_name):
        if self.first_name_first:
            return ' '.join([first_name, middle_name, last_name])
        elif self.last_name_first:
            return ' '.join([last_name, middle_name, first_name])
        elif self.last_name_with_comma_first:
            return ' '.join(['{},'.format(last_name), middle_name, first_name])

    def _reset_order_flags(self):
        self.first_name_first = False
        self.last_name_first = False
        self.last_name_with_comma_first = False


class PeopleWithMoney(People):
    def __init__(self):
        super(PeopleWithMoney, self).__init__()
        self.wealth = [random.randint(*WEALTH_RANGE) for x in range(NUM_NAMES)]

    def __iter__(self):
        for name, wealth in zip(
                super(PeopleWithMoney, self).__iter__(), self.wealth):
            yield "{} {}".format(name, wealth)

    def __call__(self):
        names_with_wealth = [n.split() for n in self.__iter__()]
        names_with_wealth_sorted = sorted(
            names_with_wealth,
            key=lambda name_with_wealth: int(name_with_wealth[-1]))
        for name_wealth in names_with_wealth_sorted:
            print(' '.join(name_wealth))


def _test_name_order(pple):
    for attr in [
            'first_name_first', 'last_name_first', 'last_name_with_comma_first'
    ]:
        pple._reset_order_flags()
        setattr(pple, attr, True)
        for person in pple:
            print(person)


def _test_callable_func(pple):
    pple()


def main():
    pple = People()
    pple_w_money = PeopleWithMoney()
    for attr in ['first_names', 'middle_names', 'last_names']:
        setattr(pple, attr, _make_random_strings(NUM_NAMES, NAME_LENGTH))
        setattr(pple_w_money, attr,
                _make_random_strings(NUM_NAMES, NAME_LENGTH))
    _test_name_order(pple)
    _test_callable_func(pple)
    for per in pple_w_money:
        print(per)
    _test_callable_func(pple_w_money)


if __name__ == "__main__":
    sys.exit(main())
