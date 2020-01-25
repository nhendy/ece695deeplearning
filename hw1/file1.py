#!/usr/bin/env python3
import sys
import random
from string import ascii_letters
import functools
from collections.abc import Iterable

# Fixed seed to init the RNG
SEED = 0
NUM_NAMES = 10
NAME_LENGTH = 5

random.seed(SEED)


def _make_random_string(length):
    rand_char = lambda: ascii_letters[random.randint(0, length - 1) % len(
        ascii_letters)]
    return ''.join([rand_char() for i in range(length)])


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

    # @property
    # def first_name_first(self):
    #     return self.first_name_first

    # @property
    # def last_name_first(self):
    #     return self.last_name_first

    # @property
    # def last_name_with_comma_first(self):
    #     return self.last_name_with_comma_first

    # @first_name_first.setter
    # def first_name_first(self, name):
    #     self._reset_order_flags()
    #     self.first_name_first = name

    # @last_name_first.setter
    # def last_name_first(self, name):
    #     self._reset_order_flags()
    #     self.first_name_first = name

    # @last_name_with_comma_first.setter
    # def last_name_first(self, name):
    #     self._reset_order_flags()
    #     self.last_name_with_comma_first = name


def _test_name_order(pple):
    for attr in [
            'last_name_with_comma_first', 'last_name_first', 'first_name_first'
    ]:
        pple._reset_order_flags()
        setattr(pple, attr, True)
        for person in pple:
            print(person)


def main():
    pple = People()
    for attr in ['first_names', 'middle_names', 'last_names']:
        setattr(pple, attr, _make_random_strings(NUM_NAMES, NAME_LENGTH))
    _test_name_order(pple)


if __name__ == "__main__":
    sys.exit(main())
