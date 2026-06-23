# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.

class BitType:
    def __init__(self, bits, signed, name=None):
        self.bits = bits
        self.signed = signed
        if name is not None:
            self.name = name
        else:
            self.update_name()
    
    @property
    def upper_bound(self):
        if self.signed:
            return 2 ** (self.bits - 1) - 1
        else:
            return 2 ** self.bits - 1
    
    @property
    def lower_bound(self):
        if self.signed:
            return -2 ** (self.bits - 1)
        else:
            return 0
    
    @property
    def range(self):
        return 2 ** self.bits

    def update_name(self):
        self.name = ''
        if self.signed:
            self.name += 'uint'
        else:
            self.name += 'int'
        self.name += '{}'.format(self.bits)

BIT_TYPE_LIST = [
    BitType(4, False, 'uint4'),
    BitType(8, True, 'int8'),
    BitType(8, False, 'uint8')
]
BIT_TYPE_DICT = {bit_type.name: bit_type for bit_type in BIT_TYPE_LIST}