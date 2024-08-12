
class FrequencyTable:
    def __init__(self, freqs):
        self.freqs = freqs
        self.total = sum(freqs)

    def get_frequency(self, symbol):
        return self.freqs[symbol]

    def get_total(self):
        return self.total

    def get_cumulative_frequency(self, symbol):
        return sum(self.freqs[:symbol])

    def get_symbol_limit(self):
        return len(self.freqs)


class BitInputStream:
    def __init__(self, file):
        self.input = file
        self.current_byte = 0
        self.num_bits_remaining = 0

    def read(self):
        if self.num_bits_remaining == 0:
            raw = self.input.read(1)
            if len(raw) == 0:
                return -1  # 表示文件末尾或读取完毕  Indicates the end of the file or completion of reading
            self.current_byte = raw[0]
            self.num_bits_remaining = 8
        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1

    def close(self):
        self.input.close()
        self.current_byte = 0
        self.num_bits_remaining = 0

class BitOutputStream:
    def __init__(self, file):
        self.output = file
        self.current_byte = 0
        self.num_bits_filled = 0

    def write(self, bit):
        if bit not in [0, 1]:
            raise ValueError("Bit must be 0 or 1")
        self.current_byte = (self.current_byte << 1) | bit
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            self.output.write(bytes([self.current_byte]))
            self.current_byte = 0
            self.num_bits_filled = 0

    def finish(self):
        if self.num_bits_filled > 0:
            self.current_byte <<= (8 - self.num_bits_filled)
            self.output.write(bytes([self.current_byte]))
            self.current_byte = 0
            self.num_bits_filled = 0

    def close(self):
        self.output.close()
        self.current_byte = 0
        self.num_bits_remaining = 0  
            
class ArithmeticDecoder:
    def __init__(self, num_bits, bitin):
        self.num_bits = num_bits
        self.bitin = bitin
        self.low = 0
        self.high = (1 << num_bits) - 1
        self.value = 0
        for _ in range(num_bits):
            self.value = (self.value << 1) | self.read_bit()
        #print(f"Initial value: {self.value}")

    def read_bit(self):
        bit = self.bitin.read()
        if bit == -1:  
            bit = 0
        return bit

    def decode(self, freqs):
        total = freqs.get_total()
        range = self.high - self.low + 1
        offset = self.value - self.low
        cum = (offset + 1) * total - 1
        cum //= range

        symbol = 0
        while freqs.get_cumulative_frequency(symbol + 1) <= cum:
            symbol += 1

        sym_low = freqs.get_cumulative_frequency(symbol)
        sym_high = freqs.get_cumulative_frequency(symbol + 1)
        self.low = self.low + range * sym_low // total
        self.high = self.low + range * sym_high // total - 1

        #print(f"Decoding symbol: {symbol}, low: {self.low}, high: {self.high}, value: {self.value}")

        while True:
            if self.high < (1 << (self.num_bits - 1)):
                pass  
            elif self.low >= (1 << (self.num_bits - 1)):
                self.value -= (1 << (self.num_bits - 1))
                self.low -= (1 << (self.num_bits - 1))
                self.high -= (1 << (self.num_bits - 1))
            elif self.low >= (1 << (self.num_bits - 2)) and self.high < (1 << (self.num_bits - 1)):
                self.value -= (1 << (self.num_bits - 2))
                self.low -= (1 << (self.num_bits - 2))
                self.high -= (1 << (self.num_bits - 2))
            else:
                break
            self.low <<= 1
            self.high <<= 1
            self.high |= 1
            self.value = (self.value << 1) | self.read_bit()
            #print(f"Adjusting: low: {self.low}, high: {self.high}, value: {self.value}")

        return symbol

class ArithmeticEncoder:
    def __init__(self, num_bits, bitout):
        self.num_bits = num_bits
        self.bitout = bitout
        self.low = 0
        self.high = (1 << num_bits) - 1
        self.underflow = 0

    def write(self, freqs, symbol):
        total = freqs.get_total()
        sym_low = freqs.get_cumulative_frequency(symbol)
        sym_high = freqs.get_cumulative_frequency(symbol + 1)
        range = self.high - self.low + 1
        self.high = self.low + (range * sym_high // total) - 1
        self.low = self.low + (range * sym_low // total)

        while True:
            if self.high < (1 << (self.num_bits - 1)):
                self.bitout.write(0)
                self._emit_underflow()
            elif self.low >= (1 << (self.num_bits - 1)):
                self.bitout.write(1)
                self.low -= (1 << (self.num_bits - 1))
                self.high -= (1 << (self.num_bits - 1))
                self._emit_underflow()
            elif self.low >= (1 << (self.num_bits - 2)) and self.high < (1 << (self.num_bits - 1)):
                self.underflow += 1
                self.low -= (1 << (self.num_bits - 2))
                self.high -= (1 << (self.num_bits - 2))
            else:
                break
            self.low <<= 1
            self.high <<= 1
            self.high |= 1

    def _emit_underflow(self):
        while self.underflow > 0:
            self.bitout.write(self.high >> (self.num_bits - 1))
            self.underflow -= 1

    def finish(self):
        self.bitout.write(self.low >> (self.num_bits - 2))
        self.bitout.finish()
