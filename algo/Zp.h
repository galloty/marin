/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>

// The prime finite field with p = 2^64 - 2^32 + 1
class Zp
{
private:
	uint64_t _n;

	static const uint64_t _p = (((1ull << 32) - 1) << 32) + 1;	// 2^64 - 2^32 + 1
	static const uint32_t _mp64 = uint32_t(-1);					// -p mod (2^64) = 2^32 - 1

private:
	Zp reduce(const uint64_t lo, const uint64_t hi) const
	{
		// lo + hi_lo * 2^64 + hi_hi * 2^96 = lo + hi_lo * (2^32 - 1) - hi_hi (mod p)
		Zp r = Zp((lo >= _p) ? lo - _p : lo);
		r += Zp(hi << 32) - uint32_t(hi);		// lhs * rhs < p^2 => hi * (2^32 - 1) < p^2 / 2^32 < p.
		r -= Zp(hi >> 32);
		return r;
	}

public:
	Zp() {}
	Zp(const uint64_t n) : _n(n) {}

	static uint64_t p() { return _p; }

	uint64_t get() const { return _n; }
	void set(const uint64_t n) { _n = n; }

	bool operator!=(const Zp & rhs) const { return (_n != rhs._n); }

	Zp & operator+=(const Zp & rhs) { const uint32_t c = (_n >= _p - rhs._n) ? _mp64 : 0; _n += rhs._n; _n += c; return *this; }
	Zp & operator-=(const Zp & rhs) { const uint32_t c = (_n < rhs._n) ? _mp64 : 0; _n -= rhs._n; _n -= c; return *this; }
	Zp & operator*=(const Zp & rhs)
	{
		const __uint128_t t = _n * __uint128_t(rhs._n);
		*this = reduce(uint64_t(t), uint64_t(t >> 64));
		return *this;
	}

	Zp operator+(const Zp & rhs) const { Zp r = *this; r += rhs; return r; }
	Zp operator-(const Zp & rhs) const { Zp r = *this; r -= rhs; return r; }
	Zp operator*(const Zp & rhs) const { Zp r = *this; r *= rhs; return r; }

	Zp muli() const { return reduce(_n << 48, _n >> (64 - 48)); }	// sqrt(-1) = 2^48 (mod p)

	Zp half() const { return Zp((_n % 2 == 0) ? _n / 2 : ((_n - 1) / 2 + (_p + 1) / 2)); }

	// Add a carry onto the number and return the carry of the first digit_width bits
	Zp digit_adc(const uint8_t digit_width, uint64_t & carry) const
	{
		const uint64_t s = _n + carry;
		const uint64_t c = (s < _n) ? 1 : 0;
		carry = (s >> digit_width) + (c << (64 - digit_width));
		return Zp(s & ((uint32_t(1) << digit_width) - 1));
	}

	Zp pow(const uint64_t e) const
	{
		if (e == 0) return Zp(1);

		Zp r = Zp(1), y = *this;
		for (uint64_t i = e; i != 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;

		return r;
	}

	Zp invert() const { return Zp(pow(_p - 2)); }
	static const Zp root_nth(const size_t n) { return Zp(7).pow((_p - 1) / n); }
};
