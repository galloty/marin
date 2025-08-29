/*
Copyright 2025, Yves Gallot

The multiplication is based on:
 - An Irrational Base Discrete Weighted Transform. x^p - 1 => x^n - 1, where n = 2^m or 5*2^m.
 - A recursive polynomial factorization approach. x^{2n} - r^2 => x^n - r and x^n + r; x^{5n} - r^5 => 5 polynomials of the form x^n - r_i.
The finite field is Z/pZ, where p = 2^64 - 2^32 + 1.
Butterfly sizes are radix-4 and radix-5.

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>

// The prime finite field with p = 2^64 - 2^32 + 1
class Zp
{
private:
	uint64_t _n;

	static const uint64_t _p = (((1ull << 32) - 1) << 32) + 1;	// 2^64 - 2^32 + 1
	static const uint32_t _mp64 = uint32_t(-1);					// -p mod (2^64) = 2^32 - 1

private:
	Zp reduce(const __uint128_t t) const
	{
		const uint64_t hi = uint64_t(t >> 64), lo = uint64_t(t);

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
	Zp & operator*=(const Zp & rhs) { *this = reduce(_n * __uint128_t(rhs._n)); return *this; }

	Zp operator+(const Zp & rhs) const { Zp r = *this; r += rhs; return r; }
	Zp operator-(const Zp & rhs) const { Zp r = *this; r -= rhs; return r; }
	Zp operator*(const Zp & rhs) const { Zp r = *this; r *= rhs; return r; }

	Zp muli() const { return reduce(__uint128_t(_n) << 48); }	// sqrt(-1) = 2^48 (mod p)

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

class Mersenne
{
private:
	const size_t _n;
	const Zp _inv_n;
	std::vector<Zp> _reg;	// the weighted representation of R0, R1, ...
	std::vector<Zp> _root;
	std::vector<Zp> _invroot;
	std::vector<Zp> _weight;
	std::vector<Zp> _invweight;
	std::vector<uint8_t> _digit_width;
	bool _even_exponent;

private:
	static constexpr size_t transform_size(const uint32_t exponent)
	{
		// Make sure the transform is long enough so that each 'digit' can't overflow after the convolution.
		uint32_t w = 0, log2_n = 1, log2_n5 = 0;
		do
		{
			++log2_n;
			// digit-width is w or w + 1
			w = exponent >> log2_n;
		// The condition is n * (2^{w + 1} - 1)^2 < 2^64 - 2^32 + 1.
		// If (w + 1) * 2 + log2(n) = 63 then n * (2^{w + 1} - 1)^2 < n * (2^{w + 1})^2 = 2^63 < 2^64 - 2^32 + 1.
		} while ((w + 1) * 2 + log2_n >= 64);

		do
		{
			++log2_n5;
			w = exponent / (5u << log2_n5);
		// log2(5) ~ 2.3219 < 2.4
		} while ((w + 1) * 2 + (log2_n5 + 2.4) >= 64);

		return std::min(size_t(1) << log2_n, size_t(5) << log2_n5);	// must be >= 4
	}

	// Bit-reversal permutation
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	// Radix-4
	static void fwd4(Zp & x0, Zp & x1, Zp & x2, Zp & x3, const Zp & r1, const Zp & r20, const Zp & r21)
	{
		const Zp u0 = x0, u2 = r1 * x2, u1 = r20 * x1, u3 = r21 * x3;
		const Zp v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = (u1 - u3).muli();
		x0 = v0 + v1; x1 = v0 - v1; x2 = v2 + v3; x3 = v2 - v3;
	}

	// Inverse radix-4
	static void bck4(Zp & x0, Zp & x1, Zp & x2, Zp & x3, const Zp & ri1, const Zp & ri20, const Zp & ri21)
	{
		const Zp u0 = x0, u1 = x1, u2 = x2, u3 = x3;
		const Zp v0 = u0 + u1, v1 = u0 - u1, v2 = u3 + u2, v3 = (u3 - u2).muli();
		x0 = v0 + v2; x2 = ri1 * (v0 - v2); x1 = ri20 * (v1 + v3); x3 = ri21 *(v1 - v3);
	}

	// 2 x radix-2
	static void fwd22(Zp & x0, Zp & x1, Zp & x2, Zp & x3, const Zp & r)
	{
		const Zp u0 = x0, u2 = r * x2, u1 = x1, u3 = r * x3;
		x0 = u0 + u2; x2 = u0 - u2; x1 = u1 + u3; x3 = u1 - u3;
	}

	// 2 x inverse radix-2
	static void bck22(Zp & x0, Zp & x1, Zp & x2, Zp & x3, const Zp & ri)
	{
		const Zp u0 = x0, u2 = x2, u1 = x1, u3 = x3;
		x0 = u0 + u2; x2 = ri * (u0 - u2); x1 = u1 + u3; x3 = ri * (u1 - u3);
	}

	// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.
	static void butterfly5(Zp & a0, Zp & a1, Zp & a2, Zp & a3, Zp & a4)
	{
		static const Zp K = Zp::root_nth(5), K2 = K * K, K3 = K * K2, K4 = K2 * K2;
		static const Zp cosu = (K + K4).half(), isinu = (K - K4).half(), cos2u = (K2 + K3).half(), isin2u = (K2 - K3).half();
		static const Zp F1 = (cosu + cos2u).half() - Zp(1), F2 = (cosu - cos2u).half(), F3 = isinu + isin2u, F4 = isin2u, F5 = isinu - isin2u;

		const Zp s1 = a1 + a4, s2 = a1 - a4, s3 = a3 + a2, s4 = a3 - a2;
		const Zp s5 = s1 + s3, s6 = s1 - s3, s7 = s2 + s4, s8 = s5 + a0;
		const Zp m0 = s8;
		const Zp m1 = F1 * s5, m2 = F2 * s6, m3 = F3 * s2, m4 = F4 * s7, m5 = F5 * s4;
		const Zp s9 = m0 + m1, s10 = s9 + m2, s11 = s9 - m2, s12 = m3 - m4;
		const Zp s13 = m4 + m5, s14 = s10 + s12, s15 = s10 - s12, s16 = s11 + s13;
		const Zp s17 = s11 - s13;
		a0 = m0; a1 = s14; a2 = s16; a3 = s17; a4 = s15;
	}

	// Radix-5
	static void fwd5(Zp & x0, Zp & x1, Zp & x2, Zp & x3, Zp & x4, const Zp & r, const Zp & r2)
	{
		const Zp r3 = r * r2, r4 = r2 * r2;
		Zp a0 = x0, a1 = r * x1, a2 = r2 * x2, a3 = r3 * x3, a4 = r4 * x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// Inverse radix-5
	static void bck5(Zp & x0, Zp & x1, Zp & x2, Zp & x3, Zp & x4, const Zp & ri, const Zp & ri2)
	{
		Zp a0 = x0, a4 = x1, a3 = x2, a2 = x3, a1 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		const Zp ri3 = ri * ri2, ri4 = ri2 * ri2;
		x0 = a0; x1 = ri * a1; x2 = ri2 * a2; x3 = ri3 * a3; x4 = ri4 * a4;
	}

	// Transform
	void forward(Zp * const x) const
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n, s5 = (n % 5 == 0) ? 5 : 4;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r4 = &_root.data()[n5 / 2];

		// Radix-4
		for (size_t s = _even_exponent ? 1 : 2, m = n / 4 / s; m >= s5; m /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp r1 = r2[j], r20 = r4[2 * j + 0], r21 = r4[2 * j + 1];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 4 * m * j + i;
					Zp x0 = x[k + 0 * m], x1 = x[k + 1 * m], x2 = x[k + 2 * m], x3 = x[k + 3 * m];
					fwd4(x0, x1, x2, x3, r1, r20, r21);
					x[k + 0 * m] = x0; x[k + 1 * m] = x1; x[k + 2 * m] = x2; x[k + 3 * m] = x3;
				}
			}
		}
	}

	// Inverse Transform
	void backward(Zp * const x) const
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n, s5 = (n % 5 == 0) ? 5 : 4;
		const Zp * const r2i = &_invroot.data()[0];
		const Zp * const r4i = &_invroot.data()[n5 / 2];

		// Inverse radix-4
		for (size_t m = s5, s = n / 4 / m; m <= n / 4; m *= 4, s /= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp ri1 = r2i[j], ri20 = r4i[2 * j + 0], ri21 = r4i[2 * j + 1];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 4 * m * j + i;
					Zp x0 = x[k + 0 * m], x1 = x[k + 1 * m], x2 = x[k + 2 * m], x3 = x[k + 3 * m];
					bck4(x0, x1, x2, x3, ri1, ri20, ri21);
					x[k + 0 * m] = x0; x[k + 2 * m] = x2; x[k + 1 * m] = x1; x[k + 3 * m] = x3;
				}
			}
		}
	}

	// Radix-4
	void forward4(Zp * const x) const
	{
		const size_t n4 = _n / 4;
		const Zp * const r2 = &_root.data()[0];

		for (size_t j = 0; j < n4; ++j)
		{
			const size_t k = 4 * j;
			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd22(x0, x1, x2, x3, r2[j]);
			x[k + 0] = x0; x[k + 2] = x2; x[k + 1] = x1; x[k + 3] = x3;
		}
	}

	// Radix-4, square, inverse radix-4
	void sqr4(Zp * const x) const
	{
		const size_t n4 = _n / 4;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r2i = &_invroot.data()[0];

		for (size_t j = 0; j < n4; ++j)
		{
			const Zp r = r2[j];
			const size_t k = 4 * j;
			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd22(x0, x1, x2, x3, r);
			const Zp t0 = x0 * x0 + x1 * x1 * r; x1 *= x0 + x0; x0 = t0;
			const Zp t2 = x2 * x2 - x3 * x3 * r; x3 *= x2 + x2; x2 = t2;
			bck22(x0, x1, x2, x3, r2i[j]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
		}
	}

	// Radix-4, mul, inverse radix-4
	void mul4(Zp * const x, const Zp * const y) const
	{
		const size_t n4 = _n / 4;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r2i = &_invroot.data()[0];

		for (size_t j = 0; j < n4; ++j)
		{
			const Zp r = r2[j];
			const size_t k = 4 * j;
			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd22(x0, x1, x2, x3, r);
			const Zp y0 = y[k + 0], y1 = y[k + 1], y2 = y[k + 2], y3 = y[k + 3];
			const Zp t0 = x0 * y0 + x1 * y1 * r; x1 = x0 * y1 + x1 * y0; x0 = t0;
			const Zp t2 = x2 * y2 - x3 * y3 * r; x3 = x2 * y3 + x3 * y2; x2 = t2;
			bck22(x0, x1, x2, x3, r2i[j]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
		}
	}

	// Radix-5
	void forward5(Zp * const x) const
	{
		const size_t n5 = _n / 5;
		const Zp * const r5 = &_root.data()[n5];

		for (size_t j = 0; j < n5; ++j)
		{
			const size_t k = 5 * j;
			fwd5(x[k + 0], x[k + 1], x[k + 2], x[k + 3], x[k + 4], r5[2 * j + 0], r5[2 * j + 1]);
		}
	}

	// Radix-5, square, inverse radix-5
	void sqr5(Zp * const x) const
	{
		const size_t n5 = _n / 5;
		const Zp * const r5 = &_root.data()[n5];
		const Zp * const r5i = &_invroot.data()[n5];

		for (size_t j = 0; j < n5; ++j)
		{
			const size_t k = 5 * j;
			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3], x4 = x[k + 4];
			fwd5(x0, x1, x2, x3, x4, r5[2 * j + 0], r5[2 * j + 1]);
			x0 *= x0; x1 *= x1; x2 *= x2; x3 *= x3; x4 *= x4;
			bck5(x0, x1, x2, x3, x4, r5i[2 * j + 0], r5i[2 * j + 1]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3; x[k + 4] = x4;
		}
	}

	// Radix-5, mul, inverse radix-5
	void mul5(Zp * const x, const Zp * const y) const
	{
		const size_t n5 = _n / 5;
		const Zp * const r5 = &_root.data()[n5];
		const Zp * const r5i = &_invroot.data()[n5];

		for (size_t j = 0; j < n5; ++j)
		{
			const size_t k = 5 * j;
			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3], x4 = x[k + 4];
			fwd5(x0, x1, x2, x3, x4, r5[2 * j + 0], r5[2 * j + 1]);
			x0 *= y[k + 0]; x1 *= y[k + 1]; x2 *= y[k + 2]; x3 *= y[k + 3]; x4 *= y[k + 4];
			bck5(x0, x1, x2, x3, x4, r5i[2 * j + 0], r5i[2 * j + 1]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3; x[k + 4] = x4;
		}
	}

	void carry_weight(Zp * const x) const
	{
		const size_t n = _n;
		const Zp inv_n = _inv_n;
		const Zp * const w = _weight.data();
		const Zp * const wi = _invweight.data();
		const uint8_t * const width = _digit_width.data();

		if (_even_exponent)
		{
			uint64_t c = 0;
			for (size_t k = 0; k < n; ++k)
			{
				const Zp u = x[k] * inv_n * wi[k];
				x[k] = u.digit_adc(width[k], c) * w[k];
			}

			while (c != 0)
			{
				for (size_t k = 0; k < n; ++k)
				{
					const Zp u = x[k] * wi[k];
					x[k] = u.digit_adc(width[k], c) * w[k];
					if (c == 0) break;
				}
			}
		}
		else
		{
			uint64_t c0 = 0, c1 = 0;
			for (size_t k = 0; k < n / 2; ++k)
			{
				const Zp u0 = x[k + 0 * n / 2], u1 = x[k + 1 * n / 2];
				// inverse radix-2
				const Zp v0 = (u0 + u1) * inv_n * wi[k + 0 * n / 2];
				const Zp v1 = (u0 - u1) * inv_n * wi[k + 1 * n / 2];
				const Zp s0 = v0.digit_adc(width[k + 0 * n / 2], c0) * w[k + 0 * n / 2];
				const Zp s1 = v1.digit_adc(width[k + 1 * n / 2], c1) * w[k + 1 * n / 2];
				// radix-2
				x[k + 0 * n / 2] = s0 + s1; x[k + 1 * n / 2] = s0 - s1;
			} 

			while ((c0 != 0) || (c1 != 0))
			{
				const uint64_t t = c0; c0 = c1; c1 = t;	// swap
				for (size_t k = 0; k < n / 2; ++k)
				{
					const Zp u0 = x[k + 0 * n / 2], u1 = x[k + 1 * n / 2];
					// inverse radix-2
					const Zp v0 = (u0 + u1).half() * wi[k + 0 * n / 2];
					const Zp v1 = (u0 - u1).half() * wi[k + 1 * n / 2];
					const Zp s0 = v0.digit_adc(width[k + 0 * n / 2], c0) * w[k + 0 * n / 2];
					const Zp s1 = v1.digit_adc(width[k + 1 * n / 2], c1) * w[k + 1 * n / 2];
					// radix-2
					x[k + 0 * n / 2] = s0 + s1; x[k + 1 * n / 2] = s0 - s1;
					if ((c0 == 0) && (c1 == 0)) break;
				}
			}
		}
	}

public:
	Mersenne(const uint32_t q) : _n(transform_size(q)), _inv_n(Zp((_n % 5 == 0) ? _n : _n / 2).invert()),
		_reg(3 * _n),	// allocate 3 registers
		_root(_n), _invroot(_n),
		_weight(_n), _invweight(_n), _digit_width(_n)
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n;

		size_t m = n5; for (; m > 1; m /= 4);
		_even_exponent = (m == 1);

		// Roots

		Zp * const root = _root.data();
		Zp * const invroot = _invroot.data();

		Zp * const r2 = &root[0];
		Zp * const r2i = &invroot[0];

		const Zp r = Zp::root_nth(2 * n5 / 2);
		for (size_t j = 0; j < n5 / 2; ++j)
		{
			const Zp rj = r.pow(bitrev(j, n5 / 2));
			r2[j] = rj; r2i[j] = rj.invert();
		}

		Zp * const r4 = &root[n5 / 2];
		Zp * const r4i = &invroot[n5 / 2];

		for (size_t j = 0; j < n5 / 4; ++j)
		{
			r4[2 * j + 0] = r2[2 * j]; r4i[2 * j + 0] = r2i[2 * j];
			r4[2 * j + 1] = r2[j] * r2[2 * j]; r4i[2 * j + 1] = r2i[j] * r2i[2 * j];
		}

		if (n % 5 == 0)
		{
			Zp * const r5 = &root[n5];
			Zp * const r5i = &invroot[n5];

			const Zp r = Zp::root_nth(5 * n5);
			for (size_t j = 0; j < n5; ++j)
			{
				const Zp rj = r.pow(bitrev(j, n5)), rji = rj.invert();
				r5[2 * j + 0] = rj; r5i[2 * j + 0] = rji;
				r5[2 * j + 1] = rj * rj; r5i[2 * j + 1] = rji * rji;
			}
		}

		// Weights

		Zp * const w = _weight.data();
		Zp * const wi = _invweight.data();
		uint8_t * const width = _digit_width.data();

		// n-th root of two
		const Zp nr2 = Zp(554).pow((Zp::p() - 1) / 192 / n);

		const uint32_t q_n = q / uint32_t(n);

		wi[0] = w[0] = Zp(1);

		uint32_t ceil_qjm1_n = 0;
		for (size_t j = 1; j <= n; ++j)
		{
			const uint64_t qj = q * uint64_t(j);
			// ceil(a / b) = floor((a - 1) / b) + 1
			const uint32_t ceil_qj_n = uint32_t((qj - 1) / n + 1);

			// bit position for digit[i] is ceil(qj / n)
			const uint32_t c = ceil_qj_n - ceil_qjm1_n;
			if ((c != q_n) && (c != q_n + 1)) throw;
			width[j - 1] = uint8_t(c);

			if (j == n) break;

			// weight is 2^[ceil(qj / n) - qj / n]
			// e = (ceil(qj / n).n - qj) / n
			// qj = k * n => e = 0
			// qj = k * n + r, r > 0 => ((k + 1).n - k.n + r) / n = (n - r) / n
			const uint32_t r = uint32_t(qj % n);
			const Zp nr2r = (r != 0) ? nr2.pow(n - r) : Zp(1);
			w[j] = nr2r; wi[j] = nr2r.invert();
			ceil_qjm1_n = ceil_qj_n;
		}
	}

	virtual ~Mersenne() {}

	size_t get_length() const { return _n; }

	enum class Reg : uint32_t { R0 = 0, R1 = 1, R2 = 2 };

	void set(const Reg dst, const uint64_t a)
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(dst) * n];

		x[0] = Zp(a);	// weight[0] = 1
		for (size_t k = 1; k < n; ++k) x[k] = Zp(0);

		// radix-2
		if (!_even_exponent) x[n / 2] = x[0];
	}

	void copy(const Reg dst, const Reg src)
	{
		const size_t n = _n;
		const Zp * const x = &_reg.data()[uint32_t(src) * n];
		Zp * const y = &_reg.data()[uint32_t(dst) * n];

		for (size_t k = 0; k < n; ++k) y[k] = x[k];
	}

	bool is_equal(const Reg src1, const Reg src2) const
	{
		const size_t n = _n;
		const Zp * const x = &_reg.data()[uint32_t(src1) * n];
		const Zp * const y = &_reg.data()[uint32_t(src2) * n];

		for (size_t k = 0; k < n; ++k) if (y[k] != x[k]) return false;
		return true;
	}

	void square(const Reg src)
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(src) * n];

		forward(x);
		if (n % 5 == 0) sqr5(x); else sqr4(x);
		backward(x);
		carry_weight(x);
	}

	void set_multiplicand(const Reg dst, const Reg src)
	{
		if (src != dst) copy(dst, src);

		const size_t n = _n;
		Zp * const y = &_reg.data()[uint32_t(dst) * n];

		forward(y);
		if (n % 5 == 0) forward5(y); else forward4(y);
	}

	void mul(const Reg dst, const Reg src)
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(dst) * n];
		const Zp * const y = &_reg.data()[uint32_t(src) * n];

		forward(x);
		if (n % 5 == 0) mul5(x, y); else mul4(x, y);
		backward(x);
		carry_weight(x);
	}

	void error()
	{
		Zp * const x = &_reg.data()[0];
		x[_n / 2] += Zp(1);
	}
};

int main()
{
	using Reg = Mersenne::Reg;

	size_t count = 0, count5 = 0;

	// 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, ...
	for (uint32_t p = 3; p <= 1207959503; p += 2)
	{
		bool isprime = true;
		for (uint32_t d = 3; p / d >= d; d += 2) if (p % d == 0) { isprime = false; break; }
		if (!isprime) continue;

		Mersenne m(p);

		++count;
		if (m.get_length() % 5 == 0) ++count5;

		// Gerbicz-Li error checking
		const uint32_t B_GL = std::max(uint32_t(std::sqrt(p)), 2u);

		// 3^{2^p}
		m.set(Reg::R0, 3);	// result = 3
		m.set(Reg::R1, 1);	// d(t) = 1
		for (uint32_t i = 0, j = p - 1; i < p; ++i, --j)
		{
			m.square(Reg::R0);
			if ((p == 1511) && (j == 0)) m.error();	// test Gerbicz-Li

			if ((j % B_GL == 0) && (j != 0))
			{
				m.set_multiplicand(Reg::R2, Reg::R0);
				m.mul(Reg::R1, Reg::R2);	// d(t + 1) = d(t) * result
			}
		}

		// 3-prp test: 3^{(2^p - 1) - 1} ?= 1  <=>  3^{2^p} ?= 9
		if (p >= 13) m.set(Reg::R2, 9); else { m.set(Reg::R2, 3); m.square(Reg::R2); }
		const bool is_prp = m.is_equal(Reg::R0, Reg::R2);

		// d(t + 1) = d(t) * result
		m.set_multiplicand(Reg::R2, Reg::R1);
		m.mul(Reg::R0, Reg::R2);

		// The exponent of the residue is 2^(p mod B)
		// See: An Efficient Modular Exponentiation Proof Scheme, §2, Darren Li, Yves Gallot, https://arxiv.org/abs/2209.15623

		// d(t)^{2^B} * 3^{2^(p mod B)} = (3 * d(t)^{2^(B - p mod B)})^{2^(p mod B)}
		for (uint32_t i = 0; i < B_GL - p % B_GL; ++i) m.square(Reg::R1);
		m.set(Reg::R2, 3);
		m.set_multiplicand(Reg::R2, Reg::R2);
		m.mul(Reg::R1, Reg::R2);
		for (uint32_t i = 0; i < p % B_GL; ++i) m.square(Reg::R1);

		if (!m.is_equal(Reg::R0, Reg::R1)) std::cout << p << ": " << m.get_length() << ", Gerbicz-Li failed!" << std::endl;

		if (is_prp) std::cout << p << ": " << m.get_length() << " (radix-5: " << count5 << "/" << count << " = " << 100.0 * count5 / count <<  "%)" << std::endl;
	}

	return EXIT_SUCCESS;
}
