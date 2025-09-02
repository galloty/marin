/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>

#include "Zp.h"

// The multiplication is based on:
//  - An Irrational Base Discrete Weighted Transform. x^p - 1 => x^n - 1, where n = 2^m or 5*2^m.
//  - A recursive polynomial factorization approach:
//    x^{2n} - r^2 => x^n - r and x^n + r;
//    x^{5n} - r^5 => 5 polynomials of the form x^n - r_i.
// The finite field is Z/pZ, where p = 2^64 - 2^32 + 1.

class Mersenne
{
protected:
	const size_t _n;
	const Zp _inv_n;
	std::vector<Zp> _reg;	// the weighted representation of R0, R1, ...
	std::vector<Zp> _root;
	std::vector<Zp> _invroot;
	std::vector<Zp> _weight;
	std::vector<Zp> _invweight;
	std::vector<uint8_t> _digit_width;

protected:
	static constexpr size_t transform_size(const uint32_t exponent)
	{
		// Make sure the transform is long enough so that each 'digit' can't overflow after the convolution.
		uint32_t w = 0, log2_n = 1, log2_n5 = 1;
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

		return std::min(size_t(1) << log2_n, size_t(5) << log2_n5);	// must be >= 4 or 5 * 4
	}

	// Bit-reversal permutation
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	// Digit-reversal permutation (n = 2^e * 5^f)
	static constexpr size_t reversal(const size_t i, const size_t n)
	{
		size_t r = 0, k = n, j = i;
		while (k % 2 == 0) { r = 2 * r + j % 2; k /= 2; j /= 2; }
		while (k % 5 == 0) { r = 5 * r + j % 5; k /= 5; j /= 5; }
		return r;
	}

	// Inverse digit-reversal permutation (n = 2^e * 5^f)
	static constexpr size_t inv_reversal(const size_t i, const size_t n)
	{
		size_t r = 0, k = n, j = i;
		while (k % 5 == 0) { r = 5 * r + j % 5; k /= 5; j /= 5; }
		while (k % 2 == 0) { r = 2 * r + j % 2; k /= 2; j /= 2; }
		return r;
	}
	
	// Radix-2
	static void fwd2(Zp & x0, Zp & x1, const Zp & r)
	{
		const Zp u0 = x0, u1 = r * x1;
		x0 = u0 + u1; x1 = u0 - u1;
	}

	// Inverse radix-2
	static void bck2(Zp & x0, Zp & x1, const Zp & ri)
	{
		const Zp u0 = x0, u1 = x1;
		x0 = u0 + u1; x1 = (u0 - u1) * ri;
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
	static void fwd5(Zp & x0, Zp & x1, Zp & x2, Zp & x3, Zp & x4, const Zp & r)
	{
		const Zp r2 = r * r, r3 = r * r2, r4 = r2 * r2;
		Zp a0 = x0, a1 = r * x1, a2 = r2 * x2, a3 = r3 * x3, a4 = r4 * x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// Inverse radix-5
	static void bck5(Zp & x0, Zp & x1, Zp & x2, Zp & x3, Zp & x4, const Zp & ri)
	{
		Zp a0 = x0, a4 = x1, a3 = x2, a2 = x3, a1 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		const Zp ri2 = ri * ri, ri3 = ri * ri2, ri4 = ri2 * ri2;
		x0 = a0; x1 = ri * a1; x2 = ri2 * a2; x3 = ri3 * a3; x4 = ri4 * a4;
	}

	// Radix-5, first stage
	static void fwd5_0(Zp & x0, Zp & x1, Zp & x2, Zp & x3, Zp & x4)
	{
		Zp a0 = x0, a1 = x1, a2 = x2, a3 = x3, a4 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// Inverse radix-5, first stage
	static void bck5_0(Zp & x0, Zp & x1, Zp & x2, Zp & x3, Zp & x4)
	{
		Zp a0 = x0, a4 = x1, a3 = x2, a2 = x3, a1 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// 2 x Radix-2, sqr, inverse radix-2
	static void sqr22(Zp & x0, Zp & x1, Zp & x2, Zp & x3, const Zp & r)
	{
		const Zp t0 = x0 * x0 + x1 * x1 * r; x1 *= x0 + x0; x0 = t0;
		const Zp t2 = x2 * x2 - x3 * x3 * r; x3 *= x2 + x2; x2 = t2;
	}

	static void mul22(Zp & x0, Zp & x1, Zp & x2, Zp & x3, const Zp & y0, const Zp & y1, const Zp & y2, const Zp & y3, const Zp & r)
	{
		const Zp t0 = x0 * y0 + x1 * y1 * r; x1 = x0 * y1 + x1 * y0; x0 = t0;
		const Zp t2 = x2 * y2 - x3 * y3 * r; x3 = x2 * y3 + x3 * y2; x2 = t2;
	}

	void carry_weight(Zp * const x, const Zp & f) const
	{
		const size_t n = _n;
		const Zp * const w = _weight.data();
		const Zp * const wi = _invweight.data();
		const uint8_t * const width = _digit_width.data();

		uint64_t c = 0;
		for (size_t k = 0; k < n; ++k)
		{
			const Zp u = x[k] * f * wi[k];
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

	void carry_weight2(Zp * const x, const Zp f) const
	{
		const size_t n = _n;
		const Zp * const w = _weight.data();
		const Zp * const wi = _invweight.data();
		const uint8_t * const width = _digit_width.data();

		uint64_t c0 = 0, c1 = 0;
		for (size_t k = 0; k < n / 2; ++k)
		{
			const Zp u0 = x[k + 0 * n / 2], u1 = x[k + 1 * n / 2];
			// inverse radix-2
			const Zp v0 = (u0 + u1) * f * wi[k + 0 * n / 2];
			const Zp v1 = (u0 - u1) * f * wi[k + 1 * n / 2];
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

public:
	Mersenne(const uint32_t q) : _n(transform_size(q)), _inv_n(Zp((_n % 5 == 0) ? _n : _n / 1).invert()),
		_reg(3 * _n),	// allocate 3 registers
		_root(_n), _invroot(_n),
		_weight(_n), _invweight(_n), _digit_width(_n)
	{
		const size_t n = _n;

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

protected:
	void _set(const Reg dst, const uint64_t a)
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(dst) * n];

		x[0] = Zp(a);	// weight[0] = 1
		for (size_t k = 1; k < n; ++k) x[k] = Zp(0);
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

	void error()
	{
		Zp * const x = &_reg.data()[0];
		x[_n / 2] += Zp(1);
	}

public:
	virtual void set(const Reg dst, const uint64_t a) = 0;
	virtual void square(const Reg src) = 0;
	virtual void set_multiplicand(const Reg dst, const Reg src) = 0;
	virtual void mul(const Reg dst, const Reg src) = 0;

	static bool check(Mersenne & m, const uint32_t p)
	{
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
		return is_prp;
	}
};
