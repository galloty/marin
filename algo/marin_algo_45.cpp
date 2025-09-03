/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <iostream>

#include "mersenne_5.h"

// Butterfly sizes are radix-4 and radix-5.
// Radix-5 is the last stage of the transform.

class Mersenne45 : public Mersenne_5
{
private:
	bool _even_exponent;

private:
	// Transform
	void forward(Zp * const x) const
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n, s5 = (n % 5 == 0) ? 5 : 4;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r4 = &_root.data()[n5 / 2];

		for (size_t s = _even_exponent ? 1 : 2, m = n / 4 / s; m >= s5; m /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp r1 = r2[j], r20 = r4[2 * j + 0], r21 = r4[2 * j + 1];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 4 * m * j + i;
					fwd4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r1, r20, r21);
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

		for (size_t m = s5, s = n / 4 / m; m <= n / 4; m *= 4, s /= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp ri1 = r2i[j], ri20 = r4i[2 * j + 0], ri21 = r4i[2 * j + 1];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 4 * m * j + i;
					bck4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], ri1, ri20, ri21);
				}
			}
		}
	}

public:
	Mersenne45(const uint32_t q) : Mersenne_5(q)
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n;

		size_t m = n5; for (; m > 1; m /= 4);
		_even_exponent = (m == 1);

		// Roots

		Zp * const r2 = &_root.data()[0];
		Zp * const r2i = &_invroot.data()[0];
		Zp * const r4 = &_root.data()[n5 / 2];
		Zp * const r4i = &_invroot.data()[n5 / 2];

		for (size_t j = 0; j < n5 / 4; ++j)
		{
			r4[2 * j + 0] = r2[2 * j]; r4i[2 * j + 0] = r2i[2 * j];
			r4[2 * j + 1] = r2[j] * r2[2 * j]; r4i[2 * j + 1] = r2i[j] * r2i[2 * j];
		}
	}

	virtual ~Mersenne45() {}

	void set(const Reg dst, const uint64_t a)
	{
		_set(dst, a);

		// radix-2
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(dst) * n];
		if (!_even_exponent) x[n / 2] = x[0];
	}

	void square(const Reg src) override
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(src) * n];

		forward(x);
		if (n % 5 == 0) sqr5(x); else sqr4(x);
		backward(x);
		const Zp f = (n % 5 == 0) ? _inv_n : _inv_n + _inv_n;
		if (_even_exponent) carry_weight(x, f); else carry_weight2(x, f);
	}

	void set_multiplicand(const Reg dst, const Reg src) override
	{
		if (src != dst) copy(dst, src);

		const size_t n = _n;
		Zp * const y = &_reg.data()[uint32_t(dst) * n];

		forward(y);
		if (n % 5 == 0) forward5(y); else forward4(y);
	}

	void mul(const Reg dst, const Reg src) override
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(dst) * n];
		const Zp * const y = &_reg.data()[uint32_t(src) * n];

		forward(x);
		if (n % 5 == 0) mul5(x, y); else mul4(x, y);
		backward(x);
		const Zp f = (n % 5 == 0) ? _inv_n : _inv_n + _inv_n;
		if (_even_exponent) carry_weight(x, f); else carry_weight2(x, f);
	}
};

int main()
{
	// 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, ...
	for (uint32_t p = 3; p <= 1207959503; p += 2)
	{
		bool isprime = true;
		for (uint32_t d = 3; p / d >= d; d += 2) if (p % d == 0) { isprime = false; break; }
		if (!isprime) continue;

		Mersenne45 m(p);
		const bool is_prp = Mersenne::check(m, p);
		if (is_prp) std::cout << p << ": " << m.get_length() << std::endl;
	}

	return EXIT_SUCCESS;
}
