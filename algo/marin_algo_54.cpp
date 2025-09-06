/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <iostream>

#include "mersenne_4.h"

// Butterfly sizes are radix-4 and radix-5.
// Radix-5 is the first stage of the transform.

class Mersenne54 : public Mersenne_4
{
private:
	bool _even_exponent;

private:

	// Transform
	void forward(Zp * const x) const
	{
		const size_t n = _n, n4 = n / 4, s5 = (n % 5 == 0) ? 5 : 4, n5 = (n % 5 == 0) ? n / 5 : n;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r4 = &_root.data()[n / 2];

		if (s5 == 5)
		{
			for (size_t j = 0; j < n5; ++j)
			{
				const size_t k = j;
				fwd5_0(x[k + 0 * n5], x[k + 1 * n5], x[k + 2 * n5], x[k + 3 * n5], x[k + 4 * n5]);
			}
		}
		else if (n >= 16)
		{
			for (size_t j = 0; j < n4; ++j)
			{
				const size_t k = j;
				fwd4_0(x[k + 0 * n4], x[k + 1 * n4], x[k + 2 * n4], x[k + 3 * n4]);
			}
		}

		for (size_t s = s5, m = n / 4 / s, s_max = n / (_even_exponent ? 8 : 16); s <= s_max; s *= 4, m /= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t sj = s + j;
				const Zp r1 = r2[sj], r20 = r4[2 * sj + 0], r21 = r4[2 * sj + 1];

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
		const size_t n = _n, n4 = n / 4, s5 = (n % 5 == 0) ? 5 : 4, n5 = (n % 5 == 0) ? n / 5 : n;
		const Zp * const r2i = &_invroot.data()[0];
		const Zp * const r4i = &_invroot.data()[n / 2];

		for (size_t m = _even_exponent ? 4 : 8, s = n / 4 / m; s >= s5; s /= 4, m *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t sj = s + j;
				const Zp r1i = r2i[sj], r20i = r4i[2 * sj + 0], r21i = r4i[2 * sj + 1];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 4 * m * j + i;
					bck4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r1i, r20i, r21i);
				}
			}
		}

		if (s5 == 5)
		{
			for (size_t j = 0; j < n5; ++j)
			{
				const size_t k = j;
				bck5_0(x[k + 0 * n5], x[k + 4 * n5], x[k + 3 * n5], x[k + 2 * n5], x[k + 1 * n5]);
			}
		}
		else if (n >= 16)
		{
			for (size_t j = 0; j < n4; ++j)
			{
				const size_t k = j;
				bck4_0(x[k + 0 * n4], x[k + 1 * n4], x[k + 2 * n4], x[k + 3 * n4]);
			}
		}
	}

public:
	Mersenne54(const uint32_t q) : Mersenne_4(q)
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n;

		size_t m = n5; for (; m > 1; m /= 4);
		_even_exponent = (m == 1);

		// Roots

		Zp * const r2 = &_root.data()[0];
		Zp * const r2i = &_invroot.data()[0];
		Zp * const r4 = &_root.data()[n / 2];
		Zp * const r4i = &_invroot.data()[n / 2];

		for (size_t s = (n % 5 == 0) ? 5 : 1; s <= n / 4; s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t sj = s + j;
				r4[2 * sj + 0] = r2[2 * sj]; r4i[2 * sj + 0] = r2i[2 * sj];
				r4[2 * sj + 1] = r2[sj] * r2[2 * sj]; r4i[2 * sj + 1] = r2i[sj] * r2i[2 * sj];
			}
		}
	}

	virtual ~Mersenne54() {}

	void set(const Reg dst, const uint64_t a)
	{
		_set(dst, a);
	}

	void square(const Reg src) override
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(src) * n];

		forward(x);
		if (_even_exponent) sqr4(x); else sqr8(x);
		backward(x);
		carry_weight(x, _inv_n + _inv_n);
	}

	void set_multiplicand(const Reg dst, const Reg src) override
	{
		if (src != dst) copy(dst, src);

		const size_t n = _n;
		Zp * const y = &_reg.data()[uint32_t(dst) * n];

		forward(y);
		if (_even_exponent) forward4(y); else forward8(y);
	}

	void mul(const Reg dst, const Reg src) override
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(dst) * n];
		const Zp * const y = &_reg.data()[uint32_t(src) * n];

		forward(x);
		if (_even_exponent) mul4(x, y); else mul8(x, y);
		backward(x);
		carry_weight(x, _inv_n + _inv_n);
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

		Mersenne54 m(p);
		const bool is_prp = Mersenne::check(m, p);
		if (is_prp) std::cout << p << ": " << m.get_length() << std::endl;
	}

	return EXIT_SUCCESS;
}
