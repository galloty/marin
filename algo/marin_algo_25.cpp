/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <iostream>

#include "mersenne_5.h"

// Butterfly sizes are radix-2 and radix-5.
// Radix-5 is the last stage of the transform.

class Mersenne25 : public Mersenne_5
{
private:

	// Transform
	void forward(Zp * const x) const
	{
		const size_t n = _n, s5 = (n % 5 == 0) ? 5 : 4;
		const Zp * const r2 = &_root.data()[0];

		for (size_t m = n / 2, s = 1; m >= s5; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp r = r2[j];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					fwd2(x[k + 0 * m], x[k + 1 * m], r);
				}
			}
		}
	}

	// Inverse Transform
	void backward(Zp * const x) const
	{
		const size_t n = _n, s5 = (n % 5 == 0) ? 5 : 4;
		const Zp * const r2i = &_invroot.data()[0];

		for (size_t m = s5, s = n / 2 / m; m <= n / 2; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Zp ri = r2i[j];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					bck2(x[k + 0 * m], x[k + 1 * m], ri);
				}
			}
		}
	}

public:
	Mersenne25(const uint32_t q) : Mersenne_5(q) {}
	virtual ~Mersenne25() {}

	void set(const Reg dst, const uint64_t a)
	{
		_set(dst, a);
	}

	void square(const Reg src) override
	{
		const size_t n = _n;
		Zp * const x = &_reg.data()[uint32_t(src) * n];

		forward(x);
		if (n % 5 == 0) sqr5(x); else sqr4(x);
		backward(x);
		const Zp f = (n % 5 == 0) ? _inv_n : _inv_n + _inv_n;
		carry_weight(x, f);
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
		carry_weight(x, f);
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

		Mersenne25 m(p);
		const bool is_prp = Mersenne::check(m, p);
		if (is_prp) std::cout << p << ": " << m.get_length() << std::endl;
	}

	return EXIT_SUCCESS;
}
