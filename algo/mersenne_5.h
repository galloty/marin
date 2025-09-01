/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <iostream>

#include "mersenne.h"

// Radix-5 is the last stage of the transform.

class Mersenne_5 : public Mersenne
{
protected:

	// Radix-4
	void forward4(Zp * const x) const
	{
		const size_t n4 = _n / 4;
		const Zp * const r2 = &_root.data()[0];

		for (size_t j = 0; j < n4; ++j)
		{
			const size_t k = 4 * j;
			const Zp r = r2[j];
			fwd2(x[k + 0], x[k + 2], r); fwd2(x[k + 1], x[k + 3], r);
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
			const Zp r = r2[j], ri = r2i[j];
			const size_t k = 4 * j;
			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd2(x0, x2, r); fwd2(x1, x3, r);
			sqr22(x0, x1, x2, x3, r);
			bck2(x0, x2, ri); bck2(x1, x3, ri);
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
			const Zp r = r2[j], ri = r2i[j];
			const size_t k = 4 * j;
			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd2(x0, x2, r); fwd2(x1, x3, r);
			mul22(x0, x1, x2, x3, y[k + 0], y[k + 1], y[k + 2], y[k + 3], r);
			bck2(x0, x2, ri); bck2(x1, x3, ri);
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
			fwd5(x[k + 0], x[k + 1], x[k + 2], x[k + 3], x[k + 4], r5[j]);
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
			fwd5(x0, x1, x2, x3, x4, r5[j]);
			x0 *= x0; x1 *= x1; x2 *= x2; x3 *= x3; x4 *= x4;
			bck5(x0, x1, x2, x3, x4, r5i[j]);
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
			fwd5(x0, x1, x2, x3, x4, r5[j]);
			x0 *= y[k + 0]; x1 *= y[k + 1]; x2 *= y[k + 2]; x3 *= y[k + 3]; x4 *= y[k + 4];
			bck5(x0, x1, x2, x3, x4, r5i[j]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3; x[k + 4] = x4;
		}
	}

public:
	Mersenne_5(const uint32_t q) : Mersenne(q)
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n;

		// Roots

		Zp * const r2 = &_root.data()[0];
		Zp * const r2i = &_invroot.data()[0];

		const Zp r = Zp::root_nth(2 * n5 / 2), ri = r.invert();
		Zp r_j = Zp(1), ri_j = Zp(1);
		for (size_t j = 0; j < n5 / 2; ++j)
		{
			const size_t jr = bitrev(j, n5 / 2);
			r2[jr] = r_j; r2i[jr] = ri_j;
			r_j *= r; ri_j *= ri;
		}

		if (n % 5 == 0)
		{
			Zp * const r5 = &_root.data()[n5];
			Zp * const r5i = &_invroot.data()[n5];

			const Zp r = Zp::root_nth(5 * n5), ri = r.invert();
			Zp r_j = Zp(1), ri_j = Zp(1);
			for (size_t j = 0; j < n5; ++j)
			{
				const size_t jr = bitrev(j, n5);
				r5[jr] = r_j; r5i[jr] = ri_j;
				r_j *= r; ri_j *= ri;
			}
		}
	}

	virtual ~Mersenne_5() {}
};
