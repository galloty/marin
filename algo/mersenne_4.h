/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <iostream>

#include "mersenne.h"

// Radix-5 is the first stage of the transform.

class Mersenne_4 : public Mersenne
{
protected:

	// Radix-4
	void forward4(Zp * const x) const
	{
		const size_t n4 = _n / 4;
		const Zp * const r2 = &_root.data()[n4];

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
		const Zp * const r2 = &_root.data()[n4];
		const Zp * const r2i = &_invroot.data()[n4];

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
		const Zp * const r2 = &_root.data()[n4];
		const Zp * const r2i = &_invroot.data()[n4];

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

	// Radix-8
	void forward8(Zp * const x) const
	{
		const size_t n8 = _n / 8;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r4 = &_root.data()[_n / 2];

		for (size_t j = 0; j < n8; ++j)
		{
			const size_t k = 8 * j;
			const Zp r1 = r2[n8 + j], r20 = r4[2 * (n8 + j) + 0], r21 = r4[2 * (n8 + j) + 1];
			fwd4(x[k + 0], x[k + 2], x[k + 4], x[k + 6], r1, r20, r21);
			fwd4(x[k + 1], x[k + 3], x[k + 5], x[k + 7], r1, r20, r21);
		}
	}

	// Radix-8, square, inverse radix-8
	void sqr8(Zp * const x) const
	{
		const size_t n8 = _n / 8;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r2i = &_invroot.data()[0];
		const Zp * const r4 = &_root.data()[_n / 2];
		const Zp * const r4i = &_invroot.data()[_n / 2];

		for (size_t j = 0; j < n8; ++j)
		{
			const size_t k = 8 * j;

			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			Zp x4 = x[k + 4], x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7];

			const Zp r1 = r2[n8 + j], r20 = r4[2 * (n8 + j) + 0], r21 = r4[2 * (n8 + j) + 1];
			fwd4(x0, x2, x4, x6, r1, r20, r21);
			fwd4(x1, x3, x5, x7, r1, r20, r21);

			sqr22(x0, x1, x2, x3, r20);
			sqr22(x4, x5, x6, x7, r20.muli());

			const Zp r1i = r2i[n8 + j], r20i = r4i[2 * (n8 + j) + 0], r21i = r4i[2 * (n8 + j) + 1];
			bck4(x0, x2, x4, x6, r1i, r20i, r21i);
			bck4(x1, x3, x5, x7, r1i, r20i, r21i);

			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
			x[k + 4] = x4; x[k + 5] = x5; x[k + 6] = x6; x[k + 7] = x7;
		}
	}

	// Radix-8, mul, inverse radix-8
	void mul8(Zp * const x, const Zp * const y) const
	{
		const size_t n8 = _n / 8;
		const Zp * const r2 = &_root.data()[0];
		const Zp * const r2i = &_invroot.data()[0];
		const Zp * const r4 = &_root.data()[_n / 2];
		const Zp * const r4i = &_invroot.data()[_n / 2];

		for (size_t j = 0; j < n8; ++j)
		{
			const size_t k = 8 * j;

			Zp x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			Zp x4 = x[k + 4], x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7];

			const Zp r1 = r2[n8 + j], r20 = r4[2 * (n8 + j) + 0], r21 = r4[2 * (n8 + j) + 1];
			fwd4(x0, x2, x4, x6, r1, r20, r21);
			fwd4(x1, x3, x5, x7, r1, r20, r21);

			mul22(x0, x1, x2, x3, y[k + 0], y[k + 1], y[k + 2], y[k + 3], r20);
			mul22(x4, x5, x6, x7, y[k + 4], y[k + 5], y[k + 6], y[k + 7], r20.muli());

			const Zp r1i = r2i[n8 + j], r20i = r4i[2 * (n8 + j) + 0], r21i = r4i[2 * (n8 + j) + 1];
			bck4(x0, x2, x4, x6, r1i, r20i, r21i);
			bck4(x1, x3, x5, x7, r1i, r20i, r21i);

			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
			x[k + 4] = x4; x[k + 5] = x5; x[k + 6] = x6; x[k + 7] = x7;
		}
	}

public:
	Mersenne_4(const uint32_t q) : Mersenne(q)
	{
		const size_t n = _n;

		// Roots

		Zp * const r2 = &_root.data()[0];
		Zp * const r2i = &_invroot.data()[0];

		for (size_t s = (n % 5 == 0) ? 5 : 1; s <= n / 4; s *= 2)
		{
			const Zp rs = Zp::root_nth(2 * s), rsi = rs.invert();
			Zp rsj = Zp(1), rsji = Zp(1);
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = inv_reversal(j, s);
				r2[s + jr] = rsj; r2i[s + jr] = rsji;
				rsj *= rs; rsji *= rsi;
			}
		}
	}

	virtual ~Mersenne_4() {}
};
