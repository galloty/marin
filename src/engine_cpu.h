/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"
#include "ibdwt.h"

#include <cstring>

struct uint64_2
{
	uint64 s0, s1;

	uint64_2() {}
	uint64_2(const uint64 u0, const uint64 u1) : s0(u0), s1(u1) {}
	// OpenCL: Implicit conversions from a scalar type to a vector type are allowed. The scalar type is then widened to the vector.
	uint64_2(const uint64 u) : s0(u), s1(u) {}
};

struct uint64_4
{
	uint64 s0, s1, s2, s3;

	uint64_4() {}
	uint64_4(const uint64 u0, const uint64 u1, const uint64 u2, const uint64 u3) : s0(u0), s1(u1), s2(u2), s3(u3) {}
	uint64_4(const uint64 u) : s0(u), s1(u), s2(u), s3(u) {}
};

struct uint8_4
{
	uint8 s0, s1, s2, s3;

	uint8_4() {}
	uint8_4(const uint8 u0, const uint8 u1, const uint8 u2, const uint8 u3) : s0(u0), s1(u1), s2(u2), s3(u3) {}
	uint8_4(const uint8 u) : s0(u), s1(u), s2(u), s3(u) {}
};

INLINE uint64_2 mod_add2(const uint64_2 lhs, const uint64_2 rhs) { return uint64_2(mod_add(lhs.s0, rhs.s0), mod_add(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_sub2(const uint64_2 lhs, const uint64_2 rhs) { return uint64_2(mod_sub(lhs.s0, rhs.s0), mod_sub(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_mul2(const uint64_2 lhs, const uint64_2 rhs) { return uint64_2(mod_mul(lhs.s0, rhs.s0), mod_mul(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_sqr2(const uint64_2 lhs) { return uint64_2(mod_sqr(lhs.s0), mod_sqr(lhs.s1)); }
INLINE uint64_2 mod_muli2(const uint64_2 lhs) { return uint64_2(mod_muli(lhs.s0), mod_muli(lhs.s1)); }

INLINE uint64_4 mod_mul4(const uint64_4 lhs, const uint64_4 rhs) { return uint64_4(mod_mul(lhs.s0, rhs.s0), mod_mul(lhs.s1, rhs.s1), mod_mul(lhs.s2, rhs.s2), mod_mul(lhs.s3, rhs.s3)); }

INLINE uint64_4 adc4(const uint64_4 lhs, const uint8_4 width, const uint64 carry)
{
	uint64_4 r;
	uint64 c = carry;
	r.s0 = adc(lhs.s0, width.s0, c);
	r.s1 = adc(lhs.s1, width.s1, c);
	r.s2 = adc(lhs.s2, width.s2, c);
	r.s3 = lhs.s3 + c;
	return r;
}

INLINE uint64_4 adc_mul4(const uint64_4 lhs, const uint32 a, const uint8_4 width, uint64 & carry)
{
	uint64_4 r;
	r.s0 = adc_mul(lhs.s0, a, width.s0, carry);
	r.s1 = adc_mul(lhs.s1, a, width.s1, carry);
	r.s2 = adc_mul(lhs.s2, a, width.s2, carry);
	r.s3 = adc_mul(lhs.s3, a, width.s3, carry);
	return r;
}

class engine_cpu : public engine
{
private:
	const size_t _n;
	const size_t _reg_count;
	const uint64 _inv_n_2;
	const bool _even;
	std::vector<uint64> _reg;	// the weighted representation of R0, R1, ...
	std::vector<uint64> _root;
	std::vector<uint64> _weight;
	std::vector<uint8> _digit_width;
	std::vector<uint64> _carry;

private:
	// Radix-4
	static void fwd4(uint64_2 & x0, uint64_2 & x1, uint64_2 & x2, uint64_2 & x3, const uint64 r1, const uint64 r20, const uint64 r21)
	{
		const uint64_2 u0 = x0, u2 = mod_mul2(r1, x2), u1 = mod_mul2(r20, x1), u3 = mod_mul2(r21, x3);
		const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3));
		x0 = mod_add2(v0, v1); x1 = mod_sub2(v0, v1); x2 = mod_add2(v2, v3); x3 = mod_sub2(v2, v3);
	}

	// Inverse radix-4
	static void bck4(uint64_2 & x0, uint64_2 & x1, uint64_2 & x2, uint64_2 & x3, const uint64 ri1, const uint64 ri20, const uint64 ri21)
	{
		const uint64_2 u0 = x0, u1 = x1, u2 = x2, u3 = x3;
		const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2));
		x0 = mod_add2(v0, v2); x2 = mod_mul2(ri1, mod_sub2(v0, v2)); x1 = mod_mul2(ri20, mod_add2(v1, v3)); x3 = mod_mul2(ri21, mod_sub2(v1, v3));
	}

	// Radix-4, first stage
	static void fwd4_0(uint64_2 & x0, uint64_2 & x1, uint64_2 & x2, uint64_2 & x3)
	{
		const uint64_2 u0 = x0, u2 = x2, u1 = x1, u3 = x3;
		const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3));
		x0 = mod_add2(v0, v1); x1 = mod_sub2(v0, v1); x2 = mod_add2(v2, v3); x3 = mod_sub2(v2, v3);
	}

	// Inverse radix-4, first stage
	static void bck4_0(uint64_2 & x0, uint64_2 & x1, uint64_2 & x2, uint64_2 & x3)
	{
		const uint64_2 u0 = x0, u1 = x1, u2 = x2, u3 = x3;
		const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2));
		x0 = mod_add2(v0, v2); x2 = mod_sub2(v0, v2); x1 = mod_add2(v1, v3); x3 = mod_sub2(v1, v3);
	}

	// 2 x radix-2
	static void fwd22(uint64_2 & x0, uint64_2 & x1, const uint64 r)
	{
		const uint64_2 u0 = x0, u1 = mod_mul2(r, x1);
		x0 = mod_add2(u0, u1); x1 = mod_sub2(u0, u1);
	}

	// 2 x inverse radix-2
	static void bck22(uint64_2 & x0, uint64_2 & x1, const uint64 ri)
	{
		const uint64_2 u0 = x0, u1 = x1;
		x0 = mod_add2(u0, u1); x1 = mod_mul2(ri, mod_sub2(u0, u1));
	}

	// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.
	static void butterfly5(uint64_2 & a0, uint64_2 & a1, uint64_2 & a2, uint64_2 & a3, uint64_2 & a4)
	{
		static const uint64 K = mod_root_nth(5), K2 = mod_sqr(K), K3 = mod_mul(K, K2), K4 = mod_sqr(K2);
		static const uint64 cosu = mod_half(mod_add(K, K4)), isinu = mod_half(mod_sub(K, K4));
		static const uint64 cos2u = mod_half(mod_add(K2, K3)), isin2u = mod_half(mod_sub(K2, K3));
		static const uint64 F1 = mod_sub(mod_half(mod_add(cosu, cos2u)), 1), F2 = mod_half(mod_sub(cosu, cos2u));
		static const uint64 F3 = mod_add(isinu, isin2u), F4 = isin2u, F5 = mod_sub(isinu, isin2u);

		const uint64_2 s1 = mod_add2(a1, a4), s2 = mod_sub2(a1, a4), s3 = mod_add2(a3, a2), s4 = mod_sub2(a3, a2);
		const uint64_2 s5 = mod_add2(s1, s3), s6 = mod_sub2(s1, s3), s7 = mod_add2(s2, s4), s8 = mod_add2(s5, a0);
		const uint64_2 m0 = s8;
		const uint64_2 m1 = mod_mul2(F1, s5), m2 = mod_mul2(F2, s6), m3 = mod_mul2(F3, s2), m4 = mod_mul2(F4, s7), m5 = mod_mul2(F5, s4);
		const uint64_2 s9 = mod_add2(m0, m1), s10 = mod_add2(s9, m2), s11 = mod_sub2(s9, m2), s12 = mod_sub2(m3, m4);
		const uint64_2 s13 = mod_add2(m4, m5), s14 = mod_add2(s10, s12), s15 = mod_sub2(s10, s12), s16 = mod_add2(s11, s13);
		const uint64_2 s17 = mod_sub2(s11, s13);
		a0 = m0; a1 = s14; a2 = s16; a3 = s17; a4 = s15;
	}

	// Radix-5, first stage
	static void fwd5_0(uint64_2 & x0, uint64_2 & x1, uint64_2 & x2, uint64_2 & x3, uint64_2 & x4)
	{
		uint64_2 a0 = x0, a1 = x1, a2 = x2, a3 = x3, a4 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// Inverse radix-5, first stage
	static void bck5_0(uint64_2 & x0, uint64_2 & x1, uint64_2 & x2, uint64_2 & x3, uint64_2 & x4)
	{
		uint64_2 a0 = x0, a1 = x1, a2 = x2, a3 = x3, a4 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x4 = a1; x3 = a2; x2 = a3; x1 = a4;
	}

	// 2 x Radix-2, sqr, inverse radix-2
	static void sqr22(uint64_2 & x01, uint64_2 & x23, const uint64 & r)
	{
		const uint64_2 s01 = mod_sqr2(x01), s23 = mod_sqr2(x23);
		x01.s1 = mod_mul(x01.s1, mod_add(x01.s0, x01.s0)); x01.s0 = mod_add(s01.s0, mod_mul(s01.s1, r));
		x23.s1 = mod_mul(x23.s1, mod_add(x23.s0, x23.s0)); x23.s0 = mod_sub(s23.s0, mod_mul(s23.s1, r));
	}

	static void mul22(uint64_2 & x01, uint64_2 & x23, const uint64_2 & y01, const uint64_2 & y23, const uint64 & r)
	{
		const uint64_2 m01 = mod_mul2(x01, y01), m23 = mod_mul2(x23, y23);
		x01.s1 = mod_add(mod_mul(x01.s0, y01.s1), mod_mul(x01.s1, y01.s0)); x01.s0 = mod_add(m01.s0, mod_mul(m01.s1, r));
		x23.s1 = mod_add(mod_mul(x23.s0, y23.s1), mod_mul(x23.s1, y23.s0)); x23.s0 = mod_sub(m23.s0, mod_mul(m23.s1, r));
	}

	void forward4(uint64_2 * const x, const size_t s, const int lm) const
	{
		const size_t n = _n, n_8 = n / 8;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r4 = &_root.data()[n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t m = size_t(1) << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;

			fwd4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r2[sj], r4[2 * sj + 0], r4[2 * sj + 1]);
		}
	}

	void forward4_0(uint64_2 * const x) const
	{
		const size_t n = _n, n_8 = n / 8;

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t k = id;
			fwd4_0(x[k + 0 * n_8], x[k + 1 * n_8], x[k + 2 * n_8], x[k + 3 * n_8]);
		}
	}

	void forward5_0(uint64_2 * const x) const
	{
		const size_t n = _n, n_10 = (n % 5 == 0) ? n / 10 : n / 2;

		for (size_t id = 0; id < n_10; ++id)
		{
			const size_t k = id;
			fwd5_0(x[k + 0 * n_10], x[k + 1 * n_10], x[k + 2 * n_10], x[k + 3 * n_10], x[k + 4 * n_10]);
		}
	}

	void backward4(uint64_2 * const x, const size_t s, const int lm) const
	{
		const size_t n = _n, n_8 = n / 8;
		const uint64 * const r2i = &_root.data()[n / 2];
		const uint64 * const r4i = &_root.data()[n + n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t m = size_t(1) << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;

			bck4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r2i[sj], r4i[2 * sj + 0], r4i[2 * sj + 1]);
		}
	}

	void backward4_0(uint64_2 * const x) const
	{
		const size_t n = _n, n_8 = n / 8;

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t k = id;
			bck4_0(x[k + 0 * n_8], x[k + 1 * n_8], x[k + 2 * n_8], x[k + 3 * n_8]);
		}
	}

	void backward5_0(uint64_2 * const x) const
	{
		const size_t n = _n, n_10 = (n % 5 == 0) ? n / 10 : n / 2;

		for (size_t id = 0; id < n_10; ++id)
		{
			const size_t k = id;
			bck5_0(x[k + 0 * n_10], x[k + 1 * n_10], x[k + 2 * n_10], x[k + 3 * n_10], x[k + 4 * n_10]);
		}
	}

	// Radix-4
	void forward_mul4(uint64_2 * const x) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 * const r2 = &_root.data()[0];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 2 * id;

			fwd22(x[k + 0], x[k + 1], r2[n_4 + j]);
		}
	}

	// Radix-4, square, inverse radix-4
	void sqr4(uint64_2 * const x) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[n / 2];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 2 * id;

			uint64_2 x01 = x[k + 0], x23 = x[k + 1];
			const uint64 r = r2[n_4 + j];
			fwd22(x01, x23, r);
			sqr22(x01, x23, r);
			bck22(x01, x23, r2i[n_4 + j]);
			x[k + 0] = x01; x[k + 1] = x23;
		}
	}

	// Radix-4, mul, inverse radix-4
	void mul4(uint64_2 * const x, const uint64_2 * const y) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[_n / 2];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 2 * id;

			uint64_2 x01 = x[k + 0], x23 = x[k + 1];
			const uint64 r = r2[n_4 + j];
			fwd22(x01, x23, r);
			mul22(x01, x23, y[k + 0], y[k + 1], r);
			bck22(x01, x23, r2i[n_4 + j]);
			x[k + 0] = x01; x[k + 1] = x23;
		}
	}

	// Radix-8
	void forward_mul8(uint64_2 * const x) const
	{
		const size_t n = _n, n_8 = _n / 8;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r4 = &_root.data()[n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t j = id, k = 4 * id;
	
			fwd4(x[k + 0], x[k + 1], x[k + 2], x[k + 3], r2[n_8 + j], r4[2 * (n_8 + j) + 0], r4[2 * (n_8 + j) + 1]);
		}
	}

	// Radix-8, square, inverse radix-8
	void sqr8(uint64_2 * const x) const
	{
		const size_t n = _n, n_8 = n / 8;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[n / 2];
		const uint64 * const r4 = &_root.data()[n];
		const uint64 * const r4i = &_root.data()[n + n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t j = id, k = 4 * id;

			uint64_2 x01 = x[k + 0], x23 = x[k + 1], x45 = x[k + 2], x67 = x[k + 3];
			const uint64 r20 = r4[2 * (n_8 + j) + 0], r21 = r4[2 * (n_8 + j) + 1];
			fwd4(x01, x23, x45, x67, r2[n_8 + j], r20, r21);
			sqr22(x01, x23, r20);
			sqr22(x45, x67, mod_muli(r20));
			bck4(x01, x23, x45, x67, r2i[n_8 + j], r4i[2 * (n_8 + j) + 0], r4i[2 * (n_8 + j) + 1]);
			x[k + 0] = x01; x[k + 1] = x23; x[k + 2] = x45; x[k + 3] = x67;
		}
	}

	// Radix-8, mul, inverse radix-8
	void mul8(uint64_2 * const x, const uint64_2 * const y) const
	{
		const size_t n = _n, n_8 = n / 8;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[n / 2];
		const uint64 * const r4 = &_root.data()[n];
		const uint64 * const r4i = &_root.data()[n + n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t j = id, k = 4 * id;

			uint64_2 x01 = x[k + 0], x23 = x[k + 1], x45 = x[k + 2], x67 = x[k + 3];
			const uint64 r20 = r4[2 * (n_8 + j) + 0], r21 = r4[2 * (n_8 + j) + 1];
			fwd4(x01, x23, x45, x67, r2[n_8 + j], r20, r21);
			mul22(x01, x23, y[k + 0], y[k + 1], r20);
			mul22(x45, x67, y[k + 2], y[k + 3], mod_muli(r20));
			bck4(x01, x23, x45, x67, r2i[n_8 + j], r4i[2 * (n_8 + j) + 0], r4i[2 * (n_8 + j) + 1]);
			x[k + 0] = x01; x[k + 1] = x23; x[k + 2] = x45; x[k + 3] = x67;
		}
	}

	// Transform, n = 4^e or 5 * 4^e
	void forward(uint64_2 * const x) const
	{
		const size_t n = _n, s5 = (n % 5 == 0) ? 5 : 4;

		if (n % 5 == 0) forward5_0(x);
		else if (n >= 16) forward4_0(x);

		const int lm_min = _even ? 1 : 2, lm_max = ilog2(n / 8 / s5);
		size_t s = s5;
		for (int lm = lm_max; lm >= lm_min; lm -= 2, s *= 4) forward4(x, s, lm);
	}

	// Inverse Transform, n = 4^e or 5 * 4^e
	void backward(uint64_2 * const x) const
	{
		const size_t n = _n, s5 = (n % 5 == 0) ? 5 : 4;

		const int lm_min = _even ? 1 : 2, lm_max = ilog2(n / 8 / s5);
		size_t s = (n / 8) >> lm_min;
		for (int lm = lm_min; lm <= lm_max; lm += 2, s /= 4) backward4(x, s, lm);

		if (s5 == 5) backward5_0(x);
		else if (n >= 16) backward4_0(x);
	}

	// Unweight, carry, mul by a, weight
	void carry_weight_mul(uint64_2 * const x2, const uint32 a = 1) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 inv_n_2 = _inv_n_2;
		uint64_4 * const x = reinterpret_cast<uint64_4 *>(x2);
		const uint64_2 * const weight2 = reinterpret_cast<const uint64_2 *>(_weight.data());
		const uint8_4 * const width4 = reinterpret_cast<const uint8_4 *>(_digit_width.data());
		uint64 * const carry = const_cast<uint64 *>(_carry.data());

		for (size_t id = 0; id < n_4; ++id)
		{
			uint64_2 w2[4]; for (size_t i = 0; i < 4; ++i) w2[i] = weight2[id + i * n_4];
			// const uint64_4 w = uint64_4(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);
			const uint64_4 wi = uint64_4(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);

			const uint8_4 wd = width4[id];

			uint64 c = 0;
			const uint64_4 u = mod_mul4(mod_mul4(x[id], inv_n_2), wi);
			x[id] = adc_mul4(u, a, wd, c);

			carry[(id != n_4 - 1) ? id + 1 : 0] = c;
		}

		for (size_t id = 0; id < n_4; ++id)
		{
			uint64_2 w2[4]; for (size_t i = 0; i < 4; ++i) w2[i] = weight2[id + i * n_4];
			const uint64_4 w = uint64_4(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);
			// const uint64_4 wi = uint64_4(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);

			const uint8_4 wd = width4[id];

			const uint64_4 u = adc4(x[id], wd, carry[id]);
			x[id] = mod_mul4(u, w);
		}
	}

public:
	engine_cpu(const uint32_t q, const size_t reg_count) : engine(), _n(ibdwt::transform_size(q)),
		_reg_count(reg_count), _inv_n_2(MOD_P - (MOD_P - 1) / (_n / 2)), _even(ibdwt::is_even(_n))
	{
		const size_t n = _n;

		_reg.resize(reg_count * n);
		_root.resize(3 * n);
		_weight.resize(2 * n);
		_digit_width.resize(n);
		_carry.resize(n / 4);

		ibdwt::roots54(n, _root.data());
		ibdwt::weights_widths(n, q, _weight.data(), _digit_width.data());
	}

	virtual ~engine_cpu() {}

	size_t get_size() const override { return _n; }

	void set(const Reg dst, const uint32 a) const override
	{
		const size_t n = _n;
		uint64 * const x = const_cast<uint64 *>(&_reg.data()[size_t(dst) * n]);

		x[0] = a;	// weight[0] = 1
		for (size_t k = 1; k < n; ++k) x[k] = 0;
	}

	void get(uint64 * const d, const Reg src) const override
	{
		const size_t n = _n;
		const uint64 * const x = &_reg.data()[size_t(src) * n];
		const uint64 * const w = &_weight.data()[0];
		const uint8 * const width = _digit_width.data();

		// unweight, carry (strong)
		uint64 c = 0;
		for (size_t k = 0; k < n; ++k)
		{
			const uint64 wi = w[2 * (k / 4 + (k % 4) * (n / 4)) + 1];
			d[k] = adc(mod_mul(x[k], wi), width[k], c);
		} 

		while (c != 0)
		{
			for (size_t k = 0; k < n; ++k)
			{
				d[k] = adc(d[k], width[k], c);
				if (c == 0) break;
			}
		}

		// encode
		for (size_t k = 0; k < n; ++k) d[k] = uint32(d[k]) | (uint64(width[k]) << 32);
	}

	void copy(const Reg dst, const Reg src) const override
	{
		const size_t n = _n;
		const uint64 * const x = &_reg.data()[size_t(src) * n];
		uint64 * const y = const_cast<uint64 *>(&_reg.data()[size_t(dst) * n]);

		for (size_t k = 0; k < n; ++k) y[k] = x[k];
	}

	bool is_equal(const Reg src1, const Reg src2) const override
	{
		const size_t n = _n;
		const uint64 * const x = &_reg.data()[size_t(src1) * n];
		const uint64 * const y = &_reg.data()[size_t(src2) * n];

		for (size_t k = 0; k < n; ++k) if (y[k] != x[k]) return false;
		return true;
	}

	void square_mul(const Reg src, const uint32 a = 1) const override
	{
		const size_t n = _n;
		uint64_2 * const x = reinterpret_cast<uint64_2 *>(const_cast<uint64 *>(&_reg.data()[size_t(src) * n]));

		forward(x);
		if (_even) sqr4(x); else sqr8(x);
		backward(x);
		carry_weight_mul(x, a);
	}

	void set_multiplicand(const Reg dst, const Reg src) const override
	{
		if (src != dst) copy(dst, src);

		const size_t n = _n;
		uint64_2 * const y = reinterpret_cast<uint64_2 *>(const_cast<uint64 *>(&_reg.data()[size_t(dst) * n]));

		forward(y);
		if (_even) forward_mul4(y); else forward_mul8(y);
	}

	void mul(const Reg dst, const Reg src) const override
	{
		const size_t n = _n;
		uint64_2 * const x = reinterpret_cast<uint64_2 *>(const_cast<uint64 *>(&_reg.data()[size_t(dst) * n]));
		const uint64_2 * const y = reinterpret_cast<const uint64_2 *>(&_reg.data()[size_t(src) * n]);

		forward(x);
		if (_even) mul4(x, y); else mul8(x, y);
		backward(x);
		carry_weight_mul(x);
	}

	void sub(const Reg src, const uint32 a) const override
	{
		const size_t n = _n;
		uint64 * const x = const_cast<uint64 *>(&_reg.data()[size_t(src) * n]);
		const uint64_2 * const weight2 = reinterpret_cast<const uint64_2 *>(&_weight.data()[0]);
		const uint8 * const width = _digit_width.data();

		uint32 c = a;
		while (c != 0)
		{
			// Unweight, sub with carry, weight
			for (size_t k = 0; k < n; ++k)
			{
				const uint64_2 w = weight2[k / 4 + (k % 4) * (n / 4)];
				x[k] = mod_mul(sbc(mod_mul(x[k], w.s1), width[k], c), w.s0);
				if (c == 0) return;
			}
		}
	}

	size_t get_register_data_size() const override  { return _n * sizeof(uint64); }

	bool get_data(std::vector<char> & data, const Reg src) const override
	{
		const size_t size = get_register_data_size();
		if (data.size() != size) return false;
		std::memcpy(data.data(), &_reg.data()[size_t(src) * _n], size);
		return true;
	}

	bool set_data(const Reg dst, const std::vector<char> & data) const override
	{
		const size_t size = get_register_data_size();
		if (data.size() != size) return false;
		std::memcpy(const_cast<uint64 *>(&_reg.data()[size_t(dst) * _n]), data.data(), size);
		return true;
	}

	size_t get_checkpoint_size() const override { return _reg_count * _n * sizeof(uint64); }

	bool get_checkpoint(std::vector<char> & data) const override
	{
		const size_t size = get_checkpoint_size();
		if (data.size() != size) return false;
		std::memcpy(data.data(), _reg.data(), size);
		return true;
	}

	bool set_checkpoint(const std::vector<char> & data) const override
	{
		const size_t size = get_checkpoint_size();
		if (data.size() != size) return false;
		std::memcpy(const_cast<uint64 *>(_reg.data()), data.data(), size);
		return true;
	}
};
