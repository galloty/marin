/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"
#include "ibdwt.h"

#include <cstring>

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
	static void fwd4(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 r1, const uint64 r20, const uint64 r21)
	{
		const uint64 u0 = x0, u2 = mod_mul(r1, x2), u1 = mod_mul(r20, x1), u3 = mod_mul(r21, x3);
		const uint64 v0 = mod_add(u0, u2), v2 = mod_sub(u0, u2), v1 = mod_add(u1, u3), v3 = mod_muli(mod_sub(u1, u3));
		x0 = mod_add(v0, v1); x1 = mod_sub(v0, v1); x2 = mod_add(v2, v3); x3 = mod_sub(v2, v3);
	}

	// Inverse radix-4
	static void bck4(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 ri1, const uint64 ri20, const uint64 ri21)
	{
		const uint64 u0 = x0, u1 = x1, u2 = x2, u3 = x3;
		const uint64 v0 = mod_add(u0, u1), v1 = mod_sub(u0, u1), v2 = mod_add(u3, u2), v3 = mod_muli(mod_sub(u3, u2));
		x0 = mod_add(v0, v2); x2 = mod_mul(ri1, mod_sub(v0, v2)); x1 = mod_mul(ri20, mod_add(v1, v3)); x3 = mod_mul(ri21, mod_sub(v1, v3));
	}

	// Radix-4, first stage
	static void fwd4_0(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3)
	{
		const uint64 u0 = x0, u2 = x2, u1 = x1, u3 = x3;
		const uint64 v0 = mod_add(u0, u2), v2 = mod_sub(u0, u2), v1 = mod_add(u1, u3), v3 = mod_muli(mod_sub(u1, u3));
		x0 = mod_add(v0, v1); x1 = mod_sub(v0, v1); x2 = mod_add(v2, v3); x3 = mod_sub(v2, v3);
	}

	// Inverse radix-4, first stage
	static void bck4_0(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3)
	{
		const uint64 u0 = x0, u1 = x1, u2 = x2, u3 = x3;
		const uint64 v0 = mod_add(u0, u1), v1 = mod_sub(u0, u1), v2 = mod_add(u3, u2), v3 = mod_muli(mod_sub(u3, u2));
		x0 = mod_add(v0, v2); x2 = mod_sub(v0, v2); x1 = mod_add(v1, v3); x3 = mod_sub(v1, v3);
	}

	// 2 x radix-2
	static void fwd22(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 r)
	{
		const uint64 u0 = x0, u2 = mod_mul(r, x2), u1 = x1, u3 = mod_mul(r, x3);
		x0 = mod_add(u0, u2); x2 = mod_sub(u0, u2); x1 = mod_add(u1, u3); x3 = mod_sub(u1, u3);
	}

	// 2 x inverse radix-2
	static void bck22(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 ri)
	{
		const uint64 u0 = x0, u2 = x2, u1 = x1, u3 = x3;
		x0 = mod_add(u0, u2); x2 = mod_mul(ri, mod_sub(u0, u2)); x1 = mod_add(u1, u3); x3 = mod_mul(ri, mod_sub(u1, u3));
	}

	// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.
	static void butterfly5(uint64 & a0, uint64 & a1, uint64 & a2, uint64 & a3, uint64 & a4)
	{
		static const uint64 K = mod_root_nth(5), K2 = mod_sqr(K), K3 = mod_mul(K, K2), K4 = mod_sqr(K2);
		static const uint64 cosu = mod_half(mod_add(K, K4)), isinu = mod_half(mod_sub(K, K4));
		static const uint64 cos2u = mod_half(mod_add(K2, K3)), isin2u = mod_half(mod_sub(K2, K3));
		static const uint64 F1 = mod_sub(mod_half(mod_add(cosu, cos2u)), 1), F2 = mod_half(mod_sub(cosu, cos2u));
		static const uint64 F3 = mod_add(isinu, isin2u), F4 = isin2u, F5 = mod_sub(isinu, isin2u);

		const uint64 s1 = mod_add(a1, a4), s2 = mod_sub(a1, a4), s3 = mod_add(a3, a2), s4 = mod_sub(a3, a2);
		const uint64 s5 = mod_add(s1, s3), s6 = mod_sub(s1, s3), s7 = mod_add(s2, s4), s8 = mod_add(s5, a0);
		const uint64 m0 = s8;
		const uint64 m1 = mod_mul(F1, s5), m2 = mod_mul(F2, s6), m3 = mod_mul(F3, s2), m4 = mod_mul(F4, s7), m5 = mod_mul(F5, s4);
		const uint64 s9 = mod_add(m0, m1), s10 = mod_add(s9, m2), s11 = mod_sub(s9, m2), s12 = mod_sub(m3, m4);
		const uint64 s13 = mod_add(m4, m5), s14 = mod_add(s10, s12), s15 = mod_sub(s10, s12), s16 = mod_add(s11, s13);
		const uint64 s17 = mod_sub(s11, s13);
		a0 = m0; a1 = s14; a2 = s16; a3 = s17; a4 = s15;
	}

	// Radix-5, first stage
	static void fwd5_0(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, uint64 & x4)
	{
		uint64 a0 = x0, a1 = x1, a2 = x2, a3 = x3, a4 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// Inverse radix-5, first stage
	static void bck5_0(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, uint64 & x4)
	{
		uint64 a0 = x0, a4 = x1, a3 = x2, a2 = x3, a1 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// 2 x Radix-2, sqr, inverse radix-2
	static void sqr22(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 & r)
	{
		const uint64 t0 = mod_add(mod_sqr(x0), mod_mul(mod_sqr(x1), r)); x1 = mod_mul(x1, mod_add(x0, x0)); x0 = t0;
		const uint64 t2 = mod_sub(mod_sqr(x2), mod_mul(mod_sqr(x3), r)); x3 = mod_mul(x3, mod_add(x2, x2)); x2 = t2;
	}

	static void mul22(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 & y0, const uint64 & y1, const uint64 & y2, const uint64 & y3, const uint64 & r)
	{
		const uint64 t0 = mod_add(mod_mul(x0, y0), mod_mul(mod_mul(x1, y1), r)); x1 = mod_add(mod_mul(x0, y1), mod_mul(x1, y0)); x0 = t0;
		const uint64 t2 = mod_sub(mod_mul(x2, y2), mod_mul(mod_mul(x3, y3), r)); x3 = mod_add(mod_mul(x2, y3), mod_mul(x3, y2)); x2 = t2;
	}

	// Radix-4
	void forward_mul4(uint64 * const x) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 * const r2 = &_root.data()[0];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 4 * id;

			fwd22(x[k + 0], x[k + 1], x[k + 2], x[k + 3], r2[n_4 + j]);
		}
	}

	// Radix-4, square, inverse radix-4
	void sqr4(uint64 * const x) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[n / 2];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 4 * id;

			const uint64 r = r2[n_4 + j];
			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd22(x0, x1, x2, x3, r);
			sqr22(x0, x1, x2, x3, r);
			bck22(x0, x1, x2, x3, r2i[n_4 + j]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
		}
	}

	// Radix-4, mul, inverse radix-4
	void mul4(uint64 * const x, const uint64 * const y) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[_n / 2];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 4 * id;

			const uint64 r = r2[n_4 + j];
			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd22(x0, x1, x2, x3, r);
			mul22(x0, x1, x2, x3, y[k + 0], y[k + 1], y[k + 2], y[k + 3], r);
			bck22(x0, x1, x2, x3, r2i[n_4 + j]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
		}
	}

	// Radix-8
	void forward_mul8(uint64 * const x) const
	{
		const size_t n = _n, n_8 = _n / 8;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r4 = &_root.data()[n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t j = id, k = 8 * id;
	
			const uint64 r1 = r2[n_8 + j], r20 = r4[2 * (n_8 + j) + 0], r21 = r4[2 * (n_8 + j) + 1];
			fwd4(x[k + 0], x[k + 2], x[k + 4], x[k + 6], r1, r20, r21);
			fwd4(x[k + 1], x[k + 3], x[k + 5], x[k + 7], r1, r20, r21);
		}
	}

	// Radix-8, square, inverse radix-8
	void sqr8(uint64 * const x) const
	{
		const size_t n = _n, n_8 = n / 8;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[n / 2];
		const uint64 * const r4 = &_root.data()[n];
		const uint64 * const r4i = &_root.data()[n + n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t j = id, k = 8 * id;

			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			uint64 x4 = x[k + 4], x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7];

			const uint64 r1 = r2[n_8 + j], r20 = r4[2 * (n_8 + j) + 0], r21 = r4[2 * (n_8 + j) + 1];
			fwd4(x0, x2, x4, x6, r1, r20, r21);
			fwd4(x1, x3, x5, x7, r1, r20, r21);

			sqr22(x0, x1, x2, x3, r20);
			sqr22(x4, x5, x6, x7, mod_muli(r20));

			const uint64 r1i = r2i[n_8 + j], r20i = r4i[2 * (n_8 + j) + 0], r21i = r4i[2 * (n_8 + j) + 1];
			bck4(x0, x2, x4, x6, r1i, r20i, r21i);
			bck4(x1, x3, x5, x7, r1i, r20i, r21i);

			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
			x[k + 4] = x4; x[k + 5] = x5; x[k + 6] = x6; x[k + 7] = x7;
		}
	}

	// Radix-8, mul, inverse radix-8
	void mul8(uint64 * const x, const uint64 * const y) const
	{
		const size_t n = _n, n_8 = n / 8;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r2i = &_root.data()[n / 2];
		const uint64 * const r4 = &_root.data()[n];
		const uint64 * const r4i = &_root.data()[n + n];

		for (size_t id = 0; id < n_8; ++id)
		{
			const size_t j = id, k = 8 * id;

			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			uint64 x4 = x[k + 4], x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7];

			const uint64 r1 = r2[n_8 + j], r20 = r4[2 * (n_8 + j) + 0], r21 = r4[2 * (n_8 + j) + 1];
			fwd4(x0, x2, x4, x6, r1, r20, r21);
			fwd4(x1, x3, x5, x7, r1, r20, r21);

			mul22(x0, x1, x2, x3, y[k + 0], y[k + 1], y[k + 2], y[k + 3], r20);
			mul22(x4, x5, x6, x7, y[k + 4], y[k + 5], y[k + 6], y[k + 7], mod_muli(r20));

			const uint64 r1i = r2i[n_8 + j], r20i = r4i[2 * (n_8 + j) + 0], r21i = r4i[2 * (n_8 + j) + 1];
			bck4(x0, x2, x4, x6, r1i, r20i, r21i);
			bck4(x1, x3, x5, x7, r1i, r20i, r21i);

			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
			x[k + 4] = x4; x[k + 5] = x5; x[k + 6] = x6; x[k + 7] = x7;
		}
	}

	// Transform, n = 4^e or 5 * 4^e
	void forward(uint64 * const x) const
	{
		const size_t n = _n, n_4 = n / 4, s5 = (n % 5 == 0) ? 5 : 4, n_5 = (n % 5 == 0) ? n / 5 : n;
		const uint64 * const r2 = &_root.data()[0];
		const uint64 * const r4 = &_root.data()[n];

		if (s5 == 5)
		{
			for (size_t id = 0; id < n_5; ++id)
			{
				const size_t k = id;
				fwd5_0(x[k + 0 * n_5], x[k + 1 * n_5], x[k + 2 * n_5], x[k + 3 * n_5], x[k + 4 * n_5]);
			}
		}
		else if (n >= 16)
		{
			for (size_t id = 0; id < n_4; ++id)
			{
				const size_t k = id;
				fwd4_0(x[k + 0 * n_4], x[k + 1 * n_4], x[k + 2 * n_4], x[k + 3 * n_4]);
			}
		}

		const int lm_min = _even ? 2 : 3, lm_max = ilog2(n_4 / s5);
		size_t s = s5;
		for (int lm = lm_max; lm >= lm_min; lm -= 2, s *= 4)
		{
			for (size_t id = 0; id < n_4; ++id)
			{
				const size_t m = size_t(1) << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;
				fwd4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r2[sj], r4[2 * sj + 0], r4[2 * sj + 1]);
			}
		}
	}

	// Inverse Transform, n = 4^e or 5 * 4^e
	void backward(uint64 * const x) const
	{
		const size_t n = _n, n_4 = n / 4, s5 = (n % 5 == 0) ? 5 : 4, n_5 = (n % 5 == 0) ? n / 5 : n;
		const uint64 * const r2i = &_root.data()[n / 2];
		const uint64 * const r4i = &_root.data()[n + n];

		const int lm_min = _even ? 2 : 3, lm_max = ilog2(n_4 / s5);
		size_t s = n_4 >> lm_min;
		for (int lm = lm_min; lm <= lm_max; lm += 2, s /= 4)
		{
			for (size_t id = 0; id < n_4; ++id)
			{
				const size_t m = size_t(1) << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;
				bck4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r2i[sj], r4i[2 * sj + 0], r4i[2 * sj + 1]);
			}
		}

		if (s5 == 5)
		{
			for (size_t id = 0; id < n_5; ++id)
			{
				const size_t k = id;
				bck5_0(x[k + 0 * n_5], x[k + 4 * n_5], x[k + 3 * n_5], x[k + 2 * n_5], x[k + 1 * n_5]);
			}
		}
		else if (n >= 16)
		{
			for (size_t id = 0; id < n_4; ++id)
			{
				const size_t k = id;
				bck4_0(x[k + 0 * n_4], x[k + 1 * n_4], x[k + 2 * n_4], x[k + 3 * n_4]);
			}
		}
	}

	// Unweight, carry, mul by a, weight
	void carry_weight_mul(uint64 * const x, const uint32 a = 1) const
	{
		const size_t n = _n, n_4 = n / 4;
		const uint64 inv_n_2 = _inv_n_2;
		const uint64 * const w = &_weight.data()[0];
		const uint8 * const width = _digit_width.data();
		uint64 * const carry = const_cast<uint64 *>(_carry.data());

		for (size_t id = 0; id < n_4; ++id)
		{
			uint64 c = 0;
			for (size_t i = 0; i < 4; ++i)
			{
				const size_t k = 4 * id + i;
				const uint64 wi = w[2 * (k / 4 + (k % 4) * (n / 4)) + 1];
				const uint64 u = mod_mul(mod_mul(x[k], inv_n_2), wi);
				x[k] = adc_mul(u, a, width[k], c);
			}
			carry[(id != n_4 - 1) ? id + 1 : 0] = c;
		}

		for (size_t id = 0; id < n_4; ++id)
		{
			uint64 c = carry[id];
			for (size_t i = 0; i < 3; ++i)
			{
				const size_t k = 4 * id + i;
				x[k] = mod_mul(adc(x[k], width[k], c), w[2 * (k / 4 + (k % 4) * (n / 4)) + 0]);
			}
			const size_t k = 4 * id + 3;
			x[k] = mod_mul(x[k] + c, w[2 * (k / 4 + (k % 4) * (n / 4)) + 0]);
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

		for (size_t k = 0; k < n; ++k) d[k] = x[k];

		// unweight, carry (strong)
		uint64 c = 0;
		for (size_t k = 0; k < n; ++k)
		{
			const uint64 wi = w[2 * (k / 4 + (k % 4) * (n / 4)) + 1];
			d[k] = adc(mod_mul(d[k], wi), width[k], c);
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
		uint64 * const x = const_cast<uint64 *>(&_reg.data()[size_t(src) * n]);

		forward(x);
		if (_even) sqr4(x); else sqr8(x);
		backward(x);
		carry_weight_mul(x, a);
	}

	void set_multiplicand(const Reg dst, const Reg src) const override
	{
		if (src != dst) copy(dst, src);

		const size_t n = _n;
		uint64 * const y = const_cast<uint64 *>(&_reg.data()[size_t(dst) * n]);

		forward(y);
		if (_even) forward_mul4(y); else forward_mul8(y);
	}

	void mul(const Reg dst, const Reg src) const override
	{
		const size_t n = _n;
		uint64 * const x = const_cast<uint64 *>(&_reg.data()[size_t(dst) * n]);
		const uint64 * const y = &_reg.data()[size_t(src) * n];

		forward(x);
		if (_even) mul4(x, y); else mul8(x, y);
		backward(x);
		carry_weight_mul(x);
	}

	void sub(const Reg src, const uint32 a) const override
	{
		const size_t n = _n;
		uint64 * const x = const_cast<uint64 *>(&_reg.data()[size_t(src) * n]);
		const uint64 * const w = &_weight.data()[0];
		const uint8 * const width = _digit_width.data();

		uint32 c = a;
		while (c != 0)
		{
			// Unweight, sub with carry, weight
			for (size_t k = 0; k < n; ++k)
			{
				const uint64 wi = w[2 * (k / 4 + (k % 4) * (n / 4)) + 1];
				x[k] = mod_mul(sbc(mod_mul(x[k], wi), width[k], c), w[2 * (k / 4 + (k % 4) * (n / 4)) + 0]);
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
