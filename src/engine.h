/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <vector>

#include "arith.h"

class engine
{
protected:
	// d is encoded: low 32-bit word is the value and high 32-bit word is the width of the base
	virtual void get(uint64 * const d, const size_t src) const = 0;	

public:
	engine() {}
	virtual ~engine() {}

	typedef size_t Reg;

	virtual size_t get_size() const = 0;
	virtual void set(const Reg dst, const uint64 a) const = 0;
	virtual void copy(const Reg dst, const Reg src) const = 0;
	virtual bool is_equal(const Reg src1, const Reg src2) const = 0;
	virtual void square_mul(const Reg src, const uint32 a = 1) const = 0;
	virtual void set_multiplicand(const Reg dst, const Reg src) const = 0;
	virtual void mul(const Reg dst, const Reg src) const = 0;
	virtual void sub(const Reg src, const uint32 a) const = 0;

	virtual size_t get_checkpoint_size() const = 0;
	virtual bool get_checkpoint(std::vector<char> & data) const = 0;
	virtual bool set_checkpoint(const std::vector<char> & data) const = 0;

	// reg_0 = reg^e, reg is erased
	void pow(const size_t reg, const uint64_t e) const
	{
		set_multiplicand(reg, reg);
		set(0, 1);
		if (e == 0) return;
		for (int i = std::bit_width(e) - 1; i >= 0; --i)
		{
			square_mul(0);
			if ((e & (static_cast<uint64_t>(1) << i)) != 0) mul(0, reg);
		}
	}

	class digit
	{
	private:
		std::vector<uint64> _data;

	public:
		digit(engine * const eng, const Reg src)
		{
			_data.resize(eng->get_size());
			eng->get(_data.data(), src);
		}

		virtual ~digit() {}

		size_t get_size() const { return _data.size(); }
		uint32 val(const size_t i) const { return uint32(_data[i]); }
		uint8 width(const size_t i) const { return uint8(_data[i] >> 32); }

		uint64 res64() const
		{
			uint64_t r64 = 0; uint8 s = 0;
			for (uint64 d :_data)
			{
				const uint64_t u = uint32(d);
				const uint8 width = uint8(d >> 32);
				r64 += u << s;
				s += width;
				if (s >= 64) break;
			}
			return r64;
		}

		bool equal_to(const uint64_t a) const
		{
			uint64_t r = a;
			for (uint64 d :_data)
			{
				const uint64_t u = uint32(d);
				const uint8 width = uint8(d >> 32);
				if ((r & ((uint64_t(1) << width) - 1)) != u) return false;
				r >>= width;
			}
			return true;
		}

		bool equal_to_Mp() const
		{
			for (uint64 d :_data)
			{
				const uint64_t u = uint32(d);
				const uint8 width = uint8(d >> 32);
				if (u != (uint64_t(1) << width) - 1) return false;
			} 
			return true;
		}
	};

	static engine * create_gpu(const uint32_t q, const size_t reg_count, const size_t device, const bool verbose);
	static engine * create_cpu(const uint32_t q);
};
